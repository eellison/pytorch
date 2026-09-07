# Eellison H1 2026 Inductor Low-Precision/Nested-Reduction Impact

Scope: eellison-authored commits reachable from current `HEAD`, dated 2026-01-01 through 2026-06-30. I used `git log ... HEAD` and `git show --name-only` only, with no `--all` and no branch changes. GitHub CLI PR metadata fetch failed through the local proxy, so PR numbers/titles/dates come from landed commit trailers and commit dates. Commits without PR trailers are listed separately as "PR unavailable".

## Benchmark Harness Pattern

Use two separately compiled callables per comparison so compile and autotune work is outside the timing path:

```python
import torch
import torch._dynamo
from torch._inductor import config, metrics

def compile_variant(fn, args, patches):
    torch._dynamo.reset()
    metrics.reset()
    with config.patch(patches):
        cfn = torch.compile(fn, fullgraph=True)
        cfn(*args)
        torch.cuda.synchronize()
        generated = metrics.generated_kernel_count
        nested = metrics.codegen_nested_reduction
    return cfn, generated, nested

def graph_replay_ms(cfn, args, iters=1000, warmup=20):
    for _ in range(warmup):
        cfn(*args)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        cfn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters
```

Primary same-checkout toggles:

- Nested reductions: `{"triton.nested_reduction": False}` versus `{"triton.nested_reduction": True}`.
- Reshape/reduction reindexing: `{"loop_reindexing_after_fusion": False}` versus `{"loop_reindexing_after_fusion": True}`.
- MX E8M0 pattern replacement: `{"pattern_matcher": False}` versus `{"pattern_matcher": True}` for the log2/ceil pattern. Direct `inductor_prims.cvt_e8m0_rceil` has no feature-off toggle.

Current HEAD caveat: `triton.nested_reduction` is off by default (`TORCHINDUCTOR_NESTED_REDUCTION=0` unless patched), so runtime impact is directly benchmarkable on HEAD but only active when this config is enabled.

## PRs And Impact

### PR #172497, 2026-01-21, `b79269a4308`

Title: `[Inductor] Add cvt_e8m0_rceil prim with PTX lowering for SM100+`

Impact: Adds `inductor_prims.cvt_e8m0_rceil` and pattern replacement for MX/MXFP8 E8M0 scale generation. On SM100+ it lowers to `cvt.rp.satfinite.ue8m0x2.f32`; on earlier CUDA hardware the log2/ceil pattern replacement uses exact bit manipulation. User-visible impact is faster and more numerically robust MX scale encoding in compiled low-precision quantization flows.

Benchmarkability: Directly benchmarkable on current HEAD. Use `pattern_matcher=False/True` for a software log2/ceil MX scale extraction function. Direct prim benchmarking requires SM100+ and does not have a config-off comparison.

Graph-replay recommendation: Benchmark `scale = (ceil(log2(abs(x)+eps)).clamp(-127,127)+127).to(uint8)` on `x` shapes `(1 << 20,)`, `(1 << 24,)`, and block-scale-like `(8192, 128)` float32 CUDA tensors. On B200/SM100, also benchmark direct `inductor_prims.cvt_e8m0_rceil(x)` as an upper-bound primitive throughput check.

### PR #177922, 2026-03-25, `1ae64875e7e`

Title: `[inductor] Add inline_asm_elementwise higher-order operator`

Impact: Enables inline PTX in eager and compiled mode, lowering to `tl.inline_asm_elementwise` in Inductor. This is the key enabler for NVFP4 packing instructions such as `cvt.rn.satfinite.e2m1x2.f32`, used later by NVFP4 nested-reduction packing.

Benchmarkability: Partially benchmarkable on current HEAD. There is no feature-off toggle; use primitive throughput or benchmark as part of PR #183638. Requires CUDA; the NVFP4 conversion path requires SM100+.

Graph-replay recommendation: On SM100+, benchmark an elementwise pack microkernel with `even` and `odd` tensors shaped `(128, 4096 // 16, 8)` and asm `cvt.rn.satfinite.e2m1x2.f32`. Compare against a non-asm emulation only as a rough control; the more meaningful benchmark is the full NVFP4 RMSNorm/amax/pack graph under PR #183638.

### PR #176345, 2026-04-14, `54028337cf5`

Title: `Simplify FloorDiv(ModularIndexing) and generalize remove_zero_terms`

Impact: Cleans up reshape/reduction index expressions, especially FloorDiv-over-ModularIndexing forms. This is supporting work for fusion paths where the same data is viewed through different loop factorizations.

Benchmarkability: Not cleanly isolatable by current config. Its impact shows up as more fusion opportunities and simpler indexing in later reshape/reduction benchmarks.

Graph-replay recommendation: Use the PR #176927 RMSNorm reshape benchmark below. To isolate only this PR, compare against the parent checkout; same-HEAD toggles will not separate this simplifier from later scheduler changes.

### PR #176927, 2026-04-15, `ad0b2d38cb7`

Title: `Reindex pointwise iteration loops to enable fusion with reductions`

Impact: Lets pointwise and reduction nodes fuse when they operate on the same data through different iteration spaces, for example `[M, N]` versus `[M*num_heads, head_dim]` after reshape. Motivation and tests are qknorm/RMSNorm-style fusion.

Benchmarkability: Directly benchmarkable on current HEAD with `loop_reindexing_after_fusion=False/True`. Expected signal is one generated kernel with reindexing enabled, more kernels and extra memory traffic without it.

Graph-replay recommendation: Use BF16 CUDA `qkv = randn(M, 10240)`, `x = qkv[:, :8192]`, `head_dim=128`, with `M=1024` for perf and `M=16` for quick validation. Function: reshape to `(-1, 128)`, compute RMSNorm over dim -1, reshape back to `(M, 8192)`, cast to BF16. Measure graph replay for the compiled function under the two config patches.

### PR #179090, 2026-04-15, `afabcbabb5c`

Title: `Prioritize write->read deps in loop reordering candidate selection`

Impact: Fixes fusion selection for FP8 grouped quantization where a large shared input read used to hide the smaller reduction-output dependency that actually blocked fusion. The commit calls out shape factorization `[8, 7168]` versus `[448, 128]`.

Benchmarkability: Partially benchmarkable on HEAD. There is no dedicated off toggle for this heuristic; it is best measured by the grouped quantization repro and by parent-vs-child checkout if isolation is required.

Graph-replay recommendation: Use `x` BF16 CUDA shape `(8, 7168)`, group size `128`, `FP8_MAX=448`. Function: `grouped=x.reshape(-1, 56, 128).float()`, `absmax=grouped.abs().amax(-1, keepdim=True)`, `scale=(absmax/448).clamp(min=1e-6)`, `x_q=(grouped/scale).clamp(-448,448).to(float16).reshape_as(x)`, then feed `(x_q, scale.squeeze(-1))` to a fake/custom opaque GEMM. Graph replay captures the fused quantization prelude.

### PR #179091, 2026-04-15, `96c328d8b28`

Title: `Use pointwise cat when cat inputs recombine the same data`

Impact: Allows qknorm/RMSNorm followed by RoPE-style recombination through `cat` to remain pointwise-fusible when the cat inputs are just recombining the same reduction output. User-visible impact is fewer kernels in transformer q/k normalization plus RoPE patterns.

Benchmarkability: Partially benchmarkable on HEAD. There is no precise off toggle; compare against parent checkout for isolation, or use kernel count/runtime as a current-head sanity check.

Graph-replay recommendation: Use qknorm/RoPE shapes from the landed test and scale up for perf: quick `(B,H,S,D)=(4,8,128,64)`, perf `(4,32,2048,128)`. Function: RMSNorm over last dim, split half, apply cos/sin rotation, `torch.cat([out1, out2], dim=-1)`. Graph replay the compiled function and check `metrics.generated_kernel_count == 1`.

### PR #182891, 2026-05-11, `de59f9192c7`

Title: `[inductor] Factor shared fusion and codegen helpers`

Impact: Preparatory refactor for nested reductions: shared GPU Triton predicate, intermediate dependency checks, row-major flatten/decompose helpers, and SIMD launch cleanup. It is enabling work, not a standalone user-visible speedup.

Benchmarkability: Not directly toggleable on HEAD. Benchmark through downstream nested-reduction PRs.

Graph-replay recommendation: Use the PR #182897 RMSNorm plus grouped `amax` benchmark. Isolate this PR only by checking out the parent/child commits.

### PR #182892, 2026-05-11, `114c93e2b43`

Title: `[inductor] Make Triton range metadata root-owned`

Impact: Moves block size, block offset, and mask metadata onto range roots so derived iteration spaces can share parent loop/grid placement. This supports nested-reduction codegen.

Benchmarkability: Not directly toggleable. User-visible impact arrives with PR #182897 and later.

Graph-replay recommendation: Same as PR #182897; inspect generated code only if isolating this refactor.

### PR #182893, 2026-05-12, `f0de237e16b`

Title: `[inductor] Add derived SIMD range roots`

Impact: Adds derived range roots, which let a nested reduction use the parent kernel placement while exposing a different logical numel/block/mask. This is central codegen scaffolding for grouped reductions nested inside an outer norm reduction.

Benchmarkability: Not directly toggleable. Downstream nested-reduction graphs exercise it.

Graph-replay recommendation: Use `LayerNorm/RMSNorm -> reshape(B, D//G, G).abs().amax(-1)` with `B=128,D=4096,G=16` under `triton.nested_reduction=True/False`.

### PR #182895, 2026-05-12, `5387f64e4d2`

Title: `[inductor] Thread block-size floors through Triton autotuning`

Impact: Lets reduction config generation and coordinate descent respect minimum XBLOCK/RBLOCK requirements. Nested grouped reductions need this so the parent tile contains whole local groups, for example `min_rblock=G`.

Benchmarkability: Partially benchmarkable through nested-reduction graphs. There is no simple current-head off toggle for just block-size floors.

Graph-replay recommendation: Use group sizes that stress block floors: `(B,D,G)=(32,4096,512)`, `(128,4096,16)`, and `(128,8192,16)`. Compile with `triton.nested_reduction=True`, verify a single nested kernel and stable runtime; compare to `triton.nested_reduction=False` for user-visible effect.

### PR #183585, 2026-05-13, `a13f6966448`

Title: `[inductor] Cache scheduler coalescing analysis`

Impact: Reduces repeated normalized read/write/coalescing analysis during tiling/fusion decisions. This is mostly compile-time support for the nested-reduction stack.

Benchmarkability: Runtime impact is indirect; compile-time impact is benchmarkable but not via graph replay. For graph replay, use downstream kernels and treat this as non-isolated support.

Graph-replay recommendation: Same runtime graphs as PR #182897; for this PR specifically, collect compile wall time separately on large fusion-heavy nested-reduction models.

### PR #182896, 2026-05-13, `8c1ad7d2a91`

Title: `[inductor] Add nested reduction scheduler legality`

Impact: Adds the legality model for fusing a producer reduction with a dependent local/grouped reduction, with REDUCED, LOCAL_REDUCTION_INPUT, and PARENT_FULL scheduler domains. User-visible perf requires later codegen, but this PR defines which patterns can be safely fused.

Benchmarkability: Indirect on current HEAD. It is exercised by `triton.nested_reduction=True` but cannot be separately turned off.

Graph-replay recommendation: Benchmark both legal and rejected forms: legal RMSNorm plus `view(B,D//16,16).abs().amax(-1)`; rejected non-power-of-two or shifted-source forms as correctness/kernel-count controls. Runtime comparison should use `triton.nested_reduction=False/True`.

### PR #183432, 2026-05-14, `546faa04929`

Title: `[inductor] Add scheduler index equivalence for nested reductions`

Impact: Lets nested full-resolution consumers fuse even when producer outputs are read through broadcasted or loop-reordered MemoryDep forms. This matters for scale/full-resolution quant epilogues after grouped amax.

Benchmarkability: Partially benchmarkable through full-resolution quant epilogue graphs. No isolated current-head toggle.

Graph-replay recommendation: Use RMSNorm plus grouped amax plus full-resolution quant: `B=128,D=4096,G=128`, `x=F.rms_norm(x,(D,),weight)`, `x_groups=x.view(B,D//G,G)`, `scale=(x_groups.abs().amax(-1)/448).clamp(min=1e-12)`, `x_quant=(x_groups/scale.unsqueeze(-1)).to(float16)`, return `x_quant.view(B,D), scale`. Compare `triton.nested_reduction=False/True`.

### PR #182897, 2026-05-20, `333b1089103` (duplicate reachable commit: `cd1cd1d5adb`)

Title: `[inductor] Lower nested reductions in SIMD codegen`

Impact: First user-visible nested grouped-reduction lowering. It fuses outer LayerNorm/RMSNorm over `D` with a dependent grouped reduction such as `x_normed.reshape(B,D//G,G).amax(-1)`, avoiding materialization and a second grouped-reduction launch. It routes `FusedNestedReductions` to `codegen_nested_reduction` and counts `metrics.codegen_nested_reduction`.

Benchmarkability: Directly benchmarkable on current HEAD using `triton.nested_reduction=False/True`. This is the cleanest same-checkout benchmark for the stack.

Graph-replay recommendation: Use:

- RMSNorm amax: `B=128,D=4096,G=16`, `x=F.rms_norm(x,(D,),weight)`, return `x.view(B,D//G,G).abs().amax(-1)`.
- LayerNorm amax: `B=64,D=4096,G=16`, manual mean/var, return grouped `amax`.
- Large group floor: `B=32,D=4096,G=512`.

Check runtime, `metrics.generated_kernel_count`, and `metrics.codegen_nested_reduction`. The expected on-config form is one nested Triton kernel.

### PR #182898, 2026-05-19, `61d445e70fb`

Title: `[inductor] Support XBLOCK nested grouped reductions`

Impact: Extends nested grouped reductions to cases where the local grouped reduction splits the parent X tile rather than the R tile. This broadens coverage to weighted/reduced-over-K forms and degenerate collapsed shapes such as `B=1`.

Benchmarkability: Directly benchmarkable with `triton.nested_reduction=False/True`.

Graph-replay recommendation: Use weighted RMSNorm reduce-K: `x` shape `(B,K,D)`, `w` shape `(B,K)`, flatten to `(B*K,D)`, RMSNorm over `D`, reshape to `(B,K,D)`, return `(w[:,:,None] * x_normed).sum(dim=1)`. Use quick `(B,K,D)=(16,16,4096)`, perf `(128,16,4096)`, and edge `B=1,K=16,D=1024`.

### PR #183638, 2026-06-22, `996f03c2bbf`

Title: `[inductor] Fuse NVFP4 nested-reduction packing`

Impact: Adds the factor-2 nested-reduction epilogue needed for NVFP4 pair packing. The grouped `amax`/scale stage runs at reduced resolution, while the inline-asm pack body runs at pair resolution and reuses parent-tile values. User-visible impact is avoiding separate materialization/launches in RMSNorm -> amax/scale -> NVFP4 pack flows.

Benchmarkability: Directly benchmarkable on current HEAD with `triton.nested_reduction=False/True`, but requires SM100+ for the NVFP4 inline asm path.

Graph-replay recommendation: On B200/SM100, use `B=128,D=4096,G=16`, BF16 `x` and `weight`. Function: `x=F.rms_norm(x,(D,),weight)`, group as `(B,D//G,G)`, `scale=(abs().amax(-1)/448).clamp(min=1e-12).to(float8_e4m3fn)`, pair as `(B,D//G,G//2,2)`, divide even/odd by `scale.float().unsqueeze(-1)`, pack with `inline_asm_elementwise(..., asm_str="cvt.rn.satfinite.e2m1x2.f32 ...")`, return packed uint8 `(B,D//2)` and scale `(B,D//G)`.

### PR unavailable, 2026-06-22, `8e6ab2141d8`

Title: `[inductor] Generalize half-resolution epilogues to sub-parent`

Impact: Generalizes the factor-2 half-resolution/NVFP4 path into an interleaved sub-parent factor. Also lets standalone interleaved epilogues work for looped reductions by recomputing source values inside the epilogue loop instead of requiring a persistent parent tile.

Benchmarkability: Directly benchmarkable by toggling `triton.nested_reduction`, but no PR number was present in the reachable commit.

Graph-replay recommendation: Use standalone half-resolution epilogue: `B=128,D=1024,G=16`, `xg=x.view(B,D//G,G)`, `amax=xg.float().abs().amax(-1)`, `scale=(amax/6).clamp(1e-12,448)`, pair into `(G//2,2)`, return even/odd half-resolution transforms plus `scale`. Compare nested off/on.

### PR unavailable, 2026-06-22, `d59fc87a47b`

Title: `[inductor] Fuse contiguous sub-parent epilogues`

Impact: Adds contiguous sub-parent source layouts for chunk/SwiGLU-style consumers after nested reductions. This targets RMSNorm followed by chunked gated epilogues, proving each epilogue read maps to a contiguous lane of the parent source and avoiding store/reload of the reduced intermediate. Limited to power-of-two sub-parent factors with a profitability guard.

Benchmarkability: Directly benchmarkable by toggling `triton.nested_reduction`, but no PR number was present in the reachable commit.

Graph-replay recommendation: Use BF16 RMSNorm chunk epilogues:

- SwiGLU: `B=128,D=8192`, `h=x+residual`, RMS normalize over `D`, multiply by `weight`, `gate,up=h.chunk(2,-1)`, return `silu(gate)*up`.
- Chunk4 gating: same inputs, `a,b,c,d=h.chunk(4,-1)`, return `silu(a)*b + tanh(c)*d`.
- Additional stress: chunk8/chunk16 additive epilogues over `D=1024` and `D=8192`.

Compare nested off/on and record whether the compiled form is one nested kernel.

## Highest-Value Benchmark Set

1. RMSNorm plus grouped amax: `B=128,D=4096,G=16`, nested off/on. This captures the core #182897 user-visible win.
2. Weighted RMSNorm reduce-K: `(B,K,D)=(128,16,4096)`, nested off/on. This captures #182898 XBLOCK coverage.
3. RMSNorm plus full-resolution quant epilogue: `B=128,D=4096,G=128`, nested off/on. This captures #183432-style index equivalence and epilogue use.
4. NVFP4 RMSNorm/amax/pack: `B=128,D=4096,G=16`, SM100+, nested off/on. This captures #177922 plus #183638.
5. RMSNorm chunk SwiGLU: `B=128,D=8192`, nested off/on. This captures contiguous sub-parent epilogues from the no-PR reachable commit.
6. MX E8M0 scale extraction: `x` shape `(1 << 24,)`, `pattern_matcher=False/True`, SM100+ preferred. This captures #172497.
7. qknorm/RoPE reshape fusion: `(B,H,S,D)=(4,32,2048,128)`, `loop_reindexing_after_fusion=False/True` where applicable. This captures #176927/#179091.

## Confidence And Caveats

Confidence is high that the PR list above is the relevant landed/reachable H1 2026 slice for nested reductions, NVFP4, MXFP8 scale conversion, and RMSNorm/qknorm fusion. The source is local reachable history and commit trailers, not local WIP branches.

Caveats:

- GitHub CLI PR detail fetch failed through the proxy, so dates are commit dates from reachable history, not PR merge timestamps from GitHub.
- Several PRs are enabling scaffolding and are not individually benchmarkable on current HEAD; their user-visible runtime impact appears only once #182897/#182898/#183638 or the no-PR sub-parent commits are enabled.
- `triton.nested_reduction` is off by default on current HEAD, so benchmark recommendations explicitly patch it on.
- NVFP4 inline asm and E8M0 PTX paths require SM100+/Blackwell for the intended hardware speedup. Pre-SM100 can still exercise some fallback logic but not the same PTX path.
- CUDA graph replay removes compile and Python launch overhead from the measurement, which is desired here; it does not measure compile time, autotune time, or dynamic-shape recompile behavior.
