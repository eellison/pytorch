# trtllm-gen attention under host tracing: trace FlashInfer's real C++ launcher (design sketch, 2026-10-08)

Status: sketch for review. Nothing is implemented. The line counts below are estimates from reading the files, not
from a patch. Paths: `L` = land/, `SITE` = L/scratch/sglang/site/flashinfer (flashinfer 0.6.18, the copy vLLM loads),
`FORK` = L/core/attn/fork/flashinfer/trtllm_trace.py (the Python port, 1245 lines, plus trtllm_witness.py at 388).

## 0. Summary

- One patched copy of the launcher sources is compiled twice:
  - **Ordinary pass:** FlashInfer's own TVM-FFI module (`fmha_gen`). The aliases are the original types, so this build
    is the stock code.
  - **Traced pass:** a torch extension. The same aliases become `c10::SymInt`, and a handful of driver/runtime names
    are shadowed inside a `traced` namespace so they record instead of launching.
- The kernel author's diff is types only: about 190 changed lines out of about 3,700 in the five files (section 9).
  It has two small exceptions: the cost-model function and `loadKernel` get a traced body in an `#ifdef` block.
- Every branch the C++ takes on a symbolic value goes through `c10::SymInt`'s comparison operators, which guard. So the
  guards are the launcher's own conditions, exact by construction, and they belong to the op: they are raised inside
  the custom op's host, which is the fork's custom op today.
- The float cost model (`selectTileSizeQForGqaGeneration`) runs only for generation calls with `max_q > 1` (spec
  decode). Under a trace it becomes `torch.cuda._host_trace.choice` over `max_kv`, evaluated by the ordinary C++ build
  of the same function. That gives fp32 semantics exactly, and the guard is on the chosen tile.
  - Decode (q = 1), context and mixed never reach the cost model, as the attn lane found (L/core/attn/STATUS.md 16:30).
- No torch C++ change is needed.
  - The record format and the conversion to `KernelLaunch` / `TmaDescriptor` live in the extension and its Python glue.
  - The core's Python API (`KernelLaunch`, `TmaDescriptor(edits=False)`, `choice`, `record_launch`, the custom-op
    redispatch) is used as it is.
- Ordinary hot path: FlashInfer's own call runs the stock machine code. The only addition is the Python wrapper's
  dispatch-stack check, which the fork has today.
- Recommendation: the C++ path (section 10 compares it with hardening the fork).
  - It covers R1 (fp8 q/KV, head 256, page 32, 6 q heads per KV head) without new porting. The fork declines all of
    those today: it accepts only bf16, head 64/128 and page 16/64.
  - It removes drift by construction.
  - The grid and guard-region tests in section 10 should be written either way: they are the soundness test for this
    path's shim too.

## 1. What the launcher is (facts from SITE/data)

- **Bindings:** `csrc/trtllm_fmha_kernel_launcher.cu` (1051 lines) holds the TVM-FFI bindings
  `trtllm_paged_attention_{decode,context}`. They take `TensorView`, `Optional<...>` and `Variant<double, Tensor>`, read
  sizes, strides and `data_ptr`, then call `trtllm_paged_attention_launcher`, which fills `TllmGenFmhaRunnerParams` and
  calls `TllmGenFmhaRunner::run`.
- **Selection and launch:** `include/flashinfer/trtllm/fmha/fmhaKernels.cuh` (1413 lines) does selection and launch:
  - `run()` loops `selectKernel` -> `loadKernel` -> `computeCtaAndClusterConfig` for at most 4 passes.
  - `setKernelParams` builds one `KernelParams` struct by value, with embedded `CUtensorMap`s.
  - The CGA fallback calls `cuOccupancyMaxActiveClusters`.
  - The launch is `cuLaunchKernelEx(&config, func, {&kernelParams}, nullptr)` with cluster, scheduling-policy and PDL
    attributes.
  - After it come the optional separate reduction kernel (`csrc/fmhaReduction.cu`) and the LSE kernel (`lse.cuh`,
    `cudaLaunchKernelEx`).
- **Params:** `kernelParams.h` (926 lines) has `setKernelParams` and the `makeTmaShapeStride*` functions.
  `buildNdTmaDescriptor` calls the driver's `cuTensorMapEncodeTiled` with element strides 1, no interleave, L2 128B and
  no OOB fill. Those are exactly the core `TmaDescriptor`'s constants; use `edits=False`, because this is the plain
  driver encode with no CuTe DSL bit.
- **Module loading:** `loadKernel` loads cubins lazily (`getCubin` -> Python callback -> `cuModuleLoadData`,
  `cuModuleGetFunction`, `cuFuncSetAttribute`) into a per-runner cache.
- **JIT build:** `gen_trtllm_gen_fmha_module` in jit/attention/modules.py builds `fmha_gen` from
  `jit_env.FLASHINFER_CSRC_DIR` and needs the cached `flashInferMetaInfo.h` (the cubin table). `FLASHINFER_CSRC_DIR` is
  derived from the package path, not from an env var.
- **Symbolic under V-stock:**
  - Values: `q.size(0)` (sum_q), `batch_size`, `max_q_len`, `max_kv_len`, and every data pointer. Possibly
    `block_tables.size(-1)`, which is static in vLLM.
  - Static: KV cache shape and strides, heads, head dims, page size, `sm_count` and workspace size. These specialize
    (guard ==), which is right for them.

## 2. Build structure: one source, two passes

- **Local copy:** L/core/trtllm_cpp/src holds a copy of SITE/data/{csrc,include} with the patch applied, plus
  `UPSTREAM.sha256`, the hashes of the five unpatched files. The site dir and caches are never touched.
- **Ordinary pass:** FlashInfer's own JIT builds it.
  - The launch script sets `FLASHINFER_WORKSPACE_BASE` to L/core/trtllm_cpp/cache and, in-process before the first
    module gen, rebinds `jit_env.FLASHINFER_CSRC_DIR` / `FLASHINFER_INCLUDE_DIR` to the copy.
  - The cubin cache stays L/scratch/vllm/cache, read-only, with `FLASHINFER_NO_DOWNLOAD=1`.
  - The aliases are the original types. A test checks that this .so's `.text` for the launcher translation unit equals
    a build of the unpatched file (objdump diff), which shows the patch is types only.
- **Traced pass:** `torch.utils.cpp_extension.load` with nvcc, MAX_JOBS <= 16, into L/core/trtllm_cpp/build. It
  compiles the launcher sources with `-DFI_HT_TRACED` plus `ht_trace.h` (the shim) and `ht_bind.cpp` (pybind). Its
  include paths are the same as `fmha_gen`'s: the metainfo include dir and the `TLLM_GEN_FMHA_CUBIN_PATH` /
  `METAINFO_HASH` defines.
- **The two namespaces:** each patched header has `FI_HT_NS_BEGIN` / `FI_HT_NS_END` around its body and a
  pass-suffixed include guard in place of `#pragma once`. That is 3-4 lines per file. In the traced pass the headers
  are included twice:
  - once as `flashinfer::...` with the aliases as plain ints (the ordinary instantiation), used by `choice` and by
    module loading;
  - once as `flashinfer::traced::...` with the aliases as SymInt.
  - Inside `traced`, unqualified names resolve to the shim first. That is how `CUtensorMap`, `cuTensorMapEncodeTiled`,
    `CUlaunchConfig`, `cuLaunchKernelEx`, `cudaLaunchConfig_t`, `cudaLaunchKernelEx`, `AlignedAllocator`,
    `TensorView`, `Optional`, `Variant`, `get_stream` and `UpPowerOfTwo` record without a source edit.
- **Cubin callback:** the traced .so includes `cubin_loader.h`, so the glue calls `FlashInferSetCubinCallback` on it
  with FlashInfer's own Python callback, as FlashInfer does for `fmha_gen`.

## 3. Which variables become SymInt

Aliases (ordinary pass -> traced pass):
- `fi_ht::s32` / `s64` / `u64` / `sz` -> `int32_t` / `int64_t` / `uint64_t` / `size_t`, or `c10::SymInt`.
- `fi_ht::ptr<T>` -> `T*`, or `c10::SymInt`, which is a byte address, as in the core's `Recorder::data_ptr`.
- Only variables that carry a symbolic value get an alias. A size read into a plain `int` keeps compiling: the shim's
  `TensorView::size(i)` returns `int64_t` through `t.size(i)`, a specialization guard. The lines that must stay
  symbolic read `fi_ht::sym_size(t, i)`, which is `t.size(i)` in the ordinary pass. The compiler finds every remaining
  site, because `int x = c10::SymInt` does not compile.

| where | becomes SymInt | stays concrete (guarded == if read from a size) |
| --- | --- | --- |
| binding bodies (launcher.cu) | `sum_seq_q`, `batch_size`, `max_q_len`, `max_kv_len`, `max_num_blocks_per_seq`, `workspace_size`, LSE strides, every `data_ptr()` | heads, head dims, page size, KV strides, `sm_count`, dtypes, flags |
| `trtllm_paged_attention_launcher` signature | the above plus the ~20 pointer parameters | the rest |
| `TllmGenFmhaRunnerParams` (fmhaRunnerParams.h) | `mBatchSize`, `mMaxSeqLenQ`, `mMaxSeqLenKv`, `mSumOfSeqLensQ/Kv`, `mMaxNumPagesPerSeqKv`, `mNumPagesInMemPool`, ~30 pointer fields | heads, head dims, strides, enums, scales |
| `CtaLaunchParams`, `computeCtaAndClusterConfig`, `run` | `mNumCtasX/Y/Z`, `mMaxNumCtasQ/Kv`, `numCtasPerSeqQ`, `numCtasPerSeqKv`, `maxAttentionWindow`, `totalNumCtas` | `mClusterDimX` (pinned: a launch attribute is node topology), all of `TllmGenSelectKernelParams` |
| kernelParams.h | `numTokens`, `batchSize`, `numKeysVals`, the shape/stride vectors (`std::vector<fi_ht::u64>`), `partialStatsBufferSize`, `maxNumCtasQ/Kv` arguments | tile shapes, swizzle, dtype, reshape factors |
| lse.cuh / fmhaReduction.cu | `n`, `num_blocks`, `num_threads` (via the shim's `UpPowerOfTwo` = core `pow2(bit_length(n - 1))`) | |

- Integer semantics: C++ `/` and `%` truncate and `c10::SymInt`'s floor; every symbolic operand here is a size or an
  address (>= 0), so they agree.
  - int32 overflow cannot be expressed. Each recorded field gets range guards that it fits its width, as FORK's
    `require(v <= 2**(8w-1)-1)` does, so a value the ordinary build would wrap is declined, not replayed.

## 4. Recording the launches

- **`KernelParams`:** in the traced pass it is a generated proxy, `traced::KernelParams = fi_ht::Traced<KernelParams>`,
  generated from the ordinary build's DWARF by an adaptation of host_tracing/minimal/gen_proxy.py, whose header is
  checked against `offsetof` asserts.
  - Members are `Field<T, offset>` that take an int or a SymInt.
  - `params.ptrPartialO = params.ptrPartialStats + n` works because a pointer field reads back its SymInt value and
    scales by `sizeof(T)`.
  - `memset(&params, 0, sizeof(KernelParams))` is shadowed by a traced `memset` overload for the proxy (zero bytes,
    clear fields).
  - `setKernelParams`'s ~100 assignments stay verbatim.
- **`CUtensorMap`:** in `traced` it is `fi_ht::TracedTensorMap`: dtype, rank, address, shape and stride SymInts, box,
  swizzle and fill.
  - `cuTensorMapEncodeTiled(&desc, ...)` resolves to the shim's overload. That overload checks the constants (element
    strides 1, interleave none, L2 128B) and stores the operands.
  - `params.tmaQ_ = desc` records a TMA field at the member's offset.
  - The checks `buildNdTmaDescriptor` already makes (`shapes[i] >= 1`, `<= 2^32`, `strides[0] == 1`, 16-byte address
    alignment through `fi_ht::aligned(p, 16)`) become guards.
- **The launch:** `cuLaunchKernelEx(&config, func, list, nullptr)` resolves to the shim's overload on the traced
  `CUlaunchConfig`, whose grid holds SymInts.
  - `list[0]` is looked up among the live proxies by address. A live proxy registers in its constructor; an unknown
    pointer declines.
  - The overload emits one launch record:
    - the `CUfunction`, its name (`cuFuncGetName`) and the param bytes;
    - fields `(offset, width, SymInt, is_pointer)` and TMA fields;
    - grid and block (SymInt), smem, cluster dim and scheduling policy (concrete), and PDL as `programmatic`.
  - `cudaLaunchKernelEx` for the LSE kernel and the separate reduction kernel is the shim's variadic overload: per
    argument a `Param<K>` as in the core's `ATen/cuda/host_trace/Recorder.h`, with the function from
    `cudaGetFuncBySymbol` on the traced .so's own copy of the kernel.
- **Glue:** the Python glue (about 60 lines, lifted from FORK `record()`) turns each record into
  `KernelLaunch(..., packed=True)`, with the layout split into pieces around the descriptors as FORK's `PIECES` does, and
  `TmaDescriptor(..., edits=False)`. The roots come from the tensor arguments. Then it calls `tr.record_launch`.
- **Byte check:** FORK's check is kept as a switch, on by default in tests and on the first trace of each op class: the
  real binding is captured on stand-ins at the hints (`capture_kernel_nodes`) and the evaluated launch is compared byte
  for byte. That covers a shim or proxy mistake (for example a member written outside the proxy) at the traced point.

## 5. Guards and dispatch decisions

- **Comparisons:**
  - `c10::SymInt`'s `operator<, <=, ==, ...` return `bool` through `guard_bool`. So `if (numCtasPerSeqKv <= 1)`,
    `numCtasPerSeqKv <= 16` (CGA), `params.mMaxSeqLenQ > 1`, `mMaxSeqLenKv > mAttentionWindowSize` and the CGA-wave
    test each guard exactly the launcher's condition, with no edit.
  - `std::min`/`std::max` on SymInt also guard (they use `operator<`). That is exact; it adds a region boundary
    wherever the min switches arms, for example the split count capped by occupancy. A later `fi_ht::min` (one token
    per site) could turn those into expressions; that is not needed for soundness.
- **Ownership:** the C++ runs inside the custom op's host (FORK's `flashinfer_ht::trtllm_paged_attention_*`, kept as
  is), so every guard belongs to the op. A tile, split, CGA or LSE-thread flip redispatches that op and never retraces.
  The kernel hash is computed from `TllmGenSelectKernelParams`, which is concrete after the guards, so the `CUfunction`
  is fixed per variant.
- **Cluster dim:** pinned (a launch attribute is node topology). In CGA mode, `clusterDimX = numCtasPerSeqKv` pins the
  split count. That is a redispatch per split value while 1 < split <= 16, the same as FORK today. It is inherent to
  the launcher, cached, and owned by the op.
- **`cuOccupancyMaxActiveClusters`:** evaluated with the concrete function, block, smem and cluster. FORK's plan_check
  found the result independent of the grid, so the shim passes grid = cluster dims. A test asserts that independence at
  a few grids, and the comparison against `numCtasX*Y*Z` is a guard.
- **Float cost model:** the traced pass gives `selectTileSizeQForGqaGeneration` an `#ifdef FI_HT_TRACED` body of
  about 6 lines. The original body stays untouched for the ordinary pass:
  1. Pin what the model reads besides `max_kv` with `guard_int`: `batch`, `max_q`, `sum_q`. These are exact guards, not
     hint reads.
  2. Call `torch.cuda._host_trace.choice(max_kv, choose, period=64, top=None)`. `choose(kv)` runs the **ordinary
     instantiation** of the same function, on the concrete `RunnerParams` with `kv`, and returns the resulting
     `TllmGenSelectKernelParams`.
  3. Copy those (all concrete) into the traced select params.
  - The period is 64 because the decision is constant on blocks `(64k, 64(k+1)]`: every candidate's `mStepKv` is 64 or
    128, and the split count steps at multiples of `2*mStepKv`. `choice_range` asserts the hint lies in its own run.
  - This is bitwise to eager by construction: the decision is computed by the shipped fp32 code, never by a port.
  - The guard is the run of `max_kv` choosing the same tile, so only a tile change redispatches.
  - Pinning batch makes spec decode redispatch per batch size. That is out of scope here; a 2-D choice would remove it.

## 6. Module loading outside capture

- **Sharing:** the traced pass's `loadKernel` body (`#ifdef`, about 5 lines) delegates to the ordinary instantiation
  in the same .so, keyed by the concrete hash. The traced and ordinary passes share one `CUmodule`/`CUfunction` cache.
- **Preload at warm-up:** the glue's `preload(kind, args)`, called at the warm-up's eager call (FORK `traced()` already
  does this; the attn lane's witness change makes the warm-up descend the custom op), runs a new ordinary-pass method.
  - The method loads every `mKernelMetaMap` entry that matches the call's static class (layout, head dims, page key,
    sparse, skip-softmax, fp16-softmax, spcompress, transform mode), across kernel type, tile Q/KV, scheduler,
    multi-CTA mode, mask and `headDimPerCtaV`.
  - That is tens of cubins, and covers any later redispatch.
  - It also fills the occupancy cache for the CGA kernels.
  - It is a public method in the same `#ifdef` area: about 15 lines.
- **Unloaded under capture:** if a trace or redispatch still reaches an unloaded kernel while the thread is capturing
  (`cudaStreamGetCaptureInfo` on the current stream), the op declines with "kernel X not loaded (a capture holds)" and
  queues X. The next eager warm-up or call loads it. That is a decline counted in `retrace_causes`, never a silent
  eager step, and the tests assert it is 0 after preload.

## 7. Binding layer

- **Why not TVM-FFI:** it converts Python ints and cannot carry a SymInt.
- **The extension:** the traced pass is a pybind torch extension. Its functions are
  `trace_decode(*args) -> records`, `trace_context(*args) -> records` and `preload(...)`.
  - Their signatures mirror the bindings: `at::Tensor` for tensors (traced tensors pass as they do into
    torch/csrc/cuda/host_trace/Aten.cpp), `c10::SymInt` for FORK `_SPECS`' `S` fields (pybind's SymInt caster),
    `double`, `bool`, `std::optional`.
  - The binding bodies stay verbatim. In `traced`:
    - `TensorView` is a shim over `at::Tensor` (`size`, `stride`, `ndim`, `dtype` as DLDataType, `data_ptr` -> SymInt
      through a Python recorder like Aten.cpp's `PyRecorder`);
    - `Optional` is `std::optional`;
    - `Variant` has `.as<T>()`;
    - `get_stream` returns the trace's stream.
- **Custom op:** FORK's `traced(binding, kind)` / `_op(kind)` (about 100 lines) is reused. Its kernel calls
  `ext.trace_<kind>(*args)` in place of FORK's `record()`, and the glue converts the records (section 4). FlashInfer's
  Python change is FORK's: 2 lines at each of the 2 call sites (decode.py:3420, prefill.py:5553).
- **Drift guard for the patch itself:** at import the glue compares `UPSTREAM.sha256` with the five site files. On a
  mismatch the traced path is off, so calls take the shim's harvest or eager route with a named reason, and a test
  fails.

## 8. vLLM hook and the runs (step 2, after review)

- **Env flag:** `ARMV_TRTLLM_CPP=1` in a copy of L/core/attn/armV (mirroring `ARMV_TRTLLM_FORK`). It puts the
  L/core/trtllm_cpp/flashinfer overlay (symlinks plus the two hooked files and the glue) ahead of the site.
- **Launcher:** `python_vllm_trtllm_cpp.sh` on the pinned build plus the attn lane's py, which has `choice` and the
  witness change.
- **Qwen3-8B V-stock:** drive `--check` for decode 1/8/64, prefill 64/512/2048, 4x128, 8x128 and mixed 32x512 / 64x64.
  The bar:
  - bitwise vs stock eager (stock `max_seq_len`);
  - report traces, variants, redispatches and `retrace_causes`;
  - 0 eager attention steps.
- **R1:** `/data/eellison/models/Qwen3.8-27B-NVFP4`, `--kv-cache-dtype fp8`, 16 full-attention layers, fp8 q,
  head 256, sm103.
  - The cubins were approved for download according to serve/STATUS.md (19:08 entry; batch phaseB3), and must be listed before
    any run. With `FLASHINFER_NO_DOWNLOAD=1`, a missing one fails loudly; then I stop and report.
  - Bar: 0 eager attention steps, and attention output bitwise vs stock eager on the same metadata (NaN-aware, because
    of serve/STATUS (iii)).
  - Whole-step bitwise is not claimed: R1's other open gaps (float8 harvest refill, the mm_fp4 DataPointer) are
    outside this lane.
- **GPU use:** GPU 1 under the gpu1.lock flock (CPUs 72-107) for checks; GPU 0 (gpu0.lock) for short timing holds.

## 9. Expected diff and effort (C++ path)

| file | lines | changed (estimate) | what |
| --- | --- | --- | --- |
| csrc/trtllm_fmha_kernel_launcher.cu (decode, context, paged launcher only; ragged/MLA under `#ifndef FI_HT_TRACED`) | 1051 | ~45 | aliases, ~4 `sym_size`, 1 `aligned` |
| fmhaRunnerParams.h | 465 | ~35 | field types |
| fmhaKernels.cuh | 1413 | ~45 | locals and `CtaLaunchParams` types (~30); `#ifdef` bodies for the cost model, `loadKernel` and preload (~15 added) |
| kernelParams.h | 926 | ~50 | shape/stride vector types and casts |
| lse.cuh, fmhaReduction.cu | 512 | ~10 | `n`/grid types |
| namespace and guard lines | | ~20 | 4 per file |
| **total** | **~3,700** | **~190 (about 5%)** | types only, plus the three `#ifdef` bodies |

New code, not in the author's files:
- `ht_trace.h` shim: about 350 lines.
- Proxy generator: about 150 lines, plus the generated header.
- `ht_bind.cpp`: about 120 lines.
- Python glue: about 200 lines, mostly FORK's custom op and records conversion.
- Build script: about 80 lines.
- Tests: about 250 lines.

Effort is about 4-5 days to Qwen3-8B decode/prefill/mixed bitwise, plus R1. Where it goes:
- shim, proxy and record: 1.5 days;
- patch and two-pass build: 1 day;
- glue and preload: 0.5 day;
- Qwen3-8B runs and fixes: 1 day;
- R1: 1 day;
- tests: 0.5 day.

Risks:
- c10 headers under nvcc in the launcher's translation unit. Torch extensions do this routinely, but `fmha_gen`'s flag
  set is unusual.
- Name shadowing missing an unqualified call that is qualified somewhere (for example `::cuLaunchKernelEx`). The byte
  check plus a "no real launch during trace" assertion catch it: the shim counts real driver launches.
- Trace and redispatch cost: each C++ SymInt op calls into the Python IR, roughly a few hundred ops per call. This is
  untested; I expect about the fork's cost. The attn lane's 36-layer redo cost is a separate core item.

## 10. Compared with making the Python fork sound and maintainable

What the fork already has:
- A per-trace byte check against the real binding at the hints.
- A concrete sweep, armF/fork/xcheck_params.py, over a few dozen decode/context configs.

What it lacks:
- **Guard soundness:** nothing checks that, at a replay point inside the port's guards, the C++ would make the same
  choice and fields.
- **Drift detection:** nothing ties the port to a FlashInfer version.
- **Coverage:** bf16 only, head 64/128, page 16/64, causal, no separate-reduction kernel. R1 is fp8/fp8/bf16, head
  256, page 32, with 6 q heads per KV head, so it needs new porting (fp8 TMA dtypes, swizzle from leading-dim bytes,
  the q-heads-per-KV grouping) and a new cross-check.
- **Port details:** the port parses the metainfo header with a regex (`_parse_meta`) and emulates fp32 in Python
  (`_f32`) for the cost model.

What would make the fork sound and maintainable:
1. **Version pin:** store sha256 of the five mirrored files and of `flashInferMetaInfo.h`. On a mismatch the fork is
   off (harvest or eager with a named reason) and a test fails. About 0.25 day.
2. **Grid differential test**, extending xcheck_params into a test.
   - For each config class (dtype triple, head dim, page, q heads per KV head, LSE, sinks), sweep batch 1-256,
     `max_kv` densely around the boundaries (multiples of 64/128/256 +-1, the occupancy cap `sm / numCtas`, split 16/17),
     `max_q` 1-4 and context `sum_q`/`max_q`.
   - Compare the port's concrete plan with the captured node: function, grid, block, smem, cluster, param bytes with
     pointers masked, and TMA bytes.
   - About 1 day. It runs in seconds per class on sm100/103 with the cubins.
3. **Guard-region test** (the soundness one, and it applies to the C++ path unchanged):
   - Trace at A.
   - For each grid point B where the variant's op guards hold, evaluate the variant's recorded launch at B with the
     program and compare it byte for byte with the C++ binding captured at B.
   - Also assert that some B outside the guards changes the C++ choice, so the guards are not vacuous.
   - About 1 day.
4. **R1 port:** fp8 q/KV, head 256, page 32 and 6 q heads per KV head, then rerun 2 and 3. About 1-2 days.
5. **Upkeep:** every FlashInfer bump that touches these files (the BF16Q+FP8KV split cap and the SM107 oversized-smem
   path are recent examples of heuristic churn) means a re-port and a re-run of 2 and 3. That cost recurs.

| | C++ (this design) | hardened Python fork |
| --- | --- | --- |
| drift from FlashInfer | none by construction: same source, re-apply a types-only patch on upgrade (rejects or compile errors flag it) | a re-port per upgrade, caught by the hash pin and the grid tests |
| guard exactness | the C++ conditions themselves (SymInt comparisons) | the port's conditions, checked on a grid |
| cost-model bitwise | runs the shipped fp32 code | Python fp32 emulation, checked on a grid |
| config coverage | everything the launcher handles (fp8, head 256, LSE, sinks, separate reduction) | bf16/head 64,128 today; each new class is port plus check |
| needs | a nvcc JIT build (minutes), a shim and a proxy generator | nothing new to build |
| one-time effort | about 4-5 days | about 2.5 days (1-3) plus 1-2 days for R1 |
| recurring | a mechanical patch rebase | a logic re-port |

Either way, write tests 2 and 3: on the C++ path they check the shim and proxy at points other than the hints. My
recommendation is the C++ path, with the fork kept as the fallback route until it lands.

## 11. Open questions for review

- OK to build FlashInfer's ordinary `fmha_gen` from the patched copy (aliases are the plain ints)? The objdump check
  shows it is the stock code. The alternative is keeping the stock module for ordinary calls and using the patch only
  in the traced pass. That is also sound, but the two builds then come from two files.
- OK to pin batch / max_q / sum_q on the spec-decode cost-model path for now (section 5), with a 2-D choice as a
  follow-up?
- The CGA redispatch per split value is the launcher's own behaviour. Is that acceptable as is? The alternative is a
  core item, such as conditional nodes or one variant per cluster dim, and it is not part of this lane.
