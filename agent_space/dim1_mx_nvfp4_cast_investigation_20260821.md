# Dim1 (columnwise) MXFP8 / NVFP4 cast investigation

Date: 2026-08-21

## Scope and environment

How Inductor handles dim1 (columnwise) quantization casts today, whether the
nested-reduction/sub-parent stack engages, and how performance compares to the
dim0 (rowwise) orientation and to torchao's dedicated dim1 Triton kernel.

- Stack tip `3f62ae27baa` ([inductor] Deduplicate nested reduction test
  helpers, resubmitted 2026-08-21), worktree `/tmp/dim1_wt` (removed after this
  run), executed via the `agent_space/run_wt.py` full-package overlay harness
  on installed torch `2.15.0a0+git7c78410`, Triton 3.8.0, NVIDIA B200,
  `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`.
- torchao 0.13.0 (`/data/users/eellison/ao`) is importable; its Triton kernels
  work but its CUDA extension still fails to load (same as the 2026-08-14
  note), so `torchao::mxfp8_quantize` CUDA is not benchmarkable here.
- torchao has NO dedicated NVFP4 dim1 kernel (only rowwise
  `NVFP4Tensor.to_nvfp4`); the NVFP4 dim1 baseline is therefore the plain-torch
  reference compiled by Inductor, with torchao's MXFP8 dim1 kernel as the
  nearest dedicated-kernel yardstick.
- Scripts: `agent_space/bench_dim1_quant_casts.py` (workloads + bench) and
  `agent_space/bench_dim1_planner_probe.py` (planner decline probe).
  Timings are `triton.testing.do_bench` medians (wall); per-kernel times are
  `torch.profiler` CUDA self-times over 20 iterations.

## Workload definitions

`x` is `[M, K]` bf16 row-major on cuda. "dim1" means scale groups along dim0:
one group is 32 (MXFP8) or 16 (NVFP4) consecutive rows of one column.

| name | semantics | payload layout | scale layout |
|---|---|---|---|
| mxfp8_dim0 | `to_mx(x, e4m3, 32, RCEIL)` | `[M, K]` row-major fp8 | `[M, K/32]` e8m0 |
| mxfp8_dim1_t | torchao dim1: `to_mx(x.t().contiguous(), ...)`, return `data.t()` | `[K, M]` contiguous (a `[M, K]` col-major view) | `[K, M/32]` e8m0 |
| mxfp8_dim1_rm | same groups, natural output | `[M, K]` row-major | `[M/32, K]` |
| nvfp4_dim0 | stack flagship: 16-groups along K, fp8 scale `(amax/6).clamp(1e-12, 448)`, adjacent COLUMN pairs packed per byte via `cvt.rn.satfinite.e2m1x2.f32` | `[M, K/2]` u8 | `[M, K/16]` fp8 |
| nvfp4_dim1_rm | transposed twin: 16-groups along M, adjacent ROW pairs packed per byte | `[M/2, K]` row-major u8 | `[M/16, K]` fp8 |
| nvfp4_dim1_t | same, transposed outputs for a GEMM consumer | `[K, M/2]` contiguous | `[K, M/16]` |
| ao_mxfp8_dim1 | `torchao ... triton_to_mxfp8_dim1(x, 32, "rceil")` | `[K, M]` contiguous | `[K, M/32]` |
| ao_mxfp8_dim0 | `triton_to_mxfp8_dim0` | `[M, K]` | `[M, K/32]` |

Correctness: for every workload/shape, nested ON and OFF outputs are bitwise
identical. Compiled `mxfp8_dim1_t` is bitwise identical to torchao's dedicated
dim1 Triton kernel at all three shapes (0 mismatched payload or scale bytes,
checked up to 117.4M payload bytes at 16384x7168).

## Q1: what Inductor emits today

`triton.nested_reduction` ON vs OFF changes NOTHING for any dim1 workload:
same kernel count, same kernels, `codegen_nested_reduction == 0` everywhere.
The staged stack engages only for `nvfp4_dim0` (2 kernels -> 1, `staged=1`).

Kernel inventory at 16384x7168 (identical structure at other shapes):

- **mxfp8_dim0**: ONE generic persistent-reduction kernel (amax + e8m0 scale +
  quantize + both stores), fully contiguous: `tl.load(in_ptr0 + (r0_1 + 32*x0))`.
  The staged stack never applies to MXFP8 (full-resolution epilogue, no packing
  rate) and is not needed.
- **mxfp8_dim1_t**: ONE kernel — generic fusion succeeds. This CONTRADICTS the
  2026-08-14 note (`agent_space/mxfp8_dim1_triton_investigation.md`), which saw
  two kernels on an older checkout; at this tip the reduction and quantize fuse.
  It is a y/x/r-tiled persistent reduction, `size_hints={'y': 8192, 'x': 512,
  'r0_': 32}` (y = K columns, x = M/32 groups, r = 32 rows):

  ```python
  tmp0 = tl.load(in_ptr0 + (y0 + 7168*r0_2 + 229376*x1), ...)   # K-strided over r, contiguous over y
  tl.store(out_ptr1 + (x1 + 512*y0), tmp13, xmask)              # e8m0 scale [K, M/32]
  tl.store(out_ptr2 + (r0_2 + 32*x1 + 16384*y0), tmp28, xmask)  # fp8 payload [K, M] contiguous
  ```

  Loads are coalesced along the y (column) axis of the tile; the store is
  contiguous along (x, r) — a transposed layout relative to the load tile, so
  Triton inserts a register/shared-memory layout conversion (the implicit
  `tl.trans`). No input re-read, no intermediates: traffic is the 356 MB ideal.
- **mxfp8_dim1_rm**: TWO kernels. K1: 1D-x persistent reduction, load
  `7168*r0_1 + 229376*(x0 // 7168) + (x0 % 7168)` (K-strided over r, coalesced
  over x), writes only the 3.7 MB u8 scale. K2: fully contiguous pointwise
  (`tl.load(in_ptr0 + x3)` — output order equals input order), re-reads the
  input, broadcast-reads the scale, writes the payload. Traffic 595 MB = 1.67x
  ideal. The fusion miss is the SwiGLU colwise loop-order class: the reduction
  iterates (group, k) with r last, while the quantize node's stride-descending
  order is (group, r, k) with r in the middle — the same op4/op6 mismatch
  documented in `agent_space/swiglu_mxfp8_two_kernel_milestone.md`. The reason
  dim1_t DOES fuse is that its transposed outputs make the epilogue's natural
  order (k, group, r), which matches the reduction's normalized (x=(k, group),
  r) loop exactly.
- **nvfp4_dim1_rm**: TWO kernels. K1: 1D-x persistent reduction over r=16
  (K-strided), writes only the 7.3 MB fp8 scale. K2: pointwise packing, reads
  row pairs `x0 + 14336*x3` and `7168 + x0 + 14336*x3` (coalesced along K),
  broadcast-reads the scale, contiguous payload store. Traffic 543 MB = 1.80x
  the 301 MB ideal.
- **nvfp4_dim1_t**: TWO kernels plus an extra intermediate. K1 stores BOTH the
  transposed fp8 scale `[K, M/16]` (a transposed store from a y=M/16, x=K tile)
  AND a 29.4 MB fp32 row-major scale temp for K2 to read coalesced. K2 packs and
  stores the payload transposed (`out_ptr0 + (y0 + 8192*x1)`). Traffic 595 MB =
  1.98x ideal.

## Q2: where exactly the staged stack declines for dim1

Evidence from `bench_dim1_planner_probe.py` (monkeypatched planner entry
points, no source edits), nvfp4_dim1_rm at 4096x4096, nested ON. Line numbers
are worktree state at `3f62ae27baa`.

**Standalone path** (`NestedReduction.sub_parent_epilogue_plan`,
`torch/_inductor/scheduler.py:650`):

1. Candidate detection PASSES. `_sub_parent_epilogue_candidate_nodes`
   (scheduler.py:840) is numeric (numel ratios + `SIMDKernel.is_compatible`),
   and the packing node's `M*K/2` elements admit `factor=2`:

   ```text
   PROBE   candidates -> factor=2 groups=[(1, ['op2'])]
   ```

2. The lane proof FAILS. `_try_get_sub_parent_source_layouts`
   (scheduler.py:1061) normalizes epilogue reads into an `(x, child_r)` domain
   where `child_r` is the TRAILING extent of the flattened iteration space.
   For dim1 the trailing axis is K (contiguous columns); the true lanes are
   along M at stride K. The probe shows the projections:

   ```text
   PROBE interleaved_lane(index=_sub_parent_child_r + 8192*((_sub_parent_x//512))
                          + 8*(ModularIndexing(_sub_parent_x, 1, 512)), factor=2)
         -> Mod(_sub_parent_child_r, 2)          # NOT statically 0 or 1
   PROBE contiguous_lane(...) -> 0               # constant, but the reconstructed
                                                 # parent index mismatches
   PROBE   source_layouts(sources=['arg0_1'], factor=2) -> None
   PROBE plan ... nodes=[('op0', '(1048576, 16)'), ('op2', '(8388608, 1)')] -> None
   ```

   In that normalization `child_r` carries the low 3 bits of k, so
   `Mod(child_r, 2)` is data-position dependent, not a constant lane; the
   interleaved match fails. The contiguous lane comes out 0, but the expected
   index `parent_index.subs(parent_r, child_r + lane*child_rnumel)` cannot
   equal a child index whose lane lives at stride K, so that match fails too.
   `None` propagates up; the pair is REJECTED from the reduction group by
   `_sub_parent_epilogue_decision` (`torch/_inductor/codegen/simd.py:2768`)
   because the packing node is sub-parent shaped (`_is_sub_parent_shaped`,
   simd.py:2743). The tiling gate `_sub_parent_tiling_is_2d` (simd.py:2884) is
   NEVER REACHED — the decline is entirely in the index-based lane proof.

3. Generic fusion also cannot take the pair. `TORCH_LOGS=fusion`:

   ```text
   cannot fuse op0 with op2: no shared data
   cannot fuse op0_op1 with op2: no shared data
   ```

   `op2`'s scale read (broadcast over 16 lanes, different var count) has no
   exact `MemoryDep` intersection with `op1`'s write — the same set-intersection
   scoring miss as SwiGLU blocker 1 — and even past scoring, a pointwise at
   `numel*rnumel/2` fits no generic reduction group. The sub-parent stack is
   the ONLY fusion home for packing epilogues, and it declines dim1.

**Nested path**: not exercised by these standalone casts (no dependent
outer-reduction -> grouped-reduction pair). For a hypothetical nested dim1
form, the grouped axis lands in X and `_nested_sub_parent_rate` rejects it
up front (`scheduler.py:943`, `if domain_context.grouped_axis is not
cls.GroupedAxis.R: return None`), pinned by
`test_producer_consumer_rejects_sub_parent_grouped_axis_x`
(test/inductor/test_nested_reduction.py:1504).

**MXFP8 dim1** never reaches a lane proof at all: the quantize epilogue is
full-resolution, so `candidates -> None` for every grouping and the decision is
DEFER (not REJECT); generic fusion then handles it (successfully for dim1_t,
2 kernels for dim1_rm as above).

## Q3: performance

Wall-clock `do_bench` medians (us). `+cd` = `coordinate_descent_tuning=True`.
Effective TB/s = ideal bytes (input + outputs only) / wall time; B200 HBM
ceiling ~8 TB/s. Nested ON/OFF timings are equal for all dim1 rows (shown once).

**MXFP8** (ideal traffic: 3.03 bytes/elem):

| shape | Ind dim0 | Ind dim1_t | Ind dim1_t +cd | Ind dim1_rm | Ind dim1_rm +cd | AO dim0 | AO dim1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024x1024 | 12.6 | 11.9 | - | 14.7 | - | 43.1 | 36.3 |
| 4096x4096 | 14.4 | 34.9 | 20.5 | 31.8 | - | 43.7 | 37.9 |
| 16384x7168 | 60.4 (58.4 +cd) | 208.9 | **88.1** | 168.9 | 105.5 | 70.7 | 95.4 |

**NVFP4** (ideal traffic: 2.5625 bytes/elem):

| shape | dim0 nested off | dim0 staged | dim0 staged +cd | dim1_rm | dim1_rm +cd | dim1_t | dim1_t +cd |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024x1024 | 11.3 | 7.3 | - | 11.3 | - | 11.4 | - |
| 4096x4096 | 25.6 | 15.4 | 16.4 | 37.9 | 23.6 | 44.0 | 40.0 |
| 16384x7168 | 142.3 | 70.7 | 64.5 | 201.6 | 99.4 | 230.4 | 207.9 |

Achieved bandwidth at 16384x7168, ACTUAL traffic / profiler kernel time:

| kernel | actual MB | us | TB/s |
|---|---:|---:|---:|
| mxfp8_dim0 (1 kernel, +cd) | 356 | 53.2 | 6.69 |
| mxfp8_dim1_t +cd (1 kernel) | 356 | 85.8 | 4.15 |
| ao_mxfp8_dim1 (1 kernel) | 356 | ~95 | 3.73 |
| mxfp8_dim1_rm +cd K1 / K2 | 239 / 356 | 35.0 / 62.5 | 6.82 / 5.70 |
| nvfp4_dim0 staged +cd (1 kernel) | 301 | 61.7 | 4.88 |
| nvfp4_dim1_rm +cd K1 / K2 | 242 / 301 | 34.5 / 49.7 | 7.02 / 6.06 |
| nvfp4_dim1_t +cd K1 / K2 | 272 / 323 | 132.1 / 64.4 | 2.06 / 5.02 |

Cross-references to the SwiGLU investigation numbers: Inductor's cd-tuned dim1
MXFP8 single kernel sustains 4.15 TB/s INCLUDING the transposed fp8 payload
store — essentially the CuTe #4743 shared-tile kernel's 4.08 TB/s, and above
AO's dedicated dim1 Triton kernel (3.73). Plain strided-load reduction kernels
reach 6.8-7.0 TB/s under cd, nearly matching the fully contiguous rowwise
kernels (6.7).

## Q4: diagnosis

The two formats fail differently, and coordinate-descent tuning changes the
default-config picture drastically:

1. **MXFP8 dim1 (torchao layout) is neither a fusion problem nor a traffic
   problem at this tip — it is a default-config problem.** One kernel, ideal
   traffic, bitwise-equal to AO. Default tiling config runs it at 1.70 TB/s;
   cd tuning takes the SAME kernel source to 4.15 TB/s (208.9 -> 88.1 us),
   beating AO's dedicated kernel at every tested shape (36.3->11.9 at 1024,
   37.9->20.5 at 4096, 95.4->88.1 at 16384x7168). This refines the SwiGLU-era
   conclusion that a shared-memory transpose pipeline is REQUIRED for
   competitive transposed stores: Triton's compiler-inserted layout conversion
   already delivers CuTe-class bandwidth for this cast — but only under a good
   launch config, echoing the milestone doc's "coordinate-descent tuning is
   required" finding for the merged colwise kernel.
2. **MXFP8 dim1_rm is the known loop-order fusion miss** (reduction wants r
   last, epilogue puts it in the middle). Two kernels, 1.67x traffic floor,
   and with cd both kernels run near ceiling, so the residual 105.5 vs ~88 us
   is almost exactly that floor. This is squarely the
   `swiglu_2kernel_proto.patch` class of fix, not a stack issue.
3. **NVFP4 dim1 is a fusion problem, and the staged stack is the only
   candidate fixer.** The lane proof declines (Q2), generic fusion cannot host
   a fractional-numel packing node, so the schedule is 2 kernels reading the
   input twice: 543 MB vs 301 MB ideal (1.80x). Under cd both kernels are
   near-ceiling (7.0 / 6.1 TB/s) — the access pattern is NOT the problem; the
   double read is. Result: 99.4 us vs the staged dim0's 64.5 us (1.54x), where
   a fused single kernel at the dim0 kernel's 4.9 TB/s would take ~62 us.
4. **NVFP4 dim1_t is both problems at once**: the unfusable packing plus a
   reduction kernel stuck at 2.06 TB/s even under cd (it must emit a transposed
   scale store and additionally materializes a 29 MB fp32 row-major scale temp).
   cd recovered the mxfp8_dim1_t kernel but not this one — the good config is
   evidently not reachable from cd's starting point here, underlining that the
   default-tiling gap is not reliably closed by tuning.

## Q5: what it would take to be competitive on dim1

- **(a) Nothing / already fine — TRUE for MXFP8 dim1 in the torchao layout,**
  with the big caveat that it needs `coordinate_descent_tuning` (or a fixed
  default config) at large shapes. The concrete follow-up is the one the
  milestone doc already names: fix the default tiling/launch-config heuristic
  for tiled persistent reductions with transposed stores so the win does not
  depend on cd tuning. That is orthogonal to the landing stack.
- **(b) Grouped-axis-X / strided-lane support in the staged stack — what NVFP4
  dim1 fusion actually requires.** Concretely: `_try_get_sub_parent_source_layouts`
  would need to prove lanes on a non-trailing axis (normalize to
  `(x_hi, lane, x_lo)` with the lane at stride K), and sub-parent codegen would
  need lane-split stores along an X-derived axis rather than splitting the R
  tree. That is new planner capability plus new codegen capability; per the
  standing recommendation, do NOT fold it into the current landing stack. If
  NVFP4 dim1 matters in practice, it is a self-contained follow-up with a
  clear ~1.5x (99 -> ~65 us) payoff at large shapes.
- **(c) The blockwise-2D shared-tile mix-order mode** (torchao #4743 follow-up)
  targets the dim0+dim1 BOTH-direction case (sibling orthogonal reductions over
  one tile). Single-direction dim1 does not need it — there is no shared-tile
  saving with only one direction — but note that once (c) exists for MXFP8
  32x32, its colwise half is exactly the mxfp8_dim1 pattern, so (c) subsumes
  the MXFP8-side concerns; it would still not cover NVFP4 pair-packing without
  (b)-style lane machinery.
- **(d) A transpose-aware tiling/smem story**: today's evidence says Triton's
  existing layout conversion is sufficient for CuTe-class bandwidth on B200
  (4.15 TB/s including the transposed store) WHEN the config is right, so a
  dedicated smem pipeline is not the near-term lever; the lever is (a)'s
  config heuristic. The nvfp4_dim1_t reduction kernel (2.06 TB/s under cd) is
  the counterexample worth a targeted look if transposed NVFP4 scale layouts
  become a real workload.

Priority: (a)'s default-config fix first (it flips MXFP8 dim1 from 2.2x slower
than AO to faster-than-AO with no scheduler change), then (c) as already
planned for the combined case, with (b) only if single-direction NVFP4 dim1
shows up in a real model.

## Contradictions with prior notes

1. `mxfp8_dim1_triton_investigation.md` (2026-08-14): "compile decomposed dim1
   = two generated kernels" — at this stack tip the torchao-layout dim1 cast
   compiles to ONE kernel (generic fusion, nested irrelevant). Its hypothesis
   that the kernel "likely needs an AO-style 2D tile load + tl.trans lowering,
   not the current strided logical transpose lowering" is half-resolved: the
   emitted kernel IS the 2D-tile form, and with cd tuning it beats AO.
2. The SwiGLU-era inference that competitive transposed stores require an
   smem-transpose pipeline (from CuTe's 4.08 TB/s) is too strong for pure
   casts: generic Triton codegen reaches 4.15 TB/s with a transposed fp8 store
   under cd tuning.
3. The old note's AO-dim1-vs-compile gap (344 vs 544-577 us at 16384x16384) no
   longer describes the current tip: at 16384x7168 it is 95.4 vs 88.1 us in
   Inductor's favor (with cd).

## Raw artifacts

- Sweep log: `/tmp/dim1_sweep.log` (transient; tables above are complete).
- Kernel dumps were under `/tmp/dim1_wt/agent_space/dim1_dumps/` (worktree
  removed; the load/store excerpts above are the load-bearing parts).
- Probe/fusion evidence reproducible via
  `agent_space/bench_dim1_planner_probe.py --workload nvfp4_dim1_rm`.
