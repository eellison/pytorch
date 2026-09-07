---
name: to-padded-blocked-prim
description: "The to_padded_blocked inductor prim (PR #195930): why it is named that, why it is not redundant with flex_gemm::to_blocked, and that it emits no shape guards"
metadata:
  node_type: memory
  type: project
  modified: 2026-09-03
---

`to_padded_blocked` (schema `inductor_to_padded_blocked`) lives in
`torch/_inductor/inductor_prims.py`, lowers in `torch/_inductor/lowering.py`,
and is tested by `test/inductor/test_to_padded_blocked.py`. Submitted
2026-09-03 as PR #195930, opened as a DRAFT on top of #195874.

Naming was deliberated with subagents 2026-09-03 and is settled -- do not
re-litigate. `scale` was dropped because in ATen `scale` means a multiplicative
factor being applied, and the op is dtype- and semantics-agnostic. `scatter`
was dropped because it names an implementation the test explicitly forbids
(`assertNotIn("index_put", ...)`); the lowering is a *gather* -- one
`Pointwise` over `[output_numel]` where each output element inverts the layout
to find its source, `ops.masked`-ing to `padding_value` when there is none.
`swizzle` was rejected on evidence: it is never a function name anywhere in
`torch/` and is already overloaded twice inside Inductor
(`cutlass_max_profiling_swizzle_options`, `CU_TENSOR_MAP_SWIZZLE_128B`).
The `inductor_` schema prefix is mandatory, not cosmetic: `make_prim` registers
into the *global* `prims::` namespace, so an unprefixed name squats there --
compare `inductor_force_stride_order`, `inductor_cvt_e8m0_rceil`.

It is NOT redundant with the pre-existing `flex_gemm::to_blocked`
(`torch/_higher_order_ops/flex_gemm.py:189`, special-cased at
`gemm_epilogue.py:477`). A sweep over every legal parameter combination of the
new prim against `to_blocked` on 256x8 reproduced none of it. Three structural
differences: `to_blocked` keeps all 128 rows in one block while this op puts
`row_outer` above `col_outer`; `to_blocked` has no logical/physical row
distinction (here `logical_row_chunk=96` expands to `physical_row_chunk=128`,
an intrinsic 1.375x); and `to_blocked` only zero-fills trailing tiles, whereas
here padding is *interior* and takes a caller-supplied value. Note that
`to_blocked` does already pad -- "padded" alone is not what distinguishes them.

The op emits zero shape guards of its own, which was an explicit user
requirement ("we also shouldnt have guards ... only statically know ntrue").
One compile serves shapes differing in both magnitude and chunk-divisibility;
`replacements == {}` and `deferred_runtime_asserts == []`. The only surviving
guards are Inductor's universal per-buffer `numel <= 2147483647` 32-bit
indexing bound, which a bare `x + 1` emits identically. This is locked in by
`test_dynamic_shapes_emit_no_layout_guards` and was validated against a
negative control (adding a `sizevars.check_leq` makes it fail).

Perf at 2048x3072: no-swizzle floor 8.5us, this pointwise fallback 10.2us
static / 10.3-10.4us dynamic, versus 23.9us for the `zeros` +
`_unsafe_index_put` baseline and 23.867us for a hand-written Triton kernel.
Still outstanding before it leaves draft: the production DCN graph end-to-end
run. Every number so far is the extracted quantize tail only, and that graph
is the real acceptance test -- see [[nested-reduction-stack]].

## 2026-09-03: post-mutation-fix verdict (SETTLED)

Once [[sub-parent-mutation-hoisting-fix]] landed, all four formulations of this
layout converge: **2 kernels and byte-identical output** (sha256-checked) on
mxfp6_quant / rmsnorm_mxfp6 / dcn_mxfp6 at 2048x3072. Best-of-tuning us:

| workload | fpad | prim | fill_scatter |
|---|---|---|---|
| mxfp6_quant | 8.94 | 8.78 | 9.08 |
| rmsnorm_mxfp6 | 13.69 | 13.43 | 12.95 |
| dcn_mxfp6 | 9.60 | 9.20 | 8.46 |

So the PR's headline claim -- 23.9us scatter vs 10.2us prim against an 8.5us
floor -- is **dead**. Post-fix the scatter forms sit at the floor. The prim wins
nothing on perf; it is a wash, and `fpad` (plain F.pad + view + permute, no new
op at all) is only 2-6% behind.

**Tried and rejected: lowering the prim source-iterating** (ir.Scatter into a
`full` prefill, so it fuses like the hand-written scatter). Correct -- identical
sha256, and all 15 existing tests pass unchanged including the no-guard test and
the "index_put never reaches codegen" assertion. But the perf split is clean and
tuning-dependent: **+2.0/+2.3/+4.6% worse at default tuning, -3.9/-4.1/-4.9%
better under coordesc**, on all three workloads. Not a free win, so not worth the
churn. Keep in the back pocket if coordesc ever becomes the default.

**What actually justifies the prim now**: not fusion, not perf. It is (1) the
single validated definition of the layout, eager and compiled agreeing via
check_padded_blocked_layout -- unvalidated params silently corrupt rather than
raise; and (2) dynamic shapes, where `fpad` emits 5 guards and raises
ConstraintViolationError on all four shapes while the prim compiles once with 0
guards. `fill_scatter` also scores 0 guards, so against *that* baseline the only
remaining argument is the abstraction boundary.

The commit message must be rewritten before this lands: paragraph 1's causal
fusion claim and the whole perf paragraph are false, and the naming paragraph's
"the lowering is a gather" reasoning is now load-bearing in a way it should not
be. Preserve `ghstack-source-id` and `Pull-Request` trailers, re-read from HEAD.

Harness: the `prim` variant now exists in
`agent_space/padding_matrix_20260903/bench_worker.py` (`xdl_prim` in
XDL_VARIANTS); raw cells in `results_prim_cmp.jsonl` and
`results_prim_scatter.jsonl`.
