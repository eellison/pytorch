# Review pass 4: PR #191974 simplified fix, temporal axis

Date: 2026-09-01. Target: `agent_space/pr191974_simplified_wt` at commit
`61b333901fa` (ordering fix + template gate committed; clean tree).

This is an AI-assisted local review document. It is not intended to be pasted
into GitHub without human review and the disclosure required by `AI_POLICY.md`.

## Verdict

- **Silent wrong results: none found.** The stage-ordering fix is sound under
  the scheduler's dependency model (verified below and by an independent static
  attack), and a ~600-run battery built entirely around multi-reduction and
  order-hazard topologies produced zero fused-kernel mismatches.
- **Resolved in the local follow-up.** Two loud compile failures
  (`InductorError: AssertionError`) reproduce on mainstream graphs that compile
  today with nested reduction off. Both were caught by the fail-loud tripwires
  this stack insisted on, so they are compile failures rather than wrong
  numerics. Both fixes and their validation are recorded below.

## Resolution

Both blockers are fixed in the uncommitted follow-up on top of `61b333901fa`.

- Nested fused nodes are excluded from `Scheduler.merge_loops`, preserving the
  loop bodies on which fusion-time planning was approved. Standalone staged
  reductions retain their existing merge behavior.
- A nested sub-parent stage is declined when an exact live-source relation
  names a value written by an ancestor of the parent reduction. This is
  intentionally conservative for the initial landing: both looped and
  persistent prologue cases keep the safe nested RMSNorm/block-reduction kernel
  but split the sub-parent epilogue into a second kernel. Reloading invalidated
  stores can recover the looped one-kernel form in a follow-up.

Validation after the fixes:

- nested suite: 411 passed, 8 skipped;
- scheduler suite: 144 passed, 6 skipped;
- config suite: 25 passed;
- eight new persistent/looped and dynamic regression cases passed with exact
  nested-fusion and kernel-count assertions;
- 22 production BF16 residual RMSNorm layouts passed in the adversarial battery;
- `PegasusForConditionalGeneration` AMP training forward/backward matched eager;
- coordinate-descent CUDA-graph checks kept both protected R-grouped paths at
  one kernel. A final paired run measured 2.185 us for block quant and 2.299 us
  for swizzle; the base commit measured 2.430 us and 2.298 us respectively.

## Blocker A: dynamic batch + materialized block input -> "plan was lost"

Repro: `tmp_dyn_basic_repro.py rms_y_out 32 1024 <mark|auto> defaults`
(`y = rms(x); return y, block_amax(y)`).

| variant | result |
| --- | --- |
| static batch | fuses, nested=1, correct |
| `mark_dynamic(x, 0)` | **crash**: nested reduction plan was lost before codegen |
| automatic dynamic (2nd batch size) | **crash** (same) |
| `y` not a graph output (inlined) | fine, nested=2 kernels=2 |
| `loop_ordering_after_fusion=False` | no crash (declines fusion at plan time) |
| persistent / looped / persistent_reductions=False | crash in all three |

Mechanism: the plan is recomputed at codegen (saved plans were removed on
purpose). Fusion-time planning runs on unmerged bodies; codegen planning runs
after `Scheduler.merge_loops`, which collapses the node to one iteration range
with a symbolic extent. `_r_grouped_stage_accesses_match` then sees a
`ModularIndexing(v, 1, 32*s)` it can only eliminate for static extents, the
index-equality proof fails, `plan_from_topology` returns None, and
`codegen_staged_reduction` asserts (simd.py:3412). This is the
fusion-time/codegen-time consistency gap the dual audit flagged, reappearing as
"recomputation gives a different answer after a body mutation".

Fix, verified: skip `merge_loops` for members of `FusedStagedReduction` (which
covers `FusedNestedReductions`). Monkeypatching exactly that in
`tmp_merge_freeze_check.py` turns the crash into nested=1, kernels=1, correct.
This also retires the last post-fusion body mutation for staged nodes, making
"codegen re-plan equals fusion-time plan" literally true instead of an
argument. Check the protected kernel corpus after the change (merged vs unmerged
bodies can change generated text). Alternatives (proving the guard on a
merge-equivalent body at fusion time, or hardening symbolic simplification) are
more code for less certainty.

Standalone F1 sub-parent path is NOT affected (`tmp_standalone_staged_check.py
standalone_dyn`: fuses, correct); its relation proofs are merge-invariant.

Regression tests to add (nested suite, both persistent configs): mark_dynamic
batch with materialized normalized output plus block reduce; the automatic
dynamic variant (two batch sizes); assert fusion and numerics on the recompiled
graph.

## Blocker B: looped parent + prologue + sub-parent epilogue -> "lost required sub-parent source"

Repro: `tmp_subparent_lost_repro.py prologue_out_norealize 4096 1 defaults`
(`p = x*w + 1; return sub_parent(rms(p))..., p`).

| variant | result |
| --- | --- |
| no prologue, D=4096 | fuses, correct |
| prologue, D=512 (persistent) | fuses, correct |
| prologue as output, D=4096 (looped) | **crash**: lost required sub-parent source 'buf0' |
| prologue realized, not output, D=4096 | **crash** (same; kernel-local buffer) |
| normalized value realized instead | fine, sub-parent split off (2 kernels) |

Production relevance (`tmp_resid_bf16_check.py`): the transformer residual
stream pattern `h = x + res; return h, quant(rmsnorm(h) * w)` **crashes at
D = 2048, 4096, 8192 in bf16 and at 4096 in fp32**. This is the shape family
the stack exists for.

Mechanism: the sub-parent node reads the prologue's buffer; the planner marks
any parent-written source as `requires_live_source` (internal). In a looped
parent the prologue is emitted inside the reduction loop, its registers are
dead after loop close, and `materialize_sources` asserts (simd.py:2543). The
standalone F1 path handles this topology by declining: its
`_order_sub_parent_parent_nodes` leading-vs-final check refuses internal
sources produced before the reduction, and `tmp_standalone_staged_check.py
standalone_prologue` confirms it falls back to two correct kernels. The nested
sub-parent planner (`_plan_nested_sub_parent_stage`, scheduler.py:1763) has no
counterpart: no deferral ordering and no leading/final source check.

Fix shapes, in order of preference:
1. Fail closed now: in the nested sub-parent planner, decline SUB_PARENT for any
   consumer whose required internal source is written by an ancestor of the
   outer reduction (a prologue). Persistence is unknown at plan time, so this
   also drops the persistent case that works today; the fallback is the
   existing two-kernel split (as `y_realize_out` shows).
2. Follow-up to recover the fusion: treat prologue-produced sources as
   reloadable. Their stores complete before the sub-parent stage, and physical
   reloads of an invalidated store add the name to `must_keep_buffers`
   (common.py:3023-3026), which blocks kernel-local removal. Note the static
   attacker's correction: for a kernel's own outputs `KernelArgs.input` returns
   the out_ptr, so survival rests on `must_keep_buffers`, not
   `args.input_buffers`; any change that trims that set breaks this option. Persistent kernels would still hit
   the live value first through the resolver; looped kernels would reload. This
   changes the F1 source contract and needs its own review and battery.
3. Port `_order_sub_parent_parent_nodes` / `required_post_reduction_index` to
   the nested path for deferrable (post-reduction) internal sources; prologues
   are never deferrable, so this does not by itself fix B but aligns the paths.

Regression tests to add: the residual pattern in bf16 at D=4096 (looped) and
D=512 (persistent), with the prologue both as output and realized-not-output;
assert correctness and, once fixed, the expected fusion outcome.

## Ordering fix (hole 5) verification, this pass

- Both original repros (`repro_parent_reads_lri{,_looped}.py`) now fall back
  (nested=0) and match eager on both calls.
- Name spaces agree: `local_stage_names` is built from `get_operation_names()`;
  `compute_ancestors` (scheduler.py:6676) adds `defining_op_name()` values. Not a
  silent no-op (also printed from an instrumented run by the static attacker).
- Ancestry includes mutation ordering: `unmet_dependencies` starts as the full
  `read_writes.reads` (:2662), `add_fake_dep` routes through `set_read_writes`
  (:2652), mutation `WeakDep`/`StarDep`s are added in `compute_dependencies`
  (:6352-6390), `dead_node_elimination` prunes only WeakDeps on removed ops
  (:2758-2772), `compute_ancestors` runs after (:5730-5734) and is never
  recomputed by `prune_redundant_deps`. RAW, WAR and WAW are covered as the
  commit message claims. Any mutator hazarding against a displaced node
  inherits its outer-reduction ancestor and is itself displaced.
- Template gate: the guard requires `ComputedBuffer` (:1032-1036); unit-tested.
- `test_reject_parent_read_from_local_reduction_input` uses the two-call pattern
  with fresh scaled inputs and asserts kernel count, so it catches the looped
  first-call-passes trap.
- Suites: nested 403 ran = 402 green + the known openssl AOTI environment
  casualty; scheduler 144 OK.

## Independent attacks this pass

Static (temporal axis; 45 fresh topologies plus code proof): no temporal hole.
Ancestry completeness proved; WAR/WAW safe by the displacement argument (any
mutator hazarding against a displaced node inherits its reduction ancestor and
is itself displaced) plus the persistent-flush loads-before-stores order
(triton.py:6900-6905); intra-grouped-stage emission order is fusion order,
kept topological by the acyclicity checks (scheduler.py:9936, 8568); the outer
group is frozen after the first nested fusion (`fuse_with` reuses node1,
`_plan_fusion_with` admits only grouped-side consumers and re-runs the full
plan); `removed_ops` is only populated on the mix-order split path; in-place
reuse cannot clobber inputs still read by later stages. Two fragilities worth
a comment in code: (1) looped-parent prologue buffers survive only through
`must_keep_buffers` (see Blocker B, option 2); (2) `_pre_fusion_custom_pass`
runs after `compute_ancestors` (scheduler.py:5746), so a custom pass that adds
nodes or deps without recomputing ancestry weakens the ordering guard.

Empirical (multi-reduction battery, `tmp_temporal_*`, 205 cases x 3 parent
configs = 615 runs, one process per run, two calls, nested-vs-unnested byte
oracle): 386 PASS, 5 PASS* (bf16 cast placement only), 196 conservative
fallbacks, 28 loud failures (all instances of blockers A and B), 0 silent
wrong results. The fixed hazard class falls back in every variant tried,
including reads through where, cat, aliases, slices, bf16 casts, chained
displaced nodes, and WAR-only WeakDep edges. Cudagraph-trees variants pass.

Low-precision note: bf16/fp16 graphs differ between nested and unnested
compiles by one ulp (and ~0.7% of NVFP4 nibbles) unless
`emulate_precision_casts=True`, under which they are byte-identical. This is
cast-pair folding from fusion, not ordering, and is expected of any new fusion,
but the PR description should say it. Compare low-precision nested results
against the unnested compile, never against eager.

## Residuals for later passes

- Conservative fallbacks worth a perf look: dynamic D and fully dynamic
  shapes disable nesting; indirect reads of realized internals unfuse;
  column-axis reductions in a chain unfuse; reduced-shaped `(x,1)` epilogues
  (sqrt of a sum, realized scalars) decline the whole nested fusion; two
  same-G block reductions of one value nest neither; grouped welford and
  argmax never nest; softmax probabilities returned plus a second row
  reduction plus block quantization always falls back (the fixed class; a
  persistent-parent schedule that keeps the displaced node in the parent
  stage after its reduction would recover it); PARENT_FULL producers feeding a
  REDUCED consumer inside the grouped stage split the consumer into its own
  kernel, the likeliest cheap win.
- The two tripwire asserts did their job twice today. Keep them; do not soften
  either into a fallback at codegen (there is no unfuse at codegen).
- Environment: `/usr/bin/openssl` shadows conda's under `conda run` on this box
  and breaks AOTI/cpp-wrapper header hashing with
  `InductorError: IndexError` from `codecache._get_file_checksum`. Not the
  change.

## Scratch index (all git-ignored, `agent_space/pr191974_failure_investigation/`)

`tmp_dyn_basic_repro.py`, `tmp_subparent_lost_repro.py`,
`tmp_merge_freeze_check.py`, `tmp_resid_bf16_check.py`,
`tmp_standalone_staged_check.py`, battery `tmp_temporal_{common,cases,worker,
driver,table}.py` with results in `tmp_temporal_results*.json`, static-attack
scripts `temporal_*.py`.
