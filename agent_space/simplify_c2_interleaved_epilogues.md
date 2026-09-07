# Simplification review: `46ead043c5e` "[inductor] Fuse interleaved epilogues in nested reductions"

Commit 2 of a 4-commit stack (+818/-69 across `simd.py`, `triton.py`, `scheduler.py`,
`cuda_combined_scheduling.py`, `xpu_combined_scheduling.py`, tests). It adds a
`PARENT_HALF` pointwise domain so an interleaved (factor-2) pair consumer can be appended
onto an already-formed `FusedNestedReductions`, plus a new backend fusion hook, a
parent-half source-layout planner, and half-resolution codegen wiring.

I read the full diff, then re-verified every idea against the stack HEAD (`bdf385fc2ed`)
by extracting the files with `git show` (the shared working tree is being mutated by a
concurrent process, so all `file:line` refs below are against `bdf385fc2ed`, not the
working tree). I also read `SIMPLIFICATION_IDEAS.MD`; it targets a superseded revision and
most of its ideas are already applied or invalidated (see "Considered and rejected").

---

## Ranked ideas

### 1. Delete the `can_fuse_nested_reduction_append` backend hook; early-return inside `SIMDScheduling.can_fuse`

**Summary:** The commit adds a brand-new virtual method to the backend interface plus two
pure-delegation overrides, to express one thing: "when node1 is a `FusedNestedReductions`,
run only the sub-parent leaf check and skip the numel/tiling ladder." That is a
three-line early return in the one backend that implements it.

**Where (all verified at HEAD):**
- `torch/_inductor/scheduler.py:11340` — `BaseScheduling.can_fuse_nested_reduction_append`
  default (`return self.can_fuse_vertical(node1, node2)`).
- `torch/_inductor/codegen/simd.py:2597` — `SIMDScheduling.can_fuse_nested_reduction_append`
  (11 lines; body is the same `_sub_parent_epilogue_leaf_violation` check that
  `SIMDScheduling.can_fuse` already runs at `simd.py:2421`, then `return True`).
- `torch/_inductor/codegen/cuda_combined_scheduling.py:110` — override delegating to
  `_triton_scheduling`.
- `torch/_inductor/codegen/xpu/xpu_combined_scheduling.py:66` — same.
- `torch/_inductor/scheduler.py:9278-9283` — `backend_can_fuse = ... if isinstance(node1,
  FusedNestedReductions) else backend.can_fuse_vertical`, used at 9291 and 9312.

**Why the delegation boilerplate is unnecessary:** the stated reason for the CUDA override
("skip the template ladder") does not hold — the ladder is already a no-op for this node
type. `is_cutlass_template` / `is_cutedsl_template` /
`is_nv_universal_gemm_template` all require `isinstance(node, SchedulerNode)`
(`cutlass/scheduling.py:58`, `cutedsl/cutedsl_scheduling.py:40`,
`nv_universal_gemm_scheduling.py:85`) and `FusedNestedReductions` is a
`FusedSchedulerNode`; `is_cutlass_fused_template` /
`is_nv_universal_gemm_fused_template` bottom out in the same predicate or in
`get_template_node()`, which is `None` here. So `CUDACombinedScheduling.can_fuse_vertical`
falls straight through to `_triton_scheduling.can_fuse_vertical`, i.e. `SIMDScheduling.can_fuse`.

**Concrete change:** in `SIMDScheduling.can_fuse`, immediately after the existing
leaf-violation check (`simd.py:2421-2424`), add:
```python
if isinstance(node1, scheduler.FusedNestedReductions):
    # Legality is decided by FusedNestedReductions.can_fuse_with; the numel/tiling
    # ladder below does not model the nested iteration space.
    return True
```
Then delete all five sites above and restore the two `self.get_backend(device).can_fuse_vertical(node1, node2)`
calls in `Scheduler._can_fuse`. Note `can_fuse_horizontal is can_fuse` in `SIMDScheduling`,
but the horizontal branch (`scheduler.py:9316`) is unreachable for this node1:
`FusedNestedReductions.can_fuse_with` returns False unless
`self.node2.get_operation_names() & other.ancestors` (`scheduler.py:3860`), so the
dependent branch is always taken.

**Est. LOC removed:** ~29 net (5 + 11 + 7 + 5 + 5 removed, 4 added), and one method off the
`BaseScheduling` interface — which is the part that matters, since every future backend
otherwise inherits a hook it will never implement.

**Risk:** low-med. Behaviour-identical by construction; the residual risk is a template
predicate I mis-read. Guard with `test_nested_reduction.py -k "parent_half or
interleaved_pair or nvfp4"` and any XPU nested-reduction coverage.

**Verdict:** yes.

---

### 2. Build `PointwiseDomainContext` through one factory instead of duplicating it in two places

**Summary:** The commit message says the bug being fixed was that the legality check and
the constructed fused node derived the parent-half domain differently, and the fix was
"one shared helper". But only the *innermost* computation was shared; the surrounding
context construction is still written out twice, verbatim, and is the actual thing that
must not diverge.

**Where:**
- `scheduler.py:1629-1643` (`NestedReduction.can_fuse`).
- `scheduler.py:3795-3811` (`FusedNestedReductions.__init__`).

The two blocks are identical modulo `group_size_int` vs `int(exact_group_size)`: both do
`iter_ranges, reduce_ranges = grouped_reduction.get_ranges()`, then
`_parent_half_domain(grouped_axis, <int group size>, outer_numel, outer_rnumel)`, then the
same six keyword arguments.

**Concrete change:** add
`PointwiseDomainContext.create(grouped_reduction, grouped_numel, grouped_rnumel, grouped_axis, group_size, outer_numel, outer_rnumel)`
as a classmethod on the dataclass; call it from both sites. `_parent_half_domain` becomes
private to that factory (or, better, `parent_half_domain` becomes a `@property` computed
from `grouped_axis` + `group_size` stored on the context, which makes the invariant
structural rather than conventional). If you take the property route,
`test_inductor_scheduler.py:326 test_nested_reduction_parent_half_domain` needs
re-pointing at the context.

Also fold the duplicate `grouped_reduction.get_ranges()` call: `scheduler.py:1605` already
computed `iter_ranges` and 1629 recomputes both.

**Est. LOC removed:** ~16.

**Risk:** low. Pure refactor of two identical expressions into one.

**Verdict:** yes — this is the commit's own stated goal, finished.

---

### 3. Replace the three parallel `parent_half_*` optionals with one `_SubParentStage | None`

**Summary:** Three correlated locals that are always all-None or all-set are threaded
through two function signatures as separate optional parameters, forcing an assert cascade
and a `| None` on a value that is never `None`.

**Where:**
- `simd.py:3245-3258` — `parent_half_family`, `parent_half_source`,
  `sub_parent_source_layouts` all initialized empty, all filled inside the same `if any(...)`.
- `simd.py:3279-3283` — three asserts, of which
  `assert sub_parent_source_layouts is not None` is *dead*: the variable is annotated
  `dict[str, SubParentSourceLayout]` and initialized to `{}` at 3245-3247, so it can never
  be `None`.
- `simd.py:3351-3360` — `_codegen_nested_grouped_schedule` gains two optional params that
  it only forwards.
- `simd.py:3465-3467, 3478-3479` — `_codegen_grouped_reduction` gains the same two, plus
  `assert source_layouts is not None` to re-establish the correlation.

**Concrete change:** a frozen dataclass next to `_GroupedReductionVars`:
```python
@dataclasses.dataclass(frozen=True)
class _SubParentStage:
    family: _DerivedIterationFamily
    source: _IterationSpace
    source_layouts: dict[str, scheduler.NestedReduction.SubParentSourceLayout]
    factor: int
```
built once at `simd.py:3253-3258`, passed as a single `sub_parent_stage: _SubParentStage | None`
to `_codegen_nested_grouped_schedule` and `_codegen_grouped_reduction`. All four asserts go
away (the `if stage is not None` narrowing covers mypy), and
`_codegen_nested_grouped_schedule` drops from 12 to 11 parameters.

**Est. LOC removed:** ~16, plus four assertion lines.

**Risk:** low. Mechanical; the correlation being encoded is already enforced by the asserts.

**Verdict:** yes.

---

### 4. Pick one vocabulary: `parent_half` or `sub_parent`

**Summary:** The same concept carries two names, frequently in the same statement. At HEAD
`sub_parent*` outnumbers `parent_half*` roughly 219 to 61 across `scheduler.py` +
`simd.py`, so `parent_half` is the minority dialect this commit introduced.

**Where (representative):**
- `simd.py:3254` — `parent_half_family = layout.make_sub_parent_family(parent_half_factor)`.
- `simd.py:3265` — `sub_parent_source_layouts = node.parent_half_source_layouts`.
- One value carries three names across three frames: `sub_parent_source_layouts`
  (`simd.py:3245`) → `parent_half_source_layouts` (`simd.py:3352`) → `source_layouts`
  (`simd.py:3396`, `simd.py:3466`).
- `scheduler.py:583 PointwiseDomain.PARENT_HALF`, `586 PARENT_HALF_FACTOR`,
  `608 parent_half_domain`, `3838 parent_half_source_layouts` vs
  `593 SubParentSourceLayout`, `3821 sub_parent_broadcast_source_names`.
- Also: `scheduler.py:934` uses `sub_parent_factor in (cls.PARENT_HALF_FACTOR, 4)` — using
  the "fused-append factor" constant as a stand-in for the literal 2 in the standalone
  plan's interleave check, which is a different concept. Write `(2, 4)` there.

**Concrete change:** rename `PARENT_HALF` → `SUB_PARENT`, `parent_half_domain` →
`sub_parent_domain`, `parent_half_family` → `sub_parent_family`,
`parent_half_source_layouts` → `sub_parent_source_layouts`, `PARENT_HALF_FACTOR` →
`NESTED_APPEND_SUB_PARENT_FACTOR` (or keep the name but stop reusing it as a literal 2).
Purely mechanical; no behaviour change.

**Est. LOC removed:** ~0. This is a readability-only item, but it is the single largest
comprehension tax in the commit: a reader currently has to prove to themselves that
"parent half" and "sub-parent at factor 2" are the same thing.

**Risk:** low (rename only). Touches ~61 sites plus test names such as
`test_dynamic_parent_half_epilogue` and `test_parent_half_append_respects_fusion_gate`.

**Verdict:** yes-if you are willing to take the rename churn in this commit rather than a
follow-up.

---

### 5. Collapse `producer_node` + `_skip_fused_nested_dispatch` — two perfectly correlated parameters

**Summary:** `_can_fuse_nested_reduction_append` gains an optional `producer_node`, and
`_can_fuse` gains an underscore-prefixed boolean whose only purpose is to suppress the
recursion that `producer_node` would otherwise cause. They are always set together;
neither can be reasoned about locally.

**Where:**
- `scheduler.py:3920` — `producer_node=self if half_resolution_nodes else None`.
- `scheduler.py:8899, 8923, 8927` — `producer_node` param, the `producer_node if
  producer_node is not None else grouped_node` ternary, and `_skip_fused_nested_dispatch=True`.
- `scheduler.py:8980, 9002` — the flag param and the guard
  `isinstance(node1, FusedNestedReductions) and not _skip_fused_nested_dispatch`.

**Concrete change, part (a) — drop the flag (low risk):** the four special-node dispatches
at `scheduler.py:9002-9012` (`FusedNestedReductions` node1/node2,
`FusedMixOrderReductions` node1/node2) belong in `_can_fuse_impl`, not `_can_fuse`.
`_can_fuse` has exactly two callers (`_can_fuse_impl:8966` and
`_can_fuse_nested_reduction_append:8922`), so moving them up makes the append path
naturally skip re-dispatch and the flag disappears. The node2 guards stay safe because
`FusedNestedReductions`/`FusedMixOrderReductions` are reductions and
`can_fuse_with` rejects `other.is_reduction()` at `scheduler.py:3856`.

**Concrete change, part (b) — drop `producer_node` (med risk):** always pass `self`. The
only reason it is conditional is that commit 1's non-half path passes `self.node2`;
`self` is a superset of `self.node2`'s buffers/deps, so this should be at worst more
permissive. If that is too spicy, at minimum rename it — `grouped_node` and
`producer_node` are both producers, and the parameter's real meaning is "the node whose
deps vertical legality is checked against".

**Est. LOC removed:** ~8 for (a); ~4 more for (b).

**Risk:** (a) low; (b) med — it changes the dep set used for the commit-1 append path, so
it needs the full `test_nested_reduction.py` suite plus `test_loop_ordering.py`.

**Verdict:** (a) yes; (b) yes-if the full suite stays green.

---

### 6. Tighten `FusedNestedReductions`'s new state: cache the half nodes, demote `grouped_pointwise_domains`, fix the method/attribute name clash

**Summary:** `__init__` computes the half-node list, throws it away, and `can_fuse_with`
recomputes it from a stored intermediate that exists only for that purpose. Separately,
an attribute and a method in the same class differ only by a leading underscore.

**Where:**
- `scheduler.py:3818-3820` — `self.grouped_pointwise_domains` stored; its only readers are
  `3823`, `3835` (both inside `__init__`) and `3894` (`candidate_domains`). Nothing outside
  `scheduler.py` reads it (verified by repo-wide grep).
- `scheduler.py:3833-3837` — `parent_half_nodes` computed as a local and discarded.
- `scheduler.py:3882-3899` — `half_resolution_nodes` and `candidate_half_resolution_nodes`,
  two near-identical comprehensions 12 lines apart, plus the `candidate_domains`
  intermediate that exists only to be filtered again.
- `scheduler.py:3838` `self.parent_half_source_layouts` (attribute) vs `scheduler.py:3923`
  `def _parent_half_source_layouts` (method).

**Concrete change:** store `self.parent_half_nodes: tuple[SchedulerNode, ...]` in
`__init__` and let `grouped_pointwise_domains` be a local. `can_fuse_with` then reads:
```python
new_half_nodes = [sn for sn, d in pointwise_domains if d is PARENT_HALF]
candidate_half_nodes = [*self.parent_half_nodes, *new_half_nodes]
```
deleting `candidate_domains` entirely. Rename the method to
`_plan_sub_parent_source_layouts` (folds into idea 4).

**Est. LOC removed:** ~10, plus one attribute off the class.

**Risk:** low.

**Verdict:** yes.

---

### 7. Share one `_rmsnorm_nvfp4` test helper and one packing-asm constant

**Summary:** The commit correctly factors the MXFP4 model into `_rmsnorm_mxfp4` and hoists
`MXFP4_RECIP_UE8M0_ASM` to module scope — and then does neither for NVFP4, duplicating the
model verbatim and inlining the same PTX string three more times.

**Where (HEAD):**
- `test/inductor/test_nested_reduction.py:946-967` (in
  `test_producer_consumer_rmsnorm_nvfp4_inline_asm`) is a verbatim copy of
  `test_nested_reduction.py:2124-2145` (in `_capture_nvfp4_kernel_sources`) — 22 lines.
- The `"{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}"` literal
  appears at lines 959, 1357, 1402, 2137, 2180, 2263 (three of those added by this commit).

**Concrete change:** add `def _rmsnorm_nvfp4(x, weight, G)` next to `_rmsnorm_mxfp4`
(`test_nested_reduction.py:2156`) and call it from both places, exactly as the MXFP4 pair
already does. Add `E2M1X2_PACK_ASM = "..."` next to `MXFP4_RECIP_UE8M0_ASM`
(`test_nested_reduction.py:30`) and use it at all six sites.

**Est. LOC removed:** ~30.

**Risk:** none (tests only, no semantic change).

**Verdict:** yes.

---

### 8. De-duplicate the RMSNorm-to-interleaved-pair model across three tests

**Summary:** Three tests carry byte-identical model bodies, so the actual thing each test
varies (dynamic shapes; a fusion gate) is buried.

**Where:** `test_nested_reduction.py:864-870`
(`test_producer_consumer_rmsnorm_interleaved_pair_epilogue`), `883-889`
(`test_dynamic_parent_half_epilogue`), `911-916`
(`test_parent_half_append_respects_fusion_gate`). The four `..._rejects_...` tests at
1237-1336 are near-variants of the same body.

**Concrete change:** one module-level
`def _rmsnorm_interleaved_pair(x, weight, G, *, return_scale=False)` used by the three
positive tests; leave the four negative tests inline, since their whole point is the
deviation (but consider a one-line comment on each naming the deviation, which the code
currently makes the reader diff for).

While there: `import torch.nn.functional as F` appears 36 times in this file, ~9 of them
added by this commit. Hoist it to module scope.

**Est. LOC removed:** ~14 for the bodies, ~9 more for the imports (~36 if you fix the
whole file).

**Risk:** none.

**Verdict:** yes.

---

### 9. Drop the `allow_reduced_broadcast` keyword

**Summary:** A boolean that gates a branch already fully determined by the value's shape.

**Where:** `simd.py:2044` (param), `2048-2056` (the branch:
`if not allow_reduced_broadcast or parent_dim != self.num_groups_str: return False`),
`simd.py:2347` (the only site that passes it, as
`allow_reduced_broadcast=name in self._must_materialize_names`).

**Why it is dead generality:** `parent_dim == self.num_groups_str` already means "this
value is at reduced resolution", and broadcasting is the only correct handling. The flag
only distinguishes "reduced-shaped value we were told to materialize" (broadcast) from
"reduced-shaped value we were not told about" (return False, which the caller at
`simd.py:2352` turns into a hard `AssertionError`). Removing it converts a compiler crash
into working code, never the reverse. In the standalone path, `must_materialize_names` is
*all* parent-node buffer names (`simd.py:3733-3737`), so the flag is already True for
everything relevant there.

**Est. LOC removed:** ~5.

**Risk:** low. Only relaxes a rejection.

**Verdict:** yes.

---

### 10. Untangle the local names and the double emptiness check in `_sub_parent_epilogue_source_deps`

**Summary:** One list gets four names in twenty lines, and the empty case is tested twice.

**Where:** `scheduler.py:910-929`:
`source_deps_for_name` → `full_resolution_source_deps` → `reduction_deps` (a pure alias,
`scheduler.py:922`) → `source_dep`. `reduction_deps` is also a lie by then: the list may
come from `source_writes`, not from any reduction's reads.

**Concrete change:** delete line 922 and merge the two guards:
```python
if not full_resolution_source_deps:
    if dep.name not in fused_buffer_names and dep.name in V.graph.removed_buffers:
        return None
    continue
if len(full_resolution_source_deps) != 1:
    return None
source_dep = full_resolution_source_deps[0]
```

**Est. LOC removed:** ~4.

**Risk:** low — semantics preserved exactly (the `fused_buffer_names` short-circuit is
folded into the removed-buffers test).

**Verdict:** yes.

---

### 11. Shrink `TritonKernel._codegen_named_constant`

**Summary:** A 15-line single-caller helper whose bulk is an internal-consistency
`AssertionError` that cannot fire, plus a two-line comment restating the code.

**Where:** `triton.py:7823-7837` (helper), `triton.py:3297-3299` (field + comment),
`triton.py:7855-7856` (the only call).

**Why the assert cannot fire today:** the only producer of named constants is
`_GroupedReductionLayout._grouped_axis_named_constants` (`simd.py:1780`), whose two
symbols are `local_reduction_size_sym → local_reduction_size` (tree-independent) and
`reduced_block_sym → FloorDiv(tree.block_size(), local_reduction_size_sym)`. Both callers
(`make_reduced_output_family:1890`, `make_sub_parent_family:1958`) pass `self.group_tree`,
so the emitted lines are identical by construction.

**Concrete change:** make `_named_constants` an `OrderedSet[str]` of emitted names, drop
the conflict branch, and inline the remaining four lines into the loop at 7855. Trim the
kernel-field comment to one line (or delete it — the field name says it).

Note: the dedup itself *is* load-bearing (the reduced-output and sub-parent families both
declare these constants and header emission order is not fixed), so do not remove it —
and do not try to drop `reduced_block_sym` from the sub-parent family: it is used as
`num_groups_str` in `_broadcast_value_to_axis_resolution` (`simd.py:2102`).

**Est. LOC removed:** ~10.

**Risk:** low.

**Verdict:** yes.

---

### 12. Grab bag

- `scheduler.py:1457` — `expected_numel: sympy.Expr | None`; no branch ever assigns `None`.
  Drop the `| None`.
- `scheduler.py:1471-1473` — `if domain_context.parent_half_domain is None: return False`
  inside the `PARENT_HALF` arm is unreachable: `_classify_grouped_pointwise_nodes` only
  emits `PARENT_HALF` when `has_parent_half_domain` (`scheduler.py:1388, 1424`), and both
  callers of `_pointwise_domains_are_compatible` (`scheduler.py:1331`, `3871`) pass the
  same context used for classification. Replace with an `assert` if mypy needs narrowing.
- `simd.py:3283` — dead `assert sub_parent_source_layouts is not None` (see idea 3).
- `scheduler.py:3887-3892` — six-line comment explaining the read-before-write hazard,
  then `scheduler.py:8919-8921` repeats a similar explanation for
  `index_equivalent_dep_names`. Both are good comments; just check the second one still
  says "Parent-full consumers" when parent-half now takes the same path
  (`scheduler.py:8907-8910`).
- `scheduler.py:8890` — the `_can_fuse_nested_reduction_append` docstring/comment at
  8919-8921 was written for `PARENT_FULL` only; update after idea 4's rename.

**Est. LOC removed:** ~10 combined.

**Risk:** low. **Verdict:** yes, bundled.

---

## Considered and rejected

- **Hardcode `factor = 2` everywhere** (prior review's idea #1). Invalid at HEAD: commits 3
  and 4 made the sub-parent machinery genuinely multi-factor (`MAX_SUB_PARENT_FACTOR = 16`,
  `scheduler.py:587`; `sub_parent_factor in (cls.PARENT_HALF_FACTOR, 4)`,
  `scheduler.py:934`; `_sub_parent_epilogue_rate` returns 4-with-3-lanes at
  `scheduler.py:837-842`). The factor parameter is live.
- **Fold `_SubParentSourceLoadMaterializer` into `_GroupedReductionOpsHandler`** (prior
  review's idea #4). Invalid at HEAD: the materializer now has two stacking sites over two
  *different* inner handlers — `_codegen_grouped_reduction` (`simd.py:3480`, over
  `_GroupedReductionOpsHandler`) and the standalone path (`simd.py:3744`, over the raw ops
  handler). Folding it in would require duplicating it.
- **Collapse the remaining assertion pair in `_SubParentPointwiseRemapHandler`** (prior
  review's ideas #2/#3). Already done — this commit deleted
  `materialize_all_store_cache_values` and cut the cascade to two distinct messages
  (`simd.py:2333-2337`, `2352-2355`), and `must_materialize_names` is no longer
  assertion-only (it now drives `allow_reduced_broadcast`). Idea 9 above finishes the job.
- **Inline `_codegen_grouped_reduction` / `_parent_half_source_layouts` as single-callers.**
  Both are ~30 and ~67 lines respectively; CLAUDE.md's rule targets 1-2 LOC helpers.
- **Share the dep-collection preamble between `sub_parent_epilogue_plan`
  (`scheduler.py:657-669`) and `_parent_half_source_layouts`
  (`scheduler.py:3935-3948`).** Tempting (both build `reduction_reads` + `source_writes`),
  but the two disagree on *which* nodes contribute to each dict — the plan uses every
  reduction node and every node's writes; the fused path uses only `grouped_reduction`'s
  reads and excludes the half nodes from writes. A shared helper would need both node
  sets as parameters and would save ~8 lines while making the difference less visible.
- **Drop the `_sub_parent_epilogue_leaf_violation` call from the append path**, on the
  theory that a `FusedNestedReductions` is codegen'd by `codegen_nested_reduction` and can
  never take the standalone sub-parent plan. Plausible, but I could not confirm it at
  runtime (the shared working tree was checked out to a different commit mid-review), and
  a wrong guess here silently re-enables a rejected fusion. Idea 1 preserves the check.

---

## Possible correctness concerns (not part of the simplification set)

1. **`producer_node` looks at the wrong node set.** `scheduler.py:3920` passes
   `producer_node=self if half_resolution_nodes else None`, where `half_resolution_nodes`
   (3882) is derived from `other` only — but the guard 20 lines earlier
   (3895-3899) deliberately uses `candidate_half_resolution_nodes`, which also includes
   `self.grouped_pointwise_domains`. If the fused node already contains a half-resolution
   consumer and `other` does not, the relaxed producer node is *not* used. If that is
   intentional it deserves a comment; if not, appending a second consumer to a
   half-resolution nested node uses the narrower `self.node2` dep set.
2. **`_parent_half_source_layouts` omits the internally-written-source guard.**
   `sub_parent_epilogue_plan` rejects plans where a planned source is both written inside
   the fusion and read by a reduction (`scheduler.py:705-709`, "looped codegen cannot
   retain an internally produced source across a downstream reduction loop"). The fused
   nested path (`scheduler.py:3950-3968`) calls the same
   `_sub_parent_epilogue_source_deps` but has no equivalent check. Worth confirming the
   nested kernel's non-persistent (looped) form cannot hit the same hazard.
