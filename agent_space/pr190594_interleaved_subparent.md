# #190594 — Fuse interleaved sub-parent reduction epilogues

`86038d6e1cd` · `gh/eellison/1066` · +2148 / -214 across 13 files

## Full stack guides and PR paths

Start with
[`/data/users/eellison/pytorch/agent_space/READ_ME_FIRST.md`](/data/users/eellison/pytorch/agent_space/READ_ME_FIRST.md)
for the current stack state, verification summary, and submission notes. Then
use the table below to open each isolated PR and its full local guide. The order
is bottom to top: every row builds on the rows above it.

| Order | PR and scope | Commit | GitHub PR | Full local guide |
|---:|---|---|---|---|
| 1 | #190594 - standalone interleaved foundation | `86038d6e1cd` | [pytorch/pytorch#190594](https://github.com/pytorch/pytorch/pull/190594) | [`/data/users/eellison/pytorch/agent_space/pr190594_interleaved_subparent.md`](/data/users/eellison/pytorch/agent_space/pr190594_interleaved_subparent.md) |
| 2 | #190595 - nested interleaved integration | `359b11d5aa1` | [pytorch/pytorch#190595](https://github.com/pytorch/pytorch/pull/190595) | [`/data/users/eellison/pytorch/agent_space/pr190595_nested_interleaved.md`](/data/users/eellison/pytorch/agent_space/pr190595_nested_interleaved.md) |
| 3 | #191775 - MXFP6 4:3 staged packing | `8dc79f8803b` | [pytorch/pytorch#191775](https://github.com/pytorch/pytorch/pull/191775) | [`/data/users/eellison/pytorch/agent_space/pr191775_mxfp6_staged_packing.md`](/data/users/eellison/pytorch/agent_space/pr191775_mxfp6_staged_packing.md) |
| 4 | #190596 - contiguous sub-parent layouts | `1e2b8890c98` | [pytorch/pytorch#190596](https://github.com/pytorch/pytorch/pull/190596) | [`/data/users/eellison/pytorch/agent_space/pr190596_contiguous_subparent.md`](/data/users/eellison/pytorch/agent_space/pr190596_contiguous_subparent.md) |
| 5 | #191974 - test-helper hygiene | `3f62ae27baa` | [pytorch/pytorch#191974](https://github.com/pytorch/pytorch/pull/191974) | [`/data/users/eellison/pytorch/agent_space/pr191974_test_hygiene.md`](/data/users/eellison/pytorch/agent_space/pr191974_test_hygiene.md) |

For an isolated diff, run `git show <commit>` with the commit from the table.
The historical #191975 cleanup is no longer a stack commit; its changes were
folded into their owner PRs. Its archival guide is
[`/data/users/eellison/pytorch/agent_space/pr191975_simplification.md`](/data/users/eellison/pytorch/agent_space/pr191975_simplification.md).

| File | Δ | What it holds |
|---|---|---|
| `torch/_inductor/scheduler.py` | +519/-91 | common staged plan/identity, planner and legality |
| `torch/_inductor/codegen/common.py` | +7 | CSE-liveness query used by lazy source projection |
| `torch/_inductor/codegen/simd.py` | +599/-88 | staged dispatch, derived iteration domain, kernel emission |
| `torch/_inductor/codegen/simd_kernel_features.py` | +14/-3 | separate tiling and index-width schedules |
| `torch/_inductor/codegen/triton.py` | +54/-3 | `tl.split` emission |
| `torch/_inductor/codegen/cuda_combined_scheduling.py` | +5/-2 | capability and staged-codegen forwarding |
| `torch/_inductor/codegen/xpu/xpu_combined_scheduling.py` | +5/-2 | capability and staged-codegen forwarding |
| `torch/_inductor/config.py` | +2/-3 | staged-reduction flag description |
| `torch/_inductor/dependencies.py` | +52 | explicit-domain `MemoryDep` normalization |
| `torch/_inductor/utils.py` | +9/-10 | float8 dtype deduplication |
| `test/inductor/test_dependencies.py` | +37 | dependency normalization coverage |
| `test/inductor/test_nested_reduction.py` | +818/-11 | behavior, identity and kernel-form coverage |
| `test/inductor/test_indexing.py` | +13 | derived-epilogue index-width coverage |

Foundation commit. The three feature PRs above it extend its vocabulary;
concepts introduced here are not re-explained there. #191974 is a final
test-only cleanup.

## Stack evolution and reservations

**Changed later:** this commit supports only factor-2 INTERLEAVED packing.
[#190595](https://github.com/pytorch/pytorch/pull/190595) must relax
`StagedReductionPlan`'s combined-stage guard and route final domain rebuilding
through `PointwiseDomainContext.create()`.
[#191775](https://github.com/pytorch/pytorch/pull/191775) derives factors from
consumer/parent sizes, adds the 4:3 output rate, widens the INTERLEAVED bound to
4, and adds MXFP6 staging.
[#190596](https://github.com/pytorch/pytorch/pull/190596) then adds CONTIGUOUS
layouts and widens one-output factors to 16. The five guides in the navigation
table describe the rebased representation; other archived files may predate
it.

**Corrected later:** #190595 keeps the temporal names already present in each
node's `read_writes` when it plans nested sources; the scheduler-global final
mutation map is not valid for that plan. #191775 additionally requires a
per-read normalized index proof before a grouped-output name can relax vertical
dependency matching. Neither correction changes this standalone commit's lane
proof.

**Final review state:** sub-parent pointwise emission uses the same remapped
emitter as nested pointwise stages. Epilogue nodes participate in index-dtype
analysis, and one kernel context makes store metadata follow actual emission.
Parent sources share one normalized index proof; independent inputs and expired
parent values reload normally. Both concrete classes pass locally (234 passed /
1 skipped). Submitted L4 CI exposed that the large-group looped test also
triggered reduction splitting; the pending one-line test patch disables that
orthogonal transform.

**Remaining reservation:** the standalone plan is deliberately rebuilt across
the fusion/codegen boundary after loop merging. This is a design choice, not a
known NVFP4 correctness failure.

## The problem

Inductor models the members of a fused kernel at a few fixed resolutions
relative to the group's `(numel, rnumel)`: the reduction itself, *reduced*
`(numel, 1)`, or *full resolution* `(numel * rnumel, 1)`.

A quantization packing epilogue fits none of them. Take the FP4 shape — a
per-group amax, then pack adjacent pairs:

```python
xg    = x.view(B, D // G, G)                       # (32, 64, 16)
scale = (xg.float().abs().amax(-1) / 6.0).clamp(...)
xp    = xg.view(B, D // G, G // 2, 2)
even  = xp[..., 0].float() / scale.unsqueeze(-1)   # (32, 64, 8)
odd   = xp[..., 1].float() / scale.unsqueeze(-1)   # (32, 64, 8)
```

Here is what the scheduler actually sees (dumped via
`_post_fusion_custom_pass`, group `(2048, 16)`):

```
op0_op1_op2_op3   group=(2048, 16)   reduction=True
    read  arg0_1   1024*d0 + 16*d1 + d2         size=(32, 64, 16)   <- reduction
    read  arg0_1   1024*d0 + 16*d1 + 2*d2       size=(32, 64, 8)    <- even lane
    read  arg0_1   1024*d0 + 16*d1 + 2*d2 + 1   size=(32, 64, 8)    <- odd lane
    write buf0     64*d0 + d1                   size=(32, 64)       <- scale
    write buf2     512*d0 + 8*d1 + d2           size=(32, 64, 8)    <- half res
    write buf3     512*d0 + 8*d1 + d2           size=(32, 64, 8)
```

The epilogue nodes have group `(numel*rnumel/2, 1)` — a **fraction** of the
parent tile. Nothing can schedule that alongside a `(2048, 16)` reduction, so
packing becomes a second kernel. The unfused graph rereads the input for the
packing stage and round-trips the reduced scale; it does not materialize and
reload the full parent tile. That extra traffic is material for quantization.

Nothing is missing from the IR here. `xp[..., 0]` is not a data-movement op —
it lowers to a load at index `2*d2`, which is exactly what the dep above shows.
That is Inductor working as designed: views, slices and broadcasts are all just
index arithmetic on loads and stores, and **fusion** is what turns co-located
generic accesses into register reuse. Locality is a scheduling outcome, not a
property of the IR.

So the generic IR is fine, and the fix belongs where the contract broke. The
scheduler could place a consumer at *reduced*, *full*, or the reduction itself,
and this consumer sits at a fraction of the parent tile. With no resolution
class for it, fusion declined — and index arithmetic that should have become a
register read became a second kernel and an HBM round-trip instead.

This PR adds that resolution class. The `tl.split` is only how codegen realises
it once the scheduler permits the fusion; it is not new expressiveness.

## Design decisions

1. **A derived range tree, not a re-tiling.** The kernel keeps the reduction's
   grid; only the epilogue stage iterates differently, over a *sub-parent* tree
   derived from the parent's R axis. This commit accepts only factor-2 pair
   packing; factor-4 interleaved staging arrives with #191775.

2. **Legality is proved on memory deps, not graph shape.** The gate is: the
   epilogue's read must be provably lane-*k* of the reduction's read of the same
   buffer. Above, that means proving `1024*d0 + 16*d1 + 2*d2 + k` is lane *k* of
   `1024*d0 + 16*d1 + d2`. Pattern-independent — it accepts anything whose
   indices line up rather than matching a known graph.

3. **Source reuse follows ordinary CSE lifetime.** The parent stage records its
   planned loads. After that stage is flushed, a still-live value is lazily
   split into lanes; an expired value is loaded at the derived child index.
   Persistent values therefore split in registers, while looped values are
   naturally invalidated at loop close and reload. There is no
   persistent-specific scheduling branch. Both forms stay in one kernel.

4. **Ambiguous, mutated, and epilogue-output consumers are rejected.**
   Sub-parent stores are emitted *after* the parent stage, so a non-epilogue
   node reading one of those buffers would load it before the store. Independent
   epilogue inputs remain ordinary kernel dependencies.

## What "lane k" means

A lane is not merely a constant offset. `k` is a constant in `[0, factor)`, but
a lane is defined by a *(child stride, constant term)* pair, and the two layouts
in this stack differ in the stride:

| Layout | index at child `c` | child stride | constant term |
|---|---|---|---|
| INTERLEAVED (this PR) | `factor*c + k` | `factor` | `k` |
| CONTIGUOUS (#190596) | `c + k*child_extent` | `1` | `k*child_extent` |

In the dep dump above, the epilogue's `2*d2 + 1` is lane 1 of the reduction's
`d2`: child stride 2, constant term 1.

`k` is *recovered* differently per layout, which matters when reading the
matchers:

```python
# interleaved - normalize both bodies into explicit (X, R) coordinates,
# then recover the same physical lane codegen will select.
parent_index = normalized_source_read_indices(parent_nodes, (x, parent_r))
child_index  = normalized_source_read_indices(epilogue_nodes, (x, child_r))
lane         = interleaved_sub_parent_lane(child_index, factor, extent_subs)
expected     = parent_index.subs(parent_r, factor * child_r + lane)

# contiguous (_contiguous_sub_parent_epilogue_emitted_lane) - zero the free
# symbols first, then bucket the constant. Mod on the whole index is useless
# with child stride 1: the child variable itself would contribute.
child_extent = FloorDiv(parent_rnumel, sub_parent_factor)
offset       = sympy_subs(dep.index, {sym: 0 for sym in dep.index.free_symbols})
lane         = simplify(FloorDiv(Mod(offset, parent_rnumel), child_extent))
```

For interleaved the `Mod` must simplify to a *static* constant, so an index like
`2*d2 + d3` where `d3` is not provably even correctly yields no lane. The extent
substitution handles a symbolic stride such as `s` when the shape environment
can prove `s == factor * (s // factor)`. Having recovered the lane, the matcher
substitutes `factor*child + k` into the reduction's index and requires static
equality — that is the actual proof.

## What it generates

One kernel. This example is the persistent form; the interesting part is:

```python
lane2_r0_index = r0_offset // 2 + tl.arange(0, R0_BLOCK // 2)[None, :]   # derived range
r0_2 = lane2_r0_index

tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), ...)                   # parent tile, loaded ONCE
tmp1, tmp2 = tl.split(tl.reshape(tmp0, [XBLOCK, (R0_BLOCK//2), 2]))   # even / odd lanes

tmp4  = tl_math.abs(tmp3)                                        # reduction runs on tmp0
tmp8  = triton_helpers.max2(tmp7, 1)[:, None]                    #   (full resolution)
tmp14 = tl.minimum(tmp12, tmp13, ...)                            # the scale

tmp16 = (tmp15 / tmp14)                                          # epilogue on lane 0
tmp18 = (tmp17 / tmp14)                                          # epilogue on lane 1

tl.store(in_out_ptr0 + (x0), tmp14, xmask)                       # scale, reduced res
tl.store(out_ptr0 + (r0_2 + 8*x0), tmp16, lane2_r0_index_mask & xmask)
tl.store(out_ptr1 + (r0_2 + 8*x0), tmp18, lane2_r0_index_mask & xmask)
```

Three things to notice. In this persistent kernel, `tmp0` is loaded once and
feeds both the reduction and the split. `lane2_r0_index` is the derived range, and
the epilogue stores are masked by it, not by `r0_mask`. And the store index
`r0_2 + 8*x0` is at half resolution, matching `buf2`'s dep above.

## The flow: fusion -> codegen

Two phases, separated in time. The scheduler decides; codegen emits. The
decision is **irrevocable** — codegen has no way to decline — so anything
codegen needs must be proven in phase 1.

```
 PHASE 1: FUSION (scheduler.py + simd.py gates)          "may these fuse?"
 -------------------------------------------------------------------------
  Scheduler.can_fuse(node1, node2)
      │
      ├─► SIMDScheduling.can_fuse
      │     ├─ ordinary reduction/reduction and pointwise/pointwise cases
      │     ├─ existing pointwise -> reduction path, unchanged
      │     ├─ for mixed reduction/pointwise candidates:
      │     │    _sub_parent_epilogue_decision
      │     │      └─ _sub_parent_epilogue_plan
      │     │           ├─ backend gates: config and capability
      │     │           ├─► NestedReduction.sub_parent_epilogue_plan
      │     │           └─ _sub_parent_tiling_is_2d
      │     └─ otherwise swap arguments into the existing mixed path
                       │   ── all legality lives here ──
                       ├─ _sub_parent_epilogue_candidate_nodes
                       │    classifies lane, reduced-output and full-parent domains
                       │    and proves each node's loop ranges are compatible
                       ├─ try_get_sub_parent_extent_subs
                       │    proves the symbolic parent extent is divisible by 2
                       ├─ _try_get_sub_parent_source_layouts
                       │    ├─ MemoryDep.normalize_with_ranges
                       │    │    remaps parent and child reads into explicit (X, R)
                       │    ├─ map_kernel_groups_to_node_sizes handles split/merged dims
                       │    └─ require a constant physical lane and index equality
                       │         under parent_r = 2*child_r + lane
                       └─ _sub_parent_epilogue_outputs_unread
                                    │
                                    ▼
                        StagedReductionPlan
                          nested_stage=None
                          sub_parent_stages=(
                       SubParentEpilogueStage(
                              factor, source_layouts, epilogue_nodes),)

      _sub_parent_epilogue_decision then has three outcomes:
        valid staged plan    -> accept the intentional numel mismatch
        invalid staged shape -> reject before generic scheduling sees it
        ordinary shape       -> continue through ordinary fusion rules

      An existing exact FusedStagedReduction may accept another sibling only
      when replanning the combined leaves still produces a valid staged plan.

  BaseScheduling.fuse
      ├─ preserve FusedStagedReduction identity for a valid extension
      └─ otherwise has_sub_parent_epilogue(combined leaves)
           └─ create FusedStagedReduction

                 ~~~ plan discarded; typed identity retained ~~~
                 ~~~ Scheduler.merge_loops mutates leaf loops ~~~

 PHASE 2: CODEGEN (simd.py + triton.py)                  "emit it"
 -------------------------------------------------------------------------
  Scheduler._codegen
      └─ isinstance(node, FusedStagedReduction)
           └─ backend.codegen_staged_reduction
                ├─ filter removed leaves
                ├─ _find_sub_parent_epilogue_plan   -- one final rebuild
                └─ _codegen_reduction_with_sub_parent_epilogue
                 │
                 ├── stage A: the reduction, at parent resolution
                 │     _SubParentSourceLoadResolver records planned loads
                 │     normal reduction emission and kernel.codegen_body()
                 │
                 └── stage B: the epilogue, in the derived lane space
                       _codegen_remapped_pointwise
                         ├─ sub_parent_family.activate(kernel)
                              swaps range trees -> nodes index the lane axis
                              without knowing they are in a derived space
                         └─ _PointwiseRemapHandler.load
                              ├─ family.resolve_load(name, index)
                              ├─ source_resolver.resolve_load(name, index)
                              │    ├─ CSE-live parent value: lazily reshape+split
                              │    └─ expired parent value: no resolution
                              └─ normal remapped load at the derived index
```

The plan does not cross the phase boundary. `FusedStagedReduction` identity
does. That identity keeps generic codegen, combo grouping and benchmark codegen
from trying to represent a derived domain they do not understand.

### The plan is rebuilt, not carried

`Scheduler.merge_loops()` is pre-existing and runs after fusion. On GPU it
rewrites leaf loop bodies, sizes and dependencies while leaving the scheduler
group unchanged. A fusion-time plan therefore must not be treated as final
codegen state.

The standalone wrapper carries identity only. Codegen rebuilds one final plan
from the active leaves after loop merging and removed-op filtering. Fusion
search can still derive plans more than once while considering candidates, but
the old double discovery in `codegen_node` and `_codegen_nodes` is gone.

The nested subclass carries only stable topology: grouped axis, reduction and
group size. Its final `StagedReductionPlan` rebuilds mutable ranges and domains
through `plan_from_topology`. Re-running full nested admission is incorrect:
after `merge_loops`, grouped-axis discovery cannot recover every axis already
approved at fusion time. This distinction is why the common representation is
not a plan cache.

## How to read it

Planner first; codegen assumes it.

**On the line numbers below.** They refer to local source commit
`86038d6e1cd`.

1. **`NestedReduction.sub_parent_epilogue_plan`** — `scheduler.py:617` — the entry
   point. Read its rejection paths in order; collectively they *are* the
   legality model. Every `return None` is a class of graph that stays unfused.
2. **`_try_get_sub_parent_source_layouts`** — `scheduler.py:803` — the source
   proof. It normalizes both sets of reads into explicit parent/child domains,
   recovers the constant physical lane and checks equality under
   `2*child + lane`. If you read one helper in this PR, read this.
3. **`_sub_parent_epilogue_candidate_nodes`** — `scheduler.py:690` — how every
   group member is classified and checked against its execution domain.
4. **`try_get_sub_parent_extent_subs`** — `scheduler.py:746` — the shared
   divisibility proof and symbolic extent normalization used by planner and
   codegen.
5. **`FusedStagedReduction` / `StagedReductionPlan`** — `scheduler.py:3457` /
   `scheduler.py:1457` — the identity that crosses phases and the ephemeral
   emission plan rebuilt within each phase.
6. **`_GroupedReductionLayout.sub_parent_iteration_values`** — `simd.py:1918` — the
   `(x, group, pair)` coordinate space, i.e. where `lane2_r0_index` comes from.
7. **`SIMDScheduling.codegen_staged_reduction`** — `simd.py:2991` — final
   revalidation and dispatch after loop merging.
8. **`_SubParentSourceLoadResolver`** — `simd.py:2210` — records parent
   loads and lazily projects one only while its value remains CSE-live.
9. **`_codegen_reduction_with_sub_parent_epilogue`** — `simd.py:3446` — the emitter;
   note the two-stage structure.
10. **`emit_split_via_reshape`** — `triton.py:6165` — the `reshape` + `tl.split`.

Explicit masked loads are not forwarded through the resident split. Both the
parent recorder and epilogue resolver check the kernel's active load mask;
masked accesses issue ordinary loads so each retains its own mask and fill
value. The fusion itself remains valid, including for concatenated inputs.

## What to pay attention to

- **Planner/codegen agreement is the structural risk.** Fusion establishes the
  typed identity; codegen rebuilds the final plan from post-`merge_loops`
  leaves. A missing final plan is a loud invariant failure, not a fallback.
  Nested codegen must rebuild mutable domains through `plan_from_topology`, not
  rerun grouped-axis discovery. #190595 must route that rebuild through its
  `PointwiseDomainContext.create()` factory so sub-parent domains cannot drift.

- **`min_rblock` is the only block floor here, and it is a legality
  constraint.** It is always `sub_parent_factor` (2 in this commit), so a lane
  group cannot straddle a loop iteration. A persistent kernel still selects its
  full padded RBLOCK through the ordinary reduction heuristic; this path does
  not impose a separate persistent floor or branch.
  There is deliberately no `min_xblock` on this path — how many rows a program
  should process to amortize the parent-tile load is a throughput question, and
  B200 data shows the binding constraint for these kernels is the fixed
  `num_warps=2` in the default candidate set anyway. Note the *nested* path's
  `min_xblock` (`simd.py:3144`) is different and is load-bearing: it picks
  whichever axis carries the grouped reduction.

- **Tiling is decided once.** `_sub_parent_tiling_is_2d` declines the fusion if
  the parent's heuristic wants a y/z tiling, and codegen then *forces*
  `create_tiling([numel], [rnumel])` rather than re-running the heuristic — so
  the fusion-time answer and the codegen-time answer cannot disagree. The
  `assert len(kernel.range_trees) == 2` remains as an invariant.

- **Tiling and index-width analysis intentionally see different schedules.**
  `SIMDKernelFeatures` receives the parent reduction schedule for tiling and the
  combined parent-plus-epilogue schedule as `indexing_node_schedule`. Thus the
  derived epilogue cannot alter the parent grid, but `select_index_dtype()` sees
  every buffer and index expression that the emitted kernel will access.

- **`_is_sub_parent_shaped` in `_sub_parent_epilogue_decision` is load-bearing.** A
  full-resolution reader of an epilogue output could join the group by generic
  numel matching, leaving a half-resolution member that generic coalescing
  analysis cannot model (`expected pointwise sizes to match pointwise_numel *
  red_numel`). Any refactor of the leaf gate must preserve it. Once a standalone
  group forms, later same-kind fusions must replan the combined leaves before
  preserving its identity. Generic `codegen_node` rejects staged identity; it
  must arrive through `codegen_staged_reduction`.

- **Rejections are silent** — ~14 `return None` paths, none logged.

## Test coverage

The current review state passes the full file (`268 passed, 1 skipped`) and
the four nested-reduction scheduler tests. Positive tests use `torch.randn`, so
an even/odd lane swap is caught. Structural tests pin direct
`FusedStagedReduction` identity, combo exclusion, benchmark bypass and rejection
of an incompatible same-group reduction.
The NVFP4 inline-asm tests remain SM100-gated, but non-asm tests exercise the
scheduling and codegen path on older GPUs.

Dynamic behavior is covered with actual runtime shape changes. One compiled
graph accepts three dynamic batch sizes and three even dynamic reduction sizes
(`510`, `768`, `1022`). A GELU producer test accepts feature sizes `512`, `768`,
and `1024` in one compiled graph and proves the producer, amax, and pair
epilogue are one kernel. The symbolic-reduction case selects looped codegen;
persistent symbolic-index coverage comes from the GELU case's fixed `R=16` and
dynamic row stride. Static non-power-of-two R is covered by `G=24`.
Divisibility is explicit: the plan requires
`parent_rnumel == factor * FloorDiv(parent_rnumel, factor)`. A positive
`cpp_wrapper` test performs a real compile and checks the staged kernel;
non-Triton backend behavior does not have a direct test.

Kernel-form tests also pin the generated source contract. The plain sub-parent
case has three input load sites when looped and one when persistent, with three
stores; the NVFP4 and pointwise-producer forms similarly have three/one input
loads and two stores. Dynamic R is necessarily looped here and pins three loads,
three stores, no `tl.split`, and `min_rblock=2`. Persistent forms require one
`tl.split`. The known looped full-resolution-sibling form currently takes three
passes; its exact load count is deliberately not pinned, and a TODO calls for
reducing it to two passes.

## Changed during the stack rewrite

- The static power-of-two `parent_rnumel` gate was deleted. The plan now accepts
  any extent provably divisible by factor 2, normalizes symbolic
  storage strides through the shared `sub_parent_extent_subs`. `min_rblock`
  stays at factor 2 for both kernel forms; an eligible persistent reduction
  selects its full padded RBLOCK through the ordinary heuristic. The factor is
  deliberately fixed at `INTERLEAVED_SUB_PARENT_FACTOR = 2` in this commit.
- `min_xblock` could produce a non-power-of-two floor, tripping `check_config`'s
  `TRITON_MAX_BLOCK["X"] % XBLOCK` assert. Reproduced at `B = 3/5/6` as a hard
  compile failure. Ultimately **deleted from this path** rather than clamped: it
  was a throughput heuristic riding on a legality mechanism, and removing it
  takes the whole non-power-of-two class with it, along with the
  `CONTIGUOUS` carve-out and a dynamic-shape hint dependence. `(3, 16, 16)` was
  added to `test_standalone_sub_parent_epilogue` and two kernel-form tests now
  assert the floor's absence.
- The `_is_sub_parent_shaped` leaf-gate fix above, typed staged identity, and
  the assertion that staged nodes never reach generic `codegen_node`.
- Eager persistent-only source materialization was replaced by one lazy load
  resolver. Parent values are projected only if they remain in CSE after the
  parent stage; otherwise the existing derived-index load path is used. The
  same `_codegen_remapped_pointwise` emitter now serves nested and sub-parent
  stages.
- Tiling moved from a post-hoc assert to a fusion-time decision, with codegen
  forcing the 2D tiling so the two cannot drift.
- `StagedReductionPlan` now represents both independently-supported forms:
  pre-existing nested reductions and standalone sub-parent epilogues.
  `FusedStagedReduction` is the common codegen identity;
  `FusedNestedReductions` remains its permanent subclass. Empty, combined and
  multiple-sub-parent states are rejected in this commit and widened only when
  a later commit emits them.
