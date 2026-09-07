# Nested Reduction: Design Space

A complete catalog of where we've been, where we are, and where we
could go. Written for a reader unfamiliar with the work but familiar
with PyTorch / Inductor at a high level.


---

## Part 1: Background


### 1.1 What "nested reduction" means here

A common pattern in modern numerical kernels is two reductions stacked
back-to-back over the same input, with the second reduction operating
on a reshape of the first reduction's working tile.

Canonical examples:

**Pattern A — RMSNorm + per-block amax (NVFP4 / FP8 quantization prep):**

```python
def f(x, weight):
    x = F.rms_norm(x, (D,), weight)              # outer reduction (norm)
    x_groups = x.view(B, D // G, G)
    amax = x_groups.abs().amax(dim=-1)           # grouped reduction
    scale = (amax / fp8_max).clamp(min=1e-12)
    x_fp8 = (x_groups / scale.unsqueeze(-1))     # full-resolution use of x
                .to(torch.float8_e4m3fn)
    return x_fp8.view(B, D), scale
```

The semantics: normalize the row, then for every group of G elements
compute the max-of-abs (one `scale` per group), then quantize the row
using those per-group scales.

**Pattern B — LayerNorm + per-block amax** (similar shape, different
norm). **Pattern C — NVFP4 packing**, where after the amax/scale the
row is split into even/odd lanes and packed two-at-a-time into bytes
(this is the "half-resolution" case).

What's special about these patterns is that **the second reduction
reads exactly the same data the first reduction already brought into
registers**. Without fusion, you write the post-RMSNorm tile to memory,
read it back, do the amax, write the scale, read both x and scale back,
quantize. With fusion: read x once, do everything in registers, write
scale and quantized output.

The performance impact is real: the vLLM RMSNorm + FP8 quantize fusion
is 2–3× faster than the hand-written CUDA kernel; NVFP4 reaches within
~10% of hand-written.


### 1.2 Why this is hard for standard Inductor fusion

PyTorch Inductor's standard fusion machinery does a great job of
fusing pointwise + reduction (epilogue fusion) and pointwise +
pointwise (loop fusion). It does *not* natively express "two
reductions over the same data with intermediate ops between them in a
single kernel."

Reasons this is hard for the standard pipeline:

1. **Different iteration extents.** The outer reduction's body
   iterates over `[B, D]` producing `[B]` outputs. The grouped
   reduction's body iterates over `[B, D//G, G]` producing
   `[B, D//G]` outputs. Standard fusion picks one iteration extent
   per kernel.

2. **Multi-stage value flow.** The post-RMSNorm tile has to live in
   registers across the kernel, then get reshaped (logically) for the
   grouped reduction, then potentially used at full-resolution AND
   reduced-resolution by epilogue ops.

3. **Standard reduction codegen reduces over the trailing range
   tree.** It can't natively express "reshape this tile, then reduce
   over an interior axis." That's the operation a grouped reduction
   needs.

So the feature is built as a special path that bends Inductor's
scheduler and codegen to handle the case.


### 1.3 The Inductor stack, briefly

Inductor has roughly four layers where fusion-pattern work *can* live:

| Layer | File | Role |
|---|---|---|
| Inductor IR | `torch/_inductor/ir.py` | `Buffer`, `Reduction`, `Pointwise`, `Loops` — the operations and their data |
| Scheduler | `torch/_inductor/scheduler.py` | Fusion decisions, ordering, scheduler nodes |
| Backend-shared codegen | `torch/_inductor/codegen/simd.py` | Range trees, indexing, kernel-level abstractions usable across Triton/CPP |
| Backend emission | `torch/_inductor/codegen/triton.py`, `cpp.py` | Actual Triton or C++ code strings |

Standard pipeline: lowering produces Inductor IR → scheduler groups
nodes into fused groups → backend codegen emits one kernel per fused
group.

For nested reduction, the question that runs through this whole
document is: **at which of these layers should the special handling
live?**


### 1.4 Anatomy of a nested-reduction kernel (today)

A fused nested-reduction Triton kernel runs in this order:

1. **Outer reduction body** — execute `node1` (e.g. RMSNorm). Keep
   `node1` outputs in registers / CSE when possible.

2. **Group reduction body** — reinterpret the outer-reduction tile as
   a grouped structure. For example, `[XBLOCK, RBLOCK]` becomes
   logically `[XBLOCK, RBLOCK/G, G]`. Reduce over `G`. Produce one
   value per group.

3. **Optional consumers**:
   - **Reduced-resolution epilogues** — pointwise ops on the
     per-group output (e.g. divide amax by 448 to compute scale).
   - **Full-resolution epilogues** — pointwise ops that consume the
     per-group reduced value broadcast back to the original tile
     shape (e.g. divide x by scale to quantize).
   - **Half-resolution consumers** — pointwise ops that operate at
     `parent_extent / 2` (e.g. NVFP4 even/odd packing).

The complication: each of these stages runs in a *different effective
iteration extent*. The outer reduction is at `[XBLOCK, RBLOCK]`. The
reduced-output stage is at `[XBLOCK, RBLOCK/G]`. The full-resolution
stage is back at `[XBLOCK, RBLOCK]`. The half-resolution stage is at
`[XBLOCK, RBLOCK/2]`.

Standard codegen assumes one iteration extent per kernel. Nested
reduction needs four.


### 1.5 The core abstraction: derived iteration families

The current implementation introduces `_DerivedIterationFamily` (in
`simd.py`) to express the multiple iteration extents. A family is
parameterized by:

- A set of range trees (the iteration variables and their extents)
- An optional index substitution map (for the reduced-output case)
- A "remapped values" map (buffer name → CSE variable) for values that
  exist in registers at this family's shape

A family is *activated* around a body's emit:

```python
with reduced_output_family.activate(kernel):
    for ep_sn in reduced_output_epilogues:
        ep_sn._body(...)
```

While the family is active, all loads/stores in the body resolve
through the family's range trees, not the kernel's primary range
trees.

Three families are used today:

- `reduced_output` — `[XBLOCK, RBLOCK/G]`. For epilogues that consume
  one value per group.
- `full_resolution` — `[XBLOCK, RBLOCK]`. For epilogues that need the
  original tile shape, with reduced values broadcast back.
- `half_resolution` — `[XBLOCK, RBLOCK/2]`. For NVFP4-style consumers
  that pack two adjacent values.


### 1.6 The current pieces

The implementation today has these moving parts (locations refer to
`single_commit_wip` baseline):

**Scheduler:**
- `NestedReduction.can_fuse` (`scheduler.py:~445`) — recognizes the
  pattern: two reductions sharing a common input, with the second
  reading a reshape of the first's output.
- `FusedNestedReductions(FusedSchedulerNode)` (`scheduler.py:~2535`) —
  the fused scheduler node wrapping node1 + node2.

**Codegen entry point:**
- `SIMDScheduling.codegen_nested_reduction(node)` (`simd.py:~2625`) —
  ~200-line orchestrator that drives the multi-stage emit.

**Codegen abstractions:**
- `_DerivedIterationFamily` (`simd.py:~1450`) — the family abstraction.
- `_GroupReductionLayout` (`simd.py:~1553`) — layout metadata for the
  grouped reduction (group_size, parent_axis, reshape_shape, etc.).
- `_GroupedReductionOpsHandler` (`simd.py:~1919`) — `WrapperHandler`
  subclass that runs during the reduction body's emit. Does the
  reshape-then-reduce, captures register values for downstream stages,
  routes the store through the reduced-output family.
- `_PointwiseRemapHandler` (`simd.py:~2017`) — `WrapperHandler` for
  consumer epilogues. Resolves loads from the family's remapped_values
  or from `cse.store_cache`.
- `DerivedIterationRangesRoot` (`simd.py:~381`) — subclass of
  `IterationRangesRoot` that backs derived families.

**Backend:**
- `TritonKernel.reduction()` (`triton.py:~4211`) — standard reduction
  emit. Doesn't know about `BlockLocalReduction` (no IR primitive
  exists yet).
- Various smaller changes: `override_mask` plumbing, `has_rmask`
  shape-aware, `_mask_name_for_symbol` helper.

This is a lot of moving parts in `simd.py` (and some in `triton.py`).
The total feature is ~+1500 LOC in `simd.py` alone. Hence the question
running through this doc: **where should this work live, ideally?**


### 1.7 Vocabulary used in this doc

- **Codegen-owned**: pattern recognition, layout, family construction,
  consumer dispatch all live in `simd.py` / `triton.py`.
- **Scheduler-owned**: pattern recognition + family/plan decisions live
  on the `FusedNestedReductions` scheduler node; codegen reads the plan.
- **IR-owned**: the operation itself (`BlockLocalReduction`) is an
  Inductor IR primitive; standard reduction codegen handles it.
- **Iteration family**: `_DerivedIterationFamily` in codegen — runs a
  pointwise body in a remapped extent (reduced-output, full-resolution,
  half-resolution).
- **Special handler**: a `WrapperHandler` subclass (`_GroupedReductionOpsHandler`,
  `_PointwiseRemapHandler`) that intercepts load/store/reduction during a
  stage's body emit.
- **Path B+**: scheduler-owned grouped-reduction plan. The scheduler's
  `FusedNestedReductions` carries a `NestedReductionPlan` with all
  pattern facts; codegen reads it instead of re-deriving.
- **Path A**: lift the reduction itself to an Inductor IR primitive
  (`BlockLocalReduction`). Standard reduction codegen learns to handle
  it.


---

## Part 2: Where we've been


### 2.1 Historical iterations

These are concrete branches that lived at some point. They informed
the current shape but were superseded. Listed roughly chronologically.

#### A1. Original spike — `nested_reduction_backup`

Where everything started. Pattern detection, layout, handlers all
inline in codegen. Multiple specialized handlers per pass (`pass1`,
`pass2`, `pass3` corresponding to outer reduction, grouped reduction,
half-resolution). No abstraction over what they shared.

This was the working prototype but was stylistically unsustainable —
every new pattern (full-res epilogue, half-res NVFP4) added another
handler.

#### A2. Derived-range probes — `nested_reduction_derived_range_attempt`, `derived_try_from_range`, `derived_try_from_range_v2`

First attempts at lifting iteration ranges into a shared abstraction.
Tried various ways to teach the kernel about derived ranges: a kernel
flag indicating "we're now in stage X," explicit range-tree swap APIs,
hierarchical range trees.

Most were abandoned because the kernel-internal indexing assumed range
trees were flat; making them hierarchical broke things downstream.
The descendant of these attempts is `DerivedIterationRangesRoot` plus
the `use_range_trees()` context manager — a flat swap rather than
hierarchical extension.

#### A3. Half-res derived try — `nested_reduction_halfres_derived_try`

Attempt to unify half-resolution with the rest via derived ranges.
Showed that the half-res case has different constraints
(lane-validity mod 2 — i.e. consumer reads must land cleanly on lane 0
or lane 1, not crossing them) that didn't fit the simple
"scaled-down range tree" model. Half-res stayed specialized but the
range tree machinery used to express it became more general.

#### A4. Semantic stack — `nested_reduction_semantic_stack`, `_v2`, `_v3`

Tried structuring the codegen as an explicit stack of stages with
explicit transition rules. The motivation: the imperative orchestrator
in `codegen_nested_reduction` is hard to reason about; a declarative
stack would make the stages and their dependencies explicit.

More verbose than the imperative orchestrator that won out. Abandoned
for being heavy without clear payoff.

#### A5. Unify probes — `nested_reduction_unify_v2`, `unify_derived`, `unify_probe`

Tried to collapse `_ReducedOutputSpace` and `_HalfResolutionSpace`
(earlier separate dataclasses) into one type. This eventually
succeeded — the current `_DerivedIterationFamily` is the descendant.
But intermediate attempts mixed in other refactors and got abandoned.

#### A6. Restack attempts — `nested_reduction_restack`, `restack_work`, `restack_commitwise`

Attempts to commit-split the work for review. Each tried different
splits (by file, by feature, by phase). The current `core_candidate`'s
2-commit split (plumbing vs feature) is the descendant of these
experiments.

#### A7. Merged attempt — `nested_reduction_merged_attempt`

A consolidation attempt before `single_commit_wip` happened. Showed
that consolidation was achievable but was itself superseded.

#### A8. Path 3 prototype — `nested_reduction_path3`

The first scheduler-ownership probe. Moved only half-resolution
discovery to the scheduler. Built on a stale baseline; per
`agent_space/explore_path_b.md`, has regressions vs current main and
~14h of cleanup is needed to reach baseline parity. Its design
insight (scheduler-owned half-res discovery) lives on in B3 below
(Path B+).

The other branches you'll see in `git worktree list`
(`commit_prep`, `pre_restack_*`, `non_local_probe`, etc.) are
intermediate save points or alternative restack attempts. They don't
represent distinct design positions.


### 2.2 What we learned from the iterations

Three things emerged from the historical churn that anchor the current
shape:

**1. The consumer side wants one abstraction.** Reduced-output,
full-resolution, half-resolution all execute pointwise bodies in
remapped iteration extents. The differences are *which extent* and
*how values from upstream stages are reshaped to fit*. One
`_DerivedIterationFamily` parameterized by these is cleaner than
three handlers.

**2. The reduction side is harder to fit a clean abstraction.** The
grouped reduction is genuinely different from a standard reduction
(reshape-then-reduce, not just reduce). Various attempts to express it
through standard codegen primitives failed; the current implementation
specializes it via `_GroupedReductionOpsHandler`.

**3. The orchestration is what makes the feature feel heavy.**
`codegen_nested_reduction` is ~200 lines of explicit multi-stage
emit. It's the function that knows "node1 first, then group reduction
body, then reduced-output epilogue, then full-resolution, then
half-resolution." Each stage has its own handler swap and family
activation. This orchestrator is the load-bearing thing — and it's the
hardest thing to dissolve into standard codegen.


---

## Part 3: Where we are


### 3.1 Current candidate landing targets

These are live worktrees with working tests. Each represents a
different answer to "what should we ship?"

#### B1. Baseline / `single_commit_wip` (= `nested_reduction_with_nvfp4_saved`)
**Codegen-owned, full feature.**

The all-in version. Everything described in Part 1 lives in codegen.

- `NestedReduction.can_fuse` recognizes the pattern in scheduler.
- `FusedNestedReductions` is just a fused-node wrapper with
  `small_dim_in_r` axis classification.
- `codegen_nested_reduction` re-derives layout, axis classification,
  internal-buffer decisions, half-res discovery (twice — early/late).
- `_GroupedReductionOpsHandler` does the reduction's reshape+reduce.
- `_PointwiseRemapHandler` runs consumer epilogues with family
  activation.
- `_DerivedIterationFamily` is the consumer-side abstraction.

Tests: 80/80. ~+2793 LOC vs main, ~+1469 in `simd.py`.

**Pros:** working, full feature, well-tested. Clean
`_DerivedIterationFamily` abstraction. NVFP4 perf result included.

**Cons:** big PR, codegen carries layer-1/layer-2 work, early/late
dual pass is a code smell (two BFS passes for half-res discovery
guarded by a runtime `RuntimeError` if they disagree), `simd.py`
footprint hard to review.

**Use when:** want to ship the full feature with no architectural
argument needed (it's just bigger).

#### B2. `core_candidate` / `core_landable`
**Codegen-owned, trimmed (no half-res / NVFP4).**

Same architecture as B1 minus the half-resolution path. The half-res /
NVFP4 codegen is removed; the saved branch
`nested_reduction_with_nvfp4_saved` preserves it for follow-up.

Two reviewable commits:
- Commit 1: backend plumbing (triton.py, runtime, config, metrics).
- Commit 2: scheduler + simd.py + tests.

Tests: 76/76 (4 NVFP4 tests removed). ~+2118 LOC vs main, ~+1008 in
`simd.py`.

**Pros:** smaller PR, cleaner commit split, NVFP4 follows separately.
Reviewer doesn't have to swallow the half-res complexity in the first
PR.

**Cons:** loses the marquee NVFP4 perf result on this PR; same
architectural shape as B1.

**Use when:** "ship soon" dominates and NVFP4 can wait.

#### B3. `path_b_plus_candidate` (just created from `ir_explore`)
**Scheduler-owned grouped-reduction plan, full feature.**

The architectural improvement that's actually achievable on top of
B1. Adds three dataclasses to scheduler.py:

```python
@dataclass(frozen=True)
class BlockLocalReductionSpec:
    reduction_node: SchedulerNode
    output_name: str
    group_size: sympy.Expr
    small_dim_in_r: bool
    requires_persistent_reduction: bool

@dataclass(frozen=True)
class NestedReductionPlan:
    group_reduction: BlockLocalReductionSpec
    reduced_output_epilogues: tuple[SchedulerNode, ...]
    full_resolution_epilogues: tuple[SchedulerNode, ...]
    half_resolution_consumer_names: tuple[str, ...] = ()
    internal_node1_outputs: tuple[str, ...] = ()
    late_internal_outputs: tuple[str, ...] = ()
```

`FusedNestedReductions._build_plan()` computes the plan at fusion
time. `codegen_nested_reduction` reads from `node.plan` instead of
re-deriving.

What dissolves from `simd.py` vs B1:
- Pattern recognition (which buffers are internal, fullres vs reduced)
- Axis classification re-derivation
- Shared-reads detection re-derivation
- Persistent-reduction prediction
- Internal-buffer ownership checks
- Early/late half-res discovery passes (and the `RuntimeError`
  agreement check)

What stays in codegen: range tree manipulation, family activation,
body emit. The reduction handler stays
(`_GroupedReductionOpsHandler`). The consumer machinery stays
(`_PointwiseRemapHandler`, `_DerivedIterationFamily`).

Tests: 80/80. ~+2683 LOC vs main, ~+1234 in `simd.py`
(simd.py is **−235 vs Baseline** for the same scope).

**Pros:** real architectural improvement, full feature kept, clean
layering between scheduler semantics and codegen execution, no
early/late dual pass. Sets up Path A (Inductor IR primitive) without
committing to it.

**Cons:** bigger PR than B2 (+655 LOC), introduces new scheduler
abstractions that reviewers must accept, two commits split by *time*
(squashed feature + refactor) rather than by *scope* (plumbing vs
feature).

**Use when:** want to ship the full feature with the architectural
improvement, willing to argue for the new scheduler abstractions in
review.

#### B4. Trimmed B3 (would-be `path_b_plus_core_candidate`)
**Scheduler-owned plan, no half-res.**

Doesn't exist yet. Would be: take B2 (core_candidate scope) and apply
the B3 refactor (scheduler-owned plan) on top. Estimated ~1 hour to
produce; would have ~+2028 LOC total but `simd.py` around ~+773
(−235 vs B2).

Could be 3 commits: plumbing / feature / refactor. Each
independently reviewable.

**Pros:** smallest PR with architectural improvement, half-res follows
on top, clean commit split.

**Cons:** doesn't exist yet; needs to be built. NVFP4 follow-up has to
land on top of the refactored shape, slightly more work than landing
NVFP4 on B2.

**Use when:** want both the smaller scope of B2 and the architectural
improvement of B3. Probably the best of both worlds *if* the 1 hour
of work to produce it is worth it.


### 3.2 What's preserved regardless of choice

Listed once because every option keeps these:

- `_DerivedIterationFamily`: every option keeps it. It's the right
  shape for "run a body in a remapped iteration extent."
- `DerivedIterationRangesRoot`: same.
- `NestedReduction.can_fuse` recognition logic: lives in scheduler in
  every option. (FX-level recognition has been ruled out as not viable
  without `MemoryDep` access — see C1 / C2 below.)
- Half-resolution discovery BFS: lives in scheduler (B3 onward).
- Tests, form-checks, the `triton.nested_reduction` config flag.
- Saved branches: B1 preserved as `nested_reduction_with_nvfp4_saved`
  and as a 3461-line patch snapshot at
  `agent_space/snapshots/nested_reduction_with_nvfp4_03baa04.patch`.


---

## Part 4: Where we could go


### 4.1 Future design choices

These aren't built but are coherent design points that have been
worked through in detail.

#### C1. Minimal Path A — IR primitive for reduction only

Add `BlockLocalReduction(Reduction)` IR class. Scheduler rewrites
node2's reduction IR to it at fusion time. `TritonKernel.reduction()`
dispatches on IR node type to do reshape-then-reduce.

**Concretely:**

```python
@ir_dataclass
class BlockLocalReduction(Reduction):
    group_size: Expr
    small_dim_in_r: bool

class TritonKernel:
    def reduction(self, dtype, src_dtype, reduction_type, value):
        if isinstance(self.current_node.data, BlockLocalReduction):
            return self._reduction_block_local(...)  # reshape-then-reduce
        return self._reduction_standard(...)
```

**Dissolves:** `_GroupedReductionOpsHandler` (~98 LOC).
**Adds:** IR class (~80), scheduler rewrite (~30), dispatch + lifted
helper (~80), kernel load-capture context (~20).
**Net:** roughly +100 LOC.

**Architectural shape:** mostly Path B+ plus a typed reduction
primitive. Doesn't change the orchestrator, doesn't change
consumer-side code.

**Why it doesn't pay off:** the complexity isn't on the reduction
side. Moves boxes without unlocking anything. Honest assessment: not
worth the churn unless full Path A follows. The reduction handler is
the *smallest* piece of complexity; lifting only that piece doesn't
help much.

**Worth it only if:** a follow-up Path A (C2) is planned, in which
case this is the first installment.

#### C2. Full Path A — no `codegen_nested_reduction` entry

Standard codegen iterates the fused group. `BlockLocalReduction` IR
primitive handled by `TritonKernel.reduction()`. Iteration families
activated automatically when standard codegen sees a tagged consumer.

Conceptually, `codegen_node_schedule_with_kernel` handles everything:

```python
for sn in fused_nodes:
    if sn.iteration_family is not None:
        family = kernel.get_or_build_family(sn.iteration_family)
        with family.activate(kernel):
            sn.codegen()
    else:
        sn.codegen()
```

No `codegen_nested_reduction`. No `_GroupedReductionOpsHandler`.
No `_PointwiseRemapHandler`.

**Dissolves:** the orchestrator (~200 LOC), the reduction handler
(~98), the consumer handler (~50), epilogue codegen functions
(~200), half-res discovery and codegen (~150), plan dataclasses
(~50).

**Adds:** IR primitive, scheduler IR rewrite, family construction
hooks, possibly an `iteration_family` field on `SchedulerNode`,
backend dispatch.

**Hard parts:**
1. Cross-extent fused group membership: half-res consumers run at a
   different iteration extent. Today they're discovered separately
   and emitted in their own pass; in C2 they have to be members of
   the fused group with their family tag. Standard scheduler fusion
   doesn't naturally accept this.
2. Index decomposition through standard codegen: today's
   `_decompose_flat_index` is called explicitly; in C2 it has to
   happen transparently when family-activated.
3. Half-resolution capture lifecycle: today's `_stage_load_values`
   captures register values during the reduction body for downstream
   half-res use. In C2 this needs another home.
4. Dynamic shapes: scheduler's existing `NestedReduction.can_fuse`
   rejects non-static cases. Path A would either inherit this
   rejection or have to handle symbolic group_size.

**Estimated:** 2-4 weeks (Option B — special-case in
`store_reduction`) or 4-8 weeks (Option A — real backend extension to
reduce over interior range tree). Per `explore_path_a.md`.

**Architectural shape:** what the IR-uplift design doc calls Path A.
The "non-local optimum" for nested reduction.

**Worth it only if:** Inductor team has a multi-week budget for this,
OR a second similar fusion pattern (cat/scatter/MLA) is on the
roadmap to amortize the cost.

#### C3. Iteration family annotation only on nested-reduction children

Compromise between B3 and C2: keep `codegen_nested_reduction` entry,
but factor consumer iteration into family-tagged emit so the
orchestrator shrinks. Don't generalize family tags onto every
SchedulerNode.

This is the "minimal-change Path A" earlier discussions explored. The
scope is iteration families remain a nested-reduction-internal concept,
just with a cleaner interface to them.

**Honest read:** even smaller win than C1. Not worth its own
implementation. Listed for completeness.

#### C4. Path D — generalize beyond nested reduction

Treat `_DerivedIterationFamily` as a general "vertical fusion within
one kernel" primitive. Use it for:

- **Cat fusion:** consumers of a concatenation that operate on each
  segment at the segment's iteration extent, sharing a single kernel.
- **Scatter fusion:** consumers of a scatter that operate at the
  scatter's index extent.
- **MLA-style heterogeneous fusion:** multi-tile-shape attention
  computations within a single kernel.

Each pattern has its own producer IR node (`BlockLocalReduction`,
future `BlockLocalCat`, `BlockLocalScatter`, ...) and shares the
consumer-side scheduler annotation + codegen activation.

**Cost:** large, multi-quarter. Justified only by 2-3 customers. Each
customer pattern has its own legality rules and recognition logic —
the abstraction reduces marginal cost per pattern but doesn't make any
single pattern cheaper.

**Why it matters:** if Path A pays for itself, Path D is the
amortization. With only nested reduction, the sunk cost of Path A is
hard to justify.

**Worth it only if:** committed to a roadmap where the abstraction
serves 2+ patterns.

#### C5. Affine masked subregions

Path-4 of `nested_reduction_non_local_optimum.md`. Extend
`_DerivedIterationFamily` to express affine masked extents like:

```
small_col = big_col - 512    valid only when 512 <= big_col < 576
```

Today's families express factorized splits (XBLOCK × G, RBLOCK / 2).
Affine masked extents would let one kernel handle, e.g., a
concatenated tensor where consumers read one segment with index offset
and validity mask.

This is the consumer-side counterpart to C4: same goal of generalizing
to non-reduction patterns, but at the iteration-family abstraction
level rather than at the IR primitive level.

**Cost:** large, blocks on the same lack-of-customer problem as C4.

**Worth it only if:** a pattern lands that needs it.


### 4.2 Quick comparison

| | Architecture | Scope | Tests | LOC vs main | Status |
|---|---|---|---|---|---|
| **B1 Baseline** | Codegen-owned | Full | 80/80 | +2793 | Working |
| **B2 core_candidate** | Codegen-owned | Trim | 76/76 | +2118 | Working |
| **B3 path_b_plus** | Scheduler-owned plan | Full | 80/80 | +2683 | Working |
| **B4 trimmed B3** | Scheduler-owned plan | Trim | n/a | ~+2028 | Not built (~1h) |
| C1 Minimal Path A | + IR primitive | Full | n/a | ~+2783 | Not built |
| C2 Full Path A | IR-owned, no entry | Full | n/a | varies | Not built (2–8 wks) |
| C3 Family-tag only | Halfway | Full | n/a | ~+2700 | Not built |
| C4 Path D | Generalized | Beyond | n/a | huge | Speculative |
| C5 Affine subregions | Family extension | Full+ | n/a | huge | Speculative |


### 4.3 Decision tree

| Constraint dominates → | Pick |
|---|---|
| Ship soon, smaller scope OK | **B2 core_candidate** |
| Ship soon, full feature | **B1 Baseline** |
| Ship architectural improvement, full feature | **B3 path_b_plus** |
| Ship architectural improvement, smaller scope, willing to spend 1h | **B4 trimmed B3** |
| Right architecture, willing to spend weeks | **C2 Full Path A** |
| Long-term Inductor evolution | **C4 + C5 after B-series ships** |


---

## Part 5: Why we are where we are


### 5.1 Why we ended up with codegen-owned (B1) first

Adding an IR primitive in Inductor is multi-week cross-cutting work;
iterating on a codegen feature with a config flag is a few days.
Prototype lived in codegen because that's where features can be added
without architectural change. This is also why iterations A1-A7 all
stayed in codegen — the cost of moving up the stack wasn't worth the
design exploration speed.

The cost of staying in codegen accumulates: each new pattern (full-res,
half-res, NVFP4) added complexity in `simd.py` rather than
distributing it. By the time the codegen-owned design was complete,
the question "should this live higher in the stack" had become loud.


### 5.2 Why Path B+ (B3) is the natural architectural improvement

It moves the work that's *already scheduler-shaped* (pattern
recognition, internal-buffer ownership, persistent-reduction
decisions) to the scheduler, leaving codegen-shaped work (range tree
manipulation, family activation, body emit) in codegen. It's the
layering that the code naturally wants.

It also doesn't require any new Inductor architecture: no IR
primitives, no fused-group fusion changes, no SchedulerNode shape
changes. Just dataclasses on `FusedNestedReductions` that codegen
reads. This is why it can be landed in the same window as the
feature — it's not a separate Inductor project.


### 5.3 Why Path A is harder than expected

The reduction itself is the smallest piece; the orchestration and
consumer machinery are the bulk. An IR primitive only addresses the
reduction. Without changing the orchestration model (which requires
either fused-group fusion of cross-extent consumers or new
scheduler-codegen interface concepts), the IR primitive doesn't unlock
much.

The other dimension: FX-level recognition was the proposal in the
design doc, but it's hard. Recognizing
`Reduction(reshape(x, [..., G, g]))` looks local but isn't:
- A standalone `BlockLocalReduction` is just a regular reduction
  (the value-add comes from co-codegen with a sibling outer reduction).
- The "is there an outer reduction over the same input" check
  requires graph-level ancestor traversal through pointwise chains.
- The "is the group axis along R or X" check
  (`small_dim_in_r`) requires `MemoryDep` stride analysis that lives
  at the scheduler layer.

So even Path A doesn't escape the scheduler — it just adds an IR
primitive plus FX-level marking, with the real legality work still in
the scheduler. The added value over B3 is: a typed IR primitive plus
standard codegen for the reduction. Without backend extension, that's
mostly cosmetic.


### 5.4 Why Path D / affine subregions are blocked

They require either:
- **(a)** A second customer beyond nested reduction to amortize the
  abstraction cost, or
- **(b)** A separate Inductor architecture project funded on its own
  merits.

This branch can't carry that work. Path D is the right move *eventually*,
but only when the second customer is concrete enough to inform the
abstraction. With only nested reduction, the abstraction would be
overfit to one pattern's needs.


---

## Part 6: Reference


### 6.1 Where each option lives

- **B1 Baseline:** `pytorch_nested_reduction_single_commit_wip` (HEAD `03baa04`)
- **B2 core_candidate:** `pytorch_nested_reduction_core_candidate` /
  `pytorch_nested_reduction_core_landable`
- **B3 path_b_plus:** `pytorch_nested_reduction_ir_explore` (branch
  `nested_reduction_path_b_plus_candidate`, HEAD `f257057`)
- **B4 trimmed B3:** would build off B2 + B3's scheduler refactor.
  Doesn't exist yet.
- **C-series:** design only. See `nested_reduction_ir_uplift.md` and
  `nested_reduction_non_local_optimum.md`.

### 6.2 Saved snapshots

Defensive backups in case any worktree state is lost:

- `nested_reduction_with_nvfp4_saved` branch — full B1 state
- `agent_space/snapshots/nested_reduction_with_nvfp4_03baa04.patch` —
  3461-line patch of the full B1 state

### 6.3 Related docs in `agent_space/`

- `nested_reduction_design_overview.md` — the iteration-space model and
  axis classification (small_dim_in_r vs small_dim_in_x). Read first
  if you want to understand *what* nested reduction is at the
  iteration-space level.
- `nested_reduction_full_design.md` — long-form design notes from the
  earlier shape of the feature.
- `nested_reduction_fullres_epilogue.md` — design notes specifically
  for the full-resolution epilogue path.
- `nested_reduction_non_local_optimum.md` — paths 1-5 of consumer-side
  abstraction generalization. Path-3 → "scheduler owns derived
  families" is what B3 partly delivers.
- `nested_reduction_ir_uplift.md` — IR-level design space (paths A-D).
  Path A is C2 in this doc. Path D is C4.
- `nested_reduction_ir_explore_comparison.md` — pre-existing
  documentation of B3 (path_b_plus) before the recent agent
  exploration.
- `explore_path_a.md` — agent's probe of C2 (full Path A).
- `explore_path_b.md` — agent's assessment of A8 (path-3 prototype).
- `path_comparison.md` — focused comparison of B-series candidates
  with concrete LOC numbers.
- `MORNING_READING_GUIDE.md` — quick orientation if the user is
  returning to this work after a gap.


### 6.4 Glossary of file:line locations referenced

In `single_commit_wip` baseline:

- `simd.py:381` — `DerivedIterationRangesRoot`
- `simd.py:1450` — `_DerivedIterationFamily`
- `simd.py:1553` — `_GroupReductionLayout`
- `simd.py:1919` — `_GroupedReductionOpsHandler`
- `simd.py:2017` — `_PointwiseRemapHandler`
- `simd.py:2625` — `codegen_nested_reduction` entry point
- `scheduler.py:445` — `NestedReduction.can_fuse`
- `scheduler.py:2535` — `FusedNestedReductions`
- `triton.py:4211` — `TritonKernel.reduction()`
