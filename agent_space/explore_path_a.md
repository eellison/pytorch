# Path A Exploration: BlockLocalReduction IR Primitive

## Key context (read first)

The assigned worktree `pytorch_nested_reduction_ir_explore` was **not** a clean
baseline. It already contained substantial uncommitted work that constitutes
a "Path B+" (scheduler-owned plan dataclasses), per
`/data/users/eellison/pytorch/agent_space/nested_reduction_ir_explore_comparison.md`.

This report therefore breaks costs into three parts:

1. **Path B+ already in place** (existing uncommitted work in the worktree).
2. **Path B+ → true IR primitive (Path A delta)**.
3. **Total: codegen-only baseline → true IR primitive**.

## What I implemented (delta on top of Path B+)

In `/data/users/eellison/pytorch_nested_reduction_ir_explore`:

1. **`torch/_inductor/ir.py`**: added `BlockLocalReduction(Reduction)` class
   (~50 lines). Subclass of `Reduction` carrying `group_size`, `small_dim_in_r`,
   `parent_reduction_size` metadata. Compiles cleanly. **Not yet wired into
   any lowering path** — just the IR shape.
   File: `torch/_inductor/ir.py:2105-2151`.

2. **`torch/_inductor/fx_passes/nested_reduction.py`**: added a stub
   `maybe_recognize_nested_reductions(gm)` graph-rewrite pass (~160 lines).
   Walks FX nodes, looks for `Reduction(view(x, [..., G, g]))` patterns,
   and tags the reduction's FX node with
   `node.meta["block_local_reduction"]` when a sibling reduction over the
   same underlying tensor is detected. **The tagging is real but the
   downstream lowering hook is not wired up.** Not invoked from
   `post_grad_passes` yet.
   File: `torch/_inductor/fx_passes/nested_reduction.py`.

Both files compile (`python -m py_compile` clean).

## What I didn't get to

- Wiring `make_reduction` (lowering.py) to dispatch to
  `BlockLocalReduction.create()` when `V.graph.current_node.meta` has the tag.
- Implementing `BlockLocalReduction.create()` itself (the equivalent of
  `Reduction.create` but stamping in `group_size` / `small_dim_in_r`).
- Replacing scheduler-side `BlockLocalReductionSpec` / `NestedReductionPlan`
  reads with reads off the IR node.
- Replacing `_GroupedReductionOpsHandler.reduction()` with a call routed
  through standard reduction codegen.
- Running any tests (the worktree has no build artifacts; per CLAUDE.md
  I must ask before building).

The unimplemented pieces are not "small wiring." They each carry the real
cost of Path A; see "Hardest challenges" below.

## Hardest challenges (concrete)

### Challenge 1: FX recognition is *graph-aware*, not local

The design doc says recognition becomes "a graph rewrite:
`Reduction(reshape(x, B, G, g))` → `BlockLocalReduction(x, group_size=g, ...)`".
A purely local rewrite is wrong. A `BlockLocalReduction` is only useful if
*paired with an outer reduction* over the same input. Standalone, it is
strictly equivalent to a plain `Reduction`. Eagerly inserting it without
the sibling check produces a special IR node that the scheduler must then
treat indistinguishably from a `Reduction` whenever the pairing fails.

The current scheduler-time `NestedReduction.can_fuse`
(`scheduler.py:445-555`) does ~110 lines of work, including:
- shared-input detection across `read_writes.reads`
  (`scheduler.py:558-566`)
- `is_small_dim_in_r` based on `MemoryDep.ranges` and stride coefficients
  (`scheduler.py:569-625`)
- coalescing analysis via `analyze_memory_coalescing`
  (`scheduler.py:533-549`)

Replicating these at FX level requires walking ancestors through pointwise
chains (mean, var, normalize) to find shared underlying buffers, and
predicting access strides without `MemoryDep` (which is a scheduler-time
construct). My stub in `fx_passes/nested_reduction.py:99-125` only covers
the trivial direct-read case and explicitly cannot determine
`small_dim_in_r`. A real implementation needs either:

- **A partial dependency analysis at FX time.** Doable but non-trivial; FX
  nodes don't carry stride information, and the indexing inside the
  reshape's view is implicit.
- **Defer the full check to scheduler.** The FX pass becomes a "candidate
  marker" and the scheduler still does the legality work — in which case
  the IR primitive is a *rename* of `BlockLocalReductionSpec`.

The honest answer is option 2 by default. Which means the FX rewrite buys
*almost nothing* over Path B+: you've moved a name from scheduler.py to
ir.py and `nested_reduction.py`, but the legality and pairing logic stays
where it is.

### Challenge 2: Codegen complexity does not collapse

`TritonKernel.reduction()` (`triton.py:4211`) reduces over
`self.range_trees[-1]`. It cannot do "reshape, then reduce over an
interior dim." For a `BlockLocalReduction` to be emitted via the *standard*
reduction codepath (the design doc's claim that
"_GroupedReductionOpsHandler.reduction() becomes standard reduction emit"),
one of these must happen:

- **A. Add an extra range tree for `g` and reduce over it.** The
  range-trees list becomes `[X, R, G]` and reduction reduces over `G`
  while a separate codegen step reduces over `R`. This requires:
  - Range-tree construction changes in `SIMDKernel.__init__`
  - Triton kernel header generation for the extra tree
  - A new "reduce over a non-trailing tree" code path in
    `TritonKernel.reduction()` (~50-100 LOC, plus testing across
    persistent/loop reductions)
  - Index expression rewrites everywhere that assumes the trailing tree
    is the reduction dim
- **B. Special-case `BlockLocalReduction` in
  `TritonKernel.store_reduction`.** Detect the IR-node type, do the
  reshape internally, call `emit_reduce`. This is essentially calling the
  existing `_GroupedReductionOpsHandler.reduction()` from a different
  entry point.

Option A is a real backend extension and would be at least ~200 LOC
across `simd.py` and `triton.py`, with significant testing risk
(persistent reductions, loop reductions, masking, dtype handling all need
to be re-validated for non-trailing reductions).

Option B is the honest minimum, and it means **the special handler does
not dissolve**. It just gets called from a different place. Saved LOC in
simd.py: maybe 50 (the handler's class wrapping disappears, the reshape
helper stays).

### Challenge 3: Iteration-family infrastructure is orthogonal

`_DerivedIterationFamily` (`simd.py:1454-1557`, ~108 LOC) and
`DerivedIterationRangesRoot` (`simd.py:381-427`) exist for *consumer-side*
remapped pointwise execution: full-resolution epilogues that consume node1's
register values and half-resolution NVFP4 consumers. These are
**downstream** of the grouped reduction itself.

The IR primitive does nothing for these. The non-local-optimum doc's
Path B and IR-uplift's Path A both explicitly preserve
`_DerivedIterationFamily`. So all of:

- `_DerivedIterationFamily` (108 LOC)
- `DerivedIterationRangesRoot` (~46 LOC)
- `_PointwiseRemapHandler` (~50 LOC)
- `make_reduced_output_family`, `make_full_resolution_family`,
  `make_half_resolution_family` on `_GroupReductionLayout`
- `_codegen_group_reduction_epilogue`, `_codegen_half_resolution_consumers`

stay in `simd.py` regardless.

### Challenge 4: Scheduler-fusion-time pairing still has to happen

`FusedNestedReductions` (`scheduler.py:2568`+) is the FusedSchedulerNode
that pairs the outer reduction with the BlockLocalReduction. Even with
an IR-level `BlockLocalReduction`, the scheduler still has to:
- Decide the outer reduction it pairs with
- Decide whether half-resolution consumers fuse in
- Build the fused-node membership

The Path B+ work moved this logic into `FusedNestedReductions._build_plan`.
The IR primitive does not eliminate this — it gives the scheduler a
better-typed input (`isinstance(reduction.data, BlockLocalReduction)`
instead of running `is_small_dim_in_r` at fusion time), but the
fused-node still has to be built.

### Challenge 5: Dynamic shapes

`BlockLocalReduction.group_size` is `sympy.Expr`. Today the codegen-only
path's `NestedReduction.can_fuse` rejects when `rnumel2` isn't a static
power-of-2 int (`scheduler.py:501-508`). At IR rewrite time, we may have
`group_size = SymInt(g)` where `g` only specializes later. Either:
- The FX rewrite specializes group_size eagerly (regresses dynamic shapes)
- Or `BlockLocalReduction` carries a symbolic group_size and codegen has
  to handle the non-pow2 / non-static cases that the codegen-only path
  currently rejects outright.

Neither is implemented. The design doc flags this as a "hard part" but
doesn't propose a resolution. My read: this is a real cost — adding
dynamic-shape tolerance to the IR-level rewrite is non-trivial.

## LOC estimate for full implementation (Path B+ → IR primitive)

A minimum-viable IR primitive that actually replaces `BlockLocalReductionSpec`:

| File | Adds | Removes | Net |
|---|---|---|---|
| `torch/_inductor/ir.py` | +120 (BlockLocalReduction + create()) | 0 | +120 |
| `torch/_inductor/lowering.py` | +30 (dispatch in make_reduction) | 0 | +30 |
| `torch/_inductor/fx_passes/nested_reduction.py` | +250 (real graph-aware rewrite) | 0 | +250 |
| `torch/_inductor/fx_passes/post_grad.py` | +5 (call site) | 0 | +5 |
| `torch/_inductor/scheduler.py` | +20 (read from IR instead of plan) | -50 (BlockLocalReductionSpec, parts of _build_plan) | -30 |
| `torch/_inductor/codegen/simd.py` | +30 (route via BlockLocalReduction) | -50 (handler simplification, option B above) | -20 |
| **Total** | **+455** | **-100** | **+355** |

If you instead pursue option A in Challenge 2 (real backend extension to
reduce over interior range trees), add ~200 LOC across `simd.py` and
`triton.py` and significant testing surface. That bumps the total to
~+550 net, but with a *real* dissolution of `_GroupedReductionOpsHandler`
(saving ~90 LOC in simd.py).

For comparison: the codegen-only original adds ~1500 LOC of nested-reduction-
specific machinery in simd.py, of which Path B+ has already moved ~289 LOC
into scheduler. Path A as described above adds another ~+355 LOC, primarily
in `fx_passes/nested_reduction.py`.

**Net: Path A doesn't reduce total LOC. It redistributes ~50-100 LOC from
simd.py into ir.py/fx_passes, while adding ~250 LOC of FX recognition.**

## What would dissolve from the current branch

Conservative estimate (option B in Challenge 2):

- `BlockLocalReductionSpec` (`scheduler.py:662-669`): replaced by IR node,
  ~10 LOC.
- Parts of `FusedNestedReductions._build_plan` that compute group_size /
  small_dim_in_r (`scheduler.py:2598-2622`): can read from IR node,
  ~25 LOC.
- `NestedReduction.is_small_dim_in_r` and the bulk of `can_fuse`'s pattern
  shape detection: subsumed by FX rewrite, ~80 LOC. *But* the legality
  predicates (cpp_wrapper guard, GPU check, coalescing analysis) stay
  somewhere — maybe in the scheduler's pairing logic.
- The `_GroupedReductionOpsHandler` class boundary
  (`simd.py:1926-2014`, ~90 LOC): becomes a method on
  `BlockLocalReduction.codegen()` or a standard reduction with metadata —
  but the *body* stays.

Aggressive estimate (option A): add the +200 LOC backend change, then
`_GroupedReductionOpsHandler` truly dissolves (~90 LOC saved in simd.py).

## What would survive unchanged

- `_DerivedIterationFamily` (`simd.py:1454-1557`, 108 LOC) — consumer-side
  remapping is orthogonal to the producer-side IR primitive.
- `DerivedIterationRangesRoot` (`simd.py:381-427`) — same reason.
- `_PointwiseRemapHandler` (`simd.py:2017-2065`) — same reason.
- Half-resolution consumer discovery
  (`scheduler.py:2677-2785`, ~110 LOC) — ancestor BFS from
  scheduler-buffer state, not derivable at FX time.
- `_codegen_group_reduction_epilogue`,
  `_codegen_half_resolution_consumers` orchestration in
  `simd.py:2809-3300` (~500 LOC) — stays; this is where the iteration
  families execute.
- The full-resolution / half-resolution NVFP4 codegen support — IR
  primitive doesn't touch this.

## Backend implications

- **Triton**: option B above requires no backend changes; option A
  requires teaching `TritonKernel.reduction()`
  (`triton.py:4211-4400+`) to reduce over a non-trailing range tree,
  plus adjusting masking and persistent-reduction code paths. ~150-200
  LOC, plus regression testing of every existing reduction codepath.
- **CPP**: nested reduction is currently disabled for CPP. Adding a true
  IR node makes this *more* expensive to extend to CPP later, because the
  CPP backend would need its own dispatch for `BlockLocalReduction`.
  Today, the CPP backend doesn't even see the construct (it's gated in
  `NestedReduction.can_fuse`). With Path A, gating happens at lowering
  time and CPP must explicitly reject or handle the IR node — either is
  straightforward but it is *new* surface area on every backend.

## Tests I ran and what passed

None. The exploration worktree has no build artifacts and CLAUDE.md
forbids initiating a build without explicit user approval. Per the task
prompt: "It is fine if you don't get the test passing."

What was validated:
- `python -m py_compile torch/_inductor/ir.py` — clean
- `python -m py_compile torch/_inductor/fx_passes/nested_reduction.py` — clean
- `python -m py_compile torch/_inductor/scheduler.py
  torch/_inductor/codegen/simd.py` — clean (Path B+ baseline preserved).

Per `nested_reduction_ir_explore_comparison.md`, the Path B+ state itself
was previously validated with `python test/inductor/test_nested_reduction.py`
(passed).

## Honest cost estimate

**Path B+ → true IR primitive: 2-4 weeks of focused work** for an
experienced Inductor contributor.

Dominant cost drivers, in order:

1. **FX-level recognition that's actually correct** (~50% of effort). The
   tagging step is easy; the legality replication is the hard part. Either
   replicate `MemoryDep`-style stride analysis at FX time, or accept that
   the FX pass is a candidate-marker and legality stays in the scheduler.
2. **Codegen routing** (~25%). The handler doesn't actually dissolve
   without a real backend change (option A in Challenge 2). Without that,
   you're moving complexity around rather than eliminating it.
3. **Dynamic shapes / symbolic group_size** (~15%). The codegen-only path
   sidesteps this by rejecting non-static cases at fusion time. An
   IR-level rewrite has to either match this rejection or actually
   support symbolic group_size — the latter is a real piece of work.
4. **Backend dispatch hygiene** (~10%). Every backend (Triton, CPP, MPS,
   Halide) gets a new IR node to reject or handle.

If option A (true backend extension) is in scope, multiply by 2x and add
significant regression-testing risk.

## Recommendation

**Do not pursue Path A in the same window as the current landing.**

Three reasons:

1. **Path B+ is a strong baseline.** It already moved the
   semantically-meaningful pieces (group_size, small_dim_in_r,
   epilogue partitioning, internal-buffer ownership, persistent-reduction
   prediction) into the scheduler. The codegen-only original was 1500 LOC
   of pattern-specific machinery in simd.py; Path B+ has reduced
   simd.py by 289 LOC. The remaining simd.py code is mostly
   `_DerivedIterationFamily`-shaped and is *not addressed* by Path A.
2. **Path A's payoff is mostly cosmetic.** Without option A's real
   backend extension, the LOC delta is roughly neutral (+355 net),
   redistributed from scheduler/simd into ir/fx_passes. The *type system*
   improves — `BlockLocalReduction` is a better concept than
   `BlockLocalReductionSpec` — but the executable behavior is unchanged.
3. **Path A's payoff that is *not* cosmetic requires option A**, which is
   a substantial backend change (~+200 LOC, full reduction codepath
   regression testing). That is a real piece of work and should be
   scoped on its own merits, not bundled with nested-reduction landing.

Conditions under which Path A becomes worth pursuing:

- **A second fusion pattern lands that needs a similar IR primitive.**
  The IR-uplift doc's Path D ("generalize beyond reductions") is the
  honest motivation: after 2-3 patterns, the cost per pattern drops if
  the IR-level scaffolding exists. With only nested reduction, the
  amortization is zero.
- **Dynamic shape support for nested reduction becomes a hard requirement.**
  The current static-rnumel2 gate is acceptable for FP8/NVFP4 quantization
  (group sizes are 64/128, fixed). If a use case needs symbolic group
  sizes, IR-level handling is a cleaner home than scheduler-time
  static-int gating.
- **Backend extension (option A) lands separately for unrelated reasons.**
  E.g., if cooperative reductions or split-block reductions need
  reduce-over-interior-tree anyway, Path A becomes free to layer on top.

Until one of those conditions holds: **land Path B+, defer Path A.** The
incremental value is small relative to the implementation cost and
testing risk.

## Files modified in this exploration

- `/data/users/eellison/pytorch_nested_reduction_ir_explore/torch/_inductor/ir.py`
  (added `BlockLocalReduction`, lines 2105-2151)
- `/data/users/eellison/pytorch_nested_reduction_ir_explore/torch/_inductor/fx_passes/nested_reduction.py`
  (new file, stub graph rewrite, 161 lines)
