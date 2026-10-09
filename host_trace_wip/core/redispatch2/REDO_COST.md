# Redispatch cost: design note (before implementation)

## Where the time goes today
`redispatch()` runs `_redo` once per unselected selector. For each one, `_redo` does the following:
1. Walks the op's recorded call. Every distinct formula becomes a fresh symbol (`env.symbol`, which takes a sha256
   of the name), every root and tensor gets rebuilt, and each formula is lowered to its row (`lo.row`).
2. Builds the dispatch signature. On a memo hit, it re-checks each cached guard by `Ctx.transfer` onto constants.
3. Without a hit, it runs the dispatch.
4. Renames the guards and launches into the variant's context and compares the outputs and allocations (`same()`, pins).

Then `lower_entries` lowers each entry's guards and launches to rows and compiles the program once per call,
and `_add_entries` adds each entry natively.

Torch-only probe on GH200 (`probe/prof_rd.py`, minops84's decode, 36 layers, step272 + RD2): a call that
redispatches one op in every layer spends about 1.0 ms per redo, 35-42 ms per call. Steps 1-2 and 4 are most of it:
`transfer` alone is about 18% (renames plus guard re-checks), and the walk and wrapping (`value`, `root`, `tensor`,
pytree) is about 25%. The attention lane's profile of the fork's context op (about 35 arguments) measured 6-7 ms
per redo with the same shape.

## Proposal: entry templates (one redo per region, instantiated per selector)
The observation is that in N layers, an op's entries differ only in which roots they read. Its argument formulas
(sizes, strides, scalars) are the same variant nodes in every layer, because the layers share the input's symbols.
So are its renamed guards, and therefore its predicate row.

- **Reads, without symbols.** A pure walk of the op's call collects what the redo would read: constants, each
  SymInt/SymBool formula as its variant node, each root's kind/dtype/layout formulas, and each tensor's layout
  formulas. Roots are collected only as positions in the walk.
  - The key is (op/launcher, kind, pytree structure, constants, the formula nodes in walk order, the root and tensor
    layouts by position).
  - No `env.symbol`, no `_TracedTensor`, no `lo.row`.
- **Template.** After a successful redo, the variant stores, under that key:
  - the entry's predicate row;
  - its lowered launches, with each pointer slot's root kept as a position in the walk's root list;
  - its section, if any (temporaries' bytes rows, drops as positions in the op's own allocations, places).
- **Instantiate.** At a later unselected selector (another layer at the same call, or any later call) whose key
  matches, with `values[predicate] == 1` (the row is already evaluated at the call):
  - copy the lowered launches, re-pointing each pointer slot to this op's root at the same position (PointerSlot
    base/root swapped; displacement rows reused);
  - give section temporaries fresh bases;
  - map drops to this op's allocations.
  - No dispatch, no rename, no lowering, no guard transfer. The cost is a dict lookup plus O(slots) dataclass copies,
    then the native `add_entry`.
- **Soundness conditions** (else the full redo runs, as today):
  - the template's guards read no root symbol (an alignment guard on a pointer is per layer);
  - its launches' scalar slots read no root;
  - the key matches exactly, so outputs, allocations and layouts are the same formulas, which `same()` already
    proved for the template's op.
- **Where it lives.** Per variant, on its `_TapeLowering`, the same place as `entry_guards`/`refused`, because
  predicate rows are that program's. It is dropped with the variant. There is no new cross-variant state.
- **Sections.** `check_section` still runs per selector on the instantiated launches. It is cheap, O(slots), and
  checks liveness at that op's own seqs. `section_places` is computed per selector too.
- **Region already holds but no entry.** This is the common N-layers case. Layer 1 builds the template; layers 2..N
  match it and instantiate it in microseconds.

## Expected effect
For N layers sharing an op and region: one ~1 ms (or 6 ms) redo plus N-1 instantiations at ~10-50 us each
(Python copies plus native `add_entry`). Target (a), a cached region costing microseconds, holds whenever the
template's conditions hold. Target (b), N layers doing the work once, holds by construction. Ops whose guards read
roots, such as Triton pointer-alignment specialization on per-layer weights, keep today's path. I'll measure how
often that happens.

## Not proposed
- A cross-variant template cache: rows are per program.
- Skipping `add_entry` by sharing one native entry across selectors: that would need a C++ selector-group concept.
  Revisit only if `add_entry` dominates after this.

## Measurements planned
- µs per redispatch before and after: `probe/prof_rd.py` (GH200), `land/core/attn/tests/prof_redispatch.py`
  (GB300, fork_orig), and the vLLM acceptance run (decode bs 2..32 bucket crossing the fork's split boundary).
- Tests: N layers whose op redispatches together give 1 full redo plus N-1 instantiations (a counter), bitwise.
  A root-reading guard (a Triton alignment specialization) does not instantiate. A section instantiated across
  layers passes check_plan.
