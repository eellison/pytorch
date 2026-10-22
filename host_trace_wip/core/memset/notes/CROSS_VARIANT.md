# Cross-variant reuse of per-op results (design, 2026-10-18; for user review before building)

User constraints (10-16):
- "sharing is fine, it's strictly better".
- "I don't want to have to split up the cudagraph at every cublas."

So sections stay few (pipelining cost model, K <= 4). A per-op result learned in one variant applies in another by node rewrites
inside the section that holds the op. Only a topology change uses a section-level alternative.

## What is already shared (as of step_ms20)

- **Keyed sites (cuBLAS, cuDNN, attention, extern):** shared across variants and replays by the provider's native binding table
  (step_ms20, `NativeBindings`).
  - A binding learned anywhere fills any variant's site row in C++ when its topology is the site's.
  - Another topology is an arm: Python adds the row and a form once per new key.
- **Redispatch results:** `lowering.templates` (`_host_trace_redispatch.Template`, learnfast redo templates keyed by what the redo
  read).
  - A template is per variant (per `Lowering`). Its key `_reads(...)` holds the variant's own IR nodes for SymInts. Its `predicate` and
    `guards` are rows of that variant's program.
  - Another variant's op with the same structure cannot find it, so it redispatches.
  - learnfast measured this on the 27B cold start: 11 of 13 stream redispatches repeat a class learned in another variant.

## What changes

1. **A variant-free template key.**
   - `_reads` gives the key with each SymInt/SymBool as a variant node. A shared key renames those symbols by first occurrence along
     the read walk (alpha-renaming).
   - The walk is deterministic (its roots in reading order), so two ops whose reads differ only in the variants' symbol names get the
     same shared key.
   - Constants, layouts as formulas over the renamed symbols, and pins on static-prefix arguments (`_for_instance`'s value pins) stay in
     the key.
2. **A shared template core**, stored once per replay (in `HostTraceReplay._shared_templates`, and in the per-contract `dispatches`
   that `_redispatch` already passes). It holds the template with:
   - its guards and launch rows as expressions over the renamed symbols and the op's roots, not as rows;
   - its roles, section and drops, as today (they are already positional).
3. **Instantiation in another variant B.** When B's op misses `lowering.templates` but its shared key is in the store:
   - substitute B's symbols for the renamed ones;
   - lower the guards into B's program (new rows, as `lower_entry` adds rows for a redo);
   - build the entry with `_instantiate` at B's roots;
   - add it with `add_entry`.
   No redispatch runs (no capture, no op dispatch).
   - The predicate row is B's own: it is evaluated at B's call like any entry guard.
   - Soundness is per-op guards as today. The entry applies only where its own guards hold at the call. The graph-level guards still
     pick the variant.
   - `verify_redo_templates` (tests) compares each shared instantiation with a full redo, as it does for the per-variant templates.
4. **Topology.** An instantiated entry whose launches match the selector's node topology is an arm-0 entry: node sets in its section,
   with no cut. One that does not (a split-K reduction kernel more, another cluster) is a section entry, as today's redispatch sections
   are:
   - a piece replacing the op's nodes in a clone of the segment;
   - a form of that segment.
   - With the executor split on, a form switch rebuilds that form's sub-execs. Making forms per section (one section's exec per
     alternative instead of the run's) is a follow-up, measured first.
5. **Memory.** The store holds one core per shared key (its rows' expressions, roles, images). Per variant: the entry B adds (rows,
   launches), as a redispatch adds today. No cut, no copy. It is a strict reduction versus today, where each variant stores its own redo
   result.

## Tests (counters, deterministic)

- A class learned in variant A is a hit in variant B: 0 redispatches, 0 traces for B's op. Bitwise against eager across A/B
  alternation.
- A topology-changing result in B uses a section entry. The executor's section count is unchanged (no cut added).
- Shared results are stored once: the store holds one core per shared key after N variants.
- A shared key never matches two ops whose reads differ (a constant, a layout formula, a pinned argument): each is a redispatch.
- `verify_redo_templates` on in the redispatch suites: every shared instantiation equals a full redo.

## Cost estimate and order

- Python only (no C++ change): the renaming, the store, instantiation into B's program.
- About 1-1.5 days with tests. Measured gate: 27B cold-start redispatch attempts (learnfast's 92 -> 25-50 estimate) and warm-up time.
- Order: build after step_ms20 lands. learnfast's 27B cold-start counters (redispatches, traces, time) are the acceptance numbers.
- Open question for the user: should the store also be shared across replays (entries of different compiled graphs with the same op
  structure)? That would be a module-level store keyed additionally by the provider set and global state. The proposal is per replay
  only.

## Built (2026-10-18): learnfast's xvr in core instead of the template store

The user asked for a per-replay store. The family's dispatch cache (`HostTraceReplay._dispatches`, per replay and contract) already is
one: a cached fresh trace of an op's body, taken wherever its guards hold, renamed into the variant. It missed across variants and
layers for one reason: its key held each input's absolute trace-time extent. learnfast's prototype (xvr) fixed that and validated it on
27B (bitwise n27wf a 97/97, b 210/210; every hit re-run and compared, 876 entries identical). The template-store design above is not
built. Core took xvr's points 1, 2 and 4:
- step_ms21: an extent is keyed as (dtype, length, non-empty, cow), plus the inputs' pairwise overlaps. A hit checks the cached trace's
  call-value checks too. An audit test fails if another extent reader appears.
- step_ms22: declval. A redo template key holds a static-prefix symbol inside a mixed formula at its value, pinned in `_for_instance`
  (GDN prefill keys were per layer).
Not taken: xvr's resolver events. Core keeps "a redo that resolved state outside its call is not stored". Topology changes use today's
section entries.
