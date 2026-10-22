# Pipelined sections: guard, patch and launch section k+1 while section k runs (memset lane, 2026-10-16)

User direction (10-16): graph splitting is the fundamental fix for per-call host cost. When a step's sizes change, the
patch is on the critical path:
- MiMo alt 97<->100 re-sets about 770 nodes, about 3 ms host per call.
- In 8B unsynced churn (churnprof2) a size-changed decode costs 8.59 ms vs 6.37, because the GPU sits idle for our 2.75
  ms dispatch.

The executor splits a run's graph into K sections (sub-execs). It evaluates, patches and launches section k+1 while
section k runs.

## Constraints (user)

1. **Nothing is materialized at a cut.**
   - Tensors crossing a cut keep the addresses the memory plan gives them (run buffer or eager order).
   - No copies, no added allocations, no added memcpy nodes, no boundary outputs, no new steps.
   - Sections partition one graph's nodes over the same addresses; the executor only launches them in order.
   - Tested: split vs unsplit have identical peak allocated, identical node sets (apart from the cuts) and zero added
     copy nodes, over alternating sizes.
2. **Cuts only where every stream is joined.**
   - A cut is legal only where every side stream forked before it has joined back before it: `_run_starts` already
     excludes every launch from a fork's first side launch through its join.
   - In addition, it must be:
     - not inside a keyed site's or a selector's node chain;
     - not on a programmatic (PDL) edge: the launch after the cut is not programmatic.
   - Graph boundaries are full barriers anyway (section k+1 starts after all of section k), so a cut inside a fork would
     only serialize, not be wrong. The rule keeps the side streams' overlap and states the user's constraint explicitly.
   - Tested on a fork/join graph (a side-stream shared expert): no cut between a fork and its join, and a manual hint
     there is ignored with a recorded reason.
3. **Graph-level guards before launch, per-op guards per section** (see Guards).

## Guards

### How they map today

- **Graph-level:** the entry's contract key and the variant's graph guards (its `valid` row and the graph guard rows,
  `dynamic_valid_rows_`, evaluated first since valid-first, ms5), plus the form (topology) choice.
- **Per-op (op-owned guards):** each top-level op's own guards are a selector. Its predicates pick the row its nodes run
  (the op's trace, a fold or a redispatch entry), and a keyed site's key rows pick its binding.
  - Each selector and keyed site lists its nodes (`LoweredSelector.nodes`, `site.nodes`).
  - A cut never falls inside a selector's or site's node chain, so **every per-op guard belongs to exactly one section at
    lowering.**
- **Launch rows:** the rest of the dynamic rows are what the section's records read (dims, smem, fields, addresses).
- **Today every dynamic row is evaluated before anything launches.**
  - MiMo bs-1 steady with a partial delta: 2,750 rows, about 11 us evaluate + bind.
  - The 8B alternating case was about 270 us evaluate (before the delta cache, ms7; about 3-30 us after it, depending on
    the change).
  - So on a size change, evaluate is a smaller cost than node setting, but it is also on the critical path today.

### Per-section evaluation order (this build)

1. Before anything launches, evaluate the graph-level guards: contract key, valid row and graph guard rows; then the
   form.
   - A failure is a variant miss as today (next variant, or a trace).
2. For section 0: evaluate its selectors' predicates and site key rows, then its launch rows. Patch, then launch.
3. For section k >= 1, while section k-1 runs: evaluate section k's selector predicates and key rows, then its launch
   rows; patch; launch.
   - Rows are partitioned per section at lowering: each row goes with the first section that reads it, and any row a
     graph-level guard reads goes up front.
   - Shared rows are evaluated once and reused.
   - Only the graph-level guard rows and section 0's rows stay exposed.
   - All rows depend only on the call's host values at call start (sizes, ints, addresses), never on device results, so
     evaluating later is the same function of the same inputs.

### A section-local miss

- A section-local miss means a selector in section k has no holding entry, or a key is not yet bound.
- **This build:** the miss is detected before anything launches. Selector predicates are cheap, so all sections'
  selector rows are evaluated up front (they are a small part of the rows). A miss takes today's path (redispatch or
  trace) before any section launches; the launch rows are what is pipelined.
- **Follow-up, designed and not built** (needs the user): per-section variants that compose.
  - A section whose own guards fail would select its alternative (another entry of that section, or an op-local
    redispatch of just that section) without touching sections already in flight.
  - It is sound because cuts are full edges and guards read only call-start host values.
  - Open issues:
    1. A new entry found mid-call must fit the memory already allocated for the run. Its temporaries can't join a run
       buffer already allocated, so they are held for that call.
    2. A redispatch needs the GIL mid-commit.
    3. Per-section variants compose across whole-graph variants. This is the cross-variant reuse the user has open for
       27B cold start (11 of 13 stream redispatches repeated a class learned in another variant).

## Per-node work without node-level checks

- **Today, per dirty record:** pack its image, memcmp it against the held image, and if it differs call
  cuGraphExecKernelNodeSetParams; then copy the image into held, and keep per-(form, whole|split) held state.
- **New, per section and per exec:** keep the section's last-set inputs: the leaf values, bases and site rows its
  records read.
  - A call whose section inputs equal them skips the section entirely: no evaluation of its launch rows, no pack, no
    compare.
  - Otherwise, the records that read a changed leaf, base or site row are found from a static index built at lowering
    (row users, already in `users_`; this is a dependency, not a guard). They are packed and set directly: no memcmp, no
    held image copy.
- **Soundness:** a record's parameters are a function of its rows, bases and site row. If none of those changed since
  its exec last set it, its parameters are what the exec holds; if one did, setting it is correct whether or not the
  bytes changed.
  - "Always" records (RNG offsets, CPU scalars read at the call) are set every call, as today.
  - No case needs a per-node compare for soundness.
  - The cost of dropping the compare is redundant sets, of records whose changed input doesn't change their bytes. These
    are counted per model. Redundant dims resets matter because of the successor penalty (`exec_update_successor_penalty`),
    so a record whose dims rows are unchanged sets only its kernel params.
- **Per-node cost (MiMo alt about 3.8 us per re-set node, vs 0.45 us bare set; build_cpp14).** The memcmp and held copy
  go away (8B: memcmp 22 us over 44 fmha nodes of 1,344 B each). What remains:
  - pack (fields, descriptors; the TMA encode is cached on ms4+ but not on build_cpp14);
  - the driver set;
  - the smem attribute raise;
  - upload at launch.
  - All but section 0's are hidden behind GPU time.
  - mem_eager measures the breakdown on the head (MiMo alt) for this note.

## Cut policy (auto, chosen at a form's first call)

- dyn(i) = 1 for a record whose rows, address or site a call can change.
- Section k+1's work is hidden when (its evaluate + pack + set) <= section k's GPU time. Approximated as
  `p * dyn(k+1) <= g * nodes(k)`, so sections grow by R = g/p.
  - Default R = 4: g is 5-12 us/node in GB300 decode; p is 1-4 us per re-set node on the head. R is a knob.
- Section 0 is the smallest legal cut holding about dyn / (1 + R + R^2 + R^3) dynamic records (at least 16 records).
- Each next section is the largest whose dyn <= R x the previous section's nodes. At most K = 4 sections; no split
  under 2.
- **Same-size step:** (K-1) x (1.3-2 us CPU + ~2.2 us GPU gap) (09-11 bench_graph_chunking), about 4 us each for K=3.
  Everything else is skipped.

### Placements on real graphs

From land/core/memset/split/cut_report.py; JSON in split/out/.

Real graphs:
- Auto policy, R=4, K<=4.
- "dyn" counts every record a size change can re-set, including records that read only an allocation address (on a
  size change the run buffer re-places, so those move). "row" counts only records that read a changing row.
- Every cut is a legal cut: no PDL edge crosses it.
- "PDL beside" lists, per cut, whether the launch just before and the launch just after it are programmatic, inside
  their own sections.

| graph (line) | records | PDL launches | legal cuts | sections (nodes) | row-dynamic per section | PDL beside cuts |
|---|---|---|---|---|---|---|
| 8B decode (ms9/ms14) | 551 | 289 | 261 | 17 / 67 / 266 / 201 | 9 / 37 / 142 / 110 | (y,n) (y,y) (n,y) |
| 8B mixed | 515 | 217 | 261 | 16 / 64 / 256 / 179 | 11 / 47 / 182 / 130 | (y,n) (y,y) (y,y) |
| MiMo V decode v0 (ms3/ms2b) | 770 | 478 | 291 | 17 / 68 / 272 / 413 | 15 / 56 / 221 / 335 | (y,y) (n,n) (n,n) |
| MiMo V alt v1 | 769 | 477 | 291 | 16 / 64 / 256 / 433 | 15 / 56 / 224 / 379 | (y,y) x3 |
| MiMo V mixed v3 | 816 | 524 | 291 | 16 / 59 / 236 / 505 | 15 / 49 / 195 / 415 | (y,y) (y,y) (n,y) |
| MiMo Vc decode v0 | 911 | 525 | 385 | 17 / 68 / 266 / 560 | 15 / 54 / 210 / 442 | (y,y) x3 |
| 27B decode v0 (ms14) | 2823 | 625 | 2101 | 34 / 136 / 541 / 2112 | 8 / 10 / 39 / 166 | none / (y,n) / (n,y) |
| 27B decode v1 | 2967 | 529 | 2197 | 35 / 140 / 560 / 2232 | 28 / 122 / 485 / 1929 | -, -, (y,n) |
| 27B mixed v0 | 3948 | 640 | 2155 | 47 / 188 / 752 / 2961 | 40 / 170 / 675 / 2661 | none |
| 27B prefill v0 | 3628 | 624 | 2139 | 43 / 172 / 688 / 2725 | 36 / 154 / 613 / 2423 | -, -, (y,n) |

Reading:
- The section sizes follow the R=4 growth.
- Section 0 is 16-47 nodes, so 2-6% of the patch stays exposed.
- The last section is the largest; its patch is covered by the third section's GPU time while its dyn <= 4 x that
  section's nodes. That holds within a few percent for 27B; mixed is at 2961 vs 4 x 752 = 3008.
- With K=4, 27B and MiMo have no room to make section 0 smaller. The node-count knob (first-section nodes) tests that.
- 27B's eager and run_buffer graphs cut identically.
- Flash-Next (side streams) is pending from its lane's vf_w run.

Synthetic check (24 x mm + relu + pointwise, 120 records): 95 legal cuts, sections [17, 68, 35], none next to PDL.

## Manual control (user: not settled; finalize later)

- This build has the auto policy plus one internal knob in node counts, not a public API:
  `torch._C._host_trace_split_nodes(first, max_per_section)`. Defaults are 0 = auto. A test- and env-level knob
  (e.g. an adapter env var) sets the first section's node count and/or the maximum nodes per section, at legal cuts.
- No trace-time marker or module-hook API yet. The per-run cut list (set_split_cuts: legal positions, plus preferred
  ones) is the interface a future hook-recorded marker would feed. A marker would add a preferred cut, taken if legal,
  else ignored with a recorded reason. User guesses for that API: split points marked through a module hook during
  tracing, and/or a node-count setting.

## Tests

- Bitwise vs eager over alternating sizes (chains of mm + pointwise + memset; an opaque site run).
- Section count and placement.
- No set in unchanged sections on same-size steps: kernel_node_sets == 0.
- Identical peak allocated and node sets split vs unsplit, and zero copy nodes.
- A fork/join graph: no cut inside the fork; a hint there is ignored with its reason.
- RNG (dropout) over alternating sizes, bitwise with the same generator, before RNG runs are allowed to split.

## Measurements (with mem_eager)

- MiMo bs-1 steady and alt 97<->100.
- 8B churnprof2 size-changed decode step.
- For each: split vs unsplit, step time and exposed host time, synced and unsynced.

## No Python on a hit (user rule: "No python in runtime")

The replay hit path must be C++ end to end; the Python slow path only serves a genuine miss (a new shape or class that
needs a trace, a redispatch or a harvest learn). `HostTraceReplay.slow_hits` counts calls that the slow path served
from an existing variant with no trace, redispatch or relower. TestReplayCounters asserts it is 0 on steady,
alternating, variant-switch and learned-cuBLAS repeats, and the MiMo/8B gate rows will assert it too.

Remaining cases where a hit still touches Python:

1. **A keyed site meets a new key whose binding the provider already holds.**
   - Python `_fill` adds the site's row from the binding, then the commit's `mark_dirty` rebuilds the whole users index
     (`index_users`: CSR over every record and row).
   - This is the 8B churn mixed step (mem_eager, cbmix2): one variant, `learns` False, no opaque calls, but every mixed
     step has a new token count. Each one costs a slow call: about 2.5-3 ms index rebuild (the "c_misc") plus about
     1.5 ms patch (443 sets), out of a 10.7 ms entry.
   - Expected failure for now: `TestOpaqueCalls.test_a_held_binding_at_a_new_key_is_a_native_hit`.
   - Plan:
     - (a) index the added rows' users incrementally on add_row (no full rebuild);
     - (b) the native entry fills the row itself. Each provider's held bindings get a native table (normalized key ->
       row template: the binding's kernel rows, fields, descriptors and scratch). Key normalization moves to C++. A keyed
       miss whose normalized key is in the table adds the row and continues, all in C++. Only a key the provider has no
       binding for (a learn) returns to Python.
   - (a) lands first (a pure optimization); (b) is the C++-only selection the rule needs.
2. **A variant that still learns** (opaque calls not yet bound) commits through `_call_slow`. That is a learn in
   progress, so allowed; once its keys bind, it relowers to a bound variant.
3. **The Python entry** (`cpp_entry` False) is a test and diagnosis mode only.

## Follow-up: cross-variant reuse of per-op results (user approved: "sharing is fine, it's strictly better")

Constraint (user): "I don't want to have to split up the cudagraph at every cublas. that would be slow."
- The number of sections stays small and is set only by the pipelining cost model (K <= 4).
- A cut is never added because an op's kernel choice varies.

Design:
- **What is shared.** A per-op result: a redispatch entry, a fold's launches, or a keyed site's binding. It is keyed
  globally by (op site class, size class), meaning the op's template key as the redo templates key it (closure roots,
  constants, layout) together with its selector's predicate values. It is no longer owned by one variant.
- **How it applies.** Inside the section that holds the op, by rewriting that op's nodes with exec node sets (kernel
  function, grid/block/smem, params), when the result's topology (node kinds, count, launch attributes) is the op's in
  that variant.
  - This is today's mechanism for a keyed site's rows and a selector's entries. It extends to entries learned in another
    variant: a variant whose selector fails looks the op's class up in the shared table before redispatching.
- **A topology change** (e.g. a split-K variant that adds a reduction kernel, or another cluster) needs a section-level
  alternative sub-graph. This is a form of that section only, chosen per call by the op's guards, still without a cut at
  the op. Today's forms are per run; they become per section, so an alternative costs one section's exec, not the whole
  graph's.
- **Soundness** comes from the per-op guards, as today. A shared result applies only where its op's own guards hold at
  the call, and the graph-level guards still pick the variant.
- **Memory.** A shared result is stored once: its rows, images and descriptors, not one copy per variant.
  - Per-variant state is only the site row index and the node handles.
  - Section alternatives cost one section exec each (instead of a whole-graph form).
  - Expected: fewer copies than today, where each variant redoes and stores its own entry.
  - learnfast's estimate for 27B cold start: 92 redispatch attempts -> 25-50, warm-up redispatch 207 s -> 60-110 s,
    stream redispatches 10 -> 2 (11 of 13 stream records repeat a class learned in another variant).
- **How often topology changes per op class.** To be measured from existing variant data (cut reports + entry
  topologies), per op class, for 8B / 27B / MiMo. Expected: rare. Known cases are cuBLAS split-K reduction kernels and
  cluster changes across M; forms today exist mostly for those.
- **Tests.** Counter tests:
  - a class learned in variant A is a hit in variant B with 0 redispatches and 0 traces;
  - a topology-changing result in B uses a section alternative, with no cut added (section count unchanged);
  - shared results are stored once (count of stored entries);
  - bitwise across A/B alternation.

Order on this line:
1. the Flash-Next crash fix;
2. the C++-only hit path (native binding table + incremental users index; slow_hits == 0 including mixed);
3. cross-variant reuse.

## Scope and order

1. **This build:**
   - always-pipelined sections;
   - graph-level guards up front;
   - per-section launch-row evaluation;
   - section-input skip without node compares;
   - the auto cut policy.
   It fixes the ms7 "split fires every step" bug (KV length moved a dims or site row every decode step, so every call
   split and also caught up the whole exec).
2. The public split API (marker / node counts) once the user settles it.
3. Per-section variant composition is a designed follow-up.

step_ms15 (run-buffer section memory) lands first. mem_eager's trusted statics lands when green. X1 goes last.

## Status 2026-10-18 (core agent after the reboot)

- (a) incremental users index: landed in step_ms19 (build_cpp_ms12a). A keyed row whose reads the site's rows already read, and that
  its pieces' reads cover, keeps the index (`row_added`); counters `_host_trace_index_counts()`.
- (b) native binding table: built in build_cpp_ms12b + snap/wip20 (gating). Each provider that supports it (`provider.native`, a
  `_host_trace_native.NativeBindings`; HarvestProvider and the tests' TableProvider) keeps one `_HostTraceBindingTable` per class of keyed
  site (op, dtypes, ranks, constant scalars, device, library state). A table key is the site key in the provider's normal form: sizes,
  strides (0 where the size is 1 for cuDNN ops), alignment class (HarvestProvider) and symbolic scalar values. Equal table keys ⇔ bind()
  returns the same binding. A keyed miss in `evaluate_sites` looks its normal key up (`fill_from_table`) and adds the row in C++ when the
  binding's topology is the site's (arm 0) and its scratch fits; otherwise `_fill`'s Python path runs as before.
  - Remaining Python on a hit: a held binding of another topology than the site's (an arm) still takes `_fill` once per new key, because it needs
    a form (graph clone and instantiate). Stored (loaded) bindings are resolved when their functions load (`_restore`), so they are in
    the tables before their first key.
  - Tests: TestOpaqueCalls.test_a_held_binding_at_a_new_key_is_a_native_hit (on: 0 slow hits, 3 fills; off: 3 slow hits),
    test_a_native_fill_needs_the_sites_topology; TestReplayCounters.test_held_library_bindings_fill_rows_natively (cuBLAS, bitwise),
    test_a_native_fill_keys_alignment_by_class.
- Item 4 dirty threshold: built with (b). A split run launches its sections only when at least `split_dirty_min` (default 256, knob
  `_host_trace_split_policy(dirty_min=)`) of its records are dirty. Fewer (steady decode, hidden behind GPU work in flight) launch the
  whole exec. Counters: `_host_trace_split_dirty_wholes()`, per section `_host_trace_split_section_counts(v, segment, form)`
  = (calls that set any node, nodes set, launches); a section's skipped calls = launches - calls that set. The cut cost: no core change is
  needed (split-cost agent, notes/SPLIT_CUT_COST.md: 1-5 us per cut after the first split-on window; keep the upload before node sets).
