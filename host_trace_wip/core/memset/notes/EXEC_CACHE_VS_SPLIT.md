# Per-size exec cache vs splitting a variant's graph (alternating / churning decode)

Status 2026-10-14: PARKED (user: hidden under async scheduling in serving; revisit for host-bound cases). Note only. Target design (coordinator): an automatic N-segment pipeline. The exec cache
was written before the hold: about 400 C++ lines in land/int6/build_cpp_ms2/src, plus a staged 90-line
delta-cache patch (land/core/memset/ms2/patch_delta_slots.py, not applied). Its build is finishing. It is
untested and parked.

Inputs. Coordinator's 8B numbers, alternating 63<->65 vs fixed, per step: patch +650 us (443 kernel + 36
memset node sets vs 44), launch/upload +207 us, evaluate +272 us, about 1,100 us host in total. Prior benches:
- graph chunking ([[paramgraph-graph-chunking-bench]]): 2.2 us GPU idle and 1.3-2 us CPU per extra graph launch,
  0.40-0.49 us host per node set;
- [[paramgraph-exec-update-successor-penalty]]: a node whose grid, block or smem is reset after upload costs its
  successor +0.07..1.4 us GPU for the exec's life;
- my microbench on build_cpp17 (land/core/memset/ms2/bench_alt.py, base_b17.json; 40 layers of mm + pointwise +
  memset, hidden 1024): alternating vs fixed, patch 127 vs 12 us, launch 19 vs 4, evaluate 33 vs 0.3, dispatch
  206 vs 27 us.

## 1. The exec cache: state and size

Per (segment, form), up to K execs of the form's graph. Slot 0 is the form's own exec; the others are
instantiated lazily, with memcpy nodes pointed at the call's live operands, and uploaded before any set.

Per slot:
- what its nodes hold, per record and per piece: held flag, row, launch dims, kernel image bytes, memset
  (dst, extents), memcpy (dst, src, bytes);
- the rows, bases and site rows its records were patched at, and its dirty mask.

A call takes:
- the slot whose key matches: the values of the rows that launch dims and memset extents read, plus the
  segment's site rows;
- else an unused slot;
- else a new one (under K);
- else the LRU slot.

A switch saves the active slot's held state and the call's values, then restores the target's and marks only
records whose rows, bases or site rows differ from the target's saved ones. Patching then sets only nodes whose
image differs from what that exec holds.

Size and cost:
- ~400 C++ LOC: Variant.h Slot/SlotHeld, and select_slot / switch_slot / instantiate_slot in Patch.cpp;
  index_users builds per-segment read_rows / shape_rows; set_argument_copies targets the active slot.
- ~90 LOC for evaluate's per-slot delta cache (K-1 saved leaf vectors + dynamic rows + site memo): a call whose
  leaves are nearer a saved vector's swaps it in. The cache's rows are saved only when the call re-evaluates
  >= 1/8 of the dynamic rows, so a KV-length-only step copies nothing.
- A switch copies the segment's held images: ~90 KB, ~10-30 us.

Pays only on recurrence within K. On a miss (churn) it is today's cost plus the switch copy. A hit removes the
node sets and the launch-time update volume, but the remaining host work (~150-200 us, the fixed-size case)
stays exposed.

## 2. Splitting into N segments

The mechanism mostly exists:
- `lowered.steps` already alternates runs (one CUDA graph each, one C++ Segment) with eager calls.
- `_host_trace_memory._split_run(lowered, i, c)` splits a run before launch c.
- `_run_starts` lists the launches a run may not start at: inside a keyed site's node chain, or inside a fork
  before its join.
- Per-step memory (StepMemory: allocate before the run, drops after), per-segment RNG increments and per-segment
  exec state are already per step.
- commit() does `allocate(step k) -> patch_and_replay(k) [async launch] -> allocate(k+1) -> patch(k+1) ...`.

So once a run is split, segment k+1's patch and launch already overlap segment k on the GPU. Splitting alone,
in lowering only, hides patch + launch/upload. Only evaluate stays exposed, because it runs before commit.

### 2a. Where cuts can go

A cut before launch c is valid if:
- `c not in _run_starts(lowered)`: not inside a keyed site's chain, not inside an unjoined fork;
- the edge into c is full, i.e. `not lowered.launches[c].launch.programmatic`, or c is a memset/memcpy (the
  driver records a full edge behind a memset even with the attribute).

Cutting there loses no PDL overlap.

Candidate count:
- Estimate for a compiled Qwen3-8B decode (~520 nodes, ~14 per layer): Inductor Triton kernels and the memsets
  launch without PDL, while cuBLAS and trtllm-gen kernels may be programmatic. That gives 3-6 full-edge cuts
  per layer, roughly 100-200 per variant.
- To count exactly on a lane's pickled or live variant:
  `[c for c, lo in enumerate(L.launches) if c not in _run_starts(L) and not getattr(lo.launch, "programmatic", False)]`
  with `L = variant.captured.lowered`.

### 2b. Open question: PDL across graph launches

Does the driver honour `cudaLaunchAttributeProgrammaticStreamSerialization` on a graph's root kernel relative
to the previous graph launch in the same stream? If yes, a cut could also go on a programmatic edge.

Probe sketch (not run):
- NVRTC kernel A (in G1) spins ~50 us, calls `cudaTriggerProgrammaticLaunchCompletion()` at entry, and writes
  `%globaltimer` at exit.
- Kernel B (root of G2, captured with the attribute) writes `%globaltimer` at entry, before
  `cudaGridDependencySynchronize()`.
- Launch G1, then G2, on one stream. If B.entry < A.exit, PDL is honoured across graph launches.
- Controls: A and B in one graph with a programmatic edge (must overlap); A and B as stream launches with the
  attribute (must overlap); G2 without the attribute (must not).

Until it's answered, cut only on full edges. Losing a PDL edge costs a few us of overlap, which is cheaper than
a stall.

### 2c. Evaluate pipelining (the only C++ part)

Before the first launch (pre-effect validation), these must be evaluated:
- the valid row's closure (all guards);
- every keyed site's key rows and every selector predicate (a keyed miss must miss before any effect);
- the form rows;
- the planned buffer's bytes row, if "planned";
- step 1's allocation sizes and segment 1's record rows.

Everything else (other segments' record rows and their allocation sizes) is evaluated in commit just before
patching segment k, i.e. while segment k-1 runs.

Pieces:
- Partition the rows at set_static_prefix / build, as `program_.partition` already does for the valid row:
  `pre`, then `rows_k` = closure(segment k's read_rows + step k's allocation rows) minus earlier ones.
- Keep the bound leaves in Frame.
- `program_.evaluate(rows_k, leaves, v)` before `patch_and_replay(k)`.
- mark_dirty per segment over that segment's read_rows (the per-segment lists the cache work already builds in
  index_users).
- Delta: intersect each DeltaRows with the partition, per segment.

About 250-350 C++ LOC.

Unknown that sets the exposed floor: the size of `pre`. In decode, guards are few shape relations, but their
closures and the site keys (per-layer cuBLAS keys read bs) may be a sizeable part of the 272 us. Measure first:
`|dynamic_valid_rows_|`, `|site key + predicate closure|`, and `|dynamic_other_rows_|` on the 8B variant.

### 2d. Choosing the cuts

Rule per variant, at lowering:
1. Segment 1 is the shortest prefix (to a valid cut) whose estimated GPU time covers segment 2's estimated host
   work.
2. Each next cut is the first valid cut where the previous segment's GPU estimate is >= the next segment's
   patch + evaluate + launch estimate.

Estimates:
- Host per node: ~2.3 us per changing node on 8B (1,100 us / 479), split as ~1.4 patch + ~0.4 launch/upload +
  ~0.6 evaluate share. Use the worst case (every size-reading node changes): a lower bound must not stall.
- GPU per node: measured, not modelled. Lower once with a default cut (segment 1 = up to the first layer's
  end). Time each segment with two events on the variant's first few replays (cheap, no profiler). Recompute
  cuts and relower if a stall shows (host still patching k+1 when k ends).
  - A cost model from keyed-site keys (GEMM M/N/K) plus a per-kernel floor would also work, but per-size GPU
    time moves with bs and KV, and the events are exact at the sizes actually served.
  - Recutting is a relower plus a host-only capture of the changed segments (ms, off the hot path), or
    clone+delete of the existing graph.

Geometric growth on 8B decode (~15 us GPU per node on average, ~2.3 us host per changing node, so one GPU node
covers ~6 nodes of host work):

| Segment 1 | Segments | Exposed host / step | Extra GPU idle |
|---|---|---|---|
| embedding + first norm (~3 nodes, ~10 us GPU) | 5 (3 / ~4 / ~25 / ~180 / rest) | `pre` + ~7 us + 4 launches (~6 us) | ~9 us |
| embedding + first layer (~15 nodes, ~225 us GPU) | 3 (15 / ~95 / rest) | `pre` + ~35 us + 2 launches (~3 us) | ~4 us |

An N=3 split with segment 1 = embedding + layer 0 is the likely optimum.

Prefill and mixed: GPU time per node is 10-100x higher, so N = 2 (a tiny segment 1 covers everything).

### 2e. Exposed host time per step, 8B compiled decode (estimates)

`pre` = bind + guard/key/allocation rows, unmeasured: assume 20-60 us.

| step | today | exec cache | N=3 pipeline (lowering only) | N=3 pipeline + evaluate split |
|---|---|---|---|---|
| fixed bs | ~150-200 | ~150-200 | ~20-40 (evaluate is small when only lengths move) | `pre` + ~10-20 |
| alternating 63<->65 | ~1,100 | hit: ~150-200 | ~272 + ~40 (evaluate exposed) | `pre` + ~35 ≈ 55-95 |
| churn (new bs every step) | ~1,100 | miss: ~1,100 + switch copy | ~272 + ~40 | `pre` + ~35 ≈ 55-95 |

Extra launches: N-1 per step (2 at N=3): ~1.5 us CPU and ~2 us GPU idle each. The last segment's outputs phase
overlaps the GPU anyway.

## 3. Do they compose?

Yes. My cache implementation is per Segment, so with N segments each segment has its own K slots, and the
delta cache is per variant.
- The pipeline hides host work.
- The cache removes it: CPU time per step (which matters when the host is shared or CPU-bound), the launch-time
  update volume, and the bs-dependent node sets.
- On hits, pipeline + cache brings exposed time down to `pre` + segment 1's (few) changes.

Recommended order:
1. Split in lowering only (Python, ~100-150 LOC; `_split_run` + full-edge filter + event timing). Measure on the
   8B alternating and churn rows.
2. Evaluate split (C++, ~300 LOC), after measuring `pre`.
3. The cache only if the 8B hit-rate analysis shows recurrence within K and CPU time per step still matters.

Gotchas for both:
- The successor penalty (+0.07..1.4 us GPU per re-gridded node's successor) is permanent per exec after its
  first re-grid, so it is the same in steady state for both designs. A cache slot could avoid it by setting the
  template's kernel nodes to the call's dims before instantiating (no exec update).
- Memsets are not set every replay: they are set when their (dst, extents) differ from what the exec holds. A
  memset at a cut point is just segment k+1's first node.
- Hooks (observed runs) replay the graph's own exec in Python: keep splitting off and use slot 0 there.

## 4. Executor-level split (opt-in; design, 2026-10-15)

User decisions so far:
- graph partitioning, not per-size slots;
- the cut lives in the executor;
- no materialized tensors or extra allocations at a cut;
- off by default.

### What changes

Lowering only annotates the safe cut points. A Segment (one lowered run, one memory plan) keeps:
- its whole exec per form, as today;
- per form, N sub-execs over consecutive record ranges, built from the same graph: cuGraphClone, then cuGraphDestroyNode of every node outside the range, then instantiate and upload.

The executor launches either the whole exec or the sub-execs back to back from the same frame, patching sub-exec k+1 while k runs.

There are no new lowered steps, outputs, Python objects or allocations:
- The run's StepMemory allocates before the first launch and drops after the last, exactly as for the whole exec.
- An intermediate crossing a cut is planned memory, live across the boundary as today.

`_host_trace_memory._split_run` is not used. It only splits the step range itself, but `plan_memory` then re-plans per step: a crossing intermediate is allocated before step k and dropped after its last step, which is a different allocation schedule plus a new lowered step.

### Safe cuts

A cut before record c is safe when all of these hold:
- `c not in _run_starts(lowered)`: not inside a keyed site's node chain or a side-stream fork before its join;
- the edge into c is full (`not launch.programmatic`), or c is a memset or memcpy;
- c is not inside a section's piece range.

Lowering passes the candidate record indices per run in the VariantSpec. Sequential launches on one stream order every cross-cut dependency, and with a full edge at the cut no PDL overlap is lost.

### RNG

replay_exec runs the generator prologue (offset increment) per launch. The first sub-exec launches through replay_exec; the others go through cudaGraphLaunch on the same stream, so the call advances the generator once, as the whole exec does. Every sub-exec's kernels read the same captured seed and offset addresses.

### Held state per exec

Today's per-form state (Segment::held[form] plus Seen) becomes per exec: (form, whole) and (form, split). In split mode a record's state is that of its node in its sub-exec. Switching between whole and split is the same O(1) switch as a form switch: save the Seen snapshot, restore the other's, mark only records whose inputs differ.

### Policy (per call, executor)

After mark_dirty, compute the patch-set size relative to each mode: one compare of the segment's read rows and bases against that mode's Seen, O(rows + bases).
- If the whole exec's set is at most T records and has no launch-configuration change, launch whole: one launch, zero split overhead. A same-size decode step (~44 args-only sets) stays exactly as today.
- Otherwise launch split.
- T is a config knob; default 64.

Catch-up: after a size change the whole exec is stale. While split mode runs with a small patch set, the host has idle time after the last sub-launch (the GPU is still running). It spends a bounded budget there patching the whole exec's stale nodes; the whole exec is not in flight. Once the whole exec's set is at most T, the next call launches whole again. This hidden host work replaces exposed work.

### Cut placement

- Initial: segment 1 = through the end of the first layer-sized chunk (~1/8 of the records); then geometric growth, each next segment at most ~6x the previous (8B: ~15 us GPU per node vs ~2.3 us host per changing node). N = 2-3 for decode.
- Refined from measurement: events between the sub-launches on the first M split replays give per-sub-exec GPU time. If the host is still patching sub-exec k+1 when k ends (host clock vs event), cuts move to the first safe cut where k's GPU time >= k+1's patch estimate. The sub-execs are re-instantiated (~1-3 ms per change, a bounded number of times per variant).

### Evaluate pipelining (phase 2, still opt-in)

- Before the first launch: the valid row's closure, every keyed site's key rows and selector predicates (a miss must miss before any effect), the forms, the planned-buffer and step allocation sizes, and sub-exec 1's rows.
- The rest is evaluated just before patching sub-exec k.
- The partition is computed once at build, as `program_.partition` does for the valid row. The delta cache intersects with it.
- Phase 1 ships without it: patch and launch are hidden, evaluate stays exposed.

### Cost

- Memory: per form, the sub-execs together hold about the whole exec's nodes, so roughly 1x the whole exec's device memory extra (~2 MiB per 450 nodes, ~4 MiB per 1000; an 8B decode segment ~2-4 MiB per form). The sub-graph clones stay alive while their execs do (node handles); this is host memory.
- Instantiate: ~1-3 ms per form, lazily at the first call that needs split (opt-in, so warm-up can pay it).
- Per split call: N-1 extra launches (~1.5 us CPU and ~2 us GPU idle each), plus the policy compare (O(rows + bases)).

### Measurements to report

- 8B unsynced churn: changed-step and mixed-step host exposure, whole vs split vs off.
- Same-size decode unchanged: policy picks whole, 0 extra launches.
- Extra launches per step, device memory per model, and the cut placements.
