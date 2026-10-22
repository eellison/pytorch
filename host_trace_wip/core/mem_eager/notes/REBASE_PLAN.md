# Rebase plan: the mem_eager line (step332_me9) onto the candidate head

Status: dry run only. No candidate snapshot is cut until gh200-a's me-line numbers are in and the coordinator has
reviewed this plan.

## What moves

Update 2026-10-15: the line's head is now step332_me10 (= me9 + the fused verify digest + TestMemoryGate). It adds
changes to torch/cuda/_host_trace_harvest.py, test/test_cuda_host_trace_harvest.py and
test/inductor/test_cudagraph_host_trace.py, none of which the candidate touches, so the conflict analysis below is
unchanged. Patch: mem_eager_F10.py.step332.patch (md5 6b7e166daf417054efe244f52f2d253d).

- me-line: `int6/snap/step332` -> `int6/snap/step332_me9`, Python only. Patch:
  `land/core/mem_eager/patches/mem_eager_F9.py.step332.patch` (md5 1e372996ff47f5ff72000852d90d1fe0).
- Its C++ part (a2's argument frees, `mem_eager_args.csrc.p1.patch`) is already on the candidate line: build_cpp17 and
  every later build derive from build_cpp15/src. Checked with a reverse dry-run apply against build_cpp_ms3/src and
  build_cpp_ms4/src: present in both. The rebase needs no new C++ build.
- Candidate heads:
  - `snap/step_ms2b` (build_cpp_ms3), the head named for the merge;
  - `snap/step_ms3` (= ms2b + per-form held state, build_cpp_ms4), dry-run too because it touches the held-state code
    the coordinator flagged.
  - The core agent's per-size ready-exec work has not landed; see "Ready-exec" below.

## Files

me-line files (step332 -> me9): 8.

| file | candidate touches it (step332 -> ms2b / ms3) | how it moves |
|---|---|---|
| torch/cuda/_host_trace_harvest.py | no | me9's file |
| torch/cuda/_host_trace_opaque.py | no | me9's file |
| torch/cuda/_host_trace_replay.py | no | me9's file |
| torch/_inductor/cudagraph_host_trace.py | no | me9's file |
| test/test_cuda_host_trace_harvest.py | no | me9's file |
| test/inductor/test_cudagraph_host_trace.py | no | me9's file |
| torch/cuda/_host_trace_tape.py | yes (ms2b and ms3 alike) | 3-way merge, 1 conflict |
| test/test_cuda_host_trace_opaque.py | yes (ms2b; ms3 adds form_held tests) | 3-way merge, clean |

The candidate's other files (_host_trace_capture.py, _host_trace_cute.py, _host_trace_lower_tape.py,
_host_trace_memory.py, _host_trace.py, __init__.pyi.in, and the aten/cute/replay tests) are untouched by the me-line.
The rebased delta has the same number of changed lines as step332 -> me9.

## Conflicts, by file and function

1. `torch/cuda/_host_trace_tape.py`, `trace()`, the `with _gc_hold, _capture(device) ...` block after the warm-up.
   - Candidate: `_capture(device) as markers` and `_Trace(..., markers=markers)` (capture markers, memset lane).
   - me-line: `if witness is not None: tr.warm_learned = witness.learned` (learn at the warm-up call, H step 2).
   - Resolution: keep both (the markers context and argument, then the warm_learned line). They are independent: the
     warm-up runs before `_capture`, and warm_learned only marks keys the trace keeps as OpaqueCalls.
   - Same single hunk against step_ms3.
2. `test/test_cuda_host_trace_opaque.py`: clean 3-way merge against both heads. The me-line edits are the
   `real_key` / out-of-band lend tests; the candidate's are the row-switch and form_held tests.

## Interactions to review (Patch.cpp, held state, native entry)

None of these conflict textually. They are behaviour the gate and the gh200-a A/B must cover.

1. Scratch sharing (me6) and per-form held state (ms3). A shared binding has fewer scratch buffers per key (conv bwd:
   1 instead of 2), and two nodes read the same scratch base.
   - Held state keys on frame values and bases (`switch_form`, `Segment::read_rows`), so a shared base is one base
     read by more records.
   - A site's scratch dict must cover every key's count (`KeyedSite.recordable`). A sibling unshared because its
     template's shared form refused has more blocks; it then relowers, as any unrecordable binding does.
2. Canonical stale high halves (me5) make 4-byte runs loose with zeroed bytes. Held copies (`Segment::Held::copy`)
   and `_restore` compare images outside slots and loose runs, so these bytes are not compared.
3. The boxed first call (me7) and the boxed trace warm-up (me8) do not go through the native entry for the entry's
   first call and its first trace.
   - `call_first_boxed` sets `_called` and `eager` the way `_miss` does.
   - `trace_boxed` re-enters `_miss` with the list; `_register` / `evaluate` read stand-ins (same metadata and
     addresses, no memory).
   - Held state starts at the first replay as before. The static prefix (parameters) is never stood in (held
     positions keep the real tensors).
4. Verify on the caching allocator (me9): no held-state contact.
5. Ready-exec (per-size ready execs, not landed). To check when it lands:
   - (a) a2's mid-tape argument frees (frame.owned in `HostTraceVariant::allocate`) against execs instantiated ahead
     for other sizes (addresses are patched at launch, so a ready exec must not cache an argument's address past a
     free);
   - (b) the first-call and trace boxed paths bypassing the native entry, if ready-exec preparation hooks there;
   - (c) per-form held copies against canonicalized loose bytes (as 2).
   If ready-exec lands before the merge, rerun this dry run against it and add its tests to the gate.

## Merge order

Recommended: one merge step (step_ms2b + me9's delta, conflict 1 resolved as above) rather than replaying me .. me9
one by one. me9 was gated as a whole, and each me step superseded the last on gh200-a. TREE.txt keeps the me-line
history (me .. me9 lines and patch md5s).

## Validation before cutting the candidate snapshot

- Gate on build_cpp_ms3: harvest, python_entry, replay, opaque, memory, inductor test_cudagraph_host_trace, cute,
  aten. Compare counts with the candidate's own gate (gate_ms3: harvest 234, memory 24, opaque 47, python_entry 598,
  replay 316) plus the me-line's added tests.
- Stress: harvest 5x, AttentionAndRng 10x, python_entry 3x (the canonicalization and sibling paths).
- The memset lane's held-state tests (opaque row-switch / form_held, fresh zeros), and on step_ms3 the form_held
  parametrizations.
- gh200-a A/B: candidate vs candidate + me on the 15 training rows and the static rows. Report
  `diag/backed_reserved.py`'s memory-backed peak next to torch's reserved (torch counts the address-only capture
  span while a learn runs).

## Dry-run results (2026-10-15)

Scratch trees: `land/core/mem_eager/scratch/rebase/tree_step_ms2b` and `tree_step_ms3`. Each is the head plus me9's 6
untouched files, the clean opaque-test merge and the resolved tape merge. Gate script `scratch/rebase/gate.sh`; logs in
F/logs/RBms2b and RBms3.

| file | rebased on ms2b (build_cpp_ms3) | rebased on ms3 (build_cpp_ms4) | candidate ms2b alone (gate_ms3) |
|---|---|---|---|
| harvest | 251 OK | 251 OK | 234 OK |
| python_entry | 615 OK (71 skipped) | 619 OK (71 skipped) | 598 OK |
| replay | 316 OK | 316 OK | 316 OK |
| opaque | 47 OK | 51 OK (form_held tests) | 47 OK |
| memory | 24 OK | 24 OK | 24 OK |
| inductor test_cudagraph_host_trace | 90 OK | 90 OK | - |
| cute | 76 OK (22 skipped) | 76 OK (22 skipped) | - |
| aten | 1761 OK | 1761 OK | - |

No illegal-address lines in any log. Harvest and python_entry carry the me-line's added tests.

Memory on the candidate head: timm mobilenetv3_large_100 AMP training, bs128, reduce-overhead with host trace, 6 steps,
measured with diag/backed_reserved.py (memory-backed reserved):

| tree | backed peak | at exit | torch peak reserved |
|---|---|---|---|
| step_ms2b | 4652 MiB | 4412 MiB | 4652 MiB |
| step_ms2b + me9 (dry run) | 3206 MiB | 3206 MiB | 3616 MiB |

-1446 MiB peak, -1206 MiB at exit. 3206 is default compile's steady reserved on this model (a1/timm_probe.py).

Stress on the dry-run trees refreshed to me10 (scratch/rebase/stress.sh; logs F/logs/RBstress_<head>): inductor file
with TestMemoryGate (95), harvest 5x, AttentionAndRng 10x, python_entry 3x per head: 37 of 38 runs green. The one
failure: rebased ms2b python_entry run 1, TestChainReplayPythonEntry.test_inplace_foreach_ops (n=3000 replay,
56/3000 elements wrong at the tail). Unchanged test since step332, a plain HostTraceReplay with no harvested op, no
boxed path: the me-line does not touch its path. 10/10 isolated on both the candidate and the rebased tree; full-file
python_entry A/B 6 + 6: candidate 0/6, rebased 0/6 (rebased overall 1 in 10 full-file runs). Base-rate A/B (12 + 12
more) running: scratch/rebase/pe_ab2.out.

Not run yet: the gh200-a A/B. Open: compile latency. gh200-a measured harness compile time
up on me4 (mobilenetv3 13.4 -> 21.6 s, inception 15.7 -> 20.1 s); steady speed is unchanged. To look at before the
merge (likely the learn at the warm-up call: digests and verify per learned key).

## Blocker: test_inplace_foreach_ops tail values (2026-10-15, under investigation)

Treated as a silent wrong-output bug (coordinator). Seen once: rebased ms2b python_entry stress run 1,
TestChainReplayPythonEntry.test_inplace_foreach_ops, n=3000 replay, 56/3000 wrong = the last 128-wide block of
add(a) (a written in place by two eager foreach steps; core agent's analysis).
- Rates (full python_entry file): rebased 1/22, candidate 0/18 (mine) + 0/29 (core agent). Isolated: 10/10 both,
  core agent 0/300 per mode on the rebased tree. Not yet significant.
- The test's own path has no me-line code: HostTraceReplay(fn) directly (no boxed_fn: no call_first_boxed,
  TraceHandback, _stand_ins), no harvested op. me-line code in the file: harvest classes earlier in the process,
  and the witness learn hook (no-op here).
- Stream ordering: every eager step preceded by a 2e8-cycle sleep on its stream: 0 wrong on both trees, cpp entry
  on and off (scratch/dbg/foreach_sleep.py).
- Harvest-state hypothesis: 4 rounds of {the file's 3 harvest classes, then the function on 20 fresh entries} in one
  process: 0 wrong on both trees (scratch/dbg/foreach_after_harvest.py).
- Shuffled-order python_entry runs (30 seeds per tree, value mismatches only): running, scratch/rebase/shuf_*.out.
  Seed 1: 0 value mismatches on both. The rebased tree's order-dependent failures are `_private_bytes() <=
  _POOL_KEEP` (me5 made _POOL_KEEP 0): test hygiene to fix (compare against the bytes at the test's start).
