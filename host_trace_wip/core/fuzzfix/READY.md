# fuzzfix: fixes for the HostTraceReplay correctness fuzzer's findings (candidate 2) -- NOT handed to int6

Base: snap/step188 (even, `.int6cpp.patch`, build_cpp7) and snap/step187 (odd, `.int6.patch`, install_land_int5b_frozen).
Trees: core/fuzzfix/<bug>_{e,o} (copies; base_{e,o} pristine), comb_{e,o} = FZ01-07 stacked, RS_{e,o} = FZ08+09+10.
Runners: run.sh (GB300 GPU 1, candidate launchers), run_f4.sh (build_f4/install), gate.sh; gh200-b (GH200, even
line only): ht/fuzzfix/{gh_mk,gh_run,gh_gate}.sh, trees built from ht/cand2_cpp's Python + compiled torch under
`python -S` (the editable finder bypassed; subprocess tests then see the stock install: 2 persist_padding failures and
quack's cutlass import are artifacts of that, they fail on base too).
Evidence logs: probe/chain*.log (GB300), gh200-b ht/fuzzfix/chain*.log, logs/.
Combined gates (FZ01-07 stacked, comb_{e,o}):
- odd, GB300 GPU 1 (logs/comb_odd): replay 189 with 1 failure, FZ05's test, because the odd line runs int5b's frozen C++
  eager step and needs the C++ half rebuilt; harvest 220 OK (base 219 + FZ04's test); oob 6 OK; aten 1668 OK.
- even, GH200 (gh200-b logs/comb2): replay 250 with 2 failures: FZ05's test (no C++ rebuild on GH200) and FZ06's
  first test version (since rewritten; the final F9/F15 tests pass on GH200 and GB300). harvest 221 = base's failures;
  oob 6 OK; aten 1670 + 1 env error. FZ05's C++ half is verified on GB300 with build_f4 (below).

FZ01-06 apply to the base in any order (offsets only, checked stacked). FZ07 is optional. FZ08 -> FZ09 -> FZ10 stack.
Overlaps: FZ02 and FZ04 both edit `_host_trace_opaque.py`/`_host_trace_replay.py` far apart; FZ06 rewrites the slot loop
of plan_memory that the CuTe lane's 98 also touched (agreed: FZ06's roots loop is the one change; 98 keeps check_plan);
FZ08 rewrites `_call_held`'s redispatch loop and `_redo` (N1's trusted redispatch is in the same files);
FZ03 edits `_traced_pointwise` in `_host_trace_tape.py` (perf lane's ATen items are elsewhere in that file).

| patch | bug | md5 even / odd | net LOC (non-blank, incl. tests) |
|---|---|---|---|
| FZ01_drawing_learn_in_range | F1 | 00e26a4473af / 83bfdc931a5c | +12 |
| FZ02_key_alias_at_the_call | F2, F6 trigger 1 | a52a368e5938 / a8bd427295e6 | +54 |
| FZ03_shape_routed_meta_guards | F5, F5b, F5c | 92ff4472075b / a6f5828a54ed | +50 |
| FZ04_twin_refuses_failing_op_guards | F6 trigger 2 (T1) | ecabfffa6b25 / f9e6ccd7ac2c | +20 |
| FZ05_eager_zero_element_strides | F4 (C++ + Python twin) | 2a06e200ee73 / 16aa0d18748e | +14 |
| FZ06_plan_liveness_from_roots | F9, F15 | a409f8d0bc9c / a5c5b06c7f20 | +37 |
| FZ07_bind_opaque_empty_spans.optional | F7, F7b, F14 root cause | 6198056d9d39 / 8286d312141e | +30 |
| FZ08_remove_respec | F7/F7b/F14 by removal | 7ee4dc2b4a6c / feac4e2cab29 | -370 / -369 |
| FZ09_redo_allocations_under_pins.on08 | size-1 "allocates otherwise" | 79550a8d4886 / f54111a06931 | +2 |
| FZ10_retrace_causes.on08_09 | retrace diagnostics + guard sites | a801838f741e / 9bb0b0815189 | +76 |

## FZ01 F1: `_learn` fills a drawing op's probabilities out of range (device assert)
- Root cause: `HostTraceReplay._learn` (_host_trace_replay.py:875 even / :854 odd) fills every floating input with
  uniform(-1, 1); bernoulli's p must lie in [0, 1]. The harvest's own refill already uses (0.25, 0.75) for drawing ops.
- Fix: a `nondeterministic_seeded` op's floating inputs are filled from (0.25, 0.75).
- Test: test_cuda_host_trace_oob.py `test_a_drawing_op_learns_on_inputs_in_its_range`. Before: device-side assert on
  both lines (GB300) and GH200; after: OK (both lines, GB300; GH200).

## FZ02 F2 (+ F6's first trigger): a key's alias distances were the trace's, not the call's
- Root cause: the lowered key's `alias` was the constant `aliasing(...)[0]` at the trace's hints
  (_host_trace_lower_tape.py:454 / :492). For a provider that doesn't key aliasing (blas) the distances are never
  guarded, so every later key carried the trace's distance. `_learn` (replay.py:859-871) then put B at +20 bytes while
  key.align said 128: misaligned address (F2). The same stale 768 became an `_eager_sites` twin guard
  `Eq(2*s18*(s61 - s61//3), 768)`, false at the call: the F6 AssertionError.
- Fix (preserve the call's facts, no recovery): the distances become key rows, appended after the scalars, and
  `_opaque_key` derives OpaqueKey.alias at the call: per storage the lowest operand, distances up from it, as
  `aliasing()` does at the trace. Also `aliasing()` now returns its distances in the sorted alias order (they were in
  generation order, a latent mismatch with key.alias in `key_guards`). `key_guards` returns None (twin refused) when the
  call's lead operands are not the trace's.
- Tests (oob): `test_a_learn_lays_aliased_operands_out_as_at_the_call` (keys checked consistent with their alignment;
  before: the stale `((1, 0, 20),)` key at call 3, both lines; after OK) and
  `test_a_refused_aliased_key_twin_guards_the_calls_key` (a refused aliased mm key: before, exactly F6's AssertionError
  on both GB300 lines and GH200; after OK).

## FZ03 F5 (+F5b, F5c): checks of operands no iterator reads never became guards
- Symbols are not shared (duck_shape=False, `_TraceShapeEnv.__init__`). Cause: `_elementwise` (tape.py:2105-2132)
  routes addmm.out at K=0 to the pointwise host ("an empty operand that does not broadcast ... is in none of its
  iterators"), whose guards come only from the witness's TensorIterators (copy of bias into out). mat2 is in no iterator,
  so the host's checks against it (expand of bias to (m, n), resize_output of out to (m, n)) run concretely in the
  witness at the hints and record nothing. A call eager rejects replays (F5); a call where eager resizes out replays
  into the old layout, writing out's padding (F5c) or past it (F5b).
- Fix (general for shape-routed ops): an op routed by `_elementwise` that ATen doesn't tag pointwise also runs its fake
  kernel on the traced tensors (`_fake_call`), with the guards owned by the op. Its meta's checks (broadcast/expand,
  out= shape) become op guards.
- Tests (aten): `test_a_shape_routed_op_guards_its_meta_checks` (F5: eager's error, no replay) and
  `test_a_shape_routed_op_resizing_out_is_eager` (F5b/F5c: padded out of another n; storage bytes and metadata equal
  eager's; the retrace declines visibly). Before: both fail (GB300 both lines for the first; GH200 both); after: OK.
  Fuzz repros F5, F5b, F5c: FAIL on base, PASS with FZ03 (GH200).

## FZ04 F6 second trigger (T1): a twin built from a tape whose own op guards fail at the call
- Root cause: `_eager_sites` (replay.py:1272) builds the twin from the trace's unbound tape. At a call the variant holds
  through an entry (a redispatch's, m != 1), that tape's own op guard `m == 1` is false at the call, and lower_tape
  asserts. The AssertionError was not caught or memoized, so every later call at that key raised.
- Fix: before building, the variant's own program (`lowered.evaluate(args)`) must hold the call, i.e. every op's own
  guards. Otherwise FoldRefused ("an op's own guards fail at the call: an entry selects it"): reported in
  eager_sites_refusals, memoized, and the call traces again.
- Test (harvest): `test_a_refused_key_at_a_redispatched_call_retraces` (the targeted repro's calls). Before: T1's
  AssertionError on both GB300 lines and GH200; after: OK, refusal counted once.

## FZ05 F4: zero-element eager-step outputs compared on strides
- Root cause: the eager-step check compares strides exactly but size-1 dims (EagerStep.cpp:651-660, Python twin
  replay.py:1474-1489). A zero-element output addresses nothing, and sort/randn_like keep the input's strides there while
  their fake kernels predict contiguous: `_Disagreement` to the user.
- Fix: a zero-element output's strides are not compared, in C++ and the Python twin (eager's tensor is what the replay
  keeps). Not addressed: F4b (flip's traced host predicts contiguous strides for a zero-size output, a silent metadata
  difference in a traced host, not the check) -- needs the traced host to follow eager's zero-size layout rule.
- Build: isolated build_f4 (build_cpp7/src + EagerStep.cpp, 45 min, rc 0). Test (replay):
  `test_a_zero_element_eager_output_keeps_eagers_strides`. Before: the F4 _Disagreement (GH200; base on build_cpp7);
  after (GB300 even, F4_e on build_f4/install): OK. Fuzz repros F4 (sort) and F4c (randn_like): FAIL on base, PASS
  with FZ05; F4b still FAIL (metadata, as above). The replay file on build_f4: OK (skipped=1). The odd line's C++ is
  int5b's frozen install, so its C++ half needs a rebuild there too.
- Note (coordinator's point): `_Disagreement` is by design raised after the replay's first effects (variant dropped); this
  fix removes the zero-element trigger. Real disagreements still raise mid-replay.

## FZ06 F9, F15: the plan freed buffers an entry reads
- Root cause: plan_memory (_host_trace_memory.py:279-291) takes a launch's uses only from its PointerSlots. Entries
  (redispatch's `_redo`, fold) are accepted when their launches hold the same `roots`, not the same slots. argmax
  respecified at n = 1 (F9) or traced over a size-1 dim (F15) has a launch with no slot into v. v was freed after its
  producer, and argmax's output or the stack temporary took its block. The entry at n = 13 / size 3 then read garbage.
  The same gap covers keyed sites: record_binding gives each node roots = all operands + scratch.
- Fix: a launch uses every root it holds (slots plus `launch.roots`).
- Tests (replay, TestReplayMemory): `test_an_entry_reads_a_root_its_respecified_launch_did_not` (F9) and
  `test_an_entry_reads_a_root_its_traced_launch_did_not` (F15). Before: wrong argmax (GH200: 12 vs 0); after: OK.
  F9 and F15 repros: FAIL on base, PASS with FZ06 (GH200). GB300: the F9 test fails on base and passes with FZ06 on both
  lines (F15 passes on GB300 base: allocator reuse differs there); F9 repro FAIL -> PASS on even (odd base passes it).

## FZ07 (optional) F7, F7b, F14: bind_opaque's empty spans
- Root cause: bind_opaque (tape.py:3318) gave an op without allocations (add.out, copy_) the span
  `range(alloc_start[p], alloc_start[p])`. With no allocation at p that is past every new allocation. respec's `_splice`
  reads spans in op order, so it emitted all 12 allocations as "records of no op", then each op's again with seqs after
  their launches. The eager-order plan placed live buffers on each other: F7 wrong values (all elements), F7b, and
  F14's "base 4 is not held".
- Fix: an empty span sits at the allocations before the op's first launch. Test: `test_a_respecified_bound_tape_keeps_its_
  allocations_in_op_order`. Before: fails both GB300 lines and GH200; after: OK. F7, F7b, F14 repros PASS with it.
- With FZ08 nothing reads those spans and the test's counters (respecs) are gone: drop FZ07 if FZ08 lands.
- F7 reproduces on GB300: the fuzzer's probe/f7b.py masks it with a dead `v1c = v1.clone()`.

## FZ08 remove respec (user decision)
- Removes `respec_refused_ops`, `HostTraceReplay._respec` and its counters (respecs, respec_s, respec_refusals),
  `respec()`, `_splice`, `_rebind` and `_redo`'s respec branches, and FoldRefused.respec/.redone. Meta changes that
  redispatch can't serve now trace the graph again. Tape.respec_guards is renamed twin_guards: it now holds only
  `_eager_sites`' refused-key guards.
- Tests: 5 respec-only tests deleted; on/off-parametrized ones keep the off expectation; counters updated. No
  conformance rows referenced respec. int6's DEFAULT_OFF.md mentions it (not edited).
- A comment where a refused redispatch now traces again (`_call_held`) notes that propagating the meta change through
  uses could serve it where all kernel and tensor access goes through custom ops; engine code does not (user, approved).
- Gates GH200 even: replay 243 (1 expected count fixed by FZ09), triton 55 OK; harvest = base's failures.
- Harness note: minops84 (and the oracle) read `r.respecs`; meas/diag_meas.py stubs it.
- F7, F7b, F14 fuzz repros: PASS on the FZ08 tree (GH200). F9 still fails there: a redispatch case, fixed by FZ06.

## FZ09 (on FZ08) a redo's allocation compared under its pins
- Root cause: `_redo`'s "allocates otherwise" compared an allocation symbol for symbol: (1, 1) against the variant's
  (s61, 1), although the redo's guards pin s61 == 1. Outputs were already compared under the pins (same()). These were
  most respec triggers in minops84 (add/mul/rsqrt/mean/softmax at T=1).
- Fix: the allocation is compared as same() compares an output. Test: `test_a_redo_agrees_with_the_variants_layout_
  under_its_guards` (FZ08: 2 traces -> FZ09: 1 trace, 1 redispatch).

## FZ10 (on FZ08, FZ09) retrace diagnostics
- `HostTraceReplay.retrace_causes` maps (class, op, guard, user line, refusal) to a count for every trace after the
  first, and a `host_trace_retrace` trace_structured artifact records each. Class: meta (a graph guard an op recorded),
  dispatch (an op's own guard failed and its redispatch refused; FoldRefused.op names the refusing op and
  LoweredSelector.rows gives each own guard's row, so the failing one is named), graph (a guard of no op's), contract,
  other. Runs only on the miss path.
- Guard sites: `guard_sloc` (_host_trace_ir.py) now records every guard's user line, the innermost frame outside
  torch, as SLoc.maybe_user_loc ("dir/file.py:line"). It is cached per (code, line), so a recording costs a dict lookup
  over the frame walk it already did. Both the IR and the sympy env take it. The minmax note (framework_loc) is unchanged.
- Tests: `test_a_retrace_records_a_guard_no_op_recorded` (class graph, the test's own line), plus assertions in the
  split-op and redo-takes tests; test_a_triton_fallback_in_a_chain counts only its own artifact.
- Gates GH200 even (R+S+D, final): replay 244 OK, triton 55 OK; harvest = base's 6; aten 1668 + 1 env (cutlass) error.

## Respec measurement (minops84 36 layers, OPAQUE=0; GH200 even; all calls bitwise)
| | respec on | FZ08 | FZ08+09 |
|---|---|---|---|
| decode (64,8192) traces / op re-runs | 1 / 30 (3 respecs) | 4 / 1319 | 3 / 892 |
| prefill (8192) traces / op re-runs | 1 / 23 (2 respecs) | 3 / 741 | 2 / 384 |
Remaining retraces (FZ10, with sites): all class dispatch, op aten.sum.dim_IntList, refusal "dispatches otherwise" (its
split reduction adds a kernel and a temporary). decode: `s != 1` at minops84.py:35 and the split formula at
minops84.py:37; prefill: the split formula at minops84.py:29 (the model's `.sum(-1)` lines). Next item: entries that may
change launch count/allocations (a re-planned section + check_plan). No meta or graph retraces.

GB300 (meas/gb300_*.log, base trees, respec on / off; traces, op re-runs; all bitwise):
| | even | odd |
|---|---|---|
| decode | 1 / 4 traces, 30 / 1319 | 4 / 6 traces, 1319 / 2185 |
| prefill | 1 / 3 traces, 23 / 741 | 1 / 3 traces, 27 / 743 |
The even numbers match GH200. The odd line's decode also has graph-guard retraces with respec on: `s == 1` / `s != 1`
size-1 guards no op owns (class graph: the odd line's known b = 1 graph guard). With respec off, the extra retraces
are the same two refusals as on even ("add allocates otherwise", which FZ09 fixes; "sum dispatches otherwise").

## Not done
- F3 (memory="packed" relocate KeyError): diagnosed. RNG keyed-site nodes (randperm's "node 2", "node 4") appear in
  `captured.segments` as LoweredLaunch objects that are not `lowered.launches`' (equal seq, other objects), so
  relocate's `moved[id(c.launch)]` misses (_host_trace_memory.py:1209). Not fixed.
- S1 (logsumexp strict mode): the C++ composite raises on numel() of a symbolic tensor. Only a message match would
  classify it (not allowed). Not fixed.
- F4b: see FZ05.

## FZ06b (test only, on candidate 3: snap/step220 even / step219 odd): FZ06's first test without respec
- FZ08 removed `respecs`, and FZ06's `test_an_entry_reads_a_root_its_respecified_launch_did_not` needed a respec
  (AttributeError at int6c3 step204). It becomes `test_an_entry_reads_a_root_its_launch_did_not` (TestReplayMemory):
  the trace at n = 1 has argmax's launch with no slot into v, and the call at n = 13 is a plain redispatch whose entry
  reads v. It asserts (traces, redispatches, eager, retrace_causes) == (1, 1, 0, {}), bitwise values, and that the
  trace's plan keeps v live through argmax's launch.
- md5: .int6cpp (on step220) 97dc3556a82f / .int6 (on step219) 016b12bace69; net -6 (+9 -15).
- GH200 even (cand3_s220 + the test, gh200-b ht/fuzzfix/c3): passes. With the roots rule removed (c3n: memory.py's
  refs back to slots only) it fails: 98's check_plan raises "allocations [2] are used at step 0 (seq 4) outside their
  lifetimes". Replay file on c3: OK (skipped=1).
- GB300: not run here. The queued run was cancelled at the coordinator's request; int6's gates on the candidate 3
  tops cover it. Handed to int6 by the coordinator after this entry.

## FZ09b (test only, odd line, on snap/step241): test_a_redo_takes_a_dispatch_whose_guards_hold after FZ08
- Finding (int6 bisect): on the odd line the test fails from step201 (FZ08) on. Call 4, (1, 2016), counts 5 softmaxes, not 1.
- Root cause: an expectation, not a redo-path bug. FZ10's retrace_causes on step241 odd: (1, 1000) retraces with class
  meta (aten.index.Tensor, guard b != 1). The odd line runs int5b's frozen binary, which lacks 79's contiguity by
  construction, so the size-1 checks of index's output view (b != 1, ...) are also read outside any op and become graph
  guards. probe/idx.py shows owners {0, None} on odd and {0} on even. So at (1, 2016) the b = 2 variant is not a
  candidate. The only candidate is the (1, 1000) variant, whose sum (16 pages, not 32) dispatches otherwise: its redo
  dispatched 1 softmax (borrowed for the other 3 layers), then the call traced (4 more). Before FZ08, respec served that
  call; now it retraces, as FZ08 intends.
- Even line: index's size-1 guards are op-owned there, so (1, 2016) redispatches on the b = 2 variant (32 pages, the
  same sum split). No other route reaches this: it is a guard-ownership difference of the odd binary, not of _redo.
- Fix: the odd test expects ([4, 1, 4, 5, 0], 3, 2) and causes [("meta", "aten.index.Tensor"), ("dispatch",
  "aten.sum.dim_IntList")]. md5 952fed5201ea, net +2 (+5 -3). GB300 GPU 1, odd step241 + patch: the test passes.

## FZ11 F20 (silent wrong output): an in-place op on a slice of an eager output was taken for a new eager output -- NOT handed to int6
- Patches: FZ11_eager_step_output_is_its_argument.on_step246.int6cpp.patch (md5 0f4664c7eeae) and .on_step245.int6.patch
  (md5 e749dd49ab0a); net +41 (+43 -2, product +5 -2).
- Root cause: `_TapeLowering.eager_call` (_host_trace_lower_tape.py:418-428) treated any output of an eager step whose root
  is an eager output's root as a fresh output (a PredictedOutput of that root). It checked "is this one of the call's
  arguments" only for other roots. In F20, sum (int32) runs as an eager step, so v0's root is eager root 0. Then
  `v1[:, :1].fill_(3)` (an expanded view of v0) also runs as an eager step and returns its argument, the (3, 1) slice.
  The lowering predicted that slice as eager root 0's fresh output, so the native step (EagerStep.cpp:686, `frame.tensors[b] =
  outs[i]`) replaced root 0's tensor with the slice. Output `("eager", 0)` (Base) then returned the slice for v0. The
  fuzzer's second instance (symti, GB300 sym2 14003026: flip (complex64) eager, zero_ on its half also an eager step
  there) is the same mechanism.
- Fix: an eager step's output that is one of its arguments is that argument (the leaf index), whatever its root. A
  tensor neither made nor passed raises an AssertionError (was a StopIteration). Other eager-root checks (the root
  registration :326, fold :766, `_redo`'s fresh-output pairing) only dedupe or pair roots and are unaffected, so
  there is no other route.
- Test (replay): `test_an_inplace_op_on_a_slice_of_an_eager_output_is_its_argument`, 18 cases: fill_ on an expanded
  slice (F20), zero_ on a flipped complex64's half (the fuzzer's second instance, with its bf16 copy), fill_/zero_/
  add_/copy_ on a plain slice; each returning the base, the slice or both. Checks shapes, strides, storage offset and
  values bitwise. GB300 GPU 1: even step246 + FZ11 18/18 OK, base 2 fail (fill_expanded base/both); odd step245 + FZ11
  18/18 OK, base the same 2 fail. The other cases pass on base too: their in-place op is traced there, not an eager step.
- Repros: F20_plain and F20_inplace_on_expanded_view_wrong_output FAIL on base, PASS with FZ11, both lines (GB300).
  The complex case (probe/f20min.py full, runs/sym2/fail_14003026.py): PASS on candidate 3 base (zero_ is traced
  there). FAIL on the symti build (core/symti/py_inc7 + install), PASS with the same change applied to it
  (wip/F20_symti_probe.diff, sym_fix tree). The minimal no-out= case without the extra outputs passes everywhere.
- Rebased onto the candidate 4 tops (MERGE_LOG: odd step273, even step278): FZ11_eager_step_output_is_its_argument.
  on_step278.int6cpp.patch (md5 8c81d3e56c7c) and .on_step273.int6.patch (md5 e9a03ddd2f51), same hunks, offsets only
  (lower_tape +2 on even, test +28 on both); net +41. GB300 check on c4_{e,o} (step278 on build_cpp9 / step273 on int5b
  frozen): odd step273 + FZ11 18/18 OK, F20_plain PASS. Even step278 not run: it needs build_cpp9/install, which is
  still empty (int6's build in progress); run the test once it lands.
- symti lane: patches/F20_eager_step_output_is_its_argument.symti_py.patch (md5 8ba1fe36b98f, FZ11's product hunk
  against land/core/symti/py; that file equals py_inc7's) and a note in land/core/symti/STATUS.md.

## FZ12 (Flash-Next blocker): a keyed site inside an op with its own guards; a native rejection declines -- NOT handed to int6
- Patches: FZ12_keyed_site_inside_an_op.on_step280.int6cpp.patch (md5 1e827ad1661a) and .on_step275.int6.patch
  (md5 249257355241); net +56 (product: lower_tape +12 -2, native +5 -3; tests +45).
- Bug 1 (lowering): lower_tape's selectors (_host_trace_lower_tape.py:644/649 on step280) took every lowered launch of
  an op, so an op whose body holds a keyed site (a library kernel's harvested GEMM, Flash-Next's QSA op around the
  harvested reshape_and_cache_flash) and records guards of its own claimed the site's node too. The native variant
  requires one owner per record (VariantBuild.cpp:196 "a site's record N").
  Choice: the site keeps its nodes and the selector takes only the op's other launches. A keyed site's node parameters
  are a per-key table row, patched from the call's key; an entry holds fixed launch parameters, so moving the site's
  launch into a selector entry would freeze one key (unsound) and would need entries to carry tables (not simpler).
  With the nodes excluded, a selector entry replaces only the op's own launches, and the keyed site still serves the
  key at the call. Redispatch already refuses an op whose redo records keyed sites (`tr.sites` -> "dispatches
  otherwise"). Fold (`_fold`) drops the folded trace's keyed-site nodes from the op's records the same way, and refuses
  ("another keyed-site layout") when the sites lie elsewhere among the op's launches.
- Bug 2: native_variant (_host_trace_native.py:309) raised AssertionError for any rejected spec, which escaped into
  user code. Now a rejection declines the trace, noted in declines ("the native variant rejects the lowering (...)"),
  and the call runs eagerly. With raise_unexpected (the suites' setting) it still raises the AssertionError, so a
  lowering bug fails the tests. Not covered: native rejections at miss time (add_entry / add_row) still propagate.
- Tests: harvest `test_a_keyed_site_inside_an_op_with_its_own_guards` (ht_nested::mm_then_branch: mm + branch on
  x.shape[0] > 16, the repro's pattern; bitwise over m = 8, 40, 8, 40, 100; no selector holds a site node) and
  replay `test_a_native_rejection_declines_the_trace` (a mocked rejection: declines + eager without
  raise_unexpected, AssertionError with it).
- Evidence (GH200, int6c3 s280 = candidate 4 even on base9 + the patch): both tests OK; on s280 both FAIL ("the native
  variant rejects: a site's record 1"). gap_5: base escapes AssertionError; with FZ12: traces 2, replays 2, bitwise 5/5,
  no decline; retrace cause = the op's guard x.shape[0] > 16 (class meta, core_gaps_repro.py:68). GB300 (even step280
  on build_cpp9, odd step275): probe/chain19_{even,odd}.log, queued.
- Fuzz lane asked (land/scratch/fuzz/FINDINGS.md, "Request from the fuzzfix lane") for nested custom-op + harvested-op
  generator patterns.
- Note for the redispatch lane (not done): an op holding a keyed site can't redispatch. `_redo` refuses a redo that
  records sites (`tr.sites` -> "dispatches otherwise"), so the op's own guards become graph guards (class meta in
  retrace_causes, as gap_5's x.shape[0] > 16) and a flip retraces the graph. To redispatch such an op:
  - the redo would record the nested call as a keyed site of its own, the call's operands renamed onto the variant's
    roots as any launch's, and its key evaluated from the call's rows;
  - the entry would be a section: the op's other launches as entry launches, plus the nested site's nodes kept as a site
    whose binding row is looked up (or learned) at the redo's key, with its scratch sized per row as add_row does;
  - native would need an entry that refers to a keyed site instead of carrying fixed nodes, the site's table shared
    between the variant's own site and the entry's (same topology) or an arm for another topology;
  - check_plan on the section's allocations.
  Until then the sound rule (FZ12) is: the site owns its nodes, the selector owns the rest, and flips of the op's own
  guards retrace.

## FZ12b (on FZ12): native rejections at a miss decline too -- NOT handed to int6
- Patches: FZ12b_native_rejection_at_a_miss.on_FZ12_step280.int6cpp.patch (md5 5e085652483a) and
  .on_FZ12_step275.int6.patch (md5 f00a80cea51a); net +47 (product +31 -10, test +26).
- `HostTraceReplay._native_mutate(variant, what, add, *args)` wraps every miss-time native mutation: `_fill`'s add_row
  (refused and bound rows) and add_form, and `_add_entries`' set_program and add_entry (fold, redispatch). A
  rejection (TypeError/ValueError/IndexError/RuntimeError) is the lowering's bug. Under raise_unexpected it raises
  AssertionError("the native variant rejects a row / a form / an entry / a program: ..."). Otherwise it is noted, the
  variant is dropped from its family and the native registry (its native state may be partial), and the call goes on
  as a miss: `_fill` returns None, `_add_entries` returns a refusal ("the native variant rejects an entry or its
  program"), so the call traces again.
- Test (replay): `test_a_native_rejection_at_a_miss_declines` [row | entry]: a mocked add_row (a new GEMM key) /
  add_entry (tactic_op's new class) rejection. Non-strict: bitwise, traces 2, eager 0, decline noted. Strict:
  AssertionError.
- Evidence GH200 (int6c3 s280 + FZ12 + FZ12b): the new test and FZ12's 2 tests OK; on FZ12 alone the new test errors
  with the raw ValueError. Replay file OK (skipped=1). Harvest file: the same 6 failures as s280 itself in this harness
  (workspace_cache_on, 2 sibling tests, 3 persist subprocess tests).
- GB300 GPU 1 (probe/chain19_{even,odd}.log; the trees held FZ12 + FZ12b by then): even step280 (build_cpp9) and odd
  step275, the harvest test 1/1 and the 3 replay tests OK, gap_5 traces 2 / replays 2 / bitwise 5/5. On the bases all
  four tests fail (the site's-record AssertionError, raw ValueErrors) and gap_5 escapes the AssertionError.
