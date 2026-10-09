# pinned (pinned host <-> device memcpy nodes) -- merge set for candidate 4, even line only

## The set (apply in this order; the F24 fix is folded in)
| # | file | md5 | applies on | +/- |
|---|---|---|---|---|
| 1 | pinned_csrc.v2.patch | e4e27a8afe7d69fb2220c90fd1162535 | int6/build_cpp8/src (C++: host_trace Variant.h, VariantBuild.cpp, Patch.cpp, Entry.cpp) | +51 -14 |
| 2 | pinned_py.step272.patch | 8d09660f148cf77f9c12a97ce19ddd9a | int6/snap/step272 (torch/cuda/_host_trace_{capture,lower_tape,native,redispatch,tape}.py) | +137 -30 |
| 3 | pinned_test.F24.patch | 26dbb79cb4651ea0a31ca86a07d8c726 | any (new file test/test_cuda_host_trace_pinned.py, 35 tests) | +332 |

- F24 is folded into #2 (capture.py: `_live_memcpy_operands`, called by `instantiate_form` before cuGraphInstantiate) and
  #3 (its tests). Do NOT also apply F24_form_live_memcpy.patch (50b06ea189c3): that is the same fix as a delta on the
  old s242 pinned tree, kept only for history.
- #1 replaces pinned_csrc.patch (bdfbcb2e56c9): v2 moves the host allocator `record_event` out of the memcpy node-set
  branch into a per-replay loop over the segment's host memcpys (`Segment::host_copies`), so it still runs when a
  node's parameters are unchanged. Needed with learncost's LC_core (memcpy nodes set only when they change): with the
  old patch a freed block reallocated at the same address would replay with no record_event. v2 composes with
  LC_core.step242's C++ in either order (dry-run checked).
- Superseded, not part of the set: pinned_py.patch (307f78cafb10, s242 base), pinned_test.patch (6ea936941e3b),
  pinned_csrc.patch (bdfbcb2e56c9), pinned_py.step264.patch.
- Rebase onto step272: conflicts with hf1 only (capture.py imports gain MEMCPY_KINDS next to CallValues;
  redispatch.py `_Root(r.name, sym, r.kind, r.index, r.host)`; tape.py `_Root.host` after hf1's `index`). hf2-8, TH
  and ceilint apply with offsets.

## Even only
The Python half needs the C++ half: the native record of a memcpy gains (cudaMemcpyKind, host argument), and the odd
line's frozen binary would read only dst/src/bytes and patch an H2D/D2H node with the D2D kind (SetParams fails at
replay); the contract key's pinned bit and the bound-call pinned reads are C++ too. Nothing goes on the odd line.

## F24 (fuzz: segfault in cuGraphInstantiate)
- Cause: a keyed-site form is instantiated from a clone of the variant's graph. The clone's memcpy nodes still hold the
  capture's operands (`_copy_operands`: a device scratch and, for H2D/D2H, a pinned scratch), freed after capture. The
  driver reads a memcpy node's pointers at instantiation, and a freed pinned block (released by
  `torch._C._host_emptyCache`) crashes it (GH200) or spins (GB300, 100% CPU).
- Fix: before instantiating a form, every memcpy node of the clone is pointed at live scratch of its kinds of memory
  (device, pinned host), held until the exec is instantiated and uploaded. A replay sets every memcpy node before each
  launch, as for the capture's exec, so the exec never runs with the scratch addresses.
- Same hazard elsewhere: instantiate_form is the only clone/re-instantiation path (the variant's own exec is
  instantiated in `_segment` while the scratch is live; C++ instantiates nothing). Kernel nodes' pointers are opaque
  parameter bytes, and memset destinations are not dereferenced at instantiation (eager-output placeholders are
  non-canonical addresses and instantiate fine), so memcpy nodes are the only ones that need live operands.
- Tests (in #3): test_a_form_instantiates_after_the_traced_buffers_are_freed[d2h, h2d, d2d] (the repro's sequence, with
  device and pinned caches emptied between calls; d2d covers the freed device scratch) and
  test_a_replay_on_the_same_buffers_after_a_new_trace (repros/F24_pinned_d2h_segv.py's sequence). On the unfixed
  tree the d2h case hangs at 100% CPU on GB300 and the F24 repro process dies after call 1.

## Results
- GH200 (gh200-b, sm_90 build of build_cpp8 + csrc v2, s242 Python + pinned + F24): pinned 35 OK, replay 254 OK
  (skipped 1, = the s242 baseline).
- GH200, the exact set (tree272 = csrc v2 binary + rebase272 = step272 + #2 + #3): see "exact set" below.
- GB300 (c8x/install = build_cpp8 + csrc v1, s242 Python + pinned + F24): F24 repro REPRO PASS (all 4 calls, the
  form path ran: replays 2), F24_pinned_d2h_segv REPRO PASS, pinned 35 OK, replay 254 OK.
- GB300 with csrc v2 (job_f24d / job_f24e, queued on gpu1.lock): see below.
- Earlier (pinned without F24, see the lane report): GB300 python_entry 515 FAIL 9 (= candidate 3), GH200 python_entry
  failures identical to cand3_s242; capture/tape/lower_tape/memory/opaque/guards/triton/aten OK on GH200; vLLM arm VP
  pinned: 0 operations outside the graph per decode step, --check 100/100 bitwise, tokens = V arm 402/402.

### exact set
- GH200 tree272 (csrc v2 binary + step272 + #2 + #3): pinned 35 OK, replay 257 OK (skipped 1), F24 repro REPRO PASS
  (logs: gh200-b ht/pinned/logs/r272/).
- GB300 c8x/install with csrc v2: F24 repro REPRO PASS (s242 Python + pinned + F24); pinned/replay on s242, rebase264
  and rebase272 queued on gpu1.lock (logs/f24d.summary, logs/f24e.summary).

# F24k (2026-10-09) -- final set, on candidate 4's even line (step288 Python, build_cpp9 C++)

## The set (two independent patches; apply both)
| # | file | md5 | applies on | +/- |
|---|---|---|---|---|
| 1 | F24k_host_memcpy_every_replay.build_cpp9.csrc.patch | f75e5bacb19242314d0bd6161b7c4e95 | int6/build_cpp9/src (host_trace/Patch.cpp, VariantBuild.cpp) | +10 -4 |
| 2 | F24k_form_memcpy_kinds.step288.patch | bca73c967531bbcd58e6c9a4aecd4f02 | int6/snap/step288 (capture.py, replay.py, test_cuda_host_trace_pinned.py) | +132 -23 |
- Even line only. #2's Python runs on any build, but its new tests need #1's binary.
- Supersedes: F24k_form_memcpy_kinds.step280.patch (b7d00a3d0101, then db0bf8bcbe59: same Python, step280 base,
  without the last test), the earlier F24k csrc (7ae969af262a: no VariantBuild hunk).
- step288 rebase: one conflict (FZ12b wraps add_form in _native_mutate; the `values` argument goes on that call).

## #1 (C++): copy records under LC
- Host memcpys set every replay: LC sets a memcpy node only when its rows change, and a pinned host node whose block
  was freed and another registered at the same address (rotated slots, `_host_emptyCache`) launched stale. Unmodified
  step280 (GH200, build_cpp9's exact host_trace sources): test_a_bound_call_reads_rotating_buffers_per_call wrong data
  (511/512), the F24 h2d form test CUDA_ERROR_LAUNCH_FAILED then abort in the pinned allocator's free, the d2h form test
  a cuGraphLaunch segfault. Host (H2D/D2H) records: always_ = 1 in index_users and no held-skip; device memcpys keep
  LC's skip.
- Copy records start not held (symm lane's finding): the build marked a memcpy as holding the traced call's addresses,
  but its node holds the capture's operands (_copy_operands). A later call whose intermediate lands at the traced
  address would skip the set and copy from or to the freed placeholder. VariantBuild leaves memcpy records not held, so
  the first replay sets them.

## #2 (Python): a form's memcpy nodes keep their capture's kind of memory
- F24's _live_memcpy_operands pointed every memcpy node of a form's clone at caching-allocator scratch; the driver ties
  a node to its operands' kind (cudaMalloc'd vs cuMem-mapped: symm_mem, expandable segments), so a cuMem argument copy
  in a segment that gains a form failed at every replay of the form ("Replacement operand type is incompatible with
  existing operand"). _fill passes the call's rows to _form; an argument operand is the call's own address (live for
  the call, the argument's kind, the memory a replay sets it to), a non-argument one caching-allocator scratch (where
  the capture's placeholders and a replay's allocations come from), at the call's width. Composes with symm's item 1.
- Tests (pinned file, now 38): test_a_form_keeps_a_mapped_argument_copy (capture under expandable segments with cuMem
  src/dst arguments, allocator back to cudaMalloc, a new mm key forms the segment); test_host_copies_across_freed_and_
  reregistered_blocks (30 replays over rotating slots and blocks freed and registered again, host output checked each
  call); test_a_copy_from_an_intermediate_at_the_traced_address (out.copy_(x * 2), the intermediate at the traced
  address on later calls).

## Results
- GH200, step280 + #2 (minus the last test) on build_cpp9 host_trace + #1 (minus the VariantBuild hunk): pinned 37 OK,
  replay 276 OK. #1 alone (unmodified step280 Python): only the kind test fails. Neither: the failures above.
- GB300 unmodified step280 on int6's build_cpp9/install: the kind test fails at the form call (same driver message).
- Final set (step288 + #2 on build_cpp9 + #1): GH200 and GB300 runs in progress (logs/c9x2.summary; gh200-b
  ht/pinned/logs/k288/summary).
