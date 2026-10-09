# redispatch2: a redispatch entry may change its op's launches and allocations (sections) -- READY for int6 candidate 4 (RD2, c7 patches)

Scheduled for candidate 4 (coordinator). The user's guard-failure model (DECISIONS.md 2026-10-07): a node-meta guard
retraces the graph; an op-owned guard whose redispatch dispatches otherwise (launch count or allocations differ: sum's
split reduction) is served by a *section*: the op's chain in the graph replaced in place by the redo's nodes (a graph
piece in a form, instantiate_form), with the op's allocations replaced too (the outputs' allocations stay the
variant's, the others become the section's temporaries sized at the call, the op's other allocations are not made).
No respec, no new tape, no rebuild of the variant.

## Patches (candidate 4: int6 snap/step272 even, step271 odd)
| patch | md5 | net LOC (non-blank) |
|---|---|---|
| RD2_section_entries.c7.int6cpp.patch (step272) | 2475f90622e7 | +679 -120 = 559 (torch/cuda +297 -59, csrc +216 -27, tests +154 -34; + pyi.in) |
| RD2_section_entries.c7.int6.patch (step271) | 0202a74bc851 | +340 -62 = 278 (torch/cuda +298 -59, tests +42 -3) |
Trees: `c7base_{e,o}` (snap copies), `c7_{e,o}` (= `c7rd2_{e,o}`, + RD2). Rebased from the c6 patches with the hint chain's
mechanism: a section temporary's address symbol takes its placeholder from its root's index (`_placeholder(a.root)`,
hf1_at_call) instead of the redo's hint, the temporary root keeps that index, and a SymBool's made symbol records its
`at_call` value as every made symbol does; `_host_trace_redispatch.py` reads no hint. Only the replay import hunk needed a
hand merge (hf6's `takes_generator`).
C++ (int6cpp only): step272's csrc = build_cpp8's (checked), so build_cpp8/src + this csrc (`c3build/__init__.pyi.in` is
build_cpp8's pyi.in + the stub); builds: GB300 `build_rd2/install`, gh200-b `ht/rd2/install` (`--target`, env untouched).
Earlier tops: `.c6.*` (step246/245), `.c5.*`, `.c4.*`, `.c3.*`, plain (superseded).

## Acceptance: vLLM Qwen3-8B open variant (trtllm fork + silu port), decode bs 5/8/64 --check (coordinator's case)
`vllm_accept.sh` (GB300 GPU 1, TAG=_2 / _3; launcher `python_vllm_rd2.sh` = the integration launcher with CAND_LINE=rd2 ->
c6_e on build_rd2), out integration/vllm/out/rd2_accept_2.json vs the c3 baseline out/c3nan_c3.json (step246, build_cpp8):
| | c3 baseline | redispatch2 |
|---|---|---|
| decode bs 2..32 bucket (decode/T1/fam) | 3 traces, 3 variants, 0 redispatches; retrace_causes: dispatch / trtllm_paged_attention_decode split-count guard (trtllm_trace.py:438) "another launch topology" x2 | 1 trace, 1 variant, 2 redispatches, retrace_causes {} |
| prefill T2 bucket (prefill/T2/fam) | 3 traces; causes: block_table.py:222 graph guard, dispatch / aten.copy_ "a memcpy" | 2 traces, 1 redispatch (the copy_'s memcpy section, final patch); cause left: the graph guard (case (ii)) |
| all entries | 7 traces, 7 variants | 5 traces before the memcpy nodes (rd2_accept_2.json), 4 with them (rd2_accept_3.json) |
| checks | 92/92 bitwise | 92/92 bitwise (both runs) |
| plan_dump used-before-allocated | 0 | 0 (check_plan ran on every build, no assertion) |
The fork's split count n sets the CGA cluster dimension (a launch attribute instantiation fixes), so the redo had the
same node and allocation counts but another topology: the section trigger is now "another launch chain" = a count
differs or a launch's fixed topology differs (kernel attributes/cluster, programmatic edge, compiled cluster, memset
element size); the piece is captured with the redo's attributes. Unit test: triton
`test_a_programmatic_choice_in_a_dispatch_unit_is_a_section` (a launch_pdl choice past 64 rows: 1 trace, 1 section).
Same change, earlier tops: `.c4.*` (step228/227: 980417d366a4 / b31b3104fd0e, = `.c3.*` on step222/221).
The F8-F17 steps touch `_host_trace_memory.py` (`_leaf_values`, hint-free arena sizing) away from this change.
Earlier base (candidate 2 + FZ06/08/09/10 + 98's check_plan + T3; trees stack_*/work_*): RD2_section_entries.int6cpp.patch
c79ae65db199, .int6.patch cbc7576ef110 (same change; superseded by c4).
C++ (int6cpp only, as 79/95): step228's csrc = build_cpp8's (checked), so the build is build_cpp8/src + this csrc and
build_cpp8's pyi.in + this stub (`c3build/__init__.pyi.in`). Built on gh200-b as ht/rd2 (`--target ht/rd2/install`,
build dir ht/rd2/build, the env untouched; rc 0). GB300 `build_rd2` (build_cpp7 base) is stale for candidate 4: a
GB300 build is build_cpp8/src + the c4 csrc (not done: GB300 off limits).
The odd line's binary is frozen int5b: its Python feature-detects `torch._C._host_trace_section_entries` (absent ->
the redo refuses "dispatches otherwise; the native replay takes no section entries", counted in FZ10's
retrace_causes; no Python rebuilt-variant path, coordinator's call).

## Acceptance: copy_ that becomes a memcpy (torch only; integration/vllm/probe/retrace_class_repro.py case (i))
`out[:n].copy_(x[:, 0])`: copy_'s host takes a memcpy at n == 1 (the strided source is contiguous) and a copy kernel
otherwise. c3: traced at n=4, retraced at n=1 ('dispatch', 'aten.copy_.default', (n-1) != 0, 'a memcpy in ...').
redispatch2 (gh200-b, c6e): n = 4, 1, 1, 6, 1: 1 trace, 1 redispatch (the memcpy section), every call bitwise,
retrace_causes {}. Case (ii) (full-width slice, a graph guard of no op's) still retraces once, by the model.
Memcpy nodes in sections: a redo whose launches include a memcpy is always a section (a plain entry has no memcpy
row); native Entry.memcpys + memcpy_row (patched every replay, as the tape's memcpy records); the piece's memcpy is
captured between two halves of a caching-allocator buffer made before the capture (a memcpy node keeps its
capture's kind of memory); the entry's predicate requires >= 1 byte and both slots inside their allocations
(the tape's own memcpy requirement, lower_tape). Core test: replay `test_a_copy_that_becomes_a_memcpy_is_a_section[first=4|1]`
(either way round: 1 trace, 1 section, bitwise, check_plan clean).

## SymBool arguments (the notebook lane's Triton case)
A redo refused any SymBool/SymFloat argument ("takes a symbolic SymBool"), so a launch with a constexpr of a size's
condition (INT64 = n >= 2**31) retraced at every specialization flip. A SymBool argument now binds as a SymInt: a
fresh 0/1 symbol b of the redo's env whose formula is where(condition, 1, 0) in the variant's context (owned there,
so T3's ownership holds), passed to the op as `b != 0`; the op's guards on it rename into the variant's symbols like
any other. A constant SymBool is its bool. SymFloat stays refused: land/core/symfloat has no DESIGN.md yet (only
scratch/), and the binding should follow its representation; SymBool does not depend on it.
Core test: triton `test_a_symbolic_bool_argument_redispatches` (a WIDE=n >= 2**31 constexpr; the counts must equal
the constant-False run's). Probe (gh200-b): before, wide 3 traces / 0 redispatches (refusal "takes a symbolic
SymBool" x2) vs constant 1 / 2; after, wide 1 / 2 = constant.

## Design (review order)
1. `_host_trace_redispatch._redo`: launch/allocation count differs (one op of the same kind, outputs same() under the
   pins as today) -> `Section`: each redo allocation that an output's root is maps to the variant's allocation for that
   output (layout compared under the pins, FZ09); every other one is a `Temporary` (its address a fresh symbol of the
   variant's context no guard reads, sizes/strides renamed through IRRename, so only from the call's inputs; T3
   refuses foreign nodes); the op's unpaired allocations are `drops`. Each section launch reads only roots the op's own
   launches hold or its temporaries; RNG/CPU-scalar/memcpy/TMA (without entry descriptors) refuse; eager-kind ops and
   the sympy rename refuse. Same-count redos keep today's arm-0 entries.
2. `lower_entries`: temporaries lowered (bytes rows, their stride requirement ANDed into the predicate), pointer slots
   into them at bases past the tape's (`("section", base)`); entries carry `LoweredSection` (or None).
3. `_host_trace_memory.check_section` (check_plan's liveness, factored as `_liveness`; a tensor is live from its seq in
   its step): the op's chain is one run, every other base a section launch reaches is live over the whole chain (at its
   first and last launch), none is a drop or another section's. Refusal -> FoldRefused, redispatch_refusals,
   retrace_causes; all entries of the call or none. check_plan itself still runs on every variant build and passes
   after sections are added (tests). `section_places`: where the temporaries are made/freed in the run's eager order.
4. Native (`Sites/VariantBuild/EagerStep/Entry.cpp`, `Variant.h`): `add_entry(site, predicate, nodes, arm, temporaries,
   drops, alloc_at, free_at)`: arm > 0 = a piece (any node count/kinds); temporaries = bytes rows at new bases
   (base_count_ grows); drops mark `Allocation.owner`. Selectors join their segment's `g.sites` (arm 0 in every existing
   form; form_of runs only for armed segments). `allocate`: a selected section's temporaries are made and freed where
   eager makes them (in the run's eager order at the op's places; a run of no order and no temporaries: among its
   tensors, under the allocator lock as eager order is); in a run-buffer run (or with replay hooks) before the run's own
   allocations, held until it is queued. A dropped allocation is not made (0 bytes / no tensor).
   Fix on the way: `evaluate_py` computes forms only once every selector selects (a selector row of -1 indexed
   entries[-2] once selectors are in g.sites: the segfault seen first).
5. `_host_trace_replay`: `_add_entries` checks every section, then adds it with its `section_binding` (capture.py:
   the launches as OpaqueKernel/OpaqueMemset at the call's values, for `_capture_piece`) as the selector's arm;
   `_form` lists a segment's selectors after its keyed sites (only where the native takes sections), a selector arm is
   always a piece. Counters: `sections`, `section_bytes` (max bytes of one entry's temporaries at the call that added it).
Default ON, no switch. Plain hits: no new work (no armed segment, no section).

## Tests
- replay: `test_a_split_reduction_is_a_section` ((x*2).sum(0)+1, T 100 -> 5000: 1 trace, 1 redispatch, 1 section,
  bitwise, check_plan passes), `test_a_split_in_an_op_is_a_section[adjacent]` (was ..._traces_its_class: 3 traces ->
  1 trace, 2 redispatches, 2/4 sections; adjacent = two pieces in one form), `test_a_section_needs_its_roots_live_over_its_op`
  (check_section passes on the real plan, rejects a plan freeing the sum's input inside the sum), `test_a_section_traces_again_where_the_native_replay_takes_none`
  (must still retrace: 2 traces, retrace cause dispatch/sum with the native refusal; also in the odd patch).
  Updated: `test_a_redo_takes_a_dispatch_whose_guards_hold` (2 traces -> 1, sections 4), `test_a_refused_redispatch_tries_the_refusing_op_first`
  (the refusing op is now enqueues_past_32: split_op is served).
- aten: `test_another_launch_chain_is_a_section[sections]` (was test_fold_refuses_another_launch_chain; layer_norm
  767/768: a section, or with sections off the fold refusal as before), `test_minmax_replays_new_sizes_in_one_trace`
  (_all cases: 2 traces -> 1 + a section). The norm/reduction/minmax memory-equality tests (replay peak == eager peak,
  now checked on these calls because they no longer trace) pass: they are what required drops + eager-order places.
- harvest: `test_new_shapes_are_local[oob=True]` 2 traces -> 1 (2 redispatches, 1 section).
- triton: `test_a_python_kernel_choice_in_a_dispatch_unit_dispatches_again[allocates]` 2 traces -> 1 (a section).

## Gates on candidate 4 (step272 / step271, the c7 patches)
gh200-b even, c7e (rd2 build + RD2) vs c7s (build8 + step272), the suites running the hint audit strict (their default):
replay 262 OK (incl. test_heuristics' new expectation: 1 trace, 3 redispatches), tape 46 OK, harvest 217 with 5 failures
+ 1 error = c7s's same set (Hopper cuBLAS/cuDNN choices, persist, workspace_cache), memory 21 OK, triton 66 OK, aten 1677 OK.
c7s replay 257 OK.
Strict hint audit on the rebased tree (TORCH_HOST_TRACE_HINT_AUDIT=strict): minops84 decode36 1 trace / 23 re-runs, 72
sections, 11/11 bitwise; prefill36 1 trace / 23, 1 section, 16/16 bitwise; no HintAuditError.
Memory and host time on candidate 4 (GH200): decode replay <= eager everywhere; prefill worst +16.5 MB at T=2000 (the
variant's plan, as without sections); plain hit 71.0-71.1 us vs c7s 71.0 (unchanged); a section form's hit 78.3-78.5 vs a
separate variant's 75.0-75.2 (the known ~3 us, follow-up).
GB300 odd (c7_o, frozen binary): replay 196 (1 failure = the base's FZ05 zero-element test), tape 46, harvest 216, memory 21
OK; triton, aten and the vLLM acceptance on c7_e: see the end of this file (running when written).

## Final gates (step246/245, the c6 patches)
gh200-b even (c6e = rd2 build + this): replay 259 (1 failure: test_heuristics' old expectation, updated and passing
on its own: 1 trace, 3 redispatches, the SymBool EVEN constexpr), tape 44, harvest 217 (5 failures + 1 error, a subset
of the base's Hopper set), memory 21, triton 65, aten 1677: all OK otherwise.
GB300 odd (c6_o, frozen binary): replay 193 (1 failure = the base's FZ05 zero-element test; test_heuristics passes
with SymBool binding, which is pure Python), triton 64 (10 skipped) OK.
GB300 even (before memcpy nodes and SymBool): replay, tape, harvest, memory, triton, aten all OK; after: the section,
memcpy, triton and aten subsets OK.

## Gates on step246/245 (GB300, before the memcpy-node change; logs/c6e, logs/c6o, logs/c6bo)
Even (c6_e on build_rd2, GPU 1): replay 257, tape 44, harvest 217, memory 21, triton 64, aten 1677: all OK.
Odd (c6_o on the frozen int5b binary) vs its base c6base_o: replay 193 / 192 with the same 1 failure
(test_a_zero_element_eager_output_keeps_eagers_strides: FZ05's C++ half is not in the frozen binary), tape 44, harvest
216, memory 21, triton 63 (10 skipped), aten 1674: all OK on both. (The candidate 2-era odd failure of
test_a_redo_takes_.. is gone.) Odd minops84 decode on c6_o (GB300): 6 traces, 23/23 bitwise; causes: aten.sum
"the native replay takes no section entries" x2 (by design on the frozen binary) and 3 index.Tensor meta retraces.
After the memcpy change: replay section/memcpy/redo subset 20 OK (gh200-b); triton programmatic + dispatch_unit OK and
aten another_launch_chain/minmax/copy 15 OK (GB300); full gh200-b gates on c6e queued (ht/rd2/logs/chain_c6.summary).

## Gates (gh200-b GH200, even line; logs ht/rd2/logs/<tag>/)
Final top step242 (c5e = rd2 build + this, c5s = build8 + step242; `gh/chain_c5.sh`):
| file | c5e | c5s |
|---|---|---|
| replay | 257 OK (skipped 1) | 254 OK (skipped 1) |
| tape | 43 OK | |
| harvest | 217: 11 failures + 1 error, the same set as c5s (Hopper cuBLAS/cuDNN choices, persist, workspace_cache) | 217: same |
| memory | 21 OK | |
| triton | 63 OK | 63 OK |
| aten | 1677 OK | |
(c5s ran only the files with known failures, for the comparison; step228's c4e had replay/tape/memory OK and the same
harvest set before it was stopped for step242.)
Step222 (c3e / c3s, every file both arms): identical failure sets (replay FZ06 respec test, harvest 11+1, triton
drift_bit_21), aten 1677 / 1676 OK.
Odd line: not gated on candidate 3/4 (GB300 only; GPU 1 off limits since 10-07 ~20:00, then all GB300 GPUs taken).
Earlier, on the candidate 2 stack (GB300 GPU 1): replay 185 with 2 non-OK (FZ06's respec test; test_a_redo_takes_..
count 5 vs 1 at (1, 2016), not seen on any even run, not compared with stack_o before the GPU was withdrawn), tape OK;
minops84 prefill 2 traces / 388 re-runs, all bitwise, the refusal "the native replay takes no section entries"
reported as the retrace cause (as designed).

## Measurements (GH200 even; identical on the candidate 2 stack, step222 and step242)
minops84 36 layers, OPAQUE=0, all calls bitwise:
| | respec era | without sections (c3s / stack) | with sections (c3e / work_e) |
|---|---|---|---|
| decode (64,8192) traces / op re-runs | 1 / 30 | 3 / 892 | 1 / 23 (72 sections, 11/11 bitwise; step242 the same) |
| prefill (8192) traces / op re-runs | 1 / 23 | 2 / 384 | 1 / 23 (1 section, 16/16 bitwise; step242 the same) |
retrace_causes: empty with sections (without: dispatch / aten.sum.dim_IntList x2 decode, x1 prefill).
GB300 (build_rd2, candidate 2 stack) prefill: 1 trace / 23, 16/16 bitwise, same section bytes.
Section bytes (largest one entry's temporaries): prefill 589,828 B (minops84), 1,048,580 B (mem.py at T=8192);
decode 0 B (its sections are split -> one kernel: drops only). Far under 16 MiB. Suite rows: not run (GB300 only).
Memory (mem.py, replay vs eager peak, same inputs): decode replay <= eager at every shape (as without sections);
prefill worst +16.5 MB at T=2000 with and without sections (the variant's own plan); at T=8192 / 5000 the section
form is +5.2 / +2.0 MB vs eager where the separately traced variant was 0 / -0.8 MB: the variant planned at 100-700
rows serving 8192 (run-buffer runs hold the section's temporaries over the run), within the max(64 MiB, 5%) rule.
The aten memory-equality tests (replay peak == eager peak) pass on calls a section serves.
Host time per hit (hit.py: 8-layer softmax + split sum, interleaved A/B on the same GPU, us/call):
| | c3s (no sections) | c3e |
|---|---|---|
| plain hit | 71.72 / 71.55 / 71.06 | 71.70 / 71.12 / 70.92 |
| plain hit after a section exists | 71.60 / 71.48 / 71.03 | 71.71 / 71.13 / 70.91 |
| hit at 6000 rows | 75.45 / 75.06 / 75.15 (its own traced variant) | 78.50 / 78.34 / 78.33 (the section form) |
| step242: plain / after a section / at 6000 | 71.80, 71.63 / 71.67, 71.55 / 74.46, 75.16 | 71.12 / 71.14 / 78.36 |
Plain hits unchanged. A hit in a section's form costs ~3 us more than a dedicated variant would: piece nodes are
packed on every replay (as keyed pieces already are) and the temporaries take allocator calls. Follow-up if it matters:
dirty tracking for piece records.

## Open
- Odd line on step241: gates and minops84 (GB300 only; the earlier candidate 2 odd run had test_a_redo_takes_.. at
  count 5 vs 1, never compared with its own base).
- GB300 even build for candidate 4 = build_cpp8/src + this csrc (`build_rd2` is on build_cpp7).
- Suite rows' section bytes (GB300 sweep harness).
- Hint reads: a temporary's address symbol takes the redo's placeholder (`_hint(a.q)`) as its hint, as the redo's
  allocation roots already do; it is never read into a guard or a row (out of the program, like an allocation's q).
