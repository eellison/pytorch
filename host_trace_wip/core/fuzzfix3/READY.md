# fuzzfix3: F8 (+F8b, F13), F10, F11, F16, F17

## Candidate 3, final tops: snap/step229 (odd) / snap/step230 (even): CURRENT

Note: int6 already applied the wip3 patches (same content) as steps 233-242 (MERGE_LOG 10-08 06:00, offsets only,
conformance 0 missing / 0 flipped on 233-242) and started candidate 3 validation on 241/242; wip4 is the same content
re-generated on fresh 229/230 copies, only for reference. To watch in int6's odd finals: on GB300 at 219 + these
patches the odd replay file had 2 F I could not compare with an odd base run (test_a_zero_element_eager_output_keeps_eagers_strides,
test_a_redo_takes_a_dispatch_whose_guards_hold); the even replay file (GH200 254 OK, GB300 220 run) has no new failure.

Re-stacked candidate 3 (without 97b(c)). Patches in `wip4/` (`.step229.int6.patch`, `.step230.int6cpp.patch`):
the step219/220 patches (`wip3/`) re-generated on fresh copies of step229/230 (`c4base_{e,o}`, torch/cuda + tests);
each applies alone with no fuzz (offsets only: F8, F16, F17 move 25 lines; F10, F11 none), and all five
together in the order F8, F10, F11, F16, F17 (offsets only). Content is unchanged from wip3.

| patch | md5 .step229.int6 / .step230.int6cpp | net (non-blank) | product | files |
|---|---|---|---|---|
| F8 | d52c4e18ede4 / 86fb888034b5 | +48 | 0 (test only) | test_cuda_host_trace_replay.py |
| F10 | 06d2573ff861 / 06d2573ff861 | +27 | +5 | _host_trace_tape.py, test_cuda_host_trace_oob.py |
| F11 | 5d262ca33c26 / 1d0599e7f0f0 | +20 | 0 | _host_trace_memory.py, test_cuda_host_trace_memory.py |
| F16 | 47a429ce17bb / 448c9344521f | +21 | +12 | _host_trace_tape.py (_free_view_strides), test_cuda_host_trace_replay.py |
| F17 | 91e4dad0626a / d2d197fb2275 | +19 | +4 | _host_trace_tape.py (_check_witness), test_cuda_host_trace_replay.py |

Verified on gh200-b (GH200): `ht/fuzzfix3/c4` = frozen int6c3/cand3_s228 + step230's Python and tests (98t is
test-only), `c4comb` = c4 + the five patches; runner gh_run4.sh = run_cand3_s228.sh's env, fresh TMPDIR per run.
GB300 not run for 229/230 (blocked); the step219/220 GB300 results below were on the same patch content.

What changed against the candidate 2 patches:
- F8/F8b/F13: none reproduces on 220 (GH200: F8 traces 1 replays 1, F8b same, F13 PASS; GB300 odd and even
  PASS): respec's `_splice`, the only reader of the bound tape's op spans that could duplicate allocations, is
  gone. The bound tape still gives ops of no allocation spans past the end (FZ07 was dropped), but every
  remaining reader (redispatch's `_redo`: `len(op.allocs)`, slices) only reads non-empty spans or slices an
  empty one, so the order no longer matters: the span-order assertion is DROPPED (it would fail on 220 for
  that benign reason), as are the respec counters. Kept: `test_a_bound_tape_at_k_one_allocates_each_buffer_once`
  (F8's and F8b's programs: bitwise vs eager through the K == 1 call, each allocation root once per variant,
  no eager call), a guard against the scenario coming back; it passes on the base (the bug is gone).
- F10: still reproduces on 220 (both lines). `_rebind` went with respec (FZ08), so the `_miss` re-bind could not
  be kept without re-adding `_splice`; the `_relower`/`_respec` parts are dropped. Instead the cause is removed
  at the trace: a keyed call with an operand on an eager step's output that has a size-1 dim no longer binds at
  trace time (the warm-up's `_check_witness` may still change that dim's stride); it is recorded as an
  OpaqueCall and binds after the trace (`bind_opaque`, oob_learn's path) at the key its rows then hold, so the
  row key and the binding's key agree and the 96 invariant (unchanged) never fires on it. _host_trace_tape.py:990
  (220). Test: `TestHostTraceStaleBinding.test_a_gemm_on_an_eager_size_1_output_binds_at_eagers_stride`.
- F11: the original repro no longer reproduces on 220 (no respec: the k=1 and 36/37 calls trace again). The
  bug is still real by construction in the refused-key twin (`_eager_sites` builds the trace's tape at the call's
  arguments, `eager.args = tuple(args)`): `_canonical`/`_forms` read the trace's hints while its rows are the
  call's. I could not drive a twin from a packed-mode program in a few tries (the calls traced again), so the
  test is now a unit test of exactly that build: `test_a_tape_lowered_at_a_later_call_plans_at_its_values_packed_{False,True}`
  traces F11's program at (36, 36), lowers its tape at (36, 37) as `_eager_sites` does, and requires the
  arenas to rename nothing; base: an arena renames a1's size into a0's (FAIL), fixed: PASS. Fix unchanged.
- F16: rebased as is (offsets only). T3's _host_trace_ir.py edits are in 219/220; `select` -> IR `where` still works
  (F16 tests pass on both lines).
- F17 (new, coordinator 10-07): see below.

### F17: the warm-up check pinned eager's size-1 stride as a constant
Exact line: _host_trace_tape.py:2937 on step230 (2927 on step229), in `_check_witness`:
`o._sym_strides = [r if n == 1 else s for n, r, s in zip(want[0], got[i][1], o._sym_strides)]`. The diagonal's own
stride never becomes 2: it stays `s18 + s50`. What becomes the constant 2 is the stride of sort's OUTPUT, where `r` is
the int the warm-up's eager sort returned. Traced value by value (probe repro/f17trace.py on 220, same code on 230):
- `v1 = a0.diagonal()` stays symbolic: strides `['s18 + s50']` (hint 2), size `Min(s13, s61)` (the view is not the problem).
- sort is an eager step; its fake kernel predicts the output contiguous: strides `['1']`. Eager's warm-up sort,
  at length 1, returns stride 2: `empty_like` of a length-1 (hence dense) tensor keeps the input's stride, the
  diagonal's real 1 + 1.
- `_check_witness` (_host_trace_tape.py:2937 on 220, 2927 on 219) sees the two differ only on a size-1 dim and
  replaces the output's stride by eager's value: `o._sym_strides = [r if n == 1 else s ...]` with `r` the warm-up's
  int: the tape's prediction row becomes `('constant', 2)` for every length.
- At (31, 12), served by a redispatch, the eager step's output check compares eager's ([12], [1]) with the
  prediction ([12], [2]): `_Disagreement`. A consumer that read the pinned stride without a check would have
  been silent.
Fix (F17 patch): the pinned stride is `select(size == 1, eager's, the trace's)` for a symbolic size (eager's for a
concrete 1, unchanged where they agree): row `('select', ...)`, no guard. After the fix the repro passes (traces 1,
redispatches 1). Test `test_a_size_1_dims_eager_stride_holds_only_at_size_1` (the repro's program, plain tensors,
4 calls, (traces, redispatches, replays, eager) == (1, 1, 3, 0)).
Captured-launch variant: I tried four (clone, empty_like + copy_, neg, cumsum of the diagonal, returned or
consumed by a launch): in all of them the traced host computes the same size-1 stride as eager, so nothing is
pinned, and the (31, 12) call traces again on base too. The pin only applies to an eager step's or traced host's
output whose prediction differs from eager on a size-1 dim, and after the trace the pinned strides are read by the
eager-step check (F17's raise), keyed sites' keys (F10) and the returned outputs' metadata; no silent variant from
built-in ops, so no second test.
Remaining gap (as flagged under F10): within the size-1 branch the value is still the warm-up's int (eager's
stride at the trace's call); a later length-1 call whose eager stride differs (another input stride) keys or
predicts that int. The eager-step check ignores size-1 dims, so it is invisible there; a GEMM key would see it.
Making it symbolic needs eager's rule per op (sort: the input's stride at length 1), which the trace does not have.

### Results (candidate 3)
GH200 on step230 (c4 = base, c4comb = all five), 10-08:
- repros: base: F10 replay_raised, F16 wrong_value, F17 replay_raised; F8, F8b, F13, F11 (packed) PASS (no longer
  reproduce). comb: all seven PASS.
- new tests, base / comb: F10 `test_a_gemm_on_an_eager_size_1_output_binds_at_eagers_stride` FAIL / PASS;
  F11 `test_a_tape_lowered_at_a_later_call_plans_at_its_values` (x2) FAIL / PASS; F16
  `test_a_gemm_on_a_size_1_view_reads_eagers_strides` FAIL / PASS; F17 `test_a_size_1_dims_eager_stride_holds_only_at_size_1`
  FAIL / PASS; F8 `test_a_bound_tape_at_k_one_allocates_each_buffer_once` (x2) PASS / PASS (guard only);
  `-k size_1` (7, incl. the renamed `..._sees_eagers_stride`) base 3 FAIL (the two new tests + the renamed one,
  which asserts eager's result where the base keeps the trace's stride) / comb 7 OK.
- gates on comb: replay 254 OK (skipped 1), tape 43 OK, memory 21 OK, oob 7 OK, aten 1676 OK, guards 12 OK,
  opaque 43 OK, capture 12 OK, cute 68 OK; harvest 217: 1 E + 11 F and triton 63: 1 F
  (test_a_record_its_launcher_disagrees_with_is_an_eager_call_drift_bit_21), the same sets on the base
  (Hopper cuBLAS picks, the -S runner's subprocess/persist tests, the private-pool/conv order effect).
GB300 on step219/220 (same patch content, before the re-stack): even F17 test base FAIL / comb PASS, F17 repro base
FAIL / comb PASS; F10, F11, F16, F8 tests and all repros as above on both lines (logs/c3gb.log, logs/c3gb2.log);
comb gates even: replay 250 (1 E: test_an_entry_reads_a_root_its_respecified_launch_did_not, `respecs` gone with
FZ08, also on the base; fixed by FZ06b in 227/228), tape, memory, oob OK; harvest 5 F (private pool + conv: the
98 test's leaked pool, the CuTe lane's pending 98t follow-up). Odd replay 188: that E + 2 F
(test_a_zero_element_eager_output_keeps_eagers_strides, test_a_redo_takes_a_dispatch_whose_guards_hold); I could not
get the odd base run for comparison before GB300 was blocked, so these two are unconfirmed against the base.
GB300 not run on 229/230 (blocked by another user's job on all 4 GPUs).

## Candidate 2 (step187/188; superseded)

Base trees: even = int6 candidate2/src_cpp (= snap/step188, build_cpp7), odd = candidate2/src_py (= snap/step187).
Work trees: `base_{e,o}` (copies), `F8_{e,o}` (F7 + F8 test), `F10_{e,o}`, `F11_{e,o}`, `F16_{e,o}`, `comb_{e,o}`
(fuzzfix's F7 + all four patches; the gate trees). GH200 (gh200-b) copies under `<path>`.
Patches in `wip/` (`.int6.patch` against step187, `.int6cpp.patch` against step188); each applies alone on its base,
all four together apply cleanly on top of fuzzfix's F7 (offsets only).

| patch | md5 (first 12) | net LOC (non-blank) | product net | files |
|---|---|---|---|---|
| F8.int6.patch / F8.int6cpp.patch | e7186fa0e485 / 4b3c76d6c39d | +53 / +53 | 0 (test only; the fix is fuzzfix's F7) | test_cuda_host_trace_replay.py |
| F10.int6.patch / F10.int6cpp.patch | ea684a325e04 / 86d7195e7794 | +52 / +52 | +31 | _host_trace_replay.py, test_cuda_host_trace_oob.py |
| F11.int6.patch / F11.int6cpp.patch | c7a07054fe5b / 00f2e4eec97e | +16 / +16 | 0 | _host_trace_memory.py, test_cuda_host_trace_memory.py |
| F16.int6.patch / F16.int6cpp.patch | 26a5bd570f02 / 4c600c0d2e3b | +21 / +21 | +12 | _host_trace_tape.py, test_cuda_host_trace_replay.py |

The .int6 and .int6cpp patches of each bug have the same hunks (line offsets differ).

## F8 / F8b / F13: respec of a bind_opaque-bound tape records allocations twice (crash)

Root cause (the same bug as fuzzfix's F7; its patch `core/fuzzfix/wip/F7.int6cpp.patch` md5 6198056d9d39,
`.int6.patch` 8286d312141e, fixes all three; my F8 patch is regression tests only, on top of it):
- `bind_opaque` (_host_trace_tape.py:3318, even line numbers) gives an op with no allocation the span
  `range(alloc_start[op.allocs.start], ...)`. The trace recorded such an op's span as an empty range at the
  trace's allocation count; when no traced allocation follows (F8: `copy_` and the eager `addmm` come after the
  last of the three traced allocations), that index is `len(tape.allocs)`, and `alloc_start[-1]` is the end of
  the NEW list, past the allocations bind_opaque inserted for the later converted GEMMs (probe: op6 `copy_` and
  op9 `addmm` got allocs 15-15 while op7/op8/op10 own 9-15).
- respec's `_splice` (_host_trace_redispatch.py:195) emits `tape.allocs[ends[1] : op.allocs.start]` as records
  of no op before each op: for `copy_` that is allocs 9..14 (the later GEMMs' outputs and workspaces), which
  those GEMMs then emit again: the respecified tape has 23 allocation records for 15 roots (4 duplicated, each
  duplicate's seq after its root's first launch).
- lower_tape maps a root to its LAST record (_host_trace_lower_tape.py:311); the eager-order plan allocates that
  duplicate in a later step (step2 tensors 15, 16, 17), so the GEMM launches of step 0 run on an address the
  commit never allocated for them: "an illegal memory access" at cuGraphLaunch (GB300, GH200). On the GH200
  (ATS: the GPU dereferences host virtual addresses) the same stale pointer wrote into host memory: F13's
  SIGSEGV/SIGBUS in the next eager step (cumsum's cudaGetDevice / the c10::Error unwinder under gdb).
- Why F8 is even-only: the odd line refuses the respec ("a memcpy in aten.copy_.default",
  _host_trace_redispatch.py:573) and retraces. F8b and F13 crash on BOTH lines (GB300 chain1: odd F8b, odd F13
  illegal address). Bisect of F8 on the even line (snap steps on their builds): step176 PASS, step178 PASS,
  step180 CRASH (85_respec_layout_under_its_pins + 86_redo_key_before_trace), 182/184/186/188 CRASH: 85 lets
  the k=1 respec proceed under its size-1 pins, which exposes the pre-existing bind_opaque span bug.
  At step178 the respec was refused ("aten.div.Tensor returned other metadata") and call 2 retraced.
- cpp_entry off still crashes (the variants' commits are native either way); the bug is in the tape, not C++.

Test (F8 patch): `TestHostTraceReplay.test_a_respecified_bound_tape_allocates_each_buffer_once_case_{k_shrinks,k_one_aliased}`
(F8's and F8b's programs as plain tensors): bitwise vs eager, after each call every variant's op spans are in op
order and each allocation root has one record; eager 0; k_one_aliased respecifies (respecs 1, no refusal) on both
lines (k_shrinks' respec is refused on the odd line, as F8's). It fails on the base at the span check after the
first call (before any crash), passes with F7. F13's program as a third case was
dropped: with F16 applied the k=1 call no longer respecifies there (the structure check of its plain-tensor form
fails on base and passes with F7 on the GH200; the oracle repro passes on comb).

## F10: _StaleBinding raised to the caller

Path: `_miss` (trace) -> `bind_opaque`/trace -> `_build` -> the 96 invariant (_host_trace_replay.py:1342-1344).
Not the twin path: the trace itself bound the GEMM. `sort` is an eager step; its fake kernel predicts a
contiguous output (stride 24 on the size-1 dim), eager's keeps the input's (27). The trace bound `v.t() @ v` at
the key with the fake's stride (a binding learned earlier at exactly that key), then the warm-up's witness
(`_check_witness`, _host_trace_tape.py:2834) set the output's size-1 stride to eager's 27. The site's rows now
give key strides `((1, 24), (27, 1), (24, 1))` while its binding is at `((1, 24), (24, 1), (24, 1))`: the
invariant fires, and `_miss` re-raises any AssertionError once a tape exists.

Fix (F10 patch): `_StaleBinding` carries the stale (site, row key) pairs; `_miss` rebinds each at its row key
(`_rebound`: bind or `_learn`, then redispatch's `_rebind`; a key that does not bind or a rebind that guards
declines visibly) and builds again; a second mismatch still raises (the invariant stays). `_relower` and
`_respec` take a `_StaleBinding` as a refusal (respec_refusals / the call traces), so it never reaches the
caller from those builds either. Test: `TestHostTraceStaleBinding.test_a_site_bound_at_a_fake_size_one_stride_binds_again`
(test_cuda_host_trace_oob.py; g learns the contiguous key, f's trace at stride 27 binds there): base raises
_StaleBinding, fixed: (traces, replays, eager, learned, harvests) = (1, 1, 1, 1, 2), the site at (27, 1).
(The test sits in its own class in the oob file: in the harvest file it ran first in its class and the process's
private-pool bytes it leaves, the same on the base, failed the later `_private_bytes() <= _POOL_KEEP` asserts.)
Note: the witness pins the size-1 stride as the constant 27 (pre-existing), so a later call at another stride
keys at 27; views of the output taken during the trace keep the fake's stride.

## F11: memory="packed": "an arena's conditions fail at the trace"

Root cause: a respecified tape is built at a later call: respec sets the program's values to the call's
(_host_trace_redispatch.py:132) while its leaves' hints stay the trace's. `_canonical` and `_forms`
(_host_trace_memory.py:871, 933) read `_hint(v)`: at call 3 (a0's dim 36, a1's dim 37) both hints were 36, so
the arena renamed a1's size symbol into a0's, and its `eq` condition failed at the build's own values; relocate
asserts (_host_trace_memory.py:1200) and `_respec` does not catch AssertionError. (At the trace hints equal
values, which is why only builds at a later call hit it: respec, and the twin, `_eager_sites`.)
Fix (F11 patch): `_leaf_values` reads the leaf symbols' values from the build's rows (`lowered.program.values`
at `lowering.row(v)`, rows the bytes rows already hold); `_canonical` and `_forms` plan at the build's call.
No refusal needed. Test: `TestPlanned.test_a_respecified_tape_plans_at_its_call_memory_{planned,packed}`:
base raises the AssertionError at call 4; fixed (traces, respecs, respec_refusals, eager) = (1, 2, {}, 1), bitwise.

## F16: bmm 1 ulp off eager (key at a free view's size-1 stride)

Root cause: `a1 @ a1.transpose(-1, -2)` is at::matmul, which reshapes each operand
(`tensor1.expand(...).reshape(...)`, LinearAlgebra.cpp) before its bmm. Eager's computeStride strides the
size-1 dim of (4, 1, 46)/(46, 1, 1) as 46; the trace's free view (`free_size_one_strides`,
_host_trace_tape.py:579-590 -> `_free_view_strides`) decides no symbolic size is 1 and keeps the input's 1.
The bmm site's key, and the harvest's operands, were `(46, 1, 1)`, which cuBLAS runs with `gemvNSP_kernel`;
eager's `(46, 46, 1)` runs `internal::gemvx::kernel` (GH200 probe: plain eager bmm on (46,1,1) picks gemvNSP,
on (46,46,1) gemvx; the replay launched gemvNSP, eager gemvx). With `free_size_one_strides=False` the key is
`(46, 46, 1)` and every call is bitwise (probe4).
Fix (F16 patch): `_free_view_strides` gives each view dim computeStride's size-1 stride where its size is 1:
the next dim's stride times its size (the last dim: the source's last stride); for a symbolic size it is a
select (`torch.cuda._host_trace.select(n == 1, eager's, free)`, a Where row, no guard), added only where it
differs from the free stride. The trace still does not branch on a size being 1; GEMM keys see eager's layout.
Behavior change: `test_a_kernel_stepping_a_size_1_dim_sees_its_free_stride` asserted the old limitation
(the T == 1 replay reads the T > 1 stride, != eager); renamed `..._sees_eagers_stride` and now asserts eager's
result. New test: `test_a_gemm_on_a_size_1_view_reads_eagers_strides` (bitwise over 4 calls, site key
`((46, 46, 1), (46, 1, 1))`); base: fails on GH200 and GB300 (values differ; the key check is deterministic on any GPU).

## Results

GH200 (gh200-b, even tree = cand2_cpp's Python, sm_90):
- before (base): F8, F8b crash (illegal address); F13 SIGSEGV (gdb: c10_cuda_check -> Error -> unwinder, in
  cumsum's eager step); F10 _StaleBinding; F11 AssertionError; F16 wrong_value (1 ulp).
- F8 tests: base FAIL (both cases), F7 PASS (both). F10 test: base raises, F10 PASS. F11 test: base FAIL (2),
  F11 PASS (2). F16 test: base FAIL, F16 PASS; `-k size_1` (6 tests) OK on F16.
- comb (F7 + all four): all six repros PASS (F8, F8b, F13, F10, F11 packed, F16).
- comb gates: replay 251 OK (skipped 1), tape 43 OK, memory 21 OK, oob 4 OK, harvest 220: the 4 known Hopper
  failures (test_workspace_cache_on error; test_scratch_only_when_referenced, test_sibling_one_capture,
  test_sibling_unverified) + test_persist_padding_pattern_{0,60}, which fail the same way on the base under this
  env (the extension's anonymous-namespace kernel name differs from the stored sidecar's): no new failure.

GB300 (gb300-box GPU 1; logs/chain1.log before, logs/chain2.log + logs/chain3.log after, gate logs logs/comb_{e,o}/):
- before (base, both lines): F10 _StaleBinding, F11 AssertionError (even and odd); F8 crash on even, odd F8
  PASS (respec refused: memcpy), F8b and F13 crash (illegal address) on BOTH lines.
- after, each patch alone on its line (even / odd):
  - F11 test: base FAIL (AssertionError x2) / FAIL; F11 PASS 2/2 / PASS 2/2; repro PASS (traces 1, respecs 2) both.
  - F10 test: base FAIL / FAIL; F10 PASS / PASS; repro PASS (traces 2, learned 2) both.
  - F8 tests: base FAIL / FAIL (both cases, at the span check); F7 + test PASS / PASS (both); comb PASS / PASS;
    F8, F8b, F13 repros on F7: PASS / PASS.
  - F16 test: base FAIL / FAIL (values differ on GB300 too); F16 PASS / PASS; `-k size_1` 6/6 OK on both;
    the F16 oracle repro passes on the GB300 base already (its data does not hit the 1 ulp there; GH200 only).
- gates on comb (F7 + F8 + F10 + F11 + F16), all PASS:
  even: replay 251 OK (skipped 1), harvest 220 OK, tape 43 OK, memory 21 OK, oob 4 OK;
  odd: replay 189 OK (skipped 1), harvest 219 OK, tape 43 OK, memory 21 OK, oob 4 OK.

GH200 broader gates on comb: guards 9, triton 56, opaque 43 (skipped 1), capture 12 OK; aten 1669 with one
error, test_rms_norm_of_a_new_hidden_size_under_a_native_override_compiles_under_the_trace: "No module named
'cutlass'" (my GH200 runner's `python -S` path lacks the CuTe DSL package; an import error, not the patches).

## Overlaps
Dry-run on comb_e (F7 + all four): fuzzfix's F1, F2, F5, F6 and fuzzfix2's T2, T3 (.int6cpp) all apply, offsets only.
- F8: depends on fuzzfix's F7 (_host_trace_tape.py bind_opaque); my test hunk is at another place of
  test_cuda_host_trace_replay.py than F7's test.
- F10: _host_trace_replay.py `_build`, `_miss`, `_relower`, `_respec`, new `_rebound` just before `_eager_sites`;
  fuzzfix's F6 edits the top of `_eager_sites` (adjacent hunk; applies with an offset). Uses redispatch's
  `_rebind` (respec's own path).
- F11: _host_trace_memory.py `_canonical`/`_forms` only (F3, packed relocate KeyError, is elsewhere in relocate).
- F16: _host_trace_tape.py `_free_view_strides` only; fuzzfix2's T2/T3 edit other parts of _host_trace_tape.py,
  T3 also _host_trace_ir.py (F16 relies on `select` -> IR `where`). Modifies an existing replay test (above).
