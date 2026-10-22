# core next round (L/core/next) -- NOT handed to int6

Base: the int6 candidate tops. `.int6cpp.patch` is against snap/step188 (even, C++ line, build_cpp7 via
int6/python_int6_build7.sh); `.int6.patch` against snap/step187 (odd, Python line, install_land_int5b_frozen via
int6/python_int6.sh). Apply with `patch -p1` from a tree root. Working copies: src188/, src187/ (= the snaps + the patch).
GPU runs: gpu1.sh / tests.sh (GPU 1, gpu1.lock, CPUs 72-107); logs in logs/.

## N1b: python_entry skip for N1's renamed hand-back test (test-only follow-up, on snap/step242) -- READY
- File: N1b_python_entry_skips_renamed_handback_test.on_step242.int6cpp.patch, md5 ca82c3f2a9ebfe65f574a9a9673ff280.
  Even line only (test_cuda_host_trace_python_entry.py exists only there). +1 -1, test/test_cuda_host_trace_python_entry.py
  (before bf6c101e76c2e6020c4114fa23aaef0a, after 7af54a29dcd20b007dd9dcced05f9fbe). Applies to step242 with no offset or fuzz.
- Cause: N1 renamed test_a_trace_that_does_not_run_hands_the_call_back to test_a_slow_call_that_does_not_run_hands_the_call_back
  (parametrized). _CPP_ONLY["TestReplayMemory"] skips by name prefix, so the old entry stopped matching and the new test began
  running under the Python entry. The patch swaps the name in that tuple (it stays sorted).
- Test-only, not a behaviour gap: the Python entry hands back the same way. Probe hb/probe_handback.py on step242, both
  entries x redispatch True/False, on GB300 (hb/probe_gb300.out) and GH200: the Handback comes at offset 1 only, then traces,
  replays, redispatches and eager are the same on both entries ((1, 3, 1, 0) with redispatch, (2, 3, 0, 0) without). Only the
  peak of the handed-back call differs: C++ 4n, Python 8n, because the Python dispatch holds the argument tuple until the
  replay ends. That is the documented reason the predecessor is in _CPP_ONLY. The finals' "0" for [True] is noise in the full
  file: run alone or as the pair, both params give 33554432 (8n) on GB300 and GH200, every time. That fits garbage left by
  earlier tests being freed during the call.
- Verified (GB300 GPU 1, gpu1.lock, step242 + patch): python_entry -k TestReplayMemoryPythonEntry: Ran 25, OK (skipped=7),
  both params "skipped 'a test of the C++ entry'". The C++ class still runs and passes the test (finals: replay 254 OK).

## N1 rebased onto candidate 3 (FZ08, no respec) -- THE FILES TO MERGE, NOT handed to int6 yet
- Files (rebase of N1 onto int6 snap/step211 odd / snap/step212 even, which carry FZ08 remove_respec + FZ09 + FZ10):
  - N1_trusted_redispatch.on_fz08.int6.patch (odd, vs step211), md5 569660bc5bc909974bfb748f03f961cc
  - N1_trusted_redispatch.on_fz08.int6cpp.patch (even, vs step212), md5 a11b6beb51172eba45785b7726122db6
  - Each +65 -14 (non-test +8 -8 in _host_trace_replay.py and _host_trace_redispatch.py; net +51). Regenerate:
    mkpatch2.sh (working trees src211 / src212 = snap + port_n1.py + port_n1_tests.py).
  - The older N1_trusted_redispatch.{int6,int6cpp}.patch (2026-10-06 18:47, vs step187/188) are superseded: they do not
    apply on FZ08 trees.
- Applies with no offset and no fuzz (patch --dry-run) on snap/step211, 212, 213-218 and the current tops step219
  (odd) / step220 (even); fresh copies src219 (odd) and gh200-b int6c3/n1_220 (even, = int6's s220 tree) patched cleanly.
- What changed vs the original N1:
  - Same code intent and the same edits: the redispatch loop in `_call_held` runs for trusted entries (the
    `if self.trusted is None else ()` skip removed) and a redispatch-served call hands back under `handback`;
    `_redo` admits a sympy (trusted) variant when the fresh env is IR (`symbolic == "ir"`), drops the dead
    `_TraceShapeEnv` fresh env and its refusal, getattr for `_id_skips`, `ctx` only where ir. Only the surrounding
    context moved (FZ08 removed `first_alloc`/respec from `_redo` and the respec loop from `_call_held`).
  - Dropped: test_a_trusted_split_respecifies (respec no longer exists). Under FZ08 a refused redispatch under trust
    retraces, as untrusted does, counted in FZ10's retrace_causes.
  - Kept tests: test_a_trusted_misaligned_input_redispatches, test_a_trusted_view_at_a_size_dependent_offset_redispatches,
    test_a_slow_call_that_does_not_run_hands_the_call_back[redispatch True|False] (rewrite of
    test_a_trace_that_does_not_run_hands_the_call_back), inductor TestInstallation.test_a_launch_alignment_flip_redispatches.
- Rebase runs (even line on gh200-b GH200, int6's int6c3/s220 tree = step220 Python on install8 (build_cpp8's C++,
  sm_90), n1_220 = s220 + the .int6cpp patch, python -S, one hold of ht/gpu.lock per run, 2026-10-07 18:32-19:45; logs
  copied to logs/gh200/). Every probe row bitwise vs eager:

  | case | s220 (no N1) traces, redispatches | n1_220 |
  |---|---|---|
  | (a) trusted input view, alignment flips (8 calls) | 2, 0 | 1, 1 |
  | (b) trusted view at a size-dependent offset (8 calls) | 3, 0 | 1, 2 |
  | (c) Inductor custom op, Triton on a moving view (10 calls) | 2, 0 | 1, 1 |
  | trusted tactic op (9 calls) | 3, 0 | 1, 2 |
  | trusted split op / two adjacent split ops (9 calls) | 3, 0 | 3, 0 (redispatch refused, retraces) |
  | untrusted (a), (b), tactic, split, reference | 1,1 / 1,2 / 1,2 / 3,0 | unchanged |

  Without N1 the trusted retraces are counted in FZ10's retrace_causes as ('other', None, 'its guards hold'); with N1
  the trusted split's retraces name the refused op ('dispatch', host_trace_test.split_op ...: "dispatches otherwise"
  x2, "allocates otherwise" x1), as untrusted ones do.
- Gates on n1_220 vs s220 (same box, same day):
  - replay: 250 run, 1 error / s220: 247 run, the same 1 error. The error is pre-existing and not N1's:
    TestReplayMemory.test_an_entry_reads_a_root_its_respecified_launch_did_not (FZ06's test) still reads `f.respecs`,
    which FZ08 removed (AttributeError). That's for int6: the FZ06 test needs FZ08's counters. The 3 extra tests are N1's
    2 trusted tests plus the hand-back test now split into [redispatch True|False].
  - triton: 63 run, 1 failure on both n1_220 and s220: test_a_record_its_launcher_disagrees_with_is_an_eager_call[drift_bit_21]
    (T_triton_soundness, a `_mm_tma` KernelLaunch recorded where an EagerCall is expected; on GH200, pre-existing).
  - inductor test_cudagraph_host_trace: 83 OK (82 + N1's test_a_launch_alignment_flip_redispatches).
- Odd line (step219 + .int6 patch, src219 on GB300 GPU 1): NOT run. batch3_once.sh got gpu1.lock twice (18:46 and
  19:15) and both times a foreign process was on GPU 1 (19:15-19:25: another user's launcher_multiapp, ~102 GB on all 4
  GPUs), so it skipped rather than run without permission. To run: `bash land/core/next/batch3_once.sh` (one hold:
  the probes before/after, the respec probe, then replay, triton and inductor on src219; logs/final_219/). It waits up
  to 10 min under the lock for GPU 1 to clear and has no override for foreign load. The odd patch's code edits are
  the same as the even one's.

## N1_trusted_redispatch.{int6cpp,int6}.patch (item 1, original, vs step187/188; SUPERSEDED by the rebase above) -- no switch
Trusted entries (Inductor's, any HostTraceReplay(trusted=TrustedInputs(...))) now redispatch / respec a failing
op-owned selector, as untrusted entries do, instead of tracing the call again.
- Why trust skipped it: no soundness reason found. The skip (`candidates if self.trusted is None else ()`,
  _host_trace_replay.py ~L685 / ~L667 odd) came in with redispatch itself (fork a9ddf715c96, int2, 2026-09-29), with
  no comment or log entry giving a reason; core READY 09c / 56 just list "trusted entries: redispatch/respec do not
  run" as not covered. It was a capability gap: a trusted trace is always sympy (nosym stage 1: "a trusted/Inductor
  trace is always sympy's"), and `_redo` refused any variant whose env is not an `_ir.Env` ("has no traced dispatch
  to run again"), so the skip only saved a refusal that was guaranteed. Trust vouches for layouts, never addresses
  (TrustedInputs docstring), and the redo runs the op on the call's own values in a fresh untrusted IR trace, so it
  records at least the guards a trace at the call would (more, not fewer: the caller's ranges decide nothing there).
- Change (non-test +8 -8 in 2 files; with tests +75 -14 per patch, net +61):
  - _host_trace_replay.py: the redispatch loop runs for trusted entries too. A redispatch-served call returns
    Handback under `handback` (as respec and a trace already do), so Inductor's boxed call still frees its arguments
    mid-tape instead of Python holding them to the end of that call.
  - _host_trace_redispatch.py `_redo`: the gate is now "the fresh env is IR" (`symbolic == "ir"`, the default), not
    "the variant's env is IR". The fresh env is always `_ir.Env()` (the `_TraceShapeEnv` fallback and its "dispatches
    again in a sympy trace" refusal were dead under that gate). A sympy (trusted) variant takes the existing ir=False
    path (renamed in sympy, as the HintsFail fallback already does); two IR-only reads are guarded
    (`_id_skips` via getattr; `ctx` only where ir). symbolic="sympy" still refuses, as before.
- Before / after (probe_trusted.py, bitwise vs eager in every row; logs/probe_before_*.log, probe_after_*.log, both
  lines identical):

  | case | before (traces, redispatches) | after |
  |---|---|---|
  | (a) trusted input view, offset k in 0,1,4,2,8,3,0,1 (8 calls) | 2, 0 | 1, 1 |
  | (b) trusted view of an intermediate at offset n*4 B, n in 1000,1001,1004,1003,1008,1002,1000,1001 | 3, 0 | 1, 2 |
  | (c) Inductor (torch.compile reduce-overhead, dynamic): custom op launching Triton on x[s0-1000:], 10 calls | 2, 0 | 1, 1 |
  | untrusted (a) / (b), reference | 1, 1 / 1, 2 | unchanged |

  (b)'s second redispatch is n=1008, where Triton's n % 16 specialization flips too.
- Tests (both lines):
  - test_cuda_host_trace_replay: test_a_trusted_misaligned_input_redispatches (traces, replays, redispatches, eager)
    = (1, 5, 1, 0); test_a_trusted_view_at_a_size_dependent_offset_redispatches = (1, 5, 2, 0); both atol=rtol=0.
  - test_a_trace_that_does_not_run_hands_the_call_back -> test_a_slow_call_that_does_not_run_hands_the_call_back
    [redispatch True|False]: it used a misaligned input to force a second trace under trust, which now redispatches.
    True: the redispatched call hands back, (traces, redispatches, eager) = (1, 1, 0), same peak 4n; False
    (`_redispatch` refused, the old path): (2, 0, 0) as before.
  - inductor/test_cudagraph_host_trace: TestInstallation.test_a_launch_alignment_flip_redispatches, case (c) above:
    (traces, redispatches, variants, eager) = (1, 1, 1, 1), bitwise.
  - test_a_trusted_split_respecifies: the split op (test_a_split_in_an_op_respecifies's) in a trusted entry, rows
    8, 40, 100, 12, 50, 128, 3, 64, 65: (traces, replays, respecs, variants, eager, respec_refusals) = (1, 8, 2, 3, 0, {}),
    bitwise. Respec under trust runs on the sympy path (no IR-internals checks, unlike the untrusted test).
- Respec / redispatch under trust (probe_trusted_respec.py, logs/respec_{before,after}_{188,187}.log; both lines
  identical, every row bitwise):

  | case (9 calls) | untrusted | trusted before | trusted after |
  |---|---|---|---|
  | split op (respec) | 1 trace, 2 respecs | 3 traces | 1 trace, 2 respecs |
  | two adjacent split ops | 1, 2 respecs | 3 | 1, 2 respecs |
  | tactic op (redispatch) | 1, 2 redispatches | 3 | 1, 2 redispatches |

- Runs, final version (batch2.sh, 2026-10-06 ~15:00-15:40 PDT, GPU 1 under gpu1.lock shared file by file with
  int6's recheck; no foreign process on GPU 1 at start, the earlier foreign job had left):
  - src188 (build_cpp7): replay 251 OK (skipped 1; step188's 247 + 4 new), triton 56, opaque 43 (skipped 1), guards 9,
    fullgraph 8, cute 66 (skipped 23: tests.sh does not set CUTE_DSL_EXAMPLES, the gate does), harvest 220,
    inductor test_cudagraph_host_trace 83 (82 + 1 new), aten test_a_trusted_trace_records_no_argument_pairs 1. All OK.
  - src187 (install_land_int5b_frozen): replay 189 OK (skipped 1; 185 + 4), triton 56 (skipped 6), opaque 37
    (skipped 1), guards 9, fullgraph 8, cute 62 (skipped 19), harvest 219, inductor 83, aten trusted 1. All OK.
  - replay re-run on both lines after test_a_trusted_split_respecifies was added: logs/final2_*/. Probes on the final
    trees (logs/probe_final_*.log) match the table above.
  - The first version (before the IR-env gate and the handback line) failed two tests on src188: the hand-back test
    (since rewritten) and triton test_a_bit_length_block_is_an_expression_of_the_size[symbolic=sympy] (sympy mode
    reached the redo; fixed by the gate).
  - Not run: aten in full (1669, 11 min), the other gate files (integration, capture, lower, lower_tape, cover, oob,
    packed, memory, python_entry), conformance.
- Behaviour notes for int6: Inductor entries whose op-owned guards fail (a user Triton kernel's or a custom op's
  launch specialization, size-hint choices inside an op) now redispatch first; a refusal falls back to the old
  fold/trace path, plus one failed redispatch attempt per class (refused selectors go first after that).
  Inductor's own copy_if_misaligned guard is graph-level and still traces (test_misaligned_input_is_a_variant
  unchanged).

## Item 2: SGLang prep C3 ("compound And/Or guards collapse to the hint", ISSUES R16) -- repro only, no patch
repro_c3.py: host code whose condition over a SymInt is compound, traced and replayed over new sizes on the same
side of the condition (100, 101, 200, 333, 1000, 4000) and across it.
- int6 (snap/step188 and snap/step187, logs/c3_*.log): does NOT pin. `(n > 4) & (n < 5000)` records one guard
  `(s > 4) & (s < 5000)`; `functools.reduce(or_, [(n >= a) & (n <= e) ...])` (_by_runs' old form) records one guard
  `(s <= 47) | ((s >= 49) & (s <= 63)) | ((s >= 70) & (s <= 4096))`; Python `and` records two relational guards. Each
  is 1 trace, 6 replays, 0 redispatches over the six sizes, bitwise. The IR keeps And/Or nodes as recorded
  (Ctx.junction, Env.guard_bool records the node or its `not`), and both lowerings evaluate them (IRLowering._bool
  "and"/"or", Lowering._chain/_any); nothing substitutes the hint.
- Every form, on int6 and on the C3-era tree (logs/c3_{188,187,sglm_abc}.log; sglm_abc = scratch/sglang/src_sglm_abc
  on install_sgl_cpp, the overlay s4 ran on; its IR junction / guard_bool / not code is identical to int6's). Same
  side: 100, 101, 200, 333, 1000, 4000, 100; flips alternate sides: 100, 48, 101, 64, 200, 69, 30, 5000, 50, 6000.

  | form | guards | same side | flips |
  |---|---|---|---|
  | `(n > 4) & (n < 5000)` | one And | 1 trace | - |
  | `n > 4 and n < 5000` | two relations | 1 | - |
  | `reduce(or_, [(n >= a) & (n <= e) ...])` (old _by_runs) | one Or(And) | 1 | 2 (one per side) |
  | symbolic_shapes.sym_or(sym_and(...)) | one Or(And) | 1 | 2 |
  | `any((n >= a) & (n <= e) ...)` | a prefix per path | 1 | 4 (one per short-circuit path) |
  | `(n - 8) or n`, then a size | Ne(n - 8, 0) + relations | 1 | - |
  | Or(And) inside a custom op | op-owned | 1 | 1 + 1 redispatch |
  | that op in 4 layers | shared op-owned | 1 | int6 1 + 1 redispatch; sglm_abc 2 (no owner_sets) |
  | Or of 74 runs incl. single points | one Or | 2 (both sides occur) | 2 |
  | m an int argument | one Or(And) | 1 | 2 |
  | trusted (sympy env) And / Or(And) | one guard | 1 | 2 |

  Every row bitwise, the same on all three trees. Traces count sides of the condition (or short-circuit paths),
  never sizes: no pin.
- Conclusion: C3 does not reproduce, not even on the tree it was reported on, so core's compound-guard handling did
  not pin m. What retraced the s4 extend entry 104 times was something else in that run (adapter or workload); its
  failing guards were never logged (the sweep json has no per-call guards) and the old `_by_runs` is gone (only the
  in-order form survives in adapter_prep_cpp.py). Nothing to fix in core; R16 can be closed as not reproducible,
  to be re-opened only with a logged failing guard from SGLang.

## T_triton_soundness.{int6cpp,int6}.patch (Triton soundness round: TMA bits, Triton launch witness, specialization audit) -- NOT handed to int6
Lane dir: core/next/triton (mkpatch.sh; probes in probe/; logs in logs/). Rebased 10-07 onto candidate 3:
.int6cpp md5 92f0ce3fd8b3, against snap/step212 (Python + C++; src212 = snap + patch); .int6 md5 fef565c9dedc, against
snap/step211 (Python only; src211). Both apply with fuzz 0 and no offsets on fresh copies of the snaps, and the
results equal src211/src212 file for file. Versus the step187/188 versions (kept as T_triton_soundness.int6.step187.patch /
.int6cpp.step188.patch, md5 da54079a3fb1 / 5879e5b2eb1e; the measured trees src187/src188/188t are those): offsets and
hunk context only, except _host_trace.py, whose hunk is re-made by content after FZ08's respec block removal
(fix_host_trace_py.py), and whose triton_tma_encode comment now says what False does under this patch (Triton
descriptor launches become eager calls; the old text described the removed CuTe-flavor encode). The C++ half is
unchanged. Checked against the newer top too (check_top.sh, patch -p1 -F0 --dry-run on the snaps): step215/216 and
step217/218 (97b(a), 98, 97b(c)) apply with fuzz 0, offsets only (_host_trace_cute.py hunks 2-3 at +15 / +56 lines,
test_cuda_host_trace_cute.py at +1 / +12); _host_trace_capture/_launch/_triton/_triton_launch/cudagraph_host_trace.py and
Patch.cpp are byte-identical between step212 and step218. Default: tma_library_bits=True (decided). Test runs on the
rebased trees (triton + cute on src211 / python_int6.sh and src212 / python_int6_build8.sh, the Python half on
build_cpp8's C++) queued on GPU 1 (core/next/triton/logs/g3.out); the gate results below are from the step187/188 trees. Non-test non-blank LOC: int6cpp +173 -58 (net +115), int6 +157 -55
(net +102); tests +114 -12.
The C++ half was verified on install_t = a copy of int6/build_cpp7/install whose libtorch_python.so is relinked from
src188's torch/csrc/cuda/host_trace/*.cpp with build_cpp7's own compile/link commands (relink/relink.sh; nothing under
int6 written; launcher python_t.sh, line "188t").

### Item 2: Triton launches get a trace-time witness (as CuTe has) -- default ON, no switch
- record_launch (user JIT and Inductor paths) now runs the kernel through its own launcher at the hints after building
  the record: JIT = CompiledKernel.run (Triton's CudaLauncher, exactly JITFunction.run's call, no hooks); Inductor =
  the tuned launcher itself (TritonCompileResult's runner or the static launcher). Addresses go in as ints (both
  launchers take an int as a tensor's address and only skip the cuPointerGetAttribute identity), TensorDescriptors are
  rebuilt over stand-in addresses, captured behind a torch.cuda._sleep kernel (so PDL shows as the edge) and never
  replayed. The node must match the record's kernel name, parameter layout, grid, block, shared bytes, explicit
  launch attributes, programmatic flag and every parameter byte (incl. the 128-byte CUtensorMaps). A mismatch goes
  through _trace_disagrees (warning + structured artifact + Declined; AssertionError under raise_trace_disagreements):
  a JIT launch becomes an EagerCall naming the field, an Inductor launch declines the trace.
- It found a real bug on its first run: _host_trace_launch._triton_clears_bit21 derived the rank as
  (len(values)-1)//2 although values has 2*rank entries, so for rank 2 it read only the first extent and cleared
  bit 21 of maps Triton keeps (e.g. a 512x136 fp16 output, max byte index 139262). Fixed (rank = len(values)//2).
  Impact before: the Python encode feeds capture images / CuTe compares; the C++ replay's rule (Patch.cpp) was right
  and patched the node at the first replay, so no wrong replay was observed, but the Python images were wrong. The
  old test_tritons_encode_is_its_launchers_bytes only compared a and b (never c), which is why it hid. With the
  witness, test_tritons_encode_is_its_launchers_bytes[broadcast|broadcast_small] and
  test_a_descriptor_whose_map_triton_alters failed on the unfixed rule (logs/triton_v1.log); new direct unit test
  test_tritons_bit_21_rule_reads_every_dimension.
- Unsound-case regression test: test_a_record_its_launcher_disagrees_with_is_an_eager_call[float_bits|bit_21] makes a
  re-derived launcher fact stale (_float_bits -> 0; the bit-21 rule -> never clear). Without the witness the replay
  would silently launch those bytes; with it the launch is an EagerCall naming "its launcher's parameter bytes" and
  the result is bitwise eager's.
- Trace-time cost (probe/witness_cost.py, GPU 1, 188t): 0.36-0.49 ms per traced launch. add (1 launch) trace 1.67 ->
  2.11 ms; add x8 3.44 -> 6.96 ms; TMA GEMM 2.30 -> 2.80 ms. Trace time only; replays unchanged.
- Consequence on the Python line (187, frozen int5b C++ without _host_trace_tma_flavors): Triton descriptors there
  were encoded with the CuTe DSL's flavor (byte-8 bit 0x02), bytes Triton never launches, so they now decline up
  front ("launch as Triton encodes them only with triton_tma_encode, in a native replay that takes it") instead of
  replaying. triton_tma_encode=False likewise makes every Triton descriptor launch an EagerCall (the old NaN-padding
  and bit-21-band declines of that mode are gone, subsumed). 187 loses Triton TMA tracing until its C++ is rebuilt.

### Item 3 fixes (default ON)
- OwnedModule loads the cubin with Triton's own driver.active.utils.load_binary (as CompiledKernel._init_handles),
  instead of cuModuleLoadData + copying one attribute: the tape's function now carries all of Triton's function setup
  (MAX_DYNAMIC_SHARED_SIZE opt-in and the cuFuncSetCacheConfig(PREFER_SHARED) it was missing; on a future driver with
  CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE=17, Triton takes that path and leaves MAX_DYNAMIC at default, which the old copy
  would have mirrored into a failing >48 KiB launch). Function attributes are invisible to a node capture, so this
  could not be witnessed; it is preserved by construction instead.
- The hand-copied _TMA_DTYPES / _TMA_TF32 table is gone: Triton's TMA_DTYPE_DEVICE_TO_HOST / TMA_TF32 are imported and
  applied as make_tensordesc_arg applies them.
- im2col detection: only meta.get("is_im2col", False), make_tensordesc_arg's own test (the "tensordesc<" /
  "input_rank" substring checks are gone).
- New test_reviewed_triton_launcher pins the reviewed launcher sources (driver.c, make_tensordesc_arg, CudaLauncher;
  3.8.0 hash 004e06f8...), as test_reviewed_triton_version pins the specialization: what the replay evaluates at
  sizes the witness never saw (the bit-21 rule, slot packing) must be re-reviewed on a Triton upgrade.

### Item 1: TMA descriptors, library bits vs the driver's map as is -- new switch, default True (decided)
- What the bits are:
  - Triton bit 21 of qword 1: not reverse-engineered. It is Triton's own code (driver.c fillTMADescriptorTiled, after
    cuTensorMapEncodeTiled: driver <= 13010 and C-int max byte index + 1 < 128 KiB -> desc_u64[1] &= ~(1<<21)), copied
    from CUTLASS commit b7ecaa60 (4.3.4, copy_traits_sm90_tma.hpp/im2col.hpp), CHANGELOG: "Work around a driver bug
    which will cause occasionally errors when executing kernels." This box runs driver 13000, inside the range. So the
    driver's raw map is the state NVIDIA's libraries work around; launching it where Triton clears the bit is *less*
    safe, not more.
  - CuTe byte 8 bit 0x02: the CuTe DSL encodes inline (no driver call); its map equals the driver's encode except this
    one constant bit (CUTE_SPIKE.md sweep, 8064 option combos: no documented option sets it). Meaning unknown; spike
    found replays correct with or without it.
- Switch torch.cuda._host_trace.tma_library_bits (default True = current behaviour, eager's bytes). False ("safe"):
  every descriptor launches cuTensorMapEncodeTiled's output with no bits edited (TmaDescriptor.edits=False, C++
  Descriptor.edits, d[7] of the native record, capability torch._C._host_trace_tma_raw). Triton: a descriptor where
  fillTMADescriptorTiled would clear bit 21 is an EagerCall (guard: index >= 128 KiB - 1 and < 2**31, Triton's own
  predicate; the old "small extents" escape, a guess of the driver's own rule, is dropped), so raw bytes == Triton's
  wherever it traces and the witness checks that with no masking. CuTe: the DSL's bit only enters the stand-in compare
  (descriptors compared with edits=True; the replay launches edits=False). Without the native capability (build_cpp7,
  frozen int5b) False declines TMA launches by name.
- Feasibility of encode-at-replay: it already is encode-at-replay (driver encode when sizes change, cuTensorMapReplace-
  Address when only the address moves, cache hit otherwise), in C++ for both lines. Raw driver cost
  (probe/encode_bench.c, GPU 1): cuTensorMapEncodeTiled 28.7 ns, cuTensorMapReplaceAddress 8.4 ns, 128-byte copy
  0.3 ns per call. Neither mode changes how often the driver is called (encodes/replaces identical, below), and the
  edit itself is a byte OR / AND. Calling Triton's own encoder at replay is not feasible: PyCUtensorMap exposes no
  accessor for its bytes (only Triton's launcher reads them), so it would need a struct-offset guess.
- A/B (probe/tma_cost.py, GPU 1, 188t, ABBA per case). Run 1 (logs/measure2.log) had another user's out-of-lock
  benchmark on GPU 1 in 6 of 8 runs; run 2 (logs/measure3.log, measure3.sh: a hold that starts with no foreign process)
  had none in 7 of 8 (one CuTe raw run saw 1). Default decided: tma_library_bits=True. Run 2 is the clean one:

  | case | mode | same | moving | alternating | kernel M=8192 | kernel M=16384 |
  |---|---|---|---|---|---|---|
  | Triton _mm_tma | library bits | 4.31, 4.02 | 4.77, 4.37 | 5.88, 4.90 | 22.82, 22.82 | 41.06, 41.06 |
  | | raw driver map | 4.06, 4.03 | 4.72, 4.53 | 5.51, 5.49 | 22.82, 22.82 | 41.06, 41.06 |
  | CuTe bw dense GEMM | library bits | 5.97, 6.58 | 7.00, 6.92 | 7.85, 7.85 | 16.48, 16.45 | 31.52, 31.49 |
  | | raw driver map | 5.09, 4.59 | 5.65, 5.38 | 6.30, 6.11 | 16.51, 16.45 | 31.43, 31.49 |

  CuTe raw came out 0.9-2.0 us/hit lower in both runs; the code path does not explain it (on a cache hit both modes
  do the same 128-byte copy plus at most one byte op; encodes/replaces are identical), so treat it as unexplained
  until profiled. Run 1 (noisy):
  host us per replay hit on small GEMMs (same inputs / new address each call / sizes alternating so a's and c's maps
  re-encode every call) and the GEMM kernel's own device time (profiler, 300 replays):

  | case | mode | same | moving | alternating | kernel M=8192 | kernel M=16384 |
  |---|---|---|---|---|---|---|
  | Triton _mm_tma (K=512, N=1088) | library bits | 4.15, 4.13 | 4.67, 4.68 | 5.66, 5.73 | 22.82, 22.85 | 41.10, 41.12 |
  | | raw driver map | 4.02, 4.50 | 4.70, 5.39 | 5.86, 6.12 | 22.82, 22.85 | 41.06, 41.12 |
  | CuTe bw dense GEMM (1,1) | library bits | 7.11, 6.35 | 7.76, 7.11 | 8.13, 8.25 | 16.48, 16.45 | 31.52, 31.49 |
  | | raw driver map | 6.59, 4.72 | 7.45, 5.32 | 8.20, 5.75 | 16.45, 16.48 | 31.49, 31.55 |

  Each run: encodes 20000, replaces 10001, 1 trace, outputs bitwise vs eager in both modes. Kernel time is identical
  to within 0.1 us (neither bit changes GPU time); Triton host cost is equal within noise, CuTe raw is ~1-2 us/hit
  lower in both runs for no reason visible in the code (above). So the trade-off is mainly not speed: "raw" costs Triton TMA coverage (on drivers <= 13010 every
  descriptor with max byte index < 128 KiB, i.e. small/decode-size maps, becomes an eager call) and launches CuTe maps
  without the DSL's bit; "library bits" launches eager's exact bytes, including Triton's driver-bug workaround, with
  the witness now checking our copy of Triton's rule at every trace.
- Tests: test_a_descriptor_whose_map_triton_alters[library_bits True|False] (False: the broadcast map is an EagerCall
  naming bit 21; (traces, replays, eager) = (1, 0, 3)), test_a_descriptor_triton_leaves_launches_the_drivers_map
  (raw, all maps >= 128 KiB: (1, 2, 0), edits False, bitwise), CuTe test_a_tma_gemm_launches_the_drivers_maps (raw:
  held native images have byte 8 bit 1 clear, (1, 2, 0), bitwise), test_a_descriptor_without_tritons_encode_is_an_eager_call.

### Item 3: Triton specialization, end to end (what Triton assumes vs what the trace guards)
Guarded (unchanged, verified by reading jit.py create_function_from_signature/run/_pack_args, BaseBackend
get_int/tensor_specialization, driver.c extractors/launch): pointer % 16 (symbolic get_tensor_specialization;
allocations/eager outputs by offset, base % 256 checked at replay in Python and C++), int % 16, int == 1 -> constexpr,
int width class i32/i64/u64 (only where unannotated; annotated widths get the slot-range guard), do_not_specialize /
do_not_specialize_on_alignment (binder flags = declarations, test), SymInt/SymBool constexprs pinned (Eq guard),
launch options and autotune keys pinned, heuristics run on SymInts, tuples of SymInts decline, SymFloat declines, u1/i8/
i16/u32/u64 slots decline, the closed attribute set (anything but tt.divisibility declines), grid limits and C-int
volume, device and stream. Every concrete-vs-symbolic specialization entry is compared at the hint.
Re-implemented launcher facts, now checked by the witness at every trace: slot packing (pointers, int widths, fp16
PyFloat_Pack2, bf16 truncation, fp32 C cast), scratch slots null, block = 32*num_warps, smem, PDL, descriptor bytes.
Found and fixed: the bit-21 rank bug, OwnedModule attributes, the copied dtype table, im2col substring detection.
Not guarded, reported only:
- JITFunction.run's used_global_vals check (eager raises RuntimeError if a global the kernel read changed) is not
  re-run at replay: a replay after such a change launches the old kernel. Same class as other Python state baked into a
  trace; checking it per call would be hot-path Python.
- knobs.runtime.debug / knobs.compilation.instrumentation_mode are read at the trace (part of the compile key); a later
  knob change is not seen by replays (global state).
- Function cache config / carveout is not part of a node capture: preserved by loading through load_binary, not checked.
- Under trusted inputs (Inductor) pointer divisibility is read at the hint in _InductorAutotuner._record. Vouched by
  Inductor's own handling: non-mutated inputs go through copy_if_misaligned (a guard under the trace), mutated inputs
  through align_inputs_from_check_idxs outside the installation, and static inputs are assumed aligned by Inductor
  itself (get_input_idxs_to_check), the same assumption without host tracing.
- The raw-mode bit-21 gate and the library-bits rule are our copy of Triton's predicate: the witness checks it at every
  trace and test_reviewed_triton_launcher pins it to the reviewed driver.c, but a replay at an unseen size relies on
  the copy (no per-size guard on the predicate in library-bits mode).

### Runs
- triton: 188t 61 OK; 187 61 OK (skipped 10: no flavors / no raw on the frozen C++); 188 (build_cpp7, Python patch on
  the old C++) 61 OK (skipped 2). cute (CUTE_DSL_EXAMPLES set): 188t 67 OK (skipped 1; 66 + the raw test), 188 67 OK
  (skipped 2), 187 62 OK (skipped 1; run before the raw test was copied to 187, where it skips: no capability).
  188t: replay 247, capture 12, opaque 43, guards 9, fullgraph 8, inductor/test_cudagraph_host_trace 82, all OK.
  187: replay 185, capture 12, opaque 37, guards 9, fullgraph 8, inductor/test_cudagraph_host_trace 82, all OK.
  Not run: harvest, aten, integration, cover, conformance.
- GPU: GPU 1 under gpu1.lock, except the 187 triton/cute/replay/capture/opaque/guards/fullgraph runs and two selected
  188t triton runs, which ran on GPU 0 under gpu0.lock before GPU 0 was reserved (the 187 inductor run then in flight was
  killed and redone on GPU 1).

## T_triton_soundness_gh200_test.patch (test-only follow-up to TT, on snap/step219 / step220) -- NOT handed to int6
core/next/triton/T_triton_soundness_gh200_test.patch, md5 ff77fb119b00, test/test_cuda_host_trace_triton.py only (identical
in step219 and step220, so one patch serves both lines; working copies triton/src219, src220). +17 -3.
- Rebased-TT runs that were pending above (logs/g3.out, GPU 1): src211 triton 63 OK (skipped 10), cute 64 OK (skipped
  2); src212 on build_cpp8 triton 63 OK (no skips: build_cpp8 carries TT's C++), cute 68 OK (skipped 1).
- GH200 failure of test_a_record_its_launcher_disagrees_with_is_an_eager_call[drift_bit_21] (logs/gh200/n1_*220): an
  environment difference, not a product bug. Triton's rule (fillTMADescriptorTiled) only clears bit 21; a stale copy of
  it is observable only where the driver's own map has the bit set. probe/bit21_driver.py (the test's broadcast map,
  extents (64, 1024), byte stride 0): GB300, driver 580.126.20 (version 13000), sets it (the drift test saw the mismatch
  there); GH200 gh200-b, driver 580.82.07 (also version 13000, inside Triton's range), leaves bit 21 clear for every
  swizzle and for 512..4096 rows, so Triton's clear is a no-op, Triton's bytes equal the stale rule's, the witness
  correctly finds nothing, and the launch is a KernelLaunch. The library-bits replay is correct either way (it equals
  Triton's bytes on both machines).
- Fix (test): the drift case first traces with the real rule, takes the a descriptor, and derives its expectation from
  the same two facts the bytes depend on: the driver's map as is (TmaDescriptor.encode(edits=False), bit 21) and the
  code's rule (_triton_clears_bit21). Drift is expected only if both hold: EagerCall naming the parameter bytes;
  otherwise the stale rule is unobservable and the launch must stay a KernelLaunch. The replay is bitwise eager's in
  both. The comment of test_a_descriptor_whose_map_triton_alters no longer claims the driver sets the bit (that test
  depends only on Triton's predicate, and passed on GH200).
- Runs: GH200 gh200-b (ht/tt/tt220 = a symlink tree of ht/int6c3/s220 with the fixed test; flock $HT_GPU_LOCK):
  test_cuda_host_trace_triton.py 63 OK (was 1 failure). GB300 GPU 1 (logs/gh200fix.out): src220 on build_cpp8 63 OK,
  src219 on python_int6.sh 63 OK (skipped 10). Probe on GB300: bit 21 set for the test's map at every swizzle, clear at
  512 rows (64 KiB of extents), set from 1024 rows (128 KiB) on; on GH200 clear at all of them, so there the drift case
  takes the KernelLaunch branch and on GB300 the EagerCall branch.

## TH_triton_no_hint_reads.{int6cpp,int6}.patch (candidate 4: the Triton hint reads, hint-audit OFFENDERS) -- NOT handed to int6
Lane dir core/next/triton_hint. Base = int6 snap/step242 (.int6cpp) / snap/step241 (.int6) + the hint audit
(core/hintaudit/hint_audit.*.patch; it applies with offsets except test_cuda_host_trace_oob.py's setUpModule, added by
hand; base/b242, base/b241). Working trees src242 / src241; mkpatch.sh. Files: _host_trace_triton_launch.py,
_host_trace_launch.py, _host_trace_capture.py, torch/_inductor/cudagraph_host_trace.py, _host_trace_hint_audit.py (the six
_TRITON OFFENDERS entries and the _stand_in transparency gone), test_cuda_host_trace_triton.py, test_cuda_host_trace_capture.py.
Non-test non-blank +243 -190 (net +53); tests +32 -17. md5 .int6cpp 02ce44dd2f17, .int6 f4c864096485. No `_hint`
call is left in the Triton files (nor in cudagraph_host_trace.py).
- `_witness` / `launcher`: the witness moved from the trace to the capture. record_launch stores KernelLaunch.witness, a
  closure that rebuilds each launcher argument from the record's slots at a call (pointers and ints from their slots,
  descriptors from their launcher and encode slots, floats and constexprs as the record's constants) and launches it
  through the kernel's own launcher; capture_tape runs it once per traced launch at the program's values at the traced
  call (lowered.evaluate(tape.args), the capture's real addresses) and compares kernel, layout, grid, block, smem,
  attributes, PDL and every parameter byte. A disagreement raises SegmentFailed for that launch (AssertionError under
  raise_trace_disagreements): the op runs eagerly, the rest replays. For Inductor the witness calls the runner the
  launcher calls (recorded from the launcher's own call: grid, stream by identity, kernel arguments by identity), so a
  launcher argument that is no kernel parameter (a template's _grid_0) is never needed at the call.
- `_intercept`: the compile no longer runs at hint values. The symbolic binder's specialization (each read a guard) is the
  key, and JITFunction.run's own lookup/compile is done on it (compute_cache_key, _pack_args with the key's constexprs,
  _do_compile); the concrete-vs-symbolic comparison is gone. A SymInt/SymBool/SymFloat of another trace (in an argument,
  a TensorDescriptor's shape/strides or a launch option) declines up front: "is a value of another trace".
- `_SymbolicSpecialization.__call__`: no hint for unaligned / unspecialized values. _unread(rule, value) runs Triton's own
  rule on two stand-ins whose single test answers True and False; one result with exactly one test means the value is no
  input (align off) and no guard is recorded, else the rule runs on the SymInt (each test a guard). Passing the SymInt
  outright where align is off would have added a % 16 guard Triton never makes (BaseBackend tests `arg % 16 == 0` before
  `align`), which test_integer_axes[_add_dnsa] forbids. A non-constexpr SymBool declines (u1 slots are not traced).
- `_resolve` / `record_launch`: `_pinned(v)` = int()/bool()/float() (the guard that pins and returns the value) instead of
  `v == _hint(v)`.
- Trusted inputs (no longer hint reads): record_launch / _descriptor skip the slot-range and grid-limit checks the caller
  vouches for (before: checked at the hint), while a descriptor's 16-byte address alignment is always a guard; in
  _InductorAutotuner._record a pointer's divisibility is a guard under trust too ("an address is never vouched for"),
  sizes are skipped (Inductor proved them).
- Regression tests: test_a_value_of_another_trace_declines (a size kept across traces by the host code). On the base it
  also declined, but only later, via "a guard on a symbol of another trace", after compiling at that trace's hint; no
  case where a Triton hint read produced a wrong replay was constructible (the guards already caught them).
  test_a_record_its_launcher_disagrees_with_is_an_eager_call now checks the capture-time witness (the call's declines and
  replays); the capture tests that hand-edit a Triton record's attributes drop its witness (a record Triton never launches).
- Runs (strict audit). GH200 gh200-b (ht/th/m242 = int6c3/s242 symlink tree + these files; sm_90 build8): triton 64,
  capture 12, replay 254 (skip 1), cute 68, inductor/test_cudagraph_host_trace 83, all OK; base b242 triton 63, capture 12
  OK. GB300 GPU 1 (src241 on python_int6.sh): triton 64 (skip 10), capture 12, inductor 83 OK; replay 192 with 2
  failures that the base b241 has too (test_a_zero_element_eager_output_keeps_eagers_strides,
  test_a_redo_takes_a_dispatch_whose_guards_hold).

### TH rebased on hintfix (step24x + audit + hf1-hf8) -- NOT handed to int6
core/next/triton_hint/TH_triton_no_hint_reads.hf.int6cpp.patch (md5 879741dac041, on hintfix/s242) and
.hf.int6.patch (md5 bd3e6f26977b, on hintfix/s241); mkpatch_hf.sh; trees hf242 / hf241 (bases hfbase/b24x = copies of
hintfix s24x). Only _host_trace_hint_audit.py needed a by-content re-apply: the six Triton entries come out of OFFENDERS
together with hf8's "owner: the Triton lane" comment and the Triton mention in its owners note. Everything else applies
with offsets. Non-test non-blank +244 -192 (net +52); tests +32 -17.
- Guards stay op-owned: every Triton specialization read and pin is recorded inside the launch's op (tr.op host=True,
  redo=...), so a flip redispatches that launch (test_a_specialization_flip_dispatches_again_into_one_variant and the
  custom-op / dispatch-unit tests pass). Values at the call: the witness. Decline: a value of another trace.
- minops (hintfix logs/new_e, decode 44,975 / prefill 50,529 hint reads): 0 come from the Triton sites. All belong to the
  symti lane's _elementwise and _pointwise_host sites. The Triton reads showed up only in the triton/capture/inductor suites,
  where strict mode now finds none.
- Runs (strict audit): GH200 gh200-b (ht/th/hm242 = int6c3/s242 symlink tree + the hf242 files, sm_90 build8): triton
  64, capture 12, replay 257 (skip 1), cute 68, inductor/test_cudagraph_host_trace 83, hint_audit 16, all OK. GB300 GPU 1
  (hf241 on python_int6.sh): triton 64 (skip 10), capture 12, inductor 83 OK; replay 195 with the 2 failures hintfix's odd
  top has too (logs/new_o/replay.out: test_a_zero_element_eager_output_keeps_eagers_strides,
  test_a_redo_takes_a_dispatch_whose_guards_hold); hint_audit 16 OK.
