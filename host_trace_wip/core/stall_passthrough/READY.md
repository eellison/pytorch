# stall_passthrough lane: READY (not merged)

Base: candidate 5 = int6 snap/step326 + int6/build_cpp13/install. Lane tree: core/stall_passthrough/wt (a copy of snap/step326, git
base commit = the snapshot, `git diff` = the patch). Launcher: core/stall_passthrough/py.sh. Python only, no C++ change.

## Task 2: CuTe passthrough hot-path overhead (READY)

Patch: `PT_loaded_function.step326.patch` (torch/cuda/_host_trace_cute.py, test/test_cuda_host_trace_cute.py; +40 / -11).

Root cause: with `_host_trace_cute.install()` active, a TVM-FFI function looked up in a module loaded from an exported object
(`cute.runtime.load_module(...).func`, i.e. quack's and FlashInfer's jit caches) came back wrapped in `_Loaded`, whose `__call__` was
`_traced_call`'s Python wrapper on every call, tracing or not: `current_trace()`, two genexprs, `any`, 9 `isinstance` and a lambda
before the real call (19 profiled calls instead of 2; passthrough/count_0.txt vs count_1.txt). A function compiled in-process
(`cute.compile`) was never wrapped outside a trace: its call graph is identical with and without install (5 calls both).

Fix: the looked-up function is now a `tvm_ffi.core.Function` subclass (`_LoadedFunction`, the looked-up handle moved into it, as
cutlass's own TVMFFIJitCompiledFunction does) with no `__call__` of its own, so a non-tracing call is TVM-FFI's C call with zero added
work. `_hook` (entered only while a trace runs, ProcessHold) patches its `__call__` with `_traced_call` like the compiled classes', and
`_unhook` deletes it again. Side effect: a loaded function is now an `isinstance(..., tvm_ffi.Function)`, as without host tracing.
Chose this over a flag check in the wrapper because a flag check still pays a Python frame per call (the user's rule is no hot-path
overhead), and over keying the program by the raw Function (no weakref support on tvm_ffi.core.Function, would need strong refs).

Test: `test_a_loaded_function_outside_a_trace_is_tvm_ffis_own_call` (loaded function is a Function with Function's own `__call__`
outside a trace, runs correctly, is still a KernelLaunch under trace(), and the call is restored after). Fails on step326 (`_Loaded is
not an instance of Function`), passes with the patch.

Microbench (passthrough/probe_passthrough.py, passthrough/rounds.sh 5: one process per arm per round, arms rotated, GB300 GPU 3 idle,
CPUs 130-131; machine load 75-95 from other users, so take the per-process spread as the noise floor). us per call, median of the 5
per-process medians [range]:

| call (no trace active) | not installed | installed, step326 | installed, patched |
|---|---|---|---|
| quack RMSNorm kernel alone (loaded from jit cache) | 4.87 [2.63-5.47] | 7.56 [6.58-8.12] | 3.79 [2.89-5.95] |
| quack rmsnorm_fwd (eager, incl. alloc) | 13.93 [13.19-14.75] | 17.09 [16.55-17.64] | 13.57 [12.62-14.94] |
| CuTe add, TVM-FFI compiled in-process | 4.83 [2.94-5.93] | 5.62 [4.35-6.01] | 4.76 [3.23-6.15] |

The loaded-function overhead (+2.7 us kernel alone, +3.2 us rmsnorm_fwd) is gone. The compiled-in-process row runs identical code in
the two installed arms, so its 0.8 us spread between them is noise; this is also what c5pair_delta.md's +1.2 us "TVM-FFI / launch
alone" CuTe add row measured (same code path as not installed, by profile). Raw: passthrough/rounds.jsonl (+ rounds_v1.jsonl, 3 arms).

Gate (GPU 3, one suite per gpu3.lock hold, int6/logs/stall_passthrough/, gate.out):
- cute: PASS 75 tests (74 + the new one), OK (skipped=1)
- replay: PASS 295, OK (skipped=1)
- harvest: PASS 230, OK
- python_entry: PASS 571, OK (skipped=70)
- aten: PASS 1731, OK

## Task 1: learn/trace stall over many sizes (DROPPED by the coordinator; no core problem found)

Before the coordinator retracted it (the r27 lane had misread a py-spy local), I checked it two ways:
- Synthetic walk (repro_sizes.py: HostTraceReplay, static prefix of weights, cuBLAS GEMMs at M = T, one variant per 64-token chunk
  count, memory=run_buffer, HarvestProvider) over T = 1..3072 in one process, at hidden 256 and 4096, and with the device filled to
  1.5 GB free: no stall, no notes or failed builds; 96 traces, each at a new chunk count, trace time grows with T only (0.05 s at T=64,
  0.24 s at T=3000). logs/r1-r3.
- The real run (real/run_learn3072.sh: the round_27cb_line.sh learn arm, c5h, --learn-range 1 3072, under diag_wrap.py which times every
  trace, prints every note and failed build, and dumps all stacks while one trace runs > 120 s), GB300 GPU 3: passed 1952 and 2000 with
  no stall: 77 traces at 15-36 s (median 20 s), 0 notes, 0 failed builds, 0 declines, 74 variants, ~4000 harvests; memory flat
  (reserved 73.7 -> 73.85 GB, device 77.5 -> 78.5 GB over 2000 sizes). Stopped at ~2050 sizes once the task was dropped. The run is just
  slow: ~1 s per size, dominated by the ~20 s traces (one per ~27 sizes).
- Minor observation, not acted on: with memory=run_buffer, walking ever-larger sizes grows the caching allocator's reserved bytes
  (synthetic hidden 4096: 0.4 -> 4.0 GB reserved over 3000 sizes at ~0.15 GB allocated): one cached block per distinct run-buffer
  size, the fragmentation _host_trace_memory's docstring already names. Small on 27B (+0.15 GB over 2000 sizes).
