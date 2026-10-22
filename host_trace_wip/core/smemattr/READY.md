# SA: a node set above its function's lowered dynamic-smem limit (GB300 whisper-tiny dynamic crash)

STATUS: READY (gates below). Patches (md5 9233e175b984 csrc, f4e26cf55b25 py):
- `SA_smem_limit.csrc.build_cpp12.patch`: C++, against int6/build_cpp12/src. It touches c10/cuda/driver_api.h,
  host_trace/Variant.h, Patch.cpp and Harvest.cpp. Incremental rebuild: 28 edges, about 3 min.
- `SA_smem_limit.py.step324.patch`: Python plus the test, against snap/step324. It touches _host_trace_capture.py and
  test_cuda_host_trace_replay.py.

## The crash

Whisper-tiny dynamic with host_trace, GB300 only, on c4 and on candidate 5 (step324 + build_cpp12). The first steady
call at batch 2 fails with:
- `cuGraphExecKernelNodeSetParams_v2` returns CUDA_ERROR_INVALID_VALUE;
- driver log: "Requested dynamic shared memory size of 6240 bytes exceeds maximum dynamic shared memory size of 49152
  bytes".

The 49152 in the message is misleading: it is not the limit that applied.

A driver hook (`set_hook.so`) swaps c10's DriverAPI slot and logs every failing set, with the function's attributes.

- **Node:** `cudnn::fusion::lean_reduction_kernel`, cuDNN SDPA's split-KV reduction on sm100.
- **What changed between shapes:** same function. Grid (1,6,5) -> (1,6,2), dynamic smem 2600 -> 6240.
- **The function's limit:** `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` = 2600.
- **What eager does:** cuDNN and cuBLASLt set that attribute to their launch's bytes before each launch. Every
  cuDNN/nvjet function in the log has limit == its last launch's smem, e.g. lean_reduction at 2600 after its batch-5
  launch. So the limit is library-owned global state, and it gets lowered too.
- **Why the set fails:** the replay re-set the node to batch 2's key (6240) while the limit held batch 5's 2600.
- **Why GB300 only:** on sm90 the cuDNN attention path has no such kernel whose smem varies with batch. nvjet kernels
  keep fixed smem per function.

Driver behavior observed:
- A set that changes `sharedMemBytes` above the limit is refused.
- A set that keeps `sharedMemBytes` (grid-only change) was not refused.
- A held node replays fine after the limit is lowered, so the launch path does not re-check.

## Fix

`raise_dynamic_smem_limit(func, bytes)` (Patch.cpp, declared in Variant.h): if `bytes > limit`, set the limit to
`bytes`, as the library's own launch at that key does.

Called before:
1. every kernel node set (Patch.cpp `patch`);
2. the harvest's region re-launch (Harvest.cpp `cuLaunchKernelEx`);
3. the Python capture's `_launch_kernel` (_host_trace_capture.py, via cuda.bindings).

`cuFuncGetAttribute` / `cuFuncSetAttribute` join the DriverAPI REPLAY list and `harvest_driver()`.

Design notes:
- The limit is never lowered.
- No per-call state is cached: the library can lower the limit at any time between replays.
- Cost: one `cuFuncGetAttribute`, 24.7 ns on GB300 (`attr_cost.c`), per kernel node set with smem > 0. Unchanged nodes
  are not set, so they pay nothing.

Alternatives not taken:
- Retry after an INVALID_VALUE: the error goes through the driver-log callback, and the retry would hide real errors.
- Re-instantiate on a function change: not the cause; the function is the same.
- Decline: eager runs this fine.

## Test

`TestNativeReplay.test_a_node_set_above_its_functions_lowered_smem_limit` (test_cuda_host_trace_replay.py):
1. `x.sum(-1)` traced at (64, 4096).
2. The reduce kernel's limit is set to 1024, standing in for a library's lower setting.
3. A call at (8, 4096) needs 2048 bytes of smem.

Results:
- Before: the same driver error, "Requested dynamic shared memory size of 2048 bytes ...".
- After: bitwise result, and the limit is 2048. The limit is restored in `finally`.

## Gates (GB300 GPU 1; b1 = isolated copy of build_cpp12 + the csrc patch, py324 = step324 + the py patch)

- New test: fails on build_cpp12 + step324 with the test only; passes with the patch.
- With the patch: replay 295 (1 skipped; the new test is the +1), python_entry 571 (70 skipped), pinned 38, opaque 45,
  harvest 230, tape 46, capture 12, memory 21, triton 66, guards 12, inductor/test_cudagraph_host_trace 85 (2 skipped),
  all OK. Same counts as the step324 / 320b12 gates.
- Not run: aten, cute, conformance, fuzz (merge lane).
- Whisper-tiny dynamic (dyn_prof), GB300:
  - step324 + build_cpp12: crashes at the first steady call;
  - with the patch: 552 replays, no error, 7 segments, 0 boundaries.
  - Checksums vs the default arm differ on all 4 shapes. This says nothing about the patch: the c4 suite has whisper
    bitwise 0/4 for trees as well as host trace. The core test is the bitwise check.

Tools here: `set_hook.c` (driver hook; SET_HOOK_ALL logs function/smem changes, SET_HOOK_FIX applies the fix in the
hook), `run_whisper.sh`, `dyn_hook.py`, `gate.sh`, `probe_*.py`, `setup_b1.sh`.
