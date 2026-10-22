# 27B mixed-step host cost (dev-box's breakdown, coordinator queue, low priority)

Source: land/scratch/dev-box/out_mix/mix27_ourssplit.analysis.txt. Per mixed step at C=64: patch 7.0 ms, evaluate 3.2 ms,
about 2,870 kernel node sets plus 290 memory node sets. C=16 adds miss 1.7 ms (native fills 7.3 per step). The split hides most of
it: about 1% of the step is exposed at C=64.

## What patch already does

The commit packs every user row and calls the setter only when the node's image or dims changed (Patch.cpp, the `h.held` compare).
The ~2,900 sets are therefore real changes: M-dependent grids and arguments across 27B's layers.

## Driver cost, no host_trace code (mixed/setcost.cu, mixed/setcost2.cu; GPU 1, CPUs 72-107, 2026-10-18)

| call | cost |
|---|---|
| cuGraphExecKernelNodeSetParams, 64 B params, no change / arg change / grid change | 0.14 / 0.17 / 0.42 us per node |
| same, grid and arg change, by param-block size: 64 B / 512 B / 2 KB / 4 KB | 0.48 / 0.58 / 1.29 / 2.55 us per node |
| cuFuncGetAttribute (raise_dynamic_smem_limit, per set with dynamic smem) | 0.026 us |
| cudaGraphExecMemcpyNodeSetParams1D / cudaGraphExecMemsetNodeSetParams | 0.45 / 0.31 us |
| cudaGraphLaunch after 3,000 sets, 64 B params: grid changes / arg changes / no change | 356 / 114 / 80 us (7 us without sets) |

Reading: the setter's cost scales with the parameter block, and the driver defers work to the next launch (item 3). With 64-512 B
blocks, 2,900 sets would cost about 1.4-1.7 ms; the measured 7 ms implies large blocks (CuTe TMA descriptors), our own per-node work
(pack of every row, compare, image copy), or both.

## Next, once the n27pr ms29 rerun is in

1. Attribute patch and evaluate on a real 27B mixed run (GPU 1, ms30, sampling perf only, not CUPTI, process not used for timings):
   libcuda share of patch versus pack/compare/copy, and evaluate's hot spots. If perf can't resolve it, add counters (bytes per set).
2. Changes only as clean, verified ones:
   - ours: pack/compare/evaluate work, verified against the current path;
   - driver share: fewer sets (only real changes are set today). The remaining lever, device-updatable kernel nodes (partial
     param updates applied on device), is a new primitive and needs the user.

## 27B attribution (2026-10-18, GPU 1, step_ms30 / build_cpp_ms12e; mixed/perf27.sh, sampling only)

perf, 20 s of the C=64 window from its start, DWARF call graphs on the main thread (499 Hz; 3,836 samples, about 19% of the
thread's wall on-CPU; the rest waits on the GPU). Python interpreter frames are 58% of on-CPU samples, native host_trace about 16%.
Within host_trace (mixed/perf_split.py, innermost host_trace frame):

| phase | share of main-thread samples | driver part | ours |
|---|---|---|---|
| evaluate_static | 5.6% | none | delta_rows 27%, HostTraceProgram::reading 22%, malloc 15%, evaluate 17% |
| patch (node-set lambda) | 4.4% | 61% (cuGraphExecKernelNodeSetParams 67% of the phase) | 20% our code + 13% memcpy/memcmp (image copy and compare) |
| commit | 1.7% | none | malloc 27%, run_buffer 16%, mark_dirty 16% |
| launch (patch_and_replay) | 1.1% | cuGraphLaunch 93% (item 3) | |
| split_ready | 1.1% | instantiate / clone / destroy node | first-time section builds of forms first used in the window |
| pack | 1.1% | | encode 31%, pack 24%, field_value 21% |

Reading: the patch phase is two thirds driver, one third ours; evaluate is all ours.

## Tried and not landed: per-leaf bitsets for delta_rows (build_cpp_ms12f + snap/wip35)

delta_rows' misses (gdb: 38 of 466 calls in 20 s) each ran a full reading() pass over the dynamic rows. The change cached, per
dynamic leaf, the rows reading it (the union over changed leaves is exact: reading is an OR over operands), OR-ed them per miss, with
reading() as an oracle (delta_rows_check, on in the suites). It was correct (suites green with the oracle on, new counter test), but
the window showed no gain: evaluate_static 5.6% -> 5.4% and delta_rows + reading + malloc 3.5% -> 3.5% of main-thread samples. The
per-leaf passes still ran in the window (74 samples; gdb saw none in its slower 20 s, so they come later in the window). Not
landed: complexity without a measured benefit. Patch kept at patches/delta_row_bits.csrc.patch.

## Harness note (not ours to change)

drive_sweep27's _ctx_probe captures its graph on the first call and times that first replay, so the first window's "before" value
reads slow (1.43-1.69 ms) in every run of ours; every later probe reads fast (1.06-1.09). One untimed replay after capture fixes it.
