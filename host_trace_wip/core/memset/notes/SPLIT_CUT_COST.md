# What a split cut costs on the GPU (split-cost lane, 2026-10-17/18)

Question: 27B NVFP4 decode bs 1 (step_ms16, build_cpp_ms11) measured 7.98 ms launch-to-completion unsplit vs 9.05 ms for 4
sections (spans 0.114 / 0.429 / 1.659 / 6.846 ms). That is +1.05 ms over 3 cuts, ~0.35 ms per cut, against a target of ~2 us.

## Answer

**The cuts are not what costs the ~1 ms.** A cut costs about 1-5 us of device time on the real 27B graph:
- 3.4 us with events between the launches;
- ~1 us with events only around the call;
- 2 us per cut on GPU 3 (relaunch: sections 7.977 vs whole 7.971-7.975 ms).

**The +1.07 ms is a per-process measurement-window effect, and the split has nothing to do with it.**
- In every process measured, the **second** measurement window is ~1.09 ms slower. That is with split on or off, for the
  whole exec and for the sections, in-call and when relaunched.
- **An off-only process shows it too** (sc16: 6 off windows, 8.04 / **9.13** / 8.04 / 8.04 / 8.04 / 8.04 ms in-call; the
  relaunched whole exec 8.03 / **9.12** / 8.03 / 8.03 / 8.03 / 8.03).
- The old A/B ran off then on in one process, so "on" always landed in window 2.
- **Any A/B that measures each arm in its own process and lands in a process's second window carries the same +1.1 ms in
  that arm.**

Signature: a fixed ~0.4 us per graph node, not slower kernels.
- 47 rebuilt chunks of ~60 nodes each were all +20-27 us in window 2, whatever the chunk's length: +5% on a 0.50 ms chunk,
  +19% on a 0.14 ms one.
- That is the same per-node shape as the CUPTI inflation seen 10-16 (0.5-0.7 us per node). No profiler is enabled in the
  harness (DRIVE_SPLIT_PASS=walls; the drive's profiler paths are env-gated off). Something in the process or on the device
  adds per-kernel launch cost for about one window.

Ruled out:
- the prompt data (seed 5 and seed 6 both slow in window 2);
- clocks and power: 2070 / 3996 MHz, P0, no clock-event reasons active, and the throttle counters (SW power cap, SW / HW
  thermal, power brake) do not move across windows;
- background GPU work in our process: GPU utilization is 0% at idle after a sync;
- other Python threads using CPU: only MainThread and tqdm_monitor.

**Trigger (next section):** the CUDA context entering a state where every graph kernel costs ~0.2-0.4 us more. A pure C++
process enters it when an 8th extra stream is given work. In our process the first drain/prefill cycle enters it and the
next cycle leaves it. vLLM's arm stays in the fast state.

## Evidence (CUDA events only, no CUPTI; GPU 2, CPUs 36-71)

Harness:
- launchshim3x (LD_PRELOAD) keeps launchshim3's per-launch events.
- It also records each launch's exec and stream, each exec's source graph (instantiate intercepted), its uploads, its
  memset sets, and its kernel sets (c10 DriverAPI's entry point intercepted).
- `ht4_bench` relaunches recorded execs back to back. It runs queued behind a 3 ms spin, or live; with events between the
  launches or none.

### Synthetic chain, standalone

`cutbench.cu`: 2,823 spin kernels of ~2.9 us, cut like split_ready (clone + destroy + instantiate AutoFree|NodePriority +
upload) at bounds 0/34/170/711/2823.

| variant | per cut |
|---|---|
| plain | 4.8 us |
| 2 / 10 node sets per section before launch | 4.8 / 5.4 us |
| re-upload after sets | 6.1 us |
| PDL every 4th kernel (705 programmatic edges) | 5.5 us |
| cluster kernels every 7th | 4.8 us |
| memset nodes every 9th | 4.9 us |
| PDL + cluster + memset + sets | 6.8 us |
| live (not queued) | 3-4 us |
| our split code on a synthetic HostTraceReplay (2,800 pointwise ops), events between / none | 4.8 / 2.7 us |

Nothing scales with section node count.

### Real 27B decode bs 1: relaunching the recorded execs

Times are queued, with no events between launches unless noted.

| window | whole exec (2,823 nodes) | 4 production sections (34/136/541/2,112) |
|---|---|---|
| off1 / on1 (runs sc2..sc13, 6 runs) | 7.885-7.891 ms | 8.960-8.966 ms |
| off2 / on2 (sc13: modes off,on,off,on) | 7.887 ms | 7.901 ms (events between), 7.894 ms (live, no events) |

In the same windows:
- In-call event spans: on1 0.103 / 0.427 / 1.646 / 6.797 ms (9.09 ms call). on2 0.098 / 0.374 / 1.441 / 6.004 ms, 7.917 ms
  of spans against 7.905 ms whole.
- The on2 split call is faster than the unsplit one (8.064 vs 8.107 ms), because the patches are pipelined.
- On1: the sections ran 12-14% slower per node in every section, not only next to the cuts.

### What was ruled out (on1 windows)

- Upload: an explicit cudaGraphUpload before each launch, or a pre-upload on a side stream, changes nothing (8.970 / 8.964).
- Events between the launches (hypothesis b): 8.962 without them.
- Host interplay (hypothesis a): relaunching with no patching and no host work still gives 8.962.
- Structure: the sections partition the whole graph's nodes exactly.
  - 2,551 kernels, 272 memsets, 113 cluster kernels in total.
  - The edges are the whole graph's minus the 3 cut edges; all 353 programmatic edges are kept.
  - One root per section; flags 9 on the whole exec and on the sections.
- Production history:
  - The sections are uploaded (stream 0, rc 0, not capturing) before their first set.
  - No kernel set changes grid, block, smem or function against the template or against the previous set.
  - No memset set changes element size, width, height, pitch or value.

### Re-splitting the whole graph ourselves (bench5/6/7, off windows)

Method:
- Clone the template, destroy the nodes outside the section, instantiate with flags 9, upload.
- Then set every node from `_host_trace_held_images`, so the addresses are valid.

| build | time |
|---|---|
| whole exec | 7.889-7.893 ms |
| K=1 clone / fresh instantiate of the template / flags 0, 1, 8 | 7.887-7.893 ms |
| K=2 at 34 / 170 / 711 / 1411 / 2111 / 2789 | +0.0 / -1.8 / -2.0 / +0.3 / -1.8 / +4.3 us |
| K=4 at 34/170/711 (production bounds) | +10.2 us (3.4 us per cut, events between) |
| K=6 | +17.7 us (3.5 us per cut) |
| core-style sets (func instead of kern, row block/smem) | K=4 7.905 ms |
| sets while the upload is pending, or while the GPU runs the whole exec, or pipelined first call, or upload on stream 0 | 7.896-7.901 ms |
| 20 pipelined calls of 2 sets per section afterwards | 7.897 ms |
| **sets before the exec's first upload** | **8.887-8.891 ms (+1.0 ms, permanent)** |

The last row is the one way found to reproduce +1 ms on purpose. It is the exec-update successor penalty (memory:
paramgraph-exec-update-successor-penalty): an update before the first upload marks every node.
- split_ready already uploads before any set, and the shim confirms production follows that order. So this is a pitfall to
  keep avoiding, not today's cause.
- The same production execs being fast in on2 rules out a penalty that lasts for the exec's life.
- Rebuilt in on1 from the split state's held images (bench8): 8.960 ms, as slow as production. So on1 slows any exec built
  or run then.

### Window 2, not split (GPU 3, CPUs 108-143, 10-18)

In-call times are event spans; "relaunch" times come at the end of each window. The whole exec is re-set from that window's
held images before it is relaunched.

| run (modes) | window 1 | window 2 | window 3+ |
|---|---|---|---|
| sc15 (off,on,off,on): relaunch whole / sections | 8.028 / - | **9.118 / 9.122** | 8.028 / -, then 8.032 / 8.041 |
| sc15 in-call | 8.044 | **9.134** (split) | 8.047, then 7.99 (split) |
| sc16 (off x6): in-call / relaunch whole | 8.040 / 8.028 | **9.133 / 9.116** | 8.04 / 8.03 in all four |
| sc17 (off x3, seed 6): relaunch whole | 8.030 | **9.114** | 8.030 |
| sc18 (off x3): relaunch whole / 47 rebuilt chunks | 8.032 / 8.254 | **9.114 / 9.329** | 8.032 / 8.252 |
| sc14 (on,off,on): in-call | **9.16** (split) | **9.14** (whole) | 7.99 (split) |

- In sc14 the relaunches at the end of each window were 7.97 in all three windows. There the slow period covered the
  in-call steps of windows 1-2 and had ended by the time of the relaunch. So it follows the process's timeline (roughly the
  second minute of measurement), not the arm.

## Window-2 trigger: the context's stream count (10-18, GPU 2/3)

**Mechanism, reproduced in a pure C++ CUDA process** (`cutbench streams=N`: 2,823 x 1 us kernels in one graph exec):
- The step goes from 5.39-5.40 ms to 5.955-5.960 ms (+0.2 us per graph node) at the 8th extra stream that is given work.
  Up to 7 extra streams (8 counting the launch stream) change nothing.
- It is context-wide: the graph is slower on its own stream, on the legacy stream, and on any of the extra streams.
- It is sticky in C++: idling 30 s, reusing the streams, destroying them, or instantiating a fresh exec of the same graph do
  not undo it.
- Creating streams without work does not trigger it; the first kernel on each does.
- It is the same with blocking or non-blocking streams and with CUDA_DEVICE_MAX_CONNECTIONS=1 / 32 (32 lowers the baseline to
  5.15 ms, but the +0.56 ms step at 8 extra streams stays).
- Interleaving up to 64 other graph execs, or instantiating and launching a new 3,628-node exec, does nothing (`interleave=`,
  `newexec=`).

**In the 27B process:**
- Giving work to 16 extra streams before the first window makes every window slow: 8.973 / 8.971 / 8.968 ms, against 7.814 /
  8.855 / 7.814 ms without (sc35 / sc36, GPU 2).
- A fixed 1,000-kernel probe graph replayed after every engine step (on the current stream; it gives work to no extra stream)
  reads 0.954 ms.
  - It jumps to 1.333 ms (+0.38 us per kernel) at the first drain/prefill cycle (t ~ 64-70 s).
  - It returns to 0.955 ms at the next cycle (t ~ 130 s) (sc39).
  - So the slow window is exactly the span between those two cycles.
- Cycle 1 is where the prefill variant's 3,628-node graph is first traced and instantiated, the TRT-LLM attention decode site
  is redone (`prov_redo`), 12 new cudaMalloc segments (114 MiB) appear, and 64 TRT-LLM kernels get
  NON_PORTABLE_CLUSTER_SIZE_ALLOWED.
  - Torch's 128 pool streams are all created before window 1 (stream creates and destroys are intercepted; none happen
    later).
  - Which stream or channel action in cycle 1 sets the state, and what in cycle 2 clears it, is not pinned yet. Triton's and
    CuTe's driver-level calls are not intercepted. The C++ repro does not show the recovery.

**vLLM arm (fullnc):**
- The probe reads 0.930 ms all run (the fast state), and there is no window effect (8.81 ms flat on GPU 3; 8.60 ms on GPU 2,
  the same with 16 forced streams).
- So comparisons against vLLM are fair when ours is measured in a fast window.
- Ours forced into the slow state is 8.97 ms, slower than vLLM's 8.60 ms. Any serving run of ours that ends up past 8 working
  streams loses the decode advantage, so it is worth checking the probe in serving runs.

**Ruled out:**
- dynolog / DCGM: the C++ process is flat for 288 s, and nv-hostengine and dynolog CPU are flat;
- in-process CUPTI: 0 libcupti calls in all windows; no KINETO/DYNO env;
- clocks / power: counters flat;
- NUMA balancing: off;
- persisting L2: the limit is 25.4 MB in every window (a side finding: something in the process sets it), and
  cudaCtxResetPersistingL2Cache does not help;
- KV block / state slot placement: window 4 uses window 2's blocks and is fast;
- prompt data;
- idle time: 60 s of idling in window 2 stays at 9.01 ms;
- exec content: kernel configs are identical across windows, and held images differ only in 6-9 sampler scalars.

### Stream census of the 27B process (ms27, 10-18)

Method:
- The shim notes the stream of every runtime and driver work call: kernel launches, copies, memsets, event records and waits,
  graph launches, uploads, host funcs.
- That covers LD_PRELOAD, c10's entry-point table, dlsym from libcuda (Triton / CuTe launchers) and cuGetProcAddress.
- It records a Python stack at each stream's first use.

Findings:
- **Only 5 streams ever get work over the whole process:**
  - the legacy stream 0x0: every replay and eager op;
  - vLLM AsyncOutput's copy stream: waits only;
  - one warm-up async-copy stream;
  - two capture-only streams: `vllm_ops_trace.capture_kernel_nodes` at t=1.2 s, and `harvest._restore` ->
    `capture_kernel_nodes` at 6.9 s. Their last uses are at 63-65 s, in cycle 1.
- No stream gets its first work in cycle 1, and no stream is still in capture during the slow window.
- **So the API-level stream count is not what our process hits.** The >8-streams C++ repro reaches the same per-kernel cost
  by another path.
- **Also not it, in C++:**
  - up to 12 execs of fork/join graphs (2-3 branches), each launched once;
  - 64 interleaved graph execs;
  - a new 3,628-node exec, with or without node sets before its first launch.
- **The state follows the drain/prefill cycles, not time:**
  - probe 1.037 -> 1.415 ms from the first cycle (t ~ 60 s) -> 1.038 ms from the second (t ~ 120 s);
  - with one extra cycle before window 1 (`SC_PRECYCLES=1`), window 1 is slow and windows 2-3 are fast (9.02 / 7.96 /
    7.96 ms).
- **What happens in cycle 1:**
  - The only new instantiation is the prefill variant's 3,628-node exec. No exec is destroyed in any window.
  - The two capture streams run their captures (208 + 3,822 uses).
  - The TRT-LLM decode site is redone.
- **Ruled out as the clearing step:** an explicit cudaGraphUpload of the prefill exec in the slow window does not clear it.
  Relaunching that exec with its stale arguments faults (illegal address), so its second launch could not be tested in
  isolation.

**Harness mitigation:** run two drain/prefill cycles before the first measured window, and keep the probe check per window.

### Trigger confirmed: an exec uploaded but not yet launched (10-18)

- **C++** (cutbench `newexec` / `uploadtest` / `lazytest`; NVIDIA_BUNDLE `graph_state_repro`):
  - Once another exec has been `cudaGraphUpload`'ed and not yet launched, every graph kernel is ~0.2 us slower:
    5.36-5.40 -> 5.94-5.96 ms for 2,823 x 1 us kernels.
  - Launching that exec once (or any one of several pending ones) clears it.
  - Upload followed immediately by launch, an implicit upload at the first launch, and instantiate without upload all have
    no effect.
  - Destroying the uploaded exec without launching it leaves the state on.
- **In-process, 27B:** the decode exec goes 7.948 -> 9.009 ms after uploading a fresh 200-kernel graph, and back to
  7.946 ms after one replay of it.
- **Our code does this at every capture.**
  - `_host_trace_capture.py:895` calls `cudaGraphUpload` right after `instantiate()`, so that later node sets are fast;
    `instantiate_form` uploads at creation too (`:1213`).
  - The prefill segment captured in the first post-warm-up prefill cycle stays uploaded and unlaunched until the next
    prefill. That is the window-2 span.
  - Not the trigger: memory reservation (`SC_RESERVE_GB=2`: the flip remains) and the TRT-LLM redo (refused: the flip
    remains).
- **Fix (with the core agent, approved):** upload lazily, right before the first launch, in the same call: upload, set nodes,
  launch. This keeps the set-before-upload successor penalty away, and C++ shows upload-then-launch has no effect.
- **The ">8 streams" effect** appears only in the cutbench harness. The minimal repro doesn't show it, even with 4 pending
  uploads. It stays an open lead in the NVIDIA bundle.
- **vLLM 8B census (fullnc):**
  - 5 streams given work; 86 execs instantiated, 79 never launched, 0 explicit uploads.
  - The probe still reads slow (1.44 ms vs 1.02-1.10 ms fast).
  - In C++, 80 instantiated-but-never-launched execs have no effect, so vLLM 8B's slow probe state is unexplained by either
    mechanism found.
- **Rough cost of the state on an 8B decode step:** ~0.38 us x ~550 nodes ~ 0.2 ms (4-5% of ~4.7 ms). This is an estimate
  from the per-node cost, not measured.
- **NVIDIA bundle** (nobody contacted): land/scratch/splitcost/NVIDIA_BUNDLE/.

### vLLM 8B: the capture itself (10-18)

- The probe is fast after model load (1.03-1.06 ms) and slow right after vLLM's CUDA-graph capture (1.43 ms). That holds for
  the 2-graph profiling capture and for the real 83-graph capture.
- Replaying all 83 graphs once does not clear it, and it stays slow through warm-up and serving.
- In C++, neither global-mode captures nor 80 instantiated execs reproduce it. The capture-path trigger is still open (listed
  in the NVIDIA bundle).
- The step-time cost to vLLM was not measurable without a fast state to compare against. Estimate: ~+0.2 ms per 8B decode step
  (+0.4 us x ~550 nodes).
- **Fairness caveat for the 8B serving ties:** vLLM's arms run in the slow state all run. Ours runs fast except in the window
  after a new segment capture.

## Fix proposal

1. **No core change for the cut cost.** The per-cut device cost is ~1-5 us, at the ~2 us target. Keep the upload before any
   set in split_ready (the bench shows +1 ms permanently without it).
2. **Harness mitigation, for every lane.**
   - Time the fixed probe graph next to every measured window, replayed on the current stream with no extra streams
     (drive27.py `SC_PROBE=1`). On GB300: about 0.93-0.95 ms means the fast state, about 1.33 ms the slow state.
   - Discard or label any window in the slow state, or skip the window after the first drain/prefill cycle.
   - One-process-per-arm A/Bs must report each arm's probe state.
   - Re-measure the split numbers in SPLIT_PIPELINE.md and startup27_h3t_vt_rw_sec_revt1.json.
3. **Core follow-up (a real ~13% decode gain whenever the state is hit).**
   - Find which activity in our trace / redo / prefill-learn path gives work to extra streams and enters the state, and keep
     the streams that ever receive work at 8 or fewer, e.g. one fixed side stream for trace and harvest work.
   - Repro: `cutbench n=2823 ns=1000 streams=12` (C++), and drive27.py `SC_FORCE_STREAMS=16` (27B).

## Files

All in land/scratch/splitcost/:
- `cutbench.cu`, `sweep1.sh`: the synthetic.
- `launchshim3x.cpp` + `shimspin.cu` (`build_shim.sh`): the shim.
- `bench4.py`: relaunch and timeline.
- `bench5.py` / `bench6.py` / `bench7.py` / `bench8.py`: re-splits with held images, set-field and protocol tests, split-state
  rebuild.
- `drive27.py`, `startup27.sh`: learnfast's driver and launcher copies with the SPLITCOST_BENCH* hooks.
- `q_sc.sh TAG` (SC_MODES=off,on,off,on): the run.
- `run.sh`: own-core gate + gpu2.lock.
- Window-2 trigger logs: `logs/tl_cpp1.log` (C++ flat 288 s), `logs/streams1.log` / `logs/streams2.log` / `logs/legacy1.log`
  (stream-count repro), `logs/sc22.log` (CUPTI), `logs/sc23.log` (short windows), `logs/sc24.log` / `logs/sc25.log` (blocks),
  `logs/sc26.log` (config diff), `logs/sc27.log` (L2), `logs/sc30.log` / `logs/sc38.log` / `logs/sc40.log` (vLLM arm), `logs/sc32.log`
  (idle decay), `logs/sc35.log` / `logs/sc36.log` (forced streams), `logs/sc39.log` (probe timeline).
- Logs: `logs/sc14.log` (GPU 3, on,off,on), `logs/sc15.log` (GPU 3, off,on,off,on: whole exec relaunched in each
  window), `logs/sc16.log` (off x6), `logs/sc17.log` (seed 6), `logs/sc18.log` + `logs/sc18_fine.json` (47-chunk
  per-window times), `logs/sc2.log` (bench4), `logs/sc5.log` (bench5), `logs/sc6.log` (bench6), `logs/sc7.log` and `logs/sc9.log`
  (bench7), `logs/sc10.log` and `logs/sc11.log` (set and memset histories), `logs/sc12.log` (bench8), `logs/sc13.log`
  (off,on,off,on).
