# nsys CUDA tracing hangs on replayed CUDA graphs (gb300-box, 2026-10-18)

Environment: GB300, driver 580.126.20, Nsight Systems 2026.1.3 (/opt/nvidia/nsight-systems/2026.1.3), CUDA 13.0.

Status: profiling replayed graphs with nsys CUDA tracing hangs. The GPU goes idle and the host synchronize never returns.
- Graph-level CUDA tracing hangs too.
- Not filed with anyone (user's decision).

## Two observations

### 1. Stock PyTorch, no host_trace code: a torch.cuda.graph's second replay hangs under nsys

Repro: land/core/memset/nsys/sa2.py, one graph replayed 6 times with a synchronize after each:
`nsys profile --trace=cuda --cuda-graph-trace=graph python sa2.py [memset] [keep] [side]`.
- PyTorch nightly 2.15.0.dev20260825+cu130 (/data/eellison/envs/pytorch-nightly): 1 replay, then a hang. Same with memset nodes,
  keep_graph=True + instantiate(), and capture on a side stream.
- Our fork (step_ms27 / build_cpp_ms12c): same.
- Without nsys: 6/6 replays (nightly and ours).
- The earlier standalone.py (exec node sets between replays) hangs the same way, and so does its no-set control. Exec updates are
  not the trigger.

### 2. Our 27B replay (vLLM, learnfast's probes, step_ms24) hangs under nsys; the trigger is the PDL launches

Hang signature: the decode warm-up stops a few replays after the first decode trace. The host sits in torch.cuda.synchronize, the
GPU is at 0% with memory allocated, and there are no CUDA errors. nsys's own --duration kills the run.

| probe | nsys mode | change | result |
|---|---|---|---|
| repro | node | none | hang |
| A | graph | none | hang |
| B | node | trace_pdl=False + keep_programmatic_edges=False (PDL launches run as eager calls; no redispatches happened) | clean |
| C | node | mm_fp4 redispatch refused | hang |
| D | node | keep_programmatic_edges=False only | hang |

In the repro's report (exported to sqlite: nsys/an/ms24_decode.sqlite):
- the stuck launch is the second execution of one exec;
- it stops after a memset node, before the fp4 convert kernel that starts an mm_fp4 chain;
- the first execution of that exec ran all of its nodes.

Smaller cases did not hang under node-level tracing:
- the synthetic cuBLAS replay (nsys/hang.py);
- 61 replay-suite tests (side streams, native replay, memory);
- vLLM 8B bs-1.
None of them has PDL launches.

## Reading

- (1) shows that nsys CUDA tracing on this box breaks a plain graph's relaunch through torch.cuda.CUDAGraph.replay. That is
  independent of our code.
- (2): our native replays survive nsys until a graph holds PDL launches (B clean, D hangs).
- Both point at CUPTI/driver graph-launch tracing on this driver. Nothing indicates a bug in host_trace.

## Workarounds

- Profile with nsys and without PDL launches: torch.cuda._host_trace.trace_pdl = False.
  - PDL Triton and CuTe launches run as eager calls. The profiled program then differs: no programmatic overlap, more eager steps, and
    different redispatch behavior.
  - Use it for structure, not timing.
- For device timelines without nsys: CUPTI kernel activity through torch.profiler works (learnfast's runs).
  - It must not run in a process whose timings are reported (harness rule).
- Not a workaround: cuGraphUpload after sets, and setting nodes only while the exec is idle (standalone.py variants).
