# Pinned host <-> device copies under host tracing (candidate 2 / candidate 3), for arm VP's single launch

## Probe
probe/memcpy_probe.py, run on gh200-b against candidate 3 (int6c3/cand3_s220, sm_90).
Log: gh200-b:<path>
Each case is a HostTraceReplay entry called 4 times. Before each call the pinned buffer is written on the host, and
the results are checked.

| case | outcome | decline (first) |
|---|---|---|
| H2D, pinned buffer as an argument (same buffer every call) | eager every call (traces 1, variants 0), results correct | `arg0 is on cpu; only CUDA tensors on one device are traced` |
| H2D, rotating pinned buffers as the argument | same | same |
| D2H into a pinned buffer passed as an argument | same | `arg1 is on cpu; ...` |
| H2D, pinned buffer held in a closure (not an argument) | eager every call | `aten.copy_.default of a tensor the trace does not track` |
| D2H into a closure-held pinned buffer | eager every call | same |
| D2D control | traced, 1 variant, 1 segment, replays correct | - |

## Where core refuses, and what it supports (candidate2/src_cpp; the same code in cand3)
1. Trace arguments must be CUDA tensors on one device: `torch/cuda/_host_trace_tape.py:2914`.
2. A CPU tensor that is not an argument and not a host step's own: `_host_trace_tape.py:778` (`_dispatch`: "of a tensor
   the trace does not track"). The jiterator twin is at line 700.
3. A pinned allocation inside a trace: `_host_trace_tape.py:502-503` ("a pinned allocation").
4. The traced copy_ host is "copy_impl and copy_device_to_device on one GPU": `aten/src/ATen/native/cuda/Copy.cu:513`
   (int6 build_cpp7 src). A contiguous same-dtype copy becomes `MemcpyRecord{dst, src, nbytes}`. The record carries
   no kind (H2D/D2H).
5. The tape's Memcpy is documented as a 1D device-to-device node: `_host_trace_tape.py:246-252`.
6. Capture issues `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice)`: `_host_trace_capture.py:602-605`.
   `_memcpy_node` refuses anything but a 1D D2D node: `_host_trace_capture.py:409-417`. The verify path is at 664-672.
7. Lowering requires a memcpy's bytes to lie inside its allocations' roots (arguments or eager allocations):
   `_host_trace_lower_tape.py:624-633`. Pointer slots are at 495-500. A selector fold refuses any memcpy:
   `_host_trace_lower_tape.py:880-881`.
8. Replay patches device pointers of memcpy nodes; the first replay sets every memcpy (`_host_trace_capture.py:703-707`).
   Nothing patches host-side (srcHost/dstHost) parameters.

## Semantics
An eager `non_blocking` copy from or to pinned memory reads or writes the host buffer when the copy runs on the
stream. A graph memcpy node with a host operand does the same at graph execution. So a traced H2D/D2H node would
match eager as long as the host buffer stays untouched while the graph is in flight. vLLM already assumes this for its
own UVA / pinned pools (2 slots, `_DEFAULT_MAX_CONCURRENCY`).

## Minimal core change for memcpy nodes (not applied; proposal only)
- Tape: accept a pinned CPU tensor as an argument root of kind "host" (tape.py:2914), usable only by copy_.
- Traced copy_: record `MemcpyRecord{dst, src, nbytes, kind}`, with kind H2D or D2H when one operand is pinned host
  (Copy.cu:513).
- Capture: `cudaMemcpyAsync` with that kind. `_memcpy_node` accepts host memory type for the host side
  (capture.py:409-417, 602-605, 664-672).
- Lowering/replay: a host pointer slot patched through the memcpy node's host field (C++ Patch.cpp, plus the
  lower_tape.py:624-633 root check for host roots).
- Rough size: ~40 lines of Python and ~40 lines of C++, plus a test on the probe's cases.

## What VP uses instead, with no core change: UVA views (implemented, validation pending)
Pinned memory has a device-accessible UVA address. vLLM itself hands kernels UVA views of pinned buffers
(UvaBackedTensor.gpu, `get_accelerator_view_from_cpu_tensor`), and those are CUDA tensors to the tracer.
ARMV_VP_UVA=1 (default with ARMV_VP=1) in armV_cand2 does two things:
- Input staging (hostcuts uva mode): the staging slots stay pinned host buffers. idx_mapping, query_start_loc and
  is_padding are slices of the slot's UVA view, so the step's prep kernels read them at graph execution and the H2D
  memcpy is gone. The slot rotates (K = 2, as vLLM), and the view is a per-call argument.
- Outputs: the VP step writes the sampled tokens and num_sampled into pinned UVA output slots with kernels
  (`torch.add(x, 0, out=uva_slot)`; 4 slots in rotation). AsyncOutput's two `async_copy_to_np` calls return those
  pinned numpy views. Its copy_event, recorded on copy_stream after `wait_stream(main)`, still orders `get_output`
  after the graph.
- Result: no copy left outside the graph. The target is 1 launch per decode step. Correctness (--check plus greedy
  tokens against V) and the launches-outside-graphs profile are queued in timing_vp_c2.sh on GPU 0.
