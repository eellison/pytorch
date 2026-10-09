# Side-stream work inside a traced region (design note for review; not implemented)

Cases: vLLM moe_forward_shared (Flash-Next, 48 eager steps per decode step), Qwen3.5-35B-A3B MoE shared experts,
vLLM AsyncOutput's copy_stream. Torch repros: core_gaps_repro.py gap_3_op (custom op), gap_3_inline.

## Today
The trace runs under a thread-local capture on a private stream P (`_capture`, _host_trace_tape.py:2999), and a
launch is recorded only on P: `_Trace.check_stream` (:398) declines anything on another stream (`side_stream`). Inside
a custom op body that decline makes the op an eager step (moe_forward_shared: an eager step per layer). Outside one,
it declines the trace.

## Eager semantics to reproduce
- A side stream S joins P's capture when it waits on an event recorded on a capturing stream: `S.wait_stream(P)` or an
  `Event.record()` / `wait()`. This is the fork.
- Work on S is ordered after the fork, and P's later work is ordered after `P.wait_stream(S)` (the join).
- Anything on S not ordered by an event runs concurrently with P.

A CUDA graph captures exactly that: fork/join become dependency edges and add no nodes.

## Recording (trace time)
1. Stream identity in the tape. Each launch record carries a logical stream id. 0 is P. k > 0 is the k-th distinct
   stream (by `cuda_stream` handle) the trace launched on, with its priority (a graph keeps priority as a node
   attribute when stream priorities differ). A launch on a stream that is not capturing (never forked from P) still
   declines, as now ("a stream never joined to the trace's"). So does an unjoined fork at the trace's end
   (cudaStreamEndCapture's error).
2. Dependencies, measured instead of inferred. At each recorded launch the trace adds a marker node to the launch's
   stream under the capture (an event-record node, cudaEventRecordExternal) and keeps its handle. After capture_end,
   the capture graph's edges between markers are the tape's ordering.
   - This catches every way user, library or C++ code orders streams: wait_stream, Event record/wait, CUDAEvent in a C++
     op, CUDAStreamGuard. Nothing is intercepted at the API level.
   - Per launch the tape keeps `after`: the launches on other streams it directly depends on. Same-stream order is
     implicit.
   - Cost: one event-record call per launch, at trace time only. The "operations the trace does not record" check counts
     the markers as expected nodes.
3. v1 scope. Kernels, memsets and memcpys, keyed sites and Triton/CuTe launches on a side stream are recorded. Decline
   for now, each with its own reason:
   - an eager step on a side stream (its replay would need a real stream and events);
   - `record_stream` (the allocator's cross-stream lifetime: not needed at replay, see Memory, but it changes eager's
     free timing).
   Allocations made while S is current are ordinary tape allocations; the stream doesn't matter at replay (below).

## Replay capture (capture_tape)
- A run (segment) whose launches use streams 0..K is captured on its origin stream O, with K private side streams of
  the recorded priorities (per device, from a pool, as private_stream).
- Launches are issued in tape order, each on its stream. Before a launch with `after` edges, an event is recorded on
  each predecessor's stream right after that predecessor, and the launch's stream waits on it. At the run's end O waits
  on every side stream, the join (a fork still open there is joined at the run's end; see Ordering).
- The capture check "each launch adds exactly one kernel node to its run's stream" applies per launch stream.
- Each node is taken from cudaStreamGetCaptureInfo's dependency set right after its launch, so node-to-launch pairing
  never depends on cudaGraphGetNodes order across streams (observed creation order, undocumented). This is the
  information we have at capture; keep it rather than recover it.
- Native replay is unchanged: one graph exec per run, launched on the caller's current stream, patched per node.

## Ordering preserved exactly (or stricter, never weaker)
- Inside a run, the graph has exactly the recorded edges plus per-stream order.
- A run boundary (an eager step on P) completes the whole run's graph on O before the next step, which is stricter than
  eager for a fork that spans the boundary but never weaker.
- A side launch never runs after the region's end: O joins every side stream at its run's end, and eager's own user
  code joined it (else the trace declined).

## Memory plan across streams
- Liveness by interval, not by the launch's own seq. For a launch L on a side stream:
  - start(L) = seq of its latest ancestor on stream 0 (its fork point);
  - end(L) = seq of its earliest descendant on stream 0 (its join), or the run's end.
  A launch on stream 0 has start = end = its seq.
  A buffer's lifetime spans its users' intervals. Two buffers whose intervals are disjoint are ordered through stream 0,
  so placing them on the same bytes is sound. Two that can run concurrently overlap and get distinct bytes.
  plan_memory takes these in place of last_seq / the first use; run_buffer, held and planned all read them. CuTe's
  check_plan gains the same intervals.
- Allocator per-stream pools: at replay every allocation (eager order, run buffer or arena) is made on O before its
  run's graph launch, and the graph's branches run inside that launch on O. So the caching allocator sees one stream and
  no cross-stream reuse arises. At trace time allocations are placeholders. The warm-up (eager) uses the user's streams
  as eager does.

## Tests (after approval)
- gap_3_op and gap_3_inline: bitwise over a size sweep, 0 eager steps.
- Assert the replay graph has two branches between the fork and the join (node dependencies), and that a buffer used
  on S keeps its bytes through the join (plan intervals).
- Declines: a never-forked stream, an unjoined fork, an eager step on S, record_stream.
- A two-side-stream fork and a fork spanning an eager step.

## Asks
1. OK to add a marker node per launch at trace time? It's the measured route; the alternative is intercepting
   wait_stream/Event at the Python level, which misses C++ ops' events.
2. OK for v1 to decline eager steps and record_stream on side streams?
3. OK that a run boundary inside a fork is stricter than eager (the run completes before the eager step)?
