# T-nccl: NCCL collectives under host tracing (design spike, 2026-09-26)

Scope: research plus short probes. No code in any wt-land* worktree.
Scratch, probes and raw notes are in `land/scratch/nccl/`:
- `PROGRESS.md`
- `xla_notes.md`
- `c10d_notes.md`
- `hitrate.py`
- `probe_mixed_child.py`, with its `probe_*.r*.json` results

Tags on every claim:
- **[V-probe]**: verified by a probe run today, or by the 2026-09-18 `nccl_facts/` probes (GB300, NCCL 2.30.7, 2 ranks, intra-node NVLink).
- **[V-src]**: verified against a source line.
  - NCCL is `<path>` (version.mk 2.30.7, the same version as the box).
  - XLA is `<path>` (rev 3be1cd85, older than GitHub main).
  - PyTorch is `/data/eellison/src/pytorch`.
- **[U]**: unverified inference.

## 0. Recommendation (short)

Treat an in-graph NCCL collective as a harvested kernel node with two pointer slots. Do not use an address-keyed child-graph cache.
- **Key.** (site, count), where the count is evaluated from symints. The comm, dtype, op and root are fixed per site.
- **Row.** Kernel function, grid, block, smem, launch attributes, the `extra` image, the qword offsets of sendbuff and recvbuff, and NCCL's serialEvent handle.
- **Hit.** Patch the two pointers and call SetParams only when they changed. This is the same C++ path as every other node, with no Python.

Memory equals eager because the collective reads and writes the same allocations eager makes.

The harvest declines unless:
- the capture is exactly [WAIT_EVENT, KERNEL, EVENT_RECORD] per kernel;
- only the pointer qwords differ between two captures with different buffers;
- a bitwise verify against eager passes.

Declines are voted across ranks. A declined site, or a collective whose wait is outside the graph, runs as an eager step between segments.

The first stage needs no upstream change. It does require graph registration to be off, or detected and declined. A per-comm NCCL knob and a small retarget API would make it robust. Both need upstream.

---

## 1. XLA's approach, concretely

Details and line numbers are in `scratch/nccl/xla_notes.md`. All of this is from the local checkout unless marked.

**Library calls (TracedCommandBuffer).** [V-src]
- `TracedCommandBufferCmd::RecordTracedCommand` (`backends/gpu/runtime/command_buffer_cmd.cc:309-336`) keeps a per-(command, command buffer) cache.
- **Key.** The deduplicated whole-allocation base addresses of every buffer the command touches (223-245). Slice offsets and sizes are not in the key.
- **Cache.** `xla_cmd_buffer_trace_cache_size` = 16 (`debug_options_flags.cc:249`), move-to-front with LRU eviction (250-298).
- **Miss.** Stream-capture the call into a nested command buffer via `cuStreamBeginCaptureToGraph` (`cuda_command_buffer.cc:602-660`).
- **Both paths.** A hit or a miss goes through `CreateChildCommand` → `cuGraphAddChildGraphNode` (which clones the graph, 415-435), or through `UpdateChildCommand` → `cuGraphExecChildGraphNodeSetParams` (474-487). A hit still pays a child update.

**Collectives are not cached at all.** [V-src]
- `CollectiveCmd::RecordTracedCommand` (1417-1439) re-captures on every record. The captured buffer is a local `unique_ptr` freed as soon as it has been cloned into the child node.
- Collectives have `requires_initialization() = true`. `CommandBufferThunk::Initialize` runs on every execution (`gpu_executable.cc:514-528`, `command_buffer_thunk.cc:176-217`).
- So every rank re-captures every collective on every step, even when no address changed.
- `command.h:191-200` explains why: all ranks must record NCCL commands together, and a rank that skips deadlocks the others.

**Lifetime of what was captured.**
- CUDA clones the child graph. When a child node is updated, the user-object references are cloned with it, so the parent exec keeps NCCL's resources alive. [V-src] (CUDA 12.8 guide text quoted in xla_notes.)
- XLA does nothing explicit about user objects (no grep hits). [V-src]
- The parent template graph pins the first trace and the exec pins the latest. [U]

**Other details.** [V-src]
- The first execution is eager, and recording happens before the run because instantiation allocates device memory, which could deadlock with in-flight NCCL (`command_buffer_thunk.cc:161-186`).
- An async start and its done must be in the same command buffer. A lone done is never converted (`command_buffer_conversion_pass.cc:262-321, 454-470`).
- COLLECTIVES is not in the default command-buffer set. `xla_gpu_enable_nccl_user_buffers` defaults to false.
- Update mode and VA remapping exist only on GitHub main. [V-src: main, per the subagent]

**What we take from XLA and what we don't.**
- We take "start and done in one graph".
- We do not take capture-per-step. It costs 370-700 us of capture plus ~150-280 us of readback per collective [V-probe, NCCL_FACTS Q3], against ~4.5 us to replay.
- We do not take the address-keyed cache. See section 3.

## 2. NCCL internals under capture

**Node shape.**
- Each collective captures as `WAIT_EVENT → KERNEL → EVENT_RECORD`. [V-probe, 09-18 and today]
- 64 grouped small all-reduces capture as 64 kernels plus 64 records and one wait. [V-probe, `probe_child`]
- The wait and record nodes come from NCCL's graph-mixing serialization, not from ProcessGroupNCCL:
  - an external `cudaStreamWaitEvent(captureStream, serialEvent, cudaEventWaitExternal)` (`misc/strongstream.cc:164-248`);
  - a serialEvent record (`strongstream.cc:347`). [V-src]
  - This corrects NCCL_FACTS Q1.
- With `graphStreamOrdering=0` (a per-comm option new in 2.30), the external wait on serialEvent is still always present; the kernel just runs on the origin stream (`enqueue.cc:1646-1662`). [V-src]

**Where the buffer addresses live.** [V-src unless noted]
- **Args storage.** `finishPlan` (`enqueue.cc:212`) chooses it when `sizeof(ncclDevKernelArgs)+batch+work <= workArgsBytes`, and `workArgsBytes = min(NCCL_WORK_ARGS_BYTES, 4 KiB)` (`init.cc:610`).
  - The work descriptors, including `ncclDevWorkColl.sendbuff/recvbuff` (`include/device.h:284`), are then inline in the kernel parameter image, passed via `extra`/`CU_LAUNCH_PARAM_BUFFER_POINTER` (`enqueue.cc:1765`).
  - The device reads them with ld.param (`device/common.h:228`).
  - Probed images are 176 B (1 channel) or 672 B (32 channels). The pointers are at qwords 9/10 or 71/72. [V-probe]
- **Persistent storage.** Used when the work does not fit, under capture only. The descriptors go into a device buffer made by `cudaMallocAsync` on the capture stream (`uploadWork`, `enqueue.cc:1248-1370`; `persistent = ncclCudaGraphValid(...)` at 1571-1592; a 1 GiB budget at 1610).
  - The pointers are then in device memory. Patching them would mean a memcpy node or a device write. Decline this case.
  - The 64-way grouped probe did not trigger it. [V-probe] Which real workloads hit it is unknown. [U]
- **Registered buffers.** `regUsed` / `netRegUsed` set means `sendbuffRmtAddrs/recvbuffRmtAddrs` point at device arrays of peer addresses (`device.h:284`). Peers hold our address, so a local patch is invalid. **Corrected 2026-09-28 (land/SYMM_DESIGN.md §2.2): peers get our address at runtime from our args via ptrExchange; registered rebind is a local ptr+offset patch (plus the per-record RmtAddrs word across records) [V-probe]. Persistent storage is patchable via a head patch kernel (§2.0 there).**
  - Registration under capture is on by default (`NCCL_GRAPH_REGISTER=1`, `enqueue.cc:283`; `register/coll_reg.cc:58-88, 180-300`).
  - IPC registration needs a cuMem handle (`transport/p2p.cc:1096-1130`). Legacy `cudaMalloc` needs `NCCL_LEGACY_CUDA_REGISTER`, which defaults to 0 (`p2p.cc:104`).
- **Per-channel counts** (`cbd countLo/Mid/Hi`, `chunkGrains`) are in the image. So is the channel mask. A new count means a new row. Rewriting counts works within one channel layout [V-probe, 09-18], but we key on count instead of relying on that.
- **Proxy ops** (net transport) carry buffer pointers only when net-registered. nsteps and nbytes are baked on the host (`enqueue.cc:2389-2431`). [V-src]
  - Multi-node captures add host-func nodes whose userData is the plan (`enqueue.cc:1703-1726`). [V-src]

**Patch or re-capture?** Patch.
- Rebinding the pointer qwords through `cuGraphExecKernelNodeSetParams` gives bitwise-equal results at 8 KiB, 2 MiB and 16 MiB, with both the default allocator and `expandable_segments:True` [V-probe, `probe_rebind`, `probe_rebind_exp`]. It costs 3-4 us [V-probe, 09-18].
  - NCCL logged "register comm ... buffer" (the graph-register cache insert, `register/register.cc:63`) for the plain-cudaMalloc run. The rebind was still exact, so no registered remote addresses were in use for RING LL/SIMPLE on 2 ranks. [V-probe]
  - NVLS and IPC-registered algorithms on more ranks or larger NVL domains were not exercised. [U]
- A captured replay on one rank interoperates with an eager call on the other, in both directions, bitwise equal at 3 sizes. [V-probe, `probe_mixed`; GPUs 0+2, see the note in section 7]
- Tuning can differ under capture:
  - `regBuff` is true under capture when `NCCL_GRAPH_REGISTER` is on, both for tuner plugins and for `CTA_POLICY_EFFICIENCY` (`enqueue.cc:2141-2162`). [V-src]
  - So in general a graph run and an eager run of the same call can choose different algorithms, which would hang when mixed. [U]
  - With `NCCL_GRAPH_REGISTER=0` both branches evaluate the same. [V-src] That no other capture-dependent tuning exists is [U].

**Symmetric-memory kernels.**
- `ncclSymkDevWork {redOpArg, nElts, inputWin, outputWin, inputOff, outputOff, rootRank, sChannelId, nChannels}` (`include/sym_kernels.h:81`).
- The offset is relative to the window's `userPtr`, or is the raw address when there is no window (`sym_kernels.cc:660`).
- They always use Args storage (`scheduler/symmetric_sched.cc:353`) and set programmatic stream serialization (PDL) (`enqueue.cc:1795-1826`). [V-src]
- Retargeting means patching `inputOff/outputOff`, which is valid only if the offset is the same on every rank. Under eager-order allocation in a symmetric pool, every rank allocates in the same order, so offsets should match. [U]
- torch symmetric-memory one-shot kernels are ordinary Triton/CUDA launches and are traced as such (the cascade 14 notes). [V-src per memory; not re-checked]

**Launch attributes a directly-traced NCCL node needs.** [V-src, `enqueue.cc:1795-1826`]
- cluster dims plus the spread policy;
- `MEM_SYNC_DOMAIN` (remote);
- the launch-completion event;
- programmatic serialization (symmetric kernels only);
- `NVLINK_UTIL_CENTRIC_SCHEDULING` (sm100, CUDA 13).

The row copies them from the captured node. [U: that our node builder can set all of them]

**Lifetime.**
- `persistentDestructor` (`enqueue.cc:1528`) only enqueues onto an MPSC queue, and the user object is `cudaUserObjectNoDestructorSync` (`strongstream.cc:117`). NCCL's graph destructor cannot block. [V-src]
  - So the 09-18 del-graph hang has another cause (unknown).
  - Today, resetting donor graphs after a child-graph swap did not hang. [V-probe, `probe_child`]
- With Args storage the image references only the comm's device state. No graph-owned buffer is involved, so a harvested row outlives its donor capture.
  - `probe_donor` [V-probe] read the image, reset the donor, churned the pool and ran 20 eager all-reduces. It then wrote the retargeted image into another graph's NCCL node, 3 times per size at 3 sizes, and every result was bitwise equal to eager.
  - Caveat: the host graph was itself an NCCL capture. A graph built purely by our builder, with no NCCL user object in it, is [U].
- `persistentRefs` matters only for host-stream proxy work (`enqueue.cc:1700-1731`). [V-src]

**A small "retarget" API in NCCL (needs upstream).**
```c
// After capturing one NCCL op on `stream`: describe the kernel node NCCL added.
ncclResult_t ncclCaptureGetOp(ncclComm_t comm, cudaStream_t stream, ncclCapturedOp_t* op);
// Rewrite send/recv in a copy of the node's params. Fails with ncclInvalidUsage when the plan
// is Persistent storage, registered (regUsed/netRegUsed/NVLS), symmetric with non-symmetric offsets,
// or has proxy ops.
ncclResult_t ncclCapturedOpRetarget(ncclCapturedOp_t op, const void* sendbuff, void* recvbuff,
                                    CUDA_KERNEL_NODE_PARAMS* params /* in-out */);
```
- NCCL already knows the work offsets in `finishPlan`, so this is roughly 150-300 LOC in NCCL. [U]
- A smaller first ask: a per-comm `ncclConfig_t.graphRegister` field. Today `NCCL_GRAPH_REGISTER` is process-global (`NCCL_PARAM`, `enqueue.cc:283`), and `ncclConfig_t` has `graphUsageMode` and `graphStreamOrdering` but no register switch (`nccl.h.in:86-107`). [V-src]
- Our harvest does not need either API: the flip-diff finds the slots in a version-agnostic way. The API would replace the diff with a contract.

## 3. Interaction with host-trace memory (P22)

**Setup.** `hitrate.py` [V-probe] records each c10d op's data_ptrs per iteration on 1 GPU, world size 1, with eager allocation order.
- Replay addresses under P22 "eager" mode are eager's addresses. That is the definition of the mode (`scratch/mem/PROGRESS.md`). [V-src]
- 3 warm-up iterations, then 30-60 measured.
- The site is the call position within the iteration.

| Scenario | Same address as previous iteration | In last 16 distinct (XLA MRU) | Distinct tuples per site | Shape key hit |
|---|---|---|---|---|
| TP decode, 8 layers, static batch size 4 | 0% (alternates with period 2) | 100% | 2 | 100% |
| TP, batch size from {1,2,4,8} | 2.8% | 72% | 14 | 100% once 4 keys are seen |
| TP, plus one variable-size eager allocation per step | 0% | 0% | 30 | 100% |
| DDP, 4 MB buckets, static or dyn | 100% | 100% | 1 | 100% |
| FSDP pattern (dedicated gather and reduce-scatter streams), static/dyn/noise | 100% | 100% | 1 | 100% |
| FSDP pattern on one stream (compiled shape), static/dyn | 100% | 100% | 1 | 100% |
| FSDP pattern on one stream, plus a 1-64 MiB eager allocation per step | 66% | 84% | 7 | 100% |

The FSDP rows use a simulation (`FsdpSim`), because `fully_shard` short-circuits at world size 1 (`_fsdp_param_group.py:402,447`) [V-src]. The simulated collectives go through the real c10d calls.

**Conclusion.** An address key hits only where buffers are persistent (DDP buckets) or live on a dedicated stream whose allocation history repeats (FSDP).
- For compute-stream temporaries it is fragile. These are the functional-collective outputs Inductor emits inside a graph (TP), and they are the case that actually lives inside host-traced graphs.
  - With static shapes and a cache of at least 2 it holds.
  - With 4 batch sizes it drops to 72%.
  - One variable-size eager allocation between calls takes it to 0%.
- Each miss would be a 0.4-0.7 ms capture plus a child update.

An offset key (offset within the run's plan) would hit 100%. But a captured NCCL graph bakes absolute addresses, so using an offset key means rebasing the pointers, which is patching.

A stable buffer hits 100% but costs memory or copies.

**Where each option sits.**
| Option | Memory vs eager | CPU per hit | CPU per miss | Hit rate |
|---|---|---|---|---|
| Address-keyed child graph (XLA) | equal for Args storage (images in params); Persistent storage adds a work buffer per cached graph [V-src] | child `ExecChildGraphNodeSetParams` (cost not measured [U]) | capture + readback, 0.5-1 ms [V-probe] | table above |
| Offset-keyed | equal only if it patches; a private stable base means extra memory | same as patch | same as patch | 100% |
| Stable buffers (copy in and out) | + the sum of collective buffers held per site or shape, e.g. 8 x 8 KiB for TP decode, but whole bucket sizes for training [U] | + 2 copy kernels per collective, 2x the bytes moved | none | 100% |
| Patch (recommended) | equal [V-src by construction] | 2 qword writes + one kernel-node SetParams when addresses changed, 3-4 us [V-probe] | one harvest per new count: 2 local captures + verify + one tiny vote all-reduce [U: ~2 ms] | 100% for a repeated count |

## 4. ProcessGroupNCCL / c10d under capture

Details are in `scratch/nccl/c10d_notes.md`. Line numbers are relative to `torch/csrc/distributed/c10d`.

**Backend.** "nccl" now selects the nccl2 backend (`c10d/nccl2/`), unless `TORCH_DIST_USE_NCCL2=0`. The old backend is "nccl-legacy" (`torch/distributed/distributed_c10d.py:876-898`). [V-src] NCCL_PRIOR_ART section 3 describes only the legacy backend.

**nccl2 under capture.** [V-src]
- The Work object is kept alive by a graph user object instead of the watchdog queue (`nccl2/ProcessGroupNCCLUtils.cpp:460-504`), and the abort check is skipped (361-365).
- The host-blocking sync is skipped, and `wait` returns early when there is no timeout (`nccl2/WorkNCCL.cpp:394-445`).
- The flight-recorder hook records nothing (`nccl2/hooks/FlightRecorderHook.cpp:473-507`).
- The NaN check runs as a pre-hook on the current stream (`NanCheckHook.cpp:44-60`).

**Legacy backend under capture.** [V-src]
- Requires NCCL >= 2.9.6 (`ProcessGroupNCCL.cpp:3759`).
- The work is never enqueued to the watchdog (3804-3807).
- Coalesced and P2P flight-recorder entries are still written (3808-3827, 4307-4320). A later dump queries captured events. [U: hazard]
- Barrier uses `cudaStreamSynchronize`, which is illegal under capture (5310-5378).

**Sequence numbers.** [V-src]
- Both backends bump their counters at capture and never at replay. Our replay does not call c10d at all, so the counters do not move either.
- If one rank re-traces (the eager learner calls c10d) while the others replay, the per-rank counters drift apart. Only debug tooling (flight recorder, desync reports) reads them. [U: no functional effect]
- Replays are invisible to the watchdog. That is the same as `torch.cuda.graph` today. [V-src]

**Async and wait.** [V-src]
- Functional collectives default to `asyncOp=true` (`Types.hpp:132`). They run on the PG's internal stream, joined by one reused event (`nccl2/ProcessGroupNCCLUtils.cpp:~506-523`).
- `wait_tensor` makes the current stream wait on the work's end event. The WorkRegistry is keyed by storage and is not capture-aware (`ProcessGroup.cpp:362-575`).
- In the tape, the pair becomes a fork edge (current stream → NCCL stream), the NCCL nodes, and a join edge at the `wait_tensor` position. The future-completion event is not a launch and is dropped. [U: tape design]

**Coalescing.**
- `_coalescing_manager` with small all-reduces gives one kernel per op, each with Args storage [V-probe].
- Inductor's bucketing emits one `all_gather_into_tensor_out` or `reduce_scatter_tensor_out` on a concatenated buffer, which is a single site [V-src, `torch/_inductor/comm_lowering.py:225-365`].

**Env to require or decline.** [V-src for the variable names]
- Decline, or document as unsupported: `TORCH_NCCL_BLOCKING_WAIT` or a work timeout (`wait` polls the event).
- Register-allocator hooks: `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK` or NCCL MemPool registration. Registered buffers give `regUsed` images, which the harvest declines.
- `NCCL_GRAPH_REGISTER`: see section 5.
- The NaN check is fine, since it is an ordinary kernel on the current stream.
- The flight recorder is fine on nccl2.

**What upstream capture supports.** [V-src]
- `test/distributed/test_c10d_cuda_graphs.py` captures and replays all ops, sync and async (173-200).
- Barrier works only on nccl2 and nccl-lazy.
- NCCL < 2.29 combined with expandable segments gives an IMA under graphs (`test/distributed/test_c10d_ops_nccl.py:347-380`).
- Communicators must be initialized and warmed up before capture.

**Functional collectives in the traced tape.** Inductor emits the following [V-src, `test/inductor/test_inductor_collectives.py:1194-1220`]:
1. `buf0 = empty_strided(...)`
2. the clone kernel, `run(arg0_1, buf0)`
3. `torch.ops._c10d_functional.all_reduce_.default(buf0, 'sum', group)`
4. `...wait_tensor.default(buf0)`

The tape sees:
- an ordinary allocation;
- an ordinary Triton kernel;
- an opaque c10d call, which becomes an NCCL site;
- a join.

all_gather and reduce_scatter are out-of-place, with fresh `empty` outputs [V-src, `torch/csrc/distributed/c10d/Functional.cpp:31-58`]. `all_to_all_single` bakes its split sizes on the host (353-384).

**Compiled FSDP2 and DDP.** Both keep their collectives out of the graph.
- FSDP2 hooks are dynamo-disabled (`torch/distributed/fsdp/_fully_shard/_fsdp_common.py:25-33`).
- DDP collectives run in the C++ reducer (`torch/csrc/distributed/c10d/reducer.cpp:1223`). [V-src]
- So these run eagerly between host-traced graphs today. Only SimpleFSDP-style traced graphs and TP/DTensor put collectives inside the graph.

## 5. Recommended design

**Tier.** A harvest-style patch tier. There is no child-graph cache and no stable buffers.
- A collective the harvest declines becomes an eager step, which splits the graph into segments (the existing segment and partition machinery [U: that it covers this]).
- The child-graph cache is dominated:
  - where addresses repeat, patching also hits;
  - where they don't, the cache misses;
  - the case patching declines (registered buffers) is exactly where a cached capture has registration-lifetime hazards (a captured registration tied to an address the allocator can reuse). [U]

**What the tape records per NCCL site.**
- op (`all_reduce_`, `all_gather_into_tensor[_out]`, `reduce_scatter_tensor[_out]`);
- group name, resolved to the comm;
- dtype, redop, root;
- count as an integer program over symints;
- operand buffer refs, which are planned allocations or inputs, like any kernel;
- the stream fork at the call and the join at `wait_tensor`.

**Harvest** (slow path, on every rank at the same SPMD point).
1. Capture the collective alone twice on scratch buffers A and B. Capture is local; no data moves. [V-src: no proxy ops intra-node; U: with registration off, nothing collective happens at capture]
2. Accept only if both hold:
   - the node list is `[WAIT_EVENT, KERNEL, EVENT_RECORD]`, with no MEM_ALLOC, MEMCPY or HOST nodes;
   - the kernel image differs between A and B exactly at the qwords equal to the A/B pointers.
   Registered or Persistent images fail this check because the RmtAddrs/workBuf pointers differ or extra nodes appear. [U: registered variant not exercised]
3. Verify: replay the patched row on fresh operands and compare bitwise with eager.
4. Vote: `all_reduce(MIN)` of the accept flag, so the decision is rank-uniform.
5. Store the row: func, grid, block, smem, attrs, image, slot offsets, and the serialEvent handle (from `cuGraphEventWaitNodeGetEvent`).
6. Drop the donor graph. [V-probe: `probe_donor`]

**Key and hit.**
- **Key.** (site, evaluated count). all_to_all with per-rank splits is declined in stage 1.
- **Node.** An NCCL site is a kernel node whose predecessors are an EVENT_WAIT node on serialEvent and the fork edge, and whose successors are an EVENT_RECORD node and the join.
- **Hit.** Evaluate the count and look up the row. If the function or image changed, SetParams. Otherwise write the pointer slots and SetParams only if they differ. No Python.
- **Event nodes.** The serialEvent wait and record keep NCCL's device-wide serialization with other graphs and with eager NCCL work. We do not drop them.
- **New node kind.** Event nodes are the only new node kind in the graph builder. Their parameters are static per comm.

**Memory.** Exactly eager. The collective uses the same planned allocations. NCCL adds nothing for Args storage. [V-src]

**Declines** (run as an eager step):
- a Persistent or registered image (harvest fails);
- multi-node host nodes (stage 1);
- all_to_all with splits;
- a `wait_tensor` outside the graph, meaning an async collective whose done is in another graph or in eager code (the XLA rule, section 1). Without this rule, the later `wait_tensor` would find no Work and not wait. [U]
- `BLOCKING_WAIT` or a timeout;
- the legacy backend barrier.

**Rank uniformity.**
- **Keys.** Keys are functions of SPMD-uniform values (counts, dtypes), so hits and misses coincide across ranks.
- **Row decisions.** Voted.
- **Residual.** Local re-traces, where one rank runs the eager learner while the others replay, rely on graph/eager interop.
  - Verified for RING LL/SIMPLE on 2 ranks [V-probe].
  - Algorithm parity in general needs `NCCL_GRAPH_REGISTER=0` [V-src], or a per-comm knob (needs upstream).

**Escaping cases.**
- **FSDP prefetch spanning graphs.** With compiled FSDP2 the collectives are outside the graph and run eagerly [V-src]. With a SimpleFSDP-style all-gather started in graph N and waited in N+1, the site declines by the wait rule.
  - Alternative (future): the tape emits a real event record at the end of graph N and registers it as the Work for the output storage. That needs a c10d hook, which is local PyTorch. [U]
- **Reduce-scatter into `.grad`.** The output is either a graph output or a mutated persistent `.grad` input. Both are ordinary addresses, and patching covers them with no copy-out. [U: that host trace's output planning treats it like any output]
- **Input produced by a previous graph after a break.** This is just an input address, and pointer rows cover it.

**Symmetric memory (T-symm, later).**
- NCCL symk kernels use the same patch route, with slots `inputOff/outputOff` plus the window handles.
- The harvest checks rank-symmetry of the offsets through the vote: all-gather the offsets and require them equal. [U]

**Implementation stages** (rough LOC, all [U]):
| Stage | Content | LOC | Upstream |
|---|---|---|---|
| N0 | Tape recognizes the c10d functional ops and `wait_tensor`, records fork/join, declines everything else as eager steps | ~250 Py | local |
| N1 | NCCL harvest provider: 2 captures, node-shape and diff check, verify, vote, row | ~300 Py | local |
| N2 | Event-wait and event-record node kinds, NCCL rows in the C++ hit path, launch-attr passthrough | ~250 C++ | local |
| N3 | 2-GPU tests: TP decode, dyn batch, a decline path, mixed re-trace | ~250 Py | local |
| N4 (opt) | Per-comm `graphRegister` config and the retarget API | ~20 + ~250 C | needs upstream (NCCL) |
| N5 (opt) | Cross-graph async (Work registration from the tape) | ~150 | local (c10d) |

N0 through N3 add up to about 1,050 LOC with no upstream change, given `NCCL_GRAPH_REGISTER=0` in the environment or the harvest declining registered images.

## 6. Open questions for Elias

1. **Setting `NCCL_GRAPH_REGISTER=0`.** Is it acceptable to set it process-wide when host-trace NCCL is enabled? It matches eager, which does not register either, and it guarantees algorithm parity. But it also changes trees and plain `torch.cuda.graph` captures in the same process. The alternative is to rely on the harvest declining registered images, and to accept a hang risk on mixed re-traces. [local vs needs upstream: the per-comm knob]
2. **Cross-graph async (a SimpleFSDP-style prefetch).** Decline it (stage 1), or add a Work-from-tape hook in c10d (local)?
3. **Pinning to NCCL internals.** Is an image-diff contract acceptable while the retarget API is not upstream? The diff is version-agnostic, but it assumes the Args layout stays inline.
4. **Multi-node.** Should it be in scope for the first stage? It adds proxy host nodes whose userData is owned by the donor, so the donor must stay alive. The hit path would be unchanged.
5. **Declined collectives.** Should one split the graph into segments (memory-exact, one extra host round trip), or should a decline refuse host trace for the whole graph (simplest)?

**Upstream asks:**
- **needs upstream (NCCL):** per-comm `graphRegister`; `ncclCaptureGetOp` / `ncclCapturedOpRetarget`.
- **needs upstream (NCCL, question):** a doc guarantee that captured and uncaptured calls pick the same algorithm when registration is off.
- **local (PyTorch c10d):** a tape-side Work registration for cross-graph waits (optional).
- **local (host trace):** event node kinds, the NCCL harvest provider, and the decline-to-eager-step path.

## 7. Probes run today and caveats

- `hitrate.py` on GPU 2 alone (world size 1). Allocation-order effects only; no real peers.
- `probe_mixed_child.py`:
  - `mixed` on GPUs 0 and 2. GPU 0 had just been picked up by the sweep's next model when the probe launched, which was my slip against the two-idle rule. The probe ran for about 30 s and used about 2 GiB.
  - `child` and `rebind` (with and without expandable segments) on GPUs 1 and 2, both idle.
  - `donor` on GPUs 0 and 1, both idle at launch.
- Not exercised: NVLS, IPC registration, Persistent storage, multi-node, more than 2 ranks, one-rank-only re-capture.
