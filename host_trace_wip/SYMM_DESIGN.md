# Distributed host trace: symmetric memory first, NCCL designed for (E-symm-first-nccl-later)

Scope fork 2026-09-28. Scratch: `land/scratch/symm/` (probes, copy bench, fetched external sources in `ext/`).
Tags: [V-src] read in source, [V-probe] measured here, [U] unverified.
NCCL source = 2.30.7 at `<path>`.

## 0. Summary

- **Symm:** port cascade 14 as a declared opaque provider ("Option A"). About 450-550 non-test LOC plus about 400 test LOC. Rendezvous tables are looked up per call by a native Lookup row, never pinned. There is no verify launch.
- **NCCL, measured against eager** (eager does no memory planning):
  - **(a) Unregistered (eager default): patch sendbuff/recvbuff.** Exact, rank-local, eager parity. Needs `NCCL_GRAPH_REGISTER=0` so the capture registers nothing eager wouldn't.
  - **(b) Opt-in registration (`register_mem_pool`, symmetric windows): also local patching.** Pointers and offsets inside a registration record are rank-local [V-probe]. A new record also needs one per-record device-array word from a per-segment table. Windows need offsets only, rank-uniform by the same assumption eager makes [V-probe].
  - **(c) Fixed registered buffer.** Beyond eager; a later optimization.
  - **(d) XLA remap.** Not chosen.
- **Rank agreement:**
  - Replay decisions may depend only on rank-uniform facts.
  - Anything collective at miss time goes through a trace-time vote.
  - Collective nodes are never verify-launched.

## 1. Symmetric memory

### 1.1 Landing line today
- The landing line (merge6/merge9) has no symm_mem references in the host-trace files [V-src grep]. Every `torch.ops.symm_mem.*` call is an EagerCall boundary.
- The landing line has no host tracer (TensorIteratorSym and similar). Its only entries are KernelLaunch (Triton/CuTe) and OpaqueProvider bindings (`_host_trace_opaque.py:54-200`; slot kinds operand/scratch/philox*; OpaqueKey = op, dtypes, sizes, strides, align, scalars, device).

### 1.2 What cascade 14 did (wt-integration-14)
- `torch/cuda/_host_trace_symm.py` (152 lines) registers traced entries for `one_shot_all_reduce{,_out,_copy,_copy_out}`. The host code of `CUDASymmetricMemoryOps.cu` was traced in the `host_trace_symm` namespace (+228 lines vs merge6).
- **Rendezvous lookups are rebound before every launch, not pinned.** The launch args (buffer_ptrs, signal_pads) come from a per-call lookup: the input's base address maps to its block, and group name + block map to the PeerAllocInfo. So a different symm buffer with the same shape serves through its own tables.
- **Rank-uniform guards.** `_check_rank_uniform_guards` declines statically if a guard recorded during the collective mentions the storage offset or base of a non-symm input. Serve-or-miss must depend only on shared facts, or one rank waits in the kernel barrier forever.
- **Declines.** The symm buffer must be a traced-function input allocated by `symm_mem.empty`.
- **Tests.** `test/test_cuda_host_trace_symm.py` (492 lines, MultiProcContinuousTest). Scenarios:
  - sizes;
  - back-to-back pad reset;
  - interleaving with eager collectives;
  - two buffers, each with its own tables;
  - an unrendezvoused buffer raises eager's error, or rendezvouses on demand;
  - a world-size change misses;
  - a rank-dependent local_input declines;
  - declines are symmetric;
  - a TP layer chain.

### 1.3 Port plan: Option A, a declared "symm" opaque provider
The kernel is ours. We know its parameter list, so there is no diffing: `learn()` declares slots by param index. Pair harvest could not find the rendezvous dependency anyway, because the block tables are constant for any two operands in the same block.

- **learn(key, args, kwargs, operands).** Runs after the eager miss call. Rendezvous is already cached, so this is rank-local.
  - Capture the real call in situ and take the one kernel node.
  - one_shot args: `(buffer_ptrs, out, local_input, input.storage_offset, numel, signal_pads, rank, world_size)`.
  - Slots:
    - `out` / `local_input`: operand;
    - `input.storage_offset`: a new int slot;
    - `buffer_ptrs` / `signal_pads`: new kinds `symm_buffer_ptrs` / `symm_signal_pads`.
- **Native Lookup row** (new; the program has no Lookup op today). Per call:
  - `find_block_covering(input base)`;
  - the interned group;
  - the cached PeerAllocInfo, whose device arrays give `buffer_ptrs_dev` and `signal_pads_dev`.
  - A miss returns "boundary" and the entry runs eagerly. Eager rendezvouses on demand, so the next call is a hit.
  - Cost is one hash lookup. Inductor comm buffers are persistent per alloc_id, so in practice it returns the same value every call.
- **No verify launch.** The kernel barriers across ranks, so an extra launch on one rank hangs or corrupts the pads (§3). Exactness comes from the declared roles, and the in-situ capture is the eager call's own launch.
- **Key.** OpaqueKey plus world size plus group. Rank is in the image and constant per process.
- **Declines.** Port `_symm_buffer` and the rank-uniform-guard rule. With no ShapeEnv on the landing line, the rule is enforced structurally: key components are all rank-uniform, and addresses only feed slots.

| Piece | File | LOC |
|---|---|---|
| Lookup row + 3 slot kinds, resolved in Patch | `torch/csrc/cuda/host_trace/{Variant.h,Program.cpp,Patch.cpp}` | ~150 |
| Lookup hook exported as `torch._C._cuda_symm_lookup`-style private binding | `aten/src/ATen/native/cuda/CUDASymmetricMemoryOps.cu` (or `SymmetricMemory.cpp`) | ~60 |
| Provider: accepts/learn/declines | `torch/cuda/_host_trace_symm.py` (new) | ~150 |
| Rank-symmetric declines (port + structural rule) | same | ~50 |
| Tests (port the 10 scenarios; no verify, pads reset, eager interleave) | `test/test_cuda_host_trace_symm.py` | ~400 |

- **Follow-ons:**
  - two_shot: about 40 LOC more.
  - multimem variants: the Lookup also returns `multicast_ptr`.
- **Upstream check [U].** The one-shot kernel appears to ignore the handle offset for MemPool sub-allocations; confirm before relying on MemPool-backed symm buffers.

## 2. NCCL, measured against eager

### 2.0 Where the pointer words live in 2.30.7 (persistent storage is not a blocker)
- **Args storage** (the work fits in 4 KiB of params, `enqueue.cc:1320-1360`). The `ncclDevWorkColl` is inline in the kernel param image. Layout (`include/device.h:284-318`):
  - bitfields (regUsed, ...) and root;
  - recvbuff, sendbuff;
  - sendbuffOffset, recvbuffOffset;
  - sendbuffRmtAddrs, recvbuffRmtAddrs;
  - the cbd counts, chunkGrains, redOpArg.
  - Patch it with a kernel-node SetParams, like any harvested row.
- **Persistent storage** (it doesn't fit; only under capture):
  - `workBuf` comes from `cudaMallocAsync(comm->memPool)` on NCCL's deviceStream, outside the graph.
  - It is filled by `cudaMemcpyAsync` from host at capture time. `kernelArgs->workBuf` points at it. The plan owns it, and graph destroy frees it.
  - To find the words, read workBuf back after capture and diff the pair, the same as for images.
  - To patch, add a head "patch kernel" node whose params carry (dst, value) pairs. It is exec-updated like any kernel and stream-ordered before the collective. Alternative: a double-buffered pinned H2D memcpy node.
  - About 80 LOC; no new mechanism. None of our probes hit Persistent storage yet (a 64-way grouped call didn't).
- **Unregistered net proxy ops** carry no user pointers.

### 2.1 Route (a): unregistered, the eager default (first NCCL route)
- **Eager:** NCCL copies through internal buffers registered at comm init. The user tensors are only read and written by the kernel.
- **Replay:** patch recvbuff/sendbuff, the only pointer qwords (71/72 at 32 channels, 9/10 at 1) [V-probe 09-18, re-confirmed].
- **Exact** (bitwise vs eager in every probe), **rank-local**, and **eager memory**.
- **Condition:** `NCCL_GRAPH_REGISTER=0`.
  - With the default (1), a capture on cuMem memory, which includes expandable segments, registers the buffer first (`ncclIpcGraphRegisterBuffer`, `coll_reg.cc:256-265`). That diverges from eager.
  - regBuff also feeds tuning (`enqueue.cc:2141-2162`), so the captured algorithm can differ from eager's (§3 hazard 3).
  - The switch is a process-global NCCL_PARAM (`enqueue.cc:283`). There is no per-comm config field.
- **Per replay:** one SetParams per collective node (3-4 µs measured 09-18; an existing Field row).
- **Key:** count (per-channel counts are in the image), dtype, op, comm, algo/proto.

### 2.2 Route (b): opt-in registration (ncclCommRegister, the NCCL MemPool, symmetric windows)
Eager opt-in path: `backend.register_mem_pool(pool, symm)` → `ProcessGroupNCCL::registerMemPool` (`ProcessGroupNCCL.cpp:1160`) → `registerSegment` per segment. That calls `ncclCommRegister`, or `ncclCommWindowRegister` when symm. Registration is per segment, and the caching allocator keeps segments. The allocator hook (`TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK`) is incompatible with expandable segments.

**How 2.30.7 gets RmtAddrs per call (IPC registration)** [V-src]:
1. `ncclRegisterCollBuffers` (`coll_reg.cc:137`) applies only to the SIMPLE protocol. It is skipped for RING in-place allreduce/reduce (`:212`) and for TREE in-place.
2. `ncclIpcLocalRegisterBuffer` (`p2p.cc:1232`) calls `ncclRegFind(comm, ptr, size)`, an address-range lookup over this comm's registration records.
3. The first use of (record, peer) goes through `ipcRegisterBuffer` (`p2p.cc:1010-1230`):
   - `cuMemRetainAllocationHandle` + export;
   - `ncclProxyCallBlocking` to the **peer's proxy thread** (`ncclProxyMsgRegister`) returns the peer-side mapping `rmtRegAddr`. This is one-sided: the peer's main thread and stream are not involved.
   - The host array is copied into the record's device array `devPeerRmtAddrs`.
   - It is cached in `regRecord->ipcInfos[peer]` until deregistration.
4. Per call, NCCL returns `offset = userbuff - record.begAddr` and `peerRmtAddrs = record.devPeerRmtAddrs`.
5. At device runtime, the sender publishes `offset + RmtAddrs[localPeer]` to the peer through the connection's ptrExchange (`prims_simple.h:755-830`). **Peers learn my address from my args at runtime.** This corrects NCCL_DESIGN.md §2 ("peers hold our address, so a local patch is invalid").

**What replay must rebind:**
- **Same record:** recvbuff, sendbuff and the offset words move with the local pointer. Probe (AllGather RING SIMPLE, 8 MiB, IPC-registered, 2 ranks): qwords 71/72/75 moved by the pointer delta; patched replay == eager bitwise, also when patching only the delta-moved words [V-probe `probe_reg.r0.json`]. Same with `NCCL_GRAPH_REGISTER=1` [V-probe `probe_reg_gr1`].
- **Other record (another segment):** additionally the RmtAddrs word (qword 77, the record's device array) changes. Patching it gave == eager [V-probe].
  - Replay needs a per-segment table (segment base → devPeerRmtAddrs value). A capture on that segment fills it at the miss, after eager's own call has created the record.
  - Filling it is rank-local safe: it is a one-sided proxy RPC, and eager does the same on its first call.
- **NVLS registration** (`nvls.cc` ~700-820) does `bootstrapIntraNodeBroadcast` + barrier. The first use of a new record is collective, so a capture must happen on all ranks together. Go through the vote (§3) or decline. Not probed; needs 4+ GPUs with NVLS.

**Symmetric windows:**
- `ncclCommWindowRegister` (`dev_runtime.cc:1333`) is collective (bootstrap allgather/barrier) and happens at registration, i.e. in eager, outside the trace.
- Per call there is a local lookup, `ncclDevrFindWindow` (`enqueue.cc:3045`).
- symk devWork (`sym_kernels.cc:660-670`) = inputWin/outputWin (vidmem) + `inputOff/outputOff = ptr - win->userPtr`. The kernel addresses peers as `ncclSymPtr(win, off)`, using **the same offset on every peer**.
- Probe (AllGather symmetric kernel, `ncclSymkDevKernel_AllGather_ST`): only qwords 44/45 (the offsets) differ; patched == eager [V-probe `probe_symm_small`].
- **Offsets must be rank-uniform. Eager assumes this too** (SGLang's allocator: "assume no individual allocation freed until pool destroyed ... asymmetry issues").
  - Replay gets eager's addresses because memory planning matches eager. So if eager's offsets are uniform, ours are.
  - This is the same symmetric-offset argument as for symm_mem.
- A different window changes the win word: the same per-window table as the IPC record. That target was not exercised; the probe's c buffer was in the same window [U].

**Fake pointers vs registration.** The harvest's pair operands come from a private pool: fake to the collective, and unregistered. For (a), that is fine. For (b), the harvest must run on the real registered operands (in situ, as in symm Option A). A fake operand would take the unregistered path, with a different kernel/algorithm and different words.

### 2.3 Route (c): fixed registered buffer (later optimization, beyond eager)
- **Variant A** (vLLM SymmMemCommunicator): copy in → collective on a persistent registered buffer → copy out. Copy cost measured on GB300, graph-captured, per copy [V-probe `copy_bench.txt`]:

  | Size | µs per copy |
  |---|---|
  | 64 KiB | 1.07 |
  | 1 MiB | 1.20 |
  | 8 MiB | 2.40 |
  | 64 MiB | 20.8 |
  | 256 MiB | 78.8 (~6.8 TB/s) |

  For comparison, 2-rank allreduce replay walls (with a ~20 µs sync floor) are 23 µs at 8 KiB, 31 at 2 MiB and 69 at 16 MiB. In+out copies add about 10% (small) to about 20% (16 MiB).
  - **Memory:** it holds one buffer the size of the largest collective, on top of eager.
  - **Exactness:** exact only when eager also does the copy. Otherwise the algorithm/kernel changes vs eager (registered vs unregistered paths differ).
- **Variant B:** the producer writes in place into the registered buffer. No copy, but it needs planning that eager doesn't do. This is real memory planning: out of scope until we agree to deviate from eager.
- Useful as a later knob: pinned buffer, zero patching. It is what Megatron/vLLM do today.

### 2.4 Route (d): XLA VA remap (considered, not chosen)
- **Correction to our earlier notes:** XLA PR #40173 (two-VA-set alternation) was **closed, not merged**.
- XLA main has `CommandBufferUpdateMode` {ALWAYS_UPDATE, SKIP_TEMP, SKIP_PROFILED} (`xla.proto:452-478`). It uses a VMM reservation, with inputs/outputs aliased per execution by `cuMemMap` + `cuMemSetAccess` (local + all P2P peers, `cuda_memory_reservation.cc:82-116`) and unmapped afterwards. "Requires the VMM allocator and assumes symmetric behavior across ranks."
- **Hazards for us:**
  - Granule alignment vs caching-allocator sub-allocation: an in-granule offset still needs patching.
  - Peers' NCCL IPC mappings of the old physical pages go stale unless every rank remaps the same way.
  - Per-replay map/unmap cost [U].
  - It needs expandable-style VMM for every tensor.
- Patching (a)/(b) is exact without any of that.

## 3. Rank-agreement protocol
- **Collective-count invariance.** Mixing graph replay and eager across ranks is fine if every rank executes each collective exactly once, in program order, with the same algorithm/config. Probe: graph replay on rank 0 against eager on rank 1 gives bitwise results at 3 sizes [V-probe `land/scratch/nccl/probe_mixed.r*.json`]. So a hit on one rank and a miss on another is benign.
- **Hazards:**
  1. **Verify launches**: an extra execution on one rank. Never verify collective nodes.
  2. **One-rank-only collective setup at miss time:** NVLS register, window register, and the first rendezvous of a new symm block.
  3. **Algorithm divergence** between captured and eager plans (GRAPH_REGISTER=1 regBuff tuning).
  4. **Grid divergence** from rank-local facts. The symm one-shot sync uses block ids, so the grid must be rank-uniform.
- **Rule:** classify each key/guard component as rank-uniform (shapes, dtypes, numel, world size, scalars, group) or rank-local (addresses, non-symm storage offsets, allocator/segment state). Topology, grid and algorithm may depend only on uniform components; rank-local components only feed patched words.
- **Vote:** at trace/miss time only, run a small CPU allreduce (gloo or store) over "capture outcome + needs-collective-setup" for any capture that could differ across ranks. A decline on any rank means all ranks decline. It costs about 100 µs per miss, and nothing on hits. Estimate: about 80 LOC plus tests.
- **Optional debug:** periodically hash the served variant id across ranks.

## 4. What open-source systems do [V-src unless marked]
- **vLLM custom allreduce:**
  - Eager copies into a pre-registered IPC buffer.
  - Under capture it uses the input pointer directly and records it in `graph_unreg_buffers_`.
  - After capture, `register_graph_buffers` exchanges IPC handles and offsets (`broadcast_object_list`).
  - Graph addresses are pinned.
- **vLLM SymmMemCommunicator:** a persistent symm buffer (max_size) rendezvoused once, then copy in → `multimem_all_reduce_` / `two_shot_all_reduce_` → copy out (= route c, variant A).
- **vLLM / SGLang `nccl_symm_mem` context:**
  - allocates from an NCCL MemPool (`ncclMemAlloc`);
  - on context exit, registers new segments via `ncclCommWindowRegister` (collective);
  - under capture, pauses the graph pool.
- **Megatron FullCudaGraphWrapper:** StaticBufferLoader copies inputs into static buffers, and one graph pool captures the whole fwd/bwd, NCCL included. Addresses are pinned, with no patching.
- **TE** [U, not read]: `make_graphed_callables` uses static buffers; userbuffers comm-GEMM overlap uses fixed pre-registered buffers.
- **XLA:** §2.4.
- **Pattern:** everyone pins addresses (static buffers or a fixed registered buffer). No one patches NCCL args. Route (a)/(b) patching is new, and the probes show it is exact.

## 5. Side note: cuDNN FE native graph API vs our harvest (E-official-api)
- `graph_interface.h:580-860`: `populate_cuda_graph(handle, uid→ptr, workspace, cudaGraph_t)` and `update_cuda_graph(...)` work on a cudaGraph_t template (one root child-graph node plus workspace memcpy/memset nodes). It is an official, documented pointer contract.
- **Recommendation:**
  - Consider it for SDPA (graph API) role discovery. It is an official contract and would replace harvest diffing there.
  - Keep our patch rows for replay: update_cuda_graph rebuilds the node params from the uid map, likely costing more per replay than our Field rows [U].
  - Conv (legacy FE path) stays on harvest.
  - Benchmark update cost before switching.
  - A backend-version gate at 92300 appears near line 1050 [U: which API it gates].

## 6. Probes not run (need hardware or time)
1. NVLS / IPC on 4+ GPUs: registration collective at first use; RmtAddrs table.
2. symk rebind across two different windows.
3. One-rank-only capture/verify: confirm the hang (hazard 1/2) and that the vote prevents it.
4. cuMem remap cost per replay (only if (d) is revisited).
5. cuDNN FE `update_cuda_graph` per-replay cost.
6. Graph-registration record lifetime under `NCCL_GRAPH_REGISTER=1` (moot if we require 0).

## 7. Open questions (each with a recommendation)
1. **Require `NCCL_GRAPH_REGISTER=0` process-wide when NCCL is traced?** Yes: it makes capture match eager. Later, propose a per-comm config field upstream to NCCL.
2. **Fund the vote protocol with the symm port, or decline everything collective-at-miss?** Land symm without a vote. Eager rendezvouses on demand before `learn`, and the Lookup misses to a boundary, so the symm port has no collective at miss time. Build the vote with NCCL route (b)/NVLS.
3. **NCCL (b): per-segment RmtAddrs/window table vs declining a new record until re-learned?** Start with "new record = miss → learn in situ" (reuses the provider path). Add the table only if segment churn shows up.
4. **Route (c) fixed buffer as a user knob?** Not in the first NCCL landing. Revisit when we agree to deviate from eager memory.
5. **Persistent workBuf patch kernel in the first NCCL landing?** Only if a target workload hits Persistent storage. Measure on real grouped collectives (FSDP) first.
