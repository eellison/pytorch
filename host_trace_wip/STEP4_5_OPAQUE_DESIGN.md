# Steps 4-5 design: opaque library calls (cuBLAS, cuDNN, attention) and RNG

Replaces section 3 of `STEP2_3_INDUCTOR_CUBLAS.md` for the step 1b stack. There, a harvested
region was a private-pool graph fragment behind the int-14 recorder. Here it is one more tape
record next to `EagerCall`, and its fallback is the 1b eager step we already have. Code
references: snapshot `land/snapshots/20260925-1bdone` (extracted at `land/scratch/step45/src`).
Items 1-2 come first. Nothing below is needed to land them, and every PR here is behind a
default-off knob.

## 0. Recommendation

- The seam is **`OpaqueBinding`**: an opaque call's parameter mapping and guards for this shape.
  **`OpaqueProvider.bind(key)`** returns it or `None`. The tape records **`OpaqueCall`**,
  which is an `EagerCall` plus the key's rows. A miss runs the call as today's 1b eager step,
  so correctness never depends on a provider.
- The first provider is the per-shape run: a harvest of capture pairs, as int-14 did, rewritten
  in Python over cuda.bindings. It covers only bf16/fp16 `mm`/`addmm`/`bmm`, with the cuBLAS
  workspace cache off.
  - On main that cache is off by default for CUDA (#194311), so the workspace is a per-call
    caching-allocator block. It becomes plain scratch, and the upstream workspace-lifetime bug
    is out of scope.
- Attention and RNG stay eager calls until the model sweep shows they matter. Then attention
  goes through the same seam, and RNG gets one extra slot kind plus PR 0.

## 1. The seam

The seam types live in `torch/cuda/_host_trace_opaque.py` (PR 4.1). It imports nothing from the
tape, so the harvest module (`_host_trace_harvest.py`, 4.3) builds these types directly.

```python
# Operands, in this order everywhere (key, slots, learn): the call's tensor leaves of
# pytree.tree_leaves((args, kwargs)) in leaf order (a tensor passed twice appears twice),
# then its fresh outputs in return order. An in-place or out= return is its argument,
# not a second operand. An operand's address is its data_ptr() (storage offset included).

@dataclass(frozen=True)
class OpaqueKey:          # what the kernel choice may read; also the binding's guard
    op: OpOverload
    dtypes: tuple[torch.dtype, ...]        # per operand
    sizes: tuple[tuple[int, ...], ...]     # per operand
    strides: tuple[tuple[int, ...], ...]   # per operand, in elements
    align: tuple[int, ...]                 # per operand: address % 256 (a fresh output: its
                                           # storage offset's bytes % 256; blocks are 256-aligned)
    scalars: tuple                         # the non-tensor leaves, in leaf order (SymInts as ints)
    settings: tuple                        # provider.settings() at this call (section 2)

@dataclass(frozen=True)
class Slot:               # bytes of a node the replay writes per call
    offset: int           # byte offset into OpaqueKernel.image (or the memset's dst)
    width: int            # 8, or 4 for cutlass 2.x packed pointers
    kind: str             # "operand" | "scratch" | "philox" (section 4)
    index: int            # operand i, or scratch buffer j
    delta: int            # bytes added to that address

@dataclass(frozen=True)
class OpaqueKernel:
    function: int         # CUfunction handle (cuGraphKernelNodeGetParams .func)
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    smem: int             # dynamic shared bytes
    attributes: tuple     # ((CUlaunchAttributeID, value), ...): cluster dims etc., sorted
    params: tuple[tuple[int, int], ...]    # (offset, size) per parameter, cuFuncGetParamInfo
    image: bytes          # the packed parameter buffer; slot bytes are don't-care
    slots: tuple[Slot, ...]

@dataclass(frozen=True)
class OpaqueMemset:       # a 1-D or 2-D memset node, as cuGraphMemsetNodeGetParams
    dst: Slot             # offset 0, width 8
    value: int
    element_size: int     # 1, 2 or 4
    width: int            # elements
    height: int
    pitch: int            # bytes

@dataclass(frozen=True)
class OpaqueBinding:
    nodes: tuple[OpaqueKernel | OpaqueMemset, ...]   # one chain, in stream order
    scratch: tuple[int, ...]                        # bytes per scratch buffer
    rng_increment: int = 0                          # philox offsets consumed
    # property topology: per node its kind and attributes; one merged exec per topology

class OpaqueProvider(Protocol):
    # At trace time, over traced tensors: may read op, dtypes, ranks, devices and constant
    # leaves, never sizes, strides or SymInt values (a branch on them records a guard).
    # Returns why it declines, or None to take the call.
    def accepts(self, op, args, kwargs) -> str | None: ...
    def settings(self) -> tuple: ...                 # read once per opaque call at a replay
    def bind(self, key: OpaqueKey) -> OpaqueBinding | None: ...   # lookup only, no GPU work
    # At a replay whose bind missed, right after the eager op ran on the caller's stream:
    # `operands` are the real tensors in operand order (fresh outputs hold eager's result).
    # The provider caches refusals itself: a refused key costs a bind and a learn lookup.
    def learn(self, key, args, kwargs, operands) -> OpaqueBinding | None: ...
```

- **Guards.** The key is the guard: a binding holds for exactly the calls whose key maps to
  it (exact match for a per-shape provider). A provider that knows more (a DynamoRIO table, a
  cuBLAS API) may own a coarser map and return slots that are expressions over key components;
  the tape never sees how the map is built.
- **Trace.** Providers are explicit: `trace(..., opaque=providers)` and
  `HostTraceReplay(..., opaque=providers)`. `_Trace.eager_call` (`_host_trace_tape.py`) makes
  all its existing checks (fake twins, eager roots, aliasing, writes). Then the first provider
  that `accepts` gets an `OpaqueCall(EagerCall, provider)`, a subclass, so every eager-call
  consumer is unchanged; otherwise the decline reasons, joined by "; ", go in
  `EagerCall.reason`. (4.1 as built.) From 4.2, an accepted call whose key binds at the trace
  (`bind_at_trace`, key at the hints) is recorded as the binding itself (`record_binding`):
  each kernel a `TritonLaunch` with `fields` over its template `image`, each memset a `Memset`,
  fresh outputs `empty_strided` allocations and scratch `uint8` allocations of the trace. The
  key's SymInts are guarded equal to their hints, so that variant holds for that key alone.
  Settings are read at the trace only. A miss, or a binding with rng, a launch attribute other
  than cluster, or a fresh output at a nonzero storage offset, stays an `OpaqueCall`.
- **Lowering** (`_host_trace_lower_tape.py`). An unbound call stays a `LoweredEagerCall`
  boundary, with a side table
  `LoweredTape.opaque: dict[step index, LoweredOpaqueCall]` holding the key rows: sizes and
  strides from the operands' view/prediction rows, `(base + offset * itemsize) & 255` per
  operand (`bitand`; an allocation or eager root's base is 0, its block 256-aligned), and the
  scalar leaves (a row per SymInt). The field `groups` holds the maximal step ranges that have
  no plain eager call and contain an opaque call. A bound call is plain launches:
  - Its kernels lower like Triton's, with slots packed at `fields`.
  - Its memsets lower to `LoweredMemset(seq, launch, slots=(dst,), width, height, pitch)`, one
    row each.
  - Its outputs are predictions, and its scratch becomes run temporaries in the memory plan.
- **Replay** (`_host_trace_replay.py`). In 4.1, `_commit` evaluates each opaque call's key,
  binds it, and always runs the call eagerly. On a miss, the provider then `learn`s from that
  run. In 4.2, a variant is chosen as follows:
  1. If a variant's opaque keys now bind, `__call__` skips it and traces again, inserting the
     new variant before the skipped one.
  2. This happens only below `max_variants`, and once per key tuple (`_Variant.tried`).
  3. At the cap, the unbound variant keeps its eager boundary; this is the topology cap.
  A bound variant is an ordinary 1b variant: one exec per (key, topology), with opaque nodes
  patched like Triton nodes and memsets through `cuGraphExecMemsetNodeSetParams`. The known
  cost is one variant per bound key. The follow-up is keyed launch tables inside one variant
  (C++ 3.2), which would give the merged per-(group, pattern) execs sketched earlier.
- **Capture** sets a kernel's cluster dims with `cuGraphKernelNodeSetAttribute` after the
  launch (cute62's route). Memsets use `cuMemsetD2D{8,16,32}Async`, and `_verify` checks
  their node parameters.
- **3.2 C++ requirements (from 4.2).**
  1. `TritonLaunch.fields: ((param, byte offset, width), ...) | None` (one per slot) and
     `image: bytes`. When `fields` is set, each layout parameter starts from
     `image[o:o+n]` (zeros if `image` is empty), and slot j is written at
     (param, offset) with that width. A pointer of width 4 is its address's low 32 bits.
     Scalars use `_SCALARS[width]`.
  2. `TritonLaunch.cluster: (x, y, z)`, set at capture when it is not (1,1,1). It is not
     patched at replay, since it is part of the topology.
  3. There is one memset record. The tape has `Memset(name, slots=(dst,), roots, value,
     element_size, width, height, pitch)`, and lowering has `LoweredMemset(seq, launch,
     slots=(dst PointerSlot,), width, height, pitch rows)`. `held = (dst address, width,
     height, pitch)`, patched with `cuGraphExecMemsetNodeSetParams`, where pitch is set only
     when height > 1. 2.3's `Memset(alloc)` becomes dst = the allocation's root at offset 0,
     value 0, element_size 1, width = pitch = nbytes, height 1.
  4. `TritonLaunch.owner` is the provider, which keeps the CUfunctions loaded.
  5. On the hit path, a variant with `lowered.opaque` must evaluate the keys and call
     `bind` before any effect. It then either skips and retraces (below the cap, keys not
     yet tried) or replays with the eager boundary.
- **What 4.3's cuBLAS bindings need from the representation (GB300 harvests, 2026-09-25).**
  1. Every nvjet kernel and `cublasLt::splitKreduce_kernel` carries
     `PROGRAMMATIC_STREAM_SERIALIZATION = 1` next to `CLUSTER_DIMENSION` (nvjet only: 2/4/8,1,1).
     Under the "launch attribute other than cluster" rule, no cuBLAS binding binds at the
     trace. Dropping PSS is correct (a PDL wait is a no-op without a programmatic edge; verified
     bitwise on split-K and plain shapes), so `bind_at_trace` should ignore PSS. Keep it in the
     binding, which is faithful to the capture.
  2. `record_binding` takes the cluster as `n.attributes[0][1]`. splitKreduce's only attribute
     is PSS, so the cluster has to be picked by attribute id.
  3. Slots are always width 8, at byte offsets 0 or 4 mod 8 within one parameter. nvjet has
     a single 1856-byte parameter passed through `extra`; cutlass align1 has a single 360-byte one.
     No 4-byte packed pointers were seen. `fields` covers all of it.
  4. No memset node appeared on any shape (split-K reduces through the workspace with a second
     kernel), so the memset record is unexercised by cuBLAS here.
  5. Scratch is one allocation per call: 32 MiB for mm/bmm, 1 MiB for the addmm bias path.

## 2. The per-shape run (the `HarvestProvider`)

- **Cache key: what cuBLAS reads.**
  - The key: `op`, dtypes, m/n/k and transposes (sizes, strides), lda/ldb/ldc and batch
    strides, each pointer's alignment, and alpha/beta. Only 1/0 is in scope, and int-14's
    declines are kept: 2-D bias, bias stride != 1, baddbmm, mixed dtypes.
  - `settings()`: device and SM count, `allow_tf32`, the fp16/bf16 reduced-precision flags,
    `preferred_blas_library()`, and `_get_sm_carveout_experimental()` (int-14 omitted the
    carveout: its bug). `CUBLAS_WORKSPACE_CONFIG`/`CUBLASLT_WORKSPACE_SIZE` are read once,
    because C++ reads them once.
  - Settings are read per replay only when the tape has an opaque call (estimated about 1 us for
    six getters, unmeasured); a C++ settings epoch is item 3's job. The key is on the opaque call, not a
    tape-contract guard, but the user asked to hold off on global-state guards: needs their OK.
    Hidden state outside this list is risk 2.
  - **Decided 09-25 (Elias): no `settings()` in the key yet** (no global-state guards): a harvest captures under the current settings, and toggling the reduced-precision flags (or carveout, BLAS library) mid-run is not detected; an accepted gap.
- **Harvest.** `learn` runs right after the miss's eager op, which has already produced the
  result. It keeps int-14's four captures:
  - A and B use address sets that flip bits between the alignment class and 2 MiB, C another
    stream, D the smeared stack.
  - Each qword is classified as operand+delta, scratch+delta, host (stack/heap by
    `/proc/self/maps`) or constant; leftover bytes are refused.
  - Params come from `cuGraphKernelNodeGetParams` and the layout from `cuFuncGetParamInfo`,
    which removes int-14's C++ image reader. The one irreducible C++ piece (~60 lines, PR 4.0)
    calls an OpOverload through the dispatcher after smearing 64 KiB of stack.
  - New refusal: a constant qword inside any caching-allocator segment is a device address the
    call does not own (the cache-on workspace, or a stale pointer).
- **Workspace.** With `TORCH_CUBLAS_WORKSPACE_CACHE` off (the CUDA default),
  `CublasLtWorkspace` (`CublasLtUtils.h:142`) and `getCurrentCUDABlasHandleWithWorkspace`
  (`CublasHandlePool.cpp:578`) allocate a fresh block per call; under capture it lands in the
  capture's pool.
  - So the workspace classifies as scratch j, and at replay it is a planned temporary in the
    run buffer: the eager pool, sized as eager's.
  - Cache on (ROCm, or the env var) hits the refusal above: decline, use eager. The upstream
    bug (captures sharing the cached per-stream workspace, cleared by `reset`) needs
    graph-keyed ownership (patch 10), only if we ever support cache-on. Not in scope.
- **Scratch discovery.** All captures share one process-wide harvest pool and stay alive until
  the harvest ends, so A's and B's scratch get distinct addresses. Scratch blocks are read from
  that pool's segments in `torch.cuda.memory_snapshot()`, which replaces int-14's alloc log
  (NRA-B1). Measure its cost; the fallback is a small C++ trace hook.
- **Verification.** Before caching, patch capture A's exec to the miss's real operands and a
  fresh output (keeping A's own scratch); launch, sync, and compare bitwise with the eager result just
  produced. It catches misclassification and any capture-versus-eager kernel choice difference,
  for one sync and one output buffer per new shape. It stays on while the knob is experimental.
- **Miss path.** An unknown key runs the 1b eager step, then `learn`. A refused key is cached
  as refused and stays eager. A per-process budget (default 64 new keys) stops harvesting.
  - Cost: 0.6-1.7 ms per shape plus the verification sync. The cache is an LRU keyed by
    `OpaqueKey`, shared across variants and entries (PYREC-B3). Lookups are lock-free; only
    `learn` takes a lock (NRA-B5).
- **Memory parity** (hard requirement: everything in the eager pool).
  - Merged exec: the opaque call's outputs are `empty_strided` from their `PredictedOutput` at
    the group's start, and its scratch comes from the run buffer, dropped once the graph is
    queued. That is the "allocated at step start" price `_host_trace_memory.py` already documents.
  - The harvest is transient and outside replay. Capture operands are views into one arena of
    about the largest operand plus 2 MiB, since captured kernels never run. The four workspaces
    live in the shared harvest pool until the captures are destroyed; its freed blocks stay
    reserved but not allocated, bounded by the largest harvest. Report both numbers in the sweep.
- **4.3 as built** (`torch/cuda/_host_trace_harvest.py` in wt-land-blas). Where it differs
  from the above:
  - Each capture gets its own fresh pool rather than one shared harvest pool.
  - Scratch comes from one unscoped `memory_snapshot()` per harvest, filtered by
    `segment_pool_id`. Scoped `memory_snapshot(pool)` can't be used: upstream
    `THCPModule_memorySnapshot` decrefs the pool-id ints it borrows, which corrupts the heap
    once pool ids pass 256.
  - The block count must equal the capture's `allocation.all.allocated` delta.
  - Verification is a direct `cuLaunchKernelEx` of the binding on the real operands, into a
    NaN-filled output with fresh scratch; no exec patching.
  - align1 cutlass kernels bind and verify, so they are not refused.
  - Cost is 5.6-5.9 ms per shape. Memory for 4096^2 bf16 with M=256 (36 MiB of operands): the
    arena is 2 x the operand windows + 2 MiB; peak +124 MiB allocated and +216 MiB reserved
    afterwards (arena + four 32 MiB workspace pools, reclaimed by `empty_cache` or an OOM
    retry).

## 3. Attention

In the land stack, no ATen attention host code is traced. All three SDPA backends are eager
calls today, which is correct at one boundary per layer.

| backend | op | path | notes |
|---|---|---|---|
| flash (in-tree FA2) | `_scaled_dot_product_flash_attention`, `_flash_attention_forward` | seam | `Flash_fwd_params` by value; splitkv's accum buffers become scratch; num_splits depends on SM count (in settings) |
| mem-efficient | `_efficient_attention_forward` | seam | int-14 traced it symbolically with an `offsetof` proxy (C++), deferred; the backward's i16 `num_splits_key` stays eager |
| cuDNN | `_scaled_dot_product_cudnn_attention` | seam, after conv | thread-local plan cache (backward on the autograd thread, so key per thread); 18 stack qwords in the 1088-byte struct (smear); faults below 16-byte alignment even in eager; needs the A457 selector fix upstream first |
| flex / Inductor Triton | compiled | traced symbolically by item 2 | the recommended path for varying sequence lengths |

- **Why seam, not symbolic:** symbolic flash or mem-efficient needs ATen C++ host changes
  (the int-14 approach; the SymInt flash work sat on a submodule bump a colleague reverted).
- **Limit:** a per-shape key misses on every new sequence length, so decode with a growing KV
  misses every step. The seam covers training and bucketed prefill; decode should use flex or
  Inductor attention (symbolic) or a fixed-capacity cache. Bindings with slots linear in the
  key (learned from two sizes, or from a table) are the follow-up, and where DynamoRIO pays most.

## 4. RNG

- **Today.** RNG ops are eager calls, so the generator advances exactly as in eager.
  Inductor's seeds come from `inductor_seeds`, one eager call that usually precedes the first
  run (one eager op, not a split; confirm in the sweep). Its Triton kernels read the seeds
  tensor as an ordinary argument, so item 2 needs nothing more.
- **In a binding** (native_dropout, bernoulli, flash dropout). Under capture, an ATen RNG
  kernel takes `PhiloxCudaState` by value: the seed/offset pointers of the generator's
  graph-safe state, and the u64 `offset_intragraph`, the prefix of earlier increments in the
  same capture.
  - The harvest adds capture E, which consumes a known increment before the op, so
    `offset_intragraph` classifies as a `"philox"` slot with a delta.
  - `rng_increment` for a key is measured at the miss's eager run (`get_offset()` before and
    after; a multiple of 4, since `increase` rounds up).
  - At replay, a merged exec's philox slots take the running prefix, and the group calls PR 0's
    `CUDAGraph.set_generator_increment(gen, sum)` before `replay()`. `replay_prologue` (1b
    already replays through `CUDAGraph.replay()`) then refills seed/offset and advances the
    generator by eager's amount.
  - Open: int-14 commit F found the state pointers can differ per capture (a side stream). The
    merged capture must read its own graph's pointers, either through a small accessor next to
    PR 0 or from a one-element `uniform_` captured into it. Decide in 5.1.
- PR 0 is ready locally (`hosttrace/00-generator-increment`, 139 lines) and waits for Elias's
  read before anything goes to GitHub.

## 5. What DynamoRIO, or a cuBLAS API, would replace

- **Replaced:** `HarvestProvider.learn`: the four captures, the smear C++, the classification
  heuristics and most of verification's role. **Unchanged:** `OpaqueCall`, key rows, groups,
  merged execs, memory and the eager fallback.
- **What a table buys:** (1) the exact key set, closing risk 2; its first use is research only,
  to find what cublasLt's selection reads and turn it into the cuBLAS-team ask. (2) Coarse keys
  with expression slots, so one binding covers a shape range (section 3's limit). (3) No capture
  per new shape, though a new CUfunction still needs one capture for its handle.
- DynamoRIO must never be a runtime dependency (EULA question open with a colleague). A checked-in
  per-cuBLAS-version table is a maintenance cost: flag it before building one. A cuBLAS API
  returning launch parameters per algo replaces the same piece with no maintenance.
- a colleague disagrees with the per-shape run, so section 1 is what to agree on with them; either
  approach then fills it.

## 6. PR slicing (in order, after items 1-2)

| PR | contents | tests | needs |
|---|---|---|---|
| 4.0 | C++: smeared dispatcher call (~60) | smear visible in a probe kernel's host qword | nothing |
| 4.1 | seam types, `OpaqueCall` record, lowering key rows, `groups`; replay binds but always runs eager (~350) | a `TableProvider` over a Triton cubin standing in for an opaque kernel: keys, misses, decline reasons | 1b |
| 4.2 | merged execs per (group, pattern), opaque node patching, outputs from predictions, scratch temporaries, `cuLaunchKernelEx` (~400) | same fake provider: bitwise vs eager over sizes, binding switches, topology cap, memory simulator from REVIEW_1B p2 | 4.1, PR 2.3 memset record |
| 4.3 | `HarvestProvider` for bf16/fp16 mm/addmm/bmm: captures, classification, segment refusal, verification, budget, LRU (~700) | M sweep 1..256 (nvjet changes at 73/80/96, split-K memsets), cache-on refusal, align1 cutlass refusal, carveout flip | 4.0-4.2 |
| 4.4 | cuDNN conv, then SDPA entries (~300 each) | priming capture; dgrad/grad_weight aliasing; per-thread plan cache | 4.3; sweep says it matters |
| 5.0 | PR 0 upstream (exists) | `test_graph_set_generator_increment` | Elias's read |
| 5.1 | philox slot kind, capture E, per-key increment, set-increment per group (~200) | dropout bitwise vs eager across sizes; generator offset equals eager after N calls | 4.3, 5.0 |

Measure boundaries removed per model in the item-2 sweep after 4.3. Build 4.4 and 5.1 only for
the models where they show up.

## 7. Risks and open questions

1. **Classification is value-matched, so heuristic:** smear depth (A422), align1/align2
   cutlass's per-call host qword, tf32 cutlass3x TMA descriptors. Recommendation: bitwise
   verification at learn time, cached refusals, knob off by default; a refused kernel stays
   eager, which is correct.
2. **Hidden state** (heuristics caches, handle or thread state, anything not in `settings()`):
   the binding runs a kernel eager wouldn't pick, breaking match-eager but not correctness.
   Recommendation: verification catches it at learn time; DynamoRIO research gives the list.
3. **Latency spikes**, 1-2 ms per new shape with the sync. Recommendation: the budget, plus
   learning the traced shape at trace time; out-of-line harvest only if the sweep shows spikes.
4. **Settings key versus "no global-state guards yet":** scoped to opaque calls and read only
   when the tape has one. Needs the user's OK.
5. **Reserved memory from the harvest pool:** allocated memory matches eager; reserved grows by
   one harvest. Recommendation: report it; release the pool per harvest only if it matters.
6. **Merged captures at replay time** cost an instantiate (hundreds of us) per new pattern.
   Recommendation: cap 8 per group, then split steps.
7. **Refused, as in int-14:** other streams, cooperative/priority attributes, multi-byte memsets.
8. **Open with a colleague:** agree on section 1 before 4.3; settle the DynamoRIO EULA question
   before any table work.
