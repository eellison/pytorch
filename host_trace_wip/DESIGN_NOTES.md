# Host trace landing stack: design decisions and big snags

A curated summary. The full log is `../DECISIONS.md` (E = Elias's decisions, A = agent-made), and per-stage detail
lives in the REVIEW_*.md files and the snapshots' PROGRESS.md. Updated as stages land.

## What it is (one paragraph)

On the first call of a compiled graph, we trace the host side: every kernel launch, with its addresses, grid and
shape-dependent arguments as integer programs over the graph's symbols. We capture one CUDA graph and plan all of its
memory in eager's caching-allocator pool. Later calls evaluate the programs in C++, patch only the graph nodes whose
parameters changed, and replay. Library calls whose launches we can't compute (cuBLAS, cuDNN, SDPA) are "harvested":
we capture them a few times, classify each parameter qword (address, size, constant, scratch), verify with a seeded
relaunch, and store one launch row per shape key.

## Design decisions

**Memory and baseline**
- **Eager is the baseline, not trees** (Elias, 2026-09-25). Memory must match eager or compile. Speed is reported vs
  eager first; trees is a reference column.
- **Everything lives in eager's pool**: no private pools and no hold mode. We don't free inputs early.
- The harvest captures into a transient MemPool per harvest, dropped afterwards (A-land-merge2-8, A-item5b-7).

**Trust and guards**
- **Inductor trust contract**: layouts, dtype, device, rank and symints are trusted.
  - Generated-code size asserts are no-ops under the trace (A-land-merge2-7).
  - The trace only follows the dynamic spec Inductor compiled for (`_trusted_inputs`).
- **Static inputs get pointer rows.** `static_input_idxs` means "address-stable under trees", not "same tensor", so
  addresses are never baked in (REVIEW_22).
- **No global-state guards.**
  - cuBLAS settings (TF32, etc.) are not in the key (E-land-1).
  - The harvest accepts fp32 as well as bf16/fp16 (E-land-3).

**Replay**
- **No Python in the hit path.** The whole hit runs in C++ (`_HostTraceEntry`, 3.2b). Learner variants, which still
  contain eager opaque steps, stay in Python and are not registered with the C++ entry (A-land-merge2-5).

**Opaque library calls**
- **Per-site key -> row tables (4.2b).**
  - The harvest fills a row once per key.
  - A hit is one hash lookup plus a node SetParams, which may swap the kernel function.
  - A scratch that doesn't fit the planned block gives a declined row, and the learner retraces (A-land-merge2-3/4).
- **One variant per topology class**: node kinds, cluster dims, programmatic edges. A new topology is a new variant,
  with no node-enable padding (A-land-merge2-2).
- **Attention and RNG are harvest families on the cuBLAS seam**, not new providers (A-item5-1).
- **RNG bindings are structurally non-keyable.** Philox seed/offset slots and the generator increment only exist at
  capture, so an RNG call with a new key retraces (A-item5b-1).
- **Checkpointed dropout**: AOT's per-graph CUDA Generators are graph inputs. Each segment's graph registers them like
  the default generator, so a replay advances them as eager would (A-item5b-6).

**Deferred or out of scope**
- **No upstream fixes yet** (E-land-1). Local workarounds are kept and listed below for extraction.
- **cuBLAS capture-and-copy tier deferred** (E-land-2).
- **CuTe DSL** runs through a copy of its MLIR text plus our own evaluator, pinned to cutlass-dsl 4.6.2. The long-term
  goal is an upstream Triton-like launch-info hook in CuTe DSL (the strongest external signal; draft only when asked). CuTe is never treated as an opaque/harvested call (E-land-4); the regex front end is to be replaced by MLIR's own parser.
- **Possible explicit cuBLAS workspace API** (Elias offered, 2026-09-26): hand the call its workspace like an operand. That would
  replace the harvest's plug/clear/allocator-history dance for cuBLAS. Not needed now; an extraction-phase simplification.
- **TODO only**:
  - NCCL (T-nccl). The spike (land/NCCL_DESIGN.md, 2026-09-26) recommends harvesting each collective as a kernel node with two pointer slots instead of the XLA-style child-graph cache: an address-keyed cache misses on the compute-stream temporaries Inductor emits (0% hit in some TP patterns under eager-order allocation), while a shape key plus patching hits 100%. It is about 1k LOC, all local, and needs `NCCL_GRAPH_REGISTER=0` (a question for Elias);
  - a generic traced-library tier (T-traced-lib);
  - symmetric memory (T-symm).

## Big snags (root cause -> resolution)

1. **Wrong grads from baked static addresses** (REVIEW_22). Trust mode treated `static_input_idxs` as constants, but
   backward's saved activations, shared compiled graphs and `p.data = ...` all move them. Fixed with pointer rows at
   about 3.5 ns each.
2. **SIGABRT with dynamic shapes** (REVIEW_ITEM5 H1; also 5 W1 decode rows in the sweep).
   - Inductor's `assert_tensor_metadata` calls the C++ `assert_size_stride` directly, bypassing our namespace seam.
   - On the base commit that function has no `HANDLE_TH_ERRORS`, so the symbolic `numel()` throw terminates the
     process. Main has the handler (#197451).
   - Fixed: the seam maps it to a no-op under the trace (trusted layouts).
3. **The ~400 MiB extra memory was mostly misattributed** (REVIEW_ITEM5 M2).
   - 264 MiB was cudagraph trees' pool. The test helper enabled host trace only around the forward pass, so the lazily
     compiled backward went to trees.
   - The real 142 MiB was the shared harvest pool, kept alive by a keeper graph and by a refusal traceback.
   - Fixed with a per-harvest MemPool plus `graph.reset()` on refusal.
   - **Lesson**: always set the host-trace config globally when measuring.
4. **RNG rows in keyed tables would replay stale offsets** (REVIEW_ITEM5 M1). The generic row filler treats unknown
   slots as scratch. Fixed structurally (non-keyable).
5. **Upstream PyTorch bugs hit** (worked around, not fixed, per E-land-1):
   - the pinned host allocator's pool-reuse assert (fresh pool ids avoid it);
   - `memory_snapshot(pool)` decrefs pool ids;
   - the cuBLAS workspace is shared across graph captures;
   - `cudaStreamGetCaptureInfo` faults across threads.
6. **Stale host bytes in ATen kernel parameters.**
   - Where: the flash copy's OffsetCalculator unused dims, efficient attention's CUTLASS Params byte 224, and
     native_dropout's TensorInfo unused dims.
   - Root cause: the host constructors fill only the used entries (`i < dims`) and never write padding, so the
     unused bytes hold whatever earlier frames left on the stack, which differs per call.
   - Resolution (zero_init fork, P3): while the harvest captures, a thread-local flag (`c10::cuda::isHostTraceHarvesting`)
     makes the OffsetCalculator and TensorInfo constructors and the mem-eff backward's Params memset the struct
     first. The stale-bytes rule and its second verification are deleted: any unexplained varying word refuses.
7. **The cuDNN SDPA fake predicts an lse the kernel doesn't return.** This declined every Inductor graph using cuDNN
   attention. Fixed in the tape (A-item5-4); the upstream fake fix is deferred.
8. **Freed temporaries coalesce in the allocator**, so the harvest can't count blocks for flash/efficient backward. It
   now reads the call's allocations from the allocator history (A-item5b-4).
9. **Inductor reuses buffers as `out=`**, so a GEMM was keyed with 4 operands and refused: 9 keys in a flash training
   step (A-item5b-8).
10. **Tight-loop timings are floors.** The entry's cost roughly doubles after any C++ dispatch (cache and branch
    predictor contention); eager's `add_` goes 4.9 -> 7.9 us the same way. Report in-situ numbers.
11. **fp32 GEMMs were eager steps** until E-land-3. The p3 MLP went from 190 to 48 us per call.
12. **Transient harvest refusals under contention.** The harvest's allocation check is per process, so other threads
    in the same process or block coalescing can trip it. REVIEW_MERGE2 confirmed this is refusal-only: every accepted
    binding is verified bitwise against eager on fresh operands.
13. **Harvest budget cliff** (REVIEW_MERGE2 M2). A process-wide cap of 64 keys meant one 4-matmul graph over 50 sizes
    used it up, and every later key ran 159 us/call through Python + eager instead of 7.8. Now a knob, default 1024
    (A-merge3).
14. **The binding key used the current device, not the operands' device** (REVIEW_MERGE2 M1). A graph on a non-current
    device would never go native, and with two GPUs a row could get another device's kernel (the setter then fails
    every call). The key now carries the operand device (merge3).
15. **Flash backward refused: allocations "differed" between captures** (merge3). Blocks were matched across the four
    captures by address, and the allocator placed them in a different order in some captures. Now matched by allocation
    order (A-merge3-8).
16. **Eager's own flash backward dq is not bitwise reproducible** (fp32 atomics), so bitwise verification could never
    accept it. Outputs that differ between two eager runs are now checked within eager's own spread (P14, A-merge3-9).
17. **Replay memory above eager in training** (merge3). A run's temporaries were packed into one buffer that lives
    through the whole run, and every tensor of a step was allocated at the step's start. Tensors can't reuse
    temporaries' bytes, and one big buffer per batch size fragments the pool: checkpointed-dropout training peaked at
    1022 MiB reserved vs 658 eager and 504 compile (segments of 370/278/186 MiB, one per batch size). It only showed up
    once flash backward bound (snag 15-16) and the whole backward became one run. Fix (wt-land-mem, P22): the default
    "eager" mode allocates in tape order and frees each temporary after its last launch; ckpt peaks at 510 MiB
    (compile 504, trees 510). Flash stays at 564 vs compile 406: the backward is one graph, and compile frees saved
    activations at their last use inside it, and a replay frees the caller's inputs only between runs, not inside one (P27).

18. **CuTe DSL's value casters mutate the IR when you read it** (cute-ir spike). The DSL registers global MLIR value
    casters; reading an operand Value runs them, and the tensor caster emits a `cute.get_iter` op per read (a caster that
    raises aborts the process). `clone()` has the same types; bytecode fails to load outside the DSL's context. So the
    live module can't be walked. Resolution: print it with MLIR's printer and load it with MLIR's parser into a context
    we own, where cute types are opaque and no DSL caster matches; the walk never reads operand Values (A-cuteir-1).

19. **Two threads sharing a stream corrupted each other's tensors** (merge4q's flaky threaded test, 9-14 of 30 runs).
    "eager" replay memory frees a run's temporaries on the host before the graph is queued. Torch ops release the GIL,
    so another thread's allocation on the same stream could land in that window, take a freed temporary's block and
    queue its kernel before our graph, which then overwrote it. The entry mutex only serializes entry calls. Fix
    (A-race-1, P25): hold the caching allocator's device lock from the run's first allocation to its launch (new c10
    `lockDeviceAllocator`). 0 of 200 after, 43 of 60 on unfixed merge4 in the same session.

20. **A tensor held to its run's end sat under the run's peak** (tf_efficientnet_b0 inference, 1239 MB steady vs
    931 default compile, 854 eager). A tensor crossing steps (here a 308 MB cuDNN conv output, read by the next run's
    first kernel) is dropped when the run holding its last use is queued; that run then allocated 314 + 617 MB. Freeing
    it on the host mid-run like a temporary is unsafe: if a segment is released in the window (OOM retry), the
    run_buffer fallback can't recover a tensor's bytes. General, not effnet-specific. Fix (effnet_split.diff, P34):
    `split_runs` splits a run just before the peak allocation when that run drops tensors, and repeats while the
    simulated peak falls. effnet: one split, 931 MB, speed unchanged; training unchanged (no split lowers it).

21. **Explicit cluster (1,1,1) was dropped at variant build, so cuBLAS kernels trapped** (Albert, mobilevit_s training
    with the blas bucket on; illegal instruction, Xid 13). Blackwell nvjet kernels such as `*_1x1_*_splitK` are
    launched by cuBLAS with an explicit `CLUSTER_DIMENSION (1,1,1)`, and they trap without it. Host trace collapsed
    "absent" and "(1,1,1)" into (1,1,1), in `KernelNode.cluster`, the opaque `KernelLaunch` and the topology. Then
    `capture._launch` only set clusters that were not (1,1,1). Harvest verification launched with the node's attributes
    verbatim, so it passed. The driver gives no signal: `CLUSTER_SIZE_MUST_BE_SET` is 0 on these kernels. Fix
    (scratch/blas_crash/fixes/launch_attributes_vs_merge4q.diff, P35) is generic, with no cluster special case:
    - read every launch attribute ID off a node;
    - keep those that differ from a plain launch's captured node, verbatim;
    - reproduce them at variant build;
    - read them back in `_verify` and decline on a mismatch;
    - an attribute the code does not decode (access policy window, an unknown ID) is kept as its raw union bytes
      and launched verbatim, so it is a template constant; the key refuses only if it differs between the
      harvest's four captures, or the node is device-updatable (merge5c-2 refinement).
    The CuTe DSL's Blackwell GEMMs had the same latent drop.

22. **cuDNN conv kernels read as per-call state, or as a constant into a live segment** (conv fork, 4.2b). cuDNN builds
    its plan and heap parameter structs uncaptured on a key's first call and leaves fields of that path in them:
    - Capture A differed from B-D. Fix: one priming capture (A-conv-3).
    - dgrad frees its workspace before it allocates grad_weight, which then sat at the workspace's address in only
      some captures. Fix: each fresh output gets a hole of exactly its bytes (A-conv-4).
    - CUTLASS 3 wgrad and strided dgrad keep operand-dependent leftovers, and also a constant pointer into cuDNN's
      freed uncaptured workspace. That workspace is in a default-pool segment that is still cached, so it
      intermittently refused 2-10 of 83 unit keys and 3 of 5 backward keys of a small training net.
      `torch.cuda.graph` captures the same bytes. Fix: stale fields verified by a second launch with another capture's
      bytes (A-conv-6, P39). The constant first went to a 0xFF decoy that had to stay unread and unwritten. In
      timm training that hung tf_efficientnet_b0 in the check launch (GPU at 100%, a flag spin) and dm_nfnet_f0 hit
      an illegal address (Xid 31). A pointer that can't be proven unread can't be bound, so the key now refuses
      like any other op (A-conv-7, P40).
      convws: small shapes retargeted (unmapped, sentinel, 512 B guard) come out bitwise, but a zeroed 512 B guard
      slot hung dm_nfnet_f0 training in a stride-2 f16 wgrad's check launch (GPU 100%): the field is live for
      some shapes. Backed out; still refusing (A-convws-1/2). Binding it needs cuDNN's semantics for it via an
      ATen workspace-recording hook (A-convws-5); torch.cuda.graph bakes the same stale pointer (A-convws-6).
      Resolved by convws2 (snag 28): the constant is kept, P40 now binds.
    - The grouped and FFT paths capture fork/join graphs. Fix: topological linearization (A-conv-5).
    - On wt-land-mem the explicit cluster (1,1,1) trap of snag 21 hit cuDNN's sm100 CUTLASS 3 convs too (fp32
      channels_last dilated). Covered by P35 on merge5c. Only conv_mem.diff/conv.diff carry the interim fix (A-conv-8).

23. **Eager-order replay memory costs 300-400 us of host time per call on some variants** (merge5c-3 sweep;
    Blenderbot and MT5 inference). Blenderbot h/d fell from 3.19 (merge2) to 2.45, with host time 205 -> 656 us per
    call, all of it inside the native `__call__`.
    - Root cause (replaymem fork, A-replaymem-1): the cost is node repatching, not the allocator calls.
      - In eager order, temporaries are freed mid-run and later requests retake their blocks. The benchmark's
        interleaved eager calls, whose batch size varies, reshape the pool between replays.
      - So the replay's addresses move from call to call. Blenderbot patches 388/388/241 of its 388 nodes per call and
        MT5 321, at about 1 us per `cuGraphExecKernelNodeSetParams`.
      - A run buffer comes back at the same address: 3 and 40 nodes.
      - The allocate loop itself costs tens of us: eager-order calls that patched nothing took 173-202 us, against
        run_buffer's 164.
      - Pool-state dependent: XGLM has the same structure and patches 3 nodes in either mode.
    - Resolution: P41, `replay_memory="auto"`, is Inductor's default. Run_buffer is taken per variant when its simulated
      peak, and each run buffer, is within max(64 MiB, 5%) of eager order's peak.
      - Blenderbot: 3.02/2.68 vs 2.37/2.73 h/d, host 268-286 vs 496-1069 us, +14 MB.
      - MT5: 3.64/3.61 vs 3.79/2.89, host 275 vs 590 us, +36 MB.
      - effnet keeps split_runs' memory. ckpt training stays eager order because of the buffer bound: 524 vs 1022 MiB
        reserved with a pure peak rule.
    - XLNet repatches about 390-411 nodes per call in either mode (about 750 us). Repatch fork (A-repatch-1..4):
      - Every moved field is an allocation address. The pool's state changes between calls, from the harness's eager
        call and from simply holding the previous output, so patching is correct. Pinning and hold mode are ruled out.
      - A moved node costs about 1 us: 0.58 us for the setter plus 0.39 us inside the next launch. cuGraphExecUpdate
        is worse.
      - Device-side batched patching (P46) went in: a head kernel node per segment with >= 16 eligible nodes sets the
        moved fields of device-updatable nodes in-graph, and a call sets only the head's parameters.
      - XLNet 630 -> 587 us held and 266 -> 215 us dropped; Blenderbot (eager) 489 -> 452 and 206 -> 195. Headline
        neutral.
    - Corrected (gemmeager, A-gemmeager-1): the 169 runs and 193-241 eager GEMM steps belong to the learning variant.
      It ran one call and is kept only to learn new cuBLAS keys; the base never serves it. The served variant has 0 eager
      steps in one segment: XLNet 436 nodes in eager order, Blenderbot 388 in run_buffer. The sweep counted every
      variant; it now counts learners apart.
      - P46 gained little because it excludes keyed-site nodes, and every harvested cuBLAS node is one.
      - Still open: XLNet's churn (eager order; auto's run buffer is over the margin). It is off the wall-time path:
        XLNet is GPU-bound at about 8.2 ms against about 0.75-1.3 ms of host time.

24. **Cross-snapshot h/d comparisons drift with the default arm** (merge5c-3 sweep). Rerunning default compile on the
    same box a day later made it 12-26% faster on XLNet (15.9 -> 12.5 ms), XGLM (12.0 -> 9.4), swin (10.1 -> 8.55),
    effnet (6.22 -> 4.60), inception (7.76 -> 6.66) and TB convnets, with host_trace latency unchanged. So a snapshot's
    h/d is only comparable with a trees/d measured in the same window. The merge5c sweep reran the trees perf arm for
    that reason; the headline uses same-run default for both.
25. **An internal error in capture became a silent decline** (merge6a). conv_vs_quality used `heapq` in
    _host_trace_capture.py without importing it. Capture catches exceptions and declines, so every capture raised
    NameError and fell back to eager with only a "(declined)" log line; the symptoms were unrelated-looking suite
    failures and input_lifetime at 160 MiB. Only `Declined` and CUDA errors are expected there; a NameError or
    AttributeError is a bug and should raise under tests. Resolved in merge6b (A-merge6b-3): `torch.cuda._host_trace.raise_unexpected`,
    set in all 13 suites' setUpModule, re-raises instead of declining in the trace, replay miss, launch hooks, CuTe
    description and harvest capture; `Declined`, OOM and CUDA capture errors still decline. Off by default.
26. **A user TorchDispatchMode saw the trace, and a replay skipped it** (warmup fork). `test_list_clearing` and
    `test_graph_partition_refcount` call the graph under a mode that asserts, at `aten.mm.out`, that the caller's
    inputs are dead. The mode fired inside the trace's symbolic run (the tape re-dispatches each op through the modes
    below it) while `_miss`/`trace` still held the arguments, not in the warm-up. The inputs' lifetime was only the
    symptom: a trace under a mode records what the mode does, and a native hit under a mode runs nothing through it
    (the new test's mode saw only its own `clone`). Resolved (P47, A-warmup-1): under any dispatch mode
    `_Installation.__call__` runs the graph's own `call` on the caller's list, which is eager Inductor, list clearing
    included. Second snag inside it: AOT pushes `_AnalyzeCustomOpInputOutputMode` (not an infra mode) around every
    compiled fn's first call, so a plain stack check sent every first call eager; that mode is exempt by exact type. The trace call still holds its arguments to its end (P27; transient, like its harvest).
27. **A refused conv key kept the served variant learning, on the Python path** (convinc; sweep6 inception_v3 inference
    at 0.76x default). P40 refuses a CUTLASS 3 fprop (relu_valpha, conv3d weight_stationary) key when a constant plan
    field (a segment base +0/+1/+2/+0x40, the same in every capture) falls in a still-cached segment, so the count is
    allocator-state dependent: 23 of 43 unique keys in the sweep row, 0-3 in reruns (repvgg_a2 3 of 16). A refused key
    stayed an OpaqueCall in the retrace, so the served variant still learned: 54 conv sites as eager steps through
    `_call_slow`, 1836 Python calls, 55 segments. Not a retry loop: 43 harvests for 43 keys (94 sites), `refused` is
    final, so P2 held; "learns" counts eager opaque steps, not harvests. Invisible: the runner's `declined_keys` counts
    declined trace classes, and a refusal was not logged. Resolved (A-convinc-1, P48): `OpaqueProvider.refusal(key)`;
    a refused, budget-spent (P1) or unrecordable key traces as a plain eager step with its key guarded, and a learner
    retraces once when a key binds or is refused, so the served variant stops learning and runs natively. Refusals log
    once per key (`log.info`, trace artifact `host_trace_harvest_refused`).
28. **The conv guard hung nfnet because a "stale pointer" was half live** (convws2; P40's refusals). A deterministic
    mutation probe (harvest, rewrite one qword of one kernel's image, launch vs eager bitwise, compute-sanitizer) on
    the 30 dm_nfnet_f0 backward configs, the named small keys and 6 inception relu_valpha fprops: the refused
    "constants in a segment" are plan fields cuDNN's first, uncaptured execute wrote and captured executes never
    rewrite (stable across harvests, deaf to stack smear and MALLOC_PERTURB_). Two kinds:
    - full pointers into that execute's freed workspace + an offset (stream_k wgrad byte 32 = old workspace +
      791040 = its old reduction workspace; dgrad 40; strided dgrad 584/616; relu_valpha 1240): 0, 0xA5..,
      an unmapped 0x7f0000000000 or a live buffer all come out bitwise, memcheck clean, buffer unwritten: unread;
    - an int32 field + 4 B padding whose padding holds a leftover VA high half: stream-K grouped wgrad byte 2008
      (CUTLASS `reduction_mode_`, low 2 -> 0 or 7 hangs), conv3d dgrad 968 (low 8 -> 0/1 wrong output). The high
      half is unread. The guard rebound the whole qword as a pointer and clobbered the live int: the nfnet hang.
    Eager has no use-after-free, and torch.cuda.graph bakes the same bytes, so there is no trees hazard. Fix: keep
    A's bytes (P40). Also, stream-K wgrad byte 512 (unread) varies only in its high half, which P39 refused as "host
    state" when one capture's value was an address; the low halves agreeing now make it lenient (A-convws2-3).

29. **The sweep's boundary count read the wrong variant** (sweep6; after gemmeager's learner split). The runner summed
    every variant's eager steps, so a family's one-call learning variant (its GEMMs still opaque) read as 241/193
    "boundaries" on XLNet/Blenderbot (A-gemmeager-1). gemmeager's fix splits by `learns`, which is still wrong two ways:
    a first variant that still learns and serves every call (inception_v3's 54 refused convs, snag 27;
    nvidia_deeprecommender) goes to "learner", and an instance called once (GPT2ForSequenceClassification's second)
    keeps only its learner, which is not served. Resolved in the sweep runner copy (merge6b1/runner/sweep_run.py):
    `served_boundaries` = eager steps of each family's first variant (the one a call tries first; the harvest
    inserts retraces before the learner) over instances with replays > 0; learner boundaries are reported apart.
    Accuracy rows run about one replay, so a learner legitimately serves there; accuracy coverage counts declines
    only. Arms still run interleaved per model in one session (snag 24).

30. **A padded out= refused the vocabulary projection's addmm** (addmm fork; merge6b2 sweep, 1 served boundary on
    torchbench BERT_pytorch and nvidia_deeprecommender). Inductor pads an output's row stride when N is odd or
    unaligned (20008, 197951), and `_harvest` refused any non-contiguous out= ("the output is not a fresh contiguous
    tensor"). The arena already binds operands at any strides and alignment class, so the out is now an operand like
    the inputs (A-addmm-1). The padded out then sends ATen's addmm down its non-Lt path: a bias `copy_` through the
    legacy `gpu_kernel_impl_nocast` elementwise kernel, then a beta=1 GEMM. That kernel's `__host__ __device__`
    lambda parameter ends in nvcc's pointer to the lambda's host heap copy, which the device never reads. It sat at
    another address in capture A than in B-D, so the unit key refused as an unexplained varying parameter. An eager
    warm-up call does not move it; a capture does. Fix: the priming capture of snag 22 now runs for every op
    (A-addmm-2, P49), about 7% more harvest time in the unit suite.

31. **An SE block's 1x1 conv backward refused on a memcpy node** (convrest; tf_efficientnet_b0 training, 14 of 92 conv
    keys). `convolution_backward` makes grad_output channels-last-contiguous. An SE conv's grad_output (N, C, 1, 1)
    has strides that only look channels-last, so ATen copies it (`cudaMemcpyAsync` D2D) into scratch inside the call.
    Native replay knows only kernel and memset records, so `graph_nodes` now turns a 1-D D2D memcpy node into a launch
    of an NVRTC copy kernel (`host_trace_memcpy`), classified and patched like any other kernel (P50). Two follow-ons
    surfaced once the key bound:
    - The copy's scratch (1792 B) took grad_weight's exact-size hole, so grad_weight landed in freed workspace
      ("allocations not its blocks"). The holes are now sized from the previous capture's allocations; the priming
      capture sizes them and is not checked.
    - With every backward conv bound, the backward became one graph, and it held the backward's `static_input_idxs`
      (its saved activations) to the run's end: peak 1219 vs 943 MiB for default compile. A backward's static
      inputs are saved activations that AOT autograd hands over, so Inductor now frees every backward input
      (P51). split_runs then splits the run (17 graphs), and the peak is 930 MiB (eager 1476).

32. **CUTLASS 3 2sm dgrad byte 512 refused as host state** (convrest; inception_v3 8, resnet50 2, repvgg_a2 1 keys, in
    bf16 and fp16). The qword is a CUDA address that is the same in captures A, B and D and different only in C (the
    other stream). It points into a torch segment (a 2 MiB small segment, or an expandable segment): an old workspace
    from that stream's own uncaptured execute. A mutation probe over all 11 keys (0, 0x3deadbee0, 0xffffffff, 1<<40,
    a guard buffer that was never written) came out bitwise every time: the field is unread. P39 now lets through a
    value that varies with the stream alone, kept as capture A's; the check launch with C's bytes is still bitwise.

33. **A PDL edge in a fork or join refused a grouped conv backward; binding it serialized cuDNN's side streams**
    (convrest; dm_nfnet_f0). The first fix did what was asked: keep PROGRAMMATIC_STREAM_SERIALIZATION only when the
    node's sole predecessor is the chain's previous node, and make every other programmatic edge a full edge. The key
    bound 60/60 and was bitwise, but nfnet went from 34 to 63 ms/step. The (768, 128/group) backward's graph forks
    into 6 branches, one per group (memset -> `wgrad_alg1_engine` -> convertTensor, each on a cuDNN side stream), and
    the chain ran them one after another: 3.4 vs about 1 ms per call, 12 calls a step. Base refused that key, so eager
    ran it with its concurrency. Three other nfnet keys have the same shape and base bound them serialized, and so do
    fp32's per-group and FFT paths. `graph_nodes` now refuses kernels on concurrent branches (memsets beside a kernel
    are still serialized): nfnet 34.1 ms/step, peak 2021 MiB, 4 keys eager (P52). The cost is on the other side for
    small branches: one ghostnet_100 depthwise key (8 groups at 112x112) now refuses too, and ghostnet runs about
    1 ms/step slower than with it serialized (12.1-14.3 vs 11.4-12.5, on a noisy host). Capturing them needs fork/join
    records in the tape and in native replay (Open).
34. **Bert training at B >= 14 had 52 `aten.sum.dim_IntList` eager boundaries** (sumdim; dynbatch; static B=16 too).
    Inductor's `MixOrderReduction.can_fuse` fuses LayerNorm backward's row and column reductions when
    nrow*ncol >= 5 Mi (guard `393216*B >= 5242880`). The wrapper then finishes each column partial with
    `ws[a:b].view(nsplit, rnumel).sum(dim=0)`, an ATen call that no harvest family took: 26 LayerNorms x (weight,
    bias). A "reduce" family ({sum.dim_IntList}) harvests it. `reduce_kernel`'s ReduceOp is not zeroed under the
    harvest flag, so its empty functor, padding and the gap before `src` hold TensorIterator's stack bytes: they
    varied with the operand set (byte 0) or, in one run, stayed constant but fell inside an allocator segment (byte
    976). Reduce takes conv's two stale-byte rules (P53). Static B=16: 77 -> 2 segments, 19.6 -> 17.7 ms. Removing
    the boundaries raised peak by 1.4 GiB until trainmem's plan fix (saved activations counted as freed; 4135 vs
    4046 MiB), because one run now spans the whole backward.
35. **timm bf16 training peaked above eager** (trainmem; convws2 measured dm_nfnet_f0 2588 vs eager 2214 MiB, and
    repvgg_a2, resnet50 and ghostnet_100 were also over). There were three causes, and all three hold arguments
    longer than eager does:
    - The Python slow path holds the call's args tuple. `_call_slow` -> `_miss` -> `call_py` never owns it, so no
      argument is dropped at its last step; the C++ fast path does this with an owned tuple (Entry.cpp
      `frame.owned`). Before convinc, a refused conv key kept the backward a learner, so every step was slow: this
      was the reported gap. After convinc it is the first call of each retraced variant: +581 MiB on nfnet at step
      2 (same allocator events as step 3, but the saved-activation frees come at the end). Fix: HostTraceReplay's
      `handback`. `_miss` returns `Handback(args)` in place of the first call; `_Installation` drops its references
      and calls again through `call_boxed`.
    - The backward's planner didn't see its saved activations. Their `static_input_idxs` are all saved tensors
      (address-stable under trees), so `freed_arguments` excluded them. The runtime already drops them, but
      `split_runs` never split for them (nfnet bwd 2133 vs default 2028). Fix: a backward holds only its
      `primals_*` inputs; parameters stay held by the module. convrest's P51 frees every input, which counts the
      parameters as freed too.
    - Open: a learner's call (warm-up step 1) still holds its args, which needs an owned `call_py` (C++). Each
      harvest's arena (two operand sets plus capture outputs) adds a transient on top, which is released: warm-up
      max nfnet 3264, resnet50 2142. Neither is in the harness's warm peak.

36. **Memcpy capture operands: placeholders (int4c) -> a 512 B scratch (int5a) -> a sized scratch (int5b).**
    - Why int5a moved off int4c's operands. int4c captured each d2d memcpy at `slot_address`: the real address for an
      argument slot, the trace's placeholder base for an allocation or eager output. Red-team S1-1 (FINDINGS.txt,
      repro `attacks/a1c_expandable_min.py` and `a1b_alloc_expandable.py`, both with
      `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) observed every replay holding a memcpy raise "invalid
      argument" ("Replacement operand type is incompatible with existing operand" from
      cuGraphExecMemcpyNodeSetParams). The entry stayed at traces=1 replays=0 and raised on every call; without
      expandable segments the same attacks passed.
    - Root cause (int5a probe `cov/int5a/exp/memcpy/probe.py`, rerun on GPU 1 as `cov/int5b/exp/memcpy/probe_*.log`):
      a memcpy node keeps the memory kind it was captured on, and SetParams1D refuses the other kind. Under the
      default allocator every capture kind can be set to both cudaMalloc and allocator memory. Under expandable
      segments a capture on a placeholder or on cudaMalloc memory refuses allocator (cuMemMap) operands, and a capture
      on allocator memory refuses cudaMalloc operands.
    - The int5a change (not a commit; part of int5a.diff, built as b_int5a4, PROGRESS 2026-09-30T13:11:16Z): capture
      inside one caching-allocator block, `torch.empty(512)`, with dst at +0 and src at +256. Red-team re-verified
      a1b 160/0 and a1c 20/20. The residual is loud: an operand of the other kind raises at replay.
    - The int5a regression. The driver checks the copy against the cudaMalloc allocation the address falls in, not
      the 512 B tensor. A copy passes while `scratch + 256 + nbytes` stays inside the segment (2 MB small pool), so
      257 B and 4 KB copies capture on int5a (measured), and a 64 MB one fails with cudaErrorInvalidValue. That
      failure was the call's, so every call of the class ran eagerly. Bert, DistilBert, Albert, Roberta and OPT all
      ran at replays=0 (cov/int5b/hf_smoke_int5a.out). The gates missed it because no gate copy ran past the segment.
    - int5b (the A/B agent's memcpy_span.diff): the scratch is `2 * span`, where span is the segment's largest
      copy rounded up to 256, with dst at +0 and src at +span. Peak capture-time memory is 2x the largest copy; the
      scratch is freed when `_segment` returns.
    - Lifetime. The node holds freed scratch addresses until it is set. Nothing executes it before then:
      instantiate and cudaGraphUpload do not run the graph. `patch_and_replay` (Patch.cpp) sets every memcpy record
      with cudaGraphExecMemcpyNodeSetParams1D on every replay, the first replay and every clone form included,
      before `replay_exec` or the Python `graph.replay()`. These are the only launches of a captured segment.
    - Tests: `test_a_device_copy_of_any_size_replays` (257 B, 4 KB and 64 MB, bitwise; on int5a the 64 MB case
      fails), plus the HF smoke gate `cov/int5b/hf_smoke.sh` (5/5 ok on int5b, 0/5 replays on int5a).
37. **A segment whose capture fails is local** (int5b). Before int5b, any error in `capture_tape` aborted the
    call's `_build`: the call ran eagerly, and its class (structural for a Declined, exact otherwise) was never
    traced again. Now `_segment` raises `SegmentFailed(seqs)`:
    - a capture-time error names the launch it failed at;
    - a verify, instantiate or upload error names all of the segment's launches.
    `HostTraceReplay._call` maps those launches to their top-level ops and traces the call again, without a
    warm-up, with those ops as eager calls (`trace(eager_ops=...)`; a Triton launch becomes its eager Triton
    call). The rest replays. The fallback stays loud: under `raise_unexpected` (the suites, SWEEP_HT_RAISE=1) a
    non-Declined error raises as itself; an OOM or interrupt is always the call's; each failure is noted in
    `declines` with its reason. An op the trace cannot run eagerly (a CuTe launch), or a second failure on an op
    already eager, declines the call as before.

## Policy decisions for Elias to review

These are agent-made (A) choices with user-visible behavior. Each is a knob or rule someone could reasonably want
different.

| # | Policy | Current choice | Alternative | Ref |
|---|---|---|---|---|
| P1 | Harvest budget: how many new cuBLAS/attention shape keys to learn per process | 1024 (a knob), then new keys run as eager calls | unlimited; per site; per graph | A-merge3 (REVIEW_MERGE2 M2) |
| P2 | Transient refusals: how often to retry a key that failed for timing/allocator reasons | 3 transient refusals, then refused for the process | retry forever; back off | harvest `transient` |
| P3 | Stale host bytes in kernel params | POLICY: a thread-local harvest flag (`c10::cuda::setHostTraceHarvesting`, set by the harvest around each capture's call through `torch._C._cuda_hostTraceSetHarvesting`). Three ATen sites read it and memset the struct before filling it: the `OffsetCalculator` and `TensorInfo` constructors, and the mem-eff attention backward's `Params` (padding at byte 224). No tolerance rule remains: any unexplained varying word refuses. Eager cost: about 2 ns per struct for the TLS read; an unconditional memset would be about 5 ns (216-504 B); both are in the noise of a 10-38 us launch. Gated per Elias's ask | unconditional memset (about 5 ns per struct, no flag); per-struct layout masks in the harvest; the old `_STALE_KERNELS` allowlist plus a second verification | A-zeroinit-1..3 |
| P4 | New topology (a different kernel count or cluster shape for a shape key) | a new variant (another graph) | pad with disabled nodes into one graph | A-land-merge2-2 |
| P5 | Scratch too big for the planned block | declined row, and the learner retraces | grow the block in place | A-land-merge2-4 |
| P6 | RNG calls with a new shape | retrace (never keyed) | key with the offsets re-derived | A-item5b-1 |
| P7 | Learner variants (still containing eager library calls) | stay in Python, not in the C++ entry | register them too | A-land-merge2-5 |
| P8 | Guards that pin an address (`int(data_ptr())`) | decline the graph | trust | merge3 (REVIEW_MERGE2 M3) |
| P9 | cuDNN SDPA fake returning an lse the kernel doesn't | patched in the tape, always on | fix the fake upstream | A-item5-4 |
| P10 | Default of `triton.cudagraph_host_trace_blas` | on whenever host trace is on | opt-in | A-land-merge1-1 |
| P11 | Harvest budget knob name/default | `triton.cudagraph_host_trace_harvest_budget` = 1024 | - | A-merge3-3 |
| P12 | `int(data_ptr())` / guards over traced addresses | decline, per exact class | trust | A-merge3-4 |
| P13 | cuBLAS workspace cache on (`TORCH_CUBLAS_WORKSPACE_CACHE=1`) | supported: the stream's cached workspaces are cleared before and after each harvest capture | refuse (old behavior) | A-merge3-6 |
| P14 | Verifying outputs eager itself can't reproduce (flash bwd dq, atomics) | outputs eager reproduces: bitwise; outputs that differ between two eager runs: within 4x eager's own run-to-run diff plus the default dtype tolerance | require `use_deterministic_algorithms`; refuse | A-merge3 (step 8) |
| P15 | `_check_trace_values` finds a trace value that disagrees with the live one (our bug, not the user's) | fall back to eager, but log it: warn once plus a `trace_structured` artifact; raise under the test suite | always raise; silent fallback (old) | QUALITY_CPP_REPLAY POLICY; implemented A-merge4q-4 (flag `_host_trace_capture.raise_trace_disagreements`) |
| P16 | TMA driver entries (`cuTensorMapEncodeTiled`/`ReplaceAddress`) missing on an older driver | OPTIONAL driver entries: TMA kernels decline, everything else works | hard failure | QUALITY_CPP_REPLAY POLICY; implemented A-merge4q-5 (NotImplementedError -> declined) |
| P17 | `_size_hint_pick` (~140 lines re-deriving Inductor's size-hint multi-kernel heuristic) | drop it: guard the exact sizes (`exact()`, one variant per size). Only graphs compiled with the opt-in `multi_kernel_hints` (default `[]`) are affected. Elias questioned the need (2026-09-26); revisit only if a `multi_kernel_hints` user needs it | interval guards as a follow-up PR | QUALITY_TAPE_INDUCTOR item 1 |
| P18 | Harvest budget scope | one global knob read at harvest time | per-installation (the old behavior: the last compile won) | QUALITY_TAPE_INDUCTOR item 12 |
| P19 | CuTe witness (a second, real capture of each CuTe launch at trace time, compared byte-for-byte with our evaluator) | keep it in the product path until CuTe DSL has a launch-info hook: it is the only check that our MLIR evaluator matches the real bytes, and it costs trace time only | debug knob / tests only (-80 LOC) | QUALITY_HARVEST_CUTE #3 |
| P20 | Harvest while the process already records allocator history | read the process's own trace, filtered by the per-harvest pool (one path; drops the 511-mod-512 plug trick) | refuse transiently while history is on | QUALITY_HARVEST_CUTE #4 |
| P21 | Fused `addmm` restrictions (1-D contiguous bias, alpha = beta = 1) | keep, with a one-line reason; other addmm forms run as eager steps | lift them | QUALITY_HARVEST_CUTE #14 |
| P22 | Replay memory strategy | two modes decided once per variant: "eager" (allocate/free in eager order; HostTraceReplay default; matches eager memory, one allocator call per allocation) and "run_buffer" (one packed buffer per run; fewer allocator calls, more memory); Inductor picks per variant with "auto" (P41). Elias (2026-09-26): replay likely wants modes that trade memory for CPU | one mode only | wt-land-mem |
| P23 | CuTe IR transport | MLIR printer -> MLIR parser into our own Context, then an IR walk plus a generic interpreter; cute type parameters still read from opaque type strings (no bindings for them in our context). Unsupported host op or unmodeled type -> eager call naming it | walk the live module (unsafe, snag 18); upstream launch-info hook or cute type bindings | A-cuteir-1/2/4 |
| P24 | "eager" replay memory when a segment is released mid-allocation (OOM retry, GC threshold, unmap) or the launch is observed | put the run's temporaries in a run buffer for that call (c10 release counter read before/after the sequence) | retry the call eagerly | A-mem-3/4 |
| P25 | Other threads during a run's allocation sequence | hold the device's caching-allocator lock (recursive; new c10 `lockDeviceAllocator`) from the run's first allocation to its launch; other threads' allocator calls wait a few microseconds. Was: assume no other thread allocates (snag 19) | detect and fall back (window stays open); private-stream temporaries (memory above eager) | A-mem-5, A-race-1/2 |
| P26 | Allocator backend not native (cudaMallocAsync) | plan "run_buffer" regardless of the knob | a cudaMallocAsync release signal | A-mem-6 |
| P27 | Freeing the caller's inputs at their last use | merge6 (A-v2fix-2): when the caller hands its references over (`HostTraceReplay.call_boxed(list)`, which Inductor's boxed call uses), a native hit drops each argument no output is or views after the step of its last use (`StepMemory.arguments`, C++ only, no cost measured on the hit path). merge6a (A-merge6-4): split_runs also splits a run where a freed argument (Inductor: a graph input not in `static_input_idxs`; `HostTraceReplay(freed_arguments=)`, default none) dies before the peak, and "auto"'s run_buffer takes those splits if they lower its own peak (input_lifetime.py steady state 128 -> 64 MiB, eager's). The warm-up/trace call and Python-path calls (learning variants) still hold every input to the end. So `test_list_clearing` / `test_graph_partition_refcount` still fail: they check at the first call's `mm.out`, which is the warm-up | a warm-up-only first call for owned inputs (the trace call still holds them); place later allocations inside dying inputs | A-mem-7, A-merge4-2, A-v2fix-2 |
| P28 | CuTe object in a disk cache without its `.cute_host` (compiled with host trace off; the flag is not in the cache key) | under host trace, a cache miss: recompile once in-process, which writes both | load it, and its calls become EagerCalls | A-cuteaot-3 |
| P29 | SymInt hashing (a compile cache keyed by sizes, e.g. quack's `jit_cache` on `x.size(-1)`) | per-ShapeEnv opt-in `ShapeEnv.hash_symints_by_value` (default False; `_TraceShapeEnv` sets it): `hash()` is `int()`, a guard on the value. Core `torch.SymInt.__hash__` changes by 2 lines. Under our trace, einops's TypeError path becomes a specialization: correct, more guards | vendored `int()` in jit_cache; global SymInt hashing | A-cuteaot-5 |
| P30 | Arming the CuTe hook in compile workers | the parent's `triton.cudagraph_host_trace` goes to workers as `TORCHINDUCTOR_HOST_TRACE_CUTE_WORKER=1`; `_compile_nvgemm` calls `install()` before compiling | always install in workers | A-cuteaot-4 |
| P31 | Vendored quack edit | `cache/jit.py` renames `<tmp>.cute_host` with the `.o` (3 lines, marked `# torch:`) | no sidecar for quack's cache (warm quack calls become EagerCalls) | A-cuteaot-2 |
| P32 | Inductor graph with no tensor inputs and no CUDA constants (e.g. HF's causal mask built from `torch.full` after a graph break) | not installed: runs uncaptured, not counted in `cudagraph_host_trace_declined` (`_Uncaptured`) | install and decline on every call (the pre-merge5b behavior: a trace has nothing to key on) | A-merge5b-2 |
| P33 | A trace with no traced launches (every call an EagerCall, e.g. a lone cuBLAS mm with nothing to harvest) | counted in `replay.uncaptured`, logged at debug, not a decline and not in the declined counter | count it as a decline (the pre-merge5b behavior) | A-merge5b-3 |
| P34 | A cross-step tensor dead mid-run while the run reaches the replay's peak ("eager" memory) | `split_runs` at plan time: split the run before the peak allocation, repeated while the simulated peak (trace-time sizes) falls; one extra graph launch per split per replay, decided once per variant (snag 20). merge6a: freed arguments count too (P27), never inside a keyed site | free it on the host mid-run like a temporary (unsafe on a segment release); accept the gap | A-effnet-1/2 |
| P35 | Kernel launch attributes of a harvested or CuTe kernel (all IDs, not just cluster/PDL) | reproduced verbatim where they differ from a plain launch's captured node; so COOPERATIVE / PRIORITY / MEM_SYNC_DOMAIN bindings, which used to decline, are now traced; an attribute the code does not decode (access policy window, an unknown ID) is a template constant: kept as its raw CUlaunchAttributeValue bytes, launched verbatim, part of the topology key; the key refuses only if it differs between the four captures (an exec's attributes cannot be patched per call: cuGraphExecKernelNodeSetParams carries none, only a whole-graph cuGraphExecUpdate could change them) or the node is device-updatable (its kernel holds a handle to its node in the library's exec); `_verify` declines a variant node whose attributes differ (snag 21) | only cluster + PDL recordable (the old rule, with the absent/(1,1,1) bug fixed); or refuse every attribute the code does not decode (the first blas-crash patch) | blas-crash, merge5c-2 |
| P36 | Host-trace declines under `cudagraph_or_error`, and the `cudagraph_skips` counter | merge6 (A-v2fix-3): a graph that never captures is a skip, as in trees: `log_cudagraph_skip_and_bump_counter` (bumps `cudagraph_skips`, raises RuntimeError under `cudagraph_or_error`). There are two such points: the policy does not install (cache entry without trusted inputs, `_NotInstalled`), and a trace declines structurally (`replay.structural`: under trust that decline is the graph's). Per-call eager fallbacks (max_variants, an empty tensor, an active capture, a Triton untuned retry, the call's own OOM) and P32/P33 (nothing to capture) are not skips | only the policy's refusal counts (declines found at the first call stay silent) | V2, A-v2fix-3 |
| P37 | Default for convolutions (cuDNN forward and backward harvested as keyed tables) | merge6a: "conv" is in the default `triton.cudagraph_host_trace_harvest` (P45), so on whenever host trace is on, like blas and attention | opt-in | A-conv-1, A-merge6-2 |
| P38 | cuDNN global settings (`enabled`, `benchmark`, `deterministic`, `allow_tf32`, fp32 precision) | not in the key: assumed unchanged between harvest and replay, like cuBLAS's (E-land-1/3) | key them (4 bools and a string) | A-conv-2 |
| P39 | Stale bytes in a convolution's kernels (cuDNN's heap structs, which ATen cannot zero) | lenient for any kernel of `_CONV` unless a capture's value is a CUDA address and the low halves differ (a varying high half alone is padding over an int32, e.g. stream-K wgrad byte 512). convrest: a value the same in A, B and D (differing only on the other stream) is lenient too, e.g. an old workspace address of CUTLASS 3 2sm dgrad byte 512 left by the stream's uncaptured execute (snag 32). Verified by a second launch with another capture's bytes | refuse the key (leaves an eager step per such conv) | A-conv-6, A-convws2-3, A-convrest-2 |
| P40 | A constant pointer into a live segment in a convolution's kernel (a cuDNN struct field pointing into the first uncaptured call's freed workspace) | keep capture A's bytes for `_CONV` (what a graph replay bakes); other ops still refuse. The fields are plan fields cuDNN's first uncaptured execute wrote and captured executes never rewrite: unread full pointers, or a stale high half over a live int32 (stream-K reduction mode, conv3d dgrad) that keeping preserves. The check launch still runs bitwise vs eager | refuse the key (eager step per wgrad/strided-dgrad/relu_valpha key, A-conv-7); a 0xFF decoy (hung/faulted, A-conv-7); a zeroed 512 B guard slot (clobbered the live low int32: hung nfnet, A-convws-2); a fresh plan per harvest or cuDNN's native graph API (unneeded: nothing to track; the native API is the fallback, E-cudnn-graph-dep) | A-convws2-1/2 |
| P41 | Replay memory default (P22's modes) [REVIEWED: accepted, E-p41-ok] | "auto", decided per variant at build time at the traced sizes. It takes run_buffer if its simulated peak is at most max(64 MiB, 5% of eager order's peak) above eager order's, and every run buffer is at most that margin; otherwise it takes eager order with split_runs. Why: eager order's addresses move with the pool's state and repatch most nodes (snag 23), while the memory saved is often about 15-40 MB. The buffer bound exists because the peak alone missed ckpt training: run buffers of 184-368 MiB, one per shape variant, doubled peak reserved (510 -> 1022 MiB). Explicit "eager"/"run_buffer" force a mode; HostTraceReplay's own default stays "eager" | eager order always (merge4-5c); run_buffer always; a pure peak margin (ckpt +512 MiB reserved); min(64 MiB, 5%) | A-replaymem-1/2 |
| P42 | Efficient attention's dropout seed/offset device outside a capture (ATen change, scratch/hffall/effattn_seed.patch, built in merge6b, A-merge6b-1) | `_efficient_attention_forward` / `_scaled_dot_product_efficient_attention` always return 0-dim int64 seed/offset on the query's CUDA device (two scalar fills per dropout forward), as flash's rng_state and the meta/fake kernel already do and as ROCm already does uncaptured; the backward takes the pointer form for device seeds and the value form for CPU ones. Under torch.cuda.graph / trees capture the seeds were already on the device, so graphs see no change | keep CPU seeds uncaptured and teach host trace a CPU-output exception for this op (a device-vs-fake mismatch special case) | A-hffall-1 |
| P43 | CPU work inside a traced graph (HF LayerDrop's `torch.rand([])`, XLNet's CPU positional embedding) | host steps: `_empty_strided_cpu` buffers are trace-owned; `cpp_fused*` kernels and CPU aten ops over them are replayed every call, first and in host order (same CPU RNG draws and same values as Inductor without graphs; no segment split). The H2D `copy_` from a host buffer is an eager boundary | a device-side memcpy node from a pinned host buffer, or folding host steps whose inputs are constant (skips the recompute but changes RNG draw order) | A-hffall-2 |
| P44 | Pinned CPU buffers (`_empty_strided_cpu_pinned`) and a device op writing a host buffer (D2H) | decline | order the non_blocking copy against the next replay's host step | A-hffall-2 |
| P45 | Which op families host trace harvests | one knob, `triton.cudagraph_host_trace_harvest: tuple[str, ...] = ("blas", "attention", "conv", "rng")`, replacing the four bools (`_blas`, `_attention`, `_rng`, `_conv`), which were always on and never tested off. A family is a value, so a new one (NCCL) is not a new knob | one bool per family; drop the knob | A-merge6-2 |
| P46 | Device-side patching threshold | `_host_trace_capture.device_patch_min = 16`: a segment with at least 16 eligible kernel nodes (not keyed, no TMA descriptor, no launch attributes/programmatic edge) gets a head kernel that patches them in-graph from one host setter; smaller segments keep per-node host setters. Why: a head kernel and its setter cost more than a few node setters; min 1 was a loss on Blenderbot (569 vs 452 us) | host setters only; a head on every segment (min 1); cuGraphExecUpdate thresholds (rejected, costlier) | A-repatch-3/4 |
| P47 [WITHDRAWN: E-p47-drop] | A compiled graph called under a TorchDispatchMode (any: DebugMode, FlopCounterMode, a test's) | Inductor's installation runs the graph's own `call` (eager Inductor): no trace, no replay, so the mode sees every op. AOT's first-call `_AnalyzeCustomOpInputOutputMode` is exempt (it only watches, and sees the trace's ops). Cost: one `torch._C._len_torch_dispatch_stack()` per call (~25 ns). Trees traces through the mode at record and skips it at replay | check in C++ `call_boxed` (no Python, needs a build); or decline only at trace time and let hits replay, as trees does | A-warmup-1 |
| P48 | An opaque call whose key its provider refused (P2, P40), past the harvest budget (P1), or whose binding is device-updatable | a plain eager step with the key guarded; a learner retraces once when a key is refused, so the served variant does not learn | keep it opaque (the variant learns on every call, on the Python path) | A-convinc-1 |
| P49 | Priming capture before the harvest's four | every op (was convolutions only, A-conv-3): an uncaptured run leaves per-process host state in the first capture's parameters (cuDNN's transient image; nvcc's heap pointer at the end of an ATen `__host__ __device__` lambda's parameter). Cost: one more raw capture per harvested key | conv only, and refuse the rest (an eager step per padded-out addmm); or extend P39's stale-bytes leniency beyond `_CONV` | A-addmm-2 |
| P50 | A memcpy node inside a harvested call (SE 1x1 conv backward's grad_output copy) | a 1-D device-to-device memcpy becomes a launch of an NVRTC copy kernel (`host_trace_memcpy`: uint4 body when both ends are 16-aligned, byte tail), classified and patched like any kernel; other memcpy shapes (2-D/3-D, host, offsets) still refuse (snag 31) | a memcpy record in the tape and native replay (a new record kind in C++); refuse the key (14 eager steps on effnet) | A-convrest-1 |
| P51 | Which inputs of an Inductor backward the replay frees at their last use | all of them: a backward's `static_input_idxs` are saved activations, which AOT autograd hands over; freeing drops only the replay's reference (snag 31) | keep static inputs held as in the forward (backward peak 1219 vs 930 MiB on effnet once its convs bound) | A-convrest-1 |
| P52 | Fork and join inside a harvested call (cuDNN side streams) | nodes in a topological order; a programmatic edge stays one only from the node's sole predecessor right before it in the chain, and any other orders the two fully; kernels on concurrent branches refuse (eager keeps the concurrency; serialized, nfnet's grouped wgrad was 3.4x slower); a memset beside a kernel is still serialized (snag 33) | serialize everything (base; nfnet 63 vs 34 ms/step once PDL keys bind); replay the branches on side streams (needs fork/join records in the tape and native replay) | A-convrest-3 |
| P53 | Stale bytes in ATen's `reduce_kernel` (ReduceOp padding, not zeroed under the harvest flag) | the "reduce" family gets P39's conv rules in Python: a varying word is checked by launching another capture's bytes, and a constant inside a segment is kept (snag 34) | zero ReduceOp's storage under `isHostTraceHarvesting()` at its one construction site, P3's mechanism (land/scratch/sumdim/reduceop_zero.proposed.diff, unbuilt; the rules then go away) | A-sumdim-2 |

## Open (as of 2026-09-26)

- **Dynamic-shape cuDNN attention via override shape** (cudnnoverride, A-cudnnoverride-1/-2): not used; attention stays on
  the per-shape sibling harvest. Needs cuDNN >= 9.21 (9.23 for the override workspace query) AND frontend >= 1.29 (fe 1.28
  never sets the backend's override-enabled attr). On 9.26 the dense BHSD/BSHD form ATen builds is silently wrong off the
  build shape (only grid + batch word updated). The THD form (uniform ragged offsets + seq lens) is bitwise vs eager for fwd
  with two plans (s <= 128 wmma, s > 128 sm100), no build envelope, image fixed except b/s words + grid. But it costs
  ~8-10 us/call at replay (memset + qkv_tma_setup + persistent kernel; +22..101% below 64x512). Bwd engines refuse override
  on 9.26. Revisit on a cuDNN release that applies dense overrides to strides/extents or adds bwd support: re-run
  land/scratch/cudnnoverride/{probe_sdpa,diff926,rag926}.py. Conv: no (ATen conv not on fe::graph, kernel changes with N).
- **Conv harvest on the zero-init line** (conv fork): zero-init (A-zeroinit-1) deleted the stale-bytes check, but cuDNN's
  heap structs need it: ATen can't zero them. conv_vs_quality.diff restores the check for `_CONV` only. It is untested
  until an install with the zero-init C++ exists. On install_land_mem, NCHW bias-add (elementwise) stale bytes that hold
  a CUDA address still refuse some keys; zero-init's OffsetCalculator memset covers them.
- **`test_two_threads_miss_and_hit_one_entry` fails on wt-land-mem** (4/4 on the base, unchanged by the conv overlay).
- **Flash training replay memory** (snag 17): 564 MiB peak reserved vs compile 406, trees 448. The saved activations
  (272 MiB) stay live through the one-graph backward. Best alternative: place a run's later allocations inside a dying
  argument's storage (the argument stays alive until the graph is queued, so an OOM-retry release cannot lose its data).
- **Flash backward 8x16x512x64 under `use_deterministic_algorithms`** refuses: in one of the four captures two blocks
  don't coalesce. 4x16x512x64 binds.
- **Efficient attention with dropout declines** on unpatched builds: its forward returns seed tensors on the CPU (A-item5-6); effattn_seed.patch (P42, A-hffall-1) fixes it in ATen.
- **GoogleFnet training declines**: a complex view of a float allocation (view_as_complex into `_fft_c2c`) needs a dtype on the native View (A-hffall-3).
- **Warm FX cache**: a graph compiled with the flag off and loaded with it on declines (A-merge3-10). Alternative:
  compute the trusted inputs on every compile so warm hits can use host trace; that changes what flag-off compiles write
  to the cache.
- **fp32 MLP peak is 33 MiB in compile vs 2.25 in eager.** The compiled graph has the same peak, so it's not ours.
- **CuTe AOT disk cache: patch ready, not yet merged** (land/scratch/cute_aot_fix/cute_aot.diff, against merge4q;
  A-cuteaot-1..7). The problem: on a cache hit, `load_module` returns a bare `tvm_ffi.core.Function` (a C type) that
  our intercept never sees. The .o holds only LLVM-lowered host code plus the cubin, so no MLIR survives.
  The fix: `export_to_c` writes our host MLIR beside the object as `<obj>.cute_host`, and `ExternalBinaryModule.__getattr__`
  wraps each TVM-FFI function it returns. The wrapped function is intercepted like a compiled one, or becomes an
  EagerCall that says why. cutedsl_cache and quack's jit_cache rename the sidecar along with the `.o`. NVGEMM compile
  workers arm the hook through an env var.
  Snag: quack's `cute_dsl_elf_fix` replaces `ExternalBinaryModule.__init__` without calling the original, so we hook
  `__new__` to record the path.
  Snag: quack's `jit_cache` hashed `x.size(-1)`, and a SymInt raised TypeError. It is now a specialization: hashing a
  SymInt from the trace's own ShapeEnv guards on the value. That is 4 core lines, P29.
  Still open for quack rmsnorm: its call passes Nones that the compile drops (9 arguments for 5 parameters), and eps is
  an f32 parameter the evaluator does not take yet. It is an EagerCall until both are handled. A cold jit_cache compile
  inside a later (non-warm-up) trace gets N as a SymInt, and cute's fake tensor rejects it.
- **Unjoined private-stream launches are silently missed** (known limitation, parity with torch.cuda.graph): a launch
  that bypasses dispatch (ctypes, cuda-python, a loaded TVM-FFI function) onto a stream not joined to the capture runs
  eagerly during the trace and is absent from replay. Stream capture itself has the same hole. For CuTe, the fix
  above makes it decline (test_a_loaded_object_declines).
- **Real-model coverage (merge2 sweep, land/COVERAGE.md)**: no crashes; 0 of 199 host-trace rows created a trees pool.
  Geomean speedup vs eager / vs compile: HF 2.88 / 1.31, timm 1.95 / 0.95, torchbench 2.69 / 1.41. Top fallback is
  the cuDNN lse fake (25 models), which merge3 fixes. Being diagnosed: "no tensor arguments" (7), CPU args (5),
  visformer_small (fully declined yet 0.33x compile), and memory on tf_efficientnet_b0 (1.30 vs 0.91 GB) and XLNet.

- **V2 bugs (land/V2.md): 1-3 fixed and 4 narrowed in land/scratch/v2fix/merge6.diff (not yet merged; A-v2fix-1..4).**
  1. Graphsafe RNG eager steps take the Python path (fix 01). A C++ form would swap the generator state (open, rare).
  2. GEMM-only graphs harvest (fix 02, `test_a_gemm_only_graph_is_captured`).
  3. Tuple boxed inputs (subsumed: `_Installation.__call__` clears only a list).
  4. P27: steady-state argument drops across steps are in. Still open: an argument dead mid-run (one-run graphs like
     `relu(x + y) @ ...` with harvested GEMMs keep x, y to the run's end: 128 vs eager 64 MiB in V2's
     input_lifetime.py), and the warm-up/trace call, which fails the two upstream refcount tests.
  Coverage from V2: conv-only graphs (81 tests; conv fork), graphs of only ATen fallbacks (lite mode, complex,
  sort, special functions), graphsafe RNG never harvested (explicit generator), `TruncToInt`/`RoundToInt` have
  no integer lowering, mm on a dtype-view of storage.

- cuDNN native graph API (populate/update_cuda_graph), tried 2026-09-27 (A-cudnngraph-1): not adopted. Across 291 conv shapes the engines ATen picks (CUTLASS 3 sm100, nvjet) are never native-graph capable. The native ones (eng71/74/83, runtime-fusion eng9) are different kernels: not bitwise with eager in general, up to 25x slower on wgrad, and about 9.5 us per conv per replay. The API itself works (update rewrites exactly the pointer fields, 135/135 bitwise). Revisit when a cuDNN release marks CUTLASS 3 engines native: rerun land/scratch/cudnngraph/probe.py + summarize.py.
  Adoption path when engines go native (user 2026-09-27: "potentially a good api eventually"): use update_cuda_graph at harvest time only (four pointer sets -> exact pointer map from cuDNN), replacing the conv byte classification and the P39/P40 stale-field rules. Per-replay patching stays ours (~1 us/node); the ~9.5 us update cost is paid once per new key.
  Related, for dynamic shapes (user raised 2026-09-27): cudnn-frontend override shape (set_override_shape_enabled + execute(..., override_uids/shapes/strides)) lets one plan/kernel serve many shapes, so the topology stays fixed and only shape-dependent param bytes move. In frontend 1.28, populate/update_cuda_graph take no overrides, the runtime workspace query with overrides needs cuDNN 9.23 (we have 9.16), and ATen conv is not on fe::graph (only MHA). Candidate for dynamic-seq cuDNN attention later; no effect on static-shape suites. The kernel cache only cuts plan-build time for runtime-compiled engines.

- **fp16 depthwise conv backward refuses on allocations** (convrest; mobilenetv2_100 2 keys, tf_efficientnet_b0 3,
  fp16 only; A-convrest-4). The call allocates grad_input-sized blocks twice, a 782 B temporary (freed), grad_weight
  576 B and a workspace. 782 and 576 round to the same 1024 B block, so grad_weight reuses the temporary's block
  under any hole layout (best fit, lowest address). The rule "a temporary freed before the output took its bytes"
  needs the temporary inside the output's 576 bytes, and it overhangs by 206 B into the block's rounding (other keys:
  2830 vs 2592, 12174 vs 12000). The one-line fix compares against the output's 512-rounded block, but that holds
  only while every replay allocation of an output is 512-rounded; a pluggable allocator or an arena that packs
  tighter would let the kernel write past the output unnoticed. Not taken without a check that the replay's
  allocator rounds the same way.
- **Capturing cuDNN's concurrent branches** (P52, snag 33): 4 dm_nfnet_f0 keys and fp32's per-group/FFT paths run as
  eager steps. Replaying a binding's branches on side streams needs a fork/join record (event record/wait) in the
  tape, the segment capture and native replay. Worth about one graph boundary per such call (eager 1.12 vs graph
  1.01 ms on nfnet's key); not worth it for nfnet alone.
- **CUPTI memset activity bytes are the instantiation-time size for exec-updated memset nodes into planned
  allocations; never use them as size evidence** (memset lane, 2026-10-14). The 27B and 8B lanes read "baked" memset
  sizes (trtllm-gen counter, GDN zeros) from CUPTI: every replay repeated its variant's trace-call bytes. Our memsets
  are not baked. The tape width is symbolic, and the native patch sets the call's width on the exec node every replay
  (torch._C._host_trace_held_images). The memory confirms the call's width ran: in a 200-call random bs walk the
  buffer is scribbled before it dies, zeros() leaves 0/200 stale calls under auto, eager and held memory, and empty()
  leaves 200/200. The real FlashInfer 0.6.18 trtllm-gen decode (27B env) holds the formula's width at every bs and is
  bitwise, while CUPTI says the trace call's 1664 B ("Memset (Unknown)"). Memsets into argument tensors report
  correctly. Placeholder-addressed nodes sometimes run as driver memset32 kernels whose grid scales with the width.
  How to check sizes: land/core/memset/check_held_memsets.py (held images around a replay), or an effect check.
  Test: test_cuda_host_trace_replay.py test_a_fresh_zeros_zeroes_its_call_size. Repros: land/core/memset/repro.
  Open question: why CUPTI reports the instantiation size. Not reproducible in pure CUDA so far: a minimal graph
  (capture at W0, cudaGraphExecMemsetNodeSetParams to W1) reports W1 under every variant tried: clone + instantiate
  flags, upload, placeholder-address capture, set through torch's libcudart, set inside the profiler window.
- **A node's held state is per node, not per row** (memset lane, 2026-10-14). Patch.cpp's skip test compared a record's packed image with the selected row's own held buffer, so it only held when the row matched `held_row`. A row switch to a launch identical to the node's (a key that moves with bs, kernel arguments that don't) set the node anyway: 402 of 443 sets per alternating 8B step. The check now also compares against `kernel_row(r, r.held_row)` (function, param sizes, dims, image), and memsets compare against their (dst, extents, value, element size). Test: test_cuda_host_trace_opaque.py test_a_row_switch_to_the_held_launch_sets_nothing.
- **Held state is per (record, form), switching is O(1)** (memset lane, 2026-10-15). Before, an exec node's held state lived on the Record and its row's KernelRow::held buffer. A form switch either forgot it (default: every node set again) or, with form_held, copied every kernel image out and in and re-packed the run, which lost in modes whose addresses move. Now Segment::held[form] holds per record and piece what that form's exec node holds (flag, row, dims, image, memset extents, memcpy operands). A per-form Seen snapshot (rows, bases, site rows, dirty mask) restores dirty_ relative to the new form's exec, so a switch packs only records whose inputs changed since that form's last replay. Default ON (never worse in ubench_formheld held/planned and bench_alt). It is the base for per-size slots: a slot is (form, exec) with the same per-slot state.
