# symm lane: READY (2026-10-08 ~22:30 PDT)

Builds:
- `c8sym/install`: pinned c8x src (int6 build_cpp8 + `pinned_csrc.patch`, md5 bdfbcb2e) + item 1 csrc + Meta kernels.
  Runner: `python_symm.sh`, Python tree `py/` (pinned py + item 1).
- `sti/install`: `land/core/symti/src` as of `sti/BASE.txt` (symti_cpp.patch aedd7ac5, symti_py.patch 52b327ec) + item 1
  (symti flavour) + Meta kernels + symm host code. Runner: `python_symm_sti.sh`, Python tree `sti/py/`.

## Item 1: memcpy node re-point across memory kinds (core bug, independent of collectives)

Root cause:
- `_host_trace_capture._copy_operands` captures every D2D memcpy node between caching-allocator placeholders.
- The driver keeps a memcpy node's operand kind: cudaMalloc'd vs cuMem-mapped (probe: `probe/memcpy_kind_probe.py`).
  `cudaGraphExecMemcpyNodeSetParams1D` returns invalid-value across kinds, for dst and src alike; within a kind any
  operand works, whatever the handle type. Memset nodes have no such restriction (`probe/memset_kind_probe.py`).
- So a symm_mem (or expandable-segment) argument fails the first replay.

Fix (user's corrected decision):
- A device-side memcpy operand that is a view of an argument is captured at the argument's own address, so the
  node gets that argument's kind. Non-argument operands keep the caching-allocator placeholder.
- There is no per-call kind check. A later call whose argument is of the other kind fails the driver update; that
  error is caught at the patch and is a miss (`Outcome::MemoryKind`) before any step runs. The native entry tries
  the family's next variant; the Python path treats it as a miss, so it traces a variant of that kind.
- A memcpy in a later step whose argument operand moved is set at commit start (`set_argument_copies`), so a
  mismatch never occurs mid-call. This costs one extra node set per such memcpy per call, only when its argument
  address changed. It costs nothing for single-run variants.

Patches:

| patch | md5 | +/- | base |
|---|---|---|---|
| `patches/item1_memcpy_kind.csrc.patch` | e41ef1c72140 | +92 -4 | current pinned c8x src (applies clean, dry-run 22:20) |
| `patches/item1_memcpy_kind.py.patch` | a81e4db9c44b | +104 -9 (test +75) | current pinned py (applies clean) |
| `patches/item1_memcpy_kind.symti.csrc.patch` | a4670205ef82 | +92 -4 | symti src |
| `patches/item1_memcpy_kind.symti.py.patch` | b6b285c351a5 | +95 -8 | symti py (no pinned H2D/D2H kinds) |

Tests (GB300 GPU 0, c8sym):
- New `TestMemcpyMemoryKinds` (replay file, default allocator config, cuMem buffers from an expandable-segment
  MemPool): 4/4 OK, plus its python_entry twins 4/4 OK. Covered: a copy into each kind across both entries and the
  static prefix; a kind switch is a miss plus retrace and stays bitwise; a later step's copy misses before any step.
- Regression (full files):
  - replay 258 OK; pinned 31 OK; capture 12 OK; memory 21 OK; tape 43 OK.
  - harvest 217 with 5 failures, python_entry 519 with 9 failures. All are the known step242 set: the "98 leak"
    `_private_bytes > _POOL_KEEP` (5 + 7) and N1's hand-back test (2). Their fixes are steps 244/246, which are not
    in the pinned base.
- Follow-up for the pinned lane: F24's `_live_memcpy_operands` re-points memcpy nodes of a form's clone at
  caching-allocator scratch. A symm-argument memcpy in a segment that gains a form would become legacy kind there
  (a MemoryKind miss each call, not a crash). It should keep an argument operand's own kind or address.

## Item 1 rebased onto int6 even step280 (C++ on build_cpp9/src) -- 2026-10-09 ~01:40 PDT

| patch | md5 | +/- | base |
|---|---|---|---|
| `patches/item1_memcpy_kind.step280.int6.patch` | 31c4116916db | +104 -9 (test +75) | `int6/snap/step280` (dry-run clean on 286 and 288 too) |
| `patches/item1_memcpy_kind.step280.int6cpp.patch` | c0c7f2bb57f6 | +92 -4 | `int6/build_cpp9/src` |

Build `c9sym/install` (copy of build_cpp9/build, paths rewritten, rc 0). Runner `python_symm9.sh`, Python tree `py280/`.
LC_core now skips unchanged memcpy sets, and `set_argument_copies` keeps `r.held_copy` equal to what the node holds.

Gates (GB300 GPU 0, c9sym + py280):
- `TestMemcpyMemoryKinds` 4/4 OK. replay 280 OK; python_entry 546 OK (includes the twins); harvest 222 OK; capture 12 OK;
  memory 21 OK; tape 46 OK.
- pinned 35 with 12 failures. All are the known-bad of 276-282 (MERGE_LOG: LC_core's skip of unchanged memcpy sets also
  skips host memcpys; F24k pending).
  - The same 10 (`test_h2d_argument` x8, `test_copies_run_when_the_graph_does`,
    `test_a_freed_argument_outlives_the_replay`) fail on int6's own step280 + build_cpp9 without item 1
    (`logs/base9_pinned_k.log`).
  - The full baseline file segfaults in cuGraphLaunch at its second test (`logs/base9_pinned.log`). With item 1 it
    completes; the other 2 failures are the two MERGE_LOG names.
- For the LC_core / pinned lanes: `VariantBuild` marks every memcpy record held at the traced values. A placeholder
  operand (any non-argument operand, captured on `_copy_operands`) is not what the node holds. A later call whose
  allocation lands at the traced address would skip the set and replay the placeholder. Item 1 removes this for
  argument operands only. Suggest starting memcpy records unheld (one line), as F24k does for host memcpys.

## Item 2: symm collectives traced through their own C++ host (symti route; approved design)

- `CUDASymmetricMemoryOps.cu`: one templated host source, `SymmHost<kSym>` (eager int64 vs SymInt under the
  recorder), for `one_shot_all_reduce{,_out,_copy,_copy_out}`, `two_shot_all_reduce_{,out}` and
  `reduce_scatter_out`. Each impl branches once on `host_trace::current_recorder()`.
  - Eager's launch statements are unchanged inside `if constexpr (!kSym)`.
  - The size TORCH_CHECKs, the 16/8/4 alignment choice and single-block vs multi-block are SymInt comparisons,
    i.e. guards of the op.
  - The symbolic grid/block use the recorder's select row for `max`/`min`.
  - The launch is recorded with `host_trace::launch`. The tables, rank and world are constants; out, local and
    offset/numel are fields.
- Rendezvous: `Recorder::real_argument(t)` (new virtual; PyRecorder implements it in `_host_trace_tape._real_argument`)
  returns the real argument whose storage `t` views. SymHost calls eager's own `rendezvous(real, group)`.
  - `_real_argument` records the identity guard `p<i>.base % 2^52 == real base % 2^52`: the base symbol's hint is the
    address under `_PLACEHOLDER_TAG`, and VA is below 2^52.
  - The guard is recorded inside the op, so it is the op's own: another buffer redispatches that op. A hit does no
    rendezvous call. A tensor that is not a view of an argument declines by name (the op is an eager step).
- Meta kernels for `two_shot_all_reduce_{,out}`, `reduce_scatter_out` and `one_shot_all_reduce{,_copy}_out`: before
  this, `two_shot_all_reduce_` declined the whole call.

| patch | md5 | +/- | base |
|---|---|---|---|
| `patches/symm_host.symti.csrc.patch` | 743c92ef8c84 | +478 -238 (Recorder.h +9, Aten.cpp +11, Ops.cu template conversion) | symti src |
| `patches/symm_host.symti.py.patch` | 8388abf56d2a | +300 (non-test ~25; new `test/test_cuda_host_trace_symm.py` ~275) | symti py |
| `patches/symm_meta_kernels.csrc.patch` | 83353056e57b | +51 | SymmetricMemory.cpp (same file in pinned and symti) |

Tests (GB300 GPU 0, sti, single rank, a one-rank group): `TestSymmSingleRank` 6/6 OK.
- One-shot and the vLLM two-shot sequence (copy-in, `two_shot_all_reduce_`, copy-out) are kernels of the graph:
  no EagerCall, 1 trace, replays, bitwise.
- Host-decision flip (the user's check): sizes across the single-block threshold and the 16/8/4 alignment classes
  in one HostTraceReplay give 1 trace, 5 redispatches (6 classes), 0 eager, bitwise.
- Another rendezvoused buffer: 1 trace, a redispatch, bitwise.
- A closure buffer: the op runs eagerly.
- A plain tensor: eager's error, and the process stays clean.

Regression on sti versus the same install with the symm lane's Python reverted (`sti/py_base`):
- replay: 3 failures in both runs (zero-element std/var fill, `test_an_eager_steps_guards_are_its_own`).
- python_entry: 28 failures in both runs, the identical set (conv harvest pool bytes, interleaved KV writer,
  decline reasons, hand-back). These are symti WIP, not this lane's.
- aten: 1716 tests, 62 failures and 1 error; the identical set with the symm lane's Python reverted (`logs/ab_sti_base_aten.log`).

2-rank: `HostTraceSymmTest` in `test/test_cuda_host_trace_symm.py`.
- It covers sizes (one-shot and the vLLM two-shot), reduce-scatter, 100 back-to-back replays with
  `_assert_pads_reset` ending in `dist.barrier()`, interleaving with NCCL, two buffers, the decision flip at 2 ranks,
  and the TP layer with real GEMMs. `tearDownClass` joins with the class timeout and terminates stuck ranks.
- Queued in the polite loop `retry_sti.sh pair_sti1` (pgid 3342528; gpu0 then gpu1 for 20 s, retry every 3 min,
  ~10 min hold). The same hold runs dsv41 partA's `tp_step.py` routes on sti. Logs: `logs/pair_sti1/`.

## Item 3: vLLM TP path

- Config: `--disable-custom-all-reduce` plus `VLLM_ALLREDUCE_USE_FLASHINFER=0`, so SymmMemCommunicator does
  copy-in, `two_shot_all_reduce_` (ws=2) and copy-out. With items 1 and 2, all three trace as graph nodes.
  `two_shot_all_reduce_` goes through its own host code, single-rank verified.
- Remaining vLLM-side limits:
  - The 4 MiB cap (`SYMM_MEM_ALL_REDUCE_MAX_SIZES["10.3"][2]`): a larger all-reduce goes to pynccl.
  - The communicator disables itself without a multicast pointer. A one-rank group has none; 2 ranks still to check.
- Lifting the buffer: the arm V adapter's existing lifting (`ArmV.attrs` -> `prefix`, which `_HostTraceBound` binds)
  covers model modules' plain CUDA tensor attributes plus explicitly listed globals (`fi.trtllm_workspace_buffer`,
  trtllm_shim COUNTERS, the block-table buffers).
  - The TP group's `device_communicator.symm_mem_comm.buffer` is neither a model module attribute nor listed, so
    stock lifting does not reach it.
  - It is reachable with no engine edit by one more `(obj, attr, tensor)` entry, the same pattern as the listed
    trtllm workspace global. `ArmV._swapped` then sets the traced argument on the communicator during the trace, and
    `SymmMemCommunicator.all_reduce` reads `self.buffer` per call.
  - Without that entry, the collective is an eager step (a closure buffer declines by name).
