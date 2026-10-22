# Torch-side issues hit by the vLLM lane (arm V)

## Conventions

- Paths are relative to `V = land/scratch/vllm` unless they start with `S = land/scratch/sglang` or with the core repo
  (`aten/`, `c10/`).
- Overlay: `src_vllmcpp` on install_sgl_cpp, launcher `python_vllm_cpp.sh`. Fork launcher: `python_vllm_cpp_fork.sh`.
- Line numbers are those of `src_vllmcpp/torch/cuda/*.py` as of 2026-10-04 13:45.
- Fix items are numbered as in `src_vllmcpp/CHANGES.txt`, or `src_vllm1/CHANGES.txt` for item 14. Each item lists its
  pre-change files, and `agent_merge/*.patch` holds the step patches.
- Every fix is opt-in. The torch default keeps the old behavior; the adapter switch (`armV/adapter.py`) turns it on.
- GPU prefix used for every repro:
  `CUDA_VISIBLE_DEVICES=2 flock land/scratch/eager/gpu2.lock taskset -c 108-143`.
- Tests run as `bash python_vllm_cpp.sh src_vllmcpp/test/<file>.py [-k name]`.
- TMA: no TMA issue came up in this lane. The trtllm-gen TmaDescriptors in the fork's records cross-check
  216/216 once item 22 is in (bench/logs/xcheck_vllm_fork.log).

## A. Harvest / out-of-band learn

A1. Interleaved K/V caches (two views of one storage) could not be an indexed extern writer.
- Symptom: reshape_and_cache_flash on vLLM's (B, H, N, 2*hs) layer, split into K/V views, was refused with "the
  index operand ... shares a storage". Without the declaration the whole cache was past the extern cap.
- Root cause: `torch/cuda/_host_trace_harvest.py`, the indexed path refused any cache sharing a storage (pre-14 file
  `src_vllm1`; see `armV/patches/14_interleaved_kv_writer.diff`).
- Fix: src_vllm1 item 14 (carried into src_vllmcpp). `indexed[op]` may end with an int, the leading cache dims the
  index spans. The default is unchanged when an op is not declared.
- Tests (ran; 176 passed, later 182/182 with the others): test_declared_interleaved_kv_writer,
  test_interleaved_kv_writer_dropped_v_write_is_refused.

A2. An alias group past _LENT refused every out-of-band learn of its op.
- Symptom: reshape_and_cache_flash, trtllm_context and trtllm_decode learns refused 36 each (one per layer). The
  relower failed, so the call retraced (bench/logs/mr2.log).
- Root cause: `_host_trace_harvest.py:721-723` (operands). `_LENT = 256 << 20` at :232. A member of an
  OpaqueKey.alias group past _LENT returns None.
- Repro: `bash bench/run_range.sh mr2x ARMV_STATIC_SHAPES=1 ARMV_LEND_GROUPS=0 -- --diag`. Then count the relowers
  and retraces of decode, or use bench/missreason.py via bench/run_mr.sh.
- Fix: item 16, `HarvestProvider(lend_alias_groups=False)`, under ARMV_LEND_GROUPS (adapter default 1). Diff
  `agent_merge/step4_vllmcpp_16_17_18.patch`.
- Test (ran): test_learned_interleaved_kv_writer. 3 learns with the switch, 0 without.

A3. A learning variant did not learn a keyed site's missing key.
- Symptom: rs3 retraced at prefill T257 after T32, and mixed T501/T502 after T129/T42 (bench/logs/mr5.log).
- Root cause: `_host_trace_replay.py:489` (`_fill`): the out-of-band learn ran only for `not variant.learns`.
- Fix: item 17, `HostTraceReplay(learn_in_learning_variants=False)`, under ARMV_LEARNER_LEARNS (adapter default 1).
- Test (ran): test_learned_in_learning_variant. 4 learns / 1 trace with the switch, 0 / 2 without.

A4. _relower's bind loop learned in the default pool.
- Symptom: test_learn_pool failed (6 of 12 learn operands in pool (0, 0)).
- Root cause: `_host_trace_replay.py:446`. The `_relower` bind loop had no learn MemPool, unlike `_fill`. This is
  the same fix as src_sglpre P7.
- Fix: item 19, only under ARMF_LEARN_POOL (default off).
- Test (ran): test_learn_pool, 181/181 (bench/logs/t_harvest_m19.log).
- Note: arm V never sets ARMF_LEARN_POOL, so in vLLM serving learn operands come from the default pool.

A5. Missing tests after an overlay merge.
- Symptom: item 14's tests were absent from src_vllmcpp because the base copied only the harvest file.
- Fix: item 18, a merge from src_vllm1.
- Tests ran: 181, then 182 with item 20.

## B. Range coverage: guard classes and misses (bench/rangesweep.py --diag; go2/mr*/rs* logs)

B1. Selector misses on learner variants retrace.
- Symptom: prefill n=1 T17, n=2 T18 and mixed nd=1/2 T18/T19 trace again after T<=16 (bench/logs/mr6.log).
- Root cause: index_select's small-index kernel at <= 16 indices (`aten/src/ATen/native/cuda/Indexing.cu:1658`) is a
  selector. `_host_trace_replay.py:480` redispatches a selector miss only on variants that do not learn. vLLM prefill
  and mixed families stay learning: each length runs once, so nothing relowers.
- Fix: item 20, `HostTraceReplay(redispatch_learners=False)`, under ARMV_REDISPATCH_LEARNERS (adapter default 0).
  mr7: prefill traces 4 -> 2, mixed 5 -> 3, 33/33 bitwise. Serving warm-up 19 -> 17 traces.
- Test (ran): test_learner_redispatches_selector, 182/182 (t_harvest_m20b.log). The first run failed on a wrong
  expectation in the test (m20.log).

B2. Size-1 contiguity guards split families into n==1 / nd==1 classes.
- Symptom: one extra trace per class per bucket.
- Sources: `torch.Tensor.is_contiguous()` and `.contiguous()` in vLLM's flashinfer.py:2187/2360/2374, and
  `armV/adapter.py` o.copy_.
- Root cause: `c10/core/Contiguity.h:66` (`_compute_contiguous_sym`) tests `sizes[d] == 1` with
  TORCH_GUARD_OR_FALSE before the strides. That records a guard for a backed size. :45 does the same for numel == 0.
- The five _C op ports hit the same thing: is_contiguous guards T == 1. `armV/vllm_ops_trace.py:147` `_dense`
  checks explicit strides instead.
- The T=1 view specialization in the qk_norm and cache cases (2 guard regions each over T 1..8192) is the same class.
- Fix: none (C++). The suggested fix is to try canonical strides before the size-1 test.
- Repro: `bash bench/run_range.sh rs2 ARMV_STATIC_SHAPES=1 -- --diag`, then look for the Eq/Ne(arg*.size(0), 1)
  guards.

B3. A full-width slice returns self, so T == max_num_batched_tokens is its own variant.
- Root cause: `aten/src/ATen/TensorIndexing.h:224`, applySlice's start 0 / stop length optimization, hit from vLLM
  block_table.py:222 `slot_mappings[:, :num_tokens_padded]`.
- Fix: adapter only. ARMV_WIDE_SLOTS=1 (default 0) makes the buffer one column wider.
- Repro: rs2 vs rs5 (bench/logs/range_rs5.log). The T=8192 classes disappear with the switch.

B4. `_TracedTensor.data_ptr()` guards numel != 0 on KV-layer views.
- Symptom: up to 144 new guards (`Ne(s*s*s*(s-128), 0)`) at the relower. bind_opaque sees the owners grow and returns
  None, so the call retraces. md2-md4 logs.
- Root cause: `_host_trace_tape.py:2199`.
- Fix: adapter only. ARMV_STATIC_SHAPES=1 (default 0) passes the weights, KV layers and fixed buffers as static
  shapes, which raises guards per variant from ~700 to ~1800. No core fix.
- Repro: `bash bench/run_range.sh rs1 -- --diag`.

B5. Every new exact length learns its GEMMs out of band (range coverage, not a bug).
- After the ports and the fork, aten::mm is the only learned op: 2180 keys over the full sweep, 2.6 learns per
  decode step on first sight (bench/logs/range_rs_after.log).
- In serving, the first-call cost of new mixed keys (44-52 s per session) is what puts V at 0.79x / 0.47x of default
  at c8 / c64 (PROGRESS.md step 5).
- Root cause: OpaqueKey carries exact sizes (`_host_trace_harvest.py:657` `_normal`), and cuBLAS picks per M.
- Fix: none here. Options are the open GEMM route (quack / CuTe, coordinator) or T bucketing in the adapter.

B6. trtllm-gen decode split count: one guard region per value.
- Symptom: with the fork, decode traces go from 4 to 12 and mixed from 10 to 21 over the full sweep.
- Root cause: the fork guards `Eq(Max(1, 19 // (bs * ...)), k)` (`$S/fork/flashinfer/trtllm_trace.py`, SELECT off;
  `range_rs_after.log` --diag entry decode/1). The replay in src_vllmcpp has no selected-launch form for a scalar
  choice.
- Fix: none in src_vllmcpp. `FLASHINFER_HT_SELECT=1` needs the SelectedLaunch replay prototype
  ($S/armF/fork/PROGRESS.md section 26), which is not here.
- Repro: `bash bench/run_items.sh rs_after fork ARMV_STATIC_SHAPES=1 ARMV_WIDE_SLOTS=1 ARMV_TRACED_IMPLS=1
  ARMV_TRTLLM_FORK=1 -- --diag`.

## C. Describing extension kernels

C1. C++ extension ops (vLLM _C) could only be harvested per key.
- Symptom: rms_norm, fused_add_rms_norm, rotary_embedding, silu_and_mul and reshape_and_cache_flash were learned per
  new T: 3815 of the 6617 learned keys in the full sweep (range_rs_before.log).
- Root cause: the tape had no way to run a described launch for a dispatcher op with no traced host
  (`_host_trace_tape.py:836` is where the new hook sits).
- Fix: item 21 (`$S/armF/externport/traced_impls.diff`), `ht.traced_impls = False` plus `register_traced_impl`.
  Launchers are in `armV/vllm_ops_trace.py` under ARMV_TRACED_IMPLS (default 0).
- Tests (ran):
  - `armV/opsport/xcheck_params.py`: 401/401 byte-identical launches per case, T 1..8192.
  - `armV/opsport/test_trace.py` TS=all: bitwise; guard regions 1/2/1/1/2/2.
  - Logs: armV/opsport/logs/full_*.log.
- No torch unit test for register_traced_impl in src_vllmcpp/test.
- Repro: `TS=all bash armV/opsport/run_unit.sh`.

C2. Packed struct parameters were missing from src_vllmcpp.
- Symptom: with the fork, every trtllm call became an EagerCall: 36 boundaries per variant
  (bench/logs/range_ck_after.log). The cause was a `KernelLaunch(packed=True)` TypeError, turned into a decline.
- Root cause: src_sglfork's F1 had not been merged. `_host_trace_launch.py` KernelLaunch.packed;
  `_host_trace_capture.py:600, 684` join the pieces. VLLM_NOTES.md does not list the requirement.
- Fix: item 22 (F1 copied unchanged; field default False).
- Test (ran): test_cuda_host_trace_packed.py, 2 passed (bench/logs/t_packed.log). Then xcheck_vllm 216/216 and
  ck_after2 51/51 bitwise.

C3. Kernel handle lookup: host-stub names differ from device names (tooling, not core).
- Symptom: fused_add_rms_norm's width-8 `enable_if` overload has a stub that demangles with `((8)==(0))||...`, while
  the device function is named with `&&`. The witness's name compare declined.
- Fix: `armV/vllm_ops_trace.py:34` `kernel()` takes the name from `cuFuncGetName(func)`. The same applies to
  externport's `_symtab` users.

C4. Ops that need a CPU sync or CUB cannot be traced (by design; workarounds are adapter-side).
- vLLM's metadata build calls `.item()` (model_states/default.py:196), so every entry declined (check2). The adapter
  traces the prepared inputs instead.
- torch.cumsum's two CUB kernels were an eager boundary (check3). The adapter uses a Triton `_kv_indptr` kernel.

## D. Process

- D1. The overlays/installs are not ABI-interchangeable (install_int5b vs install_sgl_cpp). Each test file has to run
  on its own install (CHANGES item 19 note).
- D2. Metric caveat: the per-variant "boundaries" census includes learner variants, which break at every still-opaque
  op. Report keyed variants separately; the rangesweep CENSUS line gives the distribution.
