# arm V on int6: what the integration lane added (land/scratch/integration/, nothing in int6 or scratch/vllm touched)

- ../python_vllm_int6_odd.sh: int6/python_int6_vllm.sh with HOSTTRACE_WT=int6/snap/step177 and TORCH_EXTENSIONS_DIR=vllm/ext_odd.
- ../python_vllm_int6_even.sh: the same env with HOSTTRACE_WT=int6/snap/step178, HOSTTRACE_INSTALL=HOSTTRACE_PREFIX=int6/build_cpp6/install,
  TORCH_EXTENSIONS_DIR=vllm/ext_even (vLLM site, LD_PRELOAD libcublas.so.13, the vLLM caches as in scratch/vllm/python_vllm_cpp.sh).
- bench/drive.py, bench/rangesweep.py: copies of scratch/vllm/bench/*.py with exactly two edits each:
  the armV path is absolute (scratch/vllm/armV, unchanged adapter), and gpu_memory_utilization reads INTEG_GPU_UTIL
  (default 0.6); rangesweep's max_model_len reads INTEG_MAX_MODEL_LEN (default 16384). Reason: a foreign job holds ~250 GB
  on every GPU, so 0.6 cannot initialize; the runs use 0.08 (smaller KV pool than the overlay's 1,080,192 tokens).
- probe.py: kwarg probe (adapter drops missing kwargs silently). summ.py: count table. run.sh: the run chain (gpu1.lock per command).
- bench/group_budget.py: diagnostic only. Sets HarvestProvider's budget_unit default to "group" in process, then runs the driver.
  The overlay's semantics are 1024 groups; int6's default is 1024 keys, counting cuBLAS harvests. run2.sh: round 2 (odd drives,
  range ranges at INTEG_GPU_UTIL=0.072, group-budget A/B).

## Candidate 2 (2026-10-07)
- ../python_vllm_cand2.sh: CAND_LINE=cpp (candidate2/src_cpp on int6/build_cpp7/install) or py (candidate2/src_py on
  install_land_int5b_frozen), same vLLM env as above; CAND_FORK=1 adds the FlashInfer trtllm fork ($S/fork ahead of $S/site,
  cache_fork, cubin dir, FLASHINFER_NO_DOWNLOAD=1) as scratch/vllm/python_vllm_cpp_fork.sh. Sets INTEG_ARMV=vllm/armV_cand2.
- armV_cand2/: copy of scratch/vllm/armV/*.py (live modules only; the adapter.pre*/stage* history copies dropped). Changes:
  - removed ARMV_LEARNER_LEARNS / ARMV_REDISPATCH_LEARNERS and their replay kwargs (int6 retired both at S6, steps 135/136);
  - added ARMV_PAD (default 0) + ARMV_PAD_MAX_TOKENS (512) / ARMV_PAD_MAX_REQS (256): with 1, the runner's cudagraph_manager.dispatch
    returns vLLM-default-style padded descriptors (tokens to the next capture size, uniform decode also pads requests) and
    _same_inputs accepts the padded batch; summary()["pad"] reports the padded rows. Default 0 = today's V (exact sizes).
  - no budget option: candidate 2 removed the harvest budget (96).
- retired/group_budget.py (was bench/): moot on candidate 2.
- bench/drive.py, additional edits: INTEG_ARMV picks the adapter dir; --mixed-spec NDxPL (more decode:prefill ratios, same
  form as the stock mixed block); per-workload raw step rows (R["rows"]); model-forward counts by phase (init / bench, every
  Qwen3ForCausalLM.forward call: compile, capture, warm-up, trace, eager, check); R["util"], reserved after init.
- bench/rangesweep.py: INTEG_ARMV as above.
- bench/startup.py (new): process start -> imports -> LLM() (vLLM's own model-load / compile / capture lines) -> V install
  (+ HarvestProvider.load) -> V warm-up (11 forwards, one per structural class) = ready; then first + second request per probe
  shape class with trace / harvest / restore deltas; --save writes the bindings.
- run_c2s1.sh (step 1, GPU 1), smoke_c2.sh, gpu0_job.sh + timing_lib.sh + timing_c2.sh / startup_c2.sh (GPU 0 jobs).
- 2026-10-08: adopted land/core/pinned/vllm/armV_pinned.patch into armV_cand2 (ARMV_VP_PINNED=1 with ARMV_VP_UVA=0: VP's
  staging and token outputs as plain pinned copy_(non_blocking=True), which the pinned core lane traces as memcpy nodes).
  Backup of the pre-patch files: armV_cand2_bak_prepinned/. Launcher: CAND_LINE=pinned (land/core/pinned py + c8x/install).
  Also since the last OVERLAY entry: ARMV_DUMP_LAUNCH / ARMV_DUMP_PLAN / ARMV_DUMP_TRTLLM diagnostics (summary() dumps),
  ARMV_HT_SET (host-trace module switches), ARMV_TRACED_IMPLS_NAMES (port subset), ARMV_DECODE_MAX_SEQ=sym (stock
  max_seq_len as a symbolic int, open path), drive arms Vc/compnc (stopped), INTEG_TRAP_NUMEL, INTEG_KV_BYTES.
