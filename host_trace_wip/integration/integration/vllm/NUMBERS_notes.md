
### notes

- Padded-mode correctness (out/c2_cpp_closed_pad_Vcheck.json, ARMV_PAD=1, real rows only). 281/297 are bitwise against V's
  reference eager, which uses vLLM's metadata with the max_seq_len=max_model_len policy. The other 16 are decode steps with
  9 or 5 real rows right after a mixed step (max_abs 0.5-0.75). All 16 are bitwise against stock vLLM eager
  (bitwise_vs_stock): in those padded decode steps the replay used vLLM's own max_seq_len, not the pinned policy. So every
  forward matches one valid eager reference. The pin is skipped only in padded mode; unpadded V is 228/228.
- Startup, open variant: with 0 traces and 0 harvests (the warm load), decode first calls still take 1.6-2.2 s at bs 5 and
  ~1.8 s at bs 200, against 0.10-0.20 s for closed-warm. That is per-shape work in the trtllm fork's path, not ours to cache yet.
- KV capacity at util 0.6 moves by +-11k tokens between runs of the same arm (1,072,528-1,091,712), so compare it within a round.
  In round 1, default had 1,084,032 and eager/V had 1,080,192: default got 3,840 more tokens, not fewer. The overlay's
  "~7,664 more for V" does not reproduce here.
- default prefill (T 64-512) runs ~31-42 ms in both rounds. The forward itself is ~30 ms (the fwd field), the same as the
  overlay's s5 (33 ms) and eager (35-58 ms). Its prefill is host-bound by the per-layer work between piecewise graphs; V
  replays the whole step.
- Mixed spec names are NDxPL: ND decoding requests plus one PL-token prompt added every 2 steps. So "32x512" steps
  carry 32 + 512 = 544 tokens (shape [33, 544]), not 16k. Its 4.3x sits between prefill 512 (3.8x) and the small
  mixed rows. "4x2048" steps carry 4 + 2048 = 2052 tokens (1.9x, next to prefill 2048 at 1.6x). The tokens-per-step
  sweep (t_sweep_*) covers 1k-32k tokens per step with 0/64/256 decoders.
- Open variant (traced impls + trtllm fork) on candidate 2: NaN outputs on decode at bs 4-8 (c2_cpp_open_Vcheck 193/228;
  c2_cpp_open_range 960/969, all 9 failures decode bs 4/5). Fork only (diag_fork_only): 92/92 bitwise. The ports
  (ARMV_TRACED_IMPLS=1), or the ports combined with the fork, produce the NaNs; the impls-only and both diag runs are pending.
  Open decode timing rows are therefore not valid timings of a correct run.
- GPU 0 contention, 10-07 14:13-14:56: a foreign job (pid 2587156, 50-67 GB) shared GPU 0 during t_Vcpp_open_r3 and
  t_Vpy_closed_r3. Those runs are quarantined as *.json.contended and re-queued (timing_r3b_c2.sh). Since then
  gpu0_job.sh waits for GPU 0 to have no compute process before it takes the lock, and a run that still finds one is
  skipped (SKIPPED_CONTENDED in logs/timing_idle.log).
- default-nc (10-08, pinned build, t_dnc_* / e2e_p_*; table in "stock"). vLLM with compile off and its own full graphs
  (compilation_config mode 0, cudagraph_mode FULL_DECODE_ONLY, the recipe's dgx_station_gb300 override). Requesting `FULL`
  is not accepted: vLLM logs "CUDAGraphMode.FULL is not supported with FlashInferBackend backend (support:
  AttentionCGSupport.UNIFORM_BATCH); setting cudagraph_mode=FULL_DECODE_ONLY", so the FULL runs are pooled as extra rounds.
  Correctness: its greedy tokens equal template V's at every bs and differ from eager at bs 8 (3/8 sequences, first divergence at token
  1, 4 and 19; the same in both repeats). Its captured decode graphs use the same fixed metadata policy (max_seq_len) that template V
  pinned. V-stock equals eager at every bs. default-nc runs the same kernels as eager and V-stock (stock _C ops, no Inductor), so
  default-nc vs V-stock is the same-kernel, both-graphed comparison. Decode synced: V-stock 0.91-0.99x of default-nc (bs 1: +0.32 ms;
  default-nc is itself +0.32 ms over default at bs 1, which is the Inductor-fusion GPU gap seen in PROFILE_decode). Decode async: 1.02-1.04x.
  Prefill: default-nc runs eager (28-36 ms); V-stock is 3.5-4.7x faster. Mixed 8x128: V-stock 0.10x, caused by the fork's max_kv pin for
  context calls (CORE_TASKS), not by the graphs. The host was CPU-loaded (load avg 83-155) and round-to-round spread is up to 20%
  (min-max in the table). Read decode differences under ~10% as noise.
- default's mixed 8x128 step (about 27 ms, V-stock 7.1 ms post-fix) is real host time, not a harness artifact (bench/drive_prof.py,
  out/pm_{default,vstock,fullnc}.json, traces out/prof_mixed/, analysis prof_mixed_an.py; attn line, 10-08 20:05, load avg 123-174).
  - Request mix: identical in all three arms for every measured step: 8 decoders at KV 259..273 (1 token each) + one 128-token prompt
    from KV 0 = 136 tokens, 9 requests. No Dynamo/Inductor counter moved during the measured steps (no recompiles or guard failures)
    and there was no first-use JIT. Scheduling (chunked prefill, 16384-token budget) is the same.
  - Dispatch: default ran cudagraph mode PIECEWISE at 136 tokens; 136 is itself a capture size (max capture 1024), so there is no
    padding. FULL is not available for a mixed batch: FULL_AND_PIECEWISE gives FULL only to uniform decode. V-stock and default-nc
    dispatch NONE (enforce_eager / compile off for non-uniform batches) at 9 requests.
  - GPU kernel time per step is about 5 ms in every arm: default 5.03 ms (Inductor fusions), default-nc 5.30, V-stock 5.44. All three
    run the same cuBLAS (nvjet) GEMMs and the same trtllm-gen context and decode kernels. That matches the coordinator's decode-cost
    estimate.
  - Where default's 27 ms goes: host. A PIECEWISE step is 37 graph replays with the 36 attention layers run eagerly between them.
    Under the profiler the forward was 45 ms (the profiler inflates Python about 1.7x). Of that, vllm::unified_attention_with_output
    took 34 ms (36 calls, about 0.95 ms each). Inside it:
    - FlashInfer's Python wrappers for the two trtllm calls per layer (decode + context: 72 per step);
    - an aten::zeros per call (73 per step) for the multi-CTA KV counter buffer
      (flashinfer/utils.py _resolve_trtllm_gen_multi_ctas_kv_counter_buffer);
    - the api_logging wrapper, which reads os.environ on every call for FLASHINFER_TRACE_DUMP;
    - vLLM's metadata split into decode and prefill parts.
    unified_kv_cache_update took 4.8 ms, and the 37 graph replays about 2 ms.
    Unprofiled, that is about 22 ms of host on top of about 5 ms of GPU. V-stock replays the whole step as one graph (1 launch,
    28 kernel launches outside it), so it is about GPU time plus launch.
- Scope: this is the cost of vLLM's PIECEWISE path at small mixed steps. At 4K-8K-token chunks the GPU time (55-105 ms) covers the
  same host work, which is why the g3 InferenceX rows show parity.
- The decode bs 64/128 gap at long KV (V-stock about 1.0-1.4 ms slower than default-nc) comes from host time in the replay entry
  before the graph launch. It is not a kernel or dispatch difference (bench/drive_ix3.py, prof_dec128_gpu0_c2.sh; attn line, decode
  bs 128 at KV 6056-8021; traces out/prof_dec128/, prof_dec_cmp.py, prof_dec_host.py). Unprofiled step in that job: V-stock 24.29 ms,
  default-nc 23.25 ms.
  - Same GPU work: per step GPU busy is 22.2 ms (V-stock) and 22.1 ms (default-nc), with 530 vs 531 kernels (the extra one is one
    memcpy). The kernel names are identical: the trtllm-gen decode kernel is the same Persistent (not MultiCtasKv) variant at about
    18.0 ms per step in both arms, and every GEMM picks the same cuBLAS nvjet kernel (e.g. nvjet_sm103_tst_176x128_64x5_4x1_v_bz_TNN,
    0.86 ms). Per-kernel differences are within 0.03 ms per kernel type.
  - The difference is the host time from the model forward call to cudaGraphLaunch, profiled: V-stock 782 us vs default-nc 106 us
    (run_fullgraph -> replay). Of V-stock's 782 us:
    - about 150 us is the adapter's Python key/form checks (_stock_form, _form_ok);
    - 108 us is 36 cudaGraphExecMemsetNodeSetParams calls per replay;
    - the remaining ~520 us is the C++ entry (argument binding, guard evaluation, node parameter updates), which the profiler does not
      break down.
    The GPU sits idle for that time: the GPU span grows by about 2 ms profiled while busy time is unchanged. execute_model ->
    forward also measured +0.56 ms profiled, but the V-stock run started at load 230 vs 103 and vLLM's scheduler and prepare_inputs
    (the same code in both arms) were slower by the same proportion, so that part is host load, not V-stock.
  - Implication: at bs 128 the per-replay entry cost is about 0.7-0.8 ms of host before launch (about 3% of a 23 ms step). Async
    scheduling hides part of it in serving.
- Pre-launch host time after the integration fixes (decode bs 128, KV 6056-8021; forward call -> cudaGraphLaunch, profiled):
  | V-stock run | forward -> launch | breakdown | unprofiled step |
  |---|---|---|---|
  | original adapter, attn line (prof_dec128) | 770-799 us | adapter checks ~150 us, 36 memset updates 108 us, C++ entry ~520 us | 24.29 ms |
  | armV_fast, attn line (prof_dec128c) | 692-871 us | adapter ~45 us, memset updates 109 us, C++ entry ~550 us | 24.52 ms |
  | armV_fast, attn+lc (prof_dec128b) | 458-490 us | adapter ~45 us, memset updates 0 (L1c), C++ entry ~420 us | 23.89 ms |
  default-nc's vLLM replay takes 29-41 us (unprofiled steps 23.25 / 23.37 ms). GPU busy is the same in every run (14.8 ms per
  annotated window). Both fixes are bitwise: c2_fast_attn_vstock_Vcheck and c2_fast_attnlc_vstock_Vcheck are 188/188 with the same
  traces, variants and redispatches as before.
- Capture limit under vLLM defaults and under InferenceX's batch rule (vLLM 0.29 source, land/scratch/vllm/site/vllm):
  - config/vllm.py:2003-2013 (no explicit max_cudagraph_capture_size):
    ```
    default_max_graph_size = (
        1024 if current_platform.is_device_capability_family(100) else 512
    )
    decode_query_len = self.uniform_decode_query_len
    max_num_seqs = self.scheduler_config.max_num_seqs
    max_cudagraph_capture_size = min(
        max_num_seqs * decode_query_len * 2, default_max_graph_size
    )
    ```
    The default capture list (config/vllm.py:2120-2132) is [1, 2, 4] + range(8, min(max+1, 256), 8) + range(256, max+1, 16).
  - Runtime dispatch, v1/worker/gpu/cudagraph_utils.py. _init_candidates (:306-311) maps every token count 0..(largest captured
    size) to the next captured size, for FULL and PIECEWISE. dispatch (:419-436) returns cg_mode=CUDAGraphMode.NONE for any
    (num_tokens, loras) with no candidate, i.e. above the largest captured size.
  - Our harness (vLLM defaults on GB300, sm_103 = family 100; max_num_seqs 1024, no spec decode): max capture 1024. So a mixed step
    of 1088+ tokens runs NONE: torch.compile'd code, no CUDA graph.
  - InferenceX's per-point rule, reproduced by serve/ix/ix_run.py: `args["max-num-seqs"] = max(conc, 16)`. That gives max capture
    = 2 x max(conc, 16): 32 tokens at conc 1-16, 128 at conc 64, 256 at conc 128. In InferenceX's setup every step that carries a
    prefill chunk larger than that (any real 1k or 8k prompt chunk) runs without CUDA graphs: compiled, piecewise graphs unused.
    Only decode steps (one token per request) stay graphed. The InferenceX vLLM speed-bench script's
    --max-cudagraph-capture-size 512 caps it at 512 as well.
- The g4 1024-token chunk + 64 decodes row, dispatch log and profile (bench/drive_ix4.py, prof_ix1k_gpu0_c2 on GPU 3 under the weekend
  rule; out/px_{default,fullnc,vstock}.json, trace out/prof_ix1k/):
  - Every measured step is 1088 tokens (65 requests: 64 decoders at KV 1032-2060 plus the 1024-token chunk). default dispatched
    cg_mode NONE at 1088 tokens (max capture 1024, no padding), with no Dynamo/Inductor counter change. default-nc and V-stock also
    dispatch NONE (V-stock replays its own graph inside the forward).
  - default's step has 13.9 ms of GPU kernels: the same nvjet GEMMs and trtllm-gen context + decode kernels, plus Inductor's fused
    Triton kernels. The step is host-bound. Under the profiler the forward is 109 ms:
    - 37 compiled pieces (piecewise_backend -> standalone_compile -> AOT runtime wrapper -> CompiledFxGraph), about 64 ms. Inside:
      181 Triton launches through the static launcher (18 ms), 144 mm calls (9 ms), a torch.cuda.set_device per piece (6.7 ms),
      empty_strided / reinterpret_tensor;
    - 36 eager attention calls between the pieces, about 36 ms. As in the small mixed step: FlashInfer's Python wrappers, an
      aten::zeros per call for the multi-CTA counter, and the api_logging os.environ read.
    Unprofiled (g4 r1): default 60.4 ms, default-nc (eager, vLLM _C kernels, no compiled wrappers) 33.3 ms, V-stock 16.9 ms. At this
    size, compiled code without graphs costs more host time per layer than eager does, and the GPU work (about 14 ms) is far below
    either.
