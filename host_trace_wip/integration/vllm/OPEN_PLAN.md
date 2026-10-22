# What an all-open vLLM step under arm V would take (scoping only, 2026-10-04)

Rule from the coordinator: benchmark only with open kernels. That means no closed launch logic and no learns. This file
is a plan. Nothing here is implemented, except the GEMM microbenchmark (bench/gemm_open.py, bench/gemm_open_table.py,
bench/out/gemm_open_*.json). The src_vllmcpp items are unchanged.

## 1. Opaque sites per step (Qwen3-8B, 36 layers; from bench/logs/mr7.log CENSUS)

| op | decode | prefill | mixed | where it comes from |
|---|---|---|---|---|
| aten mm (cuBLAS) | 144 | 144 | 144 | qkv, o, gate_up, down x 36 |
| _C.rms_norm | 73 | 73 | 73 | q_norm + k_norm x 36, plus the first input_layernorm |
| _C.fused_add_rms_norm | 72 | 72 | 72 | 35 input_layernorm + 36 post_attention + final norm |
| _C.rotary_embedding | 36 | 36 | 36 | |
| _C.silu_and_mul | 36 | 36 | 36 | |
| _C_cache_ops.reshape_and_cache_flash | 36 | 36 | 36 | |
| trtllm decode (vllm_ht shim) | 36 | 0 | 36 | |
| trtllm context (vllm_ht shim) | 0 | 36 | 36 | |
| total | 433 | 433 | 469 | 433 = the keyed sites of a bound variant |

- Nothing else in the step is opaque: everything else is an aten op that the trace records natively.
- The lm_head GEMM is not in the step (144 = 4 x 36). vLLM runs compute_logits outside the forward, and it does so in
  both arms. It still goes through cuBLAS in both arms, so the kernel choice below should cover it too.
- FlashInfer's per-call counter torch.zeros (no multi_ctas_kv_counter_buffer from vLLM) is already a traced fill.

## 2. Open route per op

### vLLM _C ops: Python ports of the host launch logic (externport pattern)

Reuse land/scratch/sglang/armF/externport/sgl_ops_trace.py's shared part, about 265 lines:
- `kernel(module, export, demangled)`: handles from the ELF symtab;
- `_Call`, `_select`, `_cap`, `blocks_per_sm`, `sm_count`;
- `install` -> `torch.cuda._host_trace.register_traced_impl(op, launcher)` under the `traced_impls` switch.

Factor it out, or import it. Every kernel the launchers need is in the symtab of
site/vllm/_C_stable_libtorch.abi3.so:
- rms_norm_kernel and fused_add_rms_norm_kernel bf16 instantiations;
- rotary_embedding_kernel<bf16,bf16,*>;
- 9 bf16 silu act_and_mul_kernel instantiations;
- 36 reshape_and_cache_flash_kernel instantiations.

A launcher that sees something outside its gate declines to the extern harvest, as in SGLang.

| op | C++ source (src/vllm/csrc/libtorch_stable/) | launch logic to port | T-dependence | rough size |
|---|---|---|---|---|
| rms_norm | layernorm_kernels.cu:221-293 | vec = gcd(8, hidden); rank 2/3/4 template; block = min(hidden/vec, T<256 ? 1024 : 256); grid T | block size at T<256 only when hidden/vec > 256 (hidden 4096: 512 vs 256). q/k norm (hidden 128) has block 16 and no T-dependence. The block size is an int, so it becomes `_select` (no guard). | ~40 lines |
| fused_add_rms_norm | layernorm_kernels.cu:314-372 | block = min(hidden, T<256 ? 1024 : 256); width 8 vs 0 from 16-byte alignment of input/residual/weight and strides % 8 | block via `_select`. Alignment becomes a guard; it holds for every vLLM buffer, so it adds no class. | ~40 lines |
| rotary_embedding | pos_encoding_kernels.cu:103-200 | grid T; block min(heads*rot/2, 512); neox template; strides | none (block is static) | ~35 lines |
| silu_and_mul | activation_kernels.cu:244-297 | cc>=10 and T>128 picks the 256-bit-vector template; block min(d/vec, 1024) | kernel switch at T 128/129. Today that is a guard, so it is a class boundary (see section 3). | ~30 lines |
| reshape_and_cache_flash | cache_kernels.cu:801-870 | grid = slot_mapping.size(0); block min(heads*D, 512); "auto" kv dtype -> bf16 template; strides of the interleaved K/V views | none. The adapter's indexed-writer metadata (slot_mapping, key_cache, value_cache) carries over. | ~45 lines |

- Total: about 190 lines of launchers plus a ~30-line install/test glue.
- Tests: copy externport/test_trace.py. Check each launcher's node byte for byte against the captured C++ node at a
  few T on both sides of each switch.
- `vllm_is_batch_invariant()` is read once at install (env); its value is constant for a run.
- Adapter change: drop the five ops from armV/adapter.py:207's extern list when the switch is on.
- A zero-port fallback that is worse on GPU time: run V with vLLM's `custom_ops=["none"]`. The forward_native aten
  bodies are traced natively, but they run unfused.

### Attention: the FlashInfer trtllm fork (already written)

land/scratch/sglang/fork/flashinfer/trtllm_trace.py (1206 lines) is done and checked against vLLM's call forms:
108/108 page-16 runs equal in xcheck_vllm. Wiring follows fork/VLLM_NOTES.md steps 1-5:
- a launcher variant of python_vllm_cpp.sh with the fork ahead of $S/site;
- cubin dir and no download;
- drop vllm_ht.trtllm_* from the extern list and keep the workspace attr.

Keep ARMV_DECODE_MAX_SEQ=model and ARMV_PREFILL_MAX_KV=model, so the split choice adds no classes. The default arm
calls the same trtllm-gen kernels through the unmodified binding, so the comparison stays fair.

### GEMM

GPU time per call (us, median), bench/out/gemm_open_table.md:
- measured as a CUDA graph cycling over weight copies of at least 320 MB, so weights come from HBM and host launch
  cost is out;
- GPU 2, bf16;
- FI cute-dsl refuses M > 32 by design;
- Inductor = max-autotune, TRITON backends only, autotuned per (shape, M);
- TMA = plus `triton.enable_persistent_tma_matmul`, which uses the blackwell_ws_persistent_tma templates.

| shape | M | cuBLAS | TGV (CuTe) | FI cute-dsl | Inductor Triton | Inductor Triton TMA | vLLM BI Triton | best open / cuBLAS |
|---|---|---|---|---|---|---|---|---|
| qkv | 1 | 10.4 | 9.0 | 12.8 | 12.0 | 10.8 | 62.5 | 0.86 (tgv) |
| qkv | 32 | 10.3 | 9.3 | 13.9 | 17.0 | 16.7 | 56.7 | 0.90 (tgv) |
| qkv | 128 | 10.6 | 11.3 | - | 16.8 | 16.7 | 35.7 | 1.07 (tgv) |
| qkv | 512 | 17.1 | 28.2 | - | 27.9 | 27.8 | 66.5 | 1.63 (tma) |
| qkv | 2048 | 56.8 | 94.9 | - | 79.6 | 74.4 | 190.0 | 1.31 (tma) |
| qkv | 8192 | 242.3 | 360.9 | - | 309.1 | 271.1 | 649.4 | 1.12 (tma) |
| o | 1 | 8.8 | 6.6 | 7.9 | 7.4 | 8.6 | 54.6 | 0.74 (tgv) |
| o | 32 | 8.9 | 7.5 | 8.4 | 11.5 | 11.6 | 49.6 | 0.84 (tgv) |
| o | 128 | 8.4 | 8.2 | - | 13.1 | 13.1 | 35.0 | 0.98 (tgv) |
| o | 512 | 12.2 | 17.4 | - | 20.7 | 21.0 | 35.2 | 1.43 (tgv) |
| o | 2048 | 39.2 | 60.8 | - | 52.5 | 48.4 | 114.3 | 1.23 (tma) |
| o | 8192 | 161.2 | 237.6 | - | 198.1 | 174.2 | 395.6 | 1.08 (tma) |
| gate_up | 1 | 35.8 | 33.9 | 34.0 | 33.9 | 32.9 | 147.0 | 0.92 (tma) |
| gate_up | 32 | 34.9 | 36.2 | 35.9 | 41.1 | 42.1 | 135.7 | 1.03 (fi) |
| gate_up | 128 | 37.4 | 41.0 | - | 44.2 | 42.4 | 75.5 | 1.10 (tgv) |
| gate_up | 512 | 64.2 | 147.4 | - | 87.9 | 80.5 | 200.8 | 1.25 (tma) |
| gate_up | 2048 | 213.2 | 542.6 | - | 297.1 | 265.0 | 684.2 | 1.24 (tma) |
| gate_up | 8192 | 945.2 | 2079.3 | - | 1220.9 | 1091.2 | 2635.9 | 1.15 (tma) |
| down | 1 | 20.4 | 18.0 | 18.1 | 17.4 | 17.2 | 128.4 | 0.84 (tma) |
| down | 32 | 20.0 | 19.5 | 18.9 | 30.2 | 30.3 | 122.5 | 0.95 (fi) |
| down | 128 | 24.4 | 21.1 | - | 34.9 | 34.9 | 99.1 | 0.86 (tgv) |
| down | 512 | 35.3 | 48.4 | - | 56.4 | 52.0 | 100.8 | 1.37 (tgv) |
| down | 2048 | 111.6 | 175.4 | - | 150.4 | 141.0 | 384.8 | 1.26 (tma) |
| down | 8192 | 493.2 | 761.9 | - | 591.3 | 538.3 | 1335.1 | 1.09 (tma) |
| lm_head | 1 | 204.0 | 174.5 | 173.3 | 177.8 | 178.3 | 473.2 | 0.85 (fi) |
| lm_head | 32 | 208.9 | 184.3 | 183.7 | 194.4 | 193.4 | 542.4 | 0.88 (fi) |
| lm_head | 128 | 210.0 | 218.2 | - | 211.0 | 201.9 | 301.3 | 0.96 (tma) |

GEMM GPU time per step (36 x (qkv + o + gate_up + down), plus the lm_head, which is outside the step):

| step | M (lm_head M) | cuBLAS us | best open us | ratio | TGV for M<=128, Inductor TMA for M>128 | ratio |
|---|---|---|---|---|---|---|
| decode | 1 (1) | 2923 | 2538 | 0.87 | 2601 | 0.89 |
| decode | 32 (32) | 2876 | 2759 | 0.96 | 2793 | 0.97 |
| decode | 128 (128) | 3117 | 3142 | 1.01 | 3159 | 1.01 |
| prefill | 512 (1) | 4839 | 6443 | 1.33 | 6700 | 1.38 |
| prefill | 2048 (1) | 15349 | 19213 | 1.25 | 19214 | 1.25 |
| prefill | 8192 (1) | 66513 | 74868 | 1.13 | 74869 | 1.13 |

What this says:
- At M <= 128 (all of decode), CuTe matches or beats cuBLAS. TGV is the best kernel there. FI cute-dsl is close,
  except on qkv (12.8-13.9 vs 10.3).
- At M >= 512, there is no open kernel at cuBLAS speed:
  - The best is Inductor's Blackwell persistent-TMA Triton template, at 1.07-1.27x for M >= 2048 and 1.25-1.63x at
    M = 512.
  - TGV is not built for large M; SGLang itself only uses it at M <= ~64-128.
  - vLLM's batch-invariant Triton (one fixed config) is 2.5-6x.
- So an all-open step costs about 13-38% more GEMM GPU time on prefill and mixed steps, and nothing on decode.

Options, by least change to vLLM:
1. `--linear-backend flashinfer_cutedsl`, which already exists in vLLM v0.29 (model_executor/layers/utils.py:160).
   - It sends M in 1..32 to FlashInfer cute-dsl and everything else to F.linear (cuBLAS).
   - No vLLM change, and it works in both arms: it is a registered custom op (`vllm.cuda_flashinfer_bf16_gemm`) that
     the default arm compiles as an opaque op inside its cudagraphs.
   - It is not all-open: M > 32 stays on cuBLAS, and the lm_head ignores it (vocab_parallel_embedding.py:74 calls
     `dispatch_unquantized_gemm()` with the default "auto").
   - Useful as the low-M half.
2. Recommended: one new entry in vLLM's `_FLASHINFER_BF16_BACKENDS`-style table, or an equivalent `linear_backend`
   value such as "open", about 80 lines in utils.py plus a 1-line lm_head change. Both arms select it with the same
   flag, so both run the same kernels:
   - M <= 32 (or <= 128): a CuTe low-M GEMM.
     - Either FI cute-dsl (already in the vLLM env), or TGV, which is better on qkv but lives in the sglang tree.
       TGV is pure Python CuTe DSL, so it can be imported or copied into the vLLM env.
     - V already traces TGV's CuTe launch (src_sgltgv change 14, `cute_without_tvm_ffi`). FI cute-dsl's launch path
       through the same CuTe route is not yet checked.
   - M > 32/128: a Triton matmul with a fixed config table per (N, K, M range), taken from the Inductor TMA autotune
     above (the winning configs are in bench/logs/gemm_open_tma.log).
     - V traces a Triton launch natively.
     - The default arm runs the same kernel. Either put it in a custom op, or call it as a user-defined Triton kernel
       inside the compiled graph.
     - Do not let the default arm run Inductor max-autotune on its own. It would pick different configs per M, and
       the arms would no longer run the same kernel.
   - Cost: the step GEMM ratios above (decode 0.89-1.01, prefill 1.13-1.38).
3. `VLLM_BATCH_INVARIANT=1`: no code change, and it works in both arms (aten::mm/addmm/linear overrides plus Triton
   rms_norm). But its GEMM is 2.5-6x cuBLAS, and it also changes FlashInfer to fixed split sizes and
   disable_split_kv. Not representative. Rejected for benchmarking.

Fairness: with option 2, both arms run the same GEMM kernels. The only remaining kernel difference between the arms is
by design: the default arm's Inductor-fused Triton for norm/rope/silu against V's vLLM _C kernels. Both are open.

## 3. Expected warm-up once everything is open (no learns)

Today, sr3b (V + ARMV_REDISPATCH_LEARNERS=1):
- 17 traces over 10 entries (decode b0/b1/b2 x {pos_seq form, prefill form}, prefill b1/b2, mixed b1/b2);
- 13 relowers; 7544 learns during warm-up;
- 2508 more learns during serving, which are the 44-52 s of mixed first calls.

The structural classes:
- 6 decode entries, 1 trace each;
- prefill and mixed, {n==1, n>1} per bucket (c10 size-1 contiguity guards): 8;
- 3 more not attributed from the serving log; in the rs5/mr7 breakdowns these were T<=16 selector splits on variants
  that were still learning.

With no learns:
- There are no learner variants and no _learn or _relower. Every variant is bound from its trace.
- A new exact T only re-evaluates expressions; the 115-133 ms median first call goes away.
- Selector misses redispatch on every variant, so the T<=16 embedding split folds, as decode b1 already does.

New boundaries come from kernel switches. Today these are guards; int choices like block sizes are `_select` and cost
nothing. The buckets are BS_SPLIT 2,32: b0 T<2, b1 2-32, b2 33 up.

| switch | boundary | falls in |
|---|---|---|
| TGV `_pick_tactic` ladder (148-SM one-wave rule): qkv 18/21/23/24 and gate_up 12/26/27/24 switch at M 17, 33, 65; o and down 18/21/23 switch at 33, 65 | 16/17, 32/33, 64/65 | b1 (17), the bucket edge (33, free), b2 (65) |
| silu_and_mul 256-bit template (cc>=10, T>128) | 128/129 | b2 |
| CuTe -> Triton handoff at M>128 | 128/129 (same edge) | b2 |
| one Triton config per shape for all M>128 | none (two M ranges would add one edge, e.g. 1024/1025) | b2 |
| rms_norm / fused_add block size at T<256 | `_select`, no class | - |
| trtllm decode split (max_seq_len pinned to model) | none | - |
| trtllm context | to measure: the fork guards on everything it reads; max_kv is pinned, but max_q_len (T) may reach the tile choice | prefill, mixed |

Count:

| family | today (sr3b) | all open (TGV ladder as is) | all open, one CuTe tactic per bucket |
|---|---|---|---|
| decode (2 forms) | 6 | b0 1 + b1 2 + b2 3 = 6 per form -> 12 | 1 + 1 + 2 -> 8 |
| prefill ({n==1, n>1}) | 4 | b1 2 x 2 + b2 2 x 3 = 10 | 2 + 4 = 6 |
| mixed ({nd==1, nd>1}) | 4 (+3 unattributed) | 10 | 6 |
| total | 17 (+ learns) | about 32 traces, 0 learns, 0 relowers | about 20 traces, 0 learns, 0 relowers |

- At the measured ~2-2.6 s per trace, warm-up is about 60-85 s of tracing with the TGV ladder as is, or about 40-50 s
  with one tactic per bucket. Today's warm-up is about 17 traces plus 7544 learns.
- Ways to get the count down, each behind a switch:
  - One CuTe tactic per BS bucket. Pick the tactic that fits M = 32 for all of b1; the cost is small at M <= 16, see
    the TGV row at M 1 vs 32. The CuTe boundaries then line up with the bucket edges.
  - Run silu's 256-bit template for every T. It is elementwise, so the result is bitwise equal, but it no longer
    matches eager's launch choice. Or move the CuTe/Triton handoff to 128/129 as well, so both switches share one
    edge, as in the table.
  - The SelectedLaunch prototype (fork PROGRESS.md task 1) lets a kernel switch become a select instead of a guard.
    With it, every boundary above folds, and the count returns to the 14 structural classes.
- The remaining n==1/nd==1 classes need the c10 contiguity change (PROGRESS.md 11:30). Unchanged.

## Order of work (if approved)

1. Wire the trtllm fork (VLLM_NOTES steps 1-5): removes 36-72 opaque sites per step, with no new code.
2. Port the 5 _C launchers (~220 lines with glue) behind `traced_impls`, plus byte-for-byte tests.
3. GEMM option 2: CuTe low-M plus a fixed Triton config table, the same flag in both arms. First run a per-step serving
   A/B of the default arm on cuBLAS vs the open GEMM, to put a number on the prefill cost.
4. Rerun the range warm-up and sr serving with learns off, to check the class count against section 3.
