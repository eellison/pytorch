# vLLM host-tracing plan (Qwen3-8B, TP=1, sm_100/sm_103)

Source: vLLM v0.29.0 (98dff2a8, 2026-09-08), cloned at `src/vllm`. Paths below are relative to `src/vllm/vllm/`.

## Version and environment

- v0.29.0 pins `flashinfer-python==0.6.18`, the same version as `sglang/site`, so the FlashInfer install is shared unchanged. v0.30.0 pins 0.6.18.post1 and v0.31.0 pins 0.7.0.post1.
- v0.29.0 pins torch 2.13.0. We build and run against the local torch 2.15 (`install_land_int5b_frozen`, imported through `sglang/src_sgl1/torch` via `python_vllm.sh`). The C++ extensions use the stable libtorch ABI (`_C_stable_libtorch`, `_moe_C_stable_libtorch`), which lowers the risk.
- Build: `build_vllm.sh` runs `pip install --target site --no-deps --no-build-isolation .` with `TORCH_CUDA_ARCH_LIST=10.0`, `MAX_JOBS=16` and `taskset -c 72-107`. All deps, caches and tmp live under `vllm/build` and `vllm/cache`.
- Deviations:
  - CUDA 13.0's supported list has no 10.3 entry. Blackwell kernels build as `10.0f` (family, covers sm_103). Generic kernels build as sm_100.
  - vLLM forces its own per-kernel PTX targets: Marlin and Marlin-MoE build `8.0+PTX`, scaled_mm_c2x builds `8.9+PTX`.
  - These external projects are fetched and built under `build/deps`: cutlass, vllm-flash-attn, flashmla, deepgemm, fmha_sm100, flashkda, qutlass, tml_fa4, triton_kernels.
  - The Rust frontend is optional and skipped; cargo is absent.
  - The env's setuptools 81 is newer than the `<81` pin. The pin is not enforced under `--no-build-isolation`.
- Runtime fixes, all confined to `vllm/`:
  - `_C_stable_libtorch` calls `cublasHgemm` (gptq) without a DT_NEEDED on cublas, and the loader does not find the cublas that libtorch_cuda pulls in. `python_vllm.sh` LD_PRELOADs `libcublas.so.13`.
  - The default (compiled) arm needs Inductor caches on: `HOSTTRACE_TORCH_DISABLE_CACHES=0`, `TORCHINDUCTOR_CACHE_DIR=vllm/cache/inductor`.
  - FlashInfer downloads trtllm-gen cubins on first use into `vllm/cache/.cache/flashinfer`.
- Missing runtime deps (from the env plus `sglang/site`). These were installed `--no-deps` into `vllm/site` only:
  - cachetools, blake3, py-cpuinfo
  - prometheus-fastapi-instrumentator, python-json-logger, cbor2, ijson
  - lm-format-enforcer==0.11.3, opencv-python-headless, depyf==0.20.0
  - model-hosting-container-standards
  - openai 3.24.0: `sglang/site` has 2.6.1, which lacks `NamespaceTool`.
  - jmespath (phase 6 step 4: `vllm serve` / `vllm bench serve` import model-hosting-container-standards, which needs it).
  - Not installed, not on the offline Qwen3 path: numba (ngram spec decode), mcp, opentelemetry-*, ray, nvtx.
  - Version mismatch: `sglang/site` has compressed-tensors 0.18.0 vs the 0.17.0 pin. It is unused for bf16 Qwen3 and left as is.

## 1. Eager forward path and the host-trace entry

v0.29.0 defaults to **Model Runner V2** (`v1/worker/gpu/model_runner.py`; the log line is "Using V2 Model Runner"). It applies whenever `VLLM_USE_V2_MODEL_RUNNER` is unset and the model has no unsupported features (`config/vllm.py:645`). The v1 `gpu_model_runner.py` is the fallback and is not used here. Per step, V2 `execute_model` (1517-1818) runs these stages in order:

1. Request state lives on the GPU in persistent buffers (`req_states.*.gpu`, `input_buffers.*`).
   - `prepare_inputs` (1159) does two small H2D async copies: `idx_mapping` and `query_start_loc` (`np.cumsum` on the host).
   - Triton kernels fill the fixed `input_buffers.{input_ids,positions,seq_lens}`: `prepare_prefill_inputs`, `prepare_pos_seq_lens`, `combine_sampled_and_draft_tokens`, the last also producing `logits_indices`.
2. `prepare_attn` (1368) runs the Triton `_gather_block_tables_kernel` and `_compute_slot_mappings_kernel` (`block_table.py` 166, 204).
3. `model_state.prepare_attn` (`model_states/default.py:166`) calls `build_attn_metadata`, which runs `FlashInferMetadataBuilder.build`. `max_query_len` and `max_seq_len` come from host numpy, so there is no device sync.
4. The forward has three paths:
   - FULL: `cudagraph_manager.run_fullgraph(desc)`.
   - Otherwise, under `set_forward_context(attn_metadata, ..., slot_mapping=slot_mappings_by_layer)`: PIECEWISE runs `cudagraph_manager.run_pw_graph(model, inputs)`, eager runs `self.model(**model_inputs)`.
5. `sample_tokens` (1818) does `hidden_states[input_batch.logits_indices]`, then `model.compute_logits` and the V2 sampler.

Entry point: wrap `self.model(**model_inputs)` (`Qwen3ForCausalLM.forward`) inside the forward context. This is the same cut as ArmF's `model.forward`. The per-layer metadata, `slot_mapping` and `layer.kv_cache` are all reachable there.

V2's input prep is already GPU kernels over fixed buffers, so it is a natural second stage to bring into the same trace. That covers steps 1-3 minus the two H2D copies, and is what ArmF had to build by hand (`_metadata`/`_fill`). A later stage can add the logits gather, `compute_logits` and the sampler, all of which are device ops.

## 2. Cudagraph modes

`config/compilation.py` 53-104 defines the modes.

| Mode | What it captures |
|---|---|
| NONE | Nothing. `enforce_eager=True` selects this, plus `CompilationMode.NONE` and `custom_ops=["all"]`, so the CUDA custom ops run (config/vllm.py ~1370, ~1444). |
| PIECEWISE | Needs `VLLM_COMPILE` (Dynamo plus Inductor). The FX graph is split at `splitting_ops`. The pieces between splits are compiled and captured per size; the split ops run eagerly. |
| FULL | One graph per size for the whole forward, attention included. Only valid where the backend's `get_cudagraph_support` allows it. |
| FULL_DECODE_ONLY | FULL for uniform decode batches; everything else is eager. |
| FULL_AND_PIECEWISE (default) | FULL for uniform decode batches (FlashInfer trtllm reports UNIFORM_BATCH, ~974-1015). PIECEWISE for prefill and mixed batches. |

- Default `splitting_ops`, without Inductor graph partition (compilation.py ~1153-1191): `vllm::unified_attention_with_output` and `vllm::unified_kv_cache_update`. That gives 2 split ops per layer (72 for Qwen3-8B), with a captured piece between each consecutive pair.
- Capture sizes are `[1,2,4] + range(8,256,8) + range(256,max+1,16)`. The max is 512, or 1024 on Blackwell.
- Opt-in `VLLM_USE_BREAKABLE_CUDAGRAPH` (`compilation/breakable_cudagraph.py`) is an SGLang-BCG-like eager-break path. It is not the default and is not used here.
- Under the default mode, custom ops are `"none"`, so RMSNorm, rope and silu lower through Inductor (Triton). Under eager they are the `_C` kernels. The two baselines therefore run different elementwise kernels, while attention, KV write and GEMM match.

## 3. Attention backends on sm_100/sm_103

- Priority for causal attention on SM10x (`platforms/cuda.py` 144-163): FLASHINFER, FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION. FLASHINFER is the default, and we pin `attention_backend=FLASHINFER` in every arm.
- FlashInfer on compute family 100 (`v1/attention/backends/flashinfer.py`):
  - Decode: `flashinfer.decode.trtllm_batch_decode_with_kv_cache` (~2479). The trtllm-gen kernel is chosen statically (~1046).
  - Prefill: `flashinfer.prefill.trtllm_batch_context_with_kv_cache` (~2255), used when `kv_cache_dtype=="auto"` (`utils/flashinfer.py` 517-608).
  - Both are TVM-FFI launches of the trtllm-gen FMHA cubins, the same kernels as SGLang `trtllm_mha`.
  - The decode call passes no `multi_ctas_kv_counter_buffer`, so FlashInfer allocates and zeroes a fresh counter buffer each call (`flashinfer/utils.py` 891).
  - Workspace: the module-global `trtllm_workspace_buffer` plus the builder's `_workspace_buffer`.
- KV layout on SM10x is `LBHNC` (`flashinfer.py:542`, `v1/kv_cache_layout.py`): logical `[L, B, H, N, C]` with `C = 2*head_size`, layers outermost. Each layer's cache is a contiguous `[num_blocks, num_kv_heads, block_size, 2*head_size]` view, likely one allocation across layers. The kernel gets `(k, v) = kv_cache.split(head_size, -1)` (~2062), two interleaved views of one storage.

## 4. Custom ops on the Qwen3 path vs SGLang's extern allowlist

| vLLM op (eager) | Schema writes | SGLang analogue |
|---|---|---|
| RMSNorm: `vllm_ir::rms_norm` dispatches to `_C.rms_norm` (`kernels/vllm_c.py`) | `result!` (fresh output) | `fused_inplace_qknorm` (q/k norm), sgl rmsnorm |
| `vllm_ir::fused_add_rms_norm` dispatches to `_C.fused_add_rms_norm` | `input!`, `residual!` in place | sgl fused_add_rmsnorm |
| `_C.rotary_embedding(positions, query!, key!?, head_size, cos_sin_cache, is_neox)` (`rotary_embedding/base.py:244`) | q, k in place | `apply_rope_inplace` |
| `_C.silu_and_mul(result!, input)` (`activation.py:139`) | fresh `result` | `_run_activation_inplace` |
| `F.linear` (cuBLAS, `layers/utils.py` default_unquantized_gemm) | | blas family |

- The vLLM IR ops are `torch.library` ops in the `vllm_ir` namespace with a CompositeExplicitAutograd impl (`ir/op.py` 21, 219, 522). Their priority is `["vllm_c","native"]` when Inductor is off. Under `__torch_dispatch__` the trace should see the inner `_C` op. If it sees `vllm_ir::*` instead, allowlist those.
- Qwen3 q_norm/k_norm write a fresh tensor, so rope is in place on that tensor and not on the `qkv` views. FlashInfer rope is off (`use_flashinfer=False`).

## 5. KV-cache write

`FlashInferImpl.do_kv_cache_update` (~2518-2556) calls `torch.ops._C_cache_ops.reshape_and_cache_flash(key, value, key_cache!, value_cache!, slot_mapping, kv_cache_dtype, k_scale, v_scale)`. It is reached through `vllm::unified_kv_cache_update` (`mutates_args=[]`, returns a dummy dependency). `key_cache` and `value_cache` are the two interleaved views of the one per-layer storage.

## 6. Sampler and logits

- `logits_indices` comes from the Triton `combine_sampled_and_draft_tokens` (V2) into a runner buffer. The gather is `hidden_states[logits_indices]` (aten.index), then `compute_logits` (lm_head, `F.linear`).
- V2 sampler (`v1/worker/gpu/sample/sampler.py`): `apply_sampling_params`, then the Triton `gumbel_sample` with per-request temperature and seeds in GPU buffers; temperature 0 is greedy. FlashInfer top-k/top-p sampling applies only when no request is greedy.
- `cu_num_logits_np.tolist()` (sampler.py:162) is host numpy, not a sync.

## 7. Expected break list

1. **Opaque attention ops with hidden operands.** On CUDA, `use_direct_call=False` (`attention.py:430`), so every layer calls `torch.ops.vllm.unified_kv_cache_update` and `torch.ops.vllm.unified_attention_with_output`. Their real operands (`layer.kv_cache`, the metadata, `slot_mapping`, the workspace) come from the forward context, not from op arguments, so the trace cannot see the writes.
   - Fix: the adapter sets `layer.use_direct_call=True`, so both run as Python and expose their inner ops.
2. **FlashInfer trtllm TVM-FFI calls** read `numel()` and shapes on the host, which declines the whole entry, as in SGLang before `trtllm_shim`.
   - Fix: reuse `sglang/armF/trtllm_shim.py`, extended to the kwargs vLLM passes: `kv_layout="HND"`, `q_len_per_req`, `max_q_len`, `cum_seq_lens_q`, `o_sf_scale`, `lse`/`return_lse`, `sinks`.
   - Pass a persistent counter buffer instead of FlashInfer's per-call `zeros`.
3. **`reshape_and_cache_flash` writes two interleaved views of one storage.** SGLang's `store_cache` has separate K and V buffers.
   - It needs an `indexed` declaration (`slot_mapping`, `key_cache`, `value_cache`) that also allows the two cache operands to alias. This is likely a torch-side extension of the indexed verify (OpaqueKey.alias plus indexed together).
   - Fallback: verify on a small cache with whole-storage keys.
4. **In-place extern ops.** `_C.fused_add_rms_norm` (input and residual) and `_C.rotary_embedding` (q and k) are in place.
   - Fix: allowlist them like SGLang's in-place ops. `_C.rms_norm` and `_C.silu_and_mul` write a fresh output and are plain externs.
5. **Metadata and input prep outside the forward.** V2's prep kernels (stage 1 above) are Triton over fixed buffers and should trace as-is. `FlashInferMetadataBuilder.build` runs `torch.cumsum` (~146, ~1543), H2D copies and a CPU `.item()` (~1552, CPU tensor, no sync).
   - Stage 1: keep it eager and pass its output tensors as trace args.
   - Stage 2: rebuild it in the trace, like ArmF `_metadata`/`_fill`, with the Triton inclusive scan in place of cumsum.
   - Done (19:17, armV/adapter.py `_step_md`, ARMV_METADATA=trace default): prepare_attn's Triton kernels run inside the step; the FlashInfer build's host-tensor work (`.item()`, split_decodes_and_prefills) cannot be traced, so it runs eagerly once per key as a template and the step recomputes the template's tensor fields (verified against vLLM's at the key's first call); `_kv_indptr` Triton scan for cumsum. 0 boundaries / 1 segment, 124/124 bitwise (armV/out/check4.json).
6. **Module-global workspace.** `trtllm_workspace_buffer` and the builder's `_workspace_buffer` must be declared as attrs (ArmF already lists `workspace_buffer`).
7. **`canonicalize_singleton_dim_strides`** (`utils/torch_utils.py:135`) branches on `1 in t.shape` and uses `as_strided`. This becomes a guard on symbolic batch, and a different variant when bs=1.
8. **Logits gather** (only once the trace covers `sample_tokens`). `hidden_states[logits_indices]` is aten.index. Respell it as `index_select` (SGLang change) and stage `logits_indices` into a fixed buffer.
9. **Inputs per step.** `input_ids`, `positions` and `seq_lens` are prefix slices of V2's fixed `input_buffers`, and the slot mappings and block tables are runner buffers filled by Triton. Their addresses are already stable, which is better than SGLang before staging, so ArmF's STAGE should be unnecessary. Only the slice length varies.
10. **Possible:** `vllm_ir` ops seen as CompositeExplicitAutograd wrappers, and FlashInfer prefill's `max_q_len` taken from a host int (static per variant, fine).

## Adapter plan (reusing `sglang/armF/adapter.py`)

Create a new `vllm/armV/adapter.py`, a copy of ArmF with the SGLang specifics replaced.

- **Install:** patch the V2 runner's eager forward call (`Qwen3ForCausalLM.forward`, or `self.model` on the V2 `GPUModelRunner`). For every `Attention` layer, set `use_direct_call=True` (break 1). For the default-mode comparison arm, run under `enforce_eager` so there is no Dynamo, matching how ArmF runs over SGLang's eager runner.
- **Prefix args** follow the ArmF order, params + attrs + KV + stage:
  - params and buffers from `model.named_modules()` (as ArmF);
  - attrs: `cos_sin_cache`, FlashInfer `trtllm_workspace_buffer`, the builder's `_workspace_buffer` and a new persistent counter buffer (breaks 2 and 6);
  - KV: `layer.kv_cache` per layer, likely views of one LBHNC allocation, so passing the base storage once may be enough. This replaces ArmF's `kbuf`/`vbuf` pair.
- **Per-call args:**
  - `input_ids`, `positions`;
  - the per-layer `slot_mapping` from the forward context;
  - the FlashInfer metadata tensors (`block_tables`, `seq_lens`, `cum_seq_lens_q`, `cum_seq_lens_kv`), exposed through ArmF's `_shell`/`_cuda_fields` on `FlashInferMetadata` and its trtllm decode/prefill sub-objects.
- **Provider:** `HarvestProvider(FAMILIES + ("extern",), extern_ops=EXTERN_V, indexed=INDEXED_V)` with:
  - `EXTERN_V = (_C.rms_norm, _C.fused_add_rms_norm, _C.rotary_embedding, _C.silu_and_mul, sglang_ht::trtllm_decode, sglang_ht::trtllm_context)`; the shim ops get vLLM-specific variants.
  - `INDEXED_V = {_C_cache_ops.reshape_and_cache_flash: ("slot_mapping", "key_cache", "value_cache")}` plus the alias allowance (break 3).
- **Key and checking:** per (decode|prefill, batch bucket), as ArmF. Keep ArmF's `check` mode, which compares each call against the eager forward. Keep `counters()` and `summary()` for the coverage line.
- **Not yet:** in-trace metadata (break 5 stage 2), logits/sampler in the trace, and the default-mode arm. Arms per the handoff: eager, default (FULL_AND_PIECEWISE), host-trace, all on FLASHINFER/trtllm-gen.

## Phase 5 status (2026-10-03)

Stage 3 is in `armV/adapter.py`: the three prepare_inputs Triton launches are recorded and run inside the traced step (`ARMV_INPUTS=trace`). No new dependencies. Open items:

- Real decode max_seq_len as a per-value site key/scalar, not an entry key.
- Trusted inputs (~20 us).
- The native entry's static-arg skip (the other agent's work).

## Phase 6 step 5: real decode max_seq_len through the fork's traceable trtllm launcher (design only, 2026-10-04)

Sources: `../sglang/fork/flashinfer/trtllm_trace.py` (record() pins max_kv at line ~865; Kernels.cta() ~387,
_select_tile_q ~342), `../sglang/armF/fork/PROGRESS.md` items 19-21, logs mb_select{1,2_1,3_1}.log, CPP_INTERIM PDL attempts.

**Today.** V passes max_seq_len = max_model_len (ARMV_DECODE_MAX_SEQ=model). Putting the real value in the key makes
every decode step a new key (7x slower). Model is bitwise vs stock at bs >= 10 only: below that the KV split
n = min(ceil(max_kv / (2 step_kv)), SMs / (ctas_q * heads * bs)) and seqLenPerCtaKv move with max_kv.

**Proposal.** max_kv becomes a per-step int argument, not a pin. vLLM already has it as a host int: the builder
derives it from seq_lens_cpu_upper_bound, and the STAGE prepare_inputs builds that `ub`. The bound reads it
through its `ints` list.
- Within a variant, the record emits these as expressions of max_kv, patched per step by the native integer-row
  path (Patch.cpp already sets grid/block/smem per step for a fixed function):
  - grid.x = ctas_q * n;
  - KernelParams max_kv (byte 1176), ctas_kv (1184) and seqLenPerCtaKv.
- What max_kv cannot change inside one variant:
  - the function, through the tile_q f32 cost model and Persistent/Cga/Gmem;
  - the cluster dims, a launch attribute fixed at instantiate. SetParams cannot change them
    (INVALID_CLUSTER_SIZE in mb_select1/3).
  So the guard is an interval of max_kv. Do not make the f32 cost model symbolic. At trace time the port runs
  plan() at every step_kv boundary up to max_model_len (32 points at 4096, 1024 at 131072). It merges runs with
  the same (function, cluster) into [lo, hi], and the variant guards lo <= max_kv <= hi (two rows).
- At Qwen3-8B on GB300 (mb_select2_1), the decode table has 3 functions and clusters 1/2/3/4/8:
  - bs >= 10 is one Persistent variant for every kv > 512;
  - bs 1-9 has at most 4-5 intervals and saturates at kv >= 8192.
  Decode max_kv only grows, so a request crosses each interval once. The cold cost is one trace + harvest per new
  (bs class, interval), the same as a new bs today.
- **Grid-change cost per step.** Cga variants have cluster = n, so the grid is fixed within the variant.
  Persistent has n = 1. Only Gmem variants (cluster 1; bs 1-3 and bs 6 here) re-grid, at most once per
  2*step_kv tokens of max_kv. Param bytes are the per-step cost:
  - max_kv moves by 1 every decode step, so all 36 attention nodes become dirty: about 36 x 0.85 us (SetParams,
    mb_select1), roughly 30 us host per step.
  - The kernel reads max_kv only through ceil(max_kv / step_kv) (n, seqLenPerCtaKv). If that holds, passing
    round_up(max_kv, step_kv) leaves outputs bitwise and cuts the re-sets to one step in step_kv (128).
    Check first: witness bytes and outputs for max_kv vs round_up(max_kv) at bs 1/4/8, Gmem and Cga.
- **PDL on the patch path.** Where it was measured, patching did not lose PDL:
  - mb_select1: PDL early start (-39.5 us) survives SetParams grid changes and function switches.
  - CPP_INTERIM attempts 1-2: synthetic 500-node chains re-gridded by exec SetParams v2, exec node SetParams and
    ExecUpdate lost no early start.
  Two cases are unmeasured: PDL across a whole-graph cuGraphExecUpdate (mb_select3 stopped before it), and a
  real-model re-grid, because every F bs was its own variant. The Gmem re-grid is the first real-model case.
  Measure early starts/511 and span there before and after a re-grid, as rg1/rg2 did. If PDL is lost, use C8
  (one exec per launch configuration, sglcpp_drafts/apply_grid.py). That costs one exec per (variant, n), and
  there are at most ~19 n values per Gmem interval.
- Not viable: one node per candidate config with cuGraphNodeSetEnabled. Disabled nodes still cost device time:
  36 layers x 3 disabled = +31 us per replay, x 16 = +176 us (mb_select2_1).
- **Result.** Decode is bitwise vs stock vLLM at every bs, with no per-value entry key. Expected steady-state cost:
  0-30 us host per step (depending on the rounding check) plus a rare grid patch.
