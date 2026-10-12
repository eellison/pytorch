// The traced build's entry points the binding (ht_bind.cpp) calls: FlashInfer's two paged bindings, as patched,
// with tvm::ffi::TensorView / Optional / Variant<double, Tensor> as ht_types.h's TensorView / std::optional /
// ScaleArg (the same declarations ht_traced.cu defines), and the warm-up's kernel loading.
#pragma once
#include <unordered_map>

#include "ht_types.h"

namespace fi_ht_traced::flashinfer {
using fi_ht::ScaleArg;
using fi_ht::SymInt;
using fi_ht::TensorView;

void trtllm_paged_attention_decode(
    TensorView out, std::optional<TensorView> out_scale_factor, TensorView query, TensorView key_cache,
    TensorView value_cache, TensorView workspace_buffer, TensorView multi_ctas_kv_counter_buffer,
    TensorView block_tables, TensorView seq_lens, SymInt max_q_len, SymInt max_kv_len, ScaleArg bmm1_scale,
    ScaleArg bmm2_scale, double o_sf_scale, int64_t o_sf_vec_size, int64_t o_sf_start_index, SymInt batch_size,
    int64_t window_left, int64_t sparse_mla_top_k, int64_t sm_count, bool enable_pdl, SymInt workspace_size,
    std::optional<TensorView> attention_sinks, std::optional<TensorView> cum_seq_lens_q,
    std::optional<TensorView> key_block_scales, std::optional<TensorView> value_block_scales,
    std::optional<float> skip_softmax_threshold_scale_factor, std::optional<bool> uses_shared_paged_kv_idx,
    std::optional<TensorView> lse, SymInt lse_stride_tokens, SymInt lse_stride_heads,
    bool enable_block_sparse_attention, std::optional<TensorView> sparse_mla_top_k_lens,
    int64_t bf16q_fp8kv_transform_mode, std::optional<bool> use_fp16_softmax);

void trtllm_paged_attention_context(
    TensorView out, std::optional<TensorView> out_scale_factor, TensorView query, TensorView key_cache,
    TensorView value_cache, TensorView workspace_buffer, TensorView multi_ctas_kv_counter_buffer,
    TensorView block_tables, TensorView seq_lens, SymInt max_q_len, SymInt max_kv_len, ScaleArg bmm1_scale,
    ScaleArg bmm2_scale, double o_sf_scale, int64_t o_sf_vec_size, int64_t o_sf_start_index, SymInt batch_size,
    int64_t window_left, TensorView cum_seq_lens_q, TensorView cum_seq_lens_kv, int64_t sm_count, bool enable_pdl,
    SymInt workspace_size, std::optional<TensorView> attention_sinks, std::optional<TensorView> key_block_scales,
    std::optional<TensorView> value_block_scales, std::optional<float> skip_softmax_threshold_scale_factor,
    std::optional<bool> uses_shared_paged_kv_idx, std::optional<bool> use_fp16_softmax,
    std::optional<bool> uses_spcompress, bool is_causal, std::optional<TensorView> lse, SymInt lse_stride_tokens,
    SymInt lse_stride_heads);
}  // namespace fi_ht_traced::flashinfer

namespace fi_ht {
// the names of the cubins a call of this class can select (dtypes as trtllm Data_type)
std::vector<std::string> candidates(int64_t dtype_q, int64_t dtype_kv, int64_t dtype_o, bool generation,
                                    int64_t head_dim_qk, int64_t head_dim_v, int64_t page_size);
// FlashInfer's own module (fmha_gen), whose kernels the traced build's runtime launches name
void set_stock_library(const std::string& path, const std::unordered_map<std::string, uint64_t>& symbols);
// loads one (outside a capture)
void load(int64_t dtype_q, int64_t dtype_kv, int64_t dtype_o, const std::string& name);
}  // namespace fi_ht
