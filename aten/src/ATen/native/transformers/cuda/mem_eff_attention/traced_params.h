// Host tracing (ATen/cuda/host_trace) of the memory-efficient attention hosts.
// The kernels take AttentionKernel<...>::Params by value; the host writes the
// struct through this proxy (Field.h: one same-named member per field at the
// field's offset) and launches through the typed helper of Launch.h, so the
// recorder learns which byte range of the launch image each write landed in
// and the launch record names the kernel's own function. Params is a member
// of the kernel template, so the proxy is a template too and takes each offset
// from the instantiation with offsetof (the layout the kernel translation unit
// compiled) instead of a checked-in table. Beside it: the host's checks on the
// proxy's fields (eager's check_supported and the overflow / contiguity
// macros of gemm_kernel_utils.h read the plain values; these read the
// symbolic ones, a guard where the host branches).
#pragma once
#include <ATen/cuda/host_trace/Launch.h>

#include <limits>

namespace PyTorchMemEffAttention {
namespace ht = at::cuda::host_trace;

#define TRACED_PTR(name) ht::PtrField<offsetof(P, name)> name{this, #name}
#define TRACED_INT(type, name) \
  ht::IntField<type, offsetof(P, name)> name{this, #name}
#define TRACED_FLOAT(name) \
  ht::FloatField<float, offsetof(P, name)> name{this, #name}

// AttentionKernel<...>::Params (kernel_forward.h)
template <class P>
struct TracedForwardParams : ht::TracedBase {
  P pod{};
  TRACED_PTR(query_ptr);
  TRACED_PTR(key_ptr);
  TRACED_PTR(value_ptr);
  TRACED_PTR(attn_bias_ptr);
  TRACED_PTR(seqstart_q_ptr);
  TRACED_PTR(seqstart_k_ptr);
  TRACED_PTR(seqlen_k_ptr);
  TRACED_INT(int32_t, causal_diagonal_offset);
  TRACED_PTR(output_ptr);
  TRACED_PTR(output_accum_ptr);
  TRACED_PTR(logsumexp_ptr);
  TRACED_INT(int32_t, window_size);
  TRACED_FLOAT(scale);
  TRACED_INT(int32_t, head_dim);
  TRACED_INT(int32_t, head_dim_value);
  TRACED_INT(int32_t, num_queries);
  TRACED_INT(int32_t, num_keys);
  TRACED_INT(int32_t, num_keys_absolute);
  TRACED_INT(uint8_t, custom_mask_type);
  TRACED_INT(int32_t, q_strideM);
  TRACED_INT(int32_t, k_strideM);
  TRACED_INT(int32_t, v_strideM);
  TRACED_INT(int32_t, bias_strideM);
  TRACED_INT(int32_t, o_strideM);
  TRACED_INT(int32_t, q_strideH);
  TRACED_INT(int32_t, k_strideH);
  TRACED_INT(int32_t, v_strideH);
  TRACED_INT(int64_t, bias_strideH);
  TRACED_INT(int64_t, q_strideB);
  TRACED_INT(int64_t, k_strideB);
  TRACED_INT(int64_t, v_strideB);
  TRACED_INT(int64_t, bias_strideB);
  TRACED_INT(int32_t, num_batches);
  TRACED_INT(int32_t, num_heads);
  TRACED_INT(int32_t, q_heads_per_kv);
  ht::BoolField<offsetof(P, use_dropout)> use_dropout{this, "use_dropout"};
  TRACED_INT(unsigned long long, dropout_batch_head_rng_offset);
  TRACED_FLOAT(dropout_prob);
  ht::BytesField<offsetof(P, rng_engine_inputs), sizeof(at::PhiloxCudaState)>
      rng_engine_inputs{this, "rng_engine_inputs"};
  TRACED_PTR(extragraph_offset);
  TRACED_PTR(seed);

  TracedForwardParams() : ht::TracedBase(&pod, sizeof(P)) {}

  // Params::getBlocksGrid / getThreadsGrid on the proxy's fields
  template <class Kernel>
  ht::Grid blocks_grid() const {
    return ht::Grid(
        (num_queries + Kernel::kQueriesPerBlock - 1) / Kernel::kQueriesPerBlock,
        num_heads,
        num_batches);
  }
  template <class Kernel>
  static dim3 threads_grid() {
    return dim3(Kernel::kWarpSize, Kernel::kNumWarpsPerBlock, 1);
  }
};

// AttentionBackwardKernel<...>::Params (kernel_backward.h)
template <class P>
struct TracedBackwardParams : ht::TracedBase {
  P pod{};
  TRACED_PTR(query_ptr);
  TRACED_PTR(key_ptr);
  TRACED_PTR(value_ptr);
  TRACED_PTR(bias_ptr);
  TRACED_PTR(logsumexp_ptr);
  TRACED_PTR(output_ptr);
  TRACED_PTR(grad_output_ptr);
  TRACED_PTR(delta_ptr);
  TRACED_PTR(cu_seqlens_q_ptr);
  TRACED_PTR(cu_seqlens_k_ptr);
  TRACED_PTR(grad_query_ptr);
  TRACED_PTR(grad_key_ptr);
  TRACED_PTR(grad_value_ptr);
  TRACED_PTR(grad_bias_ptr);
  TRACED_PTR(workspace);
  TRACED_PTR(workspace_gv);
  TRACED_PTR(workspace_gq);
  TRACED_INT(int32_t, window_size);
  TRACED_FLOAT(scale);
  TRACED_INT(int32_t, head_dim);
  TRACED_INT(int32_t, head_dim_value);
  TRACED_INT(int32_t, num_queries);
  TRACED_INT(int32_t, num_keys);
  TRACED_INT(int32_t, num_heads);
  TRACED_INT(int32_t, q_heads_per_kv);
  TRACED_INT(uint8_t, custom_mask_type);
  TRACED_INT(int64_t, q_strideM);
  TRACED_INT(int64_t, k_strideM);
  TRACED_INT(int64_t, v_strideM);
  TRACED_INT(int64_t, bias_strideM);
  TRACED_INT(int64_t, gO_strideM);
  TRACED_INT(int64_t, gB_strideM);
  TRACED_INT(int8_t, gQKV_strideM_multiplier);
  ht::BytesField<offsetof(P, rng_engine_inputs), sizeof(at::PhiloxCudaState)>
      rng_engine_inputs{this, "rng_engine_inputs"};
  TRACED_INT(unsigned long long, dropout_batch_head_rng_offset);
  TRACED_FLOAT(dropout_prob);
  TRACED_INT(int64_t, o_strideH);
  TRACED_INT(int32_t, q_strideH);
  TRACED_INT(int32_t, k_strideH);
  TRACED_INT(int32_t, v_strideH);
  TRACED_INT(int64_t, bias_strideH);
  TRACED_INT(int64_t, o_strideB);
  TRACED_INT(int64_t, q_strideB);
  TRACED_INT(int64_t, k_strideB);
  TRACED_INT(int64_t, v_strideB);
  TRACED_INT(int64_t, bias_strideB);
  TRACED_INT(int64_t, lse_strideB);
  TRACED_INT(int64_t, lse_strideH);
  TRACED_INT(int64_t, delta_strideB);
  TRACED_INT(int64_t, delta_strideH);
  TRACED_INT(int32_t, num_batches);
  TRACED_INT(int16_t, num_splits_key);
  TRACED_INT(int64_t, gO_strideB);
  TRACED_INT(int64_t, gQ_strideB);
  TRACED_INT(int64_t, gK_strideB);
  TRACED_INT(int64_t, gV_strideB);
  TRACED_INT(int64_t, gB_strideB);
  TRACED_INT(int64_t, gO_strideH);
  TRACED_INT(int64_t, gQ_strideH);
  TRACED_INT(int64_t, gK_strideH);
  TRACED_INT(int64_t, gV_strideH);
  TRACED_INT(int64_t, gB_strideH);

  TracedBackwardParams() : ht::TracedBase(&pod, sizeof(P)) {}

  // the Params methods the host reads, on the proxy's fields
  c10::SymInt gQ_strideM() const {
    return gQKV_strideM_multiplier * num_heads * head_dim;
  }
  c10::SymInt gK_strideM() const {
    return gQKV_strideM_multiplier * num_heads * head_dim;
  }
  c10::SymInt gV_strideM() const {
    return gQKV_strideM_multiplier * num_heads * head_dim_value;
  }
  template <class Kernel>
  ht::Grid blocks_grid() const {
    return ht::Grid(num_splits_key, num_heads, num_batches);
  }
  template <class Kernel>
  static dim3 threads_grid() {
    return dim3(Kernel::kWarpSize * Kernel::kNumWarpsPerBlock, 1, 1);
  }
  static c10::SymInt align_up(const c10::SymInt& x, int64_t a) {
    return (x + a - 1) / a * a;
  }
  template <class Kernel>
  c10::SymInt workspace_size() const {
    c10::SymInt elements = 0;
    if (Kernel::kNeedsAccumGradK) {
      elements += num_splits_key * Kernel::kBlockSizeJ * align_up(head_dim, Kernel::kBlockSizeI);
    }
    if (Kernel::kNeedsAccumGradV) {
      elements += num_splits_key * Kernel::kBlockSizeJ * align_up(head_dim_value, Kernel::kBlockSizeI);
    }
    if (Kernel::kNeedsAccumGradQ) {
      constexpr int64_t cols = Kernel::MatmulGradQ::ThreadblockShape::kN;
      constexpr int64_t per_block = sizeof(typename Kernel::GradQTempStorage) / sizeof(typename Kernel::output_accum_t);
      elements += (num_queries + Kernel::kBlockSizeI - 1) / Kernel::kBlockSizeI * ((head_dim + cols - 1) / cols) * per_block;
    }
    return num_batches * num_heads * align_up(elements, 4) * static_cast<int64_t>(sizeof(float));
  }
  bool should_zero_workspace() const {
    return num_splits_key > 1 || window_size > 0;
  }
};

#undef TRACED_PTR
#undef TRACED_INT
#undef TRACED_FLOAT

// ASSIGN_CHECK_OVERFLOW on a proxy field: the store, then the check against
// the field's width (int64 fields cannot overflow an int64 stride)
template <class T, size_t Off>
void assign_check_overflow(
    ht::IntField<T, Off>& field,
    const c10::SymInt& value,
    const char* what) {
  field = value;
  if constexpr (sizeof(T) < sizeof(int64_t)) {
    TORCH_CHECK(value < std::numeric_limits<T>::max(), what, " overflows");
  }
}

// CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA with the last stride read symbolically
#define CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA_SYM(TENSOR)                  \
  TORCH_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  TORCH_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  TORCH_CHECK(                                                         \
      TENSOR.sym_stride(-1) == 1, #TENSOR ": last dimension must be contiguous");

// `address % (alignment * sizeof(scalar_t)) == 0` on a symbolic address
// (eager's is_ptr_aligned / CHECK_ALIGNED_PTR read the pointer's bits)
inline bool ptr_aligned(const c10::SymInt& ptr, int64_t alignment_bytes) {
  return ht::aligned(ptr, alignment_bytes).guard_bool(__FILE__, __LINE__);
}

// Kernel::check_supported (kernel_forward.h) over the proxy: the same checks
// with the same messages, each a guard under a trace
template <class Kernel, class P>
void check_supported_forward(const TracedForwardParams<P>& p) {
  constexpr int64_t elem = sizeof(typename Kernel::scalar_t);
  TORCH_CHECK(
      ptr_aligned(p.query_ptr, Kernel::kAlignmentQ * elem),
      "p.query_ptr is not correctly aligned");
  TORCH_CHECK(
      ptr_aligned(p.key_ptr, Kernel::kAlignmentK * elem),
      "p.key_ptr is not correctly aligned");
  TORCH_CHECK(
      ptr_aligned(p.value_ptr, Kernel::kAlignmentV * elem),
      "p.value_ptr is not correctly aligned");
  if (Kernel::kSupportsBias) {
    TORCH_CHECK(
        ptr_aligned(p.attn_bias_ptr, Kernel::kAlignmentQ * elem),
        "p.attn_bias_ptr is not correctly aligned");
    TORCH_CHECK(
        p.num_batches <= 1 || p.bias_strideB % Kernel::kAlignmentQ == 0,
        "attn_bias is not correctly aligned (strideB). ",
        "attn_bias.stride( 0) = ", p.bias_strideB.sym(), ", and should be a "
        "multiple of ", Kernel::kAlignmentQ, ".");
    TORCH_CHECK(
        p.num_heads <= 1 || p.bias_strideH % Kernel::kAlignmentQ == 0,
        "attn_bias is not correctly aligned (strideH). "
        "attn_bias.stride(1) = ", p.bias_strideH.sym(), ", and should be a "
        "multiple of ", Kernel::kAlignmentQ, ".");
    TORCH_CHECK(
        p.num_queries <= 1 || p.bias_strideM % Kernel::kAlignmentQ == 0,
        "attn_bias is not correctly aligned (strideM). "
        "attn_bias.stride(2) = ", p.bias_strideM.sym(), ", and should be a "
        "multiple of ", Kernel::kAlignmentQ, ".");
  }
  TORCH_CHECK(
      p.q_strideM % Kernel::kAlignmentQ == 0,
      "query is not correctly aligned (strideM)");
  TORCH_CHECK(
      p.k_strideM % Kernel::kAlignmentK == 0,
      "key is not correctly aligned (strideM)");
  TORCH_CHECK(
      p.v_strideM % Kernel::kAlignmentV == 0,
      "value is not correctly aligned (strideM)");
  TORCH_CHECK(
      p.num_heads <= 1 || p.q_strideH % Kernel::kAlignmentQ == 0,
      "query is not correctly aligned (strideH)");
  TORCH_CHECK(
      p.num_heads <= 1 || p.k_strideH % Kernel::kAlignmentK == 0,
      "key is not correctly aligned (strideH)");
  TORCH_CHECK(
      p.num_heads <= 1 || p.v_strideH % Kernel::kAlignmentV == 0,
      "value is not correctly aligned (strideH)");
  TORCH_CHECK(p.q_heads_per_kv > 0, "invalid GQA group size");
  TORCH_CHECK(
      p.custom_mask_type < Kernel::NumCustomMaskTypes,
      "invalid value for `custom_mask_type`");
  if (p.window_size > 0) {
    TORCH_CHECK(
        p.custom_mask_type == Kernel::CausalFromTopLeft ||
            p.custom_mask_type == Kernel::CausalFromBottomRight,
        "custom_mask_type not supported");
  }
}

// Kernel::check_supported (kernel_backward.h) over the proxy
template <class Kernel, class P>
void check_supported_backward(const TracedBackwardParams<P>& p) {
  constexpr int64_t align = Kernel::kMinimumAlignment;
  constexpr int64_t bytes = align * sizeof(typename Kernel::scalar_t);
  TORCH_CHECK(ptr_aligned(p.query_ptr, bytes), "p.query_ptr is not correctly aligned");
  TORCH_CHECK(ptr_aligned(p.key_ptr, bytes), "p.key_ptr is not correctly aligned");
  TORCH_CHECK(ptr_aligned(p.value_ptr, bytes), "p.value_ptr is not correctly aligned");
  TORCH_CHECK(ptr_aligned(p.output_ptr, bytes), "p.output_ptr is not correctly aligned");
  TORCH_CHECK(ptr_aligned(p.grad_output_ptr, bytes), "p.grad_output_ptr is not correctly aligned");
  TORCH_CHECK(ptr_aligned(p.bias_ptr, bytes), "p.bias_ptr is not correctly aligned");
  TORCH_CHECK(
      p.num_heads <= 1 || p.lse_strideH % 8 == 0,
      "LSE is not correctly aligned (strideH)");
  TORCH_CHECK(
      p.num_batches <= 1 || p.lse_strideB % 8 == 0,
      "LSE is not correctly aligned (strideB)");
  TORCH_CHECK(
      p.num_heads <= 1 || p.q_strideH % align == 0,
      "query is not correctly aligned (strideH)");
  TORCH_CHECK(
      p.num_heads <= 1 || p.k_strideH % align == 0,
      "key is not correctly aligned (strideH)");
  TORCH_CHECK(
      p.num_heads <= 1 || p.v_strideH % align == 0,
      "value is not correctly aligned (strideH)");
  TORCH_CHECK(
      p.num_batches <= 1 || p.q_strideB % align == 0,
      "query is not correctly aligned (strideB)");
  TORCH_CHECK(
      p.num_batches <= 1 || p.k_strideB % align == 0,
      "key is not correctly aligned (strideB)");
  TORCH_CHECK(
      p.num_batches <= 1 || p.v_strideB % align == 0,
      "value is not correctly aligned (strideB)");
  TORCH_CHECK(p.q_strideM % align == 0, "query is not correctly aligned (strideM)");
  TORCH_CHECK(p.k_strideM % align == 0, "key is not correctly aligned (strideM)");
  TORCH_CHECK(p.v_strideM % align == 0, "value is not correctly aligned (strideM)");
  if (p.bias_ptr) {
    TORCH_CHECK(
        p.num_batches <= 1 || p.bias_strideB % align == 0,
        "attn_bias is not correctly aligned (strideB). ",
        "attn_bias.stride(0) = ", p.bias_strideB.sym(), ", and should be a "
        "multiple of ", align, ".");
    TORCH_CHECK(
        p.num_heads <= 1 || p.bias_strideH % align == 0,
        "attn_bias is not correctly aligned (strideH) ."
        "attn_bias.stride(1) = ", p.bias_strideH.sym(), ", and should be a "
        "multiple of ", align, ".");
    TORCH_CHECK(
        p.num_queries <= 1 || p.bias_strideM % align == 0,
        "attn_bias is not correctly aligned (strideM). "
        "attn_bias.stride(2) = ", p.bias_strideM.sym(), ", and should be a ",
        "multiple of ", align, ".");
  }
  if (p.grad_bias_ptr) {
    TORCH_CHECK(
        p.num_batches <= 1 || p.gB_strideB % align == 0,
        "attn_bias.grad is not correctly aligned (strideB)");
    TORCH_CHECK(
        p.num_heads <= 1 || p.gB_strideH % align == 0,
        "attn_bias.grad is not correctly aligned (strideH)");
    TORCH_CHECK(
        p.gB_strideM % align == 0,
        "attn_bias.grad is not correctly aligned (strideM)");
  }
  TORCH_CHECK(
      !(p.cu_seqlens_q_ptr && p.bias_ptr),
      "CuSeqlen + bias not implemented yet");
  TORCH_CHECK(
      p.custom_mask_type < Kernel::NumCustomMaskTypes,
      "Invalid value for `custom_mask_type`");
  const double dropout_prob = p.dropout_prob;
  TORCH_CHECK(
      dropout_prob <= 1.0f && dropout_prob >= 0.0f,
      "Invalid value for `dropout_prob`");
  TORCH_CHECK(
      Kernel::kApplyDropout || dropout_prob == 0.0f,
      "Set `kApplyDropout`=True to support `dropout_prob > 0`");
}

} // namespace PyTorchMemEffAttention
