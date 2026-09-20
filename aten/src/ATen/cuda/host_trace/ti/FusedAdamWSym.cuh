// The fused AdamW dtype dispatch of the traced sibling (fused_adamw_impl.cu /
// fused_adamw_amsgrad_impl.cu): the mixed-precision functor for bf16 states
// under fp32 params, the kernel's remaining arguments as eager passes them,
// over ti/MultiTensorApplySym.cuh's loop. A template each impl translation
// unit instantiates for its depth, so the multi_tensor_apply_kernel it names
// is the instantiation eager's host in that unit launches (the kernel lives
// in MultiTensorApply.cuh's anonymous namespace; DECISIONS E36).
#pragma once
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/cuda/host_trace/ti/MultiTensorApplySym.cuh>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <ATen/native/cuda/fused_adam_utils.cuh>
#include <c10/core/SymInt.h>

#include <array>
#include <type_traits>
#include <vector>

namespace at::native {

// fused_adamw_impl.cu / fused_adamw_amsgrad_impl.cu: the dtype dispatch to
// the functor (the mixed-precision functor for bf16 states under fp32
// params), the kernel's remaining arguments as eager passes them
template <int depth, bool amsgrad>
void fused_adamw_sym_(
    std::vector<std::vector<Tensor>>& tensor_lists,
    TensorList state_steps,
    bool mixed_precision,
    const c10::SymInt& lr_ptr,
    double lr,
    double beta1,
    double beta2,
    double weight_decay,
    double eps,
    bool maximize,
    const c10::SymInt& grad_scale_ptr,
    const c10::SymInt& found_inf_ptr,
    bool grads_written) {
  // the kernel writes the params and the moments; the gradient only when it
  // unscales it (a grad scale given): the address roles of A98
  std::array<bool, depth> written;
  written.fill(true);
  written[kGradIdx] = grads_written;
  using metadata_t = FusedOptimizerTensorListMetadata<depth>;
  constexpr const char* name = amsgrad ? "fused_adamw_amsgrad_kernel_cuda" : "fused_adamw_kernel_cuda";
  constexpr const char* mp_name = amsgrad ? "fused_adamw_amsgrad_mp_kernel_cuda" : "fused_adamw_mp_kernel_cuda";
  if (mixed_precision) {
    AT_DISPATCH_V2(
        tensor_lists[kExpAvgIdx][0].scalar_type(),
        mp_name,
        AT_WRAP([&]() {
          using max_t = std::conditional_t<amsgrad, scalar_t, float>;
          using functor_t = FusedAdamMathFunctorMP<float, float, float, scalar_t, scalar_t, max_t, depth, ADAM_MODE::ADAMW, amsgrad>;
          at::cuda::host_trace::multi_tensor_apply_for_fused_optimizer_sym<depth>(
              multi_tensor_apply_kernel<metadata_t, functor_t, const float*, double, double, double, double, double, bool, const float*, const float*>,
              written,
              tensor_lists,
              state_steps,
              functor_t(),
              lr_ptr,
              lr,
              beta1,
              beta2,
              weight_decay,
              eps,
              maximize,
              grad_scale_ptr,
              found_inf_ptr);
        }),
        kBFloat16);
    return;
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, tensor_lists[kParamIdx][0].scalar_type(), name, [&]() {
    using functor_t = FusedAdamMathFunctor<scalar_t, depth, ADAM_MODE::ADAMW, amsgrad>;
    at::cuda::host_trace::multi_tensor_apply_for_fused_optimizer_sym<depth>(
        multi_tensor_apply_kernel<metadata_t, functor_t, const float*, double, double, double, double, double, bool, const float*, const float*>,
        written,
        tensor_lists,
        state_steps,
        functor_t(),
        lr_ptr,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale_ptr,
        found_inf_ptr);
  });
}

} // namespace at::native
