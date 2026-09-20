#include <ATen/native/cuda/fused_adamw_impl.cuh>

#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <ATen/native/cuda/fused_adam_utils.cuh>
#include <vector>

namespace at::native {

void _fused_adamw_cuda_impl_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  auto tensor_lists = c10::make_nested<Tensor>(
      params.vec(), grads.vec(), exp_avgs.vec(), exp_avg_sqs.vec());

  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  const float* lr_ptr = nullptr;

  if (params[0].scalar_type() != exp_avgs[0].scalar_type()) {
    validate_mixed_precision_dtypes(
        params, grads, exp_avgs, exp_avg_sqs, "Mixed-precision fused AdamW");
    AT_DISPATCH_V2(
        exp_avgs[0].scalar_type(),
        "fused_adamw_mp_kernel_cuda",
        AT_WRAP([&]() {
          multi_tensor_apply_for_fused_optimizer<4>(
              tensor_lists,
              state_steps,
              FusedAdamMathFunctorMP<
                  float,
                  float,
                  float,
                  scalar_t,
                  scalar_t,
                  float,
                  4,
                  ADAM_MODE::ADAMW,
                  false>(),
              lr_ptr, // unused
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
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        params[0].scalar_type(),
        "fused_adamw_kernel_cuda",
        [&]() {
          multi_tensor_apply_for_fused_optimizer<4>(
              tensor_lists,
              state_steps,
              FusedAdamMathFunctor<scalar_t, 4, ADAM_MODE::ADAMW, false>(),
              lr_ptr, // unused
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
}

// The following overload simply has a Tensor lr
void _fused_adamw_cuda_impl_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList state_steps,
    const at::Tensor& lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  auto tensor_lists = c10::make_nested<Tensor>(
      params.vec(), grads.vec(), exp_avgs.vec(), exp_avg_sqs.vec());

  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  const float* lr_ptr = lr.const_data_ptr<float>();

  if (params[0].scalar_type() != exp_avgs[0].scalar_type()) {
    validate_mixed_precision_dtypes(
        params, grads, exp_avgs, exp_avg_sqs, "Mixed-precision fused AdamW");
    AT_DISPATCH_V2(
        exp_avgs[0].scalar_type(),
        "fused_adamw_mp_kernel_cuda",
        AT_WRAP([&]() {
          multi_tensor_apply_for_fused_optimizer<4>(
              tensor_lists,
              state_steps,
              FusedAdamMathFunctorMP<
                  float,
                  float,
                  float,
                  scalar_t,
                  scalar_t,
                  float,
                  4,
                  ADAM_MODE::ADAMW,
                  false>(),
              lr_ptr,
              1.0, // unused
              beta1,
              beta2,
              weight_decay,
              eps,
              maximize,
              grad_scale_ptr,
              found_inf_ptr);
        }),
        kBFloat16);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        params[0].scalar_type(),
        "fused_adamw_kernel_cuda",
        [&]() {
          multi_tensor_apply_for_fused_optimizer<4>(
              tensor_lists,
              state_steps,
              FusedAdamMathFunctor<scalar_t, 4, ADAM_MODE::ADAMW, false>(),
              lr_ptr,
              1.0, // unused
              beta1,
              beta2,
              weight_decay,
              eps,
              maximize,
              grad_scale_ptr,
              found_inf_ptr);
        });
  }
}

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced sibling of
// _fused_adamw_cuda_impl_, compiled here so the sibling and the real host above
// instantiate the one kernel (multi_tensor_apply_kernel over
// FusedAdamMathFunctor / FusedAdamMathFunctorMP; DECISIONS E36): the tape's
// launch is eager's function object, not a twin. multi_tensor_apply_kernel and
// the metadata structs live in MultiTensorApply.cuh's anonymous namespace, one
// instantiation per translation unit, so the entry must be compiled where
// eager's host is; the chunking loop is ti/MultiTensorApplySym.cuh's (the same
// loop over SymInt numels, the metadata block a proxy, the launches through the
// typed helper). Outside a trace the entry runs the same launches in ordinary
// mode, which is how the parity test compares it with the real op.
#include <ATen/cuda/host_trace/ti/ForeachOps.h>
#include <ATen/cuda/host_trace/ti/FusedAdamWSym.cuh>

namespace at::cuda::host_trace::ti {

void fused_adamw_traced_impl_(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::TensorList state_steps,
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
  at::native::fused_adamw_sym_<4, false>(
      tensor_lists,
      state_steps,
      mixed_precision,
      lr_ptr,
      lr,
      beta1,
      beta2,
      weight_decay,
      eps,
      maximize,
      grad_scale_ptr,
      found_inf_ptr,
      grads_written);
}

} // namespace at::cuda::host_trace::ti
