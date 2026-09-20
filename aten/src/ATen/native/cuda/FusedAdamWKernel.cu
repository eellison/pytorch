#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TypeDefault.h>
#include <ATen/native/ForeachUtils.h>
#include <c10/util/Exception.h>
#include <ATen/native/cuda/fused_adamw_amsgrad_impl.cuh>
#include <ATen/native/cuda/fused_adamw_impl.cuh>

namespace at::native {

// note(crcrpar): To observe the CI rules, i.e. 20 minutes per file to compile,
// defensively split instantiations into _impl files. this is only for CUDA 11.3
// for which it took about 20 minutes and 28 minutes in my workstation and CI,
// respectively. As a data point, it took about 20 seconds for CUDA 11.7
// installed in my environment. See
// https://github.com/pytorch/pytorch/pull/81705 for details.
void _fused_adamw_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  const bool is_mixed_precision =
      params[0].scalar_type() != exp_avgs[0].scalar_type();
  if (amsgrad) {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs},
            /*scalarList=*/{},
            /*does_op_promote_integer_inputs_to_float=*/false,
            /*skip_cross_list_dtype_check=*/is_mixed_precision),
        "params, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must have same dtype, device, and layout");
    _fused_adamw_amsgrad_cuda_impl_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  } else {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs},
            /*scalarList=*/{},
            /*does_op_promote_integer_inputs_to_float=*/false,
            /*skip_cross_list_dtype_check=*/is_mixed_precision),
        "params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout");
    _fused_adamw_cuda_impl_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  }
}

// The following overload simply has a Tensor lr
void _fused_adamw_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const at::Tensor& lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (lr.is_cpu()) {
    _fused_adamw_kernel_cuda_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr.item<double>(),
        beta1,
        beta2,
        weight_decay,
        eps,
        amsgrad,
        maximize,
        grad_scale,
        found_inf);
    return;
  }

  // Manually check devices since we specify no device check in
  // native_functions.yaml
  Device param_device = params[0].device();
  if (grad_scale.has_value()) {
    TORCH_CHECK(
        grad_scale->device() == param_device,
        "grad_scale must be on the same GPU device as the params");
  }
  if (found_inf.has_value()) {
    TORCH_CHECK(
        found_inf->device() == param_device,
        "found_inf must be on the same GPU device as the params");
  }
  TORCH_CHECK(
      lr.device() == param_device,
      "lr must be on the same GPU device as the params");

  const bool is_mixed_precision =
      params[0].scalar_type() != exp_avgs[0].scalar_type();
  if (amsgrad) {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs},
            /*scalarList=*/{},
            /*does_op_promote_integer_inputs_to_float=*/false,
            /*skip_cross_list_dtype_check=*/is_mixed_precision),
        "params, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must have same dtype, device, and layout");
    _fused_adamw_amsgrad_cuda_impl_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  } else {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs},
            /*scalarList=*/{},
            /*does_op_promote_integer_inputs_to_float=*/false,
            /*skip_cross_list_dtype_check=*/is_mixed_precision),
        "params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout");
    _fused_adamw_cuda_impl_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  }
}

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced sibling of
// _fused_adamw_kernel_cuda_, compiled here so the sibling and the real host
// above instantiate the one kernel (the impl files' multi_tensor_apply_kernel
// instantiations, reached through fused_adamw_traced_impl_ /
// fused_adamw_amsgrad_traced_impl_; DECISIONS E36): the tape's launch is
// eager's function object, not a twin. The device checks of the tensor_lr
// overload, then the fast-path restrictions with their texts, then the impl of
// the amsgrad flag; the optional tensors' addresses as SymInts. Outside a trace
// the entry runs the same launches in ordinary mode, which is how the parity
// test compares it with the real op.
#include <ATen/cuda/host_trace/ti/ForeachOps.h>
#include <c10/util/MakeNested.h>
#include <ATen/cuda/host_trace/ti/MultiTensorApplySym.cuh>
#include <ATen/native/cuda/fused_adam_utils.cuh>

#include <vector>

namespace at::cuda::host_trace::ti {

void fused_adamw_traced_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const std::optional<at::Tensor>& lr_tensor,
    double lr,
    double beta1,
    double beta2,
    double weight_decay,
    double eps,
    bool amsgrad,
    bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  // FusedAdamWKernel.cu _fused_adamw_kernel_cuda_: the device checks of the
  // tensor_lr overload, then the fast-path restrictions with its texts
  const at::Device param_device = params[0].device();
  if (grad_scale.has_value()) {
    TORCH_CHECK(
        grad_scale->device() == param_device,
        "grad_scale must be on the same GPU device as the params");
  }
  if (found_inf.has_value()) {
    TORCH_CHECK(
        found_inf->device() == param_device,
        "found_inf must be on the same GPU device as the params");
  }
  if (lr_tensor.has_value()) {
    TORCH_CHECK(
        lr_tensor->device() == param_device,
        "lr must be on the same GPU device as the params");
  }
  const bool is_mixed_precision =
      params[0].scalar_type() != exp_avgs[0].scalar_type();
  if (amsgrad) {
    TORCH_CHECK(
        fast_path_restrictions_sym(
            {params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs},
            {},
            false,
            is_mixed_precision),
        "params, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must have same dtype, device, and layout");
    if (is_mixed_precision) {
      at::native::validate_mixed_precision_dtypes(
          params,
          grads,
          exp_avgs,
          exp_avg_sqs,
          max_exp_avg_sqs,
          "Mixed-precision fused AdamW");
    }
  } else {
    TORCH_CHECK(
        fast_path_restrictions_sym(
            {params, grads, exp_avgs, exp_avg_sqs},
            {},
            false,
            is_mixed_precision),
        "params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout");
    if (is_mixed_precision) {
      at::native::validate_mixed_precision_dtypes(
          params, grads, exp_avgs, exp_avg_sqs, "Mixed-precision fused AdamW");
    }
  }
  const c10::SymInt grad_scale_ptr = grad_scale.has_value()
      ? sym_const_data_ptr<float>(*grad_scale)
      : c10::SymInt(0);
  const c10::SymInt found_inf_ptr = found_inf.has_value()
      ? sym_const_data_ptr<float>(*found_inf)
      : c10::SymInt(0);
  const c10::SymInt lr_ptr = lr_tensor.has_value()
      ? sym_const_data_ptr<float>(*lr_tensor)
      : c10::SymInt(0);
  if (amsgrad) {
    auto lists = c10::make_nested<at::Tensor>(
        params.vec(),
        grads.vec(),
        exp_avgs.vec(),
        exp_avg_sqs.vec(),
        max_exp_avg_sqs.vec());
    fused_adamw_amsgrad_traced_impl_(
        lists,
        state_steps,
        is_mixed_precision,
        lr_ptr,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale_ptr,
        found_inf_ptr,
        grad_scale.has_value());
  } else {
    auto lists = c10::make_nested<at::Tensor>(
        params.vec(), grads.vec(), exp_avgs.vec(), exp_avg_sqs.vec());
    fused_adamw_traced_impl_(
        lists,
        state_steps,
        is_mixed_precision,
        lr_ptr,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale_ptr,
        found_inf_ptr,
        grad_scale.has_value());
  }
}

} // namespace at::cuda::host_trace::ti
