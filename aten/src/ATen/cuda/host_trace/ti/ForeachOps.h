// The foreach ops and fused optimizers that opt into the traced
// multi_tensor_apply host (ti/MultiTensorApplySym.cuh; each entry is compiled
// in its eager host's translation unit, where the multi_tensor_apply_kernel
// instantiation it shares with eager lives, DECISIONS E36). Each entry is
// what the op's CUDA host does around the
// chunking loop: the foreach API checks, the fast-route decision (its slow
// path is the per-tensor op through the dispatcher, as in eager) and the
// dtype dispatch to the functor. Called by the trace mode in place of the op
// (torch/cuda/_host_trace_ti.py); outside a trace they run the same launches
// in ordinary mode, which is how the parity tests compare them with the real
// op.
#pragma once
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/SymInt.h>

#include <optional>
#include <vector>

namespace at::cuda::host_trace::ti {

// _foreach_add_.Scalar / .List (ForeachBinaryOpScalar.cu / ForeachBinaryOpList.cu)
TORCH_CUDA_CU_API void foreach_add_scalar_traced_(at::TensorList tensors, const at::Scalar& scalar);
TORCH_CUDA_CU_API void foreach_add_list_traced_(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar& alpha);
// _fused_adamw_ (FusedAdamWKernel.cu and its _impl files): `lr_tensor` is the
// tensor_lr overload's CUDA lr (its pointer becomes the kernel's lr_ptr and
// `lr` is the unused 1.0 the eager host passes), nullopt the float overload
TORCH_CUDA_CU_API void fused_adamw_traced_(
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
    const std::optional<at::Tensor>& found_inf);


// the fused AdamW launches by the amsgrad flag, each compiled in the eager impl
// file whose kernel instantiation it shares (fused_adamw_impl.cu /
// fused_adamw_amsgrad_impl.cu): the dtype dispatch to the functor over the
// lists as fused_adamw_traced_ assembled them (grads_written: the gradient
// list is written when a grad scale is given, A98)
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
    bool grads_written);
void fused_adamw_amsgrad_traced_impl_(
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
    bool grads_written);

} // namespace at::cuda::host_trace::ti
