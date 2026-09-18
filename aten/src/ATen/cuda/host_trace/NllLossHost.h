// The converted nll_loss forward / backward hosts (native/cuda/Loss.cu): the
// structured kernels allocate their outputs outside the dispatcher, so a
// traced call reaches the hosts through these entries with outputs the trace
// mode allocated (traced roots). Each entry performs the structured meta's
// checks, then runs the same host the IMPL_FUNCs run. Ordinary mode runs the
// same host on real tensors, which is how the parity test compares it with
// the real op.
#pragma once
#include <ATen/core/Tensor.h>

#include <cstdint>
#include <optional>
#include <tuple>

namespace at::native {

TORCH_CUDA_CU_API std::tuple<Tensor, Tensor> host_trace_nll_loss_forward_out(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& output,
    const Tensor& total_weight);

TORCH_CUDA_CU_API Tensor host_trace_nll_loss_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight,
    const Tensor& grad_input);

} // namespace at::native
