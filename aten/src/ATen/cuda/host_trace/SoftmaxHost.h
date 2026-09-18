// The converted softmax / log_softmax forward and backward hosts
// (native/cuda/SoftMax.cu): the structured kernels allocate their output
// outside the dispatcher, so a traced call reaches the host through these
// entries with an output the trace mode allocated (a traced root). Ordinary
// mode runs the same host on real tensors, which is how the parity test
// compares it with the real op.
#pragma once
#include <ATen/core/Tensor.h>

#include <cstdint>

namespace at::native {

TORCH_CUDA_CU_API Tensor host_trace_softmax_out(
    const Tensor& input,
    int64_t dim,
    bool half_to_float,
    bool log_softmax,
    const Tensor& output);

TORCH_CUDA_CU_API Tensor host_trace_softmax_backward_out(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype,
    bool log_softmax,
    const Tensor& grad_input);

} // namespace at::native
