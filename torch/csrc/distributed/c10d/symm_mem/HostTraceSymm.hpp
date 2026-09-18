#pragma once

#include <ATen/core/Tensor.h>
#include <c10/macros/Export.h>

#include <optional>
#include <string>

namespace c10d::symmetric_memory {

// The one-shot all-reduce host (CUDASymmetricMemoryOps.cu) for host tracing
// (ATen/cuda/host_trace): `input` may be a traced tensor with no storage, so
// the symmetric memory handle is looked up by `rendezvous_with`, the real
// buffer the traced input stands for. Outside a trace it is the ordinary op.
TORCH_API at::Tensor host_trace_one_shot_all_reduce_out(
    const at::Tensor& input,
    const at::Tensor& rendezvous_with,
    const std::optional<at::Tensor>& local_input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out);

} // namespace c10d::symmetric_memory
