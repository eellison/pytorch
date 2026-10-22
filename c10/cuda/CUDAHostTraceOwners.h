#pragma once

#include <c10/cuda/CUDAMacros.h>

#include <memory>
#include <vector>

namespace c10::cuda {

// Set on a thread while the host-trace harvest captures an op there: a
// library call appends what keeps the kernels it launches loaded (a cuDNN
// plan), for the harvested binding to hold. Returns the previous sink
using HostTraceOwners = std::vector<std::shared_ptr<void>>;
C10_CUDA_API HostTraceOwners* hostTraceOwnerSink();
C10_CUDA_API HostTraceOwners* setHostTraceOwnerSink(HostTraceOwners* sink);

} // namespace c10::cuda
