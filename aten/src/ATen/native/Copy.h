#pragma once

#include <ATen/native/DispatchStub.h>

namespace at {

class Tensor;
struct TensorIterator;
class TensorBase;

namespace native {

using copy_fn = void (*)(TensorIterator&, bool non_blocking);

DECLARE_DISPATCH(copy_fn, copy_stub)

TORCH_API void copy_ignoring_overlaps(const TensorBase &dst, const TensorBase &src);

// A CUDA graph host trace's answer to whether two of its tensors are the same
// data (one root and offset), which copy_impl asks of storages: a traced
// tensor's storage has no address to compare. Set while the trace's recorder
// runs the op's host code (torch/csrc/cuda/host_trace/Aten.cpp)
using SymbolicSameStart = bool (*)(const TensorBase& a, const TensorBase& b);
TORCH_API SymbolicSameStart set_symbolic_same_start(SymbolicSameStart hook);

} // namespace native
} // namespace at
