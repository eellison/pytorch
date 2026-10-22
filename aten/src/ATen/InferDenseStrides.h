#pragma once
// infer_dense_strides (ExpandUtils.h) over SymInt sizes and strides: a
// symbolic tensor's (a CUDA graph host trace's), each comparison a guard
#include <c10/core/SymInt.h>
#include <c10/macros/Export.h>
#include <c10/util/ArrayRef.h>

#include <vector>

namespace at {

TORCH_API std::vector<c10::SymInt> infer_dense_strides_symint(c10::SymIntArrayRef tensor_sizes, c10::SymIntArrayRef tensor_strides);

} // namespace at
