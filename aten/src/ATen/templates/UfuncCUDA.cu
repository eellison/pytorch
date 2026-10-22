#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/ufunc/${name}.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/Scalar.h>
${cuda_headers}

namespace at {

// NB: this is explicitly copied here (via codegen) rather than
// included via NativeFunctions.h to avoid recompiling this file when
// NativeFunctions.h changes
namespace meta {
${meta_declaration}
}

namespace native {
${native_declaration}
${native_definitions}
}} // namespace at::native

// the ufunc's traced host (ATen/cuda/host_trace), if it has one
#if !defined(USE_ROCM) && __has_include(<ATen/cuda/host_trace/UfuncCUDA_${name}.cuh>)
#include <ATen/cuda/host_trace/UfuncCUDA_${name}.cuh>
#endif
