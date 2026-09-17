#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <limits>

namespace at::native {

#if 0 && AT_USE_JITERATOR()
constexpr char tanh_name[] = "tanh_impl";
#endif

namespace {
template <typename scalar_t>
struct TanhFunctor {
  __device__ scalar_t operator()(scalar_t a) const {
    return ::tanh(a);
  }
};
} // namespace

void tanh_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    // Disabled due to accuracy issues
#if 0 && AT_USE_JITERATOR()
    static const auto tanh_string = jiterator_stringify(
        template <typename T> T tanh_impl(T a) { return std::tanh(a); });
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "tanh_name", [&]() {
          jitted_gpu_kernel<
              /*name=*/tanh_name,
              /*return_dtype=*/scalar_t,
              /*common_dtype=*/scalar_t,
              /*arity=*/1>(iter, tanh_string);
        });
#else
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "tanh_name", [&]() {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            return ::tanh(static_cast<opmath_t>(a));
          });
        });
#endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        common_dtype,
        "tanh_cuda",
        [&]() {
          gpu_kernel(iter, TanhFunctor<scalar_t>());
        });
  }
}

REGISTER_DISPATCH(tanh_stub, &tanh_kernel_cuda)

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced sibling of tanh_kernel_cuda, compiled here
// so the sibling and the real host above instantiate the one kernel over TanhFunctor
// (DECISIONS E36): the tape's launch is eager's function object, not a twin. Outside a trace the
// entry runs the same launches in ordinary mode, which is how the parity test compares it with
// the real op.
#include <ATen/cuda/host_trace/ti/Ops.h>
#include <ATen/cuda/host_trace/ti/UnaryEntry.cuh>

namespace at::cuda::host_trace::ti {

Tensor tanh_traced(const Tensor& self) {
  return floating_unary<at::native::TanhFunctor>(self, "tanh");
}

} // namespace at::cuda::host_trace::ti
