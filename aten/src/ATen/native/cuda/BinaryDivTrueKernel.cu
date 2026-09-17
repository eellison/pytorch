#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/TypeSafeSignMath.h>
#include <ATen/native/cuda/BinaryInternal.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>

#include <type_traits>

namespace at::native {
namespace binary_internal {

constexpr char div_name[] = "div_kernel";
void div_true_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (iter.common_dtype() == kComplexHalf) {
    using scalar_t = c10::complex<at::Half>;
#if AT_USE_JITERATOR()
    static const auto div_string = jiterator_stringify(
        template <typename T> T div_kernel(T a, T b) { return a / b; });
    opmath_jitted_gpu_kernel_with_scalars<div_name, scalar_t, scalar_t>(
        iter, div_string);
#else
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_gpu_kernel_with_scalars<scalar_t>(iter, DivFunctor<opmath_t>());
#endif
    return;
  }
  if (iter.is_cpu_scalar(2)) {
    // optimization for floating-point types: if the second operand is a CPU
    // scalar, compute a * reciprocal(b). Note that this may lose one bit of
    // precision compared to computing the division.
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, common_dtype, "div_true_cuda", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          using high_prec_t = std::conditional_t<
              c10::is_complex<scalar_t>::value,
              c10::complex<double>,
              double>;
          auto inv_b = static_cast<opmath_t>(high_prec_t(1.0) / iter.scalar_value<high_prec_t>(2));
          iter.remove_operand(2);
          gpu_kernel(
              iter,
              BUnaryFunctor<scalar_t, scalar_t, scalar_t, MulFunctor<opmath_t>>(
                  MulFunctor<opmath_t>(), inv_b));
        });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, common_dtype, "div_true_cuda", [&]() {
          DivFunctor<scalar_t> f;
          gpu_kernel_with_scalars(iter, f);
        });
  }
}
} // namespace binary_internal

REGISTER_DISPATCH(div_true_stub, &binary_internal::div_true_kernel_cuda)

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced sibling of div_true_kernel_cuda, compiled here
// so the sibling and the real host above instantiate the one kernel over DivFunctor and BUnaryFunctor over MulFunctor
// (DECISIONS E36): the tape's launch is eager's function object, not a twin. Outside a trace the
// entry runs the same launches in ordinary mode, which is how the parity test compares it with
// the real op.
#undef TORCH_ASSERT_NO_OPERATORS  // BinaryInternal.h's; the file keeps TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/cuda/host_trace/ti/LoopsSym.cuh>
#include <ATen/cuda/host_trace/ti/Ops.h>

namespace at::cuda::host_trace::ti {

Tensor div_traced(const Tensor& self, const Tensor& other, const Tensor& out) {
  TensorIteratorSym iter = TensorIteratorSym::binary_op(out, self, other);
  if (!at::isFloatingType(iter.common_dtype())) {
    decline(c10::str("host_trace: div on ", iter.common_dtype(), " is not traced (declined)"));
  }
  if (iter.is_cpu_scalar(2)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "div_traced", [&] {
      using opmath_t = at::opmath_type<scalar_t>;
      using functor_t = at::native::binary_internal::MulFunctor<opmath_t>;
      auto inv_b = static_cast<opmath_t>(double(1.0) / iter.scalar_value<double>(2));
      iter.remove_operand(2);
      ScalarFunctor<at::native::BUnaryFunctor<scalar_t, scalar_t, scalar_t, functor_t>, functor_t, opmath_t> bf(functor_t(), inv_b);
      gpu_kernel(iter, bf.get());
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "div_traced", [&] {
      at::native::binary_internal::DivFunctor<scalar_t> f;
      gpu_kernel_with_scalars(iter, f);
    });
  }
  return iter.output();
}

} // namespace at::cuda::host_trace::ti
