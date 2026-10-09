#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/DeviceAddCmulCdiv.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/PointwiseOps.h>
#include <c10/core/Scalar.h>

namespace at::native {

void addcmul_cuda_scalar_tensor2_kernel(
  TensorIteratorBase& iter,
  const Scalar& scalar_tensor2,
  const Scalar& value
);

#if AT_USE_JITERATOR()
constexpr char addcmul_name[] = "addcmul";
#endif
template <typename scalar_t>
struct AddcmulComplexFunctor {
  scalar_t alpha;
  __device__ scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return a + alpha * b * c;
  }
  auto host_trace_fields() const {
    return std::tie(alpha);
  }
};

template <typename scalar_t, typename accscalar_t>
struct AddcmulFunctor {
  accscalar_t alpha;
  __device__ scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return pointwise_op_impl<accscalar_t>(a, b, c, alpha, std::multiplies<accscalar_t>());
  }
  auto host_trace_fields() const {
    return std::tie(alpha);
  }
};

void addcmul_cuda_kernel(TensorIteratorBase& iter, const Scalar& value) {
  TORCH_CHECK(
    !iter.is_cpu_scalar(1),
    "CPU Scalar support for self argument is not supported when "
    "calling addcmul on CUDA tensors."
  );

  TORCH_CHECK(
    !iter.is_cpu_scalar(2),
    "CPU Scalar support for tensor1 argument is not supported when "
    "calling addcmul on CUDA tensors. "
    "However, CPU Scalar support for tensor2 is supported, "
    "please swap your tensor1 and tensor2 terms."
  );

  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    #if AT_USE_JITERATOR()
      AT_DISPATCH_COMPLEX_TYPES(dtype, "addcmul_cuda", [&]() {
        auto alpha = value.to<scalar_t>();
        static const auto addcmul_string = jiterator_stringify(
          template <typename T> T addcmul(T a, T b, T c, T alpha) { return a + alpha * (b * c); });
        if (iter.is_cpu_scalar(3)) {
          auto tensor2_val = iter.scalar_value<scalar_t>(3);
          iter.remove_operand(3);
          return addcmul_cuda_scalar_tensor2_kernel(iter, tensor2_val, value);
        }
        jitted_gpu_kernel<
            /*name=*/addcmul_name,
            /*return_dtype=*/scalar_t,
            /*common_dtype=*/scalar_t,
            /*arity=*/3>(
            iter,
            addcmul_string,
            /*scalar_pos=*/at::cuda::jit::BinaryFuncVariant::NoScalar,
            /*scalar_val=*/0,
            /*extra_args=*/std::make_tuple(alpha));
      });
    #else
      AT_DISPATCH_COMPLEX_TYPES(dtype, "addcmul_cuda", [&]() {
        if (iter.is_cpu_scalar(3)) {
          auto tensor2_val = iter.scalar_value<scalar_t>(3);
          iter.remove_operand(3);
          return addcmul_cuda_scalar_tensor2_kernel(iter, tensor2_val, value);
        }

        auto alpha = value.to<scalar_t>();
        gpu_kernel(iter, AddcmulComplexFunctor<scalar_t>{alpha});
      });
    #endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, dtype, "addcmul_cuda", [&]() {
      if (iter.is_cpu_scalar(3)) {
          auto tensor2_val = iter.scalar_value<scalar_t>(3);
          iter.remove_operand(3);
          return addcmul_cuda_scalar_tensor2_kernel(iter, tensor2_val, value);
      }
      // note(mkozuki): If scalar_t is fp16 or bfloat16, cast scalar to float
      // and do math in fp32 for better accuracy.
      using accscalar_t = at::acc_type<scalar_t, true>;
      auto alpha = value.to<accscalar_t>();
      gpu_kernel(iter, AddcmulFunctor<scalar_t, accscalar_t>{alpha});
    });
  }
}

#if AT_USE_JITERATOR()
constexpr char addcmul_scalar_tensor2_name[] = "addcmul_scalar_tensor2";
#endif
template <typename scalar_t>
struct AddcmulScalarTensor2ComplexFunctor {
  scalar_t alpha;
  scalar_t c;
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const {
    return a + alpha * (b * c);
  }
  auto host_trace_fields() const {
    return std::tie(alpha, c);
  }
};

template <typename scalar_t, typename accscalar_t>
struct AddcmulScalarTensor2Functor {
  accscalar_t alpha;
  accscalar_t c;
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const {
    return pointwise_op_impl<accscalar_t>(a, b, c, alpha, std::multiplies<accscalar_t>());
  }
  auto host_trace_fields() const {
    return std::tie(alpha, c);
  }
};

void addcmul_cuda_scalar_tensor2_kernel(TensorIteratorBase& iter, const Scalar& scalar_tensor2, const Scalar& value) {
  auto dtype = iter.common_dtype();

  if (at::isComplexType(dtype)) {
    #if AT_USE_JITERATOR()
      AT_DISPATCH_COMPLEX_TYPES(dtype, "addcmul_cuda", [&]() {
        auto c = scalar_tensor2.to<scalar_t>();
        auto alpha = value.to<scalar_t>();

        static const auto addcmul_scalar_tensor2_string = jiterator_stringify(
          template <typename T> T addcmul_scalar_tensor2(T a, T b, T c, T alpha) { return a + alpha * (b * c); });

        jitted_gpu_kernel<
            /*name=*/addcmul_scalar_tensor2_name,
            /*return_dtype=*/scalar_t,
            /*common_dtype=*/scalar_t,
            /*arity=*/2>(
            iter,
            addcmul_scalar_tensor2_string,
            /*scalar_pos=*/at::cuda::jit::BinaryFuncVariant::NoScalar,
            /*scalar_val=*/0,
            /*extra_args=*/std::make_tuple(c, alpha));
        });
    #else
      AT_DISPATCH_COMPLEX_TYPES(dtype, "addcmul_cuda", [&]() {
        auto c = scalar_tensor2.to<scalar_t>();
        auto alpha = value.to<scalar_t>();
        gpu_kernel(iter, AddcmulScalarTensor2ComplexFunctor<scalar_t>{alpha, c});
      });
    #endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, dtype, "addcmul_cuda", [&]() {
      // note(mkozuki): If scalar_t is fp16 or bfloat16, cast scalar to float
      // and do math in fp32 for better accuracy.
      using accscalar_t = at::acc_type<scalar_t, true>;
      auto c = scalar_tensor2.to<accscalar_t>();
      auto alpha = value.to<accscalar_t>();
      gpu_kernel(iter, AddcmulScalarTensor2Functor<scalar_t, accscalar_t>{alpha, c});
    });
  }
}

#if AT_USE_JITERATOR()
// return a + alpha * (b / static_cast<accscalar_t>(c));
constexpr char addcdiv_name[] = "addcdiv";
#endif
template <typename scalar_t>
struct AddcdivComplexFunctor {
  scalar_t alpha;
  __device__ scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return a + alpha * (b / c);
  }
  auto host_trace_fields() const {
    return std::tie(alpha);
  }
};

template <typename scalar_t, typename accscalar_t>
struct AddcdivFunctor {
  accscalar_t alpha;
  __device__ scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    //return a + alpha * (b / static_cast<accscalar_t>(c));
    return pointwise_op_impl<accscalar_t>(a, b, c, alpha, std::divides<accscalar_t>());
  }
  auto host_trace_fields() const {
    return std::tie(alpha);
  }
};

void addcdiv_cuda_kernel(TensorIteratorBase& iter, const Scalar& value) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    #if AT_USE_JITERATOR()
      AT_DISPATCH_COMPLEX_TYPES(dtype, "addcdiv_cuda", [&]() {
        auto alpha = value.to<scalar_t>();
        static const auto addcdiv_string =
            jiterator_stringify(template <typename T> T addcdiv(
                T a, T b, T c, T alpha) { return a + alpha * (b / c); });
        jitted_gpu_kernel<
            /*name=*/addcdiv_name,
            /*return_dtype=*/scalar_t,
            /*common_dtype=*/scalar_t,
            /*arity=*/3>(
            iter,
            addcdiv_string,
            /*scalar_pos=*/at::cuda::jit::BinaryFuncVariant::NoScalar,
            /*scalar_val=*/0,
            /*extra_args=*/std::make_tuple(alpha));
      });
    #else
      AT_DISPATCH_COMPLEX_TYPES(dtype, "addcdiv_cuda", [&]() {
        auto alpha = value.to<scalar_t>();
        gpu_kernel(iter, AddcdivComplexFunctor<scalar_t>{alpha});
      });
    #endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, dtype, "addcdiv_cuda", [&]() {
      // note(mkozuki): If scalar_t is fp16 or bfloat16, cast scalar to float
      // and do math in fp32 for better accuracy.
      using accscalar_t = at::acc_type<scalar_t, true>;
      auto alpha = value.to<accscalar_t>();
      gpu_kernel(iter, AddcdivFunctor<scalar_t, accscalar_t>{alpha});
    });
  }
}

template <typename scalar_t>
struct SmoothL1BackwardFunctor {
  scalar_t norm_val;
  scalar_t beta_val;
  __device__ scalar_t operator()(scalar_t input, scalar_t target, scalar_t grad_output) const {
    const auto x = input - target;
    if (x < -beta_val)
      return -norm_val * grad_output;
    else if (x > beta_val)
      return norm_val * grad_output;
    else
      return norm_val * x * grad_output / beta_val;
  }
  auto host_trace_fields() const {
    return std::tie(norm_val, beta_val);
  }
};

void smooth_l1_backward_cuda_kernel(TensorIterator& iter, const Scalar& norm, double beta) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "smooth_l1_backward_cuda", [&iter, &norm, beta] {
      auto norm_val = norm.to<scalar_t>();
      scalar_t beta_val(beta);
      gpu_kernel(iter, SmoothL1BackwardFunctor<scalar_t>{norm_val, beta_val});
  });
}

template <typename scalar_t>
struct HuberBackwardFunctor {
  scalar_t norm_val;
  scalar_t delta_val;
  __device__ scalar_t operator()(scalar_t input, scalar_t target, scalar_t grad_output) const {
    const auto x = input - target;
    if (x < -delta_val) {
      return -norm_val * grad_output * delta_val;
    } else if (x > delta_val) {
      return norm_val * grad_output * delta_val;
    } else {
      return norm_val * x * grad_output;
    }
  }
  auto host_trace_fields() const {
    return std::tie(norm_val, delta_val);
  }
};

void huber_backward_cuda_kernel(TensorIterator& iter, const Scalar& norm, double delta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "huber_backward_cuda", [&iter, &norm, delta] {
    auto norm_val = norm.to<scalar_t>();
    scalar_t delta_val(delta);
    gpu_kernel(iter, HuberBackwardFunctor<scalar_t>{norm_val, delta_val});
  });
}

template <typename scalar_t>
struct MseBackwardFunctor {
  scalar_t alpha;
  __device__ scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return alpha * (a - b) * c;
  }
  auto host_trace_fields() const {
    return std::tie(alpha);
  }
};

void mse_backward_cuda_kernel(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "mse_backward_cuda", [&]() {
    auto alpha = value.to<scalar_t>();
    gpu_kernel(iter, MseBackwardFunctor<scalar_t>{alpha});
  });
}

REGISTER_DISPATCH(addcdiv_stub, &addcdiv_cuda_kernel)
REGISTER_DISPATCH(addcmul_stub, &addcmul_cuda_kernel)
REGISTER_DISPATCH(smooth_l1_backward_stub, &smooth_l1_backward_cuda_kernel)
REGISTER_DISPATCH(huber_backward_stub, &huber_backward_cuda_kernel)
REGISTER_DISPATCH(mse_backward_stub, &mse_backward_cuda_kernel)
} // namespace at::native
