#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/NumericUtils.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

template <typename scalar_t>
struct SmoothL1Functor {
  scalar_t beta_val;
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const {
    auto z = ::abs(a - b);
    return z < beta_val ? scalar_t(0.5) * z * z / beta_val : z - scalar_t(0.5) * beta_val;
  }
  auto host_trace_fields() const {
    return std::tie(beta_val);
  }
};

void smooth_l1_kernel_cuda(TensorIteratorBase& iter, double beta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "smooth_l1_cuda", [&iter, beta]() {
    scalar_t beta_val(beta);
    gpu_kernel(iter, SmoothL1Functor<scalar_t>{beta_val});
  });
}

template <typename scalar_t>
struct HuberFunctor {
  scalar_t delta_val;
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const {
    auto z = ::abs(a - b);
    return z < delta_val ? scalar_t(0.5) * z * z : delta_val * (z - scalar_t(0.5) * delta_val);
  }
  auto host_trace_fields() const {
    return std::tie(delta_val);
  }
};

void huber_kernel_cuda(TensorIterator& iter, double delta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "huber_cuda", [&iter, delta] {
    scalar_t delta_val(delta);
    gpu_kernel(iter, HuberFunctor<scalar_t>{delta_val});
  });
}

void mse_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "mse_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      auto diff = a - b;
      return diff * diff;
    });
  });
}

void xlogy_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "xlogy_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t x, scalar_t y) -> scalar_t {
      if (at::_isnan(y)){
        return NAN;
      }
      if (x == 0){
        return 0;
      }
      return x * std::log(y);
    });
  });
}

void xlog1py_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "xlog1py_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t x, scalar_t y) -> scalar_t {
      if (at::_isnan(y)){
        return NAN;
      }
      if (x == 0){
        return 0;
      }
      return x * std::log1p(y);
    });
  });
}

void ldexp_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(0), "ldexp_cuda", [&] {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t x, int exp) -> scalar_t {
      return ::ldexp(x, exp);
    });
  });
}

REGISTER_DISPATCH(smooth_l1_stub, &smooth_l1_kernel_cuda)
REGISTER_DISPATCH(huber_stub, &huber_kernel_cuda)
REGISTER_DISPATCH(mse_stub, &mse_kernel_cuda)
REGISTER_DISPATCH(xlogy_stub, &xlogy_kernel_cuda)
REGISTER_DISPATCH(xlog1py_stub, &xlog1py_kernel_cuda)
REGISTER_DISPATCH(ldexp_stub, &ldexp_kernel_cuda)

// DO NOT ADD ANY NEW KERNELS HERE
// CUDA compilation times grow quickly.  It's perfectly acceptable to have a file per kernel.

} // namespace at::native
