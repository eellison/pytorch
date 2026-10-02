#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Fill.h>
#include <c10/core/Scalar.h>
#if !defined(USE_ROCM)
#include <ATen/cuda/host_trace/LoopsSym.cuh>
#include <ATen/cuda/host_trace/Ops.h>
#endif

namespace at::native {

template<typename scalar_t>
struct FillFunctor {
  FillFunctor(scalar_t v): value(v) {}
  __device__ __forceinline__ scalar_t operator() () const {
    return value;
  }
  auto host_trace_fields() const {
    return std::tie(value);
  }
  private:
    scalar_t value;
};

void fill_kernel_cuda(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_V2(iter.dtype(), "fill_cuda", AT_WRAP([&]() {
    gpu_kernel(iter, FillFunctor<scalar_t>(value.to<scalar_t>()));
  }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kComplexHalf, kBComplex32, kBool, kHalf, kBFloat16, AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

REGISTER_DISPATCH(fill_stub, &fill_kernel_cuda)

} // namespace at::native

#if !defined(USE_ROCM)
// Traced host (ATen/cuda/host_trace/Ops.h)
namespace at::cuda::host_trace {

// fill_kernel_cuda over fill_out's iterator
TensorBase fill_(Recorder& rec, const TensorBase& self, const Scalar& value) {
  auto iter = TensorIteratorSym::pointwise_op(rec, {self}, {self.scalar_type()}, {});
  AT_DISPATCH_V2(iter.dtype(0), "fill_cuda", AT_WRAP([&]() {
    gpu_kernel(rec, iter, at::native::FillFunctor<scalar_t>(value.to<scalar_t>()));
  }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kComplexHalf, kBComplex32, kBool, kHalf, kBFloat16, AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  return self;
}

} // namespace at::cuda::host_trace
#endif
