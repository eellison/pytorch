#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Fill.h>
#include <c10/core/Scalar.h>

namespace at::native {

template<typename scalar_t>
struct FillFunctor {
  FillFunctor(scalar_t v): value(v) {}
  __device__ __forceinline__ scalar_t operator() () const {
    return value;
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

// ---- host tracing (ATen/cuda/host_trace): the traced sibling of fill_kernel_cuda (and TensorFactories.cu's zero_cuda_), compiled here
// so the sibling and the real host above instantiate the one kernel over FillFunctor
// (DECISIONS E36): the tape's launch is eager's function object, not a twin. Outside a trace the
// entry runs the same launches in ordinary mode, which is how the parity test compares it with
// the real op.
#include <ATen/cuda/host_trace/Recorder.h>
#include <ATen/cuda/host_trace/ti/LoopsSym.cuh>
#include <ATen/cuda/host_trace/ti/Ops.h>

#include <ATen/cuda/CUDAContext.h>

namespace at::cuda::host_trace::ti {

Tensor& fill_traced(Tensor& self, const Scalar& value) {
  const ScalarType dtype = self.scalar_type();
  if (isComplexType(dtype) || isFloat8Type(dtype) || isQIntType(dtype) || isBitsType(dtype) || dtype == ScalarType::Float4_e2m1fn_x2) {
    decline(c10::str("host_trace: fill_ of a ", dtype, " tensor is not traced (declined)"));
  }
  TensorIteratorSymConfig config;
  config.check_mem_overlap_ = false;
  config.check_all_same_dtype_ = false;
  config.resize_outputs_ = false;
  TensorIteratorSym iter;
  iter.add_output(self);
  iter.build(config);
  AT_DISPATCH_V2(dtype, "fill_traced", AT_WRAP([&] {
    gpu_kernel(iter, at::native::FillFunctor<scalar_t>(value.to<scalar_t>()));
  }), AT_EXPAND(AT_ALL_TYPES), kBool, kHalf, kBFloat16, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  return self;
}

Tensor& zero_traced(Tensor& self) {
  // zero_cuda_ memsets when the data pointer is not null (an empty tensor's
  // may be) and the tensor is dense; the trace guards the element count
  // instead of reading a pointer, and the memset's byte count follows it
  const bool has_elements = self.sym_numel().sym_ne(0).guard_bool(__FILE__, __LINE__);
  if (has_elements && self.is_non_overlapping_and_dense()) {
    const c10::SymInt nbytes = self.sym_numel() * static_cast<int64_t>(self.dtype().itemsize());
    memset_async(sym_mutable_data_ptr(self), 0, nbytes, at::cuda::getCurrentCUDAStream(self.device().index()));
    return self;
  }
  return fill_traced(self, 0);
}

} // namespace at::cuda::host_trace::ti
