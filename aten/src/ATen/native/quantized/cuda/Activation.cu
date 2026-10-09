#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

namespace at::native {

template <typename scalar_t, typename underlying_t>
struct QReluFunctor {
  int64_t zero_point;
  __device__ scalar_t operator()(scalar_t value) const {
    return scalar_t(std::max<underlying_t>(value.val_, zero_point));
  }
  auto host_trace_fields() const {
    return std::tie(zero_point);
  }
};

Tensor& relu_quantized_cuda_(Tensor& self) {
  const auto zero_point = self.q_zero_point();
  AT_DISPATCH_QINT_TYPES(
    self.scalar_type(), "qrelu_cuda", [&]() {
      auto iter = TensorIterator::unary_op(self, self);
      gpu_kernel(iter, QReluFunctor<scalar_t, underlying_t>{zero_point});
  });
  return self;
}

}  // namespace at::native
