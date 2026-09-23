#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/NumericUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>


namespace at::native {

namespace {

// where_kernel_impl's device function as a named functor: the traced sibling
// at the end of this file launches it too (DECISIONS E36)
template <typename scalar_t>
struct WhereFunctor {
  __device__ scalar_t operator()(bool cond_val, scalar_t self_val, scalar_t other_val) const {
    return cond_val ? self_val : other_val;
  }
};

void where_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_V2(opaqueScalarType(iter.dtype()), "where_cuda", [&] {
      gpu_kernel_opaque(iter, WhereFunctor<scalar_t>{});
  }, AT_EXPAND(AT_OPAQUE_TYPES));
}

void isposinf_kernel_impl(TensorIteratorBase &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isposinf_cuda", [&]() {
    gpu_kernel(
      iter,
      [] GPU_LAMBDA (scalar_t a) -> bool { return a == std::numeric_limits<scalar_t>::infinity(); }
    );
  });
}

void isneginf_kernel_impl(TensorIteratorBase &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isneginf_cuda", [&]() {
    gpu_kernel(
      iter,
      [] GPU_LAMBDA (scalar_t a) -> bool { return a == -std::numeric_limits<scalar_t>::infinity(); }
    );
  });
}

void clamp_kernel_impl(TensorIteratorBase& iter) {
  AT_DISPATCH_V2(iter.common_dtype(), "clamp_cuda", AT_WRAP([&] {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t v, scalar_t lower, scalar_t upper) -> scalar_t {
      scalar_t result = (v < lower) ? lower : v;
      result = (upper < result) ? upper : result;

      if constexpr (std::numeric_limits<scalar_t>::has_quiet_NaN) {
        result = at::_isnan(upper) ? upper : result;
        result = at::_isnan(lower) ? lower : result;
        result = at::_isnan(v) ? v : result;
      }

      return result;
    });
  }), AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES), kHalf, kBFloat16);
}

// launch_clamp_scalar's device function: the members in the order its lambda captured them
template <typename scalar_t>
struct ClampScalarFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  at::native::detail::ClampLimits minmax;
  opmath_t lim0_val;
  opmath_t lim1_val;
  __device__ scalar_t operator()(scalar_t v) const {
    opmath_t val = static_cast<opmath_t>(v);
    // Propagate nan, which doesn't propagate automatically for ROCm
    if (_isnan(static_cast<opmath_t>(v))) {
      return v;
    } else if (minmax==at::native::detail::ClampLimits::Min){
      if (val == lim0_val)
        return v;
      return (val < lim0_val) ? static_cast<scalar_t>(lim0_val) : v;
    } else if (minmax==at::native::detail::ClampLimits::Max){
      if (val == lim0_val)
        return v;
      return (val > lim0_val) ? static_cast<scalar_t>(lim0_val) : v;
    } else {
      // The following replaces std::clamp(val, low, high) and is a viable solution for
      // both CUDA and ROCm since std::clamp and this replacement generates the same PTX.
      // The replacement should generate the same PTX as std::clamp. See https://godbolt.org/z/Wde9KW3v4
      opmath_t result = (val < lim0_val) ? lim0_val : val;
      return scalar_t((lim1_val < result) ? lim1_val : result);
    }
  }
};

void inline launch_clamp_scalar(TensorIteratorBase& iter, Scalar lim0, Scalar lim1, at::native::detail::ClampLimits minmax){
  AT_DISPATCH_V2(iter.common_dtype(), "clamp_scalar_cuda", AT_WRAP([&] {
    using opmath_t = at::opmath_type<scalar_t>;
    auto lim0_val = lim0.to<opmath_t>();
    auto lim1_val = lim1.to<opmath_t>();

    gpu_kernel(iter, ClampScalarFunctor<scalar_t>{minmax, lim0_val, lim1_val});
  }), AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES), kHalf, kBFloat16);
}


void clamp_scalar_kernel_impl(TensorIteratorBase& iter, const Scalar& min, const Scalar& max) {
  launch_clamp_scalar(iter, min, max, at::native::detail::ClampLimits::MinMax);
}

void clamp_min_scalar_kernel_impl(TensorIteratorBase& iter, Scalar min) {
  launch_clamp_scalar(iter, min, min, at::native::detail::ClampLimits::Min);
}

void clamp_max_scalar_kernel_impl(TensorIteratorBase& iter, Scalar max) {
  launch_clamp_scalar(iter, max, max, at::native::detail::ClampLimits::Max);
}

} // anonymous namespace


REGISTER_DISPATCH(where_kernel, &where_kernel_impl)
REGISTER_DISPATCH(isposinf_stub, &isposinf_kernel_impl)
REGISTER_DISPATCH(isneginf_stub, &isneginf_kernel_impl)
REGISTER_DISPATCH(clamp_stub, &clamp_kernel_impl)
REGISTER_DISPATCH(clamp_scalar_stub, &clamp_scalar_kernel_impl)
REGISTER_DISPATCH(clamp_min_scalar_stub, &clamp_min_scalar_kernel_impl)
REGISTER_DISPATCH(clamp_max_scalar_stub, &clamp_max_scalar_kernel_impl)

struct Msg {
 static constexpr size_t MAX_MSG_LENGTH = 256;
 char msg[MAX_MSG_LENGTH];
};
template <typename scalar_t>
__global__ void _assert_async_cuda_kernel(const scalar_t* input, Msg msg) {
  CUDA_KERNEL_ASSERT_MSG(input[0] != 0, msg.msg);
}

__global__ void _assert_async_cuda_kernel(const c10::complex<float>* input, Msg msg) {
  CUDA_KERNEL_ASSERT_MSG(input[0] != c10::complex<float>(0, 0), msg.msg);
}
__global__ void _assert_async_cuda_kernel(const c10::complex<double>* input, Msg msg) {
  CUDA_KERNEL_ASSERT_MSG(input[0] != c10::complex<double>(0, 0), msg.msg);
}

void _assert_async_msg_cuda(const Tensor& self_tensor, std::string_view assert_msg) {
  const TensorBase &self = get_tensor_base(self_tensor);
  auto n = self.numel();
  TORCH_CHECK(n != 0, "Boolean value of Tensor with no values is ambiguous");
  TORCH_CHECK(n < 2, "Boolean value of Tensor with more than one value is ambiguous");
  auto stream = at::cuda::getCurrentCUDAStream();
  Msg msg;
  size_t copy_length = assert_msg.length();
  TORCH_CHECK(copy_length < Msg::MAX_MSG_LENGTH - 1, "Message length must be smaller than " + std::to_string(Msg::MAX_MSG_LENGTH - 1));
  std::copy_n(assert_msg.data(), copy_length, msg.msg);
  msg.msg[copy_length] = '\0';  // Ensure null-termination
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, self.scalar_type(), "_assert_async_cuda", [&] {
    _assert_async_cuda_kernel<<<1, 1, 0, stream>>>(self.const_data_ptr<scalar_t>(), msg);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

void _assert_async_cuda(const Tensor& self_tensor) {
  _assert_async_msg_cuda(self_tensor, "");
}

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced sibling of launch_clamp_scalar, compiled here
// so the sibling and the real host above instantiate the one kernel over ClampScalarFunctor
// (DECISIONS E36): the tape's launch is eager's function object, not a twin. Outside a trace the
// entry runs the same launches in ordinary mode, which is how the parity test compares it with
// the real op.
#include <ATen/cuda/host_trace/ti/LoopsSym.cuh>
#include <ATen/cuda/host_trace/ti/Ops.h>

#include <c10/util/TypeSafeSignMath.h>

#include <limits>

namespace at::cuda::host_trace::ti {

namespace {
// TensorCompare.cpp prepare_clamp_bound / clamp_bound_range for the bounds
// that reach the kernel here: an integral bound on an integral dtype (a
// floating bound on one is promotion and declined before this)
std::optional<Scalar> prepare_clamp_bound(const std::optional<Scalar>& bound, ScalarType dtype, bool is_lower_bound) {
  if (!bound.has_value()) {
    return std::nullopt;
  }
  if (!isIntegralType(dtype, /*includeBool=*/false)) {
    return bound;
  }
  const int64_t value = bound->toLong();
  bool below = false;
  bool above = false;
  AT_DISPATCH_V2(dtype, "prepare_clamp_bound", AT_WRAP([&] {
    below = c10::less_than_lowest<scalar_t>(value);
    above = c10::greater_than_max<scalar_t>(value);
  }), AT_EXPAND(AT_INTEGRAL_TYPES_V2));
  if (!below && !above) {
    return bound;
  }
  const bool is_noop = is_lower_bound ? below : above;
  TORCH_CHECK(
      is_noop,
      "Clamp ",
      is_lower_bound ? "min" : "max",
      " value ",
      bound->toDouble(),
      " is outside the representable range of ",
      dtype);
  return std::nullopt;
}

bool is_nan_bound(const std::optional<Scalar>& bound) {
  return bound.has_value() && bound->toDouble() != bound->toDouble();
}
} // namespace

Tensor clamp_scalar_traced(const Tensor& self, const std::optional<Scalar>& min, const std::optional<Scalar>& max, const Tensor& out) {
  using at::native::detail::ClampLimits;
  TensorIteratorSym iter = TensorIteratorSym::unary_op(out, self);
  const ScalarType dtype = iter.common_dtype();
  if (isComplexType(dtype) || dtype == kBool || isFloat8Type(dtype) || isBitsType(dtype)) {
    decline(c10::str("host_trace: clamp on ", dtype, " is not traced (declined)"));
  }
  Tensor result = iter.output();
  if (is_nan_bound(min) || is_nan_bound(max)) {
    return fill_traced(result, std::numeric_limits<double>::quiet_NaN());
  }
  const auto lo = prepare_clamp_bound(min, dtype, /*is_lower_bound=*/true);
  const auto hi = prepare_clamp_bound(max, dtype, /*is_lower_bound=*/false);
  if (!lo && !hi) {
    if (!result.is_same(self)) {
      copy_traced(result, self);
    }
    return result;
  }
  const Scalar lim0 = lo ? *lo : *hi;
  const Scalar lim1 = hi ? *hi : *lo;
  const ClampLimits minmax = (lo && hi) ? ClampLimits::MinMax : (lo ? ClampLimits::Min : ClampLimits::Max);
  AT_DISPATCH_V2(dtype, "clamp_scalar_traced", AT_WRAP([&] {
    using opmath_t = at::opmath_type<scalar_t>;
    Zeroed<at::native::ClampScalarFunctor<scalar_t>> f(minmax, lim0.to<opmath_t>(), lim1.to<opmath_t>());
    gpu_kernel(iter, f.get());
  }), AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES), kHalf, kBFloat16);
  return result;
}

Tensor where_traced(const Tensor& condition, const Tensor& self, const Tensor& other) {
  // TensorCompare.cpp where_self_out's iterator (check_all_same_dtype off:
  // the bool condition beside the operands, which the registry's entry
  // brought to one dtype as the real op does) and where_kernel_impl's launch
  // over the opaque type of the result's element size
  TORCH_CHECK(
      condition.scalar_type() == kBool,
      "where expected condition to be a boolean tensor, but got a tensor with dtype ",
      condition.scalar_type());
  TORCH_INTERNAL_ASSERT(self.scalar_type() == other.scalar_type());
  TensorIteratorSymConfig config;
  config.check_all_same_dtype_ = false;
  config.static_dtype_ = self.scalar_type();
  TensorIteratorSym iter = TensorIteratorSym::ternary_op(Tensor(), condition, self, other, config);
  AT_DISPATCH_V2(opaqueScalarType(iter.dtype()), "where_traced", AT_WRAP([&] {
    gpu_kernel_nocast(iter, at::native::WhereFunctor<scalar_t>{});
  }), AT_EXPAND(AT_OPAQUE_TYPES));
  return iter.output();
}

} // namespace at::cuda::host_trace::ti
