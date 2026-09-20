#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/ForeachMinMaxFunctors.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_clamp_max_native.h>
#include <ATen/ops/_foreach_clamp_min_native.h>
#include <ATen/ops/_foreach_div_native.h>
#include <ATen/ops/_foreach_mul_native.h>
#include <ATen/ops/_foreach_pow_native.h>
#include <ATen/ops/_foreach_sub_native.h>

#include <ATen/ops/empty_like_native.h>
#endif

namespace at::native {

template <typename T, template <class> class Op>
std::vector<Tensor> foreach_binary_op(
    TensorList tensors,
    const Scalar& scalar) {
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors.size());
  for (const auto& t : tensors) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  auto tensor_lists =
      c10::make_nested<Tensor>(tensors.vec(), std::move(vec_res));

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<2>(
      tensor_lists,
      BinaryOpScalarFunctor<
          T,
          /* depth */ 2,
          /* r_args_depth */ 1,
          /* res_arg_index */ 1>(),
      Op<opmath_t>(),
      scalar.to<opmath_t>());
  return std::move(tensor_lists[1]);
}

template <typename T, template <class> class Op>
void foreach_binary_op_(TensorList tensors, const Scalar& scalar) {
  auto tensor_lists = c10::make_nested<Tensor>(tensors.vec());

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<1>(
      tensor_lists,
      BinaryOpScalarFunctor<
          T,
          /* depth */ 1,
          /* r_args_depth */ 1,
          /* res_arg_index */ 0>(),
      Op<opmath_t>(),
      scalar.to<opmath_t>());
  increment_version(tensors);
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_bool_half_bfloat16(
    TensorList tensors,
    const Scalar& scalar) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda",
      [&]() { return foreach_binary_op<scalar_t, Op>(tensors, scalar); });
}

template <template <class> class Op>
void all_types_complex_bool_half_bfloat16_(
    TensorList tensors,
    const Scalar& scalar) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalar); });
}

template <template <class> class Op>
std::vector<Tensor> all_types_half_bfloat16(
    TensorList tensors,
    const Scalar& scalar) {
  return AT_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda",
      [&]() { return foreach_binary_op<scalar_t, Op>(tensors, scalar); });
}

template <template <class> class Op>
void all_types_half_bfloat16_(TensorList tensors, const Scalar& scalar) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalar); });
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_half_bfloat16(
    TensorList tensors,
    const Scalar& scalar) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda",
      [&]() { return foreach_binary_op<scalar_t, Op>(tensors, scalar); });
}

template <template <class> class Op>
void all_types_complex_half_bfloat16_(
    TensorList tensors,
    const Scalar& scalar) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalar); });
}

#define FOREACH_BINARY_OP_SCALAR(FUNCTION, NAME, OP, DIVISION_OP)     \
  void foreach_tensor_##NAME##_scalar_kernel_cuda_(                   \
      TensorList tensors, const Scalar& scalar) {                     \
    check_foreach_api_restrictions(tensors);                          \
    if (!can_use_fast_route(tensors, scalar, DIVISION_OP)) {          \
      return at::native::foreach_tensor_##NAME##_scalar_kernel_slow_( \
          tensors, scalar);                                           \
    }                                                                 \
                                                                      \
    FUNCTION##_<OP>(tensors, scalar);                                 \
  }                                                                   \
                                                                      \
  std::vector<Tensor> foreach_tensor_##NAME##_scalar_kernel_cuda(     \
      TensorList tensors, const Scalar& scalar) {                     \
    check_foreach_api_restrictions(tensors);                          \
    if (!can_use_fast_route(tensors, scalar, DIVISION_OP)) {          \
      return at::native::foreach_tensor_##NAME##_scalar_kernel_slow(  \
          tensors, scalar);                                           \
    }                                                                 \
                                                                      \
    return FUNCTION<OP>(tensors, scalar);                             \
  }

FOREACH_BINARY_OP_SCALAR(
    all_types_complex_bool_half_bfloat16,
    add,
    std::plus,
    /*div_op*/ false);
FOREACH_BINARY_OP_SCALAR(
    all_types_complex_bool_half_bfloat16,
    mul,
    std::multiplies,
    /*div_op*/ false);
// See [Why is foreach_pow's division_op=true?]
FOREACH_BINARY_OP_SCALAR(
    all_types_complex_half_bfloat16,
    pow,
    power_functor,
    /*div_op*/ true);
std::vector<Tensor> foreach_scalar_pow_list_kernel_cuda(
    const Scalar& scalar,
    TensorList exponent) {
  check_foreach_api_restrictions(exponent);
  if (!can_use_fast_route(exponent)) {
    return at::native::foreach_scalar_pow_list_kernel_slow(scalar, exponent);
  }
  return all_types_complex_half_bfloat16<reverse_power_functor>(
      exponent, scalar);
}

// In the case of division, integer inputs will result in float.
// Currently multi tensor apply can only return result of the same type as
// input.
//
// Implement via multiply with reciprocal as it's faster and makes it match
// the behavior of regular Tensor div by scalar.  Loses one bit of
// precision.
Scalar scalar_reciprocal(const Scalar& scalar) {
  if (scalar.isFloatingPoint()) {
    return Scalar(1. / scalar.toDouble());
  } else if (scalar.isIntegral(/*includeBool*/ true)) {
    return Scalar(1. / static_cast<double>(scalar.toLong()));
  } else if (scalar.isComplex()) {
    return Scalar(1. / scalar.toComplexDouble());
  }
  TORCH_INTERNAL_ASSERT(
      false, "division with ", scalar.type(), " not supported");
}

void foreach_tensor_div_scalar_kernel_cuda_(
    TensorList tensors,
    const Scalar& scalar) {
  check_foreach_api_restrictions(tensors);
  if (!can_use_fast_route(tensors, scalar, true)) {
    return at::native::foreach_tensor_mul_scalar_kernel_slow_(
        tensors, scalar_reciprocal(scalar));
  }

  all_types_complex_bool_half_bfloat16_<std::multiplies>(
      tensors, scalar_reciprocal(scalar));
}

std::vector<Tensor> foreach_tensor_div_scalar_kernel_cuda(
    TensorList tensors,
    const Scalar& scalar) {
  check_foreach_api_restrictions(tensors);
  if (!can_use_fast_route(tensors, scalar, true)) {
    return at::native::foreach_tensor_mul_scalar_kernel_slow(
        tensors, scalar_reciprocal(scalar));
  }

  return all_types_complex_bool_half_bfloat16<std::multiplies>(
      tensors, scalar_reciprocal(scalar));
}

// In the case of subtraction, we dont allow scalar to be boolean following the
// torch.sub logic
void foreach_tensor_sub_scalar_kernel_cuda_(
    TensorList tensors,
    const Scalar& scalar) {
  check_foreach_api_restrictions(tensors);
  at::native::sub_check(tensors[0], scalar);

  if (!can_use_fast_route(tensors, scalar)) {
    return at::native::foreach_tensor_sub_scalar_kernel_slow_(tensors, scalar);
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda_",
      [&]() { foreach_binary_op_<scalar_t, std::minus>(tensors, scalar); });
}

std::vector<Tensor> foreach_tensor_sub_scalar_kernel_cuda(
    TensorList tensors,
    const Scalar& scalar) {
  check_foreach_api_restrictions(tensors);
  at::native::sub_check(tensors[0], scalar);

  if (!can_use_fast_route(tensors, scalar)) {
    return at::native::foreach_tensor_sub_scalar_kernel_slow(tensors, scalar);
  }

  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda",
      [&]() {
        return foreach_binary_op<scalar_t, std::minus>(tensors, scalar);
      });
}

// NOTE(crcrpar): `all_types_half_bfloat16` does not cover bool, so temporarily
// set `division_op` to true.
FOREACH_BINARY_OP_SCALAR(all_types_half_bfloat16, clamp_max, minimum, true);
FOREACH_BINARY_OP_SCALAR(all_types_half_bfloat16, clamp_min, maximum, true);

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced sibling of
// foreach_binary_op_ (the in- place form), compiled here so the sibling and the
// real host above instantiate the one kernel (multi_tensor_apply_kernel over
// BinaryOpScalarFunctor; DECISIONS E36): the tape's launch is eager's function
// object, not a twin. multi_tensor_apply_kernel and the metadata structs live
// in MultiTensorApply.cuh's anonymous namespace, one instantiation per
// translation unit, so the entry must be compiled where eager's host is; the
// chunking loop is ti/MultiTensorApplySym.cuh's (the same loop over SymInt
// numels, the metadata block a proxy, the launches through the typed helper).
// Outside a trace the entry runs the same launches in ordinary mode, which is
// how the parity test compares it with the real op.
#include <ATen/cuda/host_trace/ti/ForeachOps.h>
#include <ATen/cuda/host_trace/ti/MultiTensorApplySym.cuh>

#include <functional>
#include <vector>

namespace at::native {
namespace {

namespace ht = at::cuda::host_trace;

// ForeachBinaryOpScalar.cu foreach_binary_op_ (the in-place form, depth 1)
template <typename T, template <class> class Op>
void foreach_binary_op_sym_(TensorList tensors, const Scalar& scalar) {
  auto tensor_lists = c10::make_nested<Tensor>(tensors.vec());
  using opmath_t = at::opmath_type<T>;
  using functor_t = BinaryOpScalarFunctor<
      T,
      /*depth*/ 1,
      /*r_args_depth*/ 1,
      /*res_arg_index*/ 0>;
  ht::multi_tensor_apply_sym<1>(
      multi_tensor_apply_kernel<
          TensorListMetadata<1>,
          functor_t,
          Op<opmath_t>,
          opmath_t>,
      {true},
      tensor_lists,
      functor_t(),
      Op<opmath_t>(),
      scalar.to<opmath_t>());
  increment_version(tensors);
}

void foreach_add_scalar_sym_(TensorList tensors, const Scalar& scalar) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda_",
      [&]() { foreach_binary_op_sym_<scalar_t, std::plus>(tensors, scalar); });
}

} // anonymous namespace
} // namespace at::native

namespace at::cuda::host_trace::ti {

void foreach_add_scalar_traced_(
    at::TensorList tensors,
    const at::Scalar& scalar) {
  at::native::check_foreach_api_restrictions(tensors);
  if (!fast_path_restrictions_sym(
          {tensors}, {scalar}, /*promotes_integer_inputs_to_float=*/false)) {
    // ForeachOpsKernels.cpp foreach_tensor_add_scalar_kernel_slow_: the
    // per-tensor op through the dispatcher (its own sibling under a trace)
    for (const auto& t : tensors) {
      t.add_(scalar);
    }
    return;
  }
  at::native::foreach_add_scalar_sym_(tensors, scalar);
}

} // namespace at::cuda::host_trace::ti
