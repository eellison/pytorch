#pragma once
// add.Tensor's CUDA body written once over the host policy H: an example of
// an op body using the kernel_context boundary end to end. The functor is the
// one UfuncCUDA_add.cu generates.
//
// Not shown: the CPU-scalar routes (CUDAFunctorOnSelf_add/OnOther_add), alpha
// checks, and dtypes beyond floating point.
//
// Preview only: not in any build target and not tested.
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorIteratorSym.h>
#include <ATen/native/HostPolicy.h>
#include <ATen/native/cuda/LoopsHost.cuh>
#include <ATen/native/ufunc/add.h>

namespace at::native::host {

template <typename scalar_t>
struct CUDAFunctor_add {
  using opmath_t = at::opmath_type<scalar_t>;
  opmath_t alpha_;
  CUDAFunctor_add(opmath_t alpha) : alpha_(alpha) {}
  __device__ scalar_t operator()(scalar_t self, scalar_t other) const {
    return ufunc::add(static_cast<opmath_t>(self), static_cast<opmath_t>(other), alpha_);
  }
};

// Output metadata is final when this is called.
template <class H, class Iter>
Tensor add_kernel(H& h, Iter& iter, const Scalar& alpha) {
  Tensor out = iter.output();
  {
    auto k = h.kernel_context(); // EagerHost: empty, compiles away
    if (!k) {
      return out; // SymHost without a recorder: fake/meta stops here
    }
    // Kernel choice. With a recorder, every guard raised from here to the
    // closing brace selects this op's kernel and is not a graph guard.
    if constexpr (is_sym_v<H>) {
      iter.coalesce_dimensions(); // eager's build() has already coalesced
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "add_host", [&] {
      using opmath_t = at::opmath_type<scalar_t>;
      gpu_kernel(h, iter, CUDAFunctor_add<scalar_t>(alpha.to<opmath_t>()));
    });
  }
  return out;
}

template <class H>
Tensor add_body(H& h, const Tensor& self, const Tensor& other, const Scalar& alpha) {
  // Graph level: type promotion, output shape, strides and allocation.
  TensorIteratorConfig config;
  config.set_check_mem_overlap(true)
      .promote_inputs_to_common_dtype(true)
      .cast_common_dtype_to_outputs(true)
      .enforce_safe_casting_to_output(true)
      .add_owned_output(Tensor())
      .add_owned_const_input(self)
      .add_owned_const_input(other);
  if constexpr (is_sym_v<H>) {
    TensorIteratorSym iter(config);
    return add_kernel(h, iter, alpha);
  } else {
    auto iter = config.build();
    return add_kernel(h, iter, alpha);
  }
}

} // namespace at::native::host
