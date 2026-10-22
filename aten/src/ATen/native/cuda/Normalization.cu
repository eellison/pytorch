#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/Normalization.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Resize.h>
#include <ATen/native/cuda/Normalization.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <utility>

#if !defined(USE_ROCM)
#include <ATen/TensorUtils.h>
#include <ATen/cuda/host_trace/LoopsSym.cuh>
#include <ATen/cuda/host_trace/Ops.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_batch_norm_with_update_native.h>
#include <ATen/ops/batch_norm_backward_native.h>
#include <ATen/ops/batch_norm_backward_elemt_native.h>
#include <ATen/ops/batch_norm_backward_reduce_native.h>
#include <ATen/ops/batch_norm_elemt_native.h>
#include <ATen/ops/batch_norm_gather_stats_native.h>
#include <ATen/ops/batch_norm_gather_stats_with_counts_native.h>
#include <ATen/ops/batch_norm_stats_native.h>
#include <ATen/ops/batch_norm_update_stats_native.h>
#include <ATen/ops/cudnn_batch_norm.h>
#include <ATen/ops/cudnn_batch_norm_backward.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/miopen_batch_norm.h>
#include <ATen/ops/miopen_batch_norm_backward.h>
#include <ATen/ops/native_batch_norm_backward_native.h>
#include <ATen/ops/native_batch_norm_native.h>
#include <ATen/ops/scalar_tensor.h>
#endif

namespace at::native {

namespace {

ScalarType first_type() {
  return ScalarType::Undefined;
}

template <typename... Args>
ScalarType first_type(const Tensor& arg, const Args&... parameters) {
  return arg.defined() ? arg.scalar_type() : first_type(parameters...);
}

// A transform is mixed type if the parameters are higher precision than the input
template <typename... Args>
bool is_mixed_type(const Tensor& input, const Args&... parameters) {
  const auto parameter_type = first_type(parameters...);
  return ((parameter_type != ScalarType::Undefined) &&
          (parameter_type != input.scalar_type()));
}

inline bool batch_norm_use_channels_last_kernels(const at::Tensor& self) {
  return (
    self.is_contiguous(at::MemoryFormat::ChannelsLast) ||
    self.is_contiguous(at::MemoryFormat::ChannelsLast3d) ||
    (self.is_contiguous() && self.strides()[1] == 1)
  );
}

enum class Impl {
  Contiguous,
  ChannelsLast,
  General,
};

inline Impl batch_norm_choose_impl(const Tensor& self) {
  if (!at::cuda::detail::canUse32BitIndexMath(self)) {
    return Impl::General;
  }

  if (self.is_contiguous()) {
    return self.strides()[1] == 1 ? Impl::ChannelsLast : Impl::Contiguous;
  }

  if (self.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    return Impl::ChannelsLast;
  }

  return Impl::General;
}

inline Impl batch_norm_choose_impl(const Tensor& in1, const Tensor& in2) {
  auto imp1 = batch_norm_choose_impl(in1);
  if (imp1 == Impl::General) {
    return imp1;
  }
  auto imp2 = batch_norm_choose_impl(in2);
  return imp1 == imp2 ? imp1 : Impl::General;
}

void batch_norm_elementwise(
    const Tensor& out, const Tensor& self, const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt, const Tensor& mean_, const Tensor& invstd_) {
  switch (batch_norm_choose_impl(self)) {
  case Impl::Contiguous: {
    c10::MaybeOwned<Tensor> weight = at::borrow_from_optional_tensor(weight_opt);
    c10::MaybeOwned<Tensor> bias = at::borrow_from_optional_tensor(bias_opt);
    resize_output(out, self.sizes());
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(),
                                    "batch_norm_elementwise_cuda", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      const bool mixed_type = is_mixed_type(self, *weight, *bias);
      if (mixed_type) {
        batch_norm_elemt_cuda_template<scalar_t, accscalar_t, int32_t>(
            out, self, *weight, *bias, mean_, invstd_);
      } else {
        batch_norm_elemt_cuda_template<scalar_t, scalar_t, int32_t>(
            out, self, *weight, *bias, mean_, invstd_);
      }
    });
    return;
  }
  case Impl::ChannelsLast: {
    auto weight = at::borrow_from_optional_tensor(weight_opt);
    auto bias = at::borrow_from_optional_tensor(bias_opt);

    if (resize_output_check(out, self.sizes())) {
        resize_impl_cuda_(out.unsafeGetTensorImpl(), self.sizes(), self.strides());
    }
    if ((out.strides() == self.strides()) &&
        (!weight->defined() || weight->is_contiguous()) &&
        (!bias->defined() || bias->is_contiguous()) &&
        (!mean_.defined() || mean_.is_contiguous()) &&
        (!invstd_.defined() || invstd_.is_contiguous())) {
      batch_norm_elemt_channels_last_cuda_template(
          out, self, *weight, *bias, mean_, invstd_);
      return;
    }
    [[fallthrough]];
  }
  case Impl::General: {
    const int64_t ndim = self.dim();
    DimVector sizes(ndim, 1), strides(ndim, 0);
    // Helper to convert 1d tensors to an nd tensor that broadcasts with input
    // All elements go into the channel dimension
    auto as_nd = [&](const Tensor& t) {
      TORCH_INTERNAL_ASSERT(t.defined() && t.dim() == 1);
      sizes[1] = t.sizes()[0];
      strides[1] = t.strides()[0];
      return t.as_strided(sizes, strides);
    };

    auto weight = weight_opt.has_value() && weight_opt->defined() ?
        as_nd(*weight_opt) : at::scalar_tensor(1, mean_.options());
    auto bias = bias_opt.has_value() && bias_opt->defined() ?
        as_nd(*bias_opt) : at::scalar_tensor(0, mean_.options());
    auto mean = as_nd(mean_);
    auto invstd = as_nd(invstd_);

    auto iter = TensorIteratorConfig()
        .add_output(out)
        .add_const_input(self)
        .add_const_input(weight)
        .add_const_input(bias)
        .add_const_input(mean)
        .add_const_input(invstd)
        .check_all_same_dtype(false)
        .promote_inputs_to_common_dtype(false)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(),
                                    "batch_norm_elementwise_cuda", [&] {
      using acc_t = at::acc_type<scalar_t, true>;
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t input, acc_t weight, acc_t bias,
                                      acc_t mean, acc_t invstd) -> scalar_t {
        return (input - mean) * weight * invstd + bias;
      });
    });
    return;
  }
  }
}

template <typename scalar_t, typename accscalar_t>
struct BatchNormElementwiseBackwardTrainFunctor {
  static constexpr bool host_trace_sizes = true;
  accscalar_t norm_fct;
  __device__ scalar_t operator()(scalar_t gO, scalar_t input, accscalar_t weight,
                                 accscalar_t mean, accscalar_t invstd,
                                 accscalar_t xmu, accscalar_t dy) const {
    auto factor_1_c = invstd * invstd * xmu * norm_fct;
    auto factor_2_c = weight * invstd;
    auto m_dy_c = dy * norm_fct;
    return (gO - m_dy_c - (input - mean) * factor_1_c) * factor_2_c;
  }
};

Tensor batch_norm_elementwise_backward_train(
    const Tensor& grad_out, const Tensor& input, const Tensor& mean, const Tensor& invstd,
    const Tensor& weight, const Tensor& sum_dy, const Tensor& sum_dy_xmu) {
  switch (batch_norm_choose_impl(input, grad_out)) {
  case Impl::Contiguous: {
    return AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
                                           "batch_norm_backward_elemt", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      const bool mixed_type = is_mixed_type(input, weight);
      if (mixed_type) {
        return batch_norm_backward_elemt_cuda_template<scalar_t, accscalar_t, int32_t>(
            grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu);
      } else {
        return batch_norm_backward_elemt_cuda_template<scalar_t, scalar_t, int32_t>(
            grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu);
      }
    });
  }
  case Impl::ChannelsLast: {
    if ((!weight.defined() || weight.is_contiguous()) &&
        mean.is_contiguous() && invstd.is_contiguous()) {
      return batch_norm_backward_elemt_channels_last_cuda_template(
          grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu);
    }
    [[fallthrough]];
  }
  case Impl::General: {
    const auto ndim = input.dim();
    DimVector sizes(ndim, 1), strides(ndim, 0);
    auto as_nd = [&](const Tensor& t) {
      TORCH_INTERNAL_ASSERT(t.defined() && t.dim() == 1);
      sizes[1] = t.sizes()[0];
      strides[1] = t.strides()[0];
      return t.as_strided(sizes, strides);
    };
    auto invstd_nd = as_nd(invstd);
    auto mean_nd = as_nd(mean);
    auto sum_dy_nd = as_nd(sum_dy);
    auto sum_dy_xmu_nd = as_nd(sum_dy_xmu);
    auto weight_nd = weight.defined() ? as_nd(weight) :
        at::scalar_tensor(1.0, input.options().dtype(mean.scalar_type()));

    Tensor grad_input = at::empty(input.sizes(), grad_out.options().memory_format(input.suggest_memory_format()));
    auto iter = TensorIteratorConfig()
        .add_output(grad_input)
        .add_const_input(grad_out)
        .add_const_input(input)
        .add_const_input(weight_nd)
        .add_const_input(mean_nd)
        .add_const_input(invstd_nd)
        .add_const_input(sum_dy_xmu_nd)
        .add_const_input(sum_dy_nd)
        .check_all_same_dtype(false)
        .promote_inputs_to_common_dtype(false)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, grad_out.scalar_type(),
                                    "batch_norm_eval_backward", [&]{
      using accscalar_t = at::acc_type<scalar_t, true>;
      auto norm_fct = static_cast<accscalar_t>(1.0 / (input.numel() /input.size(1)) );
      gpu_kernel(iter, BatchNormElementwiseBackwardTrainFunctor<scalar_t, accscalar_t>{norm_fct});
    });
    return grad_input;
  }
  }
  TORCH_INTERNAL_ASSERT(false);
}

Tensor batch_norm_elementwise_backward_eval(
    const Tensor& grad_out, const Tensor& input,
    const Tensor& invstd, const Tensor& weight) {
  const auto ndim = input.dim();
  DimVector shape(ndim, 1), strides(ndim, 0);
  shape[1] = invstd.sizes()[0];
  strides[1] = invstd.strides()[0];
  auto invstd_nd = invstd.as_strided(shape, strides);
  Tensor grad_input = at::empty(input.sizes(), grad_out.options());

  if (weight.defined()) {
    strides[1] = weight.strides()[0];
    auto weight_nd = weight.as_strided(shape, strides);
    auto iter = TensorIteratorConfig()
        .add_output(grad_input)
        .add_const_input(grad_out)
        .add_const_input(invstd_nd)
        .add_const_input(weight_nd)
        .check_all_same_dtype(false)
        .promote_inputs_to_common_dtype(false)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, grad_out.scalar_type(),
                                    "batch_norm_eval_backward", [&]{
      using accscalar_t = at::acc_type<scalar_t, true>;
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t gO, accscalar_t invstd, accscalar_t weight)
                 -> scalar_t {
          return gO * weight * invstd;
      });
    });
  } else {
    auto iter = TensorIteratorConfig()
        .add_output(grad_input)
        .add_const_input(grad_out)
        .add_const_input(invstd_nd)
        .check_all_same_dtype(false)
        .promote_inputs_to_common_dtype(false)
        .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, grad_out.scalar_type(),
                                    "batch_norm_eval_backward", [&]{
      using accscalar_t = at::acc_type<scalar_t, true>;
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t gO, accscalar_t invstd) -> scalar_t {
          return gO * invstd;
      });
    });
  }
  return grad_input;
}


void batch_norm_mean_var(const Tensor& self, Tensor& save_mean, Tensor& save_var) {
  // NOTE: Epsilon is only used for InvStd, not Var. The value here is ignored.
  const double dummy_epsilon = 1e-5;
  switch (batch_norm_choose_impl(self)) {
  case Impl::Contiguous: {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, self.scalar_type(), "batch_norm_stats_cuda", [&] {
      batch_norm_stats_cuda_template<scalar_t, int32_t, Var>(
          save_mean, save_var, self, dummy_epsilon);
    });
    return;
  }
  case Impl::ChannelsLast: {
    if ((!save_mean.defined() || save_mean.is_contiguous()) &&
        (!save_var.defined() || save_var.is_contiguous())) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          kHalf, kBFloat16, self.scalar_type(), "batch_norm_stats_cuda", [&] {
        batch_norm_stats_channels_last_cuda_template<scalar_t, Var>(
            save_mean, save_var, self, dummy_epsilon);
      });
      return;
    }
    [[fallthrough]];
  }
  case Impl::General: {
    const int64_t ndim = self.dim();
    DimVector reduce_dims(ndim - 1);
    reduce_dims[0] = 0;
    for (int64_t i = 2; i < ndim; ++i) {
      reduce_dims[i - 1] = i;
    }

    // For some reason this isn't an actual operator but it exists anyway...
    at::native::var_mean_out(save_var, save_mean, self, /*dims=*/reduce_dims,
                            /*unbiased=*/false, /*keepdim=*/false);
    return;
  }
  }
}

template <typename scalar_t, typename acc_t>
struct BatchNormUpdateStatsFunctor {
  static constexpr bool host_trace_sizes = true;
  acc_t bessel_correction_factor;
  acc_t momentum;
  __device__ thrust::tuple<scalar_t, scalar_t> operator()(acc_t mean, acc_t var, scalar_t running_mean, scalar_t running_var) const {
    const auto unbiased_var = var * bessel_correction_factor;
    return thrust::tuple<scalar_t, scalar_t>{
      mean * momentum + (1 - momentum) * running_mean,
      unbiased_var * momentum + (1 - momentum) * running_var,
    };
  }
};

void batch_norm_update_stats(
    const Tensor& save_mean, const Tensor& save_var,
    const Tensor& running_mean, const Tensor& running_var,
    double momentum_, int64_t N) {

  auto iter = TensorIteratorConfig()
      .add_output(running_mean)
      .add_output(running_var)
      .add_const_input(save_mean)
      .add_const_input(save_var)
      .add_const_input(running_mean)
      .add_const_input(running_var)
      .check_all_same_dtype(false)
      .promote_inputs_to_common_dtype(false)
      .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, running_mean.scalar_type(),
                                  "batch_norm_update_stats_cuda", [&] {
      using acc_t = at::acc_type<scalar_t, true>;
      const auto bessel_correction_factor = static_cast<acc_t>(
          static_cast<double>(N) / static_cast<double>(N - 1));
      const auto momentum = static_cast<acc_t>(momentum_);
      gpu_kernel_multiple_outputs(
          iter, BatchNormUpdateStatsFunctor<scalar_t, acc_t>{bessel_correction_factor, momentum});
  });
}

template <typename scalar_t, typename acc_t>
struct BatchNormUpdateStatsAndInvertFunctor {
  static constexpr bool host_trace_sizes = true;
  acc_t bessel_correction_factor;
  acc_t momentum;
  acc_t eps;
  __device__ thrust::tuple<scalar_t, scalar_t, acc_t> operator()(acc_t mean, acc_t var, scalar_t running_mean, scalar_t running_var) const {
    const auto unbiased_var = var * bessel_correction_factor;
    return thrust::tuple<scalar_t, scalar_t, acc_t>{
      mean * momentum + (1 - momentum) * running_mean,
      unbiased_var * momentum + (1 - momentum) * running_var,
      c10::cuda::compat::rsqrt(var + eps)
    };
  }
};

void batch_norm_update_stats_and_invert(
    const Tensor& save_mean, const Tensor& save_var,
    const Tensor& running_mean, const Tensor& running_var,
    double momentum_, double epsilon, int64_t N) {

  auto iter = TensorIteratorConfig()
      .add_output(running_mean)
      .add_output(running_var)
      .add_output(save_var)
      .add_const_input(save_mean)
      .add_const_input(save_var)
      .add_const_input(running_mean)
      .add_const_input(running_var)
      .check_all_same_dtype(false)
      .promote_inputs_to_common_dtype(false)
      .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, running_mean.scalar_type(),
                                  "batch_norm_update_stats_cuda", [&] {
      using acc_t = at::acc_type<scalar_t, true>;
      const auto bessel_correction_factor = static_cast<acc_t>(
          static_cast<double>(N) / static_cast<double>(N - 1));
      const auto eps = static_cast<acc_t>(epsilon);
      const auto momentum = static_cast<acc_t>(momentum_);
      gpu_kernel_multiple_outputs(
          iter, BatchNormUpdateStatsAndInvertFunctor<scalar_t, acc_t>{bessel_correction_factor, momentum, eps});
  });
}

template <typename scalar_t, typename acc_t>
struct BatchNormInvStdFunctor {
  __device__ acc_t operator()(scalar_t var) const {
    return c10::cuda::compat::rsqrt(var + eps);
  }
  acc_t eps;
  auto host_trace_fields() const {
    return std::tie(eps);
  }
};

void batch_norm_calc_invstd(const Tensor& out_invstd, const Tensor& running_var, double epsilon) {
  auto iter = TensorIteratorConfig()
      .add_output(out_invstd)
      .add_const_input(running_var)
      .check_all_same_dtype(false)
      .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, running_var.scalar_type(),
                                  "batch_norm_invert_std_cuda", [&] {
    using acc_t = at::acc_type<scalar_t, true>;
    gpu_kernel(iter, BatchNormInvStdFunctor<scalar_t, acc_t>{static_cast<acc_t>(epsilon)});
  });
}
}

std::tuple<Tensor&, Tensor&, Tensor&> batch_norm_cuda_out(const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, bool train, double momentum, double epsilon, Tensor& output, Tensor& save_mean, Tensor& save_invstd) {
  const bool has_running_mean = (running_mean_opt.has_value() && running_mean_opt->defined());
  const bool has_running_var = (running_var_opt.has_value() && running_var_opt->defined());
  TORCH_CHECK_VALUE(has_running_mean == has_running_var,
    "running_mean and running_var must either both be None or neither be None");

  if (train) {
    batch_norm_mean_var(self, save_mean, save_invstd);
    if (has_running_mean) {
      const int64_t N = self.numel() / save_mean.numel();
      batch_norm_update_stats_and_invert(
          save_mean, save_invstd, *running_mean_opt, *running_var_opt,
          momentum, epsilon, N);
    } else {
      batch_norm_calc_invstd(save_invstd, save_invstd, epsilon);
    }
  } else {
    TORCH_CHECK(has_running_mean);
    at::native::resize_output(save_mean, running_mean_opt->sizes());
    save_mean.copy_(*running_mean_opt, /*non_blocking=*/true);
    batch_norm_calc_invstd(save_invstd, running_var_opt.value(), epsilon);
  }

  batch_norm_elementwise(output, self, weight_opt, bias_opt, save_mean, save_invstd);
  return std::tuple<Tensor&, Tensor&, Tensor&>(output, save_mean, save_invstd);
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_cuda(const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, bool train, double momentum, double epsilon) {
  auto output = at::empty_like(self);
  int64_t n_input = self.size(1);
  auto options = self.options().dtype(
      at::toAccumulateType(self.scalar_type(), /*is_cuda=*/true));
  auto save_mean = at::empty({n_input}, options);
  auto save_invstd = at::empty({n_input}, options);

  at::native::batch_norm_cuda_out(
      self,
      weight_opt,
      bias_opt,
      running_mean_opt,
      running_var_opt,
      train,
      momentum,
      epsilon,
      output,
      save_mean,
      save_invstd);
  return std::make_tuple(
      std::move(output), std::move(save_mean), std::move(save_invstd));
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _batch_norm_with_update_cuda(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    Tensor& running_mean, Tensor& running_var, double momentum, double eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = bias_opt.value_or(Tensor());
  Tensor output, save_mean, save_var, reserve;

  BatchNormBackend backend = _select_batch_norm_backend(input, weight, bias, running_mean, running_var, /*training*/true, eps);
  if (backend == BatchNormBackend::Cudnn) {
    return at::cudnn_batch_norm(input, weight, bias, running_mean, running_var, /*training*/true, momentum, eps);
  }
  if (backend == BatchNormBackend::Miopen) {
    reserve = at::empty({0}, input.options().dtype(kByte));
    std::tie(output, save_mean, save_var) =
        at::miopen_batch_norm(input, weight, bias, running_mean, running_var, /*training*/true, momentum, eps);
  } else {
    reserve = at::empty({0}, input.options().dtype(kByte));
    std::tie(output, save_mean, save_var) =
        batch_norm_cuda(input, weight_opt, bias_opt, running_mean, running_var, /*training*/true, momentum, eps);
  }
  return std::tuple<Tensor, Tensor, Tensor, Tensor>(
      std::move(output),
      std::move(save_mean),
      std::move(save_var),
      std::move(reserve));
}

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> _batch_norm_with_update_cuda_out(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    Tensor& running_mean, Tensor& running_var, double momentum, double eps,
    Tensor& out, Tensor& save_mean, Tensor& save_var, Tensor& reserve) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = bias_opt.value_or(Tensor());

  BatchNormBackend backend = _select_batch_norm_backend(input, weight, bias, running_mean, running_var, /*training*/true, eps);
  if (backend == BatchNormBackend::Cudnn) {
    std::tie(out, save_mean, save_var, reserve) =
        at::cudnn_batch_norm_out(out, save_mean, save_var, reserve, input, weight, bias, running_mean, running_var, /*training*/true, momentum, eps);
  } else if (backend == BatchNormBackend::Miopen) {
    std::tie(out, save_mean, save_var) =
        at::miopen_batch_norm_out(out, save_mean, save_var, input, weight, bias, running_mean, running_var, /*training*/true, momentum, eps);
  } else {
    std::tie(out, save_mean, save_var) =
      batch_norm_cuda_out(input, weight_opt, bias_opt, running_mean, running_var, /*update*/true, momentum, eps, out, save_mean, save_var);
  }
  return std::tuple<Tensor&, Tensor&, Tensor&, Tensor&>(out, save_mean, save_var, reserve);
}

std::tuple<Tensor, Tensor, Tensor> _batch_norm_legit_cuda(const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, Tensor& running_mean, Tensor& running_var, bool train, double momentum, double epsilon) {
  return batch_norm_cuda(self, weight_opt, bias_opt, running_mean, running_var, train, momentum, epsilon);
}

std::tuple<Tensor, Tensor, Tensor> _batch_norm_legit_no_stats_cuda(const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, bool train, double momentum, double epsilon) {
  return batch_norm_cuda(self, weight_opt, bias_opt, Tensor(), Tensor(), train, momentum, epsilon);
}

std::tuple<Tensor&, Tensor&, Tensor&> _batch_norm_legit_cuda_out(const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, Tensor& running_mean, Tensor& running_var, bool train, double momentum, double epsilon, Tensor& output, Tensor& save_mean, Tensor& save_invstd) {
  return batch_norm_cuda_out(self, weight_opt, bias_opt, running_mean, running_var, train, momentum, epsilon, output, save_mean, save_invstd);
}

std::tuple<Tensor&, Tensor&, Tensor&> _batch_norm_legit_no_stats_cuda_out(const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, bool train, double momentum, double epsilon, Tensor& output, Tensor& save_mean, Tensor& save_invstd) {
  return batch_norm_cuda_out(self, weight_opt, bias_opt, Tensor(), Tensor(), train, momentum, epsilon, output, save_mean, save_invstd);
}

std::tuple<Tensor, Tensor, Tensor> _new_batch_norm_backward_cuda(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    const std::optional<Tensor>& save_mean_opt, const std::optional<Tensor>& save_var_opt,
    bool update, double eps, std::array<bool,3> grad_input_mask, const Tensor& reserve) {
  const Tensor& dummy_bias = at::empty(1);
  const Tensor& running_mean = running_mean_opt.value_or(Tensor());
  const Tensor& running_var = running_var_opt.value_or(Tensor());
  const Tensor& save_mean = save_mean_opt.value_or(Tensor());
  const Tensor& save_var = save_var_opt.value_or(Tensor());

  BatchNormBackend backend = _select_batch_norm_backend(input, weight, dummy_bias, running_mean, running_var, /*training*/true, eps);

  if (backend == BatchNormBackend::Cudnn) {
    return at::cudnn_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, eps, reserve);
  } else if (backend == BatchNormBackend::Miopen) {
    return at::miopen_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, eps);
  } else {
    return batch_norm_backward_cuda(grad_output, input, weight, running_mean, running_var, save_mean, save_var, update, eps, grad_input_mask);
  }
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda(const Tensor& grad_out, const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, const std::optional<Tensor>& save_mean_opt, const std::optional<Tensor>& save_invstd_opt, bool train, double epsilon, std::array<bool,3> grad_input_mask) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight = at::borrow_from_optional_tensor(weight_opt);
  c10::MaybeOwned<Tensor> save_mean = at::borrow_from_optional_tensor(save_mean_opt);
  c10::MaybeOwned<Tensor> save_invstd = at::borrow_from_optional_tensor(save_invstd_opt);
  c10::MaybeOwned<Tensor> running_mean = at::borrow_from_optional_tensor(running_mean_opt);
  c10::MaybeOwned<Tensor> running_var = at::borrow_from_optional_tensor(running_var_opt);

  const bool needs_reduction = train || grad_input_mask[1] || grad_input_mask[2];

  // Fused reduction & elementwise kernel
  if (needs_reduction && grad_input_mask[0] &&
      !batch_norm_use_channels_last_kernels(input) &&
      cuda::detail::canUse32BitIndexMath(input) &&
      cuda::detail::canUse32BitIndexMath(grad_out)) {
    return AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
                                           "batch_norm_backward_cuda", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      const bool mixed_type = is_mixed_type(input, *weight, *running_mean, *running_var);
      if (mixed_type) {
          return batch_norm_backward_cuda_template<scalar_t, accscalar_t, int32_t>(
              grad_out, input, *weight, *running_mean, *running_var,
              *save_mean, *save_invstd, train, epsilon, grad_input_mask);
      } else {
          return batch_norm_backward_cuda_template<scalar_t, scalar_t, int32_t>(
              grad_out, input, *weight, *running_mean, *running_var,
              *save_mean, *save_invstd, train, epsilon, grad_input_mask);
      }
    });
  }

  // NOTE: native_batch_norm always returns save_mean and save_invstd to be reused in backward.
  // However, this is also called from cudnn_batch_norm in eval mode which doesn't give
  // save_mean and save_invstd, so it needs recalculated.
  const auto acc_type = at::toAccumulateType(input.scalar_type(), /*is_cuda=*/true);
  Tensor mean;
  TORCH_INTERNAL_ASSERT(save_mean->defined(), "save_mean should always be defined\n");
  if (save_mean->numel() != 0) {
    mean = *save_mean;
  } else if (needs_reduction) {
    TORCH_CHECK(!train && running_mean->defined());
    mean = (running_mean->scalar_type() == acc_type) ?
        *running_mean : running_mean->to(acc_type);
  }

  Tensor invstd;
  TORCH_INTERNAL_ASSERT(save_invstd->defined(), "save_invstd should always be defined\n");
  if (save_invstd->numel() != 0) {
    invstd = *save_invstd;
  } else {
    TORCH_CHECK(!train && running_var->defined());
    auto n_channels = input.sizes()[1];
    invstd = at::empty({n_channels}, input.options().dtype(acc_type));
    batch_norm_calc_invstd(invstd, *running_var, epsilon);
  }

  Tensor sum_dy, sum_dy_xmu, grad_weight, grad_bias;
  if (needs_reduction) {
    std::tie(sum_dy, sum_dy_xmu, grad_weight, grad_bias) =
        batch_norm_backward_reduce_cuda(
            grad_out, input, mean, invstd, *weight,
            grad_input_mask[0], grad_input_mask[1], grad_input_mask[2]);
  }

  Tensor grad_input;
  if (grad_input_mask[0]) {
    if (train) {
      // NOTE: sum_dy and sum_dy_xmy are defined, as train implies needs_reduction
      grad_input = batch_norm_elementwise_backward_train(
          grad_out, input, mean, invstd, *weight, sum_dy, sum_dy_xmu);
    } else {
      grad_input = batch_norm_elementwise_backward_eval(
          grad_out, input, invstd, *weight);
    }
  }

  return std::make_tuple(
      std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}

std::tuple<Tensor, Tensor> batch_norm_stats_cuda(const Tensor& self, double epsilon) {
  auto options = self.options().dtype(
      at::toAccumulateType(self.scalar_type(), /*is_cuda=*/true));
  auto n_channels = self.size(1);
  auto save_mean = at::empty({n_channels}, options);
  auto save_invstd = at::empty({n_channels}, options);

  bool use_channels_last_kernel = batch_norm_use_channels_last_kernels(self);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                  self.scalar_type(), "batch_norm_stats_cuda", [&] {
    if (cuda::detail::canUse32BitIndexMath(self)) {
      if (use_channels_last_kernel) {
        batch_norm_stats_channels_last_cuda_template<scalar_t, InvStd>(
            save_mean, save_invstd, self, epsilon);
      } else {
        batch_norm_stats_cuda_template<scalar_t, int32_t, InvStd>(
            save_mean, save_invstd, self, epsilon);
      }
    } else {
      batch_norm_stats_cuda_template<scalar_t, int64_t, InvStd>(
          save_mean, save_invstd, self, epsilon);
    }
  });
  return std::tuple<Tensor, Tensor>(save_mean, save_invstd);
}

Tensor batch_norm_elemt_cuda(
    const Tensor& self, const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt, const Tensor& mean,
    const Tensor& invstd, double epsilon) {
  auto output = at::empty_like(self);
  // FIXME: Epsilon parameter isn't required, we don't take the reciprocal
  batch_norm_elementwise(output, self, weight_opt, bias_opt, mean, invstd);
  return output;
}

Tensor& batch_norm_elemt_cuda_out(const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
                                  const Tensor& mean, const Tensor& invstd, double epsilon, Tensor& output) {
  // FIXME: Epsilon parameter isn't required, we don't take the reciprocal
  batch_norm_elementwise(output, self, weight_opt, bias_opt, mean, invstd);
  return output;
}

// accepting input(self) here to determine template data types, since running_mean/running_var are optional
std::tuple<Tensor, Tensor> batch_norm_gather_stats_cuda(const Tensor& self, const Tensor& mean, const Tensor& invstd, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, double momentum, double epsilon, int64_t count) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;
  const Tensor& running_var = running_var_opt.value_or(Tensor());

  TORCH_CHECK(mean.dim() == 2,
      "batch_norm_gather_stats: expected mean to be 2-dimensional (world_size, num_features), but got mean of sizes ",
      mean.sizes());

  std::vector<int64_t> counts(mean.size(0), count);
  Tensor counts_ = at::from_blob((void*)counts.data(), {(int64_t)counts.size()}, self.options().dtype(at::kLong).device(at::kCPU));
  counts_ = counts_.to(self.device()).to(running_mean.defined() ? running_mean.dtype() : self.dtype());
  return batch_norm_gather_stats_with_counts_cuda(self, mean, invstd, running_mean, running_var, momentum, epsilon, counts_);
}


std::tuple<Tensor, Tensor> batch_norm_gather_stats_with_counts_cuda(
    const Tensor& self, const Tensor& mean, const Tensor& invstd, const std::optional<Tensor>& running_mean_opt /* optional */, const std::optional<Tensor>& running_var_opt /* optional */, double momentum, double epsilon, const Tensor& counts) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;
  const Tensor& running_var = running_var_opt.value_or(Tensor());

  // batch_norm_reduce_statistics_kernel derives both of its loop bounds from mean alone, then
  // indexes invstd and counts with them, so any disagreement is an out-of-bounds read.
  TORCH_CHECK(mean.dim() == 2,
      "batch_norm_gather_stats_with_counts: expected mean to be 2-dimensional (world_size, num_features), but got mean of sizes ",
      mean.sizes());
  TORCH_CHECK(invstd.sizes() == mean.sizes(),
      "batch_norm_gather_stats_with_counts: expected invstd to have the same shape as mean (",
      mean.sizes(), "), but got invstd of sizes ", invstd.sizes());
  TORCH_CHECK(counts.numel() >= mean.size(0),
      "batch_norm_gather_stats_with_counts: expected counts to have at least one element per entry in mean's first dimension (",
      mean.size(0), "), but got ", counts.numel(), " elements");

  auto scalar_type = running_mean.defined() ? running_mean.scalar_type() : self.scalar_type();
  return AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "batch_norm_update_stats_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    if (cuda::detail::canUse32BitIndexMath(self)) {
      return batch_norm_gather_stats_cuda_template<scalar_t, accscalar_t, int32_t>(mean, invstd, running_mean, running_var, momentum, epsilon, counts);
    } else {
      return batch_norm_gather_stats_cuda_template<scalar_t, accscalar_t, int64_t>(mean, invstd, running_mean, running_var, momentum, epsilon, counts);
    }
  });
}

std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& mean, const Tensor& invstd, const std::optional<Tensor>& weight_opt, bool input_g, bool weight_g, bool bias_g) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  if (at::cuda::detail::canUse32BitIndexMath(grad_output) &&
      batch_norm_use_channels_last_kernels(grad_output) &&
      batch_norm_use_channels_last_kernels(input) &&
      (!weight.defined() || weight.is_contiguous()) &&
      mean.is_contiguous() && invstd.is_contiguous()){
    return batch_norm_backward_reduce_cuda_channels_last_template(
        grad_output, input, mean, invstd, weight, input_g, weight_g, bias_g);
  }

  return AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, grad_output.scalar_type(), "batch_norm_backward_reduce", [&] {
    auto mean_st = mean.dtype();
    auto invstd_st = invstd.dtype();
    TORCH_CHECK(mean_st == invstd_st, "mean and invstd need to have the same data types");
    const bool mixed_type = is_mixed_type(input, weight);
    using accscalar_t = at::acc_type<scalar_t, true>;

    if (cuda::detail::canUse32BitIndexMath(grad_output)) {
      if (mixed_type) {
        return batch_norm_backward_reduce_cuda_template<scalar_t, accscalar_t, int32_t>(grad_output, input, mean, invstd, weight, input_g, weight_g, bias_g);
      } else {
        return batch_norm_backward_reduce_cuda_template<scalar_t, scalar_t, int32_t>(grad_output, input, mean, invstd, weight, input_g, weight_g, bias_g);
      }
    } else {
      if (mixed_type) {
        return batch_norm_backward_reduce_cuda_template<scalar_t, accscalar_t, int64_t>(grad_output, input, mean, invstd, weight, input_g, weight_g, bias_g);
      } else {
        return batch_norm_backward_reduce_cuda_template<scalar_t, scalar_t, int64_t>(grad_output, input, mean, invstd, weight, input_g, weight_g, bias_g);
      }
    }
  });
}

Tensor batch_norm_backward_elemt_cuda(const Tensor& self, const Tensor& input, const Tensor& mean, const Tensor& invstd, const std::optional<Tensor>& weight_opt, const Tensor& sum_dy, const Tensor& sum_dy_xmu, const Tensor& count) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  if (at::cuda::detail::canUse32BitIndexMath(self) &&
      batch_norm_use_channels_last_kernels(self) &&
      batch_norm_use_channels_last_kernels(input))  {
    return batch_norm_backward_elemt_channels_last_cuda_template(self, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
  }

  return AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "batch_norm_backward_elemt", [&] {
    auto mean_st = mean.dtype();
    auto invstd_st = invstd.dtype();
    TORCH_CHECK(mean_st == invstd_st, "mean and invstd need to have the same data types");
    bool is_half_float = std::is_same_v<scalar_t, at::Half> && mean_st == at::kFloat;
    bool is_bfloat16_float = std::is_same_v<scalar_t, at::BFloat16> && mean_st == at::kFloat;
    using accscalar_t = at::acc_type<scalar_t, true>;
    if (cuda::detail::canUse32BitIndexMath(self)) {
      if (is_half_float || is_bfloat16_float) {
        return batch_norm_backward_elemt_cuda_template<scalar_t, accscalar_t, int32_t>(self, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
      } else {
        return batch_norm_backward_elemt_cuda_template<scalar_t, scalar_t, int32_t>(self, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
      }
    } else {
      if (is_half_float || is_bfloat16_float) {
        return batch_norm_backward_elemt_cuda_template<scalar_t, accscalar_t, int64_t>(self, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
      } else {
        return batch_norm_backward_elemt_cuda_template<scalar_t, scalar_t, int64_t>(self, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
      }
    }
  });
}

std::tuple<Tensor, Tensor> batch_norm_update_stats_cuda(
    const Tensor& self, const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt, double momentum) {
  c10::MaybeOwned<Tensor> running_mean = at::borrow_from_optional_tensor(running_mean_opt);
  c10::MaybeOwned<Tensor> running_var = at::borrow_from_optional_tensor(running_var_opt);

  const int64_t n_input = self.size(1);

  TORCH_CHECK(self.numel() != 0, "input tensor must have at least one element, but got input_sizes = ", self.sizes());
  auto options = self.options().dtype(
      at::toAccumulateType(self.scalar_type(), /*is_cuda=*/true));
  auto save_mean = at::empty({n_input}, options);
  auto save_var = at::empty({n_input}, options);

  batch_norm_mean_var(self, save_mean, save_var);
  TORCH_CHECK(running_mean->defined() == running_var->defined());
  if (running_mean->defined()) {
    const int64_t N = self.numel() / save_mean.numel();
    batch_norm_update_stats(save_mean, save_var, *running_mean, *running_var, momentum, N);
  }
  return std::tuple<Tensor, Tensor>(save_mean, save_var);
}

} // namespace at::native

#if !defined(USE_ROCM)
// Traced host (ATen/cuda/host_trace/Ops.h)
namespace at::cuda::host_trace {
namespace {

namespace an = at::native;

// GenericPackedTensorAccessor's protected members, to set in a Param
template <class A>
struct AccessorMembers : A {
  using A::data_;
  using A::sizes_;
  using A::strides_;
};

// its data_ as a plain pointer, which Param::set records as one
template <typename T, class A>
T*& accessor_data(Param<A>& p) {
  // a C-style cast, to drop the member's __restrict__
  return (T*&)(p.value().*(&AccessorMembers<A>::data_));
}

template <typename T, typename index_t>
Param<GenericPackedTensorAccessor<T, 1, RestrictPtrTraits, index_t>> accessor_or_dummy(Recorder& rec, const TensorBase& t) {
  using A = GenericPackedTensorAccessor<T, 1, RestrictPtrTraits, index_t>;
  Param<A> p;
  if (t.defined()) {
    p.set(accessor_data<T>(p), rec.data_ptr(t));
    p.set((p.value().*(&AccessorMembers<A>::sizes_))[0], t.sym_size(0));
    p.set((p.value().*(&AccessorMembers<A>::strides_))[0], t.sym_stride(0));
  }
  return p;
}

// t viewed as [N, C, F], as eager's reshape and view of a contiguous t
template <typename T, typename index_t>
Param<GenericPackedTensorAccessor<T, 3, RestrictPtrTraits, index_t>> accessor_3d(Recorder& rec, const TensorBase& t, const c10::SymDimVector& shape) {
  using A = GenericPackedTensorAccessor<T, 3, RestrictPtrTraits, index_t>;
  const auto strides = at::detail::computeStride(t.sym_sizes(), t.sym_strides(), shape);
  TORCH_INTERNAL_ASSERT(strides.has_value());
  Param<A> p;
  p.set(accessor_data<T>(p), rec.data_ptr(t));
  for (const auto i : c10::irange(3)) {
    p.set((p.value().*(&AccessorMembers<A>::sizes_))[i], shape[i]);
    p.set((p.value().*(&AccessorMembers<A>::strides_))[i], (*strides)[i]);
  }
  return p;
}

// getNumThreads
int64_t num_threads(const c10::SymInt& n) {
  for (const int64_t t : {32, 64, 128, 256}) {
    if (n <= t) {
      return t;
    }
  }
  return an::MAX_BLOCK_SIZE;
}

// lastPow2 of n >= 1
c10::SymInt last_pow2(Recorder& rec, const c10::SymInt& n) {
  if (auto c = n.maybe_as_int()) {
    return an::lastPow2(static_cast<unsigned int>(*c));
  }
  return rec.pow2(rec.bit_length(n) - 1);
}

// batch_norm_elemt_cuda_template's launch (keep in sync)
template <typename input_scalar_t, typename stat_scalar_t>
void batch_norm_elemt_launch(Recorder& rec, const TensorBase& out, const TensorBase& input, const TensorBase& weight, const TensorBase& bias, const TensorBase& mean, const TensorBase& invstd) {
  using stat_accscalar_t = at::acc_type<stat_scalar_t, true>;
  c10::SymDimVector shape{input.sym_size(0), input.sym_size(1), 1};
  for (const auto i : c10::irange(2, input.dim())) {
    shape[2] *= input.sym_size(i);
  }
  const int64_t tf = std::max(num_threads(shape[2] / 4), std::min(num_threads(shape[2]), int64_t{64}));
  const int64_t tb = std::max(int64_t{64} / tf, int64_t{1});
  const c10::SymInt blocks_y = c10::SymInt(256 * 1024 / shape[1]).min((shape[0] + tb - 1) / tb).max(1).min(static_cast<int64_t>(an::MAX_GRID_SIZE));
  launch(rec, &an::batch_norm_transform_input_kernel<input_scalar_t, stat_scalar_t, stat_accscalar_t, true, int32_t>,
         SymDim3(shape[1], blocks_y), SymDim3(tf, tb), 0,
         accessor_3d<const input_scalar_t, int32_t>(rec, input, shape), accessor_3d<input_scalar_t, int32_t>(rec, out, shape),
         accessor_or_dummy<stat_accscalar_t, int32_t>(rec, mean), accessor_or_dummy<stat_accscalar_t, int32_t>(rec, invstd),
         accessor_or_dummy<const stat_scalar_t, int32_t>(rec, weight), accessor_or_dummy<const stat_scalar_t, int32_t>(rec, bias),
         Param<stat_accscalar_t>(1e-5));
}

// batch_norm_elemt_channels_last_cuda_template's launch (keep in sync)
template <typename scalar_t, typename layerscalar_t>
void batch_norm_elemt_channels_last_launch(Recorder& rec, const TensorBase& out, const TensorBase& input, const TensorBase& weight, const TensorBase& bias, const TensorBase& mean, const TensorBase& invstd) {
  using accscalar_t = at::acc_type<scalar_t, true>;
  const c10::SymInt& stride = input.sym_size(1);
  const c10::SymInt reduction_size = input.sym_numel() / stride;
  // flexible_launch_configs; its second block_x is its first where the block is full
  const c10::SymInt block_y = last_pow2(rec, (reduction_size + an::ELEMENTS_PER_THREAD - 1) / an::ELEMENTS_PER_THREAD).min(an::MAX_BLOCK_SIZE / last_pow2(rec, stride).min(an::OPTIMAL_TILE_W));
  const c10::SymInt block_x = last_pow2(rec, stride).min(an::MAX_BLOCK_SIZE / block_y);
  const c10::SymInt grid_y = ((reduction_size + block_y * an::ELEMENTS_PER_THREAD - 1) / (block_y * an::ELEMENTS_PER_THREAD)).min(an::MAX_H_BLOCK);
  const auto address = [&rec](const TensorBase& t) { return t.defined() ? rec.data_ptr(t) : c10::SymInt(0); };
  launch(rec, &an::batch_norm_transform_input_channels_last_kernel<scalar_t, accscalar_t, layerscalar_t, an::ELEMENTS_PER_ITER>,
         SymDim3((stride + block_x - 1) / block_x, grid_y), SymDim3(block_x, block_y), 0,
         scalar_param<const scalar_t*>(address(input)), Param<const scalar_t*>(), scalar_param<const accscalar_t*>(address(mean)),
         scalar_param<const accscalar_t*>(address(invstd)), scalar_param<const layerscalar_t*>(address(weight)),
         scalar_param<const layerscalar_t*>(address(bias)), scalar_param<scalar_t*>(address(out)), scalar_param<int>(reduction_size),
         scalar_param<int>(stride), Param<bool>(false));
}

} // namespace

std::tuple<TensorBase, TensorBase, TensorBase> native_batch_norm(Recorder& rec, const TensorBase& input, const TensorBase& weight, const TensorBase& bias, const TensorBase& running_mean, const TensorBase& running_var, bool training, double eps) {
  if (training) {
    decline("a batch norm in training");
  }
  if (!running_mean.defined() || !running_var.defined()) {
    decline("a batch norm without running statistics");
  }
  const ScalarType dtype = input.scalar_type();
  if (dtype != kHalf && dtype != kBFloat16 && dtype != kFloat) {
    decline(c10::str("a ", dtype, " batch norm"));
  }
  const ScalarType acc = at::toAccumulateType(dtype, /*is_cuda=*/true);
  if (running_var.scalar_type() == kDouble) {
    decline("a double running variance");
  }
  const TensorBase& param = weight.defined() ? weight : bias;
  const ScalarType param_dtype = param.defined() ? param.scalar_type() : dtype;
  if ((param_dtype != dtype && param_dtype != acc) || (weight.defined() && bias.defined() && weight.scalar_type() != bias.scalar_type())) {
    decline("a batch norm weight or bias of another dtype");
  }
  if (input.sym_numel() == 0) {
    decline("an empty batch norm");
  }
  // batch_norm_cuda's empty_like(self) of a dense self, save_mean and save_invstd
  if (!input.is_non_overlapping_and_dense()) {
    decline("a batch norm of an input that is not dense");
  }
  const TensorBase out = at::empty_strided_symint(input.sym_sizes(), input.sym_strides(), input.options());
  const auto stat_options = input.options().dtype(acc);
  const TensorBase save_mean = at::empty_symint({input.sym_size(1)}, stat_options);
  const TensorBase save_invstd = at::empty_symint({input.sym_size(1)}, stat_options);
  copy_(rec, save_mean, running_mean);
  // batch_norm_calc_invstd
  auto iter = TensorIteratorSym::pointwise_op(rec, {save_invstd}, {acc}, {running_var});
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, running_var.scalar_type(), "batch_norm_invert_std_cuda", [&] {
    using acc_t = at::acc_type<scalar_t, true>;
    gpu_kernel(rec, iter, an::BatchNormInvStdFunctor<scalar_t, acc_t>{static_cast<acc_t>(eps)});
  });
  // batch_norm_elementwise's batch_norm_choose_impl. canUse32BitIndexMath of a
  // dense input, whose last element is at numel - 1: its linearId % size of
  // symbolic sizes costs sympy minutes on a deep conv net
  if (input.sym_numel() >= std::numeric_limits<int32_t>::max()) {
    decline("a batch norm beyond 32-bit indexing");
  }
  const bool contiguous = input.is_contiguous();
  if (contiguous && input.sym_stride(1) != 1) {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, dtype, "batch_norm_elementwise_cuda", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      if (param_dtype != dtype) {
        batch_norm_elemt_launch<scalar_t, accscalar_t>(rec, out, input, weight, bias, save_mean, save_invstd);
      } else {
        batch_norm_elemt_launch<scalar_t, scalar_t>(rec, out, input, weight, bias, save_mean, save_invstd);
      }
    });
  } else if ((contiguous || input.is_contiguous(at::MemoryFormat::ChannelsLast)) && (!weight.defined() || weight.is_contiguous()) &&
             (!bias.defined() || bias.is_contiguous())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, dtype, "batchnorm_forward", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      if (param_dtype != dtype) {
        batch_norm_elemt_channels_last_launch<scalar_t, accscalar_t>(rec, out, input, weight, bias, save_mean, save_invstd);
      } else {
        batch_norm_elemt_channels_last_launch<scalar_t, scalar_t>(rec, out, input, weight, bias, save_mean, save_invstd);
      }
    });
  } else {
    decline("a batch norm through TensorIterator (batch_norm_elementwise's general case)");
  }
  return {out, save_mean, save_invstd};
}

} // namespace at::cuda::host_trace
#endif
