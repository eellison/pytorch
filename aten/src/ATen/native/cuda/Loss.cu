#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/TensorUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/Resize.h>
#include <ATen/cuda/host_trace/Launch.h>
#include <ATen/cuda/host_trace/NllLossHost.h>
#include <ATen/cuda/host_trace/PackedAccessor.h>

#include <type_traits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/binary_cross_entropy_backward_native.h>
#include <ATen/ops/binary_cross_entropy_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/nll_loss_backward_native.h>
#include <ATen/ops/nll_loss_forward_native.h>
#include <ATen/ops/squeeze.h>
#endif

constexpr float EPSILON = 1e-12;

namespace {

using namespace at;

void binary_cross_entropy_backward_out_kernel(Tensor& grad_input, const Tensor& grad, const Tensor& input, const Tensor& target) {
  at::TensorIterator iter = TensorIteratorConfig()
      .add_output(grad_input)
      .add_const_input(grad)
      .add_const_input(input)
      .add_const_input(target)
      .build();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "binary_cross_entropy_backward_out_cuda", [&]() {
    at::native::gpu_kernel(iter, [] GPU_LAMBDA (
        scalar_t grad_val,
        scalar_t input_val,
        scalar_t target_val
      ) -> scalar_t {
        const scalar_t one = 1;
        const scalar_t epsilon = EPSILON;

        scalar_t grad_input_denominator = max(
          (one - input_val) * input_val,
          epsilon
        );

        return grad_val * (input_val - target_val) / grad_input_denominator;
      }
    );
  });
}

} // namespace

namespace at::native {

Tensor binary_cross_entropy_cuda(const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, int64_t reduction) {
    Tensor loss = at::empty_like(input);
    return at::native::binary_cross_entropy_out_cuda(
        input, target, weight_opt, reduction, loss);
}

Tensor& binary_cross_entropy_out_cuda(const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, int64_t reduction, Tensor& loss) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  Tensor loss_squeezed = at::squeeze(loss);

  TensorIterator iter = TensorIteratorConfig()
      .add_output(loss_squeezed)
      .add_owned_const_input(at::squeeze(input))
      .add_owned_const_input(at::squeeze(target))
      .build();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "binary_cross_entropy_out_cuda", [&]() {
    gpu_kernel(iter,
      [] GPU_LAMBDA (scalar_t input_val, scalar_t target_val) -> scalar_t {
        const scalar_t zero = 0;
        const scalar_t one = 1;
        const scalar_t neg_100 = -100;

        CUDA_KERNEL_ASSERT(input_val >= zero && input_val <= one);
        CUDA_KERNEL_ASSERT(target_val >= zero && target_val <= one);

        scalar_t log_input_val = std::log(input_val);
        scalar_t log_1_minus_input_val = std::log1p(-input_val);

        log_input_val = std::max(log_input_val, neg_100);
        log_1_minus_input_val = std::max(log_1_minus_input_val, neg_100);

        return ((target_val - one) * log_1_minus_input_val) - (target_val * log_input_val);
      }
    );
  });
  if (weight.defined()) {
    loss.mul_(weight);
  }

  if (reduction != at::Reduction::None) {
    Tensor loss_reduced;
    if (reduction == at::Reduction::Mean) {
      loss_reduced = loss.mean();
    } else if (reduction == at::Reduction::Sum) {
      loss_reduced = loss.sum();
    }
    loss.resize_as_(loss_reduced).copy_(loss_reduced);
  }

  return loss;
}

Tensor binary_cross_entropy_backward_cuda(const Tensor& grad, const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, int64_t reduction) {
  Tensor grad_input = at::empty_like(input);
  return at::native::binary_cross_entropy_backward_out_cuda(
      grad, input, target, weight_opt, reduction, grad_input);
}

Tensor& binary_cross_entropy_backward_out_cuda(const Tensor& grad, const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, int64_t reduction, Tensor& grad_input) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  Tensor grad_expand = grad.expand_as(input);
  binary_cross_entropy_backward_out_kernel(grad_input, grad_expand, input, target);

  if (weight.defined()) {
    grad_input.mul_(weight);
  }
  if (reduction == at::Reduction::Mean) {
    grad_input.div_(input.numel());
  }
  return grad_input;
}

// -----------------------------------
// nll_loss
// -----------------------------------
namespace {

// std::clamp(1 << round(log2(nframe / 16)), 32, 1024) on a symbolic frame
// count: round(log2(q)) == k for the integer q = nframe / 16 exactly when
// 2^(k - 1/2) <= q < 2^(k + 1/2), so the ladder below is the same function
// (checked exhaustively for nframe < 5e6) with each step a recorded guard.
int64_t nll_loss_threads(const c10::SymInt& nframe){
  const c10::SymInt q = nframe / 16;
  if (q < 46) return 32;
  if (q < 91) return 64;
  if (q < 182) return 128;
  if (q < 363) return 256;
  if (q < 725) return 512;
  return 1024;
}

// NOTE(crcrpar): `Byte` support was added for https://github.com/pytorch/pytorch/issues/59765.
#define AT_DISPATCH_NLL_LOSS_INDEX_TYPES(TYPE, NAME, ...)                     \
  AT_DISPATCH_SWITCH(TYPE, NAME,                                              \
  AT_PRIVATE_CASE_TYPE_USING_HINT(at::ScalarType::Byte, index_t, __VA_ARGS__) \
  AT_PRIVATE_CASE_TYPE_USING_HINT(at::ScalarType::Long, index_t, __VA_ARGS__))

#define CHECK_INDEX_IN_CLASS(INDEX, N_CLASSES)                           \
  if constexpr(std::is_unsigned_v<decltype(INDEX)>) {                    \
    CUDA_KERNEL_ASSERT(INDEX < N_CLASSES);                               \
  } else {                                                               \
    CUDA_KERNEL_ASSERT(INDEX >= 0 && INDEX < N_CLASSES);                 \
  }

template <typename scalar_t, typename index_t>
__global__ void nll_loss_forward_no_reduce_cuda_kernel(
    int64_t batch_size,
    PackedTensorAccessor64<scalar_t, 2> input,
    const index_t* target,
    scalar_t* output,
    const scalar_t* weights,
    int64_t n_classes,
    int64_t ignore_index) {
  CUDA_KERNEL_LOOP(index, batch_size) {
    index_t cur_target = target[index];
    if (cur_target == ignore_index) {
      output[index] = static_cast<scalar_t>(0);
      continue;
    }
    CHECK_INDEX_IN_CLASS(cur_target, n_classes);
    auto cur_weight =
        weights != nullptr ? weights[cur_target] : static_cast<scalar_t>(1);
    output[index] = -cur_weight * input[index][cur_target];
  }
}

template <typename scalar_t, typename index_t>
__global__ void nll_loss_forward_reduce_cuda_kernel_1d(
    scalar_t* output,
    scalar_t* total_weight,
    const scalar_t* input,
    const index_t* target,
    const scalar_t* weights,
    bool size_average,
    int64_t n_classes,
    int64_t ignore_index) {
  CUDA_KERNEL_ASSERT(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);

  const index_t t = *target;
  if (t != ignore_index) {
    CHECK_INDEX_IN_CLASS(t, n_classes);
    const auto cur_weight = weights != nullptr ? weights[t] : scalar_t{1};
    *total_weight = cur_weight;

    if (size_average) {
      // If we try to normalize a zero then we return a NaN
      if (cur_weight == 0) {
        *output = std::numeric_limits<scalar_t>::quiet_NaN();
      } else {
        *output = -input[t];
      }
    } else {
      *output = -cur_weight * input[t];
    }
  } else {
    // If the only element was omitted, we get 0. See the discussion in
    // https://github.com/pytorch/pytorch/pull/64572#issuecomment-926504162
    *output = scalar_t{0};
    *total_weight = scalar_t{0};
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void nll_loss_forward_reduce_cuda_kernel_2d(
    scalar_t* output,
    scalar_t* total_weight,
    const scalar_t* input,
    const index_t* target,
    const scalar_t* weights,
    bool size_average,
    int64_t nframe,
    int64_t ndim,
    int64_t n_classes,
    int64_t ignore_index) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  extern __shared__ unsigned char shmem[];
  accscalar_t* sh_inputs = reinterpret_cast<accscalar_t*>(shmem);
  accscalar_t* acc_weight = reinterpret_cast<accscalar_t*>(shmem + blockDim.x * sizeof(accscalar_t));

  sh_inputs[threadIdx.x] = static_cast<accscalar_t>(0);
  acc_weight[threadIdx.x] = static_cast<accscalar_t>(0);
  for (int i = threadIdx.x; i < nframe; i += blockDim.x) {
    index_t t = target[i];
    if (t != ignore_index) {
      CHECK_INDEX_IN_CLASS(t, n_classes);
      scalar_t cur_weight =
          weights != nullptr ? weights[t] : static_cast<scalar_t>(1);
      sh_inputs[threadIdx.x] -= input[i * ndim + t] * cur_weight;
      acc_weight[threadIdx.x] += cur_weight;
    }
  }

  __syncthreads();

  for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sh_inputs[threadIdx.x] += sh_inputs[threadIdx.x + stride];
      acc_weight[threadIdx.x] += acc_weight[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *total_weight = static_cast<scalar_t>(acc_weight[0]);
    if (size_average) {
      *output = static_cast<scalar_t>(sh_inputs[0] / acc_weight[0]);
    } else {
      *output = static_cast<scalar_t>(sh_inputs[0]);
    }
  }
}

// Host tracing (ATen/cuda/host_trace): the sizes are c10::SymInt, the
// addresses come from sym_*_data_ptr, the launches go through the typed
// helper (the no-reduce kernels' accessors through the PackedAccessor proxy)
// and every branch on a size is a recorded guard. Ordinary calls see concrete
// values and the same launches.
void nll_loss_forward_out_cuda_template(
    const Tensor& output,
    const Tensor& total_weight,
    const Tensor& input_,
    const Tensor& target_,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  namespace ht = at::cuda::host_trace;
  auto input = *input_.expect_contiguous();
  auto target = *target_.expect_contiguous();

  c10::SymInt n_classes = input.sym_size(-1);
  int64_t n_dims = input.dim();
  c10::SymInt batch_size = n_dims == 1 ? c10::SymInt(1) : input.sym_size(0);

  auto weight_ = weight.defined() ? weight.contiguous() : weight;

  if (weight_.defined()) {
  TORCH_CHECK(
      input.scalar_type() == weight_.scalar_type(),
      "expected scalar type ",
      input.scalar_type(),
      " but found ",
      weight_.scalar_type());
  }

  if (reduction == Reduction::None && n_dims == 2) {
    at::native::resize_output_symint(output, {batch_size});
    total_weight.zero_();
    if (batch_size == 0) {
      // This guards from unnecessary operations and launching CUDA kernel with
      // 0 blocks.
      return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss_forward_no_reduce_cuda_kernel",
        [&] {
          AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
              target.scalar_type(),
              "nll_loss_forward_no_reduce_cuda_kernel_index",
              [&] {
                ht::Traced<PackedTensorAccessor64<scalar_t, 2>> input_acc;
                input_acc.fill(input, ht::sym_const_data_ptr<scalar_t>(input));
                // GET_BLOCKS(batch_size) on a symbolic count
                const ht::Grid blocks(((batch_size - 1) / at::cuda::detail::CUDA_NUM_THREADS + 1)
                    .min(c10::SymInt(static_cast<int64_t>(std::numeric_limits<int>::max()))));
                ht::launch(nll_loss_forward_no_reduce_cuda_kernel<scalar_t, index_t>,
                       blocks,
                       at::cuda::detail::CUDA_NUM_THREADS,
                       0,
                       at::cuda::getCurrentCUDAStream(),
                        batch_size,
                        input_acc,
                        ht::sym_const_data_ptr<index_t>(target),
                        ht::sym_mutable_data_ptr<scalar_t>(output),
                        weight_.defined() ? ht::sym_const_data_ptr<scalar_t>(weight_)
                                          : c10::SymInt(0),
                        n_classes,
                        ignore_index);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
    return;
  }

  // produce scalar outputs for the reduction case
  at::native::resize_output_symint(output, {});
  at::native::resize_output_symint(total_weight, {});

  if (target.sym_numel() == 0) {
    // Here target (and input) have zero elements
    // Mean reduction on empty tensors produces NaN. See the discussion in
    // https://github.com/pytorch/pytorch/pull/64572#issuecomment-926504162
    if (reduction == Reduction::Mean) {
      output.fill_(std::numeric_limits<double>::quiet_NaN());
    } else {
      output.zero_();
    }
    total_weight.zero_();
    return;
  }

  if (n_dims == 1) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss_forward_reduce_cuda_kernel_1d",
        [&] {
          AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
              target.scalar_type(),
              "nll_loss_forward_reduce_cuda_kernel_1d_index",
              [&] {
                ht::launch(nll_loss_forward_reduce_cuda_kernel_1d<scalar_t, index_t>,
                    1, 1, 0, at::cuda::getCurrentCUDAStream(),
                        ht::sym_mutable_data_ptr<scalar_t>(output),
                        ht::sym_mutable_data_ptr<scalar_t>(total_weight),
                        ht::sym_const_data_ptr<scalar_t>(input),
                        ht::sym_const_data_ptr<index_t>(target),
                        weight_.defined() ? ht::sym_const_data_ptr<scalar_t>(weight_)
                                          : c10::SymInt(0),
                        reduction == at::Reduction::Mean,
                        n_classes,
                        ignore_index);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  } else if (n_dims == 2) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss_forward_reduce_cuda_kernel_2d",
        [&] {
          AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
              target.scalar_type(),
              "nll_loss_forward_reduce_cuda_kernel_2d_index",
              [&] {
                using accscalar_t = at::acc_type<scalar_t, /*is_cuda*/true>;
                int64_t nthreads = nll_loss_threads(input.sym_size(0));
                ht::launch(nll_loss_forward_reduce_cuda_kernel_2d<scalar_t, accscalar_t, index_t>,
                       1,
                       nthreads,
                       nthreads * static_cast<int64_t>(sizeof(accscalar_t)) * 2,
                       at::cuda::getCurrentCUDAStream(),
                        ht::sym_mutable_data_ptr<scalar_t>(output),
                        ht::sym_mutable_data_ptr<scalar_t>(total_weight),
                        ht::sym_const_data_ptr<scalar_t>(input),
                        ht::sym_const_data_ptr<index_t>(target),
                        weight_.defined() ? ht::sym_const_data_ptr<scalar_t>(weight_)
                                          : c10::SymInt(0),
                        reduction == at::Reduction::Mean,
                        input.sym_size(0),
                        input.sym_size(1),
                        n_classes,
                        ignore_index);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  }
}

template <typename scalar_t, typename index_t>
__global__ void nll_loss_backward_no_reduce_cuda_kernel(
  int batch_size,
  const index_t *target,
  PackedTensorAccessor64<const scalar_t, 1> grad_output,
  PackedTensorAccessor64<scalar_t, 2> grad_input,
  const scalar_t *weights,
  int64_t n_classes,
  int64_t ignore_index) {

  CUDA_KERNEL_LOOP(index, batch_size) {
    index_t cur_target = target[index];
    if (cur_target == ignore_index) {
      continue;
    }
    CHECK_INDEX_IN_CLASS(cur_target, n_classes);
    scalar_t weight = weights != nullptr ? weights[cur_target] : static_cast<scalar_t>(1);
    grad_input[index][cur_target] = -weight * grad_output[index];
  }
};

template <typename scalar_t, typename index_t>
__global__ void nll_loss_backward_reduce_cuda_kernel_1d(
  scalar_t *grad_input,
  const scalar_t *grad_output,
  const scalar_t *weights,
  const index_t *target,
  const scalar_t *total_weight,
  bool size_average,
  int64_t n_classes,
  int64_t ignore_index
) {
  const index_t t = *target;
  if (t != ignore_index) {
    CHECK_INDEX_IN_CLASS(t, n_classes);
    const auto grad = -(size_average ? *grad_output / *total_weight : *grad_output);
    grad_input[t] = weights != nullptr ? weights[t] * grad : grad;
  }
}

template <typename T> struct bwd_index_type { using type = T; };
template<> struct bwd_index_type<uint8_t> { using type = int; };
template<> struct bwd_index_type<int64_t> { using type = uint64_t; };

template <typename scalar_t, typename index_t>
__global__ void nll_loss_backward_reduce_cuda_kernel_2d(
    scalar_t* grad_input,
    const scalar_t* grad_output,
    const index_t* target,
    const scalar_t* weights,
    const scalar_t* total_weight,
    bool size_average,
    int nframe,
    int ndim,
    int64_t n_classes,
    int64_t ignore_index) {
  using bwd_index_t = typename bwd_index_type<index_t>::type;
  const auto grad = -(size_average ? *grad_output / *total_weight
                                   : *grad_output);

  for (int i = threadIdx.x; i < nframe; i += blockDim.x) {
    const index_t t = target[i];
    if (t != ignore_index) {
      CHECK_INDEX_IN_CLASS(t, n_classes);
      // NOTE(crcrpar): this index could overflow in int64_t as `t` itself can be close to the max.
      const bwd_index_t index = static_cast<bwd_index_t>(i) * ndim + t;
      if constexpr(!std::is_unsigned_v<decltype(index)>) {
        CUDA_KERNEL_ASSERT(index >= 0);
      }
      grad_input[index] = weights != nullptr ? weights[t] * grad : grad;
    }
  }
}

// Host tracing: the same conversion as the forward host above.
void nll_loss_backward_out_cuda_template(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    const Tensor& input_,
    const Tensor& target_,
    const Tensor& total_weight,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  namespace ht = at::cuda::host_trace;
  auto target = *target_.expect_contiguous();
  auto input = *input_.expect_contiguous();
  auto grad_input = *grad_input_.expect_contiguous();
  auto grad_output = *grad_output_.expect_contiguous();

  int64_t n_dims = input.dim();
  c10::SymInt n_classes = input.sym_size(-1);
  c10::SymInt batch_size = n_dims == 1 ? c10::SymInt(1) : input.sym_size(0);

  auto weight_ = weight.defined() ? weight.contiguous() : weight;

  if (reduction == at::Reduction::None && n_dims == 2) {
    if (batch_size == 0) {
      // This guards from unnecessary operations and launching CUDA kernel with 0 blocks.
      return;
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss_backward_no_reduce_cuda_kernel",
        [&] {
          AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
              target.scalar_type(),
              "nll_loss_backward_no_reduce_cuda_kernel_index",
              [&] {
                ht::Traced<PackedTensorAccessor64<const scalar_t, 1>> grad_output_acc;
                grad_output_acc.fill(grad_output, ht::sym_const_data_ptr<scalar_t>(grad_output));
                ht::Traced<PackedTensorAccessor64<scalar_t, 2>> grad_input_acc;
                grad_input_acc.fill(grad_input, ht::sym_mutable_data_ptr<scalar_t>(grad_input));
                // GET_BLOCKS(batch_size) on a symbolic count
                const ht::Grid blocks(((batch_size - 1) / at::cuda::detail::CUDA_NUM_THREADS + 1)
                    .min(c10::SymInt(static_cast<int64_t>(std::numeric_limits<int>::max()))));
                ht::launch(nll_loss_backward_no_reduce_cuda_kernel<scalar_t, index_t>,
                       blocks,
                       at::cuda::detail::CUDA_NUM_THREADS,
                       0,
                       at::cuda::getCurrentCUDAStream(),
                        batch_size,
                        ht::sym_const_data_ptr<index_t>(target),
                        grad_output_acc,
                        grad_input_acc,
                        weight.defined() ? ht::sym_const_data_ptr<scalar_t>(weight_) : c10::SymInt(0),
                        n_classes,
                        ignore_index);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
    return;
  }

  if (n_dims == 1) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss_backward_reduce_cuda_kernel_1d",
        [&] {
          AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
              target.scalar_type(),
              "nll_loss_backward_reduce_cuda_kernel_1d_index",
              [&] {
                ht::launch(nll_loss_backward_reduce_cuda_kernel_1d<scalar_t, index_t>,
                    1, 1, 0, at::cuda::getCurrentCUDAStream(),
                        ht::sym_mutable_data_ptr<scalar_t>(grad_input),
                        ht::sym_const_data_ptr<scalar_t>(grad_output),
                        weight.defined() ? ht::sym_const_data_ptr<scalar_t>(weight_)
                                         : c10::SymInt(0),
                        ht::sym_const_data_ptr<index_t>(target),
                        ht::sym_const_data_ptr<scalar_t>(total_weight),
                        reduction == at::Reduction::Mean,
                        n_classes,
                        ignore_index);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss_backward_reduce_cuda_kernel_2d",
        [&] {
          AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
              target.scalar_type(),
              "nll_loss_backward_reduce_cuda_kernel_2d_index",
              [&] {
            ht::launch(nll_loss_backward_reduce_cuda_kernel_2d<scalar_t, index_t>,
                1, nll_loss_threads(input.sym_size(0)), 0, at::cuda::getCurrentCUDAStream(),
                    ht::sym_mutable_data_ptr<scalar_t>(grad_input),
                    ht::sym_const_data_ptr<scalar_t>(grad_output),
                    ht::sym_const_data_ptr<index_t>(target),
                    weight.defined() ? ht::sym_const_data_ptr<scalar_t>(weight_) : c10::SymInt(0),
                    ht::sym_const_data_ptr<scalar_t>(total_weight),
                    reduction == at::Reduction::Mean,
                    input.sym_size(0),
                    input.sym_size(1),
                    n_classes,
                    ignore_index);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
        });
  }
}

#undef AT_DISPATCH_NLL_LOSS_INDEX_TYPES

} // namespace

TORCH_IMPL_FUNC(nll_loss_forward_out_cuda)
(const Tensor& self,
 const Tensor& target,
 const OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& output,
 const Tensor& total_weight) {
  const Tensor& weight = weight_opt.getTensorRef();
  nll_loss_forward_out_cuda_template(
      output, total_weight, self, target, weight, reduction, ignore_index);
}

TORCH_IMPL_FUNC(nll_loss_backward_out_cuda)
(const Tensor& grad_output,
 const Tensor& self,
 const Tensor& target,
 OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& total_weight,
 const Tensor& grad_input) {
  const Tensor& weight = weight_opt.getTensorRef();
  grad_input.zero_();
  nll_loss_backward_out_cuda_template(
      grad_input,
      grad_output,
      self,
      target,
      total_weight,
      weight,
      reduction,
      ignore_index);
}

// The converted hosts through outputs the caller allocated (the trace mode
// allocates them as traced roots): the structured metas' checks (LossNLL.cpp)
// on symbolic sizes, then the bodies of the two IMPL_FUNCs above.
static void nll_loss_check_inputs(const Tensor& self, const Tensor& target, const Tensor& weight) {
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(
      target.dim() <= 1,
      "0D or 1D target tensor expected, multi-target not supported");
  TORCH_CHECK(
      target.scalar_type() == kLong || target.scalar_type() == kByte,
      "expected target dtype to be Long or Byte, but got ",
      target.scalar_type());
  TORCH_CHECK(
      !weight.defined() || (weight.dim() == 1 && weight.sym_numel() == self.sym_size(-1)),
      "weight tensor should be defined either for all ",
      self.sym_size(-1),
      " classes or no classes"
      " but got weight tensor of shape: ",
      weight.sym_sizes());
}

std::tuple<Tensor, Tensor> host_trace_nll_loss_forward_out(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& output,
    const Tensor& total_weight) {
  const Tensor& weight = weight_opt.has_value() ? *weight_opt : Tensor();
  nll_loss_check_inputs(self, target, weight);
  if (self.dim() == 1 && target.dim() == 1) {
    TORCH_CHECK_VALUE(
        target.sym_size(0) == 1,
        "For 1D input, 1D target must have size 1, but got target size: ",
        target.sym_size(0));
  }
  TORCH_CHECK(
      self.dim() == 1 || (self.sym_size(0) == target.sym_size(0)),
      "size mismatch (got input: ",
      self.sym_sizes(),
      ", target: ",
      target.sym_sizes(),
      ")")
  nll_loss_forward_out_cuda_template(
      output, total_weight, self, target, weight, reduction, ignore_index);
  return std::make_tuple(output, total_weight);
}

Tensor host_trace_nll_loss_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight,
    const Tensor& grad_input) {
  const Tensor& weight = weight_opt.has_value() ? *weight_opt : Tensor();
  nll_loss_check_inputs(self, target, weight);
  auto no_batch_dim = self.dim() == 1  && target.dim() == 0;
  TORCH_CHECK(
      no_batch_dim || (self.sym_size(0) == target.sym_size(0)),
      "size mismatch (got input: ",
      self.sym_sizes(),
      ", target: ",
      target.sym_sizes(),
      ")")
  TORCH_CHECK(
      total_weight.sym_numel() == 1,
      "expected total_weight to be a  single element tensor, got: ",
      total_weight.sym_sizes(),
      " (",
      total_weight.sym_numel(),
      " elements)");
  if (reduction == Reduction::None && self.dim() == 2) {
    TORCH_CHECK(
        grad_output.dim() == 1 && grad_output.sym_size(0) == self.sym_size(0),
        "Expected grad_output of size [", self.sym_size(0), "], but got: ",
        grad_output.sym_sizes());
  } else {
    TORCH_CHECK(
        grad_output.dim() <= 1 && grad_output.sym_numel() == 1,
        "Expected a single element grad_output tensor, but got: ",
        grad_output.sym_sizes());
  }
  grad_input.zero_();
  nll_loss_backward_out_cuda_template(
      grad_input,
      grad_output,
      self,
      target,
      total_weight,
      weight,
      reduction,
      ignore_index);
  return grad_input;
}
}  // namespace at::native
