#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/DistributionTemplates.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <utility>
#include <functional>

#include <ATen/native/Distributions.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace at::native {

void bernoulli_tensor_kernel(const TensorBase &self, const TensorBase &p_, std::optional<Generator> gen_) {
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::bernoulli_kernel(self, p_, generator);
}

void bernoulli_scalar_kernel(const TensorBase &self, double p, std::optional<Generator> gen) {
  auto iter = TensorIterator::borrowing_nullary_op(self);
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::bernoulli_kernel(iter, p, generator);
}

REGISTER_DISPATCH(bernoulli_tensor_stub, &bernoulli_tensor_kernel)
REGISTER_DISPATCH(bernoulli_scalar_stub, &bernoulli_scalar_kernel)

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced entries of bernoulli_.float and bernoulli_.Tensor (the two bernoulli_impl_ of
// ATen/native/DistributionTemplates.h on the sibling iterator), compiled here so the entries and
// the real hosts above instantiate the one kernel over the header's functors (DECISIONS E36):
// the tape's launch is eager's function object, not a twin. The philox increment is declared on
// the tape as an expression of the element count (ti/DistributionSym.cuh). Outside a trace an
// entry runs the same launch in ordinary mode, which is how the parity test compares it with the
// real op.
#include <ATen/cuda/host_trace/ti/DistributionSym.cuh>
#include <ATen/cuda/host_trace/ti/Ops.h>
#include <ATen/cuda/host_trace/ti/EagerViews.cuh>
#include <ATen/native/CanUse32BitIndexMath.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace at::cuda::host_trace {

// CUDA_tensor_apply2's functor with the philox state as its member: an `rng` field of the launch
template <class scalar_t, class prob_t>
struct Traced<at::native::templates::cuda::BernoulliTensorFunctor<scalar_t, prob_t>> : TracedBase {
  using P = at::native::templates::cuda::BernoulliTensorFunctor<scalar_t, prob_t>;
  P pod;
  BytesField<offsetof(P, philox_args), sizeof(PhiloxCudaState)> philox_args{this, "philox_args"};
  explicit Traced(const PhiloxCudaState& st) : TracedBase(&pod, sizeof(P)), pod() {
    new (static_cast<void*>(philox_args)) PhiloxCudaState(st);
  }
};

} // namespace at::cuda::host_trace

namespace at::cuda::host_trace::ti {

namespace {

// IndexUtils.cu maybeOverlappingIndices on symbolic sizes and strides: every comparison a guard
bool maybe_overlapping_indices_sym(const Tensor& t) {
  std::vector<std::pair<c10::SymInt, c10::SymInt>> info; // (size, stride) of the size > 1 dims
  for (const auto i : c10::irange(t.dim())) {
    const c10::SymInt size = t.sym_size(i);
    if (size > 1) {
      const c10::SymInt stride = t.sym_stride(i);
      if (stride < 1) {
        return true;
      }
      info.emplace_back(size, stride);
    }
  }
  if (info.empty()) {
    return false;
  }
  std::sort(info.begin(), info.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
  for (size_t i = 0; i + 1 < info.size(); ++i) {
    if ((info[i].first - 1) * info[i].second >= info[i + 1].second) {
      return true;
    }
  }
  return false;
}

// CUDAApplyUtils.cuh rearrangeDims over the two TensorInfo proxies (the same dims and sizes after
// expand_inplace): strides compared as guards, dims swapped as the real one swaps them
template <class A, class B>
void rearrange_dims_sym(A& a, B& b) {
  const int dims = a.dims;
  if (b.dims != dims) {
    return;
  }
  for (int j = 0; j < dims; ++j) {
    if (c10::SymInt(a.sizes[j]) != c10::SymInt(b.sizes[j])) {
      return;
    }
  }
  for (int i = 0; i < dims - 1; ++i) {
    if (c10::SymInt(a.sizes[i]) == 1) {
      continue;
    }
    for (int j = i + 1; j < dims; ++j) {
      if (c10::SymInt(a.sizes[j]) == 1) {
        continue;
      }
      bool has_increasing_strides = false;
      bool has_decreasing_strides = false;
      const c10::SymInt a_i = a.strides[i], a_j = a.strides[j];
      const c10::SymInt b_i = b.strides[i], b_j = b.strides[j];
      for (const auto& [stride_i, stride_j] : {std::pair{a_i, a_j}, std::pair{b_i, b_j}}) {
        if (stride_i < stride_j) {
          has_increasing_strides = true;
        } else if (stride_i > stride_j) {
          has_decreasing_strides = true;
        }
      }
      if (has_increasing_strides && !has_decreasing_strides) {
        const c10::SymInt a_size_i = a.sizes[i], a_size_j = a.sizes[j];
        a.sizes[i] = a_size_j;
        a.sizes[j] = a_size_i;
        a.strides[i] = a_j;
        a.strides[j] = a_i;
        const c10::SymInt b_size_i = b.sizes[i], b_size_j = b.sizes[j];
        b.sizes[i] = b_size_j;
        b.sizes[j] = b_size_i;
        b.strides[i] = b_j;
        b.strides[j] = b_i;
      }
    }
  }
}

// CUDA_tensor_apply2<scalar_t, const prob_t, 4, Op, 512, 2>(ret, p, functor) on the trace's values:
// the grid from the symbolic count, the two TensorInfo proxies rearranged and collapsed as the real
// host leaves them, the (dims, dims) case switch, eager's kernelPointwiseApply2 instantiation
// through the typed launch
template <typename scalar_t, typename prob_t>
void bernoulli_tensor_traced_kernel(const Tensor& ret, const Tensor& p, const PhiloxCudaState& philox_args) {
  using Op = at::native::templates::cuda::BernoulliTensorFunctor<scalar_t, prob_t>;
  constexpr int step = 4;
  constexpr int max_threads_per_block = 512;
  constexpr int min_blocks_per_sm = 2;
  KernelChoice choice;
  const c10::SymInt total = ret.sym_numel();
  if (total != p.sym_numel()) {
    return;
  }
  if (ret.dim() > MAX_TENSORINFO_DIMS || p.dim() > MAX_TENSORINFO_DIMS) {
    return;
  }
  if (total == 0) {
    return;
  }
  const c10::SymInt max_grid_x(static_cast<int64_t>(at::cuda::getCurrentDeviceProperties()->maxGridSize[0]));
  const c10::SymInt grid_x = ceil_div_sym(total, c10::SymInt(max_threads_per_block * step)).min(max_grid_x);
  if (maybe_overlapping_indices_sym(ret)) {
    decline("host_trace: bernoulli_ on a tensor whose indices may overlap (CUDA_tensor_apply2's contiguous detour) is not traced (declined)");
  }
  if (!at::native::canUse32BitIndexMath(ret) || !at::native::canUse32BitIndexMath(p)) {
    decline("host_trace: bernoulli_ above 2^31 elements (64-bit indexing) is not traced (declined)");
  }
  Traced<at::cuda::detail::TensorInfo<scalar_t, unsigned int>> a;
  Traced<at::cuda::detail::TensorInfo<const prob_t, unsigned int>> b;
  a.fill(ret);
  b.fill(p);
  rearrange_dims_sym(a, b);
  a.collapseDims();
  b.collapseDims();
  Traced<Op> op(philox_args);
  auto stream = at::cuda::getCurrentCUDAStream();
  Grid grid(grid_x);
  const Block block{c10::SymInt(max_threads_per_block)};
  auto run = [&](auto adims, auto bdims) {
    launch(at::cuda::kernelPointwiseApply2<Op, scalar_t, const prob_t, unsigned int, decltype(adims)::value, decltype(bdims)::value, step, max_threads_per_block, min_blocks_per_sm>,
           grid, block, 0, stream, a, b, total, op);
  };
  auto run_b = [&](auto adims) {
    switch (b.dims) {
      case 1:
        run(adims, std::integral_constant<int, 1>{});
        break;
      case 2:
        run(adims, std::integral_constant<int, 2>{});
        break;
      default:
        run(adims, std::integral_constant<int, -1>{});
        break;
    }
  };
  switch (a.dims) {
    case 1:
      run_b(std::integral_constant<int, 1>{});
      break;
    case 2:
      run_b(std::integral_constant<int, 2>{});
      break;
    default:
      run_b(std::integral_constant<int, -1>{});
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

Tensor& bernoulli_scalar_traced(Tensor& self, double p, const std::optional<at::Generator>& gen_) {
  c10::cuda::CUDAGuard device_guard(self.device());
  auto* gen = distribution_generator("bernoulli_", self, gen_);
  TORCH_CHECK(0 <= p && p <= 1, "bernoulli_ expects p to be in [0, 1], but got p=", p);
  if (is_empty(self)) {
    return self;
  }
  assert_no_internal_overlap_sym(self);
  TensorIteratorSym iter = nullary_iterator(self);
  at::native::templates::cuda::bernoulli_kernel(iter, p, gen);
  return self;
}

Tensor& bernoulli_tensor_traced(Tensor& self, const Tensor& p_, const std::optional<at::Generator>& gen_) {
  c10::cuda::CUDAGuard device_guard(self.device());
  auto* gen = distribution_generator("bernoulli_", self, gen_);
  if (is_empty(self)) {
    return self;
  }
  assert_no_internal_overlap_sym(self);
  // templates::cuda::bernoulli_kernel(self, p_, gen) asks the generator for 10 offsets
  // (curand_uniform4 once per thread), which the generator rounds up to 12
  // (CUDAGeneratorState::increase, Note [Why enforce RNG offset % 4 == 0?]): the advance the
  // tape declares is the rounded one; the probabilities are double for a double self, float
  // otherwise
  const int64_t increment = rng_increment(c10::SymInt(12));
  PhiloxCudaState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(increment);
  }
  TORCH_CHECK(at::isFloatingType(p_.scalar_type()), "expected probabilities tensor to have floating type, got ", p_.scalar_type());
  const auto p_type = self.dtype() == at::kDouble ? at::kDouble : at::kFloat;
  auto p_cuda = p_.to(TensorOptions().device(self.device()).dtype(p_type));
  auto p = expand_inplace(self, p_cuda);
  AT_DISPATCH_ALL_TYPES_AND3(
    at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, self.scalar_type(), "bernoulli_tensor_cuda_self_", [&] {
      if constexpr (std::is_same_v<scalar_t, double>) {
        bernoulli_tensor_traced_kernel<double, double>(self, *p, rng_engine_inputs);
      } else {
        bernoulli_tensor_traced_kernel<scalar_t, float>(self, *p, rng_engine_inputs);
      }
   });
  return self;
}

} // namespace at::cuda::host_trace::ti
