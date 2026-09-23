#pragma once

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ExpandBase.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/util/Half.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/core/DistributionsHelper.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <cstdint>
#include <limits>
#include <utility>
#include <mutex>
#include <tuple>
#include <type_traits>

namespace at {
namespace native {
namespace {

// launch bounds used for kernels utilizing TensorIterator
const uint32_t block_size_bound = 256;
const uint32_t grid_size_bound = 4;
// At the time of writing, there is no curand_* call that increments the offset by more than 4.
// See: https://docs.nvidia.com/cuda/archive/11.8.0/curand/group__DEVICE.html
const uint32_t max_generator_offsets_per_curand_call = 4;

// utility function that calculates proper philox_offset
// for distributions utilizing TensorIterator. For distributions using
// TensorIterator, we are using a grid-stride loop with each
// thread yielding one element per thread. For the edge of the grid-stride
// loop, if the tensor size is large, the unroll loop will kick in and the float4
// from curand4 will start getting utilized (for common tensor sizes, we end up
// using rand.x from each thread). The philox_offset calculation was changed to
// (number of elements per thread * maximum generator increment per "curand_*" call), which makes
// sure that philox offset increment is not less than the number of randoms used
// in each thread.
std::tuple<uint64_t, dim3, dim3> calc_execution_policy(const int64_t total_elements, const uint32_t unroll_factor) {
  const uint64_t numel = static_cast<uint64_t>(total_elements);
  const uint32_t block_size = block_size_bound;
  dim3 dim_block(block_size);
  dim3 grid((numel + block_size - 1) / block_size);
  uint32_t blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
  grid.x = std::min(
      static_cast<uint32_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * blocks_per_sm,
      grid.x);
  //number of times random will be generated per thread, to offset philox counter in thc random state
  uint64_t counter_offset = ((numel - 1) / (block_size * grid.x * unroll_factor) + 1) * max_generator_offsets_per_curand_call;
  return std::make_tuple(counter_offset, grid, dim_block);
}

// grid stride loop kernel for distributions
template<typename accscalar_t, int unroll_factor, typename dist_t, typename transform_t>
C10_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__ void distribution_elementwise_grid_stride_kernel(int64_t numel,
                                                            PhiloxCudaState philox_args,
                                                            const dist_t dist_func,
                                                            const transform_t transform_func) {
  auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  int64_t idx = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);

  int64_t rounded_size = ((numel - 1)/(blockDim.x * gridDim.x * unroll_factor)+1) *
      blockDim.x * gridDim.x * unroll_factor;
  for(int64_t linear_index = idx; linear_index < rounded_size; linear_index += blockDim.x * gridDim.x * unroll_factor) {
    auto rand = dist_func(&state);
    #pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int64_t li = linear_index + blockDim.x * gridDim.x * ii;
      if (li < numel) {
        transform_func(li, static_cast<accscalar_t>((&rand.x)[ii]));
      }
    }
    __syncthreads();
  }
}

// The store closures of distribution_nullary_kernel as named functors (members in the closures'
// first-use order): eager's host below and the traced sibling
// (ATen/cuda/host_trace/ti/DistributionSym.cuh) instantiate the one kernel over them.
template<typename scalar_t, typename accscalar_t, typename transform_t>
struct DistributionTrivialStore {
  char* out_data;
  int stride0;
  transform_t transform_func;
  __device__ void operator()(int idx, accscalar_t rand) const {
    scalar_t* out = (scalar_t*)&out_data[stride0 * idx];
    *out = transform_func(rand);
  }
};

template<typename scalar_t, typename accscalar_t, typename transform_t>
struct DistributionStridedStore {
  OffsetCalculator<1> offset_calc;
  char* out_data;
  transform_t transform_func;
  __device__ void operator()(int idx, accscalar_t rand) const {
    auto offsets = offset_calc.get(idx);
    scalar_t* out = (scalar_t*)&out_data[offsets[0]];
    *out = transform_func(rand);
  }
};

/**
 * distribution_nullary_kernel is analogous to gpu_kernel in
 * ATen/native/cuda/Loops.cuh. Like gpu_kernel, it uses
 * TensorIterator to launch a kernel. However, the differences are
 *   - it launches a grid-stride loop based kernel. The kernel is not
 *     generic like elementwise_kernel in Loops.cuh and is specialized
 *     for the distribution kernels here.
 *   - For big size tensors, we can launch multiple kernels recursively
 *     (i.e. if (!iter.can_use_32bit_indexing())) and hence, the philox
 *     offset calculation is done in this function.
 *
 * FIXME: Can we specialize elementwise_kernel and launch_kernel in Loops.cuh
 * to have grid-stride loop kernel and then use that to launch our distribution
 * kernels? Note that we need a grid-stride loop kernel because, we found by testing
 * that it achieves peak effective bandwidth.
 */
template<typename scalar_t,
         typename accscalar_t,
         typename dist_func_return_t,
         typename RNG,
         typename dist_t,
         typename transform_t>
void distribution_nullary_kernel(at::TensorIteratorBase& iter,
                                 RNG gen,
                                 const dist_t& dist_func,
                                 const transform_t transform_func) {
  const int unroll_factor = sizeof(dist_func_return_t) / sizeof(accscalar_t);
  TORCH_CHECK(unroll_factor >= 1, "unroll_factor must be >= 1.");
  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }

  auto [counter_offset, grid, block] = calc_execution_policy(numel, unroll_factor);
  PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(counter_offset);
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      distribution_nullary_kernel<scalar_t, accscalar_t, dist_func_return_t>(sub_iter,
        gen, dist_func, transform_func);
    }
    return;
  }

  char* out_data = (char*)iter.data_ptr(0);

  auto stream = at::cuda::getCurrentCUDAStream();
  if (iter.is_trivial_1d()) {
    auto strides = iter.get_inner_strides();
    int stride0 = strides[0];
    distribution_elementwise_grid_stride_kernel<accscalar_t, unroll_factor><<<grid, block, 0, stream>>>(
      numel,
      rng_engine_inputs,
      dist_func,
      DistributionTrivialStore<scalar_t, accscalar_t, transform_t>{out_data, stride0, transform_func}
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto offset_calc = make_offset_calculator<1>(iter);
    distribution_elementwise_grid_stride_kernel<accscalar_t, unroll_factor><<<grid, block, 0, stream>>>(
      numel,
      rng_engine_inputs,
      dist_func,
      DistributionStridedStore<scalar_t, accscalar_t, transform_t>{offset_calc, out_data, transform_func}
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

// Binary kernel
template <typename func_t, typename inp_offset_calc_t, typename out_offset_calc_t>
__global__ void distribution_binary_elementwise_kernel(
    int numel,
    func_t f,
    PhiloxCudaState philox_args,
    typename function_traits<func_t>::result_type *output_data,
    const typename function_traits<func_t>::template arg<1>::type *input_data_1,
    const typename function_traits<func_t>::template arg<2>::type *input_data_2,
    inp_offset_calc_t inp_calc,
    out_offset_calc_t out_calc) {
  auto seeds = at::cuda::philox::unpack(philox_args);

  using input_t_1 = typename function_traits<func_t>::template arg<1>::type;
  using input_t_2 = typename function_traits<func_t>::template arg<2>::type;

  input_t_1 inputs_1[thread_work_size()];
  input_t_2 inputs_2[thread_work_size()];

  int base_index = block_work_size() * blockIdx.x;
  int remaining = std::min<int>(numel - base_index, block_work_size());

  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds),
              blockIdx.x * blockDim.x + threadIdx.x,
              std::get<1>(seeds),
              &state);

  // load data into registers
  int thread_idx = threadIdx.x;
  #pragma unroll
  for (int i = 0; i < thread_work_size(); i++) {
    if (thread_idx >= remaining) {
      break;
    }
    int input_idx = thread_idx + base_index;
    auto offsets = inp_calc.get(input_idx);
    inputs_1[i] = input_data_1[offsets[0]];
    inputs_2[i] = input_data_2[offsets[1]];

    thread_idx += num_threads();
  }

  // compute and store
  thread_idx = threadIdx.x;
  #pragma unroll
  for (int i = 0; i < thread_work_size(); i++) {
    if (thread_idx >= remaining) {
      break;
    }
    int input_idx = thread_idx + base_index;
    auto offsets = out_calc.get(input_idx);
    output_data[offsets[0]] = f(state, inputs_1[i], inputs_2[i]);
    thread_idx += num_threads();
  }
}

template <typename func_t>
void distribution_binary_kernel(TensorIteratorBase &iter, PhiloxCudaState philox_args, const func_t &f) {
  static_assert(std::is_same_v<typename function_traits<func_t>::template arg<0>::type, curandStatePhilox4_32_10_t&>, "the first argument of functor must be curandStatePhilox4_32_10_t");
  using input_t_1 = typename function_traits<func_t>::template arg<1>::type;
  using input_t_2 = typename function_traits<func_t>::template arg<2>::type;
  using output_t = typename function_traits<func_t>::result_type;

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      distribution_binary_kernel(sub_iter, philox_args, f);
    }
    return;
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(iter.can_use_32bit_indexing());

  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }

  output_t *output_data = static_cast<output_t *>(iter.data_ptr(0));
  const input_t_1 *input_data_1 = static_cast<const input_t_1 *>(iter.data_ptr(1));
  const input_t_2 *input_data_2 = static_cast<const input_t_2 *>(iter.data_ptr(2));

  int64_t grid = (numel + block_work_size() - 1) / block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();

  if (iter.is_contiguous()) {
    distribution_binary_elementwise_kernel<<<grid,num_threads(), 0, stream>>>(
        numel, f, philox_args, output_data, input_data_1, input_data_2,
        TrivialOffsetCalculator<2>(), TrivialOffsetCalculator<1>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    distribution_binary_elementwise_kernel<<<grid, num_threads(), 0, stream>>>(
        numel, f, philox_args, output_data, input_data_1, input_data_2,
        make_input_offset_calculator<2>(iter), make_output_offset_calculator(iter));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

} // namespace
}} // namespace at::native


namespace at {
namespace native {
namespace templates {
namespace cuda {

// The hosts below are templates over the iterator type: eager's at::TensorIteratorBase, or the
// SymInt-typed sibling iterator of a trace (ATen/cuda/host_trace/ti/DistributionSym.cuh, whose
// distribution_nullary_kernel overload argument-dependent lookup selects). The curand draws and
// each distribution's transform are named functors (formerly the lambdas of each host), so both
// iterators instantiate the one kernel (DECISIONS E36).

struct Curand4Functor {
  __device__ uint4 operator()(curandStatePhilox4_32_10_t* state) const {
    return curand4(state);
  }
};

struct Curand4Uint64Functor {
  __device__ ulonglong2 operator()(curandStatePhilox4_32_10_t* state) const {
    ulonglong2 ret;
    uint4 rand_val = curand4(state);
    ret.x = (static_cast<uint64_t>(rand_val.x) << 32) | rand_val.y;
    ret.y = (static_cast<uint64_t>(rand_val.z) << 32) | rand_val.w;
    return ret;
  }
};

struct CurandUniform4Functor {
  __device__ float4 operator()(curandStatePhilox4_32_10_t* state) const {
    return curand_uniform4(state);
  }
};

struct CurandUniform2DoubleFunctor {
  __device__ double2 operator()(curandStatePhilox4_32_10_t* state) const {
    return curand_uniform2_double(state);
  }
};

struct CurandNormal4Functor {
  __device__ float4 operator()(curandStatePhilox4_32_10_t* state) const {
    return curand_normal4(state);
  }
};

struct CurandNormal2DoubleFunctor {
  __device__ double2 operator()(curandStatePhilox4_32_10_t* state) const {
    return curand_normal2_double(state);
  }
};

template<typename scalar_t, typename rand_t>
struct UniformIntFromToFunctor {
  uint64_t range;
  int64_t base;
  __device__ scalar_t operator()(rand_t rand) const {
    return transformation::uniform_int_from_to<scalar_t>(rand, range, base);
  }
};

template<typename scalar_t>
struct UniformIntFullRangeFunctor {
  __device__ scalar_t operator()(uint64_t rand) const {
    return transformation::uniform_int_full_range<scalar_t>(rand);
  }
};

template<typename scalar_t, typename rand_t>
struct UniformIntFunctor {
  __device__ scalar_t operator()(rand_t rand) const {
    return transformation::uniform_int<scalar_t>(rand);
  }
};

template<typename scalar_t, typename accscalar_t>
struct NormalFunctor {
  accscalar_t mean;
  accscalar_t std;
  __device__ scalar_t operator()(accscalar_t rand) const {
    return static_cast<scalar_t>(transformation::normal<accscalar_t>(rand, mean, std));
  }
};

template<typename scalar_t, typename opmath_t>
struct UniformFunctor {
  opmath_t range;
  scalar_t from;
  scalar_t to;
  __device__ scalar_t operator()(opmath_t rand) const {
    // Compute output value before reversing the bounds
    // BEFORE TOUCHING THIS CODE READ: https://github.com/pytorch/pytorch/issues/96947
    auto value = static_cast<scalar_t>(rand * range + from);
    // reverse the bounds of curand4 from (0, 1] to [0, 1)
    // Note that this method is from legacy THCTensorRandom and is likely to give
    // you more 0-s, since, the probability of getting 1-s is higher than 0-s and
    // by reversing the bounds, we are flipping the probabilities of 1-s and 0-s.
    // BEFORE TOUCHING THIS CODE READ: https://github.com/pytorch/pytorch/issues/16706
    auto reverse_bound_value = value == to ? from : value;
    return reverse_bound_value;
  }
};

template<typename scalar_t, typename accscalar_t>
struct LogNormalFunctor {
  accscalar_t mean;
  accscalar_t std;
  __device__ scalar_t operator()(accscalar_t rand) const {
    return static_cast<scalar_t>(transformation::log_normal<accscalar_t>(transformation::normal<accscalar_t>(rand, mean, std)));
  }
};

template<typename scalar_t, typename accscalar_t>
struct GeometricFunctor {
  double p;
  __device__ scalar_t operator()(accscalar_t rand) const {
    return static_cast<scalar_t>(transformation::geometric<accscalar_t>(rand, p));
  }
};

template<typename scalar_t, typename accscalar_t>
struct ExponentialFunctor {
  accscalar_t lambda;
  __device__ scalar_t operator()(accscalar_t rand) const {
    return static_cast<scalar_t>(transformation::exponential<accscalar_t>(rand, lambda));
  }
};

template<typename scalar_t, typename accscalar_t>
struct CauchyFunctor {
  accscalar_t median;
  accscalar_t sigma;
  __device__ scalar_t operator()(accscalar_t rand) const {
    return static_cast<scalar_t>(transformation::cauchy<accscalar_t>(rand, median, sigma));
  }
};

template<typename scalar_t, typename accscalar_t>
struct BernoulliFunctor {
  double p;
  __device__ scalar_t operator()(accscalar_t rand) const {
    return static_cast<scalar_t>(transformation::bernoulli<accscalar_t>(rand, p));
  }
};

// bernoulli_ with a probability tensor: CUDA_tensor_apply2's functor, the philox state its member
template<typename scalar_t, typename prob_t>
struct BernoulliTensorFunctor {
  PhiloxCudaState philox_args;
  __device__ void operator()(
      int n, scalar_t& v1, scalar_t& v2, scalar_t& v3, scalar_t& v4,
      const prob_t& p1, const prob_t& p2, const prob_t& p3, const prob_t& p4) const {
    auto seeds = at::cuda::philox::unpack(philox_args);
    curandStatePhilox4_32_10_t state;
    curand_init(std::get<0>(seeds),
                blockIdx.x * blockDim.x + threadIdx.x,
                std::get<1>(seeds),
                &state);

    // See Note [Register spilling in curand call for CUDA < 10]
    float4 rand = curand_uniform4(&state);
    switch (n) {
      case 4: {
        CUDA_KERNEL_ASSERT(0 <= p4 && p4 <= 1);
        v4 = static_cast<scalar_t>(rand.w <= p4);
        [[fallthrough]];
      }
      case 3: {
        CUDA_KERNEL_ASSERT(0 <= p3 && p3 <= 1);
        v3 = static_cast<scalar_t>(rand.z <= p3);
        [[fallthrough]];
      }
      case 2: {
        CUDA_KERNEL_ASSERT(0 <= p2 && p2 <= 1);
        v2 = static_cast<scalar_t>(rand.y <= p2);
        [[fallthrough]];
      }
      case 1: {
        CUDA_KERNEL_ASSERT(0 <= p1 && p1 <= 1);
        v1 = static_cast<scalar_t>(rand.x <= p1);
      }
    }
  }
};

// ==================================================== Random ========================================================

template<typename RNG, typename Iter>
void random_from_to_kernel(Iter& iter, uint64_t range, int64_t base, RNG gen) {
#ifdef FBCODE_CAFFE2
  AT_DISPATCH_V2(iter.dtype(), "random_from_to_kernel_cuda", AT_WRAP([&] {
    if ((
      std::is_same_v<scalar_t, int64_t> ||
      std::is_same_v<scalar_t, double> ||
      std::is_same_v<scalar_t, float> ||
      std::is_same_v<scalar_t, at::BFloat16>) && range >= 1ULL << 32)
    {
      distribution_nullary_kernel<scalar_t, uint64_t, ulonglong2>(iter,
        gen,
        Curand4Uint64Functor{},
        UniformIntFromToFunctor<scalar_t, uint64_t>{range, base});
    } else {
      distribution_nullary_kernel<scalar_t, uint32_t, uint4>(iter,
        gen,
        Curand4Functor{},
        UniformIntFromToFunctor<scalar_t, uint32_t>{range, base});
    }
   }), AT_EXPAND(AT_ALL_TYPES), kBool, kHalf, kBFloat16, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
#else
  AT_DISPATCH_V2(iter.dtype(), "random_from_to_kernel_cuda", AT_WRAP([&] {
    if (range >= 1ULL << 28) // allow approx 5% skew in uniform int generation using %
    {
      distribution_nullary_kernel<scalar_t, uint64_t, ulonglong2>(iter,
        gen,
        Curand4Uint64Functor{},
        UniformIntFromToFunctor<scalar_t, uint64_t>{range, base});
    } else {
      distribution_nullary_kernel<scalar_t, uint32_t, uint4>(iter,
        gen,
        Curand4Functor{},
        UniformIntFromToFunctor<scalar_t, uint32_t>{range, base});
    }
   }), AT_EXPAND(AT_ALL_TYPES), kBool, kHalf, kBFloat16, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
#endif
}

// This is the special kernel to handle single specific case:
// from(inclusive) = std::numeric_limits<int64_t>::lowest()
// to(exclusive) = None (= std::numeric_limits<int64_t>::max() + 1)
template<typename RNG, typename Iter>
void random_full_64_bits_range_kernel(Iter& iter, RNG gen) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::BFloat16, iter.dtype(), "random_full_64_bits_range_kernel_cuda", [&] {
    if constexpr (std::is_same_v<scalar_t, int64_t> ||
                  std::is_same_v<scalar_t, double> ||
                  std::is_same_v<scalar_t, float> ||
                  std::is_same_v<scalar_t, at::BFloat16>) {
      distribution_nullary_kernel<scalar_t, uint64_t, ulonglong2>(iter,
        gen,
        Curand4Uint64Functor{},
        UniformIntFullRangeFunctor<scalar_t>{});
    } else {
      TORCH_CHECK(false, "random_full_64_bits_range_kernel_cuda handles only int64, double, float and bfloat16");
    }
  });
}

template<typename RNG>
struct RandomFromToKernel {
  void operator()(TensorIteratorBase& iter, uint64_t range, int64_t base, std::optional<Generator> gen) {
    random_from_to_kernel(iter, range, base, check_generator<RNG>(gen));
  }
  void operator()(TensorIteratorBase& iter, std::optional<Generator> gen) {
    random_full_64_bits_range_kernel(iter, check_generator<RNG>(gen));
  }
};

template<typename RNG, typename Iter>
void random_kernel(Iter& iter, RNG gen) {
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, iter.dtype(), "random_kernel_cuda", [&] {
    if constexpr (std::is_same_v<scalar_t, double> || std::is_same_v<scalar_t, int64_t>) {
      distribution_nullary_kernel<scalar_t, uint64_t, ulonglong2>(iter, gen,
        Curand4Uint64Functor{},
        UniformIntFunctor<scalar_t, uint64_t>{});
    } else {
      distribution_nullary_kernel<scalar_t, uint32_t, uint4>(iter,
        gen,
        Curand4Functor{},
        UniformIntFunctor<scalar_t, uint32_t>{});
    }
  });
}

template<typename RNG>
struct RandomKernel {
  void operator()(TensorIteratorBase& iter, RNG gen) {
    random_kernel(iter, gen);
  }
};

// ====================================================================================================================

template<typename scalar_t, typename accscalar_t, typename RNG, typename transform_t, typename Iter>
void uniform_and_transform(Iter& iter, RNG gen, transform_t transform) {
  if constexpr (std::is_same_v<scalar_t, double>) {
    distribution_nullary_kernel<scalar_t, accscalar_t, double2>(iter,
      gen,
      CurandUniform2DoubleFunctor{},
      transform);
  } else {
    distribution_nullary_kernel<scalar_t, accscalar_t, float4>(iter,
      gen,
      CurandUniform4Functor{},
      transform);
  }
}

template<typename scalar_t, typename accscalar_t, typename RNG, typename transform_t, typename Iter>
void normal_and_transform(Iter& iter, RNG gen, transform_t transform) {
  if constexpr (std::is_same_v<scalar_t, double>) {
    distribution_nullary_kernel<scalar_t, accscalar_t, double2>(iter,
      gen,
      CurandNormal2DoubleFunctor{},
      transform);
  } else {
    distribution_nullary_kernel<scalar_t, accscalar_t, float4>(iter,
      gen,
      CurandNormal4Functor{},
      transform);
  }
}

// ==================================================== Normal ========================================================

template<typename RNG, typename Iter>
void normal_kernel(Iter& iter, double mean_, double std_, RNG gen) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "normal_kernel_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto mean = static_cast<accscalar_t>(mean_);
    auto std = static_cast<accscalar_t>(std_);
    normal_and_transform<scalar_t, accscalar_t>(iter, gen, NormalFunctor<scalar_t, accscalar_t>{mean, std});
   });
}

template<typename RNG>
void normal_kernel(const TensorBase &self, double mean_, double std_, RNG gen) {
  auto iter = TensorIterator::borrowing_nullary_op(self);
  normal_kernel(iter, mean_, std_, gen);
}

template<typename RNG>
struct NormalKernel {
  void operator()(const TensorBase &self, double mean, double std, std::optional<Generator> gen) {
    normal_kernel(self, mean, std, check_generator<RNG>(gen));
  }
};

// ==================================================== Uniform ========================================================

template<typename RNG, typename Iter>
void uniform_kernel(Iter& iter, double from_, double to_, RNG gen) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "uniform_kernel_cuda", [&] {
    auto from = static_cast<scalar_t>(from_);
    auto to = static_cast<scalar_t>(to_);
    using opmath_t = at::opmath_type<scalar_t>;
    auto range = static_cast<opmath_t>(to-from);
    uniform_and_transform<scalar_t, opmath_t>(iter, gen, UniformFunctor<scalar_t, opmath_t>{range, from, to});
   });
}

template<typename RNG>
struct UniformKernel {
  void operator()(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
    uniform_kernel(iter, from, to, check_generator<RNG>(gen));
  }
};

// ================================================== LogNormal =======================================================

template<typename RNG, typename Iter>
void log_normal_kernel(Iter& iter, double mean_, double std_, RNG gen) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "log_normal_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto mean = static_cast<accscalar_t>(mean_);
    auto std = static_cast<accscalar_t>(std_);
    normal_and_transform<scalar_t, accscalar_t>(iter, gen, LogNormalFunctor<scalar_t, accscalar_t>{mean, std});
   });
}

template<typename RNG>
struct LogNormalKernel {
  void operator()(TensorIteratorBase& iter, double mean, double std, std::optional<Generator> gen) {
    log_normal_kernel(iter, mean, std, check_generator<RNG>(gen));
  }
};

// =================================================== Geometric ======================================================

template<typename RNG, typename Iter>
void geometric_kernel(Iter& iter, double p, RNG gen) {
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "geometric_cuda", [&] {
    using accscalar_t = at::DiscreteDistributionType<scalar_t>::type;
    uniform_and_transform<scalar_t, accscalar_t>(iter, gen, GeometricFunctor<scalar_t, accscalar_t>{p});
  });
}

template<typename RNG>
struct GeometricKernel {
  void operator()(TensorIteratorBase& iter, double p, std::optional<Generator> gen) {
    geometric_kernel(iter, p, check_generator<RNG>(gen));
  }
};

// ================================================== Exponential =====================================================

template<typename RNG, typename Iter>
void exponential_kernel(Iter& iter, double lambda_, RNG gen) {
  TORCH_CHECK(isFloatingType(iter.dtype()), "Exponential distribution is a continuous probability distribution. dtype must be a floating point but you specified ", iter.dtype());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "exponential_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto lambda = static_cast<accscalar_t>(lambda_);
    uniform_and_transform<scalar_t, accscalar_t>(iter, gen, ExponentialFunctor<scalar_t, accscalar_t>{lambda});
   });
}

template<typename RNG>
struct ExponentialKernel {
  void operator()(TensorIteratorBase& iter, double lambda, std::optional<Generator> gen) {
    exponential_kernel(iter, lambda, check_generator<RNG>(gen));
  }
};

// ==================================================== Cauchy ========================================================

template<typename RNG, typename Iter>
void cauchy_kernel(Iter& iter, double median_, double sigma_, RNG gen) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "cauchy_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto median = static_cast<accscalar_t>(median_);
    auto sigma = static_cast<accscalar_t>(sigma_);
    uniform_and_transform<scalar_t, accscalar_t>(iter, gen, CauchyFunctor<scalar_t, accscalar_t>{median, sigma});
   });
}

template<typename RNG>
struct CauchyKernel {
  void operator()(TensorIteratorBase& iter, double median, double sigma, std::optional<Generator> gen) {
    cauchy_kernel(iter, median, sigma, check_generator<RNG>(gen));
  }
};

// ==================================================== Bernoulli =====================================================

template<typename scalar_t, typename prob_t>
void bernoulli_tensor_cuda_kernel(
    const TensorBase &ret, const at::TensorBase &p,
    PhiloxCudaState philox_args) {
  auto functor = BernoulliTensorFunctor<scalar_t, prob_t>{philox_args};
  // The template argument `4` below indicates that we want to operate on four
  // element at each time. See NOTE [ CUDA_tensor_applyN helpers ] for details.
  at::cuda::CUDA_tensor_apply2<scalar_t, const prob_t, 4, decltype(functor),
                               /*max_threads_per_block=*/512,
                               /*min_blocks_per_sm==*/2>(ret, p, functor);
}

template<typename RNG>
void bernoulli_kernel(const TensorBase &self, const TensorBase &p_, RNG gen) {
  PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(10);
  }
  TORCH_CHECK(at::isFloatingType(p_.scalar_type()), "expected probabilities tensor to have floating type, got ", p_.scalar_type());
  // cast probabilities tensor to double for double `self` tensor, and to `float` for everything else
  const auto p_type = self.dtype() == at::kDouble ? at::kDouble : at::kFloat;
  auto p_cuda = p_.to(TensorOptions().device(self.device()).dtype(p_type));
  auto p = expand_inplace(self, p_cuda);
  AT_DISPATCH_ALL_TYPES_AND3(
    at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, self.scalar_type(), "bernoulli_tensor_cuda_self_", [&] {
      if constexpr (std::is_same_v<scalar_t, double>) {
        return bernoulli_tensor_cuda_kernel<double, double>(self, *p, rng_engine_inputs);
      } else {
        return bernoulli_tensor_cuda_kernel<scalar_t, float>(self, *p, rng_engine_inputs);
      }
   });
}

template<typename RNG, typename Iter>
void bernoulli_kernel(Iter& iter, double p, RNG gen) {
  AT_DISPATCH_ALL_TYPES_AND3(
    at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, iter.dtype(), "bernoulli_scalar_cuda_", [&] {
      using accscalar_t = at::DiscreteDistributionType<scalar_t>::type;
      uniform_and_transform<scalar_t, accscalar_t>(iter, gen, BernoulliFunctor<scalar_t, accscalar_t>{p});
   });
}

template<typename RNG>
struct BernoulliKernel {
  void operator()(TensorIteratorBase& iter, double p, std::optional<Generator> gen) {
    bernoulli_kernel(iter, p, check_generator<RNG>(gen));
  }
  void operator()(const TensorBase &self, const TensorBase &p_, std::optional<Generator> gen) {
    bernoulli_kernel(self, p_, check_generator<RNG>(gen));
  }
};

}}}}
