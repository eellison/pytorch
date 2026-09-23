#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/Dispatch.h>
#include <ATen/Utils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/macros/Macros.h>
#include <curand_kernel.h>
#include <utility>

#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/HostTraceFunctor_native_dropout_backward.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_masked_scale_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/native_dropout_backward_native.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace at::native {

namespace {

// philox generates 128 bits of randomness at a time. Kernel uses this explicitly by putting suitably transformed result into float4
// for all members of float4 to be consumed UNROLL has to be 4. Don't change!
// Note: VEC <= 4 (and in most real-world cases will be 4), so same logic applies.
const int UNROLL = 4;

template <
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    int ADims,
    int VEC,
    typename mask_t>
C10_LAUNCH_BOUNDS_2(256, 4)
__global__ void
fused_dropout_kernel_vec(at::cuda::detail::TensorInfo<const scalar_t, IndexType> a,
                         at::cuda::detail::TensorInfo<scalar_t, IndexType> b,
                         at::cuda::detail::TensorInfo<mask_t, IndexType> c,
                         IndexType totalElements, accscalar_t p,
                         PhiloxCudaState philox_args) {
  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  using MaskLoadT = memory::aligned_vector<mask_t, VEC>;

  auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);

  // Helps align the total number of times curand_uniform4 is called by each thread for the same totalElements
  // in the vec=2 and vec=4 cases.
  bool gridxvec_loop_state = false;
  accscalar_t scale = 1.0 / p;

  constexpr int RAND_SIZE = (VEC + 4 - 1) / 4;
  float4 rand[RAND_SIZE];

  // Note: Vectorized loads means we'll stride each thread by an additional VEC factor, as we'll load VEC elements at a time
  for (IndexType linearIndex = idx * VEC;
      linearIndex < totalElements;
      linearIndex += gridDim.x * blockDim.x * VEC) {
    // local storage
    scalar_t src[VEC];
    // We'll use this to actually cause vectorized loads later
    LoadT *value = reinterpret_cast<LoadT*>(&src);

    //curand_uniform_double was pure evil anyway, not doing what it promises, and there's nothing for Halfs, so generate float for everything
    // Note: need a new set of random values per 4 elements -- we'll handle VEC elements in this thread, so need ceil(VEC / 4)
    // sets of rand.
    if ((VEC >= 4) || (gridxvec_loop_state == 0)) {
      #pragma unroll
      for (int ii = 0; ii < RAND_SIZE; ii++) {
        rand[ii] = curand_uniform4(&state);
      }
    } else {
      // sets up the last two values we generated last iteration to be used this iteration.
      rand[0].x = rand[0].z;
      rand[0].y = rand[0].w;
      gridxvec_loop_state ^= 1;
    }

    rand[0].x = rand[0].x < p;
    rand[0].y = rand[0].y < p;
    if constexpr (VEC >= 4) {
      rand[0].z = rand[0].z < p;
      rand[0].w = rand[0].w < p;
    }

    #pragma unroll
    for (int ii = 1; ii < RAND_SIZE; ii++) {
      rand[ii].x = rand[ii].x < p;
      rand[ii].y = rand[ii].y < p;
      rand[ii].z = rand[ii].z < p;
      rand[ii].w = rand[ii].w < p;
    }

    // Note: We explicitly check for is_contiguous() before launching the vectorized kernel
    // and replace IndexToOffset call with linearIndex to allow vectorization of NHWC (or other)
    // ordering.
    // Single vectorized load
    *value = *reinterpret_cast<const LoadT*>(&a.data[linearIndex]);

    scalar_t r[VEC];
    mask_t mask[VEC];

    // Perform the actual computation
    #pragma unroll
    for (int jj = 0; jj < RAND_SIZE; jj++) {
      #pragma unroll
      for (int ii = 0; ii < std::min(VEC, 4); ii++) {
        r[jj * 4 + ii] = src[jj * 4 + ii]*(&rand[jj].x)[ii]*scale;
        mask[jj * 4 + ii] = (mask_t)(&rand[jj].x)[ii];
      }
    }

    // Vectorized writes for both mask & result
    *(reinterpret_cast<LoadT*>(&b.data[linearIndex])) = *reinterpret_cast<LoadT*>(&r[0]);
    *(reinterpret_cast<MaskLoadT*>(&c.data[linearIndex])) = *reinterpret_cast<MaskLoadT*>(&mask[0]);

    __syncthreads();
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    int ADims,
    int BDims = ADims,
    typename mask_t>
C10_LAUNCH_BOUNDS_2(256, 4)
__global__ void
fused_dropout_kernel(cuda::detail::TensorInfo<const scalar_t, IndexType> a,
                     cuda::detail::TensorInfo<scalar_t, IndexType> b,
                     cuda::detail::TensorInfo<mask_t, IndexType> c,
                     IndexType totalElements, accscalar_t p,
                     PhiloxCudaState philox_args) {
  auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);
  accscalar_t scale = 1.0 / p;

  IndexType rounded_size = ((totalElements - 1)/(blockDim.x * gridDim.x * UNROLL)+1) *
        blockDim.x * gridDim.x * UNROLL;
  for (IndexType linearIndex = idx;
       linearIndex < rounded_size;
       linearIndex += gridDim.x * blockDim.x*UNROLL) {
//curand_uniform_double was pure evil anyway, not doing what it promises, and there's nothing for Halfs, so generate float for everything
       float4 rand = curand_uniform4(&state);
       scalar_t src[UNROLL];
       rand.x = rand.x < p;
       rand.y = rand.y < p;
       rand.z = rand.z < p;
       rand.w = rand.w < p;
       for (int ii = 0; ii < UNROLL; ii++) {
           IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < totalElements) {
    // Convert `linearIndex` into an offset of `a`
               const IndexType aOffset =
                   cuda::detail::IndexToOffset<const scalar_t, IndexType, ADims>::get(li, a);
               src[ii] = a.data[aOffset];
           }
       }
       for (int ii = 0; ii < UNROLL; ii++) {
           IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < totalElements) {
    // Convert `linearIndex` into an offset of `b`
               const IndexType bOffset =
                   cuda::detail::IndexToOffset<scalar_t, IndexType, BDims>::get(li, b);
               b.data[bOffset] = src[ii]*(&rand.x)[ii]*scale;
               c.data[bOffset] = (mask_t)(&rand.x)[ii];
           }
       }
       __syncthreads();
  }
}

template<typename mask_t, typename scalar_t, typename accscalar_t>
void masked_scale_kernel(at::Tensor& ret, const at::Tensor& src, const at::Tensor& mask, accscalar_t scale){
   auto iter = at::TensorIteratorConfig()
     .check_all_same_dtype(false)
     .add_output(ret)
     .add_const_input(src)
     .add_const_input(mask)
     .build();

   if constexpr (std::is_same_v<mask_t, bool>) {
     at::native::gpu_kernel(iter, CUDAFunctor_native_dropout_backward<scalar_t>{scale});
   } else {
     at::native::gpu_kernel(
         iter,
         [=]GPU_LAMBDA(const scalar_t src_val, const mask_t mask_val) -> scalar_t {
            return (float)mask_val * src_val * scale;
         });
   }
}

template <typename scalar_t>
int get_vector_size(at::Tensor self, at::Tensor ret, at::Tensor mask) {
  int vec_size = 4;
  // get the vector size
  if (!self.is_non_overlapping_and_dense() || !ret.is_non_overlapping_and_dense() || !mask.is_non_overlapping_and_dense()) {
    vec_size = 1;
  } else {
    vec_size = memory::can_vectorize_up_to<scalar_t>((const char*)self.const_data_ptr());
#ifdef USE_ROCM
    // make sure we don't break assumption that we can't have > 16 elements / thread
    TORCH_INTERNAL_ASSERT(vec_size <= 16, "Value of VEC must be in [2, 4, 8, 16]");
#else
    const int optimal_vec_size = 16 / static_cast<int>(sizeof(scalar_t));
    vec_size = std::min<int>(optimal_vec_size, vec_size);

    // make sure we don't break assumption that we can't have > 4 elements / thread
    TORCH_INTERNAL_ASSERT(vec_size <= 8, "Value of VEC must be in [2, 4, 8]");
#endif
  }

  // check that we'd have no remainders - prefer a smaller vector size with no remainders over a larger vector and remainder.
  bool can_vectorize = true;
  do {
    can_vectorize = self.numel() % vec_size == 0 && ret.numel() % vec_size == 0 && mask.numel() % vec_size == 0;
    if (!can_vectorize) vec_size /= 2;
  } while (vec_size > 1 && !can_vectorize);
  return can_vectorize ? vec_size : 1;
}

template <typename index_type, typename mask_t>
inline void launcher(
    const Tensor& self,
    Tensor& ret,
    Tensor& mask,
    double p,
    const int64_t nelem,
    const PhiloxCudaState rng_engine_inputs,
    dim3 grid,
    dim3 dim_block) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "fused_dropout",
      [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        accscalar_t pa = (accscalar_t)(p);
        auto self_info =
            cuda::detail::getTensorInfo<const scalar_t, index_type>(self);
        auto ret_info =
            cuda::detail::getTensorInfo<scalar_t, index_type>(ret);
        auto mask_info =
            cuda::detail::getTensorInfo<mask_t, index_type>(mask);
        self_info.collapseDims();
        ret_info.collapseDims();
        mask_info.collapseDims(); // ret and mask are collapsed to 1d
                                  // contiguous tensor

        int vec_size = get_vector_size<scalar_t>(self, ret, mask);

        if (vec_size > 1) {
          switch (vec_size) {
            case 16:
              fused_dropout_kernel_vec<
                  scalar_t,
                  accscalar_t,
                  index_type,
                  1,
                  16>
                  <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                      self_info,
                      ret_info,
                      mask_info,
                      nelem,
                      pa,
                      rng_engine_inputs);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              break;
            case 8:
              fused_dropout_kernel_vec<
                  scalar_t,
                  accscalar_t,
                  index_type,
                  1,
                  8>
                  <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                      self_info,
                      ret_info,
                      mask_info,
                      nelem,
                      pa,
                      rng_engine_inputs);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              break;
            case 4:
              fused_dropout_kernel_vec<
                  scalar_t,
                  accscalar_t,
                  index_type,
                  1,
                  4>
                  <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                      self_info,
                      ret_info,
                      mask_info,
                      nelem,
                      pa,
                      rng_engine_inputs);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              break;
            case 2:
              fused_dropout_kernel_vec<
                  scalar_t,
                  accscalar_t,
                  index_type,
                  1,
                  2>
                  <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                      self_info,
                      ret_info,
                      mask_info,
                      nelem,
                      pa,
                      rng_engine_inputs);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              break;
            default:
              TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
          }
        } else {
          switch (self_info.dims) {
            case 1:
              fused_dropout_kernel<scalar_t, accscalar_t, index_type, 1>
                  <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                      self_info,
                      ret_info,
                      mask_info,
                      nelem,
                      pa,
                      rng_engine_inputs);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              break;
            default:
              if (!self.is_contiguous() && ret.is_contiguous() &&
                  mask.is_contiguous()) {
                fused_dropout_kernel<scalar_t, accscalar_t, index_type, -1, 1>
                    <<<grid,
                        dim_block,
                        0,
                        at::cuda::getCurrentCUDAStream()>>>(
                        self_info,
                        ret_info,
                        mask_info,
                        nelem,
                        pa,
                        rng_engine_inputs);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              } else {
                fused_dropout_kernel<scalar_t, accscalar_t, index_type, -1>
                    <<<grid,
                        dim_block,
                        0,
                        at::cuda::getCurrentCUDAStream()>>>(
                        self_info,
                        ret_info,
                        mask_info,
                        nelem,
                        pa,
                        rng_engine_inputs);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }
          }
        }
      });
}

} //anonymous namespace

template <typename mask_t>
std::tuple<Tensor,Tensor>
dropout_cuda(CUDAGeneratorImpl* gen, const Tensor& self, double p){
  Tensor mask = at::empty_like(self, self.options().dtype(c10::CppTypeToScalarType<mask_t>::value));
  const int64_t nelem = self.numel();
  // empty tensors should not get here, but just in case, avoid FPE
  // non-training shot-cut
  if (nelem==0) return std::tuple<Tensor,Tensor>(self.clone(), mask);

  Tensor ret = at::empty_like(self);
  const int64_t block_size = 256;
  unsigned int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/block_size;
  dim3 dim_block(block_size);
  dim3 grid((nelem + block_size -1)/block_size);
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);
//number of times random will be generated per thread, to offset philox counter in thc random state
  int64_t counter_offset = ((nelem - 1)/(block_size*grid.x*UNROLL)+1)*UNROLL;
  PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(counter_offset);
  }
  if (cuda::detail::canUse32BitIndexMath(self)){
    launcher<unsigned int, mask_t>(
        self, ret, mask, p, nelem, rng_engine_inputs, grid, dim_block);
  } else {
    launcher<uint64_t, mask_t>(
        self, ret, mask, p, nelem, rng_engine_inputs, grid, dim_block);
  }
  return std::tuple<Tensor,Tensor>(ret, mask);
}

std::tuple<Tensor,Tensor>
native_dropout_cuda(const Tensor& self, double p, std::optional<bool> train){
  // short-cut for train == false
  if (train.has_value() && !train.value()) {
    return std::make_tuple(self.clone(), at::ones_like(self, self.options().dtype(c10::CppTypeToScalarType<bool>::value)));
  }
  // short-cut
  if (p == 1) {
    // native_dropout_cuda is in derivatives.yaml, so we don't need to add data
    // dependency from output to input for autograd
    auto ret = at::zeros_like(self);
    auto mask = at::zeros_like(self, self.options().dtype(c10::CppTypeToScalarType<bool>::value));
    return std::tuple<Tensor,Tensor>(std::move(ret), std::move(mask));
  }

  auto gen = get_generator_or_default<CUDAGeneratorImpl>(std::nullopt, cuda::detail::getDefaultCUDAGenerator());
  double p1m = 1. - p;
  return dropout_cuda<bool>(gen, self, p1m);
}

// TODO: _fused_dropout_cuda is to be removed, see PR #63937
std::tuple<Tensor,Tensor>
fused_dropout_cuda(const Tensor& self, double p, std::optional<Generator> gen_){
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  return dropout_cuda<uint8_t>(gen, self, p);
}

template <typename mask_t>
Tensor dropout_backward_cuda(const Tensor& grad, const Tensor& mask, double scale){
   Tensor ret = at::empty_like(grad, grad.suggest_memory_format());
   AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, ret.scalar_type(), "masked_scale", [&] {
      using accscalar_t = acc_type<scalar_t, true>;
      masked_scale_kernel<mask_t, scalar_t>(ret, grad, mask, (accscalar_t)scale);
  });
  return ret;
}

Tensor native_dropout_backward_cuda(const Tensor& grad, const Tensor& mask, double scale){
   TORCH_CHECK(mask.scalar_type() == at::ScalarType::Bool, "Mask should be Bool Scalar Type", mask.scalar_type());
  return dropout_backward_cuda<bool>(grad, mask, scale);
}

// TODO: masked_scale_cuda is to be removed, see PR #63937
Tensor masked_scale_cuda(const Tensor& self, const Tensor& mask, double scale){
  TORCH_CHECK(mask.scalar_type() == at::ScalarType::Byte, "mask should be torch.uint8 dtype");
  return dropout_backward_cuda<uint8_t>(self, mask, scale);
}

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced sibling of masked_scale_kernel, compiled here
// so the sibling and the host above instantiate the one kernel (DECISIONS E36): the tape's
// launch is eager's function object, not a twin. Generated by torchgen from
// cuda/host_trace/ti/siblings.yaml or the op's ufunc_inner_loop; the entry, its proxies and
// its strided views. Outside a trace it runs the same launches in ordinary mode.
#include <ATen/HostTraceSibling_native_dropout_backward.cuh>

// ---- host tracing (ATen/cuda/host_trace): the traced sibling of dropout_cuda and its launcher,
// compiled here so the sibling and the real host above instantiate the one kernel
// (fused_dropout_kernel_vec / fused_dropout_kernel, this file's anonymous namespace; DECISIONS
// E36): the tape's launch is eager's function object, not a twin. The kernel's TensorInfo
// arguments are the one-dimensional proxies the real host's collapseDims leaves (contiguous
// inputs only; anything else declines by name). The philox state travels as an `rng` field:
// capture-time generator state the replay never rewrites. The increment the host hands the
// generator is declared with rng_increment, so the tape carries it as an expression of the
// element count. Outside a trace the entry runs the same launches in ordinary mode, which is how
// the parity test compares it with the real op.
#include <ATen/cuda/host_trace/Field.h>
#include <ATen/cuda/host_trace/Launch.h>
#include <ATen/cuda/host_trace/Philox.h>
#include <ATen/cuda/host_trace/Recorder.h>
#include <ATen/cuda/host_trace/ti/EagerViews.cuh>
#include <ATen/cuda/host_trace/ti/Ops.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstddef>
#include <limits>
#include <mutex>

namespace at::cuda::host_trace::ti {

namespace {

// A one-dimensional TensorInfo: data (a pointer from sym_*_data_ptr),
// sizes[0] = the element count, strides[0] = 1, dims = 1, the shape the real
// host's collapseDims leaves for a contiguous tensor; the other entries stay
// zero (the real host leaves them unwritten).
template <typename Info>
void flat_info(Info& info, const c10::SymInt& data, const c10::SymInt& numel) {
  info.data = data;
  info.sizes[0] = numel;
  info.strides[0] = 1;
  info.set_dims(1);
}

// memory::can_vectorize_up_to on the input pointer, as guards on its
// address; the fresh outputs are allocation roots (256-byte aligned).
template <typename scalar_t>
int traced_vector_size(const Tensor& self, const c10::SymInt& self_ptr, const c10::SymInt& numel) {
  const int optimal = 16 / static_cast<int>(sizeof(scalar_t));
  int vec_size = 1;
  for (int v = optimal; v > 1; v /= 2) {
    if ((self_ptr % c10::SymInt(v * static_cast<int64_t>(sizeof(scalar_t)))).sym_eq(0).guard_bool(__FILE__, __LINE__)) {
      vec_size = v;
      break;
    }
  }
  // no remainders: prefer a smaller vector with none over a larger one with one
  while (vec_size > 1 && !(numel % c10::SymInt(vec_size)).sym_eq(0).guard_bool(__FILE__, __LINE__)) {
    vec_size /= 2;
  }
  return vec_size;
}

} // namespace

std::tuple<Tensor, Tensor> native_dropout_traced(const Tensor& self, double p, std::optional<bool> train) {
  if (train.has_value() && !train.value()) {
    decline("host_trace: native_dropout with train=False is not traced (declined)");
  }
  if (p == 1) {
    decline("host_trace: native_dropout with p == 1 is not traced (declined)");
  }
  if (!at::isFloatingType(self.scalar_type())) {
    decline(c10::str("host_trace: native_dropout on ", self.scalar_type(), " is not traced (declined)"));
  }
  if (!self.is_contiguous()) {
    decline("host_trace: native_dropout on a non-contiguous input is not traced (declined)");
  }
  const c10::SymInt nelem = self.sym_numel();
  if (nelem.sym_eq(0).guard_bool(__FILE__, __LINE__)) {
    decline("host_trace: native_dropout on an empty input is not traced (declined)");
  }
  if (!nelem.sym_lt(c10::SymInt(std::numeric_limits<int>::max())).guard_bool(__FILE__, __LINE__)) {
    decline("host_trace: native_dropout above 2^31 elements (64-bit indexing) is not traced (declined)");
  }
  const double p1m = 1. - p;
  c10::cuda::CUDAGuard device_guard(self.device());
  Tensor mask = at::empty_like(self, self.options().dtype(c10::CppTypeToScalarType<bool>::value));
  Tensor ret = at::empty_like(self);

  const int64_t block_size = 256;
  const auto* props = at::cuda::getCurrentDeviceProperties();
  const unsigned int blocks_per_sm = props->maxThreadsPerMultiProcessor / block_size;
  const c10::SymInt grid_cap(static_cast<int64_t>(props->multiProcessorCount) * blocks_per_sm);
  const c10::SymInt grid_x = ((nelem + (block_size - 1)) / block_size).min(grid_cap);
  const c10::SymInt counter_offset = ((nelem - 1) / (block_size * grid_x * at::native::UNROLL) + 1) * at::native::UNROLL;
  const int64_t increment = rng_increment(counter_offset);

  auto* gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  PhiloxCudaState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(increment);
  }
  const c10::SymInt self_ptr = sym_const_data_ptr(self);
  const c10::SymInt ret_ptr = sym_mutable_data_ptr(ret);
  const c10::SymInt mask_ptr = sym_mutable_data_ptr(mask);
  auto stream = at::cuda::getCurrentCUDAStream();
  Grid grid(grid_x);
  const Block block{c10::SymInt(block_size)};

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "native_dropout_traced", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    using index_type = unsigned int;
    accscalar_t pa = (accscalar_t)(p1m);
    Traced<at::cuda::detail::TensorInfo<const scalar_t, index_type>> a;
    Traced<at::cuda::detail::TensorInfo<scalar_t, index_type>> b;
    Traced<at::cuda::detail::TensorInfo<bool, index_type>> c;
    flat_info(a, self_ptr, nelem);
    flat_info(b, ret_ptr, nelem);
    flat_info(c, mask_ptr, nelem);
    TracedPhilox philox(rng_engine_inputs);
    const int vec_size = traced_vector_size<scalar_t>(self, self_ptr, nelem);
    switch (vec_size) {
      case 8:
        launch(at::native::fused_dropout_kernel_vec<scalar_t, accscalar_t, index_type, 1, 8, bool>, grid, block, 0, stream, a, b, c, nelem, pa, philox);
        break;
      case 4:
        launch(at::native::fused_dropout_kernel_vec<scalar_t, accscalar_t, index_type, 1, 4, bool>, grid, block, 0, stream, a, b, c, nelem, pa, philox);
        break;
      case 2:
        launch(at::native::fused_dropout_kernel_vec<scalar_t, accscalar_t, index_type, 1, 2, bool>, grid, block, 0, stream, a, b, c, nelem, pa, philox);
        break;
      default:
        launch(at::native::fused_dropout_kernel<scalar_t, accscalar_t, index_type, 1, 1, bool>, grid, block, 0, stream, a, b, c, nelem, pa, philox);
        break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return std::tuple<Tensor, Tensor>(std::move(ret), std::move(mask));
}

} // namespace at::cuda::host_trace::ti
