#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/IndexKernel.h>
#include <ATen/native/IndexKernel.h>

#include <array>
#include <type_traits>
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/cub.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/native/quantized/IndexKernel.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/IndexKernelUtils.h>
#if !defined(USE_ROCM)
#include <ATen/cuda/host_trace/SymImage.h>
#endif

#include <c10/core/Scalar.h>

namespace at::native {

static constexpr int launch_bound2 = 4;

static constexpr int launch_size_nd = 128;

template<int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void index_elementwise_kernel(const int64_t N, const func_t f) {
  const auto tid = threadIdx.x;
  const auto nv = nt * vt;
  auto idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template<int nt, int vt, typename func_t>
static void launch_kernel(const int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  const dim3 block(nt);
  const dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  const auto stream = at::cuda::getCurrentCUDAStream();
  index_elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// gpu_index_kernel's elementwise functor: a named type, so that a CUDA graph
// host trace knows where each of its members lies
template <typename func_t>
struct IndexFunctor {
  OffsetCalculator<3> offset_calc;
  char* out_ptr;
  const char* in_ptr;
  std::array<int64_t, MAX_DIMS> sizes;
  std::array<int64_t, MAX_DIMS> strides;
  std::array<char*, MAX_DIMS> index_ptrs;
  int num_indices;
  func_t f;

  __device__ void operator()(int idx) const {
    const auto offsets = offset_calc.get(idx);
    char* const out_data = out_ptr + offsets[0];
    const char* const in_data = in_ptr + offsets[1];

    int64_t offset = 0;
    #pragma unroll
    for (int i = 0; i < num_indices; i++) {
      int64_t index = *reinterpret_cast<int64_t*>(index_ptrs[i] + offsets[2]);
      CUDA_KERNEL_ASSERT(-sizes[i] <= index && index < sizes[i] && "index out of bounds");
      if (index < 0) {
        index += sizes[i];
      }
      offset += index * strides[i];
    }

    f(out_data, in_data, offset);
  }
};

#if !defined(USE_ROCM)
// launch_kernel<nt, vt>(n, functor) of a CUDA graph host trace: the functor as its image
template <int nt, int vt, typename functor_t>
void sym_launch_kernel(const c10::SymInt& n, at::cuda::host_trace::ParamImage op) {
  namespace ht = at::cuda::host_trace;
  TORCH_INTERNAL_ASSERT(n <= std::numeric_limits<int32_t>::max());
  ht::ParamImage count(sizeof(int64_t), alignof(int64_t));
  count.set(0, sizeof(int64_t), n);
  std::vector<ht::ParamImage> params;
  params.push_back(std::move(count));
  params.push_back(std::move(op));
  constexpr int64_t work = nt * vt;
  ht::record_images(reinterpret_cast<const void*>(&index_elementwise_kernel<nt, vt, functor_t>), std::move(params), (n + work - 1) / work, nt, 0);
}

// gpu_index_kernel over a symbolic TensorIterator (a CUDA graph host trace):
// the same choices over its symbolic sizes, strides and addresses, the launch
// recorded. index_size and index_stride: the iterator's symbolic index
// geometry (AdvancedIndex's, set by the meta), else the constants given
template <typename func_t>
void sym_gpu_index_kernel(const TensorIteratorBase& iter, const c10::SymIntArrayRef index_size, const c10::SymIntArrayRef index_stride, const func_t& f, const bool is_gather_like) {
  namespace ht = at::cuda::host_trace;
  const auto num_indices = index_size.size();
  if (iter.sym_numel() == 0) {
    return;
  }
  // eager splits the iterator
  TORCH_CHECK_NOT_IMPLEMENTED(iter.can_use_32bit_indexing(), "host_trace: an index kernel beyond 32-bit indexing");
  ht::Recorder& rec = ht::recorder_for("an index kernel");
  const c10::SymInt out_ptr = ht::operand_address(iter, 0);
  const c10::SymInt in_ptr = ht::operand_address(iter, 1);
  const auto shape = iter.sym_shape();
  if (is_gather_like && num_indices == 1) {
    // fast_gather_kernel_eligible<16>, over SymInt (each test a guard)
    constexpr int64_t alignment = 16;
    const int64_t element_size = iter.element_size(0);
    const auto s0 = iter.sym_strides(0), s1 = iter.sym_strides(1), s2 = iter.sym_strides(2);
    const bool eligible = iter.ndim() == 2 && s2[0] == 0 && s2[1] == iter.element_size(2) && s0[0] == element_size && s1[0] == element_size &&
        s1[1] == 0 && out_ptr % alignment == 0 && in_ptr % alignment == 0 && shape[0] * element_size % alignment == 0 &&
        index_stride[0] % alignment == 0 && s0[1] % alignment == 0;
    if (eligible) {
      at::native::vectorized_gather_kernel_record<alignment, int64_t>(rec, out_ptr, in_ptr, ht::operand_address(iter, 2), shape[1], shape[0] * element_size,
                                                                      index_size[0], index_stride[0], s0[1], /*allow_neg_indices=*/true);
      return;
    }
  }
  using functor_t = IndexFunctor<func_t>;
  ht::ParamImage op(sizeof(functor_t), alignof(functor_t));
  alignas(functor_t) unsigned char raw[sizeof(functor_t)];
  const functor_t& r = *reinterpret_cast<const functor_t*>(raw);
  auto at = [&r](const auto& m) { return static_cast<size_t>(reinterpret_cast<const char*>(&m) - reinterpret_cast<const char*>(&r)); };
  ht::put_offset_calculator(op, at(r.offset_calc), 3, ht::sym_offsets(iter, 0, 3, /*element_strides=*/false));
  op.set(at(r.out_ptr), sizeof(char*), out_ptr, true);
  op.set(at(r.in_ptr), sizeof(char*), in_ptr, true);
  for (const auto i : c10::irange(num_indices)) {
    op.set(at(r.sizes) + i * sizeof(int64_t), sizeof(int64_t), index_size[i]);
    op.set(at(r.strides) + i * sizeof(int64_t), sizeof(int64_t), index_stride[i]);
    op.set(at(r.index_ptrs) + i * sizeof(char*), sizeof(char*), ht::operand_address(iter, static_cast<int>(i) + 2), true);
  }
  const int n_indices = static_cast<int>(num_indices);
  op.put(at(r.num_indices), &n_indices, sizeof(int));
  op.put(at(r.f), &f, sizeof(func_t));
  sym_launch_kernel<launch_size_nd, launch_bound2, functor_t>(iter.sym_numel(), std::move(op));
}
#endif

template <typename func_t>
void gpu_index_kernel(TensorIteratorBase& iter, const IntArrayRef index_size, const IntArrayRef index_stride, const func_t& f, const bool is_gather_like) {
#if !defined(USE_ROCM)
  if (C10_UNLIKELY(iter.is_symbolic())) {
    if (iter.has_sym_index_geometry()) {
      return sym_gpu_index_kernel(iter, iter.sym_index_sizes(), iter.sym_index_strides(), f, is_gather_like);
    }
    TORCH_CHECK_NOT_IMPLEMENTED(
        static_cast<int64_t>(index_size.size()) == iter.ntensors() - 2, "host_trace: an index kernel of a symbolic iterator without its index geometry");
    return sym_gpu_index_kernel(iter, c10::fromIntArrayRefSlow(index_size), c10::fromIntArrayRefSlow(index_stride), f, is_gather_like);
  }
#endif
  const auto num_indices = index_size.size();
  AT_ASSERT(num_indices == index_stride.size());
  AT_ASSERT(static_cast<int64_t>(num_indices) == iter.ntensors() - 2);

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_index_kernel(sub_iter, index_size, index_stride, f, is_gather_like);
    }
    return;
  }


  char* const out_ptr = static_cast<char*>(iter.data_ptr(0));
  char* const in_ptr = static_cast<char*>(iter.data_ptr(1));
  if (is_gather_like && num_indices==1) {
      const size_t element_size = iter.element_size(0);
      constexpr size_t alignment = 16;
      if (at::native::fast_gather_kernel_eligible<alignment>(iter, out_ptr, in_ptr, index_stride[0], element_size)) {
        auto slice_size = iter.shape()[0] * element_size;
        auto num_ind = iter.shape()[1];
        auto ind_dim_size = index_size[0];
        auto inp_stride_bytes = index_stride[0];
        auto out_stride_bytes = iter.strides(0)[1];
        at::native::vectorized_gather_kernel_launch<alignment, int64_t>(out_ptr, in_ptr, (int64_t*)iter.data_ptr(2), num_ind,
        slice_size, ind_dim_size, inp_stride_bytes, out_stride_bytes, /*allow_neg_indices*/true);
        return;
    }
  }

  IndexFunctor<func_t> functor{make_offset_calculator<3>(iter), out_ptr, in_ptr, {}, {}, {}, static_cast<int>(num_indices), f};
  for (unsigned i = 0; i < num_indices; i++) {
    functor.sizes[i] = index_size[i];
    functor.strides[i] = index_stride[i];
    functor.index_ptrs[i] = (char*)iter.data_ptr(i + 2);
  }
  launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), functor);
}

// The kernels are templated on an opaque, self-aligned type of the correct
// size to avoid redundant kernels for different types of the same size.
template <int N> struct alignas(N) OpaqueType { char data[N]; };

template <typename scalar_t>
void index_fill_kernel_impl(
  TensorIterator& iter,
  const int64_t dim,
  const int64_t self_dim_size,
  const int64_t self_dim_stride,
  const scalar_t fill_val) {
  if (0 == iter.numel()) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      index_fill_kernel_impl(sub_iter, dim, self_dim_size, self_dim_stride, fill_val);
    }
    return;
  }

  char* const __restrict__ self_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* const __restrict__ idx_ptr = reinterpret_cast<char*>(iter.data_ptr(1));

  const auto offset_calc = make_offset_calculator<2>(iter);

  const auto loop = [=]C10_DEVICE(int i) {
    const auto offsets = offset_calc.get(i);

    auto* __restrict__ self_data = reinterpret_cast<scalar_t*>(self_ptr + offsets[0]);
    auto idx = *reinterpret_cast<int64_t*>(idx_ptr + offsets[1]);
    CUDA_KERNEL_ASSERT(idx >= -self_dim_size && idx < self_dim_size && "index out of bounds");
    if (idx < 0) {
      idx += self_dim_size;
    }

    self_data[idx * self_dim_stride] = fill_val;
  };
  launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), loop);
}

// index_copy_kernel_impl's elementwise functor: a named type, so that a CUDA
// graph host trace knows where each of its members lies
template <typename scalar_t>
struct IndexCopyFunctor {
  OffsetCalculator<3> offset_calc;
  char* self_ptr;
  char* idx_ptr;
  char* source_ptr;
  int64_t self_dim_size;
  int64_t self_dim_stride;

  __device__ void operator()(int i) const {
    const auto offsets = offset_calc.get(i);

    auto* const __restrict__ self_data = reinterpret_cast<scalar_t*>(self_ptr + offsets[0]);
    auto idx = *reinterpret_cast<int64_t*>(idx_ptr + offsets[1]);
    const auto* const __restrict__ source_data = reinterpret_cast<scalar_t*>(source_ptr + offsets[2]);
    CUDA_KERNEL_ASSERT(idx >= 0 && idx < self_dim_size && "index_copy_(): index out of bounds");

    self_data[idx * self_dim_stride] = *source_data;
  }
};

template <typename scalar_t>
void index_copy_kernel_impl(
  TensorIterator& iter,
  const int64_t dim,
  const int64_t self_dim_size,
  const int64_t self_dim_stride) {
#if !defined(USE_ROCM)
  if (C10_UNLIKELY(iter.is_symbolic())) {
    // a CUDA graph host trace: the dim's size and stride are the iterator's
    // symbolic index geometry (set by index_copy_out)
    namespace ht = at::cuda::host_trace;
    if (iter.sym_numel() == 0) {
      return;
    }
    TORCH_CHECK_NOT_IMPLEMENTED(iter.can_use_32bit_indexing(), "host_trace: an index_copy_ kernel beyond 32-bit indexing");
    using functor_t = IndexCopyFunctor<scalar_t>;
    ht::ParamImage op(sizeof(functor_t), alignof(functor_t));
    alignas(functor_t) unsigned char raw[sizeof(functor_t)];
    const functor_t& r = *reinterpret_cast<const functor_t*>(raw);
    auto at = [&r](const auto& m) { return static_cast<size_t>(reinterpret_cast<const char*>(&m) - reinterpret_cast<const char*>(&r)); };
    ht::put_offset_calculator(op, at(r.offset_calc), 3, ht::sym_offsets(iter, 0, 3, /*element_strides=*/false));
    op.set(at(r.self_ptr), sizeof(char*), ht::operand_address(iter, 0), true);
    op.set(at(r.idx_ptr), sizeof(char*), ht::operand_address(iter, 1), true);
    op.set(at(r.source_ptr), sizeof(char*), ht::operand_address(iter, 2), true);
    op.set(at(r.self_dim_size), sizeof(int64_t), iter.sym_index_sizes()[0]);
    op.set(at(r.self_dim_stride), sizeof(int64_t), iter.sym_index_strides()[0]);
    return sym_launch_kernel<launch_size_nd, launch_bound2, functor_t>(iter.sym_numel(), std::move(op));
  }
#endif
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      index_copy_kernel_impl<scalar_t>(sub_iter, dim, self_dim_size, self_dim_stride);
    }
    return;
  }

  IndexCopyFunctor<scalar_t> functor{
      make_offset_calculator<3>(iter),
      reinterpret_cast<char*>(iter.data_ptr(0)),
      reinterpret_cast<char*>(iter.data_ptr(1)),
      reinterpret_cast<char*>(iter.data_ptr(2)),
      self_dim_size,
      self_dim_stride};
  launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), functor);
}

template <typename scalar_t>
void index_kernel_impl(TensorIteratorBase& iter, const IntArrayRef index_size, const IntArrayRef index_stride) {
  gpu_index_kernel(iter, index_size, index_stride, []C10_DEVICE(char* const out_data, const char* const in_data, const int64_t offset) {
    *reinterpret_cast<scalar_t*>(out_data) = *reinterpret_cast<const scalar_t*>(in_data + offset);
  }, true);
}

template <typename scalar_t>
void index_put_kernel_impl(TensorIterator& iter, const IntArrayRef index_size, const IntArrayRef index_stride) {
  gpu_index_kernel(iter, index_size, index_stride, []C10_DEVICE(char* const out_data, const char* const in_data, const int64_t offset) {
    *reinterpret_cast<scalar_t*>(out_data + offset) = *reinterpret_cast<const scalar_t*>(in_data);
  }, false);
}

static void index_kernel(
    TensorIteratorBase& iter,
    const IntArrayRef index_size,
    const IntArrayRef index_stride) {
  AT_DISPATCH_V2(
      iter.dtype(),
      "index_cuda",
      AT_WRAP([&] {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        index_kernel_impl<dtype>(iter, index_size, index_stride);
      }),
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
      AT_EXPAND(AT_FLOAT8_TYPES),
      kComplexHalf,
      kBComplex32,
      kHalf,
      kBool,
      kBFloat16);
}

static void index_fill_kernel(
  TensorIterator& iter,
  const int64_t dim,
  const int64_t self_dim_size,
  const int64_t self_dim_stride,
  const Scalar& source) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND5(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, kComplexHalf,
    kBComplex32,
    iter.dtype(), "index_fill_cuda", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    const auto fill_val = source.to<scalar_t>();
    const auto fill_val_opaque = *reinterpret_cast<const dtype*>(&fill_val);
    index_fill_kernel_impl<dtype>(iter, dim, self_dim_size, self_dim_stride, fill_val_opaque);
  });
}

static void index_copy_kernel(
  TensorIterator& iter,
  const int64_t dim,
  const int64_t self_dim_size,
  const int64_t self_dim_stride) {
  // See note [Writing Nondeterministic Operations]
  // Nondeterministic when index contains duplicate entries
  // this kernel will not be called when torch.use_deterministic_algorithms(True)
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND5(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, kComplexHalf,
    kBComplex32,
    iter.dtype(), "index_copy_cuda", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    index_copy_kernel_impl<dtype>(iter, dim, self_dim_size, self_dim_stride);
  });
}


static void index_put_kernel(TensorIterator& iter, const IntArrayRef index_size, const IntArrayRef index_stride, const bool accumulate) {
  TORCH_CHECK(!accumulate, "index_put does not support accumulate=true");
  AT_DISPATCH_V2(
    iter.dtype(),
    "index_put",
    AT_WRAP([&] {
      using dtype = OpaqueType<sizeof(scalar_t)>;
      index_put_kernel_impl<dtype>(iter, index_size, index_stride);
    }),
    AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
    AT_EXPAND(AT_FLOAT8_TYPES),
    kComplexHalf,
    kBComplex32,
    kHalf,
    kBool,
    kBFloat16);
}

void index_put_kernel_quantized_cuda(TensorIterator& iter, const IntArrayRef index_size, const IntArrayRef index_stride, const bool accumulate, const double scale, const int zero_point) {
  TORCH_CHECK(!accumulate, "index_put does not support accumulate=true");
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(iter.dtype(), "index_put", [&] {
    constexpr int64_t qmin = std::numeric_limits<typename scalar_t::underlying>::min();
    constexpr int64_t qmax = std::numeric_limits<typename scalar_t::underlying>::max();
    const float inv_scale = 1.0f / static_cast<float>(scale);

    gpu_index_kernel(iter, index_size, index_stride, [inv_scale, zero_point, qmin, qmax]C10_DEVICE(char* const out_data, const char* const in_data, const int64_t offset) {
      int64_t qvalue = static_cast<int64_t>(zero_point + nearbyintf(*(float*)in_data * inv_scale));
      // See https://github.com/pytorch/pytorch/issues/127666
      // and https://github.com/pytorch/pytorch/issues/128253.
      // hip-clang std::clamp __glibcxx_assert_fail host function when building on Fedora40/gcc14.
      // The following replaces std::clamp(qvalue, qmin, qmax) and is a viable solution for
      // both CUDA and ROCm since std::clamp and this replacement generates the same PTX.
      // Using #ifdef USE_ROCM to differentiate caused Windows build failures.
      // The replacement should generate the same PTX as std::clamp. See https://godbolt.org/z/Wde9KW3v4
      qvalue = (qvalue < qmin) ? qmin : (qmax < qvalue) ? qmax : qvalue;
      *(scalar_t*)(out_data + offset) = static_cast<scalar_t>(qvalue);
    }, false);
  });
}

template <typename scalar_t, typename index_t, typename func_t>
void cuda_take_put_kernel(
  TensorIterator& iter,
  const TensorBase& indexed,
  const func_t& f) {
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      cuda_take_put_kernel<scalar_t, index_t>(sub_iter, indexed, f);
    }
    return;
  }

  const auto numel = indexed.numel();
  const bool is_contiguous = indexed.is_contiguous();

  char* const __restrict__ iterated_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* const __restrict__ idx_ptr = reinterpret_cast<char*>(iter.data_ptr(1));

  const auto offset_calc = make_offset_calculator<2>(iter);
  using uindex_t = std::make_unsigned_t<index_t>;

  // OffsetCalculator needs the sizes and strides reversed
  const auto indexed_sizes = std::vector<int64_t>(indexed.sizes().rbegin(), indexed.sizes().rend());
  const auto indexed_strides = std::vector<int64_t>(indexed.strides().rbegin(), indexed.strides().rend());
  const auto* indexed_strides_data = indexed_strides.data();
  const auto offset_indexed = OffsetCalculator<1, uindex_t>(indexed.dim(),
                                                            indexed_sizes.data(),
                                                            &indexed_strides_data);

  const auto loop = [=]C10_DEVICE(int i) {
    const auto offsets = offset_calc.get(i);

    auto& iterated = *reinterpret_cast<scalar_t*>(iterated_ptr + offsets[0]);
    const auto idx = *reinterpret_cast<int64_t*>(idx_ptr + offsets[1]);
    CUDA_KERNEL_ASSERT(idx < numel && idx >= -numel && "cuda_take_put_kernel() index out of bounds");
    index_t offset = static_cast<index_t>(idx);
    if (offset < 0) {
      offset += numel;
    }
    if (!is_contiguous) {
      offset = offset_indexed.get(offset)[0];
    }

    f(iterated, offset);
  };
  launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), loop);
}

void put_kernel(TensorIterator& iter, const TensorBase& output, const bool accumulate) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, iter.dtype(), "put_cuda", [&] {
    // Cannot use `OpaqueType`, as we need the actual type for `fastSpecializedgpuAtomicAdd`
    AT_DISPATCH_INDEX_TYPES(cuda::detail::canUse32BitIndexMath(output) ? ScalarType::Int : ScalarType::Long,
        "put_cuda_index", [&] {
           auto* __restrict__ indexed_ptr = output.template data_ptr<scalar_t>();
           if (accumulate) {
             index_t numel = output.numel();
             cuda_take_put_kernel<scalar_t, index_t>(iter, output,
                 [numel, indexed_ptr] __device__(scalar_t& iterated, const index_t offset) {
                   fastSpecializedAtomicAdd(indexed_ptr, offset, numel, iterated);
                 });
           }
           else {
             cuda_take_put_kernel<scalar_t, index_t>(iter, output,
                 [indexed_ptr] __device__(scalar_t& iterated, const index_t offset) {
                   indexed_ptr[offset] = iterated;
                 });
           }
    });
  });
}

void take_kernel(
  TensorIterator& iter,
  const TensorBase& input) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, iter.dtype(), "take_cuda", [&] {
    // Cannot use `OpaqueType`, as Tensor::data_ptr<OpaqueType<N>> is not implemented
    AT_DISPATCH_INDEX_TYPES(cuda::detail::canUse32BitIndexMath(input) ? ScalarType::Int : ScalarType::Long,
      "take_cuda_index", [&] {
         const auto* __restrict__ indexed_ptr = input.template const_data_ptr<scalar_t>();
         cuda_take_put_kernel<scalar_t, index_t>(iter, input,
            [indexed_ptr] __device__(scalar_t& iterated, const index_t offset) {
               iterated = indexed_ptr[offset];
             });
     });
  });
}

namespace {

__global__ void masked_scatter_size_check(
  const int64_t* const mask_exclusive_sum,
  const bool* const mask,
  const int64_t srcSize) {
  // Convert exclusive sum to inclusive sum
  const auto totalElements = *mask_exclusive_sum + *mask;
  CUDA_KERNEL_ASSERT(totalElements <= srcSize);
}

} // anonymous namespace

template <typename scalar_t>
struct MaskedScatterFunctor {
  const scalar_t* source_ptr;
  __device__ scalar_t operator()(const scalar_t a, const bool mask, const int64_t maskPrefixSum) const {
    if (mask) {
      return source_ptr[maskPrefixSum];
    }
    return a;
  }
  auto host_trace_fields() const {
    return std::tie(source_ptr);
  }
};

void launch_masked_scatter_kernel(
    const TensorBase &self, const TensorBase &mask,
    const TensorBase &maskPrefixSum, const TensorBase &source) {
  const auto srcSize = source.numel();
  const auto mask_cont = mask.contiguous();
  const auto mask_numel = mask.numel();

  // Use a prefix sum to determine the output locations of the masked elements
  auto maskPrefixSum_data = maskPrefixSum.mutable_data_ptr<int64_t>();
  auto mask_data = mask_cont.const_data_ptr<bool>();

  at::cuda::cub::mask_exclusive_sum(
      mask_data, maskPrefixSum_data, mask_numel);

  // Asynchronously check that the number of `1` elements present in the mask
  // must be <= the number of elements available in `src`.
  masked_scatter_size_check<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      &maskPrefixSum_data[mask_numel - 1], &mask_data[mask_numel - 1], srcSize);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // We are getting elements from `src` based on an offset from
  // `maskPrefixSum`, so that should be made contiguous too
  auto source_contig = source.contiguous();

  auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(self)
      .add_input(self)
      .add_const_input(mask_cont)
      .add_input(maskPrefixSum)
      .build();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Bool,
      ScalarType::BFloat16,
      ScalarType::Half,
      self.scalar_type(),
      "masked_scatter_",
      [&]() {
        auto source_ptr = source_contig.const_data_ptr<scalar_t>();
        gpu_kernel(iter, MaskedScatterFunctor<scalar_t>{source_ptr});
        AT_CUDA_CHECK(cudaGetLastError());
      });
}

// flip_kernel_impl's elementwise functor, a named type (IndexFunctor's note)
template <typename scalar_t>
struct FlipFunctor {
  OffsetCalculator<2, uint32_t, /*signed_strides=*/true> offset_calc;
  char* out_ptr;
  const char* in_ptr;

  __device__ void operator()(const int i) const {
    const auto offsets = offset_calc.get(i);
    // offsets can be negative here, but it's fine
    scalar_t* const __restrict__ out_data = reinterpret_cast<scalar_t*>(out_ptr + offsets[0]);
    const scalar_t* const __restrict__ in_data = reinterpret_cast<const scalar_t*>(in_ptr + offsets[1]);
    *out_data = *in_data;
  }
};

template <typename scalar_t>
void flip_kernel_impl(TensorIterator& iter) {
#if !defined(USE_ROCM)
  if (C10_UNLIKELY(iter.is_symbolic())) {
    // a CUDA graph host trace: the launch below over the iterator's symbolic geometry, recorded
    namespace ht = at::cuda::host_trace;
    TORCH_CHECK_NOT_IMPLEMENTED(iter.can_use_32bit_indexing(), "host_trace: a flip beyond 32-bit indexing");
    using functor_t = FlipFunctor<scalar_t>;
    // signed and unsigned strides: one layout, the signed packed as such
    static_assert(sizeof(OffsetCalculator<2, uint32_t, true>) == sizeof(OffsetCalculator<2>));
    ht::ParamImage op(sizeof(functor_t), alignof(functor_t));
    alignas(functor_t) unsigned char raw[sizeof(functor_t)];
    const functor_t& r = *reinterpret_cast<const functor_t*>(raw);
    auto at = [&r](const auto& m) { return static_cast<size_t>(reinterpret_cast<const char*>(&m) - reinterpret_cast<const char*>(&r)); };
    ht::put_offset_calculator(op, at(r.offset_calc), 2, ht::sym_offsets(iter, 0, 2, /*element_strides=*/false));
    op.set(at(r.out_ptr), sizeof(char*), ht::operand_address(iter, 0), true);
    op.set(at(r.in_ptr), sizeof(char*), ht::operand_address(iter, 1), true);
    const c10::SymInt n = iter.sym_numel();
    TORCH_INTERNAL_ASSERT(n <= std::numeric_limits<int32_t>::max());
    ht::ParamImage count(sizeof(int64_t), alignof(int64_t));
    count.set(0, sizeof(int64_t), n);
    std::vector<ht::ParamImage> params;
    params.push_back(std::move(count));
    params.push_back(std::move(op));
    constexpr int64_t work = launch_size_nd * launch_bound2;
    ht::record_images(reinterpret_cast<const void*>(&index_elementwise_kernel<launch_size_nd, launch_bound2, functor_t>), std::move(params),
                      (n + work - 1) / work, launch_size_nd, 0);
    return;
  }
#endif
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      flip_kernel_impl<scalar_t>(sub_iter);
    }
    return;
  }

  char* const __restrict__ out_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  const char* const __restrict__ in_ptr = reinterpret_cast<const char*>(iter.data_ptr(1));

  const FlipFunctor<scalar_t> functor{make_offset_calculator<2, /*signed_strides=*/true>(iter), out_ptr, in_ptr};
  launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), functor);
}

void flip_kernel(TensorIterator& iter, const bool quantized) {
  if (quantized) {
    AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(iter.dtype(), "flip_quantized_cuda",
    [&] {
      using dtype = OpaqueType<sizeof(scalar_t)>;
      flip_kernel_impl<dtype>(iter);
    });
  } else {
    AT_DISPATCH_V2(
      iter.dtype(),
      "flip_cuda",
      AT_WRAP([&] {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        flip_kernel_impl<dtype>(iter);
      }),
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
      AT_EXPAND(AT_FLOAT8_TYPES),
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
      kComplexHalf,
      kBComplex32,
      kHalf,
      kBool,
      kBFloat16);
  }
}


REGISTER_DISPATCH(index_stub, &index_kernel)
REGISTER_DISPATCH(index_fill_stub, &index_fill_kernel)
REGISTER_DISPATCH(index_copy_stub, &index_copy_kernel)
REGISTER_DISPATCH(index_put_stub, &index_put_kernel)
REGISTER_DISPATCH(put_stub, &put_kernel)
REGISTER_DISPATCH(take_stub, &take_kernel)
REGISTER_DISPATCH(flip_stub, &flip_kernel)

REGISTER_CUDA_DISPATCH(index_put_kernel_quantized_stub, &index_put_kernel_quantized_cuda)

} // namespace at::native
