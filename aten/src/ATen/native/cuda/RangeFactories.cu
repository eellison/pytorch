#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/RangeUtils.h>
#include <cmath>
#include <limits>
#if defined(USE_ROCM)
#include <algorithm>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/linspace_native.h>
#include <ATen/ops/logspace_native.h>
#include <ATen/ops/range_native.h>
#endif

#define GPU_LAMBDA __device__ __host__

namespace {

#if defined(USE_ROCM)
constexpr int num_threads() {
  return 128;
}
#else
constexpr int num_threads() {
  return C10_WARP_SIZE * 2;
}
#endif
constexpr int thread_work_size = 1;
constexpr int block_work_size = thread_work_size * num_threads();

template<typename index_t, typename func_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void elementwise_kernel_with_index(index_t N, func_t f, typename function_traits<func_t>::result_type *data) {
  #pragma unroll
  for (int i = 0; i < thread_work_size; i++) {
    index_t idx = block_work_size * blockIdx.x + num_threads() * i + threadIdx.x;
    if (idx < N) {
      data[idx] = f(idx);
    }
  }
}

#if defined(USE_ROCM)
// HIP does not support launches with gridDim.x * blockDim.x >= 2^32:
// depending on the ROCm version the launch returns
// hipErrorInvalidConfiguration or is accepted silently with the kernel
// never executing, leaving zero-initialized output. A grid-stride kernel
// with a fixed grid sized to device occupancy avoids the limit.
template<typename index_t, typename func_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void elementwise_kernel_with_index_grid_stride(
    index_t N, func_t f,
    typename function_traits<func_t>::result_type *data) {
  index_t idx = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const index_t stride = static_cast<index_t>(gridDim.x) * blockDim.x;
  for (; idx < N; idx += stride) {
    data[idx] = f(idx);
  }
}
#endif

template<typename func_t>
void gpu_kernel_with_index(at::Tensor &output, func_t f) {
  int64_t N = output.numel();
  if (N == 0) {
    return;
  }
#if defined(USE_ROCM)
  constexpr int blocks_per_sm = 4;
  const int sm_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t orig_grid = (N + block_work_size - 1) / block_work_size;
  int64_t grid = std::min<int64_t>(
      orig_grid, static_cast<int64_t>(sm_count) * blocks_per_sm);
  grid = std::max<int64_t>(grid, 1);
  auto stream = at::cuda::getCurrentCUDAStream();
  using scalar_t = typename function_traits<func_t>::result_type;
  if (N <= std::numeric_limits<int>::max()) {
    elementwise_kernel_with_index_grid_stride<int><<<grid, num_threads(), 0, stream>>>(N, f, output.mutable_data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    elementwise_kernel_with_index_grid_stride<int64_t><<<grid, num_threads(), 0, stream>>>(N, f, output.mutable_data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
#else
  int64_t grid = (N + block_work_size - 1) / block_work_size;
  auto stream = at::cuda::getCurrentCUDAStream();
  using scalar_t = typename function_traits<func_t>::result_type;
  if (N <= std::numeric_limits<int>::max()) {
    elementwise_kernel_with_index<int><<<grid, num_threads(), 0, stream>>>(N, f, output.mutable_data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    elementwise_kernel_with_index<int64_t><<<grid, num_threads(), 0, stream>>>(N, f, output.mutable_data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
#endif
}

}  // namespace

namespace at::native {

// range_cuda_out's and arange_cuda_out's lambda as a named functor (the members
// in the closure's order): the traced sibling of arange at the end of this file
// launches it too, so the tape holds eager's own kernel (DECISIONS E36)
template <typename scalar_t, typename accscalar_t>
struct ArangeFunctor {
  accscalar_t xstart;
  accscalar_t xstep;
  GPU_LAMBDA scalar_t operator()(int64_t ind) const {
    accscalar_t inc = xstep * static_cast<accscalar_t>(ind);
    accscalar_t val = xstart + inc;
    return static_cast<scalar_t>(val);
  }
};

Tensor& linspace_cuda_out(const Scalar& start, const Scalar& end, int64_t steps, Tensor& result) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }
  bool is_contiguous = result.is_contiguous();
  Tensor r = !is_contiguous ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : result;

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    r.fill_(start);
  } else if (isIntegralType(r.scalar_type(), /*includeBool=*/false)) {
    AT_DISPATCH_INTEGRAL_TYPES(r.scalar_type(), "linspace_cuda", [&]() {
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      // Cast `end` and `start` to `float`, since range can be larger than scalar_t for integral types
      float step = (static_cast<float>(scalar_end) - static_cast<float>(scalar_start)) / (steps - 1);
      const int64_t halfway = steps / 2;
      gpu_kernel_with_index(r, [scalar_start, scalar_end, steps, step, halfway]GPU_LAMBDA(int64_t ind) -> scalar_t {
        if (ind < halfway) {
          return scalar_start + (step * ind);
        }

        return scalar_end - step * (steps - ind - 1);
      });
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, r.scalar_type(), "linspace_cuda", [&]() {
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      scalar_t step = (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
      const int64_t halfway = steps / 2;
      gpu_kernel_with_index(r, [scalar_start, scalar_end, steps, step, halfway]GPU_LAMBDA(int64_t ind) -> scalar_t {
        if (ind < halfway) {
          return scalar_start + (step * ind);
        }

        return scalar_end - step * (steps - ind - 1);
      });
    });
  }

  if (!is_contiguous) {
    result.copy_(r);
  }

  return result;
}

Tensor& logspace_cuda_out(const Scalar& start, const Scalar& end, int64_t steps, double base, Tensor& result) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }
  bool is_contiguous = result.is_contiguous();
  Tensor r = !is_contiguous ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : result;

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    if (isComplexType(r.scalar_type())){
      r.fill_(std::pow(base, start.to<c10::complex<double>>()));
    } else {
      r.fill_(std::pow(base, start.to<double>()));
    }
  } else if (isIntegralType(r.scalar_type(), /*includeBool=*/false)) {
    AT_DISPATCH_INTEGRAL_TYPES(r.scalar_type(), "logspace_cuda", [&]() {
      float scalar_base = static_cast<float>(base); // Use float to avoid promotion to double
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      float step = static_cast<float>(scalar_end - scalar_start) / (steps - 1);
      const int64_t halfway = steps / 2;
      gpu_kernel_with_index(r, [scalar_start, scalar_end, scalar_base, steps, step, halfway]GPU_LAMBDA(int64_t ind) -> scalar_t {
        if (ind < halfway) {
          return std::pow(scalar_base, scalar_start + step * ind);
        }
        return std::pow(scalar_base, scalar_end - step * (steps - ind - 1));
      });
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, r.scalar_type(), "logspace_cuda", [&]() {
      scalar_t scalar_base = static_cast<scalar_t>(base);
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      scalar_t step = (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
      const int64_t halfway = steps / 2;
      gpu_kernel_with_index(r, [scalar_start, scalar_end, scalar_base, steps, step, halfway]GPU_LAMBDA(int64_t ind) -> scalar_t {
        if (ind < halfway) {
          return std::pow(scalar_base, scalar_start + step * ind);
        }
        return std::pow(scalar_base, scalar_end - step * (steps - ind - 1));
      });
    });
  }

  if (!is_contiguous) {
    result.copy_(r);
  }

  return result;
}

Tensor& range_cuda_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, result.scalar_type(), "range_cuda", [&]() {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto xstart = start.to<accscalar_t>();
    auto xend = end.to<accscalar_t>();
    auto xstep = step.to<accscalar_t>();

    arange_check_bounds(start, end, step);

    int64_t size = static_cast<int64_t>(((xend - xstart) / xstep) + 1);

    if (result.numel() != size) {
      result.resize_({size});
    }
    bool is_contiguous = result.is_contiguous();
    Tensor r = !is_contiguous ?  at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : result;

    gpu_kernel_with_index(r, ArangeFunctor<scalar_t, accscalar_t>{xstart, xstep});

    if(!is_contiguous) {
      result.copy_(r);
    }

  });

  return result;
}

Tensor& arange_cuda_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result) {
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, result.scalar_type(), "arange_cuda", [&]() {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto xstart = start.to<accscalar_t>();
    auto xstep = step.to<accscalar_t>();

    int64_t size = compute_arange_size<scalar_t>(start, end, step);
    int64_t numel = result.numel();

    if (numel != size) {
      if(numel > 0){
        TORCH_WARN("The number of elements in the out tensor of shape ", result.sizes(),
                    " is ", numel, " which does not match the computed number of elements ", size,
                    ". Note that this may occur as a result of rounding error. "
                    "The out tensor will be resized to a tensor of shape (", size, ",).");
      }
      result.resize_({size});
    }
    bool is_contiguous = result.is_contiguous();
    Tensor r = !is_contiguous ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : result;

    gpu_kernel_with_index(r, ArangeFunctor<scalar_t, accscalar_t>{xstart, xstep});

    if(!is_contiguous) {
      result.copy_(r);
    }
  });

  return result;
}

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced sibling of arange_cuda_out, compiled here
// so the sibling and the real host above instantiate the one kernel
// (elementwise_kernel_with_index over ArangeFunctor; DECISIONS E36): the tape's launch is
// eager's function object, not a twin. Start and step are fields of the functor's proxy (an int
// field for an integral accumulate type, a float field otherwise), so a bound that is a value of
// the tape (a cache length) stays one; the registry does what the host does before the launch
// (RangeUtils.h's checks and size in SymInt arithmetic, the allocation). Outside a trace the
// entry runs the same launches in ordinary mode, which is how the parity test compares it with
// the real op.
#include <ATen/cuda/host_trace/Field.h>
#include <ATen/cuda/host_trace/Launch.h>
#include <ATen/cuda/host_trace/Recorder.h>
#include <ATen/cuda/host_trace/ti/Ops.h>

namespace at::cuda::host_trace {

// arange's start and step: int fields for an integral accumulate type, float
// fields otherwise (a SymInt bound reaches an int field as it is, a floating
// arange takes it as a SymFloat)
template <class T, size_t Off>
using ScalarField = std::conditional_t<std::is_integral_v<T>, IntField<T, Off>, FloatField<T, Off>>;
template <class scalar_t, class acc_t>
struct Traced<at::native::ArangeFunctor<scalar_t, acc_t>> : TracedBase {
  using P = at::native::ArangeFunctor<scalar_t, acc_t>;
  P pod{};
  ScalarField<acc_t, 0> xstart{this, "xstart"};
  ScalarField<acc_t, sizeof(acc_t)> xstep{this, "xstep"};
  Traced() : TracedBase(&pod, sizeof(P)) {}
};
using ArangeFloatFunctor = at::native::ArangeFunctor<float, float>;
using ArangeLongFunctor = at::native::ArangeFunctor<int64_t, int64_t>;
static_assert(offsetof(ArangeFloatFunctor, xstep) == sizeof(float), "ArangeFunctor layout");
static_assert(offsetof(ArangeLongFunctor, xstep) == sizeof(int64_t), "ArangeFunctor layout");

} // namespace at::cuda::host_trace

namespace at::cuda::host_trace::ti {

namespace {
// a bound into the functor's field: an int field takes the Scalar's SymInt
// (a bound of an integral arange is an int or a SymInt); a float field takes
// a plain integral bound converted to the accumulate type once, as
// RangeFactories.cu's start.to<accscalar_t>() does, and any other bound as
// its SymFloat (a float, or a symbolic number kept as an expression)
template <class T, size_t Off>
void assign_bound(IntField<T, Off>& field, const Scalar& s) {
  field = s.toSymInt();
}
template <class T, size_t Off>
void assign_bound(FloatField<T, Off>& field, const Scalar& s) {
  if (s.isIntegral(/*includeBool=*/true) && !s.isSymbolic()) {
    field = s.to<T>();
  } else {
    field = s.toSymFloat();
  }
}

// gpu_kernel_with_index's CUDA branch with the element count as a c10::SymInt
// and the launch through the typed helper; the kernel is the one above
template <typename func_t>
void gpu_kernel_with_index_sym(const Tensor& output, const func_t& f) {
  const c10::SymInt N = output.sym_numel();
  if (N == 0) {
    return;
  }
  if (N > std::numeric_limits<int>::max()) {
    decline("host_trace: arange over more than 2^31 - 1 elements (the 64-bit index kernel) is not traced (declined)");
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  Grid grid((N + ::block_work_size - 1) / ::block_work_size);
  launch(::elementwise_kernel_with_index<int, pod_t<func_t>>, grid, ::num_threads(), 0, stream, N, f, sym_mutable_data_ptr(output));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

Tensor& arange_traced(const Scalar& start, const Scalar& step, Tensor& out) {
  TORCH_INTERNAL_ASSERT(out.dim() == 1);
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, out.scalar_type(), "arange_traced", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    Traced<at::native::ArangeFunctor<scalar_t, accscalar_t>> f;
    assign_bound(f.xstart, start);
    assign_bound(f.xstep, step);
    gpu_kernel_with_index_sym(out, f);
  });
  return out;
}

} // namespace at::cuda::host_trace::ti
