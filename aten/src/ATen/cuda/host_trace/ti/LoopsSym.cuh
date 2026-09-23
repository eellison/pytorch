// CUDALoops.cuh's elementwise launch path for the traced sibling iterator:
// launch_vectorized_kernel, launch_legacy_kernel and gpu_kernel_impl_nocast
// with the sizes as c10::SymInt, the pointer array as a TracedArray, the
// alignment ladder as guards and the launches through the typed helper. The
// kernels are ATen's templates from CUDALoops.cuh, instantiated here for the
// same functor and array types the real op uses, so the device side is shared
// and the real launch path is never executed under a trace.
//
// One thing could not stay a re-typing of the original. IntDivider's magic
// number and shift are opaque rebinds of the divisor: the constructor's
// shift loop would otherwise guard every size into a power-of-two interval.
// The strided path's functor is CUDALoops.cuh's StridedOp / StridedCastOp,
// the named type eager launches; the proxies below view its members.
#pragma once
#include <ATen/cuda/host_trace/Launch.h>
#include <ATen/cuda/host_trace/ti/Slots.h>
#include <ATen/cuda/host_trace/ti/TensorIteratorSym.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/util/bit_cast.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace at::cuda::host_trace::ti {
using namespace at::native;

// the functor traits the launch templates read are those of the POD behind a
// proxy (Traced<AddFunctor<T>> launches as AddFunctor<T>)
template <class T>
struct function_traits : ::function_traits<pod_t<T>> {};
template <class func_t>
constexpr auto calc_io_size() {
  return at::native::calc_io_size<pod_t<func_t>>();
}

// ---- IntegerDivider.cuh / OffsetCalculator.cuh: host twins with SymInt
// members; the proxy views below assign from them member for member.
namespace detail {
inline int64_t intdivider_shift(const int64_t* a, size_t) {
  const unsigned int divisor = static_cast<unsigned int>(a[0]);
  unsigned int shift = 0;
  for (shift = 0; shift < 32; shift++) if ((1U << shift) >= divisor) break;
  return shift;
}
inline int64_t intdivider_m1(const int64_t* a, size_t n) {
  const unsigned int divisor = static_cast<unsigned int>(a[0]);
  const unsigned int shift = static_cast<unsigned int>(intdivider_shift(a, n));
  uint64_t one = 1;
  uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
  return static_cast<int64_t>(static_cast<unsigned int>(magic));
}
} // namespace detail

struct IntDividerT {
  IntDividerT() = default;
  IntDividerT(c10::SymInt d) : divisor(d) {
    // the same values IntDivider<unsigned int>'s constructor computes, as
    // functions of the divisor re-evaluated per call rather than as guards
    shift = opaque("intdivider_shift", {d}, &detail::intdivider_shift, "rebind");
    m1 = opaque("intdivider_m1", {d}, &detail::intdivider_m1, "rebind");
  }
  c10::SymInt divisor{0};
  c10::SymInt m1{0};
  c10::SymInt shift{0};
};
template <class T>
using IntDivider = IntDividerT;

template <int NARGS, typename index_t = uint32_t, bool signed_strides = false>
struct OCT {
  using stride_t = c10::SymInt;
  OCT(int dims, const c10::SymInt* sizes, const c10::SymInt* const* strides, const int64_t* element_sizes=nullptr) : dims(dims) {
    TORCH_CHECK(dims <= MAX_DIMS, "tensor has too many (>", MAX_DIMS, ") dims");
    for (int i=0; i < dims; i++){
      sizes_[i] = IntDivider<index_t>(sizes[i]);
      for (int arg = 0; arg < NARGS; arg++) {
        int64_t element_size = (element_sizes == nullptr ? 1LL : element_sizes[arg]);
        strides_[i][arg] = strides[arg][i] / element_size;
      }
    }
  }
  int dims;
  IntDivider<index_t> sizes_[MAX_DIMS];
  stride_t strides_[MAX_DIMS][std::max<int>(NARGS, 1)];
};
template <int N, typename index_t = uint32_t, bool s = false>
using OffsetCalculator = OCT<N, index_t, s>;

template<int N, bool signed_strides = false>
static OffsetCalculator<N, uint32_t, signed_strides> make_offset_calculator(const TensorIteratorSym& iter) {
  TORCH_INTERNAL_ASSERT(N <= iter.ntensors());
  std::array<const c10::SymInt*, N> strides;
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i).data();
  }
  return OffsetCalculator<N, uint32_t, signed_strides>(iter.ndim(), iter.shape().data(), strides.data());
}

// ---- the strided paths' functors are eager's (CUDALoops.cuh StridedOp /
// StridedCastOp: data, offset_calc[, dtypes], f); the proxies below view them
using at::native::StridedOp;
using at::native::StridedCastOp;

struct IntDividerView {
  using ID = ::at::cuda::detail::IntDivider<unsigned int>;
  TracedBase* o;
  size_t base;
  SlotName nm;
  IntSlot<unsigned int> divisor;
  IntSlot<unsigned int> m1;
  IntSlot<unsigned int> shift;
  IntDividerView(TracedBase* o, size_t base, SlotName nm)
      : o(o), base(base), nm(nm),
        divisor(o, base + offsetof(ID, divisor), SlotName{&this->nm, "divisor"}),
        m1(o, base + offsetof(ID, m1), SlotName{&this->nm, "m1"}),
        shift(o, base + offsetof(ID, shift), SlotName{&this->nm, "shift"}) {}
  IntDividerView(const IntDividerView&) = delete;
  IntDividerView& operator=(const IntDividerT& s) {
    divisor = s.divisor;
    m1 = s.m1;
    shift = s.shift;
    return *this;
  }
};

template <int N>
struct OffsetCalculatorView {
  using OC = ::OffsetCalculator<N, uint32_t, false>;
  TracedBase* o;
  size_t base;
  SlotName nm;
  IntSlot<int> dims;
  ArrayOf<IntDividerView, sizeof(IntDividerView::ID), MAX_DIMS> sizes_;
  ArrayOf<ArrayOf<IntSlot<uint32_t>, sizeof(uint32_t), N>, sizeof(uint32_t) * N, MAX_DIMS> strides_;
  OffsetCalculatorView(TracedBase* o, size_t base, SlotName nm)
      : o(o), base(base), nm(nm),
        dims(o, base + offsetof(OC, dims), SlotName{&this->nm, "dims"}),
        sizes_(o, base + offsetof(OC, sizes_), SlotName{&this->nm, "sizes_"}),
        strides_(o, base + offsetof(OC, strides_), SlotName{&this->nm, "strides_"}) {}
  OffsetCalculatorView(const OffsetCalculatorView&) = delete;
  OffsetCalculatorView& operator=(const OCT<N>& s) {
    dims = s.dims;
    for (int i = 0; i < s.dims; ++i) {
      sizes_[i] = s.sizes_[i];
      strides_[i] = s.strides_[i];
    }
    return *this;
  }
};

// A functor inside the StridedOp proxy. A plain functor has no symbolic
// member: its bytes are constants of the variant, copied and not recorded.
// The generated fragments (HostTraceSibling_*.cuh) specialize the view for
// the functors that carry a value.
template <class F>
struct FunctorView {
  TracedBase* o;
  size_t base;
  SlotName nm;
  FunctorView(TracedBase* o, size_t base, SlotName nm) : o(o), base(base), nm(nm) {}
  FunctorView(const FunctorView&) = delete;
  FunctorView& operator=(const F& s) {
    // a copy, not a memcpy: padding and empty members stay the zeros the proxy
    // was built with, so the image is the same bytes at the trace and the build
    new (static_cast<char*>(o->pod) + base) F(s);
    return *this;
  }
};

} // namespace at::cuda::host_trace::ti

namespace at::cuda::host_trace {

template <class F, int N>
struct Traced<ti::StridedOp<F, N>> : TracedBase {
  using P = ti::StridedOp<F, N>;
  alignas(P) unsigned char pod_bytes[sizeof(P)] = {};
  ti::ArrayOf<ti::PtrSlot, sizeof(char*), N> data;
  ti::OffsetCalculatorView<N> offset_calc;
  ti::FunctorView<F> f;
  Traced()
      : TracedBase(pod_bytes, sizeof(P)),
        data(this, offsetof(P, data), ti::SlotName{nullptr, "data"}),
        offset_calc(this, offsetof(P, offset_calc), ti::SlotName{nullptr, "offset_calc"}),
        f(this, offsetof(P, f), ti::SlotName{nullptr, "f"}) {}
};

// the dtypes array is a constant of the variant (FunctorView copies a plain value)
template <class F, int N>
struct Traced<ti::StridedCastOp<F, N>> : TracedBase {
  using P = ti::StridedCastOp<F, N>;
  alignas(P) unsigned char pod_bytes[sizeof(P)] = {};
  ti::ArrayOf<ti::PtrSlot, sizeof(char*), N> data;
  ti::OffsetCalculatorView<N> offset_calc;
  ti::FunctorView<std::array<ScalarType, N>> dtypes;
  ti::FunctorView<F> f;
  Traced()
      : TracedBase(pod_bytes, sizeof(P)),
        data(this, offsetof(P, data), ti::SlotName{nullptr, "data"}),
        offset_calc(this, offsetof(P, offset_calc), ti::SlotName{nullptr, "offset_calc"}),
        dtypes(this, offsetof(P, dtypes), ti::SlotName{nullptr, "dtypes"}),
        f(this, offsetof(P, f), ti::SlotName{nullptr, "f"}) {}
};

} // namespace at::cuda::host_trace

namespace at::cuda::host_trace::ti {

// ---- MemoryAccess.cuh: the vectorization width from the operand addresses,
// each alignment test a guard
namespace memory {
using at::native::memory::aligned_vector;
using at::native::memory::LoadWithoutCast;
using at::native::memory::StoreWithoutCast;

template<typename scalar_t>
inline int can_vectorize_up_to(const c10::SymInt& pointer) {
  constexpr int vec2_alignment = std::alignment_of_v<aligned_vector<scalar_t, 2>>;
  constexpr int vec4_alignment = std::alignment_of_v<aligned_vector<scalar_t, 4>>;
  constexpr int vec8_alignment = std::alignment_of_v<aligned_vector<scalar_t, 8>>;
  if (aligned(pointer, vec8_alignment).guard_bool(__FILE__, __LINE__)) {
   return 8;
  } else if (aligned(pointer, vec4_alignment).guard_bool(__FILE__, __LINE__)) {
    return 4;
  } else if (aligned(pointer, vec2_alignment).guard_bool(__FILE__, __LINE__)) {
    return 2;
  }
  return 1;
}

// `pointers` hold the addresses of [output, input0, input1, ...]: the width is
// the minimum over the result type and every argument type
template<typename func_t, typename array_t, size_t... I>
inline int can_vectorize_up_to_impl(const array_t& pointers, std::index_sequence<I...>) {
  using traits = function_traits<func_t>;
  int result = can_vectorize_up_to<typename traits::result_type>(pointers[0]);
  ((result = std::min<int>(result, can_vectorize_up_to<typename traits::template arg<I>::type>(pointers[I + 1]))), ...);
  return result;
}

template<typename func_t, typename array_t>
inline int can_vectorize_up_to(const array_t& pointers) {
  return can_vectorize_up_to_impl<func_t>(pointers, std::make_index_sequence<function_traits<func_t>::arity>{});
}
}  // namespace memory

// ---- CUDALoops.cuh: the launches
template <typename func_t, typename array_t>
static inline void launch_vectorized_kernel(
    c10::SymInt N,
    const func_t& f,
    array_t data) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  constexpr auto io_size = calc_io_size<func_t>();
  auto stream = at::cuda::getCurrentCUDAStream();
  using cpp_type = typename function_traits<func_t>::result_type;
  const uint16_t max_vec_size = memory::can_vectorize_up_to<func_t>(data);
  uint16_t vec_size = 16 / static_cast<uint16_t>(sizeof(cpp_type));
  cudaDeviceProp* p = at::cuda::getDeviceProperties(stream.device().index());
  // Empirical benchmarking set this threshold; retain the existing path for
  // smaller workloads to avoid performance regressions
  // (the larger thread work size decreases thread level parallelism,
  // which is important for small footprints).
  constexpr int64_t min_sm107_io_size = 16 * 1024 * 1024;
  const bool use_sm107_optimizations =
      p->major == 10 && p->minor == 7 && N * io_size >= min_sm107_io_size;
  if (use_sm107_optimizations) {
    using args_t = typename function_traits<func_t>::ArgsTuple;
    constexpr auto max_input_size = at::native::max_of_sizes(
        args_t{}, std::make_index_sequence<std::tuple_size_v<args_t>>{});
    vec_size = 32 / static_cast<uint16_t>(std::max(sizeof(cpp_type), max_input_size));
  }
  vec_size = std::min<uint16_t>(vec_size, max_vec_size);
  // due to excessive binary size the `vectorized_elementwise_kernel` of
  // the size 8 is compiled for sm_90 and sm_10x only.
  // TODO: Lift this limitation when CUDA 12.x support is fully dropped
  if (p->major != 9 && p->major != 10) {
    vec_size = std::min<uint16_t>(vec_size, 4);
  }
#if !defined(CUDA_VERSION) || CUDA_VERSION < 12080
  if constexpr (sizeof(cpp_type) < 2) {
    vec_size = std::min<uint16_t>(vec_size, 4);
  }
#endif
  int tws = elems_per_thread<io_size>();
  constexpr auto input_size = io_size - sizeof(cpp_type);
  constexpr auto tws_128b = elems_per_thread_128b<input_size>(sizeof(cpp_type));
  if (tws_128b >= 8 && use_sm107_optimizations && vec_size == 8) {
    tws = tws_128b;
  }
  int bws = tws * num_threads();
  Grid grid = (N + bws - 1) / bws;
  switch (vec_size) {
    case 8:
      if (use_sm107_optimizations) {
        launch(vectorized_elementwise_kernel<8, pod_t<func_t>, pod_t<array_t>, tws_128b >= 8>,
            grid, num_threads(), 0, stream, N, f, data);
      } else {
        launch(vectorized_elementwise_kernel<8, pod_t<func_t>, pod_t<array_t>>,
            grid, num_threads(), 0, stream, N, f, data);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 4:
      launch(vectorized_elementwise_kernel<4, pod_t<func_t>, pod_t<array_t>>,
          grid, num_threads(), 0, stream, N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 2:
      launch(vectorized_elementwise_kernel<2, pod_t<func_t>, pod_t<array_t>>,
          grid, num_threads(), 0, stream, N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 1: {
      auto input_calc = TrivialOffsetCalculator<traits::arity>();
      auto output_calc = TrivialOffsetCalculator<1>();
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();
      Grid grid_unrolled = (N + elementwise_block_work_size() - 1) / elementwise_block_work_size();
      launch(unrolled_elementwise_kernel<pod_t<func_t>, pod_t<array_t>, elementwise_thread_work_size(), decltype(input_calc), decltype(output_calc), decltype(loader), decltype(storer)>,
          grid_unrolled, num_threads(), 0, stream,
              N, f, data, input_calc, output_calc, loader, storer);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}

template <int nt, int vt, typename func_t>
static void launch_legacy_kernel(c10::SymInt N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  Grid grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  launch(elementwise_kernel<nt, vt, pod_t<func_t>>, grid, block, 0, stream, N, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename func_t>
void gpu_kernel_impl_nocast(TensorIteratorSym& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  TracedArray<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = iter.data_ptr(i);
  }

  c10::SymInt numel = iter.numel();

  bool contiguous = iter.is_contiguous();

  if (contiguous) {
    return launch_vectorized_kernel(numel, f, data);
  }
  auto offset_calc = make_offset_calculator<traits::arity + 1>(iter);
  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
  Traced<StridedOp<pod_t<func_t>, ntensors>> op;
  op.data = data;
  op.offset_calc = offset_calc;
  op.f = f;
  launch_legacy_kernel<128, unroll_factor>(numel, op);
}

// CUDALoops.cuh's needs_dynamic_casting on the sibling iterator: an operand
// whose dtype differs from the functor's static type at that position
template <typename func_t, int nargs = function_traits<func_t>::arity>
struct needs_dynamic_casting {
  static bool check(const TensorIteratorSym& iter) {
    using traits = function_traits<func_t>;
    using cpp_type = typename traits::template arg<nargs - 1>::type;
    if (iter.input_dtype(nargs - 1) != c10::CppTypeToScalarType<cpp_type>::value) {
      return true;
    }
    return needs_dynamic_casting<func_t, nargs - 1>::check(iter);
  }
};
template <typename func_t>
struct needs_dynamic_casting<func_t, 0> {
  static bool check(const TensorIteratorSym& iter) {
    using traits = function_traits<func_t>;
    using cpp_type = typename traits::result_type;
    return iter.dtype(0) != c10::CppTypeToScalarType<cpp_type>::value;
  }
};

// CUDALoops.cuh's gpu_kernel_impl dynamic-cast branch (the CUDA one: the
// unrolled kernel with LoadWithCast / StoreWithCast when contiguous, the
// legacy kernel with fetch_and_cast / cast_and_store otherwise). The cast
// policies and the dtypes array are constants of the variant.
template <typename func_t>
void gpu_kernel_impl_cast(TensorIteratorSym& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  TracedArray<char*, ntensors> data;
  std::array<ScalarType, ntensors> dtypes;
  for (int i = 0; i < ntensors; i++) {
    data[i] = iter.data_ptr(i);
    dtypes[i] = iter.dtype(i);
  }
  c10::SymInt numel = iter.numel();
  auto stream = at::cuda::getCurrentCUDAStream();

  if (iter.is_contiguous()) {
    // the real policies are built from a TensorIteratorBase; the sibling fills
    // a layout twin and bit-casts (both are trivially copyable, same size)
    using loader_t = at::native::memory::LoadWithCast<traits::arity>;
    using storer_t = at::native::memory::StoreWithCast<1>;
    struct LoaderTwin { typename loader_t::array_t dtypes; typename loader_t::size_array_t element_sizes; } lt{};
    struct StorerTwin { typename storer_t::array_t dtypes; typename storer_t::size_array_t element_sizes; } st{};
    static_assert(sizeof(LoaderTwin) == sizeof(loader_t) && sizeof(StorerTwin) == sizeof(storer_t), "cast policy layout");
    for (int i = 0; i < traits::arity; ++i) {
      lt.dtypes[i] = iter.dtype(i + 1);
      lt.element_sizes[i] = static_cast<uint32_t>(c10::elementSize(iter.dtype(i + 1)));
    }
    st.dtypes[0] = iter.dtype(0);
    st.element_sizes[0] = static_cast<uint32_t>(c10::elementSize(iter.dtype(0)));
    loader_t loader = c10::bit_cast<loader_t>(lt);
    storer_t storer = c10::bit_cast<storer_t>(st);
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    auto output_calc = TrivialOffsetCalculator<1>();
    Grid grid((numel + elementwise_block_work_size() - 1) / elementwise_block_work_size());
    launch(unrolled_elementwise_kernel<pod_t<func_t>, std::array<char*, ntensors>, elementwise_thread_work_size(), decltype(input_calc), decltype(output_calc), loader_t, storer_t>,
        grid, num_threads(), 0, stream, numel, f, data, input_calc, output_calc, loader, storer);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  auto offset_calc = make_offset_calculator<ntensors>(iter);
  Traced<StridedCastOp<pod_t<func_t>, ntensors>> op;
  op.data = data;
  op.offset_calc = offset_calc;
  op.dtypes = dtypes;
  op.f = f;
  launch_legacy_kernel<128, 4>(numel, op);
}

// Loops.cuh's gpu_kernel_opaque / gpu_kernel_nocast: the no-cast path over
// operands the functor reads as opaque values of their size (where's kernel),
// whatever their dtypes
template <typename func_t>
void gpu_kernel_nocast(TensorIteratorSym& iter, const func_t& f) {
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_cuda(),
      "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }
  if (iter.numel() == 0) {
    return;
  }
  if (!iter.can_use_32bit_indexing()) {
    iter.with_32bit_indexing(); // declines
  }
  gpu_kernel_impl_nocast(iter, f);
}

// Loops.cuh's gpu_kernel for one-output ops: the no-cast path when every
// operand has the functor's static type at its position, the dynamic-cast
// path otherwise (a copy_ between dtypes).
template <typename func_t>
void gpu_kernel(TensorIteratorSym& iter, const func_t& f) {
  // everything from here on picks the kernel and its configuration (the
  // index width, the dynamic cast, the contiguous / vectorized / strided
  // ladder); the iterator's shape and stride decisions were made in build
  KernelChoice choice;
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_cuda(),
      "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }
  if (iter.numel() == 0) {
    return;
  }
  if (!iter.can_use_32bit_indexing()) {
    iter.with_32bit_indexing(); // declines
  }
  if (needs_dynamic_casting<func_t>::check(iter)) {
    return gpu_kernel_impl_cast(iter, f);
  }
  gpu_kernel_impl_nocast(iter, f);
}

// An AUnaryFunctor / BUnaryFunctor (Loops.cuh) as a constant of the variant:
// it holds the inner functor before its scalar, and the bytes of that member
// and the padding after it must be the same at the trace and at the build's
// capture, which a stack object does not give (an empty member's trivial copy
// carries whatever byte the argument had). The functor is built into zeroed
// bytes and, for an empty inner functor, the prefix before the scalar is
// zeroed again; an inner functor with a value (CompareEqFunctor's op) has no
// padding of its own, so its copy writes exactly its bytes. The scalar's
// offset comes from a layout-compatible twin, since the members are private;
// is_layout_compatible checks the member order and types, not only the size
// (a reordered functor upstream would keep sizeof and put the scalar where
// the twin has the functor).
template <class F, class func_t, class scalar_t>
struct ScalarFunctor {
  struct Twin {
    func_t f;
    scalar_t s;
  };
  static_assert(sizeof(Twin) == sizeof(F) && std::is_standard_layout_v<F> && std::is_standard_layout_v<Twin>, "AUnaryFunctor / BUnaryFunctor layout");
  static_assert(std::is_layout_compatible_v<F, Twin>, "AUnaryFunctor / BUnaryFunctor layout twin: the members are not (func_t f; scalar s) in this order");
  static_assert(std::is_empty_v<func_t> || std::has_unique_object_representations_v<func_t>, "the inner functor's bytes are its value");
  alignas(F) unsigned char bytes[sizeof(F)] = {};
  ScalarFunctor(const func_t& f, scalar_t s) {
    new (static_cast<void*>(bytes)) F(f, s);
    if constexpr (std::is_empty_v<func_t>) {
      std::memset(bytes, 0, offsetof(Twin, s));
    }
  }
  const F& get() const {
    return *reinterpret_cast<const F*>(bytes);
  }
};

// A functor with padding (a scalar member narrower than its alignment, a
// trailing enum) built member-wise into zeroed bytes: its padding is 0 at the
// trace and at the build's capture, and every copy of it (the launch's slot,
// the StridedOp proxy) carries the zeros along.
template <class F>
struct Zeroed {
  alignas(F) unsigned char bytes[sizeof(F)] = {};
  template <class... Args>
  explicit Zeroed(Args&&... args) {
    new (static_cast<void*>(bytes)) F{std::forward<Args>(args)...};
  }
  const F& get() const {
    return *reinterpret_cast<const F*>(bytes);
  }
};

// ---- Loops.cuh's gpu_kernel_with_scalars family. A CPU scalar operand is
// read as a value at the trace (a constant of the variant, baked into the
// functor as the real host bakes it), removed from the iterator, and the
// launch is the unary AUnaryFunctor / BUnaryFunctor over the remaining
// operand: the identical functor and kernel instantiation the real op uses.
template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
void opmath_gpu_kernel_with_scalars(TensorIteratorSym& iter, const func_t& f) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);
  using traits = ::function_traits<func_t>;
  using opmath_arg1_t = typename traits::template arg<0>::type;
  using opmath_arg2_t = typename traits::template arg<1>::type;
  static_assert(traits::arity == 2, "gpu_kernel_with_scalars only supports two input arguments");
  KernelChoice choice; // the scalar-operand route picks the functor
  if (iter.is_cpu_scalar(1)) {
    ScalarFunctor<AUnaryFunctor<arg1_t, arg2_t, return_t, func_t>, func_t, opmath_arg1_t> af(f, iter.scalar_value<opmath_arg1_t>(1));
    iter.remove_operand(1);
    gpu_kernel(iter, af.get());
  } else if (iter.is_cpu_scalar(2)) {
    ScalarFunctor<BUnaryFunctor<arg1_t, arg2_t, return_t, func_t>, func_t, opmath_arg2_t> bf(f, iter.scalar_value<opmath_arg2_t>(2));
    iter.remove_operand(2);
    gpu_kernel(iter, bf.get());
  } else {
    gpu_kernel(iter, BinaryFunctor<arg1_t, arg2_t, return_t, func_t>(f));
  }
}

template <typename func_t>
void gpu_kernel_with_scalars(TensorIteratorSym& iter, const func_t& f) {
  using traits = ::function_traits<func_t>;
  static_assert(traits::arity == 2, "gpu_kernel_with_scalars only supports two input arguments");
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  using return_t = typename traits::result_type;
  opmath_gpu_kernel_with_scalars<arg1_t, arg2_t, return_t, func_t>(iter, f);
}

// the symmetric form (f(a, b) == f(b, a)): one AUnaryFunctor instantiation
// for a scalar on either side, as mul launches
template <typename scalar_t, typename return_t = scalar_t, typename func_t>
void opmath_symmetric_gpu_kernel_with_scalars(TensorIteratorSym& iter, const func_t& f) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);
  using traits = ::function_traits<func_t>;
  using opmath_arg_t = typename traits::template arg<0>::type;
  static_assert(traits::arity == 2, "gpu_kernel_with_scalars only supports two input arguments");
  static_assert(std::is_same_v<opmath_arg_t, typename traits::template arg<1>::type>, "f is not symmetric");
  KernelChoice choice; // the scalar-operand route picks the functor
  opmath_arg_t scalar_val{};
  if (iter.is_cpu_scalar(1)) {
    scalar_val = iter.scalar_value<opmath_arg_t>(1);
    iter.remove_operand(1);
  } else if (iter.is_cpu_scalar(2)) {
    scalar_val = iter.scalar_value<opmath_arg_t>(2);
    iter.remove_operand(2);
  }
  if (iter.ninputs() == 2) {
    gpu_kernel(iter, BinaryFunctor<scalar_t, scalar_t, return_t, func_t>(f));
  } else {
    ScalarFunctor<AUnaryFunctor<scalar_t, scalar_t, return_t, func_t>, func_t, opmath_arg_t> unary_f(f, scalar_val);
    gpu_kernel(iter, unary_f.get());
  }
}

} // namespace at::cuda::host_trace::ti
