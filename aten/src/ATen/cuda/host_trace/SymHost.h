// The host kind a symbolic TensorIterator's launch code runs as
// (ATen/native/cuda/CUDALoops.cuh, Loops.cuh): the same routing code eager runs
// as EagerHost, over c10::SymInt sizes, strides and addresses, recording each
// launch in the current Recorder instead of launching it. Every decision on a
// symbolic value is a guard of the trace; nothing is read from a tensor.
//
// What is instantiated per functor stays small: each launch parameter is a POD
// ArgDesc that one out-of-line recorder (SymHost.cpp) turns into its bytes.
#pragma once
#include <ATen/TensorIterator.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/host_trace/LaunchLayout.h>
#include <ATen/cuda/host_trace/Recorder.h>

#include <array>
#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace at::cuda::host_trace {

// an OffsetCalculator's host inputs: sizes and per-operand strides (bytes, or
// elements where eager's construction divides by element sizes)
struct SymOffsets {
  int dims = 0;
  c10::SmallVector<c10::SymInt, 6> sizes;
  c10::SmallVector<c10::SmallVector<c10::SymInt, 6>, 4> strides;
};

// make_offset_calculator over operands [first, first + n) of iter; with
// element_strides, make_input/output_offset_calculator's
TORCH_CUDA_CPP_API SymOffsets sym_offsets(const TensorIteratorBase& iter, int first, int n, bool element_strides);

// the routing code's value for an OffsetCalculator<N> parameter
template <int N>
struct SymOffsetCalculator {
  SymOffsets offsets;
};

// a StridedOp / StridedCastOp launch parameter: its bytes with the functor
// the stub built (and the dtypes) in place; the symbolic data and offsets are
// set at the launch
template <class Op, size_t N>
struct SymStrided {
  std::vector<uint8_t> raw;
  const void* f;
  size_t f_offset;
  size_t data_offset;
  size_t calc_offset;
  std::array<c10::SymInt, N> data;
  SymOffsets offsets;
};

// A launch parameter as the routing code holds it, for the out-of-line
// recorder: what is per functor stays a few stores
struct ArgDesc {
  enum Kind : uint8_t { Scalar, Pointers, Offsets, Bytes, Strided };
  Kind kind = Bytes;
  uint32_t size = 0;
  uint32_t align = 1;
  uint32_t n = 0;
  // Scalar: a SymInt; Pointers: SymInt[n]; Offsets: SymOffsets; Bytes, Strided: the parameter's bytes
  const void* value = nullptr;
  // Bytes, Strided: the functor object whose pending CPU scalars are fields, at f_offset
  const void* object = nullptr;
  uint32_t f_offset = 0;
  // Strided: data[n] at data_offset, the OffsetCalculator<n> at calc_offset
  const c10::SymInt* data = nullptr;
  const SymOffsets* offsets = nullptr;
  uint32_t data_offset = 0;
  uint32_t calc_offset = 0;
};

TORCH_CUDA_CPP_API void record_kernel(const void* kernel, const ArgDesc* args, size_t n, SymDim3 grid, SymDim3 block, int64_t smem);
// a launch of a function the driver gives (a jitted kernel's), its parameters as eager packs them
TORCH_CUDA_CPP_API void record_function(cudaFunction_t function, const ArgDesc* args, size_t n, SymDim3 grid, SymDim3 block, int64_t smem);

// the object a jitted kernel's scalar parameter (CUDAJitLoops.cuh's
// scalar_val) is pending as, for a ScalarScope of a host buffer's value
TORCH_CUDA_CPP_API extern const char jitted_scalar_slot;

// a parameter of `size` bytes eager passes by address, e.g. a jitted
// kernel's scalar or extra argument; `object` for its pending CPU scalar
inline ArgDesc bytes_desc(const void* value, size_t size, const void* object = nullptr) {
  ArgDesc d;
  d.kind = ArgDesc::Bytes;
  d.size = static_cast<uint32_t>(size);
  d.align = static_cast<uint32_t>(size);
  d.value = value;
  d.object = object;
  return d;
}

// A functor member a replay reads from a host buffer, while its functor object
// is being launched (opmath_*_with_scalars); without a source, a member of
// `width` bytes holding a symbolic expression's bits (a symbolic Scalar argument)
struct PendingScalar {
  const void* object;
  size_t offset;
  char cls;
  TensorBase source;
  size_t width = 0;
  c10::SymInt bits;
};
TORCH_CUDA_CPP_API std::vector<PendingScalar>& pending_scalars();

// a host trace's stand-in for a host buffer: its value is each replay's
inline bool is_host_buffer(const TensorBase& t) {
  return t.key_set().has(c10::DispatchKey::Python);
}

struct TORCH_CUDA_CPP_API ScalarScope {
  ScalarScope(const void* object, const void* member, char cls, const TensorBase& source);
  ScalarScope(const void* object, const void* member, size_t width, c10::SymInt bits);
  ~ScalarScope();
  ScalarScope(const ScalarScope&) = delete;
  ScalarScope& operator=(const ScalarScope&) = delete;
};

// an operand's symbolic address
TORCH_CUDA_CPP_API c10::SymInt operand_address(const TensorIteratorBase& iter, int arg);

[[noreturn]] TORCH_CUDA_CPP_API void unsupported_scalar_member(const char* type);
[[noreturn]] TORCH_CUDA_CPP_API void unsupported_symbol_member(const char* type);
// static_cast<float>(v)'s bits as an int32, through the program's rows
TORCH_CUDA_CPP_API c10::SymInt float_bits(const c10::SymInt& v);

// The launch parameter of type K from what the routing code holds for it
template <class K, class A>
ArgDesc arg_desc(const A& a) {
  ArgDesc d;
  d.size = sizeof(K);
  d.align = alignof(K);
  if constexpr (std::is_arithmetic_v<K>) {
    static_assert(std::is_same_v<A, c10::SymInt>, "a scalar launch parameter from a SymInt");
    d.kind = ArgDesc::Scalar;
    d.value = &a;
  } else {
    static_assert(std::is_same_v<K, A>, "a launch parameter the host trace cannot record");
    // a functor, loader, storer or trivial offset calculator: its bytes as eager passes them
    d.kind = ArgDesc::Bytes;
    d.value = &a;
    d.object = &a;
  }
  return d;
}

template <class K, size_t N>
ArgDesc arg_desc(const std::array<c10::SymInt, N>& a) {
  static_assert(std::is_same_v<K, std::array<char*, N>>);
  ArgDesc d;
  d.kind = ArgDesc::Pointers;
  d.size = sizeof(K);
  d.align = alignof(K);
  d.n = N;
  d.value = a.data();
  return d;
}

template <class K, int N>
ArgDesc arg_desc(const SymOffsetCalculator<N>& c) {
  static_assert(std::is_same_v<K, ::OffsetCalculator<N>>);
  ArgDesc d;
  d.kind = ArgDesc::Offsets;
  d.size = sizeof(K);
  d.align = alignof(K);
  d.n = N;
  d.value = &c.offsets;
  return d;
}

template <class K, class Op, size_t N>
ArgDesc arg_desc(const SymStrided<Op, N>& s) {
  static_assert(std::is_same_v<K, Op>);
  ArgDesc d;
  d.kind = ArgDesc::Strided;
  d.size = sizeof(Op);
  d.align = alignof(Op);
  d.n = N;
  d.value = s.raw.data();
  d.object = s.f;
  d.f_offset = s.f_offset;
  d.data = s.data.data();
  d.offsets = &s.offsets;
  d.data_offset = s.data_offset;
  d.calc_offset = s.calc_offset;
  return d;
}

// What CUDALoops.cuh's launch routing reads of a functor (StaticFacts<func_t>
// there), at run time: the routing runs once, over these, for every functor
// of a symbolic iterator's recorded launch. The per-functor code
// (make_sym_facts) only fills them: the kernel instantiations eager launches
// for each route, the functor's bytes and its type facts.
struct TORCH_CUDA_CPP_API SymFacts {
  static constexpr int max_tensors = 9;
  int arity = 0;
  int ntensors = 0;
  int io_size = 0;
  int tws = 0;
  int tws_128b = 0;
  int strided_unroll = 0;
  size_t result_size = 0;
  size_t max_input_size = 0;
  // the result's and the arguments' C++ types (needs_dynamic_casting) and sizes (vectorization)
  std::array<ScalarType, max_tensors> types{};
  std::array<uint32_t, max_tensors> sizes{};
  // the kernels eager launches: vectorized_elementwise_kernel<8> (and its
  // sm_107 form), <4>, <2>; the unrolled kernel without and with casts;
  // elementwise_kernel of StridedOp and of StridedCastOp. Null where eager's
  // entry point launches none (gpu_kernel_nocast: no cast routes)
  std::array<const void*, 8> kernels{};
  enum Kernel { Vec8, Vec8Sm107, Vec4, Vec2, Unrolled, UnrolledCast, Strided, StridedCast };
  // the functor: its bytes, and the object whose pending CPU scalars are fields
  const void* functor = nullptr;
  uint32_t functor_size = 0;
  uint32_t functor_align = 1;
  // the unrolled kernels' offset calculators, loaders and storers of no state
  uint32_t trivial_in_size = 1;
  uint32_t trivial_out_size = 1;
  uint32_t plain_load_size = 1;
  uint32_t plain_store_size = 1;
  // LoadWithCast<arity> and StoreWithCast<1> of iter's dtypes, as their bytes
  void (*cast_params)(const TensorIteratorBase& iter, std::vector<uint8_t>& load, std::vector<uint8_t>& store) = nullptr;
  uint32_t load_align = 1;
  uint32_t store_align = 1;
  // StridedOp<func_t, ntensors> and StridedCastOp: size, alignment, member offsets
  struct Layout {
    uint32_t size = 0, align = 1, data = 0, dtypes = 0, calc = 0, f = 0;
  };
  Layout strided;
  Layout strided_cast;

  c10::SmallVector<c10::SymInt, max_tensors> data(const TensorIteratorBase& iter) const;
  int can_vectorize(c10::ArrayRef<c10::SymInt> data) const;
  bool needs_dynamic_casting(const TensorIteratorBase& iter) const;
  void record_vectorized(int vec_size, bool sm107, const c10::SymInt& grid, const c10::SymInt& n, c10::ArrayRef<c10::SymInt> data) const;
  void record_unrolled_cast(const c10::SymInt& grid, const c10::SymInt& n, c10::ArrayRef<c10::SymInt> data, const TensorIteratorBase& iter) const;
  void record_strided(bool cast, const c10::SymInt& grid, const c10::SymInt& n, c10::ArrayRef<c10::SymInt> data, const TensorIteratorBase& iter) const;
};

// the functor argument of the routing's SymHost instantiation (its bytes are SymFacts')
struct SymFunctor {};

// gpu_kernel (nocast: gpu_kernel_nocast, which checks the dtypes with
// check_cast) of a symbolic iterator: CUDALoops.cuh's routing over facts, its
// one SymHost instantiation (SymLoops.cu)
TORCH_CUDA_CPP_API void sym_gpu_kernel(TensorIteratorBase& iter, const SymFacts& facts, bool nocast, bool check_cast);

struct SymHost {
  static constexpr bool symbolic = true;
  using int_t = c10::SymInt;

  static c10::SymInt numel(const TensorIteratorBase& iter) {
    return iter.sym_numel();
  }

  template <int N>
  static std::array<c10::SymInt, N> data(const TensorIteratorBase& iter) {
    std::array<c10::SymInt, N> d;
    for (int i = 0; i < N; i++) {
      d[i] = operand_address(iter, i);
    }
    return d;
  }

  // make_offset_calculator, or with element_strides make_input/output_offset_calculator, over operands [first, first + N)
  template <int N>
  static SymOffsetCalculator<N> offset_calculator(const TensorIteratorBase& iter, int first = 0, bool element_strides = false) {
    return {sym_offsets(iter, first, N, element_strides)};
  }

  // a parameter of type T that eager passes by address (a jitted kernel's
  // scalar) as src's value at a replay, while the scope lives
  template <typename T>
  static std::unique_ptr<ScalarScope> scalar_slot(const void* object, const TensorBase& src) {
    constexpr char cls = cpu_scalar_class<T>(false);
    if (cls == 's') {
      unsupported_scalar_member(typeid(T).name());
    }
    return std::make_unique<ScalarScope>(object, object, cls, src);
  }

  // the kernel's offset calculator type for what the routing code holds
  template <typename C>
  struct offset_calculator_of {
    using type = C;
  };
  template <int N>
  struct offset_calculator_of<SymOffsetCalculator<N>> {
    using type = ::OffsetCalculator<N>;
  };
  template <typename C>
  using offset_calculator_t = typename offset_calculator_of<C>::type;

  // the value of operand `arg`, a CPU scalar, as eager's functor holds it: a
  // constant's own value, or for a host buffer a placeholder its ScalarScope
  // marks for each replay to read from the buffer
  template <typename T>
  static T scalar_value(TensorIteratorBase& iter, int arg) {
    if (is_host_buffer(iter.tensor_base(arg))) {
      return T{};
    }
    return iter.scalar_value<T>(arg);
  }

  // marks `member` of the functor `object` as operand arg's value (or with
  // `reciprocal`, 1 / it in double precision) at a replay, while the scope lives
  template <typename F, typename M>
  static std::unique_ptr<ScalarScope> scalar_member(const F& object, const M& member, const TensorIteratorBase& iter, int arg, bool reciprocal = false) {
    const auto& src = iter.tensor_base(arg);
    if (!is_host_buffer(src)) {
      return nullptr;
    }
    return scalar_member_of(object, member, src, reciprocal);
  }

  template <typename F, typename M>
  static std::unique_ptr<ScalarScope> scalar_member_of(const F& object, const M& member, const TensorBase& src, bool reciprocal = false) {
    // a class LaunchLayout's cpu_scalar_bytes reads: ScalarTypes up to 'A' + 25, reciprocals of float / double
    const char cls = reciprocal ? cpu_scalar_class<std::decay_t<M>>(true) : cpu_scalar_class<std::decay_t<M>>(false);
    if (cls == 's') {
      unsupported_scalar_member(typeid(M).name());
    }
    return std::make_unique<ScalarScope>(&object, &member, cls, src);
  }

  // marks `member` of the functor `object` as the symbolic integer v, as
  // Scalar::to<M> converts it, at a replay, while the scope lives
  template <typename F, typename M>
  static std::unique_ptr<ScalarScope> symbol_member_of(const F& object, const M& member, const c10::SymInt& v) {
    using T = std::decay_t<M>;
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
      return std::make_unique<ScalarScope>(&object, &member, sizeof(T), v);
    } else if constexpr (std::is_same_v<T, float>) {
      return std::make_unique<ScalarScope>(&object, &member, sizeof(T), float_bits(v));
    } else {
      unsupported_symbol_member(typeid(T).name());
    }
  }

  // StridedOp / StridedCastOp{data, [dtypes,] offset_calc, f}: Op's offset
  // calculator has no default constructor, so its members are laid out on
  // zeroed bytes, the functor copy-constructed in place
  template <typename Op, size_t N, typename func_t>
  static SymStrided<Op, N> strided(std::array<c10::SymInt, N> data, SymOffsetCalculator<static_cast<int>(N)> calc, const func_t& f, std::array<ScalarType, N> dtypes = {}) {
    alignas(Op) std::array<uint8_t, sizeof(Op)> raw{};
    Op& op = *reinterpret_cast<Op*>(raw.data());
    new (&op.f) func_t(f);
    if constexpr (requires { op.dtypes; }) {
      op.dtypes = dtypes;
    }
    return SymStrided<Op, N>{
        std::vector<uint8_t>(raw.begin(), raw.end()),
        &f,
        member_offset(op, op.f),
        member_offset(op, op.data),
        member_offset(op, op.offset_calc),
        std::move(data),
        std::move(calc.offsets)};
  }

  // a jitted kernel's launch (at::cuda::jit::launch_jitted_pwise_function):
  // `typed` the parameters of known type K, then `rest` (scalar, extra
  // arguments) as their bytes
  template <typename... KArgs, typename... Args>
  static void launch_jitted(cudaFunction_t function, SymDim3 grid, SymDim3 block, c10::ArrayRef<ArgDesc> rest, const Args&... args) {
    static_assert(sizeof...(KArgs) == sizeof...(Args));
    std::vector<ArgDesc> descs{arg_desc<KArgs>(args)...};
    descs.insert(descs.end(), rest.begin(), rest.end());
    record_function(function, descs.data(), descs.size(), std::move(grid), std::move(block), 0);
  }

  template <typename... KArgs, typename... Args>
  static void launch(void (*kernel)(KArgs...), SymDim3 grid, SymDim3 block, int64_t smem, const Args&... args) {
    static_assert(sizeof...(KArgs) == sizeof...(Args));
    const ArgDesc descs[] = {arg_desc<KArgs>(args)...};
    record_kernel(reinterpret_cast<const void*>(kernel), descs, sizeof...(Args), std::move(grid), std::move(block), smem);
  }
};

} // namespace at::cuda::host_trace
