// Typed launch helper: a kernel with bare arguments launches as
//
//     at::cuda::host_trace::launch(kernel<T...>, grid, block, smem, stream, args...);
//
// templated on the kernel's signature `void(KArgs...)` and on the actual argument pack. Per position:
// the kernel's parameter type gives the kind and width (i8..i64 / u32 / u64 / f32 / f64 / ptr / struct),
// the argument's static type says whether it can be symbolic (c10::SymInt / SymFloat / SymBool and the
// proxy fields, or a plain int / float / bool / raw pointer = a constant), and a SymInt's runtime check
// says whether it is symbolic in this trace (a constant SymInt records as a constant). Recording is
// positional: the offsets come from cuFuncGetParamInfo and sizeof(KArgs[i]) is cross-checked against
// the driver's size. No sentinels, no byte matching. A `Traced<P>` proxy in a struct position attaches
// its field table; a `TracedArray<T, N>` (or a std::array<c10::SymInt, N>) in an array position records
// one element per slot. Block and smem are values on the launch record; a `const T*` parameter
// records access "r", a `T*` parameter "rw". Ordinary mode builds the argument array and calls
// cudaLaunchKernel exactly as `<<<>>>` does. Trace mode does the same under the capture, records
// positionally, and reads the node the capture created from the stream's frontier right after the launch
// (finish_trace pairs the record with that node and checks the function handle against it).
#pragma once
#include <ATen/cuda/host_trace/Field.h>

#include <ATen/core/Array.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <array>
#include <cstring>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>

namespace at::cuda::host_trace {

// Block dims as SymInts: expressions on the launch record when symbolic, hint dims for the real launch.
struct Block {
  c10::SymInt x{1}, y{1}, z{1};
  Block() = default;
  Block(c10::SymInt x_, c10::SymInt y_ = 1, c10::SymInt z_ = 1) : x(std::move(x_)), y(std::move(y_)), z(std::move(z_)) {}
  Block(dim3 d) : x(static_cast<int64_t>(d.x)), y(static_cast<int64_t>(d.y)), z(static_cast<int64_t>(d.z)) {}
};

// An array of pointers or ints passed to the kernel by value (TensorIterator's `data`): the host fills
// it element by element with SymInts (`data[i] = iter.data_ptr(i)`); the launch writes the kernel's
// std::array<T, N> / at::detail::Array<T, N> from the hints and records one param per element.
template <class T, size_t N>
struct TracedArray : std::array<c10::SymInt, N> {
  using element_type = T;
  TracedArray() { this->fill(c10::SymInt(0)); }
};

// The kernel's parameter type behind a host object: Traced<P> -> P, TracedArray<T, N> -> std::array<T, N>,
// anything else -> itself. Used where a verbatim launch template names the functor / array type in the
// kernel's template argument list (elementwise_kernel<nt, vt, func_t>): the kernel is instantiated for
// the POD while the host passes the proxy.
template <class T> struct pod_of { using type = T; };
template <class P> struct pod_of<Traced<P>> { using type = P; };
template <class T, size_t N> struct pod_of<TracedArray<T, N>> { using type = std::array<T, N>; };
template <class T> using pod_t = typename pod_of<T>::type;

namespace launch_detail {

inline Grid to_grid(const Grid& g) { return g; }
inline Grid to_grid(dim3 d) { return Grid(static_cast<int64_t>(d.x), static_cast<int64_t>(d.y), static_cast<int64_t>(d.z)); }
inline Grid to_grid(const c10::SymInt& x) { return Grid(x); }
template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
Grid to_grid(I x) { return Grid(c10::SymInt(static_cast<int64_t>(x))); }
inline Block to_block(const Block& b) { return b; }
inline Block to_block(dim3 d) { return Block(d); }
inline Block to_block(const c10::SymInt& x) { return Block(x); }
template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
Block to_block(I x) { return Block(c10::SymInt(static_cast<int64_t>(x))); }

template <class K>
constexpr const char* kind_of() {
  using U = std::remove_cv_t<K>;
  if constexpr (std::is_pointer_v<U>) return "ptr";
  else if constexpr (std::is_floating_point_v<U>) return sizeof(U) == 8 ? "f64" : "f32";
  else if constexpr (std::is_same_v<U, bool>) return "u8";
  else if constexpr (std::is_integral_v<U>) {
    if constexpr (sizeof(U) == 8) return std::is_unsigned_v<U> ? "u64" : "i64";
    else if constexpr (sizeof(U) == 4) return std::is_unsigned_v<U> ? "u32" : "i32";
    else if constexpr (sizeof(U) == 2) return "i16";
    else return "u8";
  } else if constexpr (std::is_enum_v<U>) {
    return sizeof(U) == 8 ? "i64" : sizeof(U) == 4 ? "i32" : sizeof(U) == 2 ? "i16" : "u8";
  } else return "struct";
}
template <class K>
constexpr const char* access_of() {
  using U = std::remove_cv_t<K>;
  if constexpr (std::is_pointer_v<U>) return std::is_const_v<std::remove_pointer_t<U>> ? "r" : "rw";
  else return "";
}

template <class A> struct is_traced_array : std::false_type {};
template <class T, size_t N> struct is_traced_array<TracedArray<T, N>> : std::true_type { static constexpr size_t count = N; };
template <class A> struct is_sym_array : std::false_type {};
template <size_t N> struct is_sym_array<std::array<c10::SymInt, N>> : std::true_type { static constexpr size_t count = N; };
// element type of the kernel's array parameter (std::array<E, N> / at::detail::Array<E, N>)
template <class K> struct array_elem { using type = void; };
template <class E, size_t N> struct array_elem<std::array<E, N>> { using type = E; };
template <class E, int N> struct array_elem<at::detail::Array<E, N>> { using type = E; };
// the element type an array argument records: a TracedArray names it, a std::array<c10::SymInt, N> takes the kernel's
template <class D, class U, class = void> struct elem_of { using type = typename array_elem<U>::type; };
template <class D, class U> struct elem_of<D, U, std::void_t<typename D::element_type>> { using type = typename D::element_type; };

template <class E>
void write_elem(E& dst, int64_t v) {
  if constexpr (std::is_pointer_v<E>) dst = reinterpret_cast<E>(static_cast<uintptr_t>(v));
  else dst = static_cast<E>(v);
}

// Position i of the launch: write the K-typed value the kernel receives (ordinary: the value, trace: the
// hint) into `dst`; in trace mode (rec != nullptr) append the records at image offset `off`.
template <class K, class A>
void put_arg(K& dst, const A& a, size_t off, std::vector<FieldRec>* rec, const char* kname, size_t i) {
  using U = std::remove_cv_t<K>;
  using D = std::decay_t<A>;
  const bool tracing = rec != nullptr;
  // the value the kernel receives: the hint under a trace (Hints::of is private to this template), else the value
  auto int_value = [tracing](const c10::SymInt& v) { return tracing ? Hints::of(v) : v.expect_int(); };
  auto float_value = [tracing](const c10::SymFloat& v) { return tracing ? Hints::of(v) : v.expect_float(); };
  if constexpr (std::is_base_of_v<TracedBase, D>) {
    static_assert(std::is_class_v<U>, "at::cuda::host_trace::launch: a proxy struct passed where the kernel takes a scalar");
    const TracedBase& base = a;  // the generated proxies declare their own `P pod`; the base holds the void* view
    if (base.pod_size != sizeof(U)) {
      decline(c10::str("host_trace: launch of ", kname, ": parameter ", i, " is a ", sizeof(U),
                       "-byte struct but the proxy holds ", base.pod_size, " bytes (declined)"));
    }
    std::memcpy(static_cast<void*>(&dst), base.pod, sizeof(U));
    if (tracing) {
      for (const FieldRec& f : base.fields) {
        FieldRec g = f;
        g.offset += off;
        rec->push_back(std::move(g));
      }
    }
  } else if constexpr (is_traced_array<D>::value || is_sym_array<D>::value) {
    using E = typename elem_of<D, U>::type;
    static_assert(!std::is_void_v<E>, "at::cuda::host_trace::launch: a std::array<c10::SymInt, N> needs a std::array / at::detail::Array kernel parameter");
    constexpr size_t N = std::conditional_t<is_traced_array<D>::value, is_traced_array<D>, is_sym_array<D>>::count;
    static_assert(sizeof(U) == N * sizeof(E), "at::cuda::host_trace::launch: array argument size does not match the kernel's array parameter");
    E* out = reinterpret_cast<E*>(&dst);
    for (size_t k = 0; k < N; ++k) {
      write_elem<E>(out[k], int_value(a[k]));
      if (tracing) rec->push_back({off + k * sizeof(E), sizeof(E), kind_of<E>(), SymVal::of(a[k]), "", access_of<E>()});
    }
  } else if constexpr (std::is_pointer_v<U>) {
    if constexpr (std::is_same_v<D, std::nullptr_t>) {
      dst = nullptr;
      if (tracing) rec->push_back({off, sizeof(U), "ptr", SymVal::of_int(0), "", access_of<U>()});
    } else if constexpr (std::is_pointer_v<D>) {
      // a raw pointer is a constant of the variant (a traced tensor has none); e.g. a module-owned device table
      dst = const_cast<U>(reinterpret_cast<std::add_const_t<std::remove_pointer_t<U>>*>(a));
      if (tracing) rec->push_back({off, sizeof(U), "ptr", SymVal::of_int(static_cast<int64_t>(reinterpret_cast<uintptr_t>(a))), "", access_of<U>()});
    } else {
      const c10::SymInt v = a;  // sym_*_data_ptr(), a PtrField, an IntField ...
      write_elem<U>(dst, int_value(v));
      if (tracing) rec->push_back({off, sizeof(U), "ptr", SymVal::of(v), "", access_of<U>()});
    }
  } else if constexpr (std::is_floating_point_v<U>) {
    if constexpr (std::is_arithmetic_v<D>) {
      dst = static_cast<U>(a);
      if (tracing) rec->push_back({off, sizeof(U), kind_of<U>(), SymVal::of_float(static_cast<double>(a)), ""});
    } else {
      const c10::SymFloat v = a;  // SymFloat, a FloatField, a SymInt
      dst = static_cast<U>(float_value(v));
      if (tracing) rec->push_back({off, sizeof(U), kind_of<U>(), SymVal::of(v), ""});
    }
  } else if constexpr (std::is_integral_v<U> || std::is_enum_v<U>) {
    if constexpr (std::is_arithmetic_v<D> || std::is_enum_v<D>) {
      dst = static_cast<U>(a);
      if (tracing) rec->push_back({off, sizeof(U), kind_of<U>(), SymVal::of_int(static_cast<int64_t>(a)), ""});
    } else if constexpr (std::is_same_v<D, c10::SymBool>) {
      const bool b = tracing ? Hints::of(a) : a.guard_bool(__FILE__, __LINE__);
      dst = static_cast<U>(b);
      if (tracing) rec->push_back({off, sizeof(U), kind_of<U>(), SymVal::of(a), ""});
    } else {
      const c10::SymInt v = a;  // SymInt, an IntField, a PtrField
      dst = static_cast<U>(int_value(v));
      if (tracing) rec->push_back({off, sizeof(U), kind_of<U>(), SymVal::of(v), ""});
    }
  } else {
    // a plain struct by value (an empty functor, a TrivialOffsetCalculator): a constant of the variant,
    // its bytes (padding included) stay as captured, no record
    static_assert(std::is_convertible_v<const D&, U>, "at::cuda::host_trace::launch: argument type does not match the kernel's parameter");
    new (static_cast<void*>(&dst)) U(a);
  }
}

// Storage for one kernel parameter: raw bytes, so parameter structs without a default constructor
// (StridedOp holds an OffsetCalculator) can be written by memcpy / placement copy.
template <class K>
struct Slot {
  // zeroed: a plain struct's padding and empty members are never written by
  // its copy, and the image must be the same bytes at the trace and the build
  alignas(K) std::array<unsigned char, sizeof(K)> bytes{};
  K& ref() { return *reinterpret_cast<K*>(bytes.data()); }
};

template <class Tuple, class... Args, size_t... I>
void fill_ordinary(Tuple& slots, std::index_sequence<I...> /*seq*/, const Args&... args) {
  (put_arg(std::get<I>(slots).ref(), args, 0, nullptr, "", I), ...);
}

template <class Tuple, class... Args, size_t... I>
void fill_traced(Tuple& slots, const FuncInfo& info, LaunchPacket& pk, std::index_sequence<I...> /*seq*/, const Args&... args) {
  auto one = [&](auto idx, const auto& a) {
    constexpr size_t i = decltype(idx)::value;
    auto& slot = std::get<i>(slots);
    constexpr size_t ksize = std::tuple_size_v<decltype(slot.bytes)>;
    auto [off, size] = info.params[i];
    if (size != ksize) {
      decline(c10::str("host_trace: launch of ", info.name, ": parameter ", i, " is ", size,
                       " bytes in the driver's layout but ", ksize, " bytes in the signature the host named (declined)"));
    }
    put_arg(slot.ref(), a, off, &pk.params, info.name.c_str(), i);
    std::memcpy(pk.image.data() + off, slot.bytes.data(), ksize);
  };
  (one(std::integral_constant<size_t, I>{}, args), ...);
}

template <class Tuple, size_t... I>
void arg_pointers(Tuple& slots, void** ptrs, std::index_sequence<I...> /*seq*/) {
  ((ptrs[I] = static_cast<void*>(std::get<I>(slots).bytes.data())), ...);
}

}  // namespace launch_detail

template <class... KArgs, class G, class B, class... Args>
void launch(void (*kernel)(KArgs...), const G& grid_in, const B& block_in, const c10::SymInt& smem, cudaStream_t stream,
            const Args&... args) {
  static_assert(sizeof...(KArgs) == sizeof...(Args), "at::cuda::host_trace::launch: argument count does not match the kernel signature");
  using Tuple = std::tuple<launch_detail::Slot<std::remove_cv_t<KArgs>>...>;
  constexpr size_t n = sizeof...(KArgs);
  auto seq = std::make_index_sequence<n>{};
  Tuple vals;
  std::array<void*, (n > 0 ? n : 1)> ptrs{};
  launch_detail::arg_pointers(vals, ptrs.data(), seq);
  const Grid grid = launch_detail::to_grid(grid_in);
  const Block block = launch_detail::to_block(block_in);
  TraceState* s = active();
  if (s == nullptr) {
    launch_detail::fill_ordinary(vals, seq, args...);
    const dim3 g(static_cast<unsigned>(grid.x.expect_int()), static_cast<unsigned>(grid.y.expect_int()),
                 static_cast<unsigned>(grid.z.expect_int()));
    const dim3 b(static_cast<unsigned>(block.x.expect_int()), static_cast<unsigned>(block.y.expect_int()),
                 static_cast<unsigned>(block.z.expect_int()));
    C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), g, b, ptrs.data(), static_cast<size_t>(smem.expect_int()), stream));
    return;
  }
  require_capturing_stream(s, stream, "launch()");
  const FuncInfo& info = func_info(reinterpret_cast<const void*>(kernel));
  if (info.params.size() != n) {
    decline(c10::str("host_trace: launch of ", info.name, ": the driver reports ", info.params.size(),
                     " parameters, the signature the host named has ", n, " (declined)"));
  }
  LaunchPacket pk;
  pk.typed = true;
  pk.func = reinterpret_cast<const void*>(kernel);
  pk.kernel = info.name;
  pk.image.assign(info.image_size, 0);
  launch_detail::fill_traced(vals, info, pk, seq, args...);
  pk.grid = {grid.x, grid.y, grid.z};
  pk.block = {Hints::of(block.x), Hints::of(block.y), Hints::of(block.z)};
  pk.block_expr = {block.x, block.y, block.z};
  pk.smem = smem;
  pk.stream = stream;
  const dim3 g(static_cast<unsigned>(Hints::of(grid.x)), static_cast<unsigned>(Hints::of(grid.y)), static_cast<unsigned>(Hints::of(grid.z)));
  const dim3 b(static_cast<unsigned>(pk.block[0]), static_cast<unsigned>(pk.block[1]), static_cast<unsigned>(pk.block[2]));
  const size_t sm = static_cast<size_t>(Hints::of(smem));
  typed_launch(s, std::move(pk));
  C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), g, b, ptrs.data(), sm, stream));
  typed_launched(s, stream);
}

} // namespace at::cuda::host_trace
