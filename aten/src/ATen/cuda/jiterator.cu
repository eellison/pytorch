#include <ATen/jit_macros.h>

#if AT_USE_JITERATOR()

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/jiterator.h>

#include <cstring>
#include <deque>
#include <ATen/cuda/jiterator_impl.h>
#ifndef USE_ROCM
#include <ATen/cuda/host_trace/SymDynamic.h>
#endif

#include <iostream>
#include <utility>
namespace at {
namespace native {

#ifndef USE_ROCM
// the arguments launch_jitted_pwise_function takes from the launches below,
// their offset calculators, loader and storer built again from iter; the
// scalar argument, which a NoScalar kernel never reads, of no class (eager
// passes a float of any compute type), each extra argument its Scalar's
C10_NOINLINE void report_dynamic_launch(
    const at::cuda::jit::NvrtcFunction& fn, uint32_t grid, const TensorIteratorBase& iter, int64_t N,
    bool vectorized, bool contiguous, bool dynamic_casting, const c10::SmallVector<at::Scalar>& extra_args) {
  at::cuda::host_trace::LaunchLayout l{reinterpret_cast<cudaFunction_t>(fn.function), grid, num_threads(), {}, {}};
  auto param = [&](const auto& v) { report_param(l, v); };
  param(static_cast<int>(N));
  ArrayVariant(iter).visit(param);
  if (!vectorized) {
    if (contiguous) {
      TrivialOffsetCalculatorVariant(iter.ninputs()).visit(param);
      TrivialOffsetCalculatorVariant(iter.noutputs()).visit(param);
    } else {
      OffsetCalculatorVariant</*is_input=*/true>(iter).visit(param);
      OffsetCalculatorVariant</*is_input=*/false>(iter).visit(param);
    }
    if (dynamic_casting) {
      LoadWithCastVariant(iter).visit(param);
      StoreWithCastVariant(iter).visit(param);
    } else {
      param(memory::LoadWithoutCast());
      param(memory::StoreWithoutCast());
    }
  }
  const size_t scalar_size = c10::elementSize(toOpMathType(iter.common_dtype()));
  l.classes.emplace_back(scalar_size, '.');
  l.bytes.emplace_back(scalar_size, '\0');
  for (const auto& a : extra_args) {
    const size_t size = c10::elementSize(a.type());
    l.classes.emplace_back(size, 's');
    l.bytes.emplace_back(static_cast<const char*>(a.data_ptr()), size);
  }
  at::cuda::host_trace::report_launch_layout(std::move(l));
}
#endif

// A dynamic launch's parameters as the host kind holds them: addresses of the
// values for an eager launch (launch_jitted_pwise_function), descriptors for a
// symbolic iterator's recorded one (host_trace::SymHost)
template <class H>
struct DynParams {
  std::vector<const void*> ptrs;
  void add(const void* p) {
    ptrs.push_back(p);
  }
  void launch(const at::cuda::jit::NvrtcFunction& fn, uint32_t grid) {
    at::cuda::jit::launch_jitted_pwise_function(fn, ptrs.data(), {grid, 1u, 1u}, {num_threads(), 1u, 1u});
  }
};

#ifndef USE_ROCM
template <>
struct DynParams<at::cuda::host_trace::SymHost> {
  std::vector<at::cuda::host_trace::ArgDesc> descs;
  // bytes a descriptor points at that the params own (a zeroed loader's)
  std::deque<std::vector<uint8_t>> scratch;
  void add(const at::cuda::host_trace::ArgDesc& d) {
    descs.push_back(d);
  }
  void launch(const at::cuda::jit::NvrtcFunction& fn, const c10::SymInt& grid) {
    at::cuda::host_trace::record_function(reinterpret_cast<cudaFunction_t>(fn.function), descs.data(), descs.size(), grid, num_threads(), 0);
  }
};
#endif

// the operands' data parameter: the char* array, or their symbolic addresses
template <class H>
struct DynData;

template <>
struct DynData<EagerHost> {
  explicit DynData(const TensorIteratorBase& iter) : array(iter) {
    for (auto i = 0; i < iter.ntensors(); ++i) {
      pointers.push_back(static_cast<char*>(iter.data_ptr(i)));
    }
  }
  void add_to(DynParams<EagerHost>& p) {
    p.add(array.data_ptr());
  }
  ArrayVariant array;
  c10::SmallVector<char*, 16> pointers;
};

#ifndef USE_ROCM
template <>
struct DynData<at::cuda::host_trace::SymHost> {
  explicit DynData(const TensorIteratorBase& iter) {
    for (auto i = 0; i < iter.ntensors(); ++i) {
      pointers.push_back(at::cuda::host_trace::operand_address(iter, i));
    }
  }
  void add_to(DynParams<at::cuda::host_trace::SymHost>& p) {
    p.add(at::cuda::host_trace::pointers_desc(pointers.data(), pointers.size()));
  }
  c10::SmallVector<c10::SymInt, 16> pointers;
};
#endif

// a parameter that is a value's bytes (an offset calculator of no state, a
// loader, a storer, a scalar, an extra argument)
template <class H>
void add_bytes(DynParams<H>& p, const void* value, size_t size) {
#ifndef USE_ROCM
  if constexpr (H::symbolic) {
    p.add(at::cuda::host_trace::bytes_desc(value, size));
    return;
  } else
#endif
  {
    p.add(value);
  }
}

// the variant's held value as a bytes parameter
template <class H, class Variant>
void add_variant(DynParams<H>& p, Variant& v) {
#ifndef USE_ROCM
  if constexpr (H::symbolic) {
    v.visit([&](const auto& x) {
      using T = std::decay_t<decltype(x)>;
      const void* bytes = &x;
      if constexpr (requires { x.dtypes; x.element_sizes; }) {
        // a LoadWithCast / StoreWithCast: its members on zeroed bytes, so its
        // padding is the same at every trace (eager zeroes it while harvesting)
        auto& z = p.scratch.emplace_back(sizeof(T), 0);
        const T& r = *reinterpret_cast<const T*>(z.data());
        std::memcpy(z.data() + (reinterpret_cast<const char*>(&r.dtypes) - reinterpret_cast<const char*>(&r)), &x.dtypes, sizeof(x.dtypes));
        std::memcpy(z.data() + (reinterpret_cast<const char*>(&r.element_sizes) - reinterpret_cast<const char*>(&r)), &x.element_sizes, sizeof(x.element_sizes));
        bytes = z.data();
      } else if constexpr (std::is_empty_v<T>) {
        // an empty one's byte (a trivial offset calculator, a plain loader): zero
        bytes = p.scratch.emplace_back(sizeof(T), 0).data();
      }
      auto d = at::cuda::host_trace::bytes_desc(bytes, sizeof(T));
      d.align = alignof(T);
      p.add(d);
    });
    return;
  } else
#endif
  {
    p.add(v.data_ptr());
  }
}

// the count, data and (unrolled) offsets, loader and storer parameters are in
// p; the scalar (unused, 0) and extra arguments are appended here
template <class H>
static void add_scalar_and_extra_args(DynParams<H>& p, const float& scalar_val, const c10::SmallVector<at::Scalar>& extra_args) {
  add_bytes(p, &scalar_val, sizeof(float));
  for (const auto& a : extra_args) {
    add_bytes(p, a.data_ptr(), c10::elementSize(a.type()));
  }
}

template <class H>
static void add_count(DynParams<H>& p, const typename H::int_t& N) {
#ifndef USE_ROCM
  if constexpr (H::symbolic) {
    p.add(at::cuda::host_trace::int_desc(N));
    return;
  } else
#endif
  {
    p.add(&N);
  }
}

// each route's jitted kernels: one cache, whichever host kind (eager, or a
// CUDA graph host trace's symbolic one) asks for them
struct JitCache {
  std::mutex mutex;
  std::unordered_map<std::string, at::cuda::jit::NvrtcFunction> fns;
};

JitCache& vectorized_cache() {
  static JitCache cache;
  return cache;
}

JitCache& unrolled_cache() {
  static JitCache cache;
  return cache;
}

template <class H>
static inline void launch_jitted_vectorized_kernel_dynamic(
  const std::string& name, TensorIteratorBase& iter,
  DeviceIndex dev_idx, const typename H::int_t& N, const std::string& f, DynData<H>& data,
  const c10::SmallVector<at::Scalar>& extra_args, bool return_by_ref) {
  if constexpr (!H::symbolic) {
    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  }

  int nInputs = iter.ninputs();
  int nOutputs = iter.noutputs();
  const at::ScalarType common_dtype = iter.common_dtype();

  int tws = at::cuda::jit::calc_thread_work_size(nInputs, nOutputs, common_dtype, common_dtype);
  using pointer_t = typename decltype(data.pointers)::value_type;
  int vec_size = jitted_can_vectorize_up_to<pointer_t>(iter, data.pointers);

  int bws = tws * num_threads();
  const std::conditional_t<H::symbolic, c10::SymInt, uint32_t> grid = (N + bws - 1) / bws;

  bool vectorized = vec_size > 1;

  // Different kernels are compiled depending on what we're vectorizing up to (1, 2 or 4 elements)
  std::string f_inputs_type_str = at::cuda::jit::typeName(common_dtype);
  std::string compute_type_str = at::cuda::jit::typeName(toOpMathType(common_dtype));
  std::string result_type_str = at::cuda::jit::typeName(common_dtype);
  c10::SmallVector<std::string> extra_args_types = get_extra_args_typenames(extra_args);

  // The cache key includes all the parameters to generate_code + vec_size + dev_idx
  std::stringstream ss;
  ss << nInputs << '_' << nOutputs << f;
  ss << f_inputs_type_str << compute_type_str << result_type_str;
  ss << static_cast<int>(at::cuda::jit::BinaryFuncVariant::NoScalar);
  ss << extra_args_types;
  ss << vec_size;
  ss << static_cast<int>(dev_idx);
  const std::string cache_key = std::move(ss).str();

  auto& [_jiterator_mutex, fns] = vectorized_cache();
  at::cuda::jit::NvrtcFunction* fn_ptr = &fns[cache_key];

  if (!fn_ptr->function) {
    const std::lock_guard<std::mutex> lock{_jiterator_mutex};
    if (!fn_ptr->function) { // cache miss!
      // Generates program
      auto code = at::cuda::jit::generate_code(nInputs, nOutputs, f, name,
                                               f_inputs_type_str, compute_type_str, result_type_str,
                                               /*contiguous=*/true, /*dynamic_casting=*/false,
                                               at::cuda::jit::BinaryFuncVariant::NoScalar,
                                               extra_args_types,
                                               tws,
                                               vectorized, vec_size,
                                               return_by_ref);
      std::string kernel_name = vectorized ? name + "_vectorized" + std::to_string(vec_size) : name;
      // Acquires the program
      *fn_ptr = at::cuda::jit::jit_pwise_function(code, kernel_name);
    }
  }

  // size of `extra_args` is unknown at compile-time
  float scalar_val = 0;

#ifndef USE_ROCM
  if constexpr (!H::symbolic) {
    if (at::cuda::host_trace::harvesting()) {
      report_dynamic_launch(*fn_ptr, grid, iter, N, vectorized, /*contiguous=*/true, /*dynamic_casting=*/false, extra_args);
    }
  }
#endif
  DynParams<H> p;
  add_count(p, N);
  data.add_to(p);
  if (!vectorized) {
    TrivialOffsetCalculatorVariant input_offset_calculator(iter.ninputs());
    TrivialOffsetCalculatorVariant output_offset_calculator(iter.noutputs());
    auto l = memory::LoadWithoutCast();
    auto s = memory::StoreWithoutCast();
    add_variant(p, input_offset_calculator);
    add_variant(p, output_offset_calculator);
    add_bytes(p, &l, sizeof(l));
    add_bytes(p, &s, sizeof(s));
    add_scalar_and_extra_args(p, scalar_val, extra_args);
    p.launch(*fn_ptr, grid);
    return;
  }
  add_scalar_and_extra_args(p, scalar_val, extra_args);
  p.launch(*fn_ptr, grid);
}

// ic, oc, l, s: the offset calculators', loader's and storer's parameters, already in p after N and data
template <class H>
static inline void launch_jitted_unrolled_kernel_dynamic(
  const std::string& name, TensorIteratorBase& iter,
  DeviceIndex dev_idx, const typename H::int_t& N, const std::string& f, DynParams<H>& p,
  bool contiguous, bool dynamic_casting,
  const c10::SmallVector<at::Scalar>& extra_args, bool return_by_ref) {

  if constexpr (!H::symbolic) {
    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  }

  int nInputs = iter.ninputs();
  int nOutputs = iter.noutputs();
  const at::ScalarType common_dtype = iter.common_dtype();

  int tws = at::cuda::jit::calc_thread_work_size(nInputs, nOutputs, common_dtype, common_dtype);
  int bws = tws * num_threads();
  const std::conditional_t<H::symbolic, c10::SymInt, uint32_t> grid = (N + bws - 1) / bws;

  std::string f_inputs_type_str = at::cuda::jit::typeName(common_dtype);
  std::string compute_type_str = at::cuda::jit::typeName(toOpMathType(common_dtype));
  std::string result_type_str = at::cuda::jit::typeName(common_dtype);
  c10::SmallVector<std::string> extra_args_types = get_extra_args_typenames(extra_args);

  // The cache key includes all the parameters to generate_code + dev_idx
  std::stringstream ss;
  ss << nInputs << '_' << nOutputs << f;
  ss << f_inputs_type_str << compute_type_str << result_type_str;
  ss << contiguous << dynamic_casting;
  ss << static_cast<int>(at::cuda::jit::BinaryFuncVariant::NoScalar);
  ss << extra_args_types;
  ss << dev_idx;
  const std::string cache_key = std::move(ss).str();

  auto& [_jiterator_mutex, fns] = unrolled_cache();

  at::cuda::jit::NvrtcFunction* fn_ptr = &fns[cache_key];
  if (!fn_ptr->function) {
    const std::lock_guard<std::mutex> lock{_jiterator_mutex};
    if (!fn_ptr->function) {
      auto code = at::cuda::jit::generate_code(nInputs, nOutputs, f, name,
                                               f_inputs_type_str, compute_type_str, result_type_str,
                                               contiguous, dynamic_casting,
                                               at::cuda::jit::BinaryFuncVariant::NoScalar,
                                               extra_args_types, tws, /*vectorized*/false, /*vec_size*/0, return_by_ref);
      *fn_ptr = at::cuda::jit::jit_pwise_function(code, name);
    }
  }

  float scalar_val = 0;

#ifndef USE_ROCM
  if constexpr (!H::symbolic) {
    if (at::cuda::host_trace::harvesting()) {
      report_dynamic_launch(*fn_ptr, grid, iter, N, /*vectorized=*/false, contiguous, dynamic_casting, extra_args);
    }
  }
#endif
  add_scalar_and_extra_args(p, scalar_val, extra_args);
  p.launch(*fn_ptr, grid);
}

// the input or output offset calculator: make_unique_offset_calculator's, or the symbolic iterator's
template <class H, bool IS_INPUT>
struct DynOffsets {
  explicit DynOffsets(const TensorIteratorBase& iter)
      : v(iter) {}
  void add_to(DynParams<H>& p) {
    add_variant(p, v);
  }
  OffsetCalculatorVariant<IS_INPUT> v;
};

#ifndef USE_ROCM
template <bool IS_INPUT>
struct DynOffsets<at::cuda::host_trace::SymHost, IS_INPUT> {
  explicit DynOffsets(const TensorIteratorBase& iter)
      : n(IS_INPUT ? iter.ninputs() : iter.noutputs()),
        offsets(at::cuda::host_trace::sym_offsets(iter, IS_INPUT ? iter.noutputs() : 0, n, /*element_strides=*/true)) {}
  void add_to(DynParams<at::cuda::host_trace::SymHost>& p) {
    p.add(at::cuda::host_trace::offset_calculator_desc(n, offsets));
  }
  int n;
  at::cuda::host_trace::SymOffsets offsets;
};
#endif

template <class H>
static void jitted_gpu_kernel_dynamic_impl(
    const std::string& kernel_name,
    TensorIteratorBase& iter,
    const std::string& f,
    const bool dynamic_casting,
    const c10::SmallVector<at::Scalar>& extra_args,
    bool return_by_ref) {

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.noutputs() <= 8);
  TORCH_INTERNAL_ASSERT(iter.ninputs() <= 8);

  DynData<H> data(iter);

  typename H::int_t numel = H::numel(iter);
  bool contiguous = iter.is_contiguous();

  // Decides which of 4 kernel types to launch
  // Variations are:
  //   - Case 1: no dynamic casting and contiguous
  //   - Case 2: no dynamic casting and noncontiguous
  //   - Case 3: dynamic casting and contiguous
  //   - Case 4: dynamic casting and noncontiguous
  // These cases align with the non-jitted CUDALoops.cuh cases in gpu_kernel_impl

  if (!dynamic_casting) {
    if (contiguous) {
      // Case 1: no dynamic casting and contiguous
      launch_jitted_vectorized_kernel_dynamic<H>(kernel_name, iter,
         iter.device().index(), numel, f, data, extra_args, return_by_ref);
      return;
    }

    // Case 2: no dynamic casting and noncontiguous
    DynOffsets<H, /*is_input=*/true> input_offset_calculator(iter);
    DynOffsets<H, /*is_input=*/false> output_offset_calculator(iter);
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    DynParams<H> p;
    add_count(p, numel);
    data.add_to(p);
    input_offset_calculator.add_to(p);
    output_offset_calculator.add_to(p);
    add_bytes(p, &loader, sizeof(loader));
    add_bytes(p, &storer, sizeof(storer));

    launch_jitted_unrolled_kernel_dynamic<H>(
      kernel_name, iter, iter.device().index(), numel, f, p, contiguous, dynamic_casting, extra_args, return_by_ref);

    return;
  }

  // Cases 3 and 4 are handled below
  // Both require construction of one or more storers and loaders

  // Creates load casts from inputs (note offset indexing into the iterators noutpus...n tensors)
  LoadWithCastVariant loader(iter);

  // Creates store cast to output (the 0...noutpus-1 tensor in TensorIterator)
  StoreWithCastVariant storer(iter);

  DynParams<H> p;
  add_count(p, numel);
  data.add_to(p);
  if (contiguous) {
    // Case 3: dynamic casting and contiguous
    TrivialOffsetCalculatorVariant input_offset_calculator(iter.ninputs());
    TrivialOffsetCalculatorVariant output_offset_calculator(iter.noutputs());
    add_variant(p, input_offset_calculator);
    add_variant(p, output_offset_calculator);
    add_variant(p, loader);
    add_variant(p, storer);

    launch_jitted_unrolled_kernel_dynamic<H>(
      kernel_name, iter, iter.device().index(), numel, f, p, contiguous, dynamic_casting, extra_args, return_by_ref);
    return;
  }

  // Case 4: dynamic casting and noncontiguous
  DynOffsets<H, /*is_input=*/true> input_offset_calculator(iter);
  DynOffsets<H, /*is_input=*/false> output_offset_calculator(iter);
  input_offset_calculator.add_to(p);
  output_offset_calculator.add_to(p);
  add_variant(p, loader);
  add_variant(p, storer);

  launch_jitted_unrolled_kernel_dynamic<H>(
      kernel_name, iter, iter.device().index(), numel, f, p, contiguous, dynamic_casting, extra_args, return_by_ref);
}

// Entrypoint for dynamic version of jitted GPU kernels, which accepts dynamic number of inputs
// and arbitrary types of input and extra args. This dynamic version is needed for jiterator with python interface,
// since the kernel definition is unknown at the compilation time.
// Similarly, launch_jitted_vectorized_kernel_dynamic and launch_jitted_unrolled_kernel_dynamic are created
// to handle arbitrary functions defined in python user code.
// For templated version, see note [Jiterator] in JitLoops.cuh for more details
static void jitted_gpu_kernel_dynamic(
    const std::string& kernel_name,
    TensorIteratorBase& iter,
    const std::string& f,
    const c10::SmallVector<at::Scalar>& extra_args,
    bool return_by_ref) {

  // TODO: much of preamble is common to both jitted_gpu_kernel and gpu_kernel
  //   Maybe it could be refactored?
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_cuda(),
      "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }

  // a symbolic iterator (a host trace): an empty one is a guard
  if (C10_UNLIKELY(iter.is_symbolic()) ? iter.sym_numel() == 0 : iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      jitted_gpu_kernel_dynamic(kernel_name, sub_iter, f, extra_args, return_by_ref);
    }
    return;
  }

  // Computes if dynamic casting is needed
  // Dynamic casting is needed if an input's or output's dtype differs from the common dtype
  bool needs_dynamic_casting = false;
  const at::ScalarType common_dtype = iter.common_dtype();
  for (auto i = 0; i < iter.ntensors(); ++i) {
    if (iter.dtype(i) != common_dtype) {
      needs_dynamic_casting = true;
      break;
    }
  }

#ifndef USE_ROCM
  if (C10_UNLIKELY(iter.is_symbolic())) {
    return jitted_gpu_kernel_dynamic_impl<at::cuda::host_trace::SymHost>(kernel_name, iter, f, needs_dynamic_casting, extra_args, return_by_ref);
  }
#endif
  jitted_gpu_kernel_dynamic_impl<EagerHost>(kernel_name, iter, f, needs_dynamic_casting, extra_args, return_by_ref);
}

} // namespace native

namespace cuda {

c10::SmallVector<at::Tensor> CompileAndLaunchKernel(
  const std::string& code_string,
  const std::string& kernel_name,
  const int num_outputs,
  const c10::SmallVector<at::Tensor>& tensors,
  const c10::SmallVector<at::Scalar>& extra_args,
  bool return_by_ref) {

  c10::SmallVector<at::Tensor> outs(num_outputs);
  TensorIteratorConfig config;
  config
    .set_check_mem_overlap(true)
    .allow_cpu_scalars(false)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true)
    .check_all_same_device(true);
  for (int i = 0; i < num_outputs; ++i) {
    config.add_owned_output(outs[i]);
  }
  for (const auto& t: tensors) {
    config.add_const_input(t);
  }
  TensorIterator iter = config.build();

  CUDAGuard guard(iter.device());
  at::native::jitted_gpu_kernel_dynamic(kernel_name, iter, code_string, extra_args, return_by_ref);

  c10::SmallVector<at::Tensor> outputs;
  if (num_outputs > 0) {
    outputs.reserve(num_outputs);
  }
  for (int i = 0; i < num_outputs; ++i) {
    outputs.emplace_back(iter.output(i));
  }

  return outputs;
}

}} // namespace at::cuda

#endif // AT_USE_JITERATOR()
