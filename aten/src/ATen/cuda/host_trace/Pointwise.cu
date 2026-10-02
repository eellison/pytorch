// A pointwise op's traced host from eager's own launch: the kernel node of a
// capture of the op at the hints gives the kernel and its parameter bytes (the
// functor's among them, whatever the op's), and CUDALoops.cuh's route over a
// TensorIteratorSym gives the sizes, strides and addresses in them.
#if !defined(USE_ROCM)
#include <ATen/cuda/host_trace/LoopsSym.cuh>
#include <ATen/cuda/host_trace/Ops.h>

#include <algorithm>
#include <string>

namespace at::cuda::host_trace {

namespace {

// a functor for StridedOp's layout, which ends with its functor
struct NoFunctor {
  __device__ int operator()() const {
    return 0;
  }
};

// p's bytes and fields in [lo, hi) onto parameter `param` of r
template <class K>
void overlay(KernelRecord& r, size_t param, const Param<K>& p, size_t lo = 0, size_t hi = sizeof(K)) {
  auto& image = r.params[param];
  if (image.size() < hi) {
    decline(c10::str("kernel parameter ", param, " is ", image.size(), " bytes, not ", hi));
  }
  std::copy(p.bytes.begin() + lo, p.bytes.begin() + hi, image.begin() + lo);
  for (Field f : p.fields) {
    if (f.offset >= lo && f.offset < hi) {
      f.param = param;
      r.fields.push_back(std::move(f));
    }
  }
}

// memory::can_vectorize_up_to of an operand, at most up_to
int vectorize_up_to(const c10::SymInt& address, int64_t element_size, int up_to) {
  for (int vec = up_to; vec > 1; vec /= 2) {
    if (address % (element_size * vec) == 0) {
      return vec;
    }
  }
  return 1;
}

bool contains(std::string_view name, std::string_view part) {
  return name.find(part) != std::string_view::npos;
}

template <int N>
void pointwise_launch(Recorder& rec, const TensorIteratorSym& iter, std::string_view name, KernelRecord r) {
  std::array<c10::SymInt, N> data;
  for (const auto i : c10::irange(N)) {
    data[i] = iter.data_ptr(i);
  }
  const c10::SymInt n = iter.numel();
  const bool cast = contains(name, "13StridedCastOp") || contains(name, "12LoadWithCast");
  constexpr int64_t threads = num_threads();
  Param<std::array<char*, N>> ptrs;
  for (const auto i : c10::irange(N)) {
    ptrs.set(ptrs.value()[i], data[i]);
  }
  std::string expected;
  if (iter.is_contiguous()) {
    int vec = 1;
    int64_t io_size = 0;
    for (const auto i : c10::irange(N)) {
      io_size += iter.element_size(i);
    }
    if (!cast) {
      // launch_vectorized_kernel's
      const cudaDeviceProp* p = at::cuda::getDeviceProperties(iter.device().index());
      constexpr int64_t min_sm107_io_size = 16 * 1024 * 1024;
      if (p->major == 10 && p->minor == 7 && n * io_size >= min_sm107_io_size) {
        decline("the sm_107 work size");
      }
      vec = std::min<int>(16 / iter.element_size(0), 8);
      if (p->major != 9 && p->major != 10) {
        vec = std::min(vec, 4);
      }
#if !defined(CUDA_VERSION) || CUDA_VERSION < 12080
      if (iter.element_size(0) < 2) {
        vec = std::min(vec, 4);
      }
#endif
      for (const auto i : c10::irange(N)) {
        vec = vectorize_up_to(data[i], iter.element_size(i), vec);
      }
    }
    int64_t bws = at::native::elementwise_block_work_size();
    if (vec > 1) {
      expected = c10::str("_ZN2at6native29vectorized_elementwise_kernelILi", vec, "E");
      bws = (io_size == 1 ? 16 : 8) * threads;
    } else {
      expected = "_ZN2at6native27unrolled_elementwise_kernelI";
      if (!contains(name, cast ? "12LoadWithCast" : "15LoadWithoutCast")) {
        decline(c10::str("kernel ", name, " is not the unrolled kernel's"));
      }
    }
    if (r.params.size() != (vec > 1 ? 3u : 7u)) {
      decline(c10::str("kernel ", name, " has ", r.params.size(), " parameters"));
    }
    overlay(r, 0, scalar_param<int>(n));
    overlay(r, 2, ptrs);
    r.grid = (n + bws - 1) / bws;
  } else {
    // launch_legacy_kernel<128, unroll> of a StridedOp or StridedCastOp
    const int unroll = cast || iter.element_size(0) < 4 ? 4 : 2;
    expected = c10::str("_ZN2at6native18elementwise_kernelILi128ELi", unroll, "ENS0_", cast ? "13StridedCastOpI" : "9StridedOpI");
    if (r.params.size() != 2) {
      decline(c10::str("kernel ", name, " has ", r.params.size(), " parameters"));
    }
    auto place = [&](auto op) {
      auto& body = op.value();
      for (const auto i : c10::irange(N)) {
        op.set(body.data[i], data[i]);
      }
      set_offset_calculator(rec, op, body.offset_calc, iter);
      // the members the launch sets; the node's dtypes and padding stay
      const auto* base = op.bytes.data();
      const auto* calc = reinterpret_cast<const uint8_t*>(&body.offset_calc);
      overlay(r, 1, op, 0, sizeof(body.data));
      overlay(r, 1, op, calc - base, calc - base + sizeof(body.offset_calc));
    };
    if (cast) {
      place(Param<at::native::StridedCastOp<NoFunctor, N>>());
    } else {
      place(Param<at::native::StridedOp<NoFunctor, N>>());
    }
    overlay(r, 0, scalar_param<int>(n));
    const int64_t work = 128 * unroll;
    r.grid = (n + work - 1) / work;
  }
  if (name.substr(0, expected.size()) != expected) {
    decline(c10::str("kernel ", name, " is not the iterator's route, ", expected));
  }
  r.block = threads;
  r.smem = 0;
  rec.launches.push_back(std::move(r));
}

// jitted_gpu_kernel_generic's routes (CUDAJitLoops.cuh), or with `dynamic` a
// user jiterator's (jiterator.cu), to NVRTC's extern "C"
// <op>_vectorized<vec>_kernel or <op>_kernel; the node's loader tells the
// dynamic-cast route, compute_dtype is the kernel's input type there
template <int NOUT, int NIN>
void jitted_launch(Recorder& rec, const TensorIteratorSym& iter, c10::ScalarType compute_dtype, bool dynamic, std::string_view name, KernelRecord r) {
  constexpr int N = NOUT + NIN;
  const bool vectorized = contains(name, "_vectorized");
  if (!vectorized && r.params.size() < 7) {
    decline(c10::str("kernel ", name, " has ", r.params.size(), " parameters"));
  }
  const bool cast = !vectorized && r.params[4].size() > 1;
  if (cast && compute_dtype == c10::ScalarType::Undefined) {
    decline("a jiterator dynamic-cast kernel of an iterator without a common dtype");
  }
  const int64_t in_size = cast ? c10::elementSize(compute_dtype) : iter.element_size(NOUT);
  const int64_t out_size = cast ? in_size : iter.element_size(0);
  if (cast && !dynamic && (in_size == 1 || iter.element_size(0) == 1)) {
    decline("a jiterator dynamic-cast kernel of a 1-byte type");
  }
  std::array<c10::SymInt, N> data;
  Param<std::array<char*, N>> ptrs;
  for (const auto i : c10::irange(N)) {
    data[i] = iter.data_ptr(i);
    ptrs.set(ptrs.value()[i], data[i]);
  }
  const c10::SymInt n = iter.numel();
  const int64_t bws = (std::min(in_size, out_size) == 1 ? 16 : 8) * num_threads();
  std::string suffix = "_kernel";
  if (iter.is_contiguous() && !cast) {
    // a user jiterator's vector size is jitted_can_vectorize_up_to's alone
    int vec = dynamic ? 8 : std::min<int>(16 / in_size, in_size < 2 ? 4 : 8);
    for (const auto i : c10::irange(N)) {
      vec = vectorize_up_to(data[i], i < NOUT ? out_size : in_size, vec);
    }
    if (vec > 1) {
      suffix = c10::str("_vectorized", vec, "_kernel");
    }
  } else if (!iter.is_contiguous()) {
    // make_input_offset_calculator's and make_output_offset_calculator's, element strides
    std::array<std::vector<c10::SymInt>, N> strides;
    std::array<const c10::SymInt*, N> ptr;
    for (const auto i : c10::irange(N)) {
      for (const auto& st : iter.strides(i)) {
        strides[i].push_back(st / iter.element_size(i));
      }
      ptr[i] = strides[i].data();
    }
    Param<::OffsetCalculator<NIN>> ic;
    set_offset_calculator(rec, ic, ic.value(), iter.ndim(), iter.shape().data(), ptr.data() + NOUT);
    Param<::OffsetCalculator<NOUT>> oc;
    set_offset_calculator(rec, oc, oc.value(), iter.ndim(), iter.shape().data(), ptr.data());
    overlay(r, 2, ic);
    overlay(r, 3, oc);
  }
  if (!name.ends_with(suffix) || (suffix == "_kernel" && vectorized)) {
    decline(c10::str("kernel ", name, " is not the iterator's jiterator route, *", suffix));
  }
  overlay(r, 0, scalar_param<int>(n));
  overlay(r, 1, ptrs);
  r.grid = (n + bws - 1) / bws;
  r.block = num_threads();
  r.smem = 0;
  rec.launches.push_back(std::move(r));
}

template <int NOUT>
void jitted(Recorder& rec, const TensorIteratorSym& iter, c10::ScalarType compute_dtype, bool dynamic, std::string_view name, KernelRecord r) {
  switch (iter.ninputs()) {
    case 1:
      return jitted_launch<NOUT, 1>(rec, iter, compute_dtype, dynamic, name, std::move(r));
    case 2:
      return jitted_launch<NOUT, 2>(rec, iter, compute_dtype, dynamic, name, std::move(r));
    case 3:
      return jitted_launch<NOUT, 3>(rec, iter, compute_dtype, dynamic, name, std::move(r));
    case 4:
      return jitted_launch<NOUT, 4>(rec, iter, compute_dtype, dynamic, name, std::move(r));
    case 5:
      return jitted_launch<NOUT, 5>(rec, iter, compute_dtype, dynamic, name, std::move(r));
    case 6:
      return jitted_launch<NOUT, 6>(rec, iter, compute_dtype, dynamic, name, std::move(r));
    case 7:
      return jitted_launch<NOUT, 7>(rec, iter, compute_dtype, dynamic, name, std::move(r));
    case 8:
      return jitted_launch<NOUT, 8>(rec, iter, compute_dtype, dynamic, name, std::move(r));
    default:
      decline(c10::str("a jiterator kernel of ", iter.ninputs(), " inputs"));
  }
}

// gpu_kernel_multiple_outputs's unrolled kernel, whose offset calculators
// are trivial for a contiguous iterator
template <int NOUT, int NIN>
void multiple_outputs_launch(Recorder& rec, const TensorIteratorSym& iter, std::string_view name, KernelRecord r) {
  constexpr int N = NOUT + NIN;
  if (!contains(name, c10::str("unrolled_elementwise_kernel_for_multi_outputsILi", NOUT, "E")) || contains(name, "TrivialOffsetCalculator") != iter.is_contiguous()) {
    decline(c10::str("kernel ", name, " is not the iterator's route, gpu_kernel_multiple_outputs's"));
  }
  if (r.params.size() != 5) {
    decline(c10::str("kernel ", name, " has ", r.params.size(), " parameters"));
  }
  Param<std::array<char*, N>> ptrs;
  std::array<std::vector<c10::SymInt>, N> strides;
  std::array<const c10::SymInt*, N> ptr;
  for (const auto i : c10::irange(N)) {
    ptrs.set(ptrs.value()[i], iter.data_ptr(i));
    for (const auto& st : iter.strides(i)) {
      strides[i].push_back(st / iter.element_size(i));
    }
    ptr[i] = strides[i].data();
  }
  if (!iter.is_contiguous()) {
    // make_input_offset_calculator's and make_output_offset_calculator's, element strides
    Param<::OffsetCalculator<NIN>> ic;
    set_offset_calculator(rec, ic, ic.value(), iter.ndim(), iter.shape().data(), ptr.data() + NOUT);
    Param<::OffsetCalculator<NOUT>> oc;
    set_offset_calculator(rec, oc, oc.value(), iter.ndim(), iter.shape().data(), ptr.data());
    overlay(r, 3, ic);
    overlay(r, 4, oc);
  }
  const c10::SymInt n = iter.numel();
  overlay(r, 0, scalar_param<int>(n));
  overlay(r, 2, ptrs);
  r.grid = (n + ::block_work_size() - 1) / ::block_work_size();
  r.block = num_threads();
  r.smem = 0;
  rec.launches.push_back(std::move(r));
}

template <int NOUT>
void multiple_outputs(Recorder& rec, const TensorIteratorSym& iter, std::string_view name, KernelRecord r) {
  switch (iter.ninputs()) {
    case 1:
      return multiple_outputs_launch<NOUT, 1>(rec, iter, name, std::move(r));
    case 2:
      return multiple_outputs_launch<NOUT, 2>(rec, iter, name, std::move(r));
    case 3:
      return multiple_outputs_launch<NOUT, 3>(rec, iter, name, std::move(r));
    case 4:
      return multiple_outputs_launch<NOUT, 4>(rec, iter, name, std::move(r));
    default:
      decline(c10::str("a pointwise op of ", NOUT, " outputs and ", iter.ninputs(), " inputs"));
  }
}

} // namespace

std::vector<TensorBase> pointwise(Recorder& rec, c10::ArrayRef<TensorBase> outs, c10::ArrayRef<c10::ScalarType> out_dtypes, c10::ArrayRef<TensorBase> inputs, c10::ScalarType compute_dtype, bool dynamic, std::string_view name, KernelRecord node) {
  auto iter = TensorIteratorSym::pointwise_op(rec, outs, out_dtypes, inputs);
  std::vector<TensorBase> outputs;
  for (const auto i : c10::irange(iter.noutputs())) {
    outputs.push_back(iter.output(i));
  }
  if (iter.numel() == 0) {
    if (!name.empty()) {
      decline(c10::str("kernel ", name, " of an empty pointwise op"));
    }
    return outputs;
  }
  if (name.empty()) {
    decline("no launch of a nonempty pointwise op");
  }
  if (!iter.can_use_32bit_indexing()) {
    decline("an iterator beyond 32-bit indexing");
  }
  if (name.substr(0, 2) != "_Z") {
    switch (iter.noutputs()) {
      case 1:
        jitted<1>(rec, iter, compute_dtype, dynamic, name, std::move(node));
        break;
      case 2:
        jitted<2>(rec, iter, compute_dtype, dynamic, name, std::move(node));
        break;
      case 3:
        jitted<3>(rec, iter, compute_dtype, dynamic, name, std::move(node));
        break;
      case 4:
        jitted<4>(rec, iter, compute_dtype, dynamic, name, std::move(node));
        break;
      case 5:
        jitted<5>(rec, iter, compute_dtype, dynamic, name, std::move(node));
        break;
      case 6:
        jitted<6>(rec, iter, compute_dtype, dynamic, name, std::move(node));
        break;
      case 7:
        jitted<7>(rec, iter, compute_dtype, dynamic, name, std::move(node));
        break;
      case 8:
        jitted<8>(rec, iter, compute_dtype, dynamic, name, std::move(node));
        break;
      default:
        decline(c10::str("a jiterator kernel of ", iter.noutputs(), " outputs"));
    }
    return outputs;
  }
  if (iter.noutputs() == 2) {
    multiple_outputs<2>(rec, iter, name, std::move(node));
    return outputs;
  }
  if (iter.noutputs() == 3) {
    multiple_outputs<3>(rec, iter, name, std::move(node));
    return outputs;
  }
  if (iter.noutputs() != 1) {
    decline(c10::str("a pointwise op of ", iter.noutputs(), " outputs and ", iter.ninputs(), " inputs"));
  }
  switch (iter.ntensors()) {
    case 1:
      pointwise_launch<1>(rec, iter, name, std::move(node));
      break;
    case 2:
      pointwise_launch<2>(rec, iter, name, std::move(node));
      break;
    case 3:
      pointwise_launch<3>(rec, iter, name, std::move(node));
      break;
    case 4:
      pointwise_launch<4>(rec, iter, name, std::move(node));
      break;
    default:
      decline(c10::str("a pointwise op of ", iter.ntensors(), " operands"));
  }
  return outputs;
}

} // namespace at::cuda::host_trace
#endif
