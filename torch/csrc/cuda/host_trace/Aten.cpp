#include <torch/csrc/cuda/host_trace/Variant.h>

#if !defined(USE_ROCM)
#include <ATen/TensorIterator.h>
#include <ATen/cuda/host_trace/Ops.h>
#include <ATen/cuda/host_trace/SymHost.h>
#include <ATen/native/Copy.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

namespace torch::cuda::host_trace {

namespace {

namespace ht = at::cuda::host_trace;

// a traced tensor's data_ptr() is symbolic, so ask Python. An op redispatched
// from Python (_cuda_hostTraceRedispatch) runs its host code without the GIL
struct PyRecorder : ht::Recorder {
  c10::SymInt data_ptr(const at::TensorBase& t) override {
    py::gil_scoped_acquire gil;
    auto obj = py::reinterpret_steal<py::object>(THPVariable_Wrap(t));
    return obj.attr("data_ptr")().cast<c10::SymInt>();
  }
  c10::SymInt bit_length(const c10::SymInt& x) override {
    py::gil_scoped_acquire gil;
    return host_trace.attr("bit_length")(x).cast<c10::SymInt>();
  }
  c10::SymInt pow2(const c10::SymInt& x) override {
    py::gil_scoped_acquire gil;
    return py::cast(x).attr("__rpow__")(2).cast<c10::SymInt>();
  }
  c10::SymInt f32_div(const c10::SymInt& a, const c10::SymInt& b) override {
    py::gil_scoped_acquire gil;
    return host_trace.attr("f32_div")(a, b).cast<c10::SymInt>();
  }
  c10::SymInt select(const c10::SymBool& c, const c10::SymInt& a, const c10::SymInt& b) override {
    py::gil_scoped_acquire gil;
    return host_trace.attr("select")(c, a, b).cast<c10::SymInt>();
  }
  py::module host_trace = py::module::import("torch.cuda._host_trace");
};

// a launch as (function, offsets, params, fields, grid, block, smem[,
// cpu_scalars]) or, a memcpy, (dst, src, bytes); each field (param, offset,
// width, value, is_pointer), each CPU scalar (param, offset, class, source)
py::tuple record(const std::variant<ht::KernelRecord, ht::MemcpyRecord>& launch);

// [record(launch)...]
py::list records(const PyRecorder& rec) {
  py::list launches;
  for (const auto& launch : rec.launches) {
    launches.append(record(launch));
  }
  return launches;
}

py::tuple record(const std::variant<ht::KernelRecord, ht::MemcpyRecord>& launch) {
  if (const auto* m = std::get_if<ht::MemcpyRecord>(&launch)) {
    return py::make_tuple(m->dst, m->src, m->bytes);
  }
  const auto& r = std::get<ht::KernelRecord>(launch);
  py::list params;
  for (const auto& p : r.params) {
    params.append(py::bytes(reinterpret_cast<const char*>(p.data()), p.size()));
  }
  py::list fields;
  for (const auto& f : r.fields) {
    fields.append(py::make_tuple(f.param, f.offset, f.width, f.value, f.pointer));
  }
  auto out = py::make_tuple(
      reinterpret_cast<uintptr_t>(r.function),
      r.offsets,
      params,
      fields,
      py::make_tuple(r.grid.x, r.grid.y, r.grid.z),
      py::make_tuple(r.block.x, r.block.y, r.block.z),
      r.smem);
  if (r.cpu_scalars.empty()) {
    return out;
  }
  py::list scalars;
  for (const auto& c : r.cpu_scalars) {
    scalars.append(py::make_tuple(c.param, c.offset, std::string(1, c.cls), at::Tensor(c.source)));
  }
  return py::make_tuple(out[0], out[1], out[2], out[3], out[4], out[5], out[6], py::tuple(scalars));
}

// hands each launch to `sink` as it is recorded, so the launches of the op's
// nested ops (which reach the trace in between) keep their order
struct StreamingRecorder : PyRecorder {
  explicit StreamingRecorder(py::function s) : sink(std::move(s)) {}
  void on_launch() override {
    py::gil_scoped_acquire gil;
    for (const auto& launch : launches) {
      sink(record(launch));
    }
    launches.clear();
  }
  py::function sink;
};

// at::impl::symbolic_resize's hook while a redispatch runs: the trace
// re-allocates a fresh empty output of the running op (_symbolic_resize)
bool python_symbolic_resize(const at::TensorBase& out, c10::SymIntArrayRef sizes, c10::SymIntArrayRef strides) {
  py::gil_scoped_acquire gil;
  // leaked on purpose: a static py::object would be decref'd by an exit handler after
  // Py_Finalize, without the GIL (fatal in module_dealloc when it is the last reference)
  static const py::handle tape = py::module::import("torch.cuda._host_trace_tape").release();
  const std::vector<c10::SymInt> s(sizes.begin(), sizes.end());
  const std::vector<c10::SymInt> t(strides.begin(), strides.end());
  return tape.attr("_symbolic_resize")(at::Tensor(out), s, t).cast<bool>();
}

bool python_same_start(const at::TensorBase& a, const at::TensorBase& b) {
  py::gil_scoped_acquire gil;
  static const py::handle tape = py::module::import("torch.cuda._host_trace_tape").release();
  return tape.attr("_same_start")(at::Tensor(a), at::Tensor(b)).cast<bool>();
}

// the trace's hooks for host code over its tensors: a fresh output's resize, copy_impl's same-data test
struct ResizeHook {
  ResizeHook()
      : previous(at::impl::set_symbolic_resize(&python_symbolic_resize)), previous_same(at::native::set_symbolic_same_start(&python_same_start)) {}
  ~ResizeHook() {
    at::impl::set_symbolic_resize(previous);
    at::native::set_symbolic_same_start(previous_same);
  }
  at::impl::SymbolicResize previous;
  at::native::SymbolicSameStart previous_same;
};

// While a traced tensor's storage is sealed, host code reading its data
// pointer raises: its address is symbolic, only the recorder gives it
void seal(const at::Tensor& t, bool on) {
  auto* storage = t.unsafeGetTensorImpl()->unsafe_storage().unsafeGetStorageImpl();
  if (storage == nullptr) {
    return;
  }
  if (on) {
    storage->release_data_and_set_meta_custom_data_ptr_error_msg_(
        "host_trace: host code read a traced tensor's data pointer, which is symbolic: the op's host code is not converted");
  } else {
    storage->clear_data_ptr_access_error_msg_();
  }
}

// (out, records)
py::tuple traced(const PyRecorder& rec, const at::TensorBase& out) {
  return py::make_tuple(at::Tensor(out), records(rec));
}

// a multi-output op's ((outs...), records)
template <class... T>
py::tuple traced(const PyRecorder& rec, const std::tuple<T...>& outs) {
  auto tensors = std::apply([](const auto&... o) { return py::make_tuple(at::Tensor(o)...); }, outs);
  return py::make_tuple(tensors, records(rec));
}

} // namespace

// Each op's traced host under the caller's dispatch mode; torch/cuda/_host_trace_tape.py
// maps an op's schema-ordered args to these
void initHostTraceAtenBindings(py::module& m) {
  // fn() with symbolic TensorIterators on and the launches of their kernels
  // recorded (ATen/cuda/host_trace/SymHost.h), each handed to sink as a
  // record() tuple: an op's own CUDA host code, redispatched by the trace
  m.def("_cuda_hostTraceRedispatch", [](const py::function& fn, py::function sink) {
    StreamingRecorder rec(std::move(sink));
    ht::RecorderGuard guard(&rec);
    ResizeHook resize;
    return fn();
  });
  // _cuda_hostTraceRedispatch of op(*args, **kwargs) at keyset, a Python
  // number for a Tensor operand a wrapped number, as eager's arg parser makes it
  m.def("_cuda_hostTraceRedispatchOp", [](const py::object& op, c10::DispatchKeySet keyset, const py::tuple& args, const py::dict& kwargs, py::function sink) {
    StreamingRecorder rec(std::move(sink));
    ht::RecorderGuard guard(&rec);
    ResizeHook resize;
    auto& handle = op.cast<c10::OperatorHandle&>();
    torch::jit::Stack stack;
    {
      torch::jit::ToIValueAllowNumbersAsTensors allow(true);
      stack = torch::jit::createStackForSchema(handle.schema(), torch::jit::tuple_slice(args), py::reinterpret_borrow<py::kwargs>(kwargs), /*self=*/std::nullopt);
    }
    {
      py::gil_scoped_release no_gil;
      handle.redispatchBoxed(keyset, &stack);
    }
    return torch::jit::createPyObjectForStack(std::move(stack));
  });
  m.def("_cuda_hostTraceSeal", [](const at::Tensor& t, bool on) { seal(t, on); });
  // a 0-dim CPU stand-in of a number for a Tensor operand, a wrapped number as eager's arg parser makes it
  m.def("_cuda_hostTraceWrapNumber", [](const at::Tensor& t) {
    TORCH_CHECK(t.dim() == 0 && t.is_cpu(), "host_trace: a wrapped number is a 0-dim CPU tensor");
    t.unsafeGetTensorImpl()->set_wrapped_number(true);
  });
  // a traced tensor's sizes, strides and storage offset, set (the trace's re-allocation of a fresh empty output)
  m.def("_cuda_hostTraceRestride", [](const at::Tensor& t, const std::vector<c10::SymInt>& sizes, const std::vector<c10::SymInt>& strides, const c10::SymInt& offset) {
    t.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides, offset);
  });
  // the last refusal of this thread's symbolic host code (at::detail::SymbolicRefusal), cleared:
  // None, "accessor", "output" or "generator"
  m.def("_cuda_hostTraceTakeRefusal", []() -> std::optional<std::string> {
    switch (at::detail::take_symbolic_refusal()) {
      case at::detail::SymbolicRefusal::None:
        return std::nullopt;
      case at::detail::SymbolicRefusal::Accessor:
        return "accessor";
      case at::detail::SymbolicRefusal::Output:
        return "output";
      case at::detail::SymbolicRefusal::Generator:
        return "generator";
    }
    return std::nullopt;
  });
  m.def("_cuda_hostTraceSymbolicIteratorsBuilt", []() { return at::detail::symbolic_iterators_built(); });
  // the C++ frames above the caller (a specializing int read's site)
  m.def("_cuda_hostTraceBacktrace", [](int64_t skip, int64_t frames) { return c10::get_backtrace(skip, frames, /*skip_python_frames=*/true); });
  m.def("_cuda_hostTraceCopy_", torch::wrap_pybind_function([](const at::Tensor& dst, const at::Tensor& src) {
    PyRecorder rec;
    return traced(rec, ht::copy_(rec, dst, src));
  }));
  m.def("_cuda_hostTraceToCopy", torch::wrap_pybind_function([](const at::Tensor& src, at::ScalarType dtype, at::MemoryFormat memory_format) {
    PyRecorder rec;
    return traced(rec, ht::to_copy(rec, src, dtype, memory_format));
  }));
  m.def("_cuda_hostTraceSum", torch::wrap_pybind_function([](const at::Tensor& self, const std::vector<int64_t>& dims, bool keepdim) {
    PyRecorder rec;
    return traced(rec, ht::sum(rec, self, dims, keepdim));
  }));
  m.def("_cuda_hostTraceMean", torch::wrap_pybind_function([](const at::Tensor& self, const std::vector<int64_t>& dims, bool keepdim) {
    PyRecorder rec;
    return traced(rec, ht::mean(rec, self, dims, keepdim));
  }));
  m.def("_cuda_hostTraceAmax", torch::wrap_pybind_function([](const at::Tensor& self, const std::vector<int64_t>& dims, bool keepdim) {
    PyRecorder rec;
    return traced(rec, ht::amax(rec, self, dims, keepdim));
  }));
  m.def("_cuda_hostTraceAmin", torch::wrap_pybind_function([](const at::Tensor& self, const std::vector<int64_t>& dims, bool keepdim) {
    PyRecorder rec;
    return traced(rec, ht::amin(rec, self, dims, keepdim));
  }));
  m.def("_cuda_hostTraceMaxAll", torch::wrap_pybind_function([](const at::Tensor& self, const std::optional<at::Tensor>& out) {
    PyRecorder rec;
    return traced(rec, ht::max_all(rec, self, out.value_or(at::Tensor())));
  }));
  m.def("_cuda_hostTraceMinAll", torch::wrap_pybind_function([](const at::Tensor& self, const std::optional<at::Tensor>& out) {
    PyRecorder rec;
    return traced(rec, ht::min_all(rec, self, out.value_or(at::Tensor())));
  }));
  m.def("_cuda_hostTraceMaxDim", torch::wrap_pybind_function([](const at::Tensor& self, int64_t dim, bool keepdim) {
    PyRecorder rec;
    return traced(rec, ht::max_dim(rec, self, dim, keepdim));
  }));
  m.def("_cuda_hostTraceMinDim", torch::wrap_pybind_function([](const at::Tensor& self, int64_t dim, bool keepdim) {
    PyRecorder rec;
    return traced(rec, ht::min_dim(rec, self, dim, keepdim));
  }));
  m.def("_cuda_hostTraceArgmax", torch::wrap_pybind_function([](const at::Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    PyRecorder rec;
    return traced(rec, ht::argmax(rec, self, dim, keepdim));
  }));
  m.def("_cuda_hostTraceArgmin", torch::wrap_pybind_function([](const at::Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    PyRecorder rec;
    return traced(rec, ht::argmin(rec, self, dim, keepdim));
  }));
  m.def("_cuda_hostTraceStdVar", torch::wrap_pybind_function([](const at::Tensor& self, const std::vector<int64_t>& dims, double correction, bool keepdim, bool take_sqrt) {
    PyRecorder rec;
    return traced(rec, ht::std_var(rec, self, dims, correction, keepdim, take_sqrt));
  }));
  m.def("_cuda_hostTraceSoftmax", torch::wrap_pybind_function([](const at::Tensor& self, int64_t dim, bool half_to_float) {
    PyRecorder rec;
    return traced(rec, ht::softmax(rec, self, dim, half_to_float));
  }));
  m.def("_cuda_hostTraceLogSoftmax", torch::wrap_pybind_function([](const at::Tensor& self, int64_t dim, bool half_to_float) {
    PyRecorder rec;
    return traced(rec, ht::log_softmax(rec, self, dim, half_to_float));
  }));
  m.def("_cuda_hostTraceLayerNorm", torch::wrap_pybind_function([](const at::Tensor& input, int64_t normalized_ndim, const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& bias, double eps) {
    PyRecorder rec;
    return traced(rec, ht::native_layer_norm(rec, input, normalized_ndim, weight.value_or(at::Tensor()), bias.value_or(at::Tensor()), eps));
  }));
  m.def("_cuda_hostTraceBatchNorm", torch::wrap_pybind_function([](const at::Tensor& input, const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& bias, const std::optional<at::Tensor>& running_mean, const std::optional<at::Tensor>& running_var, bool training, double eps) {
    PyRecorder rec;
    return traced(rec, ht::native_batch_norm(rec, input, weight.value_or(at::Tensor()), bias.value_or(at::Tensor()), running_mean.value_or(at::Tensor()), running_var.value_or(at::Tensor()), training, eps));
  }));
  m.def("_cuda_hostTraceRmsNorm", torch::wrap_pybind_function([](const at::Tensor& input, int64_t normalized_ndim, const std::optional<at::Tensor>& weight, std::optional<double> eps) {
    PyRecorder rec;
    return traced(rec, ht::fused_rms_norm(rec, input, normalized_ndim, weight.value_or(at::Tensor()), eps));
  }));
  m.def("_cuda_hostTraceIndexSelect", torch::wrap_pybind_function([](const at::Tensor& self, int64_t dim, const at::Tensor& index) {
    PyRecorder rec;
    return traced(rec, ht::index_select(rec, self, dim, index));
  }));
  m.def("_cuda_hostTraceArange", torch::wrap_pybind_function([](const c10::SymInt& size, const std::variant<c10::SymInt, double>& start, const std::variant<c10::SymInt, double>& step, at::ScalarType dtype, at::Device device) {
    PyRecorder rec;
    const auto scalar = [](const std::variant<c10::SymInt, double>& v) { return std::visit([](const auto& x) { return c10::Scalar(x); }, v); };
    return traced(rec, ht::arange(rec, size, scalar(start), scalar(step), dtype, device));
  }));
  m.def("_cuda_hostTraceTriu", torch::wrap_pybind_function([](const at::Tensor& self, int64_t k) {
    PyRecorder rec;
    return traced(rec, ht::triu(rec, self, k));
  }));
  m.def("_cuda_hostTraceNllLoss", torch::wrap_pybind_function([](const at::Tensor& self, const at::Tensor& target, const std::optional<at::Tensor>& weight, int64_t reduction, int64_t ignore_index) {
    PyRecorder rec;
    return traced(rec, ht::nll_loss_forward(rec, self, target, weight.value_or(at::Tensor()), reduction, ignore_index));
  }));
  m.def("_cuda_hostTraceMaxPool2d", torch::wrap_pybind_function([](const at::Tensor& input, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, bool ceil_mode) {
    PyRecorder rec;
    return traced(rec, ht::max_pool2d_with_indices(rec, input, kernel_size, stride, padding, dilation, ceil_mode));
  }));
  m.def("_cuda_hostTraceAdaptiveAvgPool2d", torch::wrap_pybind_function([](const at::Tensor& input, std::vector<int64_t> output_size) {
    PyRecorder rec;
    return traced(rec, ht::adaptive_avg_pool2d(rec, input, output_size));
  }));
  m.def("_cuda_hostTraceCat", torch::wrap_pybind_function([](const std::vector<at::Tensor>& tensors, int64_t dim) {
    PyRecorder rec;
    const std::vector<at::TensorBase> operands(tensors.begin(), tensors.end());
    return traced(rec, ht::cat(rec, operands, dim));
  }));
  // a tensor caches its contiguity and density the first time a host reads
  // them; dropping the cache has the next op read them again, under its guards
  m.def("_cuda_hostTraceRefreshContiguous", [](const at::Tensor& t) {
    std::vector<c10::SymInt> sizes(t.sym_sizes().begin(), t.sym_sizes().end());
    std::vector<c10::SymInt> strides(t.sym_strides().begin(), t.sym_strides().end());
    t.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
  });
}

} // namespace torch::cuda::host_trace
#endif
