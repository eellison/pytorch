#include <torch/csrc/cuda/host_trace/Variant.h>

#if !defined(USE_ROCM)
#include <ATen/cuda/host_trace/Ops.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>

#include <optional>
#include <tuple>
#include <variant>

namespace torch::cuda::host_trace {

namespace {

namespace ht = at::cuda::host_trace;

// a traced tensor's data_ptr() is symbolic, so ask Python
struct PyRecorder : ht::Recorder {
  c10::SymInt data_ptr(const at::TensorBase& t) override {
    auto obj = py::reinterpret_steal<py::object>(THPVariable_Wrap(t));
    return obj.attr("data_ptr")().cast<c10::SymInt>();
  }
  c10::SymInt bit_length(const c10::SymInt& x) override {
    return host_trace.attr("bit_length")(x).cast<c10::SymInt>();
  }
  c10::SymInt pow2(const c10::SymInt& x) override {
    return py::cast(x).attr("__rpow__")(2).cast<c10::SymInt>();
  }
  c10::SymInt f32_div(const c10::SymInt& a, const c10::SymInt& b) override {
    return host_trace.attr("f32_div")(a, b).cast<c10::SymInt>();
  }
  c10::SymInt select(const c10::SymBool& c, const c10::SymInt& a, const c10::SymInt& b) override {
    return host_trace.attr("select")(c, a, b).cast<c10::SymInt>();
  }
  py::module host_trace = py::module::import("torch.cuda._host_trace");
};

// [(function, offsets, params, fields, grid, block, smem)], each field
// (param, offset, width, value, is_pointer)
py::list records(const PyRecorder& rec) {
  py::list launches;
  for (const auto& r : rec.launches) {
    py::list params;
    for (const auto& p : r.params) {
      params.append(py::bytes(reinterpret_cast<const char*>(p.data()), p.size()));
    }
    py::list fields;
    for (const auto& f : r.fields) {
      fields.append(py::make_tuple(f.param, f.offset, f.width, f.value, f.pointer));
    }
    launches.append(py::make_tuple(
        reinterpret_cast<uintptr_t>(r.function),
        r.offsets,
        params,
        fields,
        py::make_tuple(r.grid.x, r.grid.y, r.grid.z),
        py::make_tuple(r.block.x, r.block.y, r.block.z),
        r.smem));
  }
  return launches;
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
  m.def("_cuda_hostTraceMul", torch::wrap_pybind_function([](const at::Tensor& a, const at::Tensor& b) {
    PyRecorder rec;
    return traced(rec, ht::mul(rec, a, b));
  }));
  m.def("_cuda_hostTraceAdd", torch::wrap_pybind_function([](const at::Tensor& a, const at::Tensor& b, std::variant<int64_t, double> alpha) {
    PyRecorder rec;
    return traced(rec, ht::add(rec, a, b, std::visit([](auto v) { return c10::Scalar(v); }, alpha)));
  }));
  m.def("_cuda_hostTraceSilu", torch::wrap_pybind_function([](const at::Tensor& a) {
    PyRecorder rec;
    return traced(rec, ht::silu(rec, a));
  }));
  m.def("_cuda_hostTraceGelu", torch::wrap_pybind_function([](const at::Tensor& a, const std::string& approximate) {
    PyRecorder rec;
    return traced(rec, ht::gelu(rec, a, approximate));
  }));
  m.def("_cuda_hostTraceRsqrt", torch::wrap_pybind_function([](const at::Tensor& a) {
    PyRecorder rec;
    return traced(rec, ht::rsqrt(rec, a));
  }));
  m.def("_cuda_hostTraceWhere", torch::wrap_pybind_function([](const at::Tensor& cond, const at::Tensor& a, const at::Tensor& b) {
    PyRecorder rec;
    return traced(rec, ht::where(rec, cond, a, b));
  }));
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
  m.def("_cuda_hostTraceRmsNorm", torch::wrap_pybind_function([](const at::Tensor& input, int64_t normalized_ndim, const std::optional<at::Tensor>& weight, std::optional<double> eps) {
    PyRecorder rec;
    return traced(rec, ht::fused_rms_norm(rec, input, normalized_ndim, weight.value_or(at::Tensor()), eps));
  }));
}

} // namespace torch::cuda::host_trace
#endif
