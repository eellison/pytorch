#include <torch/csrc/cuda/host_trace/Variant.h>

#if !defined(USE_ROCM)
#include <ATen/cuda/host_trace/Ops.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>

#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

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

// [(function, offsets, params, fields, grid, block, smem) or, a memcpy,
// (dst, src, bytes)], each field (param, offset, width, value, is_pointer)
py::list records(const PyRecorder& rec) {
  py::list launches;
  for (const auto& launch : rec.launches) {
    if (const auto* m = std::get_if<ht::MemcpyRecord>(&launch)) {
      launches.append(py::make_tuple(m->dst, m->src, m->bytes));
      continue;
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
  // node: the witness capture's kernel node (function, parameter offsets and
  // images); addresses: its functor's operand addresses (parameter, offset,
  // operand, offset into the operand)
  m.def("_cuda_hostTracePointwise", torch::wrap_pybind_function([](const std::vector<std::optional<at::Tensor>>& outs, const std::vector<at::ScalarType>& out_dtypes, const std::vector<at::Tensor>& inputs, std::optional<at::ScalarType> compute_dtype, bool dynamic, const std::string& name, uintptr_t function, const std::vector<size_t>& offsets, const std::vector<py::bytes>& images, const std::vector<std::tuple<size_t, size_t, at::Tensor, int64_t>>& addresses) {
    PyRecorder rec;
    ht::KernelRecord node;
    node.function = reinterpret_cast<cudaFunction_t>(function);
    node.offsets = offsets;
    for (const auto& image : images) {
      const auto s = static_cast<std::string_view>(image);
      node.params.emplace_back(s.begin(), s.end());
    }
    for (const auto& [param, offset, operand, delta] : addresses) {
      node.fields.push_back({param, offset, sizeof(void*), rec.data_ptr(operand) + delta, true});
    }
    const std::vector<at::TensorBase> operands(inputs.begin(), inputs.end());
    std::vector<at::TensorBase> targets;
    for (const auto& out : outs) {
      targets.push_back(out.value_or(at::Tensor()));
    }
    const auto results = ht::pointwise(rec, targets, out_dtypes, operands, compute_dtype.value_or(at::ScalarType::Undefined), dynamic, name, std::move(node));
    return py::make_tuple(std::vector<at::Tensor>(results.begin(), results.end()), records(rec));
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
