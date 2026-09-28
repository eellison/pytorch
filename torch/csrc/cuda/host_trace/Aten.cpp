#include <torch/csrc/cuda/host_trace/Variant.h>

#if !defined(USE_ROCM)
#include <ATen/cuda/host_trace/Ops.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>

namespace torch::cuda::host_trace {

namespace {

// a traced tensor's address is its data_ptr() in Python, its root's symbol
// plus its offset
struct PyRecorder : at::cuda::host_trace::Recorder {
  c10::SymInt data_ptr(const at::TensorBase& t) override {
    auto obj = py::reinterpret_steal<py::object>(THPVariable_Wrap(t));
    return obj.attr("data_ptr")().cast<c10::SymInt>();
  }
  c10::SymInt bit_length(const c10::SymInt& x) override {
    return py::module::import("torch.cuda._host_trace")
        .attr("bit_length")(x)
        .cast<c10::SymInt>();
  }
  c10::SymInt pow2(const c10::SymInt& x) override {
    return py::cast(x).attr("__rpow__")(2).cast<c10::SymInt>();
  }
};

// the op's traced host under the caller's dispatch mode: its output and its
// launches as (function, param offsets, param bytes, fields, grid, block,
// smem), a field (param, offset, width, value, is pointer)
py::tuple traced_aten(
    const std::string& name,
    const std::vector<at::Tensor>& args) {
  namespace ht = at::cuda::host_trace;
  PyRecorder rec;
  at::TensorBase out;
  if (name == "mul" && args.size() == 2) {
    out = ht::mul(rec, args[0], args[1]);
  } else if (name == "silu" && args.size() == 1) {
    out = ht::silu(rec, args[0]);
  } else {
    ht::decline("no traced host for " + name);
  }
  py::list launches;
  for (const auto& r : rec.launches) {
    py::list params;
    for (const auto& p : r.params) {
      params.append(
          py::bytes(reinterpret_cast<const char*>(p.data()), p.size()));
    }
    py::list fields;
    for (const auto& f : r.fields) {
      fields.append(
          py::make_tuple(f.param, f.offset, f.width, f.value, f.pointer));
    }
    launches.append(py::make_tuple(
        reinterpret_cast<uintptr_t>(r.function),
        r.offsets,
        params,
        fields,
        r.grid,
        r.block,
        r.smem));
  }
  return py::make_tuple(at::Tensor(out), launches);
}

} // namespace

void initHostTraceAtenBindings(py::module& m) {
  m.def("_cuda_hostTraceAten", torch::wrap_pybind_function(traced_aten));
}

} // namespace torch::cuda::host_trace
#endif
