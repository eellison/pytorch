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
  c10::SymInt f32_div(const c10::SymInt& a, const c10::SymInt& b) override {
    return py::module::import("torch.cuda._host_trace")
        .attr("f32_div")(a, b)
        .cast<c10::SymInt>();
  }
};

// the op's traced host under the caller's dispatch mode, args as its schema
// orders them: its output and its launches as (function, param offsets, param
// bytes, fields, grid, block, smem), a field (param, offset, width, value, is
// pointer)
py::tuple traced_aten(const std::string& name, const py::tuple& args) {
  namespace ht = at::cuda::host_trace;
  PyRecorder rec;
  auto tensor = [&](size_t i) {
    if (!THPVariable_Check(args[i].ptr())) {
      ht::decline(name + " of a non-tensor operand");
    }
    return args[i].cast<at::Tensor>();
  };
  at::TensorBase out;
  if (name == "mul") {
    out = ht::mul(rec, tensor(0), tensor(1));
  } else if (name == "add") {
    const py::handle alpha = args[2];
    if (py::isinstance<py::bool_>(alpha) ||
        !(py::isinstance<py::int_>(alpha) || py::isinstance<py::float_>(alpha))) {
      ht::decline("an add of a non-number alpha");
    }
    out = ht::add(
        rec,
        tensor(0),
        tensor(1),
        py::isinstance<py::int_>(alpha) ? c10::Scalar(alpha.cast<int64_t>())
                                        : c10::Scalar(alpha.cast<double>()));
  } else if (name == "silu") {
    out = ht::silu(rec, tensor(0));
  } else if (name == "gelu") {
    out = ht::gelu(rec, tensor(0), args[1].cast<std::string>());
  } else if (name == "rsqrt") {
    out = ht::rsqrt(rec, tensor(0));
  } else if (name == "where") {
    out = ht::where(rec, tensor(0), tensor(1), tensor(2));
  } else if (name == "copy_") {
    out = ht::copy_(rec, tensor(0), tensor(1));
  } else if (name == "_to_copy" || name == "clone") {
    const at::Tensor src = tensor(0);
    auto dtype = src.scalar_type();
    if (name == "_to_copy") {
      auto layout = args[2].cast<std::optional<c10::Layout>>();
      auto device = args[3].cast<std::optional<at::Device>>();
      if (layout.value_or(c10::kStrided) != c10::kStrided ||
          device.value_or(src.device()) != src.device() ||
          args[4].cast<std::optional<bool>>().value_or(false)) {
        ht::decline("a _to_copy to another layout or device, or pinned");
      }
      dtype = args[1].cast<std::optional<at::ScalarType>>().value_or(dtype);
    }
    auto memory_format =
        args[name == "clone" ? 1 : 6].cast<std::optional<at::MemoryFormat>>();
    out = ht::to_copy(
        rec, src, dtype, memory_format.value_or(at::MemoryFormat::Preserve));
  } else if (name == "sum" || name == "mean" || name == "amax") {
    // sum and mean: (self, dtype) or (self, dim, keepdim, dtype)
    const bool all = name != "amax" && args.size() == 2;
    if (name != "amax" && !args[all ? 1 : 3].is_none()) {
      ht::decline("a " + name + " with a dtype");
    }
    const auto dims = all || args[1].is_none()
        ? std::vector<int64_t>{}
        : args[1].cast<std::vector<int64_t>>();
    const bool keepdim = !all && args[2].cast<bool>();
    auto host = name == "sum" ? ht::sum : name == "mean" ? ht::mean : ht::amax;
    out = host(rec, tensor(0), dims, keepdim);
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
        py::make_tuple(r.grid.x, r.grid.y, r.grid.z),
        py::make_tuple(r.block.x, r.block.y, r.block.z),
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
