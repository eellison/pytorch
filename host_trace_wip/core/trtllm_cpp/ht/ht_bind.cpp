// pybind module of the traced trtllm-gen launcher (flashinfer/trtllm_cpp.py calls it under a host trace):
// decode(*args) / context(*args) take FlashInfer's binding arguments (traced tensors, SymInts) in the binding's
// order, run the patched launcher, and return its launches as records; candidates() / load() are the warm-up's.
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/extension.h>

#include "ht_entry.h"

namespace py = pybind11;
using fi_ht::ScaleArg;
using fi_ht::SymInt;
using fi_ht::TensorView;

namespace {

py::module& host_trace() {
  static py::module m = py::module::import("torch.cuda._host_trace");
  return m;
}

SymInt data_ptr(const at::Tensor& t) {
  auto obj = py::reinterpret_steal<py::object>(THPVariable_Wrap(t));
  return obj.attr("data_ptr")().cast<SymInt>();
}

SymInt bit_length(const SymInt& x) { return host_trace().attr("bit_length")(x).cast<SymInt>(); }

SymInt pow2(const SymInt& x) { return py::cast(x).attr("__rpow__")(2).cast<SymInt>(); }

// the binding's positional arguments by kind
struct Args {
  py::args a;
  TensorView T(size_t i) const { return TensorView(a[i].cast<at::Tensor>()); }
  std::optional<TensorView> OT(size_t i) const {
    if (a[i].is_none()) {
      return std::nullopt;
    }
    return T(i);
  }
  SymInt S(size_t i) const { return a[i].cast<SymInt>(); }
  int64_t I(size_t i) const { return S(i).guard_int(__FILE__, __LINE__); }
  double D(size_t i) const { return a[i].cast<double>(); }
  bool B(size_t i) const { return a[i].cast<bool>(); }
  std::optional<float> OF(size_t i) const {
    return a[i].is_none() ? std::nullopt : std::optional<float>(a[i].cast<float>());
  }
  std::optional<bool> OB(size_t i) const {
    return a[i].is_none() ? std::nullopt : std::optional<bool>(a[i].cast<bool>());
  }
  ScaleArg V(size_t i) const {
    ScaleArg s;
    if (THPVariable_Check(a[i].ptr())) {
      s.tensor = T(i);
    } else {
      s.value = a[i].cast<double>();
    }
    return s;
  }
};

py::list records(const fi_ht::Recorder& rec) {
  py::list out;
  for (const auto& r : rec.launches) {
    py::dict d;
    d["function"] = r.function;
    d["name"] = r.name;
    py::list params;
    for (const auto& p : r.params) {
      params.append(py::bytes(reinterpret_cast<const char*>(p.data()), p.size()));
    }
    d["params"] = params;
    d["offsets"] = r.offsets;
    py::list fields;
    for (const auto& f : r.fields) {
      fields.append(py::make_tuple(f.param, f.offset, f.width, f.value, f.pointer));
    }
    d["fields"] = fields;
    py::list tmas;
    for (const auto& t : r.tmas) {
      tmas.append(py::make_tuple(t.param, t.offset, t.dtype, t.address, t.shape, t.strides, t.box, t.swizzle, t.fill));
    }
    d["tmas"] = tmas;
    d["grid"] = py::make_tuple(r.grid[0], r.grid[1], r.grid[2]);
    d["block"] = py::make_tuple(r.block[0], r.block[1], r.block[2]);
    d["smem"] = r.smem;
    d["cluster"] = py::make_tuple(r.cluster[0], r.cluster[1], r.cluster[2]);
    d["policy"] = r.policy;
    d["pdl"] = r.pdl;
    d["packed"] = r.packed;
    out.append(d);
  }
  return out;
}

struct Recording {
  fi_ht::Recorder rec;
  fi_ht::Recorder* prev;
  Recording() : prev(fi_ht::recorder()) { fi_ht::recorder() = &rec; }
  ~Recording() { fi_ht::recorder() = prev; }
};

py::list decode(py::args args) {
  Args a{std::move(args)};
  if (a.a.size() != 35) {
    throw std::invalid_argument("decode takes the binding's 35 arguments");
  }
  Recording r;
  fi_ht_traced::flashinfer::trtllm_paged_attention_decode(
      a.T(0), a.OT(1), a.T(2), a.T(3), a.T(4), a.T(5), a.T(6), a.T(7), a.T(8), a.S(9), a.S(10), a.V(11), a.V(12),
      a.D(13), a.I(14), a.I(15), a.S(16), a.I(17), a.I(18), a.I(19), a.B(20), a.S(21), a.OT(22), a.OT(23), a.OT(24),
      a.OT(25), a.OF(26), a.OB(27), a.OT(28), a.S(29), a.S(30), a.B(31), a.OT(32), a.I(33), a.OB(34));
  return records(r.rec);
}

py::list context(py::args args) {
  Args a{std::move(args)};
  if (a.a.size() != 34) {
    throw std::invalid_argument("context takes the binding's 34 arguments");
  }
  Recording r;
  fi_ht_traced::flashinfer::trtllm_paged_attention_context(
      a.T(0), a.OT(1), a.T(2), a.T(3), a.T(4), a.T(5), a.T(6), a.T(7), a.T(8), a.S(9), a.S(10), a.V(11), a.V(12),
      a.D(13), a.I(14), a.I(15), a.S(16), a.I(17), a.T(18), a.T(19), a.I(20), a.B(21), a.S(22), a.OT(23), a.OT(24),
      a.OT(25), a.OF(26), a.OB(27), a.OB(28), a.OB(29), a.B(30), a.OT(31), a.S(32), a.S(33));
  return records(r.rec);
}

}  // namespace

PYBIND11_MODULE(fi_ht_trtllm, m) {
  fi_ht::hooks.data_ptr = data_ptr;
  fi_ht::hooks.bit_length = bit_length;
  fi_ht::hooks.pow2 = pow2;
  m.def("decode", &decode);
  m.def("context", &context);
  m.def("candidates", &fi_ht::candidates);
  m.def("load", &fi_ht::load);
  m.def("set_stock_library", &fi_ht::set_stock_library);
}
