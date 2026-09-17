#include <torch/csrc/python_headers.h>

#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_symnode.h>

#include <ATen/cuda/host_trace/Exec.h>
#include <ATen/cuda/host_trace/Hooks.h>
#include <ATen/cuda/host_trace/Recorder.h>

#include <memory>

// Python entry points for host tracing (aten/src/ATen/cuda/host_trace). The
// user-facing surface is torch/cuda/_host_trace.py; everything here is private.
// Forward declared in torch/csrc/Module.cpp, like THCPGraph_init.

namespace {

using at::cuda::host_trace::Exec;
using at::cuda::host_trace::NodeUpdate;
using at::cuda::host_trace::SymVal;
using at::cuda::host_trace::Tape;
using at::cuda::host_trace::TraceState;

// The recorder's hint reads: the Python SymNode's hint, never a guard.
int64_t int_hint(const c10::SymInt& s) {
  py::gil_scoped_acquire gil;
  c10::SymNodeImpl* n = s.toSymNodeImplUnowned();
  if (auto c = n->constant_int()) {
    return *c;
  }
  if (auto* p = dynamic_cast<torch::impl::PythonSymNodeImpl*>(n)) {
    return p->getPyObj().attr("hint").cast<int64_t>();
  }
  return n->guard_int(__FILE__, __LINE__);
}
double float_hint(const c10::SymFloat& s) {
  py::gil_scoped_acquire gil;
  c10::SymNode n = s.toSymNodeImpl();
  if (auto* p = dynamic_cast<torch::impl::PythonSymNodeImpl*>(n.get())) {
    return p->getPyObj().attr("hint").cast<double>();
  }
  return n->guard_float(__FILE__, __LINE__);
}
bool bool_hint(const c10::SymBool& s) {
  py::gil_scoped_acquire gil;
  c10::SymNodeImpl* n = s.toSymNodeImplUnowned();
  if (auto c = n->constant_bool()) {
    return *c;
  }
  if (auto* p = dynamic_cast<torch::impl::PythonSymNodeImpl*>(n)) {
    return p->getPyObj().attr("hint").cast<bool>();
  }
  return n->guard_bool(__FILE__, __LINE__);
}
c10::SymInt new_int_sym(int64_t hint, const std::string& name) {
  py::gil_scoped_acquire gil;
  py::object fn =
      py::module::import("torch.cuda._host_trace").attr("_new_int_symbol");
  return fn(hint, name).cast<c10::SymInt>();
}

py::object to_py(const SymVal& v) {
  switch (v.tag) {
    case SymVal::Tag::Int:
      return py::cast(v.i);
    case SymVal::Tag::Float:
      return py::cast(v.f);
    case SymVal::Tag::Bool:
      return py::cast(v.b);
  }
  return py::none();
}

py::list sym3(const std::array<c10::SymInt, 3>& a) {
  py::list out;
  for (const auto& s : a) {
    out.append(py::cast(s));
  }
  return out;
}

// A trace in progress: the tape it fills and the recorder state, so Python
// holds one object between begin and end.
struct TraceHandle {
  std::shared_ptr<Tape> tape = std::make_shared<Tape>();
  std::unique_ptr<TraceState> state;
  std::unique_ptr<at::cuda::host_trace::Scope> scope;
};

py::dict tape_records(const Tape& t) {
  py::dict out;
  py::list launches;
  for (const auto& L : t.launches) {
    py::dict d;
    d["seq"] = L.seq;
    d["kernel"] = L.kernel;
    d["func"] = reinterpret_cast<intptr_t>(L.func);
    d["cu_function"] = reinterpret_cast<intptr_t>(L.cu_function);
    d["param_layout"] = L.param_layout;
    d["grid"] = sym3(L.grid);
    d["block"] = py::make_tuple(L.block[0], L.block[1], L.block[2]);
    d["block_expr"] = sym3(L.block_expr);
    d["smem"] = py::cast(L.smem);
    py::list params;
    for (const auto& p : L.params) {
      py::dict q;
      q["offset"] = p.offset;
      q["size"] = p.size;
      q["kind"] = p.kind;
      q["name"] = p.name;
      q["access"] = p.access;
      q["value"] = to_py(p.v);
      params.append(std::move(q));
    }
    d["params"] = std::move(params);
    d["hint_image"] = py::bytes(
        reinterpret_cast<const char*>(L.hint_image.data()),
        L.hint_image.size());
    launches.append(std::move(d));
  }
  out["launches"] = std::move(launches);
  out["all_on_capture_stream"] = t.all_on_capture_stream;
  py::list opaque;
  for (const auto& o : t.opaque) {
    py::dict d;
    d["seq"] = o.seq;
    d["fn"] = o.fn;
    py::list args;
    for (const auto& a : o.args) {
      args.append(py::cast(a));
    }
    d["args"] = std::move(args);
    d["expected"] = o.expected;
    d["sym"] = py::cast(o.sym);
    d["kind"] = o.kind;
    auto impl = o.impl;
    // the raw address lets a compiled predicate call the host's own function
    d["impl"] = reinterpret_cast<uintptr_t>(impl);
    d["call"] = py::cpp_function([impl](const std::vector<int64_t>& v) {
      TORCH_CHECK(impl != nullptr, "host_trace: opaque call without impl");
      return impl(v);
    });
    opaque.append(std::move(d));
  }
  out["opaque"] = std::move(opaque);
  out["rng_increment"] = t.rng_increment.has_value()
      ? py::cast(*t.rng_increment)
      : py::object(py::none());
  return out;
}

} // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage)
void THCPHostTrace_init(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  at::cuda::host_trace::hooks::Hooks hooks;
  hooks.int_hint = &int_hint;
  hooks.float_hint = &float_hint;
  hooks.bool_hint = &bool_hint;
  hooks.new_int_sym = &new_int_sym;
  at::cuda::host_trace::hooks::set_hooks(hooks);

  py::register_exception<at::cuda::host_trace::TapeMismatch>(
      m, "_HostTraceTapeMismatch", PyExc_RuntimeError);
  py::register_exception<at::cuda::host_trace::Declined>(
      m, "_HostTraceDeclined", PyExc_RuntimeError);

  py::class_<TraceHandle, std::shared_ptr<TraceHandle>>(m, "_HostTraceRecorder")
      .def(py::init([](c10::DeviceIndex device) {
        // TraceState holds a stream guard, so it is neither copyable nor
        // movable: build it in place.
        auto h = std::make_shared<TraceHandle>();
        h->state = std::make_unique<TraceState>();
        h->state->t = h->tape.get();
        h->state->device = device;
        h->scope =
            std::make_unique<at::cuda::host_trace::Scope>(h->state.get());
        return h;
      }))
      .def(
          "register_root",
          [](TraceHandle& h,
             const at::Tensor& t,
             const c10::SymInt& root,
             int64_t itemsize,
             const std::string& name) {
            at::cuda::host_trace::register_root(t, root, itemsize, name);
          })
      .def(
          "next_seq",
          [](TraceHandle& h) { return at::cuda::host_trace::next_seq(); })
      .def(
          "finish",
          [](TraceHandle& h) {
            at::cuda::host_trace::finish_trace(h.state.get());
          })
      .def("records", [](TraceHandle& h) { return tape_records(*h.tape); })
      .def("end", [](TraceHandle& h) {
        h.scope.reset(); // leaves trace mode
        h.state.reset();
      });

  py::class_<Exec>(m, "_HostTraceExec")
      .def(py::init([](at::cuda::CUDAGraph& graph, c10::DeviceIndex device) {
        return std::make_unique<Exec>(graph, device);
      }))
      .def_property_readonly("num_nodes", &Exec::num_nodes)
      .def("kernel_name", &Exec::kernel_name)
      .def(
          "image",
          [](const Exec& e, size_t j) {
            auto v = e.image(j);
            return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
          })
      .def("grid", &Exec::grid)
      .def("block", &Exec::block)
      .def("smem", &Exec::smem)
      .def("instantiate", &Exec::instantiate)
      .def(
          "run",
          [](Exec& e,
             const std::vector<std::tuple<
                 size_t,
                 py::bytes,
                 std::array<unsigned, 3>,
                 std::array<unsigned, 3>,
                 unsigned>>& updates) {
            std::vector<NodeUpdate> us;
            us.reserve(updates.size());
            for (const auto& [node, image, grid, block, smem] : updates) {
              std::string s = image;
              NodeUpdate u{node, {}, grid, block, smem};
              u.image.assign(s.begin(), s.end());
              us.push_back(std::move(u));
            }
            e.run(us);
          })
      .def_property_readonly("calls", &Exec::calls)
      .def_property_readonly("dirty_nodes", &Exec::dirty_nodes);

  m.def("_host_trace_drop_storage", [](const at::Tensor& t) {
    at::cuda::host_trace::drop_storage(t);
  });
  // The storage's base address without materializing a copy-on-write
  // tensor (Storage::data() is the const accessor; Python's
  // UntypedStorage.data_ptr() is the mutable one): the trace's and the
  // replay's root binding of an input.
  m.def("_host_trace_storage_address", [](const at::Tensor& t) {
    return static_cast<int64_t>(
        reinterpret_cast<uintptr_t>(t.storage().data()));
  });
  m.def("_host_trace_sym_data_ptr", [](const at::Tensor& t) {
    return at::cuda::host_trace::sym_mutable_data_ptr(t);
  });
  m.def("_host_trace_tracing", []() {
    return at::cuda::host_trace::active() != nullptr;
  });
  // test hook: the recorder reads the captured nodes back in reverse order
  m.def("_host_trace_test_reverse_node_order", [](bool on) {
    at::cuda::host_trace::test_reverse_node_order(on);
  });
  // the name of a "not permitted inside a stream capture" CUDA error, by
  // code (torch.AcceleratorError.error_code), or None
  m.def("_host_trace_capture_error_name", [](int64_t code) -> py::object {
    const char* name =
        at::cuda::host_trace::capture_error_name(static_cast<int>(code));
    return name == nullptr ? py::object(py::none()) : py::str(name);
  });
  // (addr, nbytes) of every caching-allocator allocation on the stream
  // between the two calls: how the replay's build binds each allocation root
  m.def(
      "_host_trace_alloc_log_begin",
      [](c10::DeviceIndex device, int64_t stream) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        void* s = reinterpret_cast<void*>(static_cast<intptr_t>(stream));
        at::cuda::host_trace::alloc_log_begin(device, s);
      });
  m.def("_host_trace_alloc_log_end", []() {
    return at::cuda::host_trace::alloc_log_end();
  });
}
