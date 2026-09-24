#include <torch/csrc/python_headers.h>

#include <pybind11/stl.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_symnode.h>

#include <ATen/cuda/host_trace/Harvest.h>
#include <ATen/cuda/host_trace/Hooks.h>
#include <ATen/cuda/host_trace/Recorder.h>

#include <memory>

// Python entry points for host tracing (aten/src/ATen/cuda/host_trace). The
// user-facing surface is torch/cuda/_host_trace.py; everything here is private.
// Forward declared in torch/csrc/Module.cpp, like THCPGraph_init.

namespace {

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
c10::SymInt new_int_sym(int64_t hint, const std::string& name, bool positive) {
  py::gil_scoped_acquire gil;
  py::object fn =
      py::module::import("torch.cuda._host_trace").attr("_new_int_symbol");
  return fn(hint, name, positive).cast<c10::SymInt>();
}
c10::SymFloat round_float32(const c10::SymFloat& value) {
  py::gil_scoped_acquire gil;
  return py::module::import("torch.cuda._host_trace")
      .attr("_round_float32")(py::cast(value))
      .cast<c10::SymFloat>();
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

// The recorder on another thread for one op (Recorder.h ThreadScope): the
// trace mode enters it around each op it routes off the tracing thread.
struct ThreadEntry {
  std::shared_ptr<TraceHandle> handle;
  std::unique_ptr<at::cuda::host_trace::ThreadScope> scope;
};

// A kernel-choice context entered from Python (Recorder.h KernelChoice): a
// closed region's operand broadcast, an override's dispatch condition.
struct KernelChoiceEntry {
  std::unique_ptr<at::cuda::host_trace::KernelChoice> scope;
};

py::dict tape_records(const Tape& t) {
  py::dict out;
  py::list launches;
  for (const auto& L : t.launches) {
    py::dict d;
    d["seq"] = L.seq;
    d["kernel"] = L.kernel;
    d["func"] = reinterpret_cast<intptr_t>(L.func);
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
  out["written_roots"] = t.written_roots;
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
    d["domain"] = o.domain;
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
  hooks.round_float32 = &round_float32;
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
             bool allocation,
             std::string name) {
            at::cuda::host_trace::register_root(
                t, root, itemsize, allocation, std::move(name));
          })
      .def(
          "next_seq",
          [](TraceHandle& h) { return at::cuda::host_trace::next_seq(); })
      // the event counter as it stands (the seq the next event takes): read
      // at an op's entry and exit for the tape's op table
      .def("seq", [](TraceHandle& h) { return h.tape->seq; })
      .def(
          "finish",
          [](TraceHandle& h) {
            at::cuda::host_trace::finish_trace(h.state.get());
          })
      .def("records", [](TraceHandle& h) { return tape_records(*h.tape); })
      .def(
          "end",
          [](TraceHandle& h) {
            h.scope.reset(); // leaves trace mode
            h.state.reset();
          })
      .def("on_this_thread", [](const std::shared_ptr<TraceHandle>& h) {
        return ThreadEntry{h, nullptr};
      });

  py::class_<ThreadEntry>(m, "_HostTraceThreadEntry")
      .def(
          "__enter__",
          [](ThreadEntry& e) {
            TORCH_CHECK(
                e.handle->state != nullptr && e.scope == nullptr,
                "host_trace: on_this_thread after the trace ended, or entered twice");
            e.scope = std::make_unique<at::cuda::host_trace::ThreadScope>(
                e.handle->state.get());
          })
      .def(
          "__exit__",
          [](ThreadEntry& e,
             const py::object& /*type*/,
             const py::object& /*value*/,
             const py::object& /*tb*/) { e.scope.reset(); });

  py::class_<KernelChoiceEntry>(m, "_HostTraceKernelChoice")
      .def(py::init([]() { return KernelChoiceEntry{}; }))
      .def(
          "__enter__",
          [](KernelChoiceEntry& e) {
            TORCH_CHECK(
                e.scope == nullptr, "host_trace: kernel choice entered twice");
            e.scope = std::make_unique<at::cuda::host_trace::KernelChoice>();
          })
      .def(
          "__exit__",
          [](KernelChoiceEntry& e,
             const py::object& /*type*/,
             const py::object& /*value*/,
             const py::object& /*tb*/) { e.scope.reset(); });
  m.def("_host_trace_kernel_choice_depth", []() {
    return at::cuda::host_trace::kernel_choice_depth();
  });

  // the nodes of a graph a closed call was captured into (the graph's
  // handle as CUDAGraph.raw_cuda_graph() gives it)
  m.def(
      "_host_trace_harvest_nodes",
      [](int64_t graph, int probe_attr) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto g = reinterpret_cast<cudaGraph_t>(static_cast<intptr_t>(graph));
        auto harvested = at::cuda::host_trace::harvest_nodes(g, probe_attr);
        py::list out;
        for (const auto& h : harvested) {
          py::dict d;
          d["kind"] = h.kind == 0 ? "kernel"
              : h.kind == 1       ? "memset"
                                  : "memcpy";
          if (h.kind == 2) {
            d["name"] = h.name;
            d["src"] = h.src;
            d["dst"] = h.dst;
            d["bytes"] = h.width;
            out.append(std::move(d));
            continue;
          }
          if (h.kind == 1) {
            d["name"] = h.name;
            d["dst"] = h.dst;
            d["value"] = h.value;
            d["elem"] = h.elem;
            d["width"] = h.width;
            out.append(std::move(d));
            continue;
          }
          d["func"] = h.func;
          d["name"] = h.name;
          d["grid"] = h.grid;
          d["block"] = h.block;
          d["smem"] = h.smem;
          d["image"] = py::bytes(
              reinterpret_cast<const char*>(h.image.data()), h.image.size());
          d["layout"] = h.layout;
          d["attrs"] = h.attrs;
          out.append(std::move(d));
        }
        return out;
      },
      py::arg("graph"),
      py::arg("probe_attr") = -1);
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
  // The tensors of `args` at `positions` (a tape's written inputs) read
  // through the mutable accessor before a replay binds their addresses: a
  // copy-on-write tensor materializes here, as it would at the ordinary
  // host's launch; any other tensor is a pointer read, anything else is
  // left alone.
  m.def(
      "_host_trace_materialize",
      [](const py::sequence& args, const std::vector<int64_t>& positions) {
        for (auto p : positions) {
          py::object item = args[static_cast<size_t>(p)];
          if (THPVariable_Check(item.ptr())) {
            THPVariable_Unpack(item.ptr()).mutable_data_ptr();
          }
        }
      });
  m.def("_host_trace_tracing", []() {
    return at::cuda::host_trace::active() != nullptr;
  });
  // test hook: the recorder reads the captured nodes back in reverse order
  m.def("_host_trace_test_reverse_node_order", [](bool on) {
    at::cuda::host_trace::test_reverse_node_order(on);
  });
  // test hook: finish_trace fails right after the capture ended
  m.def("_host_trace_test_fail_capture_end", [](bool on) {
    at::cuda::host_trace::test_fail_capture_end(on);
  });
  // the name of a "not permitted inside a stream capture" CUDA error, by
  // code (torch.AcceleratorError.error_code), or None
  m.def("_host_trace_capture_error_name", [](int64_t code) -> py::object {
    const char* name =
        at::cuda::host_trace::capture_error_name(static_cast<int>(code));
    return name == nullptr ? py::object(py::none()) : py::str(name);
  });
  // (addr, nbytes) of every caching-allocator allocation a capture's pool
  // served between the two calls: how the closed regions' harvest finds the
  // library call's own allocations
  m.def(
      "_host_trace_alloc_log_begin",
      [](c10::DeviceIndex device, c10::cuda::MempoolId_t pool) {
        at::cuda::host_trace::alloc_log_begin(device, pool);
      });
  m.def("_host_trace_alloc_log_end", []() {
    return at::cuda::host_trace::alloc_log_end();
  });
}
