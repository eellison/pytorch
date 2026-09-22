#include <torch/csrc/python_headers.h>

#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <pybind11/stl.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_symnode.h>

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/cuda/host_trace/Harvest.h>
#include <ATen/cuda/host_trace/Hooks.h>
#include <ATen/cuda/host_trace/HostTable.h>
#include <ATen/cuda/host_trace/Recorder.h>

#include <cstring>
#include <memory>
#include <unordered_map>

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
      return impl(v.data(), v.size());
    });
    opaque.append(std::move(d));
  }
  out["opaque"] = std::move(opaque);
  py::list memsets;
  for (const auto& ms : t.memsets) {
    py::dict d;
    d["seq"] = ms.seq;
    d["dst"] = py::cast(ms.dst);
    d["value"] = ms.value;
    d["bytes"] = py::cast(ms.bytes);
    memsets.append(std::move(d));
  }
  out["memsets"] = std::move(memsets);
  py::list host_buffers;
  for (const auto& hb : t.host_buffers) {
    py::dict d;
    d["seq"] = hb.seq;
    d["name"] = hb.name;
    d["root"] = py::cast(hb.root);
    d["nbytes"] = hb.nbytes;
    py::list elements;
    for (const auto& e : hb.elements) {
      py::dict q;
      q["offset"] = e.offset;
      q["size"] = e.size;
      q["kind"] = e.kind;
      q["value"] = to_py(e.v);
      elements.append(std::move(q));
    }
    d["elements"] = std::move(elements);
    host_buffers.append(std::move(d));
  }
  out["host_buffers"] = std::move(host_buffers);
  py::list memcpys;
  for (const auto& mc : t.memcpys) {
    py::dict d;
    d["seq"] = mc.seq;
    d["src"] = py::cast(mc.src);
    d["dst"] = py::cast(mc.dst);
    d["bytes"] = py::cast(mc.bytes);
    d["kind"] = mc.kind;
    memcpys.append(std::move(d));
  }
  out["memcpys"] = std::move(memcpys);
  out["rng_increment"] = t.rng_increment.has_value()
      ? py::cast(*t.rng_increment)
      : py::object(py::none());
  py::list rng_slots;
  for (const auto& r : t.rng_slots) {
    py::dict d;
    d["launch"] = r.launch;
    d["offset"] = r.offset;
    d["size"] = r.size;
    d["increment"] = py::cast(r.increment);
    rng_slots.append(std::move(d));
  }
  out["rng_slots"] = std::move(rng_slots);
  return out;
}

} // namespace

namespace {
// _HostTraceBoxer: the served call's box (see the binding below)
struct HostTraceBoxer {
  HostTraceBoxer(py::object positions, std::vector<int64_t> written)
      : identity_(positions.is_none()) {
    if (!identity_) {
      for (auto p : py::cast<std::vector<int64_t>>(positions)) {
        TORCH_CHECK(p >= 0, "_HostTraceBoxer: a negative position");
        positions_.push_back(static_cast<Py_ssize_t>(p));
      }
    }
    for (auto w : written) {
      TORCH_CHECK(w >= 0, "_HostTraceBoxer: a negative written position");
      written_.push_back(static_cast<Py_ssize_t>(w));
    }
  }

  py::list call(py::handle args, py::handle arena) const {
    TORCH_CHECK(
        PyTuple_CheckExact(args.ptr()),
        "_HostTraceBoxer: the arguments must be an exact tuple");
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args.ptr());
    const Py_ssize_t n =
        identity_ ? nargs : static_cast<Py_ssize_t>(positions_.size());
    const bool with_arena = !arena.is_none();
    PyObject* box = PyList_New(n + (with_arena ? 1 : 0));
    if (box == nullptr) {
      throw py::error_already_set();
    }
    py::list out = py::reinterpret_steal<py::list>(box);
    for (Py_ssize_t i = 0; i < n; ++i) {
      const Py_ssize_t p = identity_ ? i : positions_[static_cast<size_t>(i)];
      TORCH_CHECK_INDEX(p < nargs, "_HostTraceBoxer: position ", p, " of ", nargs, " arguments");
      PyObject* item = PyTuple_GET_ITEM(args.ptr(), p);
      Py_INCREF(item);
      PyList_SET_ITEM(box, i, item);
    }
    if (with_arena) {
      Py_INCREF(arena.ptr());
      PyList_SET_ITEM(box, n, arena.ptr());
    }
    for (auto w : written_) {
      TORCH_CHECK_INDEX(w < n, "_HostTraceBoxer: written position ", w, " of a box of ", n);
      PyObject* item = PyList_GET_ITEM(box, w);
      if (THPVariable_Check(item)) {
        THPVariable_Unpack(item).mutable_data_ptr();
      }
    }
    return out;
  }

 private:
  bool identity_;
  std::vector<Py_ssize_t> positions_;
  std::vector<Py_ssize_t> written_;
};

// _HostTracePredicate: a lowered tape's compiled predicate over a box in one
// pass (see the binding below)
struct HostTracePredicate {
  HostTracePredicate(
      std::vector<int64_t> pointer_indices,
      std::vector<int64_t> offset_indices,
      const std::vector<std::tuple<std::string, int64_t, int64_t>>& facts)
      : pointer_indices_(std::move(pointer_indices)),
        offset_indices_(std::move(offset_indices)) {
    static const std::unordered_map<std::string, int> kinds = {
        {"size", 0},
        {"stride", 1},
        {"rank", 2},
        {"dtype", 3},
        {"device", 4},
        {"pinned", 5},
        {"neg", 6},
        {"conj", 7}};
    for (const auto& [kind, index, dim] : facts) {
      auto it = kinds.find(kind);
      TORCH_CHECK(
          it != kinds.end(), "_HostTracePredicate: unknown fact kind ", kind);
      facts_.emplace_back(it->second, index, dim);
    }
    values_.resize(
        pointer_indices_.size() + offset_indices_.size() + facts_.size());
  }

  bool call(py::handle box, int64_t address) {
    TORCH_CHECK(
        PyList_CheckExact(box.ptr()),
        "_HostTracePredicate: the box must be an exact list");
    const Py_ssize_t n = PyList_GET_SIZE(box.ptr());
    auto tensor = [&](int64_t index) -> const at::Tensor& {
      TORCH_CHECK_INDEX(
          index >= 0 && index < n,
          "_HostTracePredicate: box position ",
          index,
          " of ",
          n);
      PyObject* item = PyList_GET_ITEM(box.ptr(), index);
      TORCH_CHECK(
          THPVariable_Check(item),
          "_HostTracePredicate: box position ",
          index,
          " is not a Tensor");
      return THPVariable_Unpack(item);
    };
    size_t k = 0;
    for (auto index : pointer_indices_) {
      const auto& t = tensor(index);
      // the storage's const data pointer plus the offset, as the dispatcher
      // reads an input's address (a copy-on-write storage stays lazy)
      values_[k++] = reinterpret_cast<int64_t>(t.storage().data()) +
          static_cast<int64_t>(t.element_size()) * t.storage_offset();
    }
    for (auto index : offset_indices_) {
      values_[k++] = tensor(index).storage_offset();
    }
    for (const auto& [kind, index, dim] : facts_) {
      const auto& t = tensor(index);
      int64_t v = 0;
      switch (kind) {
        case 0:
          v = dim < t.dim() ? t.size(dim) : -1;
          break;
        case 1:
          v = dim < t.dim() ? t.stride(dim) : -1;
          break;
        case 2:
          v = t.dim();
          break;
        case 3:
          v = static_cast<int64_t>(t.scalar_type());
          break;
        case 4:
          v = t.is_cuda() ? t.get_device() : -1;
          break;
        case 5:
          v = (!t.is_cuda() && t.is_pinned()) ? 1 : 0;
          break;
        case 6:
          v = t.is_neg() ? 1 : 0;
          break;
        default:
          v = t.is_conj() ? 1 : 0;
      }
      values_[k++] = v;
    }
    auto fn = reinterpret_cast<int8_t (*)(int64_t*, double*)>(
        static_cast<uintptr_t>(address));
    return fn(values_.empty() ? nullptr : values_.data(), nullptr) == 1;
  }

 private:
  std::vector<int64_t> pointer_indices_;
  std::vector<int64_t> offset_indices_;
  std::vector<std::tuple<int, int64_t, int64_t>> facts_;
  std::vector<int64_t> values_;
};
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

  // The closed regions' harvest (Harvest.h): the nodes of a graph the Python
  // side captured one library call into, as dicts: kernels (func, name, grid,
  // block, smem, image, layout, attrs) and memsets (dst, value, elem, width),
  // each with its kind; and the host facts the harvest classifies slots by.
  m.def("_host_trace_stack_probe", []() {
    // an address on the calling thread's stack: classifies the host stack
    // pointers cuBLAS leaves in a kernel image (see _host_trace.py)
    volatile int local = 0;
    return reinterpret_cast<uint64_t>(const_cast<int*>(&local));
  });
  m.def("_host_trace_stack_smear", [](int pattern) {
    // fills the stack below the caller with a pattern, so that a library
    // call's uninitialized parameter padding reads it (see _harvest)
    volatile unsigned char buf[1 << 16];
    std::memset(const_cast<unsigned char*>(buf), pattern, sizeof(buf));
    return static_cast<int>(buf[sizeof(buf) / 2]);
  });
  m.def("_host_trace_blas_workspace_size", []() {
    // the scratch a closed region's variant owns: the larger of the two
    // workspaces cuBLAS / cuBLASLt calls are given, so any variant fits
    return static_cast<int64_t>(std::max(
        at::cuda::getChosenWorkspaceSize(),
        at::cuda::getCUDABlasLtWorkspaceSize()));
  });
  m.def("_host_trace_blas_workspaces", [](int64_t stream) {
    // the workspace bases cuBLAS / cuBLASLt hold for this thread's handles
    // on `stream` (CublasHandlePool.cpp's maps, read without allocating): a
    // harvest classifies a stream-dependent qword as the workspace only if
    // it equals one of them
    void* s = reinterpret_cast<void*>(static_cast<intptr_t>(stream));
    void* handles[2] = {
        static_cast<void*>(at::cuda::getCurrentCUDABlasHandle(false)),
        static_cast<void*>(at::cuda::getCurrentCUDABlasLtHandle())};
    std::vector<uint64_t> out;
    for (at::cuda::WorkspaceMapWithMutex* ws :
         {&at::cuda::cublas_handle_stream_to_workspace(),
          &at::cuda::cublaslt_handle_stream_to_workspace()}) {
      std::shared_lock<std::shared_mutex> lock(ws->mutex);
      for (void* h : handles) {
        auto it = ws->map.find(std::make_tuple(h, s));
        if (it != ws->map.end()) {
          out.push_back(reinterpret_cast<uint64_t>(it->second.first.get()));
        }
      }
    }
    return out;
  });
  // the nodes of a graph a closed call was captured into (the graph's
  // handle as CUDAGraph.raw_cuda_graph() gives it)
  m.def(
      "_host_trace_harvest_nodes",
      [](int64_t graph, int probe_attr, bool anchored) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto g = reinterpret_cast<cudaGraph_t>(static_cast<intptr_t>(graph));
        auto harvested =
            at::cuda::host_trace::harvest_nodes(g, probe_attr, anchored);
        py::list out;
        for (const auto& h : harvested) {
          py::dict d;
          d["kind"] = h.kind == 0 ? "kernel"
              : h.kind == 1       ? "memset"
                                  : "memcpy";
          d["programmatic"] = h.programmatic;
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
      py::arg("probe_attr") = -1,
      py::arg("anchored") = false);
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
  // The generator's per-capture philox pointers on the capturing stream (the
  // generator is registered with the capturing graph by this call, as a
  // kernel's own philox request would), and the intragraph offset the next
  // request would see. Used by replay adapters that write the philox state of
  // a prepared capture into a launch's rng field.
  m.def("_host_trace_generator_capture_pointers", [](py::handle generator) {
    at::Generator gen = THPGenerator_Unwrap(generator.ptr());
    auto* g = at::check_generator<at::CUDAGeneratorImpl>(gen);
    std::lock_guard<std::mutex> lock(g->mutex_);
    at::PhiloxCudaState st = g->philox_cuda_state(0);
    TORCH_CHECK(
        st.captured_, "no stream capture is active on the current stream");
    return py::make_tuple(
        reinterpret_cast<uintptr_t>(st.seed_.ptr),
        reinterpret_cast<uintptr_t>(st.offset_.ptr),
        static_cast<uint64_t>(st.offset_intragraph_));
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
  // the tensors of `args` at `positions` (a tape's written inputs), read
  // through the mutable accessor before a replay binds their addresses: a
  // copy-on-write tensor materializes here as it would at the ordinary
  // host's launch; any other tensor is untouched
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
  // The box of a served call in one pass over the argument tuple: the tensors
  // at `positions` (None: every argument) as a fresh exact list, the box
  // positions in `written` read through the mutable accessor (a copy-on-write
  // tensor materializes there, as at the ordinary host's launch; the test
  // costs one deleter compare per position) and `arena` appended when given.
  // What list(args) followed by _host_trace_materialize did in two walks.
  py::class_<HostTraceBoxer>(m, "_HostTraceBoxer")
      .def(
          py::init<py::object, std::vector<int64_t>>(),
          py::arg("positions"),
          py::arg("written"))
      .def(
          "__call__",
          &HostTraceBoxer::call,
          py::arg("args"),
          py::arg("arena") = py::none());
  // The compiled predicate of a lowered tape evaluated over a box: the pointer
  // values, the storage offsets and the Tensor facts marshalled in one pass
  // (what direct_hosttrace.check_predicate read tensor by tensor in Python),
  // then the entry point at `address` (the guard, facts or arena entry).
  py::class_<HostTracePredicate>(m, "_HostTracePredicate")
      .def(
          py::init<
              std::vector<int64_t>,
              std::vector<int64_t>,
              const std::vector<std::tuple<std::string, int64_t, int64_t>>&>(),
          py::arg("pointer_indices"),
          py::arg("offset_indices"),
          py::arg("facts"))
      .def(
          "__call__",
          &HostTracePredicate::call,
          py::arg("box"),
          py::arg("address"));
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
  // Tensor::copy_'s rule for a pinned block a stream is still reading: the
  // caching host allocator defers the block's reuse to an event on that
  // stream. A replay records its pinned inputs this way after each call.
  // Returns whether the allocator owns the block.
  m.def(
      "_host_trace_record_host_event",
      [](const at::Tensor& pinned, int64_t stream) {
        TORCH_CHECK(
            pinned.is_cpu(), "_host_trace_record_host_event: a CPU tensor");
        c10::cuda::CUDAStream s = c10::cuda::getStreamFromExternal(
            reinterpret_cast<cudaStream_t>(static_cast<intptr_t>(stream)),
            c10::cuda::current_device());
        return at::getHostAllocator(at::kCUDA)->record_event(
            pinned.data_ptr(),
            pinned.storage().data_ptr().get_context(),
            s.unwrap());
      });
}
