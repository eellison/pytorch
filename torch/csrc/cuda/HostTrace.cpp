#include <torch/csrc/python_headers.h>

#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <pybind11/stl.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_symnode.h>

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/cuda/host_trace/Exec.h>
#include <ATen/cuda/host_trace/Hooks.h>
#include <ATen/cuda/host_trace/HostTable.h>
#include <ATen/cuda/host_trace/Recorder.h>

#include <cstring>
#include <memory>

// Python entry points for host tracing (aten/src/ATen/cuda/host_trace). The
// user-facing surface is torch/cuda/_host_trace.py; everything here is private.
// Forward declared in torch/csrc/Module.cpp, like THCPGraph_init.

namespace {

using at::cuda::host_trace::Exec;
using at::cuda::host_trace::MemcpyUpdate;
using at::cuda::host_trace::MemsetUpdate;
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

  py::class_<Exec>(m, "_HostTraceExec")
      .def(py::init([](at::cuda::CUDAGraph& graph, c10::DeviceIndex device) {
        return std::make_unique<Exec>(graph, device);
      }))
      .def_property_readonly("num_nodes", &Exec::num_nodes)
      .def("kernel_name", &Exec::kernel_name)
      .def("dependencies", &Exec::dependencies)
      .def(
          "image",
          [](const Exec& e, size_t j) {
            auto v = e.image(j);
            return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
          })
      .def("grid", &Exec::grid)
      .def("block", &Exec::block)
      .def("smem", &Exec::smem)
      .def("attrs", &Exec::attrs)
      .def("node_kinds", &Exec::node_kinds)
      .def("adopt_driver", &Exec::adopt_driver)
      .def_property_readonly("num_memset_nodes", &Exec::num_memset_nodes)
      .def("memset_dst", &Exec::memset_dst)
      .def("memset_bytes", &Exec::memset_bytes)
      .def("memset_value", &Exec::memset_value)
      .def_property_readonly("num_memcpy_nodes", &Exec::num_memcpy_nodes)
      .def("memcpy_src", &Exec::memcpy_src)
      .def("memcpy_dst", &Exec::memcpy_dst)
      .def("memcpy_bytes", &Exec::memcpy_bytes)
      .def("memcpy_kind", &Exec::memcpy_kind)
      .def("instantiate", &Exec::instantiate, py::arg("replay") = true)
      .def(
          "run",
          [](Exec& e,
             const std::vector<std::tuple<
                 size_t,
                 py::bytes,
                 std::array<unsigned, 3>,
                 std::array<unsigned, 3>,
                 unsigned,
                 uint64_t,
                 std::vector<int64_t>>>& updates,
             const std::vector<
                 std::tuple<size_t, uint64_t, uint64_t, unsigned>>&
                 memset_updates,
             const std::vector<
                 std::tuple<size_t, uint64_t, uint64_t, uint64_t>>&
                 memcpy_updates) {
            std::vector<NodeUpdate> us;
            us.reserve(updates.size());
            for (const auto& [node, image, grid, block, smem, func, attrs] :
                 updates) {
              std::string s = image;
              NodeUpdate u{node, {}, grid, block, smem, func, attrs};
              u.image.assign(s.begin(), s.end());
              us.push_back(std::move(u));
            }
            std::vector<MemsetUpdate> ms;
            ms.reserve(memset_updates.size());
            for (const auto& [node, dst, bytes, value] : memset_updates) {
              ms.push_back(MemsetUpdate{node, dst, bytes, value});
            }
            std::vector<MemcpyUpdate> mc;
            mc.reserve(memcpy_updates.size());
            for (const auto& [node, src, dst, bytes] : memcpy_updates) {
              mc.push_back(MemcpyUpdate{node, src, dst, bytes});
            }
            e.run(us, ms, mc);
          },
          py::arg("updates"),
          py::arg("memset_updates") =
              std::vector<std::tuple<size_t, uint64_t, uint64_t, unsigned>>{},
          py::arg("memcpy_updates") =
              std::vector<std::tuple<size_t, uint64_t, uint64_t, uint64_t>>{})
      .def_property_readonly("dirty_nodes", &Exec::dirty_nodes)
      .def_property_readonly("dirty_memset_nodes", &Exec::dirty_memset_nodes)
      .def_property_readonly("dirty_memcpy_nodes", &Exec::dirty_memcpy_nodes)
      .def_property_readonly("graph_updates", &Exec::graph_updates);

  // A raw stream capture of one closed library call (Exec.h): the Python side
  // begins, runs the call on `stream`, ends and gets the nodes back as dicts:
  // kernels (func, name, grid, block, smem, image, layout, attrs) and
  // memsets (dst, value, elem, width), each with its kind.
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
      [](int64_t graph, int probe_attr) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto g = reinterpret_cast<cudaGraph_t>(static_cast<intptr_t>(graph));
        auto harvested = Exec::harvest_nodes(g, probe_attr);
        py::list out;
        for (const auto& h : harvested) {
          py::dict d;
          d["kind"] = h.kind == 0 ? "kernel" : "memset";
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
  m.def("_host_trace_capture_error_name", [](int64_t code) -> py::object {
    const char* name =
        at::cuda::host_trace::capture_error_name(static_cast<int>(code));
    return name == nullptr ? py::object(py::none()) : py::str(name);
  });
  // (addr, nbytes) of every caching-allocator allocation the build capture's
  // pool served between the two calls: how the replay's build binds each
  // allocation root
  m.def(
      "_host_trace_alloc_log_begin",
      [](c10::DeviceIndex device, c10::cuda::MempoolId_t pool) {
        at::cuda::host_trace::alloc_log_begin(device, pool);
      });
  m.def("_host_trace_alloc_log_end", []() {
    return at::cuda::host_trace::alloc_log_end();
  });
  // the table copies the ordinary host issued between the two calls, in
  // order: (pinned buffer, its bytes at the copy); the build checks the bytes
  // against the tape's image and binds the image's root to the buffer its
  // capture copied from
  m.def("_host_trace_host_table_log_begin", []() {
    at::cuda::host_trace::host_table_log_begin();
  });
  m.def("_host_trace_host_table_log_end", []() {
    py::list out;
    for (const auto& c : at::cuda::host_trace::host_table_log_end()) {
      out.append(py::make_tuple(
          c.buffer,
          py::bytes(
              reinterpret_cast<const char*>(c.bytes.data()), c.bytes.size())));
    }
    return out;
  });
  // Tensor::copy_'s rule for a pinned block a stream is still reading: the
  // caching host allocator defers the block's reuse to an event on that
  // stream. The replay records its staging slots this way after each launch.
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
