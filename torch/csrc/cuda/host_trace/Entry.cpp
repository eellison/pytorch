#include <torch/csrc/cuda/host_trace/Variant.h>

#if !defined(USE_ROCM)
#include <ATen/Context.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/DefaultDtype.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <structmember.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pythoncapi_compat.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <vector>

namespace torch::cuda::host_trace {

Outcome HostTraceVariant::commit(Frame& frame, PyObject** result) {
  PyObject* const* args = frame.args;
  c10::cuda::OptionalCUDAGuard guard;
  if (c10::cuda::current_device() != device_) {
    guard.set_index(device_);
    if (c10::cuda::currentStreamCaptureStatusMayInitCtx() !=
        c10::cuda::CaptureStatus::None) {
      return Outcome::Capture;
    }
  }
  frame.tensors.clear();
  frame.tensors.resize(base_count_);
  frame.bases.assign(base_count_, 0);
  bool ran = false;
  for (size_t i = 0; i <= steps_.size(); ++i) {
    const StepMemory& memory = memory_[i];
    const bool run = i < steps_.size() && steps_[i].segment >= 0;
    std::unique_lock<std::recursive_mutex> hold;
    at::Tensor scratch = allocate(
        memory, run ? &segments_[steps_[i].segment] : nullptr, frame, hold);
    bool misaligned = false;
    for (int64_t k : memory.tensors) {
      misaligned |= frame.bases[k] % kAlignment != 0;
    }
    for (const Temporary& tmp : memory.temporaries) {
      misaligned |= frame.bases[tmp.alloc] % kAlignment != 0;
    }
    if (misaligned) {
      if (!ran) {
        return Outcome::Misaligned;
      }
      PyErr_Format(
          disagreement_.ptr(),
          "an allocation that is not %lld-byte aligned",
          static_cast<long long>(kAlignment));
      throw py::error_already_set();
    }
    if (i < steps_.size()) {
      const Step& step = steps_[i];
      if (step.segment >= 0) {
        patch_and_replay(segments_[step.segment], frame);
      } else {
        run_eager(eager_[step.eager], args, frame);
      }
      ran = true;
    }
    // the buffer's bytes are freed after the run's graph is queued, so
    // stream-ordered after every kernel that uses them
    scratch.reset();
    for (int64_t b : memory.drops) {
      frame.tensors[b].reset();
    }
    if (frame.owned && !memory.arguments.empty()) {
      // a tensor's finalizer may run Python
      if (hold.owns_lock()) {
        hold.unlock();
      }
      for (int64_t a : memory.arguments) {
        PyObject* arg = PyTuple_GET_ITEM(frame.owned, a);
        PyTuple_SET_ITEM(frame.owned, a, Py_NewRef(Py_None));
        Py_DECREF(arg);
      }
    }
  }
  *result = outputs(args, frame);
  return Outcome::Hit;
}

py::tuple HostTraceVariant::call_py(py::handle args) {
  TORCH_CHECK_TYPE(PyTuple_Check(args.ptr()), "args must be a tuple");
  PyObject* const* inputs = &PyTuple_GET_ITEM(args.ptr(), 0);
  const auto count = static_cast<size_t>(PyTuple_GET_SIZE(args.ptr()));
  Frame frame;
  if (!evaluate(inputs, count, frame)) {
    return py::make_tuple(static_cast<int64_t>(Outcome::Miss), py::none());
  }
  PyObject* result = nullptr;
  const Outcome outcome = commit(frame, &result);
  return py::make_tuple(
      static_cast<int64_t>(outcome),
      result ? py::reinterpret_steal<py::object>(result) : py::none());
}

py::list HostTraceVariant::held_images() const {
  py::list out;
  for (const Record& r : records_) {
    if (!r.held) {
      out.append(py::none());
    } else if (r.kind == Kind::Memset) {
      const auto& m = r.held_memset;
      out.append(py::make_tuple(m[0], m[1], m[2], m[3]));
    } else if (r.kind == Kind::Memcpy) {
      const auto& m = r.held_copy;
      out.append(py::make_tuple(m[0], m[1], m[2]));
    } else {
      const KernelRow& k = kernel_row(r, r.held_row);
      py::tuple images(k.param_offsets.size());
      for (size_t i = 0; i < k.param_offsets.size(); ++i) {
        const auto* at = k.held + k.param_offsets[i];
        images[i] =
            py::bytes(reinterpret_cast<const char*>(at), k.param_sizes[i]);
      }
      const auto& g = r.held_dims;
      out.append(py::make_tuple(py::make_tuple(g[0], g[1], g[2]), images));
    }
  }
  return out;
}

size_t HostTraceVariant::python_steps() const {
  return std::count_if(
      eager_.begin(), eager_.end(), [](const EagerStep& s) { return !s.op; });
}

namespace {

// ---------------------------------------------------------------------------
// torch._C._HostTraceEntry: HostTraceReplay's base, whose call is a replay
// hit's whole path. Anything else is HostTraceReplay._call_slow's.

struct EntryVariant {
  py::object object; // the _HostTraceVariant
  HostTraceVariant* variant;
};

struct EntryFamily {
  std::vector<int64_t> key;
  std::vector<EntryVariant> variants;
};

struct EntryState {
  std::recursive_mutex mutex;
  bool configured = false;
  bool trusted = false;
  bool global_state = false; // check_global_state: in the contract key
  py::object active; // the tape's threading.local
  py::object disagreement;
  std::vector<EntryFamily> families;
};

struct HostTraceEntry {
  PyObject_HEAD
  EntryState* state;
  long long replays;
};

PyObject* interned(const char* s) {
  PyObject* p = PyUnicode_InternFromString(s);
  TORCH_INTERNAL_ASSERT(p);
  return p;
}

// The global state an eager call's kernels or its outputs depend on, which a
// replay does not read: a call under other state misses
void append_global_state(std::vector<int64_t>& key) {
  using at::Float32Backend;
  using at::Float32Op;
  auto& ctx = at::globalContext();
  key.insert(
      key.end(),
      {static_cast<int64_t>(c10::get_default_dtype_as_scalartype()),
       ctx.deterministicAlgorithms(),
       ctx.deterministicAlgorithmsWarnOnly(),
       ctx.deterministicFillUninitializedMemory(),
       at::GradMode::is_enabled(),
       c10::InferenceMode::is_enabled(),
       at::autocast::is_autocast_enabled(at::kCUDA),
       static_cast<int64_t>(at::autocast::get_autocast_dtype(at::kCUDA)),
       at::autocast::is_autocast_enabled(at::kCPU),
       static_cast<int64_t>(at::autocast::get_autocast_dtype(at::kCPU)),
       static_cast<int64_t>(
           ctx.float32Precision(Float32Backend::CUDA, Float32Op::MATMUL)),
       static_cast<int64_t>(ctx.allowFP16ReductionCuBLAS()),
       static_cast<int64_t>(ctx.allowBF16ReductionCuBLAS()),
       ctx.allowFP16AccumulationCuBLAS(),
       static_cast<int64_t>(ctx.blasPreferredBackend()),
       ctx._SMCarveout_EXPERIMENTAL().value_or(-1),
       ctx.userEnabledCuDNN(),
       static_cast<int64_t>(
           ctx.float32Precision(Float32Backend::CUDA, Float32Op::CONV)),
       static_cast<int64_t>(
           ctx.float32Precision(Float32Backend::CUDA, Float32Op::RNN)),
       ctx.deterministicCuDNN(),
       ctx.benchmarkCuDNN(),
       ctx.benchmarkLimitCuDNN(),
       ctx.userEnabledFlashSDP(),
       ctx.userEnabledFA3SDP(),
       ctx.userEnabledFA4SDP(),
       ctx.userEnabledMemEfficientSDP(),
       ctx.userEnabledMathSDP(),
       ctx.userEnabledCuDNNSDP(),
       ctx.userEnabledOverrideableSDP(),
       ctx.allowFP16BF16ReductionMathSDP()});
  for (at::SDPBackend b : ctx.sDPPriorityOrder()) {
    key.push_back(static_cast<int64_t>(b));
  }
}

// argument_contract(args, global_state) as ints, or false when an argument
// has no such form
bool contract_key(
    PyObject* const* args,
    size_t count,
    bool global_state,
    std::vector<int64_t>& key) {
  enum Tag : int64_t { Tensor = 1, Int, None, Bool, Float };
  key.clear();
  for (size_t i = 0; i < count; ++i) {
    PyObject* a = args[i];
    if (THPVariable_CheckExact(a)) {
      const at::Tensor& t = THPVariable_Unpack(a);
      key.insert(
          key.end(),
          {Tensor,
           reinterpret_cast<int64_t>(Py_TYPE(a)),
           static_cast<int64_t>(t.scalar_type()),
           static_cast<int64_t>(t.device().type()),
           t.device().index(),
           t.dim(),
           static_cast<int64_t>(t.layout()),
           t.is_neg(),
           t.is_conj(),
           t.requires_grad(),
           t.requires_grad() && at::GradMode::is_enabled()});
    } else if (PyLong_CheckExact(a)) {
      key.push_back(Int);
    } else if (a == Py_None) {
      key.push_back(None);
    } else if (PyBool_Check(a)) {
      key.insert(key.end(), {Bool, a == Py_True});
    } else if (PyFloat_CheckExact(a)) {
      const double d = PyFloat_AS_DOUBLE(a);
      int64_t bits = 0;
      std::memcpy(&bits, &d, sizeof(bits));
      key.insert(key.end(), {Float, bits});
    } else {
      return false;
    }
  }
  if (global_state) {
    append_global_state(key);
  }
  return true;
}

// Takes the mutex without holding the GIL while blocked: the owner may be in
// a callback that needs it
class EntryLock {
 public:
  explicit EntryLock(std::recursive_mutex& m) : m_(m) {
    if (!m_.try_lock()) {
      pybind11::gil_scoped_release no_gil;
      m_.lock();
    }
  }
  ~EntryLock() {
    m_.unlock();
  }
  EntryLock(const EntryLock&) = delete;
  EntryLock& operator=(const EntryLock&) = delete;

 private:
  std::recursive_mutex& m_;
};

PyObject* entry_slow(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs,
    bool searched) {
  static PyObject* name = interned("_call_slow");
  return PyObject_CallMethodObjArgs(
      self,
      name,
      args,
      kwargs ? kwargs : Py_None,
      searched ? Py_True : Py_False,
      nullptr);
}

PyObject* entry_new(PyTypeObject* type, PyObject* /*args*/, PyObject* /*kw*/) {
  PyObject* self = type->tp_alloc(type, 0);
  if (!self) {
    return nullptr;
  }
  auto* e = reinterpret_cast<HostTraceEntry*>(self);
  e->state = new (std::nothrow) EntryState();
  e->replays = 0;
  if (!e->state) {
    Py_DECREF(self);
    return PyErr_NoMemory();
  }
  return self;
}

int entry_traverse(PyObject* self, visitproc visit, void* arg) {
  EntryState* st = reinterpret_cast<HostTraceEntry*>(self)->state;
  if (!st) {
    return 0;
  }
  Py_VISIT(st->active.ptr());
  Py_VISIT(st->disagreement.ptr());
  for (const EntryFamily& f : st->families) {
    for (const EntryVariant& v : f.variants) {
      Py_VISIT(v.object.ptr());
    }
  }
  return 0;
}

int entry_clear(PyObject* self) {
  EntryState* st = reinterpret_cast<HostTraceEntry*>(self)->state;
  if (st) {
    // moved out first: a dropped object's finalizer may call the entry
    auto families = std::move(st->families);
    st->families.clear();
    py::object active = std::move(st->active);
    py::object disagreement = std::move(st->disagreement);
    st->configured = false;
  }
  return 0;
}

void entry_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  entry_clear(self);
  auto* e = reinterpret_cast<HostTraceEntry*>(self);
  delete e->state;
  e->state = nullptr;
  Py_TYPE(self)->tp_free(self);
}

// `owned`: the caller handed its references to the arguments over
PyObject* entry_dispatch(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs,
    bool owned) {
  auto* entry = reinterpret_cast<HostTraceEntry*>(self);
  EntryState& st = *entry->state;
  if (!st.configured) {
    return entry_slow(self, args, kwargs, false);
  }
  // inside an outer trace fn is traced inline
  static PyObject* trace_name = interned("trace");
  PyObject* tr = nullptr;
  if (PyObject_GetOptionalAttr(st.active.ptr(), trace_name, &tr) < 0) {
    return nullptr;
  }
  if (tr) {
    const bool tracing = tr != Py_None;
    Py_DECREF(tr);
    if (tracing) {
      static PyObject* fn_name = interned("fn");
      THPObjectPtr fn(PyObject_GetAttr(self, fn_name));
      return fn ? PyObject_Call(fn.get(), args, kwargs) : nullptr;
    }
  }
  EntryLock lock(st.mutex);
  if ((kwargs && PyDict_GET_SIZE(kwargs)) || st.families.empty()) {
    return entry_slow(self, args, kwargs, false);
  }
  PyObject* const* inputs = &PyTuple_GET_ITEM(args, 0);
  const auto count = static_cast<size_t>(PyTuple_GET_SIZE(args));
  const EntryFamily* family = nullptr;
  if (st.trusted) {
    family = &st.families.front();
  } else {
    thread_local std::vector<int64_t> key;
    if (contract_key(inputs, count, st.global_state, key)) {
      for (const EntryFamily& f : st.families) {
        if (f.key == key) {
          family = &f;
          break;
        }
      }
    }
  }
  if (!family ||
      c10::cuda::currentStreamCaptureStatusMayInitCtx() !=
          c10::cuda::CaptureStatus::None) {
    return entry_slow(self, args, kwargs, false);
  }
  Frame frame;
  for (const EntryVariant& ev : family->variants) {
    if (!ev.variant->evaluate(inputs, count, frame)) {
      continue;
    }
    // a call from an eager step may drop the variant or add to the families
    py::object held = ev.object;
    HostTraceVariant* variant = ev.variant;
    PyObject* result = nullptr;
    Outcome outcome = Outcome::Miss;
    frame.owned = owned && Py_REFCNT(args) == 1 ? args : nullptr;
    try {
      outcome = variant->commit(frame, &result);
    } catch (py::error_already_set& err) {
      err.restore();
      if (PyErr_ExceptionMatches(st.disagreement.ptr())) {
        PyObject *type = nullptr, *value = nullptr, *tb = nullptr;
        PyErr_Fetch(&type, &value, &tb);
        PyErr_NormalizeException(&type, &value, &tb);
        if (tb) {
          PyException_SetTraceback(value, tb);
        }
        static PyObject* name = interned("_disagreed");
        THPObjectPtr r(
            PyObject_CallMethodObjArgs(self, name, held.ptr(), value, nullptr));
        if (!r) {
          PyErr_WriteUnraisable(self);
        }
        PyErr_Restore(type, value, tb);
      }
      return nullptr;
    }
    if (outcome != Outcome::Hit) {
      static PyObject* name = interned("_unreplayed");
      THPObjectPtr code(PyLong_FromLongLong(static_cast<int64_t>(outcome)));
      return PyObject_CallMethodObjArgs(
          self, name, held.ptr(), code.get(), args, nullptr);
    }
    ++entry->replays;
    return result;
  }
  return entry_slow(self, args, kwargs, !frame.keyed_miss);
}

PyObject* entry_call(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  return entry_dispatch(self, args, kwargs, false);
  END_HANDLE_TH_ERRORS
}

// The call on a list of arguments it takes the references of and clears, as
// Inductor's boxed calls do
PyObject* entry_call_boxed(PyObject* self, PyObject* inputs) {
  HANDLE_TH_ERRORS
  TORCH_CHECK_TYPE(PyList_CheckExact(inputs), "call_boxed takes a list");
  THPObjectPtr args(PyList_AsTuple(inputs));
  if (!args ||
      PyList_SetSlice(inputs, 0, PyList_GET_SIZE(inputs), nullptr) < 0) {
    return nullptr;
  }
  return entry_dispatch(self, args.get(), nullptr, true);
  END_HANDLE_TH_ERRORS
}

PyObject* entry_native_init(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject *active = nullptr, *disagreement = nullptr;
  int trusted = 0, global_state = 0;
  if (!PyArg_ParseTuple(
          args, "OOpp", &active, &disagreement, &trusted, &global_state)) {
    return nullptr;
  }
  TORCH_CHECK_TYPE(
      PyType_Check(disagreement) &&
          PyType_IsSubtype(
              reinterpret_cast<PyTypeObject*>(disagreement),
              reinterpret_cast<PyTypeObject*>(PyExc_AssertionError)),
      "the disagreement must be an AssertionError subclass");
  EntryState& st = *reinterpret_cast<HostTraceEntry*>(self)->state;
  EntryLock lock(st.mutex);
  st.active = py::reinterpret_borrow<py::object>(active);
  st.disagreement = py::reinterpret_borrow<py::object>(disagreement);
  st.trusted = trusted;
  st.global_state = global_state;
  st.configured = true;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* entry_native_key(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK_TYPE(PyTuple_Check(args), "args must be a tuple");
  EntryState& st = *reinterpret_cast<HostTraceEntry*>(self)->state;
  std::vector<int64_t> key;
  if (!st.trusted &&
      !contract_key(
          &PyTuple_GET_ITEM(args, 0),
          PyTuple_GET_SIZE(args),
          st.global_state,
          key)) {
    Py_RETURN_NONE;
  }
  return py::cast(key).release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* entry_native_add(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject *key = nullptr, *native = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &key, &native)) {
    return nullptr;
  }
  EntryState& st = *reinterpret_cast<HostTraceEntry*>(self)->state;
  auto object = py::reinterpret_borrow<py::object>(native);
  auto* variant = object.cast<HostTraceVariant*>();
  auto k = py::handle(key).cast<std::vector<int64_t>>();
  EntryLock lock(st.mutex);
  TORCH_CHECK_VALUE(
      !st.trusted || k.empty(), "a trusted entry's variants have no key");
  for (EntryFamily& f : st.families) {
    if (f.key == k) {
      f.variants.push_back({std::move(object), variant});
      Py_RETURN_NONE;
    }
  }
  st.families.push_back({std::move(k), {{std::move(object), variant}}});
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* entry_native_remove(PyObject* self, PyObject* native) {
  HANDLE_TH_ERRORS
  EntryState& st = *reinterpret_cast<HostTraceEntry*>(self)->state;
  EntryLock lock(st.mutex);
  for (auto f = st.families.begin(); f != st.families.end(); ++f) {
    auto& vs = f->variants;
    for (auto v = vs.begin(); v != vs.end(); ++v) {
      if (v->object.ptr() == native) {
        vs.erase(v);
        if (vs.empty()) {
          st.families.erase(f);
        }
        Py_RETURN_NONE;
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyMethodDef entry_methods[] = {
    {"call_boxed", entry_call_boxed, METH_O, nullptr},
    {"_native_init", entry_native_init, METH_VARARGS, nullptr},
    {"_native_key", entry_native_key, METH_O, nullptr},
    {"_native_add", entry_native_add, METH_VARARGS, nullptr},
    {"_native_remove", entry_native_remove, METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMemberDef entry_members[] = {
    {"replays",
     T_LONGLONG,
     offsetof(HostTraceEntry, replays),
     0,
     "the calls a replay served"},
    {nullptr, 0, 0, 0, nullptr}};

PyTypeObject HostTraceEntryType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._HostTraceEntry",
    sizeof(HostTraceEntry),
};

} // namespace

} // namespace torch::cuda::host_trace
#endif

namespace torch::cuda {

void initHostTraceVariantBindings(PyObject* module) {
#if !defined(USE_ROCM)
  using namespace host_trace;
  auto m = py::handle(module).cast<py::module>();
  py::class_<HostTraceVariant>(m, "_HostTraceVariant")
      .def(py::init<py::handle>(), py::arg("spec"))
      .def("call", &HostTraceVariant::call_py, py::arg("args"))
      .def("evaluate", &HostTraceVariant::evaluate_py, py::arg("args"))
      .def("overlaps", &HostTraceVariant::overlaps_py, py::arg("args"))
      .def(
          "add_row",
          &HostTraceVariant::add_row,
          py::arg("site"),
          py::arg("key"),
          py::arg("nodes"),
          py::arg("arm") = 0,
          py::arg("piece") = false,
          py::arg("scratch") = std::vector<int64_t>())
      .def(
          "add_entry",
          &HostTraceVariant::add_entry,
          py::arg("site"),
          py::arg("predicate"),
          py::arg("nodes"))
      .def(
          "set_program",
          &HostTraceVariant::set_program,
          py::arg("program"))
      .def(
          "add_form",
          &HostTraceVariant::add_form,
          py::arg("segment"),
          py::arg("arms"),
          py::arg("exec"),
          py::arg("graph"),
          py::arg("pieces"))
      .def_property_readonly("python_steps", &HostTraceVariant::python_steps)
      .def_readonly("python_calls", &HostTraceVariant::python_calls);
#if defined(__linux__)
  initHarvestBindings(m);
#endif
  initHostTraceAtenBindings(m);
  m.def("_host_trace_global_state", [] {
    std::vector<int64_t> state;
    append_global_state(state);
    return py::tuple(py::cast(state));
  });
  // test-only
  m.def("_host_trace_held_images", [](const HostTraceVariant& v) {
    return v.held_images();
  });
  m.def("_host_trace_fail_after_setter", [](int64_t n) {
    fail_after_setter = n;
  });
  m.def("_host_trace_tma_counts", [] {
    return py::make_tuple(tma_encodes, tma_replaces);
  });
  m.def("_host_trace_buffered_runs", [] { return buffered_runs; });
  m.def("_host_trace_memory_node_sets", [] { return memory_node_sets; });
  PyTypeObject& t = HostTraceEntryType;
  t.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC;
  t.tp_doc = "HostTraceReplay's base: a replay hit's call in C++";
  t.tp_new = entry_new;
  t.tp_dealloc = entry_dealloc;
  t.tp_traverse = entry_traverse;
  t.tp_clear = entry_clear;
  t.tp_call = entry_call;
  t.tp_methods = entry_methods;
  t.tp_members = entry_members;
  if (PyType_Ready(&t) < 0) {
    throw py::error_already_set();
  }
  Py_INCREF(&t);
  if (PyModule_AddObject(module, "_HostTraceEntry", (PyObject*)&t) < 0) {
    throw py::error_already_set();
  }
#endif
}

} // namespace torch::cuda
