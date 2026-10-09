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
#include <c10/util/ScopeExit.h>
#include <structmember.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pythoncapi_compat.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>
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
  int64_t pt_misc = phase_now();
  mark_dirty(frame);
  // a commit from an eager step's call leaves the outer's nodes at its rows
  ++committing_;
  auto nested = c10::make_scope_exit([this] {
    if (--committing_ > 0) {
      std::fill(dirty_.begin(), dirty_.end(), 1);
      seen_valid_ = false;
    }
  });
  frame.tensors.clear();
  frame.tensors.resize(base_count_);
  frame.bases.assign(base_count_, 0);
  phase_add(13, pt_misc);
  int64_t pt = phase_now();
  PlanLease lease;
  if (planned_bytes_ >= 0) {
    plan(frame, lease);
  }
  phase_add(10, pt_misc);
  phase_add(4, pt);
  bool ran = false;
  for (size_t i = 0; i <= steps_.size(); ++i) {
    const StepMemory& memory = memory_[i];
    const bool run = i < steps_.size() && steps_[i].segment >= 0;
    std::unique_lock<std::recursive_mutex> hold;
    pt = phase_now();
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
    phase_add(4, pt);
    if (i < steps_.size()) {
      const Step& step = steps_[i];
      if (step.segment >= 0) {
        patch_and_replay(segments_[step.segment], frame);
      } else {
        run_eager(eager_[step.eager], args, frame);
      }
      ran = true;
    }
    pt_misc = phase_now();
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
    phase_add(13, pt_misc);
  }
  pt = phase_now();
  *result = outputs(args, frame);
  phase_add(7, pt);
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
  // the variant that served a hash of the dynamic tensors' sizes, and the
  // last one that served
  std::unordered_map<uint64_t, size_t> served;
  size_t last = 0;
};

uint64_t sizes_hash(PyObject* const* args, size_t count) {
  uint64_t h = 0xcbf29ce484222325;
  for (size_t i = 0; i < count; ++i) {
    if (!THPVariable_CheckExact(args[i])) {
      continue;
    }
    const c10::TensorImpl* impl =
        THPVariable_Unpack(args[i]).unsafeGetTensorImpl();
    if (impl->has_symbolic_sizes_strides()) {
      continue;
    }
    h = (h ^ (i << 8 | static_cast<uint64_t>(impl->dim()))) * 0x100000001b3;
    for (int64_t s : impl->sizes()) {
      h = (h ^ static_cast<uint64_t>(s)) * 0x100000001b3;
    }
  }
  return h;
}

// A leading argument as the contract key and the rows read it, recorded at
// a hit; the object is not held (only compared)
struct StaticArg {
  PyObject* object;
  PyTypeObject* type;
  c10::TensorImpl* impl;
  c10::StorageImpl* storage;
  void* data;
  int64_t storage_offset;
  c10::DispatchKeySet keys;
  caffe2::TypeMeta dtype;
  c10::Device device{c10::kCPU};
  bool requires_grad;
  c10::SmallVector<int64_t, 8> shape; // sizes, then strides
};

// A bound call trusted to pass the same leading arguments, unchanged, until
// its statics_changed(): its id (0: none) and that call's count
struct StaticOwner {
  uint64_t id = 0;
  uint64_t version = 0;
};

struct EntryState {
  std::recursive_mutex mutex;
  bool configured = false;
  bool trusted = false;
  bool global_state = false; // check_global_state: in the contract key
  py::object active; // the tape's threading.local
  py::object disagreement;
  // fn's parameters a keyword can pass, by position (None: positional-only)
  std::vector<py::object> names;
  // the constant arguments of the families' keys, which key them by index
  std::vector<py::object> constants;
  std::vector<EntryFamily> families;
  // static_prefix: the leading arguments of the last hit whose rows were all
  // evaluated, and the family it hit in. A call whose leading arguments
  // match them keys only the others and takes their rows from the variant's
  // kept call of the same generation (0: none).
  size_t static_count = 0;
  std::vector<StaticArg> statics;
  bool static_grad_mode = false;
  size_t static_family = 0;
  uint64_t static_generation = 0;
  // the bound call that last checked or recorded them
  StaticOwner static_owner;
  // bumped by each change of the families
  uint64_t epoch = 0;
};

struct HostTraceEntry {
  PyObject_HEAD
  EntryState* state;
  long long replays;
  long long static_hits;
  long long slow_calls;
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

bool is_sequence(PyObject* a) {
  return PyList_Check(a) || PyTuple_Check(a);
}

// _constant(a) == _constant(b): a list or tuple by its elements, a float or
// complex by its bits, a tensor not at all (false), anything else by type and
// ==; -1 with an error set
int same_constant(PyObject* a, PyObject* b) {
  if (is_sequence(a) || is_sequence(b)) {
    if (!is_sequence(a) || !is_sequence(b) ||
        PySequence_Fast_GET_SIZE(a) != PySequence_Fast_GET_SIZE(b)) {
      return 0;
    }
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(a) &&
         i < PySequence_Fast_GET_SIZE(b);
         ++i) {
      const int same = same_constant(
          PySequence_Fast_GET_ITEM(a, i), PySequence_Fast_GET_ITEM(b, i));
      if (same != 1) {
        return same;
      }
    }
    return PySequence_Fast_GET_SIZE(a) == PySequence_Fast_GET_SIZE(b);
  }
  if (THPVariable_Check(a) || THPVariable_Check(b)) {
    return PyErr_Occurred() ? -1 : 0;
  }
  if (PyFloat_Check(a) || PyFloat_Check(b)) {
    if (!PyFloat_Check(a) || !PyFloat_Check(b)) {
      return 0;
    }
    const double x = PyFloat_AS_DOUBLE(a), y = PyFloat_AS_DOUBLE(b);
    return std::memcmp(&x, &y, sizeof(x)) == 0;
  }
  if (PyComplex_Check(a) || PyComplex_Check(b)) {
    if (!PyComplex_Check(a) || !PyComplex_Check(b)) {
      return 0;
    }
    const Py_complex x = PyComplex_AsCComplex(a), y = PyComplex_AsCComplex(b);
    return std::memcmp(&x.real, &y.real, sizeof(double)) == 0 &&
        std::memcmp(&x.imag, &y.imag, sizeof(double)) == 0;
  }
  return Py_TYPE(a) == Py_TYPE(b) ? PyObject_RichCompareBool(a, b, Py_EQ)
                                  : 0;
}

// a constant as a key holds it: lists and tuples as tuples, so a list the
// caller changes later does not change the key
PyObject* frozen_constant(PyObject* a) {
  if (!is_sequence(a)) {
    return Py_NewRef(a);
  }
  const Py_ssize_t n = PySequence_Fast_GET_SIZE(a);
  THPObjectPtr t(PyTuple_New(n));
  for (Py_ssize_t i = 0; t && i < n; ++i) {
    PyObject* item = frozen_constant(PySequence_Fast_GET_ITEM(a, i));
    if (!item) {
      return nullptr;
    }
    PyTuple_SET_ITEM(t.get(), i, item);
  }
  return t.release();
}

// argument_contract(args, global_state) as ints, or false when an argument
// has no such form. An argument of no other kind (a str, a tuple, a dtype,
// ...) is keyed by its index among `constants` by value; `intern` adds one
// not there.
bool contract_key(
    PyObject* const* args,
    size_t count,
    bool global_state,
    std::vector<int64_t>& key,
    std::vector<py::object>& constants,
    bool intern) {
  enum Tag : int64_t { Tensor = 1, Int, None, Bool, Float, Constant };
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
           // a pinned CPU tensor (a memcpy's host argument) is its own kind
           static_cast<int64_t>(t.device().type()) |
               static_cast<int64_t>(t.is_cpu() && t.is_pinned()) << 16,
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
    } else if (!THPVariable_Check(a)) {
      size_t j = 0;
      int same = 0;
      while (j < constants.size() &&
             (same = same_constant(a, constants[j].ptr())) == 0) {
        ++j;
      }
      if (same < 0) {
        PyErr_Clear();
        return false;
      }
      if (j == constants.size()) {
        PyObject* c = intern ? frozen_constant(a) : nullptr;
        if (!c) {
          PyErr_Clear();
          return false;
        }
        constants.push_back(py::reinterpret_steal<py::object>(c));
      }
      key.insert(key.end(), {Constant, static_cast<int64_t>(j)});
    } else {
      PyErr_Clear();
      return false;
    }
  }
  if (global_state) {
    append_global_state(key);
  }
  return true;
}

bool record_static(PyObject* a, StaticArg& s) {
  if (!THPVariable_CheckExact(a)) {
    return false;
  }
  const at::Tensor& t = THPVariable_Unpack(a);
  c10::TensorImpl* impl = t.unsafeGetTensorImpl();
  if (t.layout() != at::kStrided || t.is_nested() || !impl->has_storage() ||
      impl->has_symbolic_sizes_strides()) {
    return false;
  }
  s.object = a;
  s.type = Py_TYPE(a);
  s.impl = impl;
  s.storage = impl->unsafe_storage().unsafeGetStorageImpl();
  s.data = s.storage->_mutable_data_ptr_no_checks().get();
  s.storage_offset = impl->storage_offset();
  s.keys = impl->key_set();
  s.dtype = impl->dtype();
  s.device = impl->device();
  s.requires_grad = impl->requires_grad();
  s.shape.assign(impl->sizes().begin(), impl->sizes().end());
  s.shape.append(impl->strides().begin(), impl->strides().end());
  return true;
}

bool same_static(PyObject* a, const StaticArg& s) {
  if (a != s.object || Py_TYPE(a) != s.type) {
    return false;
  }
  c10::TensorImpl* impl = THPVariable_Unpack(a).unsafeGetTensorImpl();
  if (impl != s.impl || impl->key_set() != s.keys || impl->dtype() != s.dtype ||
      !impl->has_storage() ||
      impl->unsafe_storage().unsafeGetStorageImpl() != s.storage ||
      s.storage->_mutable_data_ptr_no_checks().get() != s.data ||
      impl->storage_offset() != s.storage_offset ||
      impl->device() != s.device ||
      impl->requires_grad() != s.requires_grad ||
      impl->has_symbolic_sizes_strides()) {
    return false;
  }
  const auto sizes = impl->sizes();
  const auto strides = impl->strides();
  const size_t dim = sizes.size();
  return s.shape.size() == 2 * dim &&
      std::equal(sizes.begin(), sizes.end(), s.shape.begin()) &&
      std::equal(strides.begin(), strides.end(), s.shape.begin() + dim);
}

// Whether the leading arguments are the snapshot's
bool same_statics(const EntryState& st, PyObject* const* args, size_t count) {
  if (st.static_generation == 0 || count < st.static_count ||
      at::GradMode::is_enabled() != st.static_grad_mode) {
    return false;
  }
  // the recorded pointers are loads the checks do not wait on: fetched
  // ahead, the misses on them overlap
  constexpr size_t ahead = 8;
  const StaticArg* s = st.statics.data();
  const size_t n = st.static_count;
  for (size_t i = 0; i < n; ++i) {
    if (i + ahead < n) {
      const StaticArg& p = s[i + ahead];
      const auto* impl = reinterpret_cast<const char*>(p.impl);
      __builtin_prefetch(p.object);
      __builtin_prefetch(impl);
      __builtin_prefetch(impl + 64);
      __builtin_prefetch(impl + 128);
      __builtin_prefetch(impl + sizeof(c10::TensorImpl) - 1);
      __builtin_prefetch(p.storage);
    }
    if (!same_static(args[i], s[i])) {
      return false;
    }
  }
  return true;
}

// A trust_statics hit keeps the rows the statics' last full call evaluated,
// addresses included (only an eager step reads its arguments again). Between
// calls, re-pointing a static (set_, .data =, swap_tensors) or changing its
// layout in place (t_, as_strided_, set_ with new strides, offset or sizes,
// resize_) is unsupported unless statics_changed() follows it. The check
// below is for debugging only.
bool trusted_statics_check = false;

bool same_static_layout(const EntryState& st, PyObject* const* args) {
  // fetched ahead as in same_statics
  constexpr size_t ahead = 8;
  const StaticArg* s = st.statics.data();
  const size_t n = st.static_count;
  for (size_t i = 0; i < n; ++i) {
    if (i + ahead < n) {
      const StaticArg& p = s[i + ahead];
      const auto* impl = reinterpret_cast<const char*>(p.impl);
      __builtin_prefetch(impl);
      __builtin_prefetch(impl + 64);
      __builtin_prefetch(impl + 128);
      __builtin_prefetch(p.storage);
      __builtin_prefetch(reinterpret_cast<const char*>(p.object) + 16);
    }
    if (args[i] != s[i].object) {
      return false;
    }
    c10::TensorImpl* impl = THPVariable_Unpack(args[i]).unsafeGetTensorImpl();
    if (impl != s[i].impl || impl->dtype() != s[i].dtype ||
        !impl->has_storage() ||
        impl->unsafe_storage().unsafeGetStorageImpl() != s[i].storage ||
        s[i].storage->_mutable_data_ptr_no_checks().get() != s[i].data ||
        impl->storage_offset() != s[i].storage_offset) {
      return false;
    }
    const auto sizes = impl->sizes();
    const auto strides = impl->strides();
    const size_t dim = sizes.size();
    if (s[i].shape.size() != 2 * dim ||
        !std::equal(sizes.begin(), sizes.end(), s[i].shape.begin()) ||
        !std::equal(strides.begin(), strides.end(), s[i].shape.begin() + dim)) {
      return false;
    }
  }
  return true;
}

bool record_statics(
    PyObject* const* args,
    size_t count,
    size_t n,
    std::vector<StaticArg>& out) {
  if (count < n) {
    return false;
  }
  out.resize(n);
  for (size_t i = 0; i < n; ++i) {
    if (!record_static(args[i], out[i])) {
      return false;
    }
  }
  return true;
}

void forget_statics(EntryState& st) {
  st.static_generation = 0;
  st.static_owner = StaticOwner{};
  ++st.epoch;
}

// Releases a frame's use of a variant's cached rows
struct ReleaseCache {
  HostTraceVariant* variant;
  Frame& frame;
  ~ReleaseCache() {
    variant->release(frame);
  }
};

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
  e->static_hits = 0;
  e->slow_calls = 0;
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
  for (const py::object& c : st->constants) {
    Py_VISIT(c.ptr());
  }
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
    auto constants = std::move(st->constants);
    st->constants.clear();
    auto names = std::move(st->names);
    st->names.clear();
    st->configured = false;
    forget_statics(*st);
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

// The call on inputs[0, count), `args` their tuple. `owned`: the caller
// handed its references to the arguments over. A bound call's (no tuple)
// returns null with no error set where a tuple's would take the slow path.
PyObject* entry_dispatch(
    PyObject* self,
    PyObject* const* inputs,
    size_t count,
    PyObject* args,
    PyObject* kwargs,
    bool owned,
    const StaticOwner& owner = StaticOwner{}) {
  auto* entry = reinterpret_cast<HostTraceEntry*>(self);
  EntryState& st = *entry->state;
  int64_t pt = phase_now();
  int64_t pt_total = pt;
  auto slow = [&](bool searched) -> PyObject* {
    if (!args) {
      return nullptr;
    }
    ++entry->slow_calls;
    return entry_slow(self, args, kwargs, searched);
  };
  if (!st.configured) {
    return slow(false);
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
      if (!args) {
        return nullptr;
      }
      static PyObject* fn_name = interned("fn");
      THPObjectPtr fn(PyObject_GetAttr(self, fn_name));
      return fn ? PyObject_Call(fn.get(), args, kwargs) : nullptr;
    }
  }
  if (kwargs && PyDict_GET_SIZE(kwargs)) {
    // keywords that name the parameters after the positional arguments bind
    // to them, as _positional binds them
    const size_t k = PyDict_GET_SIZE(kwargs);
    if (!args || count + k > st.names.size()) {
      return slow(false);
    }
    THPObjectPtr full(PyTuple_New(static_cast<Py_ssize_t>(count + k)));
    if (!full) {
      return nullptr;
    }
    for (size_t i = 0; i < count + k; ++i) {
      PyObject* v = i < count
          ? inputs[i]
          : PyDict_GetItemWithError(kwargs, st.names[i].ptr());
      if (!v) {
        return PyErr_Occurred() ? nullptr : slow(false);
      }
      PyTuple_SET_ITEM(full.get(), i, Py_NewRef(v));
    }
    return entry_dispatch(
        self,
        &PyTuple_GET_ITEM(full.get(), 0),
        count + k,
        full.get(),
        nullptr,
        false,
        owner);
  }
  EntryLock lock(st.mutex);
  if (st.families.empty()) {
    return slow(false);
  }
  EntryFamily* family = nullptr;
  // the snapshot's generation when the leading arguments are its
  uint64_t generation = 0;
  thread_local std::vector<int64_t> key;
  const bool trusted_statics = owner.id &&
      owner.id == st.static_owner.id &&
      owner.version == st.static_owner.version && st.static_generation &&
      count >= st.static_count &&
      at::GradMode::is_enabled() == st.static_grad_mode &&
      (!trusted_statics_check || same_static_layout(st, inputs));
  if (trusted_statics || same_statics(st, inputs, count)) {
    if (owner.id) {
      st.static_owner = owner;
    }
    EntryFamily& f = st.families[st.static_family];
    // a tensor's contract key is 11 ints
    const size_t fixed = 11 * st.static_count;
    if (st.trusted ||
        (contract_key(
             inputs + st.static_count,
             count - st.static_count,
             st.global_state,
             key,
             st.constants,
             false) &&
         f.key.size() == fixed + key.size() &&
         std::equal(key.begin(), key.end(), f.key.begin() + fixed))) {
      family = &f;
      generation = st.static_generation;
    }
  }
  if (!family && st.trusted) {
    family = &st.families.front();
  } else if (!family) {
    if (contract_key(
            inputs, count, st.global_state, key, st.constants, false)) {
      for (EntryFamily& f : st.families) {
        if (f.key == key) {
          family = &f;
          break;
        }
      }
    }
  }
  int64_t pt_capture = phase_now();
  if (!family ||
      c10::cuda::currentStreamCaptureStatusMayInitCtx() !=
          c10::cuda::CaptureStatus::None) {
    return slow(false);
  }
  phase_add(14, pt_capture);
  const size_t n = family->variants.size();
  uint64_t shape = 0;
  size_t first = 0;
  if (n > 1 && variant_order_enabled) {
    shape = sizes_hash(inputs + st.static_count, count - st.static_count);
    auto it = family->served.find(shape);
    first = it != family->served.end() ? it->second : family->last;
    first = first < n ? first : 0;
  }
  phase_add(1, pt);
  Frame frame;
  frame.uncaptured = true;
  for (size_t k = 0; k < n; ++k) {
    const size_t i = k == 0 ? first : (k <= first ? k - 1 : k);
    const EntryVariant& ev = family->variants[i];
    ReleaseCache release{ev.variant, frame};
    const auto phases = phase_ns;
    int64_t pt_variant = phase_now();
    if (!ev.variant->evaluate_static(inputs, count, frame, generation)) {
      if (phase_timing) {
        phase_ns = phases;
        phase_add(9, pt_variant);
      }
      continue;
    }
    if (n > 1 && variant_order_enabled) {
      if (family->served.size() >= 1024) {
        family->served.clear();
      }
      family->served[shape] = i;
      family->last = i;
    }
    // a call from an eager step may drop the variant or add to the families
    py::object held = ev.object;
    HostTraceVariant* variant = ev.variant;
    PyObject* result = nullptr;
    Outcome outcome = Outcome::Miss;
    frame.owned = owned && args && Py_REFCNT(args) == 1 ? args : nullptr;
    // the leading arguments as the rows read them, before an eager step can
    // change them
    std::vector<StaticArg> records;
    const bool cached = frame.cached != nullptr;
    const bool recorded = !generation && st.static_count &&
        record_statics(inputs, count, st.static_count, records);
    const bool grad_mode = at::GradMode::is_enabled();
    const size_t family_index = family - st.families.data();
    const uint64_t epoch = st.epoch;
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
      if (!args) {
        return nullptr;
      }
      ++entry->slow_calls;
      static PyObject* name = interned("_unreplayed");
      THPObjectPtr code(PyLong_FromLongLong(static_cast<int64_t>(outcome)));
      return PyObject_CallMethodObjArgs(
          self, name, held.ptr(), code.get(), args, nullptr);
    }
    ++entry->replays;
    if (cached) {
      ++entry->static_hits;
    } else if (st.epoch == epoch) {
      if (generation && generation == st.static_generation) {
        variant->keep(frame, generation);
      } else if (recorded) {
        static uint64_t generations = 0;
        st.statics.swap(records);
        st.static_grad_mode = grad_mode;
        st.static_family = family_index;
        st.static_generation = ++generations;
        st.static_owner = owner;
        variant->keep(frame, st.static_generation);
      }
    }
    phase_add(8, pt_total);
    return result;
  }
  return slow(!frame.keyed_miss);
}

PyObject* entry_call(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  return entry_dispatch(
      self,
      &PyTuple_GET_ITEM(args, 0),
      PyTuple_GET_SIZE(args),
      args,
      kwargs,
      false);
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
  return entry_dispatch(
      self,
      &PyTuple_GET_ITEM(args.get(), 0),
      PyTuple_GET_SIZE(args.get()),
      args.get(),
      nullptr,
      true);
  END_HANDLE_TH_ERRORS
}

PyObject* entry_native_init(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject *active = nullptr, *disagreement = nullptr;
  int trusted = 0, global_state = 0;
  Py_ssize_t static_count = 0;
  PyObject* names = nullptr;
  if (!PyArg_ParseTuple(
          args,
          "OOpp|nO!",
          &active,
          &disagreement,
          &trusted,
          &global_state,
          &static_count,
          &PyTuple_Type,
          &names)) {
    return nullptr;
  }
  TORCH_CHECK_VALUE(static_count >= 0, "a negative static prefix");
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
  st.static_count = static_count;
  st.names.clear();
  for (Py_ssize_t i = 0; names && i < PyTuple_GET_SIZE(names); ++i) {
    st.names.push_back(
        py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(names, i)));
  }
  forget_statics(st);
  for (EntryFamily& f : st.families) {
    for (EntryVariant& v : f.variants) {
      v.variant->set_static_prefix(st.static_count);
    }
  }
  st.configured = true;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* entry_native_key(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK_TYPE(PyTuple_Check(args), "args must be a tuple");
  EntryState& st = *reinterpret_cast<HostTraceEntry*>(self)->state;
  EntryLock lock(st.mutex);
  std::vector<int64_t> key;
  if (!st.trusted &&
      !contract_key(
          &PyTuple_GET_ITEM(args, 0),
          PyTuple_GET_SIZE(args),
          st.global_state,
          key,
          st.constants,
          true)) {
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
  variant->set_static_prefix(st.static_count);
  forget_statics(st);
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
  forget_statics(st);
  for (auto f = st.families.begin(); f != st.families.end(); ++f) {
    auto& vs = f->variants;
    for (auto v = vs.begin(); v != vs.end(); ++v) {
      if (v->object.ptr() == native) {
        vs.erase(v);
        f->served.clear();
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
    {"static_hits",
     T_LONGLONG,
     offsetof(HostTraceEntry, static_hits),
     0,
     "the replays whose leading arguments' rows were the last hit's"},
    {"slow_calls",
     T_LONGLONG,
     offsetof(HostTraceEntry, slow_calls),
     0,
     "the calls handed to Python (_call_slow or _unreplayed)"},
    {nullptr, 0, 0, 0, nullptr}};

PyTypeObject HostTraceEntryType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._HostTraceEntry",
    sizeof(HostTraceEntry),
};

// ---------------------------------------------------------------------------
// torch._C._HostTraceBound: a caller's argument build in C++. It holds the
// leading arguments of its entries' calls, bound once, and per entry a plan:
// the instance attributes of the call's objects (and of its roots) the other
// arguments are, the guards under which they are what the caller's own build
// would pass that entry, and its output's template. A call no plan's guards
// hold for, or one that is not a replay hit, returns NotImplemented having
// run nothing.

struct BoundRead {
  size_t source; // a call object, then a root
  PyObject* name; // the plan's spec holds it
};

struct BoundBelow {
  BoundRead value;
  BoundRead index;
  std::vector<int64_t> limits;
};

struct BoundPlan {
  py::object spec;
  py::object entry;
  std::vector<PyTypeObject*> types; // per call object
  std::vector<std::pair<BoundRead, PyObject*>> same; // absent reads as None
  std::vector<BoundBelow> below; // value < limits[index], both ints
  std::vector<BoundRead> other; // no CUDA or pinned tensor
  std::vector<BoundRead> tensors; // CUDA or pinned CPU tensors
  std::vector<BoundRead> ints;
  py::object type; // the output's, or None: the flat outputs
  py::object fields; // its template's attributes
  std::vector<PyObject*> outputs; // the attributes the flat outputs are
  long long hits = 0;
};

struct BoundState {
  py::tuple prefix;
  size_t leading = 0; // the call's arguments before its objects
  size_t objects = 0;
  py::tuple roots;
  std::vector<BoundPlan> plans;
  std::vector<PyObject*> args; // the prefix, then a call's
  bool busy = false;
  // the objects read off the roots (source: a root), after the call's
  std::vector<BoundRead> derived;
  std::vector<PyObject*> none_kwargs; // keywords a call may pass as None
  // after a hit, setattr(source, name, value)
  std::vector<std::pair<BoundRead, PyObject*>> resets;
  py::object options; // (derived, none_kwargs, resets): holds their names
  py::object fallback; // called with a call no plan takes (None: none)
  StaticOwner owner;
};

struct HostTraceBound {
  PyObject_HEAD
  vectorcallfunc vectorcall;
  BoundState* state;
};

// by type: THPVariable_Check sends a non-tensor through the metaclass's
// __instancecheck__, and most of a plan's reads are not tensors. A pinned
// CPU tensor is an argument too (a memcpy's host side)
bool is_argument_tensor(PyObject* v) {
  if (!v || !THPVariableClass ||
      !PyType_IsSubtype(
          Py_TYPE(v), reinterpret_cast<PyTypeObject*>(THPVariableClass))) {
    return false;
  }
  const at::Tensor& t = THPVariable_Unpack(v);
  return t.is_cuda() || (t.is_cpu() && t.is_pinned());
}

bool bound_matches(
    const BoundPlan& p,
    PyObject* const* objects,
    PyObject* const* dicts) {
  for (size_t i = 0; i < p.types.size(); ++i) {
    if (Py_TYPE(objects[i]) != p.types[i]) {
      return false;
    }
  }
  auto get = [&](const BoundRead& r) {
    return PyDict_GetItemWithError(dicts[r.source], r.name);
  };
  for (const auto& [r, want] : p.same) {
    PyObject* v = get(r);
    if ((v ? v : Py_None) != want) {
      return false;
    }
  }
  for (const BoundBelow& b : p.below) {
    PyObject* v = get(b.value);
    PyObject* i = get(b.index);
    if (!v || !i || !PyLong_CheckExact(v) || !PyLong_CheckExact(i)) {
      return false;
    }
    int overflow = 0;
    const long long x = PyLong_AsLongLongAndOverflow(v, &overflow);
    if (overflow) {
      return false;
    }
    const long long k = PyLong_AsLongLongAndOverflow(i, &overflow);
    if (overflow || k < 0 || static_cast<size_t>(k) >= b.limits.size() ||
        x >= b.limits[k]) {
      return false;
    }
  }
  for (const BoundRead& r : p.other) {
    if (is_argument_tensor(get(r))) {
      return false;
    }
  }
  for (const BoundRead& r : p.tensors) {
    if (!is_argument_tensor(get(r))) {
      return false;
    }
  }
  for (const BoundRead& r : p.ints) {
    PyObject* v = get(r);
    if (!v || !PyLong_CheckExact(v)) {
      return false;
    }
  }
  return true;
}

PyObject* bound_output(const BoundPlan& p, PyObject* flat) {
  THPObjectPtr owned(flat);
  if (p.type.is_none()) {
    return owned.release();
  }
  TORCH_CHECK(
      PyTuple_CheckExact(flat) &&
          static_cast<size_t>(PyTuple_GET_SIZE(flat)) == p.outputs.size(),
      "host_trace: a bound entry's outputs are not its template's");
  static PyObject* empty = PyTuple_New(0);
  auto* type = reinterpret_cast<PyTypeObject*>(p.type.ptr());
  THPObjectPtr out(PyBaseObject_Type.tp_new(type, empty, nullptr));
  if (!out) {
    return nullptr;
  }
  THPObjectPtr dict(PyObject_GenericGetDict(out.get(), nullptr));
  if (!dict || PyDict_Update(dict.get(), p.fields.ptr()) < 0) {
    return nullptr;
  }
  for (size_t i = 0; i < p.outputs.size(); ++i) {
    if (PyDict_SetItem(dict.get(), p.outputs[i], PyTuple_GET_ITEM(flat, i)) <
        0) {
      return nullptr;
    }
  }
  return out.release();
}

PyObject* bound_call(
    PyObject* self,
    PyObject* const* args,
    size_t nargsf,
    PyObject* kwnames) {
  HANDLE_TH_ERRORS
  BoundState& st = *reinterpret_cast<HostTraceBound*>(self)->state;
  int64_t pt = phase_now();
  const auto n = static_cast<size_t>(PyVectorcall_NARGS(nargsf));
  if (n != st.leading + st.objects) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  const Py_ssize_t kw = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
  for (Py_ssize_t i = 0; i < kw; ++i) {
    PyObject* name = PyTuple_GET_ITEM(kwnames, i);
    if (args[n + i] != Py_None ||
        std::none_of(
            st.none_kwargs.begin(), st.none_kwargs.end(), [&](PyObject* k) {
              return k == name || PyUnicode_Compare(k, name) == 0;
            })) {
      Py_RETURN_NOTIMPLEMENTED;
    }
  }
  c10::SmallVector<PyObject*, 4> objects(args + st.leading, args + n);
  c10::SmallVector<THPObjectPtr, 2> held_derived;
  for (const BoundRead& r : st.derived) {
    THPObjectPtr d(PyObject_GenericGetDict(
        PyTuple_GET_ITEM(st.roots.ptr(), r.source), nullptr));
    PyObject* v = d ? PyDict_GetItemWithError(d.get(), r.name) : nullptr;
    if (!v) {
      PyErr_Clear();
      Py_RETURN_NOTIMPLEMENTED;
    }
    held_derived.emplace_back(Py_NewRef(v));
    objects.push_back(v);
  }
  for (py::handle root : st.roots) {
    objects.push_back(root.ptr());
  }
  c10::SmallVector<THPObjectPtr, 4> held_dicts;
  c10::SmallVector<PyObject*, 4> dicts;
  for (PyObject* o : objects) {
    PyObject* d = PyObject_GenericGetDict(o, nullptr);
    if (!d) {
      PyErr_Clear();
      Py_RETURN_NOTIMPLEMENTED;
    }
    held_dicts.emplace_back(d);
    dicts.push_back(d);
  }
  int64_t pt_match = phase_now();
  for (size_t k = 0; k < st.plans.size(); ++k) {
    const BoundPlan& p = st.plans[k];
    if (!bound_matches(p, objects.data(), dicts.data())) {
      if (PyErr_Occurred()) {
        return nullptr;
      }
      continue;
    }
    phase_add(15, pt_match);
    // held: an eager step's Python may replace the attributes, or add plans
    c10::SmallVector<THPObjectPtr, 32> held;
    const py::object entry = p.entry;
    const auto fixed = static_cast<size_t>(PyTuple_GET_SIZE(st.prefix.ptr()));
    // a call from within a call builds its own
    std::vector<PyObject*> nested;
    const bool busy = st.busy;
    if (busy) {
      nested.assign(st.args.begin(), st.args.begin() + fixed);
    }
    std::vector<PyObject*>& buffer = busy ? nested : st.args;
    buffer.resize(fixed);
    for (size_t i = 0; i < st.leading; ++i) {
      buffer.push_back(args[i]);
    }
    for (const auto* reads : {&p.tensors, &p.ints}) {
      for (const BoundRead& r : *reads) {
        PyObject* v = PyDict_GetItemWithError(dicts[r.source], r.name);
        held.emplace_back(Py_NewRef(v));
        buffer.push_back(v);
      }
    }
    st.busy = true;
    auto done = c10::make_scope_exit([&] { st.busy = busy; });
    phase_add(0, pt);
    const StaticOwner owner = st.owner;
    PyObject* flat = entry_dispatch(
        entry.ptr(),
        buffer.data(),
        buffer.size(),
        nullptr,
        nullptr,
        false,
        owner);
    if (!flat) {
      if (PyErr_Occurred()) {
        return nullptr;
      }
      Py_RETURN_NOTIMPLEMENTED;
    }
    for (const auto& [r, value] : st.resets) {
      if (PyObject_SetAttr(objects[r.source], r.name, value) < 0) {
        Py_DECREF(flat);
        return nullptr;
      }
    }
    BoundPlan& hit = st.plans[k];
    ++hit.hits;
    pt = phase_now();
    PyObject* out = bound_output(hit, flat);
    phase_add(0, pt);
    return out;
  }
  Py_RETURN_NOTIMPLEMENTED;
  END_HANDLE_TH_ERRORS
}

PyObject* bound_vectorcall(
    PyObject* self,
    PyObject* const* args,
    size_t nargsf,
    PyObject* kwnames) {
  PyObject* r = bound_call(self, args, nargsf, kwnames);
  BoundState& st = *reinterpret_cast<HostTraceBound*>(self)->state;
  if (r != Py_NotImplemented || st.fallback.is_none()) {
    return r;
  }
  Py_DECREF(r);
  const py::object fallback = st.fallback;
  return PyObject_Vectorcall(fallback.ptr(), args, nargsf, kwnames);
}

PyObject* bound_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PyObject* empty = PyTuple_New(0);
  PyObject *prefix = nullptr, *roots = nullptr, *derived = empty,
           *none_kwargs = empty, *resets = empty, *fallback = Py_None;
  Py_ssize_t leading = 0, objects = 0;
  int trust_statics = 0;
  static const char* names[] = {
      "prefix",
      "leading",
      "objects",
      "roots",
      "derived",
      "none_kwargs",
      "resets",
      "fallback",
      "trust_statics",
      nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O!nnO!|$O!O!O!Op",
          const_cast<char**>(names),
          &PyTuple_Type,
          &prefix,
          &leading,
          &objects,
          &PyTuple_Type,
          &roots,
          &PyTuple_Type,
          &derived,
          &PyTuple_Type,
          &none_kwargs,
          &PyTuple_Type,
          &resets,
          &fallback,
          &trust_statics)) {
    return nullptr;
  }
  TORCH_CHECK_TYPE(
      fallback == Py_None || PyCallable_Check(fallback),
      "fallback must be callable");
  TORCH_CHECK_VALUE(leading >= 0 && objects >= 0, "a negative count");
  THPObjectPtr self(type->tp_alloc(type, 0));
  if (!self) {
    return nullptr;
  }
  auto* b = reinterpret_cast<HostTraceBound*>(self.get());
  b->vectorcall = bound_vectorcall;
  b->state = new BoundState();
  BoundState& st = *b->state;
  st.prefix = py::reinterpret_borrow<py::tuple>(prefix);
  st.leading = leading;
  st.objects = objects;
  st.roots = py::reinterpret_borrow<py::tuple>(roots);
  st.args.assign(
      &PyTuple_GET_ITEM(prefix, 0),
      &PyTuple_GET_ITEM(prefix, 0) + PyTuple_GET_SIZE(prefix));
  st.options = py::make_tuple(
      py::handle(derived), py::handle(none_kwargs), py::handle(resets));
  st.fallback = py::reinterpret_borrow<py::object>(fallback);
  const size_t sources = st.objects +
      static_cast<size_t>(PyTuple_GET_SIZE(derived)) +
      static_cast<size_t>(PyTuple_GET_SIZE(roots));
  auto read = [](py::handle source, py::handle name, size_t limit) {
    const auto s = source.cast<size_t>();
    TORCH_CHECK_VALUE(s < limit, "no source ", s);
    TORCH_CHECK_TYPE(PyUnicode_CheckExact(name.ptr()), "a name must be a str");
    return BoundRead{s, name.ptr()};
  };
  for (py::handle g : py::handle(derived)) {
    auto t = g.cast<py::tuple>();
    st.derived.push_back(read(t[0], t[1], py::len(st.roots)));
  }
  for (py::handle name : py::handle(none_kwargs)) {
    TORCH_CHECK_TYPE(PyUnicode_CheckExact(name.ptr()), "a name must be a str");
    st.none_kwargs.push_back(name.ptr());
  }
  for (py::handle g : py::handle(resets)) {
    auto t = g.cast<py::tuple>();
    st.resets.emplace_back(read(t[0], t[1], sources), t[2].ptr());
  }
  if (trust_statics) {
    static uint64_t bounds = 0;
    st.owner.id = ++bounds;
  }
  return self.release();
  END_HANDLE_TH_ERRORS
}

// add(entry, types, same, below, other, tensors, ints, template, outputs):
// a read is (source, name); same (source, name, object); below (value read,
// index read, limits); template None: the call returns the flat outputs
PyObject* bound_add(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  BoundState& st = *reinterpret_cast<HostTraceBound*>(self)->state;
  TORCH_CHECK_TYPE(
      PyTuple_GET_SIZE(args) == 9, "add takes 9 arguments");
  py::tuple a = py::reinterpret_borrow<py::tuple>(args);
  TORCH_CHECK_TYPE(
      PyObject_TypeCheck(a[0].ptr(), &HostTraceEntryType),
      "a bound entry must be a _HostTraceEntry");
  const size_t sources = st.objects + st.derived.size() + py::len(st.roots);
  BoundPlan p;
  p.spec = a;
  p.entry = a[0];
  auto read = [&](py::handle source, py::handle name) {
    const auto s = source.cast<size_t>();
    TORCH_CHECK_VALUE(s < sources, "no source ", s);
    TORCH_CHECK_TYPE(PyUnicode_CheckExact(name.ptr()), "a name must be a str");
    return BoundRead{s, name.ptr()};
  };
  for (py::handle t : a[1].cast<py::tuple>()) {
    TORCH_CHECK_TYPE(PyType_Check(t.ptr()), "types must be types");
    p.types.push_back(reinterpret_cast<PyTypeObject*>(t.ptr()));
  }
  TORCH_CHECK_VALUE(
      p.types.size() == st.objects + st.derived.size(),
      "a type per call object and derived object");
  for (py::handle g : a[2].cast<py::tuple>()) {
    auto t = g.cast<py::tuple>();
    p.same.emplace_back(read(t[0], t[1]), t[2].ptr());
  }
  for (py::handle g : a[3].cast<py::tuple>()) {
    auto t = g.cast<py::tuple>();
    p.below.push_back(
        {read(t[0], t[1]),
         read(t[2], t[3]),
         t[4].cast<std::vector<int64_t>>()});
  }
  const std::pair<size_t, std::vector<BoundRead>*> lists[] = {
      {4, &p.other}, {5, &p.tensors}, {6, &p.ints}};
  for (const auto& [i, out] : lists) {
    for (py::handle g : a[i].cast<py::tuple>()) {
      auto t = g.cast<py::tuple>();
      out->push_back(read(t[0], t[1]));
    }
  }
  if (a[7].is_none()) {
    p.type = py::none();
  } else {
    p.type = py::reinterpret_borrow<py::object>(
        reinterpret_cast<PyObject*>(Py_TYPE(a[7].ptr())));
    THPObjectPtr fields(PyObject_GenericGetDict(a[7].ptr(), nullptr));
    if (!fields) {
      throw python_error();
    }
    p.fields = py::reinterpret_steal<py::object>(PyDict_Copy(fields.get()));
    for (py::handle name : a[8].cast<py::tuple>()) {
      TORCH_CHECK_TYPE(
          PyUnicode_CheckExact(name.ptr()), "an output must be a str");
      p.outputs.push_back(name.ptr());
    }
  }
  st.plans.push_back(std::move(p));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// The leading arguments may have changed in place (a trust_statics bound)
PyObject* bound_statics_changed(PyObject* self, PyObject* /*unused*/) {
  ++reinterpret_cast<HostTraceBound*>(self)->state->owner.version;
  Py_RETURN_NONE;
}

PyObject* bound_hits(PyObject* self, PyObject* /*unused*/) {
  HANDLE_TH_ERRORS
  BoundState& st = *reinterpret_cast<HostTraceBound*>(self)->state;
  py::list out;
  for (const BoundPlan& p : st.plans) {
    out.append(p.hits);
  }
  return out.release().ptr();
  END_HANDLE_TH_ERRORS
}

int bound_traverse(PyObject* self, visitproc visit, void* arg) {
  BoundState* st = reinterpret_cast<HostTraceBound*>(self)->state;
  if (!st) {
    return 0;
  }
  Py_VISIT(st->prefix.ptr());
  Py_VISIT(st->roots.ptr());
  Py_VISIT(st->options.ptr());
  Py_VISIT(st->fallback.ptr());
  for (const BoundPlan& p : st->plans) {
    Py_VISIT(p.spec.ptr());
    Py_VISIT(p.fields.ptr());
  }
  return 0;
}

int bound_clear(PyObject* self) {
  BoundState* st = reinterpret_cast<HostTraceBound*>(self)->state;
  if (st && !st->busy) {
    auto plans = std::move(st->plans);
    st->plans.clear();
    st->args.clear();
    py::tuple prefix = std::move(st->prefix);
    py::tuple roots = std::move(st->roots);
    st->prefix = py::tuple();
    st->roots = py::tuple();
    st->derived.clear();
    st->none_kwargs.clear();
    st->resets.clear();
    py::object options = std::move(st->options);
    py::object fallback = std::move(st->fallback);
    st->fallback = py::none();
  }
  return 0;
}

void bound_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  bound_clear(self);
  auto* b = reinterpret_cast<HostTraceBound*>(self);
  delete b->state;
  b->state = nullptr;
  Py_TYPE(self)->tp_free(self);
}

PyMethodDef bound_methods[] = {
    {"add", bound_add, METH_VARARGS, nullptr},
    {"hits", bound_hits, METH_NOARGS, nullptr},
    {"statics_changed", bound_statics_changed, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyTypeObject HostTraceBoundType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._HostTraceBound",
    sizeof(HostTraceBound),
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
  // the native encode takes a descriptor's fill and Triton flavor
  m.attr("_host_trace_tma_flavors") = true;
  // and launches the driver's map as is (a descriptor's library bits)
  m.attr("_host_trace_tma_raw") = true;
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
  m.def("_host_trace_delta_counts", [] {
    return py::make_tuple(delta_skips, delta_partials, delta_fulls, delta_rows_evaluated);
  });
  m.def("_host_trace_delta", [](bool enable) { delta_enabled = enable; });
  m.def("_host_trace_variant_order", [](bool enable) {
    variant_order_enabled = enable;
  });
  m.def("_host_trace_placement_reuse", [](bool enable) {
    placement_reuse_enabled = enable;
  });
  m.def("_host_trace_form_held", [](bool enable) {
    form_held_enabled = enable;
  });
  m.def("_host_trace_entry_descriptors", [](bool enable) {
    entry_descriptors_enabled = enable;
  });
  m.def("_host_trace_entry_descriptors_enabled", [] {
    return entry_descriptors_enabled;
  });
  m.def("_host_trace_trusted_statics_check", [](bool enable) {
    trusted_statics_check = enable;
  });
  m.def("_host_trace_delta_changed", [](const HostTraceVariant& v) {
    return v.delta_changed();
  });
  m.def("_host_trace_static_split", [](const HostTraceVariant& v) {
    return v.static_split();
  });
  m.def("_host_trace_memory_node_sets", [] { return memory_node_sets; });
  m.def("_host_trace_kernel_packs", [] { return kernel_packs; });
  m.def("_host_trace_kernel_node_sets", [] { return kernel_node_sets; });
  m.def("_host_trace_phase_times", [](bool enable) {
    const auto times = phase_ns;
    phase_ns.fill(0);
    phase_timing = enable;
    return times;
  });
  m.def("_host_trace_held_buffer_bytes", [] { return held_buffer_bytes(); });
  m.def("_host_trace_release_held_buffers", [] { release_held_buffers(); });
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
  PyTypeObject& bt = HostTraceBoundType;
  bt.tp_flags =
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HAVE_VECTORCALL;
  bt.tp_doc = "per entry, where a call's arguments are read and when";
  bt.tp_new = bound_new;
  bt.tp_dealloc = bound_dealloc;
  bt.tp_traverse = bound_traverse;
  bt.tp_clear = bound_clear;
  bt.tp_call = PyVectorcall_Call;
  bt.tp_vectorcall_offset = offsetof(HostTraceBound, vectorcall);
  bt.tp_methods = bound_methods;
  if (PyType_Ready(&bt) < 0) {
    throw py::error_already_set();
  }
  Py_INCREF(&bt);
  if (PyModule_AddObject(module, "_HostTraceBound", (PyObject*)&bt) < 0) {
    throw py::error_already_set();
  }
#endif
}

} // namespace torch::cuda
