#include <torch/csrc/cuda/host_trace/Variant.h>

#if !defined(USE_ROCM)
#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/TorchDispatchUtils.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/empty_strided_native.h>
#include <ATen/record_function.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>

#include <algorithm>
#include <sstream>

namespace torch::cuda::host_trace {

int64_t buffered_runs = 0;

// the caching allocator's block granularity (_host_trace_memory._BLOCK)
constexpr int64_t kBlock = 512;

namespace {

BlasState blas_state() {
  auto& c = at::globalContext();
  return {
      c.float32Precision(at::Float32Backend::CUDA, at::Float32Op::MATMUL),
      c.allowFP16ReductionCuBLAS(),
      c.allowBF16ReductionCuBLAS(),
      c.allowFP16AccumulationCuBLAS(),
      c._SMCarveout_EXPERIMENTAL(),
      c.blasPreferredBackend()};
}

// The global cuBLAS state from `from` to `to`, setting only what differs, as
// library_state_as does
void set_blas_state(const BlasState& from, const BlasState& to) {
  using Option = at::CuBLASReductionOption;
  auto& c = at::globalContext();
  if (from.matmul != to.matmul) {
    c.setFloat32Precision(
        at::Float32Backend::CUDA, at::Float32Op::MATMUL, to.matmul);
  }
  if (from.fp16 != to.fp16) {
    c.setAllowFP16ReductionCuBLAS(
        to.fp16 == Option::AllowReducedPrecisionWithSplitK,
        to.fp16 != Option::DisallowReducedPrecisionDisallowSplitK);
  }
  if (from.bf16 != to.bf16) {
    c.setAllowBF16ReductionCuBLAS(
        to.bf16 == Option::AllowReducedPrecisionWithSplitK,
        to.bf16 != Option::DisallowReducedPrecisionDisallowSplitK);
  }
  if (from.fp16_accumulation != to.fp16_accumulation) {
    c.setAllowFP16AccumulationCuBLAS(to.fp16_accumulation);
  }
  if (from.carveout != to.carveout) {
    c._setSMCarveout_EXPERIMENTAL(to.carveout);
  }
  if (from.backend != to.backend) {
    c.setBlasPreferredBackend(to.backend);
  }
}

} // namespace

const at::Tensor& HostTraceVariant::base_tensor(
    const BaseRef& ref,
    PyObject* const* args,
    const Frame& frame) const {
  if (ref.argument) {
    TORCH_CHECK_VALUE(
        static_cast<size_t>(ref.index) < frame.count &&
            THPVariable_Check(args[ref.index]),
        "host_trace: argument ",
        ref.index,
        " is not a tensor");
    return THPVariable_Unpack(args[ref.index]);
  }
  const at::Tensor& t = frame.tensors[ref.index];
  if (!t.defined()) {
    PyErr_Format(
        PyExc_AssertionError, "host_trace: base %lld is not held", ref.index);
    throw py::error_already_set();
  }
  return t;
}

at::Tensor HostTraceVariant::view(
    const View& v,
    PyObject* const* args,
    const Frame& frame,
    const int64_t* values) const {
  c10::SmallVector<int64_t, 8> sizes, strides;
  for (int64_t r : v.sizes) {
    sizes.push_back(values[r]);
  }
  for (int64_t r : v.strides) {
    strides.push_back(values[r]);
  }
  const at::Tensor& base = base_tensor(v.base, args, frame);
  if (!v.dtype) {
    // What the dispatcher's as_strided does for a base without autograd meta
    // (no grad, no forward grad, not a view), minus the dispatch.
    const auto tls = c10::impl::tls_local_dispatch_key_set();
    const bool direct = !torch::autograd::impl::get_autograd_meta(base) &&
        base.key_set().has(c10::DispatchKey::CUDA) &&
        base.key_set().has(c10::DispatchKey::ADInplaceOrView) &&
        c10::default_included_set.isSupersetOf(tls.included_) &&
        !tls.excluded_.has(c10::DispatchKey::ADInplaceOrView) &&
        !base.is_conj() && !base.is_neg() &&
        !at::impl::tensor_has_dispatch(base) &&
        !c10::impl::dispatch_mode_enabled() &&
        !c10::AutogradState::get_tls_state().get_view_replay_enabled() &&
        !at::hasCallbacks();
    if (!direct) {
      return base.as_strided(sizes, strides, values[v.offset]);
    }
    using torch::autograd::CreationMeta;
    return torch::autograd::as_view(
        base,
        at::native::as_strided_tensorimpl(base, sizes, strides, values[v.offset]),
        /*is_bw_differentiable=*/true,
        /*is_fw_differentiable=*/true,
        nullptr,
        nullptr,
        c10::InferenceMode::is_enabled()
            ? CreationMeta::INFERENCE_MODE
            : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT
                                          : CreationMeta::NO_GRAD_MODE));
  }
  // another dtype over the base's storage, as view_as_complex's alias
  at::Tensor t = at::detail::make_tensor<c10::TensorImpl>(
      c10::Storage(base.storage()),
      base.key_set(),
      c10::scalarTypeToTypeMeta(*v.dtype));
  t.unsafeGetTensorImpl()->set_sizes_and_strides(
      sizes, strides, values[v.offset]);
  return t;
}

bool HostTraceVariant::dropped(int64_t k, const Frame& frame) const {
  const int64_t site = allocations_[k].owner;
  if (site < 0) {
    return false;
  }
  const int64_t row = frame.site_entries[site];
  if (row <= 0) {
    return false;
  }
  const auto& drops = sites_[site].entries[row - 1]->drops;
  return std::find(drops.begin(), drops.end(), k) != drops.end();
}

int64_t HostTraceVariant::alloc_bytes(int64_t k, const Frame& frame) const {
  const Allocation& a = allocations_[k];
  if (a.site >= 0) {
    const int64_t row = frame.site_entries[a.site];
    if (row > 0) {
      return sites_[a.site].entries[row - 1]->scratch[a.scratch];
    }
  }
  return dropped(k, frame) ? 0 : frame.values[a.nbytes];
}

void HostTraceVariant::allocate_tensor(int64_t k, Frame& frame) const {
  if (dropped(k, frame)) {
    frame.bases[k] = 0;
    return;
  }
  const int64_t* v = frame.values.data();
  const Allocation& a = allocations_[k];
  c10::SmallVector<int64_t, 8> sizes, strides;
  for (int64_t r : a.sizes) {
    sizes.push_back(v[r]);
  }
  for (int64_t r : a.strides) {
    strides.push_back(v[r]);
  }
  if (a.site >= 0) {
    // a scratch buffer: bytes
    sizes[0] = alloc_bytes(k, frame);
  }
  // torch.empty_strided's kernel: the caching allocator on the current
  // stream, without the dispatcher
  at::Tensor t = at::native::empty_strided_cuda(
      sizes,
      strides,
      a.dtype,
      at::kStrided,
      at::Device(at::kCUDA, device_),
      std::nullopt);
  frame.bases[k] = reinterpret_cast<int64_t>(t.data_ptr());
  frame.tensors[k] = std::move(t);
}

at::Tensor HostTraceVariant::allocate(
    const StepMemory& memory,
    const Segment* run,
    Frame& frame,
    std::unique_lock<std::recursive_mutex>& hold,
    c10::SmallVector<c10::DataPtr, 4>& held,
    c10::SmallVector<c10::Storage, 4>& freed) const {
  auto* allocator = at::cuda::getCUDADeviceAllocator();
  // the run's selected sections with temporaries, by their place in the order
  c10::SmallVector<const Entry*, 4> sections;
  for (size_t i = 0; run && i < run->sections.size(); ++i) {
    const Site& s = sites_[run->sections[i]];
    const int64_t row = frame.site_entries[run->sections[i]];
    if (row > 0 && !s.entries[row - 1]->temporaries.empty()) {
      sections.push_back(s.entries[row - 1].get());
    }
  }
  std::sort(sections.begin(), sections.end(), [](const Entry* a, const Entry* b) {
    return a->alloc_at < b->alloc_at;
  });
  auto make = [&](const Entry& e, c10::SmallVector<c10::DataPtr, 4>& into) {
    for (size_t j = 0; j < e.temporaries.size(); ++j) {
      into.push_back(allocator->allocate(frame.values[e.temporaries[j]]));
      frame.bases[e.first_base + j] = reinterpret_cast<int64_t>(into.back().get());
    }
  };
  // Eager order frees a temporary before its run's graph is queued, so the
  // allocator stays locked until the launch: another thread's allocation on
  // the stream would take a freed temporary's bytes and queue its kernels
  // ahead of the graph. A segment released meanwhile (our own allocation's
  // OOM retry or garbage collection) may have taken them too, and replay
  // hooks run Python before the launch: then the temporaries go in a run
  // buffer, the tensors keep their addresses and nothing has launched. A
  // section's temporaries are made and freed where its op's are, in eager
  // order or, a run of no temporaries, among its tensors; else before the
  // run's own allocations (no block of one of its temporaries is free yet),
  // held until it is queued.
  const bool observed_run = run && observed(*run);
  if (observed_run || (memory.order.empty() && (sections.empty() || !memory.temporaries.empty()))) {
    // a run buffer's run places a section's temporaries with its own, over
    // its op's launches (Entry::last_seq), as eager frees them after its op
    c10::SmallVector<const Entry*, 4> placed;
    for (const Entry* e : sections) {
      if (memory.order.empty() && !memory.temporaries.empty() && e->last_seq >= 0) {
        placed.push_back(e);
      } else {
        make(*e, held);
      }
    }
    int64_t pt = phase_now();
    for (int64_t k : memory.tensors) {
      allocate_tensor(k, frame);
    }
    phase_add(11, pt);
    at::Tensor scratch = run_buffer(memory, frame, placed);
    phase_add(12, pt);
    return scratch;
  }
  hold = c10::cuda::CUDACachingAllocator::lockDeviceAllocator(device_);
  const size_t releases =
      c10::cuda::CUDACachingAllocator::getSegmentReleaseCount();
  c10::SmallVector<c10::DataPtr, 16> blocks;
  blocks.resize(memory.temporaries.size());
  c10::SmallVector<c10::DataPtr, 4> made;
  c10::SmallVector<size_t, 4> starts;
  // the release count when the order first freed an argument's bytes
  size_t argument_releases = SIZE_MAX;
  const size_t n = memory.order.empty() ? memory.tensors.size() : memory.order.size();
  for (size_t p = 0; p <= n; ++p) {
    for (size_t s = 0; s < sections.size(); ++s) {
      if (sections[s]->alloc_at == p) {
        starts.push_back(made.size());
        make(*sections[s], made);
      }
      if (sections[s]->free_at == p) {
        for (size_t j = 0; j < sections[s]->temporaries.size(); ++j) {
          made[starts[s] + j].clear();
        }
      }
    }
    if (p == n) {
      break;
    }
    if (memory.order.empty()) {
      allocate_tensor(memory.tensors[p], frame);
      continue;
    }
    const Event& e = memory.order[p];
    if (e.op == Op::Tensor) {
      allocate_tensor(e.alloc, frame);
    } else if (e.op == Op::Temporary) {
      blocks[e.block] = allocator->allocate(alloc_bytes(e.alloc, frame));
      frame.bases[e.alloc] = reinterpret_cast<int64_t>(blocks[e.block].get());
    } else if (e.op == Op::Argument) {
      if (e.block < freed.size() && freed[e.block]) {
        if (argument_releases == SIZE_MAX && freed[e.block].use_count() == 1) {
          argument_releases =
              c10::cuda::CUDACachingAllocator::getSegmentReleaseCount();
        }
        freed[e.block] = c10::Storage();
      }
    } else {
      blocks[e.block].clear();
    }
  }
  const size_t now = c10::cuda::CUDACachingAllocator::getSegmentReleaseCount();
  // the run's launches before an argument's free read its bytes, which a
  // released segment no longer holds; a temporary's are the run's own
  if (argument_releases != SIZE_MAX && now != argument_releases) {
    PyErr_SetString(
        disagreement_.ptr(),
        "host_trace: the allocator released a cached segment (an OOM retry) after the replay freed an argument at its last launch, before the run's graph was queued");
    throw py::error_already_set();
  }
  if (now == releases) {
    return at::Tensor();
  }
  ++buffered_runs;
  for (const Entry* e : sections) {
    make(*e, held);
  }
  return run_buffer(memory, frame);
}

at::Tensor HostTraceVariant::run_buffer(
    const StepMemory& memory,
    Frame& frame,
    c10::ArrayRef<const Entry*> sections) const {
  if (memory.temporaries.empty()) {
    return at::Tensor();
  }
  // _host_trace_memory.place: each temporary takes the lowest gap among
  // those live at its allocation (first fit); a placed section's temporaries
  // are made at its made_seq and live to its last_seq
  struct Live {
    int64_t offset, end, last;
  };
  const size_t count = memory.temporaries.size();
  auto round_up = [](int64_t n) { return (n + kBlock - 1) / kBlock * kBlock; };
  auto rounded = [&](size_t i) {
    return round_up(alloc_bytes(memory.temporaries[i].alloc, frame));
  };
  c10::SmallVector<const Entry*, 4> ordered(sections.begin(), sections.end());
  std::stable_sort(ordered.begin(), ordered.end(), [](const Entry* a, const Entry* b) {
    return a->made_seq < b->made_seq;
  });
  // per placed section its seqs and temporaries' sizes, which with the run's
  // own sizes decide the placement
  c10::SmallVector<int64_t, 8> section_sizes, section_key;
  for (const Entry* e : ordered) {
    section_key.push_back(e->made_seq);
    section_key.push_back(e->last_seq);
    section_key.push_back(static_cast<int64_t>(e->temporaries.size()));
    for (int64_t row : e->temporaries) {
      section_sizes.push_back(round_up(frame.values[row]));
      section_key.push_back(section_sizes.back());
    }
  }
  // the last placement holds while the rows that size the temporaries (a few:
  // many temporaries share one), the scratch buffers' sizes and the placed
  // sections and their temporaries' sizes are its
  const int64_t* v = frame.values.data();
  bool reuse = placement_reuse_enabled && memory.placed_total >= 0;
  for (size_t k = 0; reuse && k < memory.size_rows.size(); ++k) {
    reuse = v[memory.size_rows[k]] == memory.placed_row_values[k];
  }
  for (size_t j = 0; reuse && j < memory.site_temporaries.size(); ++j) {
    const size_t i = memory.site_temporaries[j];
    reuse = rounded(i) == memory.placed_sizes[i];
  }
  reuse = reuse &&
      std::equal(section_key.begin(), section_key.end(), memory.placed_section_key.begin(), memory.placed_section_key.end());
  if (!reuse) {
    memory.placed_total = -1;
    memory.placed_sizes.resize(count);
    for (size_t i = 0; i < count; ++i) {
      memory.placed_sizes[i] = rounded(i);
    }
    memory.placed_row_values.resize(memory.size_rows.size());
    for (size_t k = 0; k < memory.size_rows.size(); ++k) {
      memory.placed_row_values[k] = v[memory.size_rows[k]];
    }
    memory.placed_offsets.assign(count, 0);
    memory.placed_section_key.assign(section_key.begin(), section_key.end());
    memory.placed_section_offsets.assign(section_sizes.size(), 0);
  }
  const std::vector<int64_t>& sizes = memory.placed_sizes;
  std::vector<int64_t>& offsets = memory.placed_offsets;
  if (memory.placed_total < 0) {
    c10::SmallVector<Live, 16> live;
    auto take = [&](int64_t size, int64_t seq, int64_t last) {
      live.erase(
          std::remove_if(
              live.begin(),
              live.end(),
              [&](const Live& b) { return b.last <= seq; }),
          live.end());
      std::sort(live.begin(), live.end(), [](const Live& a, const Live& b) {
        return std::tie(a.offset, a.end, a.last) <
            std::tie(b.offset, b.end, b.last);
      });
      int64_t at = 0;
      for (const Live& b : live) {
        if (b.offset - at >= size) {
          break;
        }
        at = std::max(at, b.end);
      }
      live.push_back({at, at + size, last});
      memory.placed_total = std::max(memory.placed_total, at + size);
      return at;
    };
    // the run's temporaries in tape order and the sections by made_seq, merged
    // by seq (a section first on a tie)
    size_t s = 0, base = 0;
    for (size_t i = 0; i <= count; ++i) {
      while (s < ordered.size() && (i == count || ordered[s]->made_seq <= memory.temporaries[i].seq)) {
        for (size_t j = 0; j < ordered[s]->temporaries.size(); ++j, ++base) {
          if (section_sizes[base]) {
            memory.placed_section_offsets[base] = take(section_sizes[base], ordered[s]->made_seq, ordered[s]->last_seq);
          }
        }
        ++s;
      }
      if (i < count && sizes[i]) {
        offsets[i] = take(sizes[i], memory.temporaries[i].seq, memory.temporaries[i].last);
      }
    }
    if (memory.placed_total < 0) {
      memory.placed_total = 0;
    }
  }
  const int64_t total = memory.placed_total;
  if (total == 0) {
    for (const Entry* e : ordered) {
      for (size_t j = 0; j < e->temporaries.size(); ++j) {
        frame.bases[e->first_base + j] = 0;
      }
    }
    return at::Tensor();
  }
  at::Tensor scratch = at::native::empty_cuda(
      {total},
      at::kByte,
      at::kStrided,
      at::Device(at::kCUDA, device_),
      std::nullopt,
      std::nullopt);
  const auto at = reinterpret_cast<int64_t>(scratch.data_ptr());
  for (size_t i = 0; i < count; ++i) {
    if (sizes[i]) {
      frame.bases[memory.temporaries[i].alloc] = at + offsets[i];
    }
  }
  size_t base = 0;
  for (const Entry* e : ordered) {
    for (size_t j = 0; j < e->temporaries.size(); ++j, ++base) {
      frame.bases[e->first_base + j] = at + memory.placed_section_offsets[base];
    }
  }
  return scratch;
}

namespace {

std::mutex held_mutex;

// leaked: freeing at exit would reach a torn-down allocator
std::vector<std::unique_ptr<HeldBuffer>>& held_buffers() {
  static auto* buffers = new std::vector<std::unique_ptr<HeldBuffer>>();
  return *buffers;
}

c10::Storage planned_storage(size_t bytes) {
  return c10::Storage(
      c10::Storage::use_byte_size_t(),
      bytes,
      at::cuda::getCUDADeviceAllocator(),
      /*resizable=*/false);
}

} // namespace

int64_t held_buffer_bytes() {
  std::lock_guard<std::mutex> lock(held_mutex);
  int64_t bytes = 0;
  for (const auto& h : held_buffers()) {
    if (h->storage) {
      bytes += static_cast<int64_t>(h->storage.nbytes());
    }
  }
  return bytes;
}

void release_held_buffers() {
  std::vector<c10::Storage> freed;
  {
    std::lock_guard<std::mutex> lock(held_mutex);
    for (const auto& h : held_buffers()) {
      if (!h->busy) {
        freed.push_back(std::move(h->storage));
        h->storage = c10::Storage();
      }
    }
  }
}

PlanLease::~PlanLease() {
  if (held) {
    std::lock_guard<std::mutex> lock(held_mutex);
    held->busy = false;
  }
}

void HostTraceVariant::plan(Frame& frame, PlanLease& lease) const {
  const int64_t* v = frame.values.data();
  const auto bytes = static_cast<size_t>(v[planned_bytes_]);
  // a capture's graph keeps the addresses it reads
  if (frame.uncaptured ||
      c10::cuda::currentStreamCaptureStatusMayInitCtx() ==
          c10::cuda::CaptureStatus::None) {
    const cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(device_).stream();
    std::lock_guard<std::mutex> lock(held_mutex);
    auto& buffers = held_buffers();
    auto it = std::find_if(buffers.begin(), buffers.end(), [&](const auto& h) {
      return h->device == device_ && h->stream == stream;
    });
    if (it == buffers.end()) {
      buffers.push_back(std::make_unique<HeldBuffer>(
          HeldBuffer{device_, stream, c10::Storage()}));
      it = buffers.end() - 1;
    }
    if (!(*it)->busy) {
      lease.held = it->get();
      lease.held->busy = true;
    }
  }
  if (lease.held) {
    c10::Storage& held = lease.held->storage;
    if (!held || held.nbytes() < bytes) {
      // freed first, so the allocator can hand its bytes back; the stream
      // orders them after the kernels that used them
      held = c10::Storage();
      held = planned_storage(bytes);
    }
    lease.storage = held;
  } else {
    lease.storage = planned_storage(bytes);
  }
  const auto base = reinterpret_cast<int64_t>(lease.storage.mutable_data());
  frame.planned_lo = base;
  frame.planned_hi = base + static_cast<int64_t>(lease.storage.nbytes());
  const at::Device device(at::kCUDA, device_);
  for (const Planned& p : planned_) {
    frame.bases[p.alloc] = base + v[p.offset];
    if (!p.tensor) {
      continue;
    }
    // a storage of its own over its bytes, as a fresh allocation's, which
    // keeps the buffer alive
    const Allocation& a = allocations_[p.alloc];
    c10::StorageImpl* buffer = lease.storage.unsafeGetStorageImpl();
    c10::raw::intrusive_ptr::incref(buffer);
    c10::DataPtr data(
        reinterpret_cast<void*>(frame.bases[p.alloc]),
        buffer,
        [](void* ctx) {
          c10::raw::intrusive_ptr::decref(static_cast<c10::StorageImpl*>(ctx));
        },
        device);
    c10::Storage storage(
        c10::Storage::use_byte_size_t(),
        v[a.nbytes],
        std::move(data),
        /*allocator=*/nullptr,
        /*resizable=*/false);
    at::Tensor t = at::detail::make_tensor<c10::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::CUDA),
        c10::scalarTypeToTypeMeta(a.dtype));
    c10::SmallVector<int64_t, 8> sizes, strides;
    for (int64_t r : a.sizes) {
      sizes.push_back(v[r]);
    }
    for (int64_t r : a.strides) {
      strides.push_back(v[r]);
    }
    t.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    frame.tensors[p.alloc] = std::move(t);
  }
}

void HostTraceVariant::check_fresh(
    const EagerStep& step,
    int64_t base,
    const Frame& frame) const {
  if (base < frame.planned_lo || base >= frame.planned_hi) {
    return;
  }
  const std::string msg = step.name + " returned a view of an allocation";
  PyObject* op = step.target ? step.target.ptr() : Py_None;
  THPObjectPtr e(
      PyObject_CallFunction(disagreement_.ptr(), "sO", msg.c_str(), op));
  if (e) {
    PyErr_SetObject(disagreement_.ptr(), e.get());
  }
  throw py::error_already_set();
}

bool HostTraceVariant::observed(const Segment& run) const {
  // what CUDAGraph.replay does around the C++ replay
  if (PyDict_GET_SIZE(global_replay_start_hooks_.ptr()) ||
      PyDict_GET_SIZE(global_replay_end_hooks_.ptr())) {
    return true;
  }
  static PyObject* const names[] = {
      PyUnicode_InternFromString("_replay_start_hooks"),
      PyUnicode_InternFromString("_replay_end_hooks"),
      PyUnicode_InternFromString("_tracker"),
      PyUnicode_InternFromString("_retained"),
      PyUnicode_InternFromString("sync_before_fire")};
  PyObject* d = run.graph_dict.ptr();
  for (int i = 0; i < 2; ++i) {
    PyObject* hooks = PyDict_GetItemWithError(d, names[i]);
    if (!hooks || !PyDict_Check(hooks) || PyDict_GET_SIZE(hooks)) {
      return true;
    }
  }
  PyObject* tracker = PyDict_GetItemWithError(d, names[2]);
  PyObject* retained = PyDict_GetItemWithError(d, names[3]);
  if (tracker != Py_None || !retained) {
    return true;
  }
  THPObjectPtr sync(PyObject_GetAttr(retained, names[4]));
  return sync.get() != Py_False;
}

void HostTraceVariant::boxed(
    const EagerStep& step,
    PyObject* const* args,
    Frame& frame) const {
  const int64_t* v = frame.values.data();
  // a view two leaves share is one tensor, as in the trace
  c10::SmallVector<at::Tensor, 8> tensors(step.leaves.size());
  auto tensor = [&](int64_t leaf) -> const at::Tensor& {
    at::Tensor& t = tensors[leaf];
    if (t.defined()) {
      return t;
    }
    const int64_t index = step.leaves[leaf].index;
    for (size_t j = 0; j < tensors.size(); ++j) {
      if (tensors[j].defined() && step.leaves[j].kind == LeafKind::View &&
          step.leaves[j].index == index) {
        t = tensors[j];
        return t;
      }
    }
    // a view that is its whole base is the base, without an as_strided
    const View& view_of = views_[index];
    const at::Tensor& base = base_tensor(view_of.base, args, frame);
    bool whole = !view_of.dtype &&
        base.dim() == static_cast<int64_t>(view_of.sizes.size()) &&
        base.storage_offset() == v[view_of.offset];
    for (size_t d = 0; whole && d < view_of.sizes.size(); ++d) {
      whole = base.sizes()[d] == v[view_of.sizes[d]] &&
          base.strides()[d] == v[view_of.strides[d]];
    }
    t = whole ? base : view(view_of, args, frame, v);
    return t;
  };
  auto scalar = [&](int64_t leaf) { return v[step.leaves[leaf].index]; };
  torch::jit::Stack stack;
  stack.reserve(step.args.size());
  for (const BoxedArg& a : step.args) {
    switch (a.kind) {
      case ArgKind::Constant:
        stack.push_back(a.constant);
        break;
      case ArgKind::Tensor:
        stack.emplace_back(tensor(a.leaf));
        break;
      case ArgKind::Int:
        stack.emplace_back(scalar(a.leaf));
        break;
      case ArgKind::Double:
        stack.emplace_back(static_cast<double>(scalar(a.leaf)));
        break;
      case ArgKind::TensorList: {
        c10::List<at::Tensor> list;
        list.reserve(a.items.size());
        for (const ArgItem& item : a.items) {
          list.push_back(tensor(item.leaf));
        }
        stack.emplace_back(std::move(list));
        break;
      }
      case ArgKind::OptionalTensorList: {
        c10::List<std::optional<at::Tensor>> list;
        list.reserve(a.items.size());
        for (const ArgItem& item : a.items) {
          list.push_back(
              item.leaf >= 0 ? std::optional<at::Tensor>(tensor(item.leaf))
                             : std::nullopt);
        }
        stack.emplace_back(std::move(list));
        break;
      }
      case ArgKind::IntList: {
        c10::List<int64_t> list;
        list.reserve(a.items.size());
        for (const ArgItem& item : a.items) {
          list.push_back(
              item.leaf >= 0 ? scalar(item.leaf) : item.constant.toInt());
        }
        stack.emplace_back(std::move(list));
        break;
      }
    }
  }
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    struct RngSwap {
      std::optional<at::Generator> generator;
      at::Generator prior;
      ~RngSwap() {
        if (generator) {
          std::scoped_lock<std::mutex> lock(generator->mutex());
          generator->graphsafe_set_state(prior);
        }
      }
    } swap;
    if (step.rng_state) {
      at::Generator g =
          at::globalContext().defaultGenerator(step.rng_state->device());
      std::scoped_lock<std::mutex> lock(g.mutex());
      swap.prior = g.graphsafe_get_state();
      swap.generator = g;
      g.graphsafe_set_state(*step.rng_state);
    }
    if (step.blas) {
      const BlasState prior = blas_state();
      set_blas_state(prior, *step.blas);
      try {
        step.op->callBoxed(stack);
      } catch (...) {
        set_blas_state(*step.blas, prior);
        throw;
      }
      set_blas_state(*step.blas, prior);
    } else {
      step.op->callBoxed(stack);
    }
  }
  auto disagree = [&](const std::string& msg) {
    THPObjectPtr e(PyObject_CallFunction(
        disagreement_.ptr(), "sO", msg.c_str(), step.target.ptr()));
    if (e) {
      PyErr_SetObject(disagreement_.ptr(), e.get());
    }
    throw py::error_already_set();
  };
  const std::string& name = step.name;
  c10::SmallVector<at::Tensor, 4> outs;
  for (const c10::IValue& r : stack) {
    if (r.isTensor()) {
      if (r.toTensor().defined()) {
        outs.push_back(r.toTensor());
      }
    } else if (r.isTensorList()) {
      for (const at::Tensor& t : r.toTensorList()) {
        outs.push_back(t);
      }
    } else if (!r.isNone()) {
      disagree(name + " returned a " + r.tagKind());
    }
  }
  if (outs.size() != step.outputs.size()) {
    disagree(name + " returned " + std::to_string(outs.size()) + " tensors");
  }
  const at::Device device(at::kCUDA, device_);
  if (step.seed_offset_on_device) {
    // seed_offset_on_device: as the trace predicts them
    for (at::Tensor& o : outs) {
      if (o.is_cpu()) {
        o = o.to(o.options().device(device), /*non_blocking=*/true);
      }
    }
  }
  for (size_t i = 0; i < outs.size(); ++i) {
    const Predicted& p = step.outputs[i];
    const at::Tensor& o = outs[i];
    if (p.root < 0) {
      if (!o.is_same(tensor(p.leaf))) {
        disagree(
            name + " output " + std::to_string(i) + " is not its argument");
      }
      continue;
    }
    if (o.layout() != at::kStrided) {
      disagree(name + " output " + std::to_string(i) + " is not strided");
    }
    const auto base = reinterpret_cast<int64_t>(o.storage().data());
    bool same = o.dim() == static_cast<int64_t>(p.sizes.size()) &&
        o.storage_offset() == v[p.offset] && o.scalar_type() == p.dtype &&
        o.device() == device && (o.numel() == 0 || base % kAlignment == 0);
    for (size_t d = 0; same && d < p.sizes.size(); ++d) {
      // a size-1 dim's stride addresses nothing (_addressing), nor does a
      // zero-element output's: the replay keeps eager's
      same = o.sizes()[d] == v[p.sizes[d]] &&
          (o.sizes()[d] == 1 || o.numel() == 0 ||
           o.strides()[d] == v[p.strides[d]]);
    }
    if (!same) {
      std::vector<int64_t> sizes, strides;
      for (size_t d = 0; d < p.sizes.size(); ++d) {
        sizes.push_back(v[p.sizes[d]]);
        strides.push_back(v[p.strides[d]]);
      }
      std::ostringstream m;
      m << name << " output " << i
        << " is (sizes, strides, storage offset, dtype, device) (" << o.sizes()
        << ", " << o.strides() << ", " << o.storage_offset() << ", "
        << o.scalar_type() << ", " << o.device() << ") at address 0x"
        << std::hex << base << std::dec << "; its fake kernel predicted ("
        << c10::IntArrayRef(sizes) << ", " << c10::IntArrayRef(strides) << ", "
        << v[p.offset] << ", " << p.dtype << ", " << device << ")";
      disagree(m.str());
    }
  }
  for (size_t i = 0; i < outs.size(); ++i) {
    const Predicted& p = step.outputs[i];
    if (p.root >= 0) {
      const auto b = static_cast<size_t>(n_alloc_ + p.root);
      frame.bases[b] = reinterpret_cast<int64_t>(outs[i].storage().data());
      check_fresh(step, frame.bases[b], frame);
      frame.tensors[b] = std::move(outs[i]);
    }
  }
}

void HostTraceVariant::run_eager(
    const EagerStep& step,
    PyObject* const* args,
    Frame& frame) {
  if (step.op && !at::impl::torch_function_mode_enabled()) {
    boxed(step, args, frame);
    return;
  }
  ++python_calls;
  const int64_t* v = frame.values.data();
  THPObjectPtr leaves(PyList_New(static_cast<Py_ssize_t>(step.leaves.size())));
  if (!leaves) {
    throw py::error_already_set();
  }
  // a view two leaves share is one tensor, as in the trace
  c10::SmallVector<std::pair<int64_t, PyObject*>, 8> made;
  for (size_t i = 0; i < step.leaves.size(); ++i) {
    const Leaf& leaf = step.leaves[i];
    PyObject* x = nullptr;
    if (leaf.kind == LeafKind::Constant) {
      x = Py_NewRef(leaf.constant.ptr());
    } else if (leaf.kind == LeafKind::Scalar) {
      x = PyLong_FromLongLong(v[leaf.index]);
    } else {
      for (const auto& [index, obj] : made) {
        if (index == leaf.index) {
          x = Py_NewRef(obj);
          break;
        }
      }
      if (!x) {
        x = THPVariable_Wrap(view(views_[leaf.index], args, frame, v));
        if (x) {
          made.emplace_back(leaf.index, x);
        }
      }
    }
    if (!x) {
      throw py::error_already_set();
    }
    PyList_SET_ITEM(leaves.get(), static_cast<Py_ssize_t>(i), x);
  }
  // valid only during the call: released after it
  Py_ssize_t shape = static_cast<Py_ssize_t>(frame.values.size());
  Py_ssize_t stride = sizeof(int64_t);
  Py_buffer buffer{};
  buffer.buf = frame.values.data();
  buffer.len = shape * stride;
  buffer.itemsize = sizeof(int64_t);
  buffer.readonly = 1;
  buffer.ndim = 1;
  buffer.format = const_cast<char*>("q");
  buffer.shape = &shape;
  buffer.strides = &stride;
  THPObjectPtr values(PyMemoryView_FromBuffer(&buffer));
  if (!values) {
    throw py::error_already_set();
  }
  THPObjectPtr out(PyObject_CallFunctionObjArgs(
      step.run.ptr(), leaves.get(), values.get(), nullptr));
  if (!out) {
    // release the view without clobbering the step's error
    PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    Py_XDECREF(PyObject_CallMethod(values.get(), "release", nullptr));
    PyErr_Clear();
    PyErr_Restore(type, value, traceback);
    throw py::error_already_set();
  }
  THPObjectPtr released(PyObject_CallMethod(values.get(), "release", nullptr));
  if (!released) {
    throw py::error_already_set();
  }
  TORCH_CHECK_TYPE(PyTuple_Check(out.get()), "an eager step's outputs");
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(out.get()); ++i) {
    PyObject* pair = PyTuple_GET_ITEM(out.get(), i);
    TORCH_CHECK_TYPE(
        PyTuple_Check(pair) && PyTuple_GET_SIZE(pair) == 2 &&
            THPVariable_Check(PyTuple_GET_ITEM(pair, 1)),
        "an eager step's (root, tensor)");
    const int64_t root = THPUtils_unpackLong(PyTuple_GET_ITEM(pair, 0));
    const auto b = static_cast<size_t>(n_alloc_ + root);
    TORCH_CHECK_VALUE(root >= 0 && b < base_count_, "no eager output ", root);
    const at::Tensor& t = THPVariable_Unpack(PyTuple_GET_ITEM(pair, 1));
    frame.bases[b] = reinterpret_cast<int64_t>(t.storage().data());
    check_fresh(step, frame.bases[b], frame);
    frame.tensors[b] = t;
  }
}

PyObject* HostTraceVariant::outputs(PyObject* const* args, Frame& frame) const {
  const int64_t* v = frame.values.data();
  c10::SmallVector<THPObjectPtr, 8> outs;
  for (const Output& o : outputs_) {
    PyObject* x = nullptr;
    switch (o.kind) {
      case OutputKind::Scalar:
        x = PyLong_FromLongLong(v[o.index]);
        break;
      case OutputKind::View:
        x = THPVariable_Wrap(view(views_[o.index], args, frame, v));
        break;
      case OutputKind::Base:
        x = THPVariable_Wrap(base_tensor(o.base, args, frame));
        break;
      case OutputKind::Alias:
        x = Py_NewRef(outs[o.index].get());
        break;
    }
    if (!x) {
      throw py::error_already_set();
    }
    outs.emplace_back(x);
  }
  if (result_kind_ == 0) {
    Py_RETURN_NONE;
  }
  if (result_kind_ == 1) {
    return outs[0].release();
  }
  const auto n = static_cast<Py_ssize_t>(outs.size());
  PyObject* result = result_kind_ == 2 ? PyTuple_New(n) : PyList_New(n);
  if (!result) {
    throw py::error_already_set();
  }
  for (Py_ssize_t i = 0; i < n; ++i) {
    if (result_kind_ == 2) {
      PyTuple_SET_ITEM(result, i, outs[i].release());
    } else {
      PyList_SET_ITEM(result, i, outs[i].release());
    }
  }
  return result;
}

} // namespace torch::cuda::host_trace
#endif
