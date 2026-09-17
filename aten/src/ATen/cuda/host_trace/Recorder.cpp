#include <ATen/cuda/host_trace/Recorder.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/host_trace/Hooks.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstring>
#include <mutex>

namespace at::cuda::host_trace {

namespace {

thread_local TraceState* g_active = nullptr;
hooks::Hooks g_hooks;

constexpr const char* kNoDataPtr =
    "host_trace: data_ptr() / storage() on a traced tensor: a traced tensor has no raw "
    "address; read it with at::cuda::host_trace::sym_const_data_ptr(t) (an input) or "
    "sym_mutable_data_ptr(t) (an output), a c10::SymInt, and pass that to the kernel "
    "through launch() or a proxy field (declined)";

// allocation log
std::mutex g_log_mutex;
bool g_log_on = false;
at::DeviceIndex g_log_device = 0;
c10::cuda::MempoolId_t g_log_pool{0, 0};
std::vector<std::pair<uint64_t, uint64_t>> g_log;
c10::once_flag g_tracker_once;

bool g_test_reverse_nodes = false;
bool g_test_fail_capture_end = false;

cudaGraphNode_t frontier_node(
    TraceState* s,
    cudaStream_t stream,
    const std::string& what);

void open_packet(TraceState* s, cudaStream_t stream) {
  if (s->open_active) {
    return;
  }
  s->open = LaunchPacket{};
  // a launch's place in host order is where its events occur
  s->open.seq = s->t->seq++;
  s->open.stream = stream;
  s->open_active = true;
}
// The next recorder event, or the end of the host, follows the verbatim
// launch: its node is on the frontier of the stream it targeted.
void close_packet(TraceState* s) {
  if (!s->open_active) {
    return;
  }
  s->open.node = frontier_node(s, s->open.stream, "a verbatim launch");
  s->packets.push_back(std::move(s->open));
  s->open_active = false;
}

} // namespace

namespace hooks {
void set_hooks(const Hooks& h) {
  g_hooks = h;
}
} // namespace hooks

c10::SymFloat round_float32_symbolic(const c10::SymFloat& value) {
  TORCH_CHECK(g_hooks.round_float32 != nullptr, "host_trace: hooks not installed");
  return g_hooks.round_float32(value);
}

int64_t Hints::of(const c10::SymInt& s) {
  if (!s.is_heap_allocated()) {
    return s.as_int_unchecked();
  }
  TORCH_CHECK(
      g_hooks.int_hint != nullptr,
      "host_trace: a symbolic value reached the recorder before torch._C was "
      "initialized");
  return g_hooks.int_hint(s);
}
double Hints::of(const c10::SymFloat& s) {
  if (!s.is_symbolic()) {
    return s.expect_float();
  }
  TORCH_CHECK(g_hooks.float_hint != nullptr, "host_trace: hooks not installed");
  return g_hooks.float_hint(s);
}
bool Hints::of(const c10::SymBool& s) {
  if (!s.is_heap_allocated()) {
    return s.as_bool_unchecked();
  }
  TORCH_CHECK(g_hooks.bool_hint != nullptr, "host_trace: hooks not installed");
  return g_hooks.bool_hint(s);
}

// The recorder's own hint reads (complete here only, so no other TU can use it).
struct HintsInternal {
  static int64_t of(const c10::SymInt& s) {
    return Hints::of(s);
  }
  static double of(const c10::SymFloat& s) {
    return Hints::of(s);
  }
  static bool of(const c10::SymBool& s) {
    return Hints::of(s);
  }
};

void decline(const std::string& msg) {
  // @allow-raw-throw: registered with pybind11 as _HostTraceDeclined and exposed as Declined by torch/cuda/_host_trace.py
  throw Declined(msg);
}

const char* capture_error_name(int code) {
#ifdef USE_ROCM
  // the capture error family has no hipify mapping; the trace never captures on ROCm
  (void)code;
  return nullptr;
#else
  switch (static_cast<cudaError_t>(code)) {
    case cudaErrorStreamCaptureUnsupported:
      return "cudaErrorStreamCaptureUnsupported";
    case cudaErrorStreamCaptureInvalidated:
      return "cudaErrorStreamCaptureInvalidated";
    case cudaErrorStreamCaptureMerge:
      return "cudaErrorStreamCaptureMerge";
    case cudaErrorStreamCaptureUnmatched:
      return "cudaErrorStreamCaptureUnmatched";
    case cudaErrorStreamCaptureUnjoined:
      return "cudaErrorStreamCaptureUnjoined";
    case cudaErrorStreamCaptureIsolation:
      return "cudaErrorStreamCaptureIsolation";
    case cudaErrorStreamCaptureImplicit:
      return "cudaErrorStreamCaptureImplicit";
    case cudaErrorCapturedEvent:
      return "cudaErrorCapturedEvent";
    case cudaErrorStreamCaptureWrongThread:
      return "cudaErrorStreamCaptureWrongThread";
    // a capture begun while this thread's trace capture is open (a nested
    // CUDAGraph.capture_begin) fails with these rather than a capture code
    case cudaErrorIllegalState:
      return "cudaErrorIllegalState";
    case cudaErrorNotPermitted:
      return "cudaErrorNotPermitted";
    default:
      return nullptr;
  }
#endif
}

#define HT_DECLINE_UNLESS(cond, ...)      \
  if (C10_UNLIKELY(!(cond))) {            \
    decline(c10::str(__VA_ARGS__));       \
  }

TraceState* active() {
  return g_active;
}

namespace {
// The default CUDA generator's philox offset as the host will see it: under a
// capture philox_cuda_state(0) returns the per-capture intragraph offset.
// The Python dispatch keys are excluded so the first call, which creates the
// generator's per-capture seed/offset tensors, allocates REAL tensors.
uint64_t rng_probe(TraceState* s) {
  c10::impl::ExcludeDispatchKeyGuard no_python(c10::DispatchKeySet(
      {c10::DispatchKey::Python, c10::DispatchKey::PythonTLSSnapshot}));
  auto gen = at::cuda::detail::getDefaultCUDAGenerator(s->device);
  auto* g = at::check_generator<at::CUDAGeneratorImpl>(gen);
  std::lock_guard<std::mutex> lock(g->mutex_);
  at::PhiloxCudaState st = g->philox_cuda_state(0);
  return st.captured_ ? st.offset_intragraph_ : st.offset_.val;
}

// capture_end warns about an empty graph; on a decline the graph is empty by
// construction and the warning would only obscure the decline's own message
struct QuietWarnings final : c10::WarningHandler {
  void process(const c10::Warning& /*warning*/) override {}
};

// Abandon a capture that will not be finished (a decline, or a failed setup).
void abandon_capture(TraceState* s) {
  if (s->capturing) {
    try {
      QuietWarnings quiet;
      c10::WarningUtils::WarningHandlerGuard no_warnings(&quiet);
      // CUDAGraph ends a capture only on the stream that began it, and the
      // host may have left another stream current (finish_trace likewise)
      TORCH_INTERNAL_ASSERT(s->stream_guard.has_value());
      c10::cuda::CUDAStreamGuard on_capture_stream(
          s->stream_guard->current_stream());
      s->capture_graph->capture_end();
    } catch (...) { // NOLINT(bugprone-empty-catch): a decline is propagating
    }
    (void)cudaGetLastError();
    s->capturing = false;
  }
  s->capture_graph.reset();
  s->stream_guard.reset();
}
} // namespace

Scope::Scope(TraceState* s) : state(s), prev(g_active) {
#ifdef USE_ROCM
  // The recorder's parameter-layout queries have no HIP counterpart yet.
  decline("host tracing is CUDA-only in this version (ROCm build)");
#endif
  try {
    // The legacy default stream cannot be captured; make a pool stream current
    // so the host's getCurrentCUDAStream() launches land in the capture.
    c10::cuda::CUDAStream side =
        c10::cuda::getStreamFromPool(false, s->device);
    s->stream_guard.emplace(side);
    s->capture_stream = side.stream();
    {
      // Python dispatch keys excluded: capture_begin and the generator's first
      // per-capture use allocate real tensors, never traced ones.
      c10::impl::ExcludeDispatchKeyGuard no_python(c10::DispatchKeySet(
          {c10::DispatchKey::Python, c10::DispatchKey::PythonTLSSnapshot}));
      s->capture_graph = std::make_unique<at::cuda::CUDAGraph>(true);
      // thread-local, not relaxed: cudaMalloc, cudaDeviceSynchronize, a stream
      // synchronize or query on this stream and an event query on a captured
      // event fail inside the capture and the trace declines. The trace is
      // complete with respect to everything the capture sees; a synchronous
      // cudaMemcpy / cudaMemset from this thread is invisible to every capture
      // mode on CUDA 13.0 (measured), so it is a host-contract violation caught
      // by the HOSTTRACE_SYNC_API lint, not by the capture (Recorder.h). The
      // replay's build capture is relaxed (see torch/cuda/_host_trace.py):
      // there the ordinary host runs for real and the tape is checked byte for
      // byte against what was captured.
      s->capture_graph->capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
    }
    s->capturing = true;
    cudaStreamCaptureStatus status{};
    C10_CUDA_CHECK(cudaStreamGetCaptureInfo(
        s->capture_stream, &status, &s->capture_id));
    TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatusActive);
    s->rng_offset_before = rng_probe(s);
  } catch (...) {
    abandon_capture(s);
    throw;
  }
  // trace mode is entered only once everything above succeeded: an exception
  // must not leave a dangling active state behind a skipped destructor
  g_active = s;
}

Scope::~Scope() {
  // this scope's own state, never the thread's active pointer; capturing
  // here means the host threw (a decline): abandon quietly
  abandon_capture(state);
  g_active = prev;
}

ThreadScope::ThreadScope(TraceState* s) : prev(g_active) {
  TORCH_CHECK(
      s != nullptr && s->capturing,
      "host_trace: an op of a trace reached another thread after the trace "
      "ended");
  // this thread's own trace, if any, is the tracing thread's case (the mode
  // is on it) or another trace's, which never routes ops here
  TORCH_CHECK(
      prev == nullptr || prev == s,
      "host_trace: an op of one trace reached a thread tracing another");
  g_active = s;
}

ThreadScope::~ThreadScope() {
  g_active = prev;
}

void require_capturing_stream(
    TraceState* s,
    cudaStream_t stream,
    const char* what) {
  cudaStreamCaptureStatus status{};
  unsigned long long id = 0;
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &id));
  // a capture the driver invalidated under the trace: a call the capture
  // forbids ran since it opened, on this thread (the trace holds the cyclic
  // collector off, so it is the traced function's own: an entry it dropped,
  // whose CUDAGraph destroys its graph here, or a synchronous memory API) or
  // from another thread issuing work to the capturing stream
  HT_DECLINE_UNLESS(
      status != cudaStreamCaptureStatusInvalidated,
      "host_trace: ",
      what,
      " after the trace's capture was invalidated: a CUDA call the capture "
      "forbids ran during the traced call (a CUDAGraph torn down or a "
      "synchronous memory API on this thread, work another thread issued to "
      "the capturing stream; declined)");
  HT_DECLINE_UNLESS(
      status == cudaStreamCaptureStatusActive && id == s->capture_id,
      "host_trace: ",
      what,
      " on a stream that is not the trace's capturing stream: it would run "
      "for real, outside the capture, with the traced tensors' placeholder "
      "addresses (the host must issue it on its current stream, which the "
      "trace made the capturing stream; declined)");
}

namespace {
// The kernel node the capture created for the launch the host issued on
// `stream`: the one node on the stream's capture frontier that no earlier
// record claimed. Right after a launch the frontier is that node alone; by
// the time a verbatim record is closed the stream may have joined another
// stream, whose frontier (nodes of earlier records) is then part of its own.
cudaGraphNode_t frontier_node(
    TraceState* s,
    cudaStream_t stream,
    const std::string& what) {
#ifdef USE_ROCM
  (void)s;
  (void)stream;
  (void)what;
  TORCH_CHECK(false, "host tracing is CUDA-only in this version");
#else
  cudaStreamCaptureStatus status{};
  unsigned long long id = 0;
  cudaGraph_t graph = nullptr;
  const cudaGraphNode_t* frontier = nullptr;
  const cudaGraphEdgeData* edges = nullptr;
  size_t n = 0;
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(
      stream, &status, &id, &graph, &frontier, &edges, &n));
  HT_DECLINE_UNLESS(
      status == cudaStreamCaptureStatusActive && id == s->capture_id,
      "host_trace: ",
      what,
      " on a stream that is not in the trace's capture: the kernel ran for "
      "real, outside the capture, with the traced tensors' placeholder "
      "addresses (declined)");
  cudaGraphNode_t node = nullptr;
  size_t fresh = 0;
  for (size_t i = 0; i < n; ++i) {
    const bool claimed = std::any_of(
        s->packets.begin(), s->packets.end(), [&](const LaunchPacket& pk) {
          return pk.node == frontier[i];
        });
    if (!claimed) {
      node = frontier[i];
      ++fresh;
    }
  }
  HT_DECLINE_UNLESS(
      fresh != 0,
      "host_trace: ",
      what,
      " left no new node on the stream's capture frontier: it named another "
      "stream explicitly, or did not reach the driver (declined)");
  HT_DECLINE_UNLESS(
      fresh == 1,
      "host_trace: ",
      what,
      ": ",
      fresh,
      " nodes the recorder did not record are on the stream's capture "
      "frontier; the launch cannot be told apart from work the recorder did "
      "not see (declined)");
  return node;
#endif
}
} // namespace

void register_root(
    const at::TensorBase& t,
    c10::SymInt root,
    int64_t itemsize,
    bool allocation,
    std::string name) {
  TraceState* s = g_active;
  TORCH_CHECK(s != nullptr, "host_trace: register_root outside a trace");
  s->roots[t.unsafeGetTensorImpl()] =
      RootInfo{std::move(root), itemsize, allocation, std::move(name)};
}

void drop_storage(const at::TensorBase& t) {
  t.unsafeGetTensorImpl()->release_storage_and_set_meta_custom_data_ptr_error_msg_(
      std::string(kNoDataPtr));
}

int64_t next_seq() {
  TraceState* s = g_active;
  TORCH_CHECK(s != nullptr, "host_trace: next_seq outside a trace");
  return s->t->seq++;
}

namespace {
c10::SymInt sym_data_ptr_impl(const at::TensorBase& t, bool mutable_access) {
  // Ordinary mode and a real tensor met inside a trace: the concrete address.
  // const_data_ptr() leaves a copy-on-write tensor lazy; mutable_data_ptr()
  // materializes it, as the host's own data_ptr() did before the conversion.
  const auto concrete = [&]() {
    const void* p = mutable_access ? t.mutable_data_ptr() : t.const_data_ptr();
    return c10::SymInt(static_cast<int64_t>(reinterpret_cast<uintptr_t>(p)));
  };
  TraceState* s = g_active;
  if (s == nullptr) {
    return concrete();
  }
  auto it = s->roots.find(t.unsafeGetTensorImpl());
  if (it == s->roots.end()) {
    HT_DECLINE_UNLESS(
        !t.unsafeGetTensorImpl()->has_symbolic_sizes_strides(),
        "host_trace: a tensor with symbolic sizes that the trace did not "
        "create reached an address accessor (declined)");
    // a real tensor inside a trace (the generator's seed/offset, a
    // module-owned table): a constant address
    return concrete();
  }
  const RootInfo& r = it->second;
  // a mutable read names a root the tape writes (once per root: a view
  // carries its root's name)
  if (mutable_access) {
    auto& written = s->t->written_roots;
    if (std::find(written.begin(), written.end(), r.name) == written.end()) {
      written.push_back(r.name);
    }
  }
  // TensorImpl::data()'s rule, which the concrete accessors follow: a tensor
  // with no elements has a null data pointer whatever its storage (an empty
  // slice of a nonempty tensor; parallel_cat passes that null for an empty
  // piece). A guard on the sizes when they are symbolic.
  if (t.sym_numel() == c10::SymInt(0)) {
    return c10::SymInt(0);
  }
  return r.root + t.sym_storage_offset() * c10::SymInt(r.itemsize);
}
} // namespace

c10::SymInt sym_const_data_ptr(const at::TensorBase& t) {
  return sym_data_ptr_impl(t, /*mutable_access=*/false);
}

c10::SymInt sym_mutable_data_ptr(const at::TensorBase& t) {
  return sym_data_ptr_impl(t, /*mutable_access=*/true);
}

namespace {
// the symbol for an opaque result, with the domain the host declared for it
c10::SymInt opaque_symbol(
    const std::string& fn,
    int64_t value,
    const char* domain) {
  const bool positive = std::strcmp(domain, "positive") == 0;
  TORCH_CHECK(
      positive || std::strcmp(domain, "int") == 0,
      "host_trace: opaque ",
      fn,
      " declares an unknown domain \"",
      domain,
      "\" (\"int\" or \"positive\")");
  TORCH_CHECK(
      !positive || value >= 1,
      "host_trace: opaque ",
      fn,
      " declares a positive result but computed ",
      value);
  TORCH_CHECK(g_hooks.new_int_sym != nullptr, "host_trace: hooks not installed");
  return g_hooks.new_int_sym(value, fn, positive);
}
} // namespace

c10::SymInt opaque(
    const std::string& fn,
    std::vector<c10::SymInt> args,
    int64_t (*impl)(const std::vector<int64_t>&),
    const char* kind,
    const char* domain) {
  std::vector<int64_t> hints;
  hints.reserve(args.size());
  for (const auto& a : args) {
    hints.push_back(HintsInternal::of(a));
  }
  const int64_t r = impl(hints);
  TraceState* s = g_active;
  if (s == nullptr) {
    return c10::SymInt(r);
  }
  c10::SymInt sym = opaque_symbol(fn, r, domain);
  OpaqueRec rec{s->t->seq++, fn, std::move(args), r, sym, impl, kind, domain};
  s->t->opaque.push_back(std::move(rec));
  return sym;
}

c10::SymInt opaque(
    const char* fn,
    c10::ArrayRef<c10::SymInt> args,
    int64_t (*impl)(const std::vector<int64_t>&),
    const char* kind,
    int64_t traced_value,
    const char* domain) {
  TraceState* s = g_active;
  if (s == nullptr) {
    return c10::SymInt(traced_value);
  }
  c10::SymInt sym = opaque_symbol(fn, traced_value, domain);
  OpaqueRec rec{
      s->t->seq++, fn, args.vec(), traced_value, sym, impl, kind, domain};
  s->t->opaque.push_back(std::move(rec));
  return sym;
}

int64_t rng_increment(const c10::SymInt& v) {
  TraceState* s = g_active;
  if (s == nullptr) {
    return v.expect_int();
  }
  TORCH_CHECK(
      !s->t->rng_increment.has_value(),
      "host_trace: rng_increment declared twice in one host");
  s->t->rng_increment = v;
  s->rng_increment_hint = HintsInternal::of(v);
  return s->rng_increment_hint;
}

// A typed launch record: complete at the call; the node comes from the
// frontier right after the launch (typed_launched).
void typed_launch(TraceState* s, LaunchPacket&& pk) {
  TORCH_INTERNAL_ASSERT(s != nullptr);
  close_packet(s);
  pk.seq = s->t->seq++;
  s->packets.push_back(std::move(pk));
}

void typed_launched(TraceState* s, cudaStream_t stream) {
  LaunchPacket& pk = s->packets.back();
  pk.node = frontier_node(s, stream, "launch() of `" + pk.kernel + "`");
}

void test_reverse_node_order(bool on) {
  g_test_reverse_nodes = on;
}

bool test_reverse_nodes() {
  return g_test_reverse_nodes;
}

void test_fail_capture_end(bool on) {
  g_test_fail_capture_end = on;
}

void pending_proxy(
    const void* pod,
    size_t size,
    const std::vector<FieldRec>* fields) {
  TraceState* s = g_active;
  TORCH_INTERNAL_ASSERT(s != nullptr);
  cudaStream_t current = c10::cuda::getCurrentCUDAStream(s->device).stream();
  require_capturing_stream(
      s, current, "a verbatim launch (proxy struct converted)");
  if (s->open_active && s->open.has_proxy) {
    close_packet(s); // a second struct conversion: the next launch
  }
  open_packet(s, current);
  s->open.has_proxy = true;
  // snapshot: the host may mutate the struct before the next launch
  s->open.pod.assign(
      static_cast<const uint8_t*>(pod),
      static_cast<const uint8_t*>(pod) + size);
  s->open.fields = *fields;
}

Grid::operator dim3() const {
  TraceState* s = g_active;
  if (s != nullptr) {
    cudaStream_t current = c10::cuda::getCurrentCUDAStream(s->device).stream();
    require_capturing_stream(s, current, "a verbatim launch (Grid converted)");
    if (s->open_active && s->open.has_grid) {
      close_packet(s); // a second grid conversion: the next launch
    }
    open_packet(s, current);
    s->open.has_grid = true;
    s->open.grid = {x, y, z};
    return dim3(
        static_cast<unsigned>(HintsInternal::of(x)),
        static_cast<unsigned>(HintsInternal::of(y)),
        static_cast<unsigned>(HintsInternal::of(z)));
  }
  return dim3(
      static_cast<unsigned>(x.expect_int()),
      static_cast<unsigned>(y.expect_int()),
      static_cast<unsigned>(z.expect_int()));
}

const FuncInfo& func_info(const void* host_func) {
  static std::unordered_map<const void*, FuncInfo> cache;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = cache.find(host_func);
  if (it != cache.end()) {
    return it->second;
  }
  FuncInfo info;
#ifdef USE_ROCM
  // cudaFuncGetName and cudaFuncGetParamInfo have no HIP counterpart in
  // hipify's mappings; a trace never gets here on ROCm because Scope declines
  // first.
  TORCH_CHECK(false, "host tracing is CUDA-only in this version");
#else
  const char* name = nullptr;
  C10_CUDA_CHECK(cudaFuncGetName(&name, host_func));
  info.name = name != nullptr ? name : "";
  for (size_t i = 0;; ++i) {
    size_t off = 0, size = 0;
    if (cudaFuncGetParamInfo(host_func, i, &off, &size) != cudaSuccess) {
      (void)cudaGetLastError();
      break;
    }
    info.params.emplace_back(off, size);
    info.image_size = std::max(info.image_size, off + size);
  }
#endif
  return cache.emplace(host_func, std::move(info)).first->second;
}

namespace {

// Record one captured kernel node of a verbatim launch: grid values from the
// packet's Grid, the field table of its proxy; block and shared memory as the
// node reports them, constants of the variant by the verbatim contract (the
// host passes compile-time constants of the kernel instantiation there).
void record_launch(
    TraceState* s,
    const FuncInfo& info,
    const cudaKernelNodeParams& p,
    LaunchPacket& pk) {
  // the pairing check a verbatim record allows: the dims its Grid handed the
  // launch are the node's
  TORCH_INTERNAL_ASSERT(
      !pk.has_grid ||
          (p.gridDim.x == static_cast<unsigned>(HintsInternal::of(pk.grid[0])) &&
           p.gridDim.y == static_cast<unsigned>(HintsInternal::of(pk.grid[1])) &&
           p.gridDim.z == static_cast<unsigned>(HintsInternal::of(pk.grid[2]))),
      "host_trace: the node paired with a verbatim launch of `",
      info.name,
      "` has another grid than its Grid conversion");
  LaunchRec rec;
  rec.seq = pk.seq;
  rec.kernel = info.name;
  rec.func = p.func;
  rec.param_layout = info.params;
  rec.grid = pk.has_grid
      ? pk.grid
      : std::array<c10::SymInt, 3>{
            c10::SymInt(p.gridDim.x),
            c10::SymInt(p.gridDim.y),
            c10::SymInt(p.gridDim.z)};
  rec.block = {p.blockDim.x, p.blockDim.y, p.blockDim.z};
  rec.block_expr = {
      c10::SymInt(p.blockDim.x),
      c10::SymInt(p.blockDim.y),
      c10::SymInt(p.blockDim.z)};
  rec.smem = c10::SymInt(static_cast<int64_t>(p.sharedMemBytes));
  std::vector<uint8_t> image(info.image_size, 0);
  // A verbatim proxy must be passed exactly once, by value and unchanged.
  // Its position is structural only when the POD size has one ABI match.
  std::optional<size_t> proxy_index;
  if (pk.has_proxy) {
    for (size_t i = 0; i < info.params.size(); ++i) {
      if (info.params[i].second != pk.pod.size()) {
        continue;
      }
      if (proxy_index) {
        decline(
            "host_trace: verbatim proxy has multiple size-compatible "
            "parameters; use host_trace::launch for positional arguments");
      }
      proxy_index = i;
    }
    if (!proxy_index) {
      decline("host_trace: verbatim proxy has no size-compatible parameter");
    }
  }
  // The verbatim contract: the proxy is the launch's only formal. A bare
  // scalar or a by-value struct beside it would be a constant of the variant
  // read from the captured bytes, with nothing showing that it does not
  // follow a size; a kernel with such parameters launches through launch().
  const size_t described = pk.has_proxy ? 1 : 0;
  HT_DECLINE_UNLESS(
      info.params.size() == described,
      "host_trace: verbatim launch of `",
      info.name,
      "` passes ",
      info.params.size(),
      " parameter(s) but the tape describes ",
      described == 1 ? "only its proxy struct" : "none",
      ": a verbatim <<<>>> passes exactly one converted proxy struct and "
      "nothing else; put the other values in the proxy, or launch through "
      "at::cuda::host_trace::launch (declined)");
  if (pk.has_proxy) {
    auto [off, size] = info.params[*proxy_index];
    const uint8_t* bytes =
        static_cast<const uint8_t*>(p.kernelParams[*proxy_index]);
    std::memcpy(image.data() + off, bytes, size);
    if (std::memcmp(bytes, pk.pod.data(), size) != 0) {
      decline(
          "host_trace: verbatim proxy bytes differ from its selected "
          "parameter; pass the proxy unchanged");
    }
    for (const FieldRec& fr : pk.fields) {
      FieldRec g = fr;
      g.offset += off;
      rec.params.push_back(std::move(g));
    }
  }
  rec.hint_image = std::move(image);
  s->t->launches.push_back(std::move(rec));
}

// A typed launch's record: the packet is complete, nothing to decode.
void emit_typed(TraceState* s, const FuncInfo& info, LaunchPacket& pk) {
  LaunchRec rec;
  rec.seq = pk.seq;
  rec.kernel = pk.kernel;
  rec.func = pk.func;
  rec.param_layout = info.params;
  rec.grid = pk.grid;
  rec.block = pk.block;
  rec.block_expr = pk.block_expr;
  rec.smem = pk.smem;
  rec.params = std::move(pk.params);
  rec.hint_image = std::move(pk.image);
  s->t->launches.push_back(std::move(rec));
}

void check_rng_consumption(TraceState* s) {
  const uint64_t after = rng_probe(s);
  s->rng_consumed = after - s->rng_offset_before;
  if (s->rng_increment_hint >= 0) {
    HT_DECLINE_UNLESS(
        static_cast<uint64_t>(s->rng_increment_hint) == s->rng_consumed,
        "host_trace: rng_increment declares ",
        s->rng_increment_hint,
        " philox offsets but the host consumed ",
        s->rng_consumed,
        " (declined)");
  } else {
    HT_DECLINE_UNLESS(
        s->rng_consumed == 0,
        "host_trace: the host consumed ",
        s->rng_consumed,
        " philox offsets without declaring rng_increment: write the increment "
        "read as at::cuda::host_trace::rng_increment(expr) so the tape carries "
        "the value (declined)");
  }
}

const char* node_type_name(cudaGraphNodeType type) {
  switch (type) {
    case cudaGraphNodeTypeMemcpy:
      return "memcpy";
    case cudaGraphNodeTypeMemset:
      return "memset";
    case cudaGraphNodeTypeHost:
      return "host callback";
    case cudaGraphNodeTypeGraph:
      return "child graph";
    case cudaGraphNodeTypeEmpty:
      return "empty";
    case cudaGraphNodeTypeEventRecord:
      return "event record";
    case cudaGraphNodeTypeWaitEvent:
      return "event wait";
    default:
      return "unsupported";
  }
}

} // namespace

void finish_trace(TraceState* s) {
  TORCH_CHECK(
      s->capturing,
      "host_trace: finish_trace called twice or outside a trace");
  // On the capture's own stream, whatever the host left current: the
  // generator reads its capture state off the current stream, and CUDAGraph
  // ends a capture only on the stream that began it (abandon_capture likewise)
  TORCH_INTERNAL_ASSERT(s->stream_guard.has_value());
  c10::cuda::CUDAStreamGuard on_capture_stream(
      s->stream_guard->current_stream());
  // before the capture ends: the generator's capture state is keyed by it
  check_rng_consumption(s);
  close_packet(s);
  // The capture is over once capture_end is called, whatever it returns or
  // throws (CUDAGraph's own rule). A second call from Scope's destructor
  // would decrement the allocator's capture count again: another thread's
  // open capture would then fail at its own end.
  s->capturing = false;
  try {
    s->capture_graph->capture_end();
    TORCH_CHECK(
        !g_test_fail_capture_end,
        "host_trace: test hook: capture_end failed after the capture ended");
  } catch (const c10::AcceleratorError& e) {
    const char* name = capture_error_name(e.get_error_code());
    if (name == nullptr) {
      throw;
    }
    decline(c10::str(
        "host_trace: the trace capture ended with ",
        name,
        ": the host, or another thread, performed an operation that is "
        "illegal inside a stream capture (declined)"));
  }
  cudaGraph_t graph = s->capture_graph->raw_cuda_graph();
  size_t n = 0;
  C10_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &n));
  std::vector<cudaGraphNode_t> nodes(n);
  C10_CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &n));
  if (g_test_reverse_nodes) {
    std::reverse(nodes.begin(), nodes.end());
  }
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type{};
    C10_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
    HT_DECLINE_UNLESS(
        type == cudaGraphNodeTypeKernel,
        "host_trace: the host produced a ",
        node_type_name(type),
        " node, which this tape does not describe; memcpy, memset, events and "
        "child graphs inside a host are not traced (declined)");
  }
  // Completeness, by handle: every captured node is the node of exactly one
  // launch record. The count is the cheap test; membership names the kernel
  // the recorder did not see.
  std::unordered_map<cudaGraphNode_t, size_t> owner;
  for (size_t i = 0; i < s->packets.size(); ++i) {
    cudaGraphNode_t node = s->packets[i].node;
    TORCH_INTERNAL_ASSERT(
        node != nullptr && owner.emplace(node, i).second,
        "host_trace: launch record ",
        i,
        " has no kernel node of its own");
  }
  bool complete = nodes.size() == owner.size();
  for (size_t i = 0; complete && i < nodes.size(); ++i) {
    complete = owner.count(nodes[i]) != 0;
  }
  if (!complete) {
    for (cudaGraphNode_t node : nodes) {
      if (owner.count(node) != 0) {
        continue;
      }
      cudaKernelNodeParams p{};
      C10_CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &p));
      decline(c10::str(
          "host_trace: captured kernel `",
          func_info(p.func).name,
          "` has no launch record: launch it through "
          "at::cuda::host_trace::launch(...), or as a verbatim <<<>>> with a "
          "Grid or a proxy struct (captured ",
          nodes.size(),
          " kernel node(s), ",
          s->packets.size(),
          " launch record(s); declined)"));
    }
    TORCH_INTERNAL_ASSERT(
        false, "host_trace: a launch record claims a node outside the capture");
  }
  for (LaunchPacket& pk : s->packets) {
    if (pk.stream != s->capture_stream) {
      s->t->all_on_capture_stream = false;
    }
    cudaKernelNodeParams p{};
    C10_CUDA_CHECK(cudaGraphKernelNodeGetParams(pk.node, &p));
    HT_DECLINE_UNLESS(
        p.kernelParams != nullptr && p.extra == nullptr,
        "host_trace: captured launch used the `extra` parameter buffer; not "
        "supported (declined)");
    const FuncInfo& info = func_info(p.func);
    if (!pk.typed) {
      record_launch(s, info, p, pk);
      continue;
    }
    // the node was read from the frontier right after this launch: a
    // difference is a recorder error, not something the host did
    TORCH_INTERNAL_ASSERT(
        p.func == pk.func,
        "host_trace: the node paired with launch() of `",
        pk.kernel,
        "` is `",
        info.name,
        "`");
    for (size_t k = 0; k < info.params.size(); ++k) {
      auto [off, size] = info.params[k];
      TORCH_INTERNAL_ASSERT(
          std::memcmp(p.kernelParams[k], pk.image.data() + off, size) == 0,
          "host_trace: captured bytes of parameter ",
          k,
          " of `",
          pk.kernel,
          "` differ from the bytes launch() handed to the driver");
    }
    TORCH_INTERNAL_ASSERT(
        p.gridDim.x == static_cast<unsigned>(HintsInternal::of(pk.grid[0])) &&
            p.gridDim.y == static_cast<unsigned>(HintsInternal::of(pk.grid[1])) &&
            p.gridDim.z == static_cast<unsigned>(HintsInternal::of(pk.grid[2])) &&
            p.blockDim.x == static_cast<unsigned>(pk.block[0]) &&
            p.blockDim.y == static_cast<unsigned>(pk.block[1]) &&
            p.blockDim.z == static_cast<unsigned>(pk.block[2]) &&
            p.sharedMemBytes == static_cast<unsigned>(HintsInternal::of(pk.smem)),
        "host_trace: the captured launch configuration of `",
        pk.kernel,
        "` differs from the launch() record");
    emit_typed(s, info, pk);
  }
  s->packets.clear();
  s->capture_graph.reset(); // destroys the template and releases the pool
}

// An allocation belongs to the build when the allocator served it from the
// build capture's pool: the allocator routes to that pool by the allocating
// stream's capture, queried on the allocating thread. The tracker itself
// makes no CUDA call: it runs under the allocator's lock on whichever thread
// allocates, and a capture query on another thread's stream from there raced
// that thread's cudaStreamEndCapture and faulted inside the driver.
void alloc_log_begin(at::DeviceIndex device, c10::cuda::MempoolId_t pool) {
  TORCH_CHECK(
      pool.first != 0 || pool.second != 0,
      "host_trace: the allocation log needs the build capture's pool");
  // the tracker is the native caching allocator's; the other backends
  // (cudaMallocAsync, a pluggable allocator) raise on attaching one
  if (const std::string backend = c10::cuda::CUDACachingAllocator::name();
      backend != "native") {
    decline(
        "host_trace: a build reads its allocations from the caching "
        "allocator's trace tracker, which the " +
        backend +
        " allocator backend does not provide "
        "(PYTORCH_CUDA_ALLOC_CONF=backend:" +
        backend + ")");
  }
  c10::call_once(g_tracker_once, [] {
    c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
        [](const c10::CachingDeviceAllocator::TraceEntry& e) {
          if (e.action_ != c10::CachingDeviceAllocator::TraceEntry::ALLOC) {
            return;
          }
          std::lock_guard<std::mutex> lock(g_log_mutex);
          if (g_log_on && e.device_ == g_log_device &&
              e.mempool_ == g_log_pool) {
            g_log.emplace_back(
                static_cast<uint64_t>(e.addr_), static_cast<uint64_t>(e.size_));
          }
        });
  });
  std::lock_guard<std::mutex> lock(g_log_mutex);
  g_log.clear();
  g_log_device = device;
  g_log_pool = pool;
  g_log_on = true;
}

std::vector<std::pair<uint64_t, uint64_t>> alloc_log_end() {
  std::lock_guard<std::mutex> lock(g_log_mutex);
  g_log_on = false;
  return std::move(g_log);
}

} // namespace at::cuda::host_trace
