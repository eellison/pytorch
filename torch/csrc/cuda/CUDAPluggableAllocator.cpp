#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

namespace torch::cuda::CUDAPluggableAllocator {

int device_count = 0;

void custom_raw_deleter(void* ptr);

_AllocationMetadata::_AllocationMetadata() : size(0), device_idx(-1) {}

_AllocationMetadata::_AllocationMetadata(
    size_t size,
    c10::DeviceIndex device_idx,
    cudaStream_t stream)
    : size(size), device_idx(device_idx), stream(stream) {}

// This is a fast API to just register allocators
// based on function pointers (ie. external .so libraries)
// This avoids having to link against libtorch for C++ based custom allocators
// And also use this from python
CUDAPluggableAllocator::CUDAPluggableAllocator(
    std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, cudaStream_t)> free_fn)
    : alloc_fn_(std::move(alloc_fn)), free_fn_(std::move(free_fn)) {}

CUDAPluggableAllocator::CUDAPluggableAllocator(CUDAPluggableAllocator& other)
    : alloc_fn_(other.alloc_fn_),
      free_fn_(other.free_fn_),
      init_fn_(other.init_fn_),
      reset_fn_(other.reset_fn_),
      memory_fraction_fn_(other.memory_fraction_fn_),
      base_alloc_fn_(other.base_alloc_fn_),
      record_stream_fn_(other.record_stream_fn_),
      begin_allocate_to_pool_fn_(other.begin_allocate_to_pool_fn_),
      end_allocate_to_pool_fn_(other.end_allocate_to_pool_fn_),
      relase_pool_fn_(other.relase_pool_fn_) {}

void CUDAPluggableAllocator::set_init_fn(std::function<void(int)> init_fn) {
  init_fn_ = std::move(init_fn);
}

void CUDAPluggableAllocator::set_reset_fn(std::function<void()> reset_fn) {
  reset_fn_ = std::move(reset_fn);
}

void CUDAPluggableAllocator::set_memory_fraction_fn(
    std::function<void(double, int)> memory_fraction_fn) {
  memory_fraction_fn_ = std::move(memory_fraction_fn);
}

void CUDAPluggableAllocator::set_base_alloc_fn(
    std::function<void*(void*, size_t*)> base_alloc_fn) {
  base_alloc_fn_ = std::move(base_alloc_fn);
}

void CUDAPluggableAllocator::set_record_stream_fn(
    std::function<void(void* ptr, cudaStream_t stream)> record_stream_fn) {
  record_stream_fn_ = std::move(record_stream_fn);
}

void CUDAPluggableAllocator::set_begin_allocate_to_pool(
    std::function<
        void(int, c10::cuda::MempoolId_t, std::function<bool(cudaStream_t)>)>
        capture_begin_fn) {
  begin_allocate_to_pool_fn_ = std::move(capture_begin_fn);
}

void CUDAPluggableAllocator::set_end_allocate_to_pool_fn(
    std::function<void(int, c10::cuda::MempoolId_t)> capture_about_to_end_fn) {
  end_allocate_to_pool_fn_ = std::move(capture_about_to_end_fn);
}

void CUDAPluggableAllocator::set_release_pool(
    std::function<void(int, c10::cuda::MempoolId_t)> capture_destroy_fn) {
  relase_pool_fn_ = std::move(capture_destroy_fn);
}

void* CUDAPluggableAllocator::malloc(
    size_t size,
    c10::DeviceIndex device,
    cudaStream_t stream) {
  void* r = alloc_fn_(size, device, stream);
  if (r != nullptr) {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    allocation_metadata_.emplace(r, _AllocationMetadata(size, device, stream));
  }
  return r;
}

c10::DataPtr CUDAPluggableAllocator::allocate(size_t size) {
  c10::DeviceIndex device = -1;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device);
  void* r = this->malloc(size, device, stream);
  c10::DataPtr data_ptr = {
      r, r, raw_deleter(), c10::Device(c10::DeviceType::CUDA, device)};
  return data_ptr;
}

c10::DeleterFnPtr CUDAPluggableAllocator::raw_deleter() const {
  return &custom_raw_deleter;
}

void* CUDAPluggableAllocator::raw_alloc(size_t nbytes) {
  c10::DeviceIndex device = -1;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device);
  return malloc(nbytes, device, stream);
}

void* CUDAPluggableAllocator::raw_alloc_with_stream(
    size_t nbytes,
    cudaStream_t stream) {
  c10::DeviceIndex device = -1;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  return malloc(nbytes, device, stream);
}

void CUDAPluggableAllocator::raw_delete(void* ptr) {
  // as for the caching allocator: raw_alloc(0) may return nullptr, which is
  // not recorded
  if (ptr == nullptr) {
    return;
  }
  cudaStream_t stream{};
  c10::DeviceIndex device_idx = -1;
  size_t size = 0;
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    TORCH_CHECK(
        allocation_metadata_.count(ptr),
        "Trying to free a pointer not allocated here");
    _AllocationMetadata& metadata = allocation_metadata_[ptr];
    size = metadata.size;
    device_idx = metadata.device_idx;
    stream = metadata.stream;
    allocation_metadata_.erase(ptr);
  }
  free_fn_(ptr, size, device_idx, stream);
}

void CUDAPluggableAllocator::init(int device_count) {
  if (init_fn_) {
    init_fn_(device_count);
  }
  initialized_ = true;
}

bool CUDAPluggableAllocator::initialized() {
  return initialized_;
}

double CUDAPluggableAllocator::getMemoryFraction(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support getMemoryFraction. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::setMemoryFraction(
    double fraction,
    c10::DeviceIndex device) {
  if (memory_fraction_fn_) {
    memory_fraction_fn_(fraction, device);
  }
}

std::vector<c10::cuda::CUDACachingAllocator::StreamSegmentSize>
CUDAPluggableAllocator::getExpandableSegmentSizes(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "CUDAMallocAsyncAllocator does not yet support getExpandableSegmentSizes.");
}

void CUDAPluggableAllocator::emptyCache(
    /*unused*/ c10::cuda::MempoolId_t mempool_id) {
  if (reset_fn_) {
    reset_fn_();
  }
}

void CUDAPluggableAllocator::cacheInfo(
    c10::DeviceIndex device,
    size_t* largestBlock) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support cacheInfo. "
      "If you need it, please file an issue describing your use case.");
}

void* CUDAPluggableAllocator::getBaseAllocation(void* ptr, size_t* size) {
  if (base_alloc_fn_) {
    return base_alloc_fn_(ptr, size);
  } else {
    return ptr;
  }
}

void CUDAPluggableAllocator::recordStream(
    const c10::DataPtr& ptr,
    c10::cuda::CUDAStream stream) {
  if (record_stream_fn_) {
    record_stream_fn_(ptr.get(), stream);
  }
}

c10::CachingDeviceAllocator::DeviceStats CUDAPluggableAllocator::getDeviceStats(
    c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support getDeviceStats. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support resetAccumulatedStats. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::resetPeakStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support resetPeakStats. "
      "If you need it, please file an issue describing your use case.");
}

c10::cuda::CUDACachingAllocator::SnapshotInfo CUDAPluggableAllocator::snapshot(
    c10::cuda::MempoolId_t mempool_id,
    bool include_traces) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support snapshot. "
      "If you need it, please file an issue describing your use case.");
}

c10::cuda::CUDACachingAllocator::ShareableHandle CUDAPluggableAllocator::
    shareIpcHandle(void* ptr) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support shareIPcHandle. "
      "If you need it, please file an issue describing your use case.");
}

std::shared_ptr<void> CUDAPluggableAllocator::getIpcDevPtr(std::string handle) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support getIpcDevPtr. "
      "If you need it, please file an issue describing your use case.");
}

// CUDAGraph interactions
void CUDAPluggableAllocator::beginAllocateToPool(
    c10::DeviceIndex device,
    c10::cuda::MempoolId_t mempool_id,
    std::function<bool(cudaStream_t)> filter) {
  if (begin_allocate_to_pool_fn_) {
    begin_allocate_to_pool_fn_(device, mempool_id, std::move(filter));
  }
}

void CUDAPluggableAllocator::endAllocateToPool(
    c10::DeviceIndex device,
    c10::cuda::MempoolId_t mempool_id) {
  if (end_allocate_to_pool_fn_) {
    end_allocate_to_pool_fn_(device, mempool_id);
  }
}

void CUDAPluggableAllocator::releasePool(
    c10::DeviceIndex device,
    c10::cuda::MempoolId_t mempool_id) {
  if (relase_pool_fn_) {
    relase_pool_fn_(device, mempool_id);
  }
}

void CUDAPluggableAllocator::recordHistory(
    bool enabled,
    c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    c10::cuda::CUDACachingAllocator::RecordContext when,
    bool clearHistory,
    const std::vector<std::string>& skip_actions) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support recordHistory. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::attachOutOfMemoryObserver(
    c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support attachOutOfMemoryObserver. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::attachOomRejectionObserver(
    c10::cuda::CUDACachingAllocator::OomRejectionObserver observer) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "CUDAPluggableAllocator does not yet support attachOomRejectionObserver. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::attachAllocatorTraceTracker(
    c10::cuda::CUDACachingAllocator::AllocatorTraceTracker tracker) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not support attachAllocatorTraceTracker. "
      "attachAllocatorTraceTracker is only used inside Pytorch.");
}

std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState>
CUDAPluggableAllocator::getCheckpointState(
    c10::DeviceIndex device,
    at::cuda::MempoolId_t id) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support getCheckpointState. "
      "If you need it, please file an issue describing your use case.");
}

c10::cuda::CUDACachingAllocator::CheckpointDelta CUDAPluggableAllocator::
    setCheckpointPoolState(
        c10::DeviceIndex device,
        std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> pps) {
  TORCH_CHECK(
      false,
      "CUDAPluggableAllocator does not yet support setCheckpointPoolState. "
      "If you need it, please file an issue describing your use case.");
}

void CUDAPluggableAllocator::enablePeerAccess(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access) {
  c10::cuda::CUDAGuard device_guard(dev);
  cudaError_t err = cudaDeviceEnablePeerAccess(dev_to_access, 0);
  if (err == cudaErrorPeerAccessAlreadyEnabled) {
    // ignore and clear the error if access was already enabled
    (void)cudaGetLastError();
  } else {
    C10_CUDA_CHECK(err);
  }
}

cudaError_t CUDAPluggableAllocator::memcpyAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    cudaStream_t stream,
    bool p2p_enabled) {
  return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
}

std::string CUDAPluggableAllocator::name() {
  return "pluggable";
}

void CUDAPluggableAllocator::copy_data(
    void* dest,
    const void* src,
    std::size_t count) const {
  C10_CUDA_CHECK(cudaMemcpyAsync(
      dest,
      src,
      count,
      cudaMemcpyDeviceToDevice,
      c10::cuda::getCurrentCUDAStream()));
}

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
    current_custom_allocator;

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
getCurrentAllocator() {
  return current_custom_allocator;
}

// TODO: add more functions in the argument
std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
createCustomAllocator(
    std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, cudaStream_t)> free_fn) {
  std::shared_ptr<CUDAPluggableAllocator> allocator(
      new CUDAPluggableAllocator(std::move(alloc_fn), std::move(free_fn)));
  allocator->init(device_count);
  return allocator;
}

void changeCurrentAllocator(
    const std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>&
        allocator) {
  TORCH_CHECK(
      !c10::cuda::CUDACachingAllocator::allocator.load()->initialized(),
      "Can't swap an already initialized allocator");
  c10::cuda::CUDACachingAllocator::allocator.store(allocator.get());
  current_custom_allocator = allocator;
}

void custom_raw_deleter(void* ptr) {
  current_custom_allocator->raw_delete(ptr);
}

namespace {

namespace CCA = c10::cuda::CUDACachingAllocator;
using c10::cuda::CaptureId_t;
using c10::cuda::MempoolId_t;
using PoolKey = std::pair<c10::DeviceIndex, MempoolId_t>;

// One use_allocator context.
struct Window {
  std::shared_ptr<CCA::CUDAAllocator> allocator;
};

// A private pool whose captures a window's allocator serves.
struct Claim {
  std::shared_ptr<Window> window;
  // the capture into the pool in progress, if any
  CaptureId_t capture = 0;
};

// The DataPtr context of a tensor made by a window's allocator.
struct Owned {
  std::shared_ptr<CCA::CUDAAllocator> allocator;
  void* ptr;
};

void deleteOwned(void* ctx) {
  std::unique_ptr<Owned> owned(static_cast<Owned*>(ctx));
  owned->allocator->raw_delete(owned->ptr);
}

// The windows open on this thread, innermost last.
thread_local std::vector<std::shared_ptr<Window>> window_stack;
// A pool the caching allocator did not have at the beginAllocateToPool this
// thread ran last inside a window: claimed if a capture into it follows.
thread_local std::optional<PoolKey> fresh_pool;

class LayeredAllocator;
LayeredAllocator& layered();

// The current CUDA allocator while a use_allocator window is open on any
// thread. It forwards everything to the native caching allocator (inner),
// except the allocations inner would place into a pool claimed by a window
// while that window's capture into the pool is in progress: the window's
// allocator makes those, one call per allocation at its exact size.
class LayeredAllocator final : public CCA::CUDAAllocator {
 public:
  uint64_t beginWindow(std::shared_ptr<CCA::CUDAAllocator> allocator) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (open_windows_ == 0) {
        auto* current = CCA::allocator.load();
        TORCH_CHECK(
            current->name() == "native",
            "torch.cuda.memory.use_allocator requires the native CUDA caching "
            "allocator, but the current allocator is '",
            current->name(),
            "'");
        inner_.store(current);
        static const CCA::LayeredAllocatorHooks hooks{
            &deleteOwned, [](void* ptr) { return layered().freeRaw(ptr); }};
        CCA::setLayeredAllocatorHooks(&hooks);
        CCA::allocator.store(this);
      }
      ++open_windows_;
    }
    window_stack.push_back(
        std::make_shared<Window>(Window{std::move(allocator)}));
    return window_stack.size();
  }

  void endWindow(uint64_t token) {
    TORCH_CHECK(
        token == window_stack.size(),
        "torch.cuda.memory.use_allocator contexts must exit on the thread that "
        "entered them, innermost first");
    std::shared_ptr<Window> w = std::move(window_stack.back());
    window_stack.pop_back();
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = claimed_.begin(); it != claimed_.end();) {
      if (it->second.window != w) {
        ++it;
        continue;
      }
      if (it->second.capture != 0) {
        TORCH_WARN(
            "torch.cuda.memory.use_allocator: the context exits while a "
            "capture it claimed is in progress; the rest of that capture "
            "allocates from the caching allocator");
        active_.fetch_sub(1, std::memory_order_relaxed);
      }
      it = claimed_.erase(it);
    }
    if (--open_windows_ == 0) {
      CCA::allocator.store(inner());
    }
  }

  c10::DataPtr allocate(size_t size) override {
    if (auto c = claimant(size, std::nullopt)) {
      void* ptr = allocFrom(*c, size);
      return {
          ptr,
          new Owned{std::move(c->allocator), ptr},
          &deleteOwned,
          c10::Device(c10::DeviceType::CUDA, c->device)};
    }
    return inner()->allocate(size);
  }
  void* raw_alloc(size_t nbytes) override {
    if (auto c = claimant(nbytes, std::nullopt)) {
      return rawFrom(*c, nbytes);
    }
    return inner()->raw_alloc(nbytes);
  }
  void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
    if (auto c = claimant(nbytes, stream)) {
      return rawFrom(*c, nbytes);
    }
    return inner()->raw_alloc_with_stream(nbytes, stream);
  }

  // Claim rule, part 1: a pool inner does not have yet, begun on a thread
  // with a window open, is claimed by that window if a capture into it
  // follows (a cudagraph-trees warm-up scope is not a capture).
  void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(cudaStream_t)> filter) override {
    const bool fresh = !window_stack.empty() &&
        inner()->getPoolUseCount(device, mempool_id) == 0;
    inner()->beginAllocateToPool(device, mempool_id, std::move(filter));
    fresh_pool = fresh ? std::optional<PoolKey>(PoolKey{device, mempool_id})
                       : std::nullopt;
  }
  void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id)
      override {
    fresh_pool.reset();
    inner()->endAllocateToPool(device, mempool_id);
  }
  // Part 2: a capture begun on a thread whose innermost window claims the
  // capture's pool already (a pool shared by the window's graphs), or whose
  // pool was fresh at the beginAllocateToPool just before (torch.cuda.graph
  // with no pool, a new graph_pool_handle). A pool a graph outside the window
  // uses stays native.
  void markCaptureBegin(
      c10::DeviceIndex device,
      const c10::cuda::CUDAGraphMemory::CaptureRegistration& registration)
      override {
    const std::optional<PoolKey> fresh =
        std::exchange(fresh_pool, std::nullopt);
    inner()->markCaptureBegin(device, registration);
    // only root captures claim: a conditional node's capture allocates into
    // its root capture's pool
    if (window_stack.empty() || registration.parent_capture_id.has_value()) {
      return;
    }
    const PoolKey key{
        device,
        inner()->getAllocationPool(
            device, c10::cuda::getCurrentCUDAStream(device).stream())};
    const std::shared_ptr<Window>& w = window_stack.back();
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = claimed_.find(key);
    if (it == claimed_.end() && fresh == key) {
      it = claimed_.emplace(key, Claim{w}).first;
    }
    if (it != claimed_.end() && it->second.window == w) {
      it->second.capture = registration.capture_id;
      active_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  size_t markCaptureEnd(c10::DeviceIndex device, CaptureId_t capture_id)
      override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto& [key, claim] : claimed_) {
        if (claim.capture == capture_id) {
          claim.capture = 0;
          active_.fetch_sub(1, std::memory_order_relaxed);
        }
      }
    }
    return inner()->markCaptureEnd(device, capture_id);
  }

  // Everything else is inner's.
#define LAYERED_FORWARD(ret, name, params, args) \
  ret name params override {                     \
    return inner()->name args;                   \
  }
  LAYERED_FORWARD(c10::DeleterFnPtr, raw_deleter, () const, ())
  LAYERED_FORWARD(
      void,
      copy_data,
      (void* dest, const void* src, std::size_t count) const,
      (dest, src, count))
  LAYERED_FORWARD(
      std::optional<c10::AllocationStreamInfo>,
      allocation_stream_info,
      (const void* ptr) const,
      (ptr))
  LAYERED_FORWARD(void, raw_delete, (void* ptr), (ptr))
  LAYERED_FORWARD(bool, initialized, (), ())
  LAYERED_FORWARD(void, emptyCache, (MempoolId_t mempool_id), (mempool_id))
  LAYERED_FORWARD(
      c10::CachingDeviceAllocator::DeviceStats,
      getDeviceStats,
      (c10::DeviceIndex device),
      (device))
  LAYERED_FORWARD(
      void,
      resetAccumulatedStats,
      (c10::DeviceIndex device),
      (device))
  LAYERED_FORWARD(void, resetPeakStats, (c10::DeviceIndex device), (device))
  LAYERED_FORWARD(void, init, (int device_count), (device_count))
  LAYERED_FORWARD(
      double,
      getMemoryFraction,
      (c10::DeviceIndex device),
      (device))
  LAYERED_FORWARD(
      void,
      setMemoryFraction,
      (double fraction, c10::DeviceIndex device),
      (fraction, device))
  LAYERED_FORWARD(
      std::vector<CCA::StreamSegmentSize>,
      getExpandableSegmentSizes,
      (c10::DeviceIndex device),
      (device))
  LAYERED_FORWARD(void, enable, (bool value), (value))
  LAYERED_FORWARD(bool, isEnabled, () const, ())
  LAYERED_FORWARD(
      void,
      cacheInfo,
      (c10::DeviceIndex device, size_t* largestBlock),
      (device, largestBlock))
  LAYERED_FORWARD(
      void*,
      getBaseAllocation,
      (void* ptr, size_t* size),
      (ptr, size))
  LAYERED_FORWARD(
      void,
      recordStream,
      (const c10::DataPtr& ptr, c10::cuda::CUDAStream stream),
      (ptr, stream))
  LAYERED_FORWARD(
      CCA::SnapshotInfo,
      snapshot,
      (MempoolId_t mempool_id, bool include_traces),
      (mempool_id, include_traces))
  LAYERED_FORWARD(void, markCaptureBegin, (c10::DeviceIndex device), (device))
  LAYERED_FORWARD(void, markCaptureEnd, (c10::DeviceIndex device), (device))
  LAYERED_FORWARD(bool, isCaptureContext, (c10::DeviceIndex device), (device))
  LAYERED_FORWARD(bool, hasActiveCaptures, (c10::DeviceIndex device), (device))
  LAYERED_FORWARD(
      void,
      releasePool,
      (c10::DeviceIndex device, MempoolId_t mempool_id),
      (device, mempool_id))
  LAYERED_FORWARD(
      MempoolId_t,
      getAllocationPool,
      (c10::DeviceIndex device, cudaStream_t stream),
      (device, stream))
  LAYERED_FORWARD(
      int,
      getPoolUseCount,
      (c10::DeviceIndex device, MempoolId_t mempool_id),
      (device, mempool_id))
  LAYERED_FORWARD(
      void,
      createOrIncrefPool,
      (c10::DeviceIndex device,
       MempoolId_t mempool_id,
       std::shared_ptr<CCA::CUDAAllocator> allocator),
      (device, mempool_id, std::move(allocator)))
  LAYERED_FORWARD(
      void,
      setUseOnOOM,
      (c10::DeviceIndex device, MempoolId_t mempool_id, bool use_on_oom),
      (device, mempool_id, use_on_oom))
  LAYERED_FORWARD(
      void,
      setNoSplit,
      (c10::DeviceIndex device, MempoolId_t mempool_id),
      (device, mempool_id))
  LAYERED_FORWARD(
      bool,
      checkPoolLiveAllocations,
      (c10::DeviceIndex device,
       MempoolId_t mempool_id,
       const std::unordered_set<void*>& expected_live_allocations),
      (device, mempool_id, expected_live_allocations))
  LAYERED_FORWARD(CCA::ShareableHandle, shareIpcHandle, (void* ptr), (ptr))
  LAYERED_FORWARD(
      std::shared_ptr<void>,
      getIpcDevPtr,
      (std::string handle),
      (std::move(handle)))
  LAYERED_FORWARD(bool, isHistoryEnabled, (), ())
  LAYERED_FORWARD(
      std::shared_ptr<c10::GatheredContext>,
      getContextForPointer,
      (const void* ptr),
      (ptr))
  LAYERED_FORWARD(
      void,
      recordHistory,
      (bool enabled,
       CCA::CreateContextFn context_recorder,
       size_t alloc_trace_max_entries,
       CCA::RecordContext when,
       bool clearHistory,
       const std::vector<std::string>& skip_actions),
      (enabled,
       std::move(context_recorder),
       alloc_trace_max_entries,
       when,
       clearHistory,
       skip_actions))
  LAYERED_FORWARD(
      void,
      recordAnnotation,
      (const std::vector<std::pair<std::string, std::string>>& md),
      (md))
  LAYERED_FORWARD(void, pushCompileContext, (std::string & md), (md))
  LAYERED_FORWARD(void, popCompileContext, (), ())
  LAYERED_FORWARD(bool, supportsUserMetadata, (), ())
  LAYERED_FORWARD(
      void,
      setUserMetadata,
      (const std::string& metadata),
      (metadata))
  LAYERED_FORWARD(std::string, getUserMetadata, (), ())
  LAYERED_FORWARD(
      void,
      annotateMemory,
      (const void* ptr, const std::string& metadata),
      (ptr, metadata))
  LAYERED_FORWARD(
      void,
      attachOutOfMemoryObserver,
      (CCA::OutOfMemoryObserver observer),
      (std::move(observer)))
  LAYERED_FORWARD(
      void,
      attachOomRejectionObserver,
      (CCA::OomRejectionObserver observer),
      (std::move(observer)))
  LAYERED_FORWARD(
      void,
      attachAllocatorTraceTracker,
      (CCA::AllocatorTraceTracker tracker),
      (std::move(tracker)))
  LAYERED_FORWARD(
      void,
      enablePeerAccess,
      (c10::DeviceIndex dev, c10::DeviceIndex dev_to_access),
      (dev, dev_to_access))
  LAYERED_FORWARD(
      cudaError_t,
      memcpyAsync,
      (void* dst,
       int dstDevice,
       const void* src,
       int srcDevice,
       size_t count,
       cudaStream_t stream,
       bool p2p_enabled),
      (dst, dstDevice, src, srcDevice, count, stream, p2p_enabled))
  LAYERED_FORWARD(
      std::shared_ptr<CCA::AllocatorState>,
      getCheckpointState,
      (c10::DeviceIndex device, MempoolId_t id),
      (device, id))
  LAYERED_FORWARD(
      CCA::CheckpointDelta,
      setCheckpointPoolState,
      (c10::DeviceIndex device, std::shared_ptr<CCA::AllocatorState> pps),
      (device, std::move(pps)))
  LAYERED_FORWARD(
      void,
      restoreExpandableSegment,
      (c10::DeviceIndex device,
       cudaStream_t stream,
       MempoolId_t mempool_id,
       bool is_small,
       size_t address,
       size_t reserve_size,
       size_t segment_size,
       CCA::Expandable_Segments_Handle_Type handle_type,
       const std::vector<std::pair<size_t, size_t>>& mapped_ranges),
      (device,
       stream,
       mempool_id,
       is_small,
       address,
       reserve_size,
       segment_size,
       handle_type,
       mapped_ranges))
  LAYERED_FORWARD(
      c10::DataPtr,
      allocateWithAddress,
      (size_t size, void* addr),
      (size, addr))
  LAYERED_FORWARD(std::string, name, (), ())
#undef LAYERED_FORWARD

 private:
  struct Claimed {
    std::shared_ptr<CCA::CUDAAllocator> allocator;
    c10::DeviceIndex device;
    cudaStream_t stream;
  };

  CCA::CUDAAllocator* inner() const {
    return inner_.load(std::memory_order_relaxed);
  }

  // The window allocator serving an allocation of `size` on `stream` (the
  // current stream if none) of the current device: inner would place it into
  // a claimed pool whose capture is in progress. One relaxed load while no
  // claimed capture is in progress anywhere.
  std::optional<Claimed> claimant(
      size_t size,
      std::optional<cudaStream_t> stream) {
    if (C10_LIKELY(active_.load(std::memory_order_relaxed) == 0) || size == 0) {
      return std::nullopt;
    }
    const c10::DeviceIndex device = c10::cuda::current_device();
    const cudaStream_t s =
        stream.value_or(c10::cuda::getCurrentCUDAStream(device).stream());
    const MempoolId_t pool = inner()->getAllocationPool(device, s);
    if (pool == MempoolId_t{0, 0}) {
      return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = claimed_.find({device, pool});
    if (it == claimed_.end() || it->second.capture == 0) {
      return std::nullopt;
    }
    return Claimed{it->second.window->allocator, device, s};
  }

  static void* allocFrom(const Claimed& c, size_t size) {
    void* ptr = nullptr;
    {
      // as for the caching allocator's own segments during a capture
      // (cudaMallocMaybeCapturing): the allocator may cudaMalloc
      c10::cuda::CUDAStreamCaptureModeGuard mode{cudaStreamCaptureModeRelaxed};
      ptr = c.allocator->raw_alloc_with_stream(size, c.stream);
    }
    TORCH_CHECK_WITH(
        OutOfMemoryError,
        ptr != nullptr,
        "CUDA out of memory: the torch.cuda.memory.use_allocator allocator "
        "returned null for ",
        size,
        " bytes");
    if (C10_UNLIKELY(reinterpret_cast<uintptr_t>(ptr) % 256 != 0)) {
      c.allocator->raw_delete(ptr);
      TORCH_CHECK(
          false,
          "The torch.cuda.memory.use_allocator allocator returned ",
          ptr,
          ", which is not aligned to 256 bytes");
    }
    return ptr;
  }

  void* rawFrom(Claimed& c, size_t size) {
    void* ptr = allocFrom(c, size);
    std::lock_guard<std::mutex> lock(mutex_);
    if (raw_.emplace(ptr, std::move(c.allocator)).second) {
      raw_count_.fetch_add(1, std::memory_order_relaxed);
    }
    return ptr;
  }

  // Frees a raw allocation of a window's allocator; false if ptr is not one.
  bool freeRaw(void* ptr) {
    if (C10_LIKELY(raw_count_.load(std::memory_order_relaxed) == 0)) {
      return false;
    }
    std::shared_ptr<CCA::CUDAAllocator> owner;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = raw_.find(ptr);
      if (it == raw_.end()) {
        return false;
      }
      owner = std::move(it->second);
      raw_.erase(it);
      raw_count_.fetch_sub(1, std::memory_order_relaxed);
    }
    owner->raw_delete(ptr);
    return true;
  }

  std::atomic<CCA::CUDAAllocator*> inner_{nullptr};
  std::mutex mutex_;
  int open_windows_ = 0;
  std::map<PoolKey, Claim> claimed_;
  // claims whose capture is in progress
  std::atomic<int> active_{0};
  // live raw allocations of the windows' allocators
  std::unordered_map<void*, std::shared_ptr<CCA::CUDAAllocator>> raw_;
  std::atomic<int64_t> raw_count_{0};
};

// Never destroyed: storages made while it was current keep referring to it.
LayeredAllocator& layered() {
  static auto* allocator = new LayeredAllocator();
  return *allocator;
}

} // namespace

uint64_t beginUseAllocator(
    std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> allocator) {
  TORCH_CHECK(allocator, "use_allocator expects an allocator");
  return layered().beginWindow(std::move(allocator));
}

void endUseAllocator(uint64_t token) {
  layered().endWindow(token);
}

} // namespace torch::cuda::CUDAPluggableAllocator
