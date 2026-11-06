#include <ATen/cuda/CUDAGraph.h>
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

namespace torch::cuda::CUDAPluggableAllocator {

int device_count = 0;

void custom_raw_deleter(void* ptr);

// Runtime swap state (guarded by registry_mutex): the native allocator, and
// every pluggable allocator that was ever current (kept alive; owner lookup).
std::mutex registry_mutex;
c10::cuda::CUDACachingAllocator::CUDAAllocator* native_allocator = nullptr;
std::vector<std::shared_ptr<CUDAPluggableAllocator>> installed_allocators;
extern std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
    current_custom_allocator;

// the installed pluggable allocator that allocated ptr (other than skip)
CUDAPluggableAllocator* find_owner(
    void* ptr,
    CUDAPluggableAllocator* skip = nullptr) {
  const std::lock_guard<std::mutex> lock(registry_mutex);
  for (auto& a : installed_allocators) {
    if (a.get() != skip && a->owns(ptr)) {
      return a.get();
    }
  }
  return nullptr;
}

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
  {
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
  cudaStream_t stream{};
  c10::DeviceIndex device_idx = -1;
  size_t size = 0;
  {
    std::unique_lock<std::mutex> lock(allocator_mutex_);
    if (!allocation_metadata_.count(ptr)) {
      lock.unlock();
      // runtime swap: another installed pluggable allocator's pointer, or
      // (while swapped in) a native allocator pointer
      if (auto* owner = find_owner(ptr, this)) {
        return owner->raw_delete(ptr);
      }
      TORCH_CHECK(
          forward_native_, "Trying to free a pointer not allocated here");
      return forward_native_->raw_delete(ptr);
    }
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
  if (forward_native_ && mempool_id.first == 0 && mempool_id.second == 0) {
    forward_native_->emptyCache(mempool_id);
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
  // runtime swap: route by the DataPtr's owner
  if (forward_native_ && ptr.get_deleter() == forward_native_->raw_deleter()) {
    return forward_native_->recordStream(ptr, stream);
  }
  CUDAPluggableAllocator* owner = this;
  if (ptr.get_deleter() == &custom_raw_deleter && !owns(ptr.get())) {
    if (auto* o = find_owner(ptr.get(), this)) {
      owner = o;
    }
  }
  if (owner->record_stream_fn_) {
    owner->record_stream_fn_(ptr.get(), stream);
  }
}

void CUDAPluggableAllocator::markCaptureBegin(c10::DeviceIndex device) {
  if (forward_native_) {
    forward_native_->markCaptureBegin(device);
  }
}

void CUDAPluggableAllocator::markCaptureEnd(c10::DeviceIndex device) {
  if (forward_native_) {
    forward_native_->markCaptureEnd(device);
  }
}

bool CUDAPluggableAllocator::owns(void* ptr) {
  const std::lock_guard<std::mutex> lock(allocator_mutex_);
  return allocation_metadata_.count(ptr) != 0;
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
  C10_CUDA_CHECK(
      cudaMemcpy(dest, src, count, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
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

// registry_mutex held: remember the native allocator, keep current_custom_allocator alive / findable
static void remember_current_locked() {
  if (!native_allocator && !current_custom_allocator) {
    native_allocator = c10::cuda::CUDACachingAllocator::allocator.load();
  }
  auto p =
      std::dynamic_pointer_cast<CUDAPluggableAllocator>(current_custom_allocator);
  if (p &&
      std::find(installed_allocators.begin(), installed_allocators.end(), p) ==
          installed_allocators.end()) {
    installed_allocators.push_back(p);
  }
}

void changeCurrentAllocator(
    const std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>&
        allocator) {
  TORCH_CHECK(
      !c10::cuda::CUDACachingAllocator::allocator.load()->initialized(),
      "Can't swap an already initialized allocator");
  const std::lock_guard<std::mutex> lock(registry_mutex);
  remember_current_locked();
  c10::cuda::CUDACachingAllocator::allocator.store(allocator.get());
  current_custom_allocator = allocator;
  remember_current_locked();
}

// called by the native allocator for pointers / DataPtrs it does not own
static bool foreign_raw_delete(void* ptr) {
  auto* owner = find_owner(ptr);
  if (!owner) {
    return false;
  }
  owner->raw_delete(ptr);
  return true;
}

static void foreign_record_stream(
    const c10::DataPtr& ptr,
    c10::cuda::CUDAStream stream) {
  if (ptr.get_deleter() != &custom_raw_deleter) {
    return;
  }
  if (auto* owner = find_owner(ptr.get())) {
    owner->recordStream(ptr, stream);
  }
}

void swapCurrentAllocator(
    const std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>&
        allocator) {
  TORCH_CHECK(
      !at::cuda::is_graph_capture_active(),
      "Can't swap the CUDA allocator while a CUDA graph capture is in progress");
  const std::lock_guard<std::mutex> lock(registry_mutex);
  remember_current_locked();
  TORCH_CHECK(native_allocator, "swapCurrentAllocator: no native allocator");
  c10::cuda::CUDACachingAllocator::CUDAAllocator* target = nullptr;
  if (allocator) {
    auto p = std::dynamic_pointer_cast<CUDAPluggableAllocator>(allocator);
    TORCH_CHECK(
        p,
        "swapCurrentAllocator expects a CUDAPluggableAllocator (or None for the native allocator)");
    if (!p->initialized()) {
      p->init(c10::cuda::device_count());
    }
    if (native_allocator->initialized()) {
      p->forward_native_ = native_allocator;
    }
    target = p.get();
  } else {
    if (!native_allocator->initialized()) {
      native_allocator->init(c10::cuda::device_count());
    }
    target = native_allocator;
  }
  static const bool handlers_set = [] {
    c10::cuda::CUDACachingAllocator::setForeignPtrHandlers(
        &foreign_raw_delete, &foreign_record_stream);
    return true;
  }();
  (void)handlers_set;
  c10::cuda::CUDACachingAllocator::allocator.store(target);
  // c10::GetAllocator(kCUDA) / getDeviceAllocator(kCUDA) users (e.g.
  // torch.UntypedStorage(n, device="cuda")) follow the swap too
  c10::SetAllocator(
      c10::DeviceType::CUDA, target, std::numeric_limits<uint8_t>::max());
  current_custom_allocator = allocator;
  remember_current_locked();
}

void custom_raw_deleter(void* ptr) {
  CUDAPluggableAllocator* owner = find_owner(ptr);
  if (owner) {
    return owner->raw_delete(ptr);
  }
  // an allocator used without being installed (stock behaviour)
  std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> cur;
  {
    const std::lock_guard<std::mutex> lock(registry_mutex);
    cur = current_custom_allocator;
  }
  TORCH_CHECK(cur, "CUDAPluggableAllocator: freeing an unknown pointer");
  cur->raw_delete(ptr);
}

} // namespace torch::cuda::CUDAPluggableAllocator
