#include <ATen/cuda/MemPool.h>
#include <c10/cuda/CUDAFunctions.h>

namespace at::cuda {

// uid_ is incremented when a user creates a MemPool,
// for example: using graph_pool_handle() or c10::cuda::MemPool().
//
// uuid_ is incremented when CUDAGraph creates a MemPool
// as a result of a user not providing a pool.
//
// MempoolId_t of {0, 0} is used to denote when no MemPool has been
// passed to a function, either by user or CUDAGraphs. For example,
// default value of MempoolId_t for capture_begin function is {0, 0}.
// That's why uid_ and uuid_ start at 1.
std::atomic<CaptureId_t> MemPool::uid_{1};
std::atomic<CaptureId_t> MemPool::uuid_{1};

MemPool::MemPool(
    std::shared_ptr<CUDACachingAllocator::CUDAAllocator> allocator,
    bool is_user_created,
    bool use_on_oom,
    bool no_split,
    bool passthrough)
    : is_user_created_(is_user_created) {
  TORCH_CHECK(
      !passthrough || allocator,
      "MemPool(passthrough=True) requires a custom allocator");
  TORCH_CHECK(
      !passthrough || (!use_on_oom && !no_split),
      "MemPool(passthrough=True) does not cache blocks, so use_on_oom and "
      "no_split do not apply");
  // Checked before the pool is created, so a failure leaves nothing behind.
  TORCH_CHECK(
      !passthrough ||
          CUDACachingAllocator::get()->name() == std::string("native"),
      "MemPool(passthrough=True) requires the native CUDA caching allocator "
      "backend, but the current allocator is ",
      CUDACachingAllocator::get()->name());
  TORCH_CHECK(
      !passthrough || CUDACachingAllocator::get()->isEnabled(),
      "MemPool(passthrough=True) is not supported while CUDA memory caching "
      "is disabled (PYTORCH_NO_CUDA_MEMORY_CACHING=1): allocations would "
      "bypass the pool");
  if (is_user_created_) {
    id_ = {0, uid_++};
  } else {
    id_ = {uuid_++, 0};
  }
  device_ = c10::cuda::current_device();
  CUDACachingAllocator::createOrIncrefPool(
      device_, id_, std::move(allocator));
  if (use_on_oom) {
    CUDACachingAllocator::setUseOnOOM(device_, id_, true);
  }
  if (no_split) {
    CUDACachingAllocator::setNoSplit(device_, id_);
  }
  if (passthrough) {
    CUDACachingAllocator::setPassthrough(device_, id_);
  }
}

MemPool::~MemPool() {
  // TORCH_INTERNAL_ASSERT(use_count() == 1);
  // We used to assert that TORCH_INTERNAL_ASSERT(use_count() == 1);
  // However, this assertion is not true if a memory pool is shared
  // with a cuda graph. That CUDAGraph will increase the use count
  // until it is reset.
  CUDACachingAllocator::setUseOnOOM(device_, id_, false);
  CUDACachingAllocator::releasePool(device_, id_);
  // Releasing the pool's cached memory synchronizes on its events and frees
  // segments, which a CUDA graph capture in progress forbids, whichever thread
  // or device captures (e.g. a pool garbage-collected inside a captured region,
  // or on another thread during a capture). The pool is freeable already, so
  // the next emptyCache (or an allocation that runs out of memory) releases it
  // instead.
  for (c10::DeviceIndex d = 0; d < c10::cuda::device_count(); ++d) {
    if (CUDACachingAllocator::hasActiveCaptures(d)) {
      return;
    }
  }
  c10::cuda::CUDACachingAllocator::emptyCache(id_);
}

MempoolId_t MemPool::id() {
  return id_;
}

int MemPool::use_count() {
  return CUDACachingAllocator::getPoolUseCount(device_, id_);
}

c10::DeviceIndex MemPool::device() {
  return device_;
}

MempoolId_t MemPool::graph_pool_handle(bool is_user_created) {
  if (is_user_created) {
    return {0, uid_++};
  }
  return {uuid_++, 0};
}

} // namespace at::cuda
