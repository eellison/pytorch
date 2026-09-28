#include <torch/csrc/cuda/host_trace/Variant.h>

#if !defined(USE_ROCM) && defined(__linux__)
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <pthread.h>
#include <cstring>
#include <limits>
#include <optional>

namespace torch::cuda::host_trace {

namespace {

__attribute__((noinline)) void smear(int pattern) {
  unsigned char buf[1 << 16];
  std::memset(buf, pattern, sizeof(buf));
  asm volatile("" : : "r"(buf) : "memory");
}

// The op's CUDA kernel through the dispatcher, with the stack below it
// smeared first, so that uninitialized padding a library leaves in its
// kernels' parameters holds the pattern
py::object smeared_call(
    const std::string& name,
    const std::string& overload,
    int64_t pattern,
    py::args args,
    const py::kwargs& kwargs) {
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      name.c_str(), overload.c_str());
  auto stack = torch::jit::createStackForSchema(
      op.schema(),
      torch::jit::tuple_slice(std::move(args)),
      kwargs,
      std::nullopt);
  smear(static_cast<int>(pattern));
  op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CUDA), &stack);
  return torch::jit::createPyObjectForStack(std::move(stack));
}

std::pair<int64_t, int64_t> stack_bounds() {
  thread_local std::optional<std::pair<int64_t, int64_t>> bounds;
  if (bounds) {
    return *bounds;
  }
  pthread_attr_t attr;
  TORCH_CHECK(pthread_getattr_np(pthread_self(), &attr) == 0);
  void* lo = nullptr;
  size_t size = 0;
  const int rc = pthread_attr_getstack(&attr, &lo, &size);
  pthread_attr_destroy(&attr);
  TORCH_CHECK(rc == 0);
  const auto start = reinterpret_cast<int64_t>(lo);
  bounds.emplace(start, start + static_cast<int64_t>(size));
  return *bounds;
}

namespace alloc = c10::cuda::CUDACachingAllocator;
using Pool = std::tuple<int64_t, int64_t>;

bool same_pool(const c10::MempoolId_t& id, const Pool& pool) {
  return static_cast<int64_t>(id.first) == std::get<0>(pool) &&
      static_cast<int64_t>(id.second) == std::get<1>(pool);
}

// The pool's blocks on the device as (stream, address, size, requested
// size, active, in the small pool); and with `allocations`, the (address,
// requested bytes) of each allocation into the pool the allocator's history
// holds, oldest first
py::tuple pool_state(int64_t device, const Pool& pool, bool allocations) {
  const c10::MempoolId_t id{std::get<0>(pool), std::get<1>(pool)};
  const auto info = alloc::snapshot(id, allocations);
  py::list blocks;
  for (const auto& seg : info.segments) {
    if (seg.device != device || !same_pool(seg.owner_private_pool_id, pool)) {
      continue;
    }
    int64_t address = static_cast<int64_t>(seg.address);
    for (const auto& b : seg.blocks) {
      blocks.append(py::make_tuple(
          reinterpret_cast<int64_t>(seg.stream),
          address,
          b.size,
          b.requested_size,
          b.allocated || b.active,
          !seg.is_large));
      address += static_cast<int64_t>(b.size);
    }
  }
  if (!allocations) {
    return py::make_tuple(blocks, py::none());
  }
  py::list allocated;
  for (const auto& trace : info.device_traces) {
    for (const auto& e : trace) {
      if (e.action_ == c10::CachingDeviceAllocator::TraceEntry::ALLOC &&
          same_pool(e.mempool_, pool)) {
        allocated.append(py::make_tuple(e.addr_, e.size_));
      }
    }
  }
  return py::make_tuple(blocks, allocated);
}

// Every segment of the device as (start, end)
std::vector<std::pair<int64_t, int64_t>> segments(int64_t device) {
  std::vector<std::pair<int64_t, int64_t>> out;
  for (const auto& seg : alloc::snapshot({0, 0}, false).segments) {
    if (seg.device == device) {
      const auto start = static_cast<int64_t>(seg.address);
      out.emplace_back(start, start + static_cast<int64_t>(seg.total_size));
    }
  }
  return out;
}

// Starts, cleared, or stops the allocator's history, of allocations alone:
// the harvest's own reading while the process records none of its own
void record_allocations(bool enabled) {
  std::vector<std::string> skip;
  if (enabled) {
    skip = {
        "free_requested",
        "free_completed",
        "segment_alloc",
        "segment_free",
        "segment_map",
        "segment_unmap",
        "snapshot",
        "oom",
        "annotate"};
  }
  alloc::recordHistory(
      enabled,
      nullptr,
      std::numeric_limits<size_t>::max(),
      c10::CachingDeviceAllocator::RecordContext::NEVER,
      true,
      skip);
}

} // namespace

void initHarvestBindings(py::module& m) {
  m.def("_cuda_hostTraceSmearedCall", &smeared_call);
  m.def("_cuda_hostTraceStackBounds", &stack_bounds);
  m.def("_cuda_hostTracePool", &pool_state);
  m.def("_cuda_hostTraceSegments", &segments);
  m.def("_cuda_hostTraceRecordAllocations", &record_allocations);
  m.def("_cuda_hostTraceAllocationCount", [](int64_t device) {
    const auto stats =
        alloc::getDeviceStats(static_cast<c10::DeviceIndex>(device));
    return stats
        .allocation[static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE)]
        .allocated;
  });
  m.def("_cuda_hostTraceSetHarvesting", [](bool enabled) {
    const bool previous = c10::cuda::isHostTraceHarvesting();
    c10::cuda::setHostTraceHarvesting(enabled);
    return previous;
  });
  m.def("_cuda_hostTraceClearCublasWorkspaces", [](int64_t stream) {
    at::cuda::clearCublasWorkspacesForStream(
        reinterpret_cast<cudaStream_t>(stream));
  });
}

} // namespace torch::cuda::host_trace
#endif
