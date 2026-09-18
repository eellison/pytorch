#include <ATen/cuda/host_trace/Exec.h>
#include <ATen/cuda/host_trace/Recorder.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace at::cuda::host_trace {

Exec::Exec(at::cuda::CUDAGraph& graph, at::DeviceIndex device)
    : device_(device), graph_(&graph) {
  cudaGraph_t g = graph.raw_cuda_graph();
  size_t n = 0;
  C10_CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &n));
  std::vector<cudaGraphNode_t> raw(n);
  C10_CUDA_CHECK(cudaGraphGetNodes(g, raw.data(), &n));
  if (test_reverse_nodes()) {
    // the test hook: the array order must not be the pairing
    std::reverse(raw.begin(), raw.end());
  }
  {
    size_t ne = 0;
#ifdef USE_ROCM
    C10_CUDA_CHECK(cudaGraphGetEdges(g, nullptr, nullptr, &ne));
#else
    C10_CUDA_CHECK(cudaGraphGetEdges(g, nullptr, nullptr, nullptr, &ne));
#endif
    std::vector<cudaGraphNode_t> from(ne), to(ne);
    if (ne != 0) {
#ifdef USE_ROCM
      C10_CUDA_CHECK(cudaGraphGetEdges(g, from.data(), to.data(), &ne));
#else
      // the edge data (a programmatic dependent launch edge of a library
      // kernel) is read and dropped: the query refuses to drop it itself,
      // and an edge of any type orders its two nodes
      std::vector<cudaGraphEdgeData> data(ne);
      C10_CUDA_CHECK(
          cudaGraphGetEdges(g, from.data(), to.data(), data.data(), &ne));
#endif
    }
    const auto position = [&](cudaGraphNode_t node) {
      for (size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == node) {
          return static_cast<int64_t>(i);
        }
      }
      TORCH_INTERNAL_ASSERT(
          false, "host_trace: a capture edge names a node outside the capture");
    };
    deps_.assign(raw.size(), {});
    for (size_t i = 0; i < ne; ++i) {
      deps_[position(to[i])].push_back(position(from[i]));
    }
  }
  for (cudaGraphNode_t node : raw) {
    cudaGraphNodeType type{};
    C10_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
    if (type == cudaGraphNodeTypeMemset) {
      MemsetState ms;
      ms.node = node;
      C10_CUDA_CHECK(cudaGraphMemsetNodeGetParams(node, &ms.params));
      if (ms.params.height != 1) {
        // @allow-raw-throw: registered with pybind11 as _HostTraceTapeMismatch and caught by that name in torch/cuda/_host_trace.py
        throw TapeMismatch(
            "the capture contains a two-dimensional memset node, which this "
            "tape does not describe");
      }
      order_.emplace_back(1, memsets_.size());
      memsets_.push_back(ms);
      continue;
    }
    if (type == cudaGraphNodeTypeMemcpy) {
      MemcpyState mc;
      mc.node = node;
      C10_CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node, &mc.params));
      const cudaMemcpy3DParms& p = mc.params;
      const bool linear = p.srcArray == nullptr && p.dstArray == nullptr &&
          p.extent.height == 1 && p.extent.depth == 1 &&
          (p.kind == cudaMemcpyHostToDevice ||
           p.kind == cudaMemcpyDeviceToDevice || p.kind == cudaMemcpyDefault);
      if (!linear) {
        // @allow-raw-throw: registered with pybind11 as _HostTraceTapeMismatch and caught by that name in torch/cuda/_host_trace.py
        throw TapeMismatch(
            "the capture contains a memcpy node that is not a one-dimensional "
            "host-to-device or device-to-device copy, which this tape does "
            "not describe");
      }
      order_.emplace_back(2, memcpys_.size());
      memcpys_.push_back(mc);
      continue;
    }
    if (type != cudaGraphNodeTypeKernel) {
      // @allow-raw-throw: registered with pybind11 as _HostTraceTapeMismatch and caught by that name in torch/cuda/_host_trace.py
      throw TapeMismatch(
          "the capture contains a node that is neither a kernel, a memset nor "
          "a memcpy, which this tape does not describe");
    }
    NodeState ns;
    ns.node = node;
    C10_CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &ns.params));
    if (ns.params.kernelParams == nullptr || ns.params.extra != nullptr) {
      // @allow-raw-throw: registered with pybind11 as _HostTraceTapeMismatch and caught by that name in torch/cuda/_host_trace.py
      throw TapeMismatch("captured launch used the `extra` parameter buffer");
    }
    const FuncInfo& info = func_info(ns.params.func);
    ns.infos = info.params;
    ns.image.assign((info.image_size + 7) / 8 * 8, 0);
    for (size_t i = 0; i < ns.infos.size(); ++i) {
      std::memcpy(
          ns.image.data() + ns.infos[i].first,
          ns.params.kernelParams[i],
          ns.infos[i].second);
    }
    ns.argptrs.resize(ns.infos.size());
    order_.emplace_back(0, nodes_.size());
    nodes_.push_back(std::move(ns));
  }
  for (auto& ns : nodes_) { // pointers into the final (non-moving) images
    for (size_t i = 0; i < ns.infos.size(); ++i) {
      ns.argptrs[i] = ns.image.data() + ns.infos[i].first;
    }
    ns.params.kernelParams = ns.argptrs.data();
  }
}

Exec::~Exec() = default;

std::string Exec::kernel_name(size_t j) const {
  return func_info(nodes_.at(j).params.func).name;
}

std::vector<uint8_t> Exec::image(size_t j) const {
  const NodeState& ns = nodes_.at(j);
  const size_t n = func_info(ns.params.func).image_size;
  return std::vector<uint8_t>(
      ns.image.begin(), ns.image.begin() + static_cast<std::ptrdiff_t>(n));
}

std::array<unsigned, 3> Exec::grid(size_t j) const {
  const auto& p = nodes_.at(j).params;
  return {p.gridDim.x, p.gridDim.y, p.gridDim.z};
}

std::array<unsigned, 3> Exec::block(size_t j) const {
  const auto& p = nodes_.at(j).params;
  return {p.blockDim.x, p.blockDim.y, p.blockDim.z};
}

unsigned Exec::smem(size_t j) const {
  return nodes_.at(j).params.sharedMemBytes;
}

uint64_t Exec::memset_dst(size_t j) const {
  return reinterpret_cast<uintptr_t>(memsets_.at(j).params.dst);
}

uint64_t Exec::memset_bytes(size_t j) const {
  const auto& p = memsets_.at(j).params;
  return static_cast<uint64_t>(p.width) * p.elementSize * p.height;
}

unsigned Exec::memset_value(size_t j) const {
  return memsets_.at(j).params.value;
}

uint64_t Exec::memcpy_src(size_t j) const {
  return reinterpret_cast<uintptr_t>(memcpys_.at(j).params.srcPtr.ptr);
}

uint64_t Exec::memcpy_dst(size_t j) const {
  return reinterpret_cast<uintptr_t>(memcpys_.at(j).params.dstPtr.ptr);
}

uint64_t Exec::memcpy_bytes(size_t j) const {
  return static_cast<uint64_t>(memcpys_.at(j).params.extent.width);
}

std::string Exec::memcpy_kind(size_t j) const {
  switch (memcpys_.at(j).params.kind) {
    case cudaMemcpyHostToDevice:
      return "h2d";
    case cudaMemcpyDeviceToDevice:
      return "d2d";
    default:
      return "default";
  }
}

void Exec::instantiate(bool replay) {
  c10::cuda::CUDAGuard guard(device_);
  graph_->instantiate();
  exec_ = graph_->raw_cuda_graph_exec();
  if (replay) {
    graph_->replay();
    // the caller's stream only: a device-wide synchronize from here would
    // invalidate a trace capture on another thread
    c10::cuda::getCurrentCUDAStream(device_).synchronize();
  } else {
    C10_CUDA_CHECK(
        cudaGraphUpload(exec_, c10::cuda::getCurrentCUDAStream(device_)));
  }
  instantiated_ = true;
}

void Exec::run(
    const std::vector<NodeUpdate>& updates,
    const std::vector<MemsetUpdate>& memset_updates,
    const std::vector<MemcpyUpdate>& memcpy_updates) {
  TORCH_CHECK(instantiated_, "host_trace: this exec was never instantiated");
  c10::cuda::CUDAGuard guard(device_);
  for (const NodeUpdate& u : updates) {
    NodeState& ns = nodes_.at(u.node);
    if (!u.image.empty()) {
      TORCH_CHECK(
          u.image.size() <= ns.image.size(),
          "host_trace: argument image larger than the captured one");
      std::memcpy(ns.image.data(), u.image.data(), u.image.size());
    }
    ns.params.gridDim = dim3(u.grid[0], u.grid[1], u.grid[2]);
    ns.params.blockDim = dim3(u.block[0], u.block[1], u.block[2]);
    ns.params.sharedMemBytes = u.smem;
    C10_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(exec_, ns.node, &ns.params));
  }
  for (const MemsetUpdate& u : memset_updates) {
    MemsetState& ms = memsets_.at(u.node);
    cudaMemsetParams& p = ms.params;
    p.dst = reinterpret_cast<void*>(static_cast<uintptr_t>(u.dst));
    // one-dimensional: the captured element size when the count still
    // divides by it, else bytes (value 0 sets the same bytes either way)
    if (u.bytes % p.elementSize != 0) {
      p.elementSize = 1;
    }
    p.width = static_cast<size_t>(u.bytes / p.elementSize);
    p.height = 1;
    p.pitch = p.width * p.elementSize;
    C10_CUDA_CHECK(cudaGraphExecMemsetNodeSetParams(exec_, ms.node, &p));
  }
  for (const MemcpyUpdate& u : memcpy_updates) {
    MemcpyState& mc = memcpys_.at(u.node);
    cudaMemcpy3DParms& p = mc.params;
    // the captured copy is linear: pitch and logical width are the byte
    // count; its kind (host-to-device or device-to-device) stays the capture's
    p.srcPtr.ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(u.src));
    p.dstPtr.ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(u.dst));
    p.srcPtr.pitch = p.dstPtr.pitch = static_cast<size_t>(u.bytes);
    p.srcPtr.xsize = p.dstPtr.xsize = static_cast<size_t>(u.bytes);
    p.srcPtr.ysize = p.dstPtr.ysize = 1;
    p.extent.width = static_cast<size_t>(u.bytes);
    p.extent.height = p.extent.depth = 1;
    C10_CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams(exec_, mc.node, &p));
  }
  dirty_nodes_ += static_cast<int64_t>(updates.size());
  dirty_memset_nodes_ += static_cast<int64_t>(memset_updates.size());
  dirty_memcpy_nodes_ += static_cast<int64_t>(memcpy_updates.size());
  graph_->replay();
}

} // namespace at::cuda::host_trace
