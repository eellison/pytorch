#include <ATen/cuda/host_trace/Exec.h>
#include <ATen/cuda/host_trace/Recorder.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

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
  for (cudaGraphNode_t node : raw) {
    cudaGraphNodeType type{};
    C10_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
    if (type != cudaGraphNodeTypeKernel) {
      // @allow-raw-throw: registered with pybind11 as _HostTraceTapeMismatch and caught by that name in torch/cuda/_host_trace.py
      throw TapeMismatch(
          "the capture contains a non-kernel node, which this tape does not "
          "describe");
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

void Exec::instantiate() {
  c10::cuda::CUDAGuard guard(device_);
  graph_->instantiate();
  graph_->replay();
  // the caller's stream only: a device-wide synchronize from here would
  // invalidate a trace capture on another thread
  c10::cuda::getCurrentCUDAStream(device_).synchronize();
  exec_ = graph_->raw_cuda_graph_exec();
  instantiated_ = true;
}

void Exec::run(const std::vector<NodeUpdate>& updates) {
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
  dirty_nodes_ += static_cast<int64_t>(updates.size());
  graph_->replay();
  ++calls_;
}

} // namespace at::cuda::host_trace
