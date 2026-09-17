// The instantiated CUDA graph of a replay: the captured kernel nodes in
// order, their argument images, the captured memset nodes in order, and one
// entry point that pushes the nodes whose bytes changed and launches. Which
// bytes change, and to what, is decided on the Python side from the tape;
// nothing here evaluates an expression.
//
// Interim. The replay is meant to be lowered into a runtime that already
// replays parameterized graphs; this class and torch/cuda/_host_trace.py exist
// so the recorder can be exercised end to end. Treat the Tape (Tape.h) as the
// contract, not this API.
#pragma once
#include <ATen/cuda/CUDAGraph.h>
#include <c10/macros/Export.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace at::cuda::host_trace {

// The tape does not describe the capture it was built against: a recorder bug.
struct TORCH_CUDA_CPP_API TapeMismatch : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct NodeUpdate {
  size_t node = 0;
  std::vector<uint8_t> image; // empty: the arguments did not change
  std::array<unsigned, 3> grid{};
  std::array<unsigned, 3> block{};
  unsigned smem = 0;
};

struct MemsetUpdate {
  size_t node;
  uint64_t dst;
  uint64_t bytes;
};

class TORCH_CUDA_CPP_API Exec {
 public:
  // Reads the kernel and memset nodes of a captured (not yet instantiated)
  // graph; any other node type is a TapeMismatch.
  Exec(at::cuda::CUDAGraph& graph, at::DeviceIndex device);
  ~Exec();
  Exec(const Exec&) = delete;
  Exec& operator=(const Exec&) = delete;
  Exec(Exec&&) = delete;
  Exec& operator=(Exec&&) = delete;

  size_t num_nodes() const {
    return nodes_.size();
  }
  // the capture's dependencies by node position (cudaGraphGetNodes order):
  // the build pairs the tape's host order with a topological order of the
  // nodes, not with the array order
  std::vector<std::vector<int64_t>> dependencies() const {
    return deps_;
  }
  // the captured nodes in capture order as (kind, index within the kind):
  // kind 0 kernel, 1 memset
  std::vector<std::pair<int, size_t>> node_kinds() const {
    return order_;
  }
  std::string kernel_name(size_t j) const;
  std::vector<uint8_t> image(size_t j) const;
  std::array<unsigned, 3> grid(size_t j) const;
  std::array<unsigned, 3> block(size_t j) const;
  unsigned smem(size_t j) const;

  // the memset nodes, in capture order: destination, byte count, value
  size_t num_memset_nodes() const {
    return memsets_.size();
  }
  uint64_t memset_dst(size_t j) const;
  uint64_t memset_bytes(size_t j) const;
  unsigned memset_value(size_t j) const;

  // Instantiate, then either replay once on the current stream and wait for
  // that stream only, or upload the exec without launching it (a node
  // patched before the first upload stays slow either way; a build that must
  // execute nothing of the function uploads).
  void instantiate(bool replay = true);
  // One call: push the changed nodes and launch on the current stream.
  void run(
      const std::vector<NodeUpdate>& updates,
      const std::vector<MemsetUpdate>& memset_updates = {});

  int64_t dirty_nodes() const {
    return dirty_nodes_;
  }
  int64_t dirty_memset_nodes() const {
    return dirty_memset_nodes_;
  }

 private:
  struct NodeState {
    cudaGraphNode_t node = nullptr;
    std::vector<std::pair<size_t, size_t>> infos;
    std::vector<char> image;
    std::vector<void*> argptrs;
    cudaKernelNodeParams params{};
  };
  struct MemsetState {
    cudaGraphNode_t node = nullptr;
    cudaMemsetParams params{};
  };
  at::DeviceIndex device_;
  at::cuda::CUDAGraph* graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  bool instantiated_ = false;
  std::vector<NodeState> nodes_;
  std::vector<std::vector<int64_t>> deps_;
  std::vector<std::pair<int, size_t>> order_;
  std::vector<MemsetState> memsets_;
  int64_t dirty_nodes_ = 0;
  int64_t dirty_memset_nodes_ = 0;
};

} // namespace at::cuda::host_trace
