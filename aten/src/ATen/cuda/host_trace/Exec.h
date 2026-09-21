// The instantiated CUDA graph of a replay: the captured kernel nodes in
// order, their argument images, the captured memset and memcpy nodes in
// order, and one entry point that pushes the nodes whose bytes changed and
// launches. Which bytes change, and to what, is decided on the Python side
// from the tape; nothing here evaluates an expression.
//
// Closed regions. A node the CUDA runtime cannot read back (a library
// kernel cuBLAS launched through the driver: cudaGraphKernelNodeGetParams
// fails with cudaErrorInvalidDeviceFunction, measured) is read and updated
// through the driver entry points instead, image included when the launch
// used the `extra` parameter buffer. Such a node may change its kernel
// function on update (a different cuBLAS variant of the same node chain
// transplanted in place; a variant with another chain is another exec, built
// from the same tape). Kernel node attributes (the cluster dimension, its scheduling
// preference, cooperative, priority, the memory-sync domain) travel with the
// variant: the exec-level setter carries none of them, and a transplant
// across a cluster change with it alone fails asynchronously or hangs
// (measured), so a swap that changes an attribute goes through the graph
// node (params and attributes) and cuGraphExecUpdate. Every exec-level
// update is mirrored to the graph node, so the graph and the exec never
// disagree when an update goes that way. harvest_nodes reads the nodes of a
// graph the Python side captured one closed library call into, which is how
// it learns a variant's nodes, attributes and pointer slots.
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
#include <utility>
#include <vector>

namespace at::cuda::host_trace {

// The tape does not describe the capture it was built against: a recorder bug.
struct TORCH_CUDA_CPP_API TapeMismatch : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// The kernel node attributes tracked per node, as one integer vector:
// cluster x, y, z; cluster scheduling policy preference; cooperative;
// priority; memory-sync domain; domain map default, remote. An attribute a
// variant sets that this list does not track cannot be transplanted, so the
// harvest declines it.
constexpr size_t kNodeAttrs = 9;

struct NodeUpdate {
  size_t node = 0;
  std::vector<uint8_t> image; // empty: the arguments did not change
  std::array<unsigned, 3> grid{};
  std::array<unsigned, 3> block{};
  unsigned smem = 0;
  // a driver function handle (CUfunction): the node runs this kernel from now
  // on, with `image` laid out for it; 0 keeps the node's kernel. Only a node
  // read through the driver (a closed region's) accepts a change.
  uint64_t func = 0;
  // the variant's node attributes (kNodeAttrs values), empty to keep the
  // node's; a change routes the update through the graph node
  std::vector<int64_t> attrs;
};

struct MemsetUpdate {
  size_t node;
  uint64_t dst;
  uint64_t bytes;
  unsigned value = 0;
};

// a one-dimensional copy (host-to-device or device-to-device): new source,
// destination, byte count; the node keeps its kind
struct MemcpyUpdate {
  size_t node;
  uint64_t src;
  uint64_t dst;
  uint64_t bytes;
};

// one node of a harvested closed call: a kernel (kind 0) or a
// one-dimensional memset (kind 1: dst, value, elem, width)
struct HarvestedNode {
  int kind = 0;
  uint64_t func = 0; // CUfunction
  std::string name;
  std::array<unsigned, 3> grid{};
  std::array<unsigned, 3> block{};
  unsigned smem = 0;
  std::vector<uint8_t> image;
  std::vector<std::pair<size_t, size_t>> layout;
  std::vector<int64_t> attrs; // kNodeAttrs values
  // the node's incoming edge of the harvested capture is a programmatic
  // dependent launch edge: the library launched it with programmatic stream
  // serialization (an edge of the capture, not an attribute of the node)
  bool programmatic = false;
  uint64_t dst = 0;
  unsigned value = 0;
  unsigned elem = 1;
  uint64_t width = 0;
};

class TORCH_CUDA_CPP_API Exec {
 public:
  // Reads the kernel, memset and one-dimensional memcpy nodes (host-to-device
  // or device-to-device) of a captured (not yet instantiated) graph; any
  // other node is a TapeMismatch.
  Exec(at::cuda::CUDAGraph& graph, at::DeviceIndex device);
  // the captured nodes in capture order as (kind, index within the kind):
  // kind 0 kernel, 1 memset, 2 memcpy
  std::vector<std::pair<int, size_t>> node_kinds() const {
    return order_;
  }
  // Re-read kernel node j through the driver: a closed region's node the
  // runtime registered (a copy the host made of an operand) must be able to
  // change its kernel like the library's own nodes.
  void adopt_driver(size_t j);
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
  std::string kernel_name(size_t j) const;
  std::vector<uint8_t> image(size_t j) const;
  std::array<unsigned, 3> grid(size_t j) const;
  std::array<unsigned, 3> block(size_t j) const;
  unsigned smem(size_t j) const;
  // the node's attributes
  std::vector<int64_t> attrs(size_t j) const;

  // the memset nodes, in capture order: destination, byte count, value
  size_t num_memset_nodes() const {
    return memsets_.size();
  }
  uint64_t memset_dst(size_t j) const;
  uint64_t memset_bytes(size_t j) const;
  unsigned memset_value(size_t j) const;

  // the memcpy nodes, in capture order: source, destination, byte count
  size_t num_memcpy_nodes() const {
    return memcpys_.size();
  }
  uint64_t memcpy_src(size_t j) const;
  uint64_t memcpy_dst(size_t j) const;
  uint64_t memcpy_bytes(size_t j) const;
  // the captured node's kind: "h2d", "d2d", or "default" (cudaMemcpyDefault,
  // which names no direction)
  std::string memcpy_kind(size_t j) const;

  // Instantiate, then either replay once on the current stream and wait for
  // that stream only, or upload the exec without launching it (a node
  // patched before the first upload stays slow either way; a build that must
  // execute nothing of the function uploads).
  void instantiate(bool replay = true);
  // One call: push the changed nodes and launch on the current stream.
  void run(
      const std::vector<NodeUpdate>& updates,
      const std::vector<MemsetUpdate>& memset_updates = {},
      const std::vector<MemcpyUpdate>& memcpy_updates = {});

  int64_t dirty_nodes() const {
    return dirty_nodes_;
  }
  int64_t dirty_memset_nodes() const {
    return dirty_memset_nodes_;
  }
  int64_t dirty_memcpy_nodes() const {
    return dirty_memcpy_nodes_;
  }
  // updates that had to go through the graph node and cuGraphExecUpdate
  int64_t graph_updates() const {
    return graph_updates_;
  }

  // The nodes of a graph one closed library call (cuBLAS) was captured into
  // (a CUDAGraph with its own pool, so the call's allocations are logged like
  // a build's): kernels and one-dimensional memsets in creation order, a
  // TapeMismatch for anything else. `probe_attr` >= 0 adds one attribute id
  // to every kernel node's census (tests force the driver's refusal with an
  // id it does not know). `anchored` drops the capture's one root node: the
  // anchor kernel the harvest launches ahead of the call, so that the call's
  // first node has an incoming edge whose type says whether the library
  // launched it with programmatic stream serialization. The caller keeps the
  // graph alive.
  static std::vector<HarvestedNode> harvest_nodes(
      cudaGraph_t g,
      int probe_attr = -1,
      bool anchored = false);

 private:
  struct NodeState {
    cudaGraphNode_t node = nullptr;
    std::vector<std::pair<size_t, size_t>> infos;
    std::vector<char> image;
    std::vector<void*> argptrs;
    cudaKernelNodeParams params{};
    // a node the runtime cannot read: kept as the driver's parameter struct
    bool driver = false;
    uint64_t func = 0; // CUfunction
    std::vector<uint8_t> dparams; // CUDA_KERNEL_NODE_PARAMS, opaque here
    std::vector<int64_t> attrs;
  };
  struct MemsetState {
    cudaGraphNode_t node = nullptr;
    cudaMemsetParams params{};
  };
  struct MemcpyState {
    cudaGraphNode_t node = nullptr;
    cudaMemcpy3DParms params{};
  };
  void read_driver_node(NodeState& ns);
  void relayout(NodeState& ns, uint64_t func);
  at::DeviceIndex device_;
  at::cuda::CUDAGraph* graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  bool instantiated_ = false;
  std::vector<NodeState> nodes_;
  std::vector<std::vector<int64_t>> deps_;
  std::vector<std::pair<int, size_t>> order_;
  std::vector<MemsetState> memsets_;
  std::vector<MemcpyState> memcpys_;
  int64_t dirty_nodes_ = 0;
  int64_t dirty_memset_nodes_ = 0;
  int64_t dirty_memcpy_nodes_ = 0;
  int64_t graph_updates_ = 0;
};

} // namespace at::cuda::host_trace
