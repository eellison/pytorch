// The closed regions' harvest reader: the nodes of a graph the Python side
// captured one closed library call (cuBLAS) into, read through the driver
// entry points. The runtime cannot read a node whose kernel it never
// registered (a library kernel launched through the driver:
// cudaGraphKernelNodeGetParams fails with cudaErrorInvalidDeviceFunction,
// measured), so the nodes are read with the driver, image included when the
// launch used the `extra` parameter buffer. Kernel node attributes (the
// cluster dimension, its scheduling preference, cooperative, priority, the
// memory-sync domain) travel with a harvested variant: a replay that swaps a
// variant in must carry them (a transplant across a cluster change without
// them fails asynchronously or hangs, measured), so an attribute the driver
// does not report declines the node. The Python side (_harvest in
// torch/cuda/_host_trace.py) classifies the image bytes into operand, scratch,
// workspace and host slots over several captures; what is read here is one
// capture's nodes.
#pragma once
#include <c10/macros/Export.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace at::cuda::host_trace {

// The closed call's capture holds a node a region does not describe (a
// two-dimensional memset, a memcpy, an event) or a node attribute the driver
// does not report: the harvest declines the key by name.
struct TORCH_CUDA_CPP_API TapeMismatch : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// The kernel node attributes tracked per node, as one integer vector:
// cluster x, y, z; cluster scheduling policy preference; cooperative;
// priority; memory-sync domain; domain map default, remote. An attribute a
// variant sets that this list does not track cannot be transplanted, so the
// harvest declines it.
constexpr size_t kNodeAttrs = 9;

// one node of a captured graph: a kernel (kind 0), a one-dimensional memset
// (kind 1: dst, value, elem, width) or a one-dimensional memcpy (kind 2:
// src, dst, width in bytes)
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
  uint64_t src = 0;
  uint64_t dst = 0;
  unsigned value = 0;
  unsigned elem = 1;
  uint64_t width = 0;
};

// The nodes of a graph one closed library call (cuBLAS) was captured into
// (a CUDAGraph with its own pool, so the call's allocations are logged):
// kernels, one-dimensional memsets and one-dimensional memcpys in creation
// order (the harvest itself refuses a call that copies, _harvest_capture;
// the tests read any capture through this), a TapeMismatch for anything
// else. `probe_attr` >= 0 adds one attribute id to every kernel node's
// census (tests force the driver's refusal with an id it does not know).
// The caller keeps the graph alive.
TORCH_CUDA_CPP_API std::vector<HarvestedNode> harvest_nodes(
    cudaGraph_t g,
    int probe_attr = -1);

} // namespace at::cuda::host_trace
