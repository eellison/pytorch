#include <ATen/cuda/host_trace/Exec.h>
#include <ATen/cuda/host_trace/Recorder.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <unordered_map>

#ifndef USE_ROCM
#include <cuda.h>
#endif

namespace at::cuda::host_trace {

namespace {

#ifndef USE_ROCM
// The driver entry points the closed-region nodes need. The runtime cannot
// read a node whose kernel it never registered (a library kernel launched
// through the driver), so those nodes are read and updated with these.
struct Driver {
  CUresult (*kernelNodeGetParams)(CUgraphNode, CUDA_KERNEL_NODE_PARAMS*) =
      nullptr;
  CUresult (*kernelNodeSetParams)(CUgraphNode, const CUDA_KERNEL_NODE_PARAMS*) =
      nullptr;
  CUresult (*execKernelNodeSetParams)(
      CUgraphExec,
      CUgraphNode,
      const CUDA_KERNEL_NODE_PARAMS*) = nullptr;
  CUresult (*kernelNodeGetAttribute)(
      CUgraphNode,
      CUkernelNodeAttrID,
      CUkernelNodeAttrValue*) = nullptr;
  CUresult (*kernelNodeSetAttribute)(
      CUgraphNode,
      CUkernelNodeAttrID,
      const CUkernelNodeAttrValue*) = nullptr;
  CUresult (*funcGetParamInfo)(CUfunction, size_t, size_t*, size_t*) = nullptr;
  CUresult (*funcGetName)(const char**, CUfunction) = nullptr;
  CUresult (*execUpdate)(CUgraphExec, CUgraph, CUgraphExecUpdateResultInfo*) =
      nullptr;
  CUresult (*getErrorString)(CUresult, const char**) = nullptr;
};

template <class F>
void entry(F& fn, const char* name) {
  void* p = nullptr;
  cudaDriverEntryPointQueryResult q{};
  C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion(
      name, &p, CUDART_VERSION, cudaEnableDefault, &q));
  TORCH_CHECK(
      q == cudaDriverEntryPointSuccess && p != nullptr,
      "host_trace: the CUDA driver has no entry point for ",
      name);
  fn = reinterpret_cast<F>(p);
}

const Driver& driver() {
  static const Driver d = [] {
    Driver r;
    entry(r.kernelNodeGetParams, "cuGraphKernelNodeGetParams");
    entry(r.kernelNodeSetParams, "cuGraphKernelNodeSetParams");
    entry(r.execKernelNodeSetParams, "cuGraphExecKernelNodeSetParams");
    entry(r.kernelNodeGetAttribute, "cuGraphKernelNodeGetAttribute");
    entry(r.kernelNodeSetAttribute, "cuGraphKernelNodeSetAttribute");
    entry(r.funcGetParamInfo, "cuFuncGetParamInfo");
    entry(r.funcGetName, "cuFuncGetName");
    entry(r.execUpdate, "cuGraphExecUpdate");
    entry(r.getErrorString, "cuGetErrorString");
    return r;
  }();
  return d;
}

void check(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) {
    return;
  }
  const char* msg = nullptr;
  driver().getErrorString(r, &msg);
  TORCH_CHECK(false, "host_trace: ", what, " failed: ", msg ? msg : "?");
}

// (offset, size) per parameter of a driver function, cached: a closed
// region's variants reuse a handful of kernels
const std::vector<std::pair<size_t, size_t>>& driver_layout(CUfunction f) {
  static std::mutex mu;
  static std::unordered_map<CUfunction, std::vector<std::pair<size_t, size_t>>>
      cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(f);
  if (it != cache.end()) {
    return it->second;
  }
  std::vector<std::pair<size_t, size_t>> infos;
  for (size_t i = 0;; ++i) {
    size_t off = 0, size = 0;
    CUresult r = driver().funcGetParamInfo(f, i, &off, &size);
    if (r == CUDA_ERROR_INVALID_VALUE) {
      break;
    }
    check(r, "cuFuncGetParamInfo");
    infos.emplace_back(off, size);
  }
  return cache.emplace(f, std::move(infos)).first->second;
}

size_t image_size_of(const std::vector<std::pair<size_t, size_t>>& infos) {
  size_t n = 0;
  for (auto [off, size] : infos) {
    n = std::max(n, off + size);
  }
  return n;
}

CUfunction function_of(const CUDA_KERNEL_NODE_PARAMS& p) {
  return p.func != nullptr ? p.func : reinterpret_cast<CUfunction>(p.kern);
}

// the argument bytes of a driver-read node: from its parameter array, or
// from the packed buffer of a launch that used `extra`
std::vector<char> driver_image(
    const CUDA_KERNEL_NODE_PARAMS& p,
    const std::vector<std::pair<size_t, size_t>>& infos) {
  const size_t n = image_size_of(infos);
  std::vector<char> image((n + 7) / 8 * 8, 0);
  if (p.kernelParams != nullptr) {
    for (size_t i = 0; i < infos.size(); ++i) {
      std::memcpy(
          image.data() + infos[i].first, p.kernelParams[i], infos[i].second);
    }
    return image;
  }
  if (p.extra == nullptr) {
    // @allow-raw-throw: registered with pybind11 as _HostTraceTapeMismatch and caught by that name in torch/cuda/_host_trace.py
    throw TapeMismatch(
        "a captured kernel node has neither a parameter array nor an extra buffer");
  }
  const void* buf = nullptr;
  size_t buf_size = 0;
  for (void** e = p.extra; *e != CU_LAUNCH_PARAM_END; e += 2) {
    if (*e == CU_LAUNCH_PARAM_BUFFER_POINTER) {
      buf = e[1];
    } else if (*e == CU_LAUNCH_PARAM_BUFFER_SIZE) {
      buf_size = *static_cast<const size_t*>(e[1]);
    }
  }
  if (buf == nullptr || buf_size < n) {
    // @allow-raw-throw: registered with pybind11 as _HostTraceTapeMismatch and caught by that name in torch/cuda/_host_trace.py
    throw TapeMismatch(
        "a captured kernel node's extra buffer is smaller than its parameter layout");
  }
  std::memcpy(image.data(), buf, n);
  return image;
}

std::string driver_name(CUfunction f) {
  const char* name = nullptr;
  check(driver().funcGetName(&name, f), "cuFuncGetName");
  return name ? std::string(name) : std::string();
}

// the tracked attributes of a kernel node (Exec.h kNodeAttrs). A query the
// driver refuses never reads as a default: an attribute it does not report
// (CUDA_ERROR_INVALID_VALUE) declines the node, since a mismatch the census
// cannot see hangs a transplant; any other failure is an error. `probe` adds
// one attribute id to the census (tests force a refusal with an unknown id).
std::vector<int64_t> read_attrs(CUgraphNode node, int probe = -1) {
  std::vector<int64_t> a(kNodeAttrs, 0);
  CUkernelNodeAttrValue v{};
  auto get = [&](CUlaunchAttributeID id, const char* name) {
    std::memset(&v, 0, sizeof(v));
    CUresult r = driver().kernelNodeGetAttribute(node, id, &v);
    if (r == CUDA_ERROR_INVALID_VALUE) {
      // @allow-raw-throw: registered with pybind11 as _HostTraceTapeMismatch and caught by that name in torch/cuda/_host_trace.py
      throw TapeMismatch(
          std::string("the driver does not report kernel node attribute ") +
          name + ", which the node census tracks");
    }
    check(r, (std::string("cuGraphKernelNodeGetAttribute(") + name + ")").c_str());
  };
  get(CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, "CLUSTER_DIMENSION");
  a[0] = v.clusterDim.x;
  a[1] = v.clusterDim.y;
  a[2] = v.clusterDim.z;
  get(CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE,
      "CLUSTER_SCHEDULING_POLICY_PREFERENCE");
  a[3] = static_cast<int64_t>(v.clusterSchedulingPolicyPreference);
  get(CU_LAUNCH_ATTRIBUTE_COOPERATIVE, "COOPERATIVE");
  a[4] = v.cooperative;
  get(CU_LAUNCH_ATTRIBUTE_PRIORITY, "PRIORITY");
  a[5] = v.priority;
  get(CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN, "MEM_SYNC_DOMAIN");
  a[6] = static_cast<int64_t>(v.memSyncDomain);
  get(CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, "MEM_SYNC_DOMAIN_MAP");
  a[7] = v.memSyncDomainMap.default_;
  a[8] = v.memSyncDomainMap.remote;
  if (probe >= 0) {
    get(static_cast<CUlaunchAttributeID>(probe), "probe");
  }
  return a;
}

// set the attributes of `want` that differ from `have` on a graph node
void write_attrs(
    CUgraphNode node,
    const std::vector<int64_t>& want,
    const std::vector<int64_t>& have) {
  TORCH_CHECK(
      want.size() == kNodeAttrs && have.size() == kNodeAttrs,
      "host_trace: a node attribute vector has ",
      want.size(),
      " / ",
      have.size(),
      " entries, expected ",
      kNodeAttrs);
  CUkernelNodeAttrValue v{};
  if (want[0] != have[0] || want[1] != have[1] || want[2] != have[2]) {
    std::memset(&v, 0, sizeof(v));
    v.clusterDim.x = static_cast<unsigned>(want[0]);
    v.clusterDim.y = static_cast<unsigned>(want[1]);
    v.clusterDim.z = static_cast<unsigned>(want[2]);
    check(
        driver().kernelNodeSetAttribute(
            node, CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, &v),
        "cuGraphKernelNodeSetAttribute(cluster)");
  }
  if (want[3] != have[3]) {
    std::memset(&v, 0, sizeof(v));
    v.clusterSchedulingPolicyPreference =
        static_cast<CUclusterSchedulingPolicy>(want[3]);
    check(
        driver().kernelNodeSetAttribute(
            node, CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, &v),
        "cuGraphKernelNodeSetAttribute(cluster scheduling)");
  }
  if (want[4] != have[4]) {
    std::memset(&v, 0, sizeof(v));
    v.cooperative = static_cast<int>(want[4]);
    check(
        driver().kernelNodeSetAttribute(
            node, CU_LAUNCH_ATTRIBUTE_COOPERATIVE, &v),
        "cuGraphKernelNodeSetAttribute(cooperative)");
  }
  if (want[5] != have[5]) {
    std::memset(&v, 0, sizeof(v));
    v.priority = static_cast<int>(want[5]);
    check(
        driver().kernelNodeSetAttribute(node, CU_LAUNCH_ATTRIBUTE_PRIORITY, &v),
        "cuGraphKernelNodeSetAttribute(priority)");
  }
  if (want[6] != have[6]) {
    std::memset(&v, 0, sizeof(v));
    v.memSyncDomain = static_cast<CUlaunchMemSyncDomain>(want[6]);
    check(
        driver().kernelNodeSetAttribute(
            node, CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN, &v),
        "cuGraphKernelNodeSetAttribute(mem sync domain)");
  }
  if (want[7] != have[7] || want[8] != have[8]) {
    std::memset(&v, 0, sizeof(v));
    v.memSyncDomainMap.default_ = static_cast<unsigned char>(want[7]);
    v.memSyncDomainMap.remote = static_cast<unsigned char>(want[8]);
    check(
        driver().kernelNodeSetAttribute(
            node, CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, &v),
        "cuGraphKernelNodeSetAttribute(mem sync domain map)");
  }
}

void fill_params(
    CUDA_KERNEL_NODE_PARAMS& p,
    uint64_t func,
    const std::array<unsigned, 3>& grid,
    const std::array<unsigned, 3>& block,
    unsigned smem,
    void** argptrs) {
  p.func = reinterpret_cast<CUfunction>(func);
  p.kern = nullptr;
  p.gridDimX = grid[0];
  p.gridDimY = grid[1];
  p.gridDimZ = grid[2];
  p.blockDimX = block[0];
  p.blockDimY = block[1];
  p.blockDimZ = block[2];
  p.sharedMemBytes = smem;
  p.kernelParams = argptrs;
  p.extra = nullptr;
}
#endif

} // namespace

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
    order_.emplace_back(0, nodes_.size());
    // a kernel the runtime registered reads back here; a library kernel
    // launched through the driver does not (cudaErrorInvalidDeviceFunction)
    // and neither does a launch that used the extra buffer
    cudaError_t err = cudaGraphKernelNodeGetParams(node, &ns.params);
    if (err != cudaSuccess) {
      (void)cudaGetLastError();
    }
    if (err != cudaSuccess || ns.params.kernelParams == nullptr ||
        ns.params.extra != nullptr) {
      read_driver_node(ns);
      nodes_.push_back(std::move(ns));
      continue;
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
#ifndef USE_ROCM
    // the driver handle behind the host symbol: what a function change or a
    // driver-path push of this node compares against
    cudaFunction_t cu_function = nullptr;
    C10_CUDA_CHECK(cudaGetFuncBySymbol(&cu_function, ns.params.func));
    ns.func = reinterpret_cast<uint64_t>(cu_function);
#endif
#ifndef USE_ROCM
    ns.attrs = read_attrs(reinterpret_cast<CUgraphNode>(node));
#endif
    nodes_.push_back(std::move(ns));
  }
  for (auto& ns : nodes_) { // pointers into the final (non-moving) images
    ns.argptrs.resize(ns.infos.size());
    for (size_t i = 0; i < ns.infos.size(); ++i) {
      ns.argptrs[i] = ns.image.data() + ns.infos[i].first;
    }
    ns.params.kernelParams = ns.argptrs.data();
  }
}

Exec::~Exec() = default;

void Exec::read_driver_node(NodeState& ns) {
#ifdef USE_ROCM
  // @allow-raw-throw: registered with pybind11 as _HostTraceTapeMismatch and caught by that name in torch/cuda/_host_trace.py
  throw TapeMismatch(
      "the capture contains a kernel node the runtime cannot read; closed "
      "regions are CUDA-only in this version");
#else
  CUDA_KERNEL_NODE_PARAMS p{};
  check(
      driver().kernelNodeGetParams(
          reinterpret_cast<CUgraphNode>(ns.node), &p),
      "cuGraphKernelNodeGetParams");
  CUfunction f = function_of(p);
  ns.driver = true;
  ns.func = reinterpret_cast<uint64_t>(f);
  ns.infos = driver_layout(f);
  ns.image = driver_image(p, ns.infos);
  ns.dparams.assign(
      reinterpret_cast<const uint8_t*>(&p),
      reinterpret_cast<const uint8_t*>(&p) + sizeof(p));
  ns.attrs = read_attrs(reinterpret_cast<CUgraphNode>(ns.node));
  // the runtime-side view the accessors read
  ns.params = cudaKernelNodeParams{};
  ns.params.gridDim = dim3(p.gridDimX, p.gridDimY, p.gridDimZ);
  ns.params.blockDim = dim3(p.blockDimX, p.blockDimY, p.blockDimZ);
  ns.params.sharedMemBytes = p.sharedMemBytes;
#endif
}

void Exec::relayout(NodeState& ns, uint64_t func) {
#ifndef USE_ROCM
  ns.infos = driver_layout(reinterpret_cast<CUfunction>(func));
  ns.image.assign((image_size_of(ns.infos) + 7) / 8 * 8, 0);
  ns.argptrs.resize(ns.infos.size());
  for (size_t i = 0; i < ns.infos.size(); ++i) {
    ns.argptrs[i] = ns.image.data() + ns.infos[i].first;
  }
  ns.func = func;
#endif
}

std::string Exec::kernel_name(size_t j) const {
  const NodeState& ns = nodes_.at(j);
#ifndef USE_ROCM
  if (ns.driver) {
    return driver_name(reinterpret_cast<CUfunction>(ns.func));
  }
#endif
  return func_info(ns.params.func).name;
}

std::vector<uint8_t> Exec::image(size_t j) const {
  const NodeState& ns = nodes_.at(j);
  const size_t n = image_size_of(ns.infos);
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

std::vector<int64_t> Exec::attrs(size_t j) const {
  return nodes_.at(j).attrs;
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

void Exec::adopt_driver(size_t j) {
  NodeState& ns = nodes_.at(j);
  if (ns.driver) {
    return;
  }
  read_driver_node(ns);
  ns.argptrs.resize(ns.infos.size());
  for (size_t i = 0; i < ns.infos.size(); ++i) {
    ns.argptrs[i] = ns.image.data() + ns.infos[i].first;
  }
  ns.params.kernelParams = ns.argptrs.data();
}

void Exec::instantiate(bool replay) {
  c10::cuda::CUDAGuard guard(device_);
  graph_->instantiate();
  exec_ = graph_->raw_cuda_graph_exec();
  instantiated_ = true;
  if (replay) {
    graph_->replay();
    // the caller's stream only: a device-wide synchronize from here would
    // invalidate a trace capture on another thread
    c10::cuda::getCurrentCUDAStream(device_).synchronize();
  } else {
    C10_CUDA_CHECK(
        cudaGraphUpload(exec_, c10::cuda::getCurrentCUDAStream(device_)));
  }
}

void Exec::run(
    const std::vector<NodeUpdate>& updates,
    const std::vector<MemsetUpdate>& memset_updates,
    const std::vector<MemcpyUpdate>& memcpy_updates) {
  TORCH_CHECK(instantiated_, "host_trace: this exec was never instantiated");
  c10::cuda::CUDAGuard guard(device_);
  bool exec_update = false;
  for (const NodeUpdate& u : updates) {
    NodeState& ns = nodes_.at(u.node);
    if (ns.driver) {
#ifndef USE_ROCM
      auto& p = *reinterpret_cast<CUDA_KERNEL_NODE_PARAMS*>(ns.dparams.data());
      if (u.func != 0 && u.func != ns.func) {
        relayout(ns, u.func);
      }
      if (!u.image.empty()) {
        TORCH_CHECK(
            u.image.size() <= ns.image.size(),
            "host_trace: argument image larger than the node's layout");
        std::memcpy(ns.image.data(), u.image.data(), u.image.size());
      }
      fill_params(p, ns.func, u.grid, u.block, u.smem, ns.argptrs.data());
      CUgraphNode node = reinterpret_cast<CUgraphNode>(ns.node);
      const bool attrs_change = !u.attrs.empty() && u.attrs != ns.attrs;
      if (attrs_change) {
        // an attribute changed (the cluster dimension of another variant):
        // the exec setter would not carry it, and a mismatch hangs; the
        // graph node takes params and attributes, the exec is updated from
        // it. The driver validates params against the node's cluster and
        // the cluster against the node's grid, so the cluster is cleared
        // first, the params set, then every attribute written.
        std::vector<int64_t> neutral = ns.attrs;
        neutral[0] = neutral[1] = neutral[2] = 1;
        write_attrs(node, neutral, ns.attrs);
        check(driver().kernelNodeSetParams(node, &p), "cuGraphKernelNodeSetParams");
        write_attrs(node, u.attrs, neutral);
        ns.attrs = u.attrs;
        exec_update = true;
        ++graph_updates_;
      } else {
        // the graph node mirrors every update so that a later update
        // through the graph (cuGraphExecUpdate) finds it equal to the exec
        check(driver().kernelNodeSetParams(node, &p), "cuGraphKernelNodeSetParams");
        check(
            driver().execKernelNodeSetParams(
                reinterpret_cast<CUgraphExec>(exec_), node, &p),
            "cuGraphExecKernelNodeSetParams");
      }
      ns.params.gridDim = dim3(u.grid[0], u.grid[1], u.grid[2]);
      ns.params.blockDim = dim3(u.block[0], u.block[1], u.block[2]);
      ns.params.sharedMemBytes = u.smem;
#endif
    } else {
      TORCH_CHECK(
          u.func == 0 || u.func == ns.func,
          "host_trace: only a closed region's node changes its kernel");
      TORCH_CHECK(
          u.attrs.empty() || u.attrs == ns.attrs,
          "host_trace: only a closed region's node changes its attributes");
      if (!u.image.empty()) {
        TORCH_CHECK(
            u.image.size() <= ns.image.size(),
            "host_trace: argument image larger than the captured one");
        std::memcpy(ns.image.data(), u.image.data(), u.image.size());
      }
      ns.params.gridDim = dim3(u.grid[0], u.grid[1], u.grid[2]);
      ns.params.blockDim = dim3(u.block[0], u.block[1], u.block[2]);
      ns.params.sharedMemBytes = u.smem;
      C10_CUDA_CHECK(cudaGraphKernelNodeSetParams(ns.node, &ns.params));
      C10_CUDA_CHECK(
          cudaGraphExecKernelNodeSetParams(exec_, ns.node, &ns.params));
    }
  }
  for (const MemsetUpdate& u : memset_updates) {
    MemsetState& ms = memsets_.at(u.node);
    cudaMemsetParams& p = ms.params;
    p.dst = reinterpret_cast<void*>(static_cast<uintptr_t>(u.dst));
    p.value = u.value;
    // one-dimensional: the captured element size when the count still
    // divides by it, else bytes (value 0 sets the same bytes either way)
    if (u.bytes % p.elementSize != 0) {
      p.elementSize = 1;
    }
    p.width = static_cast<size_t>(u.bytes / p.elementSize);
    p.height = 1;
    p.pitch = p.width * p.elementSize;
    C10_CUDA_CHECK(cudaGraphMemsetNodeSetParams(ms.node, &p));
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
    C10_CUDA_CHECK(cudaGraphMemcpyNodeSetParams(mc.node, &p));
    C10_CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams(exec_, mc.node, &p));
  }
#ifndef USE_ROCM
  if (exec_update) {
    CUgraphExecUpdateResultInfo info{};
    CUresult r = driver().execUpdate(
        reinterpret_cast<CUgraphExec>(exec_),
        reinterpret_cast<CUgraph>(graph_->raw_cuda_graph()),
        &info);
    TORCH_CHECK(
        r == CUDA_SUCCESS && info.result == CU_GRAPH_EXEC_UPDATE_SUCCESS,
        "host_trace: cuGraphExecUpdate failed with result ",
        static_cast<int>(info.result));
  }
#endif
  dirty_nodes_ += static_cast<int64_t>(updates.size());
  dirty_memset_nodes_ += static_cast<int64_t>(memset_updates.size());
  dirty_memcpy_nodes_ += static_cast<int64_t>(memcpy_updates.size());
  graph_->replay();
}

std::vector<HarvestedNode> Exec::harvest_nodes(cudaGraph_t g, int probe_attr) {
  std::vector<HarvestedNode> out;
#ifdef USE_ROCM
  TORCH_CHECK(false, "host_trace: closed regions are CUDA-only in this version");
#else
  {
    size_t n = 0;
    C10_CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &n));
    std::vector<cudaGraphNode_t> nodes(n);
    C10_CUDA_CHECK(cudaGraphGetNodes(g, nodes.data(), &n));
    for (size_t i = 0; i < n; ++i) {
      cudaGraphNodeType type{};
      C10_CUDA_CHECK(cudaGraphNodeGetType(nodes[i], &type));
      if (type == cudaGraphNodeTypeMemset) {
        cudaMemsetParams mp{};
        C10_CUDA_CHECK(cudaGraphMemsetNodeGetParams(nodes[i], &mp));
        if (mp.height != 1) {
          // @allow-raw-throw: registered with pybind11 as _HostTraceTapeMismatch and caught by that name in torch/cuda/_host_trace.py
          throw TapeMismatch(
              "the closed call produced a two-dimensional memset node, which "
              "a closed region does not describe");
        }
        HarvestedNode h;
        h.kind = 1;
        h.name = "memset";
        h.dst = reinterpret_cast<uintptr_t>(mp.dst);
        h.value = mp.value;
        h.elem = mp.elementSize;
        h.width = static_cast<uint64_t>(mp.width);
        out.push_back(std::move(h));
        continue;
      }
      if (type != cudaGraphNodeTypeKernel) {
        // @allow-raw-throw: registered with pybind11 as _HostTraceTapeMismatch and caught by that name in torch/cuda/_host_trace.py
        throw TapeMismatch(c10::str(
            "the closed call produced a node of type ",
            static_cast<int>(type),
            " (neither a kernel nor a memset), which a closed region does "
            "not describe"));
      }
      const auto node = reinterpret_cast<CUgraphNode>(nodes[i]);
      CUDA_KERNEL_NODE_PARAMS p{};
      check(
          driver().kernelNodeGetParams(node, &p), "cuGraphKernelNodeGetParams");
      CUfunction f = function_of(p);
      HarvestedNode h;
      h.func = reinterpret_cast<uint64_t>(f);
      h.name = driver_name(f);
      h.grid = {p.gridDimX, p.gridDimY, p.gridDimZ};
      h.block = {p.blockDimX, p.blockDimY, p.blockDimZ};
      h.smem = p.sharedMemBytes;
      h.layout = driver_layout(f);
      std::vector<char> image = driver_image(p, h.layout);
      const size_t sz = image_size_of(h.layout);
      h.image.assign(image.begin(), image.begin() + sz);
      h.attrs = read_attrs(node, probe_attr);
      out.push_back(std::move(h));
    }
  }
#endif
  return out;
}

} // namespace at::cuda::host_trace
