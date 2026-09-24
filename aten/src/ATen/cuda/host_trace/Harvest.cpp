#include <ATen/cuda/host_trace/Harvest.h>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

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
// The driver entry points the harvest needs: a library kernel launched
// through the driver is read with these.
struct Driver {
  CUresult (*kernelNodeGetParams)(CUgraphNode, CUDA_KERNEL_NODE_PARAMS*) =
      nullptr;
  CUresult (*kernelNodeGetAttribute)(
      CUgraphNode,
      CUkernelNodeAttrID,
      CUkernelNodeAttrValue*) = nullptr;
  CUresult (*funcGetParamInfo)(CUfunction, size_t, size_t*, size_t*) = nullptr;
  CUresult (*funcGetName)(const char**, CUfunction) = nullptr;
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
    entry(r.kernelNodeGetAttribute, "cuGraphKernelNodeGetAttribute");
    entry(r.funcGetParamInfo, "cuFuncGetParamInfo");
    entry(r.funcGetName, "cuFuncGetName");
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

// the tracked attributes of a kernel node (Harvest.h kNodeAttrs). A query the
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
#endif

} // namespace

std::vector<HarvestedNode> harvest_nodes(cudaGraph_t g, int probe_attr) {
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
      if (type == cudaGraphNodeTypeMemcpy) {
        cudaMemcpy3DParms mp{};
        C10_CUDA_CHECK(cudaGraphMemcpyNodeGetParams(nodes[i], &mp));
        const bool linear = mp.srcArray == nullptr && mp.dstArray == nullptr &&
            mp.extent.height == 1 && mp.extent.depth == 1;
        if (!linear) {
          // @allow-raw-throw: registered with pybind11 as _HostTraceTapeMismatch and caught by that name in torch/cuda/_host_trace.py
          throw TapeMismatch(
              "the closed call produced a memcpy node that is not a "
              "one-dimensional copy, which a closed region does not describe");
        }
        HarvestedNode h;
        h.kind = 2;
        h.name = "memcpy";
        h.src = reinterpret_cast<uintptr_t>(mp.srcPtr.ptr);
        h.dst = reinterpret_cast<uintptr_t>(mp.dstPtr.ptr);
        h.width = static_cast<uint64_t>(mp.extent.width);
        out.push_back(std::move(h));
        continue;
      }
      if (type != cudaGraphNodeTypeKernel) {
        // @allow-raw-throw: registered with pybind11 as _HostTraceTapeMismatch and caught by that name in torch/cuda/_host_trace.py
        throw TapeMismatch(c10::str(
            "the closed call produced a node of type ",
            static_cast<int>(type),
            " (neither a kernel, a memset nor a memcpy), which a closed "
            "region does not describe"));
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
      h.image.assign(
          image.begin(),
          image.begin() + static_cast<std::vector<char>::difference_type>(sz));
      h.attrs = read_attrs(node, probe_attr);
      out.push_back(std::move(h));
    }
  }
#endif
  return out;
}

} // namespace at::cuda::host_trace
