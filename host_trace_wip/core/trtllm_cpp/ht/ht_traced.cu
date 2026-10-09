// The host-trace build of FlashInfer's trtllm-gen paged attention launcher: the patched sources
// (land/core/trtllm_cpp/src) compiled inside namespace fi_ht_traced with FI_HT_TRACED, where every symbolic size
// and address is a c10::SymInt and a launch is recorded (ht_trace.h). The stock headers come first, in the global
// namespace, as FlashInfer's own build has them; the patched ones see them through ht_trace.h's using-directives.

// ---- the stock headers (the launcher's includes)
#include <flashinfer/allocator.h>
#include <flashinfer/exception.h>
#include <flashinfer/trtllm/common.h>
#include <flashinfer/trtllm/fmha/decoder_impl_common.h>
#include <flashinfer/trtllm/fmha/fmhaRunnerParams.h>
#include <tvm/ffi/container/variant.h>

#include <algorithm>
#include <cstdint>
#include <flashinfer/trtllm/fmha/fmhaRunner.cuh>
#include <flashinfer/utils.cuh>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include "tvm/ffi/error.h"
#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}
}  // namespace flashinfer

#include <c10/cuda/CUDAStream.h>
#include <dlfcn.h>
#include <link.h>
#include <c10/util/Exception.h>

#define FI_HT_TRACED
#include "fi_ht.h"

#undef IKL_LOG_DEBUG
#define IKL_LOG_DEBUG(...) \
  do {                     \
  } while (0)

// ---- the patched sources
namespace fi_ht_traced {
#include "../src/include/flashinfer/trtllm/fmha/fmhaRunner.cuh"
#include "../src/csrc/trtllm_fmha_kernel_launcher.cu"

namespace tensorrt_llm::kernels {
void runFmhaReduction(TllmGenFmhaKernelMetaInfo const& kernelMeta, KernelParams const&, int32_t, bool, cudaStream_t) {
  if (isGmemReductionWithSeparateKernel(static_cast<MultiCtasKvMode>(kernelMeta.mMultiCtasKvMode))) {
    fi_ht::decline("the separate reduction kernel (GmemReductionWithSeparateKernel) is a later item");
  }
}
}  // namespace tensorrt_llm::kernels
}  // namespace fi_ht_traced

#include "ht_entry.h"

namespace fi_ht {

Hooks hooks;

void decline(const std::string& why) { C10_THROW_ERROR(NotImplementedError, "fi_ht: " + why); }

Recorder*& recorder() {
  thread_local Recorder* r = nullptr;
  return r;
}

std::unordered_set<const void*>& live_proxies() {
  thread_local std::unordered_set<const void*> s;
  return s;
}

std::string function_name(CUfunction f) {
  const char* name = nullptr;
  if (cuFuncGetName(&name, f) != CUDA_SUCCESS || name == nullptr) {
    decline("cuFuncGetName failed");
  }
  return name;
}

// FlashInfer's own module: its load address and its kernels' host stubs (local symbols, so not dlsym's: their
// values from the file's symbol table)
static uintptr_t stock_base = 0;
static std::unordered_map<std::string, uintptr_t> stock_symbols;

void set_stock_library(const std::string& path, const std::unordered_map<std::string, uint64_t>& symbols) {
  struct Find {
    const std::string* path;
    uintptr_t base = 0;
    bool found = false;
  } find{&path};
  dl_iterate_phdr(
      [](dl_phdr_info* info, size_t, void* data) {
        auto* f = static_cast<Find*>(data);
        if (info->dlpi_name != nullptr && *f->path == info->dlpi_name) {
          f->base = info->dlpi_addr;
          f->found = true;
          return 1;
        }
        return 0;
      },
      &find);
  if (!find.found) {
    decline("FlashInfer's module " + path + " is not loaded");
  }
  stock_base = find.base;
  stock_symbols.clear();
  for (auto const& [name, value] : symbols) {
    stock_symbols[name] = static_cast<uintptr_t>(value);
  }
}

std::pair<uint64_t, std::string> stock_kernel(const std::string& traced_name) {
  // the mangled name without this build's namespace (_ZN12fi_ht_traced10flashinfer... -> _ZN10flashinfer...)
  static const std::string tag = "12fi_ht_traced";
  auto at = traced_name.find(tag);
  if (at == std::string::npos || stock_base == 0) {
    decline("no stock twin of kernel " + traced_name);
  }
  std::string name = traced_name.substr(0, at) + traced_name.substr(at + tag.size());
  auto it = stock_symbols.find(name);
  cudaFunction_t f = nullptr;
  if (it == stock_symbols.end() ||
      cudaGetFuncBySymbol(&f, reinterpret_cast<const void*>(stock_base + it->second)) != cudaSuccess) {
    decline("no stock kernel " + name);
  }
  return {reinterpret_cast<uint64_t>(f), function_name(reinterpret_cast<CUfunction>(f))};
}

cudaStream_t current_stream() { return c10::cuda::getCurrentCUDAStream().stream(); }

void check_stream(CUstream s) {
  if (s != reinterpret_cast<CUstream>(current_stream())) {
    decline("a launch off the current stream");
  }
}

static bool capturing() {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(current_stream(), &status) != cudaSuccess) {
    return true;
  }
  return status != cudaStreamCaptureStatusNone;
}

void loading(char const* name) {
  if (recorder() != nullptr && capturing()) {
    decline(std::string("kernel ") + name + " was not loaded at the warm-up (a capture holds)");
  }
}

// the traced build's kernel table for a dtype triple (the one its runners use: TllmFmhaKernelFactory's)
static fi_ht_traced::TllmGenFmhaKernel const& kernels(int64_t dtype_q, int64_t dtype_kv, int64_t dtype_o) {
  return *fi_ht_traced::getTllmFmhaKernels(static_cast<Data_type>(dtype_q), static_cast<Data_type>(dtype_kv),
                                           static_cast<Data_type>(dtype_kv), static_cast<Data_type>(dtype_o),
                                           getSMVersion());
}

}  // namespace fi_ht

// ---- the traced records

namespace fi_ht_traced {

CUresult cuTensorMapEncodeTiled(fi_ht::TensorMap* m, CUtensorMapDataType dtype, unsigned rank, fi_ht::Ptr<void> address,
                                const fi_ht::SymInt* shape, const fi_ht::SymInt* strides, const uint32_t* box,
                                const uint32_t* element_strides, CUtensorMapInterleave interleave,
                                CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2, CUtensorMapFloatOOBfill fill) {
  if (interleave != CU_TENSOR_MAP_INTERLEAVE_NONE || l2 != CU_TENSOR_MAP_L2_PROMOTION_L2_128B) {
    fi_ht::decline("a TMA descriptor with interleave or another L2 promotion");
  }
  if (address.null()) {
    fi_ht::decline("a TMA descriptor of a null address");
  }
  m->set = true;
  m->dtype = static_cast<int>(dtype);
  m->address = address.a;
  m->shape.assign(shape, shape + rank);
  m->strides.assign(strides, strides + rank - 1);
  m->box.assign(box, box + rank);
  for (unsigned i = 0; i < rank; i++) {
    if (element_strides[i] != 1) {
      fi_ht::decline("a TMA descriptor with element strides");
    }
  }
  m->swizzle = static_cast<int>(swizzle);
  m->fill = static_cast<int>(fill);
  return CUDA_SUCCESS;
}

CUresult cuLaunchKernelEx(const fi_ht::LaunchConfig* config, CUfunction f, void** params, void** extra) {
  using fi_ht::guard_int;
  fi_ht::check_stream(config->hStream);
  if (extra != nullptr || params == nullptr || !fi_ht::live_proxies().count(params[0])) {
    fi_ht::decline("a cuLaunchKernelEx of something other than the KernelParams proxy");
  }
  fi_ht::Packer packer;
  static_cast<const fi_ht::KernelParamsFields*>(params[0])->pack(packer);
  // the kernel's parameter is its own size (the cubin's KernelParams), which the struct's padding may exceed: the
  // driver copies that many bytes
  size_t param_offset = 0, param_size = 0;
  if (cuFuncGetParamInfo(f, 0, &param_offset, &param_size) != CUDA_SUCCESS || param_offset != 0 ||
      param_size > packer.image.size()) {
    fi_ht::decline("cuFuncGetParamInfo of the FMHA kernel");
  }
  packer.image.resize(param_size);
  for (auto const& fr : packer.fields) {
    if (fr.offset + fr.width > param_size) {
      fi_ht::decline("a KernelParams member beyond the kernel's parameter");
    }
  }
  for (auto const& t : packer.tmas) {
    if (t.offset + 128 > param_size) {
      fi_ht::decline("a TMA descriptor beyond the kernel's parameter");
    }
  }
  fi_ht::LaunchRec r;
  r.function = reinterpret_cast<uint64_t>(f);
  r.name = fi_ht::function_name(f);
  r.packed = true;
  r.params.push_back(std::move(packer.image));
  r.offsets.push_back(0);
  r.fields = std::move(packer.fields);
  r.tmas = std::move(packer.tmas);
  r.grid[0] = config->gridDimX, r.grid[1] = config->gridDimY, r.grid[2] = config->gridDimZ;
  r.block[0] = config->blockDimX, r.block[1] = config->blockDimY, r.block[2] = config->blockDimZ;
  r.smem = guard_int(config->sharedMemBytes);
  for (unsigned i = 0; i < config->numAttrs; i++) {
    auto const& a = config->attrs[i];
    switch (a.id) {
      case CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION:
        // node topology: a cluster width is a guard of its own (a split flip redispatches the op)
        r.cluster[0] = guard_int(a.value.clusterDim.x);
        r.cluster[1] = guard_int(a.value.clusterDim.y);
        r.cluster[2] = guard_int(a.value.clusterDim.z);
        break;
      case CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE:
        r.policy = static_cast<int>(a.value.clusterSchedulingPolicyPreference);
        break;
      case CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION:
        r.pdl = a.value.programmaticStreamSerializationAllowed != 0;
        break;
      default:
        fi_ht::decline("launch attribute " + std::to_string(static_cast<int>(a.id)));
    }
  }
  fi_ht::recorder()->launches.push_back(std::move(r));
  return CUDA_SUCCESS;
}

// The occupancy run() reads for a Cga kernel: of the function, block, smem and cluster shape (the grid does not
// enter it; FORK's plan_check found it the same at every grid), so it is asked at a one-cluster grid.
CUresult cuOccupancyMaxActiveClusters(int* n, CUfunction f, const fi_ht::LaunchConfig* config) {
  using fi_ht::guard_int;
  ::CUlaunchConfig c{};
  ::CUlaunchAttribute attrs[4] = {};
  c.blockDimX = guard_int(config->blockDimX);
  c.blockDimY = guard_int(config->blockDimY);
  c.blockDimZ = guard_int(config->blockDimZ);
  c.sharedMemBytes = guard_int(config->sharedMemBytes);
  unsigned cluster[3] = {1, 1, 1};
  for (unsigned i = 0; i < config->numAttrs && i < 4; i++) {
    auto const& a = config->attrs[i];
    attrs[i].id = a.id;
    if (a.id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
      cluster[0] = guard_int(a.value.clusterDim.x);
      cluster[1] = guard_int(a.value.clusterDim.y);
      cluster[2] = guard_int(a.value.clusterDim.z);
      attrs[i].value.clusterDim.x = cluster[0];
      attrs[i].value.clusterDim.y = cluster[1];
      attrs[i].value.clusterDim.z = cluster[2];
    } else if (a.id == CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE) {
      attrs[i].value.clusterSchedulingPolicyPreference = a.value.clusterSchedulingPolicyPreference;
    } else if (a.id == CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION) {
      attrs[i].value.programmaticStreamSerializationAllowed = a.value.programmaticStreamSerializationAllowed;
    }
  }
  c.gridDimX = cluster[0], c.gridDimY = cluster[1], c.gridDimZ = cluster[2];
  c.attrs = attrs;
  c.numAttrs = config->numAttrs;
  return ::cuOccupancyMaxActiveClusters(n, f, &c);
}

}  // namespace fi_ht_traced

// ---- the binding's entry points (ht_entry.h)

namespace fi_ht {

std::vector<std::string> candidates(int64_t dtype_q, int64_t dtype_kv, int64_t dtype_o, bool generation,
                                    int64_t head_dim_qk, int64_t head_dim_v, int64_t page_size) {
  fi_ht_traced::TllmGenFmhaRunnerParams p;
  p.mQkvLayout = QkvLayout::PagedKv;
  p.mKernelType = generation ? FmhaKernelType::Generation : FmhaKernelType::Context;
  p.mHeadDimQk = static_cast<int>(head_dim_qk);
  p.mHeadDimV = static_cast<int>(head_dim_v);
  p.mNumTokensPerPage = static_cast<int>(page_size);
  auto const& k = kernels(dtype_q, dtype_kv, dtype_o);
  std::vector<std::string> out;
  for (uint64_t h : k.htCandidates(p)) {
    out.emplace_back(k.htName(h));
  }
  return out;
}

void load(int64_t dtype_q, int64_t dtype_kv, int64_t dtype_o, const std::string& name) {
  kernels(dtype_q, dtype_kv, dtype_o).htLoadNamed(name);
}

}  // namespace fi_ht
