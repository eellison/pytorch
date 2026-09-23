#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAGraphParams.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/native/cuda/jit_utils.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <ATen/BlasSettingsEpoch.h>
#include <ATen/Context.h>
#include <ATen/cuda/tunable/Tunable.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12040
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/cuda/driver_api.h>
#endif

namespace at::cuda::detail {

class KernelParamCache {
 public:
  uint64_t generation = 0;
  bool pointer_replay_failed = false;
  std::unordered_map<uintptr_t, std::shared_ptr<KernelNodeParams>> nodes;
};

static void validate_tensor_map_binding(const KernelTensorMapBinding& binding, size_t pointer_count, size_t value_count) {
  TORCH_CHECK_VALUE(binding.data_type <= 12 && binding.swizzle <= 3,
                   "Tensor maps require an unpacked CUDA data type and NONE/32B/64B/128B swizzle");
  const auto rank = binding.dimensions.size();
  TORCH_CHECK_VALUE(rank >= 1 && rank <= 5 && binding.strides.size() + 1 == rank &&
                       binding.box_dimensions.size() == rank,
                   "Tensor-map dimensions, strides and box dimensions must have a matching rank from 1 to 5");
  TORCH_CHECK_INDEX(binding.pointer_index < pointer_count, "Tensor-map pointer index is out of range");
  TORCH_CHECK_INDEX(binding.address_offset_value_index < value_count, "Tensor-map offset index is out of range");
  for (auto value : binding.dimensions) {
    TORCH_CHECK_INDEX(value < value_count, "Tensor-map dimension index is out of range");
  }
  for (auto value : binding.strides) {
    TORCH_CHECK_INDEX(value < value_count, "Tensor-map stride index is out of range");
  }
  for (auto dimension : binding.box_dimensions) {
    TORCH_CHECK_VALUE(dimension >= 1 && dimension <= 256, "Tensor-map box dimensions must be from 1 to 256");
  }
}

namespace {

struct TemplateSite {
  std::map<std::vector<int64_t>, int64_t> by_key;
  std::vector<std::unique_ptr<const KernelTemplateVariant>> variants;
  std::vector<KernelTemplateNode::Kind> kinds; // the chain every variant has
};

struct TemplateRegistry {
  std::shared_mutex mutex;
  std::vector<TemplateSite> sites;
};

TemplateRegistry& template_registry() {
  static auto* registry = new TemplateRegistry();
  return *registry;
}

// per thread and site: the last key selected (the region's arguments, without
// the settings), the settings epoch it was selected under and the index it
// selected; the last missed key (with the settings) for the harvest
struct TemplateSelection {
  std::vector<int64_t> key;
  uint64_t epoch = 0;
  int64_t index = -1;
  std::optional<std::vector<int64_t>> missed;
};
thread_local std::vector<TemplateSelection> tls_template_selections;

TemplateSelection& template_selection(size_t site) {
  if (tls_template_selections.size() <= site) {
    tls_template_selections.resize(site + 1);
  }
  return tls_template_selections[site];
}

// every process-global switch that changes which kernels the BLAS library runs
// for a shape, appended to every template key
// (the matmul float32 precision covers allow_tf32: its legacy setter writes it,
// and reading the legacy getter after the new setter is an error; the SM
// carveout is applied around every cuBLAS call and can change its kernel
// choice, -1 when unset). Every setter of these bumps at::blasSettingsEpoch().
using LibrarySettings = std::array<int64_t, 11>;

LibrarySettings library_settings() {
  auto& context = at::globalContext();
  auto* tuning = at::cuda::tunable::getTuningContext();
  return {
      static_cast<int64_t>(tuning->IsTunableOpEnabled()),
      static_cast<int64_t>(tuning->IsTuningEnabled()),
      static_cast<int64_t>(context.blasPreferredBackend()),
      static_cast<int64_t>(context.float32Precision(at::Float32Backend::CUDA, at::Float32Op::MATMUL)),
      static_cast<int64_t>(context.allowFP16ReductionCuBLAS()),
      static_cast<int64_t>(context.allowBF16ReductionCuBLAS()),
      static_cast<int64_t>(context.allowFP16AccumulationCuBLAS()),
      static_cast<int64_t>(context.deterministicAlgorithms()),
      static_cast<int64_t>(at::cuda::getChosenWorkspaceSize()),
      static_cast<int64_t>(at::cuda::getCUDABlasLtWorkspaceSize()),
      static_cast<int64_t>(context._SMCarveout_EXPERIMENTAL().value_or(-1)),
  };
}

// what a node reads for an attribute it was not launched with: a cluster of
// one, and the default cluster scheduling policy resolves to spread (1)
void normalize_attributes(KernelNodeAttributes& attributes) {
  for (size_t axis = 0; axis < 3; ++axis) {
    if (attributes[axis] == 0) {
      attributes[axis] = 1;
    }
  }
  if (attributes[3] == 0) {
    attributes[3] = 1;
  }
}

// hotpath-2 measurement (see the header): timers and probes, all off by default
struct HotpathTimers {
  int64_t calls = 0;
  int64_t update_ns = 0;
  int64_t templates_ns = 0;
  int64_t rest_ns = 0;
};
thread_local HotpathTimers tls_hotpath_timers;
bool hotpath_timers_on() {
  static const bool on = std::getenv("TORCH_CUDAGRAPH_HOTPATH_TIMERS") != nullptr;
  return on;
}
bool hotpath_probe(const char* name) {
  static const char* probes = std::getenv("TORCH_CUDAGRAPH_HOTPATH_PROBE");
  return probes != nullptr && std::strstr(probes, name) != nullptr;
}
inline int64_t hotpath_now() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

const KernelTemplateVariant* kernel_template_variant(int64_t site, int64_t index) {
  auto& registry = template_registry();
  std::shared_lock lock(registry.mutex);
  if (site < 0 || static_cast<size_t>(site) >= registry.sites.size()) {
    return nullptr;
  }
  const auto& variants = registry.sites[static_cast<size_t>(site)].variants;
  if (index < 0 || static_cast<size_t>(index) >= variants.size()) {
    return nullptr;
  }
  return variants[static_cast<size_t>(index)].get();
}

} // namespace

int64_t new_kernel_template_site() {
  auto& registry = template_registry();
  std::unique_lock lock(registry.mutex);
  registry.sites.emplace_back();
  return static_cast<int64_t>(registry.sites.size() - 1);
}

int64_t register_kernel_template(int64_t site, std::vector<int64_t> key, KernelTemplateVariant variant) {
  TORCH_CHECK_VALUE(!variant.nodes.empty(), "A template variant needs nodes");
  std::vector<KernelTemplateNode::Kind> kinds;
  kinds.reserve(variant.nodes.size());
  for (auto& node : variant.nodes) {
    kinds.push_back(node.kind);
    if (node.kind == KernelTemplateNode::Kind::memset) {
      TORCH_CHECK_VALUE(node.memset_element_size == 1, "Template memset nodes are byte memsets");
      TORCH_CHECK_VALUE(node.memset_width > 0, "A template memset node needs a width");
      continue;
    }
    normalize_attributes(node.attributes);
    for (const auto& slot : node.slots) {
      TORCH_CHECK_VALUE(slot.offset + sizeof(uintptr_t) <= node.image.size(), "A template slot lies outside its image");
    }
    for (auto offset : node.workspace_slots) {
      TORCH_CHECK_VALUE(offset + sizeof(uintptr_t) <= node.image.size(), "A template workspace slot lies outside its image");
    }
  }
  auto& registry = template_registry();
  std::unique_lock lock(registry.mutex);
  TORCH_CHECK_INDEX(
      site >= 0 && static_cast<size_t>(site) < registry.sites.size(), "Unknown kernel template site ", site);
  auto& entry = registry.sites[static_cast<size_t>(site)];
  auto found = entry.by_key.find(key);
  if (found != entry.by_key.end()) {
    return found->second;
  }
  if (entry.variants.empty()) {
    entry.kinds = kinds;
  } else {
    TORCH_CHECK_VALUE(
        kinds == entry.kinds,
        "Template variant has ", kinds.size(), " nodes or another node chain than its site's ", entry.kinds.size());
  }
  const auto index = static_cast<int64_t>(entry.variants.size());
  entry.variants.push_back(std::make_unique<const KernelTemplateVariant>(std::move(variant)));
  entry.by_key.emplace(std::move(key), index);
  return index;
}

namespace {

// the lookup: the key with the settings appended against the site's map; the
// selection and, on a miss, the missed key are the calling thread's
int64_t lookup_kernel_template(size_t site, std::vector<int64_t> key, TemplateSelection& selection) {
  for (auto setting : library_settings()) {
    key.push_back(setting);
  }
  int64_t index = -1;
  {
    auto& registry = template_registry();
    std::shared_lock lock(registry.mutex);
    if (site < registry.sites.size()) {
      const auto& by_key = registry.sites[site].by_key;
      auto found = by_key.find(key);
      if (found != by_key.end()) {
        index = found->second;
      }
    }
  }
  selection.index = index;
  if (index < 0) {
    selection.missed = std::move(key);
  }
  return index;
}

} // namespace

int64_t select_kernel_template(const std::vector<int64_t>& arguments) {
  // called from compiled predicates: no exceptions, an unknown site misses
  if (arguments.empty() || arguments[0] < 0) {
    return -1;
  }
  const auto site = static_cast<size_t>(arguments[0]);
  return lookup_kernel_template(
      site, std::vector<int64_t>(arguments.begin() + 1, arguments.end()), template_selection(site));
}

int64_t select_kernel_template_key(int64_t site, const int64_t* key, size_t length) {
  if (site < 0) {
    return -1;
  }
  auto& selection = template_selection(static_cast<size_t>(site));
  // the epoch before the settings: a setter writes its setting, then bumps, so a
  // result computed under an epoch read here holds for every call that reads it
  const auto epoch = at::blasSettingsEpoch();
  if (selection.index >= 0 && selection.epoch == epoch && selection.key.size() == length &&
      std::memcmp(selection.key.data(), key, length * sizeof(int64_t)) == 0) {
    return selection.index;
  }
  const auto index =
      lookup_kernel_template(static_cast<size_t>(site), std::vector<int64_t>(key, key + length), selection);
  if (index >= 0) {
    selection.key.assign(key, key + length);
    selection.epoch = epoch;
  }
  return index;
}

int64_t selected_kernel_template_at(const int64_t* arguments, size_t length) {
  if (length != 1 || arguments[0] < 0 ||
      static_cast<size_t>(arguments[0]) >= tls_template_selections.size()) {
    return -1;
  }
  return tls_template_selections[static_cast<size_t>(arguments[0])].index;
}

int64_t selected_kernel_template(const std::vector<int64_t>& arguments) {
  return selected_kernel_template_at(arguments.data(), arguments.size());
}

std::optional<std::vector<int64_t>> take_kernel_template_miss(int64_t site) {
  if (site < 0 || static_cast<size_t>(site) >= tls_template_selections.size()) {
    return std::nullopt;
  }
  auto& slot = tls_template_selections[static_cast<size_t>(site)].missed;
  auto result = std::move(slot);
  slot.reset();
  return result;
}

std::vector<KernelTemplateNode::Kind> kernel_template_site_kinds(int64_t site) {
  auto& registry = template_registry();
  std::shared_lock lock(registry.mutex);
  TORCH_CHECK_INDEX(
      site >= 0 && static_cast<size_t>(site) < registry.sites.size(), "Unknown kernel template site ", site);
  return registry.sites[static_cast<size_t>(site)].kinds;
}

std::array<int64_t, 4> hotpath_timers_take() {
  auto& t = tls_hotpath_timers;
  std::array<int64_t, 4> out{t.calls, t.update_ns, t.templates_ns, t.rest_ns};
  t = HotpathTimers{};
  return out;
}

std::vector<int64_t> kernel_template_library_settings() {
  const auto settings = library_settings();
  return {settings.begin(), settings.end()};
}

static uintptr_t displaced_pointer(uintptr_t base, int64_t offset) {
  if (offset >= 0) {
    const auto increment = static_cast<uint64_t>(offset);
    TORCH_CHECK_VALUE(increment <= std::numeric_limits<uintptr_t>::max() - base,
                     "Pointer address addition overflowed");
    return base + increment;
  }
  const auto decrement = uint64_t{0} - static_cast<uint64_t>(offset);
  TORCH_CHECK_VALUE(decrement <= base, "Pointer address subtraction underflowed");
  return base - decrement;
}

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12040
static_assert(sizeof(CUtensorMap) == 128);

CaptureFrontier get_capture_frontier(uintptr_t stream) {
  cudaStreamCaptureStatus status{};
  unsigned long long capture_id = 0;
  cudaGraph_t graph = nullptr;
  const cudaGraphNode_t* dependencies = nullptr;
  const cudaGraphEdgeData* edges = nullptr;
  size_t count = 0;
#if CUDA_VERSION >= 13000
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(
      reinterpret_cast<cudaStream_t>(stream), &status, &capture_id, &graph,
      &dependencies, &edges, &count));
#else
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo_v3(
      reinterpret_cast<cudaStream_t>(stream), &status, &capture_id, &graph,
      &dependencies, &edges, &count));
#endif
  CaptureFrontier result;
  result.status = static_cast<int>(status);
  if (status != cudaStreamCaptureStatusActive) {
    return result;
  }
  TORCH_CHECK(count == 0 || (dependencies && edges), "Missing CUDA capture dependency data");
  result.capture_id = capture_id;
  result.graph = reinterpret_cast<uintptr_t>(graph);
  result.dependencies.reserve(count);
  static_assert(sizeof(cudaGraphEdgeData) == 8);
  for (size_t index = 0; index < count; ++index) {
    std::array<uint8_t, 8> edge;
    std::memcpy(edge.data(), &edges[index], edge.size());
    result.dependencies.emplace_back(reinterpret_cast<uintptr_t>(dependencies[index]), edge);
  }
  return result;
}

struct KernelNodeParams::Impl {
  struct Image {
    std::vector<uint8_t> buffer;
    std::vector<void*> pointers;
    size_t size = 0;
    std::array<void*, 5> extra{};
  };

  CUgraphNode node;
  CUDA_KERNEL_NODE_PARAMS params{};
  std::vector<std::pair<size_t, size_t>> layout;
  std::array<Image, 2> images;
  std::vector<CUtensorMap> tensor_maps;
  size_t committed = 0;
  bool packed;

  std::vector<uint8_t>& stage() {
    const auto& source = images[committed].buffer;
    auto& target = images[committed ^ 1].buffer;
    std::copy(source.begin(), source.end(), target.begin());
    return target;
  }

  void submit(uintptr_t graph_exec, CUDA_KERNEL_NODE_PARAMS request) {
    auto& image = images[committed ^ 1];
    if (packed) {
      request.extra = image.extra.data();
    } else {
      request.kernelParams = image.pointers.data();
    }
    C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuGraphExecKernelNodeSetParams_(
        reinterpret_cast<CUgraphExec>(graph_exec), node, &request));
    committed ^= 1;
  }
};

KernelNodeParams::KernelNodeParams(uintptr_t node) : impl_(std::make_unique<Impl>()) {
  auto* api = c10::cuda::DriverAPI::get();
  TORCH_CHECK(
      api->cuGraphKernelNodeGetParams_ && api->cuGraphExecKernelNodeSetParams_,
      "CUDA driver does not support kernel parameter updates");
  auto& state = *impl_;
  state.node = reinterpret_cast<CUgraphNode>(node);
  C10_CUDA_DRIVER_CHECK(api->cuGraphKernelNodeGetParams_(state.node, &state.params));
  TORCH_CHECK(state.params.func || state.params.kern, "Kernel node has no kernel function");
  TORCH_CHECK(
      state.params.func ? api->cuFuncGetParamInfo_ != nullptr : api->cuKernelGetParamInfo_ != nullptr,
      "Kernel parameter queries require CUDA 12.4 or later");
  size_t extent = 0;
  for (size_t index = 0;; ++index) {
    size_t offset = 0;
    size_t size = 0;
    auto status = state.params.func
        ? api->cuFuncGetParamInfo_(state.params.func, index, &offset, &size)
        : api->cuKernelGetParamInfo_(state.params.kern, index, &offset, &size);
    if (status == CUDA_ERROR_INVALID_VALUE) {
      break;
    }
    C10_CUDA_DRIVER_CHECK(status);
    TORCH_CHECK(size <= std::numeric_limits<size_t>::max() - offset, "Kernel parameter extent overflows");
    extent = std::max(extent, offset + size);
    state.layout.emplace_back(offset, size);
  }

  state.packed = state.params.extra != nullptr;
  auto& bytes = state.images[0].buffer;
  if (state.packed) {
    const void* buffer = nullptr;
    const void* size_pointer = nullptr;
    for (size_t index = 0; index < 5; index += 2) {
      auto tag = state.params.extra[index];
      if (tag == CU_LAUNCH_PARAM_END) {
        break;
      }
      TORCH_CHECK(index < 4, "Invalid CUDA kernel extra parameter buffer");
      auto value = state.params.extra[index + 1];
      TORCH_CHECK(value, "Null CUDA kernel extra parameter buffer");
      if (tag == CU_LAUNCH_PARAM_BUFFER_POINTER && !buffer) {
        buffer = value;
      } else if (tag == CU_LAUNCH_PARAM_BUFFER_SIZE && !size_pointer) {
        size_pointer = value;
      } else {
        TORCH_CHECK(false, "Invalid CUDA kernel extra parameter buffer");
      }
    }
    TORCH_CHECK(buffer && size_pointer, "Missing CUDA kernel extra parameter buffer");
    size_t size = 0;
    std::memcpy(&size, size_pointer, sizeof(size));
    bytes.resize(size);
    if (size) {
      std::memcpy(bytes.data(), buffer, size);
    }
  } else {
    bytes.resize(extent);
    TORCH_CHECK(state.layout.empty() || state.params.kernelParams, "CUDA kernel node has no argument pointers");
    for (size_t index = 0; index < state.layout.size(); ++index) {
      auto [offset, size] = state.layout[index];
      TORCH_CHECK(state.params.kernelParams[index], "CUDA kernel argument pointer is null");
      if (size) {
        std::memcpy(bytes.data() + offset, state.params.kernelParams[index], size);
      }
    }
  }
  // GetParams returns node-owned memory. Future updates use only copied bytes.
  state.params.kernelParams = nullptr;
  state.params.extra = nullptr;
  state.images[1].buffer = bytes;
  for (auto& image : state.images) {
    image.size = image.buffer.size();
    image.extra = {
        CU_LAUNCH_PARAM_BUFFER_POINTER, image.buffer.data(),
        CU_LAUNCH_PARAM_BUFFER_SIZE, &image.size, CU_LAUNCH_PARAM_END};
    if (!state.packed) {
      image.pointers.reserve(state.layout.size());
      for (const auto& argument : state.layout) {
        image.pointers.push_back(image.buffer.data() + argument.first);
      }
    }
  }
}

void KernelNodeParams::validate(const std::vector<KernelArgumentUpdate>& updates) const {
  const auto& state = *impl_;
  const auto& buffer = state.images[state.committed].buffer;
  for (const auto& update : updates) {
    TORCH_CHECK_INDEX(
        update.index >= 0 && static_cast<uint64_t>(update.index) < state.layout.size(),
        "Kernel argument index ", update.index, " is out of range");
    auto [offset, size] = state.layout[update.index];
    TORCH_CHECK_VALUE(
        update.value.size() == size,
        "Kernel argument ", update.index, " requires ", size, " bytes, got ", update.value.size());
    TORCH_CHECK_VALUE(
        offset <= buffer.size() && size <= buffer.size() - offset,
        "Kernel argument ", update.index, " exceeds the captured parameter buffer");
  }
}

KernelNodeSnapshot KernelNodeParams::snapshot() const {
  const auto& state = *impl_;
  const auto& buffer = state.images[state.committed].buffer;
  KernelNodeSnapshot result{
      reinterpret_cast<uintptr_t>(state.node),
      reinterpret_cast<uintptr_t>(state.params.func),
      reinterpret_cast<uintptr_t>(state.params.kern),
      reinterpret_cast<uintptr_t>(state.params.ctx),
      {state.params.gridDimX, state.params.gridDimY, state.params.gridDimZ},
      {state.params.blockDimX, state.params.blockDimY, state.params.blockDimZ},
      state.params.sharedMemBytes, state.packed, {}};
  result.arguments.reserve(state.layout.size());
  for (size_t index = 0; index < state.layout.size(); ++index) {
    auto [offset, size] = state.layout[index];
    TORCH_CHECK_VALUE(
        offset <= buffer.size() && size <= buffer.size() - offset,
        "Kernel argument ", index, " exceeds the captured parameter buffer");
    result.arguments.push_back({
        offset, size,
        std::vector<uint8_t>(buffer.begin() + offset, buffer.begin() + offset + size)});
  }
  return result;
}

void KernelNodeParams::update(uintptr_t graph_exec, const std::vector<KernelArgumentUpdate>& updates) {
  validate(updates);
  if (updates.empty()) {
    return;
  }
  auto& state = *impl_;
  auto& staged = state.stage();
  for (const auto& update : updates) {
    auto offset = state.layout[update.index].first;
    std::copy(update.value.begin(), update.value.end(), staged.begin() + offset);
  }
  state.submit(graph_exec, state.params);
}

size_t KernelNodeParams::pointer_offset(int64_t argument, std::optional<size_t> byte_offset) const {
  const auto& state = *impl_;
  TORCH_CHECK_INDEX(
      argument >= 0 && static_cast<uint64_t>(argument) < state.layout.size(),
      "Kernel argument index ", argument, " is out of range");
  const auto [offset, size] = state.layout[argument];
  TORCH_CHECK_VALUE(
      byte_offset || size == sizeof(uintptr_t), "Kernel argument ", argument, " is not pointer-sized");
  const auto field = byte_offset.value_or(0);
  TORCH_CHECK_VALUE(
      field <= size && sizeof(uintptr_t) <= size - field,
      "Pointer field exceeds kernel argument ", argument);
  const auto& buffer = state.images[state.committed].buffer;
  TORCH_CHECK_VALUE(
      offset <= buffer.size() && size <= buffer.size() - offset,
      "Kernel argument ", argument, " exceeds the captured parameter buffer");
  return offset + field;
}

size_t KernelNodeParams::scalar_offset(
    int64_t argument, size_t width, std::optional<size_t> byte_offset) const {
  const auto& state = *impl_;
  TORCH_CHECK_INDEX(
      argument >= 0 && static_cast<uint64_t>(argument) < state.layout.size(),
      "Kernel argument index ", argument, " is out of range");
  const auto [offset, size] = state.layout[argument];
  TORCH_CHECK_VALUE(byte_offset || size == width, "Kernel scalar width differs from the selected ABI");
  const auto field = byte_offset.value_or(0);
  TORCH_CHECK_VALUE(field <= size && width <= size - field, "Scalar field exceeds kernel argument ", argument);
  const auto& buffer = state.images[state.committed].buffer;
  TORCH_CHECK_VALUE(
      offset <= buffer.size() && size <= buffer.size() - offset,
      "Kernel argument ", argument, " exceeds the captured parameter buffer");
  return offset + field;
}

void KernelNodeParams::reserve_tensor_maps(size_t count) {
  if (impl_->tensor_maps.size() < count) {
    impl_->tensor_maps.resize(count);
  }
}


static void encode_tensor_map_into(
    const KernelTensorMapBinding& binding, c10::ArrayRef<int64_t> values,
    c10::ArrayRef<uintptr_t> pointers, CUtensorMap& encoded) {
  std::array<cuuint64_t, 5> dimensions{}, strides{};
  for (size_t axis = 0; axis < binding.dimensions.size(); ++axis) {
    dimensions[axis] = static_cast<cuuint64_t>(values[binding.dimensions[axis]]);
  }
  for (size_t axis = 0; axis < binding.strides.size(); ++axis) {
    strides[axis] = static_cast<cuuint64_t>(values[binding.strides[axis]]);
  }
  const std::array<cuuint32_t, 5> element_strides{1, 1, 1, 1, 1};
  auto address = displaced_pointer(pointers[binding.pointer_index], values[binding.address_offset_value_index]);
  C10_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuTensorMapEncodeTiled(
      &encoded, static_cast<CUtensorMapDataType>(binding.data_type),
      static_cast<cuuint32_t>(binding.dimensions.size()), reinterpret_cast<void*>(address),
      dimensions.data(), strides.data(), binding.box_dimensions.data(), element_strides.data(),
      CU_TENSOR_MAP_INTERLEAVE_NONE, static_cast<CUtensorMapSwizzle>(binding.swizzle),
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      binding.nan_fill ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

std::array<uint8_t, 128> encode_tensor_map(
    const KernelTensorMapBinding& binding, c10::ArrayRef<int64_t> values, c10::ArrayRef<uintptr_t> pointers) {
  validate_tensor_map_binding(binding, pointers.size(), values.size());
  for (auto index : binding.dimensions) {
    TORCH_CHECK_VALUE(values[index] >= 1 && values[index] <= (int64_t{1} << 32),
                     "Tensor-map dimension is outside the CUDA encoder range");
  }
  for (auto index : binding.strides) {
    TORCH_CHECK_VALUE(values[index] >= 0 && values[index] < (int64_t{1} << 40),
                     "Tensor-map byte stride is outside the CUDA encoder range");
  }
  CUtensorMap encoded{};
  encode_tensor_map_into(binding, values, pointers, encoded);
  std::array<uint8_t, 128> result;
  std::memcpy(result.data(), &encoded, result.size());
  return result;
}

void KernelNodeParams::update_pointers(
    uintptr_t graph_exec,
    c10::ArrayRef<KernelPointerSlot> slots,
    c10::ArrayRef<uintptr_t> pointers) {
  auto& state = *impl_;
  const auto& buffer = state.images[state.committed].buffer;
  const bool dirty = std::any_of(slots.begin(), slots.end(), [&](const auto& slot) {
    return std::memcmp(buffer.data() + slot.offset, &pointers[slot.pointer_index], sizeof(uintptr_t)) != 0;
  });
  if (!dirty) {
    return;
  }
  auto& staged = state.stage();
  for (const auto& slot : slots) {
    std::memcpy(staged.data() + slot.offset, &pointers[slot.pointer_index], sizeof(uintptr_t));
  }
  state.submit(graph_exec, state.params);
}

int KernelNodeParams::max_threads_per_block(c10::DeviceIndex device) const {
  c10::cuda::CUDAGuard device_guard(device);
  const auto& state = *impl_;
  int result = 0;
  if (state.params.func) {
    C10_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuFuncGetAttribute(
        &result, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, state.params.func));
  } else {
    auto* api = c10::cuda::DriverAPI::get();
    TORCH_CHECK(api->cuKernelGetAttribute_, "CUDA driver does not support kernel attribute queries");
    CUdevice native_device;
    C10_CUDA_DRIVER_CHECK(api->cuDeviceGet_(&native_device, device));
    C10_CUDA_DRIVER_CHECK(api->cuKernelGetAttribute_(
        &result, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, state.params.kern, native_device));
  }
  TORCH_CHECK(result > 0, "Kernel maximum block size must be positive");
  return result;
}

void KernelNodeParams::update_replay(
    uintptr_t graph_exec,
    c10::ArrayRef<KernelPointerSlot> pointer_slots,
    c10::ArrayRef<KernelScalarSlot> scalar_slots,
    c10::ArrayRef<KernelTensorMapSlot> tensor_map_slots,
    const std::optional<std::array<size_t, 3>>& grid,
    const std::optional<size_t>& shared_memory,
    c10::ArrayRef<uintptr_t> pointers,
    c10::ArrayRef<int64_t> values,
    const std::optional<std::array<size_t, 3>>& block) {
  auto& state = *impl_;
  const auto& buffer = state.images[state.committed].buffer;
  auto request = state.params;
  auto pointer_value = [&](const KernelPointerSlot& slot) -> uintptr_t {
    return displaced_pointer(pointers[slot.pointer_index],
                             slot.address_offset_value_index ? values[*slot.address_offset_value_index] : 0);
  };
  bool tensor_maps_dirty = false;
  for (size_t index = 0; index < tensor_map_slots.size(); ++index) {
    const auto& slot = tensor_map_slots[index];
    auto& encoded = state.tensor_maps[index];
    encode_tensor_map_into(slot.binding, values, pointers, encoded);
    tensor_maps_dirty |= std::memcmp(buffer.data() + slot.offset, &encoded, sizeof(encoded)) != 0;
  }
  if (grid) {
    request.gridDimX = static_cast<unsigned int>(values[(*grid)[0]]);
    request.gridDimY = static_cast<unsigned int>(values[(*grid)[1]]);
    request.gridDimZ = static_cast<unsigned int>(values[(*grid)[2]]);
  }
  if (shared_memory) {
    request.sharedMemBytes = static_cast<unsigned int>(values[*shared_memory]);
  }
  if (block) {
    request.blockDimX = static_cast<unsigned int>(values[(*block)[0]]);
    request.blockDimY = static_cast<unsigned int>(values[(*block)[1]]);
    request.blockDimZ = static_cast<unsigned int>(values[(*block)[2]]);
  }
  const bool dirty = tensor_maps_dirty || request.gridDimX != state.params.gridDimX ||
      request.gridDimY != state.params.gridDimY || request.gridDimZ != state.params.gridDimZ ||
      request.blockDimX != state.params.blockDimX || request.blockDimY != state.params.blockDimY ||
      request.blockDimZ != state.params.blockDimZ ||
      request.sharedMemBytes != state.params.sharedMemBytes ||
      std::any_of(pointer_slots.begin(), pointer_slots.end(), [&](const auto& slot) {
        const auto pointer = pointer_value(slot);
        return std::memcmp(buffer.data() + slot.offset, &pointer, sizeof(uintptr_t)) != 0;
      }) ||
      std::any_of(scalar_slots.begin(), scalar_slots.end(), [&](const auto& slot) {
        const auto value = values[slot.value_index];
        const uint8_t byte = static_cast<uint8_t>(value);
        const uint16_t half = static_cast<uint16_t>(value);
        const int32_t narrow = slot.width == sizeof(int32_t) ? static_cast<int32_t>(value) : 0;
        const void* bytes = slot.width == 1 ? static_cast<const void*>(&byte)
            : slot.width == 2 ? static_cast<const void*>(&half)
            : slot.width == 4 ? static_cast<const void*>(&narrow) : &value;
        return std::memcmp(buffer.data() + slot.offset, bytes, slot.width) != 0;
      });
  if (!dirty) {
    return;
  }
  auto& staged = state.stage();
  for (size_t index = 0; index < tensor_map_slots.size(); ++index) {
    const auto& encoded = state.tensor_maps[index];
    std::memcpy(staged.data() + tensor_map_slots[index].offset, &encoded, sizeof(encoded));
  }
  for (const auto& slot : pointer_slots) {
    const auto pointer = pointer_value(slot);
    std::memcpy(staged.data() + slot.offset, &pointer, sizeof(uintptr_t));
  }
  for (const auto& slot : scalar_slots) {
    const auto value = values[slot.value_index];
    const uint8_t byte = static_cast<uint8_t>(value);
    const uint16_t half = static_cast<uint16_t>(value);
    const int32_t narrow = slot.width == sizeof(int32_t) ? static_cast<int32_t>(value) : 0;
    const void* bytes = slot.width == 1 ? static_cast<const void*>(&byte)
        : slot.width == 2 ? static_cast<const void*>(&half)
        : slot.width == 4 ? static_cast<const void*>(&narrow) : &value;
    std::memcpy(staged.data() + slot.offset, bytes, slot.width);
  }
  state.submit(graph_exec, request);
  // The cache describes only successful setters, including their launch fields.
  state.params.gridDimX = request.gridDimX;
  state.params.gridDimY = request.gridDimY;
  state.params.gridDimZ = request.gridDimZ;
  state.params.blockDimX = request.blockDimX;
  state.params.blockDimY = request.blockDimY;
  state.params.blockDimZ = request.blockDimZ;
  state.params.sharedMemBytes = request.sharedMemBytes;
}

void KernelNodeParams::mirror_graph_node() const {
  auto& state = *impl_;
  auto request = state.params;
  auto& image = state.images[state.committed];
  if (state.packed) {
    request.extra = image.extra.data();
  } else {
    request.kernelParams = image.pointers.data();
  }
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuGraphKernelNodeSetParams_(state.node, &request));
}

static KernelNodeAttributes read_node_attributes(CUgraphNode node) {
  auto* api = c10::cuda::DriverAPI::get();
  KernelNodeAttributes result{};
  CUkernelNodeAttrValue value{};
  auto read = [&](CUlaunchAttributeID id) {
    std::memset(&value, 0, sizeof(value));
    return api->cuGraphKernelNodeGetAttribute_(node, id, &value) == CUDA_SUCCESS;
  };
  if (read(CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION)) {
    result[0] = value.clusterDim.x;
    result[1] = value.clusterDim.y;
    result[2] = value.clusterDim.z;
  }
  if (read(CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE)) {
    result[3] = static_cast<int64_t>(value.clusterSchedulingPolicyPreference);
  }
  if (read(CU_LAUNCH_ATTRIBUTE_COOPERATIVE)) {
    result[4] = value.cooperative;
  }
  if (read(CU_LAUNCH_ATTRIBUTE_PRIORITY)) {
    result[5] = value.priority;
  }
  if (read(CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN)) {
    result[6] = static_cast<int64_t>(value.memSyncDomain);
  }
  if (read(CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP)) {
    result[7] = value.memSyncDomainMap.default_;
    result[8] = value.memSyncDomainMap.remote;
  }
  normalize_attributes(result);
  return result;
}

KernelNodeAttributes read_kernel_node_attributes(uintptr_t node) {
  return read_node_attributes(reinterpret_cast<CUgraphNode>(node));
}

// set the attributes of `want` that differ from `have` on a graph node
static void write_node_attributes(
    CUgraphNode node, const KernelNodeAttributes& want, const KernelNodeAttributes& have) {
  auto* api = c10::cuda::DriverAPI::get();
  CUkernelNodeAttrValue value{};
  auto write = [&](CUlaunchAttributeID id) {
    C10_CUDA_DRIVER_CHECK(api->cuGraphKernelNodeSetAttribute_(node, id, &value));
    std::memset(&value, 0, sizeof(value));
  };
  if (want[0] != have[0] || want[1] != have[1] || want[2] != have[2]) {
    value.clusterDim.x = static_cast<unsigned int>(want[0]);
    value.clusterDim.y = static_cast<unsigned int>(want[1]);
    value.clusterDim.z = static_cast<unsigned int>(want[2]);
    write(CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION);
  }
  if (want[3] != have[3]) {
    value.clusterSchedulingPolicyPreference = static_cast<CUclusterSchedulingPolicy>(want[3]);
    write(CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE);
  }
  if (want[4] != have[4]) {
    value.cooperative = static_cast<int>(want[4]);
    write(CU_LAUNCH_ATTRIBUTE_COOPERATIVE);
  }
  if (want[5] != have[5]) {
    value.priority = static_cast<int>(want[5]);
    write(CU_LAUNCH_ATTRIBUTE_PRIORITY);
  }
  if (want[6] != have[6]) {
    value.memSyncDomain = static_cast<CUlaunchMemSyncDomain>(want[6]);
    write(CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN);
  }
  if (want[7] != have[7] || want[8] != have[8]) {
    value.memSyncDomainMap.default_ = static_cast<unsigned char>(want[7]);
    value.memSyncDomainMap.remote = static_cast<unsigned char>(want[8]);
    write(CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP);
  }
}

void launch_kernel_image(
    uintptr_t function,
    std::array<unsigned int, 3> grid,
    std::array<unsigned int, 3> block,
    unsigned int shared_memory,
    uintptr_t stream,
    const std::vector<uint8_t>& image,
    const KernelNodeAttributes& attributes,
    bool programmatic) {
  auto* api = c10::cuda::DriverAPI::get();
  TORCH_CHECK(api->cuLaunchKernelEx_, "Kernel image launches require a CUDA 12 driver");
  CUlaunchConfig config{};
  config.gridDimX = grid[0];
  config.gridDimY = grid[1];
  config.gridDimZ = grid[2];
  config.blockDimX = block[0];
  config.blockDimY = block[1];
  config.blockDimZ = block[2];
  config.sharedMemBytes = shared_memory;
  config.hStream = reinterpret_cast<CUstream>(stream);
  std::vector<CUlaunchAttribute> launch_attributes;
  auto add = [&](CUlaunchAttributeID id) -> CUlaunchAttributeValue& {
    launch_attributes.emplace_back();
    launch_attributes.back().id = id;
    return launch_attributes.back().value;
  };
  // a cluster launch when the harvested node was one (the census reads 0 for a
  // node launched without the attribute): a cluster of one is explicit too, since
  // cuBLAS launches its sm100 nvjet "1x1" kernels with it and they raise Warp
  // Illegal Instruction when transplanted without it
  if (attributes[0] != 0 || attributes[1] != 0 || attributes[2] != 0) {
    auto& value = add(CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION);
    value.clusterDim.x = static_cast<unsigned int>(std::max<int64_t>(attributes[0], 1));
    value.clusterDim.y = static_cast<unsigned int>(std::max<int64_t>(attributes[1], 1));
    value.clusterDim.z = static_cast<unsigned int>(std::max<int64_t>(attributes[2], 1));
  }
  // the rest only where it differs from what a plain launch reads back
  // (normalize_attributes; the memory synchronization domain map defaults to
  // default 0, remote 1), so a device without domains is never asked to set them
  if (attributes[3] > 1) {
    add(CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE).clusterSchedulingPolicyPreference =
        static_cast<CUclusterSchedulingPolicy>(attributes[3]);
  }
  if (attributes[4]) {
    add(CU_LAUNCH_ATTRIBUTE_COOPERATIVE).cooperative = static_cast<int>(attributes[4]);
  }
  if (attributes[5]) {
    add(CU_LAUNCH_ATTRIBUTE_PRIORITY).priority = static_cast<int>(attributes[5]);
  }
  if (attributes[6]) {
    add(CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN).memSyncDomain = static_cast<CUlaunchMemSyncDomain>(attributes[6]);
  }
  if (attributes[7] != 0 || attributes[8] != 1) {
    auto& value = add(CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP);
    value.memSyncDomainMap.default_ = static_cast<unsigned char>(attributes[7]);
    value.memSyncDomainMap.remote = static_cast<unsigned char>(attributes[8]);
  }
  if (programmatic) {
    add(CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION)
        .programmaticStreamSerializationAllowed = 1;
  }
  config.attrs = launch_attributes.data();
  config.numAttrs = static_cast<unsigned int>(launch_attributes.size());
  std::vector<uint8_t> bytes(image);
  size_t size = bytes.size();
  void* extra[] = {
      CU_LAUNCH_PARAM_BUFFER_POINTER, bytes.data(), CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
  C10_CUDA_DRIVER_CHECK(
      api->cuLaunchKernelEx_(&config, reinterpret_cast<CUfunction>(function), nullptr, extra));
}

void KernelPointerUpdateBatch::prepare_templates(
    std::vector<KernelTemplateBinding> templates, size_t pointer_count, size_t value_count) {
  if (templates.empty()) {
    return;
  }
  auto* api = c10::cuda::DriverAPI::get();
  TORCH_CHECK(
      api->cuGraphKernelNodeGetParams_ && api->cuGraphKernelNodeSetParams_ && api->cuGraphExecKernelNodeSetParams_ &&
          api->cuGraphKernelNodeGetAttribute_ && api->cuGraphKernelNodeSetAttribute_ && api->cuGraphExecUpdate_ &&
          api->cuFuncGetParamInfo_,
      "Template bindings require a CUDA 12.4 driver");
  std::unordered_set<uintptr_t> seen;
  for (auto& binding : templates) {
    TORCH_CHECK_VALUE(!binding.nodes.empty(), "A template binding needs kernel nodes");
    TORCH_CHECK_VALUE(
        binding.operand_pointer_index.size() == binding.operand_offset_value_index.size(),
        "Template operands need a pointer index and an address offset each");
    TORCH_CHECK_INDEX(binding.variant_value_index < value_count, "Template variant value index is out of range");
    for (auto index : binding.operand_pointer_index) {
      TORCH_CHECK_INDEX(index < pointer_count, "Template operand pointer index is out of range");
    }
    for (const auto& index : binding.operand_offset_value_index) {
      if (index) {
        TORCH_CHECK_INDEX(*index < value_count, "Template operand offset value index is out of range");
      }
    }
    auto& range = value_ranges_[binding.variant_value_index];
    range.first = std::max(range.first, int64_t{0});
    Template entry{std::move(binding), {}, -1};
    for (auto handle : entry.binding.nodes) {
      TORCH_CHECK_VALUE(seen.insert(handle).second, "Duplicate template kernel node handle ", handle);
      TemplateNode state;
      state.node = handle;
      auto node = reinterpret_cast<CUgraphNode>(handle);
      cudaGraphNodeType type{};
      C10_CUDA_CHECK(cudaGraphNodeGetType(reinterpret_cast<cudaGraphNode_t>(handle), &type));
      if (type == cudaGraphNodeTypeMemset) {
        cudaMemsetParams captured{};
        C10_CUDA_CHECK(cudaGraphMemsetNodeGetParams(reinterpret_cast<cudaGraphNode_t>(handle), &captured));
        TORCH_CHECK_VALUE(
            captured.height == 1 && captured.elementSize == 1, "Template memset nodes are one-dimensional byte memsets");
        state.kind = KernelTemplateNode::Kind::memset;
        state.dst = reinterpret_cast<uintptr_t>(captured.dst);
        state.width = captured.width;
        state.value = captured.value;
        entry.nodes.push_back(std::move(state));
        continue;
      }
      TORCH_CHECK_VALUE(type == cudaGraphNodeTypeKernel, "A template binding node must be a kernel or a memset node");
      CUDA_KERNEL_NODE_PARAMS params{};
      C10_CUDA_DRIVER_CHECK(api->cuGraphKernelNodeGetParams_(node, &params));
      TORCH_CHECK(params.func, "A template kernel node has no kernel function");
      state.function = reinterpret_cast<uintptr_t>(params.func);
      state.grid = {params.gridDimX, params.gridDimY, params.gridDimZ};
      state.block = {params.blockDimX, params.blockDimY, params.blockDimZ};
      state.shared_memory = params.sharedMemBytes;
      if (params.extra) {
        const void* buffer = nullptr;
        const void* size_pointer = nullptr;
        for (size_t index = 0; index < 5; index += 2) {
          auto tag = params.extra[index];
          if (tag == CU_LAUNCH_PARAM_END) {
            break;
          }
          TORCH_CHECK(index < 4, "Invalid CUDA kernel extra parameter buffer");
          auto value = params.extra[index + 1];
          if (tag == CU_LAUNCH_PARAM_BUFFER_POINTER && !buffer) {
            buffer = value;
          } else if (tag == CU_LAUNCH_PARAM_BUFFER_SIZE && !size_pointer) {
            size_pointer = value;
          } else {
            TORCH_CHECK(false, "Invalid CUDA kernel extra parameter buffer");
          }
        }
        TORCH_CHECK(buffer && size_pointer, "Missing CUDA kernel extra parameter buffer");
        size_t size = 0;
        std::memcpy(&size, size_pointer, sizeof(size));
        const auto* begin = static_cast<const uint8_t*>(buffer);
        state.image.assign(begin, begin + size);
      } else {
        size_t extent = 0;
        std::vector<std::pair<size_t, size_t>> layout;
        for (size_t index = 0;; ++index) {
          size_t offset = 0;
          size_t size = 0;
          auto status = api->cuFuncGetParamInfo_(params.func, index, &offset, &size);
          if (status == CUDA_ERROR_INVALID_VALUE) {
            break;
          }
          C10_CUDA_DRIVER_CHECK(status);
          extent = std::max(extent, offset + size);
          layout.emplace_back(offset, size);
        }
        TORCH_CHECK(layout.empty() || params.kernelParams, "CUDA kernel node has no argument pointers");
        state.image.assign(extent, 0);
        for (size_t index = 0; index < layout.size(); ++index) {
          auto [offset, size] = layout[index];
          if (size) {
            std::memcpy(state.image.data() + offset, params.kernelParams[index], size);
          }
        }
      }
      state.attributes = read_node_attributes(node);
      entry.nodes.push_back(std::move(state));
    }
    // the captured nodes are the site's chain: the kinds every registered
    // variant has, in order
    const auto kinds = kernel_template_site_kinds(entry.binding.site);
    if (!kinds.empty()) {
      TORCH_CHECK_VALUE(kinds.size() == entry.nodes.size(), "A template binding's nodes are not its site's chain");
      for (size_t index = 0; index < kinds.size(); ++index) {
        TORCH_CHECK_VALUE(
            entry.nodes[index].kind == kinds[index], "A template binding's node ", index, " is not of its site's kind");
      }
    }
    templates_.push_back(std::move(entry));
  }
}

bool KernelPointerUpdateBatch::replay_templates(
    uintptr_t graph_exec, c10::ArrayRef<uintptr_t> pointers, c10::ArrayRef<int64_t> values) const {
  if (templates_.empty()) {
    return false;
  }
  auto* api = c10::cuda::DriverAPI::get();
  auto exec = reinterpret_cast<CUgraphExec>(graph_exec);
  bool graph_update = false;
  static const bool probe_cache = hotpath_probe("template_cache");
  for (auto& entry : templates_) {
    const int64_t index = values[entry.binding.variant_value_index];
    const KernelTemplateVariant* variant = nullptr;
    if (probe_cache && index == entry.current && entry.probe_variant != nullptr) {
      variant = entry.probe_variant;
    } else {
      variant = kernel_template_variant(entry.binding.site, index);
      entry.probe_variant = variant;
    }
    TORCH_CHECK(variant, "Template site ", entry.binding.site, " has no variant ", index);
    const bool swap = index != entry.current;
    // node i of the variant is node i of the site's chain
    TORCH_CHECK(
        variant->nodes.size() == entry.nodes.size(),
        "Template variant ", index, " has ", variant->nodes.size(), " nodes, its site's chain ", entry.nodes.size());
    for (size_t position = 0; position < entry.nodes.size(); ++position) {
      const auto& wanted = variant->nodes[position];
      auto& state = entry.nodes[position];
      TORCH_CHECK(
          state.kind == wanted.kind, "Template variant ", index, " puts a node of another kind at position ", position);
      auto node = reinterpret_cast<CUgraphNode>(state.node);
      if (wanted.kind == KernelTemplateNode::Kind::memset) {
        auto dst = static_cast<uintptr_t>(wanted.memset_delta);
        if (wanted.memset_operand) {
          TORCH_CHECK(
              *wanted.memset_operand < entry.binding.operand_pointer_index.size(), "Template memset names an unbound operand");
          const auto& offset_index = entry.binding.operand_offset_value_index[*wanted.memset_operand];
          const auto base = displaced_pointer(
              pointers[entry.binding.operand_pointer_index[*wanted.memset_operand]], offset_index ? values[*offset_index] : 0);
          dst = static_cast<uintptr_t>(static_cast<int64_t>(base) + wanted.memset_delta);
        }
        if (dst != state.dst || wanted.memset_width != state.width || wanted.memset_value != state.value) {
          cudaMemsetParams params{};
          params.dst = reinterpret_cast<void*>(dst);
          params.elementSize = 1;
          params.width = wanted.memset_width;
          params.height = 1;
          params.pitch = 0;
          params.value = wanted.memset_value;
          C10_CUDA_CHECK(cudaGraphExecMemsetNodeSetParams(
              reinterpret_cast<cudaGraphExec_t>(graph_exec), reinterpret_cast<cudaGraphNode_t>(state.node), &params));
          state.dst = dst;
          state.width = wanted.memset_width;
          state.value = wanted.memset_value;
        }
        continue;
      }
      bool dirty = swap;
      if (swap) {
        state.image = wanted.image;
      }
      auto patch = [&](size_t offset, uintptr_t address) {
        if (std::memcmp(state.image.data() + offset, &address, sizeof(address)) != 0) {
          std::memcpy(state.image.data() + offset, &address, sizeof(address));
          dirty = true;
        }
      };
      for (const auto& slot : wanted.slots) {
        TORCH_CHECK(slot.operand < entry.binding.operand_pointer_index.size(), "Template slot names an unbound operand");
        const auto& offset_index = entry.binding.operand_offset_value_index[slot.operand];
        const auto base = displaced_pointer(
            pointers[entry.binding.operand_pointer_index[slot.operand]], offset_index ? values[*offset_index] : 0);
        patch(slot.offset, static_cast<uintptr_t>(static_cast<int64_t>(base) + slot.delta));
      }
      for (auto offset : wanted.workspace_slots) {
        patch(offset, entry.binding.workspace);
      }
      if (dirty) {
        CUDA_KERNEL_NODE_PARAMS params{};
        params.func = reinterpret_cast<CUfunction>(wanted.function);
        params.gridDimX = wanted.grid[0];
        params.gridDimY = wanted.grid[1];
        params.gridDimZ = wanted.grid[2];
        params.blockDimX = wanted.block[0];
        params.blockDimY = wanted.block[1];
        params.blockDimZ = wanted.block[2];
        params.sharedMemBytes = wanted.shared_memory;
        state.image_size = state.image.size();
        state.extra = {
            CU_LAUNCH_PARAM_BUFFER_POINTER, state.image.data(), CU_LAUNCH_PARAM_BUFFER_SIZE, &state.image_size,
            CU_LAUNCH_PARAM_END};
        params.extra = state.extra.data();
        if (wanted.attributes != state.attributes) {
          // The exec-level setter carries no attributes and a mismatch hangs the
          // launch: the graph node takes params and attributes, and the exec is
          // updated from the graph before the replay. The driver validates params
          // against the node's cluster and the cluster against the grid, so the
          // cluster is cleared first, the params set, then every attribute written.
          KernelNodeAttributes neutral = state.attributes;
          neutral[0] = neutral[1] = neutral[2] = 1;
          write_node_attributes(node, neutral, state.attributes);
          C10_CUDA_DRIVER_CHECK(api->cuGraphKernelNodeSetParams_(node, &params));
          write_node_attributes(node, wanted.attributes, neutral);
          state.attributes = wanted.attributes;
          graph_update = true;
        } else {
          C10_CUDA_DRIVER_CHECK(api->cuGraphExecKernelNodeSetParams_(exec, node, &params));
        }
        state.function = wanted.function;
        state.grid = wanted.grid;
        state.block = wanted.block;
        state.shared_memory = wanted.shared_memory;
      }
    }
    if (swap) {
      entry.current = index;
      ++template_applies_;
    }
  }
  return graph_update;
}

void KernelPointerUpdateBatch::update_exec_from_graph(uintptr_t graph_exec, uintptr_t graph) const {
  // cuGraphExecUpdate applies every difference between the graph and the exec:
  // the nodes updated through the exec so far are mirrored to the graph first
  auto* api = c10::cuda::DriverAPI::get();
  auto exec = reinterpret_cast<CUgraphExec>(graph_exec);
  for (const auto& node : nodes_) {
    node.params->mirror_graph_node();
  }
  for (const auto& memset : memsets_) {
    cudaMemsetParams params{};
    params.dst = reinterpret_cast<void*>(memset.dst);
    params.pitch = memset.pitch;
    params.value = memset.value;
    params.elementSize = 1;
    params.width = memset.width;
    params.height = 1;
    C10_CUDA_CHECK(cudaGraphMemsetNodeSetParams(reinterpret_cast<cudaGraphNode_t>(memset.binding.node), &params));
  }
  for (const auto& memcpy : memcpys_) {
    cudaMemcpy3DParms params{};
    params.srcPtr = make_cudaPitchedPtr(reinterpret_cast<void*>(memcpy.src), memcpy.width, memcpy.width, 1);
    params.dstPtr = make_cudaPitchedPtr(reinterpret_cast<void*>(memcpy.dst), memcpy.width, memcpy.width, 1);
    params.extent = make_cudaExtent(memcpy.width, 1, 1);
    params.kind = memcpy.kind;
    C10_CUDA_CHECK(cudaGraphMemcpyNodeSetParams(reinterpret_cast<cudaGraphNode_t>(memcpy.node), &params));
  }
  for (auto& entry : templates_) {
    for (auto& state : entry.nodes) {
      if (state.kind == KernelTemplateNode::Kind::memset) {
        cudaMemsetParams params{};
        params.dst = reinterpret_cast<void*>(state.dst);
        params.elementSize = 1;
        params.width = state.width;
        params.height = 1;
        params.pitch = 0;
        params.value = state.value;
        C10_CUDA_CHECK(cudaGraphMemsetNodeSetParams(reinterpret_cast<cudaGraphNode_t>(state.node), &params));
        continue;
      }
      CUDA_KERNEL_NODE_PARAMS params{};
      params.func = reinterpret_cast<CUfunction>(state.function);
      params.gridDimX = state.grid[0];
      params.gridDimY = state.grid[1];
      params.gridDimZ = state.grid[2];
      params.blockDimX = state.block[0];
      params.blockDimY = state.block[1];
      params.blockDimZ = state.block[2];
      params.sharedMemBytes = state.shared_memory;
      state.image_size = state.image.size();
      state.extra = {
          CU_LAUNCH_PARAM_BUFFER_POINTER, state.image.data(), CU_LAUNCH_PARAM_BUFFER_SIZE, &state.image_size,
          CU_LAUNCH_PARAM_END};
      params.extra = state.extra.data();
      C10_CUDA_DRIVER_CHECK(api->cuGraphKernelNodeSetParams_(reinterpret_cast<CUgraphNode>(state.node), &params));
    }
  }
  CUgraphExecUpdateResultInfo info{};
  const auto result = api->cuGraphExecUpdate_(exec, reinterpret_cast<CUgraph>(graph), &info);
  TORCH_CHECK(
      result == CUDA_SUCCESS && info.result == CU_GRAPH_EXEC_UPDATE_SUCCESS,
      "cuGraphExecUpdate failed with result ", static_cast<int>(info.result));
  ++template_graph_updates_;
}
#else
std::array<uint8_t, 128> encode_tensor_map(
    const KernelTensorMapBinding&, c10::ArrayRef<int64_t>, c10::ArrayRef<uintptr_t>) {
  TORCH_CHECK(false, "Tensor-map encoding requires NVIDIA CUDA 12.4 or later");
}

CaptureFrontier get_capture_frontier(uintptr_t) {
  TORCH_CHECK(false, "CUDA graph inspection requires NVIDIA CUDA 12.4 or later");
}

struct KernelNodeParams::Impl {};

KernelNodeParams::KernelNodeParams(uintptr_t) {
  TORCH_CHECK(false, "Kernel parameter updates require NVIDIA CUDA 12.4 or later");
}

void KernelNodeParams::validate(const std::vector<KernelArgumentUpdate>&) const {
  TORCH_CHECK(false, "Kernel parameter updates require NVIDIA CUDA 12.4 or later");
}

void KernelNodeParams::update(uintptr_t, const std::vector<KernelArgumentUpdate>&) {
  TORCH_CHECK(false, "Kernel parameter updates require NVIDIA CUDA 12.4 or later");
}

size_t KernelNodeParams::pointer_offset(int64_t, std::optional<size_t>) const {
  TORCH_CHECK(false, "Kernel parameter updates require NVIDIA CUDA 12.4 or later");
}

size_t KernelNodeParams::scalar_offset(int64_t, size_t, std::optional<size_t>) const {
  TORCH_CHECK(false, "Kernel parameter updates require NVIDIA CUDA 12.4 or later");
}

void KernelNodeParams::reserve_tensor_maps(size_t) {
  TORCH_CHECK(false, "Kernel parameter updates require NVIDIA CUDA 12.4 or later");
}

int KernelNodeParams::max_threads_per_block(c10::DeviceIndex) const {
  TORCH_CHECK(false, "Kernel attribute queries require NVIDIA CUDA 12.4 or later");
}

void KernelNodeParams::update_pointers(
    uintptr_t, c10::ArrayRef<KernelPointerSlot>, c10::ArrayRef<uintptr_t>) {
  TORCH_CHECK(false, "Kernel parameter updates require NVIDIA CUDA 12.4 or later");
}

void KernelNodeParams::update_replay(
    uintptr_t,
    c10::ArrayRef<KernelPointerSlot>,
    c10::ArrayRef<KernelScalarSlot>,
    c10::ArrayRef<KernelTensorMapSlot>,
    const std::optional<std::array<size_t, 3>>&,
    const std::optional<size_t>&,
    c10::ArrayRef<uintptr_t>,
    c10::ArrayRef<int64_t>,
    const std::optional<std::array<size_t, 3>>&) {
  TORCH_CHECK(false, "Kernel parameter updates require NVIDIA CUDA 12.4 or later");
}

KernelNodeSnapshot KernelNodeParams::snapshot() const {
  TORCH_CHECK(false, "CUDA graph inspection requires NVIDIA CUDA 12.4 or later");
}
#endif

KernelNodeParams::~KernelNodeParams() = default;

#if defined(USE_ROCM) || !defined(CUDA_VERSION) || CUDA_VERSION < 12040
void KernelNodeParams::mirror_graph_node() const {
  TORCH_CHECK(false, "Kernel parameter updates require NVIDIA CUDA 12.4 or later");
}

void launch_kernel_image(
    uintptr_t, std::array<unsigned int, 3>, std::array<unsigned int, 3>, unsigned int, uintptr_t,
    const std::vector<uint8_t>&, const KernelNodeAttributes&, bool) {
  TORCH_CHECK(false, "Kernel image launches require NVIDIA CUDA 12.4 or later");
}

KernelNodeAttributes read_kernel_node_attributes(uintptr_t) {
  TORCH_CHECK(false, "Kernel node attributes require NVIDIA CUDA 12.4 or later");
}

void KernelPointerUpdateBatch::prepare_templates(std::vector<KernelTemplateBinding> templates, size_t, size_t) {
  TORCH_CHECK(templates.empty(), "Template bindings require NVIDIA CUDA 12.4 or later");
}

bool KernelPointerUpdateBatch::replay_templates(uintptr_t, c10::ArrayRef<uintptr_t>, c10::ArrayRef<int64_t>) const {
  return false;
}

std::vector<KernelTemplateNode::Kind> kernel_template_site_kinds(int64_t) {
  TORCH_CHECK(false, "Kernel templates require NVIDIA CUDA 12.4 or later");
}

void KernelPointerUpdateBatch::update_exec_from_graph(uintptr_t, uintptr_t) const {
  TORCH_CHECK(false, "Template bindings require NVIDIA CUDA 12.4 or later");
}
#endif

void KernelPointerUpdateBatch::validate_values(c10::ArrayRef<int64_t> values) const {
  TORCH_CHECK_VALUE(numeric_, "Prepared updates do not contain a numeric plan");
  TORCH_CHECK_VALUE(values.size() == value_ranges_.size(), "Numeric result count differs from preparation");
  for (size_t index = 0; index < values.size(); ++index) {
    const auto [lower, upper] = value_ranges_[index];
    TORCH_CHECK_VALUE(
        values[index] >= lower && values[index] <= upper,
        "Numeric result ", index, " is outside its scalar ABI or grid range");
  }
  for (const auto& node : nodes_) {
    if (!node.block) {
      continue;
    }
    int64_t threads = 1;
    for (auto index : *node.block) {
      TORCH_CHECK_VALUE(values[index] <= node.max_block_threads / threads,
                       "Kernel block exceeds its maximum thread count");
      threads *= values[index];
    }
  }
}

} // namespace at::cuda::detail

namespace at::cuda {

detail::CapturedGraphSnapshot CUDAGraph::inspect_captured_kernel_nodes(
    const std::vector<uintptr_t>& nodes) {
  TORCH_CHECK(keep_graph_, "Kernel node inspection requires keep_graph=True");
  TORCH_CHECK(capture_ended_ && has_graph_, "Kernel node inspection requires a completed retained graph capture");
#if defined(USE_ROCM) || !defined(CUDA_VERSION) || CUDA_VERSION < 12040
  TORCH_CHECK(false, "CUDA graph inspection requires NVIDIA CUDA 12.4 or later");
#else
  std::unordered_set<uintptr_t> seen;
  for (auto node : nodes) {
    TORCH_CHECK_VALUE(seen.insert(node).second, "Duplicate kernel node handle ", node);
  }
  c10::cuda::CUDAGuard device_guard(capture_dev_);
  size_t count = 0;
  C10_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &count));
  std::vector<cudaGraphNode_t> graph_nodes(count);
  if (count) {
    C10_CUDA_CHECK(cudaGraphGetNodes(graph_, graph_nodes.data(), &count));
    graph_nodes.resize(count);
  }
  detail::CapturedGraphSnapshot result{reinterpret_cast<uintptr_t>(graph_), capture_id_, {}, {}};
  result.nodes.reserve(graph_nodes.size());
  for (auto node : graph_nodes) {
    result.nodes.push_back(reinterpret_cast<uintptr_t>(node));
  }
  std::sort(result.nodes.begin(), result.nodes.end());
  result.kernels.reserve(nodes.size());
  for (auto handle : nodes) {
    auto node = reinterpret_cast<cudaGraphNode_t>(handle);
    TORCH_CHECK_VALUE(
        std::find(graph_nodes.begin(), graph_nodes.end(), node) != graph_nodes.end(),
        "Kernel node ", handle, " does not belong to this graph");
    cudaGraphNodeType type;
    C10_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
    TORCH_CHECK_VALUE(type == cudaGraphNodeTypeKernel, "Node ", handle, " is not a kernel node");
    result.kernels.push_back(detail::KernelNodeParams(handle).snapshot());
  }
  return result;
#endif
}

void CUDAGraph::clear_kernel_params() {
  if (kernel_params_) {
    kernel_params_->nodes.clear();
    ++kernel_params_->generation;
    kernel_params_->pointer_replay_failed = false;
  }
}

std::shared_ptr<detail::KernelParamUpdateBatch> CUDAGraph::prepare_kernel_params(
    std::vector<detail::KernelNodeUpdate> updates) {
  check_not_owned();
  TORCH_CHECK(keep_graph_, "update_kernel_params requires keep_graph=True");
  TORCH_CHECK(capture_ended_ && has_graph_, "Kernel parameter updates require a completed retained graph capture");
#if defined(USE_ROCM) || !defined(CUDA_VERSION) || CUDA_VERSION < 12040
  TORCH_CHECK(false, "Kernel parameter updates require NVIDIA CUDA 12.4 or later");
#endif
  std::unordered_set<uintptr_t> seen;
  for (const auto& update : updates) {
    TORCH_CHECK_VALUE(
        seen.insert(update.node).second,
        "Duplicate kernel node handle ", update.node);
  }
  c10::cuda::CUDAGuard device_guard(capture_dev_);
  if (!kernel_params_) {
    kernel_params_ = std::make_shared<detail::KernelParamCache>();
  }
  auto batch = std::make_shared<detail::KernelParamUpdateBatch>();
  batch->cache_ = kernel_params_;
  batch->generation_ = kernel_params_->generation;
  batch->needs_instantiation_ = !has_graph_exec_;
  batch->updates_ = std::move(updates);
  std::vector<cudaGraphNode_t> graph_nodes;
  const bool has_new_nodes = std::any_of(batch->updates_.begin(), batch->updates_.end(), [&](const auto& update) {
    return kernel_params_->nodes.count(update.node) == 0;
  });
  if (has_new_nodes) {
    size_t count = 0;
    C10_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &count));
    graph_nodes.resize(count);
    if (count) {
      C10_CUDA_CHECK(cudaGraphGetNodes(graph_, graph_nodes.data(), &count));
      graph_nodes.resize(count);
    }
  }
  batch->nodes_.reserve(batch->updates_.size());
  for (const auto& update : batch->updates_) {
    auto found = kernel_params_->nodes.find(update.node);
    std::shared_ptr<detail::KernelNodeParams> state;
    if (found != kernel_params_->nodes.end()) {
      state = found->second;
    } else {
      auto node = reinterpret_cast<cudaGraphNode_t>(update.node);
      TORCH_CHECK_VALUE(
          std::find(graph_nodes.begin(), graph_nodes.end(), node) != graph_nodes.end(),
          "Kernel node ", update.node, " does not belong to this graph");
      cudaGraphNodeType type;
      C10_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
      TORCH_CHECK_VALUE(type == cudaGraphNodeTypeKernel, "Node ", update.node, " is not a kernel node");
      state = std::make_shared<detail::KernelNodeParams>(update.node);
    }
    state->validate(update.arguments);
    batch->nodes_.push_back(std::move(state));
  }
  return batch;
}

void CUDAGraph::update_kernel_params(const detail::KernelParamUpdateBatch& batch) {
  check_not_owned();
  TORCH_CHECK(
      kernel_params_ && batch.cache_ == kernel_params_ && has_graph_exec_ && capture_ended_,
      "Prepared kernel parameter updates belong to a reset or different graph");
  TORCH_CHECK(
      batch.generation_ == kernel_params_->generation ||
          (batch.needs_instantiation_ && batch.generation_ + 1 == kernel_params_->generation),
      "Prepared kernel parameter updates were invalidated by instantiate");
  c10::cuda::CUDAGuard device_guard(capture_dev_);
  jit::initializeCudaContext();
  std::vector<std::shared_ptr<detail::KernelNodeParams>> nodes;
  nodes.reserve(batch.updates_.size());
  for (size_t index = 0; index < batch.updates_.size(); ++index) {
    const auto& update = batch.updates_[index];
    auto found = kernel_params_->nodes.find(update.node);
    // An instantiate hook may have updated an otherwise untouched argument.
    auto state = found == kernel_params_->nodes.end() ? batch.nodes_[index] : found->second;
    state->validate(update.arguments);
    nodes.push_back(std::move(state));
  }
  for (size_t index = 0; index < batch.updates_.size(); ++index) {
    kernel_params_->nodes.try_emplace(batch.updates_[index].node, nodes[index]);
  }
  try {
    for (size_t index = 0; index < batch.updates_.size(); ++index) {
      nodes[index]->update(reinterpret_cast<uintptr_t>(graph_exec_), batch.updates_[index].arguments);
    }
  } catch (...) {
    kernel_params_->pointer_replay_failed = true;
    throw;
  }
}

std::shared_ptr<detail::KernelPointerUpdateBatch> CUDAGraph::prepare_kernel_pointer_updates(
    std::vector<detail::KernelPointerBinding> bindings, size_t pointer_count) {
  for (const auto& binding : bindings) {
    TORCH_CHECK_VALUE(!binding.byte_offset, "Pointer-only bindings require whole kernel arguments");
    TORCH_CHECK_VALUE(!binding.address_offset_value_index, "Pointer address offsets require a numeric replay plan");
  }
  auto batch = prepare_kernel_replay_updates(std::move(bindings), pointer_count, {}, {}, 0, {}, {}, {}, {}, {}, {});
  batch->numeric_ = false;
  return batch;
}

std::shared_ptr<detail::KernelPointerUpdateBatch> CUDAGraph::prepare_kernel_replay_updates(
    std::vector<detail::KernelPointerBinding> bindings,
    size_t pointer_count,
    std::vector<detail::KernelScalarBinding> scalars,
    std::vector<detail::KernelGridBinding> grids,
    size_t value_count,
    std::vector<detail::KernelTensorMapBinding> tensor_maps,
    std::vector<detail::KernelMemsetBinding> memsets,
    std::vector<detail::KernelHostTableBinding> host_tables,
    std::vector<detail::KernelMemcpyBinding> memcpys,
    std::vector<detail::KernelRngBinding> rngs,
    std::vector<detail::KernelTemplateBinding> templates) {
  check_not_owned();
  TORCH_CHECK(has_graph_exec_, "Pointer updates require an instantiated graph");
  TORCH_CHECK(
      !kernel_params_ || !kernel_params_->pointer_replay_failed,
      "Pointer replay failed; re-instantiate the graph before preparing again");
  std::unordered_map<uintptr_t, size_t> positions;
  std::vector<detail::KernelNodeUpdate> requests;
  for (const auto& binding : bindings) {
    TORCH_CHECK_INDEX(binding.pointer_index < pointer_count, "Pointer index is out of range");
    if (binding.address_offset_value_index) {
      TORCH_CHECK_INDEX(*binding.address_offset_value_index < value_count, "Pointer address offset value index is out of range");
    }
    if (positions.emplace(binding.node, requests.size()).second) {
      requests.push_back({binding.node, {}});
    }
  }
  for (const auto& binding : scalars) {
    TORCH_CHECK_VALUE(
        binding.width == 1 || binding.width == 2 || binding.width == 4 || binding.width == 8,
        "Scalar width must be 1, 2, 4 or 8 bytes");
    TORCH_CHECK_INDEX(binding.value_index < value_count, "Scalar value index is out of range");
    if (positions.emplace(binding.node, requests.size()).second) {
      requests.push_back({binding.node, {}});
    }
  }
  std::unordered_set<uintptr_t> grid_nodes;
  for (const auto& binding : tensor_maps) {
    detail::validate_tensor_map_binding(binding, pointer_count, value_count);
    if (positions.emplace(binding.node, requests.size()).second) {
      requests.push_back({binding.node, {}});
    }
  }
  for (const auto& binding : grids) {
    TORCH_CHECK_VALUE(grid_nodes.insert(binding.node).second, "Duplicate kernel grid binding");
    for (auto value : binding.values) {
      TORCH_CHECK_INDEX(value < value_count, "Grid value index is out of range");
    }
    if (binding.shared_memory) {
      TORCH_CHECK_INDEX(*binding.shared_memory < value_count, "Shared-memory value index is out of range");
    }
    if (binding.block) {
      for (auto value : *binding.block) {
        TORCH_CHECK_INDEX(value < value_count, "Block value index is out of range");
      }
    }
    if (positions.emplace(binding.node, requests.size()).second) {
      requests.push_back({binding.node, {}});
    }
  }
  auto decoded = prepare_kernel_params(std::move(requests));
  auto batch = std::make_shared<detail::KernelPointerUpdateBatch>();
  batch->cache_ = decoded->cache_;
  batch->generation_ = decoded->generation_;
  batch->pointer_count_ = pointer_count;
  batch->numeric_ = true;
  batch->value_ranges_.assign(
      value_count, {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()});
  for (auto& node : decoded->nodes_) {
    batch->nodes_.push_back({node, {}, {}, std::nullopt});
  }
  std::vector<std::vector<std::pair<size_t, size_t>>> spans(batch->nodes_.size());
  for (const auto& binding : bindings) {
    auto position = positions.at(binding.node);
    auto& node = batch->nodes_[position];
    auto offset = node.params->pointer_offset(binding.argument, binding.byte_offset);
    node.slots.push_back({offset, binding.pointer_index, binding.address_offset_value_index});
    spans[position].emplace_back(offset, offset + sizeof(uintptr_t));
  }
  for (const auto& binding : scalars) {
    auto position = positions.at(binding.node);
    auto& node = batch->nodes_[position];
    auto offset = node.params->scalar_offset(binding.argument, binding.width, binding.byte_offset);
    node.scalars.push_back({offset, binding.width, binding.value_index});
    spans[position].emplace_back(offset, offset + binding.width);
    if (binding.width == 4) {
      auto& range = batch->value_ranges_[binding.value_index];
      range.first = std::max(range.first, static_cast<int64_t>(std::numeric_limits<int32_t>::min()));
      range.second = std::min(range.second, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
    }
  }
  for (auto& binding : tensor_maps) {
    auto position = positions.at(binding.node);
    auto& node = batch->nodes_[position];
    constexpr size_t tensor_map_size = 128;
    auto offset = node.params->scalar_offset(binding.argument, tensor_map_size, std::nullopt);
    spans[position].emplace_back(offset, offset + tensor_map_size);
    for (auto value : binding.dimensions) {
      auto& range = batch->value_ranges_[value];
      range.first = std::max(range.first, int64_t{1});
      range.second = std::min(range.second, int64_t{1} << 32);
    }
    for (auto value : binding.strides) {
      auto& range = batch->value_ranges_[value];
      range.first = std::max(range.first, int64_t{0});
      range.second = std::min(range.second, (int64_t{1} << 40) - 1);
    }
    node.tensor_maps.push_back({offset, std::move(binding)});
  }
  for (auto& node : batch->nodes_) {
    if (!node.tensor_maps.empty()) {
      node.params->reserve_tensor_maps(node.tensor_maps.size());
    }
  }
  for (auto& fields : spans) {
    std::sort(fields.begin(), fields.end());
    for (size_t index = 1; index < fields.size(); ++index) {
      TORCH_CHECK_VALUE(fields[index - 1].second <= fields[index].first, "Kernel argument bindings overlap");
    }
  }
  if (!grids.empty()) {
    const auto* properties = at::cuda::getDeviceProperties(capture_dev_);
    for (const auto& binding : grids) {
      auto& node = batch->nodes_[positions.at(binding.node)];
      node.grid = binding.values;
      node.shared_memory = binding.shared_memory;
      node.block = binding.block;
      for (size_t axis = 0; axis < 3; ++axis) {
        auto& range = batch->value_ranges_[binding.values[axis]];
        range.first = std::max(range.first, int64_t{1});
        range.second = std::min(range.second, static_cast<int64_t>(properties->maxGridSize[axis]));
      }
      if (binding.shared_memory) {
        auto& range = batch->value_ranges_[*binding.shared_memory];
        range.first = std::max(range.first, int64_t{0});
        range.second = std::min(range.second, static_cast<int64_t>(std::numeric_limits<unsigned int>::max()));
      }
      if (binding.block) {
        node.max_block_threads = std::min(properties->maxThreadsPerBlock,
                                         node.params->max_threads_per_block(capture_dev_));
        for (size_t axis = 0; axis < 3; ++axis) {
          auto& range = batch->value_ranges_[(*binding.block)[axis]];
          range.first = std::max(range.first, int64_t{1});
          range.second = std::min(range.second, static_cast<int64_t>(properties->maxThreadsDim[axis]));
        }
      }
    }
  }
  if (!memsets.empty()) {
    c10::cuda::CUDAGuard device_guard(capture_dev_);
    size_t count = 0;
    C10_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &count));
    std::vector<cudaGraphNode_t> graph_nodes(count);
    if (count) {
      C10_CUDA_CHECK(cudaGraphGetNodes(graph_, graph_nodes.data(), &count));
      graph_nodes.resize(count);
    }
    std::unordered_set<uintptr_t> seen;
    for (const auto& binding : memsets) {
      TORCH_CHECK_VALUE(seen.insert(binding.node).second, "Duplicate memset binding");
      TORCH_CHECK_INDEX(binding.pointer_index < pointer_count, "Memset pointer index is out of range");
      if (binding.address_offset_value_index) {
        TORCH_CHECK_INDEX(*binding.address_offset_value_index < value_count, "Memset offset value index is out of range");
      }
      TORCH_CHECK_INDEX(binding.bytes_value_index < value_count, "Memset byte count value index is out of range");
      const auto node = reinterpret_cast<cudaGraphNode_t>(binding.node);
      TORCH_CHECK_VALUE(std::find(graph_nodes.begin(), graph_nodes.end(), node) != graph_nodes.end(),
                       "Memset node does not belong to this graph");
      cudaGraphNodeType type;
      C10_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
      TORCH_CHECK_VALUE(type == cudaGraphNodeTypeMemset, "Memset binding must name a memset node");
      cudaMemsetParams params{};
      C10_CUDA_CHECK(cudaGraphMemsetNodeGetParams(node, &params));
      TORCH_CHECK_VALUE(params.height == 1 && params.elementSize == 1 && params.width > 0,
                       "Memset bindings require a nonempty one-dimensional byte memset");
      batch->memsets_.push_back(
          {binding, params.value, params.pitch, reinterpret_cast<uintptr_t>(params.dst), params.width});
      auto& range = batch->value_ranges_[binding.bytes_value_index];
      range.first = std::max(range.first, int64_t{1});
    }
  }
  for (const auto& binding : host_tables) {
    TORCH_CHECK_VALUE(!binding.slots.empty() && binding.nbytes > 0, "A host table binding needs pinned slots and a byte count");
    auto table = std::make_unique<detail::KernelPointerUpdateBatch::HostTable>();
    table->slots = binding.slots;
    table->nbytes = binding.nbytes;
    table->elements = binding.elements;
    for (const auto& element : table->elements) {
      TORCH_CHECK_VALUE(
          element.width == 1 || element.width == 2 || element.width == 4 || element.width == 8,
          "A host table element is 1, 2, 4 or 8 bytes wide");
      TORCH_CHECK_VALUE(element.offset + element.width <= table->nbytes, "A host table element lies outside the table");
      if (element.pointer) {
        TORCH_CHECK_INDEX(element.index < pointer_count, "Host table pointer index is out of range");
        if (element.address_offset_value_index) {
          TORCH_CHECK_INDEX(*element.address_offset_value_index < value_count, "Host table address offset value index is out of range");
        }
      } else {
        TORCH_CHECK_INDEX(element.index < value_count, "Host table value index is out of range");
      }
    }
    table->events.resize(table->slots.size(), nullptr);
    table->recorded.assign(table->slots.size(), false);
    for (auto& event : table->events) {
      cudaEvent_t created = nullptr;
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&created, cudaEventDisableTiming));
      event = created;
    }
    batch->host_tables_.push_back(std::move(table));
  }
  for (const auto& binding : memcpys) {
    if (binding.source_table) {
      TORCH_CHECK_INDEX(*binding.source_table < batch->host_tables_.size(), "Memcpy source table index is out of range");
    } else {
      TORCH_CHECK_INDEX(binding.source_pointer_index < pointer_count, "Memcpy source pointer index is out of range");
      if (binding.source_offset_value_index) {
        TORCH_CHECK_INDEX(*binding.source_offset_value_index < value_count, "Memcpy source offset value index is out of range");
      }
    }
    TORCH_CHECK_INDEX(binding.pointer_index < pointer_count, "Memcpy pointer index is out of range");
    if (binding.address_offset_value_index) {
      TORCH_CHECK_INDEX(*binding.address_offset_value_index < value_count, "Memcpy address offset value index is out of range");
    }
    TORCH_CHECK_INDEX(binding.bytes_value_index < value_count, "Memcpy byte count value index is out of range");
    TORCH_CHECK_VALUE(positions.find(binding.node) == positions.end(), "A memcpy binding names a kernel node");
    cudaMemcpy3DParms params{};
    C10_CUDA_CHECK(cudaGraphMemcpyNodeGetParams(reinterpret_cast<cudaGraphNode_t>(binding.node), &params));
    TORCH_CHECK_VALUE(
        params.extent.height == 1 && params.extent.depth == 1 && params.srcArray == nullptr && params.dstArray == nullptr &&
            (params.kind == cudaMemcpyHostToDevice || params.kind == cudaMemcpyDeviceToDevice ||
             params.kind == cudaMemcpyDefault),
        "Memcpy bindings support one-dimensional host-to-device and device-to-device copies");
    batch->memcpys_.push_back(
        {binding.node, binding.source_table, binding.source_pointer_index, binding.source_offset_value_index,
         binding.pointer_index, binding.address_offset_value_index, binding.bytes_value_index,
         reinterpret_cast<uintptr_t>(params.srcPtr.ptr), reinterpret_cast<uintptr_t>(params.dstPtr.ptr),
         params.extent.width, params.kind});
    auto& range = batch->value_ranges_[binding.bytes_value_index];
    range.first = std::max(range.first, int64_t{1});
  }
  for (const auto& binding : rngs) {
    TORCH_CHECK_INDEX(binding.value_index < value_count, "Generator increment value index is out of range");
    auto& range = batch->value_ranges_[binding.value_index];
    range.first = std::max(range.first, int64_t{0});
    batch->rngs_.push_back(binding);
  }
  for (const auto& binding : templates) {
    for (auto node : binding.nodes) {
      TORCH_CHECK_VALUE(positions.find(node) == positions.end(), "A template binding names a bound kernel node");
    }
  }
  batch->prepare_templates(std::move(templates), pointer_count, value_count);
  for (const auto& [node, position] : positions) {
    kernel_params_->nodes.try_emplace(node, batch->nodes_[position].params);
  }
  return batch;
}

detail::KernelPointerUpdateBatch::HostTable::~HostTable() {
  for (auto* event : events) {
    if (event != nullptr) {
      // a slot's last reader may still be queued: destruction is deferred by
      // the runtime until the recorded work completes
      C10_CUDA_CHECK_WARN(cudaEventDestroy(static_cast<cudaEvent_t>(event)));
    }
  }
}

void CUDAGraph::replay_kernel_pointer_updates(
    const detail::KernelPointerUpdateBatch& batch,
    c10::ArrayRef<uintptr_t> pointers) {
  check_not_owned();
  replay_pointer_updates_impl(batch, pointers);
}

void CUDAGraph::validate_pointer_updates(
    const detail::KernelPointerUpdateBatch& batch, size_t pointer_count) const {
  TORCH_CHECK(
      kernel_params_ && batch.cache_ == kernel_params_ && has_graph_exec_ && capture_ended_,
      "Prepared pointer updates belong to a reset or different graph");
  TORCH_CHECK(batch.generation_ == kernel_params_->generation, "Prepared pointer updates were invalidated by instantiate");
  TORCH_CHECK(!kernel_params_->pointer_replay_failed, "Pointer replay failed; re-instantiate the graph before replaying again");
  TORCH_CHECK_VALUE(pointer_count == batch.pointer_count_, "Pointer count differs from the prepared bindings");
}

void CUDAGraph::replay_pointer_updates_impl(
    const detail::KernelPointerUpdateBatch& batch,
    c10::ArrayRef<uintptr_t> pointers,
    c10::cuda::CUDACachingAllocator::PendingGraphInputs* pending) {
  TORCH_CHECK_VALUE(!batch.numeric_, "Numeric updates require a boxed numeric plan");
  validate_pointer_updates(batch, pointers.size());
  c10::cuda::CUDAGuard device_guard(capture_dev_);
  jit::initializeCudaContext();
  try {
    for (const auto& node : batch.nodes_) {
      node.params->update_pointers(reinterpret_cast<uintptr_t>(graph_exec_), node.slots, pointers);
    }
    replay_impl(pending);
  } catch (...) {
    kernel_params_->pointer_replay_failed = true;
    throw;
  }
}

void CUDAGraph::replay_kernel_updates_impl(
    const detail::KernelPointerUpdateBatch& batch,
    c10::ArrayRef<uintptr_t> pointers,
    c10::ArrayRef<int64_t> values,
    c10::cuda::CUDACachingAllocator::PendingGraphInputs* pending) {
  validate_pointer_updates(batch, pointers.size());
  for (auto& memset : batch.memsets_) {
    const auto& binding = memset.binding;
    const int64_t offset = binding.address_offset_value_index ? values[*binding.address_offset_value_index] : 0;
    memset.pending_dst = detail::displaced_pointer(pointers[binding.pointer_index], offset);
    memset.pending_width = static_cast<size_t>(values[binding.bytes_value_index]);
    TORCH_CHECK_VALUE(memset.pending_width <= std::numeric_limits<uintptr_t>::max() - memset.pending_dst,
                     "Memset address range overflowed");
  }
  c10::cuda::CUDAGuard device_guard(capture_dev_);
  jit::initializeCudaContext();
  const bool timers = detail::hotpath_timers_on();
  static const bool probe_skip = detail::hotpath_probe("skip_unchanged");
  int64_t t0 = timers ? detail::hotpath_now() : 0;
  int64_t t_update = 0;
  int64_t t_templates = 0;
  try {
    bool unchanged = false;
    if (probe_skip) {
      unchanged = batch.probe_last_valid_ && pointers.equals(batch.probe_last_pointers_) &&
          values.equals(batch.probe_last_values_);
      batch.probe_last_valid_ = false;
    }
    if (!unchanged) {
      for (const auto& node : batch.nodes_) {
        node.params->update_replay(
            reinterpret_cast<uintptr_t>(graph_exec_), node.slots, node.scalars, node.tensor_maps,
            node.grid, node.shared_memory, pointers, values, node.block);
      }
    }
    if (probe_skip) {
      batch.probe_last_pointers_.assign(pointers.begin(), pointers.end());
      batch.probe_last_values_.assign(values.begin(), values.end());
      batch.probe_last_valid_ = true;
    }
    if (timers) {
      const int64_t t1 = detail::hotpath_now();
      t_update = t1 - t0;
      t0 = t1;
    }
    for (auto& table : batch.host_tables_) {
      // render into the next ring slot once the launch that last read it is done
      const size_t slot = table->pos % table->slots.size();
      if (table->recorded[slot]) {
        C10_CUDA_CHECK(cudaEventSynchronize(static_cast<cudaEvent_t>(table->events[slot])));
        table->recorded[slot] = false;
      }
      auto* bytes = reinterpret_cast<uint8_t*>(table->slots[slot]);
      for (const auto& element : table->elements) {
        int64_t value = 0;
        if (element.pointer) {
          const int64_t offset = element.address_offset_value_index ? values[*element.address_offset_value_index] : 0;
          value = static_cast<int64_t>(pointers[element.index]) + offset;
        } else {
          value = values[element.index];
        }
        std::memcpy(bytes + element.offset, &value, element.width);
      }
      table->used = slot;
      ++table->pos;
    }
    for (auto& memcpy : batch.memcpys_) {
      uintptr_t src = 0;
      if (memcpy.source_table) {
        const auto& table = *batch.host_tables_[*memcpy.source_table];
        src = table.slots[table.used];
      } else {
        const int64_t offset = memcpy.source_offset_value_index ? values[*memcpy.source_offset_value_index] : 0;
        src = static_cast<uintptr_t>(static_cast<int64_t>(pointers[memcpy.source_pointer_index]) + offset);
      }
      const int64_t offset = memcpy.address_offset_value_index ? values[*memcpy.address_offset_value_index] : 0;
      const auto dst = static_cast<uintptr_t>(static_cast<int64_t>(pointers[memcpy.pointer_index]) + offset);
      const auto width = static_cast<size_t>(values[memcpy.bytes_value_index]);
      if (src == memcpy.src && dst == memcpy.dst && width == memcpy.width) {
        continue;
      }
      // the captured copy is linear: pitch and logical width are the byte count
      cudaMemcpy3DParms params{};
      params.srcPtr = make_cudaPitchedPtr(reinterpret_cast<void*>(src), width, width, 1);
      params.dstPtr = make_cudaPitchedPtr(reinterpret_cast<void*>(dst), width, width, 1);
      params.extent = make_cudaExtent(width, 1, 1);
      params.kind = memcpy.kind;
      C10_CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams(graph_exec_, reinterpret_cast<cudaGraphNode_t>(memcpy.node), &params));
      if (src != memcpy.src) {
        ++batch.source_rebinds_;
      }
      ++batch.memcpy_updates_;
      memcpy.src = src;
      memcpy.dst = dst;
      memcpy.width = width;
    }
    for (auto& memset : batch.memsets_) {
      if (memset.pending_dst == memset.dst && memset.pending_width == memset.width) {
        continue;
      }
      cudaMemsetParams params{};
      params.dst = reinterpret_cast<void*>(memset.pending_dst);
      params.pitch = memset.pitch;
      params.value = memset.value;
      params.elementSize = 1;
      params.width = memset.pending_width;
      params.height = 1;
      C10_CUDA_CHECK(cudaGraphExecMemsetNodeSetParams(
          graph_exec_, reinterpret_cast<cudaGraphNode_t>(memset.binding.node), &params));
      memset.dst = memset.pending_dst;
      memset.width = memset.pending_width;
    }
    for (const auto& rng : batch.rngs_) {
      set_generator_increment(rng.generator, static_cast<uint64_t>(values[rng.value_index]));
    }
    int64_t t_before_templates = timers ? detail::hotpath_now() : 0;
    if (batch.replay_templates(reinterpret_cast<uintptr_t>(graph_exec_), pointers, values)) {
      batch.update_exec_from_graph(reinterpret_cast<uintptr_t>(graph_exec_), reinterpret_cast<uintptr_t>(graph_));
    }
    if (timers) {
      const int64_t t2 = detail::hotpath_now();
      t_templates = t2 - t_before_templates;
      auto& t = detail::tls_hotpath_timers;
      ++t.calls;
      t.update_ns += t_update;
      t.templates_ns += t_templates;
      t.rest_ns += (t_before_templates - t0);
    }
    replay_impl(pending);
    if (!batch.host_tables_.empty()) {
      // the copies of this launch read their slots until this event
      const auto stream = at::cuda::getCurrentCUDAStream();
      for (auto& table : batch.host_tables_) {
        C10_CUDA_CHECK(cudaEventRecord(static_cast<cudaEvent_t>(table->events[table->used]), stream));
        table->recorded[table->used] = true;
      }
    }
  } catch (...) {
    kernel_params_->pointer_replay_failed = true;
    throw;
  }
}

detail::GraphReplayLease::GraphReplayLease(
    std::shared_ptr<CUDAGraph> graph,
    std::shared_ptr<KernelPointerUpdateBatch> batch,
    c10::DeviceIndex device)
    : graph_(std::move(graph)), batch_(std::move(batch)) {
  TORCH_CHECK(graph_ && batch_, "Native replay requires a graph and pointer batch");
  graph_->validate_pointer_updates(*batch_, batch_->pointer_count_);
  TORCH_CHECK(graph_->capture_dev_ == device, "Native replay stream is on a different device");
  const GraphReplayLease* empty = nullptr;
  TORCH_CHECK(
      graph_->replay_owner_.compare_exchange_strong(empty, this),
      "CUDA graph is already leased to a native replay owner");
}

detail::GraphReplayLease::~GraphReplayLease() {
  graph_->replay_owner_.store(nullptr);
}

void detail::GraphReplayLease::replay(
    c10::ArrayRef<uintptr_t> pointers, c10::cuda::CUDACachingAllocator::PendingGraphInputs* pending) {
  graph_->replay_pointer_updates_impl(*batch_, pointers, pending);
}

detail::InputReleasePlan detail::GraphReplayLease::prepare_input_release(
    std::vector<InputReleaseStep> steps, const std::vector<uintptr_t>& nodes,
    size_t input_count, size_t allocation_count,
    c10::ArrayRef<KernelRootUse> parameter_root_uses) const {
#if defined(USE_ROCM) || !defined(CUDA_VERSION) || CUDA_VERSION < 12040
  TORCH_CHECK(false, "Input release requires NVIDIA CUDA 12.4 or later");
#else
  graph_->validate_pointer_updates(*batch_, input_count + allocation_count);
  TORCH_CHECK_VALUE(!nodes.empty() && allocation_count &&
                       batch_->nodes_.size() == nodes.size(),
                   "Input release requires complete kernel bindings");
  auto snapshot = graph_->inspect_captured_kernel_nodes(nodes);
  TORCH_CHECK_VALUE(snapshot.nodes.size() == nodes.size(), "Release nodes must cover the complete graph");
  std::unordered_map<uintptr_t, size_t> ordinals;
  for (size_t index = 0; index < nodes.size(); ++index) {
    TORCH_CHECK_VALUE(nodes[index] && ordinals.emplace(nodes[index], index).second,
                     "Release nodes must be distinct kernel handles");
  }
  c10::cuda::CUDAGuard device_guard(graph_->capture_dev_);
  size_t edge_count = 0;
#if CUDA_VERSION >= 13000
  C10_CUDA_CHECK(cudaGraphGetEdges(graph_->graph_, nullptr, nullptr, nullptr, &edge_count));
#else
  C10_CUDA_CHECK(cudaGraphGetEdges_v2(graph_->graph_, nullptr, nullptr, nullptr, &edge_count));
#endif
  TORCH_CHECK_VALUE(edge_count == nodes.size() - 1, "Input release requires a complete linear kernel chain");
  std::vector<cudaGraphNode_t> from(edge_count), to(edge_count);
  std::vector<cudaGraphEdgeData> data(edge_count);
  if (edge_count) {
    auto expected = edge_count;
#if CUDA_VERSION >= 13000
    C10_CUDA_CHECK(cudaGraphGetEdges(graph_->graph_, from.data(), to.data(), data.data(), &edge_count));
#else
    C10_CUDA_CHECK(cudaGraphGetEdges_v2(graph_->graph_, from.data(), to.data(), data.data(), &edge_count));
#endif
    TORCH_CHECK_VALUE(edge_count == expected, "Release graph edges changed during preparation");
  }
  std::vector<bool> edges(nodes.size() - 1, false);
  for (size_t index = 0; index < edge_count; ++index) {
    const auto& edge = data[index];
    TORCH_CHECK_VALUE(edge.from_port == 0 && edge.to_port == 0 && edge.type == 0 &&
                         std::all_of(std::begin(edge.reserved), std::end(edge.reserved),
                                     [](auto value) { return value == 0; }),
                     "Input release requires full-completion graph edges");
    auto source = ordinals.find(reinterpret_cast<uintptr_t>(from[index]));
    auto target = ordinals.find(reinterpret_cast<uintptr_t>(to[index]));
    TORCH_CHECK_VALUE(source != ordinals.end() && target != ordinals.end() &&
                         source->second + 1 == target->second && !edges[source->second],
                     "Release call order differs from the captured kernel chain");
    edges[source->second] = true;
  }
  std::vector<std::vector<size_t>> uses(input_count + allocation_count);
  std::vector<bool> bound(nodes.size(), false);
  for (const auto& node : batch_->nodes_) {
    auto found = ordinals.find(node.params->snapshot().node);
    TORCH_CHECK_VALUE(found != ordinals.end() && !bound[found->second],
                     "Release nodes differ from the prepared pointer batch");
    auto ordinal = found->second;
    bound[ordinal] = true;
    for (const auto& slot : node.slots) {
      uses[slot.pointer_index].push_back(ordinal);
    }
    for (const auto& slot : node.tensor_maps) {
      uses[slot.binding.pointer_index].push_back(ordinal);
    }
  }
  for (const auto& use : parameter_root_uses) {
    auto found = ordinals.find(use.node);
    TORCH_CHECK_VALUE(found != ordinals.end() && use.pointer_index < uses.size(),
                     "Parameter root use differs from the prepared release graph");
    uses[use.pointer_index].push_back(found->second);
  }
  for (auto& row : uses) {
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
  }
  const auto absent = std::numeric_limits<size_t>::max();
  std::vector<size_t> allocations(allocation_count, absent), drops(input_count, absent);
  size_t next_allocation = 0, next_kernel = 0;
  InputReleasePlan result;
  for (size_t position = 0; position < steps.size(); ++position) {
    const auto& step = steps[position];
    if (step.kind == InputReleaseKind::Allocate) {
      TORCH_CHECK_VALUE(step.index == next_allocation && step.index < allocation_count,
                       "Release allocations must appear once in original order");
      allocations[step.index] = position;
      ++next_allocation;
      result.steps.push_back(step);
    } else if (step.kind == InputReleaseKind::Kernel) {
      TORCH_CHECK_VALUE(step.index == next_kernel && next_kernel < nodes.size(),
                       "Release kernels must appear once in captured order");
      for (size_t source = 0; source < uses.size(); ++source) {
        if (std::binary_search(uses[source].begin(), uses[source].end(), next_kernel)) {
          TORCH_CHECK_VALUE(source < input_count ? drops[source] == absent
                                                : allocations[source - input_count] != absent,
                           "Release kernel uses a dropped input or unallocated buffer");
        }
      }
      ++next_kernel;
    } else {
      TORCH_CHECK_VALUE(step.kind == InputReleaseKind::Drop && step.index < input_count &&
                           drops[step.index] == absent &&
                           (uses[step.index].empty() || uses[step.index].back() < next_kernel),
                       "Input drop must be unique and follow its last captured use");
      drops[step.index] = position;
      result.steps.push_back(step);
    }
  }
  TORCH_CHECK_VALUE(next_allocation == allocation_count && next_kernel == nodes.size(),
                   "Release program must cover every allocation and kernel");
  for (size_t input = 0; input < input_count; ++input) {
    if (drops[input] != absent) {
      result.candidates.push_back(input);
    }
  }
  TORCH_CHECK_VALUE(!result.candidates.empty(), "Release program requires saved-input drops");
  result.eligible_inputs.resize(allocation_count);
  for (size_t allocation = 0; allocation < allocation_count; ++allocation) {
    const auto& new_uses = uses[input_count + allocation];
    TORCH_CHECK_VALUE(!new_uses.empty(), "Release allocation has no captured use");
    for (size_t candidate = 0; candidate < result.candidates.size(); ++candidate) {
      auto input = result.candidates[candidate];
      if (drops[input] < allocations[allocation] &&
          (uses[input].empty() || uses[input].back() < new_uses.front())) {
        result.eligible_inputs[allocation].push_back(candidate);
      }
    }
  }
  return result;
#endif
}

void detail::GraphReplayLease::validate_values(c10::ArrayRef<int64_t> values) const {
  batch_->validate_values(values);
}

void detail::GraphReplayLease::replay(
    c10::ArrayRef<uintptr_t> pointers, c10::ArrayRef<int64_t> values,
    c10::cuda::CUDACachingAllocator::PendingGraphInputs* pending) {
  graph_->replay_kernel_updates_impl(*batch_, pointers, values, pending);
}

bool detail::GraphReplayLease::check_capture_pool_retirement() const {
  graph_->validate_pointer_updates(*batch_, batch_->pointer_count_);
  return graph_->check_capture_pool_retirement();
}

void detail::GraphReplayLease::retire_capture_pool() {
  graph_->release_capture_pools();
}

void detail::GraphReplayLease::close() {
  if (graph_->has_graph_exec_) {
    C10_CUDA_CHECK(cudaGraphExecDestroy(graph_->graph_exec_));
    graph_->graph_exec_ = nullptr;
    graph_->has_graph_exec_ = false;
  }
  if (graph_->has_graph_) {
    C10_CUDA_CHECK(cudaGraphDestroy(graph_->graph_));
    graph_->graph_ = nullptr;
    graph_->has_graph_ = false;
  }
  graph_->reset_impl();
}

} // namespace at::cuda
