#pragma once

#include <ATen/core/Generator.h>
#include <c10/core/Device.h>
#include <c10/macros/Export.h>
#include <c10/util/ArrayRef.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace c10::cuda::CUDACachingAllocator {
class PendingGraphInputs;
}

namespace at::cuda {
struct CUDAGraph;

namespace detail {

struct CaptureFrontier {
  int status = 0;
  uint64_t capture_id = 0;
  uintptr_t graph = 0;
  std::vector<std::pair<uintptr_t, std::array<uint8_t, 8>>> dependencies;
};

TORCH_CUDA_CPP_API CaptureFrontier get_capture_frontier(uintptr_t stream);

struct KernelArgumentSnapshot {
  size_t offset;
  size_t size;
  std::vector<uint8_t> value;
};

struct KernelNodeSnapshot {
  uintptr_t node;
  uintptr_t function;
  uintptr_t kernel;
  uintptr_t context;
  std::array<unsigned int, 3> grid;
  std::array<unsigned int, 3> block;
  unsigned int shared_memory;
  bool packed;
  std::vector<KernelArgumentSnapshot> arguments;
};

struct CapturedGraphSnapshot {
  uintptr_t graph;
  uint64_t capture_id;
  std::vector<uintptr_t> nodes;
  std::vector<KernelNodeSnapshot> kernels;
};

struct KernelArgumentUpdate {
  int64_t index;
  std::vector<uint8_t> value;
};

struct KernelNodeUpdate {
  uintptr_t node;
  std::vector<KernelArgumentUpdate> arguments;
};

// The caller declares pointer slots; width alone does not establish their type.
struct KernelPointerBinding {
  uintptr_t node;
  int64_t argument;
  size_t pointer_index;
  // An absent offset requires a whole argument of the binding's width.
  std::optional<size_t> byte_offset = std::nullopt;
  std::optional<size_t> address_offset_value_index = std::nullopt;
};

struct KernelPointerSlot {
  size_t offset;
  size_t pointer_index;
  std::optional<size_t> address_offset_value_index = std::nullopt;
};

struct KernelScalarBinding {
  uintptr_t node;
  int64_t argument;
  size_t width;
  size_t value_index;
  std::optional<size_t> byte_offset = std::nullopt;
};

struct KernelGridBinding {
  uintptr_t node;
  std::array<size_t, 3> values;
  std::optional<size_t> shared_memory = std::nullopt;
  std::optional<std::array<size_t, 3>> block = std::nullopt;
};

// A one-dimensional byte memset node whose destination and byte count follow
// the replay's pointers and numeric values (a host's semaphore or scratch reset).
struct KernelMemsetBinding {
  uintptr_t node;
  size_t pointer_index;
  std::optional<size_t> address_offset_value_index = std::nullopt;
  size_t bytes_value_index = 0;
};

// One element of a host staging table the replay renders natively per call:
// a pointer (a replay pointer plus an optional address offset value) or a
// numeric value, written little-endian in `width` bytes at `offset`.
struct KernelHostTableElement {
  size_t offset;
  size_t width;
  bool pointer;
  size_t index; // pointer index when `pointer`, else value index
  std::optional<size_t> address_offset_value_index = std::nullopt;
};

// A pinned host table a converted host fills and copies to the device
// (host_trace's HostTable): `slots` are the caller's pinned ring buffers, all
// `nbytes` long; each call renders `elements` into the next slot once the
// event recorded after the launch that read it has completed.
struct KernelHostTableBinding {
  std::vector<uintptr_t> slots;
  size_t nbytes;
  std::vector<KernelHostTableElement> elements;
};

// A one-dimensional host-to-device memcpy node whose source (a host table's
// current slot or a replay pointer such as a pinned input), destination and
// byte count follow the replay's pointers and numeric values.
struct KernelMemcpyBinding {
  uintptr_t node;
  std::optional<size_t> source_table = std::nullopt;
  size_t source_pointer_index = 0;
  std::optional<size_t> source_offset_value_index = std::nullopt;
  size_t pointer_index = 0;
  std::optional<size_t> address_offset_value_index = std::nullopt;
  size_t bytes_value_index = 0;
};

// The philox offsets one replay draws from `generator`, as a numeric value:
// set on the graph before each replay (CUDAGraph::set_generator_increment).
struct KernelRngBinding {
  at::Generator generator;
  size_t value_index;
};

// The launch attributes a template node tracks: cluster dimension (x, y, z),
// cluster scheduling policy, cooperative, priority, memory synchronization
// domain and its map (default, remote). An attribute a node does not carry
// reads as zero.
constexpr size_t kKernelNodeAttributes = 9;
using KernelNodeAttributes = std::array<int64_t, kKernelNodeAttributes>;

// A closed library region (a cuBLAS GEMM) runs as kernel nodes whose function,
// launch configuration, argument image and attributes the library picks per
// shape, and as byte memset nodes (a split-K semaphore reset). A template
// variant holds one such choice with the operand address qwords zeroed:
// `slots` are the qwords that hold operand `operand`'s replay address plus
// `delta`, `workspace_slots` the ones that hold the binding's own scratch
// buffer.
struct KernelTemplateSlot {
  size_t offset;
  size_t operand;
  int64_t delta;
};

// A node of a variant: a kernel node or a memset node. Node i of a variant is
// node i of its site's chain: every variant of a site has the site's node kinds
// in order (a template with another chain is another site of another exec),
// so no node is ever disabled. A memset writes `memset_width` bytes of
// `memset_value` at operand `memset_operand`'s replay address plus
// `memset_delta`, or at the fixed address `memset_delta` when it names no
// operand.
struct KernelTemplateNode {
  enum class Kind : uint8_t { kernel, memset };
  Kind kind = Kind::kernel;
  uintptr_t function = 0;
  std::array<unsigned int, 3> grid{};
  std::array<unsigned int, 3> block{};
  unsigned int shared_memory = 0;
  std::vector<uint8_t> image;
  std::vector<KernelTemplateSlot> slots;
  std::vector<size_t> workspace_slots;
  KernelNodeAttributes attributes{};
  std::optional<size_t> memset_operand = std::nullopt;
  int64_t memset_delta = 0;
  unsigned int memset_element_size = 1;
  size_t memset_width = 0;
  unsigned int memset_value = 0;
};

struct KernelTemplateVariant {
  std::vector<KernelTemplateNode> nodes;
};

// The process-wide template registry. A site is one region of one prepared
// graph; its variants are registered under integer keys (the region's operand
// sizes, strides and address alignment classes; the library's global settings
// are appended to every key) and share the site's node chain: the first
// registration fixes it, a variant with another chain is refused. `select_kernel_template` takes (site, key...),
// returns the variant index or -1 and remembers both the selection and a
// missed key for the calling thread; `selected_kernel_template` takes (site)
// and returns that selection. Both have the numeric plan's call signature: a
// dispatch predicate selects, the plan reads the selection back.
// `select_kernel_template_key` is the same select over (site, key pointer, key
// length), what a compiled predicate calls per region per call: the calling
// thread keeps per site the last key, the settings epoch it was selected under
// (at::blasSettingsEpoch) and the index, so an unchanged key under an unchanged
// epoch is one compare; a changed key runs the lookup. A hit never goes stale
// (a site's variants are only ever added, a key's index never changes); a miss
// is not kept (a registration may follow it).
TORCH_CUDA_CPP_API int64_t new_kernel_template_site();
TORCH_CUDA_CPP_API int64_t register_kernel_template(
    int64_t site, std::vector<int64_t> key, KernelTemplateVariant variant);
TORCH_CUDA_CPP_API int64_t select_kernel_template(const std::vector<int64_t>& arguments);
TORCH_CUDA_CPP_API int64_t select_kernel_template_key(int64_t site, const int64_t* key, size_t length);
TORCH_CUDA_CPP_API int64_t selected_kernel_template(const std::vector<int64_t>& arguments);
// The same read over (pointer, length), the plan's pointer-ABI opaque call
// (`pcall`): arguments[0] is the site; no vector per call.
TORCH_CUDA_CPP_API int64_t selected_kernel_template_at(const int64_t* arguments, size_t length);
TORCH_CUDA_CPP_API std::optional<std::vector<int64_t>> take_kernel_template_miss(int64_t site);
// The node kinds of a site's chain (empty before its first registration).
TORCH_CUDA_CPP_API std::vector<KernelTemplateNode::Kind> kernel_template_site_kinds(int64_t site);
TORCH_CUDA_CPP_API std::vector<int64_t> kernel_template_library_settings();
// hotpath-2 measurement: TORCH_CUDAGRAPH_HOTPATH_TIMERS=1 accumulates per thread
// (calls, ns in the per-node dirty check, ns in the template replay, ns in the
// rest of the replay updates before the launch); read and reset here.
TORCH_CUDA_CPP_API std::array<int64_t, 4> hotpath_timers_take();

// The nodes of one template site in a captured graph, in launch order (the
// site's chain, which every variant has), whose per-call state is the variant
// `variant_value_index` selects: per kernel node the function, grid, block,
// shared memory, attributes and the whole argument image with the operand
// slots patched from the replay pointers, per memset node the destination,
// width and value.
struct KernelTemplateBinding {
  std::vector<uintptr_t> nodes;
  int64_t site;
  size_t variant_value_index;
  std::vector<size_t> operand_pointer_index;
  std::vector<std::optional<size_t>> operand_offset_value_index;
  uintptr_t workspace;
};

// The tracked attributes of a captured kernel node, as a template binding
// reads them.
TORCH_CUDA_CPP_API KernelNodeAttributes read_kernel_node_attributes(uintptr_t node);

// Launch `function` on `stream` with a packed argument image and the tracked
// attributes (a capture of a template node). `programmatic` launches with
// programmatic stream serialization, so the capture records the programmatic
// dependent launch edge the library's own launch of the node recorded (an
// edge of the captured graph; no node attribute carries it).
TORCH_CUDA_CPP_API void launch_kernel_image(
    uintptr_t function,
    std::array<unsigned int, 3> grid,
    std::array<unsigned int, 3> block,
    unsigned int shared_memory,
    uintptr_t stream,
    const std::vector<uint8_t>& image,
    const KernelNodeAttributes& attributes,
    bool programmatic = false);

struct KernelTensorMapBinding {
  uintptr_t node;
  int64_t argument;
  size_t pointer_index;
  size_t address_offset_value_index;
  // CUDA order: innermost dimension first; strides are already in bytes.
  std::vector<size_t> dimensions;
  std::vector<size_t> strides;
  std::vector<uint32_t> box_dimensions;
  uint32_t data_type;
  uint32_t swizzle;
  bool nan_fill;
};

struct KernelTensorMapSlot {
  size_t offset;
  KernelTensorMapBinding binding;
};

TORCH_CUDA_CPP_API std::array<uint8_t, 128> encode_tensor_map(
    const KernelTensorMapBinding& binding,
    c10::ArrayRef<int64_t> values,
    c10::ArrayRef<uintptr_t> pointers);

struct KernelScalarSlot {
  size_t offset;
  size_t width;
  size_t value_index;
};

struct KernelRootUse {
  uintptr_t node;
  size_t pointer_index;
};

// Owns argument bytes; it does not own the graph, kernel module or device data.
class TORCH_CUDA_CPP_API KernelNodeParams {
  friend struct at::cuda::CUDAGraph;

 public:
  explicit KernelNodeParams(uintptr_t node);
  ~KernelNodeParams();
  KernelNodeParams(const KernelNodeParams&) = delete;
  KernelNodeParams& operator=(const KernelNodeParams&) = delete;

  void validate(const std::vector<KernelArgumentUpdate>& updates) const;
  void update(uintptr_t graph_exec, const std::vector<KernelArgumentUpdate>& updates);
  KernelNodeSnapshot snapshot() const;
  // Write the committed state to the graph node, so that an update of the exec
  // from the graph (cuGraphExecUpdate) finds them equal.
  void mirror_graph_node() const;

 private:
  size_t pointer_offset(int64_t argument, std::optional<size_t> byte_offset) const;
  size_t scalar_offset(int64_t argument, size_t width, std::optional<size_t> byte_offset) const;
  void reserve_tensor_maps(size_t count);
  int max_threads_per_block(c10::DeviceIndex device) const;
  void update_pointers(
      uintptr_t graph_exec,
      c10::ArrayRef<KernelPointerSlot> slots,
      c10::ArrayRef<uintptr_t> pointers);
  void update_replay(
      uintptr_t graph_exec,
      c10::ArrayRef<KernelPointerSlot> pointer_slots,
      c10::ArrayRef<KernelScalarSlot> scalar_slots,
      c10::ArrayRef<KernelTensorMapSlot> tensor_map_slots,
      const std::optional<std::array<size_t, 3>>& grid,
      const std::optional<size_t>& shared_memory,
      c10::ArrayRef<uintptr_t> pointers,
      c10::ArrayRef<int64_t> values,
      const std::optional<std::array<size_t, 3>>& block = std::nullopt);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class KernelParamCache;

class TORCH_CUDA_CPP_API KernelParamUpdateBatch {
  friend struct at::cuda::CUDAGraph;

  std::shared_ptr<KernelParamCache> cache_;
  uint64_t generation_ = 0;
  bool needs_instantiation_ = false;
  std::vector<KernelNodeUpdate> updates_;
  std::vector<std::shared_ptr<KernelNodeParams>> nodes_;
};

class KernelPointerUpdateBatch {
  friend struct at::cuda::CUDAGraph;
  friend class GraphReplayLease;

 public:
  bool has_numeric_updates() const {
    return numeric_;
  }
  size_t value_count() const {
    return value_ranges_.size();
  }
  int64_t memcpy_updates() const {
    return memcpy_updates_;
  }
  int64_t source_rebinds() const {
    return source_rebinds_;
  }
  int64_t template_applies() const {
    return template_applies_;
  }
  int64_t template_graph_updates() const {
    return template_graph_updates_;
  }

 private:
  struct Node {
    std::shared_ptr<KernelNodeParams> params;
    std::vector<KernelPointerSlot> slots;
    std::vector<KernelScalarSlot> scalars;
    std::optional<std::array<size_t, 3>> grid;
    std::optional<size_t> shared_memory = std::nullopt;
    std::vector<KernelTensorMapSlot> tensor_maps = {};
    std::optional<std::array<size_t, 3>> block = std::nullopt;
    int max_block_threads = 0;
  };
  struct Memset {
    KernelMemsetBinding binding;
    unsigned int value;
    size_t pitch;
    uintptr_t dst;
    size_t width;
    uintptr_t pending_dst = 0;
    size_t pending_width = 0;
  };
  mutable std::vector<Memset> memsets_;
  struct HostTable {
    std::vector<uintptr_t> slots;
    size_t nbytes;
    std::vector<KernelHostTableElement> elements;
    std::vector<void*> events; // cudaEvent_t per slot, recorded after the launch
    std::vector<bool> recorded;
    size_t pos = 0;
    size_t used = 0;
    HostTable() = default;
    HostTable(const HostTable&) = delete;
    HostTable& operator=(const HostTable&) = delete;
    ~HostTable();
  };
  mutable std::vector<std::unique_ptr<HostTable>> host_tables_;
  struct Memcpy {
    uintptr_t node;
    std::optional<size_t> source_table;
    size_t source_pointer_index;
    std::optional<size_t> source_offset_value_index;
    size_t pointer_index;
    std::optional<size_t> address_offset_value_index;
    size_t bytes_value_index;
    uintptr_t src; // last submitted source, destination and width
    uintptr_t dst;
    size_t width;
    cudaMemcpyKind kind; // the captured node's kind (host-to-device or device-to-device)
  };
  mutable std::vector<Memcpy> memcpys_;
  std::vector<KernelRngBinding> rngs_;
  struct TemplateNode {
    uintptr_t node;
    KernelTemplateNode::Kind kind = KernelTemplateNode::Kind::kernel;
    uintptr_t function = 0; // the exec's current state
    std::array<unsigned int, 3> grid{};
    std::array<unsigned int, 3> block{};
    unsigned int shared_memory = 0;
    KernelNodeAttributes attributes{};
    std::vector<uint8_t> image;
    size_t image_size = 0;
    std::array<void*, 5> extra{};
    uintptr_t dst = 0; // a memset node's current destination, width and value
    size_t width = 0;
    unsigned int value = 0;
  };
  struct Template {
    KernelTemplateBinding binding;
    std::vector<TemplateNode> nodes;
    int64_t current = -1; // the variant index the nodes hold
    // hotpath-2 probe (template_cache): the variant `current` names, kept so a
    // served call with an unchanged index skips the registry's shared lock
    const KernelTemplateVariant* probe_variant = nullptr;
  };
  mutable std::vector<Template> templates_;
  mutable int64_t template_applies_ = 0; // variant swaps so far
  mutable int64_t template_graph_updates_ = 0; // of which through cuGraphExecUpdate
  mutable int64_t memcpy_updates_ = 0; // memcpy nodes re-parameterized so far
  mutable int64_t source_rebinds_ = 0; // of which the source moved
  // hotpath-2 probe (skip_unchanged): the previous call's roots and values;
  // an identical pair skips the per-node dirty check (measurement only)
  mutable std::vector<uintptr_t> probe_last_pointers_;
  mutable std::vector<int64_t> probe_last_values_;
  mutable bool probe_last_valid_ = false;
  void prepare_templates(std::vector<KernelTemplateBinding> templates, size_t pointer_count, size_t value_count);
  // true when a swap went through a graph node: update_exec_from_graph follows
  bool replay_templates(uintptr_t graph_exec, c10::ArrayRef<uintptr_t> pointers, c10::ArrayRef<int64_t> values) const;
  void update_exec_from_graph(uintptr_t graph_exec, uintptr_t graph) const;
  void validate_values(c10::ArrayRef<int64_t> values) const;
  std::shared_ptr<KernelParamCache> cache_;
  uint64_t generation_ = 0;
  size_t pointer_count_ = 0;
  bool numeric_ = false;
  std::vector<std::pair<int64_t, int64_t>> value_ranges_;
  std::vector<Node> nodes_;
};

enum class InputReleaseKind { Allocate, Kernel, Drop };

struct InputReleaseStep {
  InputReleaseKind kind;
  size_t index;
};

struct InputReleasePlan {
  std::vector<InputReleaseStep> steps;
  std::vector<size_t> candidates;
  std::vector<std::vector<size_t>> eligible_inputs;
};

// Internal, serialized adoption of an unshared private capture. The enclosing
// owner drains its managed stream before close and retains this lease with all
// execution resources if completion or checked teardown fails.
class TORCH_CUDA_CPP_API GraphReplayLease {
 public:
  GraphReplayLease(
      std::shared_ptr<CUDAGraph> graph,
      std::shared_ptr<KernelPointerUpdateBatch> batch,
      c10::DeviceIndex device);
  ~GraphReplayLease();
  GraphReplayLease(const GraphReplayLease&) = delete;
  GraphReplayLease& operator=(const GraphReplayLease&) = delete;

  void replay(c10::ArrayRef<uintptr_t> pointers,
              c10::cuda::CUDACachingAllocator::PendingGraphInputs* pending = nullptr);
  InputReleasePlan prepare_input_release(
      std::vector<InputReleaseStep> steps, const std::vector<uintptr_t>& nodes,
      size_t input_count, size_t allocation_count,
      c10::ArrayRef<KernelRootUse> parameter_root_uses = {}) const;
  void validate_values(c10::ArrayRef<int64_t> values) const;
  // The boxed owner validates values before parameter updates and keeps them
  // immutable under its lock through this call.
  void replay(c10::ArrayRef<uintptr_t> pointers, c10::ArrayRef<int64_t> values,
              c10::cuda::CUDACachingAllocator::PendingGraphInputs* pending = nullptr);
  // The serialized owner checks eligibility before entering non-retryable release.
  bool check_capture_pool_retirement() const;
  void retire_capture_pool();
  void close();
  const KernelPointerUpdateBatch& batch() const {
    return *batch_;
  }

 private:
  std::shared_ptr<CUDAGraph> graph_;
  std::shared_ptr<KernelPointerUpdateBatch> batch_;
};

} // namespace detail
} // namespace at::cuda
