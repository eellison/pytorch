#pragma once

#include <torch/csrc/python_headers.h>

#if !defined(USE_ROCM)
#include <ATen/Context.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/util/SmallVector.h>
#include <cuda.h>
#include <torch/csrc/cuda/host_trace/Program.h>
#include <torch/csrc/utils/pybind.h>

#include <array>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#endif

namespace torch::cuda {
void initHostTraceVariantBindings(PyObject* module);
} // namespace torch::cuda

#if !defined(USE_ROCM)
namespace torch::cuda::host_trace {

// the trace's assumed alignment (_ALLOC_ALIGNMENT)
constexpr int64_t kAlignment = 256;

// test-only: the setters that succeed before one raises, or -1
extern int64_t fail_after_setter;
// test-only: the memset and memcpy nodes set
extern int64_t memory_node_sets;
// test-only: the kernel records packed (the dirty ones), and those of them
// whose node was set (the changed ones)
extern int64_t kernel_packs;
extern int64_t kernel_node_sets;
// test-only: kernel records whose selected row changed with nothing to set
extern int64_t kernel_row_switches;
// test-only: a form's sites whose pieces were not packed (nothing they read
// changed since that form's last replay)
extern int64_t piece_skips;
// users index rebuilds (index_users), and keyed rows added without one (row_added)
extern int64_t index_builds;
extern int64_t rows_unindexed;
// test-only: with phase_timing, each phase of the replays' host time in ns
extern bool phase_timing;
extern std::array<int64_t, 16> phase_ns;
inline int64_t phase_now() {
  return phase_timing
      ? std::chrono::steady_clock::now().time_since_epoch().count()
      : 0;
}
inline void phase_add(size_t p, int64_t& t) {
  if (phase_timing) {
    const int64_t now = phase_now();
    phase_ns[p] += now - t;
    t = now;
  }
}
// test-only: the TMA descriptors encoded, and those whose address alone was
// replaced
extern int64_t tma_encodes;
extern int64_t tma_replaces;
// test-only: descriptors whose values matched an earlier encode of theirs
extern int64_t tma_cache_hits;
// the encodes each descriptor keeps (Descriptor::encoded)
constexpr size_t kEncodedPerDescriptor = 4;
// test-only: runs whose temporaries went in a run buffer though the variant
// allocates in eager order
extern int64_t buffered_runs;
// test-only: fast-path calls that evaluated no dynamic row (no dynamic leaf
// changed), only the rows reading a changed leaf, or all of them, and the rows
// they evaluated; delta_enabled off evaluates all
extern int64_t delta_skips;
extern int64_t delta_partials;
extern int64_t delta_fulls;
extern int64_t delta_rows_evaluated;
extern bool delta_enabled;
// the static-prefix delta cache keeps up to this many other calls' dynamic
// rows besides its own (sizes that alternate re-evaluate nothing)
extern int64_t delta_slots;
// test-only: calls whose cache was exchanged with a kept call's
extern int64_t delta_exchanges;
// test-only: full evaluations that stopped at a valid row that does not hold
extern int64_t valid_first_misses;
// a family tries first the variant that last served the dynamic sizes (else
// registration order)
extern bool variant_order_enabled;

// A library sets a function's dynamic shared memory limit to its launch's
// bytes before the launch (cuDNN and cuBLASLt leave it at the last launch's),
// and the driver refuses a node set or launch above it. A replay's node set
// or launch at another key's bytes raises it the same way first.
void raise_dynamic_smem_limit(CUfunction f, unsigned int bytes);
// run_buffer keeps a run's placement while its temporaries' sizes repeat
extern bool placement_reuse_enabled;
// a form switch restores what the form's exec last held (else forgets it)
extern bool form_held_enabled;
// executor-level split (on by default, TORCH_HOST_TRACE_SPLIT=0 off): a run with legal cuts launches its form's
// sections (sub-execs over the same nodes and addresses) every call, patching
// section k+1 while section k runs. Sections grow by split_ratio in the records
// a call can re-set, at most split_sections; split_first_nodes > 0 sets the
// first section's records and split_max_nodes > 0 the most records a section
// takes (test and diagnosis knobs, 0: auto)
extern bool split_enabled;
extern int64_t split_sections;
extern double split_ratio;
extern int64_t split_first_nodes;
extern int64_t split_max_nodes;
// a call launches its run's sections only when at least this many of the run's records are dirty (its patch would
// be exposed: a size change, a mixed step); fewer (steady decode, hidden behind the GPU work in flight) launch the
// whole exec. 0: sections every call
extern int64_t split_dirty_min;
// calls whose run launched whole for fewer dirty records than split_dirty_min
extern int64_t split_dirty_wholes;
// execs uploaded (each in the call that first launches it: upload_once, split_ready)
extern int64_t exec_uploads;
// test-only: calls launched split, and forms whose sub-execs were built
extern int64_t split_calls;
// test-only: graph launches of native replays (a split call's sections each)
extern int64_t graph_launches;
extern int64_t split_builds;
// add_entry takes an entry's TMA descriptors (else refuses the entry)
extern bool entry_descriptors_enabled;

#if defined(__linux__)
// the opaque harvest's torch._C._cuda_hostTrace* calls
void initHarvestBindings(py::module& m);
#endif
// torch._C._cuda_hostTrace<Op>, the traced hosts of ATen ops
void initHostTraceAtenBindings(py::module& m);

class HostTraceVariant;

// A "planned" variant's buffer, held across calls per device and stream
struct HeldBuffer {
  c10::DeviceIndex device;
  cudaStream_t stream;
  c10::Storage storage;
  bool busy = false;
};

// the held buffers' bytes; release_held_buffers frees those no call uses
int64_t held_buffer_bytes();
void release_held_buffers();

// A commit's planned buffer: its stream's held one while the commit runs,
// else one of its own
struct PlanLease {
  HeldBuffer* held = nullptr;
  c10::Storage storage;
  PlanLease() = default;
  PlanLease(const PlanLease&) = delete;
  PlanLease& operator=(const PlanLease&) = delete;
  ~PlanLease();
};

// One call's state; a call from inside an eager step has its own
struct Frame {
  PyObject* const* args = nullptr;
  size_t count = 0;
  c10::SmallVector<int64_t, 256> values;
  // per base (allocation k, eager output n_alloc + j) while held
  c10::SmallVector<at::Tensor, 16> tensors;
  c10::SmallVector<int64_t, 16> bases;
  // per keyed site, the row its key selects: 0 its records' own, k > 0 entry
  // k - 1
  c10::SmallVector<int64_t, 4> site_entries;
  // per segment a row gives an arm, the form the rows select
  c10::SmallVector<int32_t, 4> forms;
  // a variant's site lacked the key: the call's binding is not in a table yet
  bool keyed_miss = false;
  // a variant's rows held and a pair of its arguments overlaps
  bool overlapped = false;
  // the arguments' tuple when only the call holds it (call_boxed): an
  // argument is dropped after the step of its last use
  PyObject* owned = nullptr;
  // the variant whose cached rows `values` holds (evaluate_static)
  HostTraceVariant* cached = nullptr;
  // evaluate's program evaluated every row, whether or not the call hits
  bool evaluated = false;
  // the planned buffer's [lo, hi), which no eager output may be in
  int64_t planned_lo = 0;
  int64_t planned_hi = 0;
  // entry_dispatch found no capture underway on the current stream
  bool uncaptured = false;
};

// MemoryKind: a memcpy node can only be set to operands of its capture's kind
// of memory (cudaMalloc'd or cuMem-mapped), and an argument it copies is of
// the other kind: the variant does not hold the call. No step has run
enum class Outcome : int64_t {
  Miss = 0,
  Misaligned = 1,
  Capture = 2,
  Hit = 3,
  MemoryKind = 4
};

// a memcpy node's update refused an operand of another kind of memory
struct MemoryKindMismatch {};

// A tensor an eager leaf or an output views: an argument, or a held base
struct BaseRef {
  bool argument;
  int64_t index;
};

struct View {
  BaseRef base;
  std::vector<int64_t> sizes; // rows
  std::vector<int64_t> strides; // rows
  int64_t offset; // row
  // set when the view reinterprets its base's storage (view_as_complex)
  std::optional<at::ScalarType> dtype;
};

// The global cuBLAS state a call read at the trace (library_state())
struct BlasState {
  at::Float32Precision matmul;
  at::CuBLASReductionOption fp16;
  at::CuBLASReductionOption bf16;
  bool fp16_accumulation;
  std::optional<int32_t> carveout;
  at::BlasBackend backend;
};

// A replay of one variant of torch/cuda/_host_trace_replay.py, from its
// validation to its outputs, built once from flatten_variant's VariantSpec
// (torch/cuda/_host_trace_native.py). Only a step the boxed dispatcher cannot
// express (a Triton launch) calls back into Python.
// A provider's held bindings for the keyed sites of one class (_host_trace_native.NativeBindings): per table key
// (a site key in the provider's normal form, Site::TableLink) the binding's nodes, their pointer sources as
// (operand or scratch buffer, index, delta), its scratch bytes and its topology's id. A keyed miss whose key is
// here gets its site's row in C++ (fill_from_table), as _fill's add_row would add it from the binding.
class HostTraceBindingTable {
 public:
  struct Source {
    bool scratch;
    uint32_t index;
    int64_t delta;
  };
  struct Field {
    uint32_t param;
    uint32_t offset;
    Source source;
  };
  struct Node {
    bool memset;
    uintptr_t function;
    std::array<int64_t, 7> dims; // grid xyz, block xyz, shared bytes
    std::vector<std::string> images;
    std::vector<Field> fields;
    Source dst; // a memset's
    unsigned int value;
    unsigned int element_size;
    std::array<int64_t, 3> shape; // width, height, pitch
  };
  struct Binding {
    std::vector<Node> nodes;
    std::vector<int64_t> scratch;
    int64_t topology;
  };
  // nodes as _host_trace_native.binding_template
  void add(const std::vector<int64_t>& key, py::handle nodes, const std::vector<int64_t>& scratch, int64_t topology);
  const Binding* find(const int64_t* key, size_t n) const;
  size_t size() const {
    return bindings_.size();
  }

 private:
  size_t width_ = 0;
  std::vector<int64_t> keys_;
  std::vector<Binding> bindings_;
  std::unordered_map<uint64_t, c10::SmallVector<uint32_t, 1>> index_;
};

// whether a keyed miss is filled from its site's binding table (fill_from_table); off: _fill's Python path
extern bool native_fill_enabled;
// rows fill_from_table added
extern int64_t native_fills;

class HostTraceVariant {
 public:
  explicit HostTraceVariant(py::handle spec);
  HostTraceVariant(const HostTraceVariant&) = delete;
  HostTraceVariant& operator=(const HostTraceVariant&) = delete;
  ~HostTraceVariant();

  // Every row at the call into frame.values and each keyed site's row into
  // frame.site_entries; false when the call misses
  bool evaluate(PyObject* const* args, size_t count, Frame& frame) const;
  // HostTraceReplay(static_prefix=n): the rows and argument pairs that read
  // only the first n arguments, which evaluate_static takes from a kept call
  void set_static_prefix(size_t n);
  // evaluate, with the rows that read only the leading arguments taken from
  // the call kept as `generation`, whose leading arguments the caller found
  // unchanged. frame.values is the cache until release(frame).
  bool evaluate_static(
      PyObject* const* args,
      size_t count,
      Frame& frame,
      uint64_t generation);
  void release(Frame& frame);
  // Keeps a hit's frame.values, every row evaluated, as `generation`
  void keep(Frame& frame, uint64_t generation);
  // (rows evaluate_static evaluates, rows)
  std::pair<size_t, size_t> static_split() const;
  // test-only: the leaves, as (input, kind, dim), the last partial delta
  // evaluate found changed
  py::list delta_changed() const;
  // (rows, [(site, key), ...] the tables lack, [(segment, arms), ...] the
  // forms lack, [site, ...] the selectors none of whose predicates holds) at
  // the call, or None when it misses otherwise (a site declines its key); the
  // forms only once no key is missing and every selector selects
  py::object evaluate_py(py::handle args) const;
  // Whether the call's rows hold and a pair of its arguments overlaps: the
  // call runs eagerly
  bool overlaps_py(py::handle args) const;
  // Links site to a binding table (Site::TableLink): a key it lacks whose normal form the table holds gets its row
  // in C++
  void set_site_table(
      size_t site,
      std::shared_ptr<HostTraceBindingTable> table,
      const std::vector<int8_t>& codes,
      const std::vector<int32_t>& companions,
      const std::vector<std::pair<int64_t, int64_t>>& operands,
      const std::vector<std::array<int64_t, 4>>& scratch,
      const std::vector<int64_t>& native_scratch,
      int64_t topology);
  // Adds site's row for key: nodes as _host_trace_native.binding_row, or
  // None to decline the key; the site's arm (0 its nodes' launch attributes)
  // the nodes need, whether the arm is a piece (nodes of their own, not the
  // site's), and per scratch buffer of the site its bytes at the key
  void add_row(
      size_t site,
      const std::vector<int64_t>& key,
      py::handle nodes,
      int32_t arm,
      bool piece,
      const std::vector<int64_t>& scratch);
  // Adds a selector's entry: its nodes (_host_trace_native.launch_row, with
  // rows for the launch dimensions and memset extents), run where predicate
  // holds and no earlier entry's does. With arm > 0 a section: nodes of their
  // own in place of the selector's (its forms' pieces); per temporary its
  // bytes' row, at bases from the base count at the call; the allocations of
  // the selector's op it does not make (`drops`); and where its temporaries
  // are made and freed in its run's eager order (before events alloc_at and
  // free_at; a run of no order or temporaries: in its tensors)
  void add_entry(
      size_t site,
      int64_t predicate,
      py::handle nodes,
      int32_t arm,
      const std::vector<int64_t>& temporaries,
      const std::vector<int64_t>& drops,
      int64_t alloc_at,
      int64_t free_at,
      int64_t made_seq,
      int64_t last_seq);
  // Replaces the program by one that extends it (rows added, for add_entry)
  void set_program(const HostTraceProgram& program);
  // the lowering's safe cut points of a segment (record offsets, sorted), for
  // the executor-level split (split_enabled)
  void set_split_cuts(size_t segment, const std::vector<size_t>& cuts);
  // Adds a form of the segment: exec, an instantiation of graph, a clone of
  // the segment's graph with each of its sites' kernel nodes at the launch
  // attributes of the site's arm in `arms`, or, a piece arm's site, its nodes
  // replaced by pieces' [(site, ((kind, node), ...)), ...]. The variant owns
  // the exec and the graph.
  void add_form(
      size_t segment,
      const std::vector<int32_t>& arms,
      uintptr_t exec,
      uintptr_t graph,
      py::handle pieces);
  // Runs a validated call's steps; on a hit, *result is its return value
  Outcome commit(Frame& frame, PyObject** result);
  // (outcome, result): evaluate and commit, for the entry's slow path
  py::tuple call_py(py::handle args);
  // Per record, None or what its exec node holds, as _Variant.held: a
  // kernel's (grid, per-parameter bytes), a memset's (address, width, height,
  // pitch), a memcpy's (dst, src, bytes)
  py::list held_images() const;
  // The eager steps that call Python, and the calls of them
  size_t python_steps() const;
  int64_t python_calls = 0;

 private:
  // A value a record's node holds: row's value plus delta, plus a base's
  // address for a pointer. Opaque operands and scratch buffers are bases
  struct Field {
    uint32_t param;
    uint32_t offset; // in the parameter's image
    // 1, 2, 4 or 8; a 4-byte pointer is the address's low half; 0: only a
    // descriptor reads it
    uint8_t width;
    bool pointer;
    int64_t row;
    int64_t base; // index in the bases, or -1: the row is the whole value
    int64_t delta;
  };
  // 8 bytes of an RNG kernel's generator state, plus delta: kind 0 the
  // address of its segment's captured seed, 1 of its captured offset, 2 its
  // call's intragraph offset, what the segment's earlier RNG calls take
  struct RngField {
    uint32_t param;
    uint32_t offset;
    uint8_t kind;
    int64_t delta;
  };
  // A functor member eager reads from a 0-dim CPU tensor (a CPU scalar class,
  // LaunchLayout.h), re-read from source at each replay
  struct CpuScalar {
    uint32_t param;
    uint32_t offset;
    char cls;
    at::Tensor source;
  };
  // A CUtensorMap parameter, encoded from 2 * rank fields from `first`: the
  // address, each extent, each byte stride but the first
  struct Descriptor {
    uint32_t param;
    size_t first; // in its kernel's fields
    CUtensorMapDataType dtype;
    std::vector<cuuint32_t> box;
    CUtensorMapSwizzle swizzle;
    CUtensorMapFloatOOBfill fill;
    // the library's own (TmaDescriptor.l2): the DSL's 128B, DeepGEMM's 256B
    CUtensorMapL2promotion l2;
    // nonzero: Triton's encode of elements this size, without the CuTe DSL's
    // bit and with its bit-21 clear (clear_bit21, the last encode's)
    uint32_t triton_elem;
    bool clear_bit21;
    // with the library's bits (the DSL's bit, Triton's bit-21 clear); else
    // the driver's map as is (tma_library_bits)
    bool edits;
    // the last encode's values: a change of the address alone replaces it
    std::vector<int64_t> last;
    CUtensorMap map;
    // the last few encodes (most recent first), each the values it was
    // encoded at, its map and clear_bit21: values seen before (sizes that
    // alternate) take its map, with the address replaced
    struct Encoded {
      std::vector<int64_t> values;
      CUtensorMap map;
      bool clear_bit21;
    };
    std::vector<Encoded> encoded;
  };
  // What a kernel node runs: its record's own, or a keyed site's row for a
  // key (harvested once, selected by one hash lookup of the key's rows)
  struct KernelRow {
    CUDA_KERNEL_NODE_PARAMS params; // func; grid, block, smem per call
    // grid xyz, block xyz, shared bytes
    std::array<int64_t, 7> dim_rows;
    bool constant_dims;
    std::array<int64_t, 7> dims; // a keyed row's
    size_t first_field; // while built, then fields
    size_t field_count;
    size_t first_descriptor;
    size_t descriptor_count;
    size_t offset; // of its images in its bytes, while built
    size_t nbytes;
    std::vector<size_t> param_offsets; // within its images
    std::vector<uint32_t> param_sizes;
    std::vector<void*> arguments; // params.kernelParams, into image
    const Field* fields;
    Descriptor* descriptors;
    uint8_t* image; // what a patch passes the driver
    uint8_t* held; // what the node held when it last ran this row
    std::vector<RngField> rng;
    int64_t rng_increment = 0; // the philox offsets its call takes
    std::vector<CpuScalar> cpu_scalars;
  };
  struct MemsetRow {
    Field dst;
    unsigned int value;
    unsigned int element_size;
    std::array<int64_t, 3> shape_rows; // width, height, pitch
    bool constant_shape;
    std::array<int64_t, 3> shape; // a keyed row's
  };
  // A 1D memcpy (copy_'s cudaMemcpyAsync); no keyed site's. A host-to-device
  // or device-to-host one copies from or to pinned argument host_arg
  struct MemcpyRow {
    Field dst;
    Field src;
    int64_t bytes_row;
    cudaMemcpyKind kind;
    int64_t host_arg;
  };
  // One kernel, memset or memcpy node of a segment. Its topology (kind, launch
  // attributes, programmatic edge into it) is fixed at instantiation; a keyed
  // site's rows share it (a binding of another topology is another variant's).
  enum class Kind : uint8_t { Kernel, Memset, Memcpy };
  // What its exec's node holds is per form (Segment::held)
  struct Record {
    Kind kind;
    CUgraphNode node;
    int64_t site; // or -1
    size_t site_pos; // its node's index in the site
    KernelRow kernel; // a kernel's
    MemsetRow memset; // a memset's
    MemcpyRow copy; // a memcpy's
  };
  // A key's rows, per site node the kernel, memset or (a section's) memcpy of
  // the node's kind
  struct Entry {
    std::vector<KernelRow> kernels;
    std::vector<MemsetRow> memsets;
    std::vector<MemcpyRow> memcpys;
    std::vector<Field> fields;
    std::vector<uint8_t> image;
    std::vector<uint8_t> held;
    int32_t arm = 0;
    std::vector<int64_t> scratch; // bytes per the site's scratch buffer
    std::vector<Descriptor> descriptors;
    // a section's temporaries: bytes' rows, at bases first_base, ...; its
    // op's allocations it does not make; its temporaries' places in the order
    std::vector<int64_t> temporaries;
    size_t first_base = 0;
    std::vector<int64_t> drops;
    size_t alloc_at = 0;
    size_t free_at = 0;
    // in a run buffer's run (no order): the seqs its temporaries are made at
    // (before its op's first launch) and last used at (its op's last launch),
    // over which they take the buffer's bytes first fit with the run's own
    // temporaries; last_seq -1: made before the run and held over it
    int64_t made_seq = -1;
    int64_t last_seq = -1;
  };
  // A keyed launch table: per key, filled once, the row its nodes run. A
  // selector (a top-level op's own guards) has no key: the first of its
  // predicates that holds picks the row
  struct Site {
    std::vector<int64_t> key_rows;
    // a selector's: its records' own, then per entry
    std::vector<int64_t> predicates;
    std::vector<size_t> records; // per node
    std::vector<int64_t> keys; // key_rows.size() values per key
    // per key: 0 the records' own, k > 0 entries[k - 1], -1 declined
    std::vector<int64_t> rows;
    std::unordered_map<uint64_t, c10::SmallVector<uint32_t, 1>> index;
    std::vector<std::unique_ptr<Entry>> entries;
    // per node, what its entries' kernel rows read, over all of them: the
    // index of users is rebuilt from these, not from every entry (a table
    // grows by a row per new key, and the index is rebuilt after each)
    struct Reads {
      std::vector<std::pair<int64_t, int64_t>> fields; // (row, base)
      std::vector<int64_t> dim_rows;
      bool always = false; // an RNG or CPU-scalar row
    };
    std::vector<Reads> reads;
    std::vector<int64_t> scratch; // its scratch buffers' allocations
    size_t segment; // its records' (a selector of no records: none)
    // what any of its rows' kernels reads, over every node (index_users): a
    // piece of the site is unchanged while these and the site row are; any
    // RNG or CPU scalar kernel makes it always patched
    std::vector<int64_t> piece_rows;
    std::vector<int64_t> piece_bases;
    bool piece_always = false;
    // the binding table a keyed miss is filled from: per key value its normal form (0 as is, 1 its alignment class,
    // 2 a stride, 0 where its size, key value `companions[j]`, is 1); per operand its (row, base); per scratch buffer
    // the site has, (index, row, base, the most bytes a row may take or -1); the buffers add_row sizes, in order; and
    // the topology's id a row of arm 0 has
    struct TableLink {
      std::shared_ptr<HostTraceBindingTable> table;
      std::vector<int8_t> codes;
      std::vector<int32_t> companions;
      std::vector<std::pair<int64_t, int64_t>> operands;
      std::vector<std::array<int64_t, 4>> scratch;
      std::vector<int64_t> native_scratch;
      int64_t topology;
    };
    std::unique_ptr<TableLink> link;
  };
  struct Segment {
    py::object graph; // the torch.cuda.CUDAGraph
    py::object graph_dict;
    std::shared_ptr<at::cuda::CUDAGraph> native;
    // per form its exec, [0] the graph's, and per site in `sites` its arm
    std::vector<CUgraphExec> execs;
    std::vector<std::vector<int32_t>> forms;
    size_t form; // the form last replayed (dirty_ is relative to its exec)
    size_t first;
    size_t stop;
    std::vector<size_t> sites; // the keyed sites and selectors in it
    std::vector<size_t> sections; // its selectors with a section's temporaries
    // per form but [0]: its graph, the records' nodes in it (null for one a
    // piece replaced) and the pieces' nodes, as records of their sites
    struct Clone {
      CUgraph graph;
      std::vector<CUgraphNode> nodes;
      std::vector<Record> pieces;
      // per site its pieces [first, stop), and what its rows' reads (the
      // site row, then PieceReads' rows, bases and launch dimension rows) were
      // when they were last patched (empty: not yet)
      struct Group {
        size_t site;
        size_t first;
        size_t stop;
        std::array<std::vector<int64_t>, 2> seen; // per mode (whole, split)
        std::array<uint64_t, 2> users_version{};
      };
      std::vector<Group> groups;
    };
    std::vector<Clone> clones;
    // per form, what its exec's nodes hold: per record, then per piece of the
    // form (made at the form's first replay). A memcpy's operands are known
    // (copy) even where the node is not held
    struct Held {
      bool held = false;
      int64_t row = 0; // the row the node holds, as Frame::site_entries
      std::array<int64_t, 7> dims{}; // a kernel's grid, block, smem
      std::vector<uint8_t> image; // a kernel's parameter bytes
      std::array<int64_t, 4> memset{}; // address, width, height, pitch
      std::array<int64_t, 3> copy{}; // dst, src, bytes
    };
    std::vector<std::vector<Held>> held; // per state: 2 * form + split
    // with form_held_enabled, per form left: the rows, bases and site rows its
    // records were last patched at, and which were dirty then
    struct Seen {
      bool saved = false;
      uint64_t users_version = 0;
      std::vector<int64_t> rows; // per read_rows
      std::vector<int64_t> bases; // per segment base (Users::segment_bases)
      std::vector<int64_t> sites; // per site in `sites`
      std::vector<uint8_t> dirty; // per record
    };
    std::vector<Seen> seen; // per state
    // the rows its records read (index_users)
    std::vector<int64_t> read_rows;
    // the active state's mode: the form's sub-execs (Split), else its exec
    bool split = false;
    // the lowering's safe cut points (record offsets: no programmatic edge
    // into the record, outside a keyed site's chain and a fork), sorted
    std::vector<size_t> cuts;
    // per form, its sub-execs: each a clone of the form's graph without the
    // nodes of the other ranges, over records [bounds[k], bounds[k + 1])
    struct Split {
      bool tried = false;
      bool ok = false;
      std::vector<size_t> bounds;
      std::vector<CUgraph> graphs;
      std::vector<CUgraphExec> execs;
      std::vector<CUgraphNode> nodes; // per record, in its range's graph
      std::vector<CUgraphNode> piece_nodes; // per piece of the form
      std::vector<size_t> piece_range; // per piece
      // test and diagnosis: per section, calls that set any of its nodes,
      // the nodes set, and its launches
      std::vector<int64_t> calls_set, nodes_set, launches;
    };
    std::vector<Split> splits; // per form
    // an RNG segment's: the generator its kernels draw from, its capture's
    // seed and offset addresses, and the increment its exec is set to (-1
    // the capture's)
    std::optional<at::Generator> generator;
    int64_t philox_seed = 0;
    int64_t philox_offset = 0;
    int64_t rng_increment = -1;
    // its host-to-device and device-to-host memcpy records
    std::vector<size_t> host_copies;
    // per form, whether its exec was uploaded (upload_once: in the call that first launches it)
    std::vector<uint8_t> uploaded;
  };
  struct Allocation {
    std::vector<int64_t> sizes; // rows
    std::vector<int64_t> strides; // rows
    at::ScalarType dtype;
    int64_t nbytes; // row
    // a keyed site's scratch buffer: its bytes are the key's row's
    int64_t site = -1;
    size_t scratch = 0; // its index in the site's
    // a planned arena: sized by its plan, so never a site's scratch buffer
    bool arena = false;
    // the selector whose op made it, where a section of it does not
    int64_t owner = -1;
  };
  struct Planned {
    int64_t alloc;
    int64_t offset; // row
    bool tensor; // an eager step reads or writes it
  };
  struct Temporary {
    int64_t alloc;
    int64_t seq;
    int64_t last; // the seq of its last use
  };
  // Argument: a boxed call's argument `alloc` (its position) is freed
  enum class Op : uint8_t { Tensor, Temporary, Free, Argument };
  struct Event {
    Op op;
    int64_t alloc;
    // a temporary's index among the step's temporaries, an argument's in
    // StepMemory::freed
    size_t block;
  };
  struct StepMemory {
    std::vector<int64_t> tensors;
    std::vector<Temporary> temporaries;
    std::vector<int64_t> drops;
    // eager order (_host_trace_memory.StepMemory.order), or empty
    std::vector<Event> order;
    std::vector<int64_t> arguments;
    // the arguments `order` frees, in its order
    std::vector<int64_t> freed;
    // run_buffer's last placement: the temporaries' sizes, their offsets and
    // the total (-1: none)
    mutable std::vector<int64_t> placed_sizes;
    mutable std::vector<int64_t> placed_offsets;
    mutable int64_t placed_total = -1;
    // the rows the temporaries' sizes read, and the temporaries a site's row
    // sizes (scratch buffers): the placement holds while those rows are at
    // placed_row_values and those temporaries at their placed sizes
    std::vector<int64_t> size_rows;
    std::vector<size_t> site_temporaries;
    mutable std::vector<int64_t> placed_row_values;
    // the sections placed with them (Entry::last_seq), in made_seq order:
    // per section its seqs, temporary count and sizes; their offsets
    mutable std::vector<int64_t> placed_section_key;
    mutable std::vector<int64_t> placed_section_offsets;
  };
  enum class LeafKind : uint8_t { View, Scalar, Constant };
  struct Leaf {
    LeafKind kind;
    int64_t index; // a view's index, a scalar's row
    py::object constant;
  };
  // A schema argument of a boxed step: a constant, a leaf, or a list of both
  enum class ArgKind : uint8_t {
    Constant,
    Tensor,
    Int,
    Double,
    TensorList,
    OptionalTensorList,
    IntList
  };
  struct ArgItem {
    int64_t leaf; // or -1: constant (None in a Tensor?[])
    c10::IValue constant;
  };
  struct BoxedArg {
    ArgKind kind;
    int64_t leaf;
    c10::IValue constant;
    std::vector<ArgItem> items;
  };
  // Per returned tensor: a fresh output over eager root `root`, or (root -1)
  // the leaf it returns
  struct Predicted {
    int64_t root;
    int64_t leaf;
    std::vector<int64_t> sizes; // rows
    std::vector<int64_t> strides; // rows
    int64_t offset; // row
    at::ScalarType dtype;
  };
  struct EagerStep {
    py::object run; // run(leaves, values) -> ((root, tensor), ...)
    std::vector<Leaf> leaves;
    std::optional<c10::OperatorHandle> op; // unset: run calls Python
    std::vector<BoxedArg> args;
    std::vector<Predicted> outputs;
    py::object target; // the op, for the disagreement
    std::string name; // str(target)
    std::optional<BlasState> blas; // unset: the state at the replay
    // a graphsafe RNG step's generator state, swapped into its device's
    // default generator around the call (_impl_graphsafe_rng)
    std::optional<at::Generator> rng_state;
    bool seed_offset_on_device = false; // HOST_SEED_OFFSET
  };
  struct Step {
    int64_t segment; // or -1: eager_[eager]
    size_t eager;
  };
  // Arguments pair_arguments_[a] and pair_arguments_[b], disjoint at the
  // trace, where a traced step writes one and reads the other
  // (Tape.argument_pairs)
  struct ArgumentPair {
    size_t a;
    size_t b;
  };
  enum class OutputKind : uint8_t { Scalar, View, Base, Alias };
  struct Output {
    OutputKind kind;
    int64_t index; // a scalar's row, a view's index, an alias's output
    BaseRef base;
  };

  const at::Tensor& base_tensor(
      const BaseRef& ref,
      PyObject* const* args,
      const Frame& frame) const;
  at::Tensor view(
      const View& v,
      PyObject* const* args,
      const Frame& frame,
      const int64_t* values) const;
  // The step's allocations; the run buffer holding its temporaries, if any.
  // An eager-order run leaves the device's allocator locked in `hold`. The
  // run's sections' temporaries freed after it is queued go in `held`. An
  // argument the order frees is its storage in `freed` (empty: not ours).
  at::Tensor allocate(
      const StepMemory& memory,
      const Segment* run,
      Frame& frame,
      std::unique_lock<std::recursive_mutex>& hold,
      c10::SmallVector<c10::DataPtr, 4>& held,
      c10::SmallVector<c10::Storage, 4>& freed) const;
  // whether the call's section of its op does not make allocation k
  bool dropped(int64_t k, const Frame& frame) const;
  void allocate_tensor(int64_t k, Frame& frame) const;
  int64_t alloc_bytes(int64_t k, const Frame& frame) const;
  at::Tensor run_buffer(
      const StepMemory& memory,
      Frame& frame,
      c10::ArrayRef<const Entry*> sections = {}) const;
  // "planned": the buffer, and each planned allocation's base and tensor
  void plan(Frame& frame, PlanLease& lease) const;
  // an eager step's fresh output's storage is not the planned buffer's
  void check_fresh(const EagerStep& step, int64_t base, const Frame& frame)
      const;
  bool evaluate_rows(PyObject* const* args, size_t count, Frame& frame) const;
  bool disjoint(PyObject* const* args, size_t count) const;
  // over `pairs`, reading the arguments at `used` in pair_arguments_ (null:
  // all)
  bool disjoint(
      PyObject* const* args,
      size_t count,
      const std::vector<ArgumentPair>& pairs,
      const std::vector<size_t>* used) const;
  // the keyed sites and forms at frame.values
  bool evaluate_sites(Frame& frame) const;
  bool evaluate_sites_slow(Frame& frame) const;
  // The key's index in the site's table, or -1
  int64_t find(const Site& s, const int64_t* key) const;
  // The selector's row at values, or -1
  static int64_t select(const Site& s, const int64_t* values);
  void insert(Site& s, const int64_t* key, int64_t row);
  // adds the entry's kernel rows' reads to s.reads
  // adds the entry's reads to the site's; whether any is new
  static bool note_reads(Site& s, const Entry& e);
  // whether the site's piece reads (index_users) cover the entry's
  static bool pieces_cover(const Site& s, const Entry& e);
  // a keyed row added: its reads noted, the users index rebuilt only if they are new
  void row_added(Site& s, const Entry& e);
  // site i's row for key from its binding table, added: its index in the site's keys, or -1 (no table, no binding,
  // another topology, or a scratch buffer the site lacks or holds fewer bytes of)
  int64_t fill_from_table(size_t i, const int64_t* key);
  // after a site gains an entry: the index of users is rebuilt at the next
  // commit, keeping what the records hold
  void rows_added();
  KernelRow& kernel_row(Record& r, int64_t row) const;
  // the form's held state, made at its first use (a memcpy's operands taken
  // from the current form's)
  std::vector<Segment::Held>& form_held(Segment& run, size_t form, bool split = false);
  // to another form of the run: with form_held_enabled its nodes keep what
  // they held and dirty_ marks what changed since its last replay, else none
  // is held
  void switch_form(Segment& run, size_t form, bool split, const Frame& frame);
  // marks into `out` (per record of the run) what changed since the state's
  // last replay; false when the state has no snapshot (all changed)
  bool changed_since(const Segment& run, const Segment::Seen& s, const Frame& frame, uint8_t* out) const;
  // the form's sub-execs, built at their first use; false when it has none
  bool split_ready(Segment& run, size_t form, const Frame& frame);
  // a split run's section bounds (record offsets), from its cuts
  std::vector<size_t> split_bounds(Segment& run);
  // uploads the form's exec unless it was (each exec once, in the call that first launches it)
  void upload_once(Segment& run, size_t form);
 public:
  // test-only: a form's section bounds once built (empty: not split)
  // per section of a form's split (empty: not split): (calls that set any of
  // its nodes, nodes set, launches)
  std::vector<std::tuple<int64_t, int64_t, int64_t>> split_section_counts(size_t segment, size_t form) const {
    const Segment& run = segments_.at(segment);
    std::vector<std::tuple<int64_t, int64_t, int64_t>> out;
    if (form < run.splits.size() && run.splits[form].ok) {
      const Segment::Split& sp = run.splits[form];
      for (size_t k = 0; k < sp.launches.size(); ++k) {
        out.emplace_back(sp.calls_set[k], sp.nodes_set[k], sp.launches[k]);
      }
    }
    return out;
  }
  std::vector<size_t> split_bounds_of(size_t segment, size_t form) const {
    const Segment& run = segments_.at(segment);
    return form < run.splits.size() && run.splits[form].ok ? run.splits[form].bounds : std::vector<size_t>();
  }
 private:

  const KernelRow& kernel_row(const Record& r, int64_t row) const;
  const MemsetRow& memset_row(const Record& r, int64_t row) const;
  const MemcpyRow& memcpy_row(const Record& r, int64_t row) const;
  int64_t check_row(int64_t row) const;
  int64_t check_base(int64_t base) const;
  // (param, offset, width, pointer, row, base, delta)
  Field parse_field(py::handle field, const KernelRow& k) const;
  // (row, base, delta)
  Field parse_source(py::handle source) const;
  // ((param, offset, kind, delta), ...) and the increment, for a kernel of
  // an RNG segment
  void parse_rng(
      KernelRow& k,
      const Segment& g,
      py::handle rng,
      py::handle increment) const;
  static unsigned int check_element_size(py::handle size);
  // ((param, first field, dtype, box, swizzle[, fill, triton_elem[,
  // edits]]), ...) of k, appended to `out`
  static void parse_descriptors(
      KernelRow& k,
      py::handle descriptors,
      std::vector<Descriptor>& out);
  // sets k's function; its grid, block and smem as KernelRow::dims
  static std::array<int64_t, 7> set_launch(
      KernelRow& k,
      py::handle function,
      py::handle block,
      py::handle smem,
      py::handle grid);
  static void append_images(
      KernelRow& k,
      py::handle images,
      std::vector<uint8_t>& bytes);
  static void bind_row(
      KernelRow& k,
      uint8_t* image,
      uint8_t* held,
      const Field* fields);
  int64_t field_value(const Field& f, const int64_t* v, const int64_t* bases)
      const;
  void encode(
      const KernelRow& k,
      Descriptor& d,
      const int64_t* v,
      const int64_t* bases,
      uint8_t* image) const;
  // A kernel's images at rows v and bases into image
  void pack(
      const KernelRow& k,
      const int64_t* v,
      const int64_t* bases,
      uint8_t* image) const;
  std::array<int64_t, 4> memset_shape(
      const MemsetRow& m,
      const int64_t* v,
      const int64_t* bases) const;
  std::array<int64_t, 3> memcpy_shape(
      const MemcpyRow& m,
      const int64_t* v,
      const int64_t* bases) const;
  void patch_and_replay(Segment& run, Frame& frame);
  // sets each memcpy node a later step replays whose argument operands moved
  // to them, before any step runs: false if one is of the other kind of
  // memory (the first step's own nodes fail at its patch, before it runs)
  bool set_argument_copies(Frame& frame);
  // The segment's form the rows select, or -1; `arms` its arms
  int64_t form_of(
      const Segment& g,
      const int64_t* site_entries,
      c10::SmallVector<int32_t, 8>& arms) const;
  void boxed(const EagerStep& step, PyObject* const* args, Frame& frame) const;
  void run_eager(const EagerStep& step, PyObject* const* args, Frame& frame);
  PyObject* outputs(PyObject* const* args, Frame& frame) const;
  bool observed(const Segment& run) const;

  HostTraceProgram program_;
  size_t valid_;
  size_t base_count_;
  size_t n_alloc_;
  std::vector<Segment> segments_;
  std::vector<Record> records_;
  std::vector<Field> fields_;
  std::vector<Descriptor> descriptors_;
  std::vector<Site> sites_;
  std::vector<size_t> armed_; // the segments a row gives an arm
  std::vector<uint8_t> storage_; // the images a patch passes the driver
  std::vector<uint8_t> held_bytes_; // the images the exec nodes hold
  std::vector<Allocation> allocations_;
  std::vector<View> views_;
  std::vector<EagerStep> eager_;
  std::vector<Step> steps_;
  std::vector<StepMemory> memory_; // per step, then after the last
  // "planned": the allocations in the buffer, and its bytes' row (else -1)
  std::vector<Planned> planned_;
  int64_t planned_bytes_ = -1;
  std::vector<Output> outputs_;
  std::vector<ArgumentPair> argument_pairs_;
  std::vector<size_t> pair_arguments_;
  // evaluate_rows's: the rows the valid row reads and the rest, in order, for
  // a program of valid_split_rows_ rows
  mutable std::vector<uint32_t> valid_first_rows_;
  mutable std::vector<uint32_t> valid_other_rows_;
  mutable size_t valid_split_rows_ = 0;
  // set_static_prefix's: the rows, leaves and argument pairs that read an
  // argument at or after static_prefix_, and the pairs' arguments
  size_t static_prefix_ = 0;
  std::vector<uint32_t> dynamic_rows_;
  // dynamic_rows_ as the ones the valid row reads and the rest
  std::vector<uint32_t> dynamic_valid_rows_;
  std::vector<uint32_t> dynamic_other_rows_;
  std::vector<uint32_t> dynamic_leaves_;
  std::vector<ArgumentPair> dynamic_pairs_;
  std::vector<size_t> dynamic_pair_arguments_;
  // the last call all of whose sites resolved: the rows its site keys and
  // predicates read (sorted, unique), their values, its site rows and forms,
  // and armed_.size() then (-1: none)
  mutable std::vector<int64_t> site_reads_;
  mutable std::vector<int64_t> site_read_values_;
  mutable std::vector<int64_t> site_memo_entries_;
  mutable std::vector<int32_t> site_memo_forms_;
  mutable int64_t site_memo_armed_ = -1;
  // bumped by each rebuild of site_reads_
  mutable uint64_t site_reads_version_ = 0;
  // the memo's site reads are cached_values_'s: a delta call compares only
  // those among the rows it evaluates, as the other rows are the cache's
  mutable bool site_memo_cache_ = false;
  // the kept call's values, unless a call is using them
  c10::SmallVector<int64_t, 256> cached_values_;
  uint64_t cached_generation_ = 0;
  bool cache_busy_ = false;
  // delta_valid_: cached_values_'s dynamic rows are those of delta_leaves_
  // (per dynamic_leaves_). Per mask of changed leaves (bit j for
  // dynamic_leaves_[j]) the dynamic rows that read one, split as
  // dynamic_valid_rows_ and dynamic_other_rows_. delta_marked_: mark_dirty
  // last saw cached_values_ as they are.
  struct DeltaRows {
    std::vector<uint64_t> changed;
    std::vector<uint32_t> valid_rows;
    std::vector<uint32_t> other_rows;
    // the site reads among those rows, as indices in site_reads_ when it was
    // site_reads_version_'s
    std::vector<uint32_t> site_reads;
    uint64_t site_reads_version = 0;
  };
  DeltaRows& delta_rows(c10::ArrayRef<uint64_t> changed);
  // up to delta_slots other calls' dynamic rows (of
  // the cache's generation), and the site memo with them: a call whose dynamic
  // leaves are nearer one's than the cache's (sizes that alternate) exchanges
  // it with the cache before its delta
  struct DeltaSlot {
    uint64_t generation = 0;
    uint64_t used = 0;
    std::vector<int64_t> leaves; // per dynamic_leaves_
    std::vector<int64_t> rows; // per dynamic_rows_
    bool memo = false; // site_memo_cache_ then
    int64_t memo_armed = -1;
    std::vector<int64_t> read_values;
    std::vector<int64_t> entries;
    std::vector<int32_t> forms;
  };
  // the cache's dynamic rows and memo exchanged with d's
  void exchange(DeltaSlot& d, int64_t* v, bool& memo_fresh);
  std::vector<DeltaSlot> delta_slots_;
  uint64_t delta_tick_ = 0;
  std::vector<int64_t> delta_leaves_;
  std::vector<DeltaRows> delta_memo_;
  std::vector<uint64_t> delta_last_changed_;
  bool delta_valid_ = false;
  bool delta_marked_ = false;
  // A record is re-packed only when dirty: a row, base or site row it reads
  // changed since its last patch, or it is patched at every replay (a memset,
  // a memcpy, an RNG or CPU scalar kernel). Per row, base and site the records
  // whose rows (their own and their sites') read it, as CSR; built at the
  // first commit after a change of the rows or sites.
  struct Users {
    std::vector<uint32_t> row_first;
    std::vector<uint32_t> row_records;
    std::vector<uint32_t> base_first;
    std::vector<uint32_t> base_records;
    std::vector<uint32_t> rows; // the rows some record reads
    std::vector<uint32_t> dynamic_rows; // those of dynamic_rows_
    std::vector<uint32_t> segment_first; // per segment, into segment_bases
    std::vector<uint32_t> segment_bases;
  };
  void index_users();
  // marks the records the call's rows and sites change
  void mark_dirty(const Frame& frame);
  void mark(const std::vector<uint32_t>& first, const std::vector<uint32_t>& records, size_t i);
  Users users_;
  bool users_stale_ = true;
  // stale only because entries were added: index_users keeps what the
  // records hold and the values seen (no full re-patch after a new key)
  bool users_kept_ = false;
  // bumped by index_users: a form's Seen is of a version
  uint64_t users_version_ = 0;
  std::vector<uint8_t> dirty_; // per record
  std::vector<uint8_t> always_; // per record
  // what the clean records were patched at: the rows of users_.rows, the
  // bases, the site rows; seen_generation_ the cache generation of the static
  // rows (0: unknown)
  std::vector<int64_t> seen_values_;
  std::vector<int64_t> seen_bases_;
  std::vector<int64_t> seen_sites_;
  uint64_t seen_generation_ = 0;
  bool seen_valid_ = false;
  int committing_ = 0;
  int64_t result_kind_; // none, tensor, tuple, list
  c10::DeviceIndex device_;
  py::object disagreement_;
  py::object global_replay_start_hooks_;
  py::object global_replay_end_hooks_;
};

} // namespace torch::cuda::host_trace
#endif
