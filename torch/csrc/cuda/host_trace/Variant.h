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
// a family tries first the variant that last served the dynamic sizes (else
// registration order)
extern bool variant_order_enabled;
// run_buffer keeps a run's placement while its temporaries' sizes repeat
extern bool placement_reuse_enabled;
// a form switch restores what the form's exec last held (else forgets it)
extern bool form_held_enabled;
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

enum class Outcome : int64_t { Miss = 0, Misaligned = 1, Capture = 2, Hit = 3 };

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
  // forms only once no key is missing
  py::object evaluate_py(py::handle args) const;
  // Whether the call's rows hold and a pair of its arguments overlaps: the
  // call runs eagerly
  bool overlaps_py(py::handle args) const;
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
  // holds and no earlier entry's does
  void add_entry(size_t site, int64_t predicate, py::handle nodes);
  // Replaces the program by one that extends it (rows added, for add_entry)
  void set_program(const HostTraceProgram& program);
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
  struct Record {
    Kind kind;
    CUgraphNode node;
    bool held;
    int64_t held_row; // the row the node holds, as Frame::site_entries
    int64_t site; // or -1
    size_t site_pos; // its node's index in the site
    // a kernel's
    KernelRow kernel;
    std::array<int64_t, 7> held_dims;
    // a memset's
    MemsetRow memset;
    std::array<int64_t, 4> held_memset; // address, width, height, pitch
    // a memcpy's
    MemcpyRow copy;
    std::array<int64_t, 3> held_copy; // dst, src, bytes
  };
  // A key's rows, per site node the kernel or memset of the node's kind
  struct Entry {
    std::vector<KernelRow> kernels;
    std::vector<MemsetRow> memsets;
    std::vector<Field> fields;
    std::vector<uint8_t> image;
    std::vector<uint8_t> held;
    int32_t arm = 0;
    std::vector<int64_t> scratch; // bytes per the site's scratch buffer
    std::vector<Descriptor> descriptors;
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
  };
  struct Segment {
    py::object graph; // the torch.cuda.CUDAGraph
    py::object graph_dict;
    std::shared_ptr<at::cuda::CUDAGraph> native;
    // per form its exec, [0] the graph's, and per site in `sites` its arm
    std::vector<CUgraphExec> execs;
    std::vector<std::vector<int32_t>> forms;
    size_t form; // the one whose exec the records' held parameters are of
    size_t first;
    size_t stop;
    std::vector<size_t> sites; // the keyed sites in it
    // per form but [0]: its graph, the records' nodes in it (null for one a
    // piece replaced) and the pieces' nodes, as records of their sites
    struct Clone {
      CUgraph graph;
      std::vector<CUgraphNode> nodes;
      std::vector<Record> pieces;
    };
    std::vector<Clone> clones;
    // with form_held_enabled, per form what its exec's kernel nodes held when
    // it last ran: per record, then per piece
    struct Held {
      bool held = false;
      int64_t row = 0;
      std::array<int64_t, 7> dims{};
      std::vector<uint8_t> image;
    };
    std::vector<std::vector<Held>> form_held;
    // an RNG segment's: the generator its kernels draw from, its capture's
    // seed and offset addresses, and the increment its exec is set to (-1
    // the capture's)
    std::optional<at::Generator> generator;
    int64_t philox_seed = 0;
    int64_t philox_offset = 0;
    int64_t rng_increment = -1;
    // its host-to-device and device-to-host memcpy records
    std::vector<size_t> host_copies;
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
  enum class Op : uint8_t { Tensor, Temporary, Free };
  struct Event {
    Op op;
    int64_t alloc;
    size_t block; // a temporary's index among the step's temporaries
  };
  struct StepMemory {
    std::vector<int64_t> tensors;
    std::vector<Temporary> temporaries;
    std::vector<int64_t> drops;
    // eager order (_host_trace_memory.StepMemory.order), or empty
    std::vector<Event> order;
    std::vector<int64_t> arguments;
    // run_buffer's last placement: the temporaries' sizes, their offsets and
    // the total (-1: none)
    mutable std::vector<int64_t> placed_sizes;
    mutable std::vector<int64_t> placed_offsets;
    mutable int64_t placed_total = -1;
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
  // An eager-order run leaves the device's allocator locked in `hold`.
  at::Tensor allocate(
      const StepMemory& memory,
      const Segment* run,
      Frame& frame,
      std::unique_lock<std::recursive_mutex>& hold) const;
  void allocate_tensor(int64_t k, Frame& frame) const;
  int64_t alloc_bytes(int64_t k, const Frame& frame) const;
  at::Tensor run_buffer(const StepMemory& memory, Frame& frame) const;
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
  static void note_reads(Site& s, const Entry& e);
  // after a site gains an entry: the index of users is rebuilt at the next
  // commit, keeping what the records hold
  void rows_added();
  KernelRow& kernel_row(Record& r, int64_t row) const;
  void switch_form_held(Segment& run, size_t form);
  const KernelRow& kernel_row(const Record& r, int64_t row) const;
  const MemsetRow& memset_row(const Record& r, int64_t row) const;
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
  };
  const DeltaRows& delta_rows(c10::ArrayRef<uint64_t> changed);
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
