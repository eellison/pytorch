// Host tracing for CUDA C++ kernels: run a kernel's host code once with
// symbolic sizes and no data pointers, under a stream capture, and record what
// it did (launches with every argument as a symbolic value, opaque calls, the
// generator increment) as a Tape that the Python side turns into a replay.
//
// What a traceable host does differently: size variables are c10::SymInt,
// addresses are read with sym_const_data_ptr() for inputs and
// sym_mutable_data_ptr() for outputs and in-place operands (a traced tensor
// has no storage, so data_ptr() raises; in ordinary mode the two forms are
// const_data_ptr() and mutable_data_ptr(), so an input that is copy-on-write
// stays lazy exactly as it did before the conversion), a by-value parameter
// struct goes through the Traced<T> proxy of Field.h, and a kernel with bare
// arguments launches through the typed helper of Launch.h. Everything else
// stays as written.
//
// The symbolic values are ordinary torch.SymInt / SymFloat / SymBool over a
// ShapeEnv, created on the Python side (torch/cuda/_host_trace.py) together
// with the traced tensors; C++ sees them as c10::SymInt and never inspects an
// expression. The one thing the recorder needs from a value is its hint (the
// value at the traced call), which it reads through Hints::of without
// recording a guard. That read is private to the recorder's own templates and
// sources (see Hints): a host guards (guard_int, a recorded branch) or it
// raises (expect_int), it never reads a hint.
//
// The host contract. A traceable host performs no synchronous memory API call
// (cudaMemcpy, cudaMemcpyToSymbol, cudaMemcpyFromSymbol, cudaMemset), no
// device or stream synchronize and no cudaMalloc. Asynchronous copies and
// memsets on the current stream are fine: they become nodes the completeness
// check sees. The trace is complete with respect to everything the capture
// sees. The synchronous memory calls are invisible to a stream capture in
// every mode on CUDA 13.0 (measured: they succeed, run at the trace and at the
// replay's build, and are absent from the graph), so they are a contract
// violation, caught by the HOSTTRACE_SYNC_API lint on every translation unit
// that includes these headers; a host that makes one is already wrong under
// a plain CUDA graph capture. The other forbidden calls fail inside the
// thread-local trace capture and the trace declines by the CUDA error's name.
// One-time initializations (a constant upload, a lazy module load) belong to
// the warm-up call trace() makes before the symbolic run, never to the host.
// Host-to-device data (a pointer table for a grouped kernel, ids from pinned
// memory) goes through HostTable and copy_h2d (HostTable.h): an asynchronous
// copy the tape describes and the replay re-issues from its own staging.
#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/host_trace/Tape.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/ArrayRef.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace at::cuda::host_trace {

// The host did something this tracer does not describe. Registered with
// pybind11 as torch._C._HostTraceDeclined, which torch/cuda/_host_trace.py
// exposes as Declined.
struct TORCH_CUDA_CPP_API Declined : std::runtime_error {
  using std::runtime_error::runtime_error;
};
[[noreturn]] TORCH_CUDA_CPP_API void decline(const std::string& msg);

// The name of a CUDA error that means "illegal inside a stream capture"
// (cudaErrorStreamCapture* and cudaErrorCapturedEvent), or nullptr for any
// other code. The Python side classifies a c10::AcceleratorError raised by
// the host through this, by code, never by message text.
TORCH_CUDA_CPP_API const char* capture_error_name(int code);

// A double the host narrows to a float (a float member of a parameter struct
// does it on assignment; a host narrows its own float formal by calling this).
// The ordinary value is the narrowed one; under a trace the symbolic value is
// wrapped (Float32 in torch/cuda/_host_trace.py) so that arithmetic downstream
// of the narrowing computes with what the kernel receives, as eager does.
TORCH_CUDA_CPP_API c10::SymFloat round_float32_symbolic(
    const c10::SymFloat& value);
inline c10::SymFloat round_float32(const c10::SymFloat& value) {
  if (C10_LIKELY(!value.is_symbolic())) {
    return c10::SymFloat(
        static_cast<double>(static_cast<float>(value.as_float_unchecked())));
  }
  return round_float32_symbolic(value);
}

// Complete only inside Recorder.cpp. A translation unit that defines this
// struct itself reaches the private reader: an ODR violation the language
// cannot forbid, so the HOSTTRACE_SYNC_API lint flags the name outside the
// recorder's own sources.
struct HintsInternal;
struct HostTableBase;
template <class, size_t>
struct IntField;
template <class, size_t>
struct FloatField;
template <size_t>
struct PtrField;
struct Grid;
namespace launch_detail {
template <class K, class A>
void put_arg(
    K& dst,
    const A& a,
    size_t off,
    std::vector<FieldRec>* rec,
    const char* kname,
    size_t i);
} // namespace launch_detail
template <class... KArgs, class G, class B, class... Args>
void launch(
    void (*kernel)(KArgs...),
    const G& grid,
    const B& block,
    const c10::SymInt& smem,
    cudaStream_t stream,
    const Args&... args);

// The value a symbolic scalar had at the traced call, read without a guard.
// Private: only the recorder's field types (Field.h), the typed launch
// (Launch.h), Grid and the recorder's own sources may take one, and only for
// the bytes they hand to the real launch under the capture and to the hint
// image. A host TU cannot name Hints::of; a host that needs a value guards.
class TORCH_CUDA_CPP_API Hints {
  static int64_t of(const c10::SymInt& s);
  static double of(const c10::SymFloat& s);
  static bool of(const c10::SymBool& s);
  friend struct HintsInternal;
  friend struct HostTableBase;
  friend struct Grid;
  template <class, size_t>
  friend struct IntField;
  template <class, size_t>
  friend struct FloatField;
  template <size_t>
  friend struct PtrField;
  template <class K, class A>
  friend void launch_detail::put_arg(
      K& dst,
      const A& a,
      size_t off,
      std::vector<FieldRec>* rec,
      const char* kname,
      size_t i);
  template <class... KArgs, class G, class B, class... Args>
  friend void launch(
      void (*kernel)(KArgs...),
      const G& grid,
      const B& block,
      const c10::SymInt& smem,
      cudaStream_t stream,
      const Args&... args);
};

struct RootInfo {
  c10::SymInt root; // the storage base address as a value: p<k> or 256*q<k>
  int64_t itemsize = 1;
  // made inside the traced call: its storage is never another live root's,
  // so a host's overlap test between it and any other root is No without
  // reading an address (the sibling's overlap predicates)
  bool allocation = false;
  // the tape's name of the root (p<k> an input, a<k> an allocation): two
  // traced tensors are views of one storage exactly when their names agree
  std::string name;
};

// Capture-based recording: the host runs under a stream capture, its launches
// are captured rather than executed, and each launch record names the kernel
// node the capture created for it: the launching stream's capture frontier,
// which a launch replaces with the new node alone (cudaStreamGetCaptureInfo).
// A typed launch reads it right after cudaLaunchKernel; a verbatim `<<<>>>`
// is issued by the host after the Grid or proxy conversion opened its record,
// so the record reads the frontier of the stream that was current at the
// conversion when the next recorder event closes it. finish_trace pairs
// records to nodes by that handle, never by the order cudaGraphGetNodes
// returns (creation order on the drivers measured, not a documented one).
// Two kinds of record: a typed launch (Launch.h) is complete at the call,
// with the function, positional parameters, the bytes it handed the driver
// and the grid, block and shared-memory values; a verbatim `<<<>>>` leaves
// the proxy's field snapshot and the Grid values, and its remaining bare
// arguments are constants.
struct LaunchPacket {
  int64_t seq = -1;
  // the stream the launch targets (verbatim: the current stream at the
  // conversion) and the kernel node the capture created for it
  cudaStream_t stream = nullptr;
  cudaGraphNode_t node = nullptr;
  // typed launch (Launch.h)
  bool typed = false;
  const void* func = nullptr; // the kernel's host symbol
  std::string kernel;
  std::vector<FieldRec> params; // absolute image offsets
  std::vector<uint8_t> image; // the bytes handed to the launch (hints)
  std::array<int64_t, 3> block{1, 1, 1};
  std::array<c10::SymInt, 3> block_expr{};
  c10::SymInt smem{0};
  // verbatim launch-site events
  bool has_proxy = false;
  std::vector<uint8_t> pod;
  std::vector<FieldRec> fields;
  bool has_grid = false;
  std::array<c10::SymInt, 3> grid{};
};

struct TORCH_CUDA_CPP_API TraceState {
  Tape* t = nullptr;
  at::DeviceIndex device = 0;
  // the traced tensors' roots, by TensorImpl (the Python side registers every
  // traced tensor and view and keeps them alive for the trace)
  std::unordered_map<const c10::TensorImpl*, RootInfo> roots;
  bool capturing = false;
  // The capture is an at::cuda::CUDAGraph (keep_graph) so the capture id is
  // registered and the default generator can create its per-capture state.
  std::unique_ptr<at::cuda::CUDAGraph> capture_graph;
  std::optional<c10::cuda::CUDAStreamGuard> stream_guard;
  // the capturing stream and its capture id: a launch on any other stream
  // would execute for real, with the traced tensors' placeholder addresses
  cudaStream_t capture_stream = nullptr;
  unsigned long long capture_id = 0;
  std::vector<LaunchPacket> packets;
  LaunchPacket open;
  bool open_active = false;
  // generator accounting: the default CUDA generator's offset when the trace
  // began; finish_trace compares what the host consumed with the declared
  // increments (rng_increment_hint is their sum, -1 when none was declared)
  uint64_t rng_offset_before = 0;
  int64_t rng_increment_hint = -1;
  uint64_t rng_consumed = 0;
  // one entry per rng_increment call, in host order, with the index of the
  // launch that follows it; finish_trace pairs them with the launches whose
  // arguments carry a philox state (Tape.h RngSlotRec)
  struct RngDecl {
    c10::SymInt increment;
    int64_t hint;
    size_t first_launch;
  };
  std::vector<RngDecl> rng_decls;
};

TORCH_CUDA_CPP_API TraceState* active();

// Enter/leave trace mode on this thread and begin the capture. The capture is
// thread-local: cudaMalloc, a device or stream synchronize and an event query
// from the host error inside it and the trace declines. A synchronous
// cudaMemcpy / cudaMemset from the host is invisible to any capture mode (see
// the host contract above): the lint, not the capture, catches it.
// Construction is transactional: trace mode is entered only after the stream,
// the capture and the generator state are set up; a failure unwinds them and
// leaves no trace active.
struct TORCH_CUDA_CPP_API Scope {
  explicit Scope(TraceState* s);
  ~Scope();
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  Scope(Scope&&) = delete;
  Scope& operator=(Scope&&) = delete;
  // the state this scope entered and the one it restores: the destructor
  // reads neither off the thread's active pointer
  TraceState* const state;
  TraceState* const prev;
};

// Trace mode on this thread for one op, without a capture of its own. The
// trace's dispatch mode enters it around each op it routes on a thread other
// than the tracing one: the autograd engine runs backward nodes on its device
// worker thread under a copy of the tracing thread's mode stack, and their
// launches go to the forward op's stream, the capturing one. Nothing on the
// launch path changes: a launch is accepted by the capture id of the stream it
// targets (require_capturing_stream, frontier_node), whichever thread issues
// it, and the completeness check pairs every captured node with a record.
// A thread the mode never reached (an unrelated thread) keeps ordinary mode.
struct TORCH_CUDA_CPP_API ThreadScope {
  explicit ThreadScope(TraceState* s);
  ~ThreadScope();
  ThreadScope(const ThreadScope&) = delete;
  ThreadScope& operator=(const ThreadScope&) = delete;
  ThreadScope(ThreadScope&&) = delete;
  ThreadScope& operator=(ThreadScope&&) = delete;
  TraceState* const prev;
};

// End the capture, read the kernel nodes back and record the launches. Call
// after the host returned.
TORCH_CUDA_CPP_API void finish_trace(TraceState* s);

// A traced tensor's root: its storage base address as a value, the element
// size, whether the root is an allocation of the traced call, and the tape's
// name of the root. Every traced tensor and every view of one is registered.
TORCH_CUDA_CPP_API void register_root(
    const at::TensorBase& t,
    c10::SymInt root,
    int64_t itemsize,
    bool allocation,
    std::string name);
// Drop a traced tensor's storage so data_ptr() / storage() raise with a
// message that names the sym_*_data_ptr() accessors.
TORCH_CUDA_CPP_API void drop_storage(const at::TensorBase& t);
// The next host-order sequence number (shared with the launches).
TORCH_CUDA_CPP_API int64_t next_seq();

// The byte address of a tensor's data, storage offset included, as a
// c10::SymInt. Trace mode: the tensor's root plus its symbolic storage offset
// times the element size; a real tensor met inside a trace is a constant
// address. Ordinary mode: the concrete address as an inline SymInt, no
// allocation. Two forms, mirroring TensorBase: the const form reads through
// const_data_ptr() and leaves a copy-on-write tensor lazy (use it for every
// input the kernel only reads); the mutable form reads through
// mutable_data_ptr() and materializes, as the original host's data_ptr() did
// (use it for outputs and in-place operands). Under a trace both return the
// same value: constness is a property of the ordinary path.
// Upstream placement (not done here): sym_const_data_ptr_custom /
// sym_mutable_data_ptr_custom virtuals on c10::TensorImpl whose defaults
// return the concrete address, and the accessors on at::TensorBase next to
// sym_sizes().
TORCH_CUDA_CPP_API c10::SymInt sym_const_data_ptr(const at::TensorBase& t);
TORCH_CUDA_CPP_API c10::SymInt sym_mutable_data_ptr(const at::TensorBase& t);
// The typed forms: the scalar-type check of const_data_ptr<T>() /
// mutable_data_ptr<T>() (a host that read a T* keeps it), then the address.
template <class T>
c10::SymInt sym_const_data_ptr(const at::TensorBase& t) {
  TORCH_CHECK(
      t.scalar_type() == c10::CppTypeToScalarType<T>::value,
      "expected scalar type ",
      c10::CppTypeToScalarType<T>::value,
      " but found ",
      t.scalar_type());
  return sym_const_data_ptr(t);
}
template <class T>
c10::SymInt sym_mutable_data_ptr(const at::TensorBase& t) {
  TORCH_CHECK(
      t.scalar_type() == c10::CppTypeToScalarType<T>::value,
      "expected scalar type ",
      c10::CppTypeToScalarType<T>::value,
      " but found ",
      t.scalar_type());
  return sym_mutable_data_ptr(t);
}
// `address % n == 0` on a symbolic address: the alignment test a host writes
// instead of reading the pointer's bits. A guard when the caller branches on
// it.
inline c10::SymBool aligned(const c10::SymInt& p, int64_t n) {
  return (p % c10::SymInt(n)).sym_eq(c10::SymInt(0));
}

// A function of ints the host must run on values (a heuristic). Records an
// opaque record of the given kind ("guard": a selector, a different value at
// replay is a miss; "rebind": data, re-evaluated per call). `domain` is what
// the host knows of the result by the function's definition: "int" (any
// integer) or "positive" (>= 1: last_pow2, a block or split count); a
// positive result folds the domain guards of the divisions it feeds, and a
// value outside the declared domain is an error at the trace.
TORCH_CUDA_CPP_API c10::SymInt opaque(
    const std::string& fn,
    std::vector<c10::SymInt> args,
    int64_t (*impl)(const std::vector<int64_t>&),
    const char* kind = "guard",
    const char* domain = "int");
// The same, for a function whose trace-time inputs are not evaluable on their
// hints (a lookup keyed by an input's base address, which the trace stands in
// for): the host supplies the value it obtained itself; replay evaluates
// `impl` on the real arguments. Outside a trace returns `traced_value` without
// copying anything, so a host may record several fields of one lookup per call.
TORCH_CUDA_CPP_API c10::SymInt opaque(
    const char* fn,
    c10::ArrayRef<c10::SymInt> args,
    int64_t (*impl)(const std::vector<int64_t>&),
    const char* kind,
    int64_t traced_value,
    const char* domain = "int");
// The philox increment a host hands to the generator: the concrete count the
// generator API needs, and in trace mode the value on the tape. Declared
// once per random launch, before it; the launch that follows must receive a
// philox state through an `rng` field (a proxy's PhiloxCudaState). The tape's
// increment is the sum; each launch's intragraph offset becomes the prefix
// sum before it (Tape.h RngSlotRec). finish_trace declines a host that
// consumed offsets without declaring them, declared without a random launch
// following, or handed a kernel a philox state without a declaration.
TORCH_CUDA_CPP_API int64_t rng_increment(const c10::SymInt& v);

// cudaMemsetAsync(dst, value, nbytes, stream) on a traced allocation (the
// semaphore reset of a split reduction). Ordinary mode: the call. Trace mode:
// a MemsetRec on the tape and no call, since the destination has no storage;
// the replay's build pairs the memset node of its ordinary capture with the
// record by order and updates it per call.
TORCH_CUDA_CPP_API void memset_async(
    const c10::SymInt& dst,
    int value,
    const c10::SymInt& nbytes,
    cudaStream_t stream);

// Grid dimensions as SymInts: `Grid g(M); kernel<<<g, ...>>>`. The conversion
// to dim3 hands the values to the launch record and the hints to the real
// launch.
struct TORCH_CUDA_CPP_API Grid {
  c10::SymInt x{1}, y{1}, z{1};
  Grid() = default;
  Grid(c10::SymInt x_, c10::SymInt y_ = 1, c10::SymInt z_ = 1)
      : x(std::move(x_)), y(std::move(y_)), z(std::move(z_)) {}
  operator dim3() const;
};

// Decline unless `stream` is the trace's capturing stream (the same capture
// id). A kernel launched on any other stream from the capturing thread is not
// captured: it executes for real with the traced tensors' placeholder
// addresses, and nothing can undo that afterwards. The typed launch checks
// the stream it was given. A verbatim launch checks the current stream when
// its Grid or proxy is converted; the contract is that a verbatim launch
// targets the current stream (the one the trace made the capturing stream,
// or a stream forked from it with an event). A verbatim launch that names
// another stream explicitly cannot be seen before it runs: it is outside the
// contract, the frontier read that closes its record finds no new node on
// the current stream and declines, and the placeholder addresses make it
// fault rather than touch real memory. A host that must launch on a stream
// other than the current one uses the typed launch, which checks the stream
// it is given.
TORCH_CUDA_CPP_API void require_capturing_stream(
    TraceState* s,
    cudaStream_t stream,
    const char* what);

// A by-value struct proxy converted for a verbatim launch (Field.h calls this).
TORCH_CUDA_CPP_API void pending_proxy(
    const void* pod,
    size_t size,
    const std::vector<FieldRec>* fields);
// Register a typed launch record (Launch.h) before the launch reaches the
// driver (this closes the verbatim record before it, whose node is read from
// the frontier the typed launch is about to replace); typed_launched, right
// after cudaLaunchKernel on `stream`, stores the node the capture created.
TORCH_CUDA_CPP_API void typed_launch(TraceState* s, LaunchPacket&& pk);
TORCH_CUDA_CPP_API void typed_launched(TraceState* s, cudaStream_t stream);
// Test hook: finish_trace reads the captured nodes back in reverse order.
// The pairing must not depend on that order; a test asserts it does not.
TORCH_CUDA_CPP_API void test_reverse_node_order(bool on);
TORCH_CUDA_CPP_API bool test_reverse_nodes();
// Test hook: finish_trace throws an unclassified error right after
// capture_end returned, as capture_end itself can once the driver has ended
// the capture (a generator or node readback failure). A test asserts the
// scope closing behind it does not end the capture a second time.
TORCH_CUDA_CPP_API void test_fail_capture_end(bool on);

// Parameter layout of a host kernel symbol from the driver: (offset, size) per
// parameter and the image size.
struct FuncInfo {
  std::string name;
  std::vector<std::pair<size_t, size_t>> params;
  size_t image_size = 0;
};
TORCH_CUDA_CPP_API const FuncInfo& func_info(const void* host_func);

// Ordinary-mode allocation log for the replay's build: (addr, nbytes) of every
// caching-allocator allocation served from `pool`, the build capture's private
// pool, on `device` between begin and end, in order. The allocator routes an
// allocation to that pool exactly when its stream is in the capture (the
// build stream, or a side stream forked from it), so the log holds the
// captured call's allocations and none of another thread's. The log reads
// the pool id off the allocator's trace entry and never queries a stream:
// the tracker runs on the allocating thread, and cudaStreamGetCaptureInfo on
// another thread's stream while that thread ends its capture faults inside
// the driver (580.126.20 / CUDA 13.0, measured).
TORCH_CUDA_CPP_API void alloc_log_begin(
    at::DeviceIndex device,
    c10::cuda::MempoolId_t pool);
TORCH_CUDA_CPP_API std::vector<std::pair<uint64_t, uint64_t>> alloc_log_end();

} // namespace at::cuda::host_trace
