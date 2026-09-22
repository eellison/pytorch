#include <ATen/cuda/host_trace/HostTable.h>

#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <ATen/core/CachingHostAllocator.h>
#include <ATen/cuda/host_trace/Hooks.h>
#include <c10/cuda/CUDAException.h>

#include <cstring>

namespace at::cuda::host_trace {

namespace {

void store_int(uint8_t* dst, size_t size, int64_t v) {
  std::memcpy(dst, &v, size); // little-endian: the low `size` bytes
}

void record_memcpy(
    TraceState* s,
    const c10::SymInt& src,
    const c10::SymInt& dst,
    const c10::SymInt& nbytes,
    const char* kind) {
  s->t->memcpys.push_back(MemcpyRec{s->t->seq++, src, dst, nbytes, kind});
}

// A copy record has no stream field: the replay pairs the memcpy nodes of
// its capture with the records by order, and a consumer lowering the tape
// issues the copies in host order on one stream. The capture itself would
// accept a copy on a side stream forked from the capturing stream and joined
// back (the launch rule, require_capturing_stream), but the record would not
// say which branch the copy was on, so its ordering against the other
// branch's work would be lost at replay. Until the tape carries stream
// identity (O33 in DECISIONS.md: no per-stream tape field for now), a copy
// must be on the trace's own capturing stream: an active capture with the
// trace's id, and that exact stream, as the runtime team's memset path checks.
void require_copy_stream(TraceState* s, cudaStream_t stream) {
  require_capturing_stream(s, stream, "a copy");
  if (stream != s->capture_stream) {
    decline(
        "host_trace: a copy on a side stream joined to the trace's capture: "
        "the tape's copy record has no stream, so a replay would lose the "
        "copy's ordering against the other branch; copy on the trace's own "
        "stream (declined)");
  }
}

// The destination of an H2D copy is device memory: the same rule the trace
// applies to the destination's root (a CUDA input or a host allocation).
void check_device_dst(const void* dst) {
  cudaPointerAttributes attr{};
  C10_CUDA_CHECK(cudaPointerGetAttributes(&attr, dst));
  TORCH_CHECK(
      attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged,
      "host_trace: copy_h2d destination must be device memory; got a ",
      attr.type == cudaMemoryTypeHost ? "host" : "non-device",
      " address");
}

// Returns whether the caching host allocator owns the source block.
bool ordinary_copy(
    void* dst,
    const void* src,
    size_t n,
    void* ctx,
    c10::cuda::CUDAStream stream) {
  check_device_dst(dst);
  C10_CUDA_CHECK(
      cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream.stream()));
  // Tensor::copy_'s rule (Copy.cu): the pinned block must not be handed out
  // again before this stream has read it; the caching host allocator defers
  // its reuse to an event on the stream.
  return at::getHostAllocator(at::kCUDA)->record_event(
      const_cast<void*>(src), ctx, stream.unwrap());
}
} // namespace

HostTableBase::HostTableBase(
    int64_t n,
    size_t esz,
    const char* k,
    bool ptr,
    bool flt,
    std::string nm)
    : count(n),
      elem_size(esz),
      kind(k),
      is_ptr(ptr),
      is_float(flt),
      name(std::move(nm)) {
  TORCH_CHECK(n >= 0, "host_trace: HostTable needs a non-negative element count");
  const int64_t total = n * static_cast<int64_t>(esz);
  if (active() == nullptr) {
    pinned = at::empty(
        {std::max<int64_t>(total, 1)},
        at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    return;
  }
  // nothing on the tape yet: each copy records the table as it reads it
  hint.assign(static_cast<size_t>(std::max<int64_t>(total, 1)), 0);
  traced = true;
}

int64_t HostTableBase::hint_of(const c10::SymInt& v) {
  return Hints::of(v);
}

c10::SymInt HostTableBase::data() const {
  if (active() == nullptr) {
    return c10::SymInt(
        static_cast<int64_t>(reinterpret_cast<uintptr_t>(pinned.data_ptr())));
  }
  if (!root.has_value()) {
    root = hooks::mint_int_symbol(
        static_cast<int64_t>(reinterpret_cast<uintptr_t>(hint.data())),
        c10::str("host_table ", name));
  }
  return *root;
}

void HostTableBase::record(size_t off, SymVal v) {
  FieldRec f{off, elem_size, kind, std::move(v), "", ""};
  auto it = slots.find(off);
  if (it == slots.end()) {
    slots[off] = elements.size();
    elements.push_back(std::move(f));
  } else {
    elements[it->second] = std::move(f);
  }
}

c10::SymInt HostTableBase::image(TraceState* s) const {
  HostBufferRec rec;
  // the copy's place in host order, after every element write before it, so
  // a replay renders the image once the symbols it uses (the allocations
  // whose addresses the host wrote) are bound
  rec.seq = s->t->seq++;
  rec.name = name;
  rec.root = data();
  rec.nbytes = count * static_cast<int64_t>(elem_size);
  rec.elements = elements;
  s->t->host_buffers.push_back(std::move(rec));
  // the next copy of this table, rewritten or not, is an image of its own
  root.reset();
  return s->t->host_buffers.back().root;
}

void HostTableBase::put(int64_t i, const c10::SymInt& v) {
  TORCH_CHECK(
      i >= 0 && i < count,
      "host_trace: HostTable '",
      name,
      "' element ",
      i,
      " out of range [0, ",
      count,
      ")");
  const size_t off = static_cast<size_t>(i) * elem_size;
  TraceState* s = active();
  if (s == nullptr) {
    store_int(
        static_cast<uint8_t*>(pinned.data_ptr()) + off,
        elem_size,
        v.expect_int());
    return;
  }
  if (is_ptr && !v.is_heap_allocated()) {
    // the typed guarantee: a pointer element is an address the trace created;
    // a constant is an address the host obtained some other way
    decline(c10::str(
        "host_trace: host table '",
        name,
        "': pointer element ",
        i,
        " is a constant address (",
        v.as_int_unchecked(),
        "), not a traced one; addresses come from sym_const_data_ptr(tensor) "
        "(declined)"));
  }
  store_int(hint.data() + off, elem_size, Hints::of(v));
  record(off, SymVal::of(v));
}

void HostTableBase::put_float(int64_t i, const c10::SymFloat& v) {
  TORCH_CHECK(is_float, "host_trace: HostTable '", name, "' is not a float table");
  TORCH_CHECK(
      i >= 0 && i < count,
      "host_trace: HostTable '",
      name,
      "' element ",
      i,
      " out of range [0, ",
      count,
      ")");
  const size_t off = static_cast<size_t>(i) * elem_size;
  TraceState* s = active();
  const double d = s == nullptr ? v.expect_float() : Hints::of(v);
  uint8_t* dst = s == nullptr
      ? static_cast<uint8_t*>(pinned.data_ptr()) + off
      : hint.data() + off;
  if (elem_size == 8) {
    std::memcpy(dst, &d, 8);
  } else {
    const float f = static_cast<float>(d);
    std::memcpy(dst, &f, 4);
  }
  if (s == nullptr) {
    return;
  }
  record(off, SymVal::of(v));
}

void HostTableBase::check_fits(const c10::SymInt& nbytes) const {
  const int64_t capacity = count * static_cast<int64_t>(elem_size);
  const int64_t n = Hints::of(nbytes);
  if (n < 1 || n > capacity) {
    decline(c10::str(
        "host_trace: copy_h2d of ",
        n,
        " bytes from the ",
        capacity,
        "-byte host table '",
        name,
        "'",
        n < 1 ? "; a copy of no bytes makes no node" : "",
        " (declined)"));
  }
  // On the tape as guards: a replay whose byte count leaves [1, capacity]
  // misses instead of reaching the runtime with a range past the staging slot.
  TORCH_CHECK(
      nbytes.sym_ge(1).guard_bool(__FILE__, __LINE__) &&
      nbytes.sym_le(capacity).guard_bool(__FILE__, __LINE__));
}

void HostTableBase::put_raw_ptr(int64_t i, const void* p) {
  if (active() != nullptr) {
    decline(c10::str(
        "host_trace: host table '",
        name,
        "': pointer element ",
        i,
        " is a raw address, not a traced one; addresses come from "
        "sym_const_data_ptr(tensor) (declined)"));
  }
  put(i, c10::SymInt(static_cast<int64_t>(reinterpret_cast<uintptr_t>(p))));
}

void copy_h2d(
    const c10::SymInt& dst,
    const HostTableBase& src,
    const c10::SymInt& nbytes,
    c10::cuda::CUDAStream stream) {
  TraceState* s = active();
  const int64_t capacity = src.count * static_cast<int64_t>(src.elem_size);
  if (s == nullptr) {
    const int64_t n = nbytes.expect_int();
    TORCH_CHECK(
        n >= 0 && n <= capacity,
        "host_trace: copy_h2d of ",
        n,
        " bytes from a ",
        capacity,
        "-byte host table");
    ordinary_copy(
        reinterpret_cast<void*>(static_cast<uintptr_t>(dst.expect_int())),
        src.pinned.data_ptr(),
        static_cast<size_t>(n),
        src.pinned.storage().data_ptr().get_context(),
        stream);
    return;
  }
  require_copy_stream(s, stream.stream());
  if (!src.traced) {
    decline(
        "host_trace: copy_h2d from a host table built outside the trace "
        "(declined)");
  }
  src.check_fits(nbytes);
  record_memcpy(s, src.image(s), dst, nbytes, "h2d");
}

void copy_h2d(
    const c10::SymInt& dst,
    const at::TensorBase& src,
    const c10::SymInt& nbytes,
    c10::cuda::CUDAStream stream) {
  TraceState* s = active();
  if (s == nullptr) {
    // the ordinary path refuses what the trace refuses: a device-resident or
    // pageable source, a host destination
    TORCH_CHECK(
        src.device().is_cpu(),
        "host_trace: copy_h2d source must be a HostTable or a pinned CPU "
        "tensor; got a ",
        src.device(),
        " tensor");
    const bool owned = ordinary_copy(
        reinterpret_cast<void*>(static_cast<uintptr_t>(dst.expect_int())),
        src.data_ptr(),
        static_cast<size_t>(nbytes.expect_int()),
        src.storage().data_ptr().get_context(),
        stream);
    // a block the caching host allocator does not own is fine when it is
    // pinned some other way (cudaHostRegister); pageable memory makes the
    // copy synchronous and is what the trace declines
    TORCH_CHECK(
        owned || at::globalContext().isPinnedPtr(src.data_ptr()),
        "host_trace: copy_h2d source is in pageable CPU memory; a copy "
        "source must be pinned");
    return;
  }
  require_copy_stream(s, stream.stream());
  if (!src.device().is_cpu()) {
    decline(c10::str(
        "host_trace: copy_h2d source must be a HostTable or a pinned CPU "
        "input of the traced call; got a ",
        src.device(),
        " tensor (declined)"));
  }
  const c10::SymInt addr = sym_const_data_ptr(src);
  if (!addr.is_heap_allocated()) {
    decline(
        "host_trace: copy_h2d source is a CPU tensor the trace did not "
        "create; a copy source must be a pinned CPU input of the traced call "
        "(declined)");
  }
  // the source view's bytes bound the copy: [1, numel * itemsize], hint-time
  // decline and recorded guards, as for a host table
  const c10::SymInt capacity =
      src.sym_numel() * static_cast<int64_t>(src.itemsize());
  const int64_t n = HostTableBase::hint_of(nbytes);
  const int64_t cap = HostTableBase::hint_of(capacity);
  if (n < 1 || n > cap) {
    decline(c10::str(
        "host_trace: copy_h2d of ",
        n,
        " bytes from a pinned input of ",
        cap,
        " bytes",
        n < 1 ? "; a copy of no bytes makes no node" : "",
        " (declined)"));
  }
  TORCH_CHECK(
      nbytes.sym_ge(1).guard_bool(__FILE__, __LINE__) &&
      nbytes.sym_le(capacity).guard_bool(__FILE__, __LINE__));
  record_memcpy(s, addr, dst, nbytes, "h2d");
}

void copy_d2d(
    const c10::SymInt& dst,
    const c10::SymInt& src,
    const c10::SymInt& nbytes,
    c10::cuda::CUDAStream stream) {
  TraceState* s = active();
  if (s == nullptr) {
    C10_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(static_cast<uintptr_t>(dst.expect_int())),
        reinterpret_cast<const void*>(static_cast<uintptr_t>(src.expect_int())),
        static_cast<size_t>(nbytes.expect_int()),
        cudaMemcpyDeviceToDevice,
        stream.stream()));
    return;
  }
  require_copy_stream(s, stream.stream());
  if (!src.is_heap_allocated() || !dst.is_heap_allocated()) {
    decline(
        "host_trace: copy_d2d between addresses the trace did not create (a "
        "raw pointer); both sides must be sym_*_data_ptr() values of traced "
        "tensors (declined)");
  }
  if (HostTableBase::hint_of(nbytes) < 1) {
    decline("host_trace: copy_d2d of no bytes makes no node (declined)");
  }
  TORCH_CHECK(nbytes.sym_ge(1).guard_bool(__FILE__, __LINE__));
  record_memcpy(s, src, dst, nbytes, "d2d");
}

} // namespace at::cuda::host_trace
