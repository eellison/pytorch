// Host-to-device data in a traceable host.
//
//     HostTable<const void*> ptrs(3 * G);         // a typed pinned table
//     ptrs[g] = sym_const_data_ptr<float>(A[g]);  // pointer elements are traced addresses
//     HostTable<int32_t> sizes(G);
//     sizes[g] = A[g].sym_size(0);                // ints: symbolic or literal
//     at::Tensor dev = at::empty({nbytes}, options.dtype(at::kByte));
//     copy_h2d(sym_mutable_data_ptr(dev), ptrs, ptrs.nbytes(), stream); // in-graph H2D copy
//     copy_h2d(sym_mutable_data_ptr(ids_dev), ids, B * 8, stream);  // a pinned CPU INPUT as source
//
// Ordinary mode: the table is a pinned buffer from the caching host
// allocator, element writes are plain stores, and copy_h2d is a
// cudaMemcpyAsync on the stream plus the host allocator's record_event (what
// Tensor::copy_ does), so the buffer is not reused before the copy has read
// it. Trace mode: each copy_h2d puts the table on the tape as a HostBufferRec
// (root hb<k>, one element record per slot written so far, each a value over
// the tape's symbols: the table as that copy read it) and a MemcpyRec
// {src, dst, bytes} whose src is that image's root; nothing is issued on the
// capture stream, since the destination is a storage-less allocation with no
// address yet. A table copied twice is two images, so each copy keeps its
// place in host order and its own bytes. The replay's build pairs the memcpy
// nodes of ITS capture of the ordinary host with the records by order and
// checks src, dst and bytes; per call it renders each image into a staging
// slot and updates a node only when an operand moved. A copy must be issued
// on the trace's capturing stream itself: the record has no stream field, so
// a copy on a forked side stream declines (HostTable.cpp).
//
// The typed guarantee: a pointer element must be an address the trace
// created (sym_*_data_ptr of a traced input or a host allocation). A raw
// address, a constant, or an expression with no traced root declines at trace
// time. A raw cudaMemcpyAsync in a host leaves a memcpy node the tape has no
// record for, which finish_trace declines by node type. The synchronous
// cudaMemcpy is a host-contract violation caught by the HOSTTRACE_SYNC_API
// lint (Recorder.h).
//
// A device-to-device copy (the cudaMemcpyAsync Copy.cu's copy_ issues for a
// contiguous pair of one dtype) goes through copy_d2d: the same MemcpyRec
// {src, dst, bytes} with both addresses over traced roots, nothing issued
// under the trace, the replay's memcpy node re-pointed per call.
#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/cuda/host_trace/Recorder.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace at::cuda::host_trace {

struct HostTableBase;

// Asynchronous H2D copy of `nbytes` from a host table to device memory at
// `dst` (a sym_mutable_data_ptr() value, possibly displaced), on `stream`.
TORCH_CUDA_CPP_API void copy_h2d(
    const c10::SymInt& dst,
    const HostTableBase& src,
    const c10::SymInt& nbytes,
    c10::cuda::CUDAStream stream);
// The same from a pinned CPU tensor: a traced pinned input under the trace,
// a real pinned tensor in ordinary mode. The tape's input record carries the
// device and pinned-ness, so a replay refuses a pageable or device-resident
// source before touching any node; the byte count is guarded to
// [1, numel * itemsize] of the source view. The ordinary overload refuses the
// same things: a device or pageable source, a host destination.
TORCH_CUDA_CPP_API void copy_h2d(
    const c10::SymInt& dst,
    const at::TensorBase& src,
    const c10::SymInt& nbytes,
    c10::cuda::CUDAStream stream);
// Asynchronous device-to-device copy of `nbytes` from `src` to `dst`, both
// sym_*_data_ptr() values (possibly displaced), on `stream`. Ordinary mode:
// cudaMemcpyAsync. Trace mode: a MemcpyRec on the tape and no call, since
// neither side has storage; both addresses must be values the trace created
// (a raw address declines), the byte count is guarded to at least one, and
// the copy must be on the trace's own capturing stream like copy_h2d.
TORCH_CUDA_CPP_API void copy_d2d(
    const c10::SymInt& dst,
    const c10::SymInt& src,
    const c10::SymInt& nbytes,
    c10::cuda::CUDAStream stream);

struct TORCH_CUDA_CPP_API HostTableBase {
  int64_t count;
  size_t elem_size;
  const char* kind;
  bool is_ptr;
  bool is_float;
  std::string name;
  at::Tensor pinned; // ordinary mode: the pinned buffer
  bool traced = false; // built under a trace
  std::vector<uint8_t> hint; // trace mode: the values at the traced call
  // trace mode: the element records the next copy puts on the tape (last
  // write per slot) and their root hb<k>, minted on first use and again after
  // every copy so that each copy of the table reads an image of its own
  std::vector<FieldRec> elements;
  std::unordered_map<size_t, size_t> slots; // element offset -> element index
  mutable std::optional<c10::SymInt> root;

  HostTableBase(
      int64_t n,
      size_t elem_size,
      const char* kind,
      bool is_ptr,
      bool is_float,
      std::string name);
  HostTableBase(const HostTableBase&) = delete;
  HostTableBase& operator=(const HostTableBase&) = delete;

  // The table's byte address: the pinned buffer (ordinary), the root of the
  // image the next copy records (trace).
  c10::SymInt data() const;
  c10::SymInt nbytes() const {
    return c10::SymInt(count * static_cast<int64_t>(elem_size));
  }
  void put(int64_t i, const c10::SymInt& v); // int and pointer elements
  void put_float(int64_t i, const c10::SymFloat& v);
  void put_raw_ptr(int64_t i, const void* p); // ordinary: the address; trace: declined
  // trace mode: a copy of no bytes or of more bytes than the table holds
  // declines; the bounds [1, capacity] are recorded as guards on the tape
  void check_fits(const c10::SymInt& nbytes) const;

 private:
  void record(size_t off, SymVal v);
  // trace mode: the table as this copy reads it, on the tape at the next
  // host-order position; returns the image's root and re-mints the table's
  c10::SymInt image(TraceState* s) const;
  // the pinned-tensor copy reads the byte count's hint the way the table does
  static int64_t hint_of(const c10::SymInt& v);
  friend void copy_h2d(
      const c10::SymInt& dst,
      const HostTableBase& src,
      const c10::SymInt& nbytes,
      c10::cuda::CUDAStream stream);
  friend void copy_h2d(
      const c10::SymInt& dst,
      const at::TensorBase& src,
      const c10::SymInt& nbytes,
      c10::cuda::CUDAStream stream);
  friend void copy_d2d(
      const c10::SymInt& dst,
      const c10::SymInt& src,
      const c10::SymInt& nbytes,
      c10::cuda::CUDAStream stream);
};

template <class T>
constexpr const char* table_kind() {
  using U = std::remove_cv_t<T>;
  if constexpr (std::is_pointer_v<U>) {
    return "ptr";
  } else if constexpr (std::is_same_v<U, float>) {
    return "f32";
  } else if constexpr (std::is_same_v<U, double>) {
    return "f64";
  } else if constexpr (sizeof(U) == 8) {
    return std::is_unsigned_v<U> ? "u64" : "i64";
  } else if constexpr (sizeof(U) == 4) {
    return std::is_unsigned_v<U> ? "u32" : "i32";
  } else if constexpr (sizeof(U) == 2) {
    return "i16";
  } else {
    return "u8";
  }
}

template <class T>
struct HostTable : HostTableBase {
  static_assert(
      std::is_integral_v<T> || std::is_pointer_v<T> ||
          std::is_floating_point_v<T>,
      "HostTable<T>: T must be an integer, a pointer or a float type");
  explicit HostTable(int64_t n, std::string name = "table")
      : HostTableBase(
            n,
            sizeof(T),
            table_kind<T>(),
            std::is_pointer_v<T>,
            std::is_floating_point_v<T>,
            std::move(name)) {}

  struct Ref {
    HostTable* t;
    int64_t i;
    Ref& operator=(const c10::SymInt& v) {
      t->put(i, v);
      return *this;
    }
    template <
        class U,
        std::enable_if_t<std::is_integral_v<U> || std::is_enum_v<U>, int> = 0>
    Ref& operator=(U v) {
      t->put(i, c10::SymInt(static_cast<int64_t>(v)));
      return *this;
    }
    template <class U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    Ref& operator=(const c10::SymFloat& v) {
      t->put_float(i, v);
      return *this;
    }
    template <class U = T, std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    Ref& operator=(double v) {
      t->put_float(i, c10::SymFloat(v));
      return *this;
    }
    template <class U>
    Ref& operator=(U* p) {
      t->put_raw_ptr(i, static_cast<const void*>(p));
      return *this;
    }
    Ref& operator=(std::nullptr_t) {
      t->put(i, c10::SymInt(0));
      return *this;
    }
  };
  Ref operator[](int64_t i) {
    return Ref{this, i};
  }
};

// Ordinary-mode log of the table copies a host issued between begin and end,
// in order: the pinned buffer each copy read and its bytes at that moment.
// The replay's build checks the bytes against the tape's image and binds the
// image's hb<k> to the buffer its capture copied from.
struct HostTableCopyLog {
  at::Tensor buffer;
  std::vector<uint8_t> bytes;
};
TORCH_CUDA_CPP_API void host_table_log_begin();
TORCH_CUDA_CPP_API std::vector<HostTableCopyLog> host_table_log_end();

} // namespace at::cuda::host_trace
