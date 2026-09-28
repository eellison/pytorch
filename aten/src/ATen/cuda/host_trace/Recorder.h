// The kernel launches of an ATen op's traced host: the op's host code run with
// c10::SymInt sizes and addresses under a host trace (torch/cuda/_host_trace_tape.py)
// records its launches here instead of launching them. A record is the kernel,
// each parameter's bytes and the symbolic fields over those bytes; the trace
// turns it into a KernelLaunch.
#pragma once
#include <ATen/core/TensorBase.h>
#include <c10/core/SymInt.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

namespace at::cuda::host_trace {

// a symbolic value at a byte offset of a parameter
struct Field {
  size_t param;
  size_t offset;
  size_t width;
  c10::SymInt value;
  bool pointer;
};

struct KernelRecord {
  cudaFunction_t function;
  std::vector<size_t> offsets;
  std::vector<std::vector<uint8_t>> params;
  std::vector<Field> fields;
  c10::SymInt grid;
  int64_t block;
  int64_t smem;
};

struct Recorder {
  virtual ~Recorder() = default;
  // the byte address of the tensor's data: its root's symbol plus its storage offset
  virtual c10::SymInt data_ptr(const TensorBase& t) = 0;
  // x.bit_length() and 2**x for x >= 0, as the program's bitlength and lshift rows
  virtual c10::SymInt bit_length(const c10::SymInt& x) = 0;
  virtual c10::SymInt pow2(const c10::SymInt& x) = 0;
  std::vector<KernelRecord> launches;
};

// a case the traced host does not reproduce; the trace records the op as an eager call
[[noreturn]] inline void decline(const std::string& why) {
  C10_THROW_ERROR(NotImplementedError, "host_trace: " + why);
}

// One kernel parameter of type K as the kernel receives it. Zeroed, so padding
// and empty members are the same bytes at every trace.
template <class K>
struct Param {
  static constexpr size_t align = alignof(K);
  alignas(K) std::array<uint8_t, sizeof(K)> bytes{};
  std::vector<Field> fields;

  Param() = default;
  explicit Param(const K& v) {
    new (bytes.data()) K(v);
  }
  K& pod() {
    return *reinterpret_cast<K*>(bytes.data());
  }
  // a scalar or pointer member of pod(): a constant is written, a symbolic
  // value recorded (the launch packs a 4-byte field as a signed int32)
  template <class T>
  void set(T& member, const c10::SymInt& v) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8);
    if (auto c = v.maybe_as_int()) {
      if constexpr (std::is_pointer_v<T>) {
        member = reinterpret_cast<T>(static_cast<uintptr_t>(*c));
      } else {
        member = static_cast<T>(*c);
      }
      return;
    }
    const size_t offset = reinterpret_cast<uint8_t*>(&member) - bytes.data();
    fields.push_back({0, offset, sizeof(T), v, std::is_pointer_v<T>});
  }
};

template <class K>
Param<K> scalar(const c10::SymInt& v) {
  Param<K> p;
  p.set(p.pod(), v);
  return p;
}

// `kernel<<<grid, block, smem>>>(params...)`, recorded; parameters are laid out
// at their natural alignment, as the driver's cuFuncGetParamInfo reports
template <class... KArgs>
void launch(Recorder& rec, void (*kernel)(KArgs...), c10::SymInt grid, int64_t block, int64_t smem, const Param<KArgs>&... params) {
  KernelRecord r;
  C10_CUDA_CHECK(cudaGetFuncBySymbol(&r.function, reinterpret_cast<const void*>(kernel)));
  size_t end = 0;
  auto add = [&](const auto& p) {
    end = (end + p.align - 1) / p.align * p.align;
    r.offsets.push_back(end);
    end += p.bytes.size();
    for (Field f : p.fields) {
      f.param = r.params.size();
      r.fields.push_back(std::move(f));
    }
    r.params.emplace_back(p.bytes.begin(), p.bytes.end());
  };
  (add(params), ...);
  r.grid = std::move(grid);
  r.block = block;
  r.smem = smem;
  rec.launches.push_back(std::move(r));
}

} // namespace at::cuda::host_trace
