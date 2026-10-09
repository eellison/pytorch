// Records the launches of an ATen op's host code run over c10::SymInt
// (torch/cuda/_host_trace_tape.py) instead of launching them.
#pragma once
#include <ATen/core/TensorBase.h>
#include <c10/core/SymInt.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
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

// a launch's grid or block; a SymInt or an integer is one-dimensional
struct SymDim3 {
  SymDim3(c10::SymInt x, c10::SymInt y = 1, c10::SymInt z = 1) : x(std::move(x)), y(std::move(y)), z(std::move(z)) {}
  SymDim3(int64_t x) : x(x), y(1), z(1) {}
  c10::SymInt x, y, z;
};

struct KernelRecord {
  cudaFunction_t function;
  std::vector<size_t> offsets;
  std::vector<std::vector<uint8_t>> params;
  std::vector<Field> fields;
  SymDim3 grid{0};
  SymDim3 block{0};
  c10::SymInt smem;
};

// a device-to-device cudaMemcpyAsync of `bytes` from src to dst
struct MemcpyRecord {
  c10::SymInt dst;
  c10::SymInt src;
  c10::SymInt bytes;
};

struct Recorder {
  virtual ~Recorder() = default;
  // the byte address of the tensor's data: its root's symbol plus its storage offset
  virtual c10::SymInt data_ptr(const TensorBase& t) = 0;
  // x.bit_length() and 2**x for x >= 0, as the program's bitlength and lshift rows
  virtual c10::SymInt bit_length(const c10::SymInt& x) = 0;
  virtual c10::SymInt pow2(const c10::SymInt& x) = 0;
  // static_cast<float>(a) / static_cast<float>(b)'s bits, as the program's f32div row
  virtual c10::SymInt f32_div(const c10::SymInt& a, const c10::SymInt& b) = 0;
  // `c ? a : b`, as the program's select row
  virtual c10::SymInt select(const c10::SymBool& c, const c10::SymInt& a, const c10::SymInt& b) = 0;
  std::vector<std::variant<KernelRecord, MemcpyRecord>> launches;
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
  K& value() {
    return *reinterpret_cast<K*>(bytes.data());
  }
  // a scalar or pointer member of value(): a constant is written, a symbolic
  // value recorded (the launch packs a 1- or 4-byte field as a signed integer)
  template <class T>
  void set(T& member, const c10::SymInt& v) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 4 || sizeof(T) == 8);
    if (auto c = v.maybe_as_int()) {
      if constexpr (std::is_pointer_v<T>) {
        member = reinterpret_cast<T>(static_cast<uintptr_t>(*c));
      } else {
        member = static_cast<T>(*c);
      }
      return;
    }
    fields.push_back({0, offset_of(member), sizeof(T), v, std::is_pointer_v<T>});
  }
  // a member of value() given by its bits, e.g. a float's
  template <class T>
  void set_bits(T& member, const c10::SymInt& bits) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8);
    if (auto c = bits.maybe_as_int()) {
      using bits_t = std::conditional_t<sizeof(T) == 4, int32_t, int64_t>;
      const auto b = static_cast<bits_t>(*c);
      std::memcpy(static_cast<void*>(&member), &b, sizeof(T));
      return;
    }
    fields.push_back({0, offset_of(member), sizeof(T), bits, false});
  }
  // a member of value() given as a Param of its own
  template <class T>
  void set(T& member, const Param<T>& v) {
    std::memcpy(static_cast<void*>(&member), v.bytes.data(), sizeof(T));
    const size_t offset = offset_of(member);
    for (Field f : v.fields) {
      f.offset += offset;
      fields.push_back(std::move(f));
    }
  }

 private:
  template <class T>
  size_t offset_of(T& member) {
    return reinterpret_cast<uint8_t*>(&member) - bytes.data();
  }
};

template <class K>
Param<K> scalar_param(const c10::SymInt& v) {
  Param<K> p;
  p.set(p.value(), v);
  return p;
}

// `kernel<<<grid, block, smem>>>(params...)`, recorded; parameters are laid out
// at their natural alignment, as the driver's cuFuncGetParamInfo reports
template <class... KArgs>
void launch(Recorder& rec, void (*kernel)(KArgs...), SymDim3 grid, SymDim3 block, c10::SymInt smem, const Param<KArgs>&... params) {
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
  r.block = std::move(block);
  r.smem = std::move(smem);
  rec.launches.push_back(std::move(r));
}

} // namespace at::cuda::host_trace
