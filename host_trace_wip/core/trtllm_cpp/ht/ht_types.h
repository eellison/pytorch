// The host-trace build's value types, shared by the traced launcher (ht_traced.cu) and its pybind binding
// (ht_bind.cpp): symbolic sizes and addresses, the tensor view the verbatim bindings read, and the launch records
// a call leaves.
#pragma once
#include <ATen/core/Tensor.h>
#include <c10/core/SymInt.h>
#include <cuda.h>
#include <dlpack/dlpack.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

namespace fi_ht {
using SymInt = c10::SymInt;
using sint = SymInt;
using s32 = SymInt;
using s64 = SymInt;
using u64 = SymInt;

// the call is not one this build reproduces: the binding raises NotImplementedError("fi_ht: ...")
[[noreturn]] void decline(const std::string& why);

// Python-backed operations the binding installs
struct Hooks {
  SymInt (*data_ptr)(const at::Tensor&) = nullptr;
  SymInt (*bit_length)(const SymInt&) = nullptr;  // x.bit_length() for x >= 0
  SymInt (*pow2)(const SymInt&) = nullptr;        // 2 ** x for x >= 0
};
extern Hooks hooks;

inline int64_t guard_int(const SymInt& v) { return v.guard_int(__FILE__, __LINE__); }

template <class T>
constexpr int64_t elt_size() {
  if constexpr (std::is_void_v<std::remove_cv_t<T>>) {
    return 1;
  } else {
    return sizeof(T);
  }
}

// A device address: a traced tensor's (symbolic), nullptr (0), or a constant. Null is a concrete 0, so a traced
// tensor's address is not null without a guard (and a memset struct of them is all null, as the original's).
template <class T>
struct Ptr {
  SymInt a{0};

  Ptr() = default;
  Ptr(std::nullptr_t) {}
  explicit Ptr(SymInt address) : a(std::move(address)) {}
  template <class U, std::enable_if_t<std::is_convertible_v<U*, T*>, int> = 0>
  Ptr(const Ptr<U>& o) : a(o.a) {}

  bool null() const {
    auto c = a.maybe_as_int();
    return c && *c == 0;
  }
  bool operator==(std::nullptr_t) const { return null(); }
  bool operator!=(std::nullptr_t) const { return !null(); }
  template <class U>
  bool operator==(const Ptr<U>& o) const {
    return a - o.a == 0;
  }
  template <class N>
  Ptr operator+(const N& n) const {
    return Ptr(a + SymInt(n) * elt_size<T>());
  }
};

template <class T>
std::ostream& operator<<(std::ostream& os, const Ptr<T>& p) {
  return p.null() ? os << "nullptr" : os << p.a;
}

struct Device {
  int32_t device_type;
  int32_t device_id;
};

// tvm::ffi::TensorView's surface over a (traced) tensor: sizes and strides read as ints specialize (a guard),
// sym_size stays symbolic; the address is the traced tensor's.
struct TensorView {
  at::Tensor t;
  TensorView(at::Tensor x) : t(std::move(x)) {}
  int32_t ndim() const { return static_cast<int32_t>(t.dim()); }
  int64_t size(int64_t i) const { return guard_int(t.sym_size(i)); }
  SymInt sym_size(int64_t i) const { return t.sym_size(i); }
  int64_t stride(int64_t i) const { return guard_int(t.sym_stride(i)); }
  int64_t numel() const { return guard_int(t.sym_numel()); }
  bool IsContiguous() const { return t.sym_is_contiguous().guard_bool(__FILE__, __LINE__); }
  Ptr<void> data_ptr() const { return Ptr<void>(hooks.data_ptr(t)); }
  Device device() const { return {kDLCUDA, static_cast<int32_t>(t.device().index())}; }
  DLDataType dtype() const {
    switch (t.scalar_type()) {
      case at::kFloat:
        return {kDLFloat, 32, 1};
      case at::kHalf:
        return {kDLFloat, 16, 1};
      case at::kBFloat16:
        return {kDLBfloat, 16, 1};
      case at::kFloat8_e4m3fn:
        return {kDLFloat8_e4m3fn, 8, 1};
      case at::kFloat8_e5m2:
        return {kDLFloat8_e5m2, 8, 1};
      case at::kByte:
        return {kDLUInt, 8, 1};
      case at::kChar:
        return {kDLInt, 8, 1};
      case at::kInt:
        return {kDLInt, 32, 1};
      case at::kLong:
        return {kDLInt, 64, 1};
      default:
        decline(std::string("a tensor of dtype ") + c10::toString(t.scalar_type()));
    }
  }
};

inline int64_t get_element_size(const TensorView& x) { return x.t.element_size(); }

// Variant<double, Tensor>: a scale given as a value or a device tensor
struct ScaleArg {
  std::optional<double> value;
  std::optional<TensorView> tensor;
  template <class T>
  auto as() const {
    if constexpr (std::is_same_v<T, double>) {
      return value;
    } else {
      return tensor;
    }
  }
};

// ---- the records a traced call leaves

// a symbolic value at a byte offset of a parameter (records are built with constructors, never braced lists: the
// cudafe bug of tests/test_nvcc_vector_init.cu)
struct FieldRec {
  size_t param;
  size_t offset;
  size_t width;
  SymInt value;
  bool pointer;
  FieldRec(size_t param_, size_t offset_, size_t width_, SymInt value_, bool pointer_)
      : param(param_), offset(offset_), width(width_), value(std::move(value_)), pointer(pointer_) {}
};

// a CUtensorMap parameter member: cuTensorMapEncodeTiled's operands (element strides 1, no interleave, L2 128B)
struct TmaRec {
  size_t param;
  size_t offset;
  int dtype;
  SymInt address;
  std::vector<SymInt> shape;    // every dimension's extent
  std::vector<SymInt> strides;  // in bytes, every dimension but the first
  std::vector<uint32_t> box;
  int swizzle;
  int fill;
  TmaRec(size_t param_, size_t offset_, int dtype_, SymInt address_, std::vector<SymInt> shape_,
         std::vector<SymInt> strides_, std::vector<uint32_t> box_, int swizzle_, int fill_)
      : param(param_), offset(offset_), dtype(dtype_), address(std::move(address_)), shape(std::move(shape_)),
        strides(std::move(strides_)), box(std::move(box_)), swizzle(swizzle_), fill(fill_) {}
};

struct LaunchRec {
  uint64_t function = 0;
  std::string name;
  std::vector<std::vector<uint8_t>> params;  // each parameter's bytes (constants written, symbolic members zero)
  std::vector<size_t> offsets;               // each parameter's offset in the argument buffer
  std::vector<FieldRec> fields;
  std::vector<TmaRec> tmas;
  SymInt grid[3];
  SymInt block[3];
  int64_t smem = 0;
  int64_t cluster[3] = {1, 1, 1};
  int policy = 0;  // CUclusterSchedulingPolicy
  bool pdl = false;
  bool packed = false;  // one struct parameter (cuLaunchKernelEx with a params struct)
};

struct Recorder {
  std::vector<LaunchRec> launches;
  int64_t real_launches = 0;  // launches that reached the driver during the call: must stay 0
};
Recorder*& recorder();
}  // namespace fi_ht
