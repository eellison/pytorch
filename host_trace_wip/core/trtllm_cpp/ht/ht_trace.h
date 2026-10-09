// FI_HT_TRACED: the aliases of fi_ht.h as symbolic values, and the names the patched launcher calls unqualified,
// declared in namespace fi_ht_traced so that inside it they record instead of launching: CUtensorMap and
// cuTensorMapEncodeTiled, CUlaunchConfig and cuLaunchKernelEx, cudaLaunchConfig_t and cudaLaunchKernelEx,
// AlignedAllocator, TensorView, Optional, Variant. ht_traced.cu includes the stock headers first (global
// namespace, FlashInfer's own code), then this file, then the patched sources inside namespace fi_ht_traced.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "ht_types.h"

namespace fi_ht {
using sz = SymInt;

template <class T>
using ptr = Ptr<T>;

template <class P, class T>
inline Ptr<std::remove_pointer_t<P>> ptr_cast(const Ptr<T>& p) {
  return Ptr<std::remove_pointer_t<P>>(p.a);
}
inline Ptr<void> const_cast_void(const Ptr<void const>& p) { return Ptr<void>(p.a); }
template <class N>
inline Ptr<void const> byte_offset(const Ptr<void const>& p, const N& n) {
  return Ptr<void const>(p.a + SymInt(n));
}
template <class T>
inline bool aligned(const Ptr<T>& p, int64_t alignment) {
  return p.a % alignment == 0;
}
inline SymInt sym_size(const TensorView& t, int64_t i) { return t.sym_size(i); }
inline SymInt sym_numel(const TensorView& t) { return t.t.sym_numel(); }
inline std::string to_string(const SymInt& v) {
  std::ostringstream os;
  os << v;
  return os.str();
}
// min/max of sizes: sym_min/sym_max expressions (no guard on which side wins)
template <class T>
inline SymInt min(const SymInt& a, const SymInt& b) {
  return a.min(b);
}
template <class T>
inline SymInt max(const SymInt& a, const SymInt& b) {
  return a.max(b);
}
void loading(char const* name);

// ---- KernelParams: a proxy member per POD member (gen_fields.py), packed at the launch

struct Packer {
  std::vector<uint8_t> image;
  std::vector<FieldRec> fields;
  std::vector<TmaRec> tmas;
  template <class T>
  void put(size_t off, const T& v) {
    std::memcpy(image.data() + off, &v, sizeof(T));
  }
};

// cuTensorMapEncodeTiled's operands
struct TensorMap {
  bool set = false;
  int dtype = 0;
  SymInt address;
  std::vector<SymInt> shape;
  std::vector<SymInt> strides;
  std::vector<uint32_t> box;
  int swizzle = 0;
  int fill = 0;
};

template <class T, size_t Off, class Enable = void>
struct Field {  // a member of another type (cuda::fast_mod_div): its bytes
  alignas(T) unsigned char b[sizeof(T)] = {};
  Field& operator=(const T& v) {
    std::memcpy(b, &v, sizeof(T));
    return *this;
  }
  void pack(Packer& p) const { std::memcpy(p.image.data() + Off, b, sizeof(T)); }
};

template <class T>
constexpr bool is_int_member = (std::is_integral_v<T> && !std::is_same_v<T, bool>) || std::is_enum_v<T>;

template <class T, size_t Off>
struct Field<T, Off, std::enable_if_t<is_int_member<T>>> : SymInt {
  Field() : SymInt(0) {}
  Field& operator=(const SymInt& v) {
    SymInt::operator=(v);
    return *this;
  }
  template <class U, std::enable_if_t<std::is_arithmetic_v<U> || std::is_enum_v<U>, int> = 0>
  Field& operator=(U v) {
    SymInt::operator=(SymInt(static_cast<int64_t>(static_cast<T>(v))));
    return *this;
  }
  void pack(Packer& p) const {
    if (auto c = maybe_as_int()) {
      p.put(Off, static_cast<T>(*c));
    } else {
      // the member's own width holds it: a value the stock build would wrap is a guard that fails, not a replay
      using L = std::numeric_limits<std::conditional_t<std::is_enum_v<T>, int32_t, T>>;
      if (!(*this >= static_cast<int64_t>(L::min()) && *this <= static_cast<int64_t>(std::min<uint64_t>(L::max(), INT64_MAX)))) {
        decline("a KernelParams member out of its type's range");
      }
      p.fields.emplace_back(0, Off, sizeof(T), *this, false);
    }
  }
};

template <size_t Off>
struct Field<bool, Off> {
  bool v = false;
  Field& operator=(bool x) {
    v = x;
    return *this;
  }
  operator bool() const { return v; }
  void pack(Packer& p) const { p.put(Off, v); }
};

template <class T, size_t Off>
struct Field<T, Off, std::enable_if_t<std::is_floating_point_v<T>>> {
  T v = 0;
  Field& operator=(double x) {
    v = static_cast<T>(x);
    return *this;
  }
  operator T() const { return v; }
  void pack(Packer& p) const { p.put(Off, v); }
};

template <class T, size_t Off>
struct Field<T*, Off> {
  Ptr<T> p;
  template <class U>
  Field& operator=(const Ptr<U>& x) {
    p = Ptr<T>(x.a);
    return *this;
  }
  Field& operator=(std::nullptr_t) {
    p = Ptr<T>();
    return *this;
  }
  operator Ptr<T>() const { return p; }
  template <class N>
  Ptr<T> operator+(const N& n) const {
    return p + n;
  }
  void pack(Packer& pk) const {
    if (auto c = p.a.maybe_as_int()) {
      pk.put(Off, static_cast<uint64_t>(*c));
    } else {
      pk.fields.emplace_back(0, Off, sizeof(T*), p.a, true);
    }
  }
};

template <size_t Off>
struct Field<CUtensorMap, Off> {
  TensorMap t;
  Field& operator=(const TensorMap& x) {
    t = x;
    return *this;
  }
  void pack(Packer& p) const {
    if (t.set) {
      p.tmas.emplace_back(0, Off, t.dtype, t.address, t.shape, t.strides, t.box, t.swizzle, t.fill);
    }
  }
};

// the live KernelParams proxies, by address: cuLaunchKernelEx receives `void* list[] = {&kernelParams}`
std::unordered_set<const void*>& live_proxies();

template <class Self>
struct ProxyBase {
  using fi_ht_proxy = void;
  ProxyBase() { live_proxies().insert(this); }
  ProxyBase(const ProxyBase&) { live_proxies().insert(this); }
  ProxyBase& operator=(const ProxyBase&) { return *this; }
  ~ProxyBase() { live_proxies().erase(this); }
};

}  // namespace fi_ht
// KernelParamsFields: one Field per ::KernelParams member (generated from DWARF by gen_fields.py)
#include "kernel_params_fields.h"
namespace fi_ht {

template <class T, class = void>
struct is_proxy : std::false_type {};
template <class T>
struct is_proxy<T, std::void_t<typename T::fi_ht_proxy>> : std::true_type {};

// ---- launches

struct LaunchAttribute {
  CUlaunchAttributeID id{};
  struct {
    struct {
      SymInt x{1}, y{1}, z{1};
    } clusterDim;
    CUclusterSchedulingPolicy clusterSchedulingPolicyPreference{};
    int programmaticStreamSerializationAllowed = 0;
    int sharedMemoryMode = 0;
  } value;
};

struct LaunchConfig {
  SymInt gridDimX{1}, gridDimY{1}, gridDimZ{1};
  SymInt blockDimX{1}, blockDimY{1}, blockDimZ{1};
  SymInt sharedMemBytes{0};
  CUstream hStream = nullptr;
  LaunchAttribute* attrs = nullptr;
  unsigned numAttrs = 0;
};

struct SymDim3 {
  SymInt x{1}, y{1}, z{1};
  SymDim3& operator=(const SymInt& v) {
    x = v;
    y = 1;
    z = 1;
    return *this;
  }
};

struct RtLaunchAttribute {
  cudaLaunchAttributeID id{};
  struct {
    int programmaticStreamSerializationAllowed = 0;
  } val;
};

struct RtLaunchConfig {
  SymDim3 gridDim;
  SymDim3 blockDim;
  SymInt dynamicSmemBytes{0};
  cudaStream_t stream = nullptr;
  RtLaunchAttribute* attrs = nullptr;
  unsigned numAttrs = 0;
};

std::string function_name(CUfunction f);
cudaStream_t current_stream();
// the stock twin of a kernel of this build (the same source in FlashInfer's own module): its function and name
std::pair<uint64_t, std::string> stock_kernel(const std::string& traced_name);
void check_stream(CUstream s);

template <class K, class A>
void add_param(LaunchRec& r, size_t& end, const A& a) {
  end = (end + alignof(K) - 1) / alignof(K) * alignof(K);
  r.offsets.push_back(end);
  end += sizeof(K);
  std::vector<uint8_t> bytes(sizeof(K), 0);
  const size_t at = r.params.size();
  if constexpr (std::is_pointer_v<K>) {
    if constexpr (std::is_same_v<A, std::nullptr_t>) {
      r.fields.emplace_back(at, 0, sizeof(K), SymInt(0), false);
    } else {
      r.fields.emplace_back(at, 0, sizeof(K), a.a, !a.a.maybe_as_int().has_value());
    }
  } else if constexpr (std::is_integral_v<K>) {
    SymInt v(a);
    if (!v.maybe_as_int() && !(v >= static_cast<int64_t>(std::numeric_limits<K>::min()) &&
                               v <= static_cast<int64_t>(std::min<uint64_t>(std::numeric_limits<K>::max(), INT64_MAX)))) {
      decline("a kernel argument out of its type's range");
    }
    r.fields.emplace_back(at, 0, sizeof(K), v, false);
  } else {
    K v = static_cast<K>(a);
    std::memcpy(bytes.data(), &v, sizeof(K));
  }
  r.params.push_back(std::move(bytes));
}

}  // namespace fi_ht

namespace fi_ht_traced {
// the stock namespaces' names stay visible, qualified or not, below the shadows declared here
namespace flashinfer {
using namespace ::flashinfer;
}
namespace tensorrt_llm {
using namespace ::tensorrt_llm;
namespace kernels {
using namespace ::tensorrt_llm::kernels;
}
}  // namespace tensorrt_llm

using TensorView = fi_ht::TensorView;
template <class T>
using Optional = std::optional<T>;
template <class A, class B>
using Variant = fi_ht::ScaleArg;
using CUtensorMap = fi_ht::TensorMap;
using CUlaunchConfig = fi_ht::LaunchConfig;
using CUlaunchAttribute = fi_ht::LaunchAttribute;
using cudaLaunchConfig_t = fi_ht::RtLaunchConfig;
using cudaLaunchAttribute = fi_ht::RtLaunchAttribute;

using ::memset;
template <class T, std::enable_if_t<fi_ht::is_proxy<T>::value, int> = 0>
void* memset(T* p, int, size_t) {
  *p = T();
  return p;
}

inline cudaStream_t get_stream(const fi_ht::Device&) { return fi_ht::current_stream(); }

CUresult cuTensorMapEncodeTiled(fi_ht::TensorMap* m, CUtensorMapDataType dtype, unsigned rank, fi_ht::Ptr<void> address,
                                const fi_ht::SymInt* shape, const fi_ht::SymInt* strides, const uint32_t* box,
                                const uint32_t* element_strides, CUtensorMapInterleave interleave,
                                CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2, CUtensorMapFloatOOBfill fill);
CUresult cuLaunchKernelEx(const fi_ht::LaunchConfig* config, CUfunction f, void** params, void** extra);
CUresult cuOccupancyMaxActiveClusters(int* n, CUfunction f, const fi_ht::LaunchConfig* config);

template <class... K, class... A>
cudaError_t cudaLaunchKernelEx(const fi_ht::RtLaunchConfig* c, void (*kernel)(K...), A&&... args) {
  static_assert(sizeof...(K) == sizeof...(A));
  fi_ht::check_stream(c->stream);
  fi_ht::LaunchRec r;
  cudaFunction_t f = nullptr;
  if (cudaGetFuncBySymbol(&f, reinterpret_cast<const void*>(kernel)) != cudaSuccess) {
    fi_ht::decline("cudaGetFuncBySymbol failed");
  }
  // the launch is the stock module's kernel, as eager's is
  std::tie(r.function, r.name) = fi_ht::stock_kernel(fi_ht::function_name(reinterpret_cast<CUfunction>(f)));
  size_t end = 0;
  (fi_ht::add_param<K>(r, end, args), ...);
  r.grid[0] = c->gridDim.x, r.grid[1] = c->gridDim.y, r.grid[2] = c->gridDim.z;
  r.block[0] = c->blockDim.x, r.block[1] = c->blockDim.y, r.block[2] = c->blockDim.z;
  r.smem = fi_ht::guard_int(c->dynamicSmemBytes);
  for (unsigned i = 0; i < c->numAttrs; i++) {
    if (c->attrs[i].id == cudaLaunchAttributeProgrammaticStreamSerialization) {
      r.pdl = c->attrs[i].val.programmaticStreamSerializationAllowed != 0;
    } else {
      fi_ht::decline("a runtime launch attribute other than PDL");
    }
  }
  fi_ht::recorder()->launches.push_back(std::move(r));
  return cudaSuccess;
}

// AlignedAllocator over the workspace's (symbolic) address: an aligned base is a guard, so std::align moves nothing
struct AlignedAllocator {
  fi_ht::Ptr<void> cur;
  fi_ht::SymInt remaining;
  AlignedAllocator(fi_ht::Ptr<void> base, fi_ht::SymInt size) : cur(base), remaining(std::move(size)) {}
  template <class T>
  fi_ht::Ptr<T> aligned_alloc(const fi_ht::SymInt& size, size_t alignment, const std::string& name) {
    if (cur.null() || !(cur.a % static_cast<int64_t>(alignment) == 0)) {
      fi_ht::decline("workspace allocation " + name + " at an address not " + std::to_string(alignment) + "-aligned");
    }
    if (size > remaining) {
      fi_ht::decline("workspace allocation " + name + " does not fit the workspace");
    }
    fi_ht::Ptr<T> r(cur.a);
    cur.a = cur.a + size;
    remaining = remaining - size;
    return r;
  }
};

// utils.cuh's UpPowerOfTwo: 2 ** (x - 1).bit_length()
inline fi_ht::SymInt UpPowerOfTwo(const fi_ht::SymInt& x) {
  if (auto c = x.maybe_as_int()) {
    return ::flashinfer::UpPowerOfTwo(static_cast<int>(*c));
  }
  return fi_ht::hooks.pow2(fi_ht::hooks.bit_length(x - 1));
}
}  // namespace fi_ht_traced
