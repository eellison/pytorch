// Proxy for a by-value kernel parameter struct P. A checked-in
// `template <> struct Traced<P> : TracedBase` declares one same-named member
// per field of P at the field's offset (write it by hand, generate it from the
// declaration with clang or DWARF, or, once the toolchain has C++26 static
// reflection, synthesize it; size and offset static_asserts keep it honest).
//
// A field owns no value: the POD is the storage. On the ordinary path a write
// is one store into the POD and a read is one load, so a converted host costs
// what the unconverted one did. Under a trace a write from a symbolic value
// stores the hint into the POD and records (name, offset, size, kind, value)
// in the proxy's field table, and a read returns the recorded value (or the
// POD's value as a constant when the field was never written symbolically).
// Reading an int field yields a c10::SymInt; a pointer field takes the
// c10::SymInt of sym_const_data_ptr() / sym_mutable_data_ptr(); converting the proxy to `P&` (what
// `kernel<<<...>>>(params)` does) hands the field table to the launch record.
// A host declares `using P = Traced<P>;` in the scope of its host functions,
// after the __global__ definitions, and changes nothing else.
#pragma once
#include <ATen/cuda/host_trace/Recorder.h>

#include <cstring>
#include <type_traits>
#include <unordered_map>

namespace at::cuda::host_trace {

struct TracedBase {
  void* pod;
  size_t pod_size;
  // the trace this proxy lives in, nullptr on the ordinary path: read once
  // here so the field accesses do not pay a thread-local lookup each
  TraceState* const trace;
  // trace mode only: the recorded fields, by offset
  std::vector<FieldRec> fields;
  std::unordered_map<size_t, size_t> index; // offset -> position in fields

  TracedBase(void* pod, size_t size)
      : pod(pod), pod_size(size), trace(active()) {}
  TracedBase(const TracedBase&) = delete;
  TracedBase& operator=(const TracedBase&) = delete;

  template <class T>
  T load(size_t off) const {
    T v;
    std::memcpy(&v, static_cast<const char*>(pod) + off, sizeof(T));
    return v;
  }
  template <class T>
  void store(size_t off, const T& v) {
    std::memcpy(static_cast<char*>(pod) + off, &v, sizeof(T));
  }

  void record(
      const char* name,
      size_t off,
      size_t size,
      const char* kind,
      SymVal v) {
    FieldRec r{off, size, kind, std::move(v), name, ""};
    auto it = index.find(off);
    if (it == index.end()) {
      index[off] = fields.size();
      fields.push_back(std::move(r));
    } else {
      fields[it->second] = std::move(r);
    }
  }
  const SymVal* recorded(size_t off) const {
    auto it = index.find(off);
    return it == index.end() ? nullptr : &fields[it->second].v;
  }
  void on_convert() const {
    if (trace != nullptr) {
      pending_proxy(pod, pod_size, &fields);
    }
  }
  // `params = {};` in a host: the POD is zeroed and every record dropped
  void clear_records() {
    fields.clear();
    index.clear();
  }
};

// The ordinary-path reads: the inline range of a SymInt / a plain SymFloat is
// the whole ordinary path; expect_int() does not inline (its check is a cold
// TORCH_CHECK).
inline int64_t plain_int(const c10::SymInt& v) {
  return C10_LIKELY(!v.is_heap_allocated()) ? v.as_int_unchecked()
                                            : v.expect_int();
}
inline double plain_float(const c10::SymFloat& v) {
  return C10_LIKELY(!v.is_symbolic()) ? v.as_float_unchecked()
                                      : v.expect_float();
}

template <class T>
constexpr const char* int_kind() {
  return sizeof(T) == 8 ? "i64"
      : sizeof(T) == 4  ? "i32"
      : sizeof(T) == 2  ? "i16"
                        : "u8";
}

template <class X>
struct is_int_field : std::false_type {};
template <class T, size_t Off>
struct is_int_field<IntField<T, Off>> : std::true_type {};

// The other operand of an int field's arithmetic or comparison: another int
// field, a c10::SymInt, an integer or an enum (the operands c10::SymInt itself
// accepts).
template <class X>
inline constexpr bool int_operand_v = is_int_field<std::decay_t<X>>::value ||
    std::is_same_v<std::decay_t<X>, c10::SymInt> ||
    ((std::is_integral_v<std::decay_t<X>> || std::is_enum_v<std::decay_t<X>>) &&
     !std::is_same_v<std::decay_t<X>, bool>);

struct FieldOps {
  // the plain value of an operand when the expression can be evaluated as
  // ints (the ordinary path, a constant SymInt), else false
  template <class X>
  static bool plain(const X& x, int64_t& out) {
    using D = std::decay_t<X>;
    if constexpr (std::is_same_v<D, c10::SymInt>) {
      if (C10_LIKELY(!x.is_heap_allocated())) {
        out = x.as_int_unchecked();
        return true;
      }
      return false;
    } else if constexpr (std::is_integral_v<D> || std::is_enum_v<D>) {
      out = static_cast<int64_t>(x);
      return true;
    } else {
      return x.plain(out);
    }
  }
  template <class X>
  static c10::SymInt sym(const X& x) {
    using D = std::decay_t<X>;
    if constexpr (std::is_same_v<D, c10::SymInt>) {
      return x;
    } else if constexpr (std::is_integral_v<D> || std::is_enum_v<D>) {
      return c10::SymInt(static_cast<int64_t>(x));
    } else {
      return x.sym();
    }
  }
};

// Arithmetic between an int field and an int operand yields a c10::SymInt
// (plain ints compute inline, as c10::SymInt's own operators do); a comparison
// yields a bool (a guard under a trace, as c10::SymInt's do).
#define HOST_TRACE_INT_FIELD_ARITH(op)                                       \
  template <class U, std::enable_if_t<int_operand_v<U>, int> = 0>            \
  friend c10::SymInt operator op(const IntField& a, const U& b) {            \
    int64_t x = 0, y = 0;                                                    \
    if (C10_LIKELY(a.plain(x) && FieldOps::plain(b, y))) {                   \
      return c10::SymInt(x op y);                                            \
    }                                                                        \
    return a.sym() op FieldOps::sym(b);                                      \
  }                                                                          \
  template <                                                                 \
      class U,                                                               \
      std::enable_if_t<int_operand_v<U> && !is_int_field<U>::value, int> = 0> \
  friend c10::SymInt operator op(const U& a, const IntField& b) {            \
    int64_t x = 0, y = 0;                                                    \
    if (C10_LIKELY(FieldOps::plain(a, x) && b.plain(y))) {                   \
      return c10::SymInt(x op y);                                            \
    }                                                                        \
    return FieldOps::sym(a) op b.sym();                                      \
  }
#define HOST_TRACE_INT_FIELD_CMP(op)                                         \
  template <class U, std::enable_if_t<int_operand_v<U>, int> = 0>            \
  friend bool operator op(const IntField& a, const U& b) {                   \
    int64_t x = 0, y = 0;                                                    \
    if (C10_LIKELY(a.plain(x) && FieldOps::plain(b, y))) {                   \
      return x op y;                                                         \
    }                                                                        \
    return a.sym() op FieldOps::sym(b);                                      \
  }                                                                          \
  template <                                                                 \
      class U,                                                               \
      std::enable_if_t<int_operand_v<U> && !is_int_field<U>::value, int> = 0> \
  friend bool operator op(const U& a, const IntField& b) {                   \
    int64_t x = 0, y = 0;                                                    \
    if (C10_LIKELY(FieldOps::plain(a, x) && b.plain(y))) {                   \
      return x op y;                                                         \
    }                                                                        \
    return FieldOps::sym(a) op b.sym();                                      \
  }

// Integer member.
template <class T, size_t Off>
struct IntField {
  TracedBase* o;
  const char* name;
  IntField(TracedBase* o, const char* name) : o(o), name(name) {}
  IntField(const IntField&) = delete;

  IntField& operator=(const c10::SymInt& v) {
    if (C10_LIKELY(o->trace == nullptr)) {
      o->store<T>(Off, static_cast<T>(plain_int(v)));
      return *this;
    }
    assign_traced(v);
    return *this;
  }
  template <
      class U,
      std::enable_if_t<std::is_integral_v<U> || std::is_enum_v<U>, int> = 0>
  IntField& operator=(U v) {
    if (C10_LIKELY(o->trace == nullptr)) {
      o->store<T>(Off, static_cast<T>(v));
      return *this;
    }
    assign_traced(c10::SymInt(static_cast<int64_t>(v)));
    return *this;
  }
  IntField& operator=(const IntField& other) {
    return operator=(other.sym());
  }
  // compound assignment goes through operator= so the POD and the record
  // follow the value
  template <class U, std::enable_if_t<int_operand_v<U>, int> = 0>
  IntField& operator+=(const U& v) {
    return operator=(*this + v);
  }
  template <class U, std::enable_if_t<int_operand_v<U>, int> = 0>
  IntField& operator-=(const U& v) {
    return operator=(*this - v);
  }
  template <class U, std::enable_if_t<int_operand_v<U>, int> = 0>
  IntField& operator*=(const U& v) {
    return operator=(*this * v);
  }

  bool plain(int64_t& out) const {
    if (C10_LIKELY(o->trace == nullptr)) {
      out = static_cast<int64_t>(o->template load<T>(Off));
      return true;
    }
    return false;
  }
  c10::SymInt sym() const {
    if (C10_LIKELY(o->trace == nullptr)) {
      return c10::SymInt(static_cast<int64_t>(o->template load<T>(Off)));
    }
    return sym_traced();
  }
  operator c10::SymInt() const {
    return sym();
  }
  void reset() {}

  HOST_TRACE_INT_FIELD_ARITH(+)
  HOST_TRACE_INT_FIELD_ARITH(-)
  HOST_TRACE_INT_FIELD_ARITH(*)
  HOST_TRACE_INT_FIELD_ARITH(/)
  HOST_TRACE_INT_FIELD_ARITH(%)
  HOST_TRACE_INT_FIELD_CMP(==)
  HOST_TRACE_INT_FIELD_CMP(!=)
  HOST_TRACE_INT_FIELD_CMP(<)
  HOST_TRACE_INT_FIELD_CMP(<=)
  HOST_TRACE_INT_FIELD_CMP(>)
  HOST_TRACE_INT_FIELD_CMP(>=)

 private:
  C10_NOINLINE void assign_traced(const c10::SymInt& v) {
    o->store<T>(Off, static_cast<T>(Hints::of(v)));
    o->record(name, Off, sizeof(T), int_kind<T>(), SymVal::of(v));
  }
  C10_NOINLINE c10::SymInt sym_traced() const {
    if (const SymVal* r = o->recorded(Off)) {
      return r->i;
    }
    return c10::SymInt(static_cast<int64_t>(o->template load<T>(Off)));
  }
};

#undef HOST_TRACE_INT_FIELD_ARITH
#undef HOST_TRACE_INT_FIELD_CMP

// bool member: a plain value (host booleans are constants of the variant).
template <size_t Off>
struct BoolField {
  TracedBase* o;
  const char* name;
  BoolField(TracedBase* o, const char* name) : o(o), name(name) {}
  BoolField(const BoolField&) = delete;
  BoolField& operator=(bool b) {
    const uint8_t raw = b ? 1 : 0;
    o->store<uint8_t>(Off, raw);
    if (o->trace != nullptr) {
      o->record(name, Off, 1, "u8", SymVal::of_int(raw));
    }
    return *this;
  }
  BoolField& operator=(const BoolField& other) {
    return operator=(static_cast<bool>(other));
  }
  operator bool() const {
    return o->template load<uint8_t>(Off) != 0;
  }
  void reset() {}
};

// float / double member. Reading it as a plain double in trace mode records a
// guard, so symbolic float arithmetic should stay on the SymFloat side.
template <class T, size_t Off>
struct FloatField {
  TracedBase* o;
  const char* name;
  FloatField(TracedBase* o, const char* name) : o(o), name(name) {}
  FloatField(const FloatField&) = delete;
  FloatField& operator=(const c10::SymFloat& v) {
    if (C10_LIKELY(o->trace == nullptr)) {
      o->store<T>(Off, static_cast<T>(plain_float(v)));
      return *this;
    }
    assign_traced(v);
    return *this;
  }
  FloatField& operator=(double v) {
    if (C10_LIKELY(o->trace == nullptr)) {
      o->store<T>(Off, static_cast<T>(v));
      return *this;
    }
    assign_traced(c10::SymFloat(v));
    return *this;
  }
  FloatField& operator=(float v) {
    return operator=(static_cast<double>(v));
  }
  FloatField& operator=(const FloatField& other) {
    return operator=(other.sym());
  }
  // the symbolic value, for arithmetic between two fields (FloatField *
  // FloatField would otherwise resolve to the double conversions)
  c10::SymFloat sym() const {
    if (C10_LIKELY(o->trace == nullptr)) {
      return c10::SymFloat(static_cast<double>(o->template load<T>(Off)));
    }
    return sym_traced();
  }
  operator c10::SymFloat() const {
    return sym();
  }
  operator double() const {
    if (C10_LIKELY(o->trace == nullptr)) {
      return static_cast<double>(o->template load<T>(Off));
    }
    return double_traced();
  }
  void reset() {}

 private:
  C10_NOINLINE void assign_traced(const c10::SymFloat& v) {
    o->store<T>(Off, static_cast<T>(Hints::of(v)));
    o->record(
        name, Off, sizeof(T), sizeof(T) == 8 ? "f64" : "f32", SymVal::of(v));
  }
  C10_NOINLINE c10::SymFloat sym_traced() const {
    if (const SymVal* r = o->recorded(Off)) {
      return r->f;
    }
    return c10::SymFloat(static_cast<double>(o->template load<T>(Off)));
  }
  C10_NOINLINE double double_traced() const {
    const c10::SymFloat val = sym_traced();
    if (val.is_symbolic()) {
      const double h = Hints::of(val);
      // a plain read pins the value: recorded as a guard, never silently
      TORCH_CHECK(
          val.sym_eq(c10::SymFloat(h)).guard_bool(__FILE__, __LINE__),
          "host_trace: float field read disagrees with its hint");
      return h;
    }
    return val.as_float_unchecked();
  }
};

// Pointer member: assigned from the c10::SymInt of sym_*_data_ptr() or from a
// raw T* / nullptr. A traced tensor has no raw pointer, so a raw non-null
// pointer in trace mode is a real address, a constant of the variant.
template <size_t Off>
struct PtrField {
  TracedBase* o;
  const char* name;
  PtrField(TracedBase* o, const char* name) : o(o), name(name) {}
  PtrField(const PtrField&) = delete;
  PtrField& operator=(const c10::SymInt& v) {
    if (C10_LIKELY(o->trace == nullptr)) {
      o->store<uint64_t>(Off, static_cast<uint64_t>(plain_int(v)));
      return *this;
    }
    assign_traced(v);
    return *this;
  }
  template <class U>
  PtrField& operator=(U* v) {
    const uint64_t raw = reinterpret_cast<uintptr_t>(
        const_cast<void*>(static_cast<const volatile void*>(v)));
    if (C10_LIKELY(o->trace == nullptr)) {
      o->store<uint64_t>(Off, raw);
      return *this;
    }
    assign_traced(c10::SymInt(static_cast<int64_t>(raw)));
    return *this;
  }
  PtrField& operator=(std::nullptr_t) {
    return operator=(c10::SymInt(0));
  }
  PtrField& operator=(const PtrField& other) {
    return operator=(other.sym());
  }
  c10::SymInt sym() const {
    if (C10_LIKELY(o->trace == nullptr)) {
      return c10::SymInt(
          static_cast<int64_t>(o->template load<uint64_t>(Off)));
    }
    return sym_traced();
  }
  operator c10::SymInt() const {
    return sym();
  }
  // `params.x_ptr != nullptr`: a symbolic address is a traced tensor's and
  // never null; only the constant 0 is. Answered without a guard.
  bool is_null() const {
    if (C10_LIKELY(o->trace == nullptr)) {
      return o->template load<uint64_t>(Off) == 0;
    }
    const c10::SymInt p = sym_traced();
    return !p.is_heap_allocated() && p.as_int_unchecked() == 0;
  }
  explicit operator bool() const {
    return !is_null();
  }
  friend bool operator==(const PtrField& f, std::nullptr_t) {
    return f.is_null();
  }
  friend bool operator!=(const PtrField& f, std::nullptr_t) {
    return !f.is_null();
  }
  void reset() {}

 private:
  C10_NOINLINE void assign_traced(const c10::SymInt& v) {
    o->store<uint64_t>(Off, static_cast<uint64_t>(Hints::of(v)));
    o->record(name, Off, 8, "ptr", SymVal::of(v));
  }
  C10_NOINLINE c10::SymInt sym_traced() const {
    if (const SymVal* r = o->recorded(Off)) {
      return r->i;
    }
    return c10::SymInt(static_cast<int64_t>(o->template load<uint64_t>(Off)));
  }
};

// Generator-state member (at::PhiloxCudaState): raw bytes the host writes by
// placement-new through operator void*(); recorded as kind "rng", which no
// replayer writes or compares.
template <size_t Off, size_t Size>
struct BytesField {
  TracedBase* o;
  const char* name;
  BytesField(TracedBase* o, const char* name) : o(o), name(name) {}
  BytesField(const BytesField&) = delete;
  operator void*() const {
    if (o->trace != nullptr) {
      o->record(name, Off, Size, "rng", SymVal::of_int(0));
    }
    return static_cast<char*>(o->pod) + Off;
  }
  void reset() {}
};

template <class P>
struct Traced; // specialized by the checked-in proxy headers

} // namespace at::cuda::host_trace
