// Runtime-offset proxy slots for by-value kernel parameter blocks that hold
// arrays and nested structs (TensorIterator's OffsetCalculator, the
// std::array of data pointers). Field.h's members carry their offset as a
// template argument, which fits a flat struct; arrays need element access by
// index, so these views compute the offset at access time. Same recording
// contract as Field.h: an assignment writes the POD (always) and records
// (name, offset, size, kind, value) in trace mode.
#pragma once
#include <ATen/cuda/host_trace/Launch.h>

#include <cstring>
#include <string>
#include <vector>

namespace at::cuda::host_trace::ti {

// Name of a slot: a chain of (parent, member | [index]) rendered to "a.b[i].c"
// only when a record is written; ordinary mode never builds a string.
struct SlotName {
  const SlotName* parent = nullptr;
  const char* member = nullptr;
  long index = -1;
  std::string str() const {
    std::string s = parent ? parent->str() : std::string();
    if (member) {
      if (!s.empty()) {
        s += ".";
      }
      s += member;
    }
    if (index >= 0) {
      s += "[" + std::to_string(index) + "]";
    }
    return s;
  }
};

inline const SymVal& recorded(const TracedBase* o, size_t off) {
  auto it = o->index.find(off);
  TORCH_CHECK(it != o->index.end(), "host_trace: proxy field read before set at offset ", off);
  return o->fields[it->second].v;
}

// One slot write: the typed launch's put_arg (the recorder's own writer, the
// only one besides the field types allowed a hint) puts the value the kernel
// receives into the POD and, under a trace, describes it; the description is
// then recorded under the slot's name.
template <class T, class V>
void slot_write(TracedBase* o, size_t off, const V& v, const SlotName& nm) {
  const bool tracing = active() != nullptr;
  std::vector<FieldRec> rec;
  launch_detail::put_arg<T>(*reinterpret_cast<T*>(static_cast<char*>(o->pod) + off), v, off, tracing ? &rec : nullptr, "", 0);
  if (tracing) {
    TORCH_INTERNAL_ASSERT(rec.size() == 1);
    o->record(nm.str().c_str(), off, sizeof(T), rec[0].kind.c_str(), rec[0].v);
  }
}

template <class T>
struct IntSlot {
  TracedBase* o;
  size_t off;
  SlotName nm;
  IntSlot(TracedBase* o, size_t off, SlotName nm) : o(o), off(off), nm(nm) {}
  IntSlot& operator=(const c10::SymInt& v) {
    slot_write<T>(o, off, v, nm);
    return *this;
  }
  template <class U, std::enable_if_t<std::is_integral_v<U> || std::is_enum_v<U>, int> = 0>
  IntSlot& operator=(U v) {
    operator=(c10::SymInt(static_cast<int64_t>(v)));
    return *this;
  }
  operator c10::SymInt() const {
    if (active() != nullptr) {
      return recorded(o, off).i;
    }
    T raw;
    std::memcpy(&raw, static_cast<const char*>(o->pod) + off, sizeof(T));
    return c10::SymInt(static_cast<int64_t>(raw));
  }
};

// Pointer slot: assigned from the c10::SymInt of sym_*_data_ptr() or a
// TracedArray element; a raw pointer is a real address, a constant.
struct PtrSlot {
  TracedBase* o;
  size_t off;
  SlotName nm;
  PtrSlot(TracedBase* o, size_t off, SlotName nm) : o(o), off(off), nm(nm) {}
  PtrSlot& operator=(const c10::SymInt& v) {
    slot_write<char*>(o, off, v, nm);
    return *this;
  }
  PtrSlot& operator=(std::nullptr_t) {
    operator=(c10::SymInt(0));
    return *this;
  }
};

template <class T>
struct FloatSlot {
  TracedBase* o;
  size_t off;
  SlotName nm;
  FloatSlot(TracedBase* o, size_t off, SlotName nm) : o(o), off(off), nm(nm) {}
  FloatSlot& operator=(const c10::SymFloat& v) {
    slot_write<T>(o, off, v, nm);
    return *this;
  }
};

// Fixed array of slots or nested views, one Elem(o, off, name) per access.
template <class Elem, size_t Stride, size_t N>
struct ArrayOf {
  TracedBase* o;
  size_t off;
  SlotName nm;
  ArrayOf(TracedBase* o, size_t off, SlotName nm) : o(o), off(off), nm(nm) {}
  Elem operator[](size_t i) const {
    TORCH_CHECK(i < N, "host_trace: proxy array index ", i, " out of ", N);
    return Elem(o, off + i * Stride, SlotName{&nm, nullptr, static_cast<long>(i)});
  }
  template <class A, std::enable_if_t<!std::is_same_v<std::decay_t<A>, ArrayOf>, int> = 0>
  ArrayOf& operator=(const A& src) {
    for (size_t i = 0; i < N; ++i) {
      (*this)[i] = src[i];
    }
    return *this;
  }
};

} // namespace at::cuda::host_trace::ti
