// The tape: what one traced host call did. The recorder (Recorder.h) fills it
// with the values the host handed it, as c10::SymInt / SymFloat / SymBool
// handles; their meaning (symbols, expressions, guards) lives in the Python
// ShapeEnv that created them, so nothing here interprets an expression.
//
// Stable surface. Other consumers lower a tape into their own program, so the
// following do not change without agreement: the records below, the
// positional order of a launch's parameters (by image offset) and of the
// launches themselves, the `hint_image` (the argument bytes at the traced
// call), and the recorder's C++ surface: `Traced<T>` (Field.h), `Grid` and
// `Block`, `launch()` (Launch.h), `sym_const_data_ptr()` /
// `sym_mutable_data_ptr()` and the trace scope.
// Nothing reaches the tape that the recorder did not observe. Timings are
// not part of the tape.
#pragma once
#include <c10/core/SymBool.h>
#include <c10/core/SymFloat.h>
#include <c10/core/SymInt.h>
#include <c10/macros/Export.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace at::cuda::host_trace {

// A recorded value: one of the three symbolic scalar types.
struct SymVal {
  enum class Tag : uint8_t { Int, Float, Bool };
  Tag tag = Tag::Int;
  c10::SymInt i{0};
  c10::SymFloat f{0.0};
  c10::SymBool b{false};

  static SymVal of(const c10::SymInt& v) {
    SymVal r;
    r.tag = Tag::Int;
    r.i = v;
    return r;
  }
  static SymVal of(const c10::SymFloat& v) {
    SymVal r;
    r.tag = Tag::Float;
    r.f = v;
    return r;
  }
  static SymVal of(const c10::SymBool& v) {
    SymVal r;
    r.tag = Tag::Bool;
    r.b = v;
    return r;
  }
  static SymVal of_int(int64_t v) {
    return of(c10::SymInt(v));
  }
  static SymVal of_float(double v) {
    return of(c10::SymFloat(v));
  }
};

struct FieldRec {
  size_t offset = 0;
  size_t size = 0;
  std::string kind; // ptr i64 u64 i32 u32 i16 u8 f32 f64 rng
  SymVal v;
  std::string name; // struct field name, empty for a positional argument
  // "r" / "rw" for a pointer parameter, from the kernel's constness; empty
  // when the parameter is not a pointer or the launch was verbatim
  std::string access;
};
struct LaunchRec {
  int64_t seq = 0;
  std::string kernel;
  // the kernel's host symbol (the handle a launch takes) and the driver's
  // parameter layout, (offset, size) per parameter: what a consumer needs to
  // build its own launch records; a name and image spans do not suffice
  const void* func = nullptr;
  std::vector<std::pair<size_t, size_t>> param_layout;
  std::array<c10::SymInt, 3> grid;
  std::array<int64_t, 3> block{}; // the dims at the traced call
  std::array<c10::SymInt, 3> block_expr; // the dims as values
  c10::SymInt smem;
  std::vector<FieldRec> params;
  // the argument image at the traced call; the bytes no param covers are
  // constants of the launch, kept as captured (never assumed to be padding)
  std::vector<uint8_t> hint_image;
};
struct OpaqueRec {
  int64_t seq = 0;
  std::string fn;
  std::vector<c10::SymInt> args;
  int64_t expected = 0;
  c10::SymInt sym;
  // the host's own function: the tape stays in the process, so replay calls it
  // rather than looking it up by name
  int64_t (*impl)(const std::vector<int64_t>&) = nullptr;
  // "guard": a selector, re-evaluated, a different value is a miss;
  // "rebind": data, re-evaluated and bound per call
  std::string kind;
  // the result's domain as the host declares it: "int" (any integer) or
  // "positive" (>= 1, for a result that is one by the function's definition:
  // a power-of-two rounding, a block or split count). The trace mints the
  // symbol with that property, so a relation it decides (a division by the
  // result is defined) is no guard; the recorder checks the value against it.
  std::string domain;
};

struct TORCH_CUDA_CPP_API Tape {
  std::vector<LaunchRec> launches;
  std::vector<OpaqueRec> opaque;
  // philox offsets one replay consumes; nullopt when the host draws no
  // randomness
  std::optional<c10::SymInt> rng_increment;
  // host-order counter shared by every recorded event (launches here, the
  // allocations and views on the Python side)
  int64_t seq = 0;
  // true iff every launch was issued on the trace's capturing stream (a
  // typed launch: its stream argument; a verbatim launch: the stream current
  // at its Grid or proxy conversion); false once the host forked a side
  // stream. A consumer that replays on one stream reads it before lowering;
  // the raw capture accepts fork and join either way.
  bool all_on_capture_stream = true;
  // the roots read through sym_mutable_data_ptr (an output, an in-place
  // operand), by name in first-read order: an input among them is a
  // position the tape writes, where a binding materializes a copy-on-write
  // tensor as eager's mutable read does before its kernels run
  std::vector<std::string> written_roots;
};

} // namespace at::cuda::host_trace
