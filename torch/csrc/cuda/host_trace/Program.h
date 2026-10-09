#pragma once

#include <torch/csrc/python_headers.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace torch {

// The integer program of torch/cuda/_host_trace_program.py, interpreted.
// The rows are IntegerProgram.instructions; they are checked once here, so
// evaluation does no bounds checks. The statuses are the Python `_step`'s.
class HostTraceProgram {
 public:
  enum class Status : int32_t {
    Success = 0,
    AddOverflow = 1,
    MultiplyOverflow = 2,
    DivisionDomain = 3,
    BooleanDomain = 4,
    ShiftDomain = 5,
    FloatDomain = 6,
  };

  HostTraceProgram(
      const std::vector<std::pair<std::string, std::vector<int64_t>>>& rows,
      int64_t input_count);

  size_t num_rows() const {
    return rows_.size();
  }
  size_t num_leaves() const {
    return leaves_.size();
  }
  // Reads the leaves from the call's inputs; false when there are not
  // input_count inputs or an input is not of its leaves' kind.
  bool bind(PyObject* const* inputs, size_t count, int64_t* leaves) const;
  // values[i] is row i's value when this returns Status::Success.
  Status evaluate(const int64_t* leaves, int64_t* values) const;
  // The rows and leaves that read an input at or after `prefix`, in order.
  // The other rows' values are those of any call with the same leading
  // inputs' leaves.
  void split(
      int64_t prefix,
      std::vector<uint32_t>& rows,
      std::vector<uint32_t>& leaves) const;
  // bind and evaluate of those leaves and rows alone: values holds the other
  // rows' values of a call with the same leading inputs
  bool bind(
      PyObject* const* inputs,
      size_t count,
      const std::vector<uint32_t>& subset,
      int64_t* leaves) const;
  Status evaluate(
      const std::vector<uint32_t>& subset,
      const int64_t* leaves,
      int64_t* values) const;
  // `rows` (in order) as those row `target` reads, itself included, and the
  // rest, each in order
  void partition(
      int64_t target,
      const std::vector<uint32_t>& rows,
      std::vector<uint32_t>& reads,
      std::vector<uint32_t>& rest) const;
  // the rows of `rows` (in order) that read one of `leaves`, directly or
  // through other rows of `rows`
  void reading(
      const std::vector<uint32_t>& leaves,
      const std::vector<uint32_t>& rows,
      std::vector<uint32_t>& out) const;
  // leaf i's (input, kind, dim)
  std::tuple<int64_t, int64_t, int64_t> leaf(size_t i) const {
    const Leaf& l = leaves_[i];
    return {l.input, static_cast<int64_t>(l.kind), l.dim};
  }

 private:
  enum class Op : uint8_t {
    Constant,
    Leaf,
    Add,
    Multiply,
    FloorDiv,
    CeilDiv,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    BitAnd,
    BitOr,
    BitXor,
    BitLength,
    LShift,
    F32Div,
    ToFloat,
    FSqrt,
    FDiv,
    FEq,
    FLt,
    Min,
    Max,
    Select,
  };
  enum class LeafKind : uint8_t { Boxed, Pointer, StorageOffset, Size, Stride };
  // operands_[first, first + count) are earlier rows; a constant's is its
  // value and a leaf's its index in leaves_
  struct Row {
    Op op;
    uint32_t first;
    uint32_t count;
  };
  struct Leaf {
    int64_t input;
    LeafKind kind;
    int64_t dim;
  };

  bool bind_leaf(size_t i, PyObject* const* inputs, int64_t* leaves) const;
  Status step(size_t i, const int64_t* leaves, int64_t* values) const;

  std::vector<Row> rows_;
  std::vector<int64_t> operands_;
  std::vector<Leaf> leaves_;
  int64_t input_count_;
};

void initHostTraceProgramBindings(PyObject* module);

} // namespace torch
