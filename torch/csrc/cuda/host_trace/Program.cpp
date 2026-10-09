#include <torch/csrc/cuda/host_trace/Program.h>

#include <c10/util/SmallVector.h>
#include <c10/util/safe_numerics.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace torch {

HostTraceProgram::HostTraceProgram(
    const std::vector<std::pair<std::string, std::vector<int64_t>>>& rows,
    int64_t input_count)
    : input_count_(input_count) {
  static const std::unordered_map<std::string, Op> ops = {
      {"constant", Op::Constant},
      {"add", Op::Add},
      {"multiply", Op::Multiply},
      {"floordiv", Op::FloorDiv},
      {"ceildiv", Op::CeilDiv},
      {"eq", Op::Eq},
      {"ne", Op::Ne},
      {"lt", Op::Lt},
      {"le", Op::Le},
      {"gt", Op::Gt},
      {"ge", Op::Ge},
      {"and", Op::And},
      {"bitand", Op::BitAnd},
      {"bitor", Op::BitOr},
      {"bitxor", Op::BitXor},
      {"bitlength", Op::BitLength},
      {"lshift", Op::LShift},
      {"f32div", Op::F32Div},
      {"tofloat", Op::ToFloat},
      {"fsqrt", Op::FSqrt},
      {"fdiv", Op::FDiv},
      {"feq", Op::FEq},
      {"flt", Op::FLt},
      {"min", Op::Min},
      {"max", Op::Max},
      {"select", Op::Select},
  };
  static const std::unordered_map<std::string, LeafKind> leaf_kinds = {
      {"boxed", LeafKind::Boxed},
      {"pointer", LeafKind::Pointer},
      {"storage_offset", LeafKind::StorageOffset},
      {"size", LeafKind::Size},
      {"stride", LeafKind::Stride},
  };
  TORCH_CHECK_VALUE(input_count >= 0, "negative input count ", input_count);
  for (const auto& [name, operands] : rows) {
    const auto row = static_cast<int64_t>(rows_.size());
    const auto first = static_cast<uint32_t>(operands_.size());
    const auto count = operands.size();
    if (auto kind = leaf_kinds.find(name); kind != leaf_kinds.end()) {
      const bool dimensioned =
          kind->second == LeafKind::Size || kind->second == LeafKind::Stride;
      TORCH_CHECK_VALUE(
          count == (dimensioned ? 2 : 1),
          "row ",
          row,
          ": wrong number of operands for ",
          name);
      TORCH_CHECK_VALUE(
          operands[0] >= 0 && operands[0] < input_count,
          "row ",
          row,
          ": no input ",
          operands[0]);
      TORCH_CHECK_VALUE(
          !dimensioned || operands[1] >= 0, "row ", row, ": negative dim");
      operands_.push_back(static_cast<int64_t>(leaves_.size()));
      leaves_.push_back(
          {operands[0], kind->second, dimensioned ? operands[1] : 0});
      rows_.push_back({Op::Leaf, first, 1});
      continue;
    }
    auto op = ops.find(name);
    TORCH_CHECK_VALUE(op != ops.end(), "row ", row, ": unknown op ", name);
    size_t arity = 2;
    if (op->second == Op::Constant || op->second == Op::BitLength ||
        op->second == Op::ToFloat || op->second == Op::FSqrt) {
      arity = 1;
    } else if (op->second == Op::Select) {
      arity = 3;
    } else if (op->second == Op::Min || op->second == Op::Max) {
      arity = std::max<size_t>(count, 2);
    }
    TORCH_CHECK_VALUE(
        count == arity, "row ", row, ": wrong number of operands for ", name);
    if (op->second != Op::Constant) {
      for (int64_t x : operands) {
        TORCH_CHECK_VALUE(
            x >= 0 && x < row, "row ", row, ": operand ", x, " is not earlier");
      }
    }
    operands_.insert(operands_.end(), operands.begin(), operands.end());
    rows_.push_back({op->second, first, static_cast<uint32_t>(count)});
  }
}

C10_ALWAYS_INLINE bool HostTraceProgram::bind_leaf(
    size_t i,
    PyObject* const* inputs,
    int64_t* leaves) const {
  const Leaf& leaf = leaves_[i];
  PyObject* x = inputs[leaf.input];
  if (leaf.kind == LeafKind::Boxed) {
    // exactly int, as the trace's: a bool is not an int input
    if (!PyLong_CheckExact(x)) {
      return false;
    }
    int overflow = 0;
    leaves[i] = PyLong_AsLongLongAndOverflow(x, &overflow);
    if (overflow != 0) {
      return false;
    }
    return true;
  }
  // Tensor or Parameter, as the trace's: a subclass could run Python that
  // reports other metadata or mutates `inputs`
  if (!THPVariable_CheckExact(x)) {
    return false;
  }
  const at::Tensor& t = THPVariable_Unpack(x);
  if (t.layout() != at::kStrided || t.is_nested()) {
    return false;
  }
  switch (leaf.kind) {
    case LeafKind::Pointer: {
      // data_ptr(), as Tensor.data_ptr() reads it
      const auto address = reinterpret_cast<uintptr_t>(t.data_ptr());
      if (address >
          static_cast<uintptr_t>(std::numeric_limits<int64_t>::max())) {
        return false;
      }
      leaves[i] = static_cast<int64_t>(address);
      break;
    }
    case LeafKind::StorageOffset:
      leaves[i] = t.storage_offset();
      break;
    default:
      if (leaf.dim >= t.dim()) {
        return false;
      }
      leaves[i] = leaf.kind == LeafKind::Size ? t.sizes()[leaf.dim]
                                              : t.strides()[leaf.dim];
  }
  return true;
}

bool HostTraceProgram::bind(
    PyObject* const* inputs,
    size_t count,
    int64_t* leaves) const {
  if (count != static_cast<size_t>(input_count_)) {
    return false;
  }
  for (size_t i = 0; i < leaves_.size(); ++i) {
    if (!bind_leaf(i, inputs, leaves)) {
      return false;
    }
  }
  return true;
}

bool HostTraceProgram::bind(
    PyObject* const* inputs,
    size_t count,
    const std::vector<uint32_t>& subset,
    int64_t* leaves) const {
  if (count != static_cast<size_t>(input_count_)) {
    return false;
  }
  for (uint32_t i : subset) {
    if (!bind_leaf(i, inputs, leaves)) {
      return false;
    }
  }
  return true;
}

C10_ALWAYS_INLINE HostTraceProgram::Status HostTraceProgram::step(
    size_t i,
    const int64_t* leaves,
    int64_t* values) const {
  const Row& row = rows_[i];
  const int64_t* x = operands_.data() + row.first;
  const int64_t a =
      row.op == Op::Constant || row.op == Op::Leaf ? 0 : values[x[0]];
  const int64_t b = row.count > 1 ? values[x[1]] : 0;
  int64_t& out = values[i];
  switch (row.op) {
    case Op::Constant:
      out = x[0];
      break;
    case Op::Leaf:
      out = leaves[x[0]];
      break;
    case Op::Add:
      if (c10::add_overflows(a, b, &out)) {
        return Status::AddOverflow;
      }
      break;
    case Op::Multiply:
      if (c10::mul_overflows(a, b, &out)) {
        return Status::MultiplyOverflow;
      }
      break;
    case Op::FloorDiv:
    case Op::CeilDiv:
      if (a < 0 || b <= 0) {
        return Status::DivisionDomain;
      }
      out = a / b + (row.op == Op::CeilDiv && a % b != 0);
      break;
    case Op::Eq:
      out = a == b;
      break;
    case Op::Ne:
      out = a != b;
      break;
    case Op::Lt:
      out = a < b;
      break;
    case Op::Le:
      out = a <= b;
      break;
    case Op::Gt:
      out = a > b;
      break;
    case Op::Ge:
      out = a >= b;
      break;
    case Op::And:
      if ((a != 0 && a != 1) || (b != 0 && b != 1)) {
        return Status::BooleanDomain;
      }
      out = a & b;
      break;
    case Op::BitAnd:
      out = a & b;
      break;
    case Op::BitOr:
      out = a | b;
      break;
    case Op::BitXor:
      out = a ^ b;
      break;
    case Op::BitLength:
      out = std::bit_width(
          a < 0 ? 0 - static_cast<uint64_t>(a) : static_cast<uint64_t>(a));
      break;
    case Op::LShift:
      if (b < 0) {
        return Status::ShiftDomain;
      }
      // two multiplications: 1 << 63 is not an int64, a * 2**63 can be
      out = a;
      if (a != 0 && b > 0 &&
          (b > 63 || c10::mul_overflows(a, int64_t(1) << (b - 1), &out) ||
           c10::mul_overflows(out, int64_t(2), &out))) {
        return Status::MultiplyOverflow;
      }
      break;
    case Op::F32Div:
      if (a < 0 || b <= 0) {
        return Status::DivisionDomain;
      }
      out = std::bit_cast<int32_t>(
          static_cast<float>(a) / static_cast<float>(b));
      break;
    case Op::ToFloat:
      out = std::bit_cast<int64_t>(static_cast<double>(a));
      break;
    case Op::FSqrt: {
      const double x = std::bit_cast<double>(a);
      if (!(x >= 0.0) || !std::isfinite(x)) {
        return Status::FloatDomain;
      }
      out = std::bit_cast<int64_t>(std::sqrt(x));
      break;
    }
    case Op::FDiv: {
      const double q = std::bit_cast<double>(a) / std::bit_cast<double>(b);
      if (!std::isfinite(q)) {
        return Status::FloatDomain;
      }
      out = std::bit_cast<int64_t>(q);
      break;
    }
    case Op::FEq:
      out = std::bit_cast<double>(a) == std::bit_cast<double>(b);
      break;
    case Op::FLt:
      out = std::bit_cast<double>(a) < std::bit_cast<double>(b);
      break;
    case Op::Min:
    case Op::Max:
      out = a;
      for (uint32_t k = 1; k < row.count; ++k) {
        const int64_t v = values[x[k]];
        out = row.op == Op::Min ? std::min(out, v) : std::max(out, v);
      }
      break;
    case Op::Select:
      if (a != 0 && a != 1) {
        return Status::BooleanDomain;
      }
      out = a ? b : values[x[2]];
      break;
  }
  return Status::Success;
}

HostTraceProgram::Status HostTraceProgram::evaluate(
    const int64_t* leaves,
    int64_t* values) const {
  for (size_t i = 0; i < rows_.size(); ++i) {
    if (const Status s = step(i, leaves, values); s != Status::Success) {
      return s;
    }
  }
  return Status::Success;
}

HostTraceProgram::Status HostTraceProgram::evaluate(
    const std::vector<uint32_t>& subset,
    const int64_t* leaves,
    int64_t* values) const {
  for (uint32_t i : subset) {
    if (const Status s = step(i, leaves, values); s != Status::Success) {
      return s;
    }
  }
  return Status::Success;
}

void HostTraceProgram::split(
    int64_t prefix,
    std::vector<uint32_t>& rows,
    std::vector<uint32_t>& leaves) const {
  rows.clear();
  leaves.clear();
  for (size_t i = 0; i < leaves_.size(); ++i) {
    if (leaves_[i].input >= prefix) {
      leaves.push_back(static_cast<uint32_t>(i));
    }
  }
  std::vector<bool> reads(rows_.size(), false);
  for (size_t i = 0; i < rows_.size(); ++i) {
    const Row& row = rows_[i];
    const int64_t* x = operands_.data() + row.first;
    bool r = false;
    if (row.op == Op::Leaf) {
      r = leaves_[x[0]].input >= prefix;
    } else if (row.op != Op::Constant) {
      for (uint32_t k = 0; k < row.count && !r; ++k) {
        r = reads[x[k]];
      }
    }
    reads[i] = r;
    if (r) {
      rows.push_back(static_cast<uint32_t>(i));
    }
  }
}

void HostTraceProgram::partition(
    int64_t target,
    const std::vector<uint32_t>& rows,
    std::vector<uint32_t>& reads,
    std::vector<uint32_t>& rest) const {
  reads.clear();
  rest.clear();
  std::vector<bool> read(rows_.size(), false);
  read[target] = true;
  for (int64_t i = target; i >= 0; --i) {
    const Row& row = rows_[i];
    if (!read[i] || row.op == Op::Constant || row.op == Op::Leaf) {
      continue;
    }
    const int64_t* x = operands_.data() + row.first;
    for (uint32_t k = 0; k < row.count; ++k) {
      read[x[k]] = true;
    }
  }
  for (uint32_t i : rows) {
    (read[i] ? reads : rest).push_back(i);
  }
}

void HostTraceProgram::reading(
    const std::vector<uint32_t>& leaves,
    const std::vector<uint32_t>& rows,
    std::vector<uint32_t>& out) const {
  out.clear();
  std::vector<bool> leaf(leaves_.size(), false);
  for (uint32_t l : leaves) {
    leaf[l] = true;
  }
  std::vector<bool> reads(rows_.size(), false);
  for (uint32_t i : rows) {
    const Row& row = rows_[i];
    const int64_t* x = operands_.data() + row.first;
    bool r = false;
    if (row.op == Op::Leaf) {
      r = leaf[x[0]];
    } else if (row.op != Op::Constant) {
      for (uint32_t k = 0; k < row.count && !r; ++k) {
        r = reads[x[k]];
      }
    }
    reads[i] = r;
    if (r) {
      out.push_back(i);
    }
  }
}

namespace {

using Buffer = c10::SmallVector<int64_t, 256>;

py::object to_tuple(const Buffer& ints) {
  THPObjectPtr tuple(PyTuple_New(static_cast<Py_ssize_t>(ints.size())));
  if (!tuple) {
    throw py::error_already_set();
  }
  for (size_t i = 0; i < ints.size(); ++i) {
    PyObject* v = PyLong_FromLongLong(ints[i]);
    if (!v) {
      throw py::error_already_set();
    }
    PyTuple_SET_ITEM(tuple.get(), static_cast<Py_ssize_t>(i), v);
  }
  return py::reinterpret_steal<py::object>(tuple.release());
}

py::tuple result(HostTraceProgram::Status status, const Buffer& values) {
  const bool success = status == HostTraceProgram::Status::Success;
  return py::make_tuple(
      static_cast<int32_t>(status), success ? to_tuple(values) : py::none());
}

// The leaves of `inputs` (a sequence), or false when they do not bind.
bool bind_inputs(
    const HostTraceProgram& self,
    py::handle inputs,
    Buffer& leaves) {
  // a tuple (a list is copied) owns the inputs while bind reads them
  THPObjectPtr seq(PySequence_Tuple(inputs.ptr()));
  if (!seq) {
    throw py::error_already_set();
  }
  leaves.resize(self.num_leaves());
  return self.bind(
      PySequence_Fast_ITEMS(seq.get()),
      PyTuple_GET_SIZE(seq.get()),
      leaves.data());
}

} // namespace

void initHostTraceProgramBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<HostTraceProgram>(m, "_HostTraceProgram")
      .def(
          py::init([](const py::sequence& rows, int64_t input_count) {
            std::vector<std::pair<std::string, std::vector<int64_t>>> specs;
            specs.reserve(rows.size());
            for (py::handle row : rows) {
              auto t = py::cast<py::tuple>(row);
              TORCH_CHECK_VALUE(!t.empty(), "an empty row");
              std::vector<int64_t> operands;
              for (size_t k = 1; k < t.size(); ++k) {
                operands.push_back(t[k].cast<int64_t>());
              }
              specs.emplace_back(t[0].cast<std::string>(), std::move(operands));
            }
            return HostTraceProgram(specs, input_count);
          }),
          py::arg("rows"),
          py::arg("input_count"))
      .def(
          "bind",
          [](const HostTraceProgram& self, py::handle inputs) -> py::object {
            Buffer leaves;
            if (!bind_inputs(self, inputs, leaves)) {
              return py::none();
            }
            return to_tuple(leaves);
          })
      .def(
          "evaluate",
          [](const HostTraceProgram& self, py::handle leaves) {
            THPObjectPtr seq(
                PySequence_Fast(leaves.ptr(), "leaves must be a sequence"));
            if (!seq) {
              throw py::error_already_set();
            }
            const auto count = PySequence_Fast_GET_SIZE(seq.get());
            TORCH_CHECK_VALUE(
                static_cast<size_t>(count) == self.num_leaves(),
                "expected ",
                self.num_leaves(),
                " leaves, got ",
                count);
            Buffer values(self.num_rows());
            Buffer ints(count);
            PyObject** items = PySequence_Fast_ITEMS(seq.get());
            for (Py_ssize_t i = 0; i < count; ++i) {
              int overflow = 0;
              TORCH_CHECK_VALUE(
                  PyLong_CheckExact(items[i]), "leaf ", i, " is not an int");
              ints[i] = PyLong_AsLongLongAndOverflow(items[i], &overflow);
              TORCH_CHECK_VALUE(overflow == 0, "leaf ", i, " is not an int64");
            }
            return result(self.evaluate(ints.data(), values.data()), values);
          })
      .def(
          "evaluate_inputs",
          [](const HostTraceProgram& self, py::handle inputs) -> py::object {
            Buffer leaves;
            if (!bind_inputs(self, inputs, leaves)) {
              return py::none();
            }
            Buffer values(self.num_rows());
            return result(self.evaluate(leaves.data(), values.data()), values);
          });
}

} // namespace torch
