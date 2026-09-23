#include <torch/csrc/python_headers.h>

#include <pybind11/chrono.h>
#include <pybind11/stl.h>
#include <torch/csrc/Generator.h>

#include <torch/csrc/Dtype.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/cuda/GraphParameterProgram.h>
#include <torch/csrc/cuda/Stream.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>

#include <string_view>
#include <tuple>

#include <ATen/BlasSettingsEpoch.h>
#include <ATen/EmptyTensor.h>
#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/core/CachingHostAllocator.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAGraphParams.h>
#include <ATen/ops/as_strided.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/cuda/CUDACachingAllocatorPendingGraph.h>
#include <c10/cuda/CUDAEvent.h>
#include <c10/util/safe_numerics.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <mutex>
#include <string>

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
#include <c10/cuda/driver_api.h>
#endif

// Cargo culted partially from csrc/distributed/c10d/init.cpp
// and partially from csrc/cuda/Stream.cpp.
// THCPStream_init is also declared at global scope.

// Because THCPGraph_init is forward declared in the only consumer
// (csrc/Module.cpp) I don't think we need a Graph.h.

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

struct PythonKernelPointerUpdates {
  PythonKernelPointerUpdates(
      std::shared_ptr<at::cuda::detail::KernelPointerUpdateBatch> batch,
      size_t count)
      : batch(std::move(batch)), pointers(count), referenced(count) {}

  std::shared_ptr<at::cuda::detail::KernelPointerUpdateBatch> batch;
  std::vector<uintptr_t> pointers;
  std::vector<bool> referenced;
  std::vector<at::cuda::detail::KernelPointerBinding> pointer_bindings;
  std::vector<at::cuda::detail::KernelScalarBinding> scalar_bindings;
  std::vector<at::cuda::detail::KernelGridBinding> grid_bindings;
  std::vector<at::cuda::detail::KernelMemsetBinding> memset_bindings;
  std::vector<at::cuda::detail::KernelTensorMapBinding> tensor_map_bindings;
  std::vector<at::cuda::detail::KernelHostTableBinding> host_table_bindings;
  std::vector<at::cuda::detail::KernelMemcpyBinding> memcpy_bindings;
  std::vector<at::cuda::detail::KernelRngBinding> rng_bindings;
  std::vector<at::cuda::detail::KernelTemplateBinding> template_bindings;
};

namespace {

void unpack_pointer_values(
    py::handle pointers,
    std::vector<uintptr_t>& values) {
  if (!PyTuple_CheckExact(pointers.ptr())) {
    throw py::type_error("Pointer values must be an exact tuple of integers");
  }
  if (static_cast<size_t>(PyTuple_GET_SIZE(pointers.ptr())) != values.size()) {
    throw py::value_error("Pointer count differs from the prepared bindings");
  }
  for (size_t index = 0; index < values.size(); ++index) {
    auto value = PyTuple_GET_ITEM(pointers.ptr(), index);
    if (!PyLong_CheckExact(value)) {
      throw py::type_error("Pointer values must be exact integers");
    }
    auto pointer = PyLong_AsUnsignedLongLong(value);
    if (PyErr_Occurred()) {
      throw py::error_already_set();
    }
    if (pointer > std::numeric_limits<uintptr_t>::max()) {
      PyErr_SetString(PyExc_OverflowError, "Pointer value is out of range");
      throw py::error_already_set();
    }
    values[index] = static_cast<uintptr_t>(pointer);
  }
}

std::vector<at::cuda::detail::KernelPointerBinding> unpack_pointer_bindings(
    PyObject* bindings,
    bool exact_nodes,
    bool allow_fields = false) {
  if (!PyTuple_CheckExact(bindings)) {
    throw py::type_error("Pointer bindings must be an exact tuple");
  }
  std::vector<at::cuda::detail::KernelPointerBinding> values;
  values.reserve(PyTuple_GET_SIZE(bindings));
  for (Py_ssize_t index = 0; index < PyTuple_GET_SIZE(bindings); ++index) {
    auto* row = PyTuple_GET_ITEM(bindings, index);
    const bool displaced =
        allow_fields && PyTuple_CheckExact(row) && PyTuple_GET_SIZE(row) == 5;
    const bool field = displaced ||
        (allow_fields && PyTuple_CheckExact(row) && PyTuple_GET_SIZE(row) == 4);
    if (!PyTuple_CheckExact(row) ||
        PyTuple_GET_SIZE(row) !=
            (displaced   ? 5
                 : field ? 4
                         : 3) ||
        !(exact_nodes ? PyLong_CheckExact(PyTuple_GET_ITEM(row, 0))
                      : PyLong_Check(PyTuple_GET_ITEM(row, 0))) ||
        !PyLong_CheckExact(PyTuple_GET_ITEM(row, 1)) ||
        !(PyLong_CheckExact(PyTuple_GET_ITEM(row, 2)) ||
          (displaced && Py_IsNone(PyTuple_GET_ITEM(row, 2)))) ||
        (field && !PyLong_CheckExact(PyTuple_GET_ITEM(row, 3))) ||
        (displaced && !PyLong_CheckExact(PyTuple_GET_ITEM(row, 4)))) {
      throw py::type_error(
          allow_fields
              ? "Pointer bindings must contain (node, argument, pointer_index), "
                "(node, argument, byte_offset, pointer_index), or "
                "(node, argument, byte_offset_or_None, pointer_index, address_offset_value_index) tuples"
              : "Pointer bindings must contain (node, argument, pointer_index) integer tuples");
    }
    auto node = PyLong_AsVoidPtr(PyTuple_GET_ITEM(row, 0));
    if (PyErr_Occurred()) {
      throw py::error_already_set();
    }
    auto argument = PyLong_AsLongLong(PyTuple_GET_ITEM(row, 1));
    if (PyErr_Occurred()) {
      throw py::error_already_set();
    }
    size_t pointer_index =
        PyLong_AsSize_t(PyTuple_GET_ITEM(row, field ? 3 : 2));
    if (PyErr_Occurred()) {
      throw py::error_already_set();
    }
    std::optional<size_t> byte_offset;
    if (field && !Py_IsNone(PyTuple_GET_ITEM(row, 2))) {
      byte_offset = PyLong_AsSize_t(PyTuple_GET_ITEM(row, 2));
      if (PyErr_Occurred()) {
        throw py::error_already_set();
      }
    }
    std::optional<size_t> address_offset_value_index;
    if (displaced) {
      address_offset_value_index = PyLong_AsSize_t(PyTuple_GET_ITEM(row, 4));
      if (PyErr_Occurred()) {
        throw py::error_already_set();
      }
    }
    values.push_back(
        {reinterpret_cast<uintptr_t>(node),
         argument,
         pointer_index,
         byte_offset,
         address_offset_value_index});
  }
  return values;
}

uintptr_t current_replay_context() {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  auto* api = c10::cuda::DriverAPI::get();
  TORCH_CHECK(
      api->cuCtxGetCurrent_,
      "Native graph ownership requires CUDA context queries");
  CUcontext context = nullptr;
  C10_CUDA_DRIVER_CHECK(api->cuCtxGetCurrent_(&context));
  TORCH_CHECK(
      context, "Native graph ownership requires a current CUDA context");
  return reinterpret_cast<uintptr_t>(context);
#else
  TORCH_CHECK(
      false, "Native graph ownership requires NVIDIA CUDA 12.8 or later");
#endif
}

int64_t unpack_nonnegative_integer(PyObject* value, const char* name) {
  if (!PyLong_CheckExact(value)) {
    throw py::type_error(std::string(name) + " must be an exact integer");
  }
  auto result = PyLong_AsLongLong(value);
  if (PyErr_Occurred()) {
    throw py::error_already_set();
  }
  if (result < 0) {
    throw py::value_error(std::string(name) + " must be nonnegative");
  }
  return result;
}

std::vector<int64_t> unpack_static_dimensions(PyObject* values) {
  if (!PyTuple_CheckExact(values)) {
    throw py::type_error("Output sizes and strides must be exact tuples");
  }
  std::vector<int64_t> result;
  result.reserve(PyTuple_GET_SIZE(values));
  for (Py_ssize_t index = 0; index < PyTuple_GET_SIZE(values); ++index) {
    result.push_back(unpack_nonnegative_integer(
        PyTuple_GET_ITEM(values, index), "Output dimension"));
  }
  return result;
}

std::vector<size_t> unpack_input_indices(PyObject* values, size_t input_count) {
  if (!PyTuple_CheckExact(values)) {
    throw py::type_error("Input indices must be an exact tuple");
  }
  std::vector<size_t> result;
  result.reserve(PyTuple_GET_SIZE(values));
  for (Py_ssize_t index = 0; index < PyTuple_GET_SIZE(values); ++index) {
    auto value = static_cast<size_t>(unpack_nonnegative_integer(
        PyTuple_GET_ITEM(values, index), "Input index"));
    if (value >= input_count ||
        std::find(result.begin(), result.end(), value) != result.end()) {
      throw py::value_error("Input indices must be distinct and in range");
    }
    result.push_back(value);
  }
  return result;
}

struct BoxedBufferLayout {
  caffe2::TypeMeta dtype;
  std::vector<int64_t> size;
  std::vector<int64_t> stride;
  size_t nbytes;
};

struct BoxedLayoutValue {
  size_t buffer;
  size_t dimension;
  size_t value;
  bool stride;
};

struct BoxedTensorMetadata {
  enum class Kind {
    Size,
    Stride,
    Dtype,
    Device,
    Rank,
    IsNeg,
    IsConj,
    Layout,
    Pinned
  };
  Kind kind;
  size_t input;
  int64_t dimension;

  BoxedTensorMetadata(PyObject* tag, size_t input, PyObject* dim)
      : input(input), dimension(-1) {
    if (!PyUnicode_CheckExact(tag)) {
      throw py::type_error("Tensor metadata kinds must be exact strings");
    }
    constexpr const char* names[] = {
        "size",
        "stride",
        "dtype",
        "device",
        "rank",
        "neg",
        "conj",
        "layout",
        "pinned"};
    size_t index = 0;
    while (index < std::size(names) &&
           PyUnicode_CompareWithASCIIString(tag, names[index]) != 0) {
      ++index;
    }
    if (index == std::size(names)) {
      throw py::value_error("Unsupported Tensor metadata kind");
    }
    kind = static_cast<Kind>(index);
    if (kind == Kind::Size || kind == Kind::Stride) {
      dimension = unpack_nonnegative_integer(dim, "Tensor metadata dimension");
    } else if (
        unpack_nonnegative_integer(dim, "Tensor metadata dimension") != 0) {
      throw py::value_error("Scalar Tensor metadata requires dimension zero");
    }
  }

  bool read(const at::Tensor& tensor, int64_t& value) const {
    switch (kind) {
      case Kind::Size:
      case Kind::Stride:
        if (dimension >= tensor.dim() || tensor.layout() != at::kStrided ||
            tensor.is_nested()) {
          return false;
        }
        value = kind == Kind::Size ? tensor.sizes()[dimension]
                                   : tensor.strides()[dimension];
        break;
      case Kind::Dtype:
        value = static_cast<int64_t>(tensor.scalar_type());
        break;
      case Kind::Device:
        value = tensor.is_cuda() ? tensor.get_device() : -1;
        break;
      case Kind::Rank:
        value = tensor.dim();
        break;
      case Kind::IsNeg:
        value = tensor.is_neg();
        break;
      case Kind::IsConj:
        value = tensor.is_conj();
        break;
      case Kind::Layout:
        value = static_cast<int64_t>(tensor.layout());
        break;
      case Kind::Pinned:
        value = !tensor.is_cuda() && tensor.is_pinned();
        break;
    }
    return true;
  }
};

struct CompiledBoxedEvaluation {
  using Early = int32_t (*)(const int64_t*, int64_t*);
  using Late = int32_t (*)(const int64_t*, const uintptr_t*, int64_t*);

  CompiledBoxedEvaluation(
      uintptr_t early_address,
      uintptr_t late_address,
      size_t leaf_count,
      size_t early_count,
      size_t late_count,
      size_t pointer_count,
      py::handle library_owner)
      : early(reinterpret_cast<Early>(early_address)),
        late(reinterpret_cast<Late>(late_address)),
        leaf_count(leaf_count),
        early_count(early_count),
        late_count(late_count),
        pointer_count(pointer_count) {
    if (!early || (late_count != 0 && !late) || library_owner.is_none()) {
      throw py::value_error(
          "Compiled evaluation requires native functions and a retained library owner");
    }
    owner = Py_NewRef(library_owner.ptr());
  }

  Early early;
  Late late;
  const size_t leaf_count;
  const size_t early_count;
  const size_t late_count;
  const size_t pointer_count;
  THPObjectPtr owner;
};

struct BoxedNumericPlan {
  enum class Op {
    Constant,
    Boxed,
    Pointer,
    StorageOffset,
    Size,
    Stride,
    CeilDiv,
    Multiply,
    Add,
    FloorDiv,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Select,
    Call,
    PCall, // the pointer-and-length ABI of Call
    FFromInt,
    FNeg,
    FSqrt,
    FRound32,
    FToBits32,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FPow,
    Max,
    Min
  };
  struct Instruction {
    Op op;
    int64_t first;
    size_t second = 0;
    size_t third = 0;
    std::vector<size_t> operands;
  };

  BoxedNumericPlan(PyObject* plan, size_t input_count, size_t value_count) {
    if (!PyTuple_CheckExact(plan) ||
        (PyTuple_GET_SIZE(plan) != 2 && PyTuple_GET_SIZE(plan) != 3) ||
        !PyTuple_CheckExact(PyTuple_GET_ITEM(plan, 1))) {
      throw py::type_error(
          "Numeric plan must contain integer input indices and an instruction tuple");
    }
    auto indices = unpack_input_indices(PyTuple_GET_ITEM(plan, 0), input_count);
    integer_inputs.resize(input_count);
    for (auto index : indices) {
      integer_inputs[index] = true;
    }
    for (size_t index = 0; index < input_count; ++index) {
      if (!integer_inputs[index]) {
        tensor_inputs.push_back(index);
      }
    }
    auto* program = PyTuple_GET_ITEM(plan, 1);
    if (static_cast<size_t>(PyTuple_GET_SIZE(program)) != value_count) {
      throw py::value_error(
          "Numeric instruction count differs from the prepared batch");
    }
    std::vector<bool> loaded(input_count);
    instructions.reserve(value_count);
    for (size_t index = 0; index < value_count; ++index) {
      auto* row = PyTuple_GET_ITEM(program, index);
      if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) < 2 ||
          !PyUnicode_CheckExact(PyTuple_GET_ITEM(row, 0))) {
        throw py::type_error("Numeric instructions must be tagged tuples");
      }
      auto* tag = PyTuple_GET_ITEM(row, 0);
      auto* first = PyTuple_GET_ITEM(row, 1);
      if ((PyUnicode_CompareWithASCIIString(tag, "constant") == 0 ||
           PyUnicode_CompareWithASCIIString(tag, "fconst") == 0) &&
          PyTuple_GET_SIZE(row) == 2) {
        if (!PyLong_CheckExact(first)) {
          throw py::type_error("Numeric constants must be exact integers");
        }
        auto value = PyLong_AsLongLong(first);
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        instructions.push_back({Op::Constant, value});
      } else if (
          (PyUnicode_CompareWithASCIIString(tag, "call") == 0 ||
           PyUnicode_CompareWithASCIIString(tag, "pcall") == 0) &&
          PyTuple_GET_SIZE(row) >= 3) {
        auto* owner = PyTuple_GET_ITEM(row, 2);
        if (!PyLong_CheckExact(first) || !PyCallable_Check(owner)) {
          throw py::type_error(
              "Native calls require an address and callable owner");
        }
        const auto address = PyLong_AsUnsignedLongLong(first);
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        if (address == 0 || address > std::numeric_limits<uintptr_t>::max()) {
          throw py::value_error("Native call address is out of range");
        }
        const bool pointer_abi =
            PyUnicode_CompareWithASCIIString(tag, "pcall") == 0;
        Instruction instruction{
            pointer_abi ? Op::PCall : Op::Call, static_cast<int64_t>(address)};
        for (Py_ssize_t operand = 3; operand < PyTuple_GET_SIZE(row);
             ++operand) {
          const auto slot = static_cast<size_t>(unpack_nonnegative_integer(
              PyTuple_GET_ITEM(row, operand), "Call operand"));
          if (slot >= index) {
            throw py::value_error("Call operands must precede their result");
          }
          instruction.operands.push_back(slot);
        }
        call_owners.emplace_back(Py_NewRef(owner));
        instructions.push_back(std::move(instruction));
      } else if (
          (PyUnicode_CompareWithASCIIString(tag, "ffromint") == 0 ||
           PyUnicode_CompareWithASCIIString(tag, "fneg") == 0 ||
           PyUnicode_CompareWithASCIIString(tag, "fsqrt") == 0 ||
           PyUnicode_CompareWithASCIIString(tag, "fround32") == 0 ||
           PyUnicode_CompareWithASCIIString(tag, "ftobits32") == 0) &&
          PyTuple_GET_SIZE(row) == 2) {
        const auto operand = unpack_nonnegative_integer(first, "Float operand");
        if (static_cast<uint64_t>(operand) >= index) {
          throw py::value_error("Float operands must precede their result");
        }
        Op op = Op::FFromInt;
        if (PyUnicode_CompareWithASCIIString(tag, "fneg") == 0) {
          op = Op::FNeg;
        } else if (PyUnicode_CompareWithASCIIString(tag, "fsqrt") == 0) {
          op = Op::FSqrt;
        } else if (PyUnicode_CompareWithASCIIString(tag, "fround32") == 0) {
          op = Op::FRound32;
        } else if (PyUnicode_CompareWithASCIIString(tag, "ftobits32") == 0) {
          op = Op::FToBits32;
        }
        instructions.push_back({op, operand});
      } else if (
          PyUnicode_CompareWithASCIIString(tag, "boxed") == 0 &&
          PyTuple_GET_SIZE(row) == 2) {
        auto input = static_cast<size_t>(
            unpack_nonnegative_integer(first, "Boxed integer index"));
        if (input >= input_count || !integer_inputs[input] || loaded[input]) {
          throw py::value_error(
              "Boxed integer loads must name each declared input exactly once");
        }
        loaded[input] = true;
        instructions.push_back({Op::Boxed, static_cast<int64_t>(input)});
      } else if (
          PyUnicode_CompareWithASCIIString(tag, "storage_offset") == 0 &&
          PyTuple_GET_SIZE(row) == 2) {
        auto input = static_cast<size_t>(
            unpack_nonnegative_integer(first, "Storage offset input index"));
        if (input >= input_count || integer_inputs[input]) {
          throw py::value_error(
              "Storage offset loads must name declared Tensor inputs");
        }
        instructions.push_back(
            {Op::StorageOffset, static_cast<int64_t>(input)});
      } else if (
          PyUnicode_CompareWithASCIIString(tag, "pointer") == 0 &&
          PyTuple_GET_SIZE(row) == 2) {
        // a Tensor input's data pointer as an int64 value (a host lookup keyed
        // by an address, e.g. symmetric memory's handle from its buffer)
        auto input = static_cast<size_t>(
            unpack_nonnegative_integer(first, "Data pointer input index"));
        if (input >= input_count || integer_inputs[input]) {
          throw py::value_error(
              "Data pointer loads must name declared Tensor inputs");
        }
        instructions.push_back({Op::Pointer, static_cast<int64_t>(input)});
      } else if (
          (PyUnicode_CompareWithASCIIString(tag, "size") == 0 ||
           PyUnicode_CompareWithASCIIString(tag, "stride") == 0) &&
          PyTuple_GET_SIZE(row) == 3) {
        auto input = static_cast<size_t>(
            unpack_nonnegative_integer(first, "Tensor metadata input index"));
        auto dimension = static_cast<size_t>(unpack_nonnegative_integer(
            PyTuple_GET_ITEM(row, 2), "Tensor metadata dimension"));
        if (input >= input_count || integer_inputs[input]) {
          throw py::value_error(
              "Tensor metadata loads must name declared Tensor inputs");
        }
        auto op = PyUnicode_CompareWithASCIIString(tag, "size") == 0
            ? Op::Size
            : Op::Stride;
        instructions.push_back({op, static_cast<int64_t>(input), dimension});
      } else if (
          (PyUnicode_CompareWithASCIIString(tag, "max") == 0 ||
           PyUnicode_CompareWithASCIIString(tag, "min") == 0) &&
          PyTuple_GET_SIZE(row) >= 3) {
        // ("max" | "min", operand slots...): at least two operands
        Instruction instruction{
            PyUnicode_CompareWithASCIIString(tag, "max") == 0 ? Op::Max
                                                              : Op::Min,
            0};
        for (Py_ssize_t operand = 1; operand < PyTuple_GET_SIZE(row);
             ++operand) {
          auto slot = static_cast<size_t>(unpack_nonnegative_integer(
              PyTuple_GET_ITEM(row, operand), "Max or min operand"));
          if (slot >= index) {
            throw py::value_error(
                "Max and min operands must precede their result");
          }
          instruction.operands.push_back(slot);
        }
        instructions.push_back(std::move(instruction));
      } else if (
          PyUnicode_CompareWithASCIIString(tag, "ceildiv") == 0 &&
          PyTuple_GET_SIZE(row) == 3) {
        auto left = unpack_nonnegative_integer(first, "Ceildiv operand");
        auto right = static_cast<size_t>(unpack_nonnegative_integer(
            PyTuple_GET_ITEM(row, 2), "Ceildiv operand"));
        if (static_cast<uint64_t>(left) >= index || right >= index) {
          throw py::value_error("Ceildiv operands must precede their result");
        }
        instructions.push_back({Op::CeilDiv, left, right});
      } else if (
          PyUnicode_CompareWithASCIIString(tag, "multiply") == 0 &&
          PyTuple_GET_SIZE(row) == 3) {
        auto left = unpack_nonnegative_integer(first, "Multiply operand");
        auto right = static_cast<size_t>(unpack_nonnegative_integer(
            PyTuple_GET_ITEM(row, 2), "Multiply operand"));
        if (static_cast<uint64_t>(left) >= index || right >= index) {
          throw py::value_error("Multiply operands must precede their result");
        }
        instructions.push_back({Op::Multiply, left, right});
      } else if (
          PyUnicode_CompareWithASCIIString(tag, "add") == 0 &&
          PyTuple_GET_SIZE(row) == 3) {
        auto left = unpack_nonnegative_integer(first, "Add operand");
        auto right = static_cast<size_t>(unpack_nonnegative_integer(
            PyTuple_GET_ITEM(row, 2), "Add operand"));
        if (static_cast<uint64_t>(left) >= index || right >= index) {
          throw py::value_error("Add operands must precede their result");
        }
        instructions.push_back({Op::Add, left, right});
      } else if (
          PyUnicode_CompareWithASCIIString(tag, "select") == 0 &&
          PyTuple_GET_SIZE(row) == 4) {
        auto condition = unpack_nonnegative_integer(first, "Select condition");
        auto when_true = static_cast<size_t>(unpack_nonnegative_integer(
            PyTuple_GET_ITEM(row, 2), "Select operand"));
        auto when_false = static_cast<size_t>(unpack_nonnegative_integer(
            PyTuple_GET_ITEM(row, 3), "Select operand"));
        if (static_cast<uint64_t>(condition) >= index || when_true >= index ||
            when_false >= index) {
          throw py::value_error("Select operands must precede their result");
        }
        instructions.push_back({Op::Select, condition, when_true, when_false});
      } else if (PyTuple_GET_SIZE(row) == 3) {
        Op operation;
        if (PyUnicode_CompareWithASCIIString(tag, "floordiv") == 0) {
          operation = Op::FloorDiv;
        } else if (PyUnicode_CompareWithASCIIString(tag, "eq") == 0) {
          operation = Op::Eq;
        } else if (PyUnicode_CompareWithASCIIString(tag, "ne") == 0) {
          operation = Op::Ne;
        } else if (PyUnicode_CompareWithASCIIString(tag, "lt") == 0) {
          operation = Op::Lt;
        } else if (PyUnicode_CompareWithASCIIString(tag, "le") == 0) {
          operation = Op::Le;
        } else if (PyUnicode_CompareWithASCIIString(tag, "gt") == 0) {
          operation = Op::Gt;
        } else if (PyUnicode_CompareWithASCIIString(tag, "ge") == 0) {
          operation = Op::Ge;
        } else if (PyUnicode_CompareWithASCIIString(tag, "and") == 0) {
          operation = Op::And;
        } else if (PyUnicode_CompareWithASCIIString(tag, "fadd") == 0) {
          operation = Op::FAdd;
        } else if (PyUnicode_CompareWithASCIIString(tag, "fsub") == 0) {
          operation = Op::FSub;
        } else if (PyUnicode_CompareWithASCIIString(tag, "fmul") == 0) {
          operation = Op::FMul;
        } else if (PyUnicode_CompareWithASCIIString(tag, "fdiv") == 0) {
          operation = Op::FDiv;
        } else if (PyUnicode_CompareWithASCIIString(tag, "fpow") == 0) {
          operation = Op::FPow;
        } else {
          throw py::value_error("Unsupported numeric instruction or arity");
        }
        auto left = unpack_nonnegative_integer(first, "Numeric operand");
        auto right = static_cast<size_t>(unpack_nonnegative_integer(
            PyTuple_GET_ITEM(row, 2), "Numeric operand"));
        if (static_cast<uint64_t>(left) >= index || right >= index) {
          throw py::value_error("Numeric operands must precede their result");
        }
        instructions.push_back({operation, left, right});
      } else {
        throw py::value_error("Unsupported numeric instruction or arity");
      }
    }
    if (loaded != integer_inputs) {
      throw py::value_error(
          "Every declared integer input requires one boxed load");
    }
  }

  std::vector<Instruction> leaf_bindings() const {
    std::vector<Instruction> leaves;
    for (const auto& instruction : instructions) {
      if (instruction.op != Op::Boxed && instruction.op != Op::Pointer &&
          instruction.op != Op::StorageOffset && instruction.op != Op::Size &&
          instruction.op != Op::Stride) {
        continue;
      }
      const auto duplicate =
          std::find_if(leaves.begin(), leaves.end(), [&](const auto& leaf) {
            return leaf.op == instruction.op &&
                leaf.first == instruction.first &&
                leaf.second == instruction.second;
          });
      if (duplicate == leaves.end()) {
        leaves.push_back(instruction);
      }
    }
    return leaves;
  }

  static int64_t read_input(
      const Instruction& instruction,
      std::vector<THPObjectPtr>& inputs) {
    return read_input(instruction, inputs[instruction.first].get());
  }

  static int64_t read_input(const Instruction& instruction, PyObject* input) {
    if (instruction.op == Op::Boxed) {
      if (!PyLong_CheckExact(input)) {
        throw py::type_error(
            "Declared boxed integer inputs must be exact integers");
      }
      const auto value = PyLong_AsLongLong(input);
      if (PyErr_Occurred()) {
        throw py::error_already_set();
      }
      return value;
    }
    const auto& tensor = THPVariable_Unpack(input);
    if (instruction.op == Op::StorageOffset) {
      return tensor.storage_offset();
    }
    if (instruction.op == Op::Pointer) {
      // the const accessor: a lazy copy-on-write input is not materialized by
      // reading its address (the written positions were materialized before)
      return static_cast<int64_t>(
          reinterpret_cast<uintptr_t>(tensor.const_data_ptr()));
    }
    TORCH_CHECK_VALUE(
        instruction.second < static_cast<size_t>(tensor.dim()) &&
            tensor.layout() == at::kStrided && !tensor.is_nested(),
        "Tensor metadata dimension requires an in-range strided Tensor input");
    return instruction.op == Op::Size ? tensor.sizes()[instruction.second]
                                      : tensor.strides()[instruction.second];
  }

  void evaluate(std::vector<THPObjectPtr>& inputs, std::vector<int64_t>& values)
      const {
    const auto to_bits = [](double value) {
      int64_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      return bits;
    };
    const auto from_bits = [](int64_t bits) {
      double value;
      std::memcpy(&value, &bits, sizeof(value));
      return value;
    };
    for (size_t index = 0; index < instructions.size(); ++index) {
      const auto& instruction = instructions[index];
      switch (instruction.op) {
        case Op::Max:
        case Op::Min: {
          int64_t result = values[instruction.operands.front()];
          for (auto slot : instruction.operands) {
            result = instruction.op == Op::Max ? std::max(result, values[slot])
                                               : std::min(result, values[slot]);
          }
          values[index] = result;
          break;
        }
        case Op::Constant:
          values[index] = instruction.first;
          break;
        case Op::Boxed:
        case Op::Pointer:
        case Op::StorageOffset:
        case Op::Size:
        case Op::Stride:
          values[index] = read_input(instruction, inputs);
          break;
        case Op::CeilDiv: {
          const auto left = values[instruction.first];
          const auto right = values[instruction.second];
          TORCH_CHECK_VALUE(
              left >= 0 && right > 0,
              "Ceildiv requires a nonnegative numerator and positive denominator");
          values[index] = left / right + (left % right != 0);
          break;
        }
        case Op::Multiply: {
          const auto left = values[instruction.first];
          const auto right = values[instruction.second];
          int64_t product = 0;
          TORCH_CHECK_VALUE(
              !c10::mul_overflows(left, right, &product),
              "Numeric multiplication overflowed int64");
          values[index] = product;
          break;
        }
        case Op::Add: {
          const auto left = values[instruction.first];
          const auto right = values[instruction.second];
          int64_t sum = 0;
          TORCH_CHECK_VALUE(
              !c10::add_overflows(left, right, &sum),
              "Numeric addition overflowed int64");
          values[index] = sum;
          break;
        }
        case Op::FloorDiv: {
          const auto left = values[instruction.first];
          const auto right = values[instruction.second];
          TORCH_CHECK_VALUE(
              left >= 0 && right > 0,
              "Floor division requires a nonnegative numerator and positive denominator");
          values[index] = left / right;
          break;
        }
        case Op::Eq:
          values[index] =
              values[instruction.first] == values[instruction.second];
          break;
        case Op::Ne:
          values[index] =
              values[instruction.first] != values[instruction.second];
          break;
        case Op::Lt:
          values[index] =
              values[instruction.first] < values[instruction.second];
          break;
        case Op::Le:
          values[index] =
              values[instruction.first] <= values[instruction.second];
          break;
        case Op::Gt:
          values[index] =
              values[instruction.first] > values[instruction.second];
          break;
        case Op::Ge:
          values[index] =
              values[instruction.first] >= values[instruction.second];
          break;
        case Op::And: {
          const auto left = values[instruction.first];
          const auto right = values[instruction.second];
          TORCH_CHECK_VALUE(
              (left == 0 || left == 1) && (right == 0 || right == 1),
              "Boolean and requires zero or one operands");
          values[index] = left && right;
          break;
        }
        case Op::Select: {
          const auto condition = values[instruction.first];
          TORCH_CHECK_VALUE(
              condition == 0 || condition == 1,
              "Select requires a zero or one condition");
          values[index] =
              values[condition ? instruction.second : instruction.third];
          break;
        }
        case Op::Call:
        case Op::PCall: {
          using Host = int64_t (*)(const std::vector<int64_t>&);
          using PointerHost = int64_t (*)(const int64_t*, size_t);
          std::vector<int64_t> arguments;
          arguments.reserve(instruction.operands.size());
          for (const auto operand : instruction.operands) {
            arguments.push_back(values[operand]);
          }
          const auto address = static_cast<uintptr_t>(instruction.first);
          values[index] = instruction.op == Op::PCall
              ? reinterpret_cast<PointerHost>(address)(
                    arguments.data(), arguments.size())
              : reinterpret_cast<Host>(address)(arguments);
          break;
        }
        case Op::FFromInt:
          values[index] =
              to_bits(static_cast<double>(values[instruction.first]));
          break;
        case Op::FNeg:
          values[index] = to_bits(-from_bits(values[instruction.first]));
          break;
        case Op::FSqrt:
          values[index] =
              to_bits(std::sqrt(from_bits(values[instruction.first])));
          break;
        case Op::FRound32:
          values[index] = to_bits(static_cast<double>(
              static_cast<float>(from_bits(values[instruction.first]))));
          break;
        case Op::FToBits32: {
          const float narrowed =
              static_cast<float>(from_bits(values[instruction.first]));
          int32_t bits;
          std::memcpy(&bits, &narrowed, sizeof(bits));
          values[index] = bits;
          break;
        }
        case Op::FAdd:
          values[index] = to_bits(
              from_bits(values[instruction.first]) +
              from_bits(values[instruction.second]));
          break;
        case Op::FSub:
          values[index] = to_bits(
              from_bits(values[instruction.first]) -
              from_bits(values[instruction.second]));
          break;
        case Op::FMul:
          values[index] = to_bits(
              from_bits(values[instruction.first]) *
              from_bits(values[instruction.second]));
          break;
        case Op::FDiv:
          values[index] = to_bits(
              from_bits(values[instruction.first]) /
              from_bits(values[instruction.second]));
          break;
        case Op::FPow:
          values[index] = to_bits(std::pow(
              from_bits(values[instruction.first]),
              from_bits(values[instruction.second])));
          break;
      }
    }
  }

  std::vector<bool> integer_inputs;
  std::vector<size_t> tensor_inputs;
  std::vector<Instruction> instructions;
  std::vector<THPObjectPtr> call_owners;
};

struct PythonBoxedNumeric {
  PythonBoxedNumeric(
      size_t input_count,
      py::handle numeric_plan,
      std::shared_ptr<CompiledBoxedEvaluation> compiled,
      std::vector<size_t> output_indices)
      : input_count(input_count),
        compiled(std::move(compiled)),
        output_indices(std::move(output_indices)) {
    if (!this->compiled || this->compiled->late_count != 0 ||
        !PyTuple_CheckExact(numeric_plan.ptr()) ||
        PyTuple_GET_SIZE(numeric_plan.ptr()) != 2) {
      throw py::value_error(
          "Boxed numeric evaluation requires an early-only plan");
    }
    BoxedNumericPlan numeric(
        numeric_plan.ptr(), input_count, this->compiled->early_count);
    using Op = BoxedNumericPlan::Op;
    auto* rows = PyTuple_GET_ITEM(numeric_plan.ptr(), 1);
    for (size_t index = 0; index < numeric.instructions.size(); ++index) {
      auto op = numeric.instructions[index].op;
      auto* tag = PyTuple_GET_ITEM(PyTuple_GET_ITEM(rows, index), 0);
      if (op == Op::Pointer || op == Op::Call || op == Op::PCall ||
          (op >= Op::FFromInt && op <= Op::FPow) ||
          PyUnicode_CompareWithASCIIString(tag, "fconst") == 0) {
        throw py::value_error(
            "Boxed numeric evaluation requires pure integer instructions");
      }
    }
    leaves = numeric.leaf_bindings();
    if (leaves.size() != this->compiled->leaf_count ||
        std::any_of(
            this->output_indices.begin(),
            this->output_indices.end(),
            [&](size_t index) {
              return index >= this->compiled->early_count;
            })) {
      throw py::value_error(
          "Boxed numeric bindings differ from the compiled plan");
    }
  }

  py::object call(py::handle inputs) const {
    if (!PyList_CheckExact(inputs.ptr()) ||
        static_cast<size_t>(PyList_GET_SIZE(inputs.ptr())) < input_count) {
      throw py::type_error(
          "Boxed numeric inputs require the original input prefix");
    }
    std::vector<int64_t> leaf_values(leaves.size());
    std::vector<int64_t> values(compiled->early_count);
    for (size_t index = 0; index < leaves.size(); ++index) {
      const auto& leaf = leaves[index];
      auto* input = PyList_GET_ITEM(inputs.ptr(), leaf.first);
      if (leaf.op != BoxedNumericPlan::Op::Boxed &&
          !THPVariable_CheckExact(input)) {
        throw py::type_error(
            "Boxed numeric metadata requires exact Tensor inputs");
      }
      leaf_values[index] = BoxedNumericPlan::read_input(leaf, input);
    }
    if (compiled->early(leaf_values.data(), values.data()) != 0) {
      return py::none();
    }
    py::tuple result(output_indices.size());
    for (size_t index = 0; index < output_indices.size(); ++index) {
      result[index] = py::int_(values[output_indices[index]]);
    }
    return result;
  }

  const size_t input_count;
  const std::shared_ptr<CompiledBoxedEvaluation> compiled;
  const std::vector<size_t> output_indices;
  std::vector<BoxedNumericPlan::Instruction> leaves;
};

struct CutValue {
  int64_t value;
  bool is_index;

  int64_t read(py::tuple values) const {
    if (!is_index) {
      return value;
    }
    if (value < 0 || static_cast<size_t>(value) >= values.size()) {
      throw py::value_error("Cut value index is outside the numeric batch");
    }
    auto* item = values[value].ptr();
    if (!PyLong_CheckExact(item)) {
      throw py::type_error("Cut numeric values must be exact integers");
    }
    auto result = PyLong_AsLongLong(item);
    if (PyErr_Occurred()) {
      throw py::error_already_set();
    }
    return result;
  }
};

struct CutTensor {
  std::string root;
  int64_t input_index;
  std::vector<CutValue> sizes;
  std::vector<CutValue> strides;
  CutValue offset;
};

struct PythonCut {
  struct Binding {
    size_t argument;
    int64_t element;
    std::optional<CutTensor> tensor;
    CutValue scalar{0, false};
    enum class Conversion { Integer, Floating, Boolean };
    Conversion conversion = Conversion::Integer;
  };

  PythonCut(c10::OperatorHandle op, py::tuple args, py::dict kwargs)
      : op(std::move(op)) {
    const auto& schema = this->op.schema();
    // Only the existing schema converter sees these representative values.
    const auto example = at::Tensor(at::detail::empty_meta({0}, at::kByte));
    auto tensor = py::cast(example);
    auto leaf = [&](py::handle value, size_t argument, int64_t element) {
      if (py::isinstance<CutTensor>(value)) {
        auto source = py::cast<CutTensor>(value);
        if (source.sizes.size() != source.strides.size()) {
          throw py::value_error("Cut sizes and strides have different ranks");
        }
        bindings.push_back({argument, element, std::move(source)});
        return tensor;
      }
      if (py::isinstance<CutValue>(value)) {
        bindings.push_back(
            {argument, element, std::nullopt, py::cast<CutValue>(value)});
        return py::object(py::int_(0));
      }
      if (THPVariable_Check(value.ptr())) {
        throw py::value_error("Cut tensors require prepared root bindings");
      }
      return py::reinterpret_borrow<py::object>(value);
    };
    auto argument = [&](py::handle value, size_t index) -> py::object {
      if (PyList_CheckExact(value.ptr()) || PyTuple_CheckExact(value.ptr())) {
        auto items = py::reinterpret_borrow<py::sequence>(value);
        py::list result(items.size());
        for (size_t i = 0; i < items.size(); ++i) {
          result[i] = leaf(items[i], index, i);
        }
        return result;
      }
      return leaf(value, index, -1);
    };
    py::tuple positional(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
      positional[i] = argument(args[i], i);
    }
    py::dict keywords;
    for (auto item : kwargs) {
      auto name = py::cast<std::string>(item.first);
      auto found = std::find_if(
          schema.arguments().begin(),
          schema.arguments().end(),
          [&](const c10::Argument& arg) { return arg.name() == name; });
      if (found == schema.arguments().end()) {
        throw py::value_error("Cut keyword is absent from the operator schema");
      }
      keywords[item.first] =
          argument(item.second, found - schema.arguments().begin());
    }
    try {
      stack_template = torch::jit::createStackForSchema(
          schema,
          positional,
          py::reinterpret_borrow<py::kwargs>(keywords),
          std::nullopt);
    } catch (const torch::jit::schema_match_error& error) {
      throw py::value_error(error.what());
    }
    auto scalar = [&](const c10::IValue& value) {
      if (value.isTensor()) {
        const auto& item = value.toTensor();
        return !item.defined() ||
            item.unsafeGetTensorImpl() == example.unsafeGetTensorImpl();
      }
      return value.isInt() || value.isDouble() || value.isBool() ||
          value.isComplexDouble() || value.isString() || value.isNone() ||
          value.isDevice();
    };
    for (const auto& value : stack_template) {
      if (value.isList()) {
        for (const auto& item : value.toList()) {
          if (!scalar(item)) {
            throw py::value_error("Cut arguments require flat schema lists");
          }
        }
      } else if (!scalar(value)) {
        throw py::value_error("Cut argument is outside the normalized schema");
      }
    }
    for (auto& binding : bindings) {
      auto value = stack_template.at(binding.argument);
      if (binding.element >= 0) {
        if (!value.isList() ||
            static_cast<size_t>(binding.element) >= value.toList().size()) {
          throw py::value_error(
              "Cut list binding differs from schema conversion");
        }
        value = value.toList().get(binding.element);
      }
      if (binding.tensor) {
        if (!value.isTensor()) {
          throw py::value_error(
              "Cut tensor binding requires a Tensor argument");
        }
      } else if (value.isInt()) {
        binding.conversion = Binding::Conversion::Integer;
      } else if (value.isDouble()) {
        binding.conversion = Binding::Conversion::Floating;
      } else if (value.isBool()) {
        binding.conversion = Binding::Conversion::Boolean;
      } else {
        throw py::value_error("Cut scalar binding requires a numeric argument");
      }
    }
  }

  py::object call(
      py::list inputs,
      py::dict boundary,
      py::tuple values,
      py::set known) const {
    if (at::impl::torch_function_mode_enabled() ||
        c10::impl::TorchDispatchModeTLS::stack_len() != 0) {
      return py::none();
    }
    std::vector<at::Tensor> roots;
    roots.reserve(bindings.size());
    for (const auto& binding : bindings) {
      if (!binding.tensor) {
        continue;
      }
      const auto& source = *binding.tensor;
      auto key = py::str(source.root);
      PyObject* root = PyDict_GetItemWithError(boundary.ptr(), key.ptr());
      if (root == nullptr && PyErr_Occurred()) {
        throw py::error_already_set();
      }
      if (root == nullptr && source.input_index >= 0 &&
          static_cast<size_t>(source.input_index) < inputs.size()) {
        root = PyList_GET_ITEM(inputs.ptr(), source.input_index);
      }
      if (root == nullptr || !THPVariable_CheckExact(root)) {
        return py::none();
      }
      roots.push_back(THPVariable_Unpack(root));
    }
    auto stack = stack_template;
    for (auto& value : stack) {
      if (value.isList()) {
        value = value.toList().copy();
      }
    }
    std::vector<at::Tensor> views;
    views.reserve(roots.size());
    for (const auto& binding : bindings) {
      c10::IValue value;
      if (binding.tensor) {
        const auto& source = *binding.tensor;
        std::vector<int64_t> sizes, strides;
        for (auto item : source.sizes) {
          sizes.push_back(item.read(values));
        }
        for (auto item : source.strides) {
          strides.push_back(item.read(values));
        }
        auto offset = source.offset.read(values);
        {
          py::gil_scoped_release no_gil;
          views.push_back(
              at::as_strided(roots[views.size()], sizes, strides, offset));
        }
        value = views.back();
      } else {
        auto number = binding.scalar.read(values);
        switch (binding.conversion) {
          case Binding::Conversion::Integer:
            value = number;
            break;
          case Binding::Conversion::Floating:
            value = static_cast<double>(number);
            break;
          case Binding::Conversion::Boolean:
            value = static_cast<bool>(number);
            break;
        }
      }
      if (binding.element < 0) {
        stack[binding.argument] = std::move(value);
      } else {
        stack[binding.argument].toList().set(binding.element, std::move(value));
      }
    }
    for (const auto& view : views) {
      auto storage = view.storage();
      auto invalid = storage.data() == nullptr &&
          storage.device_type() != c10::DeviceType::Meta &&
          storage.sym_nbytes() != 0;
      TORCH_CHECK(
          !invalid,
          "Attempted to access the data pointer on an invalid python storage.");
      auto pointer =
          py::int_(reinterpret_cast<uintptr_t>(storage.mutable_data()));
      if (PySet_Add(known.ptr(), pointer.ptr()) < 0) {
        throw py::error_already_set();
      }
    }
    const auto start = std::chrono::steady_clock::now();
    {
      at::NoGradGuard no_grad;
      py::gil_scoped_release no_gil;
      op.callBoxed(stack);
    }
    const auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    return py::make_tuple(
        torch::jit::createPyObjectForStack(std::move(stack)), seconds);
  }

  const c10::OperatorHandle op;
  torch::jit::Stack stack_template;
  std::vector<Binding> bindings;
};

struct BoxedOutput {
  enum class Kind { Buffer, None, Input, Value, Literal, View, Reference };
  Kind kind;
  int64_t value = 0;
};

std::unique_ptr<parameter_program::Program> unpack_parameter_program(
    PyObject* plan,
    size_t early_count,
    size_t pointer_count) {
  using namespace parameter_program;
  if (!PyTuple_CheckExact(plan) || PyTuple_GET_SIZE(plan) != 3 ||
      !PyUnicode_CheckExact(PyTuple_GET_ITEM(plan, 0)) ||
      PyUnicode_CompareWithASCIIString(
          PyTuple_GET_ITEM(plan, 0), "parameter_v1") != 0 ||
      !PyTuple_CheckExact(PyTuple_GET_ITEM(plan, 1)) ||
      !PyTuple_CheckExact(PyTuple_GET_ITEM(plan, 2))) {
    throw py::type_error(
        "Parameter plan must contain parameter_v1, instructions, and output nodes");
  }
  auto index = [](PyObject* value) -> uint32_t {
    auto parsed =
        unpack_nonnegative_integer(value, "Parameter instruction field");
    if (static_cast<uint64_t>(parsed) > std::numeric_limits<uint32_t>::max()) {
      throw py::value_error("Parameter instruction field exceeds uint32");
    }
    return static_cast<uint32_t>(parsed);
  };
  auto* rows = PyTuple_GET_ITEM(plan, 1);
  std::vector<Instruction> instructions;
  instructions.reserve(PyTuple_GET_SIZE(rows));
  for (Py_ssize_t position = 0; position < PyTuple_GET_SIZE(rows); ++position) {
    auto* row = PyTuple_GET_ITEM(rows, position);
    if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) < 2 ||
        !PyUnicode_CheckExact(PyTuple_GET_ITEM(row, 0))) {
      throw py::type_error(
          "Parameter instructions must be tagged exact tuples");
    }
    auto* name = PyTuple_GET_ITEM(row, 0);
    auto tag = [&](const char* value) {
      return PyUnicode_CompareWithASCIIString(name, value) == 0;
    };
    auto arity = [&](Py_ssize_t expected) {
      if (PyTuple_GET_SIZE(row) != expected) {
        throw py::value_error("Parameter instruction has an invalid arity");
      }
    };
    Instruction instruction{};
    if (tag("icmp")) {
      arity(4);
      instruction.op = Op::ICmp;
      instruction.width = 1;
      auto* predicate = PyTuple_GET_ITEM(row, 1);
      constexpr const char* predicates[] = {
          "eq", "ne", "ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge"};
      size_t selected = 0;
      if (!PyUnicode_CheckExact(predicate)) {
        throw py::type_error(
            "Parameter comparison requires an exact predicate string");
      }
      while (selected < std::size(predicates) &&
             PyUnicode_CompareWithASCIIString(
                 predicate, predicates[selected]) != 0) {
        ++selected;
      }
      if (selected == std::size(predicates)) {
        throw py::value_error("Unsupported parameter comparison predicate");
      }
      instruction.predicate = static_cast<Predicate>(selected);
      instruction.first = index(PyTuple_GET_ITEM(row, 2));
      instruction.second = index(PyTuple_GET_ITEM(row, 3));
    } else {
      instruction.width = index(PyTuple_GET_ITEM(row, 1));
      if (tag("constant")) {
        arity(3);
        instruction.op = Op::Constant;
        auto* value = PyTuple_GET_ITEM(row, 2);
        if (!PyLong_CheckExact(value)) {
          throw py::type_error(
              "Parameter constants must be exact unsigned integer bit patterns");
        }
        instruction.immediate = PyLong_AsUnsignedLongLong(value);
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
      } else if (tag("value") || tag("trunc") || tag("zext") || tag("sext")) {
        arity(3);
        instruction.op = tag("value") ? Op::Value
            : tag("trunc")            ? Op::Trunc
            : tag("zext")             ? Op::ZExt
                                      : Op::SExt;
        instruction.first = index(PyTuple_GET_ITEM(row, 2));
      } else if (tag("pointer")) {
        arity(4);
        instruction.op = Op::Pointer;
        instruction.first = index(PyTuple_GET_ITEM(row, 2));
        instruction.second = index(PyTuple_GET_ITEM(row, 3));
      } else if (tag("select")) {
        arity(5);
        instruction.op = Op::Select;
        instruction.first = index(PyTuple_GET_ITEM(row, 2));
        instruction.second = index(PyTuple_GET_ITEM(row, 3));
        instruction.third = index(PyTuple_GET_ITEM(row, 4));
      } else {
        struct Binary {
          const char* name;
          Op op;
          bool flags;
        };
        constexpr Binary binary[] = {
            {"add", Op::Add, true},
            {"sub", Op::Sub, true},
            {"mul", Op::Mul, true},
            {"udiv", Op::UDiv, true},
            {"sdiv", Op::SDiv, true},
            {"urem", Op::URem, false},
            {"srem", Op::SRem, false},
            {"and", Op::And, false},
            {"or", Op::Or, false},
            {"xor", Op::Xor, false},
            {"shl", Op::Shl, true},
            {"lshr", Op::LShr, true},
            {"ashr", Op::AShr, true}};
        const auto found = std::find_if(
            std::begin(binary), std::end(binary), [&](const auto& item) {
              return tag(item.name);
            });
        if (found == std::end(binary)) {
          throw py::value_error("Unsupported parameter instruction");
        }
        arity(found->flags ? 5 : 4);
        instruction.op = found->op;
        instruction.first = index(PyTuple_GET_ITEM(row, 2));
        instruction.second = index(PyTuple_GET_ITEM(row, 3));
        if (found->flags) {
          auto* flags = PyTuple_GET_ITEM(row, 4);
          if (!PyTuple_CheckExact(flags)) {
            throw py::type_error("Parameter flags must be an exact tuple");
          }
          for (Py_ssize_t flag_index = 0; flag_index < PyTuple_GET_SIZE(flags);
               ++flag_index) {
            auto* flag = PyTuple_GET_ITEM(flags, flag_index);
            uint32_t bit = 0;
            if (PyUnicode_CheckExact(flag)) {
              bit = PyUnicode_CompareWithASCIIString(flag, "nuw") == 0   ? NUW
                  : PyUnicode_CompareWithASCIIString(flag, "nsw") == 0   ? NSW
                  : PyUnicode_CompareWithASCIIString(flag, "exact") == 0 ? Exact
                                                                         : 0;
            }
            if (!bit || (instruction.flags & bit)) {
              throw py::value_error(
                  "Parameter flags must be supported and distinct");
            }
            instruction.flags |= bit;
          }
        }
      }
    }
    instructions.push_back(instruction);
  }
  std::vector<uint32_t> outputs;
  auto* output_nodes = PyTuple_GET_ITEM(plan, 2);
  outputs.reserve(PyTuple_GET_SIZE(output_nodes));
  for (Py_ssize_t position = 0; position < PyTuple_GET_SIZE(output_nodes);
       ++position) {
    outputs.push_back(index(PyTuple_GET_ITEM(output_nodes, position)));
  }
  return std::make_unique<Program>(
      std::move(instructions), std::move(outputs), early_count, pointer_count);
}

struct BoxedView {
  size_t root;
  std::vector<int64_t> size;
  std::vector<int64_t> stride;
  int64_t offset = 0;
  // the view's own dtype (a view_as_real / view_as_complex output): the offset
  // is then over the root's storage in the view's units, nothing of the root's
  // added
  std::optional<caffe2::TypeMeta> dtype;
};

struct BoxedViewValue {
  enum class Field { Size, Stride, Offset };
  size_t view;
  Field field;
  size_t dimension;
  size_t value;
};

struct BoxedReplayPlan {
  BoxedReplayPlan(
      PyObject* count,
      PyObject* inputs,
      PyObject* layouts,
      PyObject* prologue,
      PyObject* outputs,
      PyObject* numeric_plan,
      const PythonKernelPointerUpdates& updates,
      std::shared_ptr<CompiledBoxedEvaluation> compiled_evaluation,
      PyObject* pinned_positions,
      PyObject* const_positions)
      : input_count(unpack_nonnegative_integer(count, "Input count")),
        input_indices(unpack_input_indices(inputs, input_count)),
        pinned_indices(unpack_input_indices(pinned_positions, input_count)),
        has_explicit_output_indices(outputs != Py_None),
        compiled(std::move(compiled_evaluation)) {
    auto pointer_count = updates.pointers.size();
    if ((numeric_plan != Py_None) != updates.batch->has_numeric_updates()) {
      throw py::value_error(
          "Numeric batches require a numeric plan; pointer-only batches do not accept one");
    }
    if (numeric_plan != Py_None) {
      auto early_count = updates.batch->value_count();
      if (PyTuple_CheckExact(numeric_plan) &&
          PyTuple_GET_SIZE(numeric_plan) == 3) {
        if (!PyTuple_CheckExact(PyTuple_GET_ITEM(numeric_plan, 1))) {
          throw py::type_error(
              "Early numeric instructions must be an exact tuple");
        }
        early_count = static_cast<size_t>(
            PyTuple_GET_SIZE(PyTuple_GET_ITEM(numeric_plan, 1)));
        parameters = unpack_parameter_program(
            PyTuple_GET_ITEM(numeric_plan, 2), early_count, pointer_count);
        if (early_count > updates.batch->value_count() ||
            parameters->output_count() !=
                updates.batch->value_count() - early_count) {
          throw py::value_error(
              "Early instructions and parameter outputs must cover the numeric value slots");
        }
      }
      numeric = std::make_unique<BoxedNumericPlan>(
          numeric_plan, input_count, early_count);
      for (auto index : input_indices) {
        if (numeric->integer_inputs[index]) {
          throw py::value_error(
              "Integer inputs cannot supply pointer bindings");
        }
      }
    }
    if (compiled) {
      if (!numeric || compiled->early_count != numeric->instructions.size() ||
          compiled->late_count !=
              (parameters ? parameters->output_count() : 0) ||
          compiled->pointer_count != pointer_count) {
        throw py::value_error(
            "Compiled evaluation counts differ from the prepared numeric plan");
      }
      if (parameters && parameters->scratch_size() && !compiled->late) {
        throw py::value_error(
            "A nonempty parameter program requires a compiled late function");
      }
      leaves = numeric->leaf_bindings();
      if (compiled->leaf_count != leaves.size()) {
        throw py::value_error(
            "Compiled leaf bindings differ from the prepared numeric plan");
      }
    }
    if (!PyTuple_CheckExact(layouts)) {
      throw py::type_error("Buffer layouts must be an exact tuple");
    }
    if (input_count > pointer_count ||
        static_cast<size_t>(PyTuple_GET_SIZE(layouts)) !=
            pointer_count - input_count) {
      throw py::value_error(
          "Pointer count must equal input count plus allocation count");
    }
    std::vector<bool> supplied(pointer_count);
    for (auto index : input_indices) {
      supplied[index] = true;
    }
    std::fill(supplied.begin() + input_count, supplied.end(), true);
    auto referenced = updates.referenced;
    for (const auto& binding : updates.tensor_map_bindings) {
      if (!numeric ||
          binding.address_offset_value_index >= numeric->instructions.size() ||
          std::any_of(
              binding.dimensions.begin(),
              binding.dimensions.end(),
              [&](auto value) {
                return value >= numeric->instructions.size();
              }) ||
          std::any_of(
              binding.strides.begin(), binding.strides.end(), [&](auto value) {
                return value >= numeric->instructions.size();
              })) {
        throw py::value_error(
            "Tensor maps require early numeric dimensions, strides and pointer offsets");
      }
      if (binding.pointer_index < input_count &&
          numeric->integer_inputs[binding.pointer_index]) {
        throw py::value_error(
            "Tensor-map pointers must name Tensor inputs or allocations");
      }
    }
    for (const auto& binding : updates.memset_bindings) {
      if (!numeric ||
          binding.bytes_value_index >= numeric->instructions.size() ||
          (binding.address_offset_value_index &&
           *binding.address_offset_value_index >=
               numeric->instructions.size())) {
        throw py::value_error("Memset bindings require early numeric values");
      }
      if (binding.pointer_index < input_count &&
          numeric->integer_inputs[binding.pointer_index]) {
        throw py::value_error(
            "Memset pointers must name Tensor inputs or allocations");
      }
    }
    if (parameters) {
      const auto early_count = numeric->instructions.size();
      for (const auto& binding : updates.pointer_bindings) {
        if (binding.address_offset_value_index &&
            *binding.address_offset_value_index >= early_count) {
          throw py::value_error(
              "Direct pointer displacements require early numeric values");
        }
      }
      for (const auto& binding : updates.grid_bindings) {
        for (auto value : binding.values) {
          if (value >= early_count) {
            throw py::value_error("Grid bindings require early numeric values");
          }
        }
        if (binding.shared_memory && *binding.shared_memory >= early_count) {
          throw py::value_error(
              "Shared-memory bindings require early numeric values");
        }
        if (binding.block) {
          for (auto value : *binding.block) {
            if (value >= early_count) {
              throw py::value_error(
                  "Block bindings require early numeric values");
            }
          }
        }
      }
      for (const auto& binding : updates.host_table_bindings) {
        for (const auto& element : binding.elements) {
          const bool late = element.pointer
              ? (element.address_offset_value_index &&
                 *element.address_offset_value_index >= early_count)
              : element.index >= early_count;
          if (late) {
            throw py::value_error(
                "Host table bindings require early numeric values");
          }
        }
      }
      for (const auto& binding : updates.memcpy_bindings) {
        if ((binding.address_offset_value_index &&
             *binding.address_offset_value_index >= early_count) ||
            (binding.source_offset_value_index &&
             *binding.source_offset_value_index >= early_count) ||
            binding.bytes_value_index >= early_count) {
          throw py::value_error("Memcpy bindings require early numeric values");
        }
      }
      for (const auto& binding : updates.rng_bindings) {
        if (binding.value_index >= early_count) {
          throw py::value_error(
              "Generator increment bindings require early numeric values");
        }
      }
      std::vector<bool> bound(parameters->output_count());
      for (const auto& binding : updates.scalar_bindings) {
        if (binding.value_index < early_count) {
          continue;
        }
        const auto output = binding.value_index - early_count;
        if (output >= bound.size() ||
            binding.width * 8 != parameters->output_width(output)) {
          throw py::value_error(
              "Parameter output width differs from its scalar binding");
        }
        bound[output] = true;
        for (auto root : parameters->output_roots()[output]) {
          referenced[root] = true;
          parameter_root_uses.push_back({binding.node, root});
        }
      }
      if (std::find(bound.begin(), bound.end(), false) != bound.end()) {
        throw py::value_error(
            "Every parameter output requires an actual scalar binding");
      }
      for (auto root : parameters->pointer_inputs()) {
        if (!referenced[root] ||
            (root < input_count && numeric->integer_inputs[root])) {
          throw py::value_error(
              "Parameter pointer inputs require captured Tensor or allocation uses");
        }
      }
      std::sort(
          parameter_root_uses.begin(),
          parameter_root_uses.end(),
          [](const auto& a, const auto& b) {
            return a.node < b.node ||
                (a.node == b.node && a.pointer_index < b.pointer_index);
          });
      parameter_root_uses.erase(
          std::unique(
              parameter_root_uses.begin(),
              parameter_root_uses.end(),
              [](const auto& a, const auto& b) {
                return a.node == b.node && a.pointer_index == b.pointer_index;
              }),
          parameter_root_uses.end());
    }
    if (!std::equal(
            supplied.begin(),
            supplied.begin() + input_count,
            referenced.begin())) {
      throw py::value_error(
          "Boxed inputs and allocations must cover exactly the prepared pointer indices");
    }
    buffers.reserve(PyTuple_GET_SIZE(layouts));
    for (Py_ssize_t index = 0; index < PyTuple_GET_SIZE(layouts); ++index) {
      auto* layout = PyTuple_GET_ITEM(layouts, index);
      if (!PyTuple_CheckExact(layout) || PyTuple_GET_SIZE(layout) != 3 ||
          !THPDtype_Check(PyTuple_GET_ITEM(layout, 0))) {
        throw py::type_error(
            "Buffer layouts must contain dtype, size and stride tuples");
      }
      auto scalar =
          reinterpret_cast<THPDtype*>(PyTuple_GET_ITEM(layout, 0))->scalar_type;
      TORCH_CHECK_VALUE(
          !c10::isQIntType(scalar), "Quantized boxed buffers are unsupported");
      auto dtype = c10::scalarTypeToTypeMeta(scalar);
      auto* dimensions = PyTuple_GET_ITEM(layout, 1);
      auto* strides = PyTuple_GET_ITEM(layout, 2);
      auto has_tag = [](PyObject* fields) {
        if (PyTuple_CheckExact(fields)) {
          for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(fields); ++i) {
            if (PyTuple_CheckExact(PyTuple_GET_ITEM(fields, i))) {
              return true;
            }
          }
        }
        return false;
      };
      if (numeric && (has_tag(dimensions) || has_tag(strides))) {
        if (!PyTuple_CheckExact(dimensions) || !PyTuple_CheckExact(strides)) {
          throw py::type_error("Output sizes and strides must be exact tuples");
        }
        auto rank = PyTuple_GET_SIZE(dimensions);
        if (PyTuple_GET_SIZE(strides) != rank) {
          throw py::value_error(
              "Dynamic buffers require matching sizes and strides");
        }
        auto unpack = [&](PyObject* fields,
                          std::vector<std::optional<size_t>>& values) {
          std::vector<int64_t> result(rank);
          values.resize(rank);
          for (Py_ssize_t i = 0; i < rank; ++i) {
            auto* field = PyTuple_GET_ITEM(fields, i);
            if (!PyTuple_CheckExact(field)) {
              result[i] = unpack_nonnegative_integer(field, "Output dimension");
              continue;
            }
            if (PyTuple_GET_SIZE(field) != 2 ||
                !PyUnicode_CheckExact(PyTuple_GET_ITEM(field, 0)) ||
                PyUnicode_CompareWithASCIIString(
                    PyTuple_GET_ITEM(field, 0), "value") != 0) {
              throw py::value_error(
                  "Dynamic dimensions must be tagged value indices");
            }
            auto value = static_cast<size_t>(unpack_nonnegative_integer(
                PyTuple_GET_ITEM(field, 1), "Dimension value index"));
            if (value >= numeric->instructions.size()) {
              throw py::value_error("Dimension value index is out of range");
            }
            values[i] = value;
          }
          return result;
        };
        std::vector<std::optional<size_t>> size_values, stride_values;
        auto size = unpack(dimensions, size_values);
        auto stride = unpack(strides, stride_values);
        auto buffer = static_cast<size_t>(index);
        for (size_t i = 0; i < static_cast<size_t>(rank); ++i) {
          if (size_values[i]) {
            layout_values.push_back({buffer, i, *size_values[i], false});
          } else if (size[i] < 0) {
            throw py::value_error("Dynamic buffer extents must be nonnegative");
          }
          if (stride_values[i]) {
            layout_values.push_back({buffer, i, *stride_values[i], true});
          }
        }
        dynamic_buffers.push_back(buffer);
        buffers.push_back({dtype, std::move(size), std::move(stride), 0});
        continue;
      }
      auto size = unpack_static_dimensions(dimensions);
      auto stride = unpack_static_dimensions(strides);
      uint64_t numel = 0;
      TORCH_CHECK_VALUE(
          !c10::safe_multiplies_u64(size, &numel) &&
              numel <= std::numeric_limits<int64_t>::max(),
          "Boxed buffer element count overflowed");
      auto nbytes =
          at::detail::computeStorageNbytes(size, stride, dtype.itemsize());
      buffers.push_back({dtype, std::move(size), std::move(stride), nbytes});
    }
    if (outputs == Py_None) {
      output_slots.reserve(buffers.size());
      for (size_t index = 0; index < buffers.size(); ++index) {
        output_slots.push_back(
            {BoxedOutput::Kind::Buffer, static_cast<int64_t>(index)});
      }
    } else {
      if (!PyTuple_CheckExact(outputs)) {
        throw py::type_error("Output indices must be an exact tuple");
      }
      output_slots.reserve(PyTuple_GET_SIZE(outputs));
      for (Py_ssize_t index = 0; index < PyTuple_GET_SIZE(outputs); ++index) {
        auto* item = PyTuple_GET_ITEM(outputs, index);
        if (item == Py_None) {
          output_slots.push_back({BoxedOutput::Kind::None});
          continue;
        }
        if (PyTuple_CheckExact(item)) {
          if ((PyTuple_GET_SIZE(item) == 5 || PyTuple_GET_SIZE(item) == 6) &&
              PyUnicode_CheckExact(PyTuple_GET_ITEM(item, 0)) &&
              PyUnicode_CompareWithASCIIString(
                  PyTuple_GET_ITEM(item, 0), "view") == 0) {
            auto root = static_cast<size_t>(unpack_nonnegative_integer(
                PyTuple_GET_ITEM(item, 1), "View root index"));
            if (root >= pointer_count ||
                (root < input_count && numeric &&
                 numeric->integer_inputs[root])) {
              throw py::value_error(
                  "View roots must name Tensor inputs or allocations");
            }
            auto* sizes = PyTuple_GET_ITEM(item, 2);
            auto* strides = PyTuple_GET_ITEM(item, 3);
            if (!PyTuple_CheckExact(sizes) || !PyTuple_CheckExact(strides) ||
                PyTuple_GET_SIZE(sizes) != PyTuple_GET_SIZE(strides)) {
              throw py::type_error(
                  "View sizes and strides must be matching exact tuples");
            }
            auto unpack = [&](PyObject* field,
                              BoxedViewValue::Field kind,
                              size_t dimension) -> int64_t {
              if (PyTuple_CheckExact(field)) {
                if (PyTuple_GET_SIZE(field) != 2 ||
                    !PyUnicode_CheckExact(PyTuple_GET_ITEM(field, 0)) ||
                    PyUnicode_CompareWithASCIIString(
                        PyTuple_GET_ITEM(field, 0), "value") != 0) {
                  throw py::value_error(
                      "Dynamic view fields must be tagged value indices");
                }
                auto value = static_cast<size_t>(unpack_nonnegative_integer(
                    PyTuple_GET_ITEM(field, 1), "View value index"));
                if (!numeric || value >= numeric->instructions.size()) {
                  throw py::value_error(
                      "View value index must name a numeric instruction");
                }
                view_values.push_back({views.size(), kind, dimension, value});
                return 0;
              }
              if (!PyLong_CheckExact(field)) {
                throw py::type_error(
                    "View fields must be exact integers or tagged value indices");
              }
              auto value = PyLong_AsLongLong(field);
              if (PyErr_Occurred()) {
                throw py::error_already_set();
              }
              return value;
            };
            BoxedView view{root, {}, {}};
            for (Py_ssize_t dimension = 0; dimension < PyTuple_GET_SIZE(sizes);
                 ++dimension) {
              view.size.push_back(unpack(
                  PyTuple_GET_ITEM(sizes, dimension),
                  BoxedViewValue::Field::Size,
                  dimension));
              view.stride.push_back(unpack(
                  PyTuple_GET_ITEM(strides, dimension),
                  BoxedViewValue::Field::Stride,
                  dimension));
            }
            view.offset = unpack(
                PyTuple_GET_ITEM(item, 4), BoxedViewValue::Field::Offset, 0);
            if (PyTuple_GET_SIZE(item) == 6) {
              auto* dtype = PyTuple_GET_ITEM(item, 5);
              if (!THPDtype_Check(dtype)) {
                throw py::type_error("View dtype must be a torch.dtype");
              }
              auto scalar = reinterpret_cast<THPDtype*>(dtype)->scalar_type;
              TORCH_CHECK_VALUE(
                  !c10::isQIntType(scalar),
                  "Quantized view outputs are unsupported");
              view.dtype = c10::scalarTypeToTypeMeta(scalar);
            }
            output_slots.push_back(
                {BoxedOutput::Kind::View, static_cast<int64_t>(views.size())});
            views.push_back(std::move(view));
            continue;
          }
          if (PyTuple_GET_SIZE(item) != 2 ||
              !PyUnicode_CheckExact(PyTuple_GET_ITEM(item, 0))) {
            throw py::type_error(
                "Output references must be exact tagged pairs");
          }
          auto* tag = PyTuple_GET_ITEM(item, 0);
          auto* field = PyTuple_GET_ITEM(item, 1);
          if (PyUnicode_CompareWithASCIIString(tag, "output") == 0) {
            auto output =
                unpack_nonnegative_integer(field, "Output reference index");
            if (static_cast<uint64_t>(output) >= output_slots.size()) {
              throw py::value_error(
                  "Output references must name preceding Tensor outputs");
            }
            auto kind = output_slots[output].kind;
            if (kind != BoxedOutput::Kind::Buffer &&
                kind != BoxedOutput::Kind::Input &&
                kind != BoxedOutput::Kind::View &&
                kind != BoxedOutput::Kind::Reference) {
              throw py::value_error(
                  "Output references must name preceding Tensor outputs");
            }
            output_slots.push_back({BoxedOutput::Kind::Reference, output});
          } else if (PyUnicode_CompareWithASCIIString(tag, "input") == 0) {
            auto input =
                unpack_nonnegative_integer(field, "Output input index");
            if (static_cast<uint64_t>(input) >= input_count ||
                (numeric && numeric->integer_inputs[input])) {
              throw py::value_error(
                  "Borrowed outputs must name declared Tensor inputs");
            }
            output_slots.push_back({BoxedOutput::Kind::Input, input});
          } else if (PyUnicode_CompareWithASCIIString(tag, "value") == 0) {
            auto value =
                unpack_nonnegative_integer(field, "Output value index");
            if (!numeric ||
                static_cast<uint64_t>(value) >= numeric->instructions.size()) {
              throw py::value_error(
                  "Output value index must name a numeric instruction");
            }
            output_slots.push_back({BoxedOutput::Kind::Value, value});
          } else if (PyUnicode_CompareWithASCIIString(tag, "literal") == 0) {
            if (!PyLong_CheckExact(field)) {
              throw py::type_error("Output literals must be exact integers");
            }
            auto value = PyLong_AsLongLong(field);
            if (PyErr_Occurred()) {
              throw py::error_already_set();
            }
            output_slots.push_back({BoxedOutput::Kind::Literal, value});
          } else {
            throw py::value_error("Unknown output reference kind");
          }
          continue;
        }
        auto value = unpack_nonnegative_integer(item, "Output index");
        if (static_cast<uint64_t>(value) >= buffers.size() ||
            std::any_of(
                output_slots.begin(),
                output_slots.end(),
                [&](const auto& slot) {
                  return slot.kind == BoxedOutput::Kind::Buffer &&
                      slot.value == value;
                })) {
          throw py::value_error("Output indices must be distinct and in range");
        }
        output_slots.push_back({BoxedOutput::Kind::Buffer, value});
      }
    }
    if (prologue != Py_None) {
      if (!PyTuple_CheckExact(prologue) || PyTuple_GET_SIZE(prologue) != 2) {
        throw py::type_error(
            "Input prologue must contain indices and the native normalization builtin");
      }
      copy_indices =
          unpack_input_indices(PyTuple_GET_ITEM(prologue, 0), input_count);
      if (numeric) {
        for (auto index : copy_indices) {
          if (numeric->integer_inputs[index]) {
            throw py::value_error(
                "Integer inputs cannot require alignment copies");
          }
        }
      }
      auto* builtin = PyTuple_GET_ITEM(prologue, 1);
      THPObjectPtr guards(PyImport_ImportModule("torch._C._dynamo.guards"));
      if (!guards) {
        throw py::error_already_set();
      }
      auto* definition = PyModule_Check(guards.get())
          ? PyModule_GetDef(guards.get())
          : nullptr;
      auto* method = definition ? definition->m_methods : nullptr;
      while (method && method->ml_name &&
             std::strcmp(method->ml_name, "copy_if_misaligned") != 0) {
        ++method;
      }
      if (!definition || !definition->m_name ||
          std::strcmp(definition->m_name, "torch._C._dynamo.guards") != 0 ||
          !method || !method->ml_name || method->ml_flags != METH_O ||
          Py_TYPE(builtin) != &PyCFunction_Type ||
          PyCFunction_GET_FLAGS(builtin) != METH_O ||
          PyCFunction_GET_FUNCTION(builtin) != method->ml_meth ||
          PyCFunction_GET_SELF(builtin) != guards.get() ||
          PyDict_GetItemString(
              PyModule_GetDict(guards.get()), "copy_if_misaligned") !=
              builtin) {
        throw py::value_error(
            "Input prologue requires the exact native copy_if_misaligned builtin");
      }
      copy_builtin = Py_NewRef(builtin);
      copy_function = method->ml_meth;
    }
    for (auto index : pinned_indices) {
      TORCH_CHECK_VALUE(
          (!numeric || !numeric->integer_inputs[index]) &&
              std::find(input_indices.begin(), input_indices.end(), index) !=
                  input_indices.end() &&
              std::find(copy_indices.begin(), copy_indices.end(), index) ==
                  copy_indices.end(),
          "Pinned positions must name used Tensor inputs without alignment copies");
    }
    // Inputs the replay only reads: their address is taken through the const
    // accessor per call, so a copy-on-write tensor there stays lazy as it would
    // under eager. Every other input is read through data_ptr(), which
    // materializes (a frontend that names no const positions keeps that for
    // all of them).
    const_inputs.assign(input_count, false);
    for (auto index : unpack_input_indices(const_positions, input_count)) {
      TORCH_CHECK_VALUE(
          !numeric || !numeric->integer_inputs[index],
          "Const positions must name Tensor inputs");
      const_inputs[index] = true;
    }
  }

  size_t input_count;
  std::vector<size_t> input_indices;
  std::vector<size_t> pinned_indices;
  std::vector<bool> const_inputs;
  const bool has_explicit_output_indices;
  std::vector<BoxedBufferLayout> buffers;
  std::vector<BoxedLayoutValue> layout_values;
  std::vector<size_t> dynamic_buffers;
  std::vector<BoxedOutput> output_slots;
  std::vector<BoxedView> views;
  std::vector<BoxedViewValue> view_values;
  std::vector<size_t> copy_indices;
  THPObjectPtr copy_builtin;
  PyCFunction copy_function = nullptr;
  std::unique_ptr<BoxedNumericPlan> numeric;
  std::unique_ptr<parameter_program::Program> parameters;
  std::shared_ptr<CompiledBoxedEvaluation> compiled;
  std::vector<BoxedNumericPlan::Instruction> leaves;
  std::vector<at::cuda::detail::KernelRootUse> parameter_root_uses;
  std::optional<at::cuda::detail::InputReleasePlan> release;
};

struct BoxedInvocation {
  struct Buffer {
    c10::DataPtr data_ptr;
    c10::intrusive_ptr<c10::StorageImpl> storage;
  };

  explicit BoxedInvocation(const BoxedReplayPlan& plan)
      : originals(plan.input_count),
        normalized(plan.input_count),
        buffers(plan.buffers.size()),
        views(plan.views) {
    if (plan.numeric) {
      values.resize(
          plan.numeric->instructions.size() +
          (plan.parameters ? plan.parameters->output_count() : 0));
      layouts = plan.buffers;
    }
    if (plan.parameters) {
      if (!plan.compiled) {
        parameter_scratch.resize(plan.parameters->scratch_size());
      }
    }
    leaf_values.resize(plan.leaves.size());
    pinned_storages.reserve(plan.pinned_indices.size());
  }

  void clear() {
    result = nullptr;
    for (auto& buffer : buffers) {
      buffer.storage.reset();
      buffer.data_ptr.clear();
    }
    for (auto& input : normalized) {
      input = nullptr;
    }
    for (auto& input : originals) {
      input = nullptr;
    }
    pinned_storages.clear();
    pending.reset();
  }

  std::unique_ptr<c10::cuda::CUDACachingAllocator::PendingGraphInputs> pending;
  std::vector<THPObjectPtr> originals;
  std::vector<THPObjectPtr> normalized;
  std::vector<c10::Storage> pinned_storages;
  std::vector<Buffer> buffers;
  std::vector<int64_t> values;
  std::vector<int64_t> leaf_values;
  std::vector<parameter_program::Value> parameter_scratch;
  std::vector<BoxedBufferLayout> layouts;
  std::vector<BoxedView> views;
  THPObjectPtr result;
};

class PythonGraphReplayOwner {
 public:
  PythonGraphReplayOwner(
      std::shared_ptr<at::cuda::CUDAGraph> graph,
      const PythonKernelPointerUpdates& updates,
      PyObject* stream_object,
      PyObject* resources,
      c10::cuda::CUDAStream stream,
      std::unique_ptr<BoxedReplayPlan> boxed_plan = nullptr,
      std::unique_ptr<BoxedInvocation> invocation = nullptr,
      PyObject* release_plan = Py_None)
      : pointers_(updates.pointers.size()),
        stream_(stream),
        stream_owner_(Py_NewRef(stream_object)),
        resources_(Py_NewRef(resources)),
        boxed_plan_(std::move(boxed_plan)),
        invocation_(std::move(invocation)) {
    c10::cuda::CUDAGuard guard(stream_.device_index());
    context_ = current_replay_context();
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream_.stream(), &capture));
    TORCH_CHECK(
        capture == cudaStreamCaptureStatusNone,
        "Cannot bind a capturing replay stream");
    lease_ = std::make_unique<at::cuda::detail::GraphReplayLease>(
        std::move(graph), updates.batch, stream_.device_index());
    if (release_plan != Py_None) {
      if (!boxed_plan_ || !boxed_plan_->has_explicit_output_indices ||
          !PyTuple_CheckExact(release_plan) ||
          PyTuple_GET_SIZE(release_plan) != 2) {
        throw py::value_error(
            "Input release requires a boxed allocation plan and exact (steps, nodes) tuple");
      }
      auto* steps = PyTuple_GET_ITEM(release_plan, 0);
      auto* nodes = PyTuple_GET_ITEM(release_plan, 1);
      if (!PyTuple_CheckExact(steps) || !PyTuple_CheckExact(nodes)) {
        throw py::type_error("Release steps and nodes must be exact tuples");
      }
      std::vector<at::cuda::detail::InputReleaseStep> operations;
      std::vector<uintptr_t> handles;
      for (Py_ssize_t index = 0; index < PyTuple_GET_SIZE(steps); ++index) {
        auto* step = PyTuple_GET_ITEM(steps, index);
        if (!PyTuple_CheckExact(step) || PyTuple_GET_SIZE(step) != 2 ||
            !PyUnicode_CheckExact(PyTuple_GET_ITEM(step, 0))) {
          throw py::type_error(
              "Release operations require exact (kind, index) pairs");
        }
        auto* name = PyTuple_GET_ITEM(step, 0);
        at::cuda::detail::InputReleaseKind kind;
        if (PyUnicode_CompareWithASCIIString(name, "allocate") == 0) {
          kind = at::cuda::detail::InputReleaseKind::Allocate;
        } else if (PyUnicode_CompareWithASCIIString(name, "kernel") == 0) {
          kind = at::cuda::detail::InputReleaseKind::Kernel;
        } else if (PyUnicode_CompareWithASCIIString(name, "drop") == 0) {
          kind = at::cuda::detail::InputReleaseKind::Drop;
        } else {
          throw py::value_error("Unknown release operation");
        }
        operations.push_back(
            {kind,
             static_cast<size_t>(unpack_nonnegative_integer(
                 PyTuple_GET_ITEM(step, 1), "Release index"))});
      }
      for (Py_ssize_t index = 0; index < PyTuple_GET_SIZE(nodes); ++index) {
        handles.push_back(static_cast<uintptr_t>(unpack_nonnegative_integer(
            PyTuple_GET_ITEM(nodes, index), "Release node")));
      }
      auto prepared = lease_->prepare_input_release(
          std::move(operations),
          handles,
          boxed_plan_->input_count,
          boxed_plan_->buffers.size(),
          boxed_plan_->parameter_root_uses);
      for (const auto& output : boxed_plan_->output_slots) {
        auto input = output.kind == BoxedOutput::Kind::Input
            ? static_cast<size_t>(output.value)
            : output.kind == BoxedOutput::Kind::View
            ? boxed_plan_->views[output.value].root
            : boxed_plan_->input_count;
        if (input < boxed_plan_->input_count &&
            std::find(
                prepared.candidates.begin(),
                prepared.candidates.end(),
                input) != prepared.candidates.end()) {
          throw py::value_error(
              "Returned Tensor inputs cannot be released before replay");
        }
      }
      if (boxed_plan_->numeric) {
        for (auto index : prepared.candidates) {
          if (boxed_plan_->numeric->integer_inputs[index]) {
            throw py::value_error(
                "Integer inputs cannot be released as saved Tensor inputs");
          }
        }
      }
      for (auto index : boxed_plan_->pinned_indices) {
        TORCH_CHECK_VALUE(
            std::find(
                prepared.candidates.begin(),
                prepared.candidates.end(),
                index) == prepared.candidates.end(),
            "Pinned host inputs cannot be released before submission");
      }
      boxed_plan_->release = std::move(prepared);
    }
    if (boxed_plan_ && boxed_plan_->parameters) {
      call_function_ = &PythonGraphReplayOwner::call_impl<true, false, true>;
      dispatch_call_function_ =
          &PythonGraphReplayOwner::call_impl<true, true, true>;
    } else if (boxed_plan_ && boxed_plan_->numeric) {
      call_function_ = &PythonGraphReplayOwner::call_impl<true>;
      dispatch_call_function_ = &PythonGraphReplayOwner::call_impl<true, true>;
    } else {
      call_function_ = &PythonGraphReplayOwner::call_impl<false>;
      dispatch_call_function_ = &PythonGraphReplayOwner::call_impl<false, true>;
    }
  }

  void replay(py::handle pointers) {
    std::unique_lock lock(mutex_, std::try_to_lock);
    TORCH_CHECK(lock.owns_lock(), "Native graph replay owner is busy");
    TORCH_CHECK(
        !closed_ && !failed_, "Native graph replay owner is closed or failed");
    unpack_pointer_values(pointers, pointers_);
    TORCH_CHECK(
        c10::cuda::getCurrentCUDAStream(stream_.device_index()) == stream_,
        "Native graph replay requires its bound stream");
    enqueue_locked();
  }

  py::tuple call(py::handle inputs) {
    std::unique_lock lock(mutex_, std::try_to_lock);
    TORCH_CHECK(lock.owns_lock(), "Native graph replay owner is busy");
    TORCH_CHECK(
        !closed_ && !failed_, "Native graph replay owner is closed or failed");
    return (this->*call_function_)(inputs, nullptr, nullptr);
  }

  template <bool Numeric, bool Dispatched = false, bool Parameters = false>
  py::tuple call_impl(
      py::handle inputs,
      std::vector<THPObjectPtr>* snapshot = nullptr,
      const std::atomic<bool>* invalidated = nullptr) {
    static_assert(!Parameters || Numeric);
    auto& plan = *boxed_plan_;
    c10::cuda::CUDAGuard guard(stream_.device_index());
    check_boxed_stream();
    THPObjectPtr result(PyTuple_New(plan.output_slots.size()));
    if (!result) {
      throw py::error_already_set();
    }
    check_boxed_stream();
    if constexpr (Dispatched) {
      TORCH_CHECK(
          !invalidated->load(),
          "Native boxed dispatch was invalidated during admission");
    }
    // Validate after result allocation, which can trigger GC and mutate a box.
    if (!PyList_CheckExact(inputs.ptr())) {
      throw py::type_error("Boxed graph inputs must be an exact list");
    }
    const auto given = static_cast<size_t>(PyList_GET_SIZE(inputs.ptr()));
    if constexpr (Dispatched) {
      // the dispatcher's snapshot holds every input; the caller's box may omit
      // the dispatcher's hidden trailing inputs
      if (given > plan.input_count) {
        throw py::value_error(
            "Boxed graph input count differs from preparation");
      }
    } else if (given != plan.input_count) {
      throw py::value_error("Boxed graph input count differs from preparation");
    }
    if constexpr (Numeric && !Dispatched) {
      for (auto index : plan.numeric->tensor_inputs) {
        if (!THPVariable_CheckExact(PyList_GET_ITEM(inputs.ptr(), index))) {
          throw py::type_error(
              "Declared Tensor inputs must be Tensors or Parameters");
        }
      }
    } else if constexpr (!Dispatched) {
      for (size_t index = 0; index < plan.input_count; ++index) {
        if (!THPVariable_CheckExact(PyList_GET_ITEM(inputs.ptr(), index))) {
          throw py::type_error(
              "Boxed graph inputs must be Tensors or Parameters");
        }
      }
    }
    auto& invocation = *invocation_;
    for (size_t index = 0; index < plan.input_count; ++index) {
      auto* value = index < given ? PyList_GET_ITEM(inputs.ptr(), index)
                                  : (*snapshot)[index].get();
      if constexpr (Dispatched) {
        if (value != (*snapshot)[index].get()) {
          throw py::value_error(
              "Boxed inputs changed after native guard selection");
        }
        if constexpr (Numeric) {
          if (!plan.numeric->integer_inputs[index] &&
              !THPVariable_CheckExact(value)) {
            throw py::type_error(
                "Declared Tensor inputs must be Tensors or Parameters");
          }
        } else if (!THPVariable_CheckExact(value)) {
          throw py::type_error(
              "Boxed graph inputs must be Tensors or Parameters");
        }
      } else {
        invocation.originals[index] = Py_NewRef(value);
      }
    }
    if constexpr (Dispatched) {
      invocation.originals.swap(*snapshot);
    }
    invocation.result = std::move(result);
    bool started = false;
    try {
      for (auto index : plan.pinned_indices) {
        invocation.pinned_storages.push_back(
            THPVariable_Unpack(invocation.originals[index].get()).storage());
      }
      if constexpr (Numeric) {
        if (plan.compiled) {
          for (size_t index = 0; index < plan.leaves.size(); ++index) {
            invocation.leaf_values[index] = BoxedNumericPlan::read_input(
                plan.leaves[index], invocation.originals);
          }
          const auto status = plan.compiled->early(
              invocation.leaf_values.data(), invocation.values.data());
          TORCH_CHECK_VALUE(
              status == 0,
              "Compiled integer evaluation failed with status ",
              status);
        } else {
          plan.numeric->evaluate(invocation.originals, invocation.values);
        }
        if constexpr (!Parameters) {
          lease_->validate_values(invocation.values);
        }
        for (const auto& binding : plan.layout_values) {
          auto& layout = invocation.layouts[binding.buffer];
          auto& fields = binding.stride ? layout.stride : layout.size;
          auto value = invocation.values[binding.value];
          TORCH_CHECK_VALUE(
              !binding.stride || value >= 0,
              "Dynamic buffer stride must be nonnegative");
          fields[binding.dimension] = value;
        }
        for (const auto& binding : plan.view_values) {
          auto& view = invocation.views[binding.view];
          auto value = invocation.values[binding.value];
          if (binding.field == BoxedViewValue::Field::Offset) {
            view.offset = value;
          } else {
            auto& fields = binding.field == BoxedViewValue::Field::Stride
                ? view.stride
                : view.size;
            fields[binding.dimension] = value;
          }
        }
        for (auto buffer : plan.dynamic_buffers) {
          auto& layout = invocation.layouts[buffer];
          for (auto extent : layout.size) {
            TORCH_CHECK_VALUE(
                extent >= 0, "Dynamic buffer extent must be nonnegative");
          }
          uint64_t numel = 0;
          TORCH_CHECK_VALUE(
              !c10::safe_multiplies_u64(layout.size, &numel) &&
                  numel <= std::numeric_limits<int64_t>::max(),
              "Boxed buffer element count overflowed");
          layout.nbytes = at::detail::computeStorageNbytes(
              layout.size, layout.stride, layout.dtype.itemsize());
        }
      }
      const auto& layouts = Numeric ? invocation.layouts : plan.buffers;
      if (PyList_SetSlice(inputs.ptr(), 0, PY_SSIZE_T_MAX, nullptr) != 0) {
        throw py::error_already_set();
      }
      for (auto index : plan.copy_indices) {
        started = may_have_work_ = true;
        invocation.normalized[index] = plan.copy_function(
            PyCFunction_GET_SELF(plan.copy_builtin.get()),
            invocation.originals[index].get());
        if (!invocation.normalized[index]) {
          throw py::error_already_set();
        }
        check_boxed_stream();
      }
      for (auto index : plan.input_indices) {
        auto& input = invocation.normalized[index]
            ? invocation.normalized[index]
            : invocation.originals[index];
        const auto& tensor = THPVariable_Unpack(input.get());
        pointers_[index] = reinterpret_cast<uintptr_t>(
            plan.const_inputs[index] ? tensor.const_data_ptr()
                                     : tensor.data_ptr());
      }
      auto* allocator = at::cuda::getCUDADeviceAllocator();
      if (plan.release) {
        started = true;
        std::vector<c10::cuda::CUDACachingAllocator::PendingGraphInput>
            candidates;
        candidates.reserve(plan.release->candidates.size());
        bool supported = true;
        for (auto index : plan.release->candidates) {
          auto& original = invocation.originals[index];
          auto& normalized = invocation.normalized[index];
          auto& operand = normalized ? normalized : original;
          const auto& storage = THPVariable_Unpack(operand.get()).storage();
          candidates.push_back({&storage.data_ptr(), storage.nbytes(), 16});
          if (normalized && normalized.get() != original.get()) {
            const auto& source =
                THPVariable_Unpack(original.get()).storage().data_ptr();
            supported &= source.get_deleter() == allocator->raw_deleter() &&
                source.get_context() == source.get();
          }
        }
        if (supported) {
          invocation.pending =
              c10::cuda::CUDACachingAllocator::PendingGraphInputs::arm(
                  candidates, c10::cuda::CUDACachingAllocator::get(), stream_);
        }
        if (invocation.pending) {
          for (auto index : plan.release->candidates) {
            auto& normalized = invocation.normalized[index];
            auto& original = invocation.originals[index];
            if (normalized && normalized.get() != original.get()) {
              c10::cuda::CUDACachingAllocator::recordStream(
                  THPVariable_Unpack(original.get()).storage().data_ptr(),
                  stream_);
            }
          }
        }
      }
      auto allocate = [&](size_t index) {
        started = may_have_work_ = true;
        auto& buffer = invocation.buffers[index];
        if (invocation.pending && layouts[index].nbytes &&
            !plan.release->eligible_inputs[index].empty()) {
          auto claimed = invocation.pending->tryClaim(
              layouts[index].nbytes, 16, plan.release->eligible_inputs[index]);
          if (claimed) {
            buffer.data_ptr = std::move(*claimed);
          }
        }
        if (!buffer.data_ptr) {
          buffer.data_ptr = allocator->allocate(layouts[index].nbytes);
        }
        const auto address = reinterpret_cast<uintptr_t>(buffer.data_ptr.get());
        TORCH_CHECK_VALUE(
            (!layouts[index].nbytes || address) && address % 256 == 0,
            "Owned graph allocation requires 256-byte-aligned storage and a "
            "nonnull address for nonempty buffers");
        pointers_[plan.input_count + index] = address;
        check_boxed_stream();
      };
      if (invocation.pending) {
        for (const auto& step : plan.release->steps) {
          if (step.kind == at::cuda::detail::InputReleaseKind::Allocate) {
            allocate(step.index);
          } else {
            invocation.normalized[step.index] = nullptr;
            invocation.originals[step.index] = nullptr;
            check_boxed_stream();
          }
        }
      } else {
        for (size_t index = 0; index < plan.buffers.size(); ++index) {
          allocate(index);
        }
      }
      check_boxed_stream();
      if constexpr (Parameters) {
        auto* outputs =
            invocation.values.data() + plan.numeric->instructions.size();
        if (plan.compiled) {
          if (plan.compiled->late) {
            const auto status = plan.compiled->late(
                invocation.values.data(), pointers_.data(), outputs);
            TORCH_CHECK_VALUE(
                status == 0,
                "Compiled parameter evaluation failed with status ",
                status);
          }
        } else {
          plan.parameters->evaluate(
              invocation.values.data(),
              pointers_.data(),
              invocation.parameter_scratch.data(),
              outputs);
        }
        lease_->validate_values(invocation.values);
      }
      started = true;
      if constexpr (Numeric) {
        may_have_work_ = true;
        py::gil_scoped_release release;
        lease_->replay(pointers_, invocation.values, invocation.pending.get());
      } else if (invocation.pending) {
        may_have_work_ = true;
        py::gil_scoped_release release;
        lease_->replay(pointers_, invocation.pending.get());
      } else {
        enqueue_locked();
      }
      if (!plan.pinned_indices.empty()) {
        record_pinned_inputs(invocation, plan.pinned_indices);
      }
      for (size_t index = 0; index < plan.output_slots.size(); ++index) {
        const auto& slot = plan.output_slots[index];
        if (slot.kind == BoxedOutput::Kind::None) {
          PyTuple_SET_ITEM(invocation.result.get(), index, Py_NewRef(Py_None));
          continue;
        }
        if (slot.kind == BoxedOutput::Kind::Input) {
          auto& input = invocation.normalized[slot.value]
              ? invocation.normalized[slot.value]
              : invocation.originals[slot.value];
          PyTuple_SET_ITEM(
              invocation.result.get(), index, Py_NewRef(input.get()));
          continue;
        }
        if (slot.kind == BoxedOutput::Kind::Reference) {
          PyTuple_SET_ITEM(
              invocation.result.get(),
              index,
              Py_NewRef(PyTuple_GET_ITEM(invocation.result.get(), slot.value)));
          continue;
        }
        if (slot.kind == BoxedOutput::Kind::Value ||
            slot.kind == BoxedOutput::Kind::Literal) {
          auto value = slot.kind == BoxedOutput::Kind::Value
              ? invocation.values[slot.value]
              : slot.value;
          THPObjectPtr integer(PyLong_FromLongLong(value));
          if (!integer) {
            throw py::error_already_set();
          }
          PyTuple_SET_ITEM(invocation.result.get(), index, integer.release());
          continue;
        }
        const auto* view = slot.kind == BoxedOutput::Kind::View
            ? &invocation.views[slot.value]
            : nullptr;
        auto root = view ? view->root : plan.input_count + slot.value;
        at::TensorBase tensor;
        int64_t offset = view ? view->offset : 0;
        if (root < plan.input_count) {
          auto& input = invocation.normalized[root]
              ? invocation.normalized[root]
              : invocation.originals[root];
          const auto& source = THPVariable_Unpack(input.get());
          tensor = at::detail::make_tensor_base<c10::TensorImpl>(
              c10::Storage(source.storage()),
              source.key_set(),
              view && view->dtype ? *view->dtype : source.dtype());
          if (!(view && view->dtype)) {
            offset += source.storage_offset();
          }
        } else {
          auto buffer = root - plan.input_count;
          const auto& layout = layouts[buffer];
          auto& output = invocation.buffers[buffer];
          if (!output.storage) {
            // Direct forwarding leaves data_ptr here if StorageImpl allocation
            // fails.
            output.storage = c10::make_intrusive<c10::StorageImpl>(
                c10::StorageImpl::use_byte_size_t(),
                c10::SymInt(static_cast<int64_t>(layout.nbytes)),
                std::move(output.data_ptr),
                allocator,
                /*resizable=*/true);
          }
          tensor = at::detail::make_tensor_base<c10::TensorImpl>(
              c10::Storage(output.storage),
              c10::DispatchKeySet(c10::DispatchKey::CUDA),
              view && view->dtype ? *view->dtype : layout.dtype);
        }
        if (view) {
          tensor.unsafeGetTensorImpl()->set_storage_offset(offset);
          tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
              view->size, view->stride);
        } else {
          const auto& layout = layouts[slot.value];
          tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
              layout.size, layout.stride);
        }
        THPObjectPtr value(THPVariable_Wrap(std::move(tensor)));
        if (!value) {
          throw py::error_already_set();
        }
        PyTuple_SET_ITEM(invocation.result.get(), index, value.release());
      }
      result = invocation.result.release();
      invocation.clear();
      return py::reinterpret_steal<py::tuple>(result.release());
    } catch (...) {
      if (invocation.pending && !invocation.pending->finished() &&
          !invocation.pending->submission_started()) {
        try {
          invocation.pending->abortUnsubmitted();
        } catch (...) {
          quarantined_ = true;
        }
      }
      if (started) {
        failed_ = true;
      } else {
        invocation.clear();
      }
      throw;
    }
  }

  void wait_for_h2d() {
    std::unique_lock lock(mutex_, std::try_to_lock);
    TORCH_CHECK(lock.owns_lock(), "Native graph replay owner is busy");
    TORCH_CHECK(
        !closed_ && !failed_, "Native graph replay owner is closed or failed");
    if (pinned_event_) {
      {
        py::gil_scoped_release release;
        pinned_event_->synchronize();
      }
      retire_pinned_inputs();
    }
  }

  // The cold compiler caller proves complete pointer rebinding, unshared
  // capture/pool ownership and drained preparation work before this call.
  // Explicit output indices alone do not establish those obligations.
  void retire_capture_pool() {
    std::unique_lock lock(mutex_, std::try_to_lock);
    TORCH_CHECK(lock.owns_lock(), "Native graph replay owner is busy");
    TORCH_CHECK(
        !closed_ && !failed_, "Native graph replay owner is closed or failed");
    TORCH_CHECK(
        boxed_plan_ && boxed_plan_->has_explicit_output_indices,
        "Capture pool retirement requires explicit allocation output indices");
    TORCH_CHECK(
        !may_have_work_,
        "Capture pool retirement must precede replay, copies and allocations");
    c10::cuda::CUDAGuard guard(stream_.device_index());
    check_boxed_stream();
    TORCH_CHECK(
        current_replay_context() == context_,
        "Native graph replay context changed during pool retirement");
    bool releasing = false;
    try {
      py::gil_scoped_release release;
      if (!lease_->check_capture_pool_retirement()) {
        return;
      }
      releasing = true;
      lease_->retire_capture_pool();
    } catch (...) {
      if (releasing) {
        // Pool release can throw after decrementing a reference. Never retry
        // it.
        failed_ = true;
        quarantined_ = true;
      }
      throw;
    }
  }

  void close() {
    std::unique_lock lock(mutex_, std::try_to_lock);
    TORCH_CHECK(lock.owns_lock(), "Native graph replay owner is busy");
    if (closed_) {
      return;
    }
    TORCH_CHECK(!quarantined_, "Native graph replay owner is quarantined");
    try {
      c10::cuda::CUDAGuard guard(stream_.device_index());
      TORCH_CHECK(
          current_replay_context() == context_,
          "Native graph replay context changed during close");
      {
        py::gil_scoped_release release;
        if (may_have_work_) {
          C10_CUDA_CHECK(cudaStreamSynchronize(stream_.stream()));
        }
        if (invocation_ && invocation_->pending &&
            !invocation_->pending->finished()) {
          invocation_->pending->finishSubmitted();
        }
        lease_->close();
      }
      bound_pinned_.clear();
      pending_pinned_.clear();
      pinned_event_pool_.clear();
      pinned_event_.reset();
      invocation_.reset();
      boxed_plan_.reset();
      resources_ = nullptr;
      stream_owner_ = nullptr;
      lease_.reset();
      closed_ = true;
    } catch (...) {
      failed_ = true;
      quarantined_ = true;
      throw;
    }
  }

  bool closed() const {
    return closed_.load();
  }

  bool failed() const {
    return failed_.load();
  }

  // (memcpy nodes re-parameterized, of which the source moved) so far
  std::pair<int64_t, int64_t> memcpy_stats() const {
    return {lease_->batch().memcpy_updates(), lease_->batch().source_rebinds()};
  }
  std::pair<int64_t, int64_t> template_stats() const {
    return {
        lease_->batch().template_applies(),
        lease_->batch().template_graph_updates()};
  }

  // Copy nodes retain an allocation binding even when a later pin has the same
  // address. Hold storage because set_ can detach it from its Tensor.
  void bind_pinned_examples(PyObject* examples) {
    if (!PyTuple_CheckExact(examples)) {
      throw py::type_error("Pinned examples must be an exact tuple");
    }
    const auto count = static_cast<size_t>(PyTuple_GET_SIZE(examples));
    TORCH_CHECK_VALUE(
        count == 0 ||
            (boxed_plan_ && count == boxed_plan_->pinned_indices.size()),
        "Pinned examples must match the pinned positions");
    std::vector<c10::Storage> bound;
    bound.reserve(count);
    for (size_t index = 0; index < count; ++index) {
      auto* value = PyTuple_GET_ITEM(examples, index);
      TORCH_CHECK_VALUE(
          THPVariable_Check(value) && !THPVariable_Unpack(value).is_cuda() &&
              THPVariable_Unpack(value).is_pinned(),
          "Pinned examples must be pinned CPU Tensors");
      bound.push_back(THPVariable_Unpack(value).storage());
    }
    bound_pinned_ = std::move(bound);
  }

 private:
  friend class PythonBoxedGraphDispatch;

  using CallFunction = py::tuple (PythonGraphReplayOwner::*)(
      py::handle,
      std::vector<THPObjectPtr>*,
      const std::atomic<bool>*);
  CallFunction call_function_ = nullptr;
  CallFunction dispatch_call_function_ = nullptr;

  void check_boxed_stream() const {
    TORCH_CHECK(
        c10::cuda::current_device() == stream_.device_index() &&
            c10::cuda::getCurrentCUDAStream(stream_.device_index()) == stream_,
        "Native boxed graph replay requires its bound device and stream");
  }

  void enqueue_locked() {
    may_have_work_ = true;
    try {
      py::gil_scoped_release release;
      lease_->replay(pointers_);
    } catch (...) {
      failed_ = true;
      throw;
    }
  }

  struct PinnedInput {
    c10::Storage storage;
    THPObjectPtr owner;
  };

  struct PinnedSubmission {
    std::shared_ptr<c10::cuda::CUDAEvent> event;
    std::vector<PinnedInput> owners;
  };

  void retire_pinned_inputs() {
    while (!pending_pinned_.empty() && pending_pinned_.front().event->query()) {
      if (pending_pinned_.front().event != pinned_event_) {
        pinned_event_pool_.push_back(std::move(pending_pinned_.front().event));
      }
      pending_pinned_.pop_front();
    }
  }

  void record_pinned_inputs(
      BoxedInvocation& invocation,
      const std::vector<size_t>& indices) {
    retire_pinned_inputs();
    std::vector<PinnedInput> external;
    auto* allocator = at::getHostAllocator(at::kCUDA);
    for (size_t position = 0; position < indices.size(); ++position) {
      const auto index = indices[position];
      const auto& storage = invocation.pinned_storages[position];
      if (!allocator->record_event(
              reinterpret_cast<void*>(pointers_[index]),
              storage.data_ptr().get_context(),
              stream_.unwrap())) {
        auto* original = invocation.originals[index].get();
        external.push_back({storage, THPObjectPtr(Py_NewRef(original))});
      }
    }
    // An event carrying external owners cannot lose its submission boundary.
    if (!pinned_event_ ||
        (!pending_pinned_.empty() &&
         pending_pinned_.back().event == pinned_event_)) {
      if (pinned_event_pool_.empty()) {
        pinned_event_ = std::make_shared<c10::cuda::CUDAEvent>();
      } else {
        pinned_event_ = std::move(pinned_event_pool_.back());
        pinned_event_pool_.pop_back();
      }
    }
    if (!external.empty()) {
      pending_pinned_.push_back({pinned_event_, std::move(external)});
    }
    pinned_event_->record(stream_);
    bound_pinned_.swap(invocation.pinned_storages);
  }

  std::shared_ptr<c10::cuda::CUDAEvent> pinned_event_;
  std::deque<PinnedSubmission> pending_pinned_;
  std::vector<std::shared_ptr<c10::cuda::CUDAEvent>> pinned_event_pool_;
  std::vector<c10::Storage> bound_pinned_;

  std::unique_ptr<at::cuda::detail::GraphReplayLease> lease_;
  std::vector<uintptr_t> pointers_;
  c10::cuda::CUDAStream stream_;
  uintptr_t context_ = 0;
  THPObjectPtr stream_owner_;
  THPObjectPtr resources_;
  std::unique_ptr<BoxedReplayPlan> boxed_plan_;
  std::unique_ptr<BoxedInvocation> invocation_;
  std::mutex mutex_;
  bool may_have_work_ = false;
  std::atomic<bool> failed_{false};
  bool quarantined_ = false;
  std::atomic<bool> closed_{false};
};

void destroy_replay_owner(PythonGraphReplayOwner* owner) noexcept {
  if (!Py_IsInitialized() || !PyGILState_Check()) {
    return;
  }
  PyObject *type, *value, *traceback;
  PyErr_Fetch(&type, &value, &traceback);
  try {
    owner->close();
    delete owner;
  } catch (...) {
    // Unknown completion or teardown retains the lease, modules and buffers.
    PyErr_Clear();
  }
  PyErr_Restore(type, value, traceback);
}

class PythonBoxedGraphReplay {
 public:
  explicit PythonBoxedGraphReplay(std::shared_ptr<PythonGraphReplayOwner> owner)
      : owner_(std::move(owner)) {}

  py::tuple call(py::handle inputs) {
    return owner_->call(inputs);
  }

  void wait_for_h2d() {
    owner_->wait_for_h2d();
  }

  void retire_capture_pool() {
    owner_->retire_capture_pool();
  }

  void close() {
    owner_->close();
  }

  bool closed() const {
    return owner_->closed();
  }

  bool failed() const {
    return owner_->failed();
  }
  std::pair<int64_t, int64_t> memcpy_stats() const {
    return owner_->memcpy_stats();
  }
  std::pair<int64_t, int64_t> template_stats() const {
    return owner_->template_stats();
  }

 private:
  friend class PythonBoxedGraphDispatch;

  std::shared_ptr<PythonGraphReplayOwner> owner_;
};

class PythonBoxedGraphDispatch {
 public:
  PythonBoxedGraphDispatch(
      py::handle variants,
      py::handle on_miss,
      py::handle bound)
      : on_miss_(Py_NewRef(on_miss.ptr())) {
    if (!PyCallable_Check(on_miss.ptr())) {
      throw py::type_error(
          "Native boxed dispatch requires a callable miss handler");
    }
    if (!PyTuple_CheckExact(variants.ptr()) ||
        PyTuple_GET_SIZE(variants.ptr()) == 0) {
      throw py::type_error(
          "Native boxed dispatch requires a nonempty exact variant tuple");
    }
    for (Py_ssize_t index = 0; index < PyTuple_GET_SIZE(variants.ptr());
         ++index) {
      auto* row = PyTuple_GET_ITEM(variants.ptr(), index);
      if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 2) {
        throw py::type_error(
            "Dispatch variants must contain an entry and predicate");
      }
      append(
          py::cast<std::shared_ptr<PythonBoxedGraphReplay>>(
              py::handle(PyTuple_GET_ITEM(row, 0))),
          py::handle(PyTuple_GET_ITEM(row, 1)));
    }
    bind(bound);
  }

  // The trailing Tensor inputs the dispatcher binds itself (a frontend's own
  // resource, such as a planned arena): a caller's box may omit them, and a box
  // that carries them is taken as given. Replaceable between calls, and from
  // the cold callback (no call is in flight while it runs). Every input may be
  // bound (the caller's box is then empty), and a plan without inputs takes
  // the default empty tuple.
  void bind(py::handle bound) {
    std::unique_lock lock(mutex_, std::try_to_lock);
    TORCH_CHECK(lock.owns_lock(), "Native boxed dispatcher is busy");
    TORCH_CHECK(
        !closed_ && !failed_, "Native boxed dispatcher is closed or failed");
    if (!PyTuple_CheckExact(bound.ptr())) {
      throw py::type_error("Bound dispatch inputs must be an exact tuple");
    }
    const auto count = static_cast<size_t>(PyTuple_GET_SIZE(bound.ptr()));
    if (count > originals_.size()) {
      throw py::value_error(
          "Bound dispatch inputs exceed the prepared input count");
    }
    std::vector<THPObjectPtr> held;
    held.reserve(count);
    for (size_t index = 0; index < count; ++index) {
      auto* item = PyTuple_GET_ITEM(bound.ptr(), index);
      if (!THPVariable_CheckExact(item)) {
        throw py::type_error(
            "Bound dispatch inputs must be Tensors or Parameters");
      }
      if (integer_inputs_[originals_.size() - count + index]) {
        throw py::value_error(
            "Bound dispatch inputs must sit at trailing Tensor positions");
      }
      held.emplace_back(Py_NewRef(item));
    }
    hidden_ = std::move(held);
    for (auto& variant : variants_) {
      snapshot_bound_metadata(variant);
    }
  }

  void append(
      std::shared_ptr<PythonBoxedGraphReplay> entry,
      py::handle predicate) {
    std::unique_lock lock(mutex_, std::try_to_lock);
    TORCH_CHECK(lock.owns_lock(), "Native boxed dispatcher is busy");
    TORCH_CHECK(
        !closed_ && !failed_ && !invalidated_,
        "Native boxed dispatcher cannot accept variants");
    if (!entry) {
      throw py::type_error("Dispatch variants require a prepared boxed entry");
    }
    auto& owner = *entry->owner_;
    std::unique_lock entry_lock(owner.mutex_, std::try_to_lock);
    TORCH_CHECK(entry_lock.owns_lock(), "Native graph replay owner is busy");
    TORCH_CHECK(
        !owner.closed_ && !owner.failed_, "Dispatch entry is closed or failed");
    const auto& plan = *owner.boxed_plan_;
    std::vector<bool> integer_inputs(plan.input_count, false);
    if (plan.numeric) {
      integer_inputs = plan.numeric->integer_inputs;
    }
    std::vector<BoxedOutput::Kind> output_kinds;
    for (const auto& output : plan.output_slots) {
      auto kind = output.kind;
      if (kind == BoxedOutput::Kind::Input || kind == BoxedOutput::Kind::View ||
          kind == BoxedOutput::Kind::Reference) {
        kind = BoxedOutput::Kind::Buffer;
      } else if (kind == BoxedOutput::Kind::Literal) {
        kind = BoxedOutput::Kind::Value;
      }
      output_kinds.push_back(kind);
    }
    if (!variants_.empty() &&
        (integer_inputs != integer_inputs_ || output_kinds != output_kinds_ ||
         owner.stream_.device_index() != device_ ||
         owner.stream_.stream() != stream_ || owner.context_ != context_)) {
      throw py::value_error(
          "Dispatch entries must share boxed input kinds, output slot kinds, device, stream and context");
    }
    if (!PyTuple_CheckExact(predicate.ptr()) ||
        (PyTuple_GET_SIZE(predicate.ptr()) < 3 ||
         PyTuple_GET_SIZE(predicate.ptr()) > 6) ||
        !PyLong_CheckExact(PyTuple_GET_ITEM(predicate.ptr(), 1)) ||
        PyTuple_GET_ITEM(predicate.ptr(), 2) == Py_None) {
      throw py::type_error(
          "A predicate requires integer indices, a function address, a library owner and optional pointer, storage offset and Tensor fact sources");
    }
    auto indices = unpack_input_indices(
        PyTuple_GET_ITEM(predicate.ptr(), 0), plan.input_count);
    for (auto index : indices) {
      if (!integer_inputs[index]) {
        throw py::value_error(
            "Predicate sources must be declared boxed integer inputs");
      }
    }
    std::vector<size_t> pointer_indices;
    if (PyTuple_GET_SIZE(predicate.ptr()) >= 4) {
      pointer_indices = unpack_input_indices(
          PyTuple_GET_ITEM(predicate.ptr(), 3), plan.input_count);
      for (auto index : pointer_indices) {
        if (integer_inputs[index]) {
          throw py::value_error(
              "Predicate pointer sources must be declared boxed Tensor inputs");
        }
      }
    }
    std::vector<size_t> storage_offset_indices;
    if (PyTuple_GET_SIZE(predicate.ptr()) >= 5) {
      storage_offset_indices = unpack_input_indices(
          PyTuple_GET_ITEM(predicate.ptr(), 4), plan.input_count);
      for (auto index : storage_offset_indices) {
        if (integer_inputs[index]) {
          throw py::value_error(
              "Predicate storage offset sources must be declared boxed Tensor inputs");
        }
      }
    }
    std::vector<BoxedTensorMetadata> metadata_bindings;
    if (PyTuple_GET_SIZE(predicate.ptr()) == 6) {
      auto* bindings = PyTuple_GET_ITEM(predicate.ptr(), 5);
      if (!PyTuple_CheckExact(bindings)) {
        throw py::type_error(
            "Predicate Tensor metadata bindings must be an exact tuple");
      }
      metadata_bindings.reserve(PyTuple_GET_SIZE(bindings));
      for (Py_ssize_t index = 0; index < PyTuple_GET_SIZE(bindings); ++index) {
        auto* row = PyTuple_GET_ITEM(bindings, index);
        if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 3) {
          throw py::type_error(
              "Predicate Tensor metadata bindings require (kind, input, dimension) tuples");
        }
        auto input = static_cast<size_t>(unpack_nonnegative_integer(
            PyTuple_GET_ITEM(row, 1), "Tensor metadata input index"));
        if (input >= plan.input_count || integer_inputs[input]) {
          throw py::value_error(
              "Predicate Tensor metadata must name declared Tensor inputs");
        }
        metadata_bindings.emplace_back(
            PyTuple_GET_ITEM(row, 0), input, PyTuple_GET_ITEM(row, 2));
      }
    }
    auto address =
        PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(predicate.ptr(), 1));
    if (PyErr_Occurred()) {
      throw py::error_already_set();
    }
    if (address == 0 || address > std::numeric_limits<uintptr_t>::max()) {
      throw py::value_error("Predicate function address is out of range");
    }
    Variant variant{
        std::move(entry),
        std::move(indices),
        std::move(pointer_indices),
        std::move(storage_offset_indices),
        std::move(metadata_bindings),
        {},
        reinterpret_cast<Predicate>(static_cast<uintptr_t>(address)),
        THPObjectPtr(Py_NewRef(PyTuple_GET_ITEM(predicate.ptr(), 2)))};
    variant.values.resize(
        variant.indices.size() + variant.pointer_indices.size() +
        variant.storage_offset_indices.size() +
        variant.metadata_bindings.size());
    if (variants_.empty()) {
      originals_.resize(plan.input_count);
      integers_.resize(plan.input_count);
      integer_inputs_ = std::move(integer_inputs);
      for (size_t index = 0; index < integer_inputs_.size(); ++index) {
        if (integer_inputs_[index]) {
          integer_indices_.push_back(index);
        }
      }
      output_kinds_ = std::move(output_kinds);
      device_ = owner.stream_.device_index();
      stream_ = owner.stream_.stream();
      context_ = owner.context_;
    }
    variants_.push_back(std::move(variant));
    snapshot_bound_metadata(variants_.back());
  }

  py::object call(py::handle inputs) {
    std::unique_lock lock(mutex_, std::try_to_lock);
    TORCH_CHECK(
        lock.owns_lock() && !miss_active_, "Native boxed dispatcher is busy");
    TORCH_CHECK(
        !closed_ && !failed_, "Native boxed dispatcher is closed or failed");
    PythonGraphReplayOwner* selected = nullptr;
    try {
      if (!PyList_CheckExact(inputs.ptr())) {
        throw py::type_error("Boxed graph inputs must be an exact list");
      }
      const auto given = static_cast<size_t>(PyList_GET_SIZE(inputs.ptr()));
      const auto visible = originals_.size() - hidden_.size();
      if (given != originals_.size() && given != visible) {
        throw py::value_error(
            "Boxed graph input count differs from preparation");
      }
      // the bound inputs' metadata snapshot stands in for the trailing
      // positions only when the box leaves them to the bound objects; a full
      // box (the adapter's arena passed live, an output ring block after it)
      // reads every binding from the box, as before
      const bool from_bound = given < originals_.size();
      // Pin before any Python allocation; guards and replay use these same
      // objects. A box without the hidden inputs takes the bound ones.
      for (size_t index = 0; index < given; ++index) {
        originals_[index] = Py_NewRef(PyList_GET_ITEM(inputs.ptr(), index));
      }
      for (size_t index = given; index < originals_.size(); ++index) {
        originals_[index] = Py_NewRef(hidden_[index - visible].get());
      }
      for (auto index : integer_indices_) {
        auto* value = originals_[index].get();
        if (!PyLong_CheckExact(value)) {
          throw py::type_error(
              "Declared boxed integer inputs must be exact integers");
        }
        integers_[index] = PyLong_AsLongLong(value);
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
      }
      if (!invalidated_) {
        for (auto& variant : variants_) {
          const auto metadata_start = variant.indices.size() +
              variant.pointer_indices.size() +
              variant.storage_offset_indices.size();
          bool metadata_valid = !from_bound || variant.bound_metadata_valid;
          size_t last_input = originals_.size();
          const at::Tensor* tensor = nullptr;
          for (size_t index = 0;
               metadata_valid && index < variant.metadata_bindings.size();
               ++index) {
            const auto& binding = variant.metadata_bindings[index];
            if (from_bound && binding.input >= visible) {
              // a bound input the box left to the bound objects: its metadata
              // was read at the bind (the caller upholds the bound objects;
              // their pointers are still read per call below)
              continue;
            }
            if (binding.input != last_input) {
              auto* value = originals_[binding.input].get();
              if (!THPVariable_CheckExact(value)) {
                metadata_valid = false;
                break;
              }
              tensor = &THPVariable_Unpack(value);
              if (tensor->layout() != at::kStrided || tensor->is_nested()) {
                metadata_valid = false;
                break;
              }
              last_input = binding.input;
            }
            if (!binding.read(
                    *tensor, variant.values[metadata_start + index])) {
              metadata_valid = false;
              break;
            }
          }
          if (!metadata_valid) {
            continue;
          }
          for (size_t index = 0; index < variant.indices.size(); ++index) {
            variant.values[index] = integers_[variant.indices[index]];
          }
          for (size_t index = 0; index < variant.pointer_indices.size();
               ++index) {
            auto* value = originals_[variant.pointer_indices[index]].get();
            if (!THPVariable_CheckExact(value)) {
              throw py::type_error(
                  "Predicate pointer inputs must be Tensors or Parameters");
            }
            const auto& tensor = THPVariable_Unpack(value);
            if (!variant.metadata_bindings.empty() &&
                (tensor.layout() != at::kStrided || tensor.is_nested())) {
              metadata_valid = false;
              break;
            }
            static_assert(sizeof(uintptr_t) <= sizeof(uint64_t));
            const uint64_t pointer =
                reinterpret_cast<uintptr_t>(tensor.const_data_ptr());
            std::memcpy(
                &variant.values[variant.indices.size() + index],
                &pointer,
                sizeof(pointer));
          }
          if (!metadata_valid) {
            continue;
          }
          for (size_t index = 0; index < variant.storage_offset_indices.size();
               ++index) {
            auto* value =
                originals_[variant.storage_offset_indices[index]].get();
            if (!THPVariable_CheckExact(value)) {
              throw py::type_error(
                  "Predicate storage offset inputs must be Tensors or Parameters");
            }
            variant.values
                [variant.indices.size() + variant.pointer_indices.size() +
                 index] = THPVariable_Unpack(value).storage_offset();
          }
          auto matches = variant.predicate(variant.values.data(), nullptr);
          if (matches != 0 && matches != 1) {
            failed_ = true;
            throw py::value_error("Native predicate must return zero or one");
          }
          if (matches) {
            selected = variant.entry->owner_.get();
            std::unique_lock entry_lock(selected->mutex_, std::try_to_lock);
            TORCH_CHECK(
                entry_lock.owns_lock(), "Native graph replay owner is busy");
            TORCH_CHECK(
                !selected->closed_ && !selected->failed_,
                "Dispatch entry is closed or failed");
            return (selected->*selected->dispatch_call_function_)(
                inputs, &originals_, &invalidated_);
          }
        }
      }
      clear_snapshot();
    } catch (...) {
      clear_snapshot();
      if (selected && selected->failed_) {
        failed_ = true;
      }
      throw;
    }
    miss_active_ = true;
    lock.unlock();
    // The cold callback may append, but cannot recursively invoke or close us.
    THPObjectPtr result(PyObject_CallOneArg(on_miss_.get(), inputs.ptr()));
    miss_active_ = false;
    if (!result) {
      throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(result.release());
  }

  void wait_for_h2d() {
    std::unique_lock lock(mutex_, std::try_to_lock);
    TORCH_CHECK(
        lock.owns_lock() && !miss_active_, "Native boxed dispatcher is busy");
    TORCH_CHECK(
        !closed_ && !failed_, "Native boxed dispatcher is closed or failed");
    for (const auto& variant : variants_) {
      variant.entry->wait_for_h2d();
    }
  }

  void invalidate() {
    TORCH_CHECK(
        !closed_ && !failed_, "Native boxed dispatcher is closed or failed");
    invalidated_ = true;
  }

  void close() {
    std::unique_lock lock(mutex_, std::try_to_lock);
    TORCH_CHECK(
        lock.owns_lock() && !miss_active_, "Native boxed dispatcher is busy");
    if (closed_) {
      return;
    }
    invalidated_ = true;
    try {
      for (auto& variant : variants_) {
        variant.entry->close();
      }
      variants_.clear();
      hidden_.clear();
      on_miss_ = nullptr;
      closed_ = true;
    } catch (...) {
      // Keep every remaining entry and predicate library if draining fails.
      failed_ = true;
      throw;
    }
  }

  bool closed() const {
    return closed_.load();
  }

  bool failed() const {
    return failed_.load();
  }

  bool invalidated() const {
    return invalidated_.load();
  }

 private:
  using Predicate = int8_t (*)(int64_t*, double*);
  struct Variant {
    std::shared_ptr<PythonBoxedGraphReplay> entry;
    std::vector<size_t> indices;
    std::vector<size_t> pointer_indices;
    std::vector<size_t> storage_offset_indices;
    std::vector<BoxedTensorMetadata> metadata_bindings;
    std::vector<int64_t> values;
    Predicate predicate;
    THPObjectPtr library;
    // the bound inputs' metadata values hold in `values` from the bind on;
    // false when a bound input is not a strided Tensor the bindings can read
    bool bound_metadata_valid = true;
  };

  // The metadata facts of the bound (hidden) inputs, read once per bind into
  // the variant's value slots: a call reads the visible inputs' metadata only.
  void snapshot_bound_metadata(Variant& variant) {
    const auto visible = originals_.size() - hidden_.size();
    const auto metadata_start = variant.indices.size() +
        variant.pointer_indices.size() + variant.storage_offset_indices.size();
    variant.bound_metadata_valid = true;
    for (size_t index = 0; index < variant.metadata_bindings.size(); ++index) {
      const auto& binding = variant.metadata_bindings[index];
      if (binding.input < visible) {
        continue;
      }
      auto* value = hidden_[binding.input - visible].get();
      if (!THPVariable_CheckExact(value)) {
        variant.bound_metadata_valid = false;
        return;
      }
      const auto& tensor = THPVariable_Unpack(value);
      if (tensor.layout() != at::kStrided || tensor.is_nested() ||
          !binding.read(tensor, variant.values[metadata_start + index])) {
        variant.bound_metadata_valid = false;
        return;
      }
    }
  }

  void clear_snapshot() {
    for (auto& value : originals_) {
      value = nullptr;
    }
  }

  std::vector<Variant> variants_;
  std::vector<THPObjectPtr> hidden_;
  std::vector<THPObjectPtr> originals_;
  std::vector<int64_t> integers_;
  std::vector<bool> integer_inputs_;
  std::vector<size_t> integer_indices_;
  std::vector<BoxedOutput::Kind> output_kinds_;
  c10::DeviceIndex device_ = -1;
  cudaStream_t stream_ = nullptr;
  uintptr_t context_ = 0;
  THPObjectPtr on_miss_;
  std::mutex mutex_;
  std::atomic<bool> miss_active_{false};
  std::atomic<bool> invalidated_{false};
  std::atomic<bool> failed_{false};
  std::atomic<bool> closed_{false};
};

void destroy_boxed_dispatcher(PythonBoxedGraphDispatch* dispatcher) noexcept {
  if (!Py_IsInitialized() || !PyGILState_Check()) {
    return;
  }
  PyObject *type, *value, *traceback;
  PyErr_Fetch(&type, &value, &traceback);
  try {
    dispatcher->close();
    delete dispatcher;
  } catch (...) {
    PyErr_Clear();
  }
  PyErr_Restore(type, value, traceback);
}

} // namespace

void THCPGraph_init(PyObject* module) {
  // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
  // but CI linter and some builds prefer "module".
  auto torch_C_m = py::handle(module).cast<py::module>();

  using TensorMapBinding = at::cuda::detail::KernelTensorMapBinding;
  py::class_<TensorMapBinding>(torch_C_m, "_CUDAGraphTensorMapBinding")
      .def(
          py::init([](uintptr_t node,
                      int64_t argument,
                      size_t pointer_index,
                      size_t address_offset_value_index,
                      std::vector<size_t> dimensions,
                      std::vector<size_t> strides,
                      std::vector<uint32_t> box_dimensions,
                      uint32_t data_type,
                      uint32_t swizzle,
                      bool nan_fill) {
            return TensorMapBinding{
                node,
                argument,
                pointer_index,
                address_offset_value_index,
                std::move(dimensions),
                std::move(strides),
                std::move(box_dimensions),
                data_type,
                swizzle,
                nan_fill};
          }),
          py::kw_only(),
          py::arg("node"),
          py::arg("argument"),
          py::arg("pointer_index"),
          py::arg("address_offset_value_index"),
          py::arg("dimensions"),
          py::arg("strides"),
          py::arg("box_dimensions"),
          py::arg("data_type"),
          py::arg("swizzle"),
          py::arg("nan_fill"))
      .def_readonly("node", &TensorMapBinding::node)
      .def_readonly("argument", &TensorMapBinding::argument)
      .def_readonly("pointer_index", &TensorMapBinding::pointer_index)
      .def_readonly(
          "address_offset_value_index",
          &TensorMapBinding::address_offset_value_index)
      .def_readonly("dimensions", &TensorMapBinding::dimensions)
      .def_readonly("strides", &TensorMapBinding::strides)
      .def_readonly("box_dimensions", &TensorMapBinding::box_dimensions)
      .def_readonly("data_type", &TensorMapBinding::data_type)
      .def_readonly("swizzle", &TensorMapBinding::swizzle)
      .def_readonly("nan_fill", &TensorMapBinding::nan_fill);

  // The ScalarType code a predicate's "dtype" Tensor fact source compares
  // against.
  torch_C_m.def("_cuda_scalar_type_code", [](py::handle dtype) {
    if (!THPDtype_Check(dtype.ptr())) {
      throw py::type_error("_cuda_scalar_type_code requires a torch.dtype");
    }
    return static_cast<int64_t>(
        reinterpret_cast<THPDtype*>(dtype.ptr())->scalar_type);
  });

  torch_C_m.def(
      "_cuda_encode_tensor_map",
      torch::wrap_pybind_function([](const TensorMapBinding& binding,
                                     py::tuple values,
                                     py::tuple pointers) {
        std::vector<int64_t> numeric;
        numeric.reserve(values.size());
        for (auto item : values) {
          if (!PyLong_CheckExact(item.ptr())) {
            throw py::type_error("Tensor-map values must be exact integers");
          }
          auto value = PyLong_AsLongLong(item.ptr());
          if (PyErr_Occurred()) {
            throw py::error_already_set();
          }
          numeric.push_back(value);
        }
        std::vector<uintptr_t> roots(pointers.size());
        unpack_pointer_values(pointers, roots);
        std::array<uint8_t, 128> encoded;
        {
          py::gil_scoped_release release;
          encoded =
              at::cuda::detail::encode_tensor_map(binding, numeric, roots);
        }
        return py::bytes(
            reinterpret_cast<const char*>(encoded.data()), encoded.size());
      }),
      py::arg("binding"),
      py::arg("values"),
      py::arg("pointers"));

  torch_C_m.def("_graph_pool_handle", &::at::cuda::graph_pool_handle);

  shared_ptr_class_<CompiledBoxedEvaluation>(
      torch_C_m, "_CUDAGraphCompiledEvaluation")
      .def(
          py::init<
              uintptr_t,
              uintptr_t,
              size_t,
              size_t,
              size_t,
              size_t,
              py::handle>(),
          py::kw_only(),
          py::arg("early_address"),
          py::arg("late_address"),
          py::arg("leaf_count"),
          py::arg("early_count"),
          py::arg("late_count"),
          py::arg("pointer_count"),
          py::arg("library_owner"))
      .def_readonly("leaf_count", &CompiledBoxedEvaluation::leaf_count)
      .def_readonly("early_count", &CompiledBoxedEvaluation::early_count)
      .def_readonly("late_count", &CompiledBoxedEvaluation::late_count)
      .def_readonly("pointer_count", &CompiledBoxedEvaluation::pointer_count);

  shared_ptr_class_<PythonBoxedNumeric>(torch_C_m, "_CUDAGraphBoxedNumeric")
      .def(
          py::init<
              size_t,
              py::handle,
              std::shared_ptr<CompiledBoxedEvaluation>,
              std::vector<size_t>>(),
          py::kw_only(),
          py::arg("input_count"),
          py::arg("numeric_plan"),
          py::arg("compiled_evaluation"),
          py::arg("output_indices"))
      .def(
          "__call__",
          torch::wrap_pybind_function(&PythonBoxedNumeric::call),
          py::arg("inputs"));

  py::class_<CutValue>(torch_C_m, "_CUDAGraphCutValue")
      .def(py::init<int64_t, bool>(), py::arg("value"), py::arg("is_index"));
  py::class_<CutTensor>(torch_C_m, "_CUDAGraphCutTensor")
      .def(
          py::init<
              std::string,
              int64_t,
              std::vector<CutValue>,
              std::vector<CutValue>,
              CutValue>(),
          py::arg("root"),
          py::arg("input_index"),
          py::arg("sizes"),
          py::arg("strides"),
          py::arg("offset"));
  shared_ptr_class_<PythonCut>(torch_C_m, "_CUDAGraphCut")
      .def(
          py::init<c10::OperatorHandle, py::tuple, py::dict>(),
          py::arg("operator"),
          py::arg("args"),
          py::arg("kwargs"))
      .def(
          "__call__",
          torch::wrap_pybind_function(&PythonCut::call),
          py::arg("inputs"),
          py::arg("boundary"),
          py::arg("values"),
          py::arg("known"));

  torch_C_m.def(
      "_cuda_evaluate_parameter_program",
      torch::wrap_pybind_function([](py::handle plan,
                                     py::tuple early_values,
                                     py::tuple pointers) {
        std::vector<int64_t> early;
        early.reserve(early_values.size());
        for (auto item : early_values) {
          if (!PyLong_CheckExact(item.ptr())) {
            throw py::type_error(
                "Parameter evaluation requires exact early integer values");
          }
          auto value = PyLong_AsLongLong(item.ptr());
          if (PyErr_Occurred()) {
            throw py::error_already_set();
          }
          early.push_back(value);
        }
        std::vector<uintptr_t> roots(pointers.size());
        unpack_pointer_values(pointers, roots);
        auto program =
            unpack_parameter_program(plan.ptr(), early.size(), roots.size());
        std::vector<parameter_program::Value> scratch(program->scratch_size());
        std::vector<int64_t> outputs(program->output_count());
        program->evaluate(
            early.data(), roots.data(), scratch.data(), outputs.data());
        py::tuple result(outputs.size());
        for (size_t index = 0; index < outputs.size(); ++index) {
          result[index] = py::int_(outputs[index]);
        }
        return result;
      }),
      py::arg("plan"),
      py::arg("early_values"),
      py::arg("pointers"));

  torch_C_m.def(
      "_cuda_get_capture_frontier",
      torch::wrap_pybind_function([](uintptr_t stream) {
        auto snapshot = [&] {
          py::gil_scoped_release release;
          return at::cuda::detail::get_capture_frontier(stream);
        }();
        py::tuple dependencies(snapshot.dependencies.size());
        for (size_t index = 0; index < snapshot.dependencies.size(); ++index) {
          const auto& [node, edge] = snapshot.dependencies[index];
          dependencies[index] = py::make_tuple(
              node,
              py::bytes(
                  reinterpret_cast<const char*>(edge.data()), edge.size()));
        }
        return py::make_tuple(
            snapshot.status, snapshot.capture_id, snapshot.graph, dependencies);
      }),
      py::arg("stream"));

  // the kernel template registry (CUDAGraphParams.h): sites, variants keyed
  // by integers, and the two call-signature functions a predicate and a
  // numeric plan use to select a variant and read the selection back
  torch_C_m.def("_cuda_kernel_template_new_site", []() {
    return at::cuda::detail::new_kernel_template_site();
  });
  torch_C_m.def(
      "_cuda_kernel_template_register",
      torch::wrap_pybind_function([](int64_t site,
                                     std::vector<int64_t> key,
                                     py::tuple nodes) {
        // a kernel row: (function, grid, block, shared, image, slots,
        // workspace slots, attributes); a memset row: ("memset", operand or
        // None, delta or address, element size, width, value). Row i of a
        // variant is node i of its site's chain.
        at::cuda::detail::KernelTemplateVariant variant;
        for (auto item : nodes) {
          auto row = item.cast<py::tuple>();
          at::cuda::detail::KernelTemplateNode node;
          if (row.size() == 6 && py::isinstance<py::str>(row[0])) {
            if (row[0].cast<std::string>() != "memset") {
              throw py::type_error(
                  "Template memset rows are (\"memset\", operand, delta, element_size, width, value) tuples");
            }
            node.kind = at::cuda::detail::KernelTemplateNode::Kind::memset;
            if (!row[1].is_none()) {
              node.memset_operand = row[1].cast<size_t>();
            }
            node.memset_delta = row[2].cast<int64_t>();
            node.memset_element_size = row[3].cast<unsigned int>();
            node.memset_width = row[4].cast<size_t>();
            node.memset_value = row[5].cast<unsigned int>();
            variant.nodes.push_back(std::move(node));
            continue;
          }
          if (row.size() != 8) {
            throw py::type_error(
                "Template nodes are (function, grid, block, shared, image, slots, workspace_slots, attributes) tuples");
          }
          node.function = row[0].cast<uintptr_t>();
          node.grid = row[1].cast<std::array<unsigned int, 3>>();
          node.block = row[2].cast<std::array<unsigned int, 3>>();
          node.shared_memory = row[3].cast<unsigned int>();
          auto image = row[4].cast<py::bytes>();
          std::string_view bytes = image;
          node.image.assign(bytes.begin(), bytes.end());
          for (auto slot : row[5].cast<py::tuple>()) {
            auto fields = slot.cast<std::tuple<size_t, size_t, int64_t>>();
            node.slots.push_back(
                {std::get<0>(fields),
                 std::get<1>(fields),
                 std::get<2>(fields)});
          }
          node.workspace_slots = row[6].cast<std::vector<size_t>>();
          node.attributes =
              row[7].cast<at::cuda::detail::KernelNodeAttributes>();
          variant.nodes.push_back(std::move(node));
        }
        return at::cuda::detail::register_kernel_template(
            site, std::move(key), std::move(variant));
      }),
      py::arg("site"),
      py::arg("key"),
      py::arg("nodes"));
  torch_C_m.def(
      "_cuda_kernel_template_select",
      [](int64_t site, std::vector<int64_t> key) {
        key.insert(key.begin(), site);
        return at::cuda::detail::select_kernel_template(key);
      },
      py::arg("site"),
      py::arg("key"));
  torch_C_m.def("_cuda_kernel_template_selected", [](int64_t site) {
    return at::cuda::detail::selected_kernel_template({site});
  });
  torch_C_m.def("_cuda_kernel_template_take_miss", [](int64_t site) {
    return at::cuda::detail::take_kernel_template_miss(site);
  });
  torch_C_m.def("_cuda_kernel_template_select_address", []() {
    return reinterpret_cast<uintptr_t>(
        &at::cuda::detail::select_kernel_template);
  });
  torch_C_m.def("_cuda_kernel_template_select_key_address", []() {
    return reinterpret_cast<uintptr_t>(
        &at::cuda::detail::select_kernel_template_key);
  });
  torch_C_m.def("_cuda_kernel_template_settings_epoch", []() {
    return at::blasSettingsEpoch();
  });
  torch_C_m.def("_cuda_kernel_template_selected_address", []() {
    return reinterpret_cast<uintptr_t>(
        &at::cuda::detail::selected_kernel_template);
  });
  torch_C_m.def("_cuda_graph_hotpath_timers_take", []() {
    return at::cuda::detail::hotpath_timers_take();
  });
  torch_C_m.def("_cuda_kernel_template_selected_at_address", []() {
    return reinterpret_cast<uintptr_t>(
        &at::cuda::detail::selected_kernel_template_at);
  });
  torch_C_m.def("_cuda_kernel_template_library_settings", []() {
    return at::cuda::detail::kernel_template_library_settings();
  });
  torch_C_m.def("_cuda_kernel_node_attributes", [](uintptr_t node) {
    return at::cuda::detail::read_kernel_node_attributes(node);
  });
  torch_C_m.def(
      "_cuda_launch_kernel_image",
      torch::wrap_pybind_function(
          [](uintptr_t function,
             std::array<unsigned int, 3> grid,
             std::array<unsigned int, 3> block,
             unsigned int shared,
             uintptr_t stream,
             py::bytes image,
             at::cuda::detail::KernelNodeAttributes attributes,
             bool programmatic) {
            std::string_view bytes = image;
            std::vector<uint8_t> buffer(bytes.begin(), bytes.end());
            py::gil_scoped_release release;
            at::cuda::detail::launch_kernel_image(
                function,
                grid,
                block,
                shared,
                stream,
                buffer,
                attributes,
                programmatic);
          }),
      py::arg("function"),
      py::arg("grid"),
      py::arg("block"),
      py::arg("shared"),
      py::arg("stream"),
      py::arg("image"),
      py::arg("attributes"),
      py::arg("programmatic") = false);

  shared_ptr_class_<at::cuda::detail::KernelParamUpdateBatch>(
      torch_C_m, "_CUDAGraphKernelParamUpdates");
  shared_ptr_class_<PythonKernelPointerUpdates>(
      torch_C_m, "_CUDAGraphKernelPointerUpdates");
  shared_ptr_class_<PythonGraphReplayOwner>(torch_C_m, "_CUDAGraphReplayOwner")
      .def(
          "replay",
          torch::wrap_pybind_function(&PythonGraphReplayOwner::replay),
          py::arg("pointers"))
      .def("close", torch::wrap_pybind_function(&PythonGraphReplayOwner::close))
      .def_property_readonly("closed", &PythonGraphReplayOwner::closed)
      .def_property_readonly("failed", &PythonGraphReplayOwner::failed);
  shared_ptr_class_<PythonBoxedGraphReplay>(torch_C_m, "_CUDAGraphBoxedReplay")
      .def(
          "__call__",
          torch::wrap_pybind_function(&PythonBoxedGraphReplay::call),
          py::arg("inputs"))
      .def(
          "wait_for_h2d",
          torch::wrap_pybind_function(&PythonBoxedGraphReplay::wait_for_h2d))
      .def(
          "_retire_capture_pool",
          torch::wrap_pybind_function(
              &PythonBoxedGraphReplay::retire_capture_pool))
      .def("close", torch::wrap_pybind_function(&PythonBoxedGraphReplay::close))
      .def_property_readonly(
          "_boxed_call", [](const PythonBoxedGraphReplay&) { return true; })
      .def_property_readonly("closed", &PythonBoxedGraphReplay::closed)
      .def_property_readonly("failed", &PythonBoxedGraphReplay::failed)
      .def(
          "_memcpy_stats",
          torch::wrap_pybind_function(&PythonBoxedGraphReplay::memcpy_stats))
      .def(
          "_template_stats",
          torch::wrap_pybind_function(&PythonBoxedGraphReplay::template_stats));
  shared_ptr_class_<PythonBoxedGraphDispatch>(
      torch_C_m, "_CUDAGraphBoxedDispatch")
      .def(
          "__call__",
          torch::wrap_pybind_function(&PythonBoxedGraphDispatch::call),
          py::arg("inputs"))
      .def(
          "append",
          torch::wrap_pybind_function(&PythonBoxedGraphDispatch::append),
          py::arg("entry"),
          py::arg("predicate"))
      .def(
          "bind",
          torch::wrap_pybind_function(&PythonBoxedGraphDispatch::bind),
          py::arg("bound"))
      .def(
          "wait_for_h2d",
          torch::wrap_pybind_function(&PythonBoxedGraphDispatch::wait_for_h2d))
      .def(
          "invalidate",
          torch::wrap_pybind_function(&PythonBoxedGraphDispatch::invalidate))
      .def(
          "close",
          torch::wrap_pybind_function(&PythonBoxedGraphDispatch::close))
      .def_property_readonly(
          "_boxed_call", [](const PythonBoxedGraphDispatch&) { return true; })
      .def_property_readonly("closed", &PythonBoxedGraphDispatch::closed)
      .def_property_readonly("failed", &PythonBoxedGraphDispatch::failed)
      .def_property_readonly(
          "invalidated", &PythonBoxedGraphDispatch::invalidated);
  torch_C_m.def(
      "_cuda_boxed_tensor_metadata",
      torch::wrap_pybind_function([](py::handle tensor,
                                     py::handle kind,
                                     py::handle dimension) {
        if (!THPVariable_CheckExact(tensor.ptr())) {
          throw py::type_error(
              "Tensor metadata requires a Tensor or Parameter");
        }
        BoxedTensorMetadata binding(kind.ptr(), 0, dimension.ptr());
        int64_t value = 0;
        TORCH_CHECK_VALUE(
            binding.read(THPVariable_Unpack(tensor.ptr()), value),
            "Tensor metadata dimension requires an in-range strided Tensor input");
        return value;
      }),
      py::arg("tensor"),
      py::arg("kind"),
      py::arg("dimension") = 0);
  torch_C_m.def(
      "_cuda_make_boxed_dispatch",
      torch::wrap_pybind_function(
          [](py::handle variants, py::handle on_miss, py::handle bound) {
            return std::shared_ptr<PythonBoxedGraphDispatch>(
                new PythonBoxedGraphDispatch(variants, on_miss, bound),
                destroy_boxed_dispatcher);
          }),
      py::arg("variants"),
      py::arg("on_miss"),
      py::arg("bound") = py::tuple());

  shared_ptr_class_<::at::cuda::CUDAGraph>(torch_C_m, "_CUDAGraph")
      .def(py::init<bool>(), py::arg("keep_graph") = false)
      .def(
          "capture_begin",
          [](::at::cuda::CUDAGraph& self,
             std::optional<c10::cuda::MempoolId_t> pool_opt,
             const std::string& capture_error_mode) {
            cudaStreamCaptureMode capture_mode{};
            c10::cuda::MempoolId_t pool = pool_opt.has_value()
                ? pool_opt.value()
                : c10::cuda::MempoolId_t{0, 0};
            if (capture_error_mode == "global") {
              capture_mode = cudaStreamCaptureModeGlobal;
            } else if (capture_error_mode == "thread_local") {
              capture_mode = cudaStreamCaptureModeThreadLocal;
            } else if (capture_error_mode == "relaxed") {
              capture_mode = cudaStreamCaptureModeRelaxed;
            } else {
              TORCH_CHECK(
                  false,
                  "Unknown capture error mode. Expected `global`, `thread_local`, or `relaxed`, got ",
                  capture_error_mode);
            }
            return self.capture_begin(pool, capture_mode);
          },
          py::arg("pool"),
          py::arg("capture_error_mode"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "capture_end",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::capture_end))
      .def(
          "capture_end_pre",
          torch::wrap_pybind_function_no_gil(
              &at::cuda::CUDAGraph::capture_end_pre))
      .def(
          "capture_end_post",
          torch::wrap_pybind_function_no_gil(
              &at::cuda::CUDAGraph::capture_end_post))
      .def(
          "instantiate",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::instantiate))
      .def_property_readonly(
          "_has_graph_exec", &at::cuda::CUDAGraph::has_graph_exec)
      .def(
          "_check_not_owned",
          torch::wrap_pybind_function(&at::cuda::CUDAGraph::check_not_owned))
      .def(
          "_make_replay_owner",
          // Internal cold caller: an unshared private capture, no published
          // pool/raw handles, and preparation work already drained. The stream
          // is ATen-managed in the capture context; resources own modules/data.
          torch::wrap_pybind_function(
              [](std::shared_ptr<at::cuda::CUDAGraph> graph,
                 const PythonKernelPointerUpdates& updates,
                 py::handle stream,
                 py::handle resources) {
                if (updates.batch->has_numeric_updates()) {
                  throw py::value_error(
                      "Numeric updates require a boxed numeric plan");
                }
                if (!THCPStreamClass ||
                    !PyObject_TypeCheck(
                        stream.ptr(),
                        reinterpret_cast<PyTypeObject*>(THCPStreamClass))) {
                  throw py::type_error(
                      "Native graph replay requires a CUDA stream");
                }
                if (!PyTuple_CheckExact(resources.ptr())) {
                  throw py::type_error(
                      "Native graph replay resources must be an exact tuple");
                }
                auto bound_stream =
                    reinterpret_cast<THCPStream*>(stream.ptr())->cuda_stream;
                return std::shared_ptr<PythonGraphReplayOwner>(
                    new PythonGraphReplayOwner(
                        std::move(graph),
                        updates,
                        stream.ptr(),
                        resources.ptr(),
                        bound_stream),
                    destroy_replay_owner);
              }),
          py::arg("updates"),
          py::arg("stream"),
          py::arg("resources"))
      .def(
          "_make_boxed_replay",
          // Same private capture contract as _make_replay_owner. The compiler
          // binds input i and allocation r to pointer indices i and input_count
          // + r.
          torch::wrap_pybind_function(
              [](std::shared_ptr<at::cuda::CUDAGraph> graph,
                 const PythonKernelPointerUpdates& updates,
                 py::handle stream,
                 py::handle resources,
                 py::handle input_count,
                 py::handle used_input_indices,
                 py::handle output_layouts,
                 py::handle prologue,
                 py::handle output_indices,
                 py::handle numeric_plan,
                 py::handle release_plan,
                 std::shared_ptr<CompiledBoxedEvaluation> compiled_evaluation,
                 py::handle pinned_positions,
                 py::handle pinned_examples,
                 py::handle const_positions) {
                if (!THCPStreamClass ||
                    !PyObject_TypeCheck(
                        stream.ptr(),
                        reinterpret_cast<PyTypeObject*>(THCPStreamClass))) {
                  throw py::type_error(
                      "Native graph replay requires a CUDA stream");
                }
                if (!PyTuple_CheckExact(resources.ptr())) {
                  throw py::type_error(
                      "Native graph replay resources must be an exact tuple");
                }
                auto plan = std::make_unique<BoxedReplayPlan>(
                    input_count.ptr(),
                    used_input_indices.ptr(),
                    output_layouts.ptr(),
                    prologue.ptr(),
                    output_indices.ptr(),
                    numeric_plan.ptr(),
                    updates,
                    std::move(compiled_evaluation),
                    pinned_positions.ptr(),
                    const_positions.ptr());
                auto invocation = std::make_unique<BoxedInvocation>(*plan);
                auto bound_stream =
                    reinterpret_cast<THCPStream*>(stream.ptr())->cuda_stream;
                auto owner = std::shared_ptr<PythonGraphReplayOwner>(
                    new PythonGraphReplayOwner(
                        std::move(graph),
                        updates,
                        stream.ptr(),
                        resources.ptr(),
                        bound_stream,
                        std::move(plan),
                        std::move(invocation),
                        release_plan.ptr()),
                    destroy_replay_owner);
                owner->bind_pinned_examples(pinned_examples.ptr());
                return std::make_shared<PythonBoxedGraphReplay>(
                    std::move(owner));
              }),
          py::arg("updates"),
          py::arg("stream"),
          py::arg("resources"),
          py::arg("input_count"),
          py::arg("used_input_indices"),
          py::arg("output_layouts"),
          py::arg("prologue") = py::none(),
          py::arg("output_indices") = py::none(),
          py::arg("numeric_plan") = py::none(),
          py::arg("release_plan") = py::none(),
          py::kw_only(),
          py::arg("compiled_evaluation") = nullptr,
          py::arg("pinned_positions") = py::tuple(),
          py::arg("pinned_examples") = py::tuple(),
          py::arg("const_positions") = py::tuple())
      .def(
          "_prepare_kernel_params",
          torch::wrap_pybind_function([](at::cuda::CUDAGraph& self,
                                         py::handle updates) {
            if (!PyDict_Check(updates.ptr())) {
              throw py::type_error(
                  "kernel parameter updates must be dictionaries");
            }
            std::vector<at::cuda::detail::KernelNodeUpdate> values;
            values.reserve(PyDict_Size(updates.ptr()));
            Py_ssize_t node_position = 0;
            PyObject* node;
            PyObject* arguments;
            while (
                PyDict_Next(updates.ptr(), &node_position, &node, &arguments)) {
              if (!PyLong_Check(node)) {
                throw py::type_error(
                    "kernel node handles must be Python integers");
              }
              if (!PyDict_Check(arguments)) {
                throw py::type_error(
                    "kernel parameter updates must be dictionaries");
              }
              auto handle = PyLong_AsVoidPtr(node);
              if (PyErr_Occurred()) {
                throw py::error_already_set();
              }
              at::cuda::detail::KernelNodeUpdate update{
                  reinterpret_cast<uintptr_t>(handle), {}};
              update.arguments.reserve(PyDict_Size(arguments));
              Py_ssize_t argument_position = 0;
              PyObject* index;
              PyObject* value;
              while (
                  PyDict_Next(arguments, &argument_position, &index, &value)) {
                if (!PyLong_CheckExact(index)) {
                  throw py::type_error("Kernel argument index must be an int");
                }
                auto argument_index = PyLong_AsLongLong(index);
                if (PyErr_Occurred()) {
                  PyErr_Clear();
                  throw py::index_error(
                      "Kernel argument index is out of range");
                }
                if (!PyBytes_Check(value)) {
                  throw py::type_error("Kernel argument value must be bytes");
                }
                const auto* bytes =
                    reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(value));
                update.arguments.push_back(
                    {argument_index,
                     std::vector<uint8_t>(
                         bytes, bytes + PyBytes_GET_SIZE(value))});
              }
              values.push_back(std::move(update));
            }
            py::gil_scoped_release release;
            return self.prepare_kernel_params(std::move(values));
          }),
          py::arg("updates"))
      .def(
          "_apply_kernel_params",
          torch::wrap_pybind_function_no_gil(
              &at::cuda::CUDAGraph::update_kernel_params),
          py::arg("updates"))
      .def(
          "_prepare_kernel_pointer_updates",
          torch::wrap_pybind_function([](at::cuda::CUDAGraph& self,
                                         py::handle bindings,
                                         py::handle pointer_count) {
            if (!PyTuple_CheckExact(bindings.ptr()) ||
                !PyLong_CheckExact(pointer_count.ptr())) {
              throw py::type_error(
                  "Pointer bindings must be a tuple and pointer_count an int");
            }
            size_t count = PyLong_AsSize_t(pointer_count.ptr());
            if (PyErr_Occurred()) {
              throw py::error_already_set();
            }
            auto values = unpack_pointer_bindings(bindings.ptr(), false);
            py::gil_scoped_release release;
            auto batch = self.prepare_kernel_pointer_updates(values, count);
            auto result = std::make_shared<PythonKernelPointerUpdates>(
                std::move(batch), count);
            for (const auto& binding : values) {
              result->referenced[binding.pointer_index] = true;
            }
            return result;
          }),
          py::arg("bindings"),
          py::arg("pointer_count"))
      .def(
          "_prepare_kernel_replay_updates",
          torch::wrap_pybind_function([](at::cuda::CUDAGraph& self,
                                         py::handle pointer_bindings,
                                         py::handle pointer_count,
                                         py::handle scalar_bindings,
                                         py::handle grid_bindings,
                                         py::handle value_count,
                                         py::tuple tensor_map_bindings,
                                         py::handle memset_bindings,
                                         py::tuple host_table_bindings,
                                         py::tuple memcpy_bindings,
                                         py::tuple rng_bindings,
                                         py::tuple template_bindings) {
            auto pointers =
                unpack_pointer_bindings(pointer_bindings.ptr(), true, true);
            auto count = static_cast<size_t>(unpack_nonnegative_integer(
                pointer_count.ptr(), "Pointer count"));
            auto results = static_cast<size_t>(unpack_nonnegative_integer(
                value_count.ptr(), "Numeric value count"));
            if (!PyTuple_CheckExact(scalar_bindings.ptr()) ||
                !PyTuple_CheckExact(grid_bindings.ptr())) {
              throw py::type_error(
                  "Scalar and grid bindings must be exact tuples");
            }
            std::vector<at::cuda::detail::KernelScalarBinding> scalars;
            scalars.reserve(PyTuple_GET_SIZE(scalar_bindings.ptr()));
            for (Py_ssize_t index = 0;
                 index < PyTuple_GET_SIZE(scalar_bindings.ptr());
                 ++index) {
              auto* row = PyTuple_GET_ITEM(scalar_bindings.ptr(), index);
              const bool field =
                  PyTuple_CheckExact(row) && PyTuple_GET_SIZE(row) == 5;
              if (!PyTuple_CheckExact(row) ||
                  PyTuple_GET_SIZE(row) != (field ? 5 : 4) ||
                  !PyLong_CheckExact(PyTuple_GET_ITEM(row, 0))) {
                throw py::type_error(
                    "Scalar bindings must contain (node, argument, width, value_index) or "
                    "(node, argument, byte_offset, width, value_index) tuples");
              }
              auto node = PyLong_AsVoidPtr(PyTuple_GET_ITEM(row, 0));
              if (PyErr_Occurred()) {
                throw py::error_already_set();
              }
              auto argument = unpack_nonnegative_integer(
                  PyTuple_GET_ITEM(row, 1), "Scalar argument index");
              std::optional<size_t> byte_offset;
              if (field) {
                byte_offset = static_cast<size_t>(unpack_nonnegative_integer(
                    PyTuple_GET_ITEM(row, 2), "Scalar byte offset"));
              }
              const auto tail = field ? 3 : 2;
              auto width = static_cast<size_t>(unpack_nonnegative_integer(
                  PyTuple_GET_ITEM(row, tail), "Scalar width"));
              auto value = static_cast<size_t>(unpack_nonnegative_integer(
                  PyTuple_GET_ITEM(row, tail + 1), "Scalar value index"));
              scalars.push_back(
                  {reinterpret_cast<uintptr_t>(node),
                   argument,
                   width,
                   value,
                   byte_offset});
            }
            std::vector<at::cuda::detail::KernelGridBinding> grids;
            grids.reserve(PyTuple_GET_SIZE(grid_bindings.ptr()));
            for (Py_ssize_t index = 0;
                 index < PyTuple_GET_SIZE(grid_bindings.ptr());
                 ++index) {
              auto* row = PyTuple_GET_ITEM(grid_bindings.ptr(), index);
              // (node, gx, gy, gz[, shared][, bx, by, bz]): 4, 5, 7 or 8
              // entries
              const auto width =
                  PyTuple_CheckExact(row) ? PyTuple_GET_SIZE(row) : 0;
              const bool shared = width == 5 || width == 8;
              const bool block = width == 7 || width == 8;
              if (!(width == 4 || width == 5 || width == 7 || width == 8) ||
                  !PyLong_CheckExact(PyTuple_GET_ITEM(row, 0))) {
                throw py::type_error(
                    "Grid bindings must contain (node, x_value, y_value, z_value) tuples, "
                    "optionally followed by shared_value and by (bx_value, by_value, bz_value)");
              }
              auto node = PyLong_AsVoidPtr(PyTuple_GET_ITEM(row, 0));
              if (PyErr_Occurred()) {
                throw py::error_already_set();
              }
              std::array<size_t, 3> values;
              for (size_t axis = 0; axis < values.size(); ++axis) {
                values[axis] = static_cast<size_t>(unpack_nonnegative_integer(
                    PyTuple_GET_ITEM(row, axis + 1), "Grid value index"));
              }
              std::optional<size_t> shared_memory;
              if (shared) {
                shared_memory = static_cast<size_t>(unpack_nonnegative_integer(
                    PyTuple_GET_ITEM(row, 4), "Shared-memory value index"));
              }
              std::optional<std::array<size_t, 3>> block_values;
              if (block) {
                std::array<size_t, 3> dims;
                for (size_t axis = 0; axis < dims.size(); ++axis) {
                  dims[axis] = static_cast<size_t>(unpack_nonnegative_integer(
                      PyTuple_GET_ITEM(row, (shared ? 5 : 4) + axis),
                      "Block value index"));
                }
                block_values = dims;
              }
              grids.push_back(
                  {reinterpret_cast<uintptr_t>(node),
                   values,
                   shared_memory,
                   block_values});
            }
            auto tensor_maps =
                tensor_map_bindings.cast<std::vector<TensorMapBinding>>();
            if (!PyTuple_CheckExact(memset_bindings.ptr())) {
              throw py::type_error("Memset bindings must be an exact tuple");
            }
            std::vector<at::cuda::detail::KernelMemsetBinding> memsets;
            memsets.reserve(PyTuple_GET_SIZE(memset_bindings.ptr()));
            for (Py_ssize_t index = 0;
                 index < PyTuple_GET_SIZE(memset_bindings.ptr());
                 ++index) {
              auto* row = PyTuple_GET_ITEM(memset_bindings.ptr(), index);
              if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 4 ||
                  !PyLong_CheckExact(PyTuple_GET_ITEM(row, 0))) {
                throw py::type_error(
                    "Memset bindings must contain (node, pointer_index, offset_value_index, bytes_value_index) tuples");
              }
              auto node = PyLong_AsVoidPtr(PyTuple_GET_ITEM(row, 0));
              if (PyErr_Occurred()) {
                throw py::error_already_set();
              }
              auto pointer = static_cast<size_t>(unpack_nonnegative_integer(
                  PyTuple_GET_ITEM(row, 1), "Memset pointer index"));
              std::optional<size_t> offset;
              if (PyTuple_GET_ITEM(row, 2) != Py_None) {
                offset = static_cast<size_t>(unpack_nonnegative_integer(
                    PyTuple_GET_ITEM(row, 2), "Memset offset value index"));
              }
              auto bytes = static_cast<size_t>(unpack_nonnegative_integer(
                  PyTuple_GET_ITEM(row, 3), "Memset byte count value index"));
              memsets.push_back(
                  {reinterpret_cast<uintptr_t>(node), pointer, offset, bytes});
            }
            // (slots, nbytes, ((offset, width, "pointer" | "value", index,
            // offset_value_index or None), ...)) per host table
            std::vector<at::cuda::detail::KernelHostTableBinding> host_tables;
            host_tables.reserve(host_table_bindings.size());
            for (auto item : host_table_bindings) {
              auto* row = item.ptr();
              if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 3 ||
                  !PyTuple_CheckExact(PyTuple_GET_ITEM(row, 0)) ||
                  !PyTuple_CheckExact(PyTuple_GET_ITEM(row, 2))) {
                throw py::type_error(
                    "Host table bindings must contain (slots, nbytes, elements) tuples");
              }
              at::cuda::detail::KernelHostTableBinding binding;
              auto* slots = PyTuple_GET_ITEM(row, 0);
              for (Py_ssize_t k = 0; k < PyTuple_GET_SIZE(slots); ++k) {
                auto slot = PyLong_AsVoidPtr(PyTuple_GET_ITEM(slots, k));
                if (PyErr_Occurred()) {
                  throw py::error_already_set();
                }
                binding.slots.push_back(reinterpret_cast<uintptr_t>(slot));
              }
              binding.nbytes = static_cast<size_t>(unpack_nonnegative_integer(
                  PyTuple_GET_ITEM(row, 1), "Host table byte count"));
              auto* elements = PyTuple_GET_ITEM(row, 2);
              for (Py_ssize_t k = 0; k < PyTuple_GET_SIZE(elements); ++k) {
                auto* element = PyTuple_GET_ITEM(elements, k);
                if (!PyTuple_CheckExact(element) ||
                    PyTuple_GET_SIZE(element) != 5 ||
                    !PyUnicode_CheckExact(PyTuple_GET_ITEM(element, 2))) {
                  throw py::type_error(
                      "A host table element is an (offset, width, kind, index, offset_value_index) tuple");
                }
                at::cuda::detail::KernelHostTableElement e{
                    static_cast<size_t>(unpack_nonnegative_integer(
                        PyTuple_GET_ITEM(element, 0),
                        "Host table element offset")),
                    static_cast<size_t>(unpack_nonnegative_integer(
                        PyTuple_GET_ITEM(element, 1),
                        "Host table element width")),
                    PyUnicode_CompareWithASCIIString(
                        PyTuple_GET_ITEM(element, 2), "pointer") == 0,
                    static_cast<size_t>(unpack_nonnegative_integer(
                        PyTuple_GET_ITEM(element, 3),
                        "Host table element index"))};
                if (PyTuple_GET_ITEM(element, 4) != Py_None) {
                  e.address_offset_value_index =
                      static_cast<size_t>(unpack_nonnegative_integer(
                          PyTuple_GET_ITEM(element, 4),
                          "Host table element offset value index"));
                }
                binding.elements.push_back(e);
              }
              host_tables.push_back(std::move(binding));
            }
            // (node, source_table or None, source_pointer_index,
            // source_offset_value_index or None, pointer_index,
            // offset_value_index or None, bytes_value_index) per memcpy node
            std::vector<at::cuda::detail::KernelMemcpyBinding> memcpys;
            memcpys.reserve(memcpy_bindings.size());
            for (auto item : memcpy_bindings) {
              auto* row = item.ptr();
              if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 7 ||
                  !PyLong_CheckExact(PyTuple_GET_ITEM(row, 0))) {
                throw py::type_error(
                    "Memcpy bindings must contain (node, source_table, source_pointer_index, source_offset_value_index, pointer_index, offset_value_index, bytes_value_index) tuples");
              }
              auto node = PyLong_AsVoidPtr(PyTuple_GET_ITEM(row, 0));
              if (PyErr_Occurred()) {
                throw py::error_already_set();
              }
              at::cuda::detail::KernelMemcpyBinding binding;
              binding.node = reinterpret_cast<uintptr_t>(node);
              if (PyTuple_GET_ITEM(row, 1) != Py_None) {
                binding.source_table =
                    static_cast<size_t>(unpack_nonnegative_integer(
                        PyTuple_GET_ITEM(row, 1), "Memcpy source table"));
              }
              binding.source_pointer_index =
                  static_cast<size_t>(unpack_nonnegative_integer(
                      PyTuple_GET_ITEM(row, 2), "Memcpy source pointer index"));
              if (PyTuple_GET_ITEM(row, 3) != Py_None) {
                binding.source_offset_value_index =
                    static_cast<size_t>(unpack_nonnegative_integer(
                        PyTuple_GET_ITEM(row, 3),
                        "Memcpy source offset value index"));
              }
              binding.pointer_index =
                  static_cast<size_t>(unpack_nonnegative_integer(
                      PyTuple_GET_ITEM(row, 4), "Memcpy pointer index"));
              if (PyTuple_GET_ITEM(row, 5) != Py_None) {
                binding.address_offset_value_index =
                    static_cast<size_t>(unpack_nonnegative_integer(
                        PyTuple_GET_ITEM(row, 5), "Memcpy offset value index"));
              }
              binding.bytes_value_index =
                  static_cast<size_t>(unpack_nonnegative_integer(
                      PyTuple_GET_ITEM(row, 6),
                      "Memcpy byte count value index"));
              memcpys.push_back(binding);
            }
            // (generator, value_index) per generator the replay draws from
            std::vector<at::cuda::detail::KernelRngBinding> rngs;
            for (auto item : rng_bindings) {
              auto* row = item.ptr();
              if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 2) {
                throw py::type_error(
                    "Generator increment bindings must contain (generator, value_index) tuples");
              }
              rngs.push_back(
                  {THPGenerator_Unwrap(PyTuple_GET_ITEM(row, 0)),
                   static_cast<size_t>(unpack_nonnegative_integer(
                       PyTuple_GET_ITEM(row, 1),
                       "Generator increment value index"))});
            }
            // (nodes, site, variant value index, ((pointer index, offset value
            // index | None), ...), workspace address) per template site
            std::vector<at::cuda::detail::KernelTemplateBinding> templates;
            for (auto item : template_bindings) {
              auto* row = item.ptr();
              if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 5 ||
                  !PyTuple_CheckExact(PyTuple_GET_ITEM(row, 0)) ||
                  !PyTuple_CheckExact(PyTuple_GET_ITEM(row, 3))) {
                throw py::type_error(
                    "Template bindings must contain (nodes, site, variant_value_index, operands, workspace) tuples");
              }
              at::cuda::detail::KernelTemplateBinding binding;
              auto* nodes = PyTuple_GET_ITEM(row, 0);
              for (Py_ssize_t index = 0; index < PyTuple_GET_SIZE(nodes);
                   ++index) {
                binding.nodes.push_back(
                    static_cast<uintptr_t>(unpack_nonnegative_integer(
                        PyTuple_GET_ITEM(nodes, index), "Template node")));
              }
              binding.site = unpack_nonnegative_integer(
                  PyTuple_GET_ITEM(row, 1), "Template site");
              binding.variant_value_index =
                  static_cast<size_t>(unpack_nonnegative_integer(
                      PyTuple_GET_ITEM(row, 2),
                      "Template variant value index"));
              auto* operands = PyTuple_GET_ITEM(row, 3);
              for (Py_ssize_t index = 0; index < PyTuple_GET_SIZE(operands);
                   ++index) {
                auto* operand = PyTuple_GET_ITEM(operands, index);
                if (!PyTuple_CheckExact(operand) ||
                    PyTuple_GET_SIZE(operand) != 2) {
                  throw py::type_error(
                      "Template operands must be (pointer_index, offset_value_index) tuples");
                }
                binding.operand_pointer_index.push_back(
                    static_cast<size_t>(unpack_nonnegative_integer(
                        PyTuple_GET_ITEM(operand, 0),
                        "Template operand pointer index")));
                auto* offset = PyTuple_GET_ITEM(operand, 1);
                binding.operand_offset_value_index.push_back(
                    offset == Py_None
                        ? std::nullopt
                        : std::optional<size_t>(
                              static_cast<size_t>(unpack_nonnegative_integer(
                                  offset,
                                  "Template operand offset value index"))));
              }
              binding.workspace =
                  static_cast<uintptr_t>(unpack_nonnegative_integer(
                      PyTuple_GET_ITEM(row, 4), "Template workspace address"));
              templates.push_back(std::move(binding));
            }
            py::gil_scoped_release release;
            auto batch = self.prepare_kernel_replay_updates(
                pointers,
                count,
                scalars,
                grids,
                results,
                tensor_maps,
                memsets,
                host_tables,
                memcpys,
                rngs,
                templates);
            auto result = std::make_shared<PythonKernelPointerUpdates>(
                std::move(batch), count);
            for (const auto& binding : templates) {
              for (auto index : binding.operand_pointer_index) {
                result->referenced[index] = true;
              }
            }
            for (const auto& binding : pointers) {
              result->referenced[binding.pointer_index] = true;
            }
            for (const auto& binding : tensor_maps) {
              result->referenced[binding.pointer_index] = true;
            }
            for (const auto& binding : memsets) {
              result->referenced[binding.pointer_index] = true;
            }
            for (const auto& binding : host_tables) {
              for (const auto& element : binding.elements) {
                if (element.pointer) {
                  result->referenced[element.index] = true;
                }
              }
            }
            for (const auto& binding : memcpys) {
              result->referenced[binding.pointer_index] = true;
              if (!binding.source_table) {
                result->referenced[binding.source_pointer_index] = true;
              }
            }
            result->host_table_bindings = std::move(host_tables);
            result->memcpy_bindings = std::move(memcpys);
            result->rng_bindings = std::move(rngs);
            result->memset_bindings = std::move(memsets);
            result->pointer_bindings = std::move(pointers);
            result->scalar_bindings = std::move(scalars);
            result->grid_bindings = std::move(grids);
            result->tensor_map_bindings = std::move(tensor_maps);
            result->template_bindings = std::move(templates);
            return result;
          }),
          py::arg("pointer_bindings"),
          py::arg("pointer_count"),
          py::arg("scalar_bindings"),
          py::arg("grid_bindings"),
          py::arg("value_count"),
          py::arg("tensor_map_bindings") = py::tuple(),
          py::arg("memset_bindings") = py::tuple(),
          py::arg("host_table_bindings") = py::tuple(),
          py::arg("memcpy_bindings") = py::tuple(),
          py::arg("rng_bindings") = py::tuple(),
          py::arg("template_bindings") = py::tuple())
      .def(
          "_replay_kernel_pointer_updates",
          torch::wrap_pybind_function([](at::cuda::CUDAGraph& self,
                                         PythonKernelPointerUpdates& updates,
                                         py::handle pointers) {
            unpack_pointer_values(pointers, updates.pointers);
            py::gil_scoped_release release;
            self.replay_kernel_pointer_updates(
                *updates.batch, updates.pointers);
          }),
          py::arg("updates"),
          py::arg("pointers"))
      .def(
          "_inspect_captured_kernel_nodes",
          torch::wrap_pybind_function([](at::cuda::CUDAGraph& self,
                                         py::tuple nodes) {
            std::vector<uintptr_t> handles;
            handles.reserve(nodes.size());
            for (auto node : nodes) {
              if (!PyLong_Check(node.ptr())) {
                throw py::type_error(
                    "kernel node handles must be Python integers");
              }
              auto handle = PyLong_AsVoidPtr(node.ptr());
              if (PyErr_Occurred()) {
                throw py::error_already_set();
              }
              handles.push_back(reinterpret_cast<uintptr_t>(handle));
            }
            auto snapshot = [&] {
              py::gil_scoped_release release;
              return self.inspect_captured_kernel_nodes(handles);
            }();
            py::tuple graph_nodes(snapshot.nodes.size());
            for (size_t index = 0; index < snapshot.nodes.size(); ++index) {
              graph_nodes[index] = py::int_(snapshot.nodes[index]);
            }
            py::tuple kernels(snapshot.kernels.size());
            for (size_t index = 0; index < snapshot.kernels.size(); ++index) {
              const auto& kernel = snapshot.kernels[index];
              py::tuple arguments(kernel.arguments.size());
              for (size_t slot = 0; slot < kernel.arguments.size(); ++slot) {
                const auto& argument = kernel.arguments[slot];
                arguments[slot] = py::make_tuple(
                    argument.offset,
                    argument.size,
                    py::bytes(
                        reinterpret_cast<const char*>(argument.value.data()),
                        argument.value.size()));
              }
              kernels[index] = py::make_tuple(
                  kernel.node,
                  kernel.function,
                  kernel.kernel,
                  kernel.context,
                  py::make_tuple(
                      kernel.grid[0], kernel.grid[1], kernel.grid[2]),
                  py::make_tuple(
                      kernel.block[0], kernel.block[1], kernel.block[2]),
                  kernel.shared_memory,
                  kernel.packed,
                  arguments);
            }
            return py::make_tuple(
                snapshot.graph, snapshot.capture_id, graph_nodes, kernels);
          }),
          py::arg("nodes"))
      .def(
          "register_generator_state",
          [](::at::cuda::CUDAGraph& self, py::handle /*raw_generator*/) {
            TORCH_WARN_DEPRECATION(
                "CUDAGraph.register_generator_state() is deprecated, and will be removed in a future PyTorch release. It is now a no-op and can be safely removed from your code.");
          },
          py::arg("generator"))
      .def(
          "replay",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::replay))
      .def(
          "set_generator_increment",
          [](::at::cuda::CUDAGraph& self,
             py::handle generator,
             uint64_t increment) {
            at::Generator gen = THPGenerator_Unwrap(generator.ptr());
            py::gil_scoped_release no_gil;
            self.set_generator_increment(gen, increment);
          },
          py::arg("generator"),
          py::arg("increment"))
      .def(
          "reset",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::reset))
      .def(
          "pool",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::pool))
      .def(
          "pools",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::pools))
      .def(
          "_retain_pool",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::retain_pool))
      .def(
          "enable_debug_mode",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::enable_debug_mode))
      .def(
          "raw_cuda_graph",
          [](::at::cuda::CUDAGraph& self) {
            cudaGraph_t graph = self.raw_cuda_graph();
            // We return a raw int here, since otherwise pybind11 will
            // try to return the underlying struct of cudaGraph_t
            // points to, which is opaque and therefore causes a
            // compile error.
            return reinterpret_cast<uintptr_t>(graph);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "raw_cuda_graph_exec",
          [](::at::cuda::CUDAGraph& self) {
            cudaGraphExec_t graph_exec = self.raw_cuda_graph_exec();
            // We return a raw int here, since otherwise pybind11 will
            // try to return the underlying struct of cudaGraphExec_t
            // points to, which is opaque and therefore causes a
            // compile error.
            return reinterpret_cast<uintptr_t>(graph_exec);
          },
          py::call_guard<py::gil_scoped_release>())
      .def_static(
          "get_currently_capturing_graph",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::get_currently_capturing_graph),
          py::return_value_policy::reference)
      .def(
          "begin_capture_to_if_node",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::begin_capture_to_if_node),
          py::arg("scalar_cuda_pred_tensor"))
      .def(
          "begin_capture_to_while_node",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::begin_capture_to_while_node),
          py::arg("scalar_cuda_pred_tensor"))
      .def(
          "end_capture_to_conditional_node",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::end_capture_to_conditional_node))
      .def(
          "set_conditional_handle_for_current_node",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::set_conditional_handle_for_current_node),
          py::arg("scalar_cuda_pred_tensor"));
}
