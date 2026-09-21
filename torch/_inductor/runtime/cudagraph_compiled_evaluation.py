"""Compile guarded integer and typed physical-parameter DAGs during preparation."""

import ctypes
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from functools import cache
from types import MappingProxyType
from typing import Any

import sympy

import torch
from torch._guards import ShapeGuard, SLoc
from torch._inductor.codecache import CppCodeCache
from torch._inductor.runtime.cudagraph_boxed_replay import (
    _NumericProgram,
    _ParameterProgram,
)
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._sympy.functions import Identity
from torch.utils._sympy.value_ranges import ValueRanges


MIN_I64 = -(1 << 63)
MAX_I64 = (1 << 63) - 1
# the native call ops: `call` passes a std::vector, `pcall` a pointer and a length
_CALL_OPS = ("call", "pcall")
STRICT_FLOAT_FLAGS = (
    "-fno-fast-math",
    "-ffp-contract=off",
    "-fexcess-precision=standard",
)

_FLOAT_EXPRESSIONS = {
    "ffromint": "static_cast<double>({left})",
    "fneg": "-float_value({left})",
    "fsqrt": "std::sqrt(float_value({left}))",
    "fround32": "static_cast<double>(static_cast<float>(float_value({left})))",
    "fadd": "float_value({left}) + float_value({right})",
    "fsub": "float_value({left}) - float_value({right})",
    "fmul": "float_value({left}) * float_value({right})",
    "fdiv": "float_value({left}) / float_value({right})",
    "fpow": "std::pow(float_value({left}), float_value({right}))",
    "ftobits32": "float32_bits(float_value({left}))",
}
_FLOAT_CPP = r"""#include <cmath>
#include <cstdint>
#include <cstring>
static inline double float_value(int64_t bits) noexcept {
  double value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
static inline int64_t float_bits(double value) noexcept {
  int64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
static inline int64_t float32_bits(double value) noexcept {
  const float narrowed = static_cast<float>(value);
  int32_t bits;
  std::memcpy(&bits, &narrowed, sizeof(bits));
  return bits;
}
"""


@cache
def _float_preparation_library():
    lines = [_FLOAT_CPP]
    for op, expression in _FLOAT_EXPRESSIONS.items():
        value = expression.format(left="left", right="right")
        if op != "ftobits32":
            value = f"float_bits({value})"
        lines.append(
            f'extern "C" int64_t prepare_{op}(int64_t left, int64_t right) noexcept {{ return {value}; }}'
        )
    library = CppCodeCache.load("\n".join(lines), extra_flags=STRICT_FLOAT_FLAGS)
    for op in _FLOAT_EXPRESSIONS:
        function = getattr(library, f"prepare_{op}")
        function.argtypes = (ctypes.c_int64, ctypes.c_int64)
        function.restype = ctypes.c_int64
    return library


def _evaluate_float(op, left, right=0):
    return getattr(_float_preparation_library(), f"prepare_{op}")(left, right)


class EarlyStatus(IntEnum):
    SUCCESS = 0
    ADD_OVERFLOW = 1
    MULTIPLY_OVERFLOW = 2
    DIVISION_DOMAIN = 3
    BOOLEAN_DOMAIN = 4
    CALL_FAILED = 5


@dataclass(frozen=True)
class CompiledNumeric:
    library_owner: Any
    function_address: int
    leaf_bindings: tuple[tuple, ...]
    input_count: int
    integer_indices: tuple[int, ...]
    value_count: int
    cpp_source: str

    def bind_inputs(self, inputs):
        if type(inputs) not in (list, tuple) or len(inputs) != self.input_count:
            raise ValueError("Numeric inputs must have the prepared arity")
        for index, value in enumerate(inputs):
            if index in self.integer_indices:
                if type(value) is not int or not MIN_I64 <= value <= MAX_I64:
                    raise ValueError("Boxed numeric leaves require exact int64 inputs")
            elif type(value) not in (torch.Tensor, torch.nn.Parameter):
                raise ValueError("Metadata leaves require original Tensor inputs")
        leaves = []
        for kind, index, *dimension in self.leaf_bindings:
            value = inputs[index]
            if kind == "boxed":
                leaves.append(value)
            elif kind == "pointer":
                leaves.append(value.data_ptr())
            elif kind == "storage_offset":
                leaves.append(value.storage_offset())
            else:
                dim = dimension[0]
                if (
                    value.layout != torch.strided
                    or value.is_nested
                    or dim >= value.dim()
                ):
                    raise ValueError(
                        "Metadata dimensions require in-range strided Tensors"
                    )
                leaves.append(value.size(dim) if kind == "size" else value.stride(dim))
        return tuple(leaves)

    def evaluate_leaves(self, leaves):
        if len(leaves) != len(self.leaf_bindings) or any(
            type(value) is not int or not MIN_I64 <= value <= MAX_I64
            for value in leaves
        ):
            raise ValueError("Leaf values must match the prepared int64 ABI")
        arguments = (ctypes.c_int64 * len(leaves))(*leaves)
        outputs = (ctypes.c_int64 * self.value_count)()
        function = ctypes.CFUNCTYPE(
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        )(self.function_address)
        status = EarlyStatus(function(arguments, outputs))
        return status, tuple(outputs)


def _numeric_source(
    program: _NumericProgram,
) -> tuple[str, tuple[tuple, ...], int, tuple[int, ...], int]:
    """Compile validated instruction order; output slots are valid on success."""
    if type(program) is not _NumericProgram:
        raise TypeError("Expected the existing numeric preparation program")
    input_count = len(program.inputs) + len(program.tensor_inputs)
    integer_indices = tuple(program.integer_indices)
    integer_set = set(integer_indices)
    if len(integer_set) != len(integer_indices) or any(
        type(index) is not int or not 0 <= index < input_count
        for index in integer_indices
    ):
        raise ValueError("Declared integer inputs must be distinct and in range")
    lines = [
        _FLOAT_CPP,
        "#include <algorithm>",
        "#include <vector>",
        'extern "C" int32_t evaluate_early(const int64_t* leaves, int64_t* outputs) noexcept {',
    ]
    leaves = []
    leaf_indices = {}
    loaded_integers = set()
    call_arguments = False
    comparisons = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
    for index, row in enumerate(program.instructions):
        if type(row) is not tuple or not row or type(row[0]) is not str:
            raise ValueError("Instructions must be tagged tuples")
        op, *operands = row
        target = f"v{index}"
        if op in ("constant", "fconst") and len(operands) == 1:
            value = operands[0]
            if type(value) is not int or not MIN_I64 <= value <= MAX_I64:
                raise ValueError("Numeric constants must fit int64")
            literal = (
                "(-9223372036854775807LL - 1)" if value == MIN_I64 else f"{value}LL"
            )
            lines.append(f"  const int64_t {target} = {literal};")
        elif op in ("boxed", "pointer", "storage_offset", "size", "stride"):
            arity = 2 if op in ("size", "stride") else 1
            if len(operands) != arity or any(
                type(value) is not int or value < 0 for value in operands
            ):
                raise ValueError(
                    "Leaf bindings require nonnegative exact integer indices"
                )
            slot = operands[0]
            if slot >= input_count or (slot in integer_set) != (op == "boxed"):
                raise ValueError("Leaf input kind or index differs from preparation")
            if op == "boxed":
                if slot in loaded_integers:
                    raise ValueError("Each declared integer input requires one load")
                loaded_integers.add(slot)
            if row not in leaf_indices:
                leaf_indices[row] = len(leaves)
                leaves.append(row)
            lines.append(f"  const int64_t {target} = leaves[{leaf_indices[row]}];")
        elif op in ("call", "pcall"):
            if len(operands) < 2:
                raise ValueError("Native calls require an address and callable owner")
            address, owner, *arguments = operands
            if (
                type(address) is not int
                or not 0 < address < 1 << 64
                or not callable(owner)
                or any(
                    type(slot) is not int or not 0 <= slot < index for slot in arguments
                )
            ):
                raise ValueError("Native call address, owner, or operands are invalid")
            values = ", ".join(f"v{slot}" for slot in arguments)
            if op == "pcall":
                # the pointer-and-length ABI: the arguments as a stack array, no
                # vector per call (a callee that throws still fails the evaluation)
                storage, passed = (
                    (f"    const int64_t arguments[] = {{{values}}};", "arguments")
                    if arguments
                    else ("    const int64_t* arguments = nullptr;", "arguments")
                )
                invoke = f"    {target} = reinterpret_cast<int64_t (*)(const int64_t*, size_t)>(uintptr_t{{{address}ULL}})({passed}, {len(arguments)});"
            else:
                # the vector ABI: one vector per compiled evaluation, refilled per call
                if not call_arguments:
                    lines.append("  std::vector<int64_t> arguments;")
                    call_arguments = True
                storage = f"    arguments.assign({{{values}}});"
                invoke = f"    {target} = reinterpret_cast<int64_t (*)(const std::vector<int64_t>&)>(uintptr_t{{{address}ULL}})(arguments);"
            lines.extend(
                (
                    f"  int64_t {target};",
                    "  try {",
                    storage,
                    invoke,
                    "  } catch (...) {",
                    f"    return {int(EarlyStatus.CALL_FAILED)};",
                    "  }",
                )
            )
        elif op in ("max", "min"):
            if len(operands) < 2 or any(
                type(slot) is not int or not 0 <= slot < index for slot in operands
            ):
                raise ValueError("Min/max require at least two preceding operands")
            values = ", ".join(f"v{slot}" for slot in operands)
            lines.append(f"  const int64_t {target} = std::{op}({{{values}}});")
        elif op in _FLOAT_EXPRESSIONS:
            arity = (
                1 if op in ("ffromint", "fneg", "fsqrt", "fround32", "ftobits32") else 2
            )
            if len(operands) != arity or any(
                type(slot) is not int or not 0 <= slot < index for slot in operands
            ):
                raise ValueError("Float operands must precede their result")
            value = _FLOAT_EXPRESSIONS[op].format(
                left=f"v{operands[0]}", right=f"v{operands[-1]}"
            )
            if op != "ftobits32":
                value = f"float_bits({value})"
            lines.append(f"  const int64_t {target} = {value};")
        else:
            arity = 3 if op == "select" else 2
            if len(operands) != arity or any(
                type(value) is not int or not 0 <= value < index for value in operands
            ):
                raise ValueError("Arithmetic operands must precede their result")
            left, right = (f"v{value}" for value in operands[:2])
            if op in ("add", "multiply"):
                builtin = "add" if op == "add" else "mul"
                status = (
                    EarlyStatus.ADD_OVERFLOW
                    if op == "add"
                    else EarlyStatus.MULTIPLY_OVERFLOW
                )
                lines.append(f"  int64_t {target};")
                lines.append(
                    f"  if (__builtin_{builtin}_overflow({left}, {right}, &{target})) return {int(status)};"
                )
            elif op in ("ceildiv", "floordiv"):
                lines.append(
                    f"  if ({left} < 0 || {right} <= 0) return {int(EarlyStatus.DIVISION_DOMAIN)};"
                )
                extra = f" + ({left} % {right} != 0)" if op == "ceildiv" else ""
                lines.append(f"  const int64_t {target} = {left} / {right}{extra};")
            elif op in comparisons:
                lines.append(
                    f"  const int64_t {target} = {left} {comparisons[op]} {right};"
                )
            elif op == "and":
                lines.append(
                    f"  if (({left} != 0 && {left} != 1) || ({right} != 0 && {right} != 1)) return {int(EarlyStatus.BOOLEAN_DOMAIN)};"
                )
                lines.append(f"  const int64_t {target} = {left} && {right};")
            elif op == "select":
                lines.append(
                    f"  if ({left} != 0 && {left} != 1) return {int(EarlyStatus.BOOLEAN_DOMAIN)};"
                )
                lines.append(
                    f"  const int64_t {target} = {left} ? {right} : v{operands[2]};"
                )
            else:
                raise ValueError(f"Unsupported numeric instruction: {op}")
        lines.append(f"  outputs[{index}] = {target};")
    if loaded_integers != integer_set:
        raise ValueError("Every declared integer input requires one load")
    lines.extend(("  return 0;", "}"))
    source = "\n".join(lines) + "\n"
    return (
        source,
        tuple(leaves),
        input_count,
        integer_indices,
        len(program.instructions),
    )


class LateStatus(IntEnum):
    SUCCESS = 0
    EARLY_VALUE_RANGE = 1
    POINTER_RANGE = 2
    DIVISION_DOMAIN = 3
    LIVE_POISON = 4


@dataclass(frozen=True)
class CompiledParameter:
    library_owner: Any
    function_address: int
    early_count: int
    root_count: int
    output_widths: tuple[int, ...]
    output_roots: tuple[tuple[int, ...], ...]
    pointer_inputs: tuple[int, ...]
    cpp_source: str

    def evaluate(self, early, roots):
        if len(early) != self.early_count or any(
            type(value) is not int or not -(1 << 63) <= value < 1 << 63
            for value in early
        ):
            raise ValueError("Early values must match the prepared int64 ABI")
        if len(roots) != self.root_count or any(
            type(value) is not int or not 0 <= value < 1 << 64 for value in roots
        ):
            raise ValueError("Roots must match the prepared uintptr64 ABI")
        arguments = (ctypes.c_int64 * len(early))(*early)
        pointers = (ctypes.c_uint64 * len(roots))(*roots)
        outputs = (ctypes.c_int64 * len(self.output_widths))()
        function = ctypes.CFUNCTYPE(
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int64),
        )(self.function_address)
        return LateStatus(function(arguments, pointers, outputs)), tuple(outputs)


_PARAMETER_PRELUDE = r"""#include <cstdint>
#include <cstring>
struct Value { uint64_t bits = 0; bool poison = false; };
using Wide = unsigned __int128;
using SignedWide = __int128;
static inline SignedWide signed_value(uint64_t bits, uint32_t width) noexcept {
  return (bits & (uint64_t{1} << (width - 1)))
      ? SignedWide(bits) - (SignedWide(1) << width) : SignedWide(bits);
}
extern "C" int32_t evaluate_late(const int64_t* early, const uintptr_t* roots,
                                  int64_t* outputs) noexcept {
  static_assert(sizeof(uintptr_t) == sizeof(uint64_t));
"""


def _parameter_source(
    program: _ParameterProgram,
) -> tuple[
    str, int, int, tuple[int, ...], tuple[tuple[int, ...], ...], tuple[int, ...]
]:
    if type(program) is not _ParameterProgram:
        raise TypeError("Expected the existing late parameter preparation program")
    early_count = len(program.numeric.instructions)
    root_count = program.input_count + len(program.buffer_indices)
    if type(program.input_count) is not int or program.input_count < 0:
        raise ValueError("Input count must be nonnegative")
    rows, outputs = tuple(program.rows), tuple(program.outputs)
    widths, roots, pointer_inputs = [], [], []
    lines = [_PARAMETER_PRELUDE]
    flagged = {"add", "sub", "mul", "shl", "udiv", "sdiv", "lshr", "ashr"}
    binary = flagged | {"urem", "srem", "and", "or", "xor"}
    comparisons = {
        "eq": "==",
        "ne": "!=",
        "ult": "<",
        "ule": "<=",
        "ugt": ">",
        "uge": ">=",
        "slt": "<",
        "sle": "<=",
        "sgt": ">",
        "sge": ">=",
    }
    for index, row in enumerate(rows):
        if type(row) is not tuple or len(row) < 2 or type(row[0]) is not str:
            raise ValueError("Parameter instructions must be tagged tuples")
        op = row[0]
        width = 1 if op == "icmp" else row[1]
        if type(width) is not int or width not in (1, 32, 64):
            raise ValueError("Unsupported parameter width")
        arity = (
            5
            if op in flagged or op == "select"
            else 4
            if op in binary or op in ("icmp", "pointer")
            else 3
        )
        if len(row) != arity:
            raise ValueError("Invalid parameter instruction arity")
        flags = row[4] if op in flagged else ()
        allowed = {"nuw", "nsw"} if op in ("add", "sub", "mul", "shl") else {"exact"}
        if (
            type(flags) is not tuple
            or any(type(flag) is not str or flag not in allowed for flag in flags)
            or len(flags) != len(set(flags))
        ):
            raise ValueError("Unsupported or repeated parameter flags")
        operands = ()
        if op in binary or op == "icmp":
            operands = row[2:4]
        elif op in ("trunc", "zext", "sext", "select"):
            operands = row[2:]
        if any(type(value) is not int or not 0 <= value < index for value in operands):
            raise ValueError("SSA operands must precede their result")
        used_roots = set().union(*(roots[value] for value in operands))
        target = f"v{index}"
        mask = f"{(1 << width) - 1}ULL"
        lines.append(f"  Value {target};")
        if op == "constant":
            value = row[2]
            if type(value) is not int or not 0 <= value < 1 << width:
                raise ValueError("Constant exceeds its integer width")
            lines.append(f"  {target}.bits = {value}ULL;")
        elif op == "value":
            value = row[2]
            if width == 1 or type(value) is not int or not 0 <= value < early_count:
                raise ValueError("Invalid early value input")
            if width == 32:
                lines.append(
                    f"  if (early[{value}] < -2147483648LL || early[{value}] > 4294967295LL) return 1;"
                )
            lines.append(f"  {target}.bits = static_cast<uint64_t>(early[{value}]);")
        elif op == "pointer":
            root, offset = row[2:]
            if (
                width != 64
                or type(root) is not int
                or not 0 <= root < root_count
                or type(offset) is not int
                or not 0 <= offset < early_count
            ):
                raise ValueError("Invalid pointer root or byte displacement")
            used_roots.add(root)
            pointer_inputs.append(root)
            lines.extend(
                (
                    f"  if (early[{offset}] >= 0) {{",
                    f"    const uint64_t delta = static_cast<uint64_t>(early[{offset}]);",
                    f"    if (delta > UINT64_MAX - roots[{root}]) return 2;",
                    f"    {target}.bits = roots[{root}] + delta;",
                    "  } else {",
                    f"    const uint64_t delta = uint64_t{{0}} - static_cast<uint64_t>(early[{offset}]);",
                    f"    if (delta > roots[{root}]) return 2;",
                    f"    {target}.bits = roots[{root}] - delta;",
                    "  }",
                )
            )
        elif op in ("trunc", "zext", "sext"):
            operand = operands[0]
            source_width = widths[operand]
            if (op == "trunc" and source_width <= width) or (
                op != "trunc" and source_width >= width
            ):
                raise ValueError("Cast widths must narrow or extend as declared")
            lines.append(f"  {target} = v{operand};")
            if op == "sext":
                lines.append(
                    f"  if (v{operand}.bits & {1 << (source_width - 1)}ULL) {target}.bits |= ~{(1 << source_width) - 1}ULL;"
                )
        elif op == "select":
            condition, when_true, when_false = operands
            if (
                widths[condition] != 1
                or widths[when_true] != width
                or widths[when_false] != width
            ):
                raise ValueError("Invalid select operand types")
            lines.append(
                f"  {target} = v{condition}.poison ? Value{{0, true}} : (v{condition}.bits ? v{when_true} : v{when_false});"
            )
        elif op in binary or op == "icmp":
            first, second = operands
            source_width = widths[first]
            if widths[second] != source_width or (
                op != "icmp" and source_width != width
            ):
                raise ValueError("Binary operand widths differ")
            if op not in ("icmp", "and", "or", "xor") and width == 1:
                raise ValueError("Arithmetic requires i32 or i64")
            if op == "icmp" and row[1] not in comparisons:
                raise ValueError("Unsupported comparison predicate")
            a, b = f"v{first}.bits", f"v{second}.bits"
            sa, sb = (
                f"signed_value({a}, {source_width})",
                f"signed_value({b}, {source_width})",
            )
            lines.extend(
                (
                    f"  {target}.poison = v{first}.poison || v{second}.poison;",
                    f"  if (!{target}.poison) {{",
                )
            )
            if op in ("add", "sub", "mul"):
                symbol = {"add": "+", "sub": "-", "mul": "*"}[op]
                lines.append(f"    {target}.bits = {a} {symbol} {b};")
                checks = []
                if "nuw" in flags:
                    checks.append(
                        f"{a} < {b}"
                        if op == "sub"
                        else f"Wide({a}) {symbol} {b} > {mask}"
                    )
                if "nsw" in flags:
                    lines.append(
                        f"    const SignedWide signed_result = {sa} {symbol} {sb};"
                    )
                    checks.append(
                        f"signed_result < -(SignedWide(1) << {width - 1}) || signed_result >= (SignedWide(1) << {width - 1})"
                    )
                if checks:
                    lines.append(
                        f"    {target}.poison = "
                        + " || ".join(f"({check})" for check in checks)
                        + ";"
                    )
            elif op in ("udiv", "urem", "sdiv", "srem"):
                lines.append(f"    if ({b} == 0) return 3;")
                symbol = "/" if op.endswith("div") else "%"
                if op.startswith("s"):
                    lines.append(
                        f"    if ({sa} == -(SignedWide(1) << {width - 1}) && {sb} == -1) return 3;"
                    )
                    lines.append(
                        f"    {target}.bits = static_cast<uint64_t>({sa} {symbol} {sb});"
                    )
                else:
                    lines.append(f"    {target}.bits = {a} {symbol} {b};")
                if "exact" in flags:
                    left, right = (sa, sb) if op == "sdiv" else (a, b)
                    lines.append(f"    {target}.poison = {left} % {right} != 0;")
            elif op in ("and", "or", "xor"):
                lines.append(
                    f"    {target}.bits = {a} "
                    + {"and": "&", "or": "|", "xor": "^"}[op]
                    + f" {b};"
                )
            elif op in ("shl", "lshr", "ashr"):
                lines.extend(
                    (
                        f"    {target}.poison = {b} >= {width};",
                        f"    if (!{target}.poison) {{",
                    )
                )
                lines.append(
                    f"      {target}.bits = {a} {'<<' if op == 'shl' else '>>'} {b};"
                )
                if op == "shl":
                    checks = []
                    if "nuw" in flags:
                        checks.append(f"(Wide({a}) << {b}) > {mask}")
                    if "nsw" in flags:
                        lines.append(
                            f"      const SignedWide signed_result = {sa} * (SignedWide(1) << {b});"
                        )
                        checks.append(
                            f"signed_result < -(SignedWide(1) << {width - 1}) || signed_result >= (SignedWide(1) << {width - 1})"
                        )
                    if checks:
                        lines.append(
                            f"      {target}.poison = "
                            + " || ".join(f"({check})" for check in checks)
                            + ";"
                        )
                else:
                    if op == "ashr":
                        lines.append(
                            f"      if ({b} && ({a} & {1 << (width - 1)}ULL)) {target}.bits |= {mask} ^ ({mask} >> {b});"
                        )
                    if "exact" in flags:
                        lines.append(
                            f"      {target}.poison = {b} && ({a} & ((uint64_t{{1}} << {b}) - 1));"
                        )
                lines.append("    }")
            else:
                predicate = row[1]
                left, right = (sa, sb) if predicate.startswith("s") else (a, b)
                lines.append(
                    f"    {target}.bits = {left} {comparisons[predicate]} {right};"
                )
            lines.append("  }")
        else:
            raise ValueError(f"Unsupported parameter operation: {op}")
        lines.append(f"  {target}.bits &= {mask};")
        widths.append(width)
        roots.append(used_roots)
    if any(
        type(value) is not int or not 0 <= value < len(rows) or widths[value] == 1
        for value in outputs
    ):
        raise ValueError("Physical outputs must name i32 or i64 nodes")
    for output in outputs:
        lines.append(f"  if (v{output}.poison) return 4;")
    for index, output in enumerate(outputs):
        lines.append(f"  uint64_t output_{index} = v{output}.bits;")
        if widths[output] == 32:
            lines.append(
                f"  if (output_{index} & 2147483648ULL) output_{index} |= ~4294967295ULL;"
            )
        lines.append(
            f"  std::memcpy(outputs + {index}, &output_{index}, sizeof(output_{index}));"
        )
    lines.extend(("  return 0;", "}"))
    source = "\n".join(lines) + "\n"
    return (
        source,
        early_count,
        root_count,
        tuple(widths[value] for value in outputs),
        tuple(tuple(sorted(roots[value])) for value in outputs),
        tuple(pointer_inputs),
    )


@dataclass(frozen=True)
class IntegerPayloadContract:
    shape_env: ShapeEnv
    obligations: tuple[sympy.Basic, ...]
    additional_guards: tuple[sympy.Basic, ...]
    ranges: frozenset
    range_map: Mapping
    replacements: Mapping
    axioms: Mapping
    divisible: frozenset
    guards: tuple[sympy.Basic, ...]


def integer_payload_contract(shape_env: ShapeEnv) -> IntegerPayloadContract:
    if shape_env.deferred_runtime_asserts:
        raise ValueError("Deferred assertions require an ordered guard representation")
    # Resolve existing chains before their lazy compression can stale the snapshot.
    for symbol in tuple(shape_env.replacements):
        shape_env._find(symbol)
    # The same for divisibility facts a replacement has decided: simplify() drops
    # them from the set the first time it meets a FloorDiv.
    shape_env._update_divisible()
    ranges = tuple(shape_env.var_to_range.items())
    replacements = MappingProxyType(dict(shape_env.replacements))
    axioms = MappingProxyType(dict(shape_env.axioms))
    divisible = frozenset(shape_env.divisible)
    guards = tuple(guard.expr for guard in shape_env.guards)
    obligations = []
    for symbol, bounds in ranges:
        if isinstance(bounds.lower, sympy.Integer):
            obligations.append(sympy.Ge(symbol, bounds.lower, evaluate=False))
        if isinstance(bounds.upper, sympy.Integer):
            obligations.append(sympy.Le(symbol, bounds.upper, evaluate=False))
    obligations.extend(guards)
    represented = {}
    for guard in obligations:
        represented.update(shape_env.get_implications(guard))
    additional = []

    def retain(guard):
        implications = shape_env.get_implications(guard)
        if guard is sympy.true or all(
            represented.get(expr) is value for expr, value in implications
        ):
            return
        additional.append(guard)
        obligations.append(guard)
        represented.update(implications)

    # A temporary replacement can disappear before the reuse guard is compiled.
    for symbol, value in replacements.items():
        retain(sympy.Eq(symbol, value, evaluate=False))
    for value in sorted(divisible, key=sympy.default_sort_key):
        retain(sympy.Eq(value, 0, evaluate=False))
    for expression, value in axioms.items():
        if represented.get(expression) is value:
            continue
        if value is not sympy.true and value is not sympy.false:
            raise ValueError("ShapeEnv axioms require Boolean implication values")
        retain(expression if value is sympy.true else sympy.Not(expression))
    return IntegerPayloadContract(
        shape_env,
        tuple(dict.fromkeys(obligations)),
        tuple(additional),
        frozenset(ranges),
        MappingProxyType(dict(ranges)),
        replacements,
        axioms,
        divisible,
        guards,
    )


def integer_payload_contract_from_guards(
    guards: Iterable[sympy.Basic], symbols: Iterable[sympy.Symbol]
) -> IntegerPayloadContract:
    """Freeze exported integer facts in a separate context without example hints."""
    symbols = tuple(dict.fromkeys(symbols))
    if any(
        not isinstance(symbol, sympy.Symbol) or symbol.is_integer is not True
        for symbol in symbols
    ):
        raise ValueError("Integer payload sources must be original integer symbols")
    environment = ShapeEnv(duck_shape=False, specialize_zero_one=False)
    for symbol in symbols:
        environment._update_var_to_range(symbol, ValueRanges.unknown_int())
    for guard in tuple(guards):
        if not isinstance(guard, sympy.Basic):
            raise ValueError("Integer payload guards must be symbolic predicates")
        if guard.has(Identity) or any(
            isinstance(node, sympy.Expr) and node.is_integer is not True
            for node in sympy.preorder_traversal(guard)
        ):
            continue
        if not guard.free_symbols.issubset(symbols):
            raise ValueError("Integer payload guard has an unbound original symbol")
        environment.guards.append(ShapeGuard(guard, SLoc(None, None), False))
        environment.axioms.update(environment.get_implications(guard))
        if guard.free_symbols:
            environment._maybe_guard_rel(guard)
    return integer_payload_contract(environment)


def _contract_is_current(contract, shape_env):
    return (
        contract.shape_env is shape_env
        and contract.range_map == shape_env.var_to_range
        and contract.replacements == shape_env.replacements
        and contract.axioms == shape_env.axioms
        and contract.divisible == shape_env.divisible
        and not shape_env.deferred_runtime_asserts
        and len(contract.guards) == len(shape_env.guards)
        and all(
            saved == live.expr for saved, live in zip(contract.guards, shape_env.guards)
        )
    )


def _is_integer_payload(expression: sympy.Expr) -> bool:
    return (
        isinstance(expression, sympy.Expr)
        and expression.is_integer is True
        and not expression.has(Identity)
        and all(
            not isinstance(node, sympy.Expr) or node.is_integer is True
            for node in sympy.preorder_traversal(expression)
        )
    )


def simplify_integer_payload(
    expression: sympy.Expr,
    shape_env: ShapeEnv,
    contract: IntegerPayloadContract | None = None,
):
    """Return a simplified integer and its shared raw guard contract."""
    if not _is_integer_payload(expression):
        raise ValueError("Payload simplification requires an integer-only expression")
    contract = integer_payload_contract(shape_env) if contract is None else contract
    if not _contract_is_current(contract, shape_env):
        raise ValueError(
            "The payload contract requires its finalized original ShapeEnv"
        )
    known_symbols = {symbol for symbol, _ in contract.ranges}
    if not expression.free_symbols.issubset(known_symbols):
        raise ValueError("Payload contains a symbol outside its ShapeEnv contract")
    options = {"axioms": contract.obligations, "var_to_range": contract.ranges}
    simplified = shape_env.simplify(expression, **options)
    static = shape_env._maybe_evaluate_static(simplified, compute_hint=False, **options)
    if not _contract_is_current(contract, shape_env):
        raise ValueError(
            "Simplification changed its finalized original ShapeEnv contract"
        )
    if static is not None:
        simplified = static
    return simplified, contract


def _function_address(library, name):
    address = ctypes.cast(getattr(library, name), ctypes.c_void_p).value
    if address is None:
        raise RuntimeError(f"Compiled evaluator {name} has no function address")
    return address


def compile_numeric(program: _NumericProgram) -> CompiledNumeric:
    source, leaves, input_count, integer_indices, value_count = _numeric_source(program)
    library = CppCodeCache.load(source, extra_flags=STRICT_FLOAT_FLAGS)
    owners = tuple(row[2] for row in program.instructions if row[0] in _CALL_OPS)
    return CompiledNumeric(
        (library, *owners),
        _function_address(library, "evaluate_early"),
        leaves,
        input_count,
        integer_indices,
        value_count,
        source,
    )


def compile_parameter(program: _ParameterProgram) -> CompiledParameter:
    source, early_count, root_count, widths, roots, pointer_inputs = _parameter_source(
        program
    )
    library = CppCodeCache.load(source, extra_flags=STRICT_FLOAT_FLAGS)
    return CompiledParameter(
        library,
        _function_address(library, "evaluate_late"),
        early_count,
        root_count,
        widths,
        roots,
        pointer_inputs,
        source,
    )


@dataclass(frozen=True)
class CompiledEvaluation:
    library_owner: Any
    early: CompiledNumeric
    late: CompiledParameter | None
    pointer_count: int
    cpp_source: str

    @property
    def early_address(self) -> int:
        return self.early.function_address

    @property
    def late_address(self) -> int:
        return 0 if self.late is None else self.late.function_address

    @property
    def leaf_count(self) -> int:
        return len(self.early.leaf_bindings)

    @property
    def early_count(self) -> int:
        return self.early.value_count

    @property
    def late_count(self) -> int:
        return 0 if self.late is None else len(self.late.output_widths)

    @property
    def registration_kwargs(self):
        return dict(
            early_address=self.early_address,
            late_address=self.late_address,
            leaf_count=self.leaf_count,
            early_count=self.early_count,
            late_count=self.late_count,
            pointer_count=self.pointer_count,
            library_owner=self.library_owner,
        )


def compile_evaluation(
    numeric: _NumericProgram,
    parameters: _ParameterProgram | None = None,
    *,
    pointer_count: int,
) -> CompiledEvaluation:
    source, leaves, input_count, integer_indices, value_count = _numeric_source(numeric)
    if type(pointer_count) is not int or pointer_count < input_count:
        raise ValueError("Pointer count must cover the prepared boxed inputs")
    late_result = None
    if parameters is not None:
        if (
            type(parameters) is not _ParameterProgram
            or parameters.numeric is not numeric
        ):
            raise ValueError("Early and late programs require the same numeric DAG")
        late_result = _parameter_source(parameters)
        late_source, early_count, root_count, _, _, _ = late_result
        if early_count != value_count or root_count != pointer_count:
            raise ValueError(
                "Late program counts differ from the prepared early values or roots"
            )
        source += "\n" + late_source
    library = CppCodeCache.load(source, extra_flags=STRICT_FLOAT_FLAGS)
    owners = (library, *(row[2] for row in numeric.instructions if row[0] in _CALL_OPS))
    early = CompiledNumeric(
        owners,
        _function_address(library, "evaluate_early"),
        leaves,
        input_count,
        integer_indices,
        value_count,
        source,
    )
    late = None
    if late_result is not None:
        _, early_count, root_count, widths, roots, pointer_inputs = late_result
        late = CompiledParameter(
            owners,
            _function_address(library, "evaluate_late"),
            early_count,
            root_count,
            widths,
            roots,
            pointer_inputs,
            source,
        )
    return CompiledEvaluation(owners, early, late, pointer_count, source)
