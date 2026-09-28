"""Host tracing (private): the integer program a tape's guards and kernel
parameters lower to, and its evaluator.

A program is a flat list of int64 instructions over the call's inputs. It is
built once, at the trace's hints, and handed once to the interpreter in
torch/csrc/cuda/host_trace/Program.cpp, which every replay runs on the new
call's inputs; `_step` is the reference for its rules. The evaluator has no
undefined behavior: an addition, multiplication or left shift that overflows,
a division or shift outside its domain, or a boolean operand other than 0 or
1 stops the evaluation with a nonzero status, which the replay treats as a
miss.
"""

from __future__ import annotations

import operator
import struct
from enum import IntEnum
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


MIN_I64 = -(1 << 63)
MAX_I64 = (1 << 63) - 1


class Status(IntEnum):
    SUCCESS = 0
    ADD_OVERFLOW = 1
    MULTIPLY_OVERFLOW = 2
    DIVISION_DOMAIN = 3
    BOOLEAN_DOMAIN = 4
    SHIFT_DOMAIN = 5


class OutOfDomain(ValueError):
    """An instruction fails at the hints, so the program cannot hold there."""

    def __init__(self, status: Status, row: tuple) -> None:
        super().__init__(f"{row} fails at the hints: {status.name}")
        self.status = status


LEAVES = ("boxed", "pointer", "storage_offset", "size", "stride")
_COMPARISONS: dict[str, Callable[[int, int], bool]] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}
# two's complement on int64 operands: total, and the result is an int64
_BITWISE: dict[str, Callable[[int, int], int]] = {
    "bitand": operator.and_,
    "bitor": operator.or_,
    "bitxor": operator.xor,
}
_ARITY = {
    "add": 2,
    "multiply": 2,
    "floordiv": 2,
    "ceildiv": 2,
    "and": 2,
    "bitlength": 1,
    "lshift": 2,
    "f32div": 2,
    "select": 3,
    **dict.fromkeys(_COMPARISONS, 2),
    **dict.fromkeys(_BITWISE, 2),
}


def f32_bits(n: int, d: int) -> int:
    """static_cast<float>(n) / static_cast<float>(d)'s bits as an int32: each
    rounded to float32, their double quotient rounded once more, which is the
    float32 quotient."""
    f = [struct.unpack("<f", struct.pack("<f", x))[0] for x in (n, d)]
    return struct.unpack("<i", struct.pack("<f", f[0] / f[1]))[0]


def _fits(value: int) -> bool:
    return MIN_I64 <= value <= MAX_I64


def _step(op: str, args: list[int]) -> int | Status:
    """One instruction under the evaluator's rules, on Python ints."""
    if op in ("add", "multiply"):
        r = args[0] + args[1] if op == "add" else args[0] * args[1]
        if _fits(r):
            return r
        return Status.ADD_OVERFLOW if op == "add" else Status.MULTIPLY_OVERFLOW
    if op in ("floordiv", "ceildiv"):
        n, d = args
        if n < 0 or d <= 0:
            return Status.DIVISION_DOMAIN
        return n // d if op == "floordiv" else -(-n // d)
    if op in _COMPARISONS:
        return int(_COMPARISONS[op](*args))
    if op in _BITWISE:
        return _BITWISE[op](*args)
    if op == "bitlength":
        return args[0].bit_length()
    if op == "lshift":
        a, b = args
        if b < 0:
            return Status.SHIFT_DOMAIN
        r = a << min(b, 64)
        return r if _fits(r) else Status.MULTIPLY_OVERFLOW
    if op == "f32div":
        n, d = args
        if n < 0 or d <= 0:
            return Status.DIVISION_DOMAIN
        return f32_bits(n, d)
    if op in ("and", "select"):
        if any(b not in (0, 1) for b in args[: 2 if op == "and" else 1]):
            return Status.BOOLEAN_DOMAIN
        if op == "and":
            return args[0] & args[1]
        return args[1] if args[0] else args[2]
    return min(args) if op == "min" else max(args)


class IntegerProgram:
    """int64 instructions over the call's inputs, built at the hints.

    `emit(op, *operands)` appends a row and returns its index; an identical
    row returns the index it already has. The rows:
      ("constant", value)
      ("boxed", i)                      the int input i
      ("pointer", i), ("storage_offset", i),
      ("size", i, dim), ("stride", i, dim)   metadata of the tensor input i
      ("add" | "multiply" | "floordiv" | "ceildiv", a, b)
      ("eq" | "ne" | "lt" | "le" | "gt" | "ge" | "and", a, b)
      ("bitand" | "bitor" | "bitxor", a, b)
      ("bitlength", a)                  a.bit_length(), of |a|
      ("lshift", a, b)                  a * 2**b, for b >= 0
      ("f32div", a, b)                  float(a) / float(b)'s float32 bits, as
                                        an int32, for a >= 0 and b > 0
      ("min" | "max", a, b, ...)
      ("select", cond, if_true, if_false)
    where a, b, ... are earlier rows. floordiv and ceildiv are defined for a
    nonnegative numerator and a positive divisor, where truncation is floor;
    a caller lowering another division (Python's floor on negative operands,
    a modulo) builds it from these. Every row is evaluated at the hints when
    it is emitted (`values`); one that fails there raises OutOfDomain.
    """

    def __init__(self, inputs: Sequence[Any]) -> None:
        self.inputs = tuple(inputs)
        self.instructions: list[tuple] = []
        self.values: list[int] = []
        self._rows: dict[tuple, int] = {}

    def emit(self, op: str, *operands: int) -> int:
        row = (op, *operands)
        index = self._rows.get(row)
        if index is not None:
            return index
        if any(type(x) is not int for x in operands):
            raise TypeError(f"{row}: operands must be ints")
        if op == "constant":
            if len(operands) != 1 or not _fits(operands[0]):
                raise ValueError(f"{row}: a constant is one int64")
            value: int | Status = operands[0]
        elif op in LEAVES:
            value = self._leaf(row)
        else:
            if op in ("min", "max"):
                arity_ok = len(operands) >= 2
            elif op in _ARITY:
                arity_ok = len(operands) == _ARITY[op]
            else:
                raise ValueError(f"{row}: unknown instruction")
            if not arity_ok:
                raise ValueError(f"{row}: wrong number of operands")
            if any(not 0 <= x < len(self.instructions) for x in operands):
                raise ValueError(f"{row}: operands must be earlier rows")
            value = _step(op, [self.values[x] for x in operands])
            if isinstance(value, Status):
                raise OutOfDomain(value, row)
        index = len(self.instructions)
        self.instructions.append(row)
        self.values.append(value)
        self._rows[row] = index
        return index

    def _leaf(self, row: tuple) -> int:
        op, *operands = row
        if len(operands) != (2 if op in ("size", "stride") else 1):
            raise ValueError(f"{row}: wrong number of operands")
        if not 0 <= operands[0] < len(self.inputs):
            raise ValueError(f"{row}: no such input")
        value = _read_leaf(row, self.inputs[operands[0]])
        if value is None:
            raise ValueError(f"{row}: the input is not of that kind")
        return value


def _read_leaf(row: tuple, x: Any) -> int | None:
    op, _, *dim = row
    if op == "boxed":
        return x if type(x) is int and _fits(x) else None
    if not isinstance(x, torch.Tensor) or x.layout != torch.strided or x.is_nested:
        return None
    if op == "pointer":
        return x.data_ptr() if _fits(x.data_ptr()) else None
    if op == "storage_offset":
        return int(x.storage_offset())
    if not 0 <= dim[0] < x.dim():
        return None
    return x.size(dim[0]) if op == "size" else x.stride(dim[0])


def compile_program(program: IntegerProgram) -> torch._C._HostTraceProgram:
    """The program checked once by the interpreter and run per call:
    `evaluate_inputs(inputs)` is (status, every row's value), the values None
    unless the status is Status.SUCCESS, or None when the inputs are not of
    the kinds its leaves read."""
    return torch._C._HostTraceProgram(program.instructions, len(program.inputs))
