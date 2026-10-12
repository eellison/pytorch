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

import math
import operator
import struct
from enum import IntEnum
from fractions import Fraction
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
    FLOAT_DOMAIN = 6


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
    "tofloat": 1,
    "fsqrt": 1,
    "fdiv": 2,
    "feq": 2,
    "flt": 2,
    "fadd": 2,
    "fsub": 2,
    "fmul": 2,
    "f32i": 1,
    "f32r": 1,
    "f32add": 2,
    "f32sub": 2,
    "f32mul": 2,
    "f32quot": 2,
    "f32fma": 3,
    "select": 3,
    **dict.fromkeys(_COMPARISONS, 2),
    **dict.fromkeys(_BITWISE, 2),
}


def f32_round(x: float) -> float:
    """static_cast<float>(x) of a double, as a double (round to nearest even)."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def _f32_nearest(q: Fraction) -> float:
    """The exact rational q rounded once to the nearest float32 (ties to even)."""
    if q == 0:
        return 0.0
    sign, q = (-1.0 if q < 0 else 1.0), abs(q)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q:
        e -= 1
    ulp = Fraction(2) ** (max(e, -126) - 23)
    m, r = divmod(q / ulp, 1)
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and m % 2):
        m += 1
    v = m * ulp
    return sign * (math.inf if v >= 2**128 else float(v))


def f32_of_int(n: int) -> float:
    """static_cast<float>(n): the integer rounded once (int -> double -> float
    rounds twice above 2**53)."""
    return f32_round(float(n)) if abs(n) <= 1 << 53 else _f32_nearest(Fraction(n))


def _fdiv(x: float, y: float) -> float:
    if y != 0.0:
        return x / y
    return math.nan if x == 0.0 or math.isnan(x) else math.copysign(math.inf, x) * math.copysign(1.0, y)


def float_value(op: str, args: Sequence[float]) -> float:
    """A float row's value (a double; a float32 row's is a float32 value) on
    its float operands, an int for f32i. A float32 add, subtract, multiply or
    divide done in double and rounded once is the float32 operation's result
    (53 >= 2 * 24 + 2); the fused multiply-add rounds the exact value once.
    A float32 row's operands are float32 values; others convert first."""
    if op == "f32i":
        return f32_of_int(args[0])
    if op == "f32r":
        return f32_round(args[0])
    if op.startswith("f32"):
        args = [f32_round(a) for a in args]  # static_cast<float> of each operand
    if op == "f32fma":
        x, y, z = args
        if all(math.isfinite(v) for v in args):
            q = Fraction(x) * Fraction(y) + Fraction(z)
            if q == 0:  # IEEE's zero: negative only for -0 + -0
                return -0.0 if math.copysign(1.0, x * y) < 0 and math.copysign(1.0, z) < 0 and x * y == 0 else 0.0
            return _f32_nearest(q)
        return f32_round(x * y + z)
    x, y = args
    kind = op[3:] if op.startswith("f32") else op[1:]
    v = x + y if kind == "add" else x - y if kind == "sub" else x * y if kind == "mul" else _fdiv(x, y)
    return f32_round(v) if op.startswith("f32") else v


def f32_bits(n: int, d: int) -> int:
    """static_cast<float>(n) / static_cast<float>(d)'s bits as an int32."""
    return struct.unpack("<i", struct.pack("<f", float_value("f32quot", [f32_of_int(n), f32_of_int(d)])))[0]


def f64_bits(x: float) -> int:
    """A double's bits as an int64: a float row's value."""
    return struct.unpack("<q", struct.pack("<d", x))[0]


def f64(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<q", bits))[0]


# IEEE operations on doubles' bits, as the host computes them (no domain)
_FLOAT_ROWS = frozenset({"fadd", "fsub", "fmul", "f32i", "f32r", "f32add", "f32sub", "f32mul", "f32quot", "f32fma"})


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
    if op == "tofloat":
        return f64_bits(float(args[0]))
    if op == "fsqrt":
        x = f64(args[0])
        return f64_bits(math.sqrt(x)) if 0.0 <= x < math.inf else Status.FLOAT_DOMAIN
    if op == "fdiv":
        x, y = map(f64, args)
        q = x / y if y != 0.0 else math.nan
        return f64_bits(q) if math.isfinite(q) else Status.FLOAT_DOMAIN
    if op in _FLOAT_ROWS:
        return f64_bits(float_value(op, [args[0]] if op == "f32i" else [f64(a) for a in args]))
    if op == "feq":
        return int(f64(args[0]) == f64(args[1]))
    if op == "flt":
        return int(f64(args[0]) < f64(args[1]))
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
      ("tofloat", a)                    float(a)'s bits as an int64
      ("fsqrt", a), ("fdiv", a, b)      math.sqrt and / on the doubles whose
                                        bits a and b are, for a finite result
      ("feq" | "flt", a, b)             the doubles' == and <
      ("fadd" | "fsub" | "fmul", a, b)  + - * of the doubles
      ("f32i", a)                       static_cast<float>(a)'s value, as a double
      ("f32r", a)                       static_cast<float>(the double a)
      ("f32add" | "f32sub" | "f32mul" | "f32quot", a, b), ("f32fma", a, b, c)
                                        the float32 operation (std::fmaf) on
                                        float32 values held as doubles
      ("min" | "max", a, b, ...)
      ("select", cond, if_true, if_false)
    where a, b, ... are earlier rows. floordiv and ceildiv are defined for a
    nonnegative numerator and a positive divisor, where truncation is floor;
    a caller lowering another division (Python's floor on negative operands,
    a modulo) builds it from these. Every row is evaluated at the hints when
    it is emitted (`values`); one that fails there raises OutOfDomain.
    While `domains` is a list, a division or shift emitted reads admitted
    operands (numerator 0, divisor 1, shift 0 where its own are not) and
    appends the row that is 1 where its own are: it fails at no call. So do
    fsqrt (of a nonnegative double) and fdiv, admitted only for a divisor of
    magnitude at least 1, where no finite quotient overflows; an fdiv outside
    that at the hints raises OutOfDomain.
    """

    def __init__(self, inputs: Sequence[Any]) -> None:
        self.inputs = tuple(inputs)
        self.instructions: list[tuple] = []
        self.values: list[int] = []
        self._rows: dict[tuple, int] = {}
        self.domains: list[int] | None = None

    def emit(self, op: str, *operands: int) -> int:
        if self.domains is not None and op in ("floordiv", "ceildiv", "f32div", "lshift", "fsqrt", "fdiv"):
            domains, self.domains = self.domains, None
            try:
                a, *rest = operands
                zero = self.emit("constant", 0)  # also +0.0's bits
                if op == "fsqrt":
                    ok = self.emit("select", self.emit("flt", a, zero), zero, self.emit("constant", 1))
                    operands = (self.emit("select", ok, a, zero),)
                elif op == "fdiv":
                    (b,) = rest
                    one, minus_one = self.emit("constant", f64_bits(1.0)), self.emit("constant", f64_bits(-1.0))
                    small = self.emit("and", self.emit("flt", b, one), self.emit("flt", minus_one, b))
                    ok = self.emit("select", small, zero, self.emit("constant", 1))
                    if self.values[ok] != 1:
                        raise OutOfDomain(Status.FLOAT_DOMAIN, (op, *operands))
                    operands = (self.emit("select", ok, a, zero), self.emit("select", ok, b, one))
                elif op == "lshift":
                    (b,) = rest
                    ok = self.emit("ge", b, zero)
                    operands = (a, self.emit("select", ok, b, zero))
                else:
                    (b,) = rest
                    ok = self.emit("and", self.emit("ge", a, zero), self.emit("gt", b, zero))
                    operands = (self.emit("select", ok, a, zero), self.emit("select", ok, b, self.emit("constant", 1)))
                domains.append(ok)
            finally:
                self.domains = domains
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
