from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from typing import Any


def integer_width(llvm_type: str) -> int:
    match = re.fullmatch(r"i([1-9][0-9]*)", llvm_type)
    if match is None or int(match[1]) not in (1, 8, 16, 32, 64):
        raise ValueError(f"Unsupported integer type: {llvm_type}")
    return int(match[1])


@dataclass(frozen=True)
class ScalarValue:
    llvm_type: str
    value: int | float
    size: int

    def integer(self, *, signed: bool = True) -> int:
        width = integer_width(self.llvm_type)
        if type(self.value) is not int or not 0 <= self.value < 2**width:
            raise ValueError("Integer value does not represent its exact LLVM bits")
        return self.value - 2**width if signed and self.value >= 2**(width - 1) else self.value

    def data(self) -> bytes:
        if self.llvm_type in ("f32", "f64"):
            expected = 4 if self.llvm_type == "f32" else 8
            if self.size != expected or type(self.value) is not float:
                raise ValueError("Floating value has an incompatible storage width")
            return struct.pack("<f" if expected == 4 else "<d", self.value)
        if self.llvm_type.startswith("!llvm.ptr"):
            if type(self.value) is not int or not 0 <= self.value < 2**(8 * self.size):
                raise ValueError("Pointer value exceeds its compiler storage width")
            return self.value.to_bytes(self.size, "little")
        width = integer_width(self.llvm_type)
        if self.size != (width + 7) // 8:
            raise ValueError("Unsupported integer allocation size")
        return self.integer(signed=False).to_bytes(self.size, "little")


@dataclass(frozen=True)
class AggregateValue:
    llvm_type: str
    data_bytes: bytes
    fields: tuple[tuple[tuple[int, ...], ScalarValue], ...]

    def data(self) -> bytes:
        return self.data_bytes


def scalar(llvm_type: str, value: int | float, *, pointer_size: int | None = None) -> ScalarValue:
    if llvm_type in ("f32", "f64"):
        size = 4 if llvm_type == "f32" else 8
        fmt = "<f" if size == 4 else "<d"
        return ScalarValue(llvm_type, struct.unpack(fmt, struct.pack(fmt, float(value)))[0], size)
    if llvm_type.startswith("!llvm.ptr"):
        if pointer_size is None or type(pointer_size) is not int or pointer_size <= 0:
            raise ValueError("A pointer requires a compiler-provided storage size")
        result = ScalarValue(llvm_type, value, pointer_size)
    else:
        width = integer_width(llvm_type)
        if type(value) is not int:
            raise TypeError("Integer flow requires an integer value")
        result = ScalarValue(llvm_type, value % 2**width, (width + 7) // 8)
    result.data()
    return result


@dataclass(frozen=True)
class AttributeDecoding:
    constant: ScalarValue | None = None
    predicate: int | None = None


def comparison_predicates() -> tuple[int, ...]:
    from cutlass._mlir.dialects import llvm

    return tuple(int(value) for value in (
        llvm.ICmpPredicate.eq, llvm.ICmpPredicate.ne, llvm.ICmpPredicate.slt, llvm.ICmpPredicate.sle,
        llvm.ICmpPredicate.sgt, llvm.ICmpPredicate.sge, llvm.ICmpPredicate.ult, llvm.ICmpPredicate.ule,
        llvm.ICmpPredicate.ugt, llvm.ICmpPredicate.uge,
    ))


def decode_attributes(node: Any) -> AttributeDecoding:
    from cutlass._mlir import ir

    if node.kind == "constant":
        value = ir.Attribute.parse(node.value)
        if isinstance(value, ir.BoolAttr) and node.llvm_type == "i1":
            return AttributeDecoding(constant=scalar("i1", int(value.value)))
        if isinstance(value, (ir.IntegerAttr, ir.FloatAttr)) and str(value.type) == node.llvm_type:
            return AttributeDecoding(constant=scalar(node.llvm_type, value.value))
        raise ValueError("Unsupported typed LLVM constant")
    if node.kind == "llvm.select":
        attrs = {name: ir.Attribute.parse(value) for name, value in node.attributes}
        if attrs not in ({}, {"fastmathFlags": ir.Attribute.parse("#llvm.fastmath<none>")}) or len(node.operands) != 3:
            raise ValueError("Unsupported select attributes or arity")
        return AttributeDecoding()
    if node.kind == "llvm.icmp":
        attrs = dict(node.attributes)
        if set(attrs) != {"predicate"} or node.llvm_type != "i1":
            raise ValueError("Unsupported integer comparison attributes")
        predicate = ir.Attribute.parse(attrs["predicate"])
        if not isinstance(predicate, ir.IntegerAttr):
            raise ValueError("Expected a typed LLVM comparison predicate")
        if predicate.value not in comparison_predicates():
            raise ValueError("Unknown LLVM comparison predicate")
        return AttributeDecoding(predicate=predicate.value)
    supported = {"argument", "zero", "llvm.trunc", "llvm.zext", "llvm.sext", "llvm.add", "llvm.sub", "llvm.mul",
                 "llvm.udiv", "llvm.sdiv", "llvm.urem", "llvm.srem", "llvm.shl", "llvm.lshr", "llvm.ashr",
                 "llvm.and", "llvm.or", "llvm.xor"}
    if node.kind not in supported or node.attributes:
        raise ValueError(f"Unsupported scalar decoding entry: {node.kind}")
    return AttributeDecoding()


def _evaluate(flow: Any, arguments: tuple[ScalarValue | AggregateValue, ...], lookup: Any,
              predicates: tuple[int, ...]) -> ScalarValue | AggregateValue:
    def run(node: Any) -> ScalarValue | AggregateValue:
        if node.kind == "argument":
            if type(node.argument) is not int or not 0 <= node.argument < len(arguments):
                raise ValueError("Value flow does not name an original host argument")
            result = arguments[node.argument]
            if node.path:
                if not isinstance(result, AggregateValue):
                    raise ValueError("A component path requires a typed aggregate")
                fields = dict(result.fields)
                if node.path not in fields:
                    raise ValueError("Only compiler-identified scalar component paths are executable")
                result = fields[node.path]
            if result.llvm_type != node.llvm_type:
                raise ValueError("Value flow and concrete argument types differ")
            return result
        if node.kind == "constant":
            return lookup(node).constant
        if node.kind == "zero":
            return scalar(node.llvm_type, 0)
        if node.kind == "llvm.select":
            lookup(node)
            condition = run(node.operands[0])
            if not isinstance(condition, ScalarValue) or condition.llvm_type != "i1":
                raise ValueError("Select requires an i1 condition")
            result = run(node.operands[1 if condition.integer(signed=False) else 2])
            if result.llvm_type != node.llvm_type:
                raise ValueError("Select result type differs from its chosen operand")
            return result
        operands = tuple(run(item) for item in node.operands)
        if any(not isinstance(item, ScalarValue) for item in operands):
            raise ValueError("Aggregate transformations require their own compiler layout")
        if node.kind in ("llvm.trunc", "llvm.zext", "llvm.sext"):
            if node.attributes or len(operands) != 1:
                raise ValueError("Unsupported integer cast attributes or arity")
            source = operands[0]
            before, after = integer_width(source.llvm_type), integer_width(node.llvm_type)
            if (node.kind == "llvm.trunc" and after >= before) or (node.kind != "llvm.trunc" and after <= before):
                raise ValueError("Integer cast does not change widths as declared")
            return scalar(node.llvm_type, source.integer(signed=node.kind == "llvm.sext"))
        if len(operands) != 2 or operands[0].llvm_type != operands[1].llvm_type:
            raise ValueError(f"Unsupported LLVM scalar operation: {node.kind}")
        left, right = operands
        if node.kind == "llvm.icmp":
            predicate = lookup(node).predicate
            a, b = left.integer(), right.integer()
            u, v = left.integer(signed=False), right.integer(signed=False)
            comparisons = dict(zip(predicates, (a == b, a != b, a < b, a <= b, a > b, a >= b,
                                                u < v, u <= v, u > v, u >= v)))
            if predicate not in comparisons:
                raise ValueError("Unknown LLVM comparison predicate")
            return scalar("i1", int(comparisons[predicate]))
        if node.attributes or left.llvm_type != node.llvm_type:
            raise ValueError("Unsupported scalar arithmetic flags or result type")
        width = integer_width(node.llvm_type)
        a, b = left.integer(), right.integer()
        u, v = left.integer(signed=False), right.integer(signed=False)
        if node.kind == "llvm.add":
            result = a + b
        elif node.kind == "llvm.sub":
            result = a - b
        elif node.kind == "llvm.mul":
            result = a * b
        elif node.kind in ("llvm.sdiv", "llvm.srem", "llvm.udiv", "llvm.urem"):
            if b == 0 or (node.kind.startswith("llvm.s") and a == -(2**(width - 1)) and b == -1):
                raise ValueError("Undefined integer division in the host computation")
            if node.kind in ("llvm.sdiv", "llvm.srem"):
                quotient = (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
                result = quotient if node.kind == "llvm.sdiv" else a - quotient * b
            else:
                result = u // v if node.kind == "llvm.udiv" else u % v
        elif node.kind in ("llvm.shl", "llvm.lshr", "llvm.ashr"):
            if v >= width:
                raise ValueError("Undefined integer shift in the host computation")
            result = u << v if node.kind == "llvm.shl" else (a if node.kind == "llvm.ashr" else u) >> v
        elif node.kind == "llvm.and":
            result = u & v
        elif node.kind == "llvm.or":
            result = u | v
        elif node.kind == "llvm.xor":
            result = u ^ v
        else:
            raise ValueError(f"Unsupported LLVM scalar operation: {node.kind}")
        return scalar(node.llvm_type, result)

    return run(flow)


def evaluate(flow: Any, arguments: tuple[ScalarValue | AggregateValue, ...], context: Any,
             *, decoded: Any = None) -> ScalarValue | AggregateValue:
    from cutlass._mlir import ir

    if decoded is not None:
        from torch._inductor.runtime._cudagraph._compiler.decoded_values import DecodedValues

        if type(decoded) is not DecodedValues:
            raise TypeError("Expected an owned decoded scalar program")
        decoded.check(context)
        if not any(flow is root for root in decoded.roots):
            raise ValueError("Scalar evaluation is not an instruction of the decoded CFG")
    lookup = decode_attributes if decoded is None else decoded.lookup
    with context, ir.Location.unknown(), ir.raw_values():
        return _evaluate(flow, arguments, lookup, comparison_predicates())
