from __future__ import annotations

import io
from collections import deque
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from torch._inductor.runtime._cudagraph._compiler.argument_flow import ValueFlow
from torch._inductor.runtime._cudagraph._compiler.values import AggregateValue, evaluate, ScalarValue


CFG_VERSION = 1
_BINARY = {
    "llvm.add", "llvm.sub", "llvm.mul", "llvm.udiv", "llvm.sdiv", "llvm.urem", "llvm.srem",
    "llvm.shl", "llvm.lshr", "llvm.ashr", "llvm.and", "llvm.or", "llvm.xor",
}
_CASTS = {"llvm.trunc", "llvm.zext", "llvm.sext"}


@dataclass(frozen=True)
class ValueType:
    llvm_type: str
    leaves: tuple[tuple[tuple[int, ...], str], ...] | None = None


@dataclass(frozen=True)
class Instruction:
    result: int
    expression: ValueFlow


@dataclass(frozen=True)
class Edge:
    block: int
    arguments: tuple[int, ...]


@dataclass(frozen=True)
class Terminator:
    kind: str
    values: tuple[int, ...] = ()
    edges: tuple[Edge, ...] = ()


@dataclass(frozen=True)
class Block:
    arguments: tuple[int, ...]
    instructions: tuple[Instruction, ...]
    terminator: Terminator


def _snapshot(module: Any) -> tuple[str, bytes]:
    buffer = io.BytesIO()
    module.operation.write_bytecode(buffer)
    return str(module), buffer.getvalue()


def _type(typ: Any, path: tuple[int, ...] = (), depth: int = 0) -> ValueType:
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm

    if depth > 16:
        raise ValueError("LLVM helper aggregate nesting exceeds the supported depth")
    name = str(typ)
    if isinstance(typ, llvm.StructType):
        if typ.opaque:
            raise ValueError("Opaque LLVM helper aggregate types are unsupported")
        leaves = []
        for index, member in enumerate(typ.body):
            child = _type(member, (*path, index), depth + 1)
            leaves.extend(child.leaves if child.leaves is not None else (((*path, index), child.llvm_type),))
        return ValueType(name, tuple(leaves))
    if (isinstance(typ, llvm.PointerType) or isinstance(typ, ir.IntegerType) and typ.width in (1, 8, 16, 32, 64)
            or isinstance(typ, (ir.F32Type, ir.F64Type))):
        return ValueType(name)
    raise ValueError(f"Unsupported LLVM helper type: {typ}")


def _validate_value(value: Any, typ: ValueType) -> None:
    expected = AggregateValue if typ.leaves is not None else ScalarValue
    if type(value) is not expected or value.llvm_type != typ.llvm_type:
        raise ValueError("LLVM helper argument or result has the wrong type")
    if typ.leaves is None:
        value.data()
        return
    if (type(value.data_bytes) is not bytes
            or tuple((path, item.llvm_type) for path, item in value.fields) != typ.leaves):
        raise ValueError("LLVM helper aggregate fields lack exact typed leaf coverage")
    for _, item in value.fields:
        if type(item) is not ScalarValue:
            raise ValueError("LLVM helper aggregate leaves must be scalar values")
        item.data()


@dataclass(frozen=True)
class CFGProgram:
    module: Any = field(repr=False)
    context: Any = field(repr=False)
    function_name: str
    argument_types: tuple[str, ...]
    result_types: tuple[str, ...]
    types: tuple[ValueType, ...]
    blocks: tuple[Block, ...]
    text: str = field(repr=False)
    bytecode: bytes = field(repr=False)
    _owners: tuple[Any, Any] = field(repr=False)
    _fingerprint: str = field(repr=False)

    def _digest(self) -> str:
        return sha256(repr((CFG_VERSION, self.function_name, self.argument_types, self.result_types,
                            self.types, self.blocks, self.text, self.bytecode)).encode()).hexdigest()

    def check(self) -> None:
        from cutlass._mlir import ir

        if (len(self._owners) != 2 or self.module is not self._owners[0] or self.context is not self._owners[1]
                or self.module.context != self.context or self._digest() != self._fingerprint):
            raise RuntimeError("LLVM helper program ownership or typed control flow changed")
        with self.context, ir.Location.unknown(), ir.raw_values():
            if _snapshot(self.module) != (self.text, self.bytecode):
                raise RuntimeError("The original LLVM helper Module changed")

    def evaluate(self, arguments: tuple[ScalarValue | AggregateValue, ...]) -> tuple[ScalarValue | AggregateValue, ...]:
        self.check()
        if type(arguments) is not tuple or len(arguments) != len(self.argument_types):
            raise ValueError("LLVM helper arguments do not match its exact entry signature")
        slots: list[Any] = [None] * len(self.types)
        block_index, incoming = 0, arguments
        for _ in self.blocks:
            block = self.blocks[block_index]
            for slot, value in zip(block.arguments, incoming):
                _validate_value(value, self.types[slot])
                slots[slot] = value
            for instruction in block.instructions:
                result = evaluate(instruction.expression, tuple(slots), self.context)
                _validate_value(result, self.types[instruction.result])
                slots[instruction.result] = result
            terminator = block.terminator
            if terminator.kind == "return":
                result = tuple(slots[slot] for slot in terminator.values)
                if tuple(value.llvm_type for value in result) != self.result_types:
                    raise RuntimeError("LLVM helper return lost its verified types")
                return result
            selected = 0
            if terminator.kind == "conditional":
                condition = slots[terminator.values[0]]
                if type(condition) is not ScalarValue or condition.llvm_type != "i1":
                    raise RuntimeError("LLVM helper branch requires a concrete i1")
                selected = 0 if condition.integer(signed=False) else 1
            edge = terminator.edges[selected]
            incoming = tuple(slots[slot] for slot in edge.arguments)
            block_index = edge.block
        raise RuntimeError("An acyclic LLVM helper did not reach a return")


def _expression(op: Any, slots: dict[Any, int]) -> ValueFlow:
    from cutlass._mlir import ir

    if len(op.results) != 1 or op.regions or op.successors:
        raise ValueError(f"Unsupported LLVM helper operation or effect: {op.name}")
    typ = str(op.results[0].type)
    refs = tuple(ValueFlow("argument", str(value.type), argument=slots[value]) for value in op.operands)
    attrs = set(op.attributes)
    if op.name == "llvm.mlir.constant" and not refs and attrs == {"value"}:
        value = op.attributes["value"]
        if not isinstance(value, (ir.IntegerAttr, ir.FloatAttr, ir.BoolAttr)):
            raise ValueError("LLVM helper constants must have scalar numeric attributes")
        return ValueFlow("constant", typ, value=str(value))
    if op.name == "llvm.mlir.zero" and not refs and not attrs and isinstance(op.results[0].type, ir.IntegerType):
        return ValueFlow("zero", typ)
    if op.name == "llvm.extractvalue" and len(refs) == 1 and attrs == {"position"}:
        path = tuple(op.attributes["position"])
        if not path or any(type(index) is not int or index < 0 for index in path):
            raise ValueError("LLVM helper component paths require nonnegative indices")
        source = _type(op.operands[0].type)
        if source.leaves is None or (path, typ) not in source.leaves:
            raise ValueError("Only exact scalar-leaf aggregate projections are supported")
        return ValueFlow("argument", typ, argument=refs[0].argument, path=path)
    if op.name == "llvm.select":
        attributes = {name: op.attributes[name] for name in attrs}
        if (len(refs) != 3 or attributes not in ({}, {"fastmathFlags": ir.Attribute.parse("#llvm.fastmath<none>")})
                or str(op.operands[0].type) != "i1" or op.operands[1].type != op.operands[2].type
                or op.operands[1].type != op.results[0].type):
            raise ValueError("Unsupported LLVM helper select attributes or types")
        return ValueFlow(op.name, typ, operands=refs,
                         attributes=tuple((name, str(op.attributes[name])) for name in sorted(attrs)))
    if op.name in _BINARY | _CASTS | {"llvm.icmp"}:
        count = 1 if op.name in _CASTS else 2
        if len(refs) != count or attrs != ({"predicate"} if op.name == "llvm.icmp" else set()):
            raise ValueError("Unsupported LLVM helper arithmetic attributes or arity")
        if any(not isinstance(value.type, ir.IntegerType) for value in (*op.operands, *op.results)):
            raise ValueError("LLVM helper arithmetic requires supported integer operands/results")
        attributes = tuple((name, str(op.attributes[name])) for name in sorted(attrs))
        return ValueFlow(op.name, typ, operands=refs, attributes=attributes)
    raise ValueError(f"Unsupported LLVM helper operation or effect: {op.name}")


def read_cfg_function(module: Any, function_name: str) -> CFGProgram:
    from cutlass._mlir import ir

    if not isinstance(module, ir.Module) or type(function_name) is not str or not function_name:
        raise TypeError("Expected an explicitly owned LLVM Module and helper function name")
    with module.context, ir.Location.unknown(), ir.raw_values():
        before = _snapshot(module)
        if not module.operation.verify():
            raise ValueError("LLVM helper Module failed verification")
        matches = [view.operation for view in module.body.operations if view.operation.name == "llvm.func"
                   and view.operation.attributes["sym_name"].value == function_name]
        if len(matches) != 1 or len(matches[0].regions) != 1 or not matches[0].regions[0].blocks:
            raise ValueError("Expected one defined LLVM helper function")
        function = matches[0]
        allowed = {"sym_name", "function_type", "CConv", "linkage", "visibility_", "sym_visibility",
                   "dso_local", "arg_attrs", "res_attrs", "no_inline", "unnamed_addr"}
        if set(function.attributes) - allowed:
            raise ValueError("Unsupported LLVM helper function attributes")
        if "CConv" in function.attributes and function.attributes["CConv"] != ir.Attribute.parse("#llvm.cconv<ccc>"):
            raise ValueError("Unsupported LLVM helper calling convention")
        for name in ("arg_attrs", "res_attrs"):
            if name in function.attributes and any(len(item) > int("cudagraph.source_arg" in item)
                                                  for item in function.attributes[name]):
                raise ValueError("Unsupported LLVM helper argument/result assumptions")
        blocks = tuple(function.regions[0].blocks)
        block_ids = {block: index for index, block in enumerate(blocks)}
        slots, types = {}, []
        for block in blocks:
            for value in (*block.arguments, *(result for view in block.operations for result in view.operation.results)):
                slots[value] = len(types)
                types.append(_type(value.type))
        result_blocks, returns = [], set()
        for block in blocks:
            operations = tuple(view.operation for view in block.operations)
            if not operations:
                raise ValueError("LLVM helper blocks require explicit terminators")
            instructions = tuple(Instruction(slots[op.results[0]], _expression(op, slots))
                                 for op in operations[:-1] if len(op.results) == 1)
            if len(instructions) != len(operations) - 1:
                raise ValueError("Unsupported LLVM helper operation or effect without one result")
            end = operations[-1]
            if end.results or end.regions:
                raise ValueError("Unsupported LLVM helper terminator")

            def edge(successor: Any, values: Any) -> Edge:
                if successor not in block_ids or tuple(value.type for value in values) != tuple(successor.arguments.types):
                    raise ValueError("LLVM helper branch operands differ from destination block arguments")
                return Edge(block_ids[successor], tuple(slots[value] for value in values))

            if end.name == "llvm.return" and not end.successors and not end.attributes and len(end.operands) <= 1:
                result_types = tuple(str(value.type) for value in end.operands)
                returns.add(result_types)
                terminator = Terminator("return", tuple(slots[value] for value in end.operands))
            elif end.name == "llvm.br" and len(end.successors) == 1 and not end.attributes:
                terminator = Terminator("jump", edges=(edge(end.successors[0], end.operands),))
            elif (end.name == "llvm.cond_br" and len(end.successors) == 2
                  and set(end.attributes) == {"operandSegmentSizes"}):
                sizes = tuple(end.attributes["operandSegmentSizes"])
                if len(sizes) != 3 or sizes[0] != 1 or min(sizes) < 0 or sum(sizes) != len(end.operands):
                    raise ValueError("LLVM conditional branch has invalid operand segments")
                if str(end.operands[0].type) != "i1":
                    raise ValueError("LLVM conditional branch requires i1")
                split = 1 + sizes[1]
                terminator = Terminator("conditional", (slots[end.operands[0]],), (
                    edge(end.successors[0], end.operands[1:split]), edge(end.successors[1], end.operands[split:]),
                ))
            else:
                raise ValueError(f"Unsupported LLVM helper terminator: {end.name}")
            result_blocks.append(Block(tuple(slots[value] for value in block.arguments), instructions, terminator))
        successors = [set(edge.block for edge in block.terminator.edges) for block in result_blocks]
        reachable, pending = set(), [0]
        while pending:
            index = pending.pop()
            if index not in reachable:
                reachable.add(index)
                pending.extend(successors[index])
        if len(reachable) != len(blocks):
            raise ValueError("Unreachable LLVM helper blocks are unsupported")
        counts = [sum(index in targets for targets in successors) for index in range(len(blocks))]
        ready = deque(index for index, count in enumerate(counts) if not count)
        visited = 0
        while ready:
            index = ready.popleft()
            visited += 1
            for target in successors[index]:
                counts[target] -= 1
                if not counts[target]:
                    ready.append(target)
        if visited != len(blocks):
            raise ValueError("Cyclic LLVM helper control flow is unsupported")
        if len(returns) != 1:
            raise ValueError("LLVM helper must have one consistent return type")
        result_types = next(iter(returns))
        argument_types = tuple(str(value.type) for value in blocks[0].arguments)
        result_type = result_types[0] if result_types else "void"
        expected = ir.Type.parse(f"!llvm.func<{result_type} ({', '.join(argument_types)})>")
        if function.attributes["function_type"].value != expected:
            raise ValueError("LLVM helper has an unsupported variadic or mismatched signature")
        if _snapshot(module) != before:
            raise RuntimeError("LLVM helper inspection changed the original Module")
        result = CFGProgram(module, module.context, function_name, argument_types, result_types, tuple(types),
                            tuple(result_blocks), *before, (module, module.context), "")
        object.__setattr__(result, "_fingerprint", result._digest())
        result.check()
        return result
