from dataclasses import dataclass, field
from typing import Any
from torch._inductor.runtime._cudagraph._compiler.cfg_values import CFGProgram
from cutlass._mlir.dialects.llvm import IntegerOverflowFlags
from torch._inductor.runtime._cudagraph._compiler.metadata_regions import MetadataAggregate, _check_computation, _validate
from torch._inductor.runtime._cudagraph._compiler.values import ScalarValue, integer_width


_OVERFLOW = frozenset({"llvm.add", "llvm.sub", "llvm.mul", "llvm.shl", "llvm.trunc"})


_NSW, _NUW = int(IntegerOverflowFlags.nsw), int(IntegerOverflowFlags.nuw)


def _flags(op: Any) -> int:
    from cutlass._mlir import ir

    assembly = op.get_asm(print_generic_op_form=True, use_local_scope=True)
    start = assembly.find("<{")
    end = assembly.find("}>", start)
    if (start < 0 or end < start or assembly.find("<{", start + 2) >= 0
            or assembly.find("}>", end + 2) >= 0):
        raise ValueError("Expected one complete compiler-printed LLVM property dictionary")
    properties = ir.Attribute.parse(assembly[start + 1:end + 1])
    if not isinstance(properties, ir.DictAttr) or len(properties) != 1 or "overflowFlags" not in properties:
        raise ValueError("Only the exact LLVM overflow property is supported")
    value = properties["overflowFlags"]
    allowed = int(IntegerOverflowFlags.nsw | IntegerOverflowFlags.nuw)
    if (not isinstance(value, ir.IntegerAttr) or value.type != ir.IntegerType.get_signless(32)
            or value.value < 0 or value.value & ~allowed):
        raise ValueError(f"Unknown compiler LLVM overflow flags: {type(value).__name__} {value}")
    return value.value


@dataclass(frozen=True)
class PropertyCFG:
    cfg: CFGProgram
    flags: tuple[tuple[int, int], ...]
    _owners: tuple[Any, ...] = field(repr=False)

    def check(self) -> None:
        if (type(self.cfg) is not CFGProgram or self._owners != (id(self.cfg), self.flags)
                or type(self.flags) is not tuple):
            raise RuntimeError("LLVM property plan ownership changed")
        self.cfg.check()


def bind_properties(cfg: CFGProgram) -> PropertyCFG:
    from cutlass._mlir import ir

    _check_computation(cfg)
    flags = []
    with cfg.context, ir.Location.unknown(), ir.raw_values():
        functions = [view.operation for view in cfg.module.body.operations
                     if view.operation.name == "llvm.func"
                     and view.operation.attributes["sym_name"].value == cfg.function_name]
        if len(functions) != 1:
            raise ValueError("Expected the exact owned LLVM helper")
        blocks = tuple(functions[0].regions[0].blocks)
        if len(blocks) != len(cfg.blocks):
            raise ValueError("LLVM property plan lost its original CFG blocks")
        for original, block in zip(blocks, cfg.blocks):
            operations = tuple(view.operation for view in original.operations)[:-1]
            if len(operations) != len(block.instructions):
                raise ValueError("LLVM property plan lost its original instructions")
            for op, instruction in zip(operations, block.instructions):
                if op.name in _OVERFLOW:
                    if instruction.expression.kind != op.name:
                        raise ValueError("LLVM properties refer to another instruction")
                    flags.append((instruction.result, _flags(op)))
        cfg.check()
    flags = tuple(flags)
    result = PropertyCFG(cfg, flags, (id(cfg), flags))
    result.check()
    return result


def _check_overflow(flow: Any, slots: list[Any], flags: int) -> None:
    if flow.kind not in _OVERFLOW or not flags:
        return
    operands = []
    for operand in flow.operands:
        if operand.kind != "argument" or operand.path or operand.operands or operand.attributes:
            raise ValueError("Overflow checks require exact compiler instruction operand slots")
        value = slots[operand.argument]
        if type(value) is not ScalarValue or value.llvm_type != operand.llvm_type:
            raise ValueError("Overflow checks require concrete typed integer operands")
        operands.append(value)
    width = integer_width(flow.llvm_type)
    if flow.kind == "llvm.trunc":
        if len(operands) != 1 or integer_width(operands[0].llvm_type) <= width:
            raise ValueError("Invalid narrowing integer instruction")
        signed, unsigned = operands[0].integer(), operands[0].integer(signed=False)
    else:
        if len(operands) != 2 or any(value.llvm_type != flow.llvm_type for value in operands):
            raise ValueError("Overflow checks require two exact result-width operands")
        a, b = (value.integer() for value in operands)
        u, v = (value.integer(signed=False) for value in operands)
        if flow.kind == "llvm.add":
            signed, unsigned = a + b, u + v
        elif flow.kind == "llvm.sub":
            signed, unsigned = a - b, u - v
        elif flow.kind == "llvm.mul":
            signed, unsigned = a * b, u * v
        else:
            if v >= width:
                raise ValueError("Undefined integer shift in the host computation")
            signed, unsigned = a * 2**v, u * 2**v
    if flags & _NSW and not -(2**(width - 1)) <= signed < 2**(width - 1):
        raise ValueError("Compiler nsw property rejects signed integer overflow")
    if flags & _NUW and not 0 <= unsigned < 2**width:
        raise ValueError("Compiler nuw property rejects unsigned integer overflow")


def _evaluate_cfg(cfg: Any, arguments: tuple[Any, ...], flags: dict[int, int], run_scalar: Any) -> tuple[ScalarValue, ...]:
    if type(arguments) is not tuple or len(arguments) != len(cfg.argument_types):
        raise ValueError("Metadata helper arguments differ from its original signature")
    slots = [None] * len(cfg.types)
    block_index, incoming = 0, arguments
    for _ in cfg.blocks:
        block = cfg.blocks[block_index]
        if len(incoming) != len(block.arguments):
            raise ValueError("Metadata helper edge lost its block arguments")
        for slot, value in zip(block.arguments, incoming):
            _validate(value, cfg.types[slot])
            slots[slot] = value
        for instruction in block.instructions:
            flow = instruction.expression
            if flow.kind == "argument":
                result = slots[flow.argument]
                if flow.path:
                    if type(result) is not MetadataAggregate:
                        raise ValueError("Metadata projection requires its original aggregate")
                    result = dict(result.fields)[flow.path]
            else:
                if flow.kind in _OVERFLOW:
                    _check_overflow(flow, slots, flags[instruction.result])
                numeric = tuple(value if type(value) is ScalarValue and not value.llvm_type.startswith("!llvm.ptr")
                                else None for value in slots)
                result = run_scalar(flow, numeric)
            _validate(result, cfg.types[instruction.result])
            slots[instruction.result] = result
        terminator = block.terminator
        if terminator.kind == "return":
            result = tuple(slots[slot] for slot in terminator.values)
            if (any(type(value) is not ScalarValue for value in result)
                    or tuple(value.llvm_type for value in result) != cfg.result_types):
                raise ValueError("Metadata helper may return only its exact numeric scalar types")
            return result
        selected = 0
        if terminator.kind == "conditional":
            condition = slots[terminator.values[0]]
            if type(condition) is not ScalarValue or condition.llvm_type != "i1":
                raise ValueError("Metadata helper requires a concrete i1 branch")
            selected = 0 if condition.integer(signed=False) else 1
        elif terminator.kind != "jump":
            raise ValueError("Unsupported metadata helper terminator")
        edge = terminator.edges[selected]
        incoming = tuple(slots[slot] for slot in edge.arguments)
        block_index = edge.block
    raise RuntimeError("Acyclic metadata helper did not reach a return")
