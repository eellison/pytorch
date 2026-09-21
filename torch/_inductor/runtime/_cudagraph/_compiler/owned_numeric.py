import struct
from typing import Any, NamedTuple

from torch._inductor.runtime._cudagraph._compiler.cfg_values import Block, Edge, Instruction, Terminator, ValueType
from torch._inductor.runtime._cudagraph._compiler.decoded_values import DecodedValues, _flow_state
from torch._inductor.runtime._cudagraph._compiler.metadata_regions import _check_computation
from torch._inductor.runtime._cudagraph._compiler.overflow_properties import PropertyCFG, _evaluate_cfg, _NSW, _NUW, _OVERFLOW
from torch._inductor.runtime._cudagraph._compiler.values import AttributeDecoding, ScalarValue, _evaluate, comparison_predicates


class _Flow(NamedTuple):
    kind: str
    llvm_type: str
    argument: int | None
    path: tuple[int, ...]
    operands: tuple[Any, ...]
    attributes: tuple[tuple[str, str], ...]
    constant: tuple[str, bytes, int] | None
    predicate: int | None


class _Type(NamedTuple):
    llvm_type: str
    leaves: tuple[tuple[tuple[int, ...], str], ...] | None


class _Instruction(NamedTuple):
    result: int
    expression: _Flow


class _Edge(NamedTuple):
    block: int
    arguments: tuple[int, ...]


class _Terminator(NamedTuple):
    kind: str
    values: tuple[int, ...]
    edges: tuple[_Edge, ...]


class _Block(NamedTuple):
    arguments: tuple[int, ...]
    instructions: tuple[_Instruction, ...]
    terminator: _Terminator


class _CFG(NamedTuple):
    argument_types: tuple[str, ...]
    result_types: tuple[str, ...]
    types: tuple[_Type, ...]
    blocks: tuple[_Block, ...]


class OwnedNumeric:
    __slots__ = ("_cfg", "_flags", "_predicates", "_source_order", "_seal")

    def __new__(cls):
        raise TypeError("OwnedNumeric must be created by freeze_numeric")

    def __setattr__(self, name, value):
        raise AttributeError("An owned numeric program is immutable")

    @property
    def source_order(self) -> tuple[int, ...]:
        return self._source_order

    @property
    def argument_types(self) -> tuple[str, ...]:
        return self._cfg.argument_types

    @property
    def result_types(self) -> tuple[str, ...]:
        return self._cfg.result_types

    def check(self) -> None:
        owned = self._cfg, self._flags, self._predicates, self._source_order
        if (type(self) is not OwnedNumeric or type(self._cfg) is not _CFG
                or type(self._seal) is not tuple or len(owned) != len(self._seal)
                or any(value is not original for value, original in zip(owned, self._seal))):
            raise RuntimeError("Owned numeric payload or source ordering changed")


def freeze_numeric(properties: PropertyCFG, decoded: DecodedValues, source_order: tuple[int, ...]) -> OwnedNumeric:
    if (type(properties) is not PropertyCFG or type(decoded) is not DecodedValues
            or decoded.cfg is not properties.cfg):
        raise ValueError("Numeric preparation requires one exact property and decoding CFG")
    properties.check()
    cfg = properties.cfg
    decoded.check(cfg.context)
    _check_computation(cfg)
    if (type(source_order) is not tuple or len(source_order) != len(cfg.argument_types)
            or any(type(index) is not int or index < 0 for index in source_order)
            or len(set(source_order)) != len(source_order)):
        raise ValueError("Numeric preparation requires exact original source formal indices")
    for types in (cfg.argument_types, cfg.result_types):
        if type(types) is not tuple or any(type(value) is not str for value in types):
            raise ValueError("Numeric signatures must have exact primitive type descriptions")
    flags = properties.flags
    if (type(flags) is not tuple or any(type(pair) is not tuple or len(pair) != 2
            or any(type(value) is not int for value in pair) or pair[0] < 0
            or pair[1] < 0 or pair[1] & ~(_NSW | _NUW) for pair in flags)):
        raise ValueError("Unsupported owned arithmetic property record")
    expected = tuple(item.result for block in cfg.blocks for item in block.instructions
                     if item.expression.kind in _OVERFLOW)
    if tuple(slot for slot, _ in flags) != expected:
        raise ValueError("Owned arithmetic properties lack complete instruction coverage")
    predicates = comparison_predicates()
    if len(predicates) != 10 or any(type(value) is not int for value in predicates) or len(set(predicates)) != 10:
        raise ValueError("Unsupported typed comparison predicate identities")
    copied, active = {}, set()

    def flow_copy(flow):
        _flow_state(flow)
        if id(flow) in active:
            raise ValueError("Cyclic numeric expressions cannot be copied")
        if id(flow) in copied:
            return copied[id(flow)]
        active.add(id(flow))
        decoding = decoded.lookup(flow)
        constant = decoding.constant
        bits = None if constant is None else (constant.llvm_type, constant.data(), constant.size)
        result = _Flow(flow.kind, flow.llvm_type, flow.argument, tuple(index for index in flow.path),
                       tuple(flow_copy(item) for item in flow.operands),
                       tuple((name, value) for name, value in flow.attributes), bits, decoding.predicate)
        active.remove(id(flow))
        copied[id(flow)] = result
        return result

    def slots(indices):
        if type(indices) is not tuple or any(type(index) is not int or not 0 <= index < len(cfg.types) for index in indices):
            raise ValueError("Owned CFG requires exact in-range scalar slots")
        return tuple(index for index in indices)

    types = []
    for typ in cfg.types:
        if type(typ) is not ValueType or type(typ.llvm_type) is not str:
            raise ValueError("Unsupported owned CFG type record")
        leaves = None
        if typ.leaves is not None:
            if (type(typ.leaves) is not tuple or any(type(pair) is not tuple or len(pair) != 2
                    or type(pair[0]) is not tuple or any(type(index) is not int or index < 0 for index in pair[0])
                    or type(pair[1]) is not str for pair in typ.leaves)):
                raise ValueError("Unsupported owned aggregate leaf type record")
            leaves = tuple((tuple(index for index in path), name) for path, name in typ.leaves)
        types.append(_Type(typ.llvm_type, leaves))
    blocks = []
    for block in cfg.blocks:
        if type(block) is not Block or type(block.instructions) is not tuple:
            raise ValueError("Unsupported owned CFG block record")
        instructions = []
        for instruction in block.instructions:
            if type(instruction) is not Instruction:
                raise ValueError("Unsupported owned CFG instruction record")
            result, = slots((instruction.result,))
            instructions.append(_Instruction(result, flow_copy(instruction.expression)))
        end = block.terminator
        if type(end) is not Terminator or type(end.kind) is not str or end.kind not in {"return", "jump", "conditional"}:
            raise ValueError("Unsupported owned CFG terminator")
        if type(end.edges) is not tuple:
            raise ValueError("Unsupported owned CFG successors")
        edges = []
        for edge in end.edges:
            if type(edge) is not Edge or type(edge.block) is not int or not 0 <= edge.block < len(cfg.blocks):
                raise ValueError("Unsupported owned CFG successor")
            edges.append(_Edge(edge.block, slots(edge.arguments)))
        blocks.append(_Block(slots(block.arguments), tuple(instructions),
                             _Terminator(end.kind, slots(end.values), tuple(edges))))
    payload = _CFG(tuple(name for name in cfg.argument_types), tuple(name for name in cfg.result_types),
                   tuple(types), tuple(blocks))
    flags = tuple((slot, bits) for slot, bits in flags)
    order = tuple(index for index in source_order)
    properties.check()
    decoded.check(cfg.context)
    result = object.__new__(OwnedNumeric)
    object.__setattr__(result, "_cfg", payload)
    object.__setattr__(result, "_flags", flags)
    object.__setattr__(result, "_predicates", predicates)
    object.__setattr__(result, "_source_order", order)
    object.__setattr__(result, "_seal", (payload, flags, predicates, order))
    result.check()
    return result


def _attributes(flow: _Flow) -> AttributeDecoding:
    value = None
    if flow.constant is not None:
        llvm_type, data, size = flow.constant
        number = struct.unpack("<f" if llvm_type == "f32" else "<d", data)[0] if llvm_type in {"f32", "f64"} else int.from_bytes(data, "little")
        value = ScalarValue(llvm_type, number, size)
    return AttributeDecoding(value, flow.predicate)


def evaluate_owned(program: OwnedNumeric, arguments: tuple[Any, ...]) -> tuple[ScalarValue, ...]:
    if type(program) is not OwnedNumeric:
        raise TypeError("Expected a factory-owned numeric program")
    program.check()
    return _evaluate_cfg(program._cfg, arguments, dict(program._flags),
                         lambda flow, numeric: _evaluate(flow, numeric, _attributes, program._predicates))
