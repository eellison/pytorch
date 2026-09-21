from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from torch._inductor.runtime._cudagraph._compiler.argument_flow import ValueFlow
from torch._inductor.runtime._cudagraph._compiler.cfg_values import CFGProgram
from torch._inductor.runtime._cudagraph._compiler.metadata_regions import _NUMERIC, _check_computation
from torch._inductor.runtime._cudagraph._compiler.values import AttributeDecoding, ScalarValue, decode_attributes


def _flow_state(flow: ValueFlow) -> tuple[Any, ...]:
    if (type(flow) is not ValueFlow or type(flow.kind) is not str or type(flow.llvm_type) is not str
            or flow.argument is not None and type(flow.argument) is not int
            or type(flow.path) is not tuple or any(type(index) is not int for index in flow.path)
            or flow.value is not None and type(flow.value) is not str
            or type(flow.operands) is not tuple or any(type(item) is not ValueFlow for item in flow.operands)
            or type(flow.attributes) is not tuple
            or any(type(pair) is not tuple or len(pair) != 2 or any(type(item) is not str for item in pair)
                   for pair in flow.attributes)):
        raise ValueError("Unsupported or changed immutable scalar flow")
    return (id(flow), flow.kind, flow.llvm_type, flow.argument, flow.path, flow.value,
            tuple(id(item) for item in flow.operands), flow.attributes)


@dataclass(frozen=True)
class DecodedEntry:
    flow: ValueFlow
    attributes: AttributeDecoding


@dataclass(frozen=True)
class DecodedValues:
    cfg: CFGProgram = field(repr=False)
    context: Any = field(repr=False)
    roots: tuple[ValueFlow, ...] = field(repr=False)
    entries: tuple[DecodedEntry, ...]
    by_id: Any = field(repr=False)
    _owners: tuple[Any, ...] = field(repr=False)
    _seal: tuple[Any, ...] = field(repr=False)

    def _state(self) -> tuple[Any, ...]:
        entries = []
        for entry in self.entries:
            if type(entry) is not DecodedEntry or type(entry.attributes) is not AttributeDecoding:
                raise RuntimeError("Decoded scalar entry changed its typed record")
            source = _flow_state(entry.flow)
            constant = entry.attributes.constant
            predicate = entry.attributes.predicate
            if (constant is not None and type(constant) is not ScalarValue
                    or predicate is not None and type(predicate) is not int
                    or constant is not None and predicate is not None):
                raise RuntimeError("Decoded scalar payload has an unsupported type or incompatible fields")
            if (constant is not None and entry.flow.kind != "constant"
                    or predicate is not None and entry.flow.kind != "llvm.icmp"
                    or entry.flow.kind == "constant" and constant is None
                    or entry.flow.kind == "llvm.icmp" and predicate is None):
                raise RuntimeError("Decoded scalar payload does not match its exact instruction kind")
            if constant is not None and (type(constant.llvm_type) is not str or type(constant.size) is not int
                                         or type(constant.value) not in (int, float)
                                         or constant.llvm_type != entry.flow.llvm_type or constant.llvm_type not in _NUMERIC):
                raise RuntimeError("Decoded constant changed its primitive scalar fields")
            state = None if constant is None else (id(constant), constant.llvm_type, constant.size, constant.data())
            entries.append((id(entry), source, id(entry.attributes), state, entry.attributes.predicate))
        return (tuple(id(root) for root in self.roots), tuple(entries),
                tuple((key, id(value)) for key, value in self.by_id.items()))

    def check(self, context: Any) -> None:
        owned = self.cfg, self.context, self.roots, self.entries, self.by_id
        if (len(owned) != len(self._owners) or any(value is not owner for value, owner in zip(owned, self._owners))
                or type(self.cfg) is not CFGProgram or context is not self.context or self.cfg.context is not self.context
                or type(self.roots) is not tuple or type(self.entries) is not tuple
                or type(self.by_id) is not MappingProxyType or self._state() != self._seal):
            raise RuntimeError("Decoded scalar flow, payload or compiler ownership changed")
        self.cfg.check()
        actual = tuple(instruction.expression for block in self.cfg.blocks for instruction in block.instructions)
        if len(actual) != len(self.roots) or any(root is not expected for root, expected in zip(actual, self.roots)):
            raise RuntimeError("Decoded scalar instructions lost the exact original CFG")

    def lookup(self, flow: ValueFlow) -> AttributeDecoding:
        entry = self.by_id.get(id(flow))
        if entry is None or entry.flow is not flow:
            raise ValueError("Foreign scalar flow has no decoding in this CFG")
        return entry.attributes


def prepare_decodings(cfg: CFGProgram) -> DecodedValues:
    from cutlass._mlir import ir

    _check_computation(cfg)
    roots = tuple(instruction.expression for block in cfg.blocks for instruction in block.instructions)
    entries, visited, active = [], set(), set()

    def visit(flow):
        _flow_state(flow)
        if id(flow) in active:
            raise ValueError("Cyclic scalar expression has no supported decoding")
        if id(flow) in visited:
            return
        active.add(id(flow))
        attributes = decode_attributes(flow)
        for operand in flow.operands:
            visit(operand)
        active.remove(id(flow))
        visited.add(id(flow))
        entries.append(DecodedEntry(flow, attributes))

    with cfg.context, ir.Location.unknown(), ir.raw_values():
        for root in roots:
            visit(root)
    entries = tuple(entries)
    lookup = MappingProxyType({id(entry.flow): entry for entry in entries})
    owners = cfg, cfg.context, roots, entries, lookup
    result = DecodedValues(*owners, owners, ())
    object.__setattr__(result, "_seal", result._state())
    result.check(cfg.context)
    return result
