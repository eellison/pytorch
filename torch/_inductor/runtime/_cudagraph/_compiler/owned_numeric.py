import struct
from typing import Any, NamedTuple

from torch._inductor.runtime._cudagraph._compiler.cfg_values import (
    Block,
    Edge,
    Instruction,
    Terminator,
    ValueType,
)
from torch._inductor.runtime._cudagraph._compiler.decoded_values import (
    _flow_state,
    DecodedValues,
)
from torch._inductor.runtime._cudagraph._compiler.metadata_regions import (
    _check_computation,
    _NUMERIC,
)
from torch._inductor.runtime._cudagraph._compiler.overflow_properties import (
    _evaluate_cfg,
    _NSW,
    _NUW,
    _OVERFLOW,
    PropertyCFG,
)
from torch._inductor.runtime._cudagraph._compiler.values import (
    _evaluate,
    AttributeDecoding,
    comparison_predicates,
    ScalarValue,
)


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
        if (
            type(self) is not OwnedNumeric
            or type(self._cfg) is not _CFG
            or type(self._seal) is not tuple
            or len(owned) != len(self._seal)
            or any(value is not original for value, original in zip(owned, self._seal))
        ):
            raise RuntimeError("Owned numeric payload or source ordering changed")


def freeze_numeric(
    properties: PropertyCFG, decoded: DecodedValues, source_order: tuple[int, ...]
) -> OwnedNumeric:
    if (
        type(properties) is not PropertyCFG
        or type(decoded) is not DecodedValues
        or decoded.cfg is not properties.cfg
    ):
        raise ValueError(
            "Numeric preparation requires one exact property and decoding CFG"
        )
    properties.check()
    cfg = properties.cfg
    decoded.check(cfg.context)
    _check_computation(cfg)
    if (
        type(source_order) is not tuple
        or len(source_order) != len(cfg.argument_types)
        or any(type(index) is not int or index < 0 for index in source_order)
        or len(set(source_order)) != len(source_order)
    ):
        raise ValueError(
            "Numeric preparation requires exact original source formal indices"
        )
    for types in (cfg.argument_types, cfg.result_types):
        if type(types) is not tuple or any(type(value) is not str for value in types):
            raise ValueError(
                "Numeric signatures must have exact primitive type descriptions"
            )
    flags = properties.flags
    if type(flags) is not tuple or any(
        type(pair) is not tuple
        or len(pair) != 2
        or any(type(value) is not int for value in pair)
        or pair[0] < 0
        or pair[1] < 0
        or pair[1] & ~(_NSW | _NUW)
        for pair in flags
    ):
        raise ValueError("Unsupported owned arithmetic property record")
    expected = tuple(
        item.result
        for block in cfg.blocks
        for item in block.instructions
        if item.expression.kind in _OVERFLOW
    )
    if tuple(slot for slot, _ in flags) != expected:
        raise ValueError(
            "Owned arithmetic properties lack complete instruction coverage"
        )
    predicates = comparison_predicates()
    if (
        len(predicates) != 10
        or any(type(value) is not int for value in predicates)
        or len(set(predicates)) != 10
    ):
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
        bits = (
            None
            if constant is None
            else (constant.llvm_type, constant.data(), constant.size)
        )
        result = _Flow(
            flow.kind,
            flow.llvm_type,
            flow.argument,
            tuple(index for index in flow.path),
            tuple(flow_copy(item) for item in flow.operands),
            tuple((name, value) for name, value in flow.attributes),
            bits,
            decoding.predicate,
        )
        active.remove(id(flow))
        copied[id(flow)] = result
        return result

    def slots(indices):
        if type(indices) is not tuple or any(
            type(index) is not int or not 0 <= index < len(cfg.types)
            for index in indices
        ):
            raise ValueError("Owned CFG requires exact in-range scalar slots")
        return tuple(index for index in indices)

    types = []
    for typ in cfg.types:
        if type(typ) is not ValueType or type(typ.llvm_type) is not str:
            raise ValueError("Unsupported owned CFG type record")
        leaves = None
        if typ.leaves is not None:
            if type(typ.leaves) is not tuple or any(
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not tuple
                or any(type(index) is not int or index < 0 for index in pair[0])
                or type(pair[1]) is not str
                for pair in typ.leaves
            ):
                raise ValueError("Unsupported owned aggregate leaf type record")
            leaves = tuple(
                (tuple(index for index in path), name) for path, name in typ.leaves
            )
        types.append(_Type(typ.llvm_type, leaves))
    blocks = []
    for block in cfg.blocks:
        if type(block) is not Block or type(block.instructions) is not tuple:
            raise ValueError("Unsupported owned CFG block record")
        instructions = []
        for instruction in block.instructions:
            if type(instruction) is not Instruction:
                raise ValueError("Unsupported owned CFG instruction record")
            (result,) = slots((instruction.result,))
            instructions.append(_Instruction(result, flow_copy(instruction.expression)))
        end = block.terminator
        if (
            type(end) is not Terminator
            or type(end.kind) is not str
            or end.kind not in {"return", "jump", "conditional"}
        ):
            raise ValueError("Unsupported owned CFG terminator")
        if type(end.edges) is not tuple:
            raise ValueError("Unsupported owned CFG successors")
        edges = []
        for edge in end.edges:
            if (
                type(edge) is not Edge
                or type(edge.block) is not int
                or not 0 <= edge.block < len(cfg.blocks)
            ):
                raise ValueError("Unsupported owned CFG successor")
            edges.append(_Edge(edge.block, slots(edge.arguments)))
        blocks.append(
            _Block(
                slots(block.arguments),
                tuple(instructions),
                _Terminator(end.kind, slots(end.values), tuple(edges)),
            )
        )
    payload = _CFG(
        tuple(name for name in cfg.argument_types),
        tuple(name for name in cfg.result_types),
        tuple(types),
        tuple(blocks),
    )
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
        number = (
            struct.unpack("<f" if llvm_type == "f32" else "<d", data)[0]
            if llvm_type in {"f32", "f64"}
            else int.from_bytes(data, "little")
        )
        value = ScalarValue(llvm_type, number, size)
    return AttributeDecoding(value, flow.predicate)


def evaluate_owned(
    program: OwnedNumeric, arguments: tuple[Any, ...]
) -> tuple[ScalarValue, ...]:
    if type(program) is not OwnedNumeric:
        raise TypeError("Expected a factory-owned numeric program")
    program.check()
    return _evaluate_cfg(
        program._cfg,
        arguments,
        dict(program._flags),
        lambda flow, numeric: _evaluate(
            flow, numeric, _attributes, program._predicates
        ),
    )


_NUMERIC_RECORDS = (_Flow, _Type, _Instruction, _Edge, _Terminator, _Block, _CFG)


def _matches(value, annotation):
    from types import UnionType
    from typing import get_args, get_origin, Union

    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        return True
    if origin in (UnionType, Union):
        return any(_matches(value, item) for item in args)
    if origin is tuple or annotation is tuple:
        if type(value) is not tuple:
            return False
        if annotation is tuple:
            return True
        if len(args) == 2 and args[1] is Ellipsis:
            return all(_matches(item, args[0]) for item in value)
        return len(value) == len(args) and all(
            _matches(item, typ) for item, typ in zip(value, args)
        )
    return type(value) is annotation


def _dump_records(value, kind, records):
    import json

    nodes, indices, active = [], {}, set()
    stack = [(value, False)]

    def reference(item):
        if item is None or type(item) in (str, int, bool):
            return item
        if type(item) is bytes:
            return ["bytes", item.hex()]
        return ["ref", indices[id(item)]]

    while stack:
        item, ready = stack.pop()
        if item is None or type(item) in (str, int, bool, bytes) or id(item) in indices:
            continue
        if type(item) is OwnedNumeric:
            item.check()
            children = (item._cfg, item._flags, item._predicates, item._source_order)
            tag = "OwnedNumeric"
        elif type(item) is tuple or type(item) in records:
            children = tuple(item)
            tag = "tuple" if type(item) is tuple else type(item).__name__
        else:
            raise TypeError("Unsupported typed payload record")
        if ready:
            active.remove(id(item))
            indices[id(item)] = len(nodes)
            nodes.append([tag, [reference(child) for child in children]])
        else:
            if id(item) in active:
                raise ValueError("Cyclic typed payload")
            active.add(id(item))
            stack.append((item, True))
            stack.extend((child, False) for child in reversed(children))
    return json.dumps(
        {"version": 1, "kind": kind, "nodes": nodes, "root": reference(value)},
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _load_records(data, kind, records):
    import json
    from typing import get_type_hints

    if type(data) is not bytes:
        raise TypeError("Typed payload transport requires bytes")
    payload = json.loads(data)
    if (
        type(payload) is not dict
        or set(payload) != {"version", "kind", "nodes", "root"}
        or type(payload["version"]) is not int
        or payload["version"] != 1
        or payload["kind"] != kind
        or type(payload["nodes"]) is not list
    ):
        raise ValueError("Unsupported typed payload format or version")
    classes = {record.__name__: record for record in records}
    schemas = {
        name: tuple(get_type_hints(record).values()) for name, record in classes.items()
    }
    values = []

    def reference(item):
        if item is None or type(item) in (str, int, bool):
            return item
        if type(item) is list and len(item) == 2:
            tag, value = item
            if tag == "bytes" and type(value) is str:
                return bytes.fromhex(value)
            if tag == "ref" and type(value) is int and 0 <= value < len(values):
                return values[value]
        raise ValueError("Invalid or forward typed payload reference")

    for row in payload["nodes"]:
        if (
            type(row) is not list
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not list
        ):
            raise ValueError("Invalid typed payload record")
        tag, encoded = row
        fields = tuple(reference(item) for item in encoded)
        if tag == "tuple":
            result = fields
        elif tag == "OwnedNumeric":
            if len(fields) != 4:
                raise ValueError("Invalid owned numeric record")
            result = object.__new__(OwnedNumeric)
            for name, value in zip(
                ("_cfg", "_flags", "_predicates", "_source_order"), fields
            ):
                object.__setattr__(result, name, value)
            object.__setattr__(result, "_seal", fields)
            _validate_numeric(result)
        elif tag in classes:
            schema = schemas[tag]
            if len(fields) != len(schema) or not all(
                _matches(item, typ) for item, typ in zip(fields, schema)
            ):
                raise ValueError(f"Invalid {tag} field types")
            result = classes[tag](*fields)
        else:
            raise ValueError(f"Unknown typed payload record: {tag}")
        values.append(result)
    return reference(payload["root"])


def _validate_numeric(program):
    from .cfg_values import _BINARY, _CASTS

    program.check()
    cfg, flags, predicates, order = (
        program._cfg,
        program._flags,
        program._predicates,
        program._source_order,
    )
    if (
        not _matches(flags, tuple[tuple[int, int], ...])
        or not _matches(predicates, tuple[int, ...])
        or len(predicates) != 10
        or len(set(predicates)) != 10
        or not _matches(order, tuple[int, ...])
        or len(order) != len(cfg.argument_types)
        or any(index < 0 for index in order)
        or len(set(order)) != len(order)
        or not cfg.blocks
    ):
        raise ValueError("Invalid numeric source ordering or arithmetic properties")
    expected = tuple(
        item.result
        for block in cfg.blocks
        for item in block.instructions
        if item.expression.kind in _OVERFLOW
    )
    if tuple(slot for slot, _ in flags) != expected or any(
        bits < 0 or bits & ~(_NSW | _NUW) for _, bits in flags
    ):
        raise ValueError("Arithmetic flags lack exact instruction coverage")
    count = len(cfg.types)
    defined, successors, incoming = set(), [], [set() for _ in cfg.blocks]
    for block_index, block in enumerate(cfg.blocks):
        slots = (*block.arguments, *(item.result for item in block.instructions))
        if any(not 0 <= slot < count or slot in defined for slot in slots) or len(
            set(slots)
        ) != len(slots):
            raise ValueError("Invalid or duplicate numeric slot")
        defined.update(slots)
        end = block.terminator
        if (
            end.kind not in {"return", "jump", "conditional"}
            or len(end.edges) != {"return": 0, "jump": 1, "conditional": 2}[end.kind]
            or end.kind == "jump"
            and end.values
            or end.kind == "conditional"
            and len(end.values) != 1
        ):
            raise ValueError("Invalid numeric terminator")
        targets = set()
        for edge in end.edges:
            if not 0 <= edge.block < len(cfg.blocks) or any(
                not 0 <= slot < count for slot in edge.arguments
            ):
                raise ValueError("Invalid numeric successor")
            target = cfg.blocks[edge.block]
            if tuple(cfg.types[slot] for slot in edge.arguments) != tuple(
                cfg.types[slot] for slot in target.arguments
            ):
                raise ValueError("Numeric successor argument types differ")
            targets.add(edge.block)
            incoming[edge.block].add(block_index)
        successors.append(targets)
    if (
        defined != set(range(count))
        or incoming[0]
        or tuple(cfg.types[slot].llvm_type for slot in cfg.blocks[0].arguments)
        != cfg.argument_types
    ):
        raise ValueError("Numeric signature or slot coverage differs")
    visited, available = set(), {}
    pending = [0]
    while pending:
        block_index = pending.pop()
        if block_index in visited:
            continue
        block = cfg.blocks[block_index]
        ready = (
            set.intersection(*(available[index] for index in incoming[block_index]))
            if incoming[block_index]
            else set()
        )
        ready.update(block.arguments)
        for instruction in block.instructions:
            root = instruction.expression
            stack, seen = [root], set()
            while stack:
                flow = stack.pop()
                if type(flow) is not _Flow:
                    raise ValueError("Invalid numeric operand record")
                if id(flow) in seen:
                    continue
                seen.add(id(flow))
                if flow.llvm_type not in _NUMERIC or any(
                    type(child) is not _Flow for child in flow.operands
                ):
                    raise ValueError("Unsupported numeric operand type")
                kind, operands = flow.kind, flow.operands
                arity = (
                    1
                    if kind in _CASTS
                    else 2
                    if kind in _BINARY | {"llvm.icmp"}
                    else 3
                    if kind == "llvm.select"
                    else 0
                )
                if (
                    kind
                    not in _BINARY
                    | _CASTS
                    | {"argument", "constant", "zero", "llvm.icmp", "llvm.select"}
                    or len(operands) != arity
                ):
                    raise ValueError("Unsupported numeric instruction or arity")
                if kind == "argument":
                    if flow.argument not in ready:
                        raise ValueError(
                            "Numeric argument is not defined on every incoming path"
                        )
                    typ = cfg.types[flow.argument]
                    if (
                        (not flow.path and typ.llvm_type != flow.llvm_type)
                        or flow.path
                        and (
                            typ.leaves is None
                            or (flow.path, flow.llvm_type) not in typ.leaves
                        )
                    ):
                        raise ValueError("Numeric argument projection type differs")
                elif flow.argument is not None or flow.path:
                    raise ValueError("Only numeric argument leaves have source slots")
                if (kind == "constant") != (flow.constant is not None):
                    raise ValueError("Numeric constant coverage differs")
                if flow.constant is not None:
                    typ, data, size = flow.constant
                    if (
                        typ != flow.llvm_type
                        or size != max(1, int(typ[1:]) // 8)
                        or len(data) != size
                    ):
                        raise ValueError("Numeric constant bytes or width differ")
                if (
                    (kind == "llvm.icmp") != (flow.predicate is not None)
                    or flow.predicate is not None
                    and flow.predicate not in predicates
                ):
                    raise ValueError("Numeric predicate coverage differs")
                if len({name for name, _ in flow.attributes}) != len(flow.attributes):
                    raise ValueError("Duplicate numeric attribute")
                stack.extend(operands)
            if cfg.types[instruction.result].llvm_type != root.llvm_type:
                raise ValueError("Numeric instruction result type differs")
            ready.add(instruction.result)
        end = block.terminator
        if any(
            slot not in ready
            for slot in (
                *end.values,
                *(slot for edge in end.edges for slot in edge.arguments),
            )
        ):
            raise ValueError("Numeric terminator uses an undefined slot")
        result_types = tuple(cfg.types[slot].llvm_type for slot in end.values)
        if (
            end.kind == "return"
            and result_types != cfg.result_types
            or end.kind == "conditional"
            and result_types != ("i1",)
        ):
            raise ValueError("Numeric terminator result type differs")
        visited.add(block_index)
        available[block_index] = ready
        pending.extend(
            target for target in successors[block_index] if incoming[target] <= visited
        )
    if len(visited) != len(cfg.blocks):
        raise ValueError("Cyclic or unreachable numeric blocks")


def dump_numeric(program: OwnedNumeric) -> bytes:
    if type(program) is not OwnedNumeric:
        raise TypeError("Expected an owned numeric program")
    _validate_numeric(program)
    return _dump_records(program, "cute.numeric", _NUMERIC_RECORDS)


def load_numeric(data: bytes) -> OwnedNumeric:
    result = _load_records(data, "cute.numeric", _NUMERIC_RECORDS)
    if type(result) is not OwnedNumeric:
        raise ValueError("Expected an owned numeric root")
    return result
