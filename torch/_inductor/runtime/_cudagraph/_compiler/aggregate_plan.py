from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch._inductor.runtime._cudagraph._compiler.accessors import Component, ComponentMapping


AGGREGATE_PLAN_VERSION = 1


@dataclass(frozen=True)
class AggregateLeaf:
    path: tuple[int, ...]
    llvm_type: str
    component: Component


@dataclass(frozen=True)
class TensorAggregate:
    source_arg_index: int
    llvm_arg_index: int
    llvm_type: str
    leaves: tuple[AggregateLeaf, ...]
    constants: tuple[Component, ...]


def _integer_vector_parts(typ: Any):
    from cutlass._mlir import ir

    if not isinstance(typ, ir.VectorType):
        return None
    element = typ.element_type
    if (len(typ.shape) != 1 or typ.shape[0] <= 0 or any(typ.scalable_dims)
            or not isinstance(element, ir.IntegerType) or not element.is_signless
            or element.width not in (1, 8, 16, 32, 64)):
        raise ValueError("Expected a fixed rank-one vector of supported signless integers")
    return typ.shape[0], element


def _scalar_leaves(typ: Any, path: tuple[int, ...] = (), active: frozenset[Any] = frozenset()) -> tuple[tuple[tuple[int, ...], str], ...]:
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm
    from torch._inductor.runtime._cudagraph._compiler.llvm_types import integer_array_parts

    if isinstance(typ, (llvm.PointerType, ir.IntegerType)):
        return ((path, str(typ)),)
    vector = _integer_vector_parts(typ)
    if vector is not None:
        count, element = vector
        return tuple(((*path, index), str(element)) for index in range(count))
    array = integer_array_parts(typ)
    if array is not None:
        count, element = array
        return tuple(((*path, index), str(element)) for index in range(count))
    if not isinstance(typ, llvm.StructType) or typ.opaque:
        raise ValueError("Tensor aggregate contains an opaque or unsupported component type")
    if typ in active:
        raise ValueError("Recursive by-value tensor aggregates are unsupported")
    return tuple(leaf for index, child in enumerate(typ.body)
                 for leaf in _scalar_leaves(child, (*path, index), active | {typ}))


def _build_formal(source_arg_index: int, llvm_arg_index: int, typ: Any,
                  components: tuple[Component, ...]) -> TensorAggregate:
    """Match typed leaves only; this helper does not confer compiler ownership."""
    typed_leaves = _scalar_leaves(typ)
    expected = dict(typed_leaves)
    assigned, constants = {}, []
    for component in components:
        source = component.source
        if component.spec.source_arg_index != source_arg_index or component.llvm_argument_type != str(typ):
            raise ValueError("Aggregate component belongs to a different original formal or type")
        if source.kind in ("constant", "zero"):
            if (component.spec.property == "pointer" or source.argument is not None
                    or source.path or source.operands or source.attributes):
                raise ValueError("Invalid constant tensor property")
            constants.append(component)
            continue
        if (source.kind != "argument" or source.argument != source_arg_index
                or source.value is not None or source.operands or source.attributes):
            raise ValueError("Aggregate reconstruction requires direct component paths; casts are unsupported")
        path = source.path
        if type(path) is not tuple or any(type(index) is not int or index < 0 for index in path):
            raise ValueError("Invalid tensor aggregate component path")
        if path not in expected:
            raise ValueError("Accessor supplies an extra or non-scalar aggregate component")
        if path in assigned:
            raise ValueError("Multiple accessor sources claim the same aggregate leaf")
        if source.llvm_type != expected[path]:
            raise ValueError("Accessor and actual aggregate leaf types disagree")
        assigned[path] = component
    if set(assigned) != set(expected):
        raise ValueError("Tensor aggregate has missing scalar-leaf sources")
    leaves = tuple(AggregateLeaf(path, leaf_type, assigned[path]) for path, leaf_type in typed_leaves)
    return TensorAggregate(source_arg_index, llvm_arg_index, str(typ), leaves, tuple(constants))


def _derive(mapping: ComponentMapping) -> tuple[TensorAggregate, ...]:
    from torch._inductor.runtime._cudagraph._compiler.accessors import ComponentMapping, _function, _tagged_arguments
    from cutlass._mlir import ir

    if type(mapping) is not ComponentMapping:
        raise TypeError("Expected an accepted compiler-owned ComponentMapping")
    mapping.check()
    groups: dict[int, list[Component]] = {}
    for component in mapping.components:
        groups.setdefault(component.spec.source_arg_index, []).append(component)
    program = mapping.program
    with program.context, ir.raw_values():
        host = _function(program.module, program.function_name, "llvm.func")
        arguments = _tagged_arguments(host)
        if not groups or set(groups) - set(arguments):
            raise ValueError("Tensor components do not identify original LLVM formals")
        result = tuple(_build_formal(source_index, llvm_index, argument.type, tuple(groups[source_index]))
                       for llvm_index, (source_index, argument) in enumerate(arguments.items())
                       if source_index in groups)
    mapping.check()
    return result


def _plan_state(formals: tuple[TensorAggregate, ...]) -> tuple[Any, ...]:
    return tuple((formal.source_arg_index, formal.llvm_arg_index, formal.llvm_type,
                  tuple((leaf.path, leaf.llvm_type, id(leaf.component)) for leaf in formal.leaves),
                  tuple(id(component) for component in formal.constants)) for formal in formals)


@dataclass(frozen=True)
class AggregatePlan:
    mapping: ComponentMapping = field(repr=False, compare=False)
    formals: tuple[TensorAggregate, ...]
    _owners: tuple[Any, ...] = field(repr=False, compare=False)
    _state: tuple[Any, ...] = field(repr=False)

    def check(self) -> None:
        if (self.mapping is not self._owners[0] or self.formals is not self._owners[1]
                or _plan_state(self.formals) != self._state):
            raise RuntimeError("Tensor aggregate plan ownership or leaf associations changed")
        if _derive(self.mapping) != self.formals:
            raise RuntimeError("Tensor aggregate plan differs from its compiler component mapping")


def build_aggregate_plan(mapping: ComponentMapping) -> AggregatePlan:
    """Cover actual scalar leaves without computing offsets or packing bytes."""
    formals = _derive(mapping)
    return AggregatePlan(mapping, formals, (mapping, formals), _plan_state(formals))
