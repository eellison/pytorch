from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from torch._inductor.runtime._cudagraph._compiler.accessors import _snapshot
from torch._inductor.runtime._cudagraph._compiler.emitter_v2 import SOURCE_ARGUMENT, _argument_attrs, _op, _walk
from torch._inductor.runtime._cudagraph._compiler.source_dispatch import SourceDispatch, SourceLaunchSite, _pure, check_dispatch_source
from torch._inductor.runtime._cudagraph._compiler.tma_requirements import project_tma_property, TmaDimensionRequirement, TmaStrideRequirement


@dataclass(frozen=True)
class ScalarRequest:
    symbol: str
    role: str
    index: int
    value: Any = field(repr=False, compare=False)
    site: SourceLaunchSite | None = field(default=None, repr=False, compare=False)


def _tma_projection(request: ScalarRequest) -> TmaStrideRequirement | TmaDimensionRequirement | None:
    if request.role not in ("tma_stride", "tma_shape", "tma_dimension_stride"):
        return None
    if request.site is None:
        raise ValueError("TMA helper lacks its exact launch requirement")
    stride = request.role == "tma_stride"
    requirements = request.site.tma_strides if stride else request.site.tma_dimensions
    if not 0 <= request.index < len(requirements):
        raise ValueError("TMA helper lacks its exact launch requirement")
    requirement = requirements[request.index]
    if (type(requirement) is not (TmaStrideRequirement if stride else TmaDimensionRequirement)
            or requirement.tensor != request.value
            or request.role == "tma_dimension_stride" and not requirement.grouped):
        raise ValueError("TMA helper changed its original constructor operand")
    return requirement


def _slice(source: SourceDispatch, request: ScalarRequest) -> tuple[tuple[Any, ...], tuple[str, int, int]]:
    from cutlass._mlir import ir

    if (type(request) is not ScalarRequest or type(request.symbol) is not str
            or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9.]*", request.symbol) is None
            or type(request.role) is not str or not request.role
            or type(request.index) is not int or request.index < 0):
        raise ValueError("Expected a named scalar request with an explicit role and index")
    projection = _tma_projection(request)
    if (not isinstance(request.value, ir.Value)
            or projection is None and str(request.value.type) not in {"i1", "i32", "i64"}):
        raise ValueError("Expected an actual supported scalar SSA output")
    if request.site is None:
        if request.value != source.predicate:
            raise ValueError("A root helper must use the exact admitted dispatch predicate")
        arm = ()
    else:
        if not any(request.site is site for site in source.sites):
            raise ValueError("The requested site is not owned by the admitted source")
        operations = tuple(view.operation for view in request.site.block.operations)
        arm = operations[:operations.index(request.site.launch)]
    available = (*source.prefix, *arm)
    needed, visiting, kernels = set(), set(), {}

    def require(value: Any) -> None:
        if value in source.arguments:
            return
        owner = _op(value.owner)
        if owner not in available:
            raise ValueError("Scalar output or dependency escapes the selected arm and dominating prefix")
        if owner in needed:
            return
        if owner in visiting:
            raise ValueError("Cyclic source scalar dependency")
        _pure(source.module, owner, kernels)
        if kernels and (request.site is None or any(kernel != request.site.kernel for kernel in kernels)):
            raise ValueError("A shared-memory query must identify the selected site's exact kernel")
        visiting.add(owner)
        tree = _walk(owner)
        internal = {result for operation in tree for result in operation.results}
        for operation in tree:
            for operand in operation.operands:
                if operand not in internal:
                    require(operand)
        visiting.remove(owner)
        needed.add(owner)

    require(request.value)
    if request.value in source.arguments:
        origin = ("argument", source.arguments.index(request.value), 0)
    else:
        owner = _op(request.value.owner)
        result = tuple(owner.results).index(request.value)
        origin = (("prefix", source.prefix.index(owner), result) if owner in source.prefix
                  else ("arm", arm.index(owner), result))
    return tuple(operation for operation in available if operation in needed), origin


@dataclass(frozen=True)
class DispatchScalarHelper:
    symbol: str
    role: str
    index: int
    operation: Any = field(repr=False, compare=False)
    source: SourceDispatch = field(repr=False, compare=False)
    site: SourceLaunchSite | None = field(repr=False, compare=False)
    output_value: Any = field(repr=False, compare=False)
    input_origin: tuple[Any, ...] = field(repr=False, compare=False)
    source_ids: tuple[int, ...]
    source_types: tuple[str, ...]
    result_type: str
    output_origin: tuple[str, int, int]
    operations: tuple[Any, ...] = field(repr=False, compare=False)
    _seal: tuple[Any, ...] = field(repr=False)

    @property
    def module(self) -> Any:
        return self.source.module

    @property
    def context(self) -> Any:
        return self.source.context

    def _state(self) -> tuple[Any, ...]:
        return (self.symbol, self.role, self.index, id(self.operation), id(self.source), id(self.site),
                id(self.output_value), id(self.input_origin), self.source_ids, self.source_types,
                self.result_type, self.output_origin, id(self.operations))

    def check(self) -> None:
        if self._state() != self._seal:
            raise RuntimeError("Dispatch scalar helper ownership or source association changed")
        source = self.source
        source.check()
        self._check_body(source)

    def _check_body(self, source: SourceDispatch) -> None:
        from cutlass._mlir import ir

        if self.source is not source or self._state() != self._seal:
            raise RuntimeError("Dispatch scalar helper ownership or source association changed")
        with source.context, ir.raw_values():
            request = ScalarRequest(self.symbol, self.role, self.index, self.output_value, self.site)
            operations, origin = _slice(source, request)
            attrs, ids, _ = _argument_attrs(source.host, len(source.arguments))
            helpers = tuple(view.operation for view in source.module.body.operations
                            if "sym_name" in view.operation.attributes
                            and view.operation.attributes["sym_name"].value == self.symbol)
            if (helpers != (self.operation,) or operations != self.operations or origin != self.output_origin
                    or self.input_origin != source.arguments or ids != self.source_ids
                    or source.source_types != self.source_types
                    or ("i64" if _tma_projection(request) is not None else str(self.output_value.type)) != self.result_type):
                raise RuntimeError("Dispatch scalar helper lost its exact source correspondence")
            helper_attrs, helper_ids, tagged = _argument_attrs(self.operation, len(self.input_origin))
            expected = tuple(ir.DictAttr.get({**{item.name: item.attr for item in attr},
                SOURCE_ARGUMENT: ir.IntegerAttr.get(ir.IntegerType.get_signless(64), index)})
                for attr, index in zip(attrs, self.source_ids))
            if not tagged or helper_ids != self.source_ids or helper_attrs != expected:
                raise RuntimeError("Dispatch scalar helper formal IDs changed")


@dataclass(frozen=True)
class DispatchHelpers:
    source: SourceDispatch = field(repr=False, compare=False)
    helpers: tuple[DispatchScalarHelper, ...]
    _owners: tuple[Any, ...] = field(repr=False, compare=False)

    def check(self) -> None:
        if type(self) is not DispatchHelpers:
            raise TypeError("Expected the exact dispatch helper collection")
        source, helpers, owners = self.source, self.helpers, self._owners
        if (type(source) is not SourceDispatch or type(helpers) is not tuple
                or any(type(helper) is not DispatchScalarHelper for helper in helpers)):
            raise TypeError("Expected exact dispatch source and scalar helpers")
        if (type(owners) is not tuple or len(owners) != 2
                or source is not owners[0] or helpers is not owners[1]):
            raise RuntimeError("Dispatch helper collection ownership changed")
        source.check()
        for helper in helpers:
            if helper.source is not source:
                raise RuntimeError("Dispatch helpers do not share the final source owner")
            DispatchScalarHelper._check_body(helper, source)
        source.check()
        if self.source is not source or self.helpers is not helpers or self._owners is not owners:
            raise RuntimeError("Dispatch helper collection ownership changed")
        for helper in helpers:
            if helper.source is not source or helper._state() != helper._seal:
                raise RuntimeError("Dispatch scalar helper ownership or source association changed")


def emit_dispatch_helpers(source: SourceDispatch, requests: tuple[ScalarRequest, ...]) -> DispatchHelpers:
    """Append arm-local SSA slices and return a fresh admission of the final Module."""
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import func

    if type(source) is not SourceDispatch or type(requests) is not tuple or not requests:
        raise ValueError("Expected an admitted source and a nonempty tuple of scalar requests")
    source.check()
    with source.context, ir.Location.unknown(), ir.raw_values():
        slices = tuple(_slice(source, request) for request in requests)
        original_ops = tuple(view.operation for view in source.module.body.operations)
        names = {op.attributes["sym_name"].value for op in original_ops if "sym_name" in op.attributes}
        if len({request.symbol for request in requests}) != len(requests) or any(request.symbol in names for request in requests):
            raise ValueError("Helper symbols must be new and unique")
        attrs, source_ids, _ = _argument_attrs(source.host, len(source.arguments))
        before = _snapshot(source.module.operation)
        originals = tuple(_snapshot(operation) for operation in original_ops)
        created = []
        try:
            for request, (operations, _) in zip(requests, slices):
                projection = _tma_projection(request)
                result_type = ir.IntegerType.get_signless(64) if projection is not None else request.value.type
                with ir.InsertionPoint(source.module.body):
                    helper = func.FuncOp(request.symbol, ([value.type for value in source.arguments], [result_type]),
                                         visibility="public")
                created.append(helper.operation)
                helper.operation.attributes["no_inline"] = ir.UnitAttr.get()
                helper.arg_attrs = [ir.DictAttr.get({**{item.name: item.attr for item in attr},
                    SOURCE_ARGUMENT: ir.IntegerAttr.get(ir.IntegerType.get_signless(64), index)})
                    for attr, index in zip(attrs, source_ids)]
                block = helper.add_entry_block()
                mapping = dict(zip(source.arguments, block.arguments))
                for operation in operations:
                    cloned = operation.clone(ip=ir.InsertionPoint(block))
                    old_tree, new_tree = _walk(operation), _walk(cloned)
                    if len(old_tree) != len(new_tree):
                        raise RuntimeError("Deep clone changed the source operation tree")
                    for old, new in zip(old_tree, new_tree):
                        if (old.name != new.name or len(old.operands) != len(new.operands)
                                or tuple(value.type for value in old.results) != tuple(value.type for value in new.results)
                                or {key: old.attributes[key] for key in old.attributes}
                                != {key: new.attributes[key] for key in new.attributes}):
                            raise RuntimeError("Deep clone changed typed operation structure")
                        mapping.update(zip(old.results, new.results))
                    for old, new in zip(old_tree, new_tree):
                        for index, operand in enumerate(old.operands):
                            if operand not in mapping:
                                raise ValueError("Cloned dependency escapes the helper formal and result mapping")
                            new.operands[index] = mapping[operand]
                with ir.InsertionPoint(block):
                    value = mapping[request.value]
                    if projection is not None:
                        property = "shape" if request.role == "tma_shape" else "stride"
                        value = project_tma_property(projection, value, property)
                    if value.type != result_type:
                        raise ValueError("Projected helper changed its exact scalar result type")
                    func.ReturnOp([value])
            if not source.module.operation.verify():
                raise RuntimeError("Dispatch helpers failed typed Module verification")
            if (tuple(view.operation for view in source.module.body.operations) != (*original_ops, *created)
                    or tuple(_snapshot(operation) for operation in original_ops) != originals):
                raise RuntimeError("Helper insertion changed an original host or device operation")
            final = check_dispatch_source(source.module, source.function_name, source.metadata)
            helpers = []
            for request, operation, (operations, origin) in zip(requests, created, slices):
                if request.site is None:
                    site = None
                else:
                    matches = tuple(site for site in final.sites if site.launch == request.site.launch)
                    if len(matches) != 1:
                        raise RuntimeError("The original launch did not survive helper insertion uniquely")
                    site = matches[0]
                helper = DispatchScalarHelper(request.symbol, request.role, request.index, operation, final, site,
                    request.value, final.arguments, source_ids, final.source_types,
                    "i64" if _tma_projection(request) is not None else str(request.value.type),
                    origin, operations, ())
                helpers.append(replace(helper, _seal=helper._state()))
            helpers = tuple(helpers)
            result = DispatchHelpers(final, helpers, (final, helpers))
            result.check()
            return result
        except BaseException:
            for operation in reversed(created):
                operation.erase()
            if _snapshot(source.module.operation) != before:
                raise RuntimeError("Failed helper emission did not restore the original Module") from None
            raise
