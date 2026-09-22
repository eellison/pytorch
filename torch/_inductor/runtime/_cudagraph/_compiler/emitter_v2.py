from __future__ import annotations

import io
import re
from dataclasses import dataclass, field, replace
from typing import Any


SOURCE_ARGUMENT = "cudagraph.source_arg"
_PROJECTIONS = {
    "cute.get_iter", "cute.get_layout", "cute.get_shape", "cute.get_stride",
    "cute.get_leaves", "cute.get_scalars", "cute.make_int_tuple", "cute.make_tile",
    "cute.ceil_div", "cute.to_int_tuple", "cute.tuple_add", "cute.tuple_mul",
    "cute.tuple_div", "cute.tuple_mod",
}
_METADATA_OPS = {
    "cute.static": ("StaticOp", 0),
    "cute.make_shape": ("MakeShapeOp", None),
    "cute.make_stride": ("MakeStrideOp", None),
    "cute.make_ordered_layout": ("MakeOrderedLayoutOp", None),
    "cute.make_composed_layout": ("MakeComposedLayoutOp", 3),
    "cute.tile_to_shape": ("TileToShapeOp", 3),
    "cute.raked_product": ("RakedProductOp", 2),
    "cute.right_inverse": ("RightInverseOp", 1),
    "cute.composition": ("CompositionOp", 2),
    "cute.tuple.product_each": ("TupleProductEachOp", 1),
    "cute.make_atom": ("MakeAtomOp", None),
    "cute.make_tiled_copy": ("MakeTiledCopyOp", 1),
    "cute.make_tiled_mma": ("MakeTiledMmaOp", 1),
    "cute.composed_get_inner": ("ComposedGetInnerOp", 1),
    "cute.composed_get_offset": ("ComposedGetOffsetOp", 1),
    "cute.composed_get_outer": ("ComposedGetOuterOp", 1),
}
_TYPED_HOST_OPS = {
    "cute.coalesce": ("cute", "CoalesceOp", (1, 2), 1, (), ()),
    "cute.cosize": ("cute", "CosizeOp", (1,), 1, (), ("mode",)),
    "cute.deref_arith_tuple_iter": ("cute", "DereferenceArithTupleIteratorOp", (1,), 1, (), ()),
    "cute.dice": ("cute", "DiceOp", (1,), 1, ("coord",), ()),
    "cute.get": ("cute", "GetOp", (1,), 1, (), ("mode",)),
    "cute.make_coord": ("cute", "MakeCoordOp", None, 1, (), ()),
    "cute.make_identity_layout": ("cute", "MakeIdentityLayoutOp", (1,), 1, (), ()),
    "cute.mma.make_fragment": ("cute", "MmaMakeFragmentOp", (2,), 1, ("operand_id",), ()),
    "cute.recast_iter": ("cute", "RecastIterOp", (1,), 1, (), ()),
    "cute.recast_layout": ("cute", "RecastLayoutOp", (1,), 1, ("new_type_bits", "old_type_bits"), ()),
    "cute.size": ("cute", "SizeOp", (1,), 1, (), ("mode",)),
    "cute.slice": ("cute", "SliceOp", (2,), 1, (), ()),
    "cute.tiled.mma.partition_shape": ("cute", "TiledMmaPartitionShapeOp", (2,), 1, ("operand_id",), ()),
    "cute.tiled_divide": ("cute", "TiledDivideOp", (2,), 1, (), ()),
    "cute.tuple_sub": ("cute", "TupleSubOp", (2,), 1, (), ()),
    "cute_nvgpu.atom.make_non_exec_tiled_tma_load": (
        "cute_nvgpu", "AtomCopyMakeNonExecTiledTmaLoadOp", (3,), 2, ("kind",), ("num_multicast", "tma_format")),
    "cute_nvgpu.atom.make_non_exec_tiled_tma_store": (
        "cute_nvgpu", "AtomCopyMakeNonExecTiledTmaStoreOp", (3,), 2, (), ("tma_format",)),
    "vector.from_elements": ("vector", "FromElementsOp", None, 1, (), ()),
}
_INTEGER_OPS = {
    "arith.addi", "arith.subi", "arith.muli", "arith.divsi", "arith.divui", "arith.floordivsi",
    "arith.remsi", "arith.remui", "arith.andi", "arith.ori", "arith.xori",
    "arith.shli", "arith.shrsi", "arith.shrui", "arith.extsi", "arith.extui",
    "arith.trunci", "arith.select",
}
_OVERFLOW_OPS = {"arith.addi", "arith.subi", "arith.muli", "arith.shli", "arith.trunci"}


def _snapshot(operation: Any) -> tuple[str, bytes]:
    buffer = io.BytesIO()
    operation.write_bytecode(buffer)
    return str(operation), buffer.getvalue()


def _op(value: Any) -> Any:
    from cutlass._mlir import ir

    return value.operation if isinstance(value, ir.OpView) else value


def _walk(operation: Any) -> tuple[Any, ...]:
    result = [operation]
    for region in operation.regions:
        for block in region.blocks:
            for view in block.operations:
                result.extend(_walk(view.operation))
    return tuple(result)


def _source(module: Any, function: Any) -> tuple[Any, ...]:
    from cutlass._mlir import ir

    if (not isinstance(function, ir.Operation) or function.name != "func.func"
            or function not in tuple(view.operation for view in module.body.operations)
            or len(function.regions) != 1 or len(function.regions[0].blocks) != 1):
        raise ValueError("Expected the exact original single-block source function in its Module")
    return tuple(function.regions[0].blocks[0].arguments)


def _argument_attrs(function: Any, count: int) -> tuple[tuple[Any, ...], tuple[int, ...], bool]:
    from cutlass._mlir import ir

    attrs = tuple(function.attributes["arg_attrs"]) if "arg_attrs" in function.attributes else tuple(
        ir.DictAttr.get({}) for _ in range(count)
    )
    if len(attrs) != count or any(not isinstance(attr, ir.DictAttr) for attr in attrs):
        raise ValueError("Source argument attributes do not cover the original formals")
    tagged = [SOURCE_ARGUMENT in attr for attr in attrs]
    if any(tagged) and not all(tagged):
        raise ValueError("Source formal IDs must be complete")
    ids = []
    for index, attr in enumerate(attrs):
        value = attr[SOURCE_ARGUMENT] if tagged[index] else None
        if value is not None and (not isinstance(value, ir.IntegerAttr)
                or str(value.type) != "i64" or value.value < 0):
            raise ValueError("Source formal ID must be a nonnegative i64")
        ids.append(index if value is None else value.value)
    if len(set(ids)) != len(ids):
        raise ValueError("Source formal IDs must be unique")
    return attrs, tuple(ids), all(tagged)


def _kernel(module: Any, operation: Any) -> Any:
    from cutlass._mlir import ir

    if (operation.operands or len(operation.results) != 1 or str(operation.results[0].type) != "i64"
            or set(operation.attributes) != {"kernel_name"}
            or not isinstance(operation.attributes["kernel_name"], ir.SymbolRefAttr)):
        raise ValueError("Shared-memory query requires one exact existing kernel symbol")
    names = tuple(operation.attributes["kernel_name"].value)
    if len(names) != 2:
        raise ValueError("Shared-memory query requires module and kernel symbols")
    modules = [view.operation for view in module.body.operations if view.operation.name == "gpu.module"
               and view.operation.attributes["sym_name"].value == names[0]]
    if len(modules) != 1 or len(modules[0].regions[0].blocks) != 1:
        raise ValueError("Shared-memory query GPU module is absent or ambiguous")
    kernels = [view.operation for view in modules[0].regions[0].blocks[0].operations
               if view.operation.name == "cuda.kernel" and view.operation.attributes["sym_name"].value == names[1]]
    if len(kernels) != 1:
        raise ValueError("Shared-memory query does not identify an existing cuda.kernel")
    return kernels[0]


def _validate_tree(module: Any, operation: Any, kernels: dict[Any, Any]) -> None:
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import cute, cute_nvgpu, vector

    if operation.successors:
        raise ValueError(f"Unsupported source dependency or effect: {operation.name}")
    if operation.name == "scf.if":
        if (operation.attributes or len(operation.operands) != 1 or str(operation.operands[0].type) != "i1"
                or not operation.results or len(operation.regions) != 2
                or any(not isinstance(value.type, ir.IntegerType) for value in operation.results)):
            raise ValueError("Expected a pure scalar-result scf.if with two regions")
        for region in operation.regions:
            if len(region.blocks) != 1 or region.blocks[0].arguments:
                raise ValueError("Scalar-result branches require one block without arguments")
            body = tuple(view.operation for view in region.blocks[0].operations)
            if (not body or body[-1].name != "scf.yield" or body[-1].attributes
                    or body[-1].results or body[-1].regions or body[-1].successors
                    or tuple(value.type for value in body[-1].operands) != tuple(value.type for value in operation.results)):
                raise ValueError("Branch yield does not preserve the selected scalar result types")
            for nested in body[:-1]:
                _validate_tree(module, nested, kernels)
        return
    if operation.regions:
        raise ValueError(f"Unsupported source dependency or effect: {operation.name}")
    attrs = set(operation.attributes)
    valid = False
    if operation.name in _PROJECTIONS:
        valid = not attrs
    elif operation.name in _TYPED_HOST_OPS:
        dialect, view_name, counts, results, required, optional = _TYPED_HOST_OPS[operation.name]
        namespace = {"cute": cute, "cute_nvgpu": cute_nvgpu, "vector": vector}[dialect]
        valid = (isinstance(operation.opview, getattr(namespace, view_name))
                 and set(required) <= attrs <= set(required) | set(optional)
                 and (counts is None or len(operation.operands) in counts)
                 and len(operation.results) == results and operation.verify())
    elif operation.name in _METADATA_OPS:
        view_name, count = _METADATA_OPS[operation.name]
        valid = (isinstance(operation.opview, getattr(cute, view_name)) and not attrs
                 and len(operation.results) == 1
                 and (count is None or len(operation.operands) == count))
    elif operation.name == "cute.make_layout":
        segments = operation.attributes["operandSegmentSizes"] if "operandSegmentSizes" in attrs else None
        valid = (isinstance(operation.opview, cute.MakeLayoutOp) and attrs == {"operandSegmentSizes"}
                 and isinstance(segments, ir.DenseI32ArrayAttr) and len(segments) == 2
                 and all(count in (0, 1) for count in segments)
                 and sum(segments) == len(operation.operands) and len(operation.results) == 1)
    elif operation.name == "cute.select":
        valid = (isinstance(operation.opview, cute.SelectOp) and attrs == {"mode"}
                 and isinstance(operation.attributes["mode"], ir.DenseI32ArrayAttr)
                 and len(operation.operands) == 1 and len(operation.results) == 1)
    elif operation.name == "cute.make_view":
        valid = (isinstance(operation.opview, cute.MakeViewOp) and not attrs
                 and len(operation.operands) in (1, 2) and len(operation.results) == 1)
    elif operation.name in _INTEGER_OPS:
        valid = not attrs and all(isinstance(value.type, ir.IntegerType) for value in (*operation.operands, *operation.results))
        if operation.name in _OVERFLOW_OPS:
            valid = (attrs <= {"overflowFlags"}
                     and (not attrs or operation.attributes["overflowFlags"] == ir.Attribute.parse("#arith.overflow<none>"))
                     and all(isinstance(value.type, ir.IntegerType) for value in (*operation.operands, *operation.results)))
    elif operation.name == "arith.constant":
        if attrs == {"value"} and not operation.operands and len(operation.results) == 1:
            value = operation.attributes["value"]
            result_type = operation.results[0].type
            valid = ((isinstance(value, ir.IntegerAttr) and value.type == result_type)
                     or (isinstance(value, ir.BoolAttr) and str(result_type) == "i1")
                     or (isinstance(value, ir.FloatAttr) and value.type == result_type
                         and str(result_type) in ("f32", "f64")))
    elif operation.name == "arith.cmpi":
        valid = (attrs == {"predicate"} and isinstance(operation.attributes["predicate"], ir.IntegerAttr)
                 and 0 <= operation.attributes["predicate"].value <= 9
                 and all(isinstance(value.type, ir.IntegerType) for value in (*operation.operands, *operation.results)))
    elif operation.name == "cute.kernel_smem_size":
        kernel = _kernel(module, operation)
        kernels[kernel] = _snapshot(kernel)
        valid = True
    if not valid:
        attributes = {key: str(operation.attributes[key]) for key in attrs}
        raise ValueError(f"Unsupported source dependency or effect: {operation.name}, attributes={attributes}")


@dataclass(frozen=True)
class OwnedScalarHelper:
    symbol: str
    operation: Any = field(repr=False, compare=False)
    module: Any = field(repr=False, compare=False)
    context: Any = field(repr=False, compare=False)
    source_function: Any = field(repr=False, compare=False)
    source_ids: tuple[int, ...]
    source_types: tuple[str, ...]
    result_types: tuple[str, ...]
    output_values: tuple[Any, ...] = field(repr=False, compare=False)
    output_origins: tuple[tuple[str, int, int], ...]
    _source_body: tuple[str, ...] = field(repr=False)
    _source_header: tuple[tuple[str, str], ...] = field(repr=False)
    _original_arg_attrs: tuple[str, ...] = field(repr=False)
    _source_tagged: bool = field(repr=False)
    _helper_body: tuple[str, bytes] = field(repr=False)
    _kernels: tuple[Any, ...] = field(repr=False, compare=False)
    _seal: tuple[Any, ...] = field(repr=False)

    def _state(self) -> tuple[Any, ...]:
        return (self.symbol, id(self.operation), id(self.module), id(self.context), id(self.source_function),
                self.source_ids, self.source_types, self.result_types, id(self.output_values), self.output_origins,
                self._source_body, self._source_header, self._original_arg_attrs, self._source_tagged,
                self._helper_body, id(self._kernels))

    def check(self) -> None:
        from cutlass._mlir import ir

        if self._state() != self._seal or self.module.context != self.context:
            raise RuntimeError("Scalar helper ownership or specification changed")
        with self.context, ir.raw_values():
            arguments = _source(self.module, self.source_function)
            body = tuple(view.operation for view in self.source_function.regions[0].blocks[0].operations)
            attrs, ids, tagged = _argument_attrs(self.source_function, len(arguments))
            plain_attrs = tuple(str(ir.DictAttr.get({item.name: item.attr for item in attr if item.name != SOURCE_ARGUMENT})) for attr in attrs)
            header = tuple((key, str(self.source_function.attributes[key])) for key in self.source_function.attributes if key != "arg_attrs")
            helpers = [view.operation for view in self.module.body.operations
                       if "sym_name" in view.operation.attributes and view.operation.attributes["sym_name"].value == self.symbol]
            if (tuple(str(value.type) for value in arguments) != self.source_types or ids != self.source_ids
                    or (self._source_tagged and not tagged)
                    or plain_attrs != self._original_arg_attrs or header != self._source_header
                    or tuple(str(op) for op in body) != self._source_body
                    or helpers != [self.operation] or _snapshot(self.operation) != self._helper_body
                    or any(_snapshot(kernel) != state for kernel, state in self._kernels)):
                raise RuntimeError("Scalar helper source, body, or formal association changed")
            for value, (kind, index, result) in zip(self.output_values, self.output_origins):
                actual = arguments[index] if kind == "argument" else body[index].results[result]
                if value != actual:
                    raise RuntimeError("Scalar helper output source changed")


def emit_scalar_helper(module: Any, source_function: Any, output_values: Any, helper_symbol: str) -> OwnedScalarHelper:
    """Append a typed SSA slice, retaining complete needed scalar branch regions."""
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import func

    if type(helper_symbol) is not str or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9.]*", helper_symbol) is None:
        raise ValueError("Expected a new ordinary helper symbol")
    source = _op(source_function)
    outputs = tuple(output_values)
    with module.context, ir.Location.unknown(), ir.raw_values():
        if not module.operation.verify():
            raise ValueError("Source Module failed verification")
        arguments = _source(module, source)
        operations = tuple(view.operation for view in source.regions[0].blocks[0].operations)
        attrs, source_ids, tagged = _argument_attrs(source, len(arguments))
        if not outputs or any(not isinstance(value, ir.Value) or not isinstance(value.type, ir.IntegerType) for value in outputs):
            raise ValueError("Expected actual scalar integer SSA outputs")
        names = {view.operation.attributes["sym_name"].value for view in module.body.operations if "sym_name" in view.operation.attributes}
        if helper_symbol in names:
            raise ValueError("Helper symbol already exists")
        needed, visiting, kernels = set(), set(), {}

        def require(value: Any) -> None:
            if value in arguments:
                return
            owner = _op(value.owner)
            if owner not in operations:
                raise ValueError("Output or dependency escapes the original source block")
            if owner in needed:
                return
            if owner in visiting:
                raise ValueError("Cyclic source dependency")
            visiting.add(owner)
            _validate_tree(module, owner, kernels)
            tree = _walk(owner)
            internal = {result for operation in tree for result in operation.results}
            for operation in tree:
                for operand in operation.operands:
                    if operand not in internal:
                        require(operand)
            visiting.remove(owner)
            needed.add(owner)

        for value in outputs:
            require(value)
        origins = tuple(("argument", arguments.index(value), 0) if value in arguments else
                        ("result", operations.index(_op(value.owner)), tuple(_op(value.owner).results).index(value)) for value in outputs)
        original = tuple((view.operation, _snapshot(view.operation)) for view in module.body.operations)
        before = _snapshot(module.operation)
        with ir.InsertionPoint(module.body):
            helper = func.FuncOp(helper_symbol, ([value.type for value in arguments], [value.type for value in outputs]), visibility="public")
        try:
            helper.operation.attributes["no_inline"] = ir.UnitAttr.get()
            helper.arg_attrs = [ir.DictAttr.get({**{item.name: item.attr for item in attr}, SOURCE_ARGUMENT: ir.IntegerAttr.get(ir.IntegerType.get_signless(64), source_ids[index])}) for index, attr in enumerate(attrs)]
            block = helper.add_entry_block()
            mapping = dict(zip(arguments, block.arguments))
            for operation in operations:
                if operation not in needed:
                    continue
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
                func.ReturnOp([mapping[value] for value in outputs])
            if (tuple(view.operation for view in module.body.operations) != tuple(op for op, _ in original) + (helper.operation,)
                    or any(_snapshot(op) != state for op, state in original) or not module.operation.verify()):
                raise RuntimeError("Scalar helper emission changed original IR or failed verification")
            result = OwnedScalarHelper(
                helper_symbol, helper.operation, module, module.context, source, source_ids,
                tuple(str(value.type) for value in arguments), tuple(str(value.type) for value in outputs), outputs, origins,
                tuple(str(op) for op in operations), tuple((key, str(source.attributes[key])) for key in source.attributes if key != "arg_attrs"),
                tuple(str(ir.DictAttr.get({item.name: item.attr for item in attr if item.name != SOURCE_ARGUMENT})) for attr in attrs),
                tagged, _snapshot(helper.operation), tuple(kernels.items()), (),
            )
            result = replace(result, _seal=result._state())
            result.check()
            return result
        except BaseException:
            helper.operation.erase()
            if _snapshot(module.operation) != before:
                raise RuntimeError("Failed scalar helper emission did not restore original Module") from None
            raise
