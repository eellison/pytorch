from __future__ import annotations

import io
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch._inductor.runtime._cudagraph._compiler.argument_flow import ValueFlow


ACCESSOR_VERSION = 1
SOURCE_ARGUMENT = "cudagraph.source_arg"
_CASTS = frozenset({"llvm.trunc", "llvm.zext", "llvm.sext", "llvm.bitcast", "llvm.addrspacecast"})


def _snapshot(operation: Any) -> tuple[str, bytes]:
    buffer = io.BytesIO()
    operation.write_bytecode(buffer)
    return str(operation), buffer.getvalue()


def _function(module: Any, name: str, kind: str) -> Any:
    matches = [view.operation for view in module.body.operations
               if view.operation.name == kind and view.operation.attributes["sym_name"].value == name]
    if len(matches) != 1 or len(matches[0].regions) != 1 or not matches[0].regions[0].blocks:
        raise ValueError("Expected one defined function with the exact symbol")
    return matches[0]


@dataclass(frozen=True)
class _SourceOwner:
    module: Any = field(repr=False, compare=False)
    context: Any = field(repr=False, compare=False)
    host: Any = field(repr=False, compare=False)
    function_name: str
    source_types: tuple[str, ...]
    metadata: Any
    _owners: tuple[Any, ...] = field(repr=False, compare=False)
    _fingerprint: str = field(repr=False)

    def _digest(self) -> str:
        state = (ACCESSOR_VERSION, self.function_name, self.source_types,
                 self.metadata.params, self.metadata.symbols, self.metadata.ret)
        return sha256(repr(state).encode()).hexdigest()

    def check(self) -> None:
        from cutlass._mlir import ir

        if (any(value is not owner for value, owner in zip((self.module, self.context, self.host), self._owners))
                or self.module.context != self.context or self._digest() != self._fingerprint):
            raise RuntimeError("Component source ownership or signature changed")
        with self.context, ir.raw_values():
            host = _function(self.module, self.function_name, "func.func")
            if host != self.host or tuple(str(arg.type) for arg in host.regions[0].blocks[0].arguments) != self.source_types:
                raise RuntimeError("Original component source formals changed")


@dataclass(frozen=True)
class AccessorSpec:
    symbol: str
    host_function_name: str
    source_arg_index: int
    metadata_path: tuple[int, ...]
    name: str
    property: str
    property_path: tuple[int, ...]
    source_type: str
    result_type: str
    _source: _SourceOwner = field(repr=False, compare=False)
    _function: Any = field(repr=False, compare=False)
    _body: tuple[str, bytes] = field(repr=False)
    _seal: tuple[Any, ...] = field(repr=False)

    def _state(self) -> tuple[Any, ...]:
        return (ACCESSOR_VERSION, self.symbol, self.host_function_name, self.source_arg_index,
                self.metadata_path, self.name, self.property, self.property_path, self.source_type,
                self.result_type, id(self._source), id(self._function), self._body)

    def check(self) -> None:
        from cutlass._mlir import ir

        if self._state() != self._seal:
            raise RuntimeError("Component accessor specification changed")
        self._source.check()
        with self._source.context, ir.raw_values():
            function = _function(self._source.module, self.symbol, "func.func")
            if function != self._function or _snapshot(function) != self._body:
                raise RuntimeError("Original component accessor body changed")


def _emit_function(module: Any, symbol: str, argument_type: Any, index: int,
                   prop: str, axis: int | None, parameter: Any, metadata: Any) -> Any:
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import builtin, cute, func, llvm

    with ir.InsertionPoint(module.body):
        function = func.FuncOp(symbol, ([argument_type], []), visibility="public")
    try:
        function.operation.attributes["no_inline"] = ir.UnitAttr.get()
        function.arg_attrs = [ir.DictAttr.get({SOURCE_ARGUMENT: ir.IntegerAttr.get(ir.IntegerType.get_signless(64), index)})]
        block = function.add_entry_block()
        with ir.InsertionPoint(block):
            argument = block.arguments[0]
            pointer = cute.GetIterOp(argument).result
            layout = cute.GetLayoutOp(argument).result
            if (not isinstance(pointer.type, cute.PtrType) or not isinstance(layout.type, cute.LayoutType)
                    or cute.MemRefType.get(pointer.type, layout.type) != argument_type):
                raise ValueError("Component accessors require an ordinary pointer and layout MemRef")
            if prop == "pointer":
                result = builtin.UnrealizedConversionCastOp(
                    [llvm.PointerType.get(pointer.type.address_space)], [pointer],
                ).result
            else:
                dimensions = parameter.shape if prop == "shape" else parameter.strides
                projected = (cute.GetShapeOp(layout).result if prop == "shape"
                             else cute.GetStrideOp(layout).result)
                leaves = cute.GetLeavesOp(projected).results
                if projected.type.rank != len(dimensions) or len(leaves) != len(dimensions):
                    raise ValueError("Metadata and typed tensor property leaves disagree")
                scalars = cute.GetScalarsOp(leaves[axis]).results
                if len(scalars) != 1 or not isinstance(scalars[0].type, ir.IntegerType):
                    raise ValueError("A tensor property must project to one integer scalar")
                result = scalars[0]
                kind, value = dimensions[axis]
                if kind == "symbol" and result.type.width != metadata.symbols[value][1]:
                    raise ValueError("Tensor property width differs from its signature symbol")
            function.operation.attributes["function_type"] = ir.TypeAttr.get(
                ir.FunctionType.get([argument_type], [result.type]),
            )
            func.ReturnOp([result])
        if not function.operation.verify():
            raise RuntimeError("Generated component accessor failed verification")
        return function.operation
    except BaseException:
        function.operation.erase()
        raise


def emit_accessors(module: Any, host_function_name: str, metadata: Any) -> tuple[AccessorSpec, ...]:
    """Add typed accessors before the original artifact is serialized or lowered.

    Flat Tensor metadata supplies property indices, never aggregate layouts.
    The caller must compile this actual Module through its owned artifact.
    """
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import cute
    from torch._inductor.runtime._cudagraph._compiler.entry_signature import snapshot_metadata

    signature = snapshot_metadata(metadata)
    if signature.symbol_name != host_function_name:
        raise ValueError("Component metadata does not name the original host")
    with module.context, ir.Location.unknown(), ir.raw_values():
        if not module.operation.verify():
            raise ValueError("Component source Module failed verification")
        host = _function(module, host_function_name, "func.func")
        arguments = tuple(host.regions[0].blocks[0].arguments)
        forwarded = [param.ir_arg_index for param in signature.params if param.ir_arg_index is not None]
        if len(set(forwarded)) != len(forwarded) or set(forwarded) != set(range(len(arguments))):
            raise ValueError("Signature metadata must identify every original source formal")
        tensors = [(path, param) for path, param in enumerate(signature.params) if param.kind == "Tensor"]
        if not tensors or {param.ir_arg_index for _, param in tensors} != {
            index for index, arg in enumerate(arguments) if isinstance(arg.type, cute.MemRefType)
        }:
            raise ValueError("Tensor metadata must identify exactly the source MemRef formals")
        for _, parameter in tensors:
            if len(parameter.shape) != len(parameter.strides):
                raise ValueError("Tensor shape and stride metadata ranks differ")
            for kind, value in (*parameter.shape, *parameter.strides):
                if kind == "symbol" and not 0 <= value < len(signature.symbols):
                    raise ValueError("Tensor metadata references an unknown dimension symbol")
        owner = _SourceOwner(module, module.context, host, host_function_name,
                             tuple(str(arg.type) for arg in arguments), signature,
                             (module, module.context, host), "")
        owner = replace(owner, _fingerprint=owner._digest())
        original = tuple((view.operation, _snapshot(view.operation)) for view in module.body.operations)
        before = _snapshot(module.operation)
        names = {op.attributes["sym_name"].value for op, _ in original if "sym_name" in op.attributes}
        prefix = "__cudagraph_component_" + sha256(host_function_name.encode()).hexdigest()[:16]
        requests = []
        for path, parameter in tensors:
            properties = [("pointer", ())] + [(prop, (axis,)) for prop in ("shape", "stride")
                                              for axis in range(len(parameter.shape))]
            for prop, property_path in properties:
                suffix = "_".join(str(item) for item in property_path)
                symbol = f"{prefix}_{parameter.ir_arg_index}_{prop}{suffix}"
                if symbol in names:
                    raise ValueError("Component accessor symbol already exists")
                names.add(symbol)
                requests.append((symbol, path, parameter, prop, property_path))
        generated, specs = [], []
        try:
            for symbol, path, parameter, prop, property_path in requests:
                index = parameter.ir_arg_index
                function = _emit_function(module, symbol, arguments[index].type, index, prop,
                                          property_path[0] if property_path else None, parameter, signature)
                generated.append(function)
                result_type = ir.FunctionType(function.attributes["function_type"].value).results[0]
                spec = AccessorSpec(symbol, host_function_name, index, (path,), parameter.name, prop,
                                    property_path, str(arguments[index].type), str(result_type), owner,
                                    function, _snapshot(function), ())
                specs.append(replace(spec, _seal=spec._state()))
            current = tuple(view.operation for view in module.body.operations)
            if (current != tuple(op for op, _ in original) + tuple(generated)
                    or any(_snapshot(op) != state for op, state in original)
                    or not module.operation.verify()):
                raise RuntimeError("Accessor emission changed preexisting host or device IR")
        except BaseException:
            for function in reversed(generated):
                function.erase()
            if _snapshot(module.operation) != before:
                raise RuntimeError("Failed accessor emission did not restore the source Module") from None
            raise
        return tuple(specs)


@dataclass(frozen=True)
class Component:
    spec: AccessorSpec
    llvm_argument_type: str
    source: ValueFlow


def _tagged_arguments(function: Any) -> dict[int, Any]:
    from cutlass._mlir import ir

    arguments = function.regions[0].blocks[0].arguments
    attrs = function.attributes.get("arg_attrs")
    if not isinstance(attrs, ir.ArrayAttr) or len(attrs) != len(arguments):
        raise ValueError("Compiled formals lack preserved source argument markers")
    result = {}
    for argument, dictionary in zip(arguments, attrs):
        marker = dictionary[SOURCE_ARGUMENT] if SOURCE_ARGUMENT in dictionary else None
        if (not isinstance(marker, ir.IntegerAttr) or marker.type != ir.IntegerType.get_signless(64)
                or marker.value < 0 or marker.value in result):
            raise ValueError("Compiled source argument markers are missing or ambiguous")
        result[marker.value] = argument
    return result


def _host_source(flow: ValueFlow, source_index: int) -> ValueFlow:
    if flow.kind == "argument":
        if flow.argument != 0:
            raise ValueError("Component accessor references an unexpected argument")
        return replace(flow, argument=source_index)
    return replace(flow, operands=tuple(_host_source(item, source_index) for item in flow.operands))


def _read_component(function: Any, spec: AccessorSpec, host_argument: Any) -> Component:
    from torch._inductor.runtime._cudagraph._compiler.argument_flow import _Values
    from cutlass._mlir import ir

    if len(function.regions[0].blocks) != 1:
        raise ValueError("Component accessor must contain one block")
    arguments = _tagged_arguments(function)
    if set(arguments) != {spec.source_arg_index}:
        raise ValueError("Component accessor source argument identity changed")
    argument = arguments[spec.source_arg_index]
    if argument.type != host_argument.type:
        raise ValueError("Accessor and original host formal lowered to different types")
    operations = tuple(view.operation for view in function.regions[0].blocks[0].operations)
    if not operations or operations[-1].name != "llvm.return" or len(operations[-1].operands) != 1:
        raise ValueError("Component accessor must return exactly one value")
    for op in operations:
        if op.regions or op.successors:
            raise ValueError("Component accessor contains control flow")
        if op.name == "llvm.return":
            valid = op == operations[-1] and not op.results and not op.attributes
        elif op.name == "llvm.extractvalue":
            valid = len(op.operands) == len(op.results) == 1 and set(op.attributes) == {"position"}
        elif op.name == "llvm.mlir.constant":
            valid = (not op.operands and len(op.results) == 1 and set(op.attributes) == {"value"}
                     and isinstance(op.attributes["value"], ir.IntegerAttr))
        elif op.name == "llvm.mlir.zero":
            valid = not op.operands and len(op.results) == 1 and not op.attributes
        else:
            valid = op.name in _CASTS and len(op.operands) == len(op.results) == 1 and not op.attributes
        if not valid:
            raise ValueError(f"Unsupported component accessor operation: {op.name}")
    value = operations[-1].operands[0]
    if str(value.type) != spec.result_type:
        raise ValueError("Component accessor result type changed during lowering")
    flow = _Values((argument,)).read(value)
    terminal = flow
    while terminal.kind in _CASTS:
        terminal = terminal.operands[0]
    parameter = spec._source.metadata.params[spec.metadata_path[0]]
    if spec.property == "pointer":
        if terminal.kind != "argument":
            raise ValueError("Tensor pointer accessor lost its original argument provenance")
    else:
        dimension = (parameter.shape if spec.property == "shape" else parameter.strides)[spec.property_path[0]]
        if dimension[0] == "symbol" and terminal.kind != "argument":
            raise ValueError("Dynamic tensor property lost its original argument provenance")
        if dimension[0] == "constant":
            defining = value.owner.operation if isinstance(value.owner, ir.OpView) else value.owner
            exact = (flow.kind == "constant" and isinstance(defining, ir.Operation)
                     and defining.attributes["value"].value == dimension[1])
            if not exact and not (flow.kind == "zero" and dimension[1] == 0 and isinstance(value.type, ir.IntegerType)):
                raise ValueError("Static tensor property differs from its declared signature constant")
    return Component(spec, str(argument.type), _host_source(flow, spec.source_arg_index))


def _analyze(program: Any, specs: tuple[AccessorSpec, ...]) -> tuple[Component, ...]:
    from cutlass._mlir import ir
    from torch._inductor.runtime._cudagraph._compiler.entry_signature import snapshot_metadata

    program.check()
    if type(specs) is not tuple or not specs or any(type(spec) is not AccessorSpec for spec in specs):
        raise TypeError("Expected retained component accessor specifications")
    owner = specs[0]._source
    if (program.source_module is not owner.module or program.source_context is not owner.context
            or program.function_name != owner.function_name or any(spec._source is not owner for spec in specs)):
        raise RuntimeError("Accessors do not belong to the original compiler Module")
    metadata = [item for item in program.source_metadata if item.symbol_name == owner.function_name]
    if len(metadata) != 1:
        raise RuntimeError("Expected original compiler-owned source metadata")
    actual = snapshot_metadata(metadata[0])
    if (actual.params, actual.symbols, actual.ret) != (owner.metadata.params, owner.metadata.symbols, owner.metadata.ret):
        raise RuntimeError("Accessor metadata differs from the original compiler signature")
    for spec in specs:
        spec.check()
    expected = {(index, prop, path) for index, parameter in enumerate(owner.metadata.params) if parameter.kind == "Tensor"
                for prop, path in [("pointer", ())] + [(kind, (axis,)) for kind in ("shape", "stride")
                                                      for axis in range(len(parameter.shape))]}
    actual_keys = [(spec.metadata_path[0], spec.property, spec.property_path) for spec in specs]
    if len(set(actual_keys)) != len(actual_keys) or set(actual_keys) != expected:
        raise RuntimeError("Component accessor set is incomplete or duplicated")
    with program.context, ir.raw_values():
        if not program.module.operation.verify():
            raise ValueError("Compiled component Module failed verification")
        host = _function(program.module, program.function_name, "llvm.func")
        arguments = _tagged_arguments(host)
        if set(arguments) != set(range(len(owner.source_types))):
            raise ValueError("Original host lowering dropped, split, or invented formals")
        result = tuple(_read_component(_function(program.module, spec.symbol, "llvm.func"), spec,
                                       arguments[spec.source_arg_index]) for spec in specs)
    program.check()
    return result


@dataclass(frozen=True)
class ComponentMapping:
    program: Any = field(repr=False, compare=False)
    specs: tuple[AccessorSpec, ...]
    components: tuple[Component, ...]
    _owners: tuple[Any, ...] = field(repr=False, compare=False)

    def check(self) -> None:
        if any(value is not owner for value, owner in zip((self.program, self.specs, self.components), self._owners)):
            raise RuntimeError("Compiled component mapping ownership changed")
        if _analyze(self.program, self.specs) != self.components:
            raise RuntimeError("Compiled tensor component correspondence changed")


def analyze_accessors(compiled_program: Any, specs: tuple[AccessorSpec, ...]) -> ComponentMapping:
    """Read compiler-produced component paths and constants; never byte offsets."""
    components = _analyze(compiled_program, specs)
    return ComponentMapping(compiled_program, specs, components, (compiled_program, specs, components))
