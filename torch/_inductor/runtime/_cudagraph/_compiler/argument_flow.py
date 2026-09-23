from __future__ import annotations

import io
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from torch._inductor.runtime._cudagraph._compiler.branch_abi import _allocation, _calls, _exact_uses, _function, _gep_indices, _integer, _ordered, _owner, _uses
from torch._inductor.runtime.cutedsl_parameter_analysis import _raw_value, _walk_block


FLOW_VERSION = 1
_LAUNCHES = {"_cudaLaunchKernelEx": (3, 1, 2), "_cudaLaunchKernel": (10, 0, 7)}
_HOST_CALLS = frozenset(_LAUNCHES) | {
    "printf", "_cudaGetDevice", "_cudaDeviceGetAttribute", "_cuKernelGetAttribute",
    "_cudaFuncSetAttribute", "_cudaKernelSetAttributeForDevice", "_cudaOccupancyMaxActiveBlocksPerMultiprocessor",
}
_UNARY = {"llvm.trunc", "llvm.zext", "llvm.sext", "llvm.bitcast", "llvm.addrspacecast",
          "llvm.fptrunc", "llvm.fpext", "llvm.fptosi", "llvm.fptoui", "llvm.sitofp", "llvm.uitofp",
          "llvm.ptrtoint", "llvm.inttoptr"}
_BINARY = {"llvm.add", "llvm.sub", "llvm.mul", "llvm.udiv", "llvm.sdiv", "llvm.urem", "llvm.srem",
           "llvm.shl", "llvm.lshr", "llvm.ashr", "llvm.and", "llvm.or", "llvm.xor", "llvm.icmp",
           "llvm.fadd", "llvm.fsub", "llvm.fmul", "llvm.fdiv", "llvm.frem", "llvm.fcmp"}

_HOST_OPS = _UNARY | _BINARY | {
    "llvm.alloca", "llvm.load", "llvm.store", "llvm.call", "llvm.getelementptr",
    "llvm.br", "llvm.cond_br", "llvm.return", "llvm.unreachable", "llvm.intr.trap",
    "llvm.mlir.addressof", "llvm.mlir.constant", "llvm.mlir.zero", "llvm.mlir.undef", "llvm.mlir.poison",
    "llvm.extractvalue", "llvm.insertvalue", "llvm.select",
}
# These are the emitted compiler/runtime entry ABIs, independent of user kernels.
_RUNTIME_ABI = {
    "_cudaLaunchKernelEx": "i32 (ptr, ptr, ptr)",
    "_cudaLaunchKernel": "i32 (ptr, i32, i32, i32, i32, i32, i32, ptr, i64, ptr)",
    "_cudaLibraryLoadData": "i32 (ptr, ptr, ptr, ptr, i32, ptr, ptr, i32)",
    "_cudaLibraryGetKernel": "i32 (ptr, ptr, ptr)",
    "_cudaOccupancyMaxActiveBlocksPerMultiprocessor": "i32 (ptr, ptr, i32, i64)",
    "_cuKernelGetAttribute": "i32 (ptr, i32, ptr, i32)",
    "_cudaDeviceGetAttribute": "i32 (ptr, i32, i32)",
    "_cudaGetDevice": "i32 (ptr)",
    "_cudaKernelSetAttributeForDevice": "i32 (ptr, i32, i32, i32)",
    "_cudaFuncSetAttribute": "i32 (ptr, i32, i32)",
    "printf": "i32 (ptr, ...)",
}


def _check_host_operations(module: Any, function_name: str) -> None:
    from cutlass._mlir import ir

    c_convention = ir.Attribute.parse("#llvm.cconv<ccc>")
    functions = {}
    for view in module.body.operations:
        op = view.operation
        if op.name == "llvm.func":
            name = op.attributes["sym_name"].value
            if name in functions:
                raise ValueError("Duplicate LLVM function symbol")
            functions[name] = op
    for name, signature in _RUNTIME_ABI.items():
        if name not in functions:
            continue
        function = functions[name]
        if any(region.blocks for region in function.regions):
            raise ValueError(f"Runtime callee must be an external declaration: {name}")
        if function.attributes["function_type"].value != ir.Type.parse(f"!llvm.func<{signature}>"):
            raise ValueError(f"Runtime callee has an unsupported emitted ABI: {name}")
        convention = function.attributes.get("CConv")
        if convention is not None and convention != c_convention:
            raise ValueError(f"Runtime callee has an unsupported calling convention: {name}")
    pending = [function_name, "cuda_num_binaries", "cuda_init", "cuda_load", "cuda_load_to_device"]
    checked = set()
    while pending:
        name = pending.pop()
        if name in checked:
            continue
        if name not in functions:
            raise ValueError(f"Missing host or registration function: {name}")
        checked.add(name)
        function = functions[name]
        for region in function.regions:
            for block in region.blocks:
                for view in block.operations:
                    op = view.operation
                    if op.regions or op.name not in _HOST_OPS:
                        raise ValueError(f"Unsupported host operation or effect: {op.name}")
                    if op.name != "llvm.call":
                        continue
                    callee = op.attributes.get("callee")
                    if callee is None or callee.value not in functions:
                        raise ValueError("Unknown or indirect host call is unsupported")
                    target = functions[callee.value]
                    if callee.value not in _RUNTIME_ABI and not any(region.blocks for region in target.regions):
                        raise ValueError(f"Unknown external host effect: {callee.value}")
                    pending.append(callee.value)
                    convention = op.attributes.get("CConv")
                    if callee.value in _RUNTIME_ABI and convention is not None:
                        if convention != c_convention:
                            raise ValueError(f"Runtime call has an unsupported calling convention: {callee.value}")


@dataclass(frozen=True)
class ValueFlow:
    kind: str
    llvm_type: str
    argument: int | None = None
    path: tuple[int, ...] = ()
    value: str | None = None
    operands: tuple[ValueFlow, ...] = ()
    attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ParameterFlow:
    index: int
    llvm_type: str
    source: ValueFlow


@dataclass(frozen=True)
class BinaryImage:
    library_slot: int
    global_name: str
    sha256: str
    data: bytes = field(repr=False)


@dataclass(frozen=True)
class KernelRegistration:
    kernel_symbol: str
    handle_global: str
    library_slot: int
    binary_global: str
    binary_sha256: str


@dataclass(frozen=True)
class LaunchFlow:
    index: int
    block_index: int
    operation_index: int
    callee: str
    registration: KernelRegistration
    parameters: tuple[ParameterFlow, ...]


def _snapshot(module: Any) -> tuple[str, bytes]:
    buffer = io.BytesIO()
    module.operation.write_bytecode(buffer)
    return str(module), buffer.getvalue()


@dataclass(frozen=True)
class ArgumentFlow:
    module: Any = field(repr=False)
    context: Any = field(repr=False)
    function_name: str
    host_types: tuple[str, ...]
    binaries: tuple[BinaryImage, ...]
    registrations: tuple[KernelRegistration, ...]
    launches: tuple[LaunchFlow, ...]
    module_sha256: str
    _text: str = field(repr=False)
    _bytecode: bytes = field(repr=False)
    _owners: tuple[Any, Any] = field(repr=False)
    _fingerprint: str = field(repr=False)

    def _digest(self) -> str:
        state = (FLOW_VERSION, self.function_name, self.host_types, self.binaries, self.registrations,
                 self.launches, self.module_sha256)
        return sha256(repr(state).encode()).hexdigest()

    def check(self) -> None:
        if (self.module is not self._owners[0] or self.context is not self._owners[1]
                or self.module.context != self.context or self._digest() != self._fingerprint
                or sha256(self._bytecode).hexdigest() != self.module_sha256
                or any(sha256(image.data).hexdigest() != image.sha256 for image in self.binaries)):
            raise RuntimeError("Argument-flow source association changed")
        with self.context:
            if _snapshot(self.module) != (self._text, self._bytecode):
                raise RuntimeError("The owned argument-flow Module changed")


def _plain_memory(op: Any, *, allow_nontemporal: bool = False) -> None:
    from cutlass._mlir import ir

    allowed = {"alignment", "ordering"} | ({"nontemporal"} if allow_nontemporal else set())
    if set(op.attributes) - allowed:
        raise ValueError("Unsupported memory access attributes in argument packing")
    if "nontemporal" in op.attributes and not isinstance(op.attributes["nontemporal"], ir.UnitAttr):
        raise ValueError("Expected a unit nontemporal memory hint")
    ordering = op.attributes.get("ordering")
    if ordering is not None and (not isinstance(ordering, ir.IntegerAttr) or ordering.value != 0):
        raise ValueError("Atomic ABI storage accesses are unsupported")


def _local_allocation(value: Any, count: int, typ: Any) -> Any:
    op = _allocation(value, count, typ)
    if set(op.attributes) - {"elem_type", "alignment"}:
        raise ValueError("Unsupported ABI allocation attributes")
    return op


def _library_slot(value: Any, argument: Any) -> tuple[int, Any]:
    from cutlass._mlir.dialects import llvm

    gep = _owner(value, "llvm.getelementptr")
    indices = _gep_indices(gep)
    if (gep.operands[0] != argument or len(indices) != 1 or indices[0] < 0
            or gep.attributes["elem_type"].value != llvm.PointerType.get()):
        raise ValueError("Expected a constant library-array slot")
    return indices[0], gep


def _registrations(module: Any) -> tuple[tuple[BinaryImage, ...], tuple[KernelRegistration, ...]]:
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm

    globals_by_name = {}
    for view in module.body.operations:
        op = view.operation
        if op.name == "llvm.mlir.global":
            name = op.attributes["sym_name"].value
            if name in globals_by_name:
                raise ValueError("Duplicate LLVM global name")
            globals_by_name[name] = op
    counter = _function(module, "cuda_num_binaries")
    counter_ops = [view.operation for view in counter.regions[0].blocks[0].operations]
    if (len(counter.regions[0].blocks) != 1 or len(counter_ops) != 2
            or counter_ops[-1].name != "llvm.return" or len(counter_ops[-1].operands) != 1):
        raise ValueError("Expected a constant compiler library count")
    count = _integer(counter_ops[-1].operands[0])
    if count <= 0:
        raise ValueError("Expected at least one embedded library")
    init = _function(module, "cuda_init")
    init_calls = _calls(init, "_cudaLibraryLoadData")
    if len(init_calls) != count or not init.regions[0].blocks[0].arguments:
        raise ValueError("Library count and initializer calls disagree")
    images = {}
    for call in init_calls:
        if len(call.operands) != 8:
            raise ValueError("Unsupported library-load calling convention")
        index, slot = _library_slot(call.operands[0], init.regions[0].blocks[0].arguments[0])
        address = _owner(call.operands[1], "llvm.mlir.addressof")
        name = address.attributes["global_name"].value
        global_op = globals_by_name.get(name)
        value = None if global_op is None else global_op.attributes.get("value")
        if (index in images or index >= count or not isinstance(value, ir.StringAttr) or not value.value_bytes
                or "constant" not in global_op.attributes
                or global_op.attributes["global_type"].value != ir.Type.parse(f"!llvm.array<{len(value.value_bytes)} x i8>")):
            raise ValueError("Library initialization lacks an exact immutable image")
        _ordered(slot, call)
        _ordered(address, call)
        _exact_uses(slot.results[0], [(call, 0)])
        _exact_uses(address.results[0], [(call, 1)])
        images[index] = BinaryImage(index, name, sha256(value.value_bytes).hexdigest(), value.value_bytes)
    if set(images) != set(range(count)):
        raise ValueError("Incomplete library-slot initialization")

    mapping = None
    registration_calls = []
    for entry in ("cuda_load", "cuda_load_to_device"):
        loader = _function(module, entry)
        if not loader.regions[0].blocks[0].arguments:
            raise ValueError("Loader has no library-array formal")
        current = {}
        for call in _calls(loader, "_cudaLibraryGetKernel"):
            if len(call.operands) != 3:
                raise ValueError("Unsupported kernel-registration calling convention")
            address = _owner(call.operands[0], "llvm.mlir.addressof")
            name = address.attributes["global_name"].value
            library = _owner(call.operands[1], "llvm.load")
            _plain_memory(library)
            index, slot = _library_slot(library.operands[0], loader.regions[0].blocks[0].arguments[0])
            if index not in images:
                raise ValueError("Kernel registration references an uninitialized library")
            memory = call.operands[2]
            stores = [op for op, operand in _uses(memory) if op.name == "llvm.store" and operand == 1]
            if len(stores) != 1:
                raise ValueError("Kernel symbol storage must have one writer")
            store = stores[0]
            _plain_memory(store)
            literal = _owner(store.operands[0], "llvm.mlir.constant")
            text = literal.attributes.get("value")
            if not isinstance(text, ir.StringAttr) or not text.value_bytes.endswith(b"\0") or b"\0" in text.value_bytes[:-1]:
                raise ValueError("Kernel symbol must be an exact NUL-terminated literal")
            symbol = text.value_bytes[:-1].decode("utf-8")
            if not symbol or name in current or any(item.kernel_symbol == symbol and item.library_slot == index for item in current.values()):
                raise ValueError("Ambiguous kernel symbol or handle registration")
            typ = literal.results[0].type
            if typ != ir.Type.parse(f"!llvm.array<{len(text.value_bytes)} x i8>"):
                raise ValueError("Kernel symbol literal has an inconsistent type")
            allocation = _local_allocation(memory, 1, typ)
            _ordered(allocation, store, call)
            _ordered(literal, store)
            _ordered(address, call)
            _ordered(slot, library, call)
            _exact_uses(memory, [(store, 1), (call, 2)])
            _exact_uses(slot.results[0], [(library, 0)])
            _exact_uses(library.results[0], [(call, 1)])
            image = images[index]
            current[name] = KernelRegistration(symbol, name, index, image.global_name, image.sha256)
            registration_calls.append(call)
        if not current or (mapping is not None and current != mapping):
            raise ValueError("Compiler loader entry points disagree on exact kernel registrations")
        mapping = current
    for name in mapping:
        op = globals_by_name.get(name)
        if op is None or op.attributes["global_type"].value != llvm.PointerType.get():
            raise ValueError("Kernel registration does not name a pointer global")
    image_names = {image.global_name for image in images.values()}
    for op in _walk_block(module.body):
        if op.name != "llvm.mlir.addressof":
            continue
        name = op.attributes["global_name"].value
        if name in image_names:
            if any(use not in init_calls or operand != 1 for use, operand in _uses(op.results[0])):
                raise ValueError("Unknown binary-image writer or escaped address")
        if name in mapping:
            for use, operand in _uses(op.results[0]):
                if use in registration_calls and operand == 0:
                    continue
                if use.name != "llvm.load" or operand != 0:
                    raise ValueError("Unknown kernel-handle writer or escaped address")
                _plain_memory(use)
    return tuple(images[index] for index in range(count)), tuple(mapping[name] for name in sorted(mapping))


class _Values:
    def __init__(self, arguments: Any, *, local_pointer_width: int | None = None) -> None:
        if local_pointer_width is not None and (type(local_pointer_width) is not int or local_pointer_width <= 0):
            raise ValueError("Expected a checked positive local target pointer width")
        arguments = tuple(arguments)
        self.arguments = {_raw_value(value): index for index, value in enumerate(arguments)}
        self.argument_types = tuple(value.type for value in arguments)
        self.cache: dict[Any, ValueFlow] = {}
        self.active: set[Any] = set()
        self.local_pointer_width = local_pointer_width

    def _local_load(self, load: Any) -> ValueFlow:
        from cutlass._mlir import ir
        from cutlass._mlir.dialects import llvm
        from torch._inductor.runtime._cudagraph._compiler.llvm_types import integer_array_parts

        positions = {view.operation: index for index, view in enumerate(load.block.operations)}
        pointers = {}

        def producer(value):
            op = _raw_value(value).owner
            return op.operation if isinstance(op, ir.OpView) else op

        def resolve(value):
            value = _raw_value(value)
            if value in pointers:
                return pointers[value]
            op = producer(value)
            if (not isinstance(op, ir.Operation) or op not in positions
                    or len(op.results) != 1 or op.regions or op.successors
                    or value.type != llvm.PointerType.get()):
                raise ValueError("Local memory requires same-block default-address-space pointer provenance")
            if op.name == "llvm.alloca":
                element = op.attributes["elem_type"].value
                count = _integer(op.operands[0])
                if (not isinstance(element, ir.IntegerType) or not element.is_signless
                        or element.width not in (1, 8, 16, 32, 64) or count < 0):
                    raise ValueError("Local allocation must be a constant-count supported integer array")
                _local_allocation(value, count, element)
                result = op, 0
            elif op.name == "llvm.getelementptr":
                if set(op.attributes) != {"elem_type", "rawConstantIndices"}:
                    raise ValueError("Unsupported local pointer-index attributes")
                allocation, index = resolve(op.operands[0])
                count = _integer(allocation.operands[0])
                element = allocation.attributes["elem_type"].value
                indices = _gep_indices(op)
                indexed = op.attributes["elem_type"].value
                if indexed == element and len(indices) == 1:
                    index += indices[0]
                elif integer_array_parts(indexed) == (count, element) and len(indices) == 2 and indices[0] == 0:
                    index += indices[1]
                else:
                    raise ValueError("Local GEP does not index the original homogeneous allocation")
                if not 0 <= index <= count:
                    raise ValueError("Local GEP is outside its original allocation")
                _ordered(producer(op.operands[0]), op)
                result = allocation, index
            elif op.name == "llvm.inttoptr":
                if len(op.operands) != 1 or op.attributes:
                    raise ValueError("Unsupported local pointer round trip")
                cast = _owner(op.operands[0], "llvm.ptrtoint")
                typ = op.operands[0].type
                if (len(cast.operands) != 1 or cast.attributes or cast.operands[0].type != value.type
                        or not isinstance(typ, ir.IntegerType) or not typ.is_signless
                        or self.local_pointer_width is None or typ.width < self.local_pointer_width):
                    raise ValueError("Local pointer round trip lacks sufficient checked target width")
                result = resolve(cast.operands[0])
                _ordered(producer(cast.operands[0]), cast, op)
            else:
                raise ValueError("Unsupported local pointer provenance: " + op.name)
            pointers[value] = result
            return result

        if len(load.operands) != 1 or len(load.results) != 1:
            raise ValueError("Expected one local load address and result")
        allocation, load_index = resolve(load.operands[0])
        count = _integer(allocation.operands[0])
        element = allocation.attributes["elem_type"].value

        def load_path(op, index):
            _plain_memory(op, allow_nontemporal=True)
            typ = op.results[0].type
            if typ == element and 0 <= index < count:
                return None
            path = ()
            while isinstance(typ, llvm.StructType) and not typ.opaque and len(typ.body) == 1:
                path += (0,)
                typ = typ.body[0]
            if index != 0 or integer_array_parts(typ) != (count, element):
                raise ValueError("Local aggregate load must match the complete homogeneous allocation")
            return path

        path = load_path(load, load_index)
        pending, seen, stores = [allocation.results[0]], set(), {}
        while pending:
            pointer = _raw_value(pending.pop())
            if pointer in seen:
                continue
            seen.add(pointer)
            current, index = resolve(pointer)
            if current != allocation:
                raise ValueError("Local pointer changed its original allocation")
            for use, operand in _uses(pointer):
                if use not in positions:
                    raise ValueError("Local allocation escapes its straight-line block")
                _ordered(producer(pointer), use)
                if use.name == "llvm.getelementptr" and operand == 0:
                    resolve(use.results[0])
                    pending.append(use.results[0])
                elif use.name == "llvm.ptrtoint" and operand == 0:
                    for consumer, number in _uses(use.results[0]):
                        if consumer.name != "llvm.inttoptr" or number != 0:
                            raise ValueError("Local pointer integer escapes its exact round trip")
                        resolve(consumer.results[0])
                        pending.append(consumer.results[0])
                elif use.name == "llvm.store" and operand == 1:
                    _plain_memory(use, allow_nontemporal=True)
                    if len(use.operands) != 2 or use.operands[0].type != element or not 0 <= index < count:
                        raise ValueError("Local store must write one original integer element")
                    stores[use] = index
                elif use.name == "llvm.load" and operand == 0:
                    if len(use.operands) != 1 or len(use.results) != 1:
                        raise ValueError("Expected one local load address and result")
                    load_path(use, index)
                else:
                    raise ValueError("Unsupported local allocation alias or escape")
        latest = {}
        for store in sorted(stores, key=positions.__getitem__):
            if positions[store] < positions[load]:
                latest[stores[store]] = store.operands[0]
        if path is None:
            return self._read(latest[load_index]) if load_index in latest else ValueFlow("undef", str(element))
        result = ValueFlow("undef", str(load.results[0].type))
        for index, value in sorted(latest.items()):
            result = ValueFlow("llvm.insertvalue", result.llvm_type, path=(*path, index),
                               operands=(result, self._read(value)))
        return result

    @staticmethod
    def _children(typ):
        from torch._inductor.runtime._cudagraph._compiler.aggregate_plan import _integer_vector_parts
        from cutlass._mlir.dialects import llvm
        from torch._inductor.runtime._cudagraph._compiler.llvm_types import integer_array_parts

        if isinstance(typ, llvm.StructType):
            if typ.opaque:
                raise ValueError("Opaque aggregate provenance has no field types")
            return tuple(typ.body)
        array = integer_array_parts(typ)
        if array is not None:
            count, element = array
            return (element,) * count
        vector = _integer_vector_parts(typ)
        if vector is not None:
            count, element = vector
            return (element,) * count
        return None

    def _at(self, typ, path):
        for index in path:
            children = self._children(typ)
            if type(index) is not int or children is None or not 0 <= index < len(children):
                raise ValueError("Aggregate component path requires exposed LLVM struct field types")
            typ = children[index]
        return typ

    def _leaves(self, typ, path=()):
        children = self._children(typ)
        if children is None:
            return ((path, typ),)
        return tuple(leaf for index, child in enumerate(children)
                     for leaf in self._leaves(child, (*path, index)))

    def _project(self, source, typ, path):
        from torch._inductor.runtime._cudagraph._compiler.aggregate_plan import _integer_vector_parts
        from cutlass._mlir import ir

        target = self._at(typ, path)
        if not path:
            return source
        if source.kind == "argument":
            return ValueFlow("argument", str(target), source.argument, source.path + path)
        if source.kind in ("zero", "poison", "undef"):
            return ValueFlow(source.kind, str(target))
        if source.kind == "constant" and isinstance(typ, ir.VectorType):
            count, element = _integer_vector_parts(typ)
            attribute = ir.Attribute.parse(source.value)
            if (not isinstance(attribute, ir.DenseIntElementsAttr) or attribute.type != typ
                    or len(attribute) != count or len(path) != 1):
                raise ValueError("Constant vector lanes require the exact typed dense integer attribute")
            literal = ir.IntegerAttr.get(element, int(attribute[path[0]]))
            return ValueFlow("constant", str(target), value=str(literal))
        if source.kind == "llvm.insertvalue":
            base, inserted = source.operands
            position = source.path
            if path[:len(position)] == position:
                return self._project(inserted, self._at(typ, position), path[len(position):])
            projected = self._project(base, typ, path)
            if position[:len(path)] == path:
                return ValueFlow("llvm.insertvalue", str(target), path=position[len(path):],
                                 operands=(projected, inserted))
            return projected
        return ValueFlow("llvm.extractvalue", str(target), path=path, operands=(source,))

    def _normalize(self, source, typ):
        if self._children(typ) is None or source.kind == "argument":
            return source
        leaves = self._leaves(typ)
        if not leaves:
            return source
        projected = tuple((path, leaf, self._project(source, typ, path)) for path, leaf in leaves)
        argument = projected[0][2].argument
        if (argument is not None and self.argument_types[argument] == typ
                and all(value == ValueFlow("argument", str(leaf), argument, path)
                        for path, leaf, value in projected)):
            return ValueFlow("argument", str(typ), argument)
        return source

    def _defined(self, source):
        from cutlass._mlir import ir

        typ = ir.Type.parse(source.llvm_type)
        if source.kind in ("poison", "undef"):
            if self._leaves(typ):
                raise ValueError(f"Live uninitialized parameter provenance: {source.kind}")
        elif source.kind == "llvm.insertvalue":
            for path, _ in self._leaves(typ):
                self._defined(self._project(source, typ, path))
        else:
            for operand in source.operands:
                self._defined(operand)

    def read(self, value: Any) -> ValueFlow:
        result = self._read(value)
        self._defined(result)
        return result

    def read_parameter(self, value: Any) -> ValueFlow:
        from cutlass._mlir import ir

        result = self._read(value)
        for _, typ, source in parameter_leaves(result, value.type):
            if source.kind == "undef":
                if (not isinstance(typ, ir.IntegerType) or not typ.is_signless
                        or typ.width not in (1, 8, 16, 32, 64)):
                    raise ValueError("Undefined parameter leaves require a supported signless integer type")
            else:
                self._defined(source)
        return result

    def _read(self, value: Any) -> ValueFlow:
        from cutlass._mlir import ir
        from torch._inductor.runtime._cudagraph._compiler.overflow_properties import _flags, _OVERFLOW

        value = _raw_value(value)
        if value in self.cache:
            return self.cache[value]
        if value in self.active:
            raise ValueError("Cyclic argument provenance is unsupported")
        typ = str(value.type)
        if value in self.arguments:
            result = ValueFlow("argument", typ, self.arguments[value])
        else:
            op = value.owner
            if isinstance(op, ir.OpView):
                op = op.operation
            if not isinstance(op, ir.Operation):
                raise ValueError("Non-entry block arguments and merged provenance are unsupported")
            if len(op.results) != 1 or op.regions or op.successors:
                raise ValueError("Unsupported multi-result or control-flow provenance")
            self.active.add(value)
            try:
                attributes = {name: str(op.attributes[name]) for name in op.attributes}
                if op.name in _OVERFLOW:
                    flags = _flags(op)
                    attribute = str(ir.IntegerAttr.get(ir.IntegerType.get_signless(32), flags))
                    if "overflowFlags" in attributes and attributes["overflowFlags"] != attribute:
                        raise ValueError("LLVM overflow attribute differs from its compiler property")
                    if flags:
                        attributes["overflowFlags"] = attribute
                    else:
                        attributes.pop("overflowFlags", None)
                if op.name in ("llvm.udiv", "llvm.sdiv", "llvm.lshr", "llvm.ashr"):
                    if "isExact" in op.attributes and not isinstance(op.attributes["isExact"], ir.UnitAttr):
                        raise ValueError("LLVM exactness requires the original unit attribute")
                attrs = tuple(sorted(attributes.items()))
                if op.name == "llvm.mlir.constant" and not op.operands and set(op.attributes) == {"value"}:
                    attribute = op.attributes["value"]
                    if isinstance(value.type, ir.VectorType) and (
                            not isinstance(attribute, ir.DenseIntElementsAttr) or attribute.type != value.type):
                        raise ValueError("Vector constant requires its exact typed dense integer attribute")
                    result = ValueFlow("constant", typ, value=str(attribute))
                elif op.name == "llvm.mlir.zero" and not op.operands and not op.attributes:
                    result = ValueFlow("zero", typ)
                elif op.name in ("llvm.mlir.poison", "llvm.mlir.undef") and not op.operands and not op.attributes:
                    result = ValueFlow(op.name.removeprefix("llvm.mlir."), typ)
                elif op.name == "llvm.extractvalue" and len(op.operands) == 1 and set(op.attributes) == {"position"}:
                    source = self._read(op.operands[0])
                    path = tuple(op.attributes["position"])
                    if not path or self._at(op.operands[0].type, path) != value.type:
                        raise ValueError("Aggregate extraction changed its exact field type")
                    result = self._project(source, op.operands[0].type, path)
                elif op.name == "llvm.insertvalue" and len(op.operands) == 2 and set(op.attributes) == {"position"}:
                    path = tuple(op.attributes["position"])
                    if (not path or op.operands[0].type != value.type
                            or self._at(value.type, path) != op.operands[1].type):
                        raise ValueError("Aggregate insertion changed its exact field type")
                    result = ValueFlow(op.name, typ, path=path,
                                       operands=tuple(self._read(item) for item in op.operands))
                elif op.name == "llvm.load":
                    result = self._local_load(op)
                elif ((op.name in _UNARY and len(op.operands) == 1)
                      or (op.name in _BINARY and len(op.operands) == 2)
                      or (op.name == "llvm.getelementptr" and op.operands)):
                    result = ValueFlow(op.name, typ, operands=tuple(self._read(item) for item in op.operands), attributes=attrs)
                else:
                    raise ValueError(f"Unresolved or unsupported parameter provenance: {op.name}")
            finally:
                self.active.remove(value)
        result = self._normalize(result, value.type)
        self.cache[value] = result
        return result


def parameter_leaves(source: ValueFlow, actual_type: Any) -> tuple:
    from cutlass._mlir import ir

    if not isinstance(source, ValueFlow) or not isinstance(actual_type, ir.Type):
        raise TypeError("Expected parameter provenance and its actual compiler type")
    if source.llvm_type != str(actual_type):
        raise ValueError("Parameter provenance differs from its actual compiler type")
    reader = _Values(())
    return tuple((path, typ, reader._project(source, actual_type, path))
                 for path, typ in reader._leaves(actual_type))


def _pack(call: Any, calls: list[Any], values: _Values) -> tuple[ParameterFlow, ...]:
    from cutlass._mlir.dialects import llvm

    callee = call.attributes["callee"].value
    array = call.operands[_LAUNCHES[callee][2]]
    allocation = _owner(array, "llvm.alloca")
    count = _integer(allocation.operands[0])
    if count < 0:
        raise ValueError("Negative kernel parameter count")
    _local_allocation(array, count, llvm.PointerType.get())
    _ordered(allocation, call)
    projections = [op for op, operand in _uses(array) if op.name == "llvm.getelementptr" and operand == 0]
    consumers = [(other, _LAUNCHES[other.attributes["callee"].value][2]) for other in calls
                 if other.operands[_LAUNCHES[other.attributes["callee"].value][2]] == array]
    _exact_uses(array, [(op, 0) for op in projections] + consumers)
    if len(projections) != count:
        raise ValueError("Kernel parameter array has missing or extra slot projections")
    parameters = {}
    for projection in projections:
        indices = _gep_indices(projection)
        if (len(indices) != 1 or not 0 <= indices[0] < count or indices[0] in parameters
                or projection.attributes["elem_type"].value != llvm.PointerType.get()):
            raise ValueError("Ambiguous or out-of-range parameter-array index")
        uses = _uses(projection.results[0])
        if len(uses) != 1 or uses[0][0].name != "llvm.store" or uses[0][1] != 1:
            raise ValueError("A parameter-array slot needs exactly one pointer store")
        pointer_store = uses[0][0]
        _plain_memory(pointer_store)
        memory = pointer_store.operands[0]
        stores = [op for op, operand in _uses(memory) if op.name == "llvm.store" and operand == 1]
        if len(stores) != 1:
            raise ValueError("Parameter bytes need one complete dominating writer")
        store = stores[0]
        _plain_memory(store)
        typ = store.operands[0].type
        backing = _local_allocation(memory, 1, typ)
        _exact_uses(memory, [(store, 1), (pointer_store, 0)])
        _ordered(backing, store, pointer_store, call)
        _ordered(allocation, projection, pointer_store, call)
        for consumer, _ in consumers:
            _ordered(pointer_store, consumer)
        index = indices[0]
        parameters[index] = ParameterFlow(index, str(typ), values.read_parameter(store.operands[0]))
    return tuple(parameters[index] for index in range(count))


def analyze_argument_flow(module: Any, function_name: str, *, local_pointer_width: int | None = None) -> ArgumentFlow:
    from cutlass._mlir import ir

    if not isinstance(module, ir.Module) or type(function_name) is not str or not function_name:
        raise TypeError("Expected an explicitly owned LLVM-dialect Module and host function name")
    context = module.context
    with context, ir.raw_values():
        text, bytecode = _snapshot(module)
        if not module.operation.verify():
            raise ValueError("LLVM-dialect Module failed verification")
        _check_host_operations(module, function_name)
        host = _function(module, function_name)
        blocks = list(host.regions[0].blocks)
        host_types = tuple(str(value.type) for value in blocks[0].arguments)
        calls = []
        positions = {}
        for block_index, block in enumerate(blocks):
            for op_index, view in enumerate(block.operations):
                op = view.operation
                if op.regions:
                    raise ValueError("Nested regions in a lowered host function are unsupported")
                if op.name != "llvm.call":
                    continue
                callee = op.attributes.get("callee")
                if callee is None or callee.value not in _HOST_CALLS:
                    raise ValueError("Unknown or indirect host call may contain an unobserved invocation")
                if callee.value in _LAUNCHES:
                    if len(op.operands) != _LAUNCHES[callee.value][0]:
                        raise ValueError("Unsupported kernel-launch calling convention")
                    calls.append(op)
                    positions[op] = (block_index, op_index)
        if not calls:
            raise ValueError("Host function contains no supported kernel invocation")
        binaries, registrations = _registrations(module)
        handles = {record.handle_global: record for record in registrations}
        values = _Values(blocks[0].arguments, local_pointer_width=local_pointer_width)
        launches = []
        for index, call in enumerate(calls):
            callee = call.attributes["callee"].value
            handle = _owner(call.operands[_LAUNCHES[callee][1]], "llvm.load")
            _plain_memory(handle)
            address = _owner(handle.operands[0], "llvm.mlir.addressof")
            name = address.attributes["global_name"].value
            if name not in handles:
                raise ValueError("Launch function lacks an exact binary/symbol registration")
            launches.append(LaunchFlow(index, *positions[call], callee, handles[name], _pack(call, calls, values)))
        if _snapshot(module) != (text, bytecode):
            raise RuntimeError("Argument-flow inspection changed the Module")
    result = ArgumentFlow(module, context, function_name, host_types, binaries, registrations, tuple(launches),
                          sha256(bytecode).hexdigest(), text, bytecode, (module, context), "")
    object.__setattr__(result, "_fingerprint", result._digest())
    result.check()
    return result
