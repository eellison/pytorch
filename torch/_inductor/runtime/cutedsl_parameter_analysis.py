"""Conservative device-kernel analysis during CuTe trace finalization.

Use CuTeParameterTrace as cute.compile's trace_finalize_hooks, then finish it
with that compilation's result. Supported certificates describe only typed
device-formal uses. Host argument provenance, CUDA ABI mapping, launch behavior,
aliases and lifetimes still require independent validation before rebinding.
Integers or compile-time constants encoding addresses are outside this contract.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from hashlib import sha256
from importlib.metadata import version
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from cutlass._mlir import ir
    from cutlass.base_dsl.jit_executor import JitCompiledFunction


ANALYSIS_VERSION = 3
_TRACE_ATTR = "torch.cudagraph.parameter_trace_sha256"
_VERSION_ATTR = "torch.cudagraph.parameter_analysis_version"
_FLOAT_OPS = {"arith.addf", "arith.subf", "arith.mulf"}
_INT_OPS = {"arith.addi", "arith.subi", "arith.muli"}
_INDEX_OPS = {
    f"nvvm.read.ptx.sreg.{register}.{axis}"
    for register in ("tid", "ctaid", "ntid")
    for axis in ("x", "y", "z")
} | {"nvvm.read.ptx.sreg.dynamic.smem.size"}
_ALLOWLIST = _FLOAT_OPS | _INT_OPS | _INDEX_OPS | {
    "arith.constant",
    "arith.cmpi",
    "arith.sitofp",
    "cute.get_iter",
    "cute.get_layout",
    "cute.make_coord",
    "cute.memref.load",
    "cute.memref.store",
    "cuda.return",
}


@dataclass(frozen=True)
class PointerFormal:
    ir_index: int
    ir_type: str
    required_alignment: int
    read: bool
    written: bool


@dataclass(frozen=True)
class KernelAnalysis:
    compiled_hash: str
    trace_sha256: str
    compiler_version: str
    kernel_symbol: str
    kernel_signature: tuple[str, ...]
    supported: bool
    reason: str
    pointer_formals: tuple[PointerFormal, ...]
    operations: tuple[str, ...]
    analyzer_version: int
    cuda_abi_verified: bool
    host_mapping_verified: bool


class _Unsupported(ValueError):
    pass


def _numeric_type(typ: ir.Type) -> bool:
    from cutlass._mlir import ir

    return (
        isinstance(typ, ir.IntegerType) and typ.width in (1, 8, 16, 32, 64)
    ) or isinstance(typ, (ir.F16Type, ir.BF16Type, ir.F32Type, ir.F64Type))


def _raw_value(value: Any) -> ir.Value:
    from cutlass._mlir import ir

    return ir.Value(value if isinstance(value, ir.Value) else value.value)


def _walk_block(block: ir.Block) -> Iterator[ir.Operation]:
    for view in block.operations:
        op = view.operation
        yield op
        for region in op.regions:
            for child in region.blocks:
                yield from _walk_block(child)


def _inspect_kernel(kernel: ir.Operation) -> tuple[PointerFormal, ...]:
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import cute

    if len(kernel.regions) != 1 or len(kernel.regions[0].blocks) != 1:
        raise _Unsupported("expected one device entry block")
    block = kernel.regions[0].blocks[0]
    root_ops = [view.operation for view in block.operations]
    if not root_ops or root_ops[-1].name != "cuda.return":
        raise _Unsupported("expected a CUDA kernel return")
    attrs = kernel.attributes
    allowed_attrs = {
        "sym_name", "function_type", "cute.kernel", "gpu.kernel", "cu_attrs",
        "nvvm.reqntid", "smem.partition_num", "arg_attrs",
    }
    if set(attrs) - allowed_attrs:
        raise _Unsupported(f"unsupported kernel or argument attributes: {sorted(set(attrs) - allowed_attrs)}")
    signature = attrs["function_type"].value
    if not isinstance(signature, ir.FunctionType) or signature.results:
        raise _Unsupported("expected a void device kernel")
    arg_types = list(block.arguments.types)
    if list(signature.inputs) != arg_types:
        raise _Unsupported("device signature and block argument types differ")
    if "arg_attrs" in attrs and (
        not isinstance(attrs["arg_attrs"], ir.ArrayAttr)
        or len(attrs["arg_attrs"]) != len(arg_types)
        or any(not isinstance(attr, ir.DictAttr) or len(attr) for attr in attrs["arg_attrs"])
    ):
        raise _Unsupported("unsupported device argument attributes")
    for typ in arg_types:
        if isinstance(typ, cute.MemRefType):
            if typ.address_space != 1 or typ.is_swizzled or not _numeric_type(typ.value_type):
                raise _Unsupported("unsupported device tensor type")
        elif not _numeric_type(typ):
            raise _Unsupported("unsupported device formal type")

    shape_ops = {
        "cute.get_shape", "cute.get_leaves", "cute.to_int_tuple",
        "cute.get_scalars", "cute.make_int_tuple", "cute.size",
    }
    metadata_types = (cute.CoordType, cute.LayoutType, cute.ShapeType, cute.IntTupleType)
    # Preflight every branch before retrieving SSA values: CuTe's casters build IR.
    for op in _walk_block(block):
        if op.name not in _ALLOWLIST | shape_ops | {"scf.if", "scf.yield"}:
            raise _Unsupported(f"unsupported operation {op.name}")
        if op.successors:
            raise _Unsupported("unstructured control flow is unsupported")
        if op.name == "scf.if":
            if len(op.regions) != 2 or any(len(region.blocks) != 1 for region in op.regions):
                raise _Unsupported("expected two single-block conditional branches")
            for region in op.regions:
                child = region.blocks[0]
                children = list(child.operations)
                if child.arguments or not children or children[-1].operation.name != "scf.yield":
                    raise _Unsupported("unsupported conditional block arguments or terminator")
        elif op.regions:
            raise _Unsupported(f"unsupported control flow {op.name}")
        allowed = (
            {"value"} if op.name == "arith.constant"
            else {"fastmath"} if op.name in _FLOAT_OPS
            else {"overflowFlags"} if op.name in _INT_OPS
            else {"predicate"} if op.name == "arith.cmpi"
            else {"mode"} if op.name == "cute.size"
            else set()
        )
        if set(op.attributes) - allowed:
            raise _Unsupported(f"unsupported attributes on {op.name}")
        for typ in op.results.types:
            if _numeric_type(typ) or isinstance(typ, metadata_types):
                continue
            if (
                isinstance(typ, cute.PtrType)
                and typ.address_space == 1
                and not typ.is_swizzled
                and _numeric_type(typ.value_type)
            ):
                continue
            if op.name == "scf.if" and isinstance(typ, cute.MemRefType) and typ in arg_types:
                continue
            raise _Unsupported(f"unsupported result type on {op.name}")

    origins: dict[ir.Value, int] = {}
    alignments: dict[int, int] = {}
    reads: set[int] = set()
    writes: set[int] = set()
    # Keep caster-created pure operations attached to live trace IR, never detached.
    with ir.InsertionPoint(root_ops[-1]):
        for index, value in enumerate(block.arguments):
            if not isinstance(arg_types[index], cute.MemRefType):
                continue
            pointer_type = value.iterator.type
            if (
                not isinstance(pointer_type, cute.PtrType)
                or pointer_type.address_space != 1
                or pointer_type.is_swizzled
                or pointer_type.value_type != arg_types[index].value_type
                or not isinstance(value.layout.type, cute.LayoutType)
            ):
                raise _Unsupported("expected an ordinary global tensor")
            alignment = pointer_type.alignment
            if alignment <= 0 or alignment & (alignment - 1):
                raise _Unsupported("unsupported pointer alignment")
            origins[_raw_value(value)] = index
            alignments[index] = alignment

    def inspect_block(current: ir.Block) -> list[ir.Value]:
        ops = [view.operation for view in current.operations]
        with ir.InsertionPoint(ops[-1]):
            for op in ops:
                args = [_raw_value(value) for value in op.operands]
                result_types = list(op.results.types)
                if op.name == "arith.constant":
                    attr = op.attributes.get("value")
                    if (
                        args or len(result_types) != 1
                        or not isinstance(attr, (ir.IntegerAttr, ir.FloatAttr))
                        or attr.type != result_types[0]
                    ):
                        raise _Unsupported("unsupported numeric constant")
                elif op.name in _FLOAT_OPS | _INT_OPS | {"arith.cmpi"}:
                    if (
                        len(args) != 2 or len(result_types) != 1
                        or any(not _numeric_type(value.type) for value in args)
                        or not _numeric_type(result_types[0])
                    ):
                        raise _Unsupported(f"unsupported numeric operation {op.name}")
                elif op.name == "arith.sitofp":
                    if (
                        len(args) != 1 or len(result_types) != 1
                        or not isinstance(args[0].type, ir.IntegerType) or args[0].type.width != 32
                        or not isinstance(result_types[0], ir.F32Type)
                    ):
                        raise _Unsupported("expected signed int32 to float32 conversion")
                elif op.name in _INDEX_OPS:
                    if args or len(result_types) != 1 or not isinstance(result_types[0], ir.IntegerType):
                        raise _Unsupported("unsupported thread-index operation")
                elif op.name in ("cute.get_iter", "cute.get_layout"):
                    if len(args) != 1 or args[0] not in origins or len(result_types) != 1:
                        raise _Unsupported("unsupported tensor provenance")
                    expected = cute.PtrType if op.name == "cute.get_iter" else cute.LayoutType
                    if not isinstance(result_types[0], expected):
                        raise _Unsupported("unsupported tensor projection")
                    if op.name == "cute.get_iter" and result_types[0].alignment != alignments[origins[args[0]]]:
                        raise _Unsupported("unaccounted pointer alignment assumption")
                elif op.name in shape_ops:
                    if op.name == "cute.size":
                        mode = op.attributes.get("mode")
                        if not isinstance(mode, ir.DenseI32ArrayAttr) or len(mode):
                            raise _Unsupported("only full-shape size queries are supported")
                    if not args or not result_types:
                        raise _Unsupported("expected shape-only operands and results")
                    if any(not isinstance(value.type, metadata_types + (ir.IntegerType,)) for value in args):
                        raise _Unsupported("shape operation uses a pointer or tensor")
                    if any(not isinstance(typ, metadata_types + (ir.IntegerType,)) for typ in result_types):
                        raise _Unsupported("shape operation produces a pointer or tensor")
                elif op.name == "cute.make_coord":
                    if (
                        len(result_types) != 1 or not isinstance(result_types[0], cute.CoordType)
                        or any(not isinstance(value.type, ir.IntegerType) for value in args)
                    ):
                        raise _Unsupported("unsupported coordinate construction")
                elif op.name in ("cute.memref.load", "cute.memref.store"):
                    is_load = op.name == "cute.memref.load"
                    if (
                        len(args) != (2 if is_load else 3)
                        or args[0] not in origins
                        or not isinstance(args[1].type, cute.CoordType)
                        or (is_load and result_types != [args[0].type.value_type])
                        or (not is_load and (result_types or args[2].type != args[0].type.value_type))
                    ):
                        raise _Unsupported("unsupported tensor load/store provenance")
                    (reads if is_load else writes).add(origins[args[0]])
                elif op.name == "scf.if":
                    if len(args) != 1 or not isinstance(args[0].type, ir.IntegerType) or args[0].type.width != 1:
                        raise _Unsupported("expected a scalar boolean condition")
                    branches = [inspect_block(region.blocks[0]) for region in op.regions]
                    if any([value.type for value in values] != result_types for values in branches):
                        raise _Unsupported("conditional yield types differ")
                    for index, typ in enumerate(result_types):
                        if isinstance(typ, cute.MemRefType):
                            origin = origins.get(branches[0][index])
                            if origin is None or origins.get(branches[1][index]) != origin:
                                raise _Unsupported("conditional tensor has ambiguous pointer provenance")
                            origins[_raw_value(op.results[index])] = origin
                        elif not _numeric_type(typ):
                            raise _Unsupported("unsupported conditional result")
                elif op.name == "scf.yield":
                    if result_types:
                        raise _Unsupported("conditional yield cannot produce results")
                    return args
                elif op.name == "cuda.return":
                    if args or result_types:
                        raise _Unsupported("kernel must return no values")
            return []

    inspect_block(block)
    if not reads or not writes:
        raise _Unsupported("expected an ordinary load/compute/store kernel")
    return tuple(
        PointerFormal(index, str(arg_types[index]), alignment, index in reads, index in writes)
        for index, alignment in alignments.items()
    )


class CuTeParameterTrace:
    """A single-compilation trace hook producing a device-local certificate."""

    def __init__(self) -> None:
        self.trace_mlir = ""
        self._gpu_module = ""
        self._function_name = ""
        self._calls = 0
        self._analysis = KernelAnalysis(
            "", "", "", "", (), False, "missing CuTe trace", (), (),
            ANALYSIS_VERSION, False, False,
        )

    def __call__(self, owner: Any, module: ir.Module, function_name: str) -> None:
        self._calls += 1
        result = self._analysis
        try:
            from cutlass._mlir import ir
            from cutlass.cutlass_dsl.cutlass import CuTeDSL

            if self._calls != 1 or type(owner) is not CuTeDSL:
                raise _Unsupported("expected one ordinary CuTe compilation")
            if _TRACE_ATTR in module.operation.attributes or _VERSION_ATTR in module.operation.attributes:
                raise _Unsupported("trace already carries a parameter certificate")
            self._function_name = function_name
            result = replace(result, compiler_version=version("nvidia-cutlass-dsl"))
            gpu_modules = [op for op in module.body.operations if op.operation.name == "gpu.module"]
            if len(gpu_modules) != 1:
                raise _Unsupported("expected one GPU module")
            gpu_module = gpu_modules[0].operation
            self._gpu_module = gpu_module.attributes["sym_name"].value
            if len(gpu_module.regions) != 1 or len(gpu_module.regions[0].blocks) != 1:
                raise _Unsupported("unsupported GPU module structure")
            kernels = list(gpu_module.regions[0].blocks[0].operations)
            if len(kernels) != 1 or kernels[0].operation.name != "cuda.kernel":
                raise _Unsupported("expected one device kernel without helpers or globals")
            kernel = kernels[0].operation
            body = [op for region in kernel.regions for block in region.blocks for op in _walk_block(block)]
            result = replace(
                result,
                kernel_symbol=kernel.attributes["sym_name"].value,
                kernel_signature=tuple(str(typ) for typ in kernel.attributes["function_type"].value.inputs),
                operations=tuple(sorted({op.name for op in body})),
            )
            hooks = tuple(owner._trace_finalize_hooks) + tuple(owner._scoped_trace_finalize_hooks.get())
            if hooks != (self,):
                raise _Unsupported("additional trace hooks could change the certified kernel")
            with module.context, module.operation.location:
                formals = _inspect_kernel(kernel)
            if not module.operation.verify():
                raise _Unsupported("CuTe module verification failed")
            result = replace(
                result, supported=True, reason="supported device-kernel-local subset",
                pointer_formals=formals,
            )
        except Exception as exc:
            result = replace(result, supported=False, reason=f"{type(exc).__name__}: {exc}", pointer_formals=())
        try:
            self.trace_mlir = str(module)
            digest = sha256(self.trace_mlir.encode()).hexdigest()
            with module.context:
                module.operation.attributes[_TRACE_ATTR] = ir.StringAttr.get(digest)
                module.operation.attributes[_VERSION_ATTR] = ir.IntegerAttr.get(
                    ir.IntegerType.get_signless(32), ANALYSIS_VERSION
                )
            self._analysis = replace(result, trace_sha256=digest)
        except Exception as exc:
            self._analysis = replace(result, supported=False, reason=f"{type(exc).__name__}: {exc}", pointer_formals=())

    def finish(self, compiled: JitCompiledFunction) -> KernelAnalysis:
        result = self._analysis
        try:
            from cutlass._mlir import ir
            from cutlass.cutlass_dsl.cuda_jit_executor import CudaDialectJitCompiledFunction

            if self._calls != 1 or type(compiled) is not CudaDialectJitCompiledFunction:
                raise _Unsupported("expected the selected non-FFI CuTe compilation")
            if compiled.load_from_binary or compiled.total_added_arguments:
                raise _Unsupported("loaded artifacts or added workspace arguments are unsupported")
            if compiled.function_name != self._function_name or tuple(compiled.kernel_info) != (result.kernel_symbol,):
                raise _Unsupported("selected kernel identity differs from the trace")
            attrs = compiled.ir_module.operation.attributes
            if (
                not isinstance(attrs.get(_TRACE_ATTR), ir.StringAttr)
                or attrs[_TRACE_ATTR].value != result.trace_sha256
                or not isinstance(attrs.get(_VERSION_ATTR), ir.IntegerAttr)
                or attrs[_VERSION_ATTR].value != ANALYSIS_VERSION
                or result.trace_sha256 != sha256(self.trace_mlir.encode()).hexdigest()
            ):
                raise _Unsupported("missing or mismatched selected trace identity")
            binaries = [
                op.operation.attributes.get("value")
                for op in compiled.ir_module.body.operations
                if op.operation.name == "llvm.mlir.global"
                and op.operation.attributes.get("sym_name").value == f"{self._gpu_module}_binary"
            ]
            if len(binaries) != 1 or not isinstance(binaries[0], ir.StringAttr):
                raise _Unsupported("missing selected embedded CUDA binary")
            binary = binaries[0].value_bytes
            if not binary:
                raise _Unsupported("empty selected CUDA binary")
            return replace(result, compiled_hash=sha256(binary).hexdigest())
        except Exception as exc:
            return replace(result, supported=False, reason=f"{type(exc).__name__}: {exc}", pointer_formals=())
