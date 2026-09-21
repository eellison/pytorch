from __future__ import annotations

import ctypes
import struct
import sys
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, TYPE_CHECKING

from .cudagraph_replay_recipe import Guard, IntExpr
from .cutedsl_parameter_analysis import ANALYSIS_VERSION, CuTeParameterTrace, _raw_value, _walk_block

if TYPE_CHECKING:
    import torch
    from cutlass.cutlass_dsl.cuda_jit_executor import CudaDialectJitCompiledFunction


@dataclass(frozen=True)
class _TensorFormal:
    index: int


@dataclass(frozen=True)
class _Condition:
    predicate: int
    left: IntExpr
    right: IntExpr


@dataclass(frozen=True)
class HostRecipe:
    function_name: str
    compiled_hash: str
    trace_sha256: str
    compiler_version: str
    analyzer_version: int
    kernel_symbol: str
    parameter_types: tuple[str, ...]
    argument_sources: tuple[int, ...]
    grid: tuple[IntExpr, IntExpr, IntExpr]
    block: tuple[int, int, int]
    shared: IntExpr
    guards: tuple[Guard, ...]
    shared_limit: int

    def evaluate(self, *, input_extent: int, output_extent: int) -> tuple[tuple[int, int, int], int]:
        arguments = [input_extent, output_extent]
        if any(type(value) is not int or not 1 <= value < 2**31 for value in arguments):
            raise ValueError("Expected positive signed-int32 tensor lengths")
        if input_extent != output_extent:
            raise ValueError("Tensor extent equality guard failed")
        if not all(guard.evaluate(arguments) for guard in self.guards):
            raise ValueError("Host arithmetic or inactive-branch guard failed")
        grid = tuple(expression.evaluate(arguments) for expression in self.grid)
        if any(not 0 < value <= limit for value, limit in zip(grid, (2**31 - 1, 65535, 65535))):
            raise ValueError("Evaluated grid exceeds CUDA limits")
        shared = self.shared.evaluate(arguments)
        if not 0 <= shared <= self.shared_limit:
            raise ValueError("Evaluated shared bytes exceed the selected function limit")
        return grid, shared


def extract_host_recipe(
    compiled: CudaDialectJitCompiledFunction,
    trace: CuTeParameterTrace,
    *,
    shared_limit: int = 49152,
) -> HostRecipe:
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import cute as cute_ir

    if type(trace) is not CuTeParameterTrace:
        raise TypeError("Expected the compilation's CuTeParameterTrace")
    analysis = trace.finish(compiled)
    if not analysis.supported or not analysis.compiled_hash or analysis.analyzer_version != ANALYSIS_VERSION:
        raise ValueError("Expected a supported selected device certificate")
    if type(shared_limit) is not int or not 0 < shared_limit < 2**31:
        raise ValueError("Expected the selected function's dynamic shared-memory limit")
    if (
        len(analysis.kernel_signature) != 2 or analysis.kernel_signature[0] != analysis.kernel_signature[1]
        or tuple((formal.ir_index, formal.read, formal.written, formal.required_alignment)
                 for formal in analysis.pointer_formals) != ((0, True, False, 16), (1, False, True, 16))
        or compiled.kernel_info[analysis.kernel_symbol] or compiled.kernel_extra_args
    ):
        raise ValueError("Expected one read-only input and one write-only output without extra launch metadata")
    shape_ops = {
        "cute.get_shape", "cute.get_leaves", "cute.to_int_tuple",
        "cute.get_scalars", "cute.make_int_tuple", "cute.size",
    }
    allowed = shape_ops | {
        "cute.get_iter", "cute.get_layout", "cute.make_tile", "cute.ceil_div",
        "cute.kernel_smem_size", "arith.constant", "arith.cmpi", "scf.if", "scf.yield",
        "cute.print", "cuda.launch_cfg.create", "cuda.launch_ex", "cuda.cast",
        "cuda.return_if_error", "func.return",
        "cute.tuple_add", "cute.tuple_mul", "arith.extsi",
        "cuda.launch_cfg.programmatic_stream_serialization_allowed", "cuda.launch_cfg.cooperative",
    }
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(trace.trace_mlir)
        hosts = [view.operation for view in module.body.operations
                 if view.operation.name == "func.func"
                 and view.operation.attributes["sym_name"].value == compiled.function_name]
        gpu = [view.operation for view in module.body.operations if view.operation.name == "gpu.module"]
        if len(hosts) != 1 or len(gpu) != 1:
            raise ValueError("Expected the selected host function and GPU module")
        host = hosts[0]
        if len(host.regions) != 1 or len(host.regions[0].blocks) != 1:
            raise ValueError("Host control-flow blocks are unsupported")
        block = host.regions[0].blocks[0]
        argument_types = list(block.arguments.types)
        if len(argument_types) != 3 or tuple(str(t) for t in argument_types[:2]) != analysis.kernel_signature:
            raise ValueError("Host tensor formals do not match the certified device types")
        expected_type = cute.runtime.make_fake_tensor(
            cutlass.Float32, (cute.sym_int(32, symbol="N"),), stride=(1,), assumed_align=16,
        ).mlir_type
        if argument_types[:2] != [expected_type, expected_type]:
            raise ValueError("Expected rank-one dynamic-int32 float32 memrefs with stride one and alignment 16")
        kernel = gpu[0].regions[0].blocks[0].operations[0].operation
        required = kernel.attributes.get("nvvm.reqntid")
        if not isinstance(required, ir.DenseI32ArrayAttr):
            raise ValueError("Expected a certified fixed CUDA block")
        threads = tuple(required)
        if len(threads) != 3 or threads[1:] != (1, 1) or not 1 <= threads[0] <= 1024:
            raise ValueError("Expected a one-dimensional fixed CUDA block")
        expected_symbol = [gpu[0].attributes["sym_name"].value, analysis.kernel_symbol]
        operations = list(_walk_block(block))
        for op in operations:
            if op.name not in allowed:
                raise ValueError(f"Unsupported host operation {op.name}")
            if op.successors or (op.regions and op.name != "scf.if"):
                raise ValueError("Unsupported host control flow")
            attrs = (
                {"value"} if op.name == "arith.constant"
                else {"predicate"} if op.name == "arith.cmpi"
                else {"mode"} if op.name == "cute.size"
                else {"kernel_name"} if op.name == "cute.kernel_smem_size"
                else {"maxNumAttrs"} if op.name == "cuda.launch_cfg.create"
                else {"assume_kernel_attr", "callee"} if op.name == "cuda.launch_ex"
                else {"fmt", "format", "stderr"} if op.name == "cute.print"
                else set()
            )
            if set(op.attributes) - attrs:
                raise ValueError(f"Unsupported attributes on {op.name}: {set(op.attributes) - attrs}")
        root_ops = [view.operation for view in block.operations]
        if not root_ops or root_ops[-1].name != "func.return":
            raise ValueError("Expected one final host return")
        environment = {}
        stream, status, config = object(), object(), object()
        grid = None
        shared = None
        guards = []
        sources = None
        flags = set()
        launches = 0
        with ir.InsertionPoint(root_ops[-1]):
            arguments = [_raw_value(value) for value in block.arguments]
            environment.update(zip(arguments, (_TensorFormal(0), _TensorFormal(1), stream)))
            for op in root_ops:
                operands = [_raw_value(value) for value in op.operands]
                args = [environment[value] for value in operands]
                result = None
                if op.name in ("cute.get_iter", "cute.get_layout"):
                    if len(args) != 1 or type(args[0]) is not _TensorFormal:
                        raise ValueError("Unproven host tensor projection")
                    result = ("pointer", args[0].index) if op.name == "cute.get_iter" else IntExpr("boxed", args[0].index)
                elif op.name == "cute.make_int_tuple" and not args:
                    typ = op.results.types[0]
                    if not isinstance(typ, cute_ir.IntTupleType) or not cute_ir.is_static(typ):
                        raise ValueError("Expected a typed static integer tuple")
                    value = cute_ir.unpack_x_tuple(typ, [])
                    if type(value) is not int:
                        raise ValueError("Only scalar typed tuple constants are supported")
                    result = IntExpr("constant", value)
                elif op.name in ("cute.tuple_add", "cute.tuple_mul"):
                    if len(args) != 2 or any(type(value) is not IntExpr for value in args):
                        raise ValueError("Expected scalar typed tuple arithmetic")
                    result = IntExpr("add" if op.name == "cute.tuple_add" else "mul", args=tuple(args))
                elif op.name == "arith.extsi":
                    if (
                        len(args) != 1 or type(args[0]) is not IntExpr
                        or [value.type for value in operands] != [ir.IntegerType.get_signless(32)]
                        or list(op.results.types) != [ir.IntegerType.get_signless(64)]
                    ):
                        raise ValueError("Only signed int32 to int64 extension is supported")
                    result = args[0]
                elif op.name in shape_ops:
                    if len(args) != 1 or type(args[0]) is not IntExpr:
                        raise ValueError("Only one-dimensional host shape arithmetic is supported")
                    if op.name == "cute.size":
                        mode = op.attributes.get("mode")
                        if not isinstance(mode, ir.DenseI32ArrayAttr) or len(mode):
                            raise ValueError("Expected full-shape size arithmetic")
                    result = args[0]
                elif op.name == "cute.make_tile":
                    if args or tuple(op.results.types) != (cute_ir.TileType.get(f"{threads[0]}:1"),):
                        raise ValueError("Static host tile must match the certified block")
                    result = IntExpr("constant", threads[0])
                elif op.name == "cute.ceil_div":
                    if len(args) != 2 or any(type(value) is not IntExpr for value in args):
                        raise ValueError("Unexpected ceiling-division operands")
                    if args[1] != IntExpr("constant", threads[0]):
                        raise ValueError("Expected the certified static tile divisor")
                    divisor = IntExpr("constant", -threads[0])
                    result = IntExpr("neg", args=(IntExpr("floordiv", args=(args[0], divisor)),))
                elif op.name == "cute.kernel_smem_size":
                    if args or op.attributes["kernel_name"].value != expected_symbol:
                        raise ValueError("Shared-memory query changed the selected kernel")
                    result = IntExpr("constant", 0)
                elif op.name == "arith.constant":
                    attr = op.attributes["value"]
                    if args or not isinstance(attr, ir.IntegerAttr):
                        raise ValueError("Expected an integer host constant")
                    result = IntExpr("constant", attr.value)
                elif op.name == "arith.cmpi":
                    if len(args) != 2 or any(type(value) is not IntExpr for value in args):
                        raise ValueError("Expected a scalar signed comparison")
                    predicate = op.attributes["predicate"].value
                    if predicate not in (2, 4):
                        raise ValueError("Unsupported host comparison")
                    result = _Condition(predicate, *args)
                elif op.name == "scf.if":
                    if len(args) != 1 or type(args[0]) is not _Condition or len(op.results) or len(op.regions) != 2:
                        raise ValueError("Only guarded inactive host diagnostics are supported")
                    branch = op.regions[0]
                    if len(branch.blocks) != 1 or branch.blocks[0].arguments:
                        raise ValueError("Unsupported host diagnostic block")
                    body = [view.operation for view in branch.blocks[0].operations]
                    diagnostic_ops = shape_ops | {
                        "cute.get_iter", "cute.get_layout", "cute.tuple_add", "cute.tuple_mul",
                        "arith.constant", "arith.extsi", "cute.print", "scf.yield",
                    }
                    if (
                        not body or body[-1].name != "scf.yield"
                        or sum(item.name == "cute.print" for item in body) != 1
                        or any(item.name not in diagnostic_ops or item.regions for item in body)
                    ):
                        raise ValueError("Host branching with other effects is unsupported")
                    if body[-1].operands or any(item.results for item in body if item.name in ("cute.print", "scf.yield")):
                        raise ValueError("Host diagnostic cannot yield values")
                    if len(op.regions) == 2:
                        other = op.regions[1]
                        if other.blocks and (len(other.blocks) != 1 or [v.operation.name for v in other.blocks[0].operations] != ["scf.yield"]):
                            raise ValueError("Unexpected active host branch")
                    condition = args[0]
                    guards.append(Guard("ge" if condition.predicate == 2 else "le", condition.left, condition.right))
                elif op.name == "cuda.launch_cfg.create":
                    if grid is not None or len(args) != 8 or args[-1] is not stream:
                        raise ValueError("Expected one launch on the original host stream")
                    if args[:3] != [IntExpr("constant", value) for value in threads] or type(args[3]) is not IntExpr:
                        raise ValueError("Launch block or shared-byte expression is unsupported")
                    if any(type(value) is not IntExpr for value in args[4:7]):
                        raise ValueError("Expected symbolic integer grid dimensions")
                    grid = tuple(args[4:7])
                    shared = args[3]
                    result = config
                elif op.name.startswith("cuda.launch_cfg."):
                    if args != [config, IntExpr("constant", 0)] or launches or op.name in flags:
                        raise ValueError("Only explicit disabled cooperative/PDL attributes are supported")
                    flags.add(op.name)
                elif op.name == "cuda.launch_ex":
                    if launches or not args or args[0] is not config or op.attributes["callee"].value != expected_symbol:
                        raise ValueError("Unexpected kernel or configuration in host launch")
                    if len(args) != 3 or any(type(value) is not _TensorFormal for value in args[1:]):
                        raise ValueError("Kernel parameters must be original complete tensor formals")
                    sources = tuple(value.index for value in args[1:])
                    if sources != (0, 1):
                        raise ValueError("Host/device aggregate argument mapping differs")
                    launches += 1
                    result = status
                elif op.name == "cuda.cast":
                    if args != [status]:
                        raise ValueError("Expected the launch status cast")
                    result = status
                elif op.name == "cuda.return_if_error":
                    if args != [status]:
                        raise ValueError("Expected normal launch status propagation")
                elif op.name == "func.return":
                    if args != [IntExpr("constant", 0)]:
                        raise ValueError("Expected a successful host return")
                else:
                    raise ValueError(f"Unsupported active host effect {op.name}")
                results = [_raw_value(value) for value in op.results]
                if results:
                    if len(results) != 1 or result is None:
                        raise ValueError("Unsupported host result arity")
                    environment[results[0]] = result
                    typ = results[0].type
                    if type(result) is IntExpr and isinstance(typ, (ir.IntegerType, cute_ir.IntTupleType)):
                        width = typ.width
                        if width not in (32, 64):
                            raise ValueError("Only signed int32/int64 host arithmetic is supported")
                        guards.extend((Guard("ge", result, IntExpr("constant", -(2 ** (width - 1)))),
                                       Guard("le", result, IntExpr("constant", 2 ** (width - 1) - 1))))
        if launches != 1 or grid is None or len(flags) != 2:
            raise ValueError("Incomplete host launch recipe")
    return HostRecipe(
        compiled.function_name, analysis.compiled_hash, sha256(trace.trace_mlir.encode()).hexdigest(),
        analysis.compiler_version, analysis.analyzer_version, analysis.kernel_symbol, analysis.kernel_signature,
        sources, grid, threads, shared, tuple(guards), shared_limit,
    )


@dataclass(frozen=True)
class KernelABI:
    layout: tuple[tuple[int, int], ...]
    pointer_fields: tuple[tuple[int, int], ...]
    scalar_fields: tuple[tuple[int, int, str], ...]
    padding_fields: tuple[tuple[int, int, int], ...]


def check_lowered_abi(compiled: CudaDialectJitCompiledFunction) -> KernelABI:
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm

    functions = [view.operation for view in compiled.ir_module.body.operations
                 if view.operation.name == "llvm.func"
                 and view.operation.attributes["sym_name"].value == compiled.function_name]
    if len(functions) != 1:
        raise ValueError("Expected the exact selected host function")
    host = functions[0]
    arguments = list(host.regions[0].blocks[0].arguments)
    types = list(host.regions[0].blocks[0].arguments.types)
    if len(types) != 3 or types[0] != types[1] or not isinstance(types[2], llvm.PointerType):
        raise ValueError("Expected two equal tensor aggregates and a stream")
    aggregate = types[0]
    if not isinstance(aggregate, llvm.StructType) or aggregate.packed or len(aggregate.body) != 2:
        raise ValueError("Unsupported memref aggregate")
    pointer, layout = aggregate.body
    if not isinstance(pointer, llvm.PointerType) or pointer.address_space != 1:
        raise ValueError("Expected global device pointer as first aggregate field")
    if not isinstance(layout, llvm.StructType) or layout.packed or len(layout.body) != 2:
        raise ValueError("Unsupported dynamic layout aggregate")
    extent, empty = layout.body
    if (
        not isinstance(extent, ir.IntegerType) or extent.width != 32
        or not isinstance(empty, llvm.StructType) or empty.packed or len(empty.body)
        or ctypes.sizeof(ctypes.c_void_p) != 8 or sys.byteorder != "little"
    ):
        raise ValueError("Expected one int32 extent and no dynamic stride")
    operations = [view.operation for block in host.regions[0].blocks for view in block.operations]
    calls = [op for op in operations if op.name == "llvm.call"
             and op.attributes.get("callee") is not None
             and op.attributes["callee"].value == "_cudaLaunchKernelEx"]
    if len(calls) != 1 or len(calls[0].operands) != 3:
        raise ValueError("Expected one selected CUDA extended launch")
    array = calls[0].operands[2]
    allocate_array = array.owner
    if (
        allocate_array.name != "llvm.alloca"
        or not isinstance(allocate_array.attributes["elem_type"].value, llvm.PointerType)
        or allocate_array.operands[0].owner.attributes["value"].value != 2
    ):
        raise ValueError("Expected a two-entry kernel-argument pointer array")
    projections = [op for op in operations if op.name == "llvm.getelementptr" and op.operands[0] == array]
    if len(projections) != 2:
        raise ValueError("Unexpected uses of the kernel argument array")
    seen = set()
    for projection in projections:
        if len(projection.operands) != 2:
            raise ValueError("Expected explicit constant argument index")
        index_op = projection.operands[1].owner
        if index_op.name != "llvm.mlir.constant":
            raise ValueError("Dynamic CUDA parameter indexing is unsupported")
        index = index_op.attributes["value"].value
        if index not in (0, 1) or index in seen:
            raise ValueError("Expected one pointer-array slot per tensor formal")
        seen.add(index)
        stores = [op for op in operations if op.name == "llvm.store" and op.operands[1] == projection.results[0]]
        if len(stores) != 1:
            raise ValueError("Ambiguous CUDA parameter slot packing")
        memory = stores[0].operands[0]
        allocation = memory.owner
        if allocation.name != "llvm.alloca" or allocation.attributes["elem_type"].value != aggregate:
            raise ValueError("CUDA parameter does not contain the complete tensor aggregate")
        values = [op for op in operations if op.name == "llvm.store" and op.operands[1] == memory]
        if len(values) != 1 or values[0].operands[0] != arguments[index]:
            raise ValueError("Host-to-device formal order or aggregate bytes changed")
    return KernelABI(
        ((0, 16), (16, 16)), ((0, 0), (1, 0)),
        ((0, 8, "i32"), (1, 8, "i32")), ((0, 12, 4), (1, 12, 4)),
    )


def runtime_argument(tensor: torch.Tensor, expected_type: str) -> tuple[Any, bytes]:
    import torch
    import cutlass.cute as cute
    from cutlass._mlir import ir

    if (
        type(tensor) is not torch.Tensor or tensor.dtype != torch.float32 or tensor.device.type != "cuda"
        or tensor.ndim != 1 or tensor.stride() != (1,) or not 1 <= tensor.numel() < 2**31
    ):
        raise ValueError("Expected a rank-one CUDA float32 tensor with a signed-int32 extent")
    argument = cute.runtime.from_dlpack(tensor, assumed_align=16, use_32bit_stride=True).mark_layout_dynamic(leading_dim=0)
    with ir.Context():
        if argument.mlir_type != ir.Type.parse(expected_type):
            raise ValueError(f"Runtime tensor type {argument.mlir_type} differs from selected dynamic memref")
    pointers = argument.__c_pointers__()
    if len(pointers) != 1:
        raise ValueError("Expected one runtime aggregate descriptor")
    raw = ctypes.string_at(pointers[0], 16)
    pointer, extent = struct.unpack("=Qi", raw[:12])
    if pointer != tensor.data_ptr() or extent != tensor.numel():
        raise ValueError("Runtime memref descriptor differs from typed pointer/extent fields")
    return argument, raw
