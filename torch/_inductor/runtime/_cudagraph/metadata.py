"""Inherited compiler inputs attached to the original terminal-traced wrapper."""

from dataclasses import dataclass
from inspect import CO_VARARGS, CO_VARKEYWORDS
from types import CodeType, FunctionType, MethodType
from weakref import ref, ReferenceType

import torch
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import InputContract, IntegerRange, TensorInput
from torch._inductor.runtime.cudagraph_arg_mapping import grid_expression_inputs


class MetadataDeclined(ValueError):
    pass


@dataclass(frozen=True)
class TerminalMetadata:
    inputs: InputContract
    saved_input_indices: tuple[int, ...] = ()


@dataclass(frozen=True, eq=False)
class _Attachment:
    function: ReferenceType
    code: CodeType
    metadata: TerminalMetadata


def _check_metadata(metadata):
    if type(metadata) is not TerminalMetadata or type(metadata.inputs) is not InputContract:
        raise MetadataDeclined("Expected inherited terminal compiler inputs")
    inputs = metadata.inputs
    if (type(inputs.kinds) is not tuple or not inputs.kinds
            or any(type(kind) is not str or kind not in ("integer", "tensor") for kind in inputs.kinds)
            or type(inputs.tensor_inputs) is not tuple or type(inputs.integer_ranges) is not tuple
            or type(inputs.device_index) is not int or inputs.device_index < 0):
        raise MetadataDeclined("Malformed terminal input contract")
    integers = tuple(index for index, kind in enumerate(inputs.kinds) if kind == "integer")
    tensors = tuple(index for index, kind in enumerate(inputs.kinds) if kind == "tensor")
    if (not tensors
            or any(type(row) is not IntegerRange or type(row.index) is not int
                   or type(row.lower) is not int or row.lower < 1
                   or row.upper is not None and (type(row.upper) is not int or row.upper < row.lower)
                   for row in inputs.integer_ranges)
            or tuple(row.index for row in inputs.integer_ranges) != integers
            or any(type(row) is not TensorInput or type(row.index) is not int
                   or type(row.dtype) is not torch.dtype or type(row.size) is not tuple
                   or type(row.stride) is not tuple or len(row.size) != len(row.stride)
                   for row in inputs.tensor_inputs)
            or tuple(row.index for row in inputs.tensor_inputs) != tensors):
        raise MetadataDeclined("Terminal input facts do not cover exact boxed slots")
    saved = metadata.saved_input_indices
    if (type(saved) is not tuple or any(type(index) is not int or index not in tensors for index in saved)
            or len(saved) != len(set(saved))):
        raise MetadataDeclined("Saved inputs must name distinct Tensor boxed slots")
    for tensor in inputs.tensor_inputs:
        for value in (*tensor.size, *tensor.stride):
            if type(value) is int and value >= 0:
                continue
            origins = grid_expression_inputs(value)
            if origins is None or not set(origins).issubset(integers):
                raise MetadataDeclined("Input layout lost its compiler integer expression origins")


def collect_terminal_metadata(wrapper):
    from torch._inductor.runtime._cudagraph._compiler.compiler_input_contract.collector import compiler_inputs, CompilerInputDeclined
    from torch._inductor import config, ir
    from torch._inductor.codegen.wrapper import (
        AllocateLine, AssertAlignmentLine, AssertSizeStrideLine, CommentLine, CuTeCallLine,
        EnterDeviceContextManagerLine, ExitDeviceContextManagerLine,
        FreeIfNotReusedLine, FreeLine, GroupedAssertSizeStrideLine, InputAlignmentLine,
        KernelCallLine, KernelDefinitionLine, LineContext, NullLine, PythonWrapperCodegen,
        ReuseLine, SymbolicCallArgLine,
    )
    from torch._inductor.virtualized import V
    from torch.fx.experimental.proxy_tensor import _coor_enabled

    graph = V.graph
    partition_names = getattr(wrapper, "all_partition_names", None if config.graph_partition else [])
    if (type(wrapper) is not PythonWrapperCodegen or graph.cpp_wrapper or graph.aot_mode
            or graph.partition_maps or partition_names != [] or _coor_enabled()
            or graph.effectful_ops or graph.constants or not graph.operations
            or any(type(op) not in (ir.ComputedBuffer, ir.TritonTemplateBuffer, ir.MultiTemplateBuffer,
                                   ir.UserDefinedTritonKernel, ir.UserDefinedCuTeKernel) for op in graph.operations)
            or any(type(op) in (ir.TritonTemplateBuffer, ir.MultiTemplateBuffer) and op.mutated_inputs
                   for op in graph.operations)
            or wrapper._multistream_alignment_copies or torch.version.hip is not None
            or config.cuda_backend != "triton" or not config.use_static_triton_launcher
            or config.generate_intermediate_hooks or config.profiler_mark_wrapper_call
            or config.profile_bandwidth or config.nan_asserts or config.annotate_training
            or config.incremental_autotune or config.triton.debug_sync_graph
            or config.triton.debug_sync_kernel or config.triton.proton_profiling
            or config.triton.store_cubin
            or config.aot_inductor.debug_intermediate_value_printer != "0"):
        return None
    if any(type(line) not in (
        AllocateLine, AssertAlignmentLine, AssertSizeStrideLine, CommentLine, CuTeCallLine,
        EnterDeviceContextManagerLine, ExitDeviceContextManagerLine,
        FreeIfNotReusedLine, FreeLine, GroupedAssertSizeStrideLine, InputAlignmentLine,
        KernelCallLine, KernelDefinitionLine, LineContext, NullLine, ReuseLine, SymbolicCallArgLine,
    ) for line in wrapper.lines):
        return None
    try:
        inputs = compiler_inputs(wrapper)
    except CompilerInputDeclined:
        return None
    saved = ()
    if config.cudagraph_saved_input_schedule:
        from torch._inductor.runtime._cudagraph._compiler.compiler_saved_inputs.transport import collect_terminal_saved_inputs

        saved = collect_terminal_saved_inputs(wrapper, inputs)
    metadata = TerminalMetadata(inputs, saved)
    _check_metadata(metadata)
    return metadata


def emit_terminal_metadata(result, function_name, metadata):
    if metadata is None:
        return
    if type(function_name) is not str or not function_name.isidentifier():
        raise MetadataDeclined("Expected the generated function's exact global name")
    _check_metadata(metadata)
    result.writeline("import torch")
    result.writeline('from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import InputContract, IntegerRange, TensorInput')
    result.writeline("from torch._inductor.runtime.cudagraph_arg_mapping import IntExpr")
    result.writeline('from torch._inductor.runtime._cudagraph.metadata import TerminalMetadata, attach_terminal_metadata')
    result.writeline(f"attach_terminal_metadata({function_name}, {metadata!r})")


def _callable_body(function):
    if type(function) is FunctionType:
        return function
    if type(function) is MethodType:
        body, owner = function.__func__, function.__self__
        partitions = getattr(owner, "partitions", None)
        if (type(body) is FunctionType
                and body.__globals__.get("runner") is owner
                and body.__globals__.get("Runner") is type(owner)
                and type(owner).__dict__.get(body.__name__) is body
                and type(partitions) is list and not partitions):
            return body
    return None


def attach_terminal_metadata(function, metadata):
    body = _callable_body(function)
    if (body is None or body.__closure__ or body.__defaults__
            or body.__kwdefaults__ or body.__code__.co_argcount != (2 if type(function) is MethodType else 1)
            or body.__code__.co_kwonlyargcount
            or body.__code__.co_flags & (CO_VARARGS | CO_VARKEYWORDS)
            or body.__globals__.get(body.__name__) is not function
            or "_cudagraph_terminal_attachment" in body.__dict__):
        raise MetadataDeclined("Expected an unattached original one-box generated function")
    _check_metadata(metadata)
    body._cudagraph_terminal_attachment = _Attachment(ref(function), body.__code__, metadata)


def _attachment_matches(function, attachment):
    body = _callable_body(function)
    return (body is not None and type(attachment) is _Attachment
            and body.__dict__.get("_cudagraph_terminal_attachment") is attachment
            and attachment.function() is function and body.__code__ is attachment.code
            and body.__globals__.get(body.__name__) is function
            and not body.__defaults__ and not body.__kwdefaults__)


def check_terminal_attachment(function, attachment):
    if not _attachment_matches(function, attachment):
        raise MetadataDeclined("Generated callable lost its original terminal metadata attachment")
    _check_metadata(attachment.metadata)
    if not _attachment_matches(function, attachment):
        raise MetadataDeclined("Generated callable changed during terminal metadata validation")
    return attachment.metadata


def read_terminal_metadata(artifact):
    from torch._inductor.output_code import CompiledFxGraph
    from torch._inductor.utils import _InputAlignmentWrapper

    if type(artifact) is not CompiledFxGraph:
        raise MetadataDeclined("Expected the original compiler output artifact")
    ordinary = artifact.current_callable
    function = ordinary.model if type(ordinary) is _InputAlignmentWrapper else ordinary
    if (type(function) not in (FunctionType, MethodType)
            or function is not artifact._cudagraph_original_callable):
        raise MetadataDeclined("Active callable is not this artifact's original generated function")
    attachment = function.__dict__.get("_cudagraph_terminal_attachment")
    result = function, check_terminal_attachment(function, attachment)
    if type(ordinary) is _InputAlignmentWrapper:
        tensors = {row.index for row in result[1].inputs.tensor_inputs}
        for indices in (ordinary.inputs_to_check, ordinary.mutated_input_idxs):
            if (type(indices) is not tuple or any(type(index) is not int or index not in tensors for index in indices)
                    or len(indices) != len(set(indices))):
                raise MetadataDeclined("Alignment wrapper indices lost their compiler tensor slots")
    if (artifact.current_callable is not ordinary
            or artifact._cudagraph_original_callable is not function):
        raise MetadataDeclined("Active callable changed during terminal metadata validation")
    return result
