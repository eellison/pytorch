"""Logical compiler sources for owned CuTe calls; not a replay ABI."""

from dataclasses import dataclass
from inspect import CO_VARARGS, CO_VARKEYWORDS
from types import FunctionType
from weakref import ref

import sympy
import torch
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource, InputSource, IntegerInput, IntExpr, pointwise_expression_inputs,
)

from .invocation import check_implementation, InvocationDeclined, InvocationEntry, resolve_entry


class DescriptorDeclined(ValueError):
    pass


@dataclass(frozen=True)
class TensorFact:
    source: InputSource | BufferSource
    dtype: torch.dtype
    device: str
    size: tuple[int | IntExpr, ...]
    stride: tuple[int | IntExpr, ...]


@dataclass(frozen=True)
class CuTeCall:
    entry_key: str
    entry_global: str
    provider_kind: str
    formals: tuple[str, str]
    operands: tuple[TensorFact, TensorFact]


@dataclass(frozen=True)
class CuTeDescriptors:
    version: int
    input_names: tuple[str, ...]
    integer_inputs: tuple[IntegerInput, ...]
    tensors: tuple[TensorFact, ...]
    calls: tuple[CuTeCall, ...]
    outputs: tuple[InputSource | BufferSource, ...]


def _check_descriptors(records):
    if (type(records) is not CuTeDescriptors or type(records.version) is not int
            or records.version != 1 or type(records.input_names) is not tuple
            or any(type(name) is not str for name in records.input_names)
            or len(set(records.input_names)) != len(records.input_names)
            or any(type(getattr(records, field)) is not tuple
                   for field in ("integer_inputs", "tensors", "calls", "outputs"))):
        raise DescriptorDeclined("Unknown CuTe logical descriptor schema")
    integers = set()
    for row in records.integer_inputs:
        if (type(row) is not IntegerInput or type(row.boxed_index) is not int
                or not 0 <= row.boxed_index < len(records.input_names)
                or type(row.symbol) is not str or row.boxed_index in integers or row.source_index is not None):
            raise DescriptorDeclined("Malformed boxed shape source")
        integers.add(row.boxed_index)
    tensors = {}
    for row in records.tensors:
        if (type(row) is not TensorFact or type(row.dtype) is not torch.dtype
                or type(row.device) is not str or row.device not in ("cpu", "cuda")
                and not (row.device.startswith("cuda:") and row.device[5:].isdigit())
                or type(row.size) is not tuple or type(row.stride) is not tuple
                or len(row.size) != len(row.stride)):
            raise DescriptorDeclined("Malformed Tensor layout fact")
        source = row.source
        if type(source) is InputSource:
            if type(source.index) is not int or not 0 <= source.index < len(records.input_names) or source.index in integers:
                raise DescriptorDeclined("Malformed Tensor boxed source")
        elif type(source) is BufferSource:
            if type(source.name) is not str or not source.name:
                raise DescriptorDeclined("Malformed allocation source")
        else:
            raise DescriptorDeclined("Unknown Tensor source")
        if source in tensors:
            raise DescriptorDeclined("Repeated Tensor source")
        for value in (*row.size, *row.stride):
            if type(value) is int and value >= 0:
                continue
            origins = pointwise_expression_inputs(value) if type(value) is IntExpr else None
            if origins is None or not set(origins).issubset(integers):
                raise DescriptorDeclined("Tensor layout lost its boxed shape origin")
        tensors[source] = row
    if {source.index for source in tensors if type(source) is InputSource} | integers != set(range(len(records.input_names))):
        raise DescriptorDeclined("Descriptor does not cover the boxed input order")
    for call in records.calls:
        if (type(call) is not CuTeCall or type(call.entry_key) is not str
                or type(call.entry_global) is not str or not call.entry_global.isidentifier()
                or call.provider_kind not in ("cute", "cpu_standin") or type(call.formals) is not tuple
                or len(call.formals) != 2 or any(type(name) is not str for name in call.formals)
                or len(set(call.formals)) != 2 or type(call.operands) is not tuple or len(call.operands) != 2
                or any(type(row) is not TensorFact or tensors.get(row.source) != row for row in call.operands)
                or type(call.operands[1].source) is not BufferSource
                or call.operands[0].source == call.operands[1].source):
            raise DescriptorDeclined("Malformed CuTe call or nonfresh destination")
    if not records.calls or any(type(source) not in (InputSource, BufferSource) or source not in tensors for source in records.outputs):
        raise DescriptorDeclined("Missing CuTe calls or unknown output source")


def collect_descriptors(wrapper):
    from torch._inductor import ir
    from torch._inductor.codegen.wrapper import (
        AllocateLine, CuTeCallLine, FreeIfNotReusedLine, FreeLine, PythonWrapperCodegen, ReuseLine,
    )
    from torch._inductor.virtualized import V
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    graph = V.graph
    check_implementation()
    if type(wrapper) is not PythonWrapperCodegen or graph.cpp_wrapper or graph.aot_mode:
        raise DescriptorDeclined("CuTe descriptors require the ordinary Python wrapper")
    names, inputs = tuple(wrapper.get_graph_input_names()), wrapper.get_graph_inputs()
    symbols, integers = {}, []
    if graph.sizevars.shape_env.deferred_runtime_asserts:
        raise DescriptorDeclined("Deferred shape assertions are not entry guarantees")
    for index, name in enumerate(names):
        value = inputs[name]
        if isinstance(value, sympy.Expr):
            if not isinstance(value, sympy.Symbol) or value.is_integer is not True or free_unbacked_symbols(value):
                raise DescriptorDeclined("Unsupported boxed shape symbol")
            simplified = graph.sizevars.simplify(value)
            if (not isinstance(simplified, sympy.Symbol) or simplified.is_integer is not True
                    or free_unbacked_symbols(simplified) or simplified in symbols):
                raise DescriptorDeclined("Ambiguous boxed shape symbol")
            expression = IntExpr("boxed", index)
            symbols[value] = symbols[simplified] = expression
            integers.append(IntegerInput(value.name, index))

    live, facts, nodes = {}, [], {}

    def add(node, source):
        if type(node) not in (ir.InputBuffer, ir.ComputedBuffer):
            raise DescriptorDeclined("CuTe sources require typed inputs or fresh allocations")
        name, layout = node.get_name(), node.get_layout()
        if (name in nodes or type(layout) is not ir.FixedLayout
                or wrapper._cudagraph_literal(layout.offset) != 0):
            raise DescriptorDeclined("Repeated source or unsupported Tensor layout")
        values = tuple(wrapper._cudagraph_integer(value, symbols) for value in (*layout.size, *layout.stride))
        if any(value is None for value in values):
            raise DescriptorDeclined("Tensor layout has no supported compiler expression")
        rank = len(layout.size)
        fact = TensorFact(source, layout.dtype, str(layout.device), values[:rank], values[rank:])
        live[name] = node, fact
        nodes[name] = node
        facts.append(fact)

    for index, name in enumerate(names):
        node = inputs[name]
        if isinstance(node, sympy.Expr):
            continue
        while type(node) in (ir.TensorBox, ir.StorageBox):
            node = node.data
        if type(node) is not ir.InputBuffer or node.get_name() != name:
            raise DescriptorDeclined("Input name is not its original typed Tensor")
        add(node, InputSource(index))

    calls = []
    for line in wrapper.lines:
        if type(line) is AllocateLine:
            if line.comm_buffer or type(line.node) is not ir.ComputedBuffer:
                raise DescriptorDeclined("Unsupported allocation kind")
            if not any(op is line.node for op in graph.operations):
                raise DescriptorDeclined("Allocation is not an original compiler operation")
            add(line.node, BufferSource(line.node.get_name()))
        elif type(line) is ReuseLine:
            raise DescriptorDeclined("Reused allocations are outside the first CuTe descriptor")
        elif type(line) in (FreeLine, FreeIfNotReusedLine):
            live.pop(line.node.get_name(), None)
        elif type(line) is CuTeCallLine:
            op = line.node
            if type(op) is not ir.UserDefinedCuTeKernel or not any(candidate is op for candidate in graph.operations):
                raise DescriptorDeclined("Call is not an original typed CuTe operation")
            try:
                entry = resolve_entry(op.entry_key)
            except InvocationDeclined as error:
                raise DescriptorDeclined("CuTe entry is unavailable") from error
            if entry is not op.entry or wrapper.cute_invocation_entries.get(line.entry_global) is not entry:
                raise RuntimeError("CuTe compiler call changed its owned entry")
            operands = []
            for node in op.inputs:
                row = live.get(node.get_name())
                if row is None or row[0] is not node:
                    raise DescriptorDeclined("Call operand differs from its live compiler source")
                operands.append(row[1])
            calls.append(CuTeCall(entry.key, line.entry_global, entry.kind, entry.formals, tuple(operands)))
    outputs = []
    for node in wrapper.get_graph_outputs():
        if node is None or type(node) is ir.NoneAsConstantBuffer:
            continue
        while type(node) in (ir.TensorBox, ir.StorageBox):
            node = node.data
        if type(node) is ir.ShapeAsConstantBuffer:
            value = wrapper._cudagraph_integer(node.expr, symbols)
            if value is None or type(value) is int and not -(2 ** 63) <= value < 2 ** 63:
                raise DescriptorDeclined("Output integer lost its compiler expression")
            continue
        if type(node) not in (ir.InputBuffer, ir.ComputedBuffer):
            raise DescriptorDeclined("Unsupported compiler output")
        row = live.get(node.get_name())
        if row is None or row[0] is not node:
            raise DescriptorDeclined("Output differs from its live compiler source")
        outputs.append(row[1].source)
    records = CuTeDescriptors(1, names, tuple(integers), tuple(facts), tuple(calls), tuple(outputs))
    _check_descriptors(records)
    return records


@dataclass(frozen=True, eq=False)
class _Attachment:
    function: object
    code: object
    records: object
    snapshot: str


def attach_descriptors(function, records):
    if (type(function) is not FunctionType or function.__closure__ or function.__defaults__
            or function.__kwdefaults__ or function.__code__.co_argcount != 1
            or function.__code__.co_kwonlyargcount or function.__code__.co_flags & (CO_VARARGS | CO_VARKEYWORDS)
            or function.__globals__.get(function.__name__) is not function
            or "_cute_invocation_attachment" in function.__dict__):
        raise DescriptorDeclined("Expected the original one-box generated function")
    function._cute_invocation_descriptors = records
    function._cute_invocation_attachment = _Attachment(ref(function), function.__code__, records, repr(records))


class BoundCuTeCalls:
    def __init__(self, function, attachment, entries):
        self.function = ref(function)
        self.attachment = attachment
        self.entries = entries
        self.records = attachment.records
        self._seal = attachment, entries, self.records

    def check(self):
        function = self.function()
        attachment, entries, records = self._seal
        if (function is None or self.attachment is not attachment or self.entries is not entries
                or self.records is not records or _read_attachment(function) is not attachment
                or type(entries) is not tuple or len(entries) != len(records.calls)
                or any(type(entry) is not InvocationEntry for entry in entries)):
            raise DescriptorDeclined("CuTe logical binding changed or expired")
        for call, entry in zip(records.calls, entries):
            current = function.__globals__.get(call.entry_global)
            if current is not entry:
                from .callsite import _CompilerInvocation
                from .conversion import ConversionDeclined

                if type(current) is not _CompilerInvocation:
                    raise DescriptorDeclined("CuTe call no longer binds its emitted live entry")
                try:
                    original = current.check_binding(function, call)
                except ConversionDeclined as error:
                    raise DescriptorDeclined("CuTe compiler-call association changed") from error
                if original is not entry:
                    raise DescriptorDeclined("CuTe compiler call changed its original entry")
            if (resolve_entry(call.entry_key) is not entry
                    or call.formals != entry.formals or call.provider_kind != entry.kind
                    or any(torch.device(row.device).type != entry.provider.device_type for row in call.operands)):
                raise DescriptorDeclined("CuTe call no longer binds its emitted live entry")
        if _read_attachment(function) is not attachment:
            raise DescriptorDeclined("CuTe callable changed during binding")

    def __reduce_ex__(self, protocol):
        raise TypeError("Live CuTe bindings are not cache transport")


def _read_attachment(function):
    check_implementation()
    if type(function) is not FunctionType:
        raise DescriptorDeclined("Expected an original generated function")
    attachment = function.__dict__.get("_cute_invocation_attachment")
    if (type(attachment) is not _Attachment or attachment.function() is not function
            or function.__code__ is not attachment.code or function.__defaults__ or function.__kwdefaults__
            or function.__globals__.get(function.__name__) is not function
            or function.__dict__.get("_cute_invocation_descriptors") is not attachment.records):
        raise DescriptorDeclined("CuTe callable lost its original descriptor attachment")
    _check_descriptors(attachment.records)
    if repr(attachment.records) != attachment.snapshot:
        raise DescriptorDeclined("CuTe descriptor contents changed")
    return attachment


def bind_descriptors(function):
    attachment = _read_attachment(function)
    try:
        entries = tuple(resolve_entry(call.entry_key) for call in attachment.records.calls)
        binding = BoundCuTeCalls(function, attachment, entries)
        binding.check()
    except InvocationDeclined as error:
        raise DescriptorDeclined("CuTe entry is unavailable") from error
    return binding


def bind_compiler_invocation(function):
    from .callsite import _CompilerInvocation
    from .conversion import ConversionDeclined
    from .envelope import bind_envelope, EnvelopeDeclined

    try:
        compiler = bind_envelope(function)
    except EnvelopeDeclined:
        return None
    namespace = function.__globals__
    bindings, created = [], []
    try:
        for index, (call, entry) in enumerate(zip(compiler.cute.calls, compiler.binding.entries)):
            current = namespace.get(call.entry_global)
            if type(current) is _CompilerInvocation:
                if current.check_binding(function, call) is not entry:
                    raise DescriptorDeclined("CuTe compiler call changed its original entry")
                bindings.append(current)
            else:
                binding = _CompilerInvocation(compiler, call_index=index)
                created.append((call, entry, binding))
                bindings.append(binding)
    except ConversionDeclined:
        for _, _, binding in created:
            binding.close()
        return None
    except BaseException:
        for _, _, binding in created:
            binding.close()
        raise
    try:
        for call, entry, binding in created:
            if namespace.get(call.entry_global) is not entry:
                raise DescriptorDeclined("CuTe compiler entry changed before installation")
        for call, _, binding in created:
            namespace[call.entry_global] = binding
        compiler.check()
    except BaseException:
        for call, entry, binding in created:
            if namespace.get(call.entry_global) is binding:
                namespace[call.entry_global] = entry
            binding.close()
        raise
    return bindings[0] if len(bindings) == 1 else tuple(bindings)


def emit_descriptors(result, function_name, records):
    _check_descriptors(records)
    if type(function_name) is not str or not function_name.isidentifier():
        raise ValueError("Expected the generated function's exact global name")
    result.writeline('from torch._inductor.runtime._cudagraph._compiler.compiler_cute_handoff.descriptor import CuTeDescriptors, CuTeCall, TensorFact, attach_descriptors')
    result.writeline("from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource, InputSource, IntegerInput, IntExpr")
    result.writeline(f"attach_descriptors({function_name}, {records!r})")
