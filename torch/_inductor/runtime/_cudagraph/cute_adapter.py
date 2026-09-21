"""Bind traced CuTe operands to their observed ordinary compiler artifact."""

from dataclasses import dataclass, field, replace
import struct

import sympy
import torch
from torch._inductor.runtime._cudagraph._compiler.compiler_cute_handoff.invocation import InvocationEntry
from torch._inductor.runtime._cudagraph._compiler.cute_bridge.lowering import _Operands, _lower_cute_call
from torch._inductor.runtime._cudagraph._compiler.cute_bridge.numeric import Comparison, lower_numeric
from torch._inductor.runtime._cudagraph._compiler.cute_bridge.parameter_program import lower_parameter
from torch._inductor.runtime._cudagraph._compiler.cute_bridge.provider import _constant_consumer, CuTeKernelOwner
from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.entry import OrdinaryEntry
from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.artifact import TmaDimensionDomain, TmaStrideDomain
from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.factory import prepare_dispatch_artifact, rebind_ordinary_artifact
from torch._inductor.runtime._cudagraph._compiler.entry_signature import _number_state, _tensor_state, build_entry_signature
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import FXTraceDeclined
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_adapter.invocation import normalize_artifact_invocation
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner import ObservedOrdinaryEntry
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_reuse.binding import bind_ordinary_metadata
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_reuse.correspondence import join_ordinary_correspondence
from torch._inductor.runtime._cudagraph._compiler.python_entry import _Recording, EntryCall, opaque_entry
from torch._inductor.runtime._cudagraph._compiler.stream_layout import compile_stream_layout
from torch._inductor.runtime._cudagraph._compiler.target_layout import compile_target_layout
from torch._inductor.runtime._cudagraph._compiler.tma_dimension import TmaDimension
from torch._inductor.runtime._cudagraph.address_guard_printer import UIntGCD
from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource, ExpressionSource, InputSource, IntegerSource, IntExpr, ParameterSource, PointerSource
from torch._subclasses.fake_tensor import FakeTensor

from .cute_types import CuteInvokeEvent, CuTeCall
from .direct_cute import DirectCuTe
from .address_scalars import lower_address_scalar, pointer_alignment_guard, symbolic_integer


class CuTeDeclined(FXTraceDeclined):
    pass


@dataclass
class _CompilerTemplate:
    entry: InvocationEntry
    compilation: object
    artifact: object


class CuTePreparation:
    def __init__(self):
        self.templates = {}

    def bind(self, entry, binding):
        key = id(entry), id(binding.compilation)
        template = self.templates.get(key)
        if template is not None:
            if template.entry is not entry or template.compilation is not binding.compilation:
                raise CuTeDeclined("CuTe preparation mixed ordinary compilation owners")
            return rebind_ordinary_artifact(template.artifact, binding)
        invocation = normalize_artifact_invocation(join_ordinary_correspondence(binding))
        artifact = prepare_dispatch_artifact(invocation, compile_target_layout(invocation.plan),
                                             compile_stream_layout(invocation.formals, invocation.stream_source_index))
        template = _CompilerTemplate(entry, binding.compilation, artifact)
        self.templates[key] = template
        return artifact


class _TraceView:
    providers = ()

    def __init__(self, entry, sink, fake_mode):
        self.entry = entry
        self.sink = sink
        self.fake_mode = fake_mode
        self._entry = entry
        self._invoke = type(entry).invoke
        self._code = self._invoke.__code__
        self.check()

    def check(self):
        if (type(self.entry) not in (InvocationEntry, DirectCuTe) or self.entry is not self._entry
                or type(self.entry).invoke is not self._invoke or self._invoke.__code__ is not self._code):
            raise CuTeDeclined("CuTe trace view lost its original invocation entry")
        self.entry.check()
        if type(self.entry) is InvocationEntry:
            provider = self.entry.provider
            if self.entry.kind != "cute" or provider.run.__func__ is not OrdinaryEntry._invoke_owned:
                raise CuTeDeclined("CuTe tracing requires an owned ordinary executor")
        if type(self.owner) is not ObservedOrdinaryEntry or self.owner._selected is None:
            raise CuTeDeclined("CuTe tracing requires a warmed owned ordinary executor")

    @property
    def owner(self):
        return self.entry.owner if type(self.entry) is DirectCuTe else self.entry.provider.owner

    def check_operands(self, operands):
        if len(operands) != len(self.owner.argument_kinds):
            raise CuTeDeclined("CuTe invocation arguments differ from its registered signature")
        for kind, value in zip(self.owner.argument_kinds, operands):
            if kind == "tensor":
                if type(value) is not FakeTensor or value.fake_mode is not self.fake_mode:
                    raise CuTeDeclined("CuTe tensor escaped the local tracing mode")
            elif kind != "integer" or type(value) not in (int, torch.SymInt):
                raise CuTeDeclined("CuTe scalar requires an integer argument")
            elif type(value) is torch.SymInt and value.node.shape_env is not self.fake_mode.shape_env:
                raise CuTeDeclined("CuTe scalar escaped the local symbolic environment")

    def invoke(self, *operands):
        self.check()
        self.check_operands(operands)
        owner = self.owner
        conversion = None
        if owner.conversion is not None:
            conversion = owner.conversion.trace(operands)
            operands = tuple(output.value for output in conversion.outputs)
            self.check_operands(operands)
            for event in conversion.events:
                self.sink(event)
        entry = owner.entry
        recording = _Recording((entry,), self.fake_mode, self.fake_mode.shape_env)
        recording.record(entry, operands, {})
        call, = recording.calls
        self.sink(CuteInvokeEvent(self.entry, operands, call, conversion))
        self.check()

    __call__ = invoke


def make_cute_trace_view(entry, sink, fake_mode):
    return _TraceView(entry, sink, fake_mode)


@dataclass(frozen=True, eq=False)
class _InvocationTrace:
    terminal: object
    event: CuteInvokeEvent
    borrow: object
    operand_state: tuple

    @property
    def calls(self):
        return (self.event.call,)

    @property
    def shape_env(self):
        return self.terminal.shape_env

    @property
    def fake_mode(self):
        index = self.terminal.contract.tensor_inputs[0].index
        return self.terminal.placeholders[index].fake_mode

    @property
    def input_contract(self):
        return self.terminal.contract

    def check(self):
        self.terminal.compiler_binding.check()
        self.borrow.check()
        event, call = self.event, self.event.call
        conversion = event.conversion
        if conversion is None:
            if self.borrow.owner.conversion is not None:
                raise CuTeDeclined("CuTe invocation lost its user conversion trace")
        else:
            conversion.check()
            if (conversion.conversion is not self.borrow.owner.conversion
                    or len(conversion.outputs) != len(event.operands)
                    or any(output.value is not operand for output, operand in zip(conversion.outputs, event.operands))):
                raise CuTeDeclined("CuTe invocation lost its converted argument sources")
        if (type(event) is not CuteInvokeEvent or not any(item is event for item in self.terminal.events)
                or self.borrow.entry is not event.entry
                or self.fake_mode.shape_env is not self.shape_env
                or any(value.fake_mode is not self.fake_mode for value in event.operands
                       if type(value) is FakeTensor)
                or tuple(_tensor_state(value) if type(value) is FakeTensor else _number_state(value)
                         for value in event.operands) != self.operand_state
                or type(call) is not EntryCall or call.entry is not self.borrow.owner.entry
                or call.target is not call.entry.target or call.config or call.entry_index != 0
                or call.keyword_arguments or len(call.arguments) != len(event.operands)
                or len(call.operands) != len(event.operands)
                or call.node.graph is not self.terminal.graph_module.graph
                or call.node.op != "call_function" or call.node.target is not opaque_entry
                or call.node.args != (0, *call.arguments) or call.node.kwargs or not call.node.is_impure()
                or any(operand.value is not value or operand.path != ("args", index)
                       or operand.fx_argument is not call.arguments[index]
                       for index, (operand, value) in enumerate(zip(call.operands, event.operands)))):
            raise CuTeDeclined("CuTe invocation lost its recorded operands or ordinary owner")


_FAILED_RECEIPTS = []


@dataclass(eq=False)
class _InvocationResources:
    borrow: object
    owners: list = field(default_factory=list)
    closed: bool = False

    def close(self):
        if self.closed:
            return
        failure = None
        for owner in reversed(self.owners):
            try:
                owner.close()
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    failure.add_note(f"Additional CuTe owner cleanup failed: {error}")
        if failure is None:
            try:
                self.borrow.close()
            except BaseException as error:
                failure = error
        if failure is not None:
            if not any(item is self for item in _FAILED_RECEIPTS):
                _FAILED_RECEIPTS.append(self)
            raise failure
        self.closed = True
        _FAILED_RECEIPTS[:] = [item for item in _FAILED_RECEIPTS if item is not self]


@dataclass(frozen=True, eq=False)
class CuTeReceipt:
    trace: _InvocationTrace
    invocation: object
    owner: CuTeKernelOwner
    resources: _InvocationResources

    @property
    def closed(self):
        return self.resources.closed

    @property
    def borrow(self):
        return self.trace.borrow

    def check(self):
        if self.closed:
            raise CuTeDeclined("CuTe terminal receipt is closed")
        self.trace.check()
        self.invocation.check()
        self.owner.check()
        if (self.resources.borrow is not self.borrow
                or not any(owner is self.owner for owner in self.resources.owners)
                or self.invocation.signature.trace is not self.trace
                or self.owner.artifact.signature is not self.invocation.signature
                or self.owner.artifact._guards.binding is not self.invocation):
            raise CuTeDeclined("CuTe terminal receipt mixed compiler or launch owners")

    def close(self):
        self.resources.close()


def _symbolic(value, symbols):
    return symbolic_integer(value, symbols, CuTeDeclined)


def _condition(predicate, symbols):
    value = predicate.expression
    if type(value) is Comparison:
        operators = {"eq": sympy.Eq, "ne": sympy.Ne, "slt": sympy.Lt,
                     "sle": sympy.Le, "sgt": sympy.Gt, "sge": sympy.Ge}
        if value.predicate not in operators:
            raise CuTeDeclined("CuTe unsigned predicates need a typed guard lowering")
        return operators[value.predicate](_symbolic(value.left, symbols), _symbolic(value.right, symbols))
    if type(value) is IntExpr and predicate.llvm_type == "i1":
        return sympy.Eq(_symbolic(value, symbols), 1)
    raise CuTeDeclined("CuTe dispatch has no supported exact predicate")


def _check_fields(bound, operands, sources, root_alignments, trace):
    owner, site = bound.module, bound.module.site
    artifact = owner.artifact
    if (struct.calcsize("P") != 8 or tuple(size for _, size in owner.parameter_layout) != owner.parameter_sizes
            or owner.parameter_sizes != site.fields.parameter_sizes
            or bound.padding != tuple(tuple(row) for row in site.fields.padding)
            or bound.constants != tuple((row.parameter, row.byte_offset, row.data) for row in site.fields.constants)
            or bound.undefined != tuple((row.parameter, row.byte_offset, row.byte_size) for row in site.fields.undefined)):
        raise CuTeDeclined("CuTe physical parameters differ from the loaded compiler ABI")
    schema = {(row.parameter, row.byte_offset): ("pointer", row.source) for row in site.fields.pointers}
    schema.update({(row.parameter, row.byte_offset): (row.dtype, row.source) for row in site.fields.integers})
    fields = {(row.parameter, row.byte_offset): row for row in bound.fields}
    if (len(schema) != len(site.fields.pointers) + len(site.fields.integers)
            or len(fields) != len(bound.fields) or set(fields) != set(schema)):
        raise CuTeDeclined("CuTe physical fields require complete unique ABI coverage")
    spans = [[] for _ in owner.parameter_sizes]
    guards = []
    for key, row in fields.items():
        kind, origin = schema[key]
        if (row.kind != kind or kind not in ("pointer", "i32", "i64")
                or type(row.parameter) is not int or not 0 <= row.parameter < len(spans)
                or type(row.byte_offset) is not int or row.byte_offset < 0):
            raise CuTeDeclined("CuTe field changed its physical type or position")
        spans[row.parameter].append((row.byte_offset, row.byte_offset + (4 if kind == "i32" else 8)))
        expected = (lower_parameter(origin.expression, operands)[0]
                    if origin.kind == "compiler_expression" else operands.field(origin))
        if kind != "pointer" and type(expected) is not ParameterSource:
            expected = ExpressionSource(expected) if type(expected) is IntExpr else IntegerSource(expected)
        if row.source != expected:
            raise CuTeDeclined("CuTe field lost its exact operand property")
        if type(expected) is ParameterSource:
            if expected.width != (32 if kind == "i32" else 64):
                raise CuTeDeclined("CuTe parameter computation changed its physical width")
            if any(pointer.root not in root_alignments for pointer in expected.pointers):
                raise CuTeDeclined("CuTe parameter computation lost its traced storage root")
            continue
        if kind != "pointer":
            continue
        formal = operands.formals[origin.ir_arg_index]
        alignment = formal.data_alignment
        if row.source not in sources:
            raise CuTeDeclined("CuTe pointer lost its traced source")
        guards.append(pointer_alignment_guard(row.source, alignment, root_alignments, trace))
    for formal in artifact.formals:
        if formal.kind == "Tensor":
            for constant in formal.constants:
                value = operands.property(formal, constant.property, constant.property_path)
                if type(value) is not int or value != constant.value:
                    raise CuTeDeclined("CuTe operand changed an elided compiler constant")
    for parameter, offset, data in bound.constants:
        if (type(parameter) is not int or not 0 <= parameter < len(spans)
                or type(offset) is not int or offset < 0 or type(data) is not bytes or not data):
            raise CuTeDeclined("CuTe literal bytes lost their exact compiler span")
        spans[parameter].append((offset, offset + len(data)))
    for parameter, offset, width in (*bound.padding, *bound.undefined):
        if (type(parameter) is not int or not 0 <= parameter < len(spans)
                or type(offset) is not int or offset < 0 or type(width) is not int or width <= 0):
            raise CuTeDeclined("CuTe padding lost its exact compiler span")
        spans[parameter].append((offset, offset + width))
    for size, intervals in zip(owner.parameter_sizes, spans, strict=True):
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor or not start < end <= size:
                raise CuTeDeclined("CuTe fields and padding do not cover the complete parameter")
            cursor = end
        if cursor != size:
            raise CuTeDeclined("CuTe parameter contains bytes without an exact source")
    return tuple(guards)


def _integer_requirement_guards(requirement, value):
    if requirement.kind == "integer_divisibility":
        if type(requirement.divisor) is not int or requirement.divisor <= 0:
            raise CuTeDeclined("CuTe signature requires a positive integer divisor")
        return (sympy.Eq(sympy.Mod(value, requirement.divisor), 0),)
    if requirement.kind not in ("integer_range", "storage_offset_nonnegative"):
        raise CuTeDeclined("CuTe signature has an unsupported runtime requirement")
    guards = []
    if requirement.minimum is not None:
        guards.append(sympy.Ge(value, requirement.minimum))
    if requirement.maximum is not None:
        guards.append(sympy.Le(value, requirement.maximum))
    return tuple(guards)


def _shared_resources(artifact, site, operands, symbols):
    expected = {("shared", 0), ("kernel_smem", 0),
                *(("diagnostic", index) for index in range(len(site.diagnostics)))}
    values, obligations = {}, []
    for consumer in artifact.consumers:
        if consumer.site_id != site.site_id or consumer.role not in ("shared", "kernel_smem", "diagnostic"):
            continue
        key = consumer.role, consumer.index
        if key not in expected or key in values:
            raise CuTeDeclined("CuTe shared resources lost their exact compiler consumers")
        lowered = lower_numeric(consumer.numeric, operands.numeric)
        if len(lowered.values) != 1 or lowered.values[0].llvm_type != consumer.result_type:
            raise CuTeDeclined("CuTe shared resource has an unsupported result signature")
        values[key] = lowered.values[0]
        obligations.extend(lowered.obligations)
    if set(values) != expected:
        raise CuTeDeclined("CuTe shared resources lack complete compiler consumers")
    shared, needed = (values[role, 0] for role in ("shared", "kernel_smem"))
    if any(value.llvm_type not in ("i32", "i64") or type(value.expression) is not IntExpr
           for value in (shared, needed)):
        raise CuTeDeclined("CuTe shared sizes require exact integer recipes")
    request, requirement = (_symbolic(value.expression, symbols) for value in (shared, needed))
    guards = [sympy.Ge(request, 0), sympy.Le(request, 2 ** 32 - 1),
              sympy.Ge(requirement, 0), sympy.Ge(request, requirement)]
    for index, diagnostic in enumerate(site.diagnostics):
        condition = _condition(values["diagnostic", index], symbols)
        guards.append(condition if diagnostic.expected else sympy.Not(condition))
    return shared.expression, tuple(guards), tuple(obligations)


def _tma_values(artifact, site, role, indices, operands, symbols):
    consumers = tuple(consumer for consumer in artifact.consumers
                      if consumer.site_id == site.site_id and consumer.role == role)
    if tuple(consumer.index for consumer in consumers) != indices:
        raise CuTeDeclined(f"CuTe {role} lost its exact compiler consumers")
    values, obligations = {}, []
    for consumer in consumers:
        lowered = lower_numeric(consumer.numeric, operands.numeric)
        if (len(lowered.values) != 1 or lowered.values[0].llvm_type != consumer.result_type
                or consumer.result_type not in ("i32", "i64")
                or type(lowered.values[0].expression) is not IntExpr):
            raise CuTeDeclined(f"CuTe {role} has an unsupported integer recipe")
        values[consumer.index] = _symbolic(lowered.values[0].expression, symbols)
        obligations.extend(lowered.obligations)
    return values, tuple(obligations)


def _tma_stride_guards(artifact, site, operands, symbols):
    divisors = site.tma_stride_divisors
    strides, obligations = _tma_values(artifact, site, "tma_stride", tuple(range(len(divisors))), operands, symbols)
    guards = []
    for index, divisor in enumerate(divisors):
        if type(divisor) is not int or divisor <= 0:
            raise CuTeDeclined("CuTe TMA stride requires a positive integer divisor")
        guards.append(sympy.Eq(sympy.Mod(strides[index], divisor), 0))
    for domain in site.tma_stride_domains:
        if (type(domain) is not TmaStrideDomain or domain.element_bytes not in (1, 2, 4, 8)
                or not domain.indices or any(type(index) is not int or not 0 <= index < len(strides)
                                            for index in domain.indices)):
            raise CuTeDeclined("CuTe TMA stride domain lost its exact compiler consumers")
        stride = UIntGCD(*(strides[index] for index in domain.indices))
        guards.append(sympy.Lt(stride, (1 << 40) // domain.element_bytes))
    return tuple(guards), tuple(obligations)


def _tma_dimension_guards(artifact, site, operands, symbols):
    domains = site.tma_dimension_domains
    if any(type(domain) is not TmaDimensionDomain or type(domain.grouped) is not bool
           or not domain.indices or (not domain.grouped and len(domain.indices) != 1)
           or any(type(index) is not int for index in domain.indices) for domain in domains):
        raise CuTeDeclined("CuTe TMA dimension domain lost its exact compiler consumers")
    indices = tuple(index for domain in domains for index in domain.indices)
    if indices != tuple(range(len(indices))):
        raise CuTeDeclined("CuTe TMA dimension domains must retain every ordered source")
    shapes, obligations = _tma_values(artifact, site, "tma_shape", indices, operands, symbols)
    indices = tuple(index for domain in domains if domain.grouped for index in domain.indices)
    strides, stride_obligations = _tma_values(artifact, site, "tma_dimension_stride", indices, operands, symbols)
    guards = []
    for domain in domains:
        values = (tuple(value for index in domain.indices for value in (shapes[index], strides[index]))
                  if domain.grouped else (shapes[domain.indices[0]],))
        dimension = TmaDimension(*values)
        guards.extend((sympy.Ge(dimension, 1), sympy.Le(dimension, 1 << 32)))
    return tuple(guards), (*obligations, *stride_obligations)


def lower_cute_calls(trace, event, tensors, expression, *, stream, preparation, root_alignments):
    if type(event) is not CuteInvokeEvent or type(stream) is not int or not 0 <= stream < 2 ** 64:
        raise CuTeDeclined("CuTe lowering requires an actual traced invocation and raw stream")
    tensor_operands = tuple(value for value in event.operands if type(value) is FakeTensor)
    sources = tuple(tensors.get(id(value)) for value in tensor_operands)
    if any(type(source) is not PointerSource or type(source.root) not in (InputSource, BufferSource)
           for source in sources):
        raise CuTeDeclined("CuTe tensor arguments require traced storage roots")
    if any(value.device != torch.device("cuda", trace.contract.device_index) for value in tensor_operands):
        raise CuTeDeclined("CuTe operand escaped its inherited CUDA device")
    borrow = event.entry.borrow_native()
    resources = _InvocationResources(borrow)
    try:
        local = _InvocationTrace(trace, event, borrow,
            tuple(_tensor_state(value) if type(value) is FakeTensor else _number_state(value)
                  for value in event.operands))
        local.check()
        compilation = borrow.owner.compilation()
        signature = build_entry_signature(local, event.call, policy=borrow.owner.policy,
                                          metadata=compilation.metadata)
        binding = bind_ordinary_metadata(signature, compilation)
        artifact = preparation.bind(event.entry, binding)
        common_guards = []

        def parameter_expression(value, abi_type):
            late = lower_address_scalar(value, abi_type, trace, expression)
            if late is None:
                return None
            source, guards = late
            common_guards.extend(guards)
            return source

        operands = _Operands(local, artifact, tensors, expression, parameter_expression=parameter_expression,
                             storage_offset_indices=(binding.index for binding in trace.storage_offset_bindings))
        symbols = {origin.value: symbol for symbol, origin in trace.symbol_sources.items()}
        symbols.update((("storage_offset", binding.index), binding.symbol)
                       for binding in trace.storage_offset_bindings)
        address_symbols = {binding.symbol for binding in trace.address_bindings}
        metadata_symbols = {binding.symbol for binding in trace.storage_offset_bindings}
        predicates = [item for item in artifact.consumers if item.site_id is None and item.role == "predicate"]
        decision = condition = None
        if artifact.sites and all(site.arm is None for site in artifact.sites):
            if predicates:
                raise CuTeDeclined("Unconditional CuTe artifact contains a dispatch predicate")
            sites = artifact.sites
        else:
            if len(predicates) != 1:
                raise CuTeDeclined("CuTe artifact lacks one original dispatch predicate")
            decision = lower_numeric(predicates[0].numeric, operands.numeric)
            if len(decision.values) != 1:
                raise CuTeDeclined("CuTe dispatch predicate has multiple results")
            condition = _condition(decision.values[0], symbols)
            selected = condition.xreplace(trace.shape_env.backed_var_to_val)
            if selected not in (sympy.true, sympy.false):
                raise CuTeDeclined("CuTe dispatch predicate has no exact preparation value")
            sites = tuple(site for site in artifact.sites if site.arm is bool(selected))
            if len(sites) != 1:
                raise CuTeDeclined("CuTe dispatch does not select exactly one compiler site")
        uses = {use.path: use for operand in signature.operands if operand.tensor is not None
                for use in (*operand.tensor.shape, *operand.tensor.strides, operand.tensor.storage_offset)}
        uses.update((operand.scalar.use.path, operand.scalar.use) for operand in signature.operands
                    if operand.scalar is not None)
        for requirement in binding.requirements:
            if requirement.kind == "effective_pointer_alignment":
                if (type(requirement.alignment) is not int or requirement.alignment <= 0
                        or requirement.alignment & (requirement.alignment - 1)):
                    raise CuTeDeclined("CuTe signature requires a positive power-of-two alignment")
                matches = [tensors.get(id(operand.tensor.value)) for operand in signature.operands
                           if operand.tensor is not None and operand.path == requirement.path]
                if len(matches) != 1 or type(matches[0]) is not PointerSource:
                    raise CuTeDeclined("CuTe alignment requirement lost its original operand")
                common_guards.append(pointer_alignment_guard(matches[0], requirement.alignment,
                                                             root_alignments, trace))
                continue
            if requirement.path not in uses:
                raise CuTeDeclined("CuTe signature has an unsupported runtime requirement")
            original = uses[requirement.path].value
            if (type(original) is torch.SymInt and original.node.shape_env is trace.shape_env
                    and original.node.expr.free_symbols.intersection(address_symbols | metadata_symbols)):
                value = original.node.expr
            else:
                value = _symbolic(expression(original), symbols)
            common_guards.extend(_integer_requirement_guards(requirement, value))
        calls = []
        for site in sites:
            shared, resource_guards, resource_obligations = _shared_resources(artifact, site, operands, symbols)
            tma_guards, tma_obligations = _tma_stride_guards(artifact, site, operands, symbols)
            dimension_guards, dimension_obligations = _tma_dimension_guards(artifact, site, operands, symbols)
            owner = CuTeKernelOwner(artifact, site, stream=stream,
                block=tuple(_constant_consumer(artifact, site, "block", axis) for axis in range(3)),
                shared=shared.value if shared.op == "constant" else None)
            resources.owners.append(owner)
            receipt = CuTeReceipt(local, binding, owner, resources)
            bound, predicate, obligations = _lower_cute_call(artifact, site, owner, operands)
            if owner.shared is None:
                bound = replace(bound, shared=shared)
            field_guards = _check_fields(bound, operands, sources, root_alignments, trace)
            guards = [*common_guards, *field_guards, *resource_guards, *tma_guards, *dimension_guards,
                      sympy.Le(_symbolic(shared, symbols), owner.max_dynamic_shared)]
            if condition is None:
                if predicate is not None:
                    raise CuTeDeclined("Unconditional CuTe lowering introduced a dispatch predicate")
            else:
                guards.append(condition if site.arm else sympy.Not(condition))
                if predicate is None or _condition(predicate, symbols) != condition:
                    raise CuTeDeclined("CuTe physical lowering changed its original predicate")
            decision_obligations = () if decision is None else decision.obligations
            for obligation in (*decision_obligations, *obligations, *resource_obligations,
                               *tma_obligations, *dimension_obligations):
                value = _symbolic(obligation.expression, symbols)
                guards.extend((sympy.Ge(value, obligation.lower), sympy.Le(value, obligation.upper)))
            receipt.check()
            calls.append(CuTeCall(bound, sources, tuple(guards), receipt))
        return tuple(calls)
    except BaseException as error:
        try:
            resources.close()
        except BaseException as cleanup:
            error.add_note(f"CuTe preparation resources retained after cleanup failed: {cleanup}")
        if isinstance(error, (ValueError, TypeError)) and not isinstance(error, CuTeDeclined):
            raise CuTeDeclined(str(error)) from error
        raise
