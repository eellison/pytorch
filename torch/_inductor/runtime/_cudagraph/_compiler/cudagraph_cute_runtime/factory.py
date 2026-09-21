from __future__ import annotations

from hashlib import sha256
from typing import Any

from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_adapter.invocation import ArtifactInvocation, normalize_artifact_invocation

from ..entry_signature import EntrySignature
from ..owned_numeric import OwnedNumeric
from ..values import ScalarValue
from .artifact import (
    ARTIFACT_VERSION,
    ArtifactSite,
    BinaryImage,
    ConstantProperty,
    ConstantField,
    Consumer,
    Diagnostic,
    DispatchArtifact,
    FieldSource,
    Formal,
    IntegerField,
    Leaf,
    NodeFields,
    Padding,
    Parameter,
    ParameterExpression,
    UndefinedField,
    PointerField,
    Registration,
    Stream,
    TmaDimensionDomain,
    TmaStrideDomain,
    _FunctionGuard,
    _LiveGuards,
    _OrdinaryGuards,
    _Payload,
    _check_guards,
)
from .signature import prepare_entry_source_guard


_RECORDS = (Leaf, ConstantProperty, Formal, Stream, BinaryImage, Registration, Parameter, FieldSource,
            PointerField, IntegerField, ConstantField, Padding, ParameterExpression, UndefinedField, NodeFields, Diagnostic, Consumer, TmaDimensionDomain, TmaStrideDomain, ArtifactSite, _Payload)


def _immutable(value: Any) -> None:
    if value is None or type(value) in (str, int, bool, bytes):
        return
    if type(value) is OwnedNumeric:
        value.check()
        return
    if type(value) is tuple or type(value) in _RECORDS:
        for item in value:
            _immutable(item)
        return
    raise TypeError("Dispatch artifact contains a mutable or unsupported compiler record")


def _property_path(prop: str, path: tuple[int, ...], parameter: Any) -> None:
    if prop == "pointer" and path == ():
        return
    dimensions = parameter.shape if prop == "shape" else parameter.strides
    if (prop not in {"shape", "stride"} or type(path) is not tuple or len(path) != 1
            or type(path[0]) is not int or not 0 <= path[0] < len(dimensions)):
        raise ValueError("Unsupported or out-of-range tensor property path")


def _copy_formals(invocation: ArtifactInvocation, layout: Any, stream_layout: Any) -> tuple[Formal, ...]:
    from torch._inductor.runtime._cudagraph._compiler.values import evaluate

    invocation = normalize_artifact_invocation(invocation)
    signature = invocation.signature
    actual = invocation.bound
    if (len(actual.params) != len(signature.operands)
            or actual.metadata != invocation.joined.admission.source.metadata):
        raise ValueError("Artifact metadata lacks exact original Python operand coverage")
    stream_kind = "Stream" if invocation.kind == "ordinary" else "EnvStream"
    by_source = {}
    for index, (parameter, operand) in enumerate(zip(actual.params, signature.operands)):
        metadata = parameter.metadata
        if (parameter.source is not operand
                or type(metadata.ir_arg_index) is not int or metadata.ir_arg_index < 0
                or metadata.ir_arg_index in by_source or metadata.kind not in {"Tensor", "Var", stream_kind}
                or metadata.name != operand.name):
            raise ValueError("Artifact requires exact Tensor, integer and original stream source bindings")
        by_source[metadata.ir_arg_index] = (index, operand, metadata)
    tensors = {formal.source_arg_index: formal for formal in invocation.plan.formals}
    targets = {formal.source_arg_index: formal for formal in layout.formals}
    joined_properties = {id(item.component): item for item in invocation.properties}
    lowered = sorted(invocation.formals.formals, key=lambda formal: formal.llvm_arg_index)
    flow = invocation.joined.mapping.flow
    if (tuple(formal.llvm_arg_index for formal in lowered) != tuple(range(len(flow.host_types)))
            or {formal.ir_arg_index for formal in lowered} != set(by_source)
            or len(lowered) != len(by_source) or set(tensors) != set(targets)):
        raise ValueError("Artifact source and lowered argument coverage differs")
    result = []
    streams = 0
    for formal in lowered:
        index, operand, metadata = by_source[formal.ir_arg_index]
        if (formal.metadata.name != metadata.name or formal.metadata.kind != metadata.kind
                or formal.metadata.ir_arg_index != metadata.ir_arg_index
                or formal.metadata.abi_arg_index != metadata.abi_arg_index
                or formal.llvm_type != flow.host_types[formal.llvm_arg_index]):
            raise ValueError("Artifact formal lost its metadata or lowered ordinal")
        leaves, constants = [], []
        if metadata.kind == stream_kind:
            streams += 1
            if (operand.origin != "environment_stream" or operand.tensor is not None or operand.scalar is not None
                    or formal.ir_arg_index != stream_layout.source_index or formal.llvm_type != stream_layout.llvm_type):
                raise ValueError("Artifact stream is not the injected original environment formal")
            size, alignment = stream_layout.size, stream_layout.alignment
        elif metadata.kind == "Var":
            if (operand.tensor is not None or operand.scalar is None or operand.scalar.kind != "integer"
                    or operand.scalar.bits not in (32, 64) or formal.llvm_type != f"i{operand.scalar.bits}"):
                raise ValueError("Artifact integer formal lost its original scalar signature")
            size = alignment = operand.scalar.bits // 8
            leaves.append(Leaf((), formal.llvm_type, "value", (), 0, size, alignment))
        else:
            if operand.tensor is None or operand.scalar is not None or operand.tensor.device.type != "cuda":
                raise ValueError("Artifact tensor signature must retain its actual CUDA source")
            tensor, target = tensors[formal.ir_arg_index], targets[formal.ir_arg_index]
            if (tensor.llvm_arg_index != formal.llvm_arg_index or target.llvm_arg_index != formal.llvm_arg_index
                    or tensor.llvm_type != formal.llvm_type or target.llvm_type != formal.llvm_type):
                raise ValueError("Artifact aggregate or layout lost the original formal")
            fields = {item.path: item for item in target.fields}
            if len(fields) != len(target.fields) or set(fields) != {leaf.path for leaf in tensor.leaves}:
                raise ValueError("Artifact tensor leaves do not cover the actual compiler layout")
            for component in (*tuple(leaf.component for leaf in tensor.leaves), *tensor.constants):
                prop = joined_properties.get(id(component))
                spec = component.spec
                _property_path(spec.property, spec.property_path, metadata)
                if (prop is None or prop.component is not component or prop.source is not operand
                        or spec.source_arg_index != formal.ir_arg_index):
                    raise ValueError("Artifact component lost its exact original Python operand")
                if spec.property == "pointer":
                    expected_use = None
                else:
                    uses = operand.tensor.shape if spec.property == "shape" else operand.tensor.strides
                    expected_use = uses[spec.property_path[0]]
                if prop.use is not expected_use:
                    raise ValueError("Artifact tensor property lost its exact FX or symbolic use")
            for leaf in tensor.leaves:
                field, spec, value = fields[leaf.path], leaf.component.spec, leaf.component.source
                if (field.llvm_type != leaf.llvm_type or value.kind != "argument"
                        or value.argument != formal.ir_arg_index or value.path != leaf.path
                        or value.value is not None or value.operands or value.attributes):
                    raise ValueError("Artifact leaves require direct compiler-identified component paths")
                if spec.property == "pointer":
                    if not leaf.llvm_type.startswith("!llvm.ptr") or field.size != 8:
                        raise ValueError("Artifact pointer leaf requires its actual 8-byte pointer layout")
                elif leaf.llvm_type not in {"i32", "i64"} or field.size != int(leaf.llvm_type[1:]) // 8:
                    raise ValueError("Artifact tensor integer field has an unsupported width")
                leaves.append(Leaf(leaf.path, leaf.llvm_type, spec.property, spec.property_path,
                                   field.offset, field.size, field.alignment))
            for component in tensor.constants:
                spec = component.spec
                if spec.property not in {"shape", "stride"} or component.source.kind not in {"constant", "zero"}:
                    raise ValueError("Artifact static properties require actual constant accessors")
                value = evaluate(component.source, (), invocation.program.context)
                if type(value) is not ScalarValue or value.llvm_type not in {"i32", "i64"}:
                    raise ValueError("Artifact static property has an unsupported scalar type")
                value.data()
                constants.append(ConstantProperty(spec.property, spec.property_path, value.llvm_type, value.integer(), value.size))
            size, alignment = target.size, target.alignment
        result.append(Formal(formal.ir_arg_index, formal.llvm_arg_index, index, metadata.name,
            formal.metadata.path, metadata.kind, formal.source_type, formal.llvm_type, metadata.shape, metadata.strides,
            metadata.dtype, metadata.element_dtype, metadata.data_address_space, metadata.device_kind,
            metadata.data_alignment, size, alignment, tuple(leaves), tuple(constants)))
    if streams != 1 or not tensors or set(tensors) != {item.source_arg_index for item in result if item.kind == "Tensor"}:
        raise ValueError("Artifact requires Tensor formals and exactly one original environment stream")
    return tuple(result)


def _copy_sites(invocation: ArtifactInvocation, fields: Any, formals: tuple[Formal, ...], consumers: tuple[Consumer, ...],
                binaries: tuple[BinaryImage, ...]) -> tuple[ArtifactSite, ...]:
    from torch._inductor.runtime._cudagraph._compiler.values import evaluate

    invocation = normalize_artifact_invocation(invocation)
    dispatch = invocation
    by_source = {formal.source_arg_index: formal for formal in formals}
    by_llvm = {formal.llvm_arg_index: formal for formal in formals}
    tensors = {formal.source_arg_index: formal for formal in invocation.plan.formals}

    def expression_copy(value):
        source = None
        if value.kind == "argument":
            formal = by_llvm.get(value.argument)
            if formal is None:
                raise ValueError("Parameter computation has no original compiler formal")
            source = formal.source_arg_index
        elif value.argument is not None:
            raise ValueError("Only argument leaves can carry an original formal")
        return ParameterExpression(value.kind, value.llvm_type, source, value.path, value.value,
            tuple(expression_copy(item) for item in value.operands), value.attributes)
    sites = []
    for site_id, (site, field_site) in enumerate(zip(dispatch.joined.sites, fields.sites)):
        if field_site.site is not site or len(field_site.nodes) != 1:
            raise ValueError("Artifact fields must identify one exact original dispatch site")
        node, = field_site.nodes
        launch = site.compiled.launch
        parameters = []
        literals = {}
        for binding, parameter in zip(site.compiled.parameters, launch.parameters):
            if (binding.parameter is not parameter or binding.device_parameter_index != parameter.index
                    or parameter.index != len(parameters)):
                raise ValueError("Artifact parameter lost its exact compiled launch slot")
            if binding.formal is None and not (parameter.source.kind in ("constant", "zero")
                                               and parameter.llvm_type in ("i32", "i64")):
                physical = field_site.physical[parameter.index]
                if physical.parameter is not parameter or node.parameter_sizes[parameter.index] != physical.size:
                    raise ValueError("Constructed parameter changed its compiler-owned field layout")
                parameters.append(Parameter(parameter.index, parameter.llvm_type, None, None,
                                            physical.size, physical.alignment))
                continue
            if binding.formal is None:
                value = parameter.source
                if (value.kind not in ("constant", "zero") or value.llvm_type not in ("i32", "i64")
                        or parameter.llvm_type != value.llvm_type or value.argument is not None
                        or value.path or value.operands or value.attributes
                        or value.kind == "zero" and value.value is not None):
                    raise ValueError("Artifact literal parameter lost its exact compiler source")
                literal = evaluate(value, (), invocation.program.context)
                if (type(literal) is not ScalarValue or literal.llvm_type != parameter.llvm_type
                        or literal.size != int(value.llvm_type[1:]) // 8
                        or node.parameter_sizes[parameter.index] != literal.size):
                    raise ValueError("Artifact literal differs from its exact physical integer width")
                literals[id(value)] = value, literal.integer()
                parameters.append(Parameter(parameter.index, parameter.llvm_type, None, None,
                                            literal.size, literal.size))
                continue
            formal = by_source[binding.formal.ir_arg_index]
            if (formal.kind not in ("Tensor", "Var") or binding.parameter is not parameter
                    or binding.device_parameter_index != parameter.index or parameter.index != len(parameters)
                    or parameter.source.argument != formal.llvm_arg_index
                    or parameter.llvm_type != formal.llvm_type or node.parameter_sizes[parameter.index] != formal.size):
                raise ValueError("Artifact parameters require exact whole Tensor or integer formal forwarding")
            parameters.append(Parameter(parameter.index, parameter.llvm_type, formal.source_arg_index,
                                        formal.llvm_arg_index, formal.size, formal.alignment))

        def source_copy(source):
            if source.kind == "compiler_expression":
                if (source.ir_arg_index is not None or source.formal_name is not None
                        or source.metadata_path or source.property != "value"
                        or not any(leaf.source is source.value and leaf.path == source.property_path
                                   for row in field_site.physical for leaf in row.fields)):
                    raise ValueError("Parameter field lost its exact compiler expression")
                return FieldSource(source.kind, None, None, (), "value", source.property_path,
                                   source.value.llvm_type, (), expression=expression_copy(source.value))
            if source.kind == "compiler_constant":
                literal = literals.get(id(source.value))
                if (literal is None or literal[0] is not source.value or source.ir_arg_index is not None
                        or source.formal_name is not None or source.metadata_path
                        or source.property != "value" or source.property_path):
                    raise ValueError("Artifact literal field lost its exact compiler parameter source")
                return FieldSource(source.kind, None, None, (), "value", (), source.value.llvm_type, (), literal[1])
            formal = by_source[source.ir_arg_index]
            if source.formal_name != formal.name or source.metadata_path != formal.metadata_path:
                raise ValueError("Artifact native field lost its exact original formal")
            if formal.kind == "Var":
                value = source.value
                if (source.kind != "scalar_formal" or source.property != "value" or source.property_path
                        or value.kind != "argument" or value.argument != formal.llvm_arg_index
                        or value.path or value.operands or value.attributes or value.value is not None
                        or value.llvm_type != formal.llvm_type
                        or not any(binding.parameter.source is value for binding in site.compiled.parameters)):
                    raise ValueError("Artifact scalar field lost its exact compiler formal source")
            else:
                leaves = [leaf for leaf in tensors[source.ir_arg_index].leaves if leaf.path == source.value.path]
                if (source.kind != "tensor_property" or len(leaves) != 1
                        or leaves[0].component.source is not source.value
                        or leaves[0].component.spec.property != source.property
                        or leaves[0].component.spec.property_path != source.property_path):
                    raise ValueError("Artifact native field lost its exact component source")
            return FieldSource(source.kind, source.ir_arg_index, source.formal_name, source.metadata_path,
                               source.property, source.property_path, source.value.llvm_type, source.value.path)

        if node.fixed or len(parameters) != len(launch.parameters):
            raise ValueError("Artifact native fields require complete non-fixed parameter coverage")
        pointers = tuple(PointerField(item.parameter, item.byte_offset, source_copy(item.source)) for item in node.pointers)
        integers = tuple(IntegerField(item.parameter, item.byte_offset, item.dtype, source_copy(item.source)) for item in node.integers)
        padding = tuple(Padding(item.parameter, item.byte_offset, item.byte_size) for item in node.padding)
        undefined = tuple(UndefinedField(item.parameter, item.byte_offset, item.byte_size,
                                         expression_copy(item.source)) for item in node.undefined)
        constants = tuple(ConstantField(item.parameter, item.byte_offset, item.data, expression_copy(item.source))
                          for item in node.constants)
        copied_fields = NodeFields(node.launch, node.kernel_symbol, node.parameter_sizes, pointers, integers, (), padding,
                                   undefined, constants)
        registration = Registration(*tuple(getattr(launch.registration, name) for name in Registration._fields))
        images = [image for image in binaries if image.library_slot == registration.library_slot]
        if (len(images) != 1 or images[0].global_name != registration.binary_global
                or images[0].sha256 != registration.binary_sha256 or registration.kernel_symbol != site.source.callee[1]
                or node.launch != launch.index or node.kernel_symbol != registration.kernel_symbol):
            raise ValueError("Artifact site lost its actual embedded binary or function registration")
        diagnostics = tuple(Diagnostic(item.kind, item.expected, item.limit, item.arch) for item in site.source.diagnostics)
        consumer_ids = tuple(item.consumer_id for item in consumers if item.site_id == site_id)
        expected = {("grid", axis): "i32" for axis in range(3)} | {("block", axis): "i32" for axis in range(3)}
        expected.update({("shared", 0): "i64", ("kernel_smem", 0): "i64"})
        expected.update({("diagnostic", index): "i1" for index in range(len(diagnostics))})
        stride_divisors = tuple(item.divisor for item in site.source.tma_strides)
        stride_groups = {}
        for index, requirement in enumerate(site.source.tma_strides):
            if requirement.group is not None:
                key = requirement.constructor, requirement.group, requirement.new_bits // 8
                stride_groups.setdefault(key, []).append(index)
        stride_domains = tuple(TmaStrideDomain(tuple(indices), key[2]) for key, indices in stride_groups.items())
        expected.update({("tma_stride", index): "i64" for index in range(len(stride_divisors))})
        dimension_groups = {}
        for index, requirement in enumerate(site.source.tma_dimensions):
            key = requirement.constructor, requirement.group, requirement.grouped
            dimension_groups.setdefault(key, []).append(index)
            expected["tma_shape", index] = "i64"
            if requirement.grouped:
                expected["tma_dimension_stride", index] = "i64"
        dimension_domains = tuple(TmaDimensionDomain(tuple(indices), key[2]) for key, indices in dimension_groups.items())
        actual = {(consumers[index].role, consumers[index].index): consumers[index].result_type for index in consumer_ids}
        if actual != expected or len(consumer_ids) != len(expected):
            raise ValueError("Artifact site lacks complete exact launch-field and diagnostic consumers")
        sites.append(ArtifactSite(site_id, site.source.arm, launch.index, site.source.callee, registration,
            tuple(parameters), copied_fields, consumer_ids, diagnostics, site.source.stream_source_arg_index,
            tuple(operation.name for operation in site.source.attributes), site.source.cluster, stride_divisors,
            stride_domains, dimension_domains))
    unconditional = bool(sites) and all(site.arm is None for site in sites)
    conditional = len(sites) == 2 and {site.arm for site in sites} == {True, False}
    if (not (unconditional or conditional)
            or conditional and len({site.callee for site in sites}) != len(sites)
            or {site.launch_index for site in sites} != set(range(len(dispatch.joined.mapping.flow.launches)))):
        raise ValueError("Artifact requires complete admitted source and compiled launch coverage")
    return tuple(sites)


def prepare_dispatch_artifact(entry: Any, layout: Any, stream_layout: Any) -> DispatchArtifact:
    from torch._inductor.runtime._cudagraph._compiler.entry import DispatchEntry
    from torch._inductor.runtime._cudagraph._compiler.fields_dispatch import derive_dispatch_fields
    from torch._inductor.runtime._cudagraph._compiler.prepared_dispatch import prepare_dispatch

    if type(entry) not in (DispatchEntry, ArtifactInvocation):
        raise TypeError("Artifact preparation requires the genuine joined DispatchEntry")
    invocation = normalize_artifact_invocation(entry)
    invocation.check_layout(layout, stream_layout)
    prepared = prepare_dispatch(invocation, layout, stream_layout)
    fields = derive_dispatch_fields(invocation, layout)
    fields.check()
    dispatch = invocation
    program, signature = invocation.program, invocation.signature
    if type(signature) is not EntrySignature or layout.query.host_target != stream_layout.host_target:
        raise ValueError("Artifact preparation requires one Python, compiler, and layout owner")
    formals = _copy_formals(invocation, layout, stream_layout)
    metadata = invocation.metadata
    symbols = tuple((name, bits, divisibility) for name, bits, divisibility in metadata.symbols)
    for name, bits, divisibility in symbols:
        if (type(name) is not str or type(bits) is not int or bits not in {32, 64}
                or divisibility is not None and (type(divisibility) is not int or divisibility <= 0)):
            raise ValueError("Unsupported compiler dimension symbol")
    for formal in formals:
        for dimensions in (formal.shape, formal.strides):
            for kind, value in dimensions:
                if (type(value) is not int or kind not in {"constant", "symbol"}
                        or kind == "symbol" and not 0 <= value < len(symbols)):
                    raise ValueError("Artifact tensor property lost its compiler dimension identity")
    source_sites = dispatch.joined.sites
    consumers = []
    for index, item in enumerate(prepared.consumers):
        bound = item.bound
        site_ids = [site_id for site_id, site in enumerate(source_sites) if bound.site is site.source]
        if bound.site is not None and len(site_ids) != 1:
            raise ValueError("Artifact helper lost its exact original launch consumer")
        site_id = None if bound.site is None else site_ids[0]
        if (item.numeric.source_order != bound.source_order or item.numeric.result_types != (bound.helper.result_type,)
                or item.numeric.argument_types != tuple(next(formal.llvm_type for formal in formals
                    if formal.source_arg_index == source_index) for source_index in bound.source_order)):
            raise ValueError("Artifact helper lost its actual source tags or scalar result type")
        consumers.append(Consumer(index, bound.helper.symbol, site_id, bound.role, bound.index,
                                  bound.helper.result_type, item.numeric.source_order, item.numeric))
    consumers = tuple(consumers)
    predicates = [item for item in consumers if item.site_id is None]
    if dispatch.joined.admission.source.predicate is None:
        if predicates:
            raise ValueError("Unconditional artifact cannot contain a dispatch predicate")
    elif len(predicates) != 1 or (predicates[0].role, predicates[0].index, predicates[0].result_type) != ("predicate", 0, "i1"):
        raise ValueError("Artifact requires its exact original root predicate")
    flow = dispatch.joined.mapping.flow
    binaries = tuple(BinaryImage(image.library_slot, image.global_name, image.sha256, image.data) for image in flow.binaries)
    if (not binaries or len({image.library_slot for image in binaries}) != len(binaries)
            or any(type(image.data) is not bytes or sha256(image.data).hexdigest() != image.sha256 for image in binaries)):
        raise ValueError("Artifact embedded library bytes or identities changed")
    sites = _copy_sites(invocation, fields, formals, consumers, binaries)
    stream_kind = "Stream" if invocation.kind == "ordinary" else "EnvStream"
    stream_formal, = (formal for formal in formals if formal.kind == stream_kind)
    stream = Stream(stream_formal.source_arg_index, stream_formal.llvm_arg_index, stream_formal.operand_index,
                    stream_formal.llvm_type, stream_formal.size, stream_formal.alignment)
    if any(site.stream_source_index != stream.source_index for site in sites):
        raise ValueError("Artifact sites use different original environment streams")
    payload = _Payload(ARTIFACT_VERSION, program.function_name, program.arch, layout.query.host_target,
        invocation.source_sha256, invocation.compiled_sha256, layout.object_sha256, stream_layout.object_sha256,
        tuple(flow.host_types), symbols, tuple((formal.source_arg_index, formal.operand_index) for formal in formals),
        formals, stream, binaries, consumers, sites)
    _immutable(payload)
    if invocation.kind == "ordinary":
        guards = _OrdinaryGuards(invocation.bound, prepare_entry_source_guard(signature))
    else:
        bundle = program.bundle
        originals = (bundle.original_host, bundle.original_host.__wrapped__, bundle.original_kernel, bundle.original_kernel.__wrapped__)
        if (tuple(item.function for item in bundle._functions) != originals
                or originals[1].__code__ is not bundle.host_code or originals[3].__code__ is not bundle.kernel_code):
            raise ValueError("Artifact code pins do not cover the original compilation functions")
        functions = tuple(_FunctionGuard(fn, fn.__code__, getattr(fn, "__wrapped__", None), tuple(fn.__annotations__.items()),
                                        tuple(cell.cell_contents for cell in fn.__closure__ or ())) for fn in originals)
        namespaces = tuple((fn.__globals__, tuple(fn.__globals__.items())) for fn in (originals[1], originals[3]))
        guards = _LiveGuards(functions, namespaces, (originals[1], originals[3]), tuple(signature.operands),
                             prepare_entry_source_guard(signature))
    invocation.check()
    prepared.check()
    fields.check()
    _check_guards(signature, guards)
    result = object.__new__(DispatchArtifact)
    object.__setattr__(result, "_payload", payload)
    object.__setattr__(result, "_signature", signature)
    object.__setattr__(result, "_guards", guards)
    object.__setattr__(result, "_seal", (result, payload, signature, guards))
    result.check()
    return result


def rebind_ordinary_artifact(template: DispatchArtifact, binding: Any) -> DispatchArtifact:
    from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_reuse.binding import OrdinaryBinding

    if type(template) is not DispatchArtifact or type(binding) is not OrdinaryBinding:
        raise TypeError("Artifact rebinding requires an ordinary artifact and exact metadata binding")
    template.check()
    if type(template._guards) is not _OrdinaryGuards:
        raise TypeError("Artifact rebinding requires an original ordinary compilation")
    binding.check()
    original = template._guards.binding
    if binding.compilation is not original.compilation or binding.metadata is not original.metadata:
        raise ValueError("Artifact rebinding requires the identical ordinary compilation and metadata")
    signature = binding.signature
    if (len(template.formals) != len(binding.params)
            or tuple(sorted(template.operand_bindings)) != tuple(
                (row.metadata.ir_arg_index, index) for index, row in enumerate(binding.params))):
        raise ValueError("Artifact rebinding changed the original formal-to-operand map")
    for formal in template.formals:
        parameter = binding.params[formal.operand_index]
        operand = signature.operands[formal.operand_index]
        if (parameter.source is not operand or formal.source_arg_index != parameter.metadata.ir_arg_index
                or formal.name != operand.name or formal.kind != parameter.metadata.kind):
            raise ValueError("Artifact rebinding lost an exact original operand endpoint")
        if formal.kind != "Tensor":
            continue
        tensor = operand.tensor
        if tensor is None or operand.scalar is not None:
            raise ValueError("Artifact rebinding lost its original Tensor operand")
        # Compiler routes are immutable; their symbolic endpoints belong to this binding.
        for component in (*formal.leaves, *formal.constants):
            _property_path(component.property, component.property_path, parameter.metadata)
            if component.property == "pointer":
                continue
            uses = tensor.shape if component.property == "shape" else tensor.strides
            dimensions = parameter.metadata.shape if component.property == "shape" else parameter.metadata.strides
            axis, = component.property_path
            kind, value = dimensions[axis]
            if kind == "symbol" and not any(uses[axis] is use for use in binding.symbols[value].uses):
                raise ValueError("Artifact rebinding lost an exact original symbolic property")
    guards = _OrdinaryGuards(binding, prepare_entry_source_guard(signature))
    payload = template._payload
    result = object.__new__(DispatchArtifact)
    object.__setattr__(result, "_payload", payload)
    object.__setattr__(result, "_signature", signature)
    object.__setattr__(result, "_guards", guards)
    object.__setattr__(result, "_seal", (result, payload, signature, guards))
    result.check()
    return result
