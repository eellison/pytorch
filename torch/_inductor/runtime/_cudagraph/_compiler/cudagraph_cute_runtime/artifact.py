from __future__ import annotations

from math import gcd
from typing import Any, NamedTuple, TYPE_CHECKING

from ..entry_signature import EntrySignature
from ..metadata_regions import MetadataAggregate, UnavailablePointer
from ..owned_numeric import evaluate_owned, OwnedNumeric
from ..tma_dimension import tma_dimension
from ..values import scalar, ScalarValue


if TYPE_CHECKING:
    from types import FunctionType

    from .signature import EntrySourceGuard


ARTIFACT_VERSION = 1


class TensorProperties(NamedTuple):
    shape: tuple[int, ...]
    stride: tuple[int, ...]


class Leaf(NamedTuple):
    path: tuple[int, ...]
    llvm_type: str
    property: str
    property_path: tuple[int, ...]
    offset: int
    size: int
    alignment: int


class ConstantProperty(NamedTuple):
    property: str
    property_path: tuple[int, ...]
    llvm_type: str
    value: int
    size: int


class Formal(NamedTuple):
    source_arg_index: int
    llvm_arg_index: int
    operand_index: int
    name: str
    metadata_path: tuple[int, ...]
    kind: str
    source_type: str
    llvm_type: str
    shape: tuple[tuple[str, int], ...]
    strides: tuple[tuple[str, int], ...]
    dtype: tuple[str, int, int] | None
    element_dtype: tuple[str, int, int] | None
    data_address_space: str | None
    device_kind: int | None
    data_alignment: int | None
    size: int
    alignment: int
    leaves: tuple[Leaf, ...]
    constants: tuple[ConstantProperty, ...]


class Stream(NamedTuple):
    source_index: int
    llvm_arg_index: int
    operand_index: int
    llvm_type: str
    size: int
    alignment: int


class BinaryImage(NamedTuple):
    library_slot: int
    global_name: str
    sha256: str
    data: bytes


class Registration(NamedTuple):
    kernel_symbol: str
    handle_global: str
    library_slot: int
    binary_global: str
    binary_sha256: str


class Parameter(NamedTuple):
    index: int
    llvm_type: str
    source_arg_index: int | None
    llvm_arg_index: int | None
    size: int
    alignment: int


class ParameterExpression(NamedTuple):
    kind: str
    llvm_type: str
    source_arg_index: int | None
    path: tuple[int, ...]
    value: str | None
    operands: tuple
    attributes: tuple[tuple[str, str], ...]


class FieldSource(NamedTuple):
    kind: str
    ir_arg_index: int | None
    formal_name: str | None
    metadata_path: tuple[int, ...]
    property: str
    property_path: tuple[int, ...]
    llvm_type: str
    component_path: tuple[int, ...]
    constant: int | None = None
    expression: ParameterExpression | None = None


class PointerField(NamedTuple):
    parameter: int
    byte_offset: int
    source: FieldSource


class IntegerField(NamedTuple):
    parameter: int
    byte_offset: int
    dtype: str
    source: FieldSource


class Padding(NamedTuple):
    parameter: int
    byte_offset: int
    byte_size: int


class UndefinedField(NamedTuple):
    parameter: int
    byte_offset: int
    byte_size: int
    source: ParameterExpression


class ConstantField(NamedTuple):
    parameter: int
    byte_offset: int
    data: bytes
    source: ParameterExpression


class NodeFields(NamedTuple):
    launch: int
    kernel_symbol: str
    parameter_sizes: tuple[int, ...]
    pointers: tuple[PointerField, ...]
    integers: tuple[IntegerField, ...]
    fixed: tuple[()]
    padding: tuple[Padding, ...]
    undefined: tuple[UndefinedField, ...] = ()
    constants: tuple[ConstantField, ...] = ()

    @property
    def pointer_descriptors(self):
        return tuple((item.parameter, item.byte_offset) for item in self.pointers)

    @property
    def scalar_descriptors(self):
        return tuple(
            (item.parameter, item.byte_offset, item.dtype) for item in self.integers
        )


class Diagnostic(NamedTuple):
    kind: str
    expected: bool
    limit: int | None
    arch: str | None


class Consumer(NamedTuple):
    consumer_id: int
    symbol: str
    site_id: int | None
    role: str
    index: int
    result_type: str
    source_order: tuple[int, ...]
    numeric: OwnedNumeric


class TmaStrideDomain(NamedTuple):
    indices: tuple[int, ...]
    element_bytes: int


class TmaDimensionDomain(NamedTuple):
    indices: tuple[int, ...]
    grouped: bool


class ArtifactSite(NamedTuple):
    site_id: int
    arm: bool | None
    launch_index: int
    callee: tuple[str, ...]
    registration: Registration
    parameters: tuple[Parameter, ...]
    fields: NodeFields
    consumer_ids: tuple[int, ...]
    diagnostics: tuple[Diagnostic, ...]
    stream_source_index: int
    default_attributes: tuple[str, ...]
    cluster: tuple[int, int, int] | None = None
    tma_stride_divisors: tuple[int, ...] = ()
    tma_stride_domains: tuple[TmaStrideDomain, ...] = ()
    tma_dimension_domains: tuple[TmaDimensionDomain, ...] = ()


class _Payload(NamedTuple):
    version: int
    function_name: str
    arch: str
    host_target: str
    source_sha256: str
    compiled_sha256: str
    layout_sha256: str
    stream_layout_sha256: str
    host_types: tuple[str, ...]
    symbols: tuple[tuple[str, int, int | None], ...]
    operand_bindings: tuple[tuple[int, int], ...]
    formals: tuple[Formal, ...]
    stream: Stream
    binaries: tuple[BinaryImage, ...]
    consumers: tuple[Consumer, ...]
    sites: tuple[ArtifactSite, ...]


class _FunctionGuard(NamedTuple):
    function: FunctionType
    code: Any
    wrapped: Any
    annotations: tuple[tuple[str, Any], ...]
    closure: tuple[Any, ...]


class _LiveGuards(NamedTuple):
    functions: tuple[_FunctionGuard, ...]
    namespaces: tuple[tuple[dict[str, Any], tuple[tuple[str, Any], ...]], ...]
    bodies: tuple[FunctionType, ...]
    operands: tuple[Any, ...]
    entry: EntrySourceGuard


class _OrdinaryGuards(NamedTuple):
    binding: Any
    entry: EntrySourceGuard


class _CapturedGuards(NamedTuple):
    binding: Any
    entry: EntrySourceGuard


def _check_guards(
    signature: EntrySignature, guards: _LiveGuards | _OrdinaryGuards | _CapturedGuards
) -> None:
    if type(guards) is _CapturedGuards:
        from ..ordinary_artifact_reuse.binding import CapturedBinding

        if (
            type(guards.binding) is not CapturedBinding
            or guards.binding.signature is not signature
        ):
            raise RuntimeError("Artifact lost its captured signature binding")
        guards.binding.check()
        guards.entry.check(signature)
        return
    if type(guards) is _OrdinaryGuards:
        from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_reuse.binding import (
            OrdinaryBinding,
        )

        if (
            type(guards.binding) is not OrdinaryBinding
            or guards.binding.signature is not signature
        ):
            raise RuntimeError("Artifact lost its exact ordinary signature binding")
        guards.binding.check()
        guards.entry.check(signature)
        return
    guards.entry.check(signature)
    if len(signature.operands) != len(guards.operands) or any(
        actual is not old for actual, old in zip(signature.operands, guards.operands)
    ):
        raise RuntimeError("Artifact lost its live Python operand bindings")
    for guard in guards.functions:
        fn = guard.function
        current = tuple(fn.__annotations__.items())
        if (
            fn.__code__ is not guard.code
            or getattr(fn, "__wrapped__", None) is not guard.wrapped
            or fn.__defaults__
            or fn.__kwdefaults__
            or len(current) != len(guard.annotations)
            or any(
                name != old_name or value is not old
                for (name, value), (old_name, old) in zip(current, guard.annotations)
            )
            or len(fn.__closure__ or ()) != len(guard.closure)
            or any(
                cell.cell_contents is not old
                for cell, old in zip(fn.__closure__ or (), guard.closure)
            )
        ):
            raise RuntimeError("Artifact original host or kernel code changed")
    for namespace, items in guards.namespaces:
        if len(namespace) != len(items) or any(
            name not in namespace or namespace[name] is not old for name, old in items
        ):
            raise RuntimeError("Artifact original host or kernel globals changed")
    if any(getattr(fn, "_preprocessed", False) for fn in guards.bodies):
        raise RuntimeError("Artifact original functions were preprocessed")


class DispatchArtifact:
    __slots__ = ("_payload", "_signature", "_guards", "_seal")

    def __new__(cls):
        raise TypeError("DispatchArtifact must be created by prepare_dispatch_artifact")

    def __setattr__(self, name, value):
        raise AttributeError("A dispatch artifact is immutable")

    @property
    def signature(self) -> EntrySignature:
        return self._signature

    @property
    def formals(self):
        return self._payload.formals

    @property
    def symbols(self):
        return self._payload.symbols

    @property
    def stream(self):
        return self._payload.stream

    @property
    def operand_bindings(self):
        return self._payload.operand_bindings

    @property
    def consumers(self):
        return self._payload.consumers

    @property
    def sites(self):
        return self._payload.sites

    @property
    def binaries(self):
        return self._payload.binaries

    @property
    def host_types(self):
        return self._payload.host_types

    @property
    def host_target(self):
        return self._payload.host_target

    @property
    def arch(self):
        return self._payload.arch

    @property
    def function_name(self):
        return self._payload.function_name

    @property
    def source_sha256(self):
        return self._payload.source_sha256

    @property
    def compiled_sha256(self):
        return self._payload.compiled_sha256

    @property
    def layout_sha256(self):
        return self._payload.layout_sha256

    @property
    def stream_layout_sha256(self):
        return self._payload.stream_layout_sha256

    def check(self) -> None:
        owned = self, self._payload, self._signature, self._guards
        if (
            type(self) is not DispatchArtifact
            or type(self._payload) is not _Payload
            or type(self._signature) is not EntrySignature
            or type(self._guards) not in (_LiveGuards, _OrdinaryGuards, _CapturedGuards)
            or type(self._seal) is not tuple
            or len(self._seal) != len(owned)
            or any(actual is not old for actual, old in zip(owned, self._seal))
        ):
            raise RuntimeError(
                "Dispatch artifact identity or immutable payload changed"
            )
        _check_guards(self._signature, self._guards)
        for consumer in self.consumers:
            consumer.numeric.check()

    def check_site(self, site: ArtifactSite) -> None:
        self.check()
        if type(site) is not ArtifactSite or not any(
            site is original for original in self.sites
        ):
            raise ValueError("Dispatch site belongs to another artifact")

    def bind_properties(self, properties: dict[int, TensorProperties]) -> RuntimeInputs:
        return bind_properties(self, properties)

    def select(self, inputs: RuntimeInputs) -> RuntimeSelection:
        return select(self, inputs)

    def select_sites(self, inputs: RuntimeInputs) -> tuple[RuntimeSelection, ...]:
        return select_sites(self, inputs)


def _value_state(value: Any) -> tuple[Any, ...]:
    if (
        type(value) not in (MetadataAggregate, UnavailablePointer, ScalarValue)
        or type(value.llvm_type) is not str
    ):
        raise TypeError("Runtime metadata requires an exact LLVM type string")
    if type(value) is MetadataAggregate:
        if type(value.fields) is not tuple or any(
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not tuple
            or any(type(index) is not int or index < 0 for index in pair[0])
            or type(pair[1]) not in (ScalarValue, UnavailablePointer)
            for pair in value.fields
        ):
            raise ValueError("Runtime metadata has invalid aggregate fields")
        return (
            id(value),
            value.llvm_type,
            id(value.fields),
            tuple((path, _value_state(item)) for path, item in value.fields),
        )
    if (
        type(value) not in (UnavailablePointer, ScalarValue)
        or type(value.size) is not int
        or value.size <= 0
    ):
        raise TypeError(
            "Runtime metadata requires exact scalar or unavailable-pointer records"
        )
    if type(value) is UnavailablePointer:
        return id(value), value.llvm_type, value.size
    if type(value.value) not in (int, float):
        raise TypeError("Runtime scalar payloads require exact Python numbers")
    number = value.value.hex() if type(value.value) is float else value.value
    return (
        id(value),
        value.llvm_type,
        value.size,
        type(value.value),
        number,
        value.data(),
    )


class RuntimeInputs:
    __slots__ = ("_artifact", "_properties", "_source_values", "_seal")

    def __new__(cls):
        raise TypeError("RuntimeInputs must be created by artifact.bind_properties")

    def __setattr__(self, name, value):
        raise AttributeError("Runtime input ownership is immutable")

    @property
    def artifact(self):
        return self._artifact

    @property
    def properties(self):
        return self._properties

    @property
    def source_values(self):
        return self._source_values

    @property
    def values(self):
        return self._source_values

    def check(self) -> None:
        owned = self, self._artifact, self._properties, self._source_values
        if (
            type(self) is not RuntimeInputs
            or type(self._artifact) is not DispatchArtifact
            or type(self._seal) is not tuple
            or len(self._seal) != 5
            or any(actual is not old for actual, old in zip(owned, self._seal[:4]))
        ):
            raise RuntimeError("Runtime inputs changed artifact or value ownership")
        self._artifact.check()
        if (
            tuple((index, _value_state(value)) for index, value in self.source_values)
            != self._seal[4]
        ):
            raise RuntimeError("Runtime input scalar or aggregate metadata changed")


def bind_properties(
    artifact: DispatchArtifact, properties: dict[int, TensorProperties]
) -> RuntimeInputs:
    if type(artifact) is not DispatchArtifact:
        raise TypeError("Expected a factory-owned dispatch artifact")
    artifact.check()
    tensors = tuple(formal for formal in artifact.formals if formal.kind == "Tensor")
    if (
        type(properties) is not dict
        or any(type(index) is not int for index in properties)
        or set(properties) != {formal.source_arg_index for formal in tensors}
    ):
        raise ValueError(
            "Tensor properties must identify every original tensor formal exactly"
        )
    symbols, copied, values = {}, [], []
    for formal in tensors:
        value = properties[formal.source_arg_index]
        if type(value) is not TensorProperties:
            raise TypeError("Expected immutable tensor shape and stride metadata")
        for prop, actual, dimensions in (
            ("shape", value.shape, formal.shape),
            ("stride", value.stride, formal.strides),
        ):
            if type(actual) is not tuple or len(actual) != len(dimensions):
                raise ValueError("Tensor metadata changed its compiler rank")
            for number, (kind, constant) in zip(actual, dimensions):
                if type(number) is not int or number < (1 if prop == "shape" else 0):
                    raise ValueError(
                        "This metadata subset requires positive shapes and nonnegative strides"
                    )
                if kind == "constant":
                    if number != constant:
                        raise ValueError(
                            "Tensor metadata violates a static compiler property"
                        )
                elif kind == "symbol":
                    _, bits, divisibility = artifact.symbols[constant]
                    if not 0 <= number < 2 ** (bits - 1):
                        raise ValueError(
                            "Tensor metadata exceeds its signed compiler dimension width"
                        )
                    if divisibility is not None and number % divisibility:
                        raise ValueError(
                            "Tensor metadata violates compiler dimension divisibility"
                        )
                    if constant in symbols and symbols[constant] != number:
                        raise ValueError(
                            "Tensor metadata disagrees on a shared compiler dimension symbol"
                        )
                    symbols[constant] = number
                else:
                    raise ValueError("Unsupported compiler tensor dimension")
        copied.append(
            (
                formal.source_arg_index,
                TensorProperties(tuple(value.shape), tuple(value.stride)),
            )
        )
        leaves = []
        for leaf in formal.leaves:
            if leaf.property == "pointer":
                item = UnavailablePointer(leaf.llvm_type, leaf.size)
            else:
                numbers = value.shape if leaf.property == "shape" else value.stride
                number = numbers[leaf.property_path[0]]
                if not 0 <= number < 2 ** (int(leaf.llvm_type[1:]) - 1):
                    raise ValueError(
                        "Tensor property exceeds its actual lowered integer leaf"
                    )
                item = scalar(leaf.llvm_type, number)
                if item.size != leaf.size:
                    raise ValueError(
                        "Numeric tensor leaf differs from its compiler storage width"
                    )
            leaves.append((leaf.path, item))
        for constant in formal.constants:
            numbers = value.shape if constant.property == "shape" else value.stride
            if numbers[constant.property_path[0]] != constant.value:
                raise ValueError(
                    "Tensor metadata differs from the actual compiled constant accessor"
                )
        values.append(
            (
                formal.source_arg_index,
                MetadataAggregate(formal.llvm_type, tuple(leaves)),
            )
        )
    stream = artifact.stream
    values.append(
        (stream.source_index, UnavailablePointer(stream.llvm_type, stream.size))
    )
    props, ordered = tuple(sorted(copied)), tuple(sorted(values))
    artifact.check()
    result = object.__new__(RuntimeInputs)
    object.__setattr__(result, "_artifact", artifact)
    object.__setattr__(result, "_properties", props)
    object.__setattr__(result, "_source_values", ordered)
    state = tuple((index, _value_state(value)) for index, value in ordered)
    object.__setattr__(result, "_seal", (result, artifact, props, ordered, state))
    result.check()
    return result


class _Selection(NamedTuple):
    artifact: DispatchArtifact
    inputs: RuntimeInputs
    site: ArtifactSite
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared: int
    kernel_smem: int
    evaluated: tuple[str, ...]
    values: tuple[tuple[str, int, ScalarValue], ...]


class RuntimeSelection:
    __slots__ = ("_record", "_seal")

    def __new__(cls):
        raise TypeError(
            "RuntimeSelection must be created by artifact.select or artifact.select_sites"
        )

    def __setattr__(self, name, value):
        raise AttributeError("Runtime selection ownership is immutable")

    @property
    def artifact(self):
        return self._record.artifact

    @property
    def inputs(self):
        return self._record.inputs

    @property
    def site(self):
        return self._record.site

    @property
    def grid(self):
        return self._record.grid

    @property
    def block(self):
        return self._record.block

    @property
    def shared(self):
        return self._record.shared

    @property
    def kernel_smem(self):
        return self._record.kernel_smem

    @property
    def evaluated(self):
        return self._record.evaluated

    @property
    def values(self):
        return self._record.values

    def check(self) -> None:
        if (
            type(self) is not RuntimeSelection
            or type(self._record) is not _Selection
            or type(self._seal) is not tuple
            or len(self._seal) != 3
            or self._seal[0] is not self
            or self._seal[1] is not self._record
        ):
            raise RuntimeError("Runtime selection ownership changed")
        self.inputs.check()
        self.artifact.check_site(self.site)
        if self.inputs.artifact is not self.artifact:
            raise ValueError("Runtime selection belongs to another artifact")
        if (
            tuple(
                (role, index, _value_state(value)) for role, index, value in self.values
            )
            != self._seal[2]
        ):
            raise RuntimeError("Runtime selection scalar results changed")


def select(artifact: DispatchArtifact, inputs: RuntimeInputs) -> RuntimeSelection:
    results = select_sites(artifact, inputs)
    if len(results) != 1:
        raise ValueError(
            "Dispatch selects multiple ordered sites; use artifact.select_sites"
        )
    return results[0]


def select_sites(
    artifact: DispatchArtifact, inputs: RuntimeInputs
) -> tuple[RuntimeSelection, ...]:
    if type(artifact) is not DispatchArtifact or type(inputs) is not RuntimeInputs:
        raise TypeError("Expected a dispatch artifact and its checked runtime inputs")
    artifact.check()
    inputs.check()
    if inputs.artifact is not artifact:
        raise ValueError("Runtime inputs belong to another dispatch artifact")
    source_values = dict(inputs.source_values)
    evaluated, values = [], []

    def compute(consumer):
        arguments = tuple(source_values[index] for index in consumer.source_order)
        result = evaluate_owned(consumer.numeric, arguments)
        if (
            len(result) != 1
            or type(result[0]) is not ScalarValue
            or result[0].llvm_type != consumer.result_type
        ):
            raise ValueError("Dispatch helper changed its exact scalar result type")
        _value_state(result[0])
        evaluated.append(consumer.symbol)
        values.append((consumer.role, consumer.index, result[0]))
        return result[0]

    predicates = tuple(
        consumer for consumer in artifact.consumers if consumer.site_id is None
    )
    if (
        not predicates
        and artifact.sites
        and all(site.arm is None for site in artifact.sites)
    ):
        sites = artifact.sites
    else:
        (predicate,) = predicates
        arm = bool(compute(predicate).integer(signed=False))
        (site,) = (site for site in artifact.sites if site.arm is arm)
        sites = (site,)
    prefix_evaluated, prefix_values = tuple(evaluated), tuple(values)
    results = []
    for site in sites:
        evaluated, values = list(prefix_evaluated), list(prefix_values)
        selected = {}
        for index in site.consumer_ids:
            consumer = artifact.consumers[index]
            selected[consumer.role, consumer.index] = compute(consumer)
        grid = tuple(selected["grid", axis].integer() for axis in range(3))
        block = tuple(selected["block", axis].integer() for axis in range(3))
        shared = selected["shared", 0].integer()
        kernel_smem = selected["kernel_smem", 0].integer()
        for index, diagnostic in enumerate(site.diagnostics):
            if (
                bool(selected["diagnostic", index].integer(signed=False))
                is not diagnostic.expected
            ):
                raise ValueError(
                    f"Original compiler shared-memory diagnostic rejected dispatch: {diagnostic.kind}"
                )
        for index, divisor in enumerate(site.tma_stride_divisors):
            if selected["tma_stride", index].integer() % divisor:
                raise ValueError(
                    "TMA descriptor outer byte stride must be a multiple of 16"
                )
        for domain in site.tma_stride_domains:
            stride = gcd(
                *(
                    selected["tma_stride", index].integer(signed=False)
                    for index in domain.indices
                )
            )
            if stride >= (1 << 40) // domain.element_bytes:
                raise ValueError("TMA descriptor outer byte stride must be below 2**40")
        for domain in site.tma_dimension_domains:
            numbers = []
            for index in domain.indices:
                numbers.append(selected["tma_shape", index].integer(signed=False))
                if domain.grouped:
                    numbers.append(
                        selected["tma_dimension_stride", index].integer(signed=False)
                    )
            if not 1 <= tma_dimension(*numbers) <= 1 << 32:
                raise ValueError("TMA descriptor dimension must be in [1, 2**32]")
        if (
            any(number <= 0 for number in (*grid, *block))
            or not 0 <= shared < 2**32
            or kernel_smem < 0
        ):
            raise ValueError(
                "Dispatch launch dimensions or shared-memory values are invalid"
            )
        inputs.check()
        record = _Selection(
            artifact,
            inputs,
            site,
            grid,
            block,
            shared,
            kernel_smem,
            tuple(evaluated),
            tuple(values),
        )
        result = object.__new__(RuntimeSelection)
        object.__setattr__(result, "_record", record)
        state = tuple(
            (role, index, _value_state(value)) for role, index, value in record.values
        )
        object.__setattr__(result, "_seal", (result, record, state))
        result.check()
        results.append(result)
    return tuple(results)
