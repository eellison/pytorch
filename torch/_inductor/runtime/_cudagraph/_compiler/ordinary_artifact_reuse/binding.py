"""Bind ordinary compiler metadata to original traced operand properties."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from cutlass._mlir import ir

from torch._inductor.runtime._cudagraph._compiler.accessors import _function
from torch._inductor.runtime._cudagraph._compiler.entry_signature import (
    BoundParameter,
    EntrySignature,
    MetadataSnapshot,
    ParameterMetadata,
    RuntimeRequirement,
    snapshot_metadata,
    ValueUse,
)

from .tensor_formal_properties import tensor_properties


@dataclass(frozen=True, eq=False)
class OrdinarySymbol:
    index: int
    name: str
    bits: int
    divisibility: int | None
    uses: tuple[ValueUse, ...]


def _expression(use):
    return (
        use.expression
        if use.shape_env is None
        else use.shape_env.simplify(use.expression)
    )


def _bind_parameters(signature, metadata, *, tensor_types=None, captured=False):
    """The caller must separately establish the compilation's generation owner."""
    if type(signature) is not EntrySignature or type(metadata) is not MetadataSnapshot:
        raise TypeError("Expected an exact entry signature and typed metadata snapshot")
    signature.check()
    if (
        metadata.abi != "Abi.Tbd"
        or metadata.ret != signature._expected.ret
        or len(metadata.params) != len(signature.operands)
        or len(signature._expected.params) != len(signature.operands)
    ):
        raise ValueError("Ordinary metadata changed its ABI, return or formal coverage")
    for name, bits, divisibility in metadata.symbols:
        if (
            type(name) is not str
            or type(bits) is not int
            or bits not in (32, 64)
            or divisibility is not None
            and (type(divisibility) is not int or divisibility <= 0)
        ):
            raise ValueError("Unsupported ordinary symbol width or divisibility")
    if tensor_types is not None and set(tensor_types) != {
        parameter.ir_arg_index
        for parameter in metadata.params
        if parameter.kind == "Tensor"
    }:
        raise ValueError(
            "Compiler tensor types do not cover the exact metadata formals"
        )
    uses = [[] for _ in metadata.symbols]
    params, requirements = [], list(signature.requirements)

    def require_divisibility(use, divisor):
        if divisor is None or divisor == 1:
            return
        if type(divisor) is not int or divisor <= 0:
            raise ValueError("Compiler tensor property has invalid divisibility")
        if use.shape_env is None:
            if type(use.value) is not int or use.value % divisor:
                raise ValueError(
                    "Static tensor property violates compiler divisibility"
                )
            return
        requirement = RuntimeRequirement(
            "integer_divisibility", use.path, divisor=divisor
        )
        if requirement not in requirements:
            requirements.append(requirement)

    def bind_dimension(dimension, use, bits, minimum, formal=None):
        kind, value = dimension
        if type(value) is not int:
            raise ValueError(
                "Ordinary dimension lacks an exact literal or symbol index"
            )
        maximum = (1 << (bits - 1)) - 1
        if kind == "constant":
            if (
                use.shape_env is not None
                or type(use.value) is not int
                or use.value != value
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    "Ordinary static dimension differs from its original property"
                )
        elif kind == "symbol":
            if not 0 <= value < len(uses) or metadata.symbols[value][1] != bits:
                raise ValueError(
                    "Ordinary dynamic dimension changed its source or integer width"
                )
            if use.shape_env is None:
                if type(use.value) is not int or not minimum <= use.value <= maximum:
                    raise ValueError(
                        "Original constant exceeds the compiler dimension range"
                    )
            elif use.shape_env is not signature.trace.shape_env:
                raise ValueError(
                    "Ordinary dynamic dimension changed its source or integer width"
                )
            previous = uses[value]
            if previous and (
                use.shape_env is not previous[0].shape_env
                or _expression(use) != _expression(previous[0])
            ):
                raise ValueError(
                    "One ordinary symbol refers to distinct original properties"
                )
            previous.append(use)
            if use.shape_env is not None:
                requirement = RuntimeRequirement(
                    "integer_range", use.path, minimum, maximum
                )
                if requirement not in requirements:
                    requirements.append(requirement)
            require_divisibility(use, metadata.symbols[value][2])
        else:
            raise ValueError("Unsupported ordinary dimension kind")
        if formal is not None:
            if formal.constant is not None:
                if (
                    use.shape_env is not None
                    or type(use.value) is not int
                    or use.value != formal.constant
                ):
                    raise ValueError(
                        "Compiler static tensor property differs from its original property"
                    )
            else:
                if formal.bits != bits:
                    raise ValueError(
                        "Compiler tensor property width differs from its metadata"
                    )
                require_divisibility(use, formal.divisibility)

    for index, (source, actual, expected) in enumerate(
        zip(signature.operands, metadata.params, signature._expected.params)
    ):
        if (
            source.formal_index != index
            or actual.name != source.name
            or actual.ir_arg_index != index
            or not captured
            and actual.abi_arg_index != index
        ):
            raise ValueError("Ordinary metadata lost its exact flat formal indices")
        tensor = source.tensor
        if source.scalar is not None:
            supported = (
                source.scalar.kind == "integer"
                and source.scalar.bits in (32, 64)
                or captured
                and source.scalar.kind == "float"
                and source.scalar.bits == 32
                and source.scalar.use.shape_env is None
                and type(source.scalar.use.value) is float
            )
            if (
                tensor is not None
                or source.operand is None
                or not supported
                or actual.kind != "Var"
                or replace(actual, abi_arg_index=expected.abi_arg_index) != expected
            ):
                raise ValueError(
                    "Ordinary integer formal changed its exact scalar signature"
                )
        elif tensor is None:
            if (
                source.origin != "environment_stream"
                or source.name != signature.policy.stream_name
                or source.operand is not None
                or expected.kind != "EnvStream"
                or (
                    actual != expected
                    if captured and actual.kind == "EnvStream"
                    else actual
                    != ParameterMetadata(
                        "Stream",
                        source.name,
                        index,
                        actual.abi_arg_index if captured else index,
                    )
                )
            ):
                raise ValueError(
                    "Ordinary stream must be the owner's explicit unsupplied formal"
                )
        else:
            if (
                source.operand is None
                or source.scalar is not None
                or actual.kind != "Tensor"
                or len(actual.shape) != len(tensor.shape)
                or len(actual.strides) != len(tensor.strides)
                or replace(
                    actual,
                    shape=expected.shape,
                    strides=expected.strides,
                    abi_arg_index=expected.abi_arg_index,
                )
                != expected
            ):
                raise ValueError(
                    "Ordinary Tensor dtype, device, alignment or layout contract differs"
                )
            properties = ((None,) * len(tensor.shape), (None,) * len(tensor.strides))
            if tensor_types is not None:
                properties = tensor_properties(tensor_types[index])
                if tuple(map(len, properties)) != (
                    len(tensor.shape),
                    len(tensor.strides),
                ):
                    raise ValueError(
                        "Compiler tensor property rank differs from its metadata"
                    )
            for dimension, use, formal in zip(
                actual.shape, tensor.shape, properties[0], strict=True
            ):
                bind_dimension(dimension, use, signature.policy.shape_bits, 1, formal)
            for dimension, use, formal in zip(
                actual.strides, tensor.strides, properties[1], strict=True
            ):
                bind_dimension(dimension, use, signature.policy.stride_bits, 0, formal)
        params.append(BoundParameter(source, actual))
    if any(not group for group in uses):
        raise ValueError("Ordinary metadata contains an unbound dimension symbol")
    symbols = tuple(
        OrdinarySymbol(index, *symbol, tuple(group))
        for index, (symbol, group) in enumerate(zip(metadata.symbols, uses))
    )
    return tuple(params), symbols, tuple(requirements)


def _check_compilation(signature, compilation):
    from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner import (
        OrdinaryCompilation,
    )

    if (
        type(signature) is not EntrySignature
        or type(compilation) is not OrdinaryCompilation
    ):
        raise TypeError(
            "Expected the exact ordinary compilation capsule and entry signature"
        )
    compilation.check()
    signature.check()
    owner = compilation.entry_owner
    if (
        signature.call.entry is not owner.entry
        or signature.target is not owner.entry.target
        or signature.signature != owner.signature
        or signature.policy != owner.policy
        or compilation.metadata.symbol_name != compilation.function_name
        or snapshot_metadata(compilation.function_metadata) != compilation.metadata
    ):
        raise ValueError(
            "Entry signature belongs to another ordinary compilation generation"
        )
    tensors = {
        row.name: row.tensor for row in signature.operands if row.tensor is not None
    }
    dtypes = dict(compilation.tensor_dtypes)
    if set(tensors) != set(owner.tensor_names) or set(tensors) != set(dtypes):
        raise ValueError("Traced Tensor formals differ from the ordinary compilation")
    if owner.conversion is not None:
        if any(
            tensor.dtype != dtypes[name] or tensor.device != compilation.device
            for name, tensor in tensors.items()
        ):
            raise ValueError(
                "Converted Tensor differs from its ordinary dtype or device"
            )
        return
    for name, policy in owner.tensor_policies.items():
        tensor = tensors[name]
        if (
            tensor.dtype != dtypes[name]
            or tensor.device != compilation.device
            or len(tensor.shape) != len(policy.shape)
            or any(
                use.shape_env is None
                if size is None
                else use.shape_env is not None or use.value != size
                for use, size in zip(tensor.shape, policy.shape)
            )
        ):
            raise ValueError(
                "Traced Tensor differs from the ordinary dtype, device or static dimensions"
            )
        strides, product = [None] * len(policy.shape), 1
        for axis in reversed(policy.stride_order):
            strides[axis] = product
            product *= _expression(tensor.shape[axis])
        if tuple(_expression(use) for use in tensor.strides) != tuple(strides):
            raise ValueError("Traced strides differ from the declared ordinary layout")


def _binding_state(params, symbols, requirements):
    return (
        tuple((id(row.source), row.metadata) for row in params),
        tuple(
            (
                row.index,
                row.name,
                row.bits,
                row.divisibility,
                tuple(id(use) for use in row.uses),
            )
            for row in symbols
        ),
        requirements,
    )


def _bind_compilation_parameters(signature, compilation):
    with compilation.source_context, ir.raw_values():
        host = _function(
            compilation.source_module, compilation.function_name, "func.func"
        )
        arguments = host.regions[0].blocks[0].arguments
        tensor_types = {
            parameter.ir_arg_index: arguments[parameter.ir_arg_index].type
            for parameter in compilation.metadata.params
            if parameter.kind == "Tensor"
        }
        return _bind_parameters(
            signature, compilation.metadata, tensor_types=tensor_types
        )


@dataclass(frozen=True, eq=False)
class OrdinaryBinding:
    signature: EntrySignature
    compilation: object = field(repr=False)
    metadata: MetadataSnapshot
    params: tuple[BoundParameter, ...]
    symbols: tuple[OrdinarySymbol, ...]
    requirements: tuple[RuntimeRequirement, ...]
    _seal: tuple = field(repr=False)

    def _state(self):
        return (
            id(self.signature),
            id(self.compilation),
            id(self.metadata),
            _binding_state(self.params, self.symbols, self.requirements),
        )

    def check(self):
        _check_compilation(self.signature, self.compilation)
        if (
            self.metadata is not self.compilation.metadata
            or self._state() != self._seal
        ):
            raise RuntimeError(
                "Ordinary binding ownership or property association changed"
            )
        actual = _bind_compilation_parameters(self.signature, self.compilation)
        if _binding_state(*actual) != _binding_state(
            self.params, self.symbols, self.requirements
        ):
            raise RuntimeError(
                "Ordinary binding no longer matches its original properties"
            )


def bind_ordinary_metadata(signature: EntrySignature, compilation) -> OrdinaryBinding:
    _check_compilation(signature, compilation)
    params, symbols, requirements = _bind_compilation_parameters(signature, compilation)
    result = OrdinaryBinding(
        signature, compilation, compilation.metadata, params, symbols, requirements, ()
    )
    object.__setattr__(result, "_seal", result._state())
    return result


@dataclass(frozen=True, eq=False)
class CapturedBinding:
    signature: EntrySignature
    program: object = field(repr=False)
    metadata: MetadataSnapshot
    params: tuple[BoundParameter, ...]
    symbols: tuple[OrdinarySymbol, ...]
    requirements: tuple[RuntimeRequirement, ...]
    _seal: tuple = field(repr=False)

    @property
    def kernel_owner(self):
        return self.program.kernel_owner

    @property
    def kernel_payload(self):
        return self.program.descriptor.payload

    def _state(self):
        return (
            id(self.signature),
            id(self.program),
            id(self.metadata),
            id(self.kernel_owner),
            id(self.kernel_payload),
            _binding_state(self.params, self.symbols, self.requirements),
        )

    def check(self):
        from torch.cuda._host_trace_cute_desc import Program

        if type(self.program) is not Program:
            raise TypeError("Captured binding requires its exact recorded program")
        self.program.check()
        if (
            self.metadata is not self.program.descriptor.metadata
            or self.signature.call.entry is not self.program.entry
            or self.signature.policy != self.program.policy
            or self._state() != self._seal
        ):
            raise RuntimeError(
                "Captured binding changed its program or compiler payload"
            )
        actual = _bind_parameters(self.signature, self.metadata, captured=True)
        if _binding_state(*actual) != _binding_state(
            self.params, self.symbols, self.requirements
        ):
            raise RuntimeError(
                "Captured binding lost its original property correspondence"
            )


def bind_captured_metadata(signature, program):
    program.check()
    metadata = program.descriptor.metadata
    params, symbols, requirements = _bind_parameters(signature, metadata, captured=True)
    result = CapturedBinding(
        signature, program, metadata, params, symbols, requirements, ()
    )
    object.__setattr__(result, "_seal", result._state())
    result.check()
    return result
