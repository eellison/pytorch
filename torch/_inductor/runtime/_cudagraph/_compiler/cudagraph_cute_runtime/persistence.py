from hashlib import sha256

from ..owned_numeric import (
    _dump_records,
    _load_records,
    _NUMERIC_RECORDS,
    _validate_numeric,
)
from .artifact import _Payload, ARTIFACT_VERSION, ParameterExpression
from .factory import _immutable, _RECORDS


def _validate_payload(payload):
    if type(payload) is not _Payload or payload.version != ARTIFACT_VERSION:
        raise ValueError("Unsupported dispatch payload version")
    _immutable(payload)
    hashes = (
        payload.source_sha256,
        payload.compiled_sha256,
        payload.layout_sha256,
        payload.stream_layout_sha256,
    )
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in hashes
    ):
        raise ValueError("Invalid compiler object identity")
    formals = {item.source_arg_index: item for item in payload.formals}
    if (
        len(formals) != len(payload.formals)
        or any(index < 0 for index in formals)
        or tuple(item.llvm_arg_index for item in payload.formals)
        != tuple(range(len(payload.host_types)))
        or tuple(item.llvm_type for item in payload.formals) != payload.host_types
        or sorted(item.operand_index for item in payload.formals)
        != list(range(len(payload.formals)))
        or payload.operand_bindings
        != tuple(
            (item.source_arg_index, item.operand_index) for item in payload.formals
        )
    ):
        raise ValueError("Payload source formal coverage differs")
    streams = [item for item in payload.formals if item.kind in {"Stream", "EnvStream"}]
    if len(streams) != 1:
        raise ValueError("Payload requires one original stream formal")
    stream = streams[0]
    if tuple(payload.stream) != (
        stream.source_arg_index,
        stream.llvm_arg_index,
        stream.operand_index,
        stream.llvm_type,
        stream.size,
        stream.alignment,
    ):
        raise ValueError("Payload stream binding differs")
    for name, bits, divisibility in payload.symbols:
        if (
            not name
            or bits not in (32, 64)
            or divisibility is not None
            and divisibility <= 0
        ):
            raise ValueError("Invalid compiler dimension symbol")
    for formal in payload.formals:
        if (
            formal.kind not in {"Tensor", "Var", "Stream", "EnvStream"}
            or formal.size <= 0
            or formal.alignment <= 0
        ):
            raise ValueError("Invalid compiler formal layout")
        for dimensions in (formal.shape, formal.strides):
            if any(
                kind not in {"constant", "symbol"}
                or kind == "symbol"
                and not 0 <= value < len(payload.symbols)
                for kind, value in dimensions
            ):
                raise ValueError("Invalid compiler dimension reference")
        if any(
            leaf.offset < 0
            or leaf.size <= 0
            or leaf.offset + leaf.size > formal.size
            or leaf.alignment <= 0
            for leaf in formal.leaves
        ):
            raise ValueError("Formal leaf exceeds its compiler layout")
    binaries = {item.library_slot: item for item in payload.binaries}
    if (
        not binaries
        or len(binaries) != len(payload.binaries)
        or any(slot < 0 for slot in binaries)
        or any(
            sha256(item.data).hexdigest() != item.sha256 for item in payload.binaries
        )
    ):
        raise ValueError("Compiler binary bytes or identities differ")
    consumers = payload.consumers
    if tuple(item.consumer_id for item in consumers) != tuple(range(len(consumers))):
        raise ValueError("Payload consumer identities differ")
    for consumer in consumers:
        _validate_numeric(consumer.numeric)
        if (
            any(index not in formals for index in consumer.source_order)
            or consumer.source_order != consumer.numeric.source_order
            or consumer.numeric.result_types != (consumer.result_type,)
            or consumer.numeric.argument_types
            != tuple(formals[index].llvm_type for index in consumer.source_order)
        ):
            raise ValueError("Consumer lost its exact numeric source mapping")
    sites = payload.sites
    conditional = len(sites) == 2 and {site.arm for site in sites} == {True, False}
    if (
        not sites
        or not (all(site.arm is None for site in sites) or conditional)
        or tuple(site.site_id for site in sites) != tuple(range(len(sites)))
        or {site.launch_index for site in sites}
        != set(range(1 + max(site.launch_index for site in sites)))
        or conditional
        and len({site.callee for site in sites}) != 2
    ):
        raise ValueError("Payload launch coverage differs")
    predicates = [item for item in consumers if item.site_id is None]
    if (
        len(predicates) != int(conditional)
        or any(
            (item.role, item.index, item.result_type) != ("predicate", 0, "i1")
            for item in predicates
        )
        or any(
            item.site_id is not None and not 0 <= item.site_id < len(sites)
            for item in consumers
        )
    ):
        raise ValueError("Payload dispatch predicate coverage differs")
    for site in sites:
        registration, fields = site.registration, site.fields
        binary = binaries.get(registration.library_slot)
        if (
            binary is None
            or binary.global_name != registration.binary_global
            or binary.sha256 != registration.binary_sha256
            or len(site.callee) < 2
            or site.callee[1] != registration.kernel_symbol
            or fields.launch != site.launch_index
            or fields.kernel_symbol != registration.kernel_symbol
            or site.stream_source_index != payload.stream.source_index
        ):
            raise ValueError(
                "Launch lost its compiler binary, function or stream association"
            )
        if (
            tuple(item.index for item in site.parameters)
            != tuple(range(len(site.parameters)))
            or tuple(item.size for item in site.parameters) != fields.parameter_sizes
            or fields.fixed
        ):
            raise ValueError("Launch parameter layout differs")
        for parameter in site.parameters:
            if parameter.size <= 0 or parameter.alignment <= 0:
                raise ValueError("Invalid parameter size or alignment")
            if parameter.source_arg_index is not None:
                formal = formals.get(parameter.source_arg_index)
                if formal is None or (
                    parameter.llvm_arg_index,
                    parameter.llvm_type,
                    parameter.size,
                    parameter.alignment,
                ) != (
                    formal.llvm_arg_index,
                    formal.llvm_type,
                    formal.size,
                    formal.alignment,
                ):
                    raise ValueError("Parameter lost its exact original formal")
            elif parameter.llvm_arg_index is not None:
                raise ValueError("Parameter has no original source formal")
        ranges = [[] for _ in site.parameters]
        for item in (
            *fields.pointers,
            *fields.integers,
            *fields.padding,
            *fields.undefined,
            *fields.constants,
        ):
            if not 0 <= item.parameter < len(ranges):
                raise ValueError("Native field has no parameter")
            if hasattr(item, "dtype"):
                if (
                    item.dtype not in {"i32", "i64", "f32"}
                    or item.source.llvm_type != item.dtype
                ):
                    raise ValueError(
                        "Unsupported or inconsistent native scalar field type"
                    )
                size = int(item.dtype[1:]) // 8
            elif hasattr(item, "byte_size"):
                size = item.byte_size
            elif hasattr(item, "data"):
                size = len(item.data)
            else:
                size = 8
            if (
                item.byte_offset < 0
                or size <= 0
                or item.byte_offset + size > fields.parameter_sizes[item.parameter]
            ):
                raise ValueError("Native field exceeds its compiler layout")
            ranges[item.parameter].append((item.byte_offset, item.byte_offset + size))
        for size, spans in zip(fields.parameter_sizes, ranges):
            cursor = 0
            for begin, end in sorted(spans):
                if begin != cursor:
                    raise ValueError("Native field coverage has a hole or overlap")
                cursor = end
            if cursor != size:
                raise ValueError("Native fields do not cover their parameter")
        for item in (*fields.pointers, *fields.integers):
            source = item.source
            if source.kind in {"tensor_property", "scalar_formal"}:
                formal = formals.get(source.ir_arg_index)
                if (
                    formal is None
                    or source.formal_name != formal.name
                    or source.metadata_path != formal.metadata_path
                ):
                    raise ValueError("Native field source binding differs")
            elif (
                source.kind not in {"compiler_expression", "compiler_constant"}
                or source.ir_arg_index is not None
            ):
                raise ValueError("Unknown native field source")
            if (source.kind == "compiler_expression") != (
                source.expression is not None
            ):
                raise ValueError("Native field expression coverage differs")
            if (source.kind == "compiler_constant") != (source.constant is not None):
                raise ValueError("Native field constant coverage differs")
        expressions = [item.source for item in (*fields.undefined, *fields.constants)]
        expressions.extend(
            item.source.expression
            for item in (*fields.pointers, *fields.integers)
            if item.source.expression is not None
        )
        seen = set()
        while expressions:
            expression = expressions.pop()
            if type(expression) is not ParameterExpression:
                raise ValueError("Invalid parameter expression record")
            if id(expression) in seen:
                continue
            seen.add(id(expression))
            if (
                (expression.kind == "argument")
                != (expression.source_arg_index is not None)
                or expression.source_arg_index is not None
                and expression.source_arg_index not in formals
            ):
                raise ValueError("Parameter expression lost its source formal")
            expressions.extend(expression.operands)
        expected = {("grid", axis): "i32" for axis in range(3)} | {
            ("block", axis): "i32" for axis in range(3)
        }
        expected.update({("shared", 0): "i64", ("kernel_smem", 0): "i64"})
        expected.update(
            {("diagnostic", index): "i1" for index in range(len(site.diagnostics))}
        )
        expected.update(
            {
                ("tma_stride", index): "i64"
                for index in range(len(site.tma_stride_divisors))
            }
        )
        for domain in site.tma_stride_domains:
            if domain.element_bytes <= 0 or any(
                not 0 <= index < len(site.tma_stride_divisors)
                for index in domain.indices
            ):
                raise ValueError("Invalid TMA stride domain")
        for domain in site.tma_dimension_domains:
            for index in domain.indices:
                expected["tma_shape", index] = "i64"
                if domain.grouped:
                    expected["tma_dimension_stride", index] = "i64"
        if (
            site.consumer_ids
            != tuple(
                item.consumer_id for item in consumers if item.site_id == site.site_id
            )
            or len(site.consumer_ids) != len(expected)
            or {
                (consumers[index].role, consumers[index].index): consumers[
                    index
                ].result_type
                for index in site.consumer_ids
            }
            != expected
        ):
            raise ValueError(
                "Payload lacks exact launch-field and diagnostic consumers"
            )


def dump_payload(payload: _Payload) -> bytes:
    _validate_payload(payload)
    return _dump_records(payload, "cute.payload", (*_NUMERIC_RECORDS, *_RECORDS))


def load_payload(data: bytes) -> _Payload:
    result = _load_records(data, "cute.payload", (*_NUMERIC_RECORDS, *_RECORDS))
    _validate_payload(result)
    return result
