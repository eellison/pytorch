"""Persist the shared typed CuTe reader's payload and bind its original operands."""

from __future__ import annotations

import dataclasses
import inspect
import json
import keyword
import sys
from dataclasses import dataclass
from typing import Any

import torch


VERSION = 3


class Unexpressed(ValueError):
    pass


def available():
    module = sys.modules.get("cutlass._mlir.ir")
    return module is not None and getattr(module, "raw_values", None) is not None


def capture(function, args, kwargs):
    from cutlass.cutlass_dsl.cutlass import CuTeDSL

    from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner import (
        _OrdinaryCapture,
    )

    signature = inspect.signature(function)
    keywords = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return _OrdinaryCapture(CuTeDSL._get_dsl(), signature, args, keywords)


def _tuple_tree(value):
    if type(value) is list:
        return tuple(_tuple_tree(item) for item in value)
    if type(value) is dict:
        return {key: _tuple_tree(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class Descriptor:
    payload: Any
    metadata: Any
    parameters: tuple
    _recipe: str
    declined: str | None = None

    @property
    def function_name(self):
        return "" if self.payload is None else self.payload.function_name

    @property
    def recipe(self):
        return json.loads(self._recipe)

    def check(self):
        if self.declined is not None:
            raise Unexpressed(self.declined)
        from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.factory import (
            _immutable,
        )

        _immutable(self.payload)
        if any(site.arm is not None for site in self.payload.sites):
            registrations = {}
            for site in self.payload.sites:
                symbol = site.registration.kernel_symbol
                previous = registrations.setdefault(symbol, site.registration)
                if previous != site.registration:
                    raise Unexpressed(
                        "Observed handles cannot distinguish equal symbols from different registrations"
                    )
        if (
            self.metadata.symbol_name != self.payload.function_name
            or self.metadata.symbols != self.payload.symbols
            or len(self.metadata.params) != len(self.payload.formals)
        ):
            raise Unexpressed("Saved metadata differs from its typed compiler payload")
        by_source = {row.ir_arg_index: row for row in self.metadata.params}
        for formal in self.payload.formals:
            row = by_source.get(formal.source_arg_index)
            if row is None or (
                row.name,
                row.kind,
                row.shape,
                row.strides,
                row.dtype,
                row.data_alignment,
            ) != (
                formal.name,
                formal.kind,
                formal.shape,
                formal.strides,
                formal.dtype,
                formal.data_alignment,
            ):
                raise Unexpressed("Saved formal lost its compiler source mapping")

    def to_json(self):
        from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.persistence import (
            dump_payload,
        )

        if self.declined is None:
            self.check()
        return json.dumps(
            {
                "version": VERSION,
                "payload": None
                if self.payload is None
                else dump_payload(self.payload).decode(),
                "metadata": None
                if self.metadata is None
                else dataclasses.asdict(self.metadata),
                "parameters": self.parameters,
                "recipe": self._recipe,
                "declined": self.declined,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, text):
        from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.persistence import (
            load_payload,
        )
        from torch._inductor.runtime._cudagraph._compiler.entry_signature import (
            MetadataSnapshot,
            ParameterMetadata,
        )

        value = json.loads(text)
        if (
            type(value) is not dict
            or set(value)
            != {"version", "payload", "metadata", "parameters", "recipe", "declined"}
            or value["version"] != VERSION
        ):
            raise Unexpressed("The cache has no supported typed CuTe descriptor")
        if value["declined"] is not None:
            return cls(None, None, (), value["recipe"], value["declined"])
        metadata = _tuple_tree(value["metadata"])
        metadata["params"] = tuple(
            ParameterMetadata(**row) for row in metadata["params"]
        )
        metadata["ret"] = ParameterMetadata(**metadata["ret"])
        result = cls(
            load_payload(value["payload"].encode()),
            MetadataSnapshot(**metadata),
            _tuple_tree(value["parameters"]),
            value["recipe"],
        )
        result.check()
        return result


def build(compiled, function, args, kwargs, captured, compile_options):
    from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.captured import (
        capture_dispatch_payload,
    )

    target = function if inspect.isfunction(function) else type(function)
    recipe = json.dumps(
        {
            "callable": {
                "module": getattr(target, "__module__", ""),
                "qualname": getattr(target, "__qualname__", ""),
            }
        },
        sort_keys=True,
    )
    try:
        if captured is None:
            raise Unexpressed("The compilation has no typed source capture")
        payload = capture_dispatch_payload(compiled, captured)
        metadata = dataclasses.replace(
            captured.metadata,
            params=tuple(
                row for row in captured.metadata.params if row.ir_arg_index is not None
            ),
        )
        parameters = []
        by_name = {row.name: row for row in captured.metadata.params}
        for name, parameter in compiled.execution_args.signature.parameters.items():
            logical = by_name.get(name)
            if logical is None:
                raise Unexpressed("Runtime argument has no compiler metadata")
            if logical.kind == "EnvStream":
                continue
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise Unexpressed("Variadic compiled-call signatures are unsupported")
            default = parameter.default
            present = default is not inspect.Parameter.empty
            if (
                present
                and default is not None
                and type(default) not in (bool, int, float, str)
            ):
                raise Unexpressed("Unsupported compiled-call default value")
            parameters.append(
                (
                    name,
                    parameter.kind.name,
                    present,
                    default if present else None,
                    logical.kind == "Unit",
                )
            )
        result = Descriptor(payload, metadata, tuple(parameters), recipe)
        result.check()
        return result
    except (ValueError, TypeError, RuntimeError) as error:
        return Descriptor(
            None,
            None,
            (),
            recipe,
            f"Shared typed CuTe reader: {type(error).__name__}: {error}",
        )


def _arguments(descriptor, args, kwargs):
    parameters = tuple(
        inspect.Parameter(
            name,
            getattr(inspect.Parameter, kind),
            default=default if present else inspect.Parameter.empty,
        )
        for name, kind, present, default, _ in descriptor.parameters
    )
    bound = inspect.Signature(parameters).bind(*args, **kwargs)
    bound.apply_defaults()
    if any(
        unit and bound.arguments[name] is not None
        for name, _, _, _, unit in descriptor.parameters
    ):
        raise Unexpressed("A compiler-elided Unit operand must remain None")
    return bound.arguments


class _ProgramBorrow:
    def __init__(self, program):
        self.program = program
        self._token = object()

    def check(self):
        if self._token is None:
            raise RuntimeError("Captured program borrow is closed")
        self.program.check()

    def close(self):
        self._token = None
        self.program = None


class Program:
    def __init__(self, descriptor, name, module, kernel_owner, eager, device, context):
        import cutlass
        import cutlass.cute as cute

        from cuda.bindings import driver

        from torch._inductor.runtime._cudagraph._compiler.entry_signature import (
            SignaturePolicy,
        )
        from torch._inductor.runtime._cudagraph._compiler.python_entry import (
            PythonEntry,
        )

        descriptor.check()
        self.descriptor, self.name, self.module = descriptor, name, module
        self.kernel_owner, self.eager = kernel_owner, tuple(eager)
        self._device, self.context = torch.device(device), int(context)
        self.conversion = None
        self.owner = self
        names = tuple(row.name for row in descriptor.metadata.params)
        if any(
            not name.isidentifier() or keyword.iskeyword(name) for name in names
        ) or len(set(names)) != len(names):
            raise Unexpressed("Compiler formals cannot form a flat Python binding")
        streams = [
            row
            for row in descriptor.metadata.params
            if row.kind in ("Stream", "EnvStream")
        ]
        if len(streams) != 1:
            raise Unexpressed("Typed descriptor requires one compiler stream")
        widths = []
        for property_name in ("shape", "strides"):
            bits = {
                descriptor.metadata.symbols[index][1]
                for row in descriptor.metadata.params
                if row.kind == "Tensor"
                for kind, index in getattr(row, property_name)
                if kind == "symbol"
            }
            if len(bits) > 1:
                raise Unexpressed(
                    f"Mixed compiler {property_name} widths require a per-dimension signature policy"
                )
            widths.append(next(iter(bits), 32))
        self.policy = SignaturePolicy(*widths, 1, streams[0].name)
        namespace = {}
        exec(
            compile(
                f"def invoke({', '.join(names)}):\n    pass\n",
                "<typed CuTe operands>",
                "exec",
            ),
            namespace,
        )
        target = namespace["invoke"]
        scalar_types = {
            ("DTypeCode.Int", 32, 1): cutlass.Int32,
            ("DTypeCode.Int", 64, 1): cutlass.Int64,
            ("DTypeCode.Float", 32, 1): cutlass.Float32,
        }
        target.__annotations__ = {
            row.name: cute.Tensor
            if row.kind == "Tensor"
            else driver.CUstream
            if row.kind in ("Stream", "EnvStream")
            else scalar_types[row.dtype]
            for row in descriptor.metadata.params
        }
        self.entry = PythonEntry(target)
        self._seal = (
            descriptor,
            kernel_owner,
            self.eager,
            self.entry,
            self.policy,
            self._device,
            self.context,
        )
        self.check()

    def check(self):
        from torch.cuda import _host_trace_cute_dsl as dsl

        state = (
            self.descriptor,
            self.kernel_owner,
            self.eager,
            self.entry,
            self.policy,
            self._device,
            self.context,
        )
        if any(a is not b for a, b in zip(state, self._seal)):
            raise RuntimeError("Captured program owners changed")
        record = dsl._compiles.get(self.kernel_owner)
        if record is None or record.descriptor is not self.descriptor:
            raise RuntimeError(
                "Captured program lost its exact compiled or loaded owner"
            )
        if (
            type(self.kernel_owner) is dsl.LoadedProgram
            and self.kernel_owner.descriptor is not self.descriptor
        ):
            raise RuntimeError("Loaded program changed its paired descriptor")
        self.descriptor.check()

    def borrow_native(self):
        self.check()
        return _ProgramBorrow(self)


def _captured_launches(program, sites):
    from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.loading import (
        CapturedKernel,
    )

    program.check()
    if len(sites) != len(program.eager):
        raise Unexpressed("Observed launches differ from the selected typed host sites")
    result = []
    for site, node in zip(sites, program.eager):
        if node["name"] != site.registration.kernel_symbol:
            raise Unexpressed(
                "Observed launch order differs from the selected typed host"
            )
        result.append(
            CapturedKernel(
                site,
                int(node["func"]),
                program.kernel_owner,
                program.context,
                program._device.index,
            )
        )
    return tuple(result)


def record(tr, program, args, kwargs, operand, name):
    from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.factory import (
        bind_captured_artifact,
    )
    from torch._inductor.runtime._cudagraph._compiler.cute_bridge.provider import (
        CuTeKernelOwner,
    )
    from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_reuse.binding import (
        bind_captured_metadata,
    )
    from torch.cuda import _host_trace_cute as recorder
    from torch.utils._python_dispatch import _disable_current_modes

    def decline(reason):
        raise recorder._host_trace().Declined(
            f"host_trace: CuTe DSL kernel {name}: {reason} (declined)"
        )

    try:
        program.check()
        values = _arguments(program.descriptor, args, kwargs)
        originals, origins, names, read_only = [], {}, [], []
        for formal in program.descriptor.metadata.params:
            if formal.kind in ("Stream", "EnvStream"):
                if (
                    formal.name in values
                    and int(values[formal.name]) != tr.cute.stream.cuda_stream
                ):
                    decline("Compiled call uses another stream")
                continue
            value = values[formal.name]
            if formal.kind == "Tensor":
                value, readonly = operand(value)
                if readonly:
                    read_only.append(formal.name)
                if (
                    type(value) is not recorder._host_trace()._TracedTensor
                    or value.device != tr.device
                ):
                    decline(f"Tensor {formal.name} is not an operand of this trace")
                twin = value._fake
                origins[id(twin)] = value._root, value._sym_offset, value.dtype
                originals.append(twin)
            else:
                originals.append(value.value if hasattr(value, "value") else value)
            names.append(formal.name)
        converted = tuple(originals)
        with _disable_current_modes():
            operands = recorder._specialize(
                tr, converted, program.descriptor.metadata, names, origins, decline
            )
            invocation = recorder._bind(
                tr,
                tr.cute,
                program,
                program,
                program.descriptor,
                operands,
                converted,
                origins,
                None,
                decline,
                read_only=frozenset(read_only),
                binding_factory=bind_captured_metadata,
                artifact_factory=bind_captured_artifact,
            )
        captured = _captured_launches(program, invocation.sites)

        def owner_provider(artifact, site, *, stream, block, shared):
            matches = [row for row in captured if row.site is site]
            if len(matches) != 1:
                raise Unexpressed("Selected typed site lost its exact captured launch")
            return CuTeKernelOwner(
                artifact,
                site,
                stream=stream,
                block=block,
                shared=shared,
                captured=matches[0],
            )

        invocation = dataclasses.replace(invocation, owner_provider=owner_provider)
        recorder._record(tr, tr.cute, invocation, name)
    except (ValueError, TypeError, RuntimeError) as error:
        decline(str(error))


_side_streams = {}


def read_eager_launches(call, device):
    if torch.cuda.is_current_stream_capturing():
        raise Unexpressed("The warm-up ran inside a stream capture")
    stream = _side_streams.get(device)
    if stream is None:
        stream = _side_streams[device] = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    done = False
    with torch.cuda.device(device), torch.cuda.stream(stream):
        graph.capture_begin(capture_error_mode="thread_local")
        try:
            call()
            done = True
        finally:
            try:
                graph.capture_end()
            except Exception:
                if done:
                    raise
    nodes = torch._C._host_trace_harvest_nodes(graph.raw_cuda_graph(), -1, False)
    if any(node["kind"] != "kernel" for node in nodes):
        raise Unexpressed("The compiled host captured a non-kernel node")
    return nodes
