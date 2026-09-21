"""Private converted CUDA host boundary for a mixed terminal trace."""

from dataclasses import dataclass

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource, InputSource
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.fx.experimental.proxy_tensor import _sym_register, get_proxy_mode
from torch.fx.experimental.symbolic_shapes import DimDynamic

from .cuda_tape_import import expression, recorded_guard_pins, TapeSources
from .direct_invocation import ACTIVE
from .extraction import ComputedIntegerBinding


class DirectCudaHost:
    def __init__(self, entry):
        self.entry = entry

    def __call__(self, *arguments):
        handler = ACTIVE.get()
        if handler is None:
            return self.entry(*arguments)
        return handler.cuda(self, arguments)


@dataclass(frozen=True)
class CudaInvocationDeclined:
    reason: str


@dataclass(frozen=True, eq=False)
class CudaHostInvokeEvent:
    owner: object
    arguments: tuple
    allocations: tuple
    addresses: dict
    call_index: int
    computed: tuple = ()


@dataclass(frozen=True, eq=False)
class CudaHostCompleteEvent:
    owner: object
    arguments: tuple
    allocations: tuple
    addresses: dict
    computed: tuple = ()


@dataclass(frozen=True, eq=False)
class CudaHostMemsetEvent:
    owner: object
    arguments: tuple
    allocations: tuple
    addresses: dict
    index: int
    computed: tuple = ()


@dataclass(frozen=True, eq=False)
class CudaHostTableEvent:
    owner: object
    arguments: tuple
    allocations: tuple
    addresses: dict
    index: int
    computed: tuple = ()


@dataclass(frozen=True, eq=False)
class CudaHostMemcpyEvent:
    owner: object
    arguments: tuple
    allocations: tuple
    addresses: dict
    index: int
    computed: tuple = ()


class CudaInvocation:
    r"""Retain an observed CUDA host call and its lowering for a mixed trace."""

    providers = ()

    def __init__(self, adapter, tape, lowered, output_kind, output_identities):
        self.adapter = adapter
        self.entry = adapter.entry
        self.code = getattr(self.entry, "__code__", None)
        self.tape = tape
        self.lowered = lowered
        self.output_kind = output_kind
        self.output_identities = output_identities

    def check(self):
        if (
            self.adapter.entry is not self.entry
            or getattr(self.entry, "__code__", None) is not self.code
            or self.lowered.tape is not self.tape
        ):
            raise UnsupportedCapture("Converted CUDA host invocation changed")
        for call in self.lowered.calls:
            call.module.check()

    def trace(self, state, arguments):
        r"""Import symbolic allocations, ordered effects and outputs into ``state``.

        Kernel calls become events; their device computation is not executed.
        """
        self.check()
        roots = state.tensor_roots
        addresses = {}

        def address(resolution):
            key = resolution.root
            if key not in addresses:
                addresses[key] = resolution.root_tensor.data_ptr()
            return addresses[key]

        sources = TapeSources(
            self.tape,
            dict(enumerate(arguments)),
            roots,
            address,
            lambda value: value,
            mapping=self.lowered.symbols.mapping,
        )
        allocations, computed = [], []
        tensors: dict[InputSource | BufferSource, torch.Tensor] = {
            InputSource(index): tensor for index, tensor in enumerate(sources.arguments)
        }

        def materialize(value):
            term = state.mode.shape_env.simplify(sources.translate(value))
            return (
                int(term)
                if isinstance(term, sympy.Integer)
                else state.mode.shape_env.create_symintnode(term, hint=None)
            )

        schedule = sorted(
            [(record.seq, "allocate", record) for record in self.tape.allocs]
            + [
                (launch["seq"], "launch", index)
                for index, launch in enumerate(self.tape.launches)
            ]
            + [
                (seq, "memset", index)
                for index, (seq, _, _, _) in enumerate(self.lowered.memsets)
            ]
            + [
                (table.seq, "table", index)
                for index, table in enumerate(getattr(self.lowered, "host_tables", ()))
            ]
            + [
                (seq, "memcpy", index)
                for index, (seq, _, _, _) in enumerate(
                    getattr(self.lowered, "memcpys", ())
                )
            ]
            + [(record["seq"], "computed", record) for record in self.tape.opaque],
            key=lambda row: row[0],
        )
        if len({seq for seq, _, _ in schedule}) != len(schedule):
            raise UnsupportedCapture(
                "Converted CUDA host events have duplicate sequence numbers"
            )
        if len(self.tape.launches) != len(self.lowered.calls):
            raise UnsupportedCapture(
                "Converted CUDA host lost its lowered launch correspondence"
            )
        for _, kind, item in schedule:
            if kind == "computed":
                arguments_expr = tuple(sources.translate(arg) for arg in item["args"])
                traced_arguments = tuple(materialize(arg) for arg in item["args"])
                environment = state.mode.shape_env
                source = LocalSource(f"computed_{len(state.computed_integer_bindings)}")
                symbol = environment.create_unspecified_symbol(
                    item["expected"], source, DimDynamic.DYNAMIC
                )
                environment.constrain_symbol_range(symbol, -(1 << 63), (1 << 63) - 1)
                value = environment.create_symintnode(
                    symbol, hint=item["expected"], source=source
                )
                proxy = get_proxy_mode()
                if proxy is not None:

                    def invoke(*values, native=item["call"]):
                        return native(values)

                    _sym_register(proxy.tracer, invoke, traced_arguments, value)
                state.computed_integer_bindings.append(
                    ComputedIntegerBinding(
                        value,
                        source,
                        symbol,
                        arguments_expr,
                        item["impl"],
                        item["call"],
                        item["kind"],
                        item["expected"],
                    )
                )
                sources.bind_computed(item, value)
                computed.append((item, value))
            elif kind == "allocate":
                size = tuple(materialize(value) for value in item.sizes)
                stride = tuple(materialize(value) for value in item.strides)
                tensor = torch.empty_strided(
                    size, stride, dtype=item.dtype, device=state.device
                )
                sources.bind_allocation(item, tensor)
                allocations.append((item, tensor))
                tensors[BufferSource(item.name)] = tensor
            else:
                event_type = {
                    "memset": CudaHostMemsetEvent,
                    "memcpy": CudaHostMemcpyEvent,
                    "table": CudaHostTableEvent,
                }.get(kind, CudaHostInvokeEvent)
                state.record(
                    event_type(
                        self,
                        tuple(arguments),
                        tuple(allocations),
                        addresses,
                        item,
                        tuple(computed),
                    )
                )
        identity_pins = recorded_guard_pins(self.tape)
        outputs = []
        for record, identity in zip(
            self.tape.outputs, self.output_identities, strict=True
        ):
            local = self.lowered.symbols.mapping.root_source(record.root)
            source = tensors[local]
            if identity is None:
                size = tuple(materialize(value) for value in record.sizes)
                stride = tuple(materialize(value) for value in record.strides)
                offset = materialize(record.offset)
                if type(local) is InputSource:
                    offset = (
                        offset
                        - materialize(self.tape.inputs[local.index].offset)
                        + source.storage_offset()
                    )
                output = source.as_strided(size, stride, offset)
            else:
                kind, index = identity
                output = arguments[index] if kind == "argument" else outputs[index]
                original = (
                    next(item for item in self.tape.inputs if item.position == index)
                    if kind == "argument"
                    else self.tape.outputs[index]
                )
                expected = (*record.sizes, *record.strides, record.offset)
                actual = (*original.sizes, *original.strides, original.offset)
                # Local layout equalities can depend on guards imported at completion.
                if (
                    original.root is not record.root
                    or roots(output).root != roots(source).root
                    or original.dtype != record.dtype
                    or output.dtype != record.dtype
                    or len(actual) != len(expected)
                    or any(
                        expression(a).xreplace(identity_pins)
                        != expression(b).xreplace(identity_pins)
                        for a, b in zip(actual, expected, strict=True)
                    )
                ):
                    raise UnsupportedCapture(
                        "Converted CUDA host output identity lost its traced layout"
                    )
            outputs.append(output)
        state.record(
            CudaHostCompleteEvent(
                self, tuple(arguments), tuple(allocations), addresses, tuple(computed)
            )
        )
        if self.output_kind is torch.Tensor:
            if len(outputs) != 1:
                raise UnsupportedCapture("Converted CUDA host output arity changed")
            return outputs[0]
        return self.output_kind(outputs)
