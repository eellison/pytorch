# Owner(s): ["module: inductor"]

import ctypes
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    InputContract,
    IntegerRange,
    TensorInput,
)
from torch._inductor.runtime._cudagraph.direct_cuda_host import (
    CudaInvocation,
    DirectCudaHost,
)
from torch._inductor.runtime._cudagraph.direct_host import (
    _direct_origin,
    _TracedInvocations,
)
from torch._inductor.runtime._cudagraph.direct_invocation import activate
from torch._inductor.runtime._cudagraph.extraction import trace_host
from torch._inductor.runtime._cudagraph.frontend import lower_terminal
from torch._inductor.runtime._cudagraph.guard_export import prepare_guard
from torch._inductor.runtime.cudagraph_arg_mapping import IntExpr
from torch._inductor.runtime.cudagraph_host_trace_mapping import HostTraceSymbolMapping
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.cuda._host_trace import _TraceShapeEnv
from torch.fx.experimental.symbolic_shapes import DimDynamic
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


ADAPTER = None
OFFSET = 2
INPUT_IDENTITY = True


def ordinary(source):
    raise AssertionError("Symbolic tracing must not invoke the ordinary entry")


def host(box):
    rows, source = box
    box.clear()
    backing = torch.empty_strided(
        (rows * 128 + 4,), (1,), dtype=source.dtype, device=source.device
    )
    view = backing[OFFSET : rows * 128 + OFFSET].view(rows, 128)
    first, second = ADAPTER(view)
    if first is not second or (INPUT_IDENTITY and first is not view):
        raise AssertionError("Recorded output identity was not preserved")
    return first, second


def make_owner(offset, input_identity):
    env = _TraceShapeEnv()

    def symbol(name, hint):
        source = LocalSource(name)
        expr = env.create_unspecified_symbol(hint, source, DimDynamic.DYNAMIC)
        return env.create_symintnode(expr, hint=hint, source=source)

    rows = symbol("rows", 7)
    width = symbol("width", 128)
    root = SimpleNamespace(name="p0", itemsize=4, sym=symbol("base", 4096))
    source = SimpleNamespace(
        position=0,
        dtype=torch.float32,
        device=torch.device("cpu"),
        pinned=False,
        sizes=[rows, width],
        strides=[symbol("stride0", 128), symbol("stride1", 1)],
        offset=symbol("offset", offset),
        root=root,
    )
    if rows != 7:
        raise AssertionError("The local trace must select its equality branch")
    output = SimpleNamespace(
        root=root,
        sizes=[7, width],
        strides=source.strides,
        offset=source.offset,
        dtype=source.dtype,
    )
    tape = SimpleNamespace(
        nargs=1,
        shape_env=env,
        inputs=[source],
        allocs=[],
        launches=[],
        opaque=[],
        host_buffers=[],
        constants=(),
        memsets=[],
        memcpys=[],
        outputs=[output, output],
        guards=env.tape_guards()[0],
    )
    lowered = SimpleNamespace(
        tape=tape,
        symbols=SimpleNamespace(mapping=HostTraceSymbolMapping(tape)),
        calls=(),
        memsets=(),
        memcpys=(),
        host_buffers=(),
        extra_guards=(),
    )
    identities = (
        (("argument", 0), ("argument", 0)) if input_identity else (None, ("output", 0))
    )
    return CudaInvocation(DirectCudaHost(ordinary), tape, lowered, tuple, identities)


@instantiate_parametrized_tests
class TestCudaOutputIdentity(TestCase):
    def trace(self, owner, offset=2, input_identity=True):
        self.enterContext(
            mock.patch.dict(
                globals(),
                ADAPTER=owner.adapter,
                OFFSET=offset,
                INPUT_IDENTITY=input_identity,
            )
        )
        n = IntExpr("boxed", 0)
        contract = InputContract(
            ("integer", "tensor"),
            (TensorInput(1, torch.float32, (n, 128), (128, 1)),),
            (IntegerRange(0, 2, 127),),
            device_index=0,
        )
        origin, _ = _direct_origin(host, contract)
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            trace = trace_host(
                host,
                contract,
                [7, torch.empty(7, 128)],
                (),
                None,
                direct=True,
                context_factory=lambda state: activate(
                    _TracedInvocations(state, (owner,))
                ),
            )
        return replace(trace, compiler_binding=origin)

    @parametrize("offset", (0, 2))
    @parametrize("input_identity", (False, True))
    def test_local_equality_preserves_identity_and_reuse_guard(
        self, offset, input_identity
    ):
        owner = make_owner(offset, input_identity)
        trace = self.trace(owner, offset, input_identity)
        program = lower_terminal(trace, (owner,))
        self.addCleanup(program.close)
        guard = prepare_guard(program, [7, torch.empty(7, 128)])
        self.assertIsNotNone(guard)
        self.assertEqual(guard.boxed_pointer_indices, ())
        predicate = ctypes.CFUNCTYPE(
            ctypes.c_int8,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
        )(guard.function_address)
        for n in (6, 7, 8):
            boxed = [n, torch.empty(n, 128)]
            values = [boxed[index] for index in guard.boxed_integer_indices]
            values.extend(
                boxed[index].storage_offset()
                for index in guard.boxed_storage_offset_indices
            )
            words = (ctypes.c_int64 * len(values))(*values)
            self.assertEqual(predicate(words, None), int(n == 7))

    @parametrize("snapshot", ("missing", "changed"))
    def test_identity_requires_the_recorded_guard_snapshot(self, snapshot):
        owner = make_owner(2, True)
        if snapshot == "missing":
            owner.tape.guards = []
        else:
            width = owner.tape.inputs[0].sizes[1]
            if width != 128:
                raise AssertionError("The local trace must select its width branch")
        self.assertNotEqual(owner.tape.guards, owner.tape.shape_env.tape_guards()[0])
        with self.assertRaisesRegex(
            UnsupportedCapture, "identity lost its traced layout"
        ):
            self.trace(owner)

    def test_repeated_output_requires_the_constructed_dtype(self):
        owner = make_owner(2, False)
        owner.tape.outputs[0].dtype = torch.int32
        with self.assertRaisesRegex(
            UnsupportedCapture, "identity lost its traced layout"
        ):
            self.trace(owner, input_identity=False)

    @parametrize("difference", ("size", "stride", "offset", "dtype"))
    def test_identity_still_requires_the_recorded_layout(self, difference):
        owner = make_owner(2, True)
        output = owner.tape.outputs[0]
        if difference == "size":
            output.sizes = [8, output.sizes[1]]
        elif difference == "stride":
            output.strides = [129, output.strides[1]]
        elif difference == "offset":
            output.offset = output.offset + 1
        else:
            output.dtype = torch.int32
        with self.assertRaisesRegex(
            UnsupportedCapture, "identity lost its traced layout"
        ):
            self.trace(owner)


if __name__ == "__main__":
    run_tests()
