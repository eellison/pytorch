# Owner(s): ["module: inductor"]

from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    FXTraceDeclined,
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
from torch._inductor.runtime._cudagraph.direct_hosttrace import _HostTable
from torch._inductor.runtime._cudagraph.direct_invocation import activate
from torch._inductor.runtime._cudagraph.extraction import trace_host
from torch._inductor.runtime._cudagraph.frontend import (
    DirectHostTable,
    DirectMemcpy,
    DirectPhysicalCall,
    lower_terminal,
)
from torch._inductor.runtime._cudagraph.replay import _release_steps
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource,
    InputSource,
    IntExpr,
    PointerSource,
)
from torch._inductor.runtime.cudagraph_boxed_replay import (
    _NumericProgram,
    _PhysicalCall,
    _PhysicalField,
)
from torch._inductor.runtime.cudagraph_host_trace_mapping import HostTraceSymbolMapping
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


ZERO = IntExpr("constant", 0)
ONE = IntExpr("constant", 1)
ADAPTER = None


def symbol(env, name, hint):
    source = LocalSource(name)
    value = env.create_unspecified_symbol(hint, source, DimDynamic.DYNAMIC)
    return env.create_symintnode(value, hint=hint, source=source)


def ordinary(x):
    raise AssertionError("No ordinary call during symbolic extraction")


def host(box):
    _, x = box
    box.clear()
    return (ADAPTER(x[2:]),)


def twice(box):
    _, x = box
    box.clear()
    return (ADAPTER(ADAPTER(x[2:])),)


def pin_host(box):
    _, x = box
    box.clear()
    return (x[:] if x.is_pinned() and x[:].is_pinned() else x[:1],)


def explicit_pin_query(box):
    _, x = box
    box.clear()
    if POSITIONAL:
        x.is_pinned("cuda")
    else:
        x.is_pinned(device="cuda")
    return (x,)


POSITIONAL = False


def owner(adapter, *, table_seq=2, pinned=False):
    env = ShapeEnv(duck_shape=False, specialize_zero_one=False)
    n = symbol(env, "n", 8)
    root = SimpleNamespace(name="p0", itemsize=4, sym=symbol(env, "address", 4096))
    inp = SimpleNamespace(
        position=0,
        dtype=torch.float32,
        sizes=[n],
        strides=[symbol(env, "stride", 1)],
        offset=symbol(env, "offset", 0),
        root=root,
    )
    allocs = []
    for index, size in enumerate(([n], [32])):
        q = env.create_unbacked_symint()
        allocs.append(
            SimpleNamespace(
                name=f"alloc{index}",
                seq=index,
                q=q,
                root=SimpleNamespace(name=f"a{index}", itemsize=4, sym=256 * q),
                dtype=torch.float32,
                sizes=size,
                strides=[1],
            )
        )
    tape = SimpleNamespace(
        nargs=1,
        shape_env=env,
        inputs=[inp],
        allocs=allocs,
        launches=[{"seq": 4}],
        outputs=[
            SimpleNamespace(root=allocs[0].root, sizes=[n], strides=[1], offset=0)
        ],
        guards=[sympy.Ge(n.node._expr, 2, evaluate=False)],
        opaque=[],
        constants=(),
        host_buffers=[],
        memcpys=[{"seq": 3}],
        memsets=[],
    )
    table = _HostTable(
        table_seq,
        "same_name",
        24,
        ((20, b"ABCD"),),
        (
            (0, 8, PointerSource(InputSource(0), ZERO)),
            (8, 8, PointerSource(BufferSource("alloc0"), ZERO)),
            (16, 4, IntExpr("size", 0, (ZERO,))),
        ),
    )
    call = _PhysicalCall(
        (_PhysicalField(0, 0, "pointer", PointerSource(BufferSource("alloc1"), ZERO)),),
        SimpleNamespace(check=lambda: None),
        (ONE, ONE, ONE),
        (),
    )
    source = PointerSource(InputSource(0), ZERO) if pinned else 0
    lowered = SimpleNamespace(
        tape=tape,
        symbols=SimpleNamespace(mapping=HostTraceSymbolMapping(tape)),
        calls=(call,),
        memsets=(),
        host_tables=() if pinned else (table,),
        memcpys=(
            (
                3,
                source,
                PointerSource(BufferSource("alloc1"), ZERO),
                IntExpr("constant", 24),
            ),
        ),
        extra_guards=(),
    )
    return CudaInvocation(adapter, tape, lowered, torch.Tensor, (None,))


@instantiate_parametrized_tests
class TestMixedHostTraceH2D(TestCase):
    def trace(
        self, body=host, *, repeated=False, pinned=False, table_seq=2, contract_pin=True
    ):
        adapter = DirectCudaHost(ordinary)
        first = owner(adapter, pinned=pinned, table_seq=table_seq)
        second = CudaInvocation(
            adapter, first.tape, first.lowered, torch.Tensor, (None,)
        )
        self.enterContext(mock.patch.dict(globals(), ADAPTER=adapter))
        n = IntExpr("boxed", 0)
        row = TensorInput(
            1,
            torch.float32,
            (n,),
            (1,),
            device=torch.device("cpu") if pinned else None,
            pinned=pinned and contract_pin,
        )
        contract = InputContract(
            ("integer", "tensor"), (row,), (IntegerRange(0, 4, 64),), 0
        )
        origin, _ = _direct_origin(body, contract)
        observations = (first, second) if repeated else (first,)
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            trace = trace_host(
                body,
                contract,
                [10, torch.empty(10)],
                (),
                None,
                direct=True,
                context_factory=lambda state: activate(
                    _TracedInvocations(state, observations)
                ),
            )
        return replace(trace, compiler_binding=origin)

    @parametrize("repeated", (False, True))
    def test_tables_rebase_and_keep_invocation_identity(self, repeated):
        program = lower_terminal(
            self.trace(twice if repeated else host, repeated=repeated), ()
        )
        tables = [event for event in program.events if type(event) is DirectHostTable]
        copies = [event for event in program.events if type(event) is DirectMemcpy]
        calls = [event for event in program.events if type(event) is DirectPhysicalCall]
        self.assertEqual(len(tables), 2 if repeated else 1)
        self.assertEqual(len(copies), len(tables))
        self.assertEqual(len(calls), len(tables))
        for table, copy in zip(tables, copies, strict=True):
            self.assertIs(copy.source, table)
            self.assertEqual(table.table.constants, ((20, b"ABCD"),))
            self.assertLess(program.events.index(table), program.events.index(copy))
        self.assertEqual(tables[0].table.elements[0][2].root, InputSource(1))
        if repeated:
            self.assertIsNot(tables[0].owner, tables[1].owner)
            self.assertNotEqual(
                tables[0].table.elements[1][2].root, tables[1].table.elements[1][2].root
            )
            self.assertEqual(
                tables[1].table.elements[0][2].root, tables[0].table.elements[1][2].root
            )

    @parametrize("table_seq", (5, 6))
    def test_copy_before_table_declines(self, table_seq):
        with self.assertRaisesRegex(FXTraceDeclined, "precedes its table"):
            lower_terminal(self.trace(table_seq=table_seq), ())

    def test_pinned_source_maps_to_outer_box_position(self):
        program = lower_terminal(self.trace(pinned=True), ())
        copy = next(event for event in program.events if type(event) is DirectMemcpy)
        self.assertEqual(copy.source.root, InputSource(1))

    def test_pageable_contract_does_not_inherit_pinned_fact(self):
        with self.assertRaisesRegex(FXTraceDeclined, "pinned CPU input contract"):
            lower_terminal(self.trace(pinned=True, contract_pin=False), ())

    @parametrize("pinned", (False, True))
    def test_fake_host_observes_declared_device_and_pinning(self, pinned):
        n = IntExpr("boxed", 0)
        contract = InputContract(
            ("integer", "tensor"),
            (TensorInput(1, torch.float32, (n,), (1,), torch.device("cpu"), pinned),),
            (IntegerRange(0, 4, 64),),
            0,
        )
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            trace = trace_host(
                pin_host,
                contract,
                [10, torch.empty(10)],
                (),
                None,
                direct=True,
                context_factory=lambda state: activate(_TracedInvocations(state, ())),
            )
        self.assertEqual(trace.outputs[0].device, torch.device("cpu"))
        self.assertEqual(int(trace.outputs[0].numel()), 10 if pinned else 1)

    @parametrize("positional", (False, True))
    def test_explicit_pin_query_does_not_drop_its_device(self, positional):
        self.enterContext(mock.patch.dict(globals(), POSITIONAL=positional))
        with self.assertRaisesRegex(FXTraceDeclined, "default device query"):
            self.trace(explicit_pin_query, pinned=True)

    def test_pointer_table_uses_disable_kernel_only_reclamation(self):
        program = replace(lower_terminal(self.trace(), ()), saved_input_indices=(1,))
        calls = [event for event in program.events if type(event) is DirectPhysicalCall]
        self.assertIsNone(_release_steps(program, calls))
        table = next(
            event.table for event in program.events if type(event) is DirectHostTable
        )
        kernel_roots = {
            field.source.root for call in calls for field in call.bound.fields
        }
        self.assertNotIn(table.elements[0][2].root, kernel_roots)
        self.assertNotIn(table.elements[1][2].root, kernel_roots)

    def test_pointer_offset_and_scalar_use_outer_symbol(self):
        program = lower_terminal(self.trace(), ())
        table = next(
            event.table for event in program.events if type(event) is DirectHostTable
        )
        for count in (10, 23):
            numeric = _NumericProgram(program, [count, torch.empty(count)])
            self.assertEqual(
                numeric.values[numeric.add(table.elements[0][2].byte_offset)], 8
            )
            self.assertEqual(
                numeric.values[numeric.add(table.elements[2][2])], count - 2
            )


if __name__ == "__main__":
    run_tests()
