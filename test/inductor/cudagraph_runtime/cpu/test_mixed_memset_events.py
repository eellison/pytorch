# Owner(s): ["module: inductor"]

import ctypes
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.codecache import CppCodeCache
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    AllocateEvent,
    FXTraceDeclined,
    InputContract,
    IntegerRange,
    TensorInput,
)
from torch._inductor.runtime._cudagraph._compiler.host_program import Normalize
from torch._inductor.runtime._cudagraph.direct_cuda_host import (
    CudaHostCompleteEvent,
    CudaHostInvokeEvent,
    CudaHostMemsetEvent,
    CudaInvocation,
    DirectCudaHost,
)
from torch._inductor.runtime._cudagraph.direct_host import (
    _direct_origin,
    _TracedInvocations,
)
from torch._inductor.runtime._cudagraph.direct_invocation import activate
from torch._inductor.runtime._cudagraph.extraction import trace_host
from torch._inductor.runtime._cudagraph.frontend import (
    DirectMemset,
    DirectPhysicalCall,
    lower_terminal,
)
from torch._inductor.runtime._cudagraph.guard_export import prepare_guard
from torch._inductor.runtime._cudagraph.replay import _CaptureCall, _release_steps
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource,
    InputSource,
    IntExpr,
    PointerSource,
)
from torch._inductor.runtime.cudagraph_boxed_replay import _PhysicalCall, _PhysicalField
from torch._inductor.runtime.cudagraph_host_trace_mapping import HostTraceSymbolMapping
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


ADAPTER = None
copy_if_misaligned = torch._C._dynamo.guards.copy_if_misaligned
ZERO = IntExpr("constant", 0)
ONE = IntExpr("constant", 1)


def local_symbol(env, name, hint):
    source = LocalSource(name)
    value = env.create_unspecified_symbol(hint, source, DimDynamic.DYNAMIC)
    return env.create_symintnode(value, hint=hint, source=source)


def ordinary(source):
    raise AssertionError("Structural tracing must not invoke the ordinary kernel")


def host(box):
    _, source, other = box
    box.clear()
    output = ADAPTER(source)
    copy_if_misaligned(other)
    return output, other


def normalize_written_input(box):
    _, source, other = box
    box.clear()
    output = ADAPTER(source)
    copy_if_misaligned(source)
    return output, other


def repeated_host(box):
    _, source, other = box
    box.clear()
    first = ADAPTER(source)
    return ADAPTER(first), other


def make_owner(adapter, *, write_input=False, kernel_sequences=(2, 4)):
    env = ShapeEnv(duck_shape=False, specialize_zero_one=False)
    n = local_symbol(env, "input_size", 8)
    input_root = SimpleNamespace(
        name="p0", itemsize=4, sym=local_symbol(env, "input_base", 4096)
    )
    source = SimpleNamespace(
        position=0,
        dtype=torch.float32,
        sizes=[n],
        strides=[local_symbol(env, "input_stride", 1)],
        offset=local_symbol(env, "input_offset", 0),
        root=input_root,
    )
    q = env.create_unbacked_symint()
    root = SimpleNamespace(name="a0", itemsize=4, sym=256 * q)
    allocation = SimpleNamespace(
        name="alloc0",
        seq=0,
        q=q,
        root=root,
        dtype=torch.float32,
        sizes=[n],
        strides=[1],
    )
    tape = SimpleNamespace(
        nargs=1,
        shape_env=env,
        inputs=[source],
        allocs=[allocation],
        launches=[{"seq": seq} for seq in kernel_sequences],
        outputs=[SimpleNamespace(root=root, sizes=[n], strides=[1], offset=0)],
        guards=[sympy.Ge(n.node._expr, 4, evaluate=False)],
        opaque=[],
        host_buffers=[],
        constants=(),
        memsets=[SimpleNamespace(seq=1), SimpleNamespace(seq=3)],
    )
    module = SimpleNamespace(check=lambda: None)
    call = _PhysicalCall(
        (_PhysicalField(0, 0, "pointer", PointerSource(BufferSource("alloc0"), ZERO)),),
        module,
        (ONE, ONE, ONE),
        (),
    )
    destination = PointerSource(
        InputSource(0) if write_input else BufferSource("alloc0"), ZERO
    )
    lowered = SimpleNamespace(
        tape=tape,
        symbols=SimpleNamespace(mapping=HostTraceSymbolMapping(tape)),
        calls=tuple(call for _ in kernel_sequences),
        memsets=(
            (1, destination, IntExpr("constant", 4), 0),
            (3, destination, IntExpr("constant", 4), 17),
        ),
        extra_guards=(),
    )
    return CudaInvocation(adapter, tape, lowered, torch.Tensor, (None,))


@instantiate_parametrized_tests
class TestMixedMemsetEvents(TestCase):
    def trace(
        self,
        body=host,
        *,
        write_input=False,
        kernel_sequences=(2, 4),
        repeated=False,
        owner=None,
    ):
        adapter = DirectCudaHost(ordinary) if owner is None else owner.adapter
        if owner is None:
            owner = make_owner(
                adapter, write_input=write_input, kernel_sequences=kernel_sequences
            )
        self.enterContext(mock.patch.dict(globals(), ADAPTER=adapter))
        n = IntExpr("boxed", 0)
        contract = InputContract(
            ("integer", "tensor", "tensor"),
            (
                TensorInput(1, torch.float32, (n,), (1,)),
                TensorInput(2, torch.float32, (n,), (1,)),
            ),
            (IntegerRange(0, 2, 64),),
            device_index=0,
        )
        origin, _ = _direct_origin(body, contract)
        observations = (owner, owner) if repeated else (owner,)
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            trace = trace_host(
                body,
                contract,
                [8, torch.empty(8), torch.empty(8)],
                (),
                None,
                direct=True,
                context_factory=lambda state: activate(
                    _TracedInvocations(state, observations)
                ),
            )
        return replace(trace, compiler_binding=origin), owner

    def test_memsets_preserve_kernel_ordinals_and_normalization_position(self):
        trace, owner = self.trace()
        records = tuple(
            event
            for event in trace.events
            if type(event)
            in (CudaHostMemsetEvent, CudaHostInvokeEvent, CudaHostCompleteEvent)
        )
        self.assertEqual(
            tuple(type(event) for event in records),
            (
                CudaHostMemsetEvent,
                CudaHostInvokeEvent,
                CudaHostMemsetEvent,
                CudaHostInvokeEvent,
                CudaHostCompleteEvent,
            ),
        )
        self.assertEqual(
            tuple(
                event.call_index
                for event in records
                if type(event) is CudaHostInvokeEvent
            ),
            (0, 1),
        )
        program = lower_terminal(trace, (owner,))
        self.addCleanup(program.close)
        normalizations = tuple(
            event for event in program.events if type(event) is Normalize
        )
        self.assertEqual(normalizations, (Normalize(2, 2),))
        self.assertEqual(
            sum(type(event) is DirectMemset for event in program.events), 2
        )
        self.assertEqual(
            sum(type(event) is DirectPhysicalCall for event in program.events), 2
        )

    def test_memset_write_prevents_later_input_normalization(self):
        trace, owner = self.trace(normalize_written_input, write_input=True)
        with self.assertRaisesRegex(FXTraceDeclined, "normalization must precede"):
            lower_terminal(trace, (owner,))

    def test_completion_guard_survives_memset_only_local_invocation(self):
        trace, owner = self.trace(kernel_sequences=())
        program = lower_terminal(trace, (owner,))
        self.addCleanup(program.close)
        guard = prepare_guard(program, [8, torch.empty(8), torch.empty(8)])
        self.assertIsNotNone(guard)
        predicate = ctypes.CFUNCTYPE(
            ctypes.c_int8,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
        )(guard.function_address)
        for n, expected in ((2, 0), (3, 0), (4, 1), (8, 1), (16, 1)):
            boxed = [n, torch.empty(n), torch.empty(n)]
            values = [boxed[index] for index in guard.boxed_integer_indices]
            values.extend(
                boxed[index].data_ptr() for index in guard.boxed_pointer_indices
            )
            values.extend(
                boxed[index].storage_offset()
                for index in guard.boxed_storage_offset_indices
            )
            words = (ctypes.c_uint64 * len(values))(*values)
            self.assertEqual(
                predicate(ctypes.cast(words, ctypes.POINTER(ctypes.c_int64)), None),
                expected,
            )

    def test_two_invocations_rebase_same_local_memset_allocation_independently(self):
        trace, owner = self.trace(repeated_host, repeated=True)
        program = lower_terminal(trace, (owner,))
        self.addCleanup(program.close)
        memsets = tuple(
            event for event in program.events if type(event) is DirectMemset
        )
        self.assertEqual(len(memsets), 4)
        roots = tuple(event.destination.root for event in memsets)
        self.assertEqual(roots[0], roots[1])
        self.assertEqual(roots[2], roots[3])
        self.assertNotEqual(roots[0], roots[2])
        self.assertEqual(tuple(event.value for event in memsets), (0, 17, 0, 17))

    def test_repeated_imports_keep_callback_symbols_and_runtime_arguments(self):
        adapter = DirectCudaHost(ordinary)
        owner = make_owner(adapter)
        library = CppCodeCache.load("""#include <cstdint>
#include <vector>
static int64_t calls = 0;
extern "C" int64_t count() { return calls; }
extern "C" void reset() { calls = 0; }
extern "C" int64_t callback(const std::vector<int64_t>& args) {
  ++calls;
  return args[0];
}
""")
        library.count.restype = ctypes.c_int64
        value = local_symbol(owner.tape.shape_env, "computed_local", 8)
        n = owner.tape.inputs[0].sizes[0]
        owner.tape.opaque.append(
            {
                "sym": value,
                "args": [n],
                "expected": 8,
                "kind": "rebind",
                "seq": 0,
                "fn": "identity",
                "impl": ctypes.cast(library.callback, ctypes.c_void_p).value,
                "call": library.callback,
            }
        )
        owner.tape.guards.append(
            sympy.Eq(value.node._expr, n.node._expr, evaluate=False)
        )
        for allocation in owner.tape.allocs:
            allocation.seq += 1
        for launch in owner.tape.launches:
            launch["seq"] += 1
        owner.lowered.memsets = tuple(
            (seq + 1, *rest) for seq, *rest in owner.lowered.memsets
        )
        owner.lowered.symbols.mapping = HostTraceSymbolMapping(owner.tape)
        trace, _ = self.trace(repeated_host, repeated=True, owner=owner)
        bindings = trace.computed_integer_bindings
        self.assertEqual(len(bindings), 2)
        self.assertNotEqual(bindings[0].symbol, bindings[1].symbol)
        self.assertEqual(tuple(binding.expected for binding in bindings), (8, 8))
        program = lower_terminal(trace, (owner,))
        self.addCleanup(program.close)
        guard = prepare_guard(program, [8, torch.empty(8), torch.empty(8)])
        library.reset()
        self.assertEqual(guard.boxed_pointer_indices, ())
        predicate = ctypes.CFUNCTYPE(
            ctypes.c_int8,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
        )(guard.function_address)
        for n in (8, 16):
            values = (ctypes.c_int64 * len(guard.boxed_integer_indices))(
                *[n for _ in guard.boxed_integer_indices]
            )
            self.assertEqual(predicate(values, None), 1)
        self.assertEqual(library.count(), 4)

    def test_memset_cannot_use_allocation_before_its_outer_event(self):
        trace, owner = self.trace()
        events = list(trace.events)
        allocation = next(
            index for index, event in enumerate(events) if type(event) is AllocateEvent
        )
        memset = next(
            index
            for index, event in enumerate(events)
            if type(event) is CudaHostMemsetEvent
        )
        events[allocation], events[memset] = events[memset], events[allocation]
        with self.assertRaisesRegex(
            FXTraceDeclined, "no preceding traced storage root"
        ):
            lower_terminal(replace(trace, events=tuple(events)), (owner,))

    def test_memset_rejects_same_named_allocation_from_another_tape(self):
        trace, owner = self.trace()
        foreign = make_owner(owner.adapter)
        events = tuple(
            replace(
                event, allocations=((foreign.tape.allocs[0], event.allocations[0][1]),)
            )
            if type(event) is CudaHostMemsetEvent
            else event
            for event in trace.events
        )
        with self.assertRaisesRegex(
            UnsupportedCapture, "does not belong to this local tape"
        ):
            lower_terminal(replace(trace, events=events), (owner,))

    @parametrize("position", (0, 1, 2))
    def test_saved_input_is_retained_through_nonkernel_write(self, position):
        trace, owner = self.trace(write_input=True)
        program = lower_terminal(trace, (owner,))
        calls = tuple(
            _CaptureCall(event.bound, 1, (1, 1, 1), (), ())
            for event in program.events
            if type(event) is DirectPhysicalCall
        )
        kernel_events = tuple(
            event for event in program.events if type(event) is DirectPhysicalCall
        )
        memset = next(event for event in program.events if type(event) is DirectMemset)
        events = list(kernel_events)
        events.insert(position, memset)
        program = replace(program, events=tuple(events), saved_input_indices=(1,))
        self.assertIsNone(_release_steps(program, calls))


if __name__ == "__main__":
    run_tests()
