# Owner(s): ["module: inductor"]

import sys
import unittest
from unittest import mock

import triton
import triton.language as tl

import torch
from torch._inductor.runtime._cudagraph import direct_host, replay
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    InputContract,
    IntegerRange,
    TensorInput,
)
from torch._inductor.runtime._cudagraph.direct_cuda_host import DirectCudaHost
from torch._inductor.runtime._cudagraph.direct_hosttrace import _function_by_symbol
from torch._inductor.runtime._cudagraph.direct_triton import DirectTriton
from torch._inductor.runtime._cudagraph.frontend import DirectMemset, DirectPhysicalCall
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource,
    IntExpr,
    PointerSource,
)
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch._inductor.runtime.cudagraph_compiled_evaluation import (
    compile_numeric,
    EarlyStatus,
)
from torch._inductor.runtime.cudagraph_launch_association import (
    RecordedGraphNode,
    RecordedKernelLaunch,
)
from torch.cuda import _host_trace
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@triton.jit(do_not_specialize=["N"], do_not_specialize_on_alignment=["N"])
def before(X, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i, i < N, other=0) + 1, i < N)


@triton.jit(do_not_specialize=["N"], do_not_specialize_on_alignment=["N"])
def after(X, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i, i < N, other=0) * 2, i < N)


FIRST = SECOND = REDUCE = None


def reduction(source):
    return torch.sum(source, -1)


def host(box):
    rows, columns, source = box
    box.clear()
    storage = torch.empty_strided(
        (rows * columns + 4,), (1,), dtype=source.dtype, device=source.device
    )
    middle = storage[2 : rows * columns + 2].view(rows, columns)
    FIRST[lambda meta: (triton.cdiv(rows * columns, meta["BLOCK"]),)](
        source, middle, rows * columns, BLOCK=256
    )
    reduced = REDUCE(middle)
    output = torch.empty_like(reduced)
    SECOND[lambda meta: (triton.cdiv(rows, meta["BLOCK"]),)](
        reduced, output, rows, BLOCK=128
    )
    return (output,)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.hip is None
    and _get_torch_cuda_version() >= (12, 8),
    "requires NVIDIA CUDA >= 12.8",
)
class TestMixedCudaMemset(TestCase):
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    def test_semaphore_reset_between_triton_kernels(self, device, dtype):
        with torch.cuda.device(device):
            first, second = DirectTriton(before), DirectTriton(after)
            self.addCleanup(first.close)
            self.addCleanup(second.close)
            self.enterContext(
                mock.patch.dict(
                    globals(),
                    FIRST=first,
                    SECOND=second,
                    REDUCE=DirectCudaHost(reduction),
                )
            )
            rows, columns = IntExpr("boxed", 0), IntExpr("boxed", 1)
            contract = InputContract(
                ("integer", "integer", "tensor"),
                (TensorInput(2, dtype, (rows, columns), (columns, 1)),),
                (IntegerRange(0, 1, 32), IntegerRange(1, 131072, 300000)),
                torch.cuda.current_device(),
            )
            runtime = direct_host.DirectHost(host, contract)
            self.addCleanup(runtime.close)
            captures = []
            prepare = replay._make_replay

            def inspect_capture(*args, **kwargs):
                launches = args[6]
                events = kwargs["capture_events"]
                memsets = kwargs["memsets"]
                self.assertEqual(len(launches), 3)
                self.assertEqual(len(memsets), 1)
                self.assertEqual(
                    [type(event) for event in events],
                    [
                        RecordedKernelLaunch,
                        RecordedGraphNode,
                        RecordedKernelLaunch,
                        RecordedKernelLaunch,
                    ],
                )
                self.assertEqual(events[1].kind, "memset")
                node, destination, count, value = memsets[0]
                self.assertIs(type(destination), PointerSource)
                self.assertIs(type(destination.root), BufferSource)
                self.assertEqual(value, 0)
                self.assertEqual(node, events[1].after[3][0][0])
                self.assertIn(destination.root, args[7])
                captures.append((destination, count))
                return prepare(*args, **kwargs)

            self.enterContext(
                mock.patch.object(replay, "_make_replay", side_effect=inspect_capture)
            )
            held, frames, variants = [], [], []
            for index, columns in enumerate((262144, 200000, 262144, 200000)):
                source = torch.randn(8, columns, device=device, dtype=dtype)
                box = [8, columns, source]
                if index == 0:
                    (output,) = runtime(box)
                    self.assertIsNotNone(
                        runtime.entry, "Mixed semaphore capture declined"
                    )
                else:

                    def profile(frame, event, arg):
                        if event == "call":
                            frames.append(
                                (frame.f_code.co_filename, frame.f_code.co_name)
                            )

                    try:
                        sys.setprofile(profile)
                        (output,) = runtime.entry(box)
                    finally:
                        sys.setprofile(None)
                    self.assertEqual(frames, [])
                self.assertEqual(box, [])
                (expected,) = host([8, columns, source])
                held.append((source, output, expected))
                variants.append(len(runtime.variants))
            torch.cuda.synchronize(device)
            self.assertEqual(variants, [1, 1, 1, 1])
            self.assertEqual(len(captures), 1)
            self.assertEqual(len({source.data_ptr() for source, _, _ in held}), 4)
            self.assertEqual(len({output.data_ptr() for _, output, _ in held}), 4)
            for _, output, expected in held:
                self.assertEqual(output, expected, atol=0, rtol=0)
            program = runtime.variants[0].program
            self.assertEqual(
                sum(type(event) is DirectMemset for event in program.events), 1
            )
            self.assertEqual(
                sum(type(event) is DirectPhysicalCall for event in program.events), 1
            )
            call = next(
                event.bound
                for event in program.events
                if type(event) is DirectPhysicalCall
            )
            memset = next(
                event for event in program.events if type(event) is DirectMemset
            )
            numeric = _NumericProgram(program, (8, 262144, held[0][0]))
            block = call.block or tuple(
                IntExpr("constant", value) for value in call.module.block
            )
            shared = (
                call.shared
                if call.shared is not None
                else IntExpr("constant", call.module.shared)
            )
            slots = tuple(
                numeric.add(value)
                for value in (*call.grid, *block, shared, memset.byte_count)
            )
            compiled = compile_numeric(numeric)
            counts = []
            for source, _, _ in held:
                columns = source.size(1)
                status, values = compiled.evaluate_leaves(
                    compiled.bind_inputs((8, columns, source))
                )
                self.assertEqual(status, EarlyStatus.SUCCESS)
                storage = torch.empty(8 * columns + 4, device=device, dtype=dtype)
                middle = storage[2 : 8 * columns + 2].view(8, columns)
                tape = _host_trace.trace(reduction, (middle,), warm_up=False)
                self.assertEqual(len(tape.launches), 1)
                self.assertEqual(len(tape.memsets), 1)
                selected = tape.launches[0]
                self.assertEqual(
                    call.module.function, _function_by_symbol(int(selected["func"]))
                )
                expected = (
                    *selected["grid"],
                    *selected["block"],
                    selected["smem"],
                    tape.memsets[0]["bytes"],
                )
                expected = tuple(
                    value.node.hint if type(value) is torch.SymInt else value
                    for value in expected
                )
                self.assertEqual(tuple(values[slot] for slot in slots), expected)
                counts.append(values[slots[-1]])
            self.assertEqual(len(set(counts)), 1)


instantiate_device_type_tests(TestMixedCudaMemset, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
