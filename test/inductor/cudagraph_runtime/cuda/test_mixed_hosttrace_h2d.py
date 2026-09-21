# Owner(s): ["module: inductor"]

import gc
import os
import sys
from unittest import mock

import triton
import triton.language as tl

import torch
from torch._inductor.runtime._cudagraph import direct_host
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    InputContract,
    IntegerRange,
    TensorInput,
)
from torch._inductor.runtime._cudagraph.direct_cuda_host import DirectCudaHost
from torch._inductor.runtime._cudagraph.direct_triton import DirectTriton
from torch._inductor.runtime._cudagraph.frontend import (
    DirectHostTable,
    DirectKernelCall,
    DirectMemcpy,
    DirectPhysicalCall,
)
from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource, IntExpr
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from host_trace_h2d_probe import probe  # noqa: E402


@triton.jit(do_not_specialize=["N"], do_not_specialize_on_alignment=["N"])
def pointwise(X, Y, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(Y + i, tl.load(X + i, i < N, other=0) + 1, i < N)


KERNEL = GROUPED = HISTORY = None
REPEAT = False


def grouped(*args):
    return probe().grouped_mul(*args)


def host(box):
    n, x, y = box
    box.clear()
    backing = torch.empty((n + 4,), dtype=x.dtype, device=x.device)
    view = backing[2 : n + 2]
    block = 64 if n < 128 else 128
    KERNEL[lambda meta: (triton.cdiv(n, meta["B"]),)](x, view, n, B=block)
    first = GROUPED(view, y, x, y)
    middle = GROUPED(first, first) if REPEAT else first
    output = torch.empty_like(middle)
    count = middle.numel()
    KERNEL[lambda meta: (triton.cdiv(count, meta["B"]),)](middle, output, count, B=128)
    return (output,)


def copy_history_host(box):
    n, source = box
    box.clear()
    backing = torch.empty((n + 4,), dtype=source.dtype, device=source.device)
    view = backing[2 : n + 2]
    block = 64 if n < 128 else 128
    KERNEL[lambda meta: (triton.cdiv(n, meta["B"]),)](source, view, n, B=block)
    copied = HISTORY(view)
    output = torch.empty_like(copied)
    count = copied.numel()
    KERNEL[lambda meta: (triton.cdiv(count, meta["B"]),)](copied, output, count, B=128)
    return (output,)


class TestMixedHostTraceH2D(TestCase):
    def runtime(self, repeated):
        kernel = DirectTriton(pointwise)
        self.addCleanup(kernel.close)
        self.enterContext(
            mock.patch.dict(
                globals(),
                KERNEL=kernel,
                GROUPED=DirectCudaHost(grouped),
                REPEAT=repeated,
            )
        )
        n = IntExpr("boxed", 0)
        contract = InputContract(
            ("integer", "tensor", "tensor"),
            (
                TensorInput(1, torch.float32, (n,), (1,)),
                TensorInput(2, torch.float32, (n,), (1,)),
            ),
            (IntegerRange(0, 1, 10000),),
            torch.cuda.current_device(),
        )
        runtime = direct_host.DirectHost(host, contract)
        self.addCleanup(runtime.close)
        return runtime

    @parametrize("repeated", (False, True))
    def test_composition(self, device, repeated):
        runtime = self.runtime(repeated)
        held = []
        for size, variants in ((64, 1), (96, 1), (33, 1), (256, 2), (64, 2)):
            x = torch.randn(size, device=device)
            y = torch.randn(size, device=device)
            wanted = torch.cat(((x + 1) * y, x * y))
            if repeated:
                wanted = wanted * wanted
            box = [size, x, y]
            (output,) = runtime(box)
            self.assertEqual(box, [])
            self.assertEqual(output, wanted + 1)
            held.append(output)
            self.assertEqual(len(runtime.variants), variants)
        tables = [
            event
            for event in runtime.variants[0].program.events
            if type(event) is DirectHostTable
        ]
        self.assertEqual(len(tables), 4 if repeated else 2)
        pointer_table = tables[0].table
        self.assertIsInstance(pointer_table.elements[0][2].root, BufferSource)
        self.assertIsInstance(pointer_table.elements[2][2].root, BufferSource)

    @parametrize("repeated", (False, True))
    def test_queued_native_hits_keep_tables_and_outputs_alive(self, device, repeated):
        runtime = self.runtime(repeated)
        runtime([64, torch.randn(64, device=device), torch.randn(64, device=device)])
        self.assertEqual(len(runtime.variants), 1)
        inputs = [
            (size, torch.randn(size, device=device), torch.randn(size, device=device))
            for size in (64, 96, 33, 80, 70, 64)
        ]
        expected = []
        for _, x, y in inputs:
            value = torch.cat(((x + 1) * y, x * y))
            expected.append((value * value if repeated else value) + 1)
        outputs = []
        with mock.patch.object(
            direct_host,
            "_observe_direct",
            side_effect=AssertionError("Native hit invoked the Python host"),
        ):
            for args in inputs:
                outputs.append(runtime(list(args))[0])
                pressure = torch.empty(2048, device=device)
                pressure.fill_(17)
        inputs.clear()
        gc.collect()
        torch.cuda.synchronize(device)
        for output, reference in zip(outputs, expected, strict=True):
            self.assertEqual(output, reference)
        self.assertEqual(len(runtime.variants), 1)

    @parametrize("rewrite", (False, True))
    def test_table_copy_history_composes_with_native_old_variant_hits(
        self, device, rewrite
    ):
        kernel = DirectTriton(pointwise)
        self.addCleanup(kernel.close)
        fn = (
            probe().copy_rewrite_copy
            if rewrite
            else probe().copy_twice
        )
        self.enterContext(
            mock.patch.dict(globals(), KERNEL=kernel, HISTORY=DirectCudaHost(fn))
        )
        n = IntExpr("boxed", 0)
        contract = InputContract(
            ("integer", "tensor"),
            (TensorInput(1, torch.float32, (n,), (1,)),),
            (IntegerRange(0, 1, 10000),),
            torch.cuda.current_device(),
        )
        runtime = direct_host.DirectHost(copy_history_host, contract)
        self.addCleanup(runtime.close)
        sizes = (64, 96, 256, 512, 80, 64)
        inputs = [torch.randn(size, device=device) for size in sizes]
        self.assertEqual(len({source.data_ptr() for source in inputs}), len(inputs))
        held = []
        for index, (size, source) in enumerate(zip(sizes, inputs, strict=True)):
            box = [size, source]
            if index in (0, 2):
                (output,) = runtime(box)
            else:
                with mock.patch.object(
                    direct_host,
                    "_observe_direct",
                    side_effect=AssertionError(
                        "Native table hit invoked the Python host"
                    ),
                ):
                    (output,) = runtime(box)
            self.assertEqual(box, [])
            expected = (
                torch.tensor(
                    [size * 100, size * 200, size * 300]
                    if rewrite
                    else [size * 100 + i for i in range(4)] * 2,
                    device=device,
                )
                + 1
            )
            held.append((output, expected))
            self.assertEqual(len(runtime.variants), 1 if index < 2 else 2)

        for variant in runtime.variants:
            events = variant.program.events
            tables = [event for event in events if type(event) is DirectHostTable]
            copies = [event for event in events if type(event) is DirectMemcpy]
            calls = [event for event in events if type(event) is DirectPhysicalCall]
            self.assertEqual(len(tables), 2)
            self.assertEqual(len(copies), 2)
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                sum(type(event) is DirectKernelCall for event in events), 2
            )
            self.assertEqual([table.index for table in tables], [0, 1])
            self.assertEqual(tables[0].table.name, tables[1].table.name)
            self.assertIsNot(tables[0], tables[1])
            for table, copy in zip(tables, copies, strict=True):
                self.assertIs(copy.source, table)
                self.assertIs(table.owner, calls[0].owner)
            self.assertNotEqual(copies[0].destination.root, copies[1].destination.root)
            ordered = [tables[0], copies[0], tables[1], copies[1], calls[0]]
            positions = [
                next(i for i, event in enumerate(events) if event is item)
                for item in ordered
            ]
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(
                [len(table.table.elements) for table in tables],
                [1, 2] if rewrite else [4, 4],
            )

        inputs.clear()
        gc.collect()
        torch.cuda.synchronize(device)
        runtime.close()
        for output, expected in held:
            self.assertEqual(output, expected, atol=0, rtol=0)


instantiate_device_type_tests(TestMixedHostTraceH2D, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
