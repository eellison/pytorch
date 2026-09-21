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
from torch._inductor.runtime._cudagraph.direct_hosttrace import HostTraceReplay
from torch._inductor.runtime._cudagraph.direct_triton import DirectTriton
from torch._inductor.runtime.cudagraph_arg_mapping import IntExpr
from torch.multiprocessing.reductions import StorageWeakRef
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
from host_trace_h2d_probe import probe


@triton.jit(do_not_specialize=["N"], do_not_specialize_on_alignment=["N"])
def add_one(X, Y, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(Y + i, tl.load(X + i, i < N, other=0) + 1, i < N)


KERNEL = GATHER = None
ROWS, WIDTH = 128, 16


def gather(table, indices):
    return probe().gather(table, indices)


def mixed_host(box):
    n, table, indices = box
    box.clear()
    first = torch.empty_like(table)
    KERNEL[(ROWS * WIDTH // 128,)](table, first, ROWS * WIDTH, B=128)
    middle = GATHER(first, indices[2 : n + 2])
    output = torch.empty_like(middle)
    KERNEL[lambda meta: (triton.cdiv(n * WIDTH, meta["B"]),)](
        middle, output, n * WIDTH, B=128
    )
    return (output,)


def unused_pinned_formal(constant, unused, table, indices):
    if constant != 7:
        raise AssertionError("Unexpected closed scalar")
    return gather(table, indices)


class TestMixedPinnedH2D(TestCase):
    def runtime(self):
        kernel = DirectTriton(add_one)
        self.addCleanup(kernel.close)
        self.enterContext(
            mock.patch.dict(globals(), KERNEL=kernel, GATHER=DirectCudaHost(gather))
        )
        n = IntExpr("boxed", 0)
        capacity = IntExpr("add", args=(n, IntExpr("constant", 4)))
        contract = InputContract(
            ("integer", "tensor", "tensor"),
            (
                TensorInput(1, torch.float32, (ROWS, WIDTH), (WIDTH, 1)),
                TensorInput(
                    2, torch.int64, (capacity,), (1,), torch.device("cpu"), True
                ),
            ),
            (IntegerRange(0, 1, ROWS),),
            torch.cuda.current_device(),
        )
        runtime = direct_host.DirectHost(mixed_host, contract)
        self.addCleanup(runtime.close)
        return runtime

    def indices(self, n):
        return torch.randint(0, ROWS, (n + 4,), dtype=torch.int64).pin_memory()

    def test_ordinary_intermediate_released_after_preparation(self, device):
        runtime = self.runtime()
        ordinary = KERNEL._run_ordinary
        temporaries = []

        def observe_output(*args, **kwargs):
            result = ordinary(*args, **kwargs)
            if args[2] == ROWS * WIDTH:
                temporaries.append(StorageWeakRef(args[1].untyped_storage()))
            return result

        table = torch.randn(ROWS, WIDTH, device=device)
        n = 16
        ids = self.indices(n)
        expected = table[ids[2 : n + 2].to(device)] + 1 + 1
        with mock.patch.object(KERNEL, "_run_ordinary", new=observe_output):
            (output,) = runtime([n, table, ids])
        torch.cuda.synchronize(device)
        gc.collect()
        self.assertEqual(len(temporaries), 1)
        self.assertTrue(temporaries[0].expired())
        self.assertEqual(output, expected, atol=0, rtol=0)
        runtime.close()
        self.assertEqual(output + 1, expected + 1, atol=0, rtol=0)

    def test_changed_pinned_views_and_native_hits(self, device):
        runtime = self.runtime()
        table = torch.randn(ROWS, WIDTH, device=device)
        for index, n in enumerate((16, 32, 48, 16)):
            ids = self.indices(n)
            reference = table[ids[2 : n + 2].to(device)] + 1 + 1
            if index:
                with mock.patch.object(
                    direct_host,
                    "_observe_direct",
                    side_effect=AssertionError("Expected native hit"),
                ):
                    (output,) = runtime([n, table, ids])
            else:
                (output,) = runtime([n, table, ids])
            self.assertEqual(output, reference)
            self.assertEqual(len(runtime.variants), 1)

    def test_drop_pinned_sources_before_synchronization(self, device):
        runtime = self.runtime()
        table = torch.randn(ROWS, WIDTH, device=device)
        runtime([16, table, self.indices(16)])
        inputs = [(n, self.indices(n)) for n in (16, 32, 48, 16, 24)]
        expected = [table[ids[2 : n + 2].to(device)] + 1 + 1 for n, ids in inputs]
        with mock.patch.object(
            direct_host,
            "_observe_direct",
            side_effect=AssertionError("Expected native hit"),
        ):
            outputs = [runtime([n, table, ids])[0] for n, ids in inputs]
        inputs.clear()
        gc.collect()
        pressure = [
            torch.empty(52, dtype=torch.int64, pin_memory=True) for _ in range(8)
        ]
        for value in pressure:
            value.fill_(0)
        torch.cuda.synchronize(device)
        for output, reference in zip(outputs, expected, strict=True):
            self.assertEqual(output, reference)

    def test_explicit_wait_before_rewriting_pinned_source(self, device):
        runtime = self.runtime()
        table = torch.randn(ROWS, WIDTH, device=device)
        n = 32
        ids = self.indices(n)
        runtime([n, table, ids])
        for _ in range(3):
            runtime.entry.wait_for_h2d()
            ids.copy_(torch.randint(0, ROWS, ids.shape))
            reference = table[ids[2 : n + 2].to(device)] + 1 + 1
            (output,) = runtime([n, table, ids])
            self.assertEqual(output, reference)
        self.assertEqual(len(runtime.variants), 1)

    def test_bound_pinned_sources_survive_host_cache_flush(self, device):
        runtime = self.runtime()
        table = torch.randn(ROWS, WIDTH, device=device)
        n = 16
        ids = self.indices(n)
        captured = StorageWeakRef(ids.untyped_storage())
        expected = table[ids[2 : n + 2].to(device)] + 1 + 1
        (output,) = runtime([n, table, ids])
        self.assertEqual(output, expected, atol=0, rtol=0)
        del ids
        torch.cuda.synchronize(device)
        gc.collect()
        self.assertFalse(captured.expired())
        torch._C._host_emptyCache()

        ids = self.indices(n)
        served = StorageWeakRef(ids.untyped_storage())
        expected = table[ids[2 : n + 2].to(device)] + 1 + 1
        with mock.patch.object(
            direct_host,
            "_observe_direct",
            side_effect=AssertionError("Expected native hit"),
        ):
            (output,) = runtime([n, table, ids])
        self.assertEqual(output, expected, atol=0, rtol=0)
        del ids
        torch.cuda.synchronize(device)
        gc.collect()
        self.assertTrue(captured.expired())
        self.assertFalse(served.expired())
        torch._C._host_emptyCache()
        ids = self.indices(n)
        rebound = StorageWeakRef(ids.untyped_storage())
        expected = table[ids[2 : n + 2].to(device)] + 1 + 1
        with mock.patch.object(
            direct_host,
            "_observe_direct",
            side_effect=AssertionError("Expected native hit"),
        ):
            (output,) = runtime([n, table, ids])
        self.assertEqual(output, expected, atol=0, rtol=0)
        self.assertEqual(len(runtime.variants), 1)
        del ids
        torch.cuda.synchronize(device)
        gc.collect()
        self.assertTrue(served.expired())
        self.assertFalse(rebound.expired())
        runtime.close()
        gc.collect()
        self.assertTrue(rebound.expired())

    def test_standalone_closed_scalar_and_unused_cpu_formal(self, device):
        table = torch.randn(ROWS, WIDTH, device=device)
        unused = torch.empty(7, pin_memory=True)
        first = self.indices(16)[2:18]
        runtime = HostTraceReplay(unused_pinned_formal, (7, unused, table, first))
        self.addCleanup(runtime.close)
        for n in (16, 32, 16):
            ids = self.indices(n)[2 : n + 2]
            output = runtime(7, unused, table, ids)
            runtime.wait_for_h2d()
            self.assertEqual(output, table[ids.to(device)])
        self.assertEqual(runtime.misses, 0)


instantiate_device_type_tests(TestMixedPinnedH2D, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
