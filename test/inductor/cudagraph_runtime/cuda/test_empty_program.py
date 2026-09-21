# Owner(s): ["module: inductor"]

import unittest
from unittest import mock

import torch
from torch._inductor.runtime._cudagraph import direct_host, replay
from torch._inductor.runtime._cudagraph.api import (
    DirectHost,
    InputContract,
    IntegerRange,
    IntExpr,
    TensorInput,
)
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


def view_outputs(box):
    n, source = box
    box.clear()
    start = 1 if n < 8 else 0
    view = source[start : start + n - 1]
    return n, None, source, view, view, source[start : start + n - 1], n


def allocation_outputs(box):
    n, source = box
    box.clear()
    rows = n if n < 8 else n + 2
    owned = torch.empty_strided(
        (rows, 2), (3, 1), dtype=source.dtype, device=source.device
    )
    return n, None, source, owned, owned, owned[:, 1], n


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.hip is None
    and _get_torch_cuda_version() >= (12, 8),
    "requires NVIDIA CUDA >= 12.8",
)
class TestEmptyProgram(TestCase):
    @parametrize("allocate", (False, True))
    def test_outputs_without_kernel_nodes(self, device, allocate):
        self.enterContext(torch.cuda.device(device))
        n = IntExpr("boxed", 0)
        contract = InputContract(
            ("integer", "tensor"),
            (TensorInput(1, torch.float32, (n,), (1,)),),
            (IntegerRange(0, 2, 32),),
            device_index=torch.cuda.current_device(),
        )
        runtime = DirectHost(allocation_outputs if allocate else view_outputs, contract)
        self.addCleanup(runtime.close)
        prepare = replay._make_replay

        def prepare_empty(graph, *args, **kwargs):
            self.assertEqual(graph._inspect_captured_kernel_nodes(())[2:], ((), ()))
            return prepare(graph, *args, **kwargs)

        self.enterContext(mock.patch.object(replay, "_make_replay", new=prepare_empty))
        samples = [
            torch.arange(n + offset, dtype=torch.float32, device=device)[offset:]
            for n, offset in ((5, 1), (6, 3), (10, 2), (12, 4), (4, 2))
        ]
        self.assertEqual(len({source.data_ptr() for source in samples}), len(samples))
        held = []
        for step, source in enumerate(samples):
            n = source.numel()
            box = [n, source]
            if step in (1, 3, 4):
                with mock.patch.object(
                    direct_host,
                    "_observe_direct",
                    side_effect=AssertionError("accepted hit reran the Python host"),
                ):
                    actual = runtime.entry(box)
            else:
                actual = runtime(box)
            self.assertEqual(box, [])
            self.assertIsNotNone(runtime.entry, "empty program was not prepared")
            self.assertEqual(len(runtime.variants), 1 if step < 2 else 2)
            self.assertEqual((actual[0], actual[1], actual[6]), (n, None, n))
            self.assertIs(actual[2], source)
            self.assertIs(actual[3], actual[4])
            self.assertIsNot(actual[3], actual[5])
            if allocate:
                rows = n if n < 8 else n + 2
                owned, view = actual[3], actual[5]
                self.assertEqual(owned.shape, (rows, 2))
                self.assertEqual(owned.stride(), (3, 1))
                self.assertEqual(owned.storage_offset(), 0)
                self.assertEqual(
                    (owned.dtype, owned.device), (source.dtype, source.device)
                )
                self.assertEqual(view.shape, (rows,))
                self.assertEqual(view.stride(), (3,))
                self.assertEqual(
                    view.data_ptr(), owned.data_ptr() + owned.element_size()
                )
                self.assertNotIn(
                    owned.data_ptr(), [result[3].data_ptr() for result in held]
                )
                owned.fill_(step + 17)
            else:
                start = 1 if n < 8 else 0
                expected = source[start : start + n - 1]
                self.assertEqual(actual[3], expected)
                self.assertEqual(actual[5], expected)
                self.assertEqual(actual[3].data_ptr(), expected.data_ptr())
                self.assertEqual(actual[5].data_ptr(), expected.data_ptr())
            held.append(actual)

        runtime.close()
        for step, (source, actual) in enumerate(zip(samples, held, strict=True)):
            self.assertIs(actual[2], source)
            self.assertIs(actual[3], actual[4])
            if allocate:
                self.assertEqual(actual[3], torch.full_like(actual[3], step + 17))
                self.assertEqual(actual[5], torch.full_like(actual[5], step + 17))
            else:
                start = 1 if source.numel() < 8 else 0
                self.assertEqual(actual[3], source[start : start + source.numel() - 1])
                self.assertEqual(actual[5], actual[3])


instantiate_device_type_tests(TestEmptyProgram, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
