# Owner(s): ["module: inductor"]

import os
import sys
import unittest

import torch
from torch._inductor.runtime._cudagraph.host_trace import HostTraceReplay
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from host_trace_h2d_probe import probe  # noqa: E402


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.hip is None
    and _get_torch_cuda_version() >= (12, 8),
    "requires NVIDIA CUDA >= 12.8",
)
class TestHostTraceSequence(TestCase):
    def test_reduction_memsets_between_launches(self, device):
        def fn(x):
            return torch.sum(torch.sin(x), -1), torch.sum(x, -1)

        replay = HostTraceReplay(fn)
        self.addCleanup(replay.close)
        retained = []
        for shape in ((8, 262144), (12, 262144), (8, 200000)):
            for repeat in range(2):
                x = torch.randn(shape, device=device)
                expected = fn(x)
                before = len(replay.variants)
                box = [x]
                actual = replay(box)
                self.assertEqual(box, [])
                self.assertEqual(actual, expected, atol=0, rtol=0)
                if repeat:
                    self.assertEqual(len(replay.variants), before)
                retained.append((actual, expected))
        tape = replay.variants[0].program.tape
        self.assertEqual(len(tape.memsets), 2)
        self.assertLess(tape.launches[0]["seq"], tape.memsets[0]["seq"])
        self.assertLess(tape.memsets[0]["seq"], tape.launches[1]["seq"])
        replay.close()
        for actual, expected in retained:
            self.assertEqual(actual, expected, atol=0, rtol=0)

    def test_h2d_between_launches(self, device):
        def fn(table, ids):
            return probe().gather(torch.sin(table), ids)

        replay = HostTraceReplay(fn)
        self.addCleanup(replay.close)
        retained = []
        for count in (16, 32, 24):
            table = torch.randn((512, 64), device=device)
            ids = torch.randint(0, 512, (count,)).pin_memory()
            expected = table.sin()[ids.to(device)]
            box = [table, ids]
            actual = replay(box)
            self.assertEqual(box, [])
            self.assertEqual(actual, (expected,), atol=0, rtol=0)
            variants = len(replay.variants)
            self.assertEqual(replay([table, ids]), actual, atol=0, rtol=0)
            self.assertEqual(len(replay.variants), variants)
            retained.append((actual, expected))
        tape = replay.variants[0].program.tape
        self.assertEqual(len(tape.memcpys), 1)
        self.assertLess(tape.launches[0]["seq"], tape.memcpys[0]["seq"])
        self.assertLess(tape.memcpys[0]["seq"], tape.launches[1]["seq"])
        replay.close()
        for actual, expected in retained:
            self.assertEqual(actual, (expected,), atol=0, rtol=0)


instantiate_device_type_tests(TestHostTraceSequence, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
