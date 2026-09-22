# Owner(s): ["module: inductor"]
import unittest

import torch
from torch.cuda import _host_trace as ht
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_TRITON


if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def add_scalar(x, value: tl.float32, BLOCK: tl.constexpr):
        offset = tl.arange(0, BLOCK)
        tl.store(x + offset, tl.load(x + offset) + value)


@unittest.skipUnless(HAS_TRITON, "requires Triton")
class TestHostTraceTritonExecution(TestCase):
    @parametrize("warm_up", (False, True))
    def test_caught_unsupported_abi_cannot_publish_incomplete_tape(
        self, device, warm_up
    ):
        caught = []

        def host(x):
            try:
                add_scalar[(1,)](x, 2.0, BLOCK=32)
            except RuntimeError as error:
                caught.append(error)
            return x

        x = torch.arange(32, device=device, dtype=torch.float32)
        expected = x + (2 if warm_up else 0)
        with self.assertRaisesRegex(ht.Declined, "supported compiler ABI") as declined:
            ht.trace(host, (x,), warm_up=warm_up)
        self.assertEqual(len(caught), 1)
        self.assertIs(declined.exception.__cause__, caught[0])
        self.assertEqual(x, expected, atol=0, rtol=0)


instantiate_device_type_tests(TestHostTraceTritonExecution, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
