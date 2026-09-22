# Owner(s): ["module: inductor"]
import unittest

import torch
from torch._inductor.runtime._cudagraph.direct_hosttrace import HostTraceReplay
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_TRITON


if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit(
        do_not_specialize=["n"],
        do_not_specialize_on_alignment=["x_ptr", "y_ptr"],
    )
    def generic(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(x_ptr + index, index < n, other=0)
        tl.store(y_ptr + index, value * 2 + 1, index < n)

    @triton.jit(do_not_specialize_on_alignment=["x_ptr", "y_ptr", "n"])
    def unaligned(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(x_ptr + index, index < n, other=0)
        tl.store(y_ptr + index, value * 2 + 1, index < n)

    @triton.jit(do_not_specialize_on_alignment=["x_ptr", "y_ptr", "n"])
    def typed(x_ptr, y_ptr, n: tl.int64, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(x_ptr + index, index < n, other=0)
        tl.store(y_ptr + index, value * 2 + 1, index < n)


@unittest.skipUnless(HAS_TRITON, "requires Triton")
class TestHostTraceTritonSpecialization(TestCase):
    @parametrize("kind", ("generic", "unaligned", "typed"))
    def test_declared_specialization_replays(self, device, kind):
        kernel = {"generic": generic, "unaligned": unaligned, "typed": typed}[kind]

        def host(x):
            y = torch.empty_like(x)
            n = x.numel()
            kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](x, y, n, BLOCK=128)
            return y

        replay = HostTraceReplay(host, (torch.arange(16, device=device).float(),))
        self.addCleanup(replay.close)
        held = []
        counts = (1, 1, 1, 1, 1) if kind == "generic" else (1, 1, 2, 2, 2)
        for rows, offset, count in zip((16, 17, 1, 16, 17), (1, 4, 1, 4, 1), counts):
            x = torch.arange(rows + offset, device=device).float()[offset:]
            actual = replay(x)
            expected = x * 2 + 1
            self.assertEqual(actual, expected, atol=0, rtol=0)
            self.assertEqual(len(replay.variants), count)
            self.assertEqual(replay.declines, [])
            self.assertEqual(replay.ordinary, 0)
            held.append((actual, expected))
        replay.close()
        for actual, expected in held:
            self.assertEqual(actual, expected, atol=0, rtol=0)


instantiate_device_type_tests(
    TestHostTraceTritonSpecialization, globals(), only_for="cuda"
)


if __name__ == "__main__":
    run_tests()
