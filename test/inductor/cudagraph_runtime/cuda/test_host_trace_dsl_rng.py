# Owner(s): ["module: inductor"]
"""RNG launch indices survive Python DSL calls inserted into the recorder tape."""

import triton
import triton.language as tl

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@triton.jit
def add_one_kernel(x, y, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + offsets, offsets < n, other=0)
    tl.store(y + offsets, values + 1, offsets < n)


def add_one(x):
    y = torch.empty_like(x)
    add_one_kernel[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), BLOCK=256)
    return y


class TestDslRngComposition(TestCase):
    @parametrize("placement", ("before", "between", "after"))
    def test_rng_indices_and_replay(self, device, placement):
        from torch._inductor.runtime._cudagraph.direct_hosttrace import HostTraceReplay
        from torch.cuda import _host_trace

        def fn(x):
            if placement == "before":
                x = add_one(x)
            x = torch.native_dropout(x, 0.1, True)[0]
            if placement == "between":
                x = add_one(x)
            x = torch.native_dropout(x, 0.2, True)[0]
            if placement == "after":
                x = add_one(x)
            return x

        x = torch.randn(4096, device=device)
        tape = _host_trace.trace(fn, (x,))
        expected = [
            i
            for i, launch in enumerate(tape.launches)
            if any(p["kind"] == "rng" for p in launch["params"])
        ]
        actual = [slot["launch"] for slot in tape.rng_slots]
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 2)
        replay = HostTraceReplay(fn, (x,))
        self.addCleanup(replay.close)
        self.assertEqual(len(replay.variants), 1)
        generator = torch.cuda.default_generators[torch.cuda.current_device()]
        for size in (4096, 8192):
            inputs = torch.randn(size, device=device)
            torch.cuda.manual_seed(1234)
            expected = [fn(inputs) for _ in range(3)]
            expected_offset = generator.get_offset()
            torch.cuda.manual_seed(1234)
            actual = [replay(inputs) for _ in range(3)]
            self.assertEqual(generator.get_offset(), expected_offset)
            self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual((replay.misses, replay.declines), (0, []))


instantiate_device_type_tests(TestDslRngComposition, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
