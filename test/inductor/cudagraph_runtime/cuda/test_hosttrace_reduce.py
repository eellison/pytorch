# Owner(s): ["module: inductor"]
"""Reductions through the traced TensorIterator sibling, lowered into the shared native replay."""

import gc

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


def _x(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device="cuda").to(dtype)


class TestHostTraceReduce(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from torch._inductor.runtime._cudagraph import direct_hosttrace

        self.module = direct_hosttrace

    def _replay(self, fn, args):
        gc.collect()
        replay = self.module.HostTraceReplay(fn, args)
        self.addCleanup(replay.close)
        return replay

    def _check(self, replay, args, fn):
        got = replay(*args)
        want = fn(*args)
        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got, want, atol=0, rtol=0)

    def test_sum_serves_new_shapes_bitwise(self):
        fn = lambda t: torch.sum(t, -1)  # noqa: E731
        replay = self._replay(fn, (_x(64, 4096),))
        self.assertEqual(len(replay.lowered.calls), 1)
        served = 0
        for M, N in (
            (64, 4096),
            (32, 4096),
            (128, 4096),
            (48, 3000),
            (64, 2048),
            (1024, 512),
            (16, 4096),
        ):
            before = replay.misses
            self._check(replay, (_x(M, N),), fn)
            served += replay.misses == before
        self.assertGreaterEqual(served, 5)

    def test_block_is_an_expression(self):
        fn = lambda t: torch.sum(t, -1)  # noqa: E731
        replay = self._replay(fn, (_x(64, 4096),))
        call = replay.lowered.calls[0]
        # the reduction picks its block from the shape: a block binding, not a constant
        self.assertTrue(call.block is not None or call.module.block is not None)

    def test_split_path_updates_the_semaphore_memset(self):
        fn = lambda t: torch.sum(t, -1)  # noqa: E731
        replay = self._replay(fn, (_x(8, 262144),))
        self.assertEqual(len(replay.lowered.memsets), 1)
        self.assertEqual(len(replay.lowered.calls), 1)
        served = 0
        for M, N in ((8, 262144), (8, 200000), (5, 262144), (12, 262144), (16, 262144)):
            before = replay.misses
            self._check(replay, (_x(M, N),), fn)
            served += replay.misses == before
        self.assertGreaterEqual(served, 3)
        # two replays in a row: the memset node re-zeroes the semaphores each launch
        x = _x(8, 200000)
        a = replay(x).clone()
        b = replay(x).clone()
        want = fn(x)
        self.assertEqual(a, want, atol=0, rtol=0)
        self.assertEqual(b, want, atol=0, rtol=0)

    def test_mean_and_amax(self):
        for fn in (
            lambda t: torch.mean(t, -1),
            lambda t: torch.amax(t, -1),
            lambda t: torch.sum(t, 0),
        ):
            replay = self._replay(fn, (_x(64, 4096),))
            for M, N in ((48, 3000), (64, 2048), (16, 4096)):
                self._check(replay, (_x(M, N),), fn)

    def test_dtype_and_dims_changes_miss(self):
        fn = lambda t: torch.sum(t, -1)  # noqa: E731
        replay = self._replay(fn, (_x(64, 4096),))
        self._check(replay, (_x(64, 4096, dtype=torch.float16),), fn)
        self._check(replay, (_x(4, 64, 4096),), fn)
        self.assertEqual(replay.misses, 2)

    def test_agrees_with_eager_at_other_shapes(self):
        from torch.testing._internal.host_trace_oracle import Oracle

        fn = lambda t: torch.sum(t, -1)  # noqa: E731
        oracle = Oracle(fn, (_x(64, 4096),))
        self.addCleanup(oracle.close)
        self.assertIsNone(oracle.refused)
        for M, N in ((32, 4096), (48, 3000), (16, 4096)):
            oracle.check((_x(M, N),))


if __name__ == "__main__":
    run_tests()
