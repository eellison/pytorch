# Owner(s): ["module: inductor"]
"""Softmax, cast copies and unary ops (host-tracing commit 7) lowered into the shared native replay."""

import gc

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


def _rand(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device="cuda", dtype=dtype)


class TestHostTraceOps7(TestCase):
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

    def _served(self, replay, fn, cases):
        out = []
        for args in cases:
            want = fn(*args)
            before = replay.misses
            got = replay(*args)
            hit = replay.misses == before
            if hit:
                self.assertEqual(got.dtype, want.dtype)
                self.assertEqual(got, want, atol=0, rtol=0)
            out.append(hit)
        return out

    def test_softmax_persistent_path_serves_its_bucket(self):
        fn = lambda t: torch.softmax(t, -1)  # noqa: E731
        replay = self._replay(fn, (_rand(64, 400),))
        self.assertEqual(len(replay.lowered.calls), 1)
        served = self._served(
            replay,
            fn,
            [
                (_rand(64, 300),),
                (_rand(8, 512),),
                (_rand(3, 257),),
                (_rand(64, 256),),
                (_rand(64, 600),),
                (_rand(64, 4096),),
            ],
        )
        # inside the log2 bucket served; another bucket or the block path a named miss
        self.assertEqual(served, [True, True, True, False, False, False])

    def test_softmax_block_path_register_and_smem_kernels(self):
        fn = lambda t: torch.softmax(t, -1)  # noqa: E731
        replay = self._replay(fn, (_rand(8, 8192, dtype=torch.float32),))
        served = self._served(
            replay,
            fn,
            [
                (_rand(3, 8192, dtype=torch.float32),),
                (_rand(8, 8000, dtype=torch.float32),),
                (_rand(8, 6000, dtype=torch.float32),),
                (_rand(8, 12000, dtype=torch.float32),),
                (_rand(8, 1024, dtype=torch.float32),),
            ],
        )
        self.assertEqual(served, [True, True, False, False, False])
        smem = self._replay(fn, (_rand(8, 12000, dtype=torch.float32),))
        self.assertEqual(
            self._served(smem, fn, [(_rand(5, 11000, dtype=torch.float32),)]), [True]
        )

    def test_log_softmax_3d(self):
        fn = lambda t: torch.log_softmax(t, -1)  # noqa: E731
        replay = self._replay(fn, (_rand(2, 8, 1000, dtype=torch.float16),))
        self.assertEqual(
            self._served(replay, fn, [(_rand(3, 5, 900, dtype=torch.float16),)]), [True]
        )

    def test_cast_copies(self):
        fn = lambda t: t.to(torch.float32)  # noqa: E731
        replay = self._replay(fn, (_rand(64, 4096),))
        self.assertEqual(len(replay.lowered.calls), 1)
        served = self._served(
            replay,
            fn,
            [
                (_rand(48, 3000),),
                (_rand(5, 7),),
                (_rand(64, 4096).t().contiguous().t(),),
            ],
        )
        self.assertEqual(served, [True, True, False])
        # a source dtype change misses (the argument contract), never a wrong cast
        before = replay.misses
        replay(_rand(64, 4096, dtype=torch.float16))
        self.assertEqual(replay.misses, before + 1)
        dyn = self._replay(lambda t: t.to(torch.int32), (_rand(64, 4096),))
        self.assertEqual(
            self._served(
                dyn, lambda t: t.to(torch.int32), [(_rand(48, 3000),), (_rand(5, 7),)]
            ),
            [True, True],
        )

    def test_unary_ops_and_the_rotary_composition(self):
        for fn in (torch.sin, torch.cos, torch.exp, torch.neg):
            replay = self._replay(fn, (_rand(64, 4096),))
            self.assertEqual(
                self._served(
                    replay,
                    fn,
                    [(_rand(48, 3000),), (_rand(5, 7),), (_rand(1024, 1024),)],
                ),
                [True] * 3,
            )
        neg_one = torch.tensor(-1.0, device="cuda", dtype=torch.bfloat16)

        def rotary(x, cos, sin, neg_one):
            h = x.shape[-1] // 2
            return x * cos + torch.cat([x[..., h:] * neg_one, x[..., :h]], -1) * sin

        def make(b, dh):
            ang = torch.arange(dh // 2, device="cuda", dtype=torch.float32) * 0.1
            return (
                _rand(b, 8, 1, dh),
                torch.cat([ang.cos(), ang.cos()]).to(torch.bfloat16),
                torch.cat([ang.sin(), ang.sin()]).to(torch.bfloat16),
                neg_one,
            )

        replay = self._replay(rotary, make(4, 64))
        self.assertEqual(
            self._served(replay, rotary, [make(4, 64), make(8, 64), make(2, 64)]),
            [True] * 3,
        )


if __name__ == "__main__":
    run_tests()
