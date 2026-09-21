# Owner(s): ["module: inductor"]
"""index_select, embedding and cat (host-tracing commit 8) lowered into the shared native replay."""

import gc
import os
import sys

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from host_trace_h2d_probe import probe  # noqa: E402


def _ids(n, hi=1000):
    return torch.randint(0, hi, (n,), device="cuda")


class TestHostTraceEmbedCat(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from torch._inductor.runtime._cudagraph import direct_hosttrace

        self.module = direct_hosttrace
        torch.manual_seed(0)

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
                self.assertEqual(got.shape, want.shape)
                self.assertEqual(got.stride(), want.stride())
                self.assertEqual(got, want, atol=0, rtol=0)
            out.append(hit)
        return out

    def test_index_select_small_route(self):
        w = torch.randn(1000, 64, device="cuda", dtype=torch.bfloat16)
        fn = lambda w, i: torch.index_select(w, 0, i)  # noqa: E731
        replay = self._replay(fn, (w, _ids(8)))
        self.assertEqual(len(replay.lowered.calls), 1)
        served = self._served(
            replay,
            fn,
            [
                (w, _ids(8)),
                (w, _ids(3)),
                (w, _ids(16)),
                (w, _ids(1)),
                (w, _ids(5)),
                (w, _ids(32)),
            ],
        )
        # 1 id folds a dim (another instantiation), > 16 ids is the gather route: named misses
        self.assertEqual(served, [True, True, True, False, True, False])
        w2 = torch.randn(500, 64, device="cuda", dtype=torch.bfloat16)
        self.assertEqual(self._served(replay, fn, [(w2, _ids(8, 500))]), [True])

    def test_index_select_gather_route(self):
        w = torch.randn(4096, 128, device="cuda", dtype=torch.bfloat16)
        fn = lambda w, i: torch.index_select(w, 0, i)  # noqa: E731
        replay = self._replay(fn, (w, _ids(64, 4096)))
        self.assertEqual(
            self._served(
                replay,
                fn,
                [(w, _ids(200, 4096)), (w, _ids(17, 4096)), (w, _ids(1000, 4096))],
            ),
            [True] * 3,
        )

    def test_embedding_1d_2d_and_from_pinned_ids(self):
        table = torch.randn(1000, 128, device="cuda", dtype=torch.bfloat16)
        fn = lambda t, i: F.embedding(i, t)  # noqa: E731
        replay = self._replay(fn, (table, _ids(4)))
        self.assertEqual(
            self._served(
                replay, fn, [(table, _ids(8)), (table, _ids(16)), (table, _ids(2))]
            ),
            [True] * 3,
        )
        replay2 = self._replay(
            fn, (table, torch.randint(0, 1000, (2, 4), device="cuda"))
        )
        self.assertEqual(
            self._served(
                replay2,
                fn,
                [
                    (table, torch.randint(0, 1000, s, device="cuda"))
                    for s in ((4, 4), (1, 8), (3, 5))
                ],
            ),
            [True] * 3,
        )

        def pinned_embed(pinned, table):
            dev = torch.empty(pinned.shape[0], dtype=torch.int64, device="cuda")
            probe().copy_into(dev, pinned)
            return F.embedding(dev, table)

        pin = lambda n: torch.randint(0, 1000, (n,), dtype=torch.int64).pin_memory()  # noqa: E731
        replay3 = self._replay(pinned_embed, (pin(4), table))
        self.assertEqual(
            self._served(
                replay3,
                pinned_embed,
                [(pin(4), table), (pin(8), table), (pin(3), table)],
            ),
            [True] * 3,
        )

    def test_cat_batched_path(self):
        fn = lambda x, y: torch.cat([x, y], -1)  # noqa: E731

        def pair(m, a, b):
            return (
                torch.randn(m, a, device="cuda", dtype=torch.bfloat16),
                torch.randn(m, b, device="cuda", dtype=torch.bfloat16),
            )

        replay = self._replay(fn, pair(8, 32, 32))
        served = self._served(
            replay,
            fn,
            [
                pair(8, 32, 32),
                pair(16, 32, 32),
                pair(4, 64, 64),
                pair(8, 48, 16),
                pair(3, 8, 8),
                pair(3, 6, 6),
            ],
        )
        # 12-byte slices are not 16-byte multiples: another kernel, a named miss
        self.assertEqual(served, [True, True, True, True, True, False])
        many = lambda *xs: torch.cat(xs, 0)  # noqa: E731

        def xs(n, m):
            return tuple(
                torch.randn(2, m, device="cuda", dtype=torch.bfloat16) for _ in range(n)
            )

        # up to 128 inputs is one batched launch
        replay2 = self._replay(many, xs(100, 32))
        self.assertEqual(len(replay2.lowered.calls), 1)
        self.assertEqual(
            self._served(replay2, many, [xs(100, 32), xs(100, 64)]), [True, True]
        )
        # above 128 the host splits into two launches; the second launch's grid is an
        # n-ary max over the remaining inputs whose lowered chain shares its prefix
        # with the first launch's (the interned DAG; preparation used to hang here)
        replay3 = self._replay(many, xs(130, 32))
        self.assertEqual(len(replay3.lowered.calls), 2)
        self.assertEqual(
            self._served(replay3, many, [xs(130, 32), xs(130, 64), xs(130, 6)]),
            [True, True, False],
        )

    def test_rotate_half(self):
        neg_one = torch.tensor(-1.0, device="cuda", dtype=torch.bfloat16)

        def rotate_half(x, neg_one):
            h = x.shape[-1] // 2
            return torch.cat([x[..., h:] * neg_one, x[..., :h]], -1)

        def make(b, dh):
            return (
                torch.randn(b, 8, 1, dh, device="cuda", dtype=torch.bfloat16),
                neg_one,
            )

        replay = self._replay(rotate_half, make(4, 64))
        self.assertEqual(
            self._served(replay, rotate_half, [make(4, 64), make(8, 64)]), [True, True]
        )


if __name__ == "__main__":
    run_tests()
