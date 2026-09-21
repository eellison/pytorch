# Owner(s): ["module: inductor"]
"""Flash attention with dropout lowered into the shared native replay: the generator increment."""

import gc

import torch
import torch.nn.functional as F
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_utils import run_tests, TestCase


def sdpa_dropout(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)


def _qkv(B, S, H=8, D=64):
    return tuple(
        torch.randn(B, H, S, D, device="cuda", dtype=torch.float16) for _ in range(3)
    )


class TestHostTraceDropout(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available() or not PLATFORM_SUPPORTS_FLASH_ATTENTION:
            self.skipTest("CUDA with flash attention required")
        from torch._inductor.runtime._cudagraph import direct_hosttrace

        self.module = direct_hosttrace
        self._flash = torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        )
        self._flash.__enter__()
        self.addCleanup(self._flash.__exit__, None, None, None)

    def _replay(self, args):
        gc.collect()
        replay = self.module.HostTraceReplay(sdpa_dropout, args)
        self.addCleanup(replay.close)
        return replay

    def test_replays_draw_the_same_random_stream_as_eager(self):
        torch.manual_seed(0)
        replay = self._replay(_qkv(4, 128))
        self.assertIsNotNone(replay.tape.rng_increment)
        self.assertEqual(len(replay.lowered.rng_fields), 1)
        inputs = [_qkv(4, 128), _qkv(2, 128), _qkv(8, 128), _qkv(1, 128), _qkv(4, 128)]
        torch.manual_seed(1234)
        want = [sdpa_dropout(*x) for x in inputs]
        eager_state = torch.cuda.get_rng_state()
        torch.manual_seed(1234)
        got = [replay(*x) for x in inputs]
        self.assertEqual(replay.misses, 0)
        for g, w in zip(got, want):
            self.assertEqual(g, w, atol=0, rtol=0)
        # the generator advanced by the same increments (b * h * 32 per call)
        self.assertEqual(torch.cuda.get_rng_state(), eager_state)

    def test_seeds(self):
        replay = self._replay(_qkv(4, 64))
        args = _qkv(4, 64)
        torch.manual_seed(7)
        a = replay(*args)
        torch.manual_seed(7)
        b = replay(*args)
        torch.manual_seed(8)
        c = replay(*args)
        self.assertEqual(a, b, atol=0, rtol=0)
        self.assertFalse(torch.equal(a, c))


if __name__ == "__main__":
    run_tests()
