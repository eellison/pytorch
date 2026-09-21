# Owner(s): ["module: inductor"]
"""The flash attention forward traced by torch.cuda._host_trace, lowered into the shared native replay."""

import gc
import unittest

import torch

from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_utils import run_tests, TestCase


flash_forward = torch.ops.aten._flash_attention_forward.default


def flash(q, k, v, causal, scale):
    return flash_forward(q, k, v, None, None, 0, 0, 0.0, causal, False, scale=scale)


def num_splits_heuristic(batch_nheads_mblocks, num_SMs, num_n_blocks, max_splits):
    import numpy as np

    f = np.float32
    if f(batch_nheads_mblocks) >= f(0.8) * f(num_SMs):
        return 1
    max_splits = min(max_splits, num_SMs, num_n_blocks)

    def ceildiv(a, b):
        return (a + b - 1) // b

    def eligible(s):
        return s == 1 or ceildiv(num_n_blocks, s) != ceildiv(num_n_blocks, s - 1)

    eff, best = [], f(0)
    for s in range(1, max_splits + 1):
        if not eligible(s):
            eff.append(f(0))
            continue
        n_waves = f(batch_nheads_mblocks * s) / f(num_SMs)
        e = n_waves / np.ceil(n_waves)
        best = max(best, e)
        eff.append(e)
    for s in range(1, max_splits + 1):
        if eligible(s) and float(eff[s - 1]) >= 0.85 * float(best):
            return s
    return 1


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
@unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "flash attention not supported")
class TestHostTraceFlash(TestCase):
    H = 32
    D = 128

    def setUp(self):
        super().setUp()
        from torch._inductor.runtime._cudagraph import direct_hosttrace

        self.module = direct_hosttrace

    def _qkv(self, B, Sq, Sk, H=None, Hk=None, dtype=torch.bfloat16, offset=0):
        H = H or self.H
        Hk = Hk or H

        def make(S, heads):
            flat = torch.randn(
                B * S * heads * self.D + offset, device="cuda", dtype=dtype
            )
            return flat[offset:].view(B, S, heads, self.D)

        return make(Sq, H), make(Sk, Hk), make(Sk, Hk)

    def _replay(self, *shape, causal=False, scale=0.125, **kw):
        gc.collect()
        args = (*self._qkv(*shape, **kw), causal, scale)
        replay = self.module.HostTraceReplay(flash, args)
        self.addCleanup(replay.close)
        return replay

    def _check(self, replay, args):
        got = replay(*args)
        ref = flash(*args)
        self.assertEqual(got[0], ref[0], atol=0, rtol=0)
        self.assertEqual(got[1], ref[1], atol=0, rtol=0)

    def test_lowering_shape(self):
        replay = self._replay(4, 512, 512)
        lowered = replay.lowered
        self.assertEqual(len(lowered.calls), 1)
        self.assertEqual(lowered.registration[0], ())
        call = lowered.calls[0]
        self.assertEqual(len(call.module.parameter_layout), 1)  # one struct by value
        kinds = {field.kind for field in call.fields}
        self.assertTrue({"pointer", "i32", "i64"} <= kinds)

    def test_serves_six_shapes_bitwise(self):
        replay = self._replay(4, 512, 512)
        for B, Sq, Sk in [
            (4, 512, 512),
            (1, 512, 512),
            (8, 512, 512),
            (2, 1024, 256),
            (4, 384, 1024),
            (3, 512, 64),
        ]:
            self._check(replay, (*self._qkv(B, Sq, Sk), False, 0.125))
        self.assertEqual(replay.misses, 0)
        self.assertEqual(replay.served, 6)

    def test_misses_build_reusable_variants(self):
        replay = self._replay(4, 512, 512)
        # seqlen_q 500 flips the even-MN branch: a guard
        self._check(replay, (*self._qkv(4, 500, 512), False, 0.125))
        # causal is a constant of the contract: a separate family on its miss
        self._check(replay, (*self._qkv(4, 512, 512), True, 0.125))
        # float16 after a bfloat16 trace: the dtype fact
        self._check(
            replay, (*self._qkv(4, 512, 512, dtype=torch.float16), False, 0.125)
        )
        self.assertEqual(replay.misses, 3)
        self.assertEqual(replay.served, 3)
        self.assertEqual(replay.ordinary, 0)
        self.assertEqual(len(replay.variants), 4)
        # storage offsets that keep flash's 16-byte alignment are served: the
        # addresses are expressions of the offsets
        for offset in (8, 64):
            self._check(replay, (*self._qkv(2, 512, 512, offset=offset), False, 0.125))
        self.assertEqual(replay.served, 5)

    def test_misses_build_their_own_variants(self):
        replay = self._replay(4, 512, 512)
        # seqlen_q 500 flips the even-MN branch: a guard, so a second variant
        self._check(replay, (*self._qkv(4, 500, 512), False, 0.125))
        # causal is a constant of the contract: its own family of variants
        self._check(replay, (*self._qkv(4, 512, 512), True, 0.125))
        self.assertEqual((replay.misses, replay.ordinary), (2, 0))
        self.assertEqual(len(replay.variants), 3)
        # float16 after a bfloat16 trace: the dtype fact misses and the fp16 variant
        # is built beside the others. Its debug-mask output, which no launch writes,
        # is a runtime buffer of that variant (an fp16 one; the bf16 variants keep
        # theirs): no family-wide shape or dtype of a "fresh" output any more
        fp16 = (*self._qkv(4, 512, 512, dtype=torch.float16), False, 0.125)
        self._check(replay, fp16)
        self._check(replay, fp16)
        self.assertEqual((replay.misses, replay.ordinary), (3, 0))
        self.assertEqual(len(replay.variants), 4)
        self.assertEqual(replay.declines, [])
        # the philox seed and offset (no dropout) and the debug mask: nothing writes them
        self.assertEqual(replay.variants[3].lowered.unwritten_outputs, (2, 3, 4))
        self.assertEqual(replay(*fp16)[4].dtype, torch.float16)
        # the three classes built at their misses hit from now on
        self._check(replay, (*self._qkv(4, 500, 512), False, 0.125))
        self._check(replay, (*self._qkv(4, 512, 512), True, 0.125))
        # storage offsets that keep flash's 16-byte alignment are served by the first
        # variant: the addresses are expressions of the offsets
        for offset in (8, 64):
            self._check(replay, (*self._qkv(2, 512, 512, offset=offset), False, 0.125))
        self.assertEqual((replay.misses, replay.served), (3, 9))

    def test_scale_from_the_head_dim(self):
        # scale=None: the host derives 1/sqrt(d) as a SymFloat; a float field in the plan
        replay = self._replay(4, 512, 512, scale=None)
        call = replay.lowered.calls[0]
        ops = {
            field.source.expression.op
            for field in call.fields
            if field.kind != "pointer"
        }
        self.assertIn("ftobits32", ops)
        for B, Sq, Sk in [(2, 512, 512), (1, 1024, 1024)]:
            self._check(replay, (*self._qkv(B, Sq, Sk), False, None))
        self.assertEqual(replay.misses, 0)

    def _check_split_variants(self, replay, batches, Sk):
        num_n_blocks = (Sk + 127) // 128
        sms = 2 * torch.cuda.get_device_properties(0).multi_processor_count
        seen = {replay.tape.opaque[0]["expected"]}
        representatives = {}
        served = missed = 0
        for B in batches:
            expect = num_splits_heuristic(B * 8, sms, num_n_blocks, 128)
            args = (*self._qkv(B, 1, Sk, H=8, Hk=8), False, 0.125)
            before = replay.misses
            variants = len(replay.variants)
            self._check(replay, args)
            new_variant = expect not in seen
            self.assertEqual(replay.misses, before + new_variant)
            self.assertEqual(len(replay.variants), variants + new_variant)
            if new_variant:
                self.assertEqual(replay.variants[-1].tape.opaque[0]["expected"], expect)
                seen.add(expect)
                missed += 1
            else:
                served += 1
            representatives.setdefault(expect, B)
        before = replay.misses
        variants = len(replay.variants)
        for B in reversed(tuple(representatives.values())):
            self._check(replay, (*self._qkv(B, 1, Sk, H=8, Hk=8), False, 0.125))
            self.assertEqual(replay.misses, before)
            self.assertEqual(len(replay.variants), variants)
        self.assertEqual(replay.ordinary, 0)
        return served, missed

    def test_split_kv_and_the_heuristic(self):
        Sk = 8192
        replay = self._replay(1, 1, Sk, H=8, Hk=8)
        self.assertEqual(
            [o["fn"] for o in replay.tape.opaque], ["num_splits_heuristic"]
        )
        self.assertGreater(replay.tape.opaque[0]["expected"], 1)
        self.assertEqual(len(replay.lowered.calls), 2)  # split kernel and combine
        served, missed = self._check_split_variants(replay, range(1, 33), Sk)
        self.assertGreater(served, 0)
        self.assertGreater(missed, 0)

    def test_split_count_ties(self):
        Sk = 4096
        num_n_blocks = (Sk + 127) // 128
        sms = (
            2 * torch.cuda.get_device_properties(0).multi_processor_count
        )  # flash's host doubles the SM count
        ties = [
            B
            for B in range(1, 65)
            if any((B * 8 * s) % sms == 0 for s in range(1, min(128, num_n_blocks) + 1))
        ]
        self.assertTrue(ties)
        replay = self._replay(ties[0], 1, Sk, H=8, Hk=8)
        self._check_split_variants(replay, ties, Sk)

    def test_agrees_with_the_interim_replay(self):
        from torch.cuda import _host_trace

        replay = self._replay(4, 512, 512)
        args = (*self._qkv(4, 512, 512), False, 0.125)
        interim = _host_trace.build(replay.tape, flash, args)
        for B, Sq, Sk in [(1, 512, 512), (8, 512, 512), (2, 1024, 256)]:
            args = (*self._qkv(B, Sq, Sk), False, 0.125)
            native = replay(*args)
            ours = interim.replay(args)
            self.assertEqual(native[0], ours[0], atol=0, rtol=0)
            self.assertEqual(native[1], ours[1], atol=0, rtol=0)


if __name__ == "__main__":
    run_tests()
