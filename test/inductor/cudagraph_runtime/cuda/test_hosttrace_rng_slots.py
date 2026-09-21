# Owner(s): ["module: inductor"]
"""Per-launch rng slots (host-tracing commit 9) through the shared native replay: several random
launches in one tape draw eager's stream from the same seed."""

import gc
import unittest.mock

import torch
import torch.nn.functional as F
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_utils import run_tests, TestCase


SEED = 1234


def _gen():
    return torch.cuda.default_generators[torch.cuda.current_device()]


def _sequence(fn, args, n):
    torch.cuda.manual_seed(SEED)
    outs, offsets = [], []
    for _ in range(n):
        out = fn(*args)
        if isinstance(out, torch.Tensor):
            out = [out]
        outs.append([t.clone() for t in out])
        torch.cuda.synchronize()
        offsets.append(_gen().get_offset())
    return outs, offsets


def two_dropouts(x):
    d1 = torch.native_dropout(x, 0.1, True)[0]
    d2 = torch.native_dropout(d1[:, :1024].contiguous(), 0.2, True)[0]
    return d1, d2


def step2(x, w):
    h = F.layer_norm(x, (x.shape[-1],), w, None)
    d = torch.native_dropout(h, 0.1, True)[0]
    y = torch.native_dropout(F.silu(d + x), 0.2, True)[0]
    return (y * w).sum(-1)


def one_dropout(x):
    return torch.native_dropout(x, 0.2, True)[0] * 2


def dropout_then_flash(q, k, v):
    h = torch.native_dropout(q, 0.1, True)[0]
    return torch.ops.aten._flash_attention_forward.default(
        h, k, v, None, None, 0, 0, 0.1, False, False
    )[0]


class TestHostTraceRngSlots(TestCase):
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
        self.assertEqual(len(replay.tape.rng_slots), 2)
        return replay

    def _equal(self, replay, fn, args, n):
        ref, ref_offsets = _sequence(fn, args, n)
        got, offsets = _sequence(lambda *a: replay(*a), args, n)
        self.assertEqual(replay.misses, 0)
        self.assertEqual(offsets, ref_offsets)
        for g, r in zip(got, ref):
            for a, b in zip(g, r):
                self.assertEqual(a, b, atol=0, rtol=0)

    def test_two_dropouts_and_other_batches(self):
        x = torch.randn(4, 4096, device="cuda")
        replay = self._replay(two_dropouts, (x,))
        self._equal(replay, two_dropouts, (x,), 5)
        for B in (8, 2, 16):
            self._equal(replay, two_dropouts, (torch.randn(B, 4096, device="cuda"),), 3)

    def test_two_stage_composition(self):
        w = 1 + 0.1 * torch.randn(1024, device="cuda")
        x = torch.randn(4, 1024, device="cuda")
        replay = self._replay(step2, (x, w))
        self._equal(replay, step2, (x, w), 10)

    def test_prepared_capture_owns_the_philox_pointers(self):
        # The trace's capture and the preparation's capture each allocate their own
        # philox seed/offset tensors; the tape's pointer bytes are the trace's. Take
        # every free small block on the default stream between the two captures so the
        # preparation cannot land on the same addresses (a side-stream host sees this
        # through torch.cuda.graph's empty_cache releasing the trace's segment).
        from torch.cuda import _host_trace

        held, trace = [], _host_trace.trace

        def trace_then_fill(fn, args, **kwargs):
            tape = trace(fn, args, **kwargs)
            with torch.cuda.stream(torch.cuda.default_stream()):
                key = "segment.small_pool.current"
                segments = torch.cuda.memory_stats()[key]
                while torch.cuda.memory_stats()[key] == segments:
                    held.extend(
                        torch.empty(1, dtype=torch.int64, device="cuda")
                        for _ in range(64)
                    )
            return tape

        x = torch.randn(4, 1024, device="cuda")
        with unittest.mock.patch.object(_host_trace, "trace", trace_then_fill):
            replay = self.module.HostTraceReplay(one_dropout, (x,))
        self.addCleanup(replay.close)
        ((launch, _, _),) = replay.lowered.rng_fields
        params = replay.tape.launches[launch]["params"]
        (rng,) = [p for p in params if p["kind"] == "rng" and p["size"] == 16]
        image = bytes(replay.tape.launches[launch]["hint_image"])
        traced_seed = int.from_bytes(image[rng["offset"] : rng["offset"] + 8], "little")
        self.assertIn(traced_seed, {t.data_ptr() for t in held})
        self._equal(replay, one_dropout, (x,), 3)
        self._equal(replay, one_dropout, (torch.randn(8, 1024, device="cuda"),), 2)

    def test_a_tape_without_rng_slots_declines_by_name(self):
        # The recorder declares a philox launch's intragraph offset as a per-launch slot
        # param beside the 16-byte pointer field (cascade 11's record shape); a tape
        # whose slots were stripped leaves the traced offset in the image, and a replay
        # must not draw from it: the lowering declines by name instead of admitting
        # the traced bytes.
        from torch.cuda import _host_trace

        trace = _host_trace.trace

        def trace_without_slots(fn, args, **kwargs):
            tape = trace(fn, args, **kwargs)
            for launch in tape.launches:
                launch["params"] = [
                    p
                    for p in launch["params"]
                    if p["name"] != "philox_offset_intragraph"
                ]
            tape.rng_slots = []
            return tape

        for fn, x in (
            (one_dropout, torch.randn(4, 1024, device="cuda")),
            (two_dropouts, torch.randn(4, 4096, device="cuda")),
        ):
            # the decline comes after the constructor's warm-up ran, so the entry
            # is constructed in the declined state (E24: the warm-up was the call)
            reason = "without per-launch rng slots"
            with unittest.mock.patch.object(_host_trace, "trace", trace_without_slots):
                with self.assertWarnsRegex(RuntimeWarning, reason):
                    replay = self.module.HostTraceReplay(fn, (x,))
            self.assertEqual((len(replay.variants), len(replay.declines)), (0, 1))
            replay.close()
        # the recorded shape: the pointer field is exactly the two pointers, the slot
        # its own u64 param, the flag a constant byte, the tail the traced padding
        x = torch.randn(4, 1024, device="cuda")
        replay = self.module.HostTraceReplay(one_dropout, (x,))
        self.addCleanup(replay.close)
        ((launch, parameter, offset),) = replay.lowered.rng_fields
        params = replay.tape.launches[launch]["params"]
        by_name = {p["name"]: p for p in params}
        pointers = [p for p in params if p["kind"] == "rng" and p["size"] == 16]
        self.assertEqual(len(pointers), 1)
        self.assertEqual(
            by_name["philox_offset_intragraph"]["offset"], pointers[0]["offset"] + 16
        )
        self.assertEqual(by_name["philox_offset_intragraph"]["size"], 8)
        self.assertEqual(by_name["philox_captured"]["size"], 1)
        admitted = [
            (o - offset, len(data))
            for p, o, data in replay.lowered.calls[launch].constants
            if p == parameter and o < offset + 16 and o + len(data) > offset
        ]
        self.assertEqual(admitted, [])
        self._equal(replay, one_dropout, (x,), 3)
        self._equal(replay, one_dropout, (torch.randn(8, 1024, device="cuda"),), 2)

    def test_dropout_then_flash_dropout(self):
        if not PLATFORM_SUPPORTS_FLASH_ATTENTION:
            self.skipTest("flash attention not supported")

        def qkv(B, S):
            return tuple(
                torch.randn(B, S, 8, 64, device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            )

        replay = self._replay(dropout_then_flash, qkv(4, 128))
        self._equal(replay, dropout_then_flash, qkv(4, 128), 5)
        for B in (2, 8):
            self._equal(replay, dropout_then_flash, qkv(B, 128), 3)


if __name__ == "__main__":
    run_tests()
