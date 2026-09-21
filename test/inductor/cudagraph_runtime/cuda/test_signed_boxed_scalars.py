"""Signed boxed scalars retain their selected ABI and local branch guards."""

import json
import struct
import sys
from unittest import mock


import torch
import triton
import triton.language as tl
from torch._inductor.runtime._cudagraph import direct_host, replay
from torch._inductor.runtime._cudagraph.api import DirectHost, DirectTriton, InputContract, IntegerRange, TensorInput
from torch._inductor.runtime.cudagraph_arg_mapping import ExpressionSource, IntExpr
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@triton.jit
def add_scalars(source, output, narrow, wide, BIAS: tl.constexpr):
    tl.store(output, tl.load(source) + narrow.to(tl.int64) + BIAS)
    tl.store(output + 1, tl.load(source + 1) + wide + BIAS)


ADD = None


def host(box):
    narrow, wide, source = box
    box.clear()
    bias = 5 if narrow < 0 else 9
    output = torch.empty_strided((2,), (1,), dtype=source.dtype, device=source.device)
    ADD[(1,)](source, output, narrow, wide, BIAS=bias)
    return (output,)


class TestSignedBoxedScalars(TestCase):
    def test_zero_negative_and_native_reuse(self, device):
        with torch.cuda.device(device):
            adapter = DirectTriton(add_scalars)
            self.addCleanup(adapter.close)
            self.enterContext(mock.patch.dict(globals(), {"ADD": adapter}))
            contract = InputContract(
                ("integer", "integer", "tensor"),
                (TensorInput(2, torch.int64, (2,), (1,)),),
                (IntegerRange(0, -1024, 1024), IntegerRange(1, -(1 << 41), -(1 << 39))),
                device_index=torch.cuda.current_device(),
            )
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            observations = self.enterContext(mock.patch.object(
                direct_host, "_observe_direct", wraps=direct_host._observe_direct))
            captures = self.enterContext(mock.patch.object(replay, "_make_replay", wraps=replay._make_replay))
            samples = [(narrow, -(1 << 40) + step * 16,
                        torch.tensor((step * 100, step * 100 + 1), dtype=torch.int64, device=device))
                       for step, narrow in enumerate((0, -16, 32, -32, 0))]
            self.assertEqual(len({source.data_ptr() for _, _, source in samples}), len(samples))
            held, hits, paths = [], 0, []
            for step, (narrow, wide, source) in enumerate(samples):
                box, frames = [narrow, wide, source], []
                before = observations.call_count

                def profile(frame, event, result):
                    if event == "call":
                        frames.append((frame.f_code.co_filename, frame.f_code.co_name))

                if step == 0:
                    actual, = runtime(box)
                else:
                    try:
                        sys.setprofile(profile)
                        actual, = runtime.entry(box)
                    finally:
                        sys.setprofile(None)
                missed = observations.call_count != before
                self.assertEqual(missed, step < 2)
                if not missed:
                    self.assertEqual(frames, [])
                    hits += 1
                self.assertEqual(box, [])
                ordinary, = host([narrow, wide, source])
                bias = 5 if narrow < 0 else 9
                expected = source + torch.tensor((narrow + bias, wide + bias), dtype=torch.int64, device=device)
                self.assertEqual(actual, expected)
                self.assertEqual(ordinary, expected)
                self.assertNotEqual(actual.data_ptr(), ordinary.data_ptr())
                held.append((actual, ordinary, expected))
                paths.append({"narrow": narrow, "wide": wide, "miss": missed})

            self.assertEqual(observations.call_count, 2)
            self.assertEqual(captures.call_count, 2)
            self.assertEqual(len(runtime.variants), 2)
            self.assertEqual(hits, 3)
            for index, capture in enumerate(captures.call_args_list):
                bound, = capture.args[5]
                launch, = capture.args[6]
                narrow, wide = bound.arguments[2:4]
                self.assertEqual((narrow.triton_type, wide.triton_type), ("i32", "i64"))
                self.assertEqual(narrow.source, ExpressionSource(IntExpr("boxed", 0)))
                self.assertEqual(wide.source, ExpressionSource(IntExpr("boxed", 1)))
                self.assertEqual(struct.unpack("i", launch.argument_bytes[2]), (samples[index][0],))
                self.assertEqual(struct.unpack("q", launch.argument_bytes[3]), (samples[index][1],))
                self.assertTrue(runtime.variants[index].guard.expressions)
            runtime.close()
            adapter.close()
            for actual, ordinary, expected in held:
                self.assertEqual(actual, ordinary)
                self.assertEqual(actual, expected)

            print("SIGNED_BOXED_SCALARS_RESULT=" + json.dumps({
                "accepted": True,
                "samples": paths, "variants": 2, "captures": 2, "native_hits": hits,
                "ordinary_references": len(samples), "scalar_widths": [32, 64],
                "python_frames_on_native_hits": 0, "held_outputs_after_close": True,
            }, sort_keys=True), flush=True)


instantiate_device_type_tests(TestSignedBoxedScalars, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
