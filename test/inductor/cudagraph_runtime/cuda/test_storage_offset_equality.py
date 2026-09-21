"""A locally specialized storage offset must remain a native reuse predicate."""

import json
import sys
from unittest import mock


import torch
import triton
import triton.language as tl
from torch._inductor.runtime._cudagraph import direct_host
from torch._inductor.runtime._cudagraph.api import DirectHost, DirectTriton, InputContract, TensorInput
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@triton.jit
def add_bias(source, output, BIAS: tl.constexpr):
    index = tl.arange(0, 8)
    tl.store(output + index, tl.load(source + index) + BIAS)


ADD = None


def host(box):
    source, = box
    box.clear()
    bias = 1 if source.storage_offset() == 3 else 2
    output = torch.empty_strided((8,), (1,), dtype=source.dtype, device=source.device)
    ADD[(1,)](source, output, BIAS=bias)
    return (output,)


class TestStorageOffsetEquality(TestCase):
    def test_local_equality_misses_then_reuses(self, device):
        with torch.cuda.device(device):
            adapter = DirectTriton(add_bias)
            self.addCleanup(adapter.close)
            self.enterContext(mock.patch.dict(globals(), {"ADD": adapter}))
            contract = InputContract(("tensor",), (TensorInput(0, torch.float32, (8,), (1,)),), (),
                                     device_index=torch.cuda.current_device())
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            observations = self.enterContext(mock.patch.object(
                direct_host, "_observe_direct", wraps=direct_host._observe_direct))
            samples = []
            for step, offset in enumerate((3, 7, 3)):
                storage = torch.arange(32, device=device, dtype=torch.float32) + step * 100
                samples.append(storage[offset:offset + 8])
            self.assertEqual(len({source.data_ptr() for source in samples}), 3)
            self.assertEqual({source.data_ptr() % 16 for source in samples}, {12})
            report = {"accepted": False,
                      "samples": [], "native_hits": 0}
            held = []
            try:
                for step, source in enumerate(samples):
                    box, frames = [source], []
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
                    if not missed:
                        self.assertEqual(frames, [])
                        report["native_hits"] += 1
                    self.assertEqual(box, [])
                    ordinary, = host([source])
                    expected = source + (1 if source.storage_offset() == 3 else 2)
                    report["samples"].append({
                        "offset": source.storage_offset(), "miss": missed,
                        "variants": len(runtime.variants),
                        "guards": [list(variant.program.guards.expressions) for variant in runtime.variants],
                        "actual_first": actual[0].item(), "expected_first": expected[0].item(),
                    })
                    self.assertEqual(ordinary, expected)
                    self.assertEqual(actual, expected)
                    held.append((actual, ordinary, expected))
                self.assertEqual(observations.call_count, 2)
                self.assertEqual(len(runtime.variants), 2)
                self.assertEqual(report["native_hits"], 1)
                runtime.close()
                adapter.close()
                for actual, ordinary, expected in held:
                    self.assertEqual(actual, ordinary)
                    self.assertEqual(actual, expected)
                report["accepted"] = True
            finally:
                print("STORAGE_OFFSET_EQUALITY_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


instantiate_device_type_tests(TestStorageOffsetEquality, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
