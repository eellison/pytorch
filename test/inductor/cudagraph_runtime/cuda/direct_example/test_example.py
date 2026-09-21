"""Exercise the self-contained example through ordinary and native execution."""

import json
import sys
from unittest import mock

import example
import torch
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch._inductor.runtime._cudagraph.frontend import DirectKernelCall
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


class TestOrdinaryUserEntry(TestCase):
    def test_composition_and_output_lifetime(self, device):
        with torch.cuda.device(device):
            runtime, owner, add = example.make_example(torch.cuda.current_device())
            self.addCleanup(owner.close)
            self.addCleanup(add.close)
            self.addCleanup(runtime.close)
            self.enterContext(mock.patch.object(torch, "compile", side_effect=AssertionError("No host compiler")))
            self.enterContext(mock.patch.object(torch.func, "functionalize", side_effect=AssertionError("No functionalization")))
            samples = [(rows, torch.randn((rows, 128), device=device)) for rows in (5, 7, 35, 96)]
            self.assertEqual(len({source.data_ptr() for _, source in samples}), len(samples))
            held, frames = [], []
            native_calls = 0

            def profile(frame, event, result):
                if event == "call":
                    frames.append((frame.f_code.co_filename, frame.f_code.co_name))

            for index, (rows, source) in enumerate(samples):
                box = [rows, source]
                if index == 0:
                    actual, = runtime(box)
                else:
                    try:
                        sys.setprofile(profile)
                        actual, = runtime.entry(box)
                    finally:
                        sys.setprofile(None)
                    self.assertEqual(frames, [])
                    native_calls += 1
                self.assertEqual(box, [])
                ordinary, = example.host([rows, source])
                expected = (source + 1) * 2 + rows + 1
                self.assertEqual(actual, ordinary)
                self.assertEqual(actual, expected)
                self.assertEqual(tuple(actual.shape), (rows, 128))
                self.assertNotEqual(actual.data_ptr(), ordinary.data_ptr())
                held.append((actual, actual.clone()))
                self.assertEqual(len(runtime.variants), 1)

            compile_count = owner._capture.calls
            self.assertEqual(compile_count, 1)
            variant_count = len(runtime.variants)
            program = runtime.variants[0].program
            calls = tuple(event for event in program.events if type(event) in (DirectKernelCall, CuTeCall))
            self.assertEqual(tuple(type(call) for call in calls), (DirectKernelCall, CuTeCall, DirectKernelCall))
            self.assertEqual(len(program.allocations), 2)
            self.assertEqual(native_calls, 3)
            self.assertEqual(len({actual.data_ptr() for actual, _ in held}), len(held))
            runtime.close()
            self.assertTrue(runtime.closed)
            self.assertEqual(owner._native_borrows, set())
            add.close()
            owner.close()
            for actual, expected in held:
                self.assertEqual(actual, expected)

            result = {
                "accepted": True,

                "samples": [rows for rows, _ in samples],
                "native_calls": native_calls,
                "variants": variant_count,
                "cute_compilations": compile_count,
                "program_calls": len(calls),
                "allocations": len(program.allocations),
                "ordinary_references": len(samples),
                "held_outputs": len(held),
                "held_outputs_after_close": True,
                "python_frames": frames,
            }
            print("ORDINARY_USER_ENTRY_RESULT=" + json.dumps(result, sort_keys=True))


instantiate_device_type_tests(TestOrdinaryUserEntry, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
