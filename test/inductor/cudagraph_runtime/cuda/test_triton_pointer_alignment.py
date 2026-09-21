"""Reuse ordinary Triton pointer alignment through original input/view guards."""

import ctypes
import hashlib
import json
from pathlib import Path
import sys
from unittest import mock


import torch
import triton
import triton.language as tl
from torch._inductor.runtime._cudagraph import direct_host, replay
from torch._inductor.runtime._cudagraph.api import DirectHost, DirectTriton, InputContract, TensorInput
from torch._inductor.runtime._cudagraph.frontend import DirectKernelCall
from torch._inductor.runtime.cudagraph_arg_mapping import InputSource, IntExpr, PointerSource
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@triton.jit
def combine_views(raw, absolute, output):
    index = tl.arange(0, 8)
    tl.store(output + index, tl.load(raw + index) + 2 * tl.load(absolute + index))


COMBINE = None


def host(box):
    source, = box
    box.clear()
    absolute = source.as_strided((8,), (1,), storage_offset=0)
    output = torch.empty_strided((8,), (1,), dtype=source.dtype, device=source.device)
    COMBINE[(1,)](source, absolute, output)
    return (output,)


class TestTritonPointerAlignment(TestCase):
    def test_raw_and_view_alignment_variants_without_copy(self, device):
        with torch.cuda.device(device):
            adapter = DirectTriton(combine_views)
            self.addCleanup(adapter.close)
            self.enterContext(mock.patch.dict(globals(), {"COMBINE": adapter}))
            contract = InputContract(("tensor",), (TensorInput(0, torch.float32, (8,), (1,)),), (),
                                     device_index=torch.cuda.current_device())
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            observations = self.enterContext(mock.patch.object(direct_host, "_observe_direct",
                                                                wraps=direct_host._observe_direct))
            preparations = self.enterContext(mock.patch.object(direct_host, "_prepare_observed",
                                                               wraps=direct_host._prepare_observed))
            captures = []
            original_make_replay = replay._make_replay

            def capture(*args, **kwargs):
                _, _, allocations, _, copies, calls, launches, _, _ = args
                self.assertEqual(len(allocations), 1)
                self.assertEqual(copies, ())
                self.assertEqual(len(launches), 1)
                call, = calls
                sources = {argument.formal: argument.source for argument in call.arguments}
                self.assertEqual(sources["raw"], PointerSource(InputSource(0), IntExpr("constant", 0)))
                self.assertIs(type(sources["absolute"]), PointerSource)
                self.assertEqual(sources["absolute"].root, InputSource(0))
                numeric = kwargs["numeric"]
                offset = numeric.values[numeric.add(sources["absolute"].byte_offset)]
                self.assertEqual(offset, -4 * kwargs["capture_inputs"][0].storage_offset())
                captures.append(offset)
                return original_make_replay(*args, **kwargs)

            self.enterContext(mock.patch.object(replay, "_make_replay", side_effect=capture))
            samples = []
            for step, offset in enumerate((0, 1, 4, 5, 0, 1)):
                storage = torch.arange(32, dtype=torch.float32, device=device) + step * 100
                self.assertEqual(storage.data_ptr() % 16, 0)
                source = storage[offset:offset + 8]
                self.assertEqual(source.storage_offset(), offset)
                samples.append((source, storage))
            self.assertEqual(len({source.data_ptr() for source, _ in samples}), 6)
            held = []
            for step, (source, storage) in enumerate(samples):
                box, frames = [source], []

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
                    if step == 1:
                        self.assertTrue(frames)
                    else:
                        self.assertEqual(frames, [])
                self.assertEqual(box, [])
                expected = source + 2 * storage[:8]
                self.assertEqual(actual, expected)
                ordinary, = host([source])
                self.assertEqual(ordinary, expected)
                self.assertEqual(len(runtime.variants), min(step + 1, 2))
                held.append((actual, expected))
            self.assertEqual(observations.call_count, 2)
            self.assertEqual(preparations.call_count, 2)
            self.assertEqual(captures, [0, -4])
            alignments, predicates = [], []
            for variant in runtime.variants:
                call, = [event for event in variant.program.events if type(event) is DirectKernelCall]
                alignments.append({row.formal: row.alignment for row in call.owner.pointers})
                guard = variant.guard
                self.assertEqual(guard.boxed_integer_indices, ())
                self.assertEqual(guard.boxed_pointer_indices, (0,))
                self.assertEqual(guard.boxed_storage_offset_indices, (0,))
                predicates.append(ctypes.CFUNCTYPE(ctypes.c_int8, ctypes.POINTER(ctypes.c_int64),
                                                  ctypes.POINTER(ctypes.c_double))(guard.function_address))
            self.assertGreater(alignments[0]["raw"], 4)
            self.assertLessEqual(alignments[1]["raw"], 4)
            self.assertGreater(alignments[0]["absolute"], 4)
            self.assertEqual(alignments[1]["absolute"], alignments[0]["absolute"])
            alignment = max(alignments[0]["raw"], alignments[0]["absolute"])
            base = alignment * 64
            self.assertEqual(predicates[0]((ctypes.c_int64 * 2)(base, 0), None), 1)
            self.assertEqual(predicates[0]((ctypes.c_int64 * 2)(base + 4, 1), None), 0)
            self.assertEqual(predicates[0]((ctypes.c_int64 * 2)(base, 1), None), 0)
            self.assertEqual(predicates[1]((ctypes.c_int64 * 2)(base + 4, 1), None), 1)
            self.assertEqual(predicates[1]((ctypes.c_int64 * 2)(base + 4, 0), None), 0)
            runtime.close()
            adapter.close()
            for actual, expected in held:
                self.assertEqual(actual, expected)
            print("TRITON_POINTER_ALIGNMENT_RESULT=" + json.dumps({


                "samples": 6, "offsets": [source.storage_offset() for source, _ in samples],
                "ordinary_observations": 2, "preparations": 2, "captures": captures,
                "variants": 2, "native_hits": 4, "ordinary_references": 6,
                "compiler_alignments": alignments, "held_outputs": len(held), "normalization_copies": 0,
                "view_guard_acceptances": 2, "view_guard_rejections": 3,
            }, sort_keys=True))


instantiate_device_type_tests(TestTritonPointerAlignment, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
