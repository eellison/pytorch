"""Replay absolute input views and metadata guards on fresh offset inputs."""

import hashlib
import json
from pathlib import Path
import sys
from unittest import mock


import torch
import triton
import triton.language as tl
from torch._inductor.runtime._cudagraph.api import DirectHost, DirectTriton, InputContract, TensorInput
from torch._inductor.runtime._cudagraph.frontend import DirectKernelCall
from torch._inductor.runtime.cudagraph_arg_mapping import InputSource, IntExpr, PointerSource, TensorViewOutput
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@triton.jit(do_not_specialize_on_alignment=["absolute", "relative", "output"])
def copy_views(absolute, relative, output, BIAS: tl.constexpr):
    index = tl.arange(0, 4)
    tl.store(output + index, tl.load(absolute + index) + BIAS)
    tl.store(output + 4 + index, tl.load(relative + index) + BIAS)


COPY = None


def host(box):
    source, = box
    box.clear()
    absolute = source.as_strided((4,), (1,), storage_offset=0)
    relative = source[1:5]
    bias = 1 if source.storage_offset() % 2 == 0 else 2
    output = torch.empty_strided((8,), (1,), dtype=source.dtype, device=source.device)
    COPY[(1,)](absolute, relative, output, bias)
    return output, absolute, relative, source.storage_offset()


class TestStorageOffsetReplay(TestCase):
    def test_views_metadata_and_lifetime(self, device):
        with torch.cuda.device(device):
            adapter = DirectTriton(copy_views)
            self.addCleanup(adapter.close)
            self.enterContext(mock.patch.dict(globals(), {"COPY": adapter}))
            contract = InputContract(("tensor",), (TensorInput(0, torch.float32, (8,), (1,)),), (),
                                     device_index=torch.cuda.current_device())
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            held, addresses, native_hits = [], set(), 0
            offsets = (2, 3, 4, 1, 0, 6)
            for step, offset in enumerate(offsets):
                storage = torch.arange(32, dtype=torch.float32, device=device) + step * 100
                source = storage[offset:offset + 8]
                self.assertEqual(source.storage_offset(), offset)
                self.assertNotIn(source.data_ptr(), addresses)
                addresses.add(source.data_ptr())
                box, frames = [source], []

                def profile(frame, event, result):
                    if event == "call":
                        frames.append((frame.f_code.co_filename, frame.f_code.co_name))

                if step == 0:
                    actual = runtime(box)
                else:
                    try:
                        sys.setprofile(profile)
                        actual = runtime.entry(box)
                    finally:
                        sys.setprofile(None)
                    if step >= 2:
                        self.assertEqual(frames, [])
                        native_hits += 1
                self.assertEqual(box, [])
                ordinary = host([source])
                expected = torch.cat((storage[:4], source[1:5])) + (1 if offset % 2 == 0 else 2)
                self.assertEqual(actual[0], expected)
                self.assertEqual(ordinary[0], expected)
                self.assertEqual(actual[1], storage[:4])
                self.assertEqual(actual[2], source[1:5])
                self.assertEqual(actual[3], offset)
                self.assertEqual(actual[1].data_ptr(), storage.data_ptr())
                self.assertEqual(actual[1].storage_offset(), 0)
                self.assertEqual(actual[2].data_ptr(), source.data_ptr() + source.element_size())
                self.assertEqual(actual[2].storage_offset(), offset + 1)
                held.append((actual, expected, storage[:4].clone(), source[1:5].clone(), offset))
            self.assertEqual(native_hits, 4)
            self.assertEqual(len(runtime.variants), 2)
            for variant in runtime.variants:
                self.assertEqual(variant.guard.boxed_storage_offset_indices, (0,))
                self.assertEqual(variant.guard.boxed_integer_indices, ())
                self.assertEqual(variant.guard.boxed_pointer_indices, ())
                self.assertEqual(len(variant.guard.registration), 5)
                call, = [event for event in variant.program.events if type(event) is DirectKernelCall]
                arguments = {argument.formal: argument.source for argument in call.arguments}
                absolute = arguments["absolute"]
                self.assertIs(type(absolute), PointerSource)
                self.assertEqual(absolute.root, InputSource(0))
                pending, metadata_indices = [absolute.byte_offset], set()
                while pending:
                    expression = pending.pop()
                    if type(expression) is IntExpr:
                        if expression.op == "storage_offset":
                            metadata_indices.add(expression.value)
                        pending.extend(expression.args)
                self.assertEqual(metadata_indices, {0})
                self.assertEqual(arguments["relative"].byte_offset, IntExpr("constant", 4))
                self.assertIs(type(variant.program.outputs[1]), TensorViewOutput)
            runtime.close()
            adapter.close()
            for actual, expected, absolute, relative, offset in held:
                self.assertEqual(actual[0], expected)
                self.assertEqual(actual[1], absolute)
                self.assertEqual(actual[2], relative)
                self.assertEqual(actual[3], offset)
            print("STORAGE_OFFSET_REPLAY_RESULT=" + json.dumps({


                "samples": len(offsets), "offsets": offsets, "native_hits": native_hits,
                "variants": 2, "ordinary_references": len(offsets), "held_outputs": len(held),
                "absolute_aliases": len(held), "relative_aliases": len(held),
                "metadata_guard_indices": [0], "signed_absolute_displacement": True,
            }, sort_keys=True))


instantiate_device_type_tests(TestStorageOffsetReplay, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
