"""CuTe packs signed absolute-view displacements from original input metadata."""

import hashlib
import json
from pathlib import Path
import struct
import sys
from unittest import mock


from torch._inductor.runtime._cudagraph import _sdk

_sdk.activate()

from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver
import torch
from torch._inductor.runtime._cudagraph import replay
from torch._inductor.runtime._cudagraph.api import (
    DirectCuTe, DirectHost, InputContract, ObservedOrdinaryEntry, PythonEntry, SignaturePolicy, TensorInput,
)
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch._inductor.runtime.cudagraph_arg_mapping import InputSource, IntExpr, PointerSource
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@cute.kernel
def copy_views(absolute: cute.Tensor, relative: cute.Tensor, output: cute.Tensor):
    index = cute.arch.thread_idx()[0]
    output[index] = absolute[index]
    output[index + 4] = relative[index]


@cute.jit
def launch_views(absolute: cute.Tensor, relative: cute.Tensor, output: cute.Tensor, stream: driver.CUstream):
    copy_views(absolute, relative, output).launch(grid=(1, 1, 1), block=(4, 1, 1), smem=0, stream=stream)


def convert_arguments(absolute, relative, output):
    return tuple(from_dlpack(tensor, assumed_align=4, use_32bit_stride=False)
                 for tensor in (absolute, relative, output))


CUTE = None


def host(box):
    source, = box
    box.clear()
    absolute = source.as_strided((4,), (1,), storage_offset=0)
    relative = source[1:5]
    output = torch.empty_strided((8,), (1,), dtype=source.dtype, device=source.device)
    CUTE(absolute, relative, output)
    return output, absolute, relative


class TestCuTeStorageOffsetReplay(TestCase):
    def test_absolute_relative_fields_and_native_reuse(self, device):
        with torch.cuda.device(device):
            owner = ObservedOrdinaryEntry(PythonEntry(launch_views), copy_views,
                policy=SignaturePolicy(32, 64, 4, "stream"), conversion=convert_arguments)
            self.addCleanup(owner.close)
            self.enterContext(mock.patch.dict(globals(), {"CUTE": DirectCuTe(owner)}))
            contract = InputContract(("tensor",), (TensorInput(0, torch.float32, (8,), (1,)),), (),
                                     device_index=torch.cuda.current_device())
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            captures = []
            make_replay = replay._make_replay

            def capture(*args, **kwargs):
                _, input_count, allocations, _, copies, calls, launches, buffers, _ = args
                self.assertEqual(input_count, 1)
                self.assertEqual(len(allocations), 1)
                self.assertEqual(copies, ())
                call, = calls
                launch, = launches
                numeric = kwargs["numeric"]
                offsets = []
                self.assertEqual(len(call.fields), 3)
                for field in call.fields:
                    source = field.source
                    self.assertIs(type(source), PointerSource)
                    self.assertEqual(field.kind, "pointer")
                    tensor = (kwargs["capture_inputs"][source.root.index]
                              if type(source.root) is InputSource else buffers[source.root])
                    offset = numeric.values[numeric.add(source.byte_offset)]
                    expected = tensor.data_ptr() + offset
                    payload = launch.argument_bytes[field.parameter][field.byte_offset:field.byte_offset + 8]
                    self.assertEqual(struct.unpack("P", payload)[0], expected)
                    offsets.append(offset)
                self.assertEqual(offsets, [-8, 4, 0])
                captures.append(tuple(offsets))
                return make_replay(*args, **kwargs)

            self.enterContext(mock.patch.object(replay, "_make_replay", side_effect=capture))
            held, addresses = [], set()
            for step, offset in enumerate((2, 3, 0)):
                storage = torch.arange(32, dtype=torch.float32, device=device) + step * 100
                source = storage[offset:offset + 8]
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
                    self.assertEqual(frames, [])
                self.assertEqual(box, [])
                ordinary = host([source])
                expected = torch.cat((storage[:4], source[1:5]))
                self.assertEqual(actual[0], expected)
                self.assertEqual(ordinary[0], expected)
                self.assertEqual(actual[1].data_ptr(), storage.data_ptr())
                self.assertEqual(actual[1].storage_offset(), 0)
                self.assertEqual(actual[2].data_ptr(), source.data_ptr() + 4)
                self.assertEqual(actual[2].storage_offset(), offset + 1)
                held.append((actual, expected, storage[:4].clone(), source[1:5].clone()))
            variant, = runtime.variants
            self.assertEqual(owner._capture.calls, 1)
            self.assertEqual(captures, [(-8, 4, 0)])
            self.assertIsNone(variant.guard)
            call, = [event for event in variant.program.events if type(event) is CuTeCall]
            self.assertEqual(tuple(pointer.root for pointer in call.pointers[:2]), (InputSource(0), InputSource(0)))
            self.assertEqual(call.pointers[1].byte_offset, IntExpr("constant", 4))
            self.assertEqual(tuple(formal.data_alignment for formal in call.bound.module.artifact.formals
                                   if formal.kind == "Tensor"), (4, 4, 4))
            runtime.close()
            owner.close()
            for actual, expected, absolute, relative in held:
                self.assertEqual(actual[0], expected)
                self.assertEqual(actual[1], absolute)
                self.assertEqual(actual[2], relative)
            print("CUTE_STORAGE_OFFSET_RESULT=" + json.dumps({


                "samples": 3, "offsets": [2, 3, 0], "native_hits": 2, "variants": 1,
                "ordinary_references": 3, "compilations": 1, "captures": captures,
                "compiler_alignment": [4, 4, 4], "held_outputs": len(held),
            }, sort_keys=True))


instantiate_device_type_tests(TestCuTeStorageOffsetReplay, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
