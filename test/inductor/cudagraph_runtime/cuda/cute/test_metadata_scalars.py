"""Original Tensor storage offsets feed real CuTe Int64 scalar formals."""

import hashlib
import json
from pathlib import Path
import struct
import sys
from unittest import mock


from torch._inductor.runtime._cudagraph import _sdk

_sdk.activate()

import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver
import torch
from torch._inductor.runtime._cudagraph import replay
from torch._inductor.runtime._cudagraph.api import (
    DirectCuTe, DirectHost, InputContract, ObservedOrdinaryEntry, PythonEntry, SignaturePolicy, TensorInput,
)
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@cute.kernel
def add_offset(source: cute.Tensor, output: cute.Tensor, offset: cutlass.Int64):
    index = cute.arch.thread_idx()[0]
    output[index] = source[index] + cutlass.Float32(offset)


@cute.jit
def launch_offset(source: cute.Tensor, output: cute.Tensor, offset: cutlass.Int64, stream: driver.CUstream):
    add_offset(source, output, offset).launch(grid=(1, 1, 1), block=(8, 1, 1), smem=0, stream=stream)


def convert_arguments(source, output, offset):
    return (from_dlpack(source, assumed_align=4, use_32bit_stride=False),
            from_dlpack(output, assumed_align=4, use_32bit_stride=False), cutlass.Int64(offset))


CUTE = None


def host(box):
    source, = box
    box.clear()
    output = torch.empty_strided((8,), (1,), dtype=source.dtype, device=source.device)
    CUTE(source, output, source.storage_offset())
    return (output,)


class TestMetadataScalars(TestCase):
    def test_storage_offset_int64_formal(self, device):
        with torch.cuda.device(device):
            owner = ObservedOrdinaryEntry(PythonEntry(launch_offset), add_offset,
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
                _, input_count, allocations, _, copies, calls, launches, _, _ = args
                self.assertEqual(input_count, 1)
                self.assertEqual(len(allocations), 1)
                self.assertEqual(copies, ())
                call, = calls
                launch, = launches
                field, = [field for field in call.fields if field.kind == "i64"]
                payload = launch.argument_bytes[field.parameter][field.byte_offset:field.byte_offset + 8]
                actual, = struct.unpack("q", payload)
                self.assertEqual(actual, kwargs["capture_inputs"][0].storage_offset())
                self.assertIn(("storage_offset", 0), kwargs["numeric"].instructions)
                captures.append(actual)
                return make_replay(*args, **kwargs)

            self.enterContext(mock.patch.object(replay, "_make_replay", side_effect=capture))
            held, inputs, addresses = [], [], set()
            for step, offset in enumerate((0, 1, 3, 6)):
                storage = torch.arange(32, dtype=torch.float32, device=device) + step * 100
                source = storage[offset:offset + 8]
                inputs.append(source)
                self.assertEqual(source.storage_offset(), offset)
                self.assertNotIn(source.data_ptr(), addresses)
                addresses.add(source.data_ptr())
                expected = source + offset
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
                    self.assertEqual(frames, [])
                self.assertEqual(box, [])
                ordinary, = host([source])
                self.assertEqual(actual, expected)
                self.assertEqual(ordinary, expected)
                held.append((actual, expected))
            variant, = runtime.variants
            call, = [event for event in variant.program.events if type(event) is CuTeCall]
            self.assertEqual(len([field for field in call.bound.fields if field.kind == "i64"]), 1)
            self.assertEqual(variant.program.integer_inputs, ())
            self.assertEqual(captures, [0])
            self.assertEqual(owner._capture.calls, 1)
            runtime.close()
            owner.close()
            for actual, expected in held:
                self.assertEqual(actual, expected)
            print("METADATA_SCALAR_RESULT=" + json.dumps({


                "offsets": [0, 1, 3, 6], "samples": 4, "fresh_input_addresses": len(addresses),
                "native_hits": 3, "python_frames_on_hits": 0, "variants": 1,
                "ordinary_references": 4, "compilations": 1, "captures": captures,
                "scalar_type": "i64", "held_outputs": len(held),
            }, sort_keys=True))


instantiate_device_type_tests(TestMetadataScalars, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
