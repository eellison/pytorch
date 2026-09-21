"""Absolute-view addresses feed real CuTe Int64 scalar formals."""

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
from torch._inductor.runtime._cudagraph import direct_host, replay
from torch._inductor.runtime._cudagraph.api import (
    DirectCuTe, DirectHost, InputContract, ObservedOrdinaryEntry, PythonEntry, SignaturePolicy, TensorInput,
)
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch._inductor.runtime.cudagraph_arg_mapping import InputSource, ParameterSource
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@cute.kernel
def store_address(output: cute.Tensor, address: cutlass.Int64):
    index = cute.arch.thread_idx()[0]
    output[index] = address


@cute.jit
def launch_address(output: cute.Tensor, address: cutlass.Int64, stream: driver.CUstream):
    store_address(output, address).launch(grid=(1, 1, 1), block=(8, 1, 1), smem=0, stream=stream)


def convert_arguments(output, address):
    return from_dlpack(output, assumed_align=8, use_32bit_stride=False), cutlass.Int64(address)


CUTE = None


def host(box):
    source, = box
    box.clear()
    absolute = source.as_strided((8,), (1,), storage_offset=0)
    output = torch.empty_strided((8,), (1,), dtype=torch.int64, device=source.device)
    CUTE(output, absolute.data_ptr())
    return (output,)


class TestCuTeAbsoluteAddressScalar(TestCase):
    def test_absolute_view_int64_formal(self, device):
        with torch.cuda.device(device):
            owner = ObservedOrdinaryEntry(PythonEntry(launch_address), store_address,
                policy=SignaturePolicy(32, 64, 8, "stream"), conversion=convert_arguments)
            self.addCleanup(owner.close)
            self.enterContext(mock.patch.dict(globals(), {"CUTE": DirectCuTe(owner)}))
            contract = InputContract(("tensor",), (TensorInput(0, torch.float32, (8,), (1,)),), (),
                                     device_index=torch.cuda.current_device())
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            observations = self.enterContext(mock.patch.object(direct_host, "_observe_direct",
                                                                wraps=direct_host._observe_direct))
            preparations = self.enterContext(mock.patch.object(direct_host, "_prepare_observed",
                                                               wraps=direct_host._prepare_observed))
            captures = []
            make_replay = replay._make_replay

            def inspect_capture(*args, **kwargs):
                _, input_count, allocations, _, copies, calls, launches, _, _ = args
                self.assertEqual(input_count, 1)
                self.assertEqual(len(allocations), 1)
                self.assertEqual(copies, ())
                call, = calls
                launch, = launches
                field, = [field for field in call.fields if type(field.source) is ParameterSource]
                self.assertEqual(field.kind, "i64")
                self.assertEqual(field.source.width, 64)
                self.assertEqual({pointer.root for pointer in field.source.pointers}, {InputSource(0)})
                source = kwargs["capture_inputs"][0]
                expected = source.as_strided((8,), (1,), storage_offset=0).data_ptr()
                image = launch.argument_bytes[field.parameter]
                self.assertEqual(image[field.byte_offset:field.byte_offset + 8], struct.pack("q", expected))
                result = make_replay(*args, **kwargs)
                self.assertIn(("storage_offset", 0), kwargs["numeric"].instructions)
                captures.append((field.parameter, field.byte_offset, field.source.width))
                return result

            self.enterContext(mock.patch.object(replay, "_make_replay", side_effect=inspect_capture))
            samples = []
            for offset in (2, 0, 1, 6):
                storage = torch.empty(32, dtype=torch.float32, device=device)
                source = storage[offset:offset + 8]
                self.assertEqual(source.storage_offset(), offset)
                self.assertLess(storage.data_ptr(), 1 << 63)
                samples.append((source, storage))
            self.assertEqual(len({source.data_ptr() for source, _ in samples}), 4)
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
                    self.assertEqual(frames, [])
                self.assertEqual(box, [])
                expected = torch.full((8,), storage.data_ptr(), dtype=torch.int64, device=device)
                self.assertEqual(actual.dtype, torch.int64)
                self.assertEqual(actual, expected)
                ordinary, = host([source])
                self.assertEqual(ordinary, expected)
                self.assertEqual(len(runtime.variants), 1)
                held.append((actual, expected))
            self.assertEqual(observations.call_count, 1)
            self.assertEqual(preparations.call_count, 1)
            self.assertEqual(owner._capture.calls, 1)
            self.assertEqual(len(captures), 1)
            self.assertEqual(len({actual.data_ptr() for actual, _ in held}), 4)
            variant, = runtime.variants
            call, = [event for event in variant.program.events if type(event) is CuTeCall]
            self.assertEqual(tuple(pointer.root for pointer in call.pointers),
                             (variant.program.allocations[0].source,))
            formal, = [formal for formal in call.bound.module.artifact.formals if formal.name == "address"]
            self.assertEqual(formal.kind, "Var")
            leaf, = formal.leaves
            self.assertEqual(leaf.llvm_type, "i64")
            self.assertEqual(variant.program.integer_inputs, ())
            self.assertEqual(variant.guard.boxed_pointer_indices, (0,))
            self.assertEqual(variant.guard.boxed_storage_offset_indices, (0,))
            runtime.close()
            self.assertEqual(owner._native_borrows, set())
            owner.close()
            for actual, expected in held:
                self.assertEqual(actual, expected)
            print("CUTE_ABSOLUTE_ADDRESS_SCALAR_RESULT=" + json.dumps({


                "offsets": [source.storage_offset() for source, _ in samples], "samples": 4,
                "fresh_input_addresses": 4, "native_hits": 3, "python_frames_on_hits": 0,
                "variants": 1, "ordinary_observations": 1, "preparations": 1,
                "ordinary_references": 4, "torch_references": 4, "compilations": 1,
                "captures": captures, "scalar_type": "i64", "scalar_only_input_root": True,
                "held_outputs_after_close": len(held),
            }, sort_keys=True))


instantiate_device_type_tests(TestCuTeAbsoluteAddressScalar, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
