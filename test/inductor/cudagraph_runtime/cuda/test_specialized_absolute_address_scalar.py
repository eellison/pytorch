"""Store absolute-view address integers without dereferencing integer addresses."""

import ctypes
import hashlib
import json
from pathlib import Path
import struct
import sys
from unittest import mock


import torch
import triton
import triton.language as tl
from torch._inductor.runtime._cudagraph import direct_host, replay
from torch._inductor.runtime._cudagraph.api import DirectHost, DirectTriton, InputContract, IntegerRange, TensorInput
from torch._inductor.runtime._cudagraph.frontend import DirectKernelCall
from torch._inductor.runtime.cudagraph_arg_mapping import InputSource, ParameterSource, PointerSource
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@triton.jit(do_not_specialize=["count"], do_not_specialize_on_alignment=["count"])
def store_address(output, address, count):
    index = tl.arange(0, 32)
    tl.store(output + index, address.to(tl.int64), index < count)


STORE = None


def host(box):
    count, source = box
    box.clear()
    absolute = source.as_strided((8,), (1,), storage_offset=0)
    output = torch.empty_strided((count,), (1,), dtype=torch.int64, device=source.device)
    STORE[(1,)](output, absolute.data_ptr(), count)
    return (output,)


class TestAbsoluteAddressScalar(TestCase):
    def test_metadata_address_arithmetic_and_native_reuse(self, device):
        with torch.cuda.device(device):
            adapter = DirectTriton(store_address)
            self.addCleanup(adapter.close)
            self.enterContext(mock.patch.dict(globals(), {"STORE": adapter}))
            contract = InputContract(("integer", "tensor"),
                (TensorInput(1, torch.float32, (8,), (1,)),), (IntegerRange(0, 1, 32),),
                device_index=torch.cuda.current_device())
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            observations = self.enterContext(mock.patch.object(direct_host, "_observe_direct",
                                                                wraps=direct_host._observe_direct))
            preparations = self.enterContext(mock.patch.object(direct_host, "_prepare_observed",
                                                               wraps=direct_host._prepare_observed))
            captures = []
            original_make_replay = replay._make_replay

            def inspect_capture(*args, **kwargs):
                _, _, allocations, _, copies, calls, launches, _, _ = args
                self.assertEqual(len(allocations), 1)
                self.assertEqual(copies, ())
                call, = calls
                launch, = launches
                scalar, = [argument for argument in call.arguments if argument.formal == "address"]
                self.assertEqual(scalar.triton_type, "i64")
                self.assertIs(type(scalar.source), ParameterSource)
                self.assertEqual(scalar.source.width, 64)
                self.assertEqual({pointer.root for pointer in scalar.source.pointers}, {InputSource(1)})
                pointer_roots = {argument.source.root for argument in call.arguments
                                 if type(argument.source) is PointerSource}
                self.assertNotIn(InputSource(1), pointer_roots)
                source = kwargs["capture_inputs"][1]
                expected = source.as_strided((8,), (1,), storage_offset=0).data_ptr()
                self.assertEqual(launch.argument_bytes[scalar.call_arg_index], struct.pack("q", expected))
                result = original_make_replay(*args, **kwargs)
                self.assertIn(("storage_offset", 1), kwargs["numeric"].instructions)
                captures.append({"scalar_slot": scalar.call_arg_index, "scalar_width": 64,
                                 "storage_offset_index": 1, "scalar_only_input_root": 1})
                return result

            self.enterContext(mock.patch.object(replay, "_make_replay", side_effect=inspect_capture))
            samples = []
            for count, offset in ((8, 2), (12, 0), (5, 1), (24, 6)):
                storage = torch.arange(32, dtype=torch.float32, device=device)
                source = storage[offset:offset + 8]
                self.assertEqual(source.storage_offset(), offset)
                self.assertLess(storage.data_ptr(), 1 << 63)
                samples.append((count, source, storage))
            self.assertEqual(len({source.data_ptr() for _, source, _ in samples}), 4)
            held = []
            for step, (count, source, storage) in enumerate(samples):
                box, frames = [count, source], []

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
                expected = torch.full((count,), storage.data_ptr(), dtype=torch.int64, device=device)
                self.assertEqual(actual, expected)
                ordinary, = host([count, source])
                self.assertEqual(ordinary, expected)
                self.assertEqual(len(runtime.variants), 1)
                held.append((actual, expected))
            self.assertEqual(observations.call_count, 1)
            self.assertEqual(preparations.call_count, 1)
            self.assertEqual(len(captures), 1)
            self.assertEqual(len({actual.data_ptr() for actual, _ in held}), 4)
            variant, = runtime.variants
            call, = [event for event in variant.program.events if type(event) is DirectKernelCall]
            selected, = [row for row in call.owner.formals if row.formal == "address"]
            self.assertEqual(selected.triton_type, "i64")
            divisibility, = [value for name, value in selected.attributes if name == "tt.divisibility"]
            self.assertIs(type(divisibility), int)
            self.assertGreater(divisibility, 1)
            for _, _, storage in samples:
                self.assertEqual(storage.data_ptr() % divisibility, 0)
            guard = variant.guard
            self.assertTrue(set(guard.boxed_integer_indices).issubset({0}))
            self.assertEqual(guard.boxed_pointer_indices, (1,))
            self.assertEqual(guard.boxed_storage_offset_indices, (1,))
            self.assertEqual(len(guard.registration), 5)
            predicate = ctypes.CFUNCTYPE(ctypes.c_int8, ctypes.POINTER(ctypes.c_int64),
                                        ctypes.POINTER(ctypes.c_double))(guard.function_address)

            def accepts(pointer, offset):
                values = [8 for _ in guard.boxed_integer_indices]
                values.extend((ctypes.c_int64(pointer).value, offset))
                return predicate((ctypes.c_int64 * len(values))(*values), None)

            self.assertEqual(accepts(divisibility * 16 + 12, 3), 1)
            self.assertEqual(accepts(((1 << 63) - 1) // divisibility * divisibility, 0), 1)
            self.assertEqual(accepts(((1 << 63) + divisibility - 1) // divisibility * divisibility, 0), 0)
            self.assertEqual(accepts((1 << 64) - divisibility, 1 << 62), 0)
            runtime.close()
            adapter.close()
            for actual, expected in held:
                self.assertEqual(actual, expected)
            print("ABSOLUTE_ADDRESS_SCALAR_RESULT=" + json.dumps({


                "samples": 4, "offsets": [source.storage_offset() for _, source, _ in samples],
                "native_hits": 3, "variants": 1, "ordinary_observations": 1,
                "preparations": 1, "captures": captures, "ordinary_references": 4,
                "held_outputs": len(held), "guard_acceptances": 2, "guard_rejections": 2,
                "scalar_only_input_root": True, "selected_scalar_attributes": selected.attributes,
            }, sort_keys=True))


instantiate_device_type_tests(TestAbsoluteAddressScalar, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
