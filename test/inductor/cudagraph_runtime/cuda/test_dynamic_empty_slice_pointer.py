"""A positive symbolic slice end can select empty and nonempty pointer formals."""

import ctypes
import json
import struct
import sys
from unittest import mock


import torch
import triton
import triton.language as tl
from torch._inductor.runtime._cudagraph import direct_host, replay
from torch._inductor.runtime._cudagraph.api import DirectHost, DirectTriton, InputContract, IntegerRange, TensorInput
from torch._inductor.runtime.cudagraph_arg_mapping import ParameterSource, PointerSource
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@triton.jit
def store_pointer(source, output):
    tl.store(output, source.to(tl.int64))


STORE = None


def host(box):
    end, source = box
    box.clear()
    view = source[4:end]
    output = torch.empty_strided((1,), (1,), dtype=torch.int64, device=source.device)
    STORE[(1,)](view, output)
    return (output,)


class TestDynamicEmptySlicePointer(TestCase):
    @parametrize("first_empty", (True, False))
    def test_nullness_guard_and_positive_pointer(self, device, first_empty):
        with torch.cuda.device(device):
            adapter = DirectTriton(store_pointer)
            self.addCleanup(adapter.close)
            self.enterContext(mock.patch.dict(globals(), {"STORE": adapter}))
            contract = InputContract(("integer", "tensor"),
                (TensorInput(1, torch.float32, (16,), (1,)),), (IntegerRange(0, 4, 12),),
                device_index=torch.cuda.current_device())
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            observed = self.enterContext(mock.patch.object(direct_host, "_observe_direct",
                                                           wraps=direct_host._observe_direct))
            prepared = self.enterContext(mock.patch.object(direct_host, "_prepare_observed",
                                                           wraps=direct_host._prepare_observed))
            captures = []
            make_replay = replay._make_replay

            def capture(*args, **kwargs):
                _, _, _, _, copies, calls, launches, _, _ = args
                self.assertEqual(copies, ())
                call, = calls
                launch, = launches
                source, = [argument for argument in call.arguments if argument.formal == "source"]
                end, tensor = kwargs["capture_inputs"]
                pointer = tensor[4:end].data_ptr()
                self.assertEqual(launch.argument_bytes[source.call_arg_index], struct.pack("P", pointer))
                self.assertEqual(source.triton_type, "*fp32")
                if end == 4:
                    self.assertEqual(source.source, ParameterSource("constant", 64, 0))
                else:
                    self.assertIs(type(source.source), PointerSource)
                    self.assertEqual(pointer, tensor.data_ptr() + 16)
                result = make_replay(*args, **kwargs)
                captures.append({"empty": end == 4, "source": repr(source.source),
                                 "captured_pointer": pointer, "abi": call.module.arg_tys})
                return result

            self.enterContext(mock.patch.object(replay, "_make_replay", side_effect=capture))
            ends = (4, 7, 4, 9) if first_empty else (7, 4, 9, 4)
            samples = [torch.arange(16, dtype=torch.float32, device=device) for _ in ends]
            self.assertEqual(len({source.data_ptr() for source in samples}), len(samples))
            held = []
            for step, (end, source) in enumerate(zip(ends, samples, strict=True)):
                pointer = source[4:end].data_ptr()
                self.assertEqual(pointer, 0 if end == 4 else source.data_ptr() + 16)
                expected = torch.tensor([ctypes.c_int64(pointer).value], dtype=torch.int64, device=device)
                ordinary, = host([end, source])
                self.assertEqual(ordinary, expected)
                box, frames = [end, source], []

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
                    if step >= 2:
                        self.assertEqual(frames, [])
                self.assertEqual(box, [])
                self.assertEqual(actual, expected)
                self.assertEqual(len(runtime.variants), min(step + 1, 2))
                held.append((actual, expected))
            self.assertEqual((observed.call_count, prepared.call_count, len(captures)), (2, 2, 2))
            self.assertEqual({row["empty"] for row in captures}, {False, True})
            self.assertEqual({row["abi"] for row in captures}, {"OO"})
            self.assertTrue(all(variant.guard is not None for variant in runtime.variants))
            torch.cuda.synchronize(device)
            runtime.close()
            adapter.close()
            for actual, expected in held:
                self.assertEqual(actual, expected)
            print("DYNAMIC_EMPTY_VIEW_POINTER=" + json.dumps({
                "first_empty": first_empty, "samples": len(samples), "native_hits": 2,
                "ordinary_misses": observed.call_count, "variants": 2, "captures": captures,
                "held_outputs_after_close": len(held),

            }), flush=True)


instantiate_device_type_tests(TestDynamicEmptySlicePointer, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
