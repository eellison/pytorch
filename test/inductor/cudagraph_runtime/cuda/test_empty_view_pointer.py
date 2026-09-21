"""An empty Tensor pointer formal must retain the ordinary null pointer value."""

import json
import struct
import sys
from unittest import mock


import torch
import triton
import triton.language as tl
from torch._inductor.runtime._cudagraph import direct_host, replay
from torch._inductor.runtime._cudagraph.api import DirectHost, DirectTriton, InputContract, TensorInput
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@triton.jit
def store_pointer(source, output):
    tl.store(output, source.to(tl.int64))


STORE = None


def host(box):
    source, = box
    box.clear()
    empty = source[4:4]
    output = torch.empty_strided((1,), (1,), dtype=torch.int64, device=source.device)
    STORE[(1,)](empty, output)
    return (output,)


class TestEmptyViewPointer(TestCase):
    def test_empty_view_pointer_matches_ordinary(self, device):
        with torch.cuda.device(device):
            adapter = DirectTriton(store_pointer)
            self.addCleanup(adapter.close)
            self.enterContext(mock.patch.dict(globals(), {"STORE": adapter}))
            contract = InputContract(("tensor",), (TensorInput(0, torch.float32, (16,), (1,)),), (),
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
                captures.append({"source": repr(source.source), "kind": source.triton_type,
                                 "captured_pointer": struct.unpack("P", launch.argument_bytes[source.call_arg_index])[0]})
                return make_replay(*args, **kwargs)

            self.enterContext(mock.patch.object(replay, "_make_replay", side_effect=capture))
            samples = [torch.arange(16, dtype=torch.float32, device=device) for _ in range(2)]
            self.assertNotEqual(samples[0].data_ptr(), samples[1].data_ptr())
            expected = torch.zeros(1, dtype=torch.int64, device=device)
            held = []
            for step, source in enumerate(samples):
                empty = source[4:4]
                self.assertEqual(empty.numel(), 0)
                self.assertEqual(empty.storage_offset(), 4)
                self.assertEqual(empty.data_ptr(), 0)
                self.assertEqual(empty.const_data_ptr(), 0)
                ordinary, = host([source])
                self.assertEqual(ordinary, expected)
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
                print("EMPTY_VIEW_POINTER=" + json.dumps({
                    "sample": step, "input_pointer": source.data_ptr(),
                    "ordinary_pointer": empty.data_ptr(), "result": actual.item(),
                    "captured": captures, "native_python_frames": frames,

                }), flush=True)
                self.assertEqual(actual, expected)
                held.append(actual)
                self.assertEqual(len(runtime.variants), 1)
            self.assertEqual((observed.call_count, prepared.call_count, len(captures)), (1, 1, 1))
            torch.cuda.synchronize(device)
            runtime.close()
            adapter.close()
            for output in held:
                self.assertEqual(output, expected)


instantiate_device_type_tests(TestEmptyViewPointer, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
