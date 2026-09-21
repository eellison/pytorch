# Owner(s): ["module: inductor"]

"""Nonempty views of empty inputs must not bind to a null graph parameter."""

from unittest import mock

import triton
import triton.language as tl

import torch
from torch._inductor.runtime._cudagraph import direct_host, replay
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    FXTraceDeclined,
)
from torch._inductor.runtime._cudagraph.api import (
    DirectHost,
    DirectTriton,
    InputContract,
    TensorInput,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@triton.jit
def store_pointer(source, output):
    tl.store(output, source.to(tl.int64))


STORE = None


def host(box):
    (source,) = box
    box.clear()
    view = source.as_strided((4,), (1,), storage_offset=0)
    output = torch.empty_strided((1,), (1,), dtype=torch.int64, device=source.device)
    STORE[(1,)](view, output)
    return (output,)


class TestNonemptyViewFromEmpty(TestCase):
    @parametrize("offset", (0, 4))
    def test_ordinary_fallback_preserves_preparation_decline(self, device, offset):
        with torch.cuda.device(device):
            adapter = DirectTriton(store_pointer)
            self.addCleanup(adapter.close)
            self.enterContext(mock.patch.dict(globals(), {"STORE": adapter}))
            contract = InputContract(
                ("tensor",),
                (TensorInput(0, torch.float32, (0,), (1,)),),
                (),
                device_index=torch.cuda.current_device(),
            )
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            preparations = self.enterContext(
                mock.patch.object(
                    direct_host, "prepare_terminal", wraps=direct_host.prepare_terminal
                )
            )
            captures = self.enterContext(
                mock.patch.object(replay, "_make_replay", wraps=replay._make_replay)
            )
            observations = self.enterContext(
                mock.patch.object(
                    direct_host, "_observe_direct", wraps=direct_host._observe_direct
                )
            )
            storage = [
                torch.arange(16, dtype=torch.float32, device=device) for _ in range(2)
            ]
            self.assertNotEqual(storage[0].data_ptr(), storage[1].data_ptr())
            held = []
            for backing in storage:
                source = backing[offset:offset]
                view = source.as_strided((4,), (1,), storage_offset=0)
                self.assertEqual(source.data_ptr(), 0)
                self.assertEqual(source.storage_offset(), offset)
                self.assertEqual(view.data_ptr(), backing.data_ptr())
                self.assertTrue(torch._C._is_alias_of(source, view))
                expected = torch.tensor(
                    [backing.data_ptr()], dtype=torch.int64, device=device
                )
                (ordinary,) = host([source])
                self.assertEqual(ordinary, expected)
                previous_observations = observations.call_count
                box = [source]
                (actual,) = runtime(box)
                self.assertEqual(box, [])
                self.assertEqual(actual, expected)
                self.assertEqual(observations.call_count, previous_observations + 1)
                with self.assertRaises(FXTraceDeclined) as error:
                    direct_host.prepare_direct(host, contract, (source,))
                self.assertEqual(
                    str(error.exception),
                    "A nonempty view cannot use an empty input's null data pointer as its root",
                )
                self.assertEqual(preparations.call_count, 0)
                self.assertEqual(captures.call_count, 0)
                self.assertEqual(len(runtime.variants), 0)
                self.assertIsNone(runtime.entry)
                held.append((ordinary, actual, expected))
            runtime.close()
            adapter.close()
            for ordinary, actual, expected in held:
                self.assertEqual(ordinary, expected)
                self.assertEqual(actual, expected)


instantiate_device_type_tests(TestNonemptyViewFromEmpty, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
