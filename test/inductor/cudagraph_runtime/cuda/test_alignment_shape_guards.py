"""Compose selected pointer alignment with local host shape dispatch."""

import sys
from unittest import mock

import torch
import triton
import triton.language as tl
from torch._inductor.runtime._cudagraph import direct_host
from torch._inductor.runtime._cudagraph.api import (
    DirectHost,
    DirectTriton,
    InputContract,
    IntegerRange,
    IntExpr,
    TensorInput,
)
from torch._inductor.runtime._cudagraph.frontend import DirectKernelCall
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@triton.jit(do_not_specialize=["count"], do_not_specialize_on_alignment=["count"])
def add_views(left, right, output, count, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < count
    value = tl.load(left + index, mask, other=0) + 2 * tl.load(right + index, mask, other=0)
    tl.store(output + index, value, mask)


ADD = None


def host(box):
    count, left, right = box
    box.clear()
    block = 64 if count <= 128 else 128
    output = torch.empty_strided((count,), (1,), dtype=left.dtype, device=left.device)
    ADD[(triton.cdiv(count, block),)](left, right, output, count, BLOCK=block)
    return (output,)


class TestAlignmentShapeGuards(TestCase):
    @parametrize("aligned_first", (True, False))
    def test_independent_pointer_and_shape_guards(self, device, aligned_first):
        if torch.version.hip:
            self.skipTest("requires CUDA parameterized graph replay")
        if aligned_first:
            cases = (
                (64, 0, 0, True),
                (96, 4, 4, False),
                (96, 1, 4, True),
                (96, 4, 1, True),
                (96, 1, 1, True),
                (192, 4, 4, True),
                (255, 1, 1, True),
                (80, 0, 0, False),
                (223, 1, 1, False),
            )
        else:
            cases = (
                (96, 1, 1, True),
                (64, 0, 0, False),
                (96, 4, 4, False),
                (192, 1, 1, True),
                (224, 0, 0, False),
                (80, 1, 1, False),
                (255, 1, 1, False),
            )
        self.enterContext(torch.cuda.device(device))
        adapter = DirectTriton(add_views)
        self.addCleanup(adapter.close)
        self.enterContext(mock.patch.dict(globals(), {"ADD": adapter}))
        count = IntExpr("boxed", 0)
        contract = InputContract(
            ("integer", "tensor", "tensor"),
            tuple(TensorInput(index, torch.float32, (count,), (1,)) for index in (1, 2)),
            (IntegerRange(0, 1, 512),),
            device_index=torch.cuda.current_device(),
        )
        runtime = DirectHost(host, contract)
        self.addCleanup(runtime.close)
        observations = self.enterContext(
            mock.patch.object(direct_host, "_observe_direct", wraps=direct_host._observe_direct)
        )
        preparations = self.enterContext(
            mock.patch.object(direct_host, "_prepare_observed", wraps=direct_host._prepare_observed)
        )
        samples = []
        for size, left_offset, right_offset, miss in cases:
            tensors = []
            for offset in (left_offset, right_offset):
                storage = torch.randn(size + 8, device=device)
                tensor = storage[offset:offset + size]
                self.assertEqual(storage.data_ptr() % 16, 0)
                self.assertEqual(tensor.data_ptr() - storage.data_ptr(), offset * tensor.element_size())
                self.assertEqual(tensor.data_ptr() % 16, (offset * tensor.element_size()) % 16)
                tensors.append(tensor)
            samples.append((size, *tensors, miss))
        self.assertEqual(len({tensor.data_ptr() for _, left, right, _ in samples
                              for tensor in (left, right)}), 2 * len(samples))

        misses = 0
        held = []
        for index, (size, left, right, miss) in enumerate(samples):
            frames = []

            def profile(frame, event, result):
                if event == "call":
                    frames.append(frame.f_code)

            box = [size, left, right]
            if index == 0:
                actual, = runtime(box)
            else:
                try:
                    sys.setprofile(profile)
                    actual, = runtime.entry(box)
                finally:
                    sys.setprofile(None)
                self.assertEqual(bool(frames), miss)
            misses += int(miss)
            self.assertEqual(box, [])
            self.assertEqual(observations.call_count, misses)
            self.assertEqual(preparations.call_count, misses)
            self.assertEqual(len(runtime.variants), misses)
            expected = left + 2 * right
            self.assertEqual(actual, expected)
            self.assertEqual((actual,), host([size, left, right]))
            held.append((actual, expected))

        call, = (event for event in runtime.variants[0].program.events if type(event) is DirectKernelCall)
        alignments = {row.formal: row.alignment for row in call.owner.pointers}
        for name in ("left", "right"):
            if aligned_first:
                self.assertGreater(alignments[name], 4)
            else:
                self.assertLessEqual(alignments[name], 4)
        self.assertEqual(len({actual.data_ptr() for actual, _ in held}), len(held))
        runtime.close()
        adapter.close()
        for actual, expected in held:
            self.assertEqual(actual, expected)


instantiate_device_type_tests(TestAlignmentShapeGuards, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
