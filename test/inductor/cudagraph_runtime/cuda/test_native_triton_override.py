# Owner(s): ["module: inductor"]

import sys
from unittest import mock

import torch
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
from torch._inductor.runtime.cudagraph_arg_mapping import InputSource, PointerSource
from torch._native.ops.bmm_outer_product import triton_kernels
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


NATIVE_HOST = triton_kernels.bmm_outer_product


def host(box):
    _, left, right = box
    box.clear()
    return (NATIVE_HOST(left, right),)


class TestNativeTritonOverride(TestCase):
    @parametrize("changes", ("addresses", "shape", "alignment"))
    def test_read_only_arguments(self, device, changes):
        with torch.cuda.device(device):
            if changes == "shape":
                sizes, offsets = (48, 80, 112, 48), (0, 0, 0, 0)
                counts = (1, 1, 2, 2)
            elif changes == "alignment":
                sizes, offsets = (48, 48, 48, 48), (0, 1, 4, 0)
                counts = (1, 2, 2, 2)
            else:
                sizes, offsets = (48, 48, 48, 48), (0, 0, 0, 0)
                counts = (1, 1, 1, 1)
            samples = []
            for size, offset in zip(sizes, offsets):
                storage = torch.randn(2 * size + offset, device=device)
                left = storage[offset:].view(2, size, 1)
                right = torch.randn(2, 1, 8, device=device)
                samples.append((size, left, right))
            self.assertEqual(len({left.data_ptr() for _, left, _ in samples}), 4)
            self.assertEqual(
                tuple(left.storage_offset() for _, left, _ in samples), offsets
            )
            expected = [torch.bmm(left, right) for _, left, right in samples]

            kernel = triton_kernels._bmm_outer_product_kernel.jit_kernel
            adapter = DirectTriton(kernel)
            self.addCleanup(adapter.close)
            self.enterContext(
                mock.patch.object(triton_kernels, "_bmm_outer_product_kernel", adapter)
            )
            config = triton_kernels._bmm_outer_product_launch_config
            self.enterContext(
                mock.patch.object(
                    triton_kernels,
                    "_bmm_outer_product_launch_config",
                    config.__wrapped__,
                )
            )
            rows = IntExpr("boxed", 0)
            contract = InputContract(
                ("integer", "tensor", "tensor"),
                (
                    TensorInput(1, torch.float32, (2, rows, 1), (rows, 1, 1)),
                    TensorInput(2, torch.float32, (2, 1, 8), (8, 8, 1)),
                ),
                (IntegerRange(0, 33, 192),),
                device_index=torch.cuda.current_device(),
            )
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            observe = self.enterContext(
                mock.patch.object(
                    direct_host, "_observe_direct", wraps=direct_host._observe_direct
                )
            )
            held = []
            for index, (sample, reference) in enumerate(zip(samples, expected)):
                box, frames = list(sample), []

                def profile(frame, event, arg):
                    if event == "call":
                        frames.append((frame.f_code.co_filename, frame.f_code.co_name))

                if index == 0:
                    (actual,) = runtime(box)
                else:
                    try:
                        sys.setprofile(profile)
                        (actual,) = runtime.entry(box)
                    finally:
                        sys.setprofile(None)
                    if counts[index] != counts[index - 1]:
                        self.assertTrue(frames)
                    else:
                        self.assertEqual(frames, [])
                self.assertEqual(actual, reference)
                self.assertEqual(box, [])
                self.assertEqual(len(runtime.variants), counts[index])
                held.append((actual, reference))
            self.assertEqual(observe.call_count, counts[-1])
            alignments = []
            for variant in runtime.variants:
                (call,) = [
                    event
                    for event in variant.program.events
                    if type(event) is DirectKernelCall
                ]
                self.assertIs(call.owner.jit, kernel)
                pointers = {
                    argument.formal: argument.source for argument in call.arguments
                }
                self.assertEqual(
                    pointers["A_ptr"],
                    PointerSource(InputSource(1), IntExpr("constant", 0)),
                )
                self.assertEqual(
                    pointers["B_ptr"],
                    PointerSource(InputSource(2), IntExpr("constant", 0)),
                )
                alignments.append(
                    {row.formal: row.alignment for row in call.owner.pointers}
                )
            if changes == "alignment":
                self.assertGreater(alignments[0]["A_ptr"], 4)
                self.assertLessEqual(alignments[1]["A_ptr"], 4)
            runtime.close()
            adapter.close()
            for actual, reference in held:
                self.assertEqual(actual, reference)


instantiate_device_type_tests(TestNativeTritonOverride, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
