"""Cold resolver controls; uninitialized child objects stand in for compiled owners."""

import os
from unittest import mock

from torch._inductor.runtime._cudagraph._compiler.selected_kernel import resolve_warmed_kernel
from torch._inductor import config
from torch._inductor.codegen.multi_kernel import MultiKernelCall, SizeHintMultiKernelCall
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


@instantiate_parametrized_tests
class TestWarmedMultiKernelSelection(TestCase):
    def carrier(self):
        children = [CachingAutotuner.__new__(CachingAutotuner) for _ in range(2)]
        with config.patch({"triton.multi_kernel": 1}), mock.patch.dict(
            os.environ, {"TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE": "1"}
        ):
            return MultiKernelCall("selection_control", children, {
                0: [slice(0, 2), slice(4, 7)],
                1: [slice(0, 2), slice(7, 10)],
            })

    @parametrize("picked", (0, 1))
    def test_selected_child_and_actual_argument_slices(self, picked):
        carrier = self.carrier()
        carrier.picked_kernel = picked
        winner, receipt = resolve_warmed_kernel(carrier)
        self.assertIs(winner, carrier._kernels[picked])
        self.assertIs(receipt.owner, carrier)
        arguments = tuple(object() for _ in range(10))
        expected = carrier._get_filtered_args(arguments, picked)
        actual = [arguments[index] for index in receipt.argument_indices]
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected, strict=True):
            self.assertIs(left, right)
        receipt.check()
        direct, no_dispatch = resolve_warmed_kernel(winner)
        self.assertIs(direct, winner)
        self.assertIsNone(no_dispatch)

    @parametrize("changed", ("winner", "argument_map", "child"))
    def test_changed_selection_declines(self, changed):
        carrier = self.carrier()
        carrier.picked_kernel = 1
        _, receipt = resolve_warmed_kernel(carrier)
        original_indices = receipt.argument_indices
        if changed == "winner":
            carrier.picked_kernel = 0
        elif changed == "argument_map":
            carrier.arg_index[1][-1] = slice(4, 7)
        else:
            carrier._kernels[1] = CachingAutotuner.__new__(CachingAutotuner)
        with self.assertRaisesRegex(ValueError, "selection changed"):
            receipt.check()
        self.assertEqual(receipt.argument_indices, original_indices)

    def test_unselected_and_shape_dispatch_decline(self):
        with self.assertRaisesRegex(ValueError, "fixed-choice MultiKernelCall"):
            resolve_warmed_kernel(self.carrier())
        with self.assertRaisesRegex(ValueError, "fixed-choice MultiKernelCall"):
            resolve_warmed_kernel(SizeHintMultiKernelCall.__new__(SizeHintMultiKernelCall))


if __name__ == "__main__":
    run_tests()
