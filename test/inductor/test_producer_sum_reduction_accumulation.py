# Owner(s): ["module: inductor"]

import unittest

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.utils import counters
from torch._inductor import metrics
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


class SumSumSum6107A2F54029(torch.nn.Module):
    def forward(
        self,
        mm_193,
        fmod_2,
        primals_20,
        mul_10,
        div_120,
        view_1358,
        shape_0,
        shape_1,
        shape_2,
        shape_3,
        shape_4,
        shape_5,
        shape_6,
    ):
        reshape_default = torch.ops.aten.reshape.default(mm_193, shape_0)
        reshape_default_1 = torch.ops.aten.reshape.default(reshape_default, shape_1)
        reshape_default_2 = torch.ops.aten.reshape.default(reshape_default_1, shape_2)
        permute_default = torch.ops.aten.permute.default(
            reshape_default_2, [0, 1, 3, 2, 4, 5]
        )
        clone_default = torch.ops.aten.clone.default(
            permute_default, memory_format=torch.contiguous_format
        )
        reshape_default_3 = torch.ops.aten.reshape.default(clone_default, shape_3)
        index_tensor = torch.ops.aten.index.Tensor(
            reshape_default_3, [None, None, fmod_2]
        )
        index_tensor_1 = torch.ops.aten.index.Tensor(index_tensor, [None, fmod_2])
        mul_tensor = torch.ops.aten.mul.Tensor(index_tensor_1, primals_20)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 128)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, mul_10)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_10, sum_dim_int_list_1)
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3)
        mul_tensor_4 = torch.ops.aten.mul.Tensor(div_120, sub_tensor_1)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(index_tensor_1, mul_10)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1, 2])
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(index_tensor_1, [0, 1, 2])
        add_tensor = torch.ops.aten.add.Tensor(view_1358, mul_tensor_4)
        reshape_default_4 = torch.ops.aten.reshape.default(add_tensor, shape_4)
        reshape_default_5 = torch.ops.aten.reshape.default(reshape_default_4, shape_5)
        permute_default_1 = torch.ops.aten.permute.default(reshape_default_5, [1, 0])
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(
            reshape_default_5, [0], True
        )
        reshape_default_6 = torch.ops.aten.reshape.default(sum_dim_int_list_4, shape_6)
        return (
            sum_dim_int_list_2,
            sum_dim_int_list_3,
            permute_default_1,
            reshape_default_6,
        )


class SumSumSumDC96C4651516(SumSumSum6107A2F54029):
    pass


class SumSumSum3213336F4C0B(torch.nn.Module):
    def forward(
        self,
        mm_127,
        mm_129,
        mm_131,
        mul_279,
        arg8_1,
        arg109_1,
        arg297_1,
        arg108_1,
        shape_0,
        shape_1,
        shape_2,
        shape_3,
        shape_4,
    ):
        view_default = torch.ops.aten.view.default(mm_127, shape_0)
        view_default_1 = torch.ops.aten.view.default(mm_129, shape_1)
        add_tensor = torch.ops.aten.add.Tensor(view_default, view_default_1)
        view_default_2 = torch.ops.aten.view.default(mm_131, shape_2)
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default_2)
        permute_default = torch.ops.aten.permute.default(add_tensor_1, [1, 0, 2])
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_279, permute_default)
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor_2, arg8_1)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, arg109_1)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(
            mul_tensor_2, [2], True
        )
        mul_tensor_3 = torch.ops.aten.mul.Tensor(arg109_1, sum_dim_int_list_1)
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3)
        mul_tensor_4 = torch.ops.aten.mul.Tensor(arg297_1, sub_tensor_1)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_2, arg109_1)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1])
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1])
        convert_element_type_default = torch.ops.prims.convert_element_type.default(
            arg108_1, torch.float32
        )
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0)
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_6)
        view_default_3 = torch.ops.aten.view.default(mul_tensor_7, shape_3)
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [1, 0])
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(
            view_default_3, [0], True
        )
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list_4, shape_4)
        return (
            sum_dim_int_list_2,
            sum_dim_int_list_3,
            permute_default_1,
            view_default_4,
        )


class SumSumSumBaf315Cfc5F0(torch.nn.Module):
    def forward(
        self,
        mm_273,
        mm_275,
        mm_277,
        arg11_1,
        arg231_1,
        arg537_1,
        add_181,
        shape_0,
        shape_1,
        shape_2,
        shape_3,
        shape_4,
    ):
        view_default = torch.ops.aten.view.default(mm_273, shape_0)
        view_default_1 = torch.ops.aten.view.default(mm_275, shape_1)
        add_tensor = torch.ops.aten.add.Tensor(view_default, view_default_1)
        view_default_2 = torch.ops.aten.view.default(mm_277, shape_2)
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default_2)
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor_1, arg11_1)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 2048)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, arg231_1)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(
            mul_tensor_2, [2], True
        )
        mul_tensor_3 = torch.ops.aten.mul.Tensor(arg231_1, sum_dim_int_list_1)
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3)
        mul_tensor_4 = torch.ops.aten.mul.Tensor(arg537_1, sub_tensor_1)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_1, arg231_1)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1])
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1])
        add_tensor_2 = torch.ops.aten.add.Tensor(add_181, mul_tensor_4)
        view_default_3 = torch.ops.aten.view.default(add_tensor_2, shape_3)
        permute_default = torch.ops.aten.permute.default(view_default_3, [1, 0])
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(
            view_default_3, [0], True
        )
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list_4, shape_4)
        return (
            sum_dim_int_list_2,
            sum_dim_int_list_3,
            permute_default,
            view_default_4,
        )


@unittest.skipIf(not HAS_GPU, "requires GPU")
@inductor_config.patch(
    {
        "producer_sum_reduction_accumulation": True,
        "triton.cooperative_reductions": False,
        "triton.force_cooperative_reductions": False,
    }
)
class ProducerSumReductionAccumulationTest(TestCase):
    def setUp(self):
        super().setUp()
        counters.clear()
        metrics.reset()
        torch._dynamo.reset()

    def assert_inductor_counters(self, expected):
        actual = counters["inductor"]
        for name, value in expected.items():
            self.assertEqual(actual[name], value, f"{name}: {dict(actual)}")

    def assert_outputs_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_output, expected_output in zip(actual, expected):
            self.assertEqual(actual_output.shape, expected_output.shape)
            torch.testing.assert_close(
                actual_output,
                expected_output,
                rtol=1e-3,
                atol=1e-1,
            )

    @largeTensorTest("1GB", device=GPU_TYPE, inductor=True)
    def test_sum_sum_sum_baf315cfc5f0_rejects_below_min_bytes(self):
        args = (
            torch.randn(4096, 2048, device=GPU_TYPE),
            torch.randn(4096, 2048, device=GPU_TYPE),
            torch.randn(4096, 2048, device=GPU_TYPE),
            torch.randn(2048, device=GPU_TYPE),
            torch.randn(32, 128, 2048, device=GPU_TYPE),
            torch.randn(32, 128, 1, device=GPU_TYPE),
            torch.randn(32, 128, 2048, device=GPU_TYPE),
            [32, 128, 2048],
            [32, 128, 2048],
            [32, 128, 2048],
            [4096, 2048],
            [2048],
        )

        mod = SumSumSumBaf315Cfc5F0()
        expected = mod(*args)
        actual = torch.compile(mod)(*args)

        self.assert_outputs_close(actual, expected)
        self.assert_inductor_counters(
            {
                "producer_sum_reduction_accumulation_candidates": 0,
                "producer_sum_reduction_accumulation_selected": 0,
                "producer_sum_reduction_accumulation_codegen": 0,
            }
        )

    @largeTensorTest("1GB", device=GPU_TYPE, inductor=True)
    def test_sum_sum_sum_3213336f4c0b_rejects_workspace_overhead(self):
        args = (
            torch.randn(8192, 768, device=GPU_TYPE),
            torch.randn(8192, 768, device=GPU_TYPE),
            torch.randn(8192, 768, device=GPU_TYPE),
            torch.randn(8, 1024, 768, device=GPU_TYPE),
            torch.randn(768, device=GPU_TYPE),
            torch.randn(8, 1024, 768, device=GPU_TYPE),
            torch.randn(8, 1024, 1, device=GPU_TYPE),
            torch.randint(0, 2, (8, 1024, 768), device=GPU_TYPE, dtype=torch.bool),
            [1024, 8, 768],
            [1024, 8, 768],
            [1024, 8, 768],
            [8192, 768],
            [768],
        )

        mod = SumSumSum3213336F4C0B()
        expected = mod(*args)
        actual = torch.compile(mod)(*args)

        self.assert_outputs_close(actual, expected)
        self.assert_inductor_counters(
            {
                "producer_sum_reduction_accumulation_candidates": 0,
                "producer_sum_reduction_accumulation_selected": 0,
                "producer_sum_reduction_accumulation_codegen": 0,
            }
        )

    @largeTensorTest("8GB", device=GPU_TYPE, inductor=True)
    def check_swin_extra_saved_partials(self, mod):
        args = (
            torch.randn(401408, 128, device=GPU_TYPE),
            torch.arange(56, device=GPU_TYPE),
            torch.randn(128, device=GPU_TYPE),
            torch.randn(128, 56, 56, 128, device=GPU_TYPE),
            torch.randn(128, 56, 56, 1, device=GPU_TYPE),
            torch.randn(128, 56, 56, 128, device=GPU_TYPE),
            [8192, 49, 128],
            [8192, 7, 7, 128],
            [128, 8, 8, 7, 7, 128],
            [128, 56, 56, 128],
            [128, 3136, 128],
            [401408, 128],
            [128],
        )

        expected = mod(*args)
        actual = torch.compile(mod)(*args)

        self.assert_outputs_close(actual, expected)
        self.assertEqual(metrics.generated_kernel_count, 1)
        self.assert_inductor_counters(
            {
                "producer_sum_reduction_accumulation_candidates": 1,
                "producer_sum_reduction_accumulation_selected": 1,
                "producer_sum_reduction_accumulation_extra_reductions": 1,
                "producer_sum_reduction_accumulation_codegen": 1,
                "producer_sum_reduction_accumulation_reject_domain_mismatch": 0,
                "producer_sum_reduction_accumulation_reject_subnode_domain_mismatch": 0,
                "producer_sum_reduction_accumulation_reject_not_mix_order": 0,
            }
        )

    @largeTensorTest("8GB", device=GPU_TYPE, inductor=True)
    def test_sum_sum_sum_6107a2f54029_selects_extra_saved_partials(self):
        self.check_swin_extra_saved_partials(SumSumSum6107A2F54029())

    @largeTensorTest("8GB", device=GPU_TYPE, inductor=True)
    def test_sum_sum_sum_dc96c4651516_selects_extra_saved_partials(self):
        self.check_swin_extra_saved_partials(SumSumSumDC96C4651516())


if __name__ == "__main__":
    run_tests()
