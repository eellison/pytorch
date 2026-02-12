# Owner(s): ["module: inductor"]

import unittest

import torch
import torch._inductor.config as inductor_config
import torch._inductor.metrics as inductor_metrics
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_GPU_AND_TRITON


class TestReductionEpilogueFusion(TestCase):
    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @inductor_config.patch(
        {
            "triton.small_reduction_epilogue": True,
            "triton.small_reduction_epilogue_fusion": True,
        }
    )
    def test_loop_based_small_reduction_epilogue(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            y = x.amax(dim=-1)
            return y.reshape(4, 16).amax(dim=-1)

        x = torch.randn(64, 4096, device="cuda")
        expected = x.amax(dim=-1).reshape(4, 16).amax(dim=-1)

        inductor_metrics.generated_kernel_count = 0
        actual = fn(x)
        self.assertEqual(actual, expected)
        self.assertEqual(inductor_metrics.generated_kernel_count, 1)

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @inductor_config.patch(
        {
            "triton.small_reduction_epilogue": True,
            "triton.small_reduction_epilogue_fusion": True,
        }
    )
    def test_loop_based_small_reduction_keepdim_store(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            y = x.amax(dim=-1)
            # Vary final store shape/layout by keeping reduced dim.
            return y.reshape(4, 16).amax(dim=-1, keepdim=True)

        x = torch.randn(64, 4096, device="cuda")
        expected = x.amax(dim=-1).reshape(4, 16).amax(dim=-1, keepdim=True)
        actual = fn(x)
        self.assertEqual(actual, expected)

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @inductor_config.patch(
        {
            "triton.small_reduction_epilogue": True,
            "triton.small_reduction_epilogue_fusion": True,
        }
    )
    def test_loop_based_small_reduction_dynamic_shapes(self):
        @torch.compile(fullgraph=True, dynamic=True)
        def fn(x):
            groups = x.shape[0] // 16
            y = x.amax(dim=-1)
            return y.reshape(groups, 16).amax(dim=-1)

        x0 = torch.randn(64, 4096, device="cuda")
        x1 = torch.randn(128, 4096, device="cuda")

        expected0 = x0.amax(dim=-1).reshape(4, 16).amax(dim=-1)
        expected1 = x1.amax(dim=-1).reshape(8, 16).amax(dim=-1)

        actual0 = fn(x0)
        actual1 = fn(x1)
        self.assertEqual(actual0, expected0)
        self.assertEqual(actual1, expected1)

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @inductor_config.patch(
        {
            "triton.small_reduction_epilogue": True,
            "triton.small_reduction_epilogue_fusion": True,
        }
    )
    def test_in_register_candidate_falls_back_safely(self):
        @torch.compile(fullgraph=True)
        def fn(x, weight, bias):
            y = torch.nn.functional.layer_norm(x, [4096], weight, bias)
            return y.reshape(64, 256, 16).abs().amax(dim=-1)

        x = torch.randn(4, 16, 4096, device="cuda")
        weight = torch.randn(4096, device="cuda")
        bias = torch.randn(4096, device="cuda")

        expected = (
            torch.nn.functional.layer_norm(x, [4096], weight, bias)
            .reshape(64, 256, 16)
            .abs()
            .amax(dim=-1)
        )
        actual = fn(x, weight, bias)
        self.assertEqual(actual, expected)

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @inductor_config.patch(
        {
            "triton.small_reduction_epilogue": True,
            "triton.small_reduction_epilogue_fusion": True,
        }
    )
    def test_in_register_reduction_epilogue(self):
        """In-register epilogue: LN output is returned (materialized) so a
        full-size intermediate buffer exists for the grouped amax."""

        @torch.compile(fullgraph=True)
        def fn(x, weight, bias):
            y = torch.nn.functional.layer_norm(x, [4096], weight, bias)
            z = y.reshape(64, 256, 16).abs().amax(dim=-1)
            return y, z

        x = torch.randn(4, 16, 4096, device="cuda")
        weight = torch.randn(4096, device="cuda")
        bias = torch.randn(4096, device="cuda")

        y_ref = torch.nn.functional.layer_norm(x, [4096], weight, bias)
        z_ref = y_ref.reshape(64, 256, 16).abs().amax(dim=-1)

        y_actual, z_actual = fn(x, weight, bias)
        self.assertEqual(y_actual, y_ref)
        self.assertEqual(z_actual, z_ref)

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @inductor_config.patch(
        {
            "triton.small_reduction_epilogue": True,
            "triton.small_reduction_epilogue_fusion": True,
        }
    )
    def test_in_register_reduction_broadcast_back(self):
        """NVFP4-style pattern: LN -> abs -> amax -> broadcast scale -> divide.
        The reduction result is broadcast back and used for division, all in
        one kernel via the in-register epilogue."""

        @torch.compile(fullgraph=True)
        def fn(x, weight, bias):
            y = torch.nn.functional.layer_norm(x, [4096], weight, bias)
            y_grouped = y.reshape(64, 256, 16)
            scale = y_grouped.abs().amax(dim=-1, keepdim=True)
            y_scaled = y_grouped / scale
            return y_scaled.reshape(64, 4096), scale.squeeze(-1)

        x = torch.randn(4, 16, 4096, device="cuda")
        weight = torch.randn(4096, device="cuda")
        bias = torch.randn(4096, device="cuda")

        y_ref = torch.nn.functional.layer_norm(x, [4096], weight, bias)
        y_grouped_ref = y_ref.reshape(64, 256, 16)
        scale_ref = y_grouped_ref.abs().amax(dim=-1, keepdim=True)
        scaled_ref = (y_grouped_ref / scale_ref).reshape(64, 4096)
        scale_out_ref = scale_ref.squeeze(-1)

        actual_scaled, actual_scale = fn(x, weight, bias)
        self.assertEqual(actual_scaled, scaled_ref)
        self.assertEqual(actual_scale, scale_out_ref)


class TestBlockLocalReduction(TestCase):
    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @inductor_config.patch(
        {
            "triton.block_local_reduction": True,
        }
    )
    def test_ln_grouped_amax(self):
        """LayerNorm followed by grouped amax (NVFP4-like pattern)."""
        batch, hidden, group_size = 4, 256, 16
        weight = torch.randn(hidden, device="cuda")
        bias = torch.randn(hidden, device="cuda")

        @torch.compile(fullgraph=True)
        def fn(x):
            normed = torch.nn.functional.layer_norm(x, [hidden], weight, bias)
            return (
                normed.view(batch, hidden // group_size, group_size).abs().amax(dim=-1)
            )

        x = torch.randn(batch, hidden, device="cuda")
        self.assertEqual(fn(x), fn.__wrapped__(x))

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @inductor_config.patch(
        {
            "triton.block_local_reduction": True,
        }
    )
    def test_ln_grouped_amax_dynamic(self):
        """LayerNorm + grouped amax with dynamic shapes."""
        hidden, group_size = 256, 16
        weight = torch.randn(hidden, device="cuda")
        bias = torch.randn(hidden, device="cuda")

        @torch.compile(fullgraph=True, dynamic=True)
        def fn(x):
            normed = torch.nn.functional.layer_norm(x, [hidden], weight, bias)
            return normed.view(-1, hidden // group_size, group_size).abs().amax(dim=-1)

        for b in [4, 8, 2]:
            x = torch.randn(b, hidden, device="cuda")
            self.assertEqual(fn(x), fn.__wrapped__(x))

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @inductor_config.patch(
        {
            "triton.block_local_reduction": True,
        }
    )
    def test_ln_grouped_amax_group32(self):
        """Grouped amax with group_size=32."""
        batch, hidden, group_size = 4, 256, 32
        weight = torch.randn(hidden, device="cuda")
        bias = torch.randn(hidden, device="cuda")

        @torch.compile(fullgraph=True)
        def fn(x):
            normed = torch.nn.functional.layer_norm(x, [hidden], weight, bias)
            return (
                normed.view(batch, hidden // group_size, group_size).abs().amax(dim=-1)
            )

        x = torch.randn(batch, hidden, device="cuda")
        self.assertEqual(fn(x), fn.__wrapped__(x))

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @inductor_config.patch(
        {
            "triton.block_local_reduction": True,
        }
    )
    def test_ln_grouped_sum(self):
        """Grouped sum reduction (not amax)."""
        batch, hidden, group_size = 4, 256, 16
        weight = torch.randn(hidden, device="cuda")
        bias = torch.randn(hidden, device="cuda")

        @torch.compile(fullgraph=True)
        def fn(x):
            normed = torch.nn.functional.layer_norm(x, [hidden], weight, bias)
            return normed.view(batch, hidden // group_size, group_size).sum(dim=-1)

        x = torch.randn(batch, hidden, device="cuda")
        self.assertEqual(fn(x), fn.__wrapped__(x), atol=1e-4, rtol=1e-4)

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @inductor_config.patch(
        {
            "triton.block_local_reduction": True,
        }
    )
    def test_ln_grouped_amax_keepdim(self):
        """Amax with keepdim=True (size-1 dim in child ranges)."""
        batch, hidden, group_size = 4, 256, 16
        weight = torch.randn(hidden, device="cuda")
        bias = torch.randn(hidden, device="cuda")

        @torch.compile(fullgraph=True)
        def fn(x):
            normed = torch.nn.functional.layer_norm(x, [hidden], weight, bias)
            return (
                normed.view(batch, hidden // group_size, group_size)
                .abs()
                .amax(dim=-1, keepdim=True)
            )

        x = torch.randn(batch, hidden, device="cuda")
        self.assertEqual(fn(x), fn.__wrapped__(x), atol=1e-4, rtol=1e-4)

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @inductor_config.patch(
        {
            "triton.block_local_reduction": True,
        }
    )
    def test_ln_amax_broadcast_divide(self):
        """NVFP4 pattern: LN -> grouped amax -> broadcast back -> divide."""
        batch, hidden, group_size = 4, 256, 16
        weight = torch.randn(hidden, device="cuda")
        bias = torch.randn(hidden, device="cuda")

        @torch.compile(fullgraph=True)
        def fn(x):
            normed = torch.nn.functional.layer_norm(x, [hidden], weight, bias)
            reshaped = normed.view(batch, hidden // group_size, group_size)
            scale = reshaped.abs().amax(dim=-1, keepdim=True)
            return (reshaped / scale.clamp(min=1e-12)).view(batch, hidden)

        x = torch.randn(batch, hidden, device="cuda")
        self.assertEqual(fn(x), fn.__wrapped__(x), atol=1e-4, rtol=1e-4)

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @inductor_config.patch(
        {
            "triton.block_local_reduction": True,
        }
    )
    def test_simple_sum_amax(self):
        """across-X pattern: sum reduction followed by grouped amax."""

        @torch.compile(fullgraph=True)
        def fn(x):
            return x.sum(dim=-1).view(4, 16, 16).abs().amax(dim=-1)

        x = torch.randn(4, 256, 32, device="cuda")
        self.assertEqual(fn(x), fn.__wrapped__(x))

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @inductor_config.patch(
        {
            "triton.block_local_reduction": True,
        }
    )
    def test_ln_amax_multiple_stores(self):
        """Block-local reduction output feeds multiple downstream stores."""
        batch, hidden, group_size = 4, 256, 16
        weight = torch.randn(hidden, device="cuda")
        bias = torch.randn(hidden, device="cuda")

        @torch.compile(fullgraph=True)
        def fn(x):
            normed = torch.nn.functional.layer_norm(x, [hidden], weight, bias)
            reshaped = normed.view(batch, hidden // group_size, group_size)
            amax = reshaped.abs().amax(dim=-1, keepdim=True)
            scale = amax.clamp(min=1e-12)
            scaled = normed / scale.view(batch, hidden // group_size).repeat_interleave(
                group_size, dim=-1
            )
            return normed, scaled, amax.squeeze(-1)

        x = torch.randn(batch, hidden, device="cuda")
        expected = fn.__wrapped__(x)
        actual = fn(x)
        for r, e in zip(actual, expected):
            self.assertEqual(r, e, atol=1e-4, rtol=1e-4)


    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @inductor_config.patch(
        {
            "triton.block_local_reduction": True,
        }
    )
    def test_intermediate_buf_external_reader(self):
        """Intermediate buffer (LN output) read by both the block-local child
        AND a separate downstream kernel. The intermediate must be stored to
        global memory so the external reader sees valid data."""
        batch, hidden, group_size = 4, 256, 16
        weight = torch.randn(hidden, device="cuda")
        bias = torch.randn(hidden, device="cuda")

        @torch.compile(fullgraph=True)
        def fn(x):
            normed = torch.nn.functional.layer_norm(x, [hidden], weight, bias)
            # Block-local path: normed -> reshape -> abs -> amax
            amax = normed.view(batch, hidden // group_size, group_size).abs().amax(
                dim=-1
            )
            # External reader: separate op on the same normed buffer
            normed_sum = normed.sum(dim=-1)
            return amax, normed_sum

        x = torch.randn(batch, hidden, device="cuda")
        expected = fn.__wrapped__(x)
        actual = fn(x)
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1], atol=1e-4, rtol=1e-4)


    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @inductor_config.patch(
        {
            "triton.block_local_reduction": True,
        }
    )
    def test_intermediate_buf_fused_external_reader(self):
        """Intermediate buffer read by block-local child AND a fused downstream
        kernel (pointwise chain). This tests that the intermediate buffer
        analysis correctly accounts for users across fusion boundaries."""
        batch, hidden, group_size = 4, 256, 16
        weight = torch.randn(hidden, device="cuda")
        bias = torch.randn(hidden, device="cuda")

        @torch.compile(fullgraph=True)
        def fn(x):
            normed = torch.nn.functional.layer_norm(x, [hidden], weight, bias)
            amax = normed.view(batch, hidden // group_size, group_size).abs().amax(
                dim=-1
            )
            # Pointwise chain on normed — likely fused into another kernel
            doubled = normed * 2.0
            shifted = doubled + 1.0
            return amax, shifted

        x = torch.randn(batch, hidden, device="cuda")
        expected = fn.__wrapped__(x)
        actual = fn(x)
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1], atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    run_tests()
