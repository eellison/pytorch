# Owner(s): ["module: inductor"]
"""
Tests for SmallReductionEpilogue fusion optimization.

This optimization fuses a large reduction followed by a smaller reduction
that operates on groups within the first reduction's output.

Example pattern:
    LayerNorm (reduce over 4096) -> GroupSum (reduce over 16 groups)

The fusion keeps intermediate data in registers between passes, avoiding
expensive global memory round-trips.
"""

import torch
import torch._inductor.config as inductor_config
import torch.nn as nn
from torch._dynamo.utils import same
from torch._inductor import metrics
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


class SmallReductionEpilogueTestBase(TestCase):
    def setUp(self):
        super().setUp()
        metrics.reset()
        torch._dynamo.reset()

    def check_numeric(self, f, args, tol=1e-4):
        """Check that compiled output matches eager output."""
        ref = f(*args)
        with inductor_config.patch({
            "aggressive_fusion": True,
            "triton.small_reduction_epilogue": True,
            "triton.small_reduction_epilogue_fusion": True
        }):
            act = torch.compile(f)(*args)
        self.assertTrue(same(ref, act, tol=tol), f"ref:\n{ref}\nact:\n{act}")


@instantiate_parametrized_tests
class SmallReductionEpilogueTest(SmallReductionEpilogueTestBase):
    """Test SmallReductionEpilogue fusion with various configurations."""

    @parametrize("batch_size", [4, 8, 16, 32])
    @parametrize("hidden_size", [64, 128, 256])
    @parametrize("group_size", [8, 16, 32, 64])
    def test_explicit_layer_norm_group_sum(self, batch_size, hidden_size, group_size):
        """Test fusion with explicit mean/var operations followed by group sum."""
        if hidden_size % group_size != 0:
            self.skipTest("hidden_size must be divisible by group_size")

        groups = hidden_size // group_size
        if groups == 1:
            self.skipTest("groups=1 (single group) not supported by fusion")

        def f(x, w, b):
            # Explicit layer norm
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            y = (x - mean) / (var + 1e-5).sqrt() * w + b
            # Group sum
            y_grouped = y.view(batch_size, groups, group_size)
            return y_grouped.sum(dim=-1)

        x = torch.randn(batch_size, hidden_size, device=GPU_TYPE)
        w = torch.randn(hidden_size, device=GPU_TYPE)
        b = torch.randn(hidden_size, device=GPU_TYPE)

        self.check_numeric(f, (x, w, b))

    @parametrize("batch_size", [4, 8, 16])
    @parametrize("hidden_size", [128, 256])
    def test_layer_norm_group_sum_inference(self, batch_size, hidden_size):
        """Test fusion with nn.functional.layer_norm in inference mode."""
        group_size = 16
        groups = hidden_size // group_size

        def f(x, w, b):
            y = torch.nn.functional.layer_norm(x, [hidden_size], w, b)
            y_grouped = y.view(batch_size, groups, group_size)
            return y_grouped.sum(dim=-1)

        ln = nn.LayerNorm(hidden_size).to(GPU_TYPE)
        x = torch.randn(batch_size, hidden_size, device=GPU_TYPE)

        with torch.inference_mode():
            ref = f(x, ln.weight, ln.bias)
            with inductor_config.patch({
                "aggressive_fusion": True,
                "triton.small_reduction_epilogue": True,
                "triton.small_reduction_epilogue_fusion": True
            }):
                act = torch.compile(f)(x, ln.weight, ln.bias)
            self.assertTrue(same(ref, act, tol=1e-4))

    def test_group_max(self):
        """Test fusion with group max instead of group sum."""
        batch_size, hidden_size, group_size = 4, 128, 16
        groups = hidden_size // group_size

        def f(x, w, b):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            y = (x - mean) / (var + 1e-5).sqrt() * w + b
            y_grouped = y.view(batch_size, groups, group_size)
            return y_grouped.max(dim=-1).values

        x = torch.randn(batch_size, hidden_size, device=GPU_TYPE)
        w = torch.randn(hidden_size, device=GPU_TYPE)
        b = torch.randn(hidden_size, device=GPU_TYPE)

        self.check_numeric(f, (x, w, b))

    def test_group_mean(self):
        """Test fusion with group mean instead of group sum."""
        batch_size, hidden_size, group_size = 4, 128, 16
        groups = hidden_size // group_size

        def f(x, w, b):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            y = (x - mean) / (var + 1e-5).sqrt() * w + b
            y_grouped = y.view(batch_size, groups, group_size)
            return y_grouped.mean(dim=-1)

        x = torch.randn(batch_size, hidden_size, device=GPU_TYPE)
        w = torch.randn(hidden_size, device=GPU_TYPE)
        b = torch.randn(hidden_size, device=GPU_TYPE)

        self.check_numeric(f, (x, w, b))

    def test_group_min(self):
        """Test fusion with group min instead of group sum."""
        batch_size, hidden_size, group_size = 4, 128, 16
        groups = hidden_size // group_size

        def f(x, w, b):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            y = (x - mean) / (var + 1e-5).sqrt() * w + b
            y_grouped = y.view(batch_size, groups, group_size)
            return y_grouped.min(dim=-1).values

        x = torch.randn(batch_size, hidden_size, device=GPU_TYPE)
        w = torch.randn(hidden_size, device=GPU_TYPE)
        b = torch.randn(hidden_size, device=GPU_TYPE)

        self.check_numeric(f, (x, w, b))

    def test_group_prod(self):
        """Test fusion with group prod instead of group sum."""
        batch_size, hidden_size, group_size = 4, 128, 16
        groups = hidden_size // group_size

        def f(x, w, b):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            y = (x - mean) / (var + 1e-5).sqrt() * w + b
            y_grouped = y.view(batch_size, groups, group_size)
            return y_grouped.prod(dim=-1)

        x = torch.randn(batch_size, hidden_size, device=GPU_TYPE)
        w = torch.randn(hidden_size, device=GPU_TYPE)
        b = torch.randn(hidden_size, device=GPU_TYPE)

        # Use larger tolerance for prod since errors compound
        self.check_numeric(f, (x, w, b), tol=1e-3)

    def test_transpose_contiguous(self):
        """Test fusion output followed by transpose and contiguous."""
        batch_size, hidden_size, group_size = 4, 128, 16
        groups = hidden_size // group_size

        def f(x, w, b):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            y = (x - mean) / (var + 1e-5).sqrt() * w + b
            y_grouped = y.view(batch_size, groups, group_size)
            result = y_grouped.sum(dim=-1)
            return result.T.contiguous()

        x = torch.randn(batch_size, hidden_size, device=GPU_TYPE)
        w = torch.randn(hidden_size, device=GPU_TYPE)
        b = torch.randn(hidden_size, device=GPU_TYPE)

        self.check_numeric(f, (x, w, b))


@instantiate_parametrized_tests
class SmallReductionEpilogueEdgeCasesTest(SmallReductionEpilogueTestBase):
    """Test edge cases for SmallReductionEpilogue."""

    def test_single_batch_fallback(self):
        """Test that B=1 falls back to non-fused path (scalar handling issues)."""
        batch_size, hidden_size, group_size = 1, 128, 16
        groups = hidden_size // group_size

        def f(x, w, b):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            y = (x - mean) / (var + 1e-5).sqrt() * w + b
            y_grouped = y.view(batch_size, groups, group_size)
            return y_grouped.sum(dim=-1)

        x = torch.randn(batch_size, hidden_size, device=GPU_TYPE)
        w = torch.randn(hidden_size, device=GPU_TYPE)
        b = torch.randn(hidden_size, device=GPU_TYPE)

        # Should still produce correct results even if fusion is skipped
        self.check_numeric(f, (x, w, b))

    def test_large_hidden_dim(self):
        """Test with larger hidden dimensions."""
        batch_size, hidden_size, group_size = 4, 4096, 16
        groups = hidden_size // group_size

        def f(x, w, b):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            y = (x - mean) / (var + 1e-5).sqrt() * w + b
            y_grouped = y.view(batch_size, groups, group_size)
            return y_grouped.sum(dim=-1)

        x = torch.randn(batch_size, hidden_size, device=GPU_TYPE)
        w = torch.randn(hidden_size, device=GPU_TYPE)
        b = torch.randn(hidden_size, device=GPU_TYPE)

        self.check_numeric(f, (x, w, b))

    def test_non_power_of_two_groups(self):
        """Test with non-power-of-2 number of groups (should fall back)."""
        batch_size, hidden_size = 4, 120
        group_size = 10  # Not power of 2
        groups = hidden_size // group_size  # 12 groups

        def f(x, w, b):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            y = (x - mean) / (var + 1e-5).sqrt() * w + b
            y_grouped = y.view(batch_size, groups, group_size)
            return y_grouped.sum(dim=-1)

        x = torch.randn(batch_size, hidden_size, device=GPU_TYPE)
        w = torch.randn(hidden_size, device=GPU_TYPE)
        b = torch.randn(hidden_size, device=GPU_TYPE)

        # Should still produce correct results
        self.check_numeric(f, (x, w, b))


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
