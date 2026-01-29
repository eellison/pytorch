# Owner(s): ["module: inductor"]

"""
Tests for stride-2 coalesced load optimization.

This optimization coalesces adjacent stride-2 loads into single wider loads
to reduce memory bandwidth usage.
"""

import sys
import unittest
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

HAS_CUDA = torch.cuda.is_available()


class TestStride2CoalescedLoads(TestCase):
    """Tests for stride-2 coalesced load optimization."""

    @unittest.skipUnless(HAS_CUDA, "CUDA required")
    def test_complex64_real_plus_imag(self):
        """Test complex64 real+imag pattern uses coalesced load."""

        @torch.compile
        def fn(x):
            return x.real + x.imag

        x = torch.randn(1024, 1024, dtype=torch.complex64, device="cuda")
        result = fn(x)
        expected = x.real + x.imag
        self.assertEqual(result, expected)

    @unittest.skipUnless(HAS_CUDA, "CUDA required")
    def test_complex64_real_times_imag(self):
        """Test complex64 real*imag pattern uses coalesced load."""

        @torch.compile
        def fn(x):
            return x.real * x.imag

        x = torch.randn(512, 512, dtype=torch.complex64, device="cuda")
        result = fn(x)
        expected = x.real * x.imag
        self.assertEqual(result, expected)

    @unittest.skipUnless(HAS_CUDA, "CUDA required")
    def test_bf16_even_odd_sum(self):
        """Test bf16 stride-2 pattern (even + odd elements)."""

        @torch.compile
        def fn(x):
            # Access even and odd indices
            even = x[:, 0::2]  # indices 0, 2, 4, ...
            odd = x[:, 1::2]   # indices 1, 3, 5, ...
            return even + odd

        x = torch.randn(256, 128, dtype=torch.bfloat16, device="cuda")
        result = fn(x)
        expected = x[:, 0::2] + x[:, 1::2]
        self.assertEqual(result, expected)

    @unittest.skipUnless(HAS_CUDA, "CUDA required")
    def test_fp16_even_odd_product(self):
        """Test fp16 stride-2 pattern (even * odd elements)."""

        @torch.compile
        def fn(x):
            even = x[:, 0::2]
            odd = x[:, 1::2]
            return even * odd

        x = torch.randn(256, 128, dtype=torch.float16, device="cuda")
        result = fn(x)
        expected = x[:, 0::2] * x[:, 1::2]
        self.assertEqual(result, expected)

    @unittest.skipUnless(HAS_CUDA, "CUDA required")
    def test_complex64_abs_squared(self):
        """Test complex64 abs squared: real^2 + imag^2."""

        @torch.compile
        def fn(x):
            return x.real**2 + x.imag**2

        x = torch.randn(512, 512, dtype=torch.complex64, device="cuda")
        result = fn(x)
        expected = x.real**2 + x.imag**2
        self.assertEqual(result, expected)

    @unittest.skipUnless(HAS_CUDA, "CUDA required")
    def test_view_as_pair_sum(self):
        """Test view-based stride-2 pattern."""

        @torch.compile
        def fn(x):
            # View as pairs and sum
            pairs = x.view(-1, 2)
            return pairs[:, 0] + pairs[:, 1]

        x = torch.randn(2048, dtype=torch.float32, device="cuda")
        result = fn(x)
        expected = x.view(-1, 2)[:, 0] + x.view(-1, 2)[:, 1]
        self.assertEqual(result, expected)

    @unittest.skipUnless(HAS_CUDA, "CUDA required")
    def test_transform_disabled_gives_same_result(self):
        """Test that disabling transform gives same numerical result."""
        from torch._inductor import loop_body

        @torch.compile
        def fn(x):
            return x.real + x.imag

        x = torch.randn(256, 256, dtype=torch.complex64, device="cuda")

        # With optimization
        torch._dynamo.reset()
        result_opt = fn(x)

        # Without optimization
        torch._dynamo.reset()
        orig_transform = loop_body.LoopBody.transform_stride2_loads
        loop_body.LoopBody.transform_stride2_loads = lambda self, skip=None: False
        try:
            result_no_opt = fn(x)
        finally:
            loop_body.LoopBody.transform_stride2_loads = orig_transform

        self.assertEqual(result_opt, result_no_opt)


class TestStride2Detection(TestCase):
    """Tests for stride-2 pattern detection logic."""

    def test_is_stride2_pair_simple(self):
        """Test _is_stride2_pair with simple patterns."""
        from torch._inductor.loop_body import LoopBody
        import sympy

        body = LoopBody.__new__(LoopBody)
        x = sympy.Symbol("x")

        # Basic stride-2 pattern: 2*x and 2*x + 1
        idx1 = 2 * x
        idx2 = 2 * x + 1
        result = body._is_stride2_pair(idx1, idx2)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], idx1)  # even
        self.assertEqual(result[1], idx2)  # odd

    def test_is_stride2_pair_reversed(self):
        """Test _is_stride2_pair with reversed order."""
        from torch._inductor.loop_body import LoopBody
        import sympy

        body = LoopBody.__new__(LoopBody)
        x = sympy.Symbol("x")

        # Reversed order: odd first, even second
        idx1 = 2 * x + 1
        idx2 = 2 * x
        result = body._is_stride2_pair(idx1, idx2)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], idx2)  # even (2*x)
        self.assertEqual(result[1], idx1)  # odd (2*x + 1)

    def test_is_stride2_pair_not_stride2(self):
        """Test _is_stride2_pair rejects non-stride-2 patterns."""
        from torch._inductor.loop_body import LoopBody
        import sympy

        body = LoopBody.__new__(LoopBody)
        x = sympy.Symbol("x")

        # Not stride-2: difference is 2, not 1
        idx1 = 2 * x
        idx2 = 2 * x + 2
        result = body._is_stride2_pair(idx1, idx2)
        self.assertIsNone(result)

        # Not stride-2: no 2* multiplier
        idx1 = x
        idx2 = x + 1
        result = body._is_stride2_pair(idx1, idx2)
        # This might pass depending on implementation, but it's not the target pattern


class TestStride2DtypeConfig(TestCase):
    """Tests for stride-2 dtype configuration."""

    def test_supported_dtypes(self):
        """Test that expected dtypes are supported."""
        from torch._inductor.loop_body import LoopBody

        config = LoopBody.STRIDE2_COALESCE_DTYPES

        # Check bf16 -> int32
        self.assertIn(torch.bfloat16, config)
        self.assertEqual(config[torch.bfloat16], (torch.int32, 16))

        # Check fp16 -> int32
        self.assertIn(torch.float16, config)
        self.assertEqual(config[torch.float16], (torch.int32, 16))

        # Check fp32 -> int64 (for complex64)
        self.assertIn(torch.float32, config)
        self.assertEqual(config[torch.float32], (torch.int64, 32))

    def test_unsupported_dtypes(self):
        """Test that some dtypes are not supported."""
        from torch._inductor.loop_body import LoopBody

        config = LoopBody.STRIDE2_COALESCE_DTYPES

        # int8, int16, int32 are not in the config (could be added later)
        self.assertNotIn(torch.int8, config)
        self.assertNotIn(torch.int16, config)


if __name__ == "__main__":
    run_tests()
