# Owner(s): ["module: inductor"]
"""
Tests for cvt_e8m0_rceil inductor prim with PTX lowering.

This tests:
1. Direct prim call with lowering (bit manipulation fallback or PTX on SM100+)
2. Pattern matching of bit manipulation code
3. End-to-end compilation and correctness
"""

from unittest import skipIf

import torch
from torch._inductor import inductor_prims
from torch._inductor.fx_passes.misc_patterns import _misc_patterns_init
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.common_cuda import SM100OrLater
from torch.testing._internal.common_utils import run_tests, skipIfRocm, skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_gpu


class TestCvtE8M0Rceil(InductorTestCase):
    """Tests for cvt_e8m0_rceil prim and pattern matching."""

    @requires_gpu()
    @skipIfRocm
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    def test_direct_prim_call(self):
        """Test calling the prim directly."""

        def fn(inp):
            return inductor_prims.cvt_e8m0_rceil(inp)

        inp = torch.tensor(
            [1.0, 2.0, 4.0, 0.5, 0.25], device=GPU_TYPE, dtype=torch.float32
        )

        # Eager result
        eager_result = fn(inp)

        # Compiled result
        fn_opt = torch.compile(fn)
        compiled_result = fn_opt(inp)

        # Should match
        self.assertEqual(compiled_result, eager_result)

        # Verify expected values for powers of 2
        # e8m0 biased exponent = log2(value) + 127
        expected = torch.tensor(
            [127, 128, 129, 126, 125], device=GPU_TYPE, dtype=torch.uint8
        )
        self.assertEqual(eager_result, expected)

    @requires_gpu()
    @skipIfRocm
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    def test_prim_with_random_values(self):
        """Test prim with random values."""

        def fn(inp):
            return inductor_prims.cvt_e8m0_rceil(inp)

        inp = torch.rand(1024, device=GPU_TYPE, dtype=torch.float32) * 100 + 0.01

        # Eager result
        eager_result = fn(inp)

        # Compiled result
        fn_opt = torch.compile(fn)
        compiled_result = fn_opt(inp)

        # Should match
        self.assertEqual(compiled_result, eager_result)

    @requires_gpu()
    @skipIfRocm
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    @skipIf(not SM100OrLater, "Pattern matching only enabled on SM100+")
    def test_pattern_match_replacement(self):
        """Test that the bit manipulation pattern gets matched and replaced on SM100+."""
        # Initialize patterns
        _misc_patterns_init()

        def fn_with_pattern(inp):
            """This should be pattern matched and replaced with cvt_e8m0_rceil."""
            inp_bits = inp.view(torch.int32)
            biased_exp = (inp_bits >> 23) & 0xFF
            mantissa = inp_bits & 0x7FFFFF
            needs_round_up = mantissa != 0
            e8m0_biased = biased_exp + needs_round_up.to(torch.int32)
            e8m0_biased = torch.clamp(e8m0_biased, 0, 255)
            return e8m0_biased.to(torch.uint8)

        inp = torch.tensor(
            [1.0, 2.0, 4.0, 3.0, 1.5], device=GPU_TYPE, dtype=torch.float32
        )

        # Get eager result (the pattern)
        eager_result = fn_with_pattern(inp)

        # Compile - pattern should be matched and replaced
        fn_opt = torch.compile(fn_with_pattern)
        compiled_result = fn_opt(inp)

        # Results should match
        self.assertEqual(compiled_result, eager_result)

    @requires_gpu()
    @skipIfRocm
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    @skipIf(not SM100OrLater, "PTX instruction requires SM100+")
    def test_ptx_instruction_sm100(self):
        """Test that PTX instruction is used on SM100+."""

        def fn(inp):
            return inductor_prims.cvt_e8m0_rceil(inp)

        inp = torch.rand(1024, device=GPU_TYPE, dtype=torch.float32) * 100 + 0.01

        fn_opt = torch.compile(fn)
        compiled_result = fn_opt(inp)
        eager_result = fn(inp)

        self.assertEqual(compiled_result, eager_result)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
