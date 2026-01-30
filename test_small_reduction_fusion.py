"""
Test small reduction epilogue fusion.

This tests the infrastructure for detecting and handling the pattern:
  reduction1 (e.g., amax over 4096) -> reduction2 (e.g., amax over 16)

The goal is to fuse these into a single kernel to avoid intermediate buffer.
"""

import torch
import torch._inductor.config as config

# Enable the small reduction features
config.triton.small_reduction_epilogue = True
config.triton.small_reduction_epilogue_fusion = True


def test_amax_chain():
    """Test amax(4096) -> amax(16) fusion pattern."""

    @torch.compile(fullgraph=True)
    def fn(x):
        # x: [4, 16, 4096]
        # First reduction: amax over last dim -> [4, 16]
        y = x.amax(dim=-1)
        # Reshape for second reduction: [4, 16] -> [4, 16]
        # Second reduction: amax over groups of 16 -> [4]
        z = y.reshape(4, 16).amax(dim=-1)
        return z

    # Input: 4 groups of 16 rows, each row has 4096 elements
    x = torch.randn(4, 16, 4096, device="cuda")

    # Run the model
    result = fn(x)

    # Verify correctness
    expected = x.amax(dim=-1).reshape(4, 16).amax(dim=-1)
    torch.testing.assert_close(result, expected)

    print("test_amax_chain passed!")
    return result


def test_layernorm_amax():
    """
    Test LayerNorm -> amax(16) fusion pattern.

    This is the NVFP4 quantization pattern:
    - LayerNorm normalizes over 4096 elements
    - amax computes scale factors over groups of 16
    """

    @torch.compile(fullgraph=True)
    def fn(x, weight, bias):
        # x: [4, 16, 4096]
        # LayerNorm over last dim -> [4, 16, 4096]
        y = torch.nn.functional.layer_norm(x, [4096], weight, bias)
        # amax over groups of 16 rows for scaling
        y_reshaped = y.reshape(64, 4096)  # [64, 4096]
        # Take amax of abs values for each row -> [64]
        amax_per_row = y_reshaped.abs().amax(dim=-1)
        # Group into 16s and take amax -> [4]
        amax_per_group = amax_per_row.reshape(4, 16).amax(dim=-1)
        return y, amax_per_group

    # Input
    x = torch.randn(4, 16, 4096, device="cuda")
    weight = torch.randn(4096, device="cuda")
    bias = torch.randn(4096, device="cuda")

    # Run
    y, amax = fn(x, weight, bias)

    # Verify
    y_ref = torch.nn.functional.layer_norm(x, [4096], weight, bias)
    torch.testing.assert_close(y, y_ref)

    amax_ref = y_ref.reshape(64, 4096).abs().amax(dim=-1).reshape(4, 16).amax(dim=-1)
    torch.testing.assert_close(amax, amax_ref)

    print("test_layernorm_amax passed!")
    return y, amax


def test_simple_reduction_chain():
    """Simple test of reduction -> small reduction."""

    @torch.compile(fullgraph=True)
    def fn(x):
        # x: [64, 4096]
        # First reduction: sum over last dim -> [64]
        y = x.sum(dim=-1)
        # Second reduction: sum over groups of 16 -> [4]
        z = y.reshape(4, 16).sum(dim=-1)
        return z

    x = torch.randn(64, 4096, device="cuda")
    result = fn(x)

    expected = x.sum(dim=-1).reshape(4, 16).sum(dim=-1)
    # Use looser tolerance for sum reductions (floating point ordering differences)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    print("test_simple_reduction_chain passed!")
    return result


def test_edge_cases():
    """Test various edge cases with different numel2 and rnumel2 values."""

    # Test with different rnumel2 values
    for rnumel2 in [2, 4, 8, 16, 32]:
        @torch.compile(fullgraph=True)
        def fn(x, numel2=4, rnumel2=rnumel2):
            # x: [numel2 * rnumel2, 4096]
            y = x.sum(dim=-1)
            z = y.reshape(numel2, rnumel2).sum(dim=-1)
            return z

        numel1 = 4 * rnumel2
        x = torch.randn(numel1, 4096, device="cuda")
        result = fn(x)
        expected = x.sum(dim=-1).reshape(4, rnumel2).sum(dim=-1)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    # Test with amin reduction
    @torch.compile(fullgraph=True)
    def fn_amin(x):
        y = x.amin(dim=-1)
        z = y.reshape(4, 16).amin(dim=-1)
        return z

    x = torch.randn(64, 4096, device="cuda")
    result = fn_amin(x)
    expected = x.amin(dim=-1).reshape(4, 16).amin(dim=-1)
    torch.testing.assert_close(result, expected)

    print("test_edge_cases passed!")


if __name__ == "__main__":
    # Enable logging to see fusion decisions
    import logging
    import torch._logging
    torch._logging.set_logs(fusion=True)

    print("Testing small reduction fusion...")
    print(f"small_reduction_epilogue = {config.triton.small_reduction_epilogue}")
    print(f"small_reduction_epilogue_fusion = {config.triton.small_reduction_epilogue_fusion}")

    test_simple_reduction_chain()
    test_amax_chain()
    test_layernorm_amax()
    test_edge_cases()

    print("\nAll tests passed!")
