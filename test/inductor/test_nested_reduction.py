# Owner(s): ["module: inductor"]

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.utils import same
from torch._inductor import metrics
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


def _run_and_capture_kernel_source(f, args, kernel_signature: str) -> str:
    """Compile one representative pattern and return its fused Triton source."""
    def capture():
        with inductor_config.patch("triton.nested_reduction", True):
            compiled = torch.compile(f)
            return compiled(*args)

    with fresh_inductor_cache():
        _, source_codes = run_and_get_code(capture)
    metrics.reset()
    torch._dynamo.reset()

    # run_and_get_code can return multiple snippets; use a stable substring to
    # select the fused Triton kernel corresponding to the requested test case.
    return next(
        code
        for code in reversed(source_codes)
        if kernel_signature in code and "@triton_heuristics" in code
    )


def _capture_nvfp4_kernel_source(batch_size: int) -> str:
    B, D = batch_size, 4096
    import torch.nn.functional as F
    from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise

    G = 16

    def f(x, weight):
        x = F.rms_norm(x, (D,), weight)
        x = x.view(B, D // G, G)
        amax = x.abs().amax(dim=-1)
        scale = (amax / 448.0).clamp(min=1e-12).to(torch.float8_e4m3fn)
        xg = x.view(B, D // G, G // 2, 2)
        scale_f = scale.float().unsqueeze(-1)
        even = xg[..., 0].float() / scale_f
        odd = xg[..., 1].float() / scale_f
        packed = inline_asm_elementwise(
            even,
            odd,
            asm_str=(
                "{.reg .b8 t;"
                " cvt.rn.satfinite.e2m1x2.f32 t, $2, $1;"
                " cvt.u32.u8 $0, t;}"
            ),
            constraints="=r,f,f",
            dtype=torch.int32,
            is_pure=True,
            pack=1,
        )
        return packed.to(torch.uint8).view(B, D // 2), scale

    x = torch.randn(B, D, device="cuda")
    w = torch.randn(D, device="cuda")
    return _run_and_capture_kernel_source(
        f,
        (x, w),
        "cvt.rn.satfinite.e2m1x2.f32",
    )


def _capture_amax_kernel_source(batch_size: int) -> str:
    B, D = batch_size, 4096
    import torch.nn.functional as F

    G = 16

    def f(x, weight):
        x = F.rms_norm(x, (D,), weight)
        return x.view(B, D // G, G).abs().amax(dim=-1)

    x = torch.randn(B, D, device="cuda")
    w = torch.randn(D, device="cuda")
    return _run_and_capture_kernel_source(
        f,
        (x, w),
        "triton_per_fused",
    )


def _capture_fullres_kernel_source(batch_size: int) -> str:
    B, D = batch_size, 4096
    import torch.nn.functional as F

    G = 128
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    def f(x, weight):
        x = F.rms_norm(x, (D,), weight)
        x_groups = x.view(B, D // G, G)
        amax = x_groups.abs().amax(dim=-1)
        scale = (amax / fp8_max).clamp(min=1e-12)
        x_fp8 = (x_groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        return x_fp8.view(B, D).float(), scale

    x = torch.randn(B, D, device="cuda")
    w = torch.randn(D, device="cuda")
    return _run_and_capture_kernel_source(
        f,
        (x, w),
        "triton_per_fused",
    )


class TestBase(TestCase):
    def setUp(self):
        super().setUp()
        metrics.reset()
        torch._dynamo.utils.clear_compilation_metrics()

    def check_numeric(self, f, args, tol=1e-2):
        ref = f(*args)
        act = torch.compile(f)(*args)
        self.assertTrue(same(ref, act, tol=tol))

    def check_fusion(self, expected_kernels=1):
        """Verify nested reduction fired and produced the expected kernels."""
        self.assertEqual(
            inductor_config.triton.nested_reduction,
            metrics.codegen_nested_reduction,
        )
        if inductor_config.triton.nested_reduction and expected_kernels is not None:
            self.assertEqual(metrics.generated_kernel_count, expected_kernels)

    def assert_nvfp4_kernel_form(self, batch_size: int) -> None:
        code = _capture_nvfp4_kernel_source(batch_size)
        (
            FileCheck()
            .check_count("tl.load(in_ptr0 +", 1, exactly=True)
            .check_count("tl.load(in_ptr1 +", 1, exactly=True)
            .check("tl.split(")
            .check_count("tl.store(out_ptr", 2, exactly=True)
            .run(code)
        )

    def assert_amax_kernel_form(self, batch_size: int) -> None:
        code = _capture_amax_kernel_source(batch_size)
        (
            FileCheck()
            .check_count("tl.load(in_ptr0 +", 1, exactly=True)
            .check_count("tl.load(in_ptr1 +", 1, exactly=True)
            .check_not("tl.split(")
            .check_count("tl.store(out_ptr", 1, exactly=True)
            .run(code)
        )

    def assert_fullres_kernel_form(self, batch_size: int) -> None:
        code = _capture_fullres_kernel_source(batch_size)
        (
            FileCheck()
            .check_count("tl.load(in_ptr0 +", 1, exactly=True)
            .check_count("tl.load(in_ptr1 +", 1, exactly=True)
            .check_not("tl.split(")
            .check("tl.broadcast_to")
            .check_count("tl.store(out_ptr", 2, exactly=True)
            .run(code)
        )


@instantiate_parametrized_tests
class NestedReductionTest(TestBase):
    """Tests for fusing dependent cross-axis reductions into a single kernel."""

    # ---- Pattern 1: small dim in x (rmsnorm-like) ----
    # K must be large enough that Inductor treats the sum as a reduction
    # rather than unrolling it as pointwise (threshold is around K>=16).

    @parametrize("B", [32, 256])
    @parametrize("K", [16, 32])
    @parametrize("D", [4096])
    def test_rmsnorm_weighted_sum(self, B, K, D):
        def f(x, w):
            BK, Dim = x.shape[0] * x.shape[1], x.shape[2]
            x_flat = x.reshape(BK, Dim)
            rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
            x_normed = (x_flat / rms).reshape(x.shape)
            return (w[:, :, None] * x_normed).sum(dim=1)

        x = torch.randn(B, K, D, device=GPU_TYPE)
        w = torch.randn(B, K, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_fusion()

    @parametrize("K", [16, 32])
    def test_rmsnorm_weighted_max(self, K):
        def f(x, w):
            BK, D = x.shape[0] * x.shape[1], x.shape[2]
            x_flat = x.reshape(BK, D)
            rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
            x_normed = (x_flat / rms).reshape(x.shape)
            return (w[:, :, None] * x_normed).amax(dim=1)

        x = torch.randn(64, K, 4096, device=GPU_TYPE)
        w = torch.randn(64, K, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_fusion()

    @parametrize("reduce_fn", ["sum", "amax", "amin"])
    def test_rmsnorm_weighted_reduce_B1(self, reduce_fn):
        """B=1 flattened small_dim_in_x still fuses for pass-2 reductions."""

        B, K, D = 1, 16, 1024
        rfn = {"sum": torch.Tensor.sum, "amax": torch.Tensor.amax,
               "amin": torch.Tensor.amin}[reduce_fn]

        def f(x, w):
            x_flat = x.reshape(B * K, D)
            rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
            x_normed = (x_flat / rms).reshape(B, K, D)
            return rfn(w[:, :, None] * x_normed, dim=1)

        x = torch.randn(B, K, D, device=GPU_TYPE)
        w = torch.randn(B, K, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_fusion()
        if inductor_config.triton.nested_reduction:
            self.assertEqual(metrics.generated_kernel_count, 1)

    def test_layernorm_weighted_sum(self):
        """Welford mean+var both consumed by node2."""

        def f(x, w):
            B, K, D = x.shape
            x_flat = x.reshape(B * K, D)
            mean = x_flat.mean(dim=-1, keepdim=True)
            var = x_flat.var(dim=-1, keepdim=True, correction=0)
            x_normed = ((x_flat - mean) / torch.sqrt(var + 1e-6)).reshape(B, K, D)
            return (w[:, :, None] * x_normed).sum(dim=1)

        x = torch.randn(64, 16, 4096, device=GPU_TYPE)
        w = torch.randn(64, 16, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_fusion()

    def test_layernorm_weighted_sum_B1(self):
        """B=1 flattened LayerNorm small_dim_in_x still fuses."""

        B, K, D = 1, 16, 1024

        def f(x, w):
            x_flat = x.reshape(B * K, D)
            mean = x_flat.mean(dim=-1, keepdim=True)
            var = x_flat.var(dim=-1, keepdim=True, correction=0)
            x_normed = ((x_flat - mean) / torch.sqrt(var + 1e-6)).reshape(B, K, D)
            return (w[:, :, None] * x_normed).sum(dim=1)

        x = torch.randn(B, K, D, device=GPU_TYPE)
        w = torch.randn(B, K, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_fusion()
        if inductor_config.triton.nested_reduction:
            self.assertEqual(metrics.generated_kernel_count, 1)

    # ---- Pattern 2: small dim in r (NVFP4-like) ----

    @parametrize("B", [32, 256])
    @parametrize("G", [8, 16, 32])
    def test_layernorm_block_amax(self, B, G):
        def f(x, G):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, correction=0)
            x_normed = (x - mean) / torch.sqrt(var + 1e-6)
            return x_normed.reshape(x.shape[0], x.shape[1] // G, G).abs().amax(dim=-1)

        x = torch.randn(B, 4096, device=GPU_TYPE)
        self.check_numeric(f, (x, G))
        self.check_fusion()

    @parametrize("G", [8, 16])
    def test_rmsnorm_block_amax(self, G):
        def f(x, G):
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            x_normed = x / rms
            return x_normed.reshape(x.shape[0], x.shape[1] // G, G).abs().amax(dim=-1)

        x = torch.randn(128, 8192, device=GPU_TYPE)
        self.check_numeric(f, (x, G))
        self.check_fusion()

    def test_layernorm_block_sum(self):
        def f(x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, correction=0)
            x_normed = (x - mean) / torch.sqrt(var + 1e-6)
            return x_normed.reshape(x.shape[0], x.shape[1] // 16, 16).sum(dim=-1)

        x = torch.randn(64, 4096, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.check_fusion()

    def test_layernorm_block_min(self):
        def f(x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, correction=0)
            x_normed = (x - mean) / torch.sqrt(var + 1e-6)
            return x_normed.reshape(x.shape[0], x.shape[1] // 16, 16).amin(dim=-1)

        x = torch.randn(64, 4096, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.check_fusion()

    # ---- Epilogue dtype conversion ----

    def test_bf16_epilogue_pattern1(self):
        def f(x, w):
            B, K, D = x.shape
            x_flat = x.reshape(B * K, D)
            rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
            x_normed = (x_flat / rms).reshape(B, K, D)
            return (w[:, :, None] * x_normed).sum(dim=1).to(torch.bfloat16)

        x = torch.randn(64, 16, 4096, device=GPU_TYPE)
        w = torch.randn(64, 16, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_fusion()

    def test_bf16_epilogue_pattern2(self):
        def f(x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, correction=0)
            x_normed = (x - mean) / torch.sqrt(var + 1e-6)
            return (
                x_normed.reshape(x.shape[0], x.shape[1] // 16, 16)
                .abs()
                .amax(dim=-1)
                .to(torch.bfloat16)
            )

        x = torch.randn(64, 4096, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.check_fusion()

    # ---- Downstream pointwise fusion ----

    def test_pointwise_epilogue_pattern1(self):
        """Fuse out * scale + bias after nested reduction (pattern 1)."""

        def f(x, w, scale, bias):
            B, K, D = x.shape
            x_flat = x.reshape(B * K, D)
            rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
            x_normed = (x_flat / rms).reshape(B, K, D)
            out = (w[:, :, None] * x_normed).sum(dim=1)
            return out * scale + bias

        x = torch.randn(64, 16, 4096, device=GPU_TYPE)
        w = torch.randn(64, 16, device=GPU_TYPE)
        scale = torch.randn(64, 4096, device=GPU_TYPE)
        bias = torch.randn(64, 4096, device=GPU_TYPE)
        self.check_numeric(f, (x, w, scale, bias))
        self.check_fusion()

    def test_pointwise_epilogue_pattern2(self):
        """Fuse out * scale + bias after nested reduction (pattern 2)."""

        def f(x, scale, bias):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, correction=0)
            x_normed = (x - mean) / torch.sqrt(var + 1e-6)
            out = (
                x_normed.reshape(x.shape[0], x.shape[1] // 16, 16)
                .abs()
                .amax(dim=-1)
            )
            return out * scale + bias

        x = torch.randn(64, 4096, device=GPU_TYPE)
        scale = torch.randn(64, 256, device=GPU_TYPE)
        bias = torch.randn(64, 256, device=GPU_TYPE)
        self.check_numeric(f, (x, scale, bias))
        self.check_fusion()

    # ---- Edge cases ----

    @parametrize(
        "B,D,G",
        [(256, 4096, 16), (128, 4096, 32), (256, 8192, 32)],
    )
    def test_edge_B_equals_D_over_G(self, B, D, G):
        """When B == D/G, size-based matching is ambiguous."""

        def f(x, G):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, correction=0)
            x_normed = (x - mean) / torch.sqrt(var + 1e-6)
            return x_normed.reshape(x.shape[0], x.shape[1] // G, G).abs().amax(dim=-1)

        x = torch.randn(B, D, device=GPU_TYPE)
        self.check_numeric(f, (x, G))
        self.check_fusion()

    @parametrize("BK", [16, 32])
    def test_edge_B_equals_K(self, BK):
        """When B == K, size-based matching is ambiguous."""

        def f(x, w):
            B, K, D = x.shape
            x_flat = x.reshape(B * K, D)
            rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
            x_normed = (x_flat / rms).reshape(B, K, D)
            return (w[:, :, None] * x_normed).sum(dim=1)

        x = torch.randn(BK, BK, 4096, device=GPU_TYPE)
        w = torch.randn(BK, BK, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_fusion()

    # ---- Dynamic shapes ----

    def test_dynamic_shapes_pattern1(self):
        def f(x, w):
            B, K, D = x.shape
            x_flat = x.reshape(B * K, D)
            rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
            x_normed = (x_flat / rms).reshape(B, K, D)
            return (w[:, :, None] * x_normed).sum(dim=1)

        compiled = torch.compile(f, dynamic=True)

        for B in [32, 64, 128]:
            x = torch.randn(B, 16, 4096, device=GPU_TYPE)
            w = torch.randn(B, 16, device=GPU_TYPE)
            ref = f(x, w)
            act = compiled(x, w)
            self.assertTrue(same(ref, act, tol=1e-2))

    def test_dynamic_shapes_pattern2(self):
        def f(x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, correction=0)
            x_normed = (x - mean) / torch.sqrt(var + 1e-6)
            return x_normed.reshape(x.shape[0], x.shape[1] // 16, 16).abs().amax(dim=-1)

        compiled = torch.compile(f, dynamic=True)

        for B in [32, 64, 256]:
            x = torch.randn(B, 4096, device=GPU_TYPE)
            ref = f(x)
            act = compiled(x)
            self.assertTrue(same(ref, act, tol=1e-2))

    # ---- Producer-consumer: node2 reads node1's materialized output ----
    # Instead of node1 and node2 sharing a common input, node2 reads
    # node1's output. This triggers the producer-consumer path in
    # NestedReduction.can_fuse.

    def test_producer_consumer_rmsnorm_amax(self):
        """RMS norm materializes output, amax reads it."""
        import torch.nn.functional as F

        B, D, G = 128, 4096, 16

        def f(x, weight):
            x = F.rms_norm(x, (D,), weight)
            return x.view(B, D // G, G).abs().amax(dim=-1)

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        if inductor_config.triton.nested_reduction:
            self.assert_amax_kernel_form(B)
        self.check_numeric(f, (x, w))
        self.check_fusion()

    def test_producer_consumer_rmsnorm_scale(self):
        """RMS norm + amax + scale epilogue (clamp + to_fp8)."""
        import torch.nn.functional as F

        B, D, G = 128, 4096, 16

        def f(x, weight):
            x = F.rms_norm(x, (D,), weight)
            x = x.view(B, D // G, G)
            amax = x.abs().amax(dim=-1)
            scale = (amax / 448.0).clamp(min=1e-12).to(torch.float8_e4m3fn)
            return scale.float()

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        self.check_numeric(f, (x, w), tol=0.01)
        self.check_fusion()

    def test_producer_consumer_rmsnorm_fp8_quant(self):
        """RMS norm + amax + scale + full-res quantize epilogue.

        The quantize step (x / scale -> fp8) reads the full-resolution
        normalized output, testing the full-res epilogue fusion path
        (small_dim_in_r only).
        """
        import torch.nn.functional as F

        B, D, G = 128, 4096, 128
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        def f(x, weight):
            x = F.rms_norm(x, (D,), weight)
            x_groups = x.view(B, D // G, G)
            amax = x_groups.abs().amax(dim=-1)
            scale = (amax / fp8_max).clamp(min=1e-12)
            x_fp8 = (x_groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
            return x_fp8.view(B, D).float(), scale

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        if inductor_config.triton.nested_reduction:
            self.assert_fullres_kernel_form(B)
        ref = f(x, w)
        act = torch.compile(f)(x, w)
        # Scale should be exact between compiled variants
        self.assertTrue(same(ref[1], act[1], tol=1e-2))
        # FP8 values differ from eager due to intermediate precision
        # (f32 registers vs bf16 round-trip), but should be close
        self.assertTrue(same(ref[0], act[0], tol=0.5))
        self.check_fusion()

    def test_producer_consumer_rmsnorm_fp8_quant_B1(self):
        """B=1 edge case: flattened iteration range still fuses."""
        import torch.nn.functional as F

        B, D, G = 1, 4096, 128
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        def f(x, weight):
            x = F.rms_norm(x, (D,), weight)
            x_groups = x.view(B, D // G, G)
            amax = x_groups.abs().amax(dim=-1)
            scale = (amax / fp8_max).clamp(min=1e-12)
            x_fp8 = (x_groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
            return x_fp8.view(B, D).float(), scale

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        ref = f(x, w)
        act = torch.compile(f)(x, w)
        self.assertTrue(same(ref[1], act[1], tol=1e-2))
        self.assertTrue(same(ref[0], act[0], tol=0.5))
        self.check_fusion()

    def test_no_fullres_epilogue_small_dim_in_x(self):
        """Full-res epilogues must NOT fuse for small_dim_in_x patterns."""

        def f(x, w):
            B, K, D = x.shape
            x_flat = x.reshape(B * K, D)
            rms = torch.sqrt(
                torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6
            )
            x_normed = (x_flat / rms).reshape(B, K, D)
            s = (w[:, :, None] * x_normed).sum(dim=1)
            return x_normed + s[:, None, :]

        x = torch.randn(64, 16, 4096, device=GPU_TYPE)
        w = torch.randn(64, 16, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        # The nested reduction fuses the norm + weighted sum, but the
        # full-res addition must stay as a separate kernel.
        self.check_fusion(expected_kernels=2)

    def test_no_fullres_epilogue_small_dim_in_x_B1(self):
        """B=1 still keeps the full-res epilogue out of the nested kernel."""

        B, K, D = 1, 16, 1024

        def f(x, w):
            x_flat = x.reshape(B * K, D)
            rms = torch.sqrt(
                torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6
            )
            x_normed = (x_flat / rms).reshape(B, K, D)
            s = (w[:, :, None] * x_normed).sum(dim=1)
            return x_normed + s[:, None, :]

        x = torch.randn(B, K, D, device=GPU_TYPE)
        w = torch.randn(B, K, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        if inductor_config.triton.nested_reduction:
            self.assertEqual(metrics.codegen_nested_reduction, 1)
            self.assertEqual(metrics.generated_kernel_count, 2)

    def test_pass3_rmsnorm_nvfp4(self):
        """Pass 3: fuse pair-wise NVFP4 quantization into the kernel.

        This exercises the current half-resolution family path end-to-end:
        full-resolution values are split into even/odd halves, pass 2 scale
        is broadcast, and inline_asm_elementwise runs through the remapped
        handler.  The family/scaffolding is broader than NVFP4, but the
        concrete lane legality in this test is the NVFP4-specific even/odd
        pattern we support today.
        Requires sm_100+ for cvt.rn.satfinite.e2m1x2.f32.
        """
        import torch.nn.functional as F
        from torch._higher_order_ops.inline_asm_elementwise import (
            inline_asm_elementwise,
        )

        cc = torch.cuda.get_device_capability()
        if cc[0] < 10:
            self.skipTest("requires sm_100+ (Blackwell)")

        B, D, G = 128, 4096, 16

        def f(x, weight):
            x = F.rms_norm(x, (D,), weight)
            x = x.view(B, D // G, G)
            amax = x.abs().amax(dim=-1)
            scale = (amax / 448.0).clamp(min=1e-12).to(torch.float8_e4m3fn)
            xg = x.view(B, D // G, G // 2, 2)
            scale_f = scale.float().unsqueeze(-1)
            even = xg[..., 0].float() / scale_f
            odd = xg[..., 1].float() / scale_f
            packed = inline_asm_elementwise(
                even, odd,
                asm_str=(
                    "{.reg .b8 t;"
                    " cvt.rn.satfinite.e2m1x2.f32 t, $2, $1;"  # noqa: B950
                    " cvt.u32.u8 $0, t;}"
                ),
                constraints="=r,f,f",
                dtype=torch.int32,
                is_pure=True,
                pack=1,
            )
            return packed.to(torch.uint8).view(B, D // 2), scale

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        if inductor_config.triton.nested_reduction:
            self.assert_nvfp4_kernel_form(B)

        # Eager can't run inline_asm_elementwise, so use the unfused
        # compiled path as the correctness oracle.
        torch._dynamo.reset()
        with inductor_config.patch("triton.nested_reduction", False):
            ref = torch.compile(f)(x, w)
        metrics.reset()
        torch._dynamo.reset()
        act = torch.compile(f)(x, w)
        self.assertTrue(same(ref[0], act[0]))
        self.assertTrue(same(ref[1].float(), act[1].float(), tol=1e-2))
        self.check_fusion()

    def test_pass3_rmsnorm_nvfp4_B1(self):
        """Pass 3 with B=1: edge case where XBLOCK=1.

        Verifies that the iteration range context correctly handles
        the trivial x-dimension (numel=1).
        """
        import torch.nn.functional as F
        from torch._higher_order_ops.inline_asm_elementwise import (
            inline_asm_elementwise,
        )

        cc = torch.cuda.get_device_capability()
        if cc[0] < 10:
            self.skipTest("requires sm_100+ (Blackwell)")

        B, D, G = 1, 4096, 16

        def f(x, weight):
            x = F.rms_norm(x, (D,), weight)
            x = x.view(B, D // G, G)
            amax = x.abs().amax(dim=-1)
            scale = (amax / 448.0).clamp(min=1e-12).to(torch.float8_e4m3fn)
            xg = x.view(B, D // G, G // 2, 2)
            scale_f = scale.float().unsqueeze(-1)
            even = xg[..., 0].float() / scale_f
            odd = xg[..., 1].float() / scale_f
            packed = inline_asm_elementwise(
                even, odd,
                asm_str=(
                    "{.reg .b8 t;"
                    " cvt.rn.satfinite.e2m1x2.f32 t, $2, $1;"  # noqa: B950
                    " cvt.u32.u8 $0, t;}"
                ),
                constraints="=r,f,f",
                dtype=torch.int32,
                is_pure=True,
                pack=1,
            )
            return packed.to(torch.uint8).view(B, D // 2), scale

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        if inductor_config.triton.nested_reduction:
            self.assert_nvfp4_kernel_form(B)
        torch._dynamo.reset()
        with inductor_config.patch("triton.nested_reduction", False):
            ref = torch.compile(f)(x, w)
        metrics.reset()
        torch._dynamo.reset()
        act = torch.compile(f)(x, w)
        self.assertTrue(same(ref[0], act[0]))
        self.assertTrue(same(ref[1].float(), act[1].float(), tol=1e-2))
        self.check_fusion()


@inductor_config.patch(
    "triton.nested_reduction",
    not inductor_config.triton.nested_reduction,
)
class NoNestedReductionTest(NestedReductionTest):
    """Run all the same tests with nested_reduction toggled off."""

    pass


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
