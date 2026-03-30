# Owner(s): ["module: inductor"]

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.utils import same
from torch._inductor import metrics
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


class TestBase(TestCase):
    def setUp(self):
        super().setUp()
        metrics.reset()
        torch._dynamo.utils.clear_compilation_metrics()

    def check_numeric(self, f, args, tol=1e-2):
        ref = f(*args)
        act = torch.compile(f)(*args)
        if isinstance(ref, tuple):
            for i, (r, a) in enumerate(zip(ref, act)):
                self.assertTrue(
                    same(r, a, tol=tol),
                    f"output[{i}] max_diff={(r - a).abs().max().item()}",
                )
        else:
            self.assertTrue(
                same(ref, act, tol=tol),
                f"max_diff={(ref - act).abs().max().item()}",
            )

    def check_fusion(self):
        """Verify nested reduction fired iff config is enabled."""
        self.assertEqual(
            inductor_config.triton.nested_reduction,
            metrics.codegen_nested_reduction,
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
        if inductor_config.triton.nested_reduction:
            self.assertEqual(metrics.generated_kernel_count, 1)

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
        if inductor_config.triton.nested_reduction:
            self.assertEqual(metrics.generated_kernel_count, 1)

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
        if inductor_config.triton.nested_reduction:
            self.assertEqual(metrics.generated_kernel_count, 1)

    def test_pass3_rmsnorm_nvfp4(self):
        """Pass 3: fuse pair-wise NVFP4 quantization into the kernel.

        Tests the _IterationRangeContext split/broadcast path end-to-end:
        full-resolution values are split into even/odd halves, pass 2
        scale is broadcast, and inline_asm_elementwise runs through
        the remapped handler.
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

        # inline_asm_elementwise eager uses jiterator which may not
        # support the target arch, so only verify the compiled path
        # produces 1 fused kernel and runs without error.
        compiled = torch.compile(f)
        compiled(x, w)
        self.check_fusion()
        if inductor_config.triton.nested_reduction:
            self.assertEqual(metrics.generated_kernel_count, 1)

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
        compiled = torch.compile(f)
        compiled(x, w)
        self.check_fusion()
        if inductor_config.triton.nested_reduction:
            self.assertEqual(metrics.generated_kernel_count, 1)


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
