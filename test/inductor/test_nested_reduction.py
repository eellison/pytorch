# Owner(s): ["module: inductor"]

"""End-to-end nested-reduction behavior tests.

Generated-kernel structure checks live in test_nested_reduction_internals.py
so this file can stay focused on numerics, fusion policy, and edge cases.
"""

import torch
import torch._inductor.config as inductor_config
from torch._inductor import metrics
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU

from torch._inductor.choices import InductorChoices
from torch._inductor.virtualized import V


def _choices_context(force_persistent: bool | None):
    import contextlib

    if force_persistent is None:
        return contextlib.nullcontext()

    class _Choices(InductorChoices):
        @staticmethod
        def should_use_cooperative_reduction(*args, **kwargs):
            return False

        @staticmethod
        def should_use_persistent_reduction(*args, **kwargs):
            return force_persistent

    return V.set_choices_handler(_Choices())


class TestBase(TestCase):
    force_persistent_outer_reduction: bool | None = None

    def setUp(self):
        super().setUp()
        metrics.reset()
        torch._dynamo.utils.clear_compilation_metrics()
        self._choices_ctx = _choices_context(self.force_persistent_outer_reduction)
        self._choices_ctx.__enter__()

    def tearDown(self):
        self._choices_ctx.__exit__(None, None, None)
        super().tearDown()

    def check_numeric(self, f, args, tol=1e-2):
        ref = f(*args)
        act = torch.compile(f)(*args)
        self.assertEqual(act, ref, atol=tol, rtol=tol)

    def check_fusion(self, expected_kernels=1):
        if inductor_config.triton.nested_reduction:
            self.assertEqual(metrics.codegen_nested_reduction, 1)
            if expected_kernels is not None:
                self.assertEqual(metrics.generated_kernel_count, expected_kernels)
        else:
            self.assertEqual(metrics.codegen_nested_reduction, 0)


def _rmsnorm(x_flat):
    return x_flat / torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)


def _layernorm(x_flat):
    mean = x_flat.mean(dim=-1, keepdim=True)
    var = x_flat.var(dim=-1, keepdim=True, correction=0)
    return (x_flat - mean) / torch.sqrt(var + 1e-6)


@instantiate_parametrized_tests
class _NestedReductionBase:
    """Tests for fusing dependent cross-axis reductions into a single kernel."""

    # ---- Pattern 1: small dim in x (weighted norm + reduce over K) ----

    def _pattern1(self, norm, reduce_fn, B, K, D):
        rfn = {"sum": torch.Tensor.sum, "amax": torch.Tensor.amax,
               "amin": torch.Tensor.amin}[reduce_fn]

        def f(x, w):
            x_normed = norm(x.reshape(x.shape[0] * K, D)).reshape(x.shape)
            return rfn(w[:, :, None] * x_normed, dim=1)

        x = torch.randn(B, K, D, device=GPU_TYPE)
        w = torch.randn(B, K, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_fusion()

    @parametrize("B", [32, 256])
    @parametrize("K", [16, 32])
    def test_rmsnorm_weighted_sum(self, B, K):
        self._pattern1(_rmsnorm, "sum", B, K, 4096)

    @parametrize("K", [16, 32])
    def test_rmsnorm_weighted_max(self, K):
        self._pattern1(_rmsnorm, "amax", 64, K, 4096)

    @parametrize("reduce_fn", ["sum", "amax", "amin"])
    def test_rmsnorm_weighted_reduce_B1(self, reduce_fn):
        """B=1 flattened small_dim_in_x still fuses."""
        self._pattern1(_rmsnorm, reduce_fn, 1, 16, 1024)

    def test_layernorm_weighted_sum(self):
        self._pattern1(_layernorm, "sum", 64, 16, 4096)

    def test_layernorm_weighted_sum_B1(self):
        self._pattern1(_layernorm, "sum", 1, 16, 1024)

    def test_fullres_prologue_small_dim_in_x_loop_order(self):
        """Full-res prologue uses the grouped reduction's logical loop order."""

        B, K, D = 16, 16, 1024

        def f(x, w, bias):
            x_flat = x.reshape(B * K, D)
            rms = torch.sqrt(
                torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6
            )
            y = torch.ops._inductor_test.realize(
                torch.relu((x_flat / rms).reshape(B, K, D) + bias[:, None, :])
            )
            return y, (w[:, :, None] * y).sum(dim=1)

        x = torch.randn(B, K, D, device=GPU_TYPE)
        w = torch.randn(B, K, device=GPU_TYPE)
        bias = torch.randn(B, D, device=GPU_TYPE)
        self.check_numeric(f, (x, w, bias))
        self.check_fusion()

    # ---- Pattern 2: small dim in r (norm + block reduce) ----

    def _pattern2(self, norm, reduce_fn, B, D, G):
        rfn = {"sum": torch.Tensor.sum, "amax": torch.Tensor.amax,
               "amin": torch.Tensor.amin}[reduce_fn]

        def f(x):
            x_normed = norm(x)
            grouped = x_normed.reshape(x.shape[0], x.shape[1] // G, G)
            if reduce_fn == "amax":
                return grouped.abs().amax(dim=-1)
            return rfn(grouped, dim=-1)

        x = torch.randn(B, D, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.check_fusion()

    @parametrize(
        "B,D,G",
        [
            (32, 4096, 8),
            (32, 4096, 16),
            (32, 4096, 32),
            (256, 4096, 8),
            (256, 4096, 16),
            (256, 4096, 32),
            (4, 384, 128),
        ],
    )
    def test_layernorm_block_amax(self, B, D, G):
        self._pattern2(_layernorm, "amax", B, D, G)

    @parametrize("G", [8, 16])
    def test_rmsnorm_block_amax(self, G):
        self._pattern2(_rmsnorm, "amax", 128, 8192, G)

    def test_layernorm_block_sum(self):
        self._pattern2(_layernorm, "sum", 64, 4096, 16)

    def test_layernorm_block_min(self):
        self._pattern2(_layernorm, "amin", 64, 4096, 16)

    def test_layernorm_block_amax_group_size_512(self):
        self._pattern2(_layernorm, "amax", 32, 4096, 512)

    def test_layernorm_block_amax_non_power_of_2_xblock_cap(self):
        """D=6144 computes a non-power-of-two max_xblock before rounding."""
        self._pattern2(_layernorm, "amax", 16, 6144, 128)

    # ---- Epilogue dtype conversion ----

    def test_bf16_epilogue_pattern1(self):
        def f(x, w):
            x_normed = _rmsnorm(x.reshape(x.shape[0] * 16, 4096)).reshape(x.shape)
            return (w[:, :, None] * x_normed).sum(dim=1).to(torch.bfloat16)

        x = torch.randn(64, 16, 4096, device=GPU_TYPE)
        w = torch.randn(64, 16, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_fusion()

    def test_bf16_epilogue_pattern2(self):
        def f(x):
            return _layernorm(x).reshape(x.shape[0], -1, 16).abs().amax(dim=-1).to(torch.bfloat16)

        x = torch.randn(64, 4096, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.check_fusion()

    # ---- Downstream pointwise fusion ----

    def test_pointwise_epilogue_pattern1(self):
        """Fuse out * scale + bias after nested reduction (pattern 1)."""

        def f(x, w, scale, bias):
            x_normed = _rmsnorm(x.reshape(x.shape[0] * 16, 4096)).reshape(x.shape)
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
            out = (
                _layernorm(x).reshape(x.shape[0], x.shape[1] // 16, 16)
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

    @parametrize("dynamic", [False, True])
    def test_shapes_pattern1(self, dynamic):
        """Dynamic small-dim-in-x keeps only B/D dynamic."""
        K = 16

        def f(x, w):
            B, D = x.shape[0], x.shape[2]
            x_flat = x.reshape(B * K, D)
            rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
            x_normed = (x_flat / rms).reshape(B, K, D)
            return (w[:, :, None] * x_normed).sum(dim=1)

        compiled = torch.compile(f, dynamic=dynamic)
        for B, D in (
            [(32, 1024), (64, 2048), (128, 4096)] if dynamic else [(32, 4096)]
        ):
            x = torch.randn(B, K, D, device=GPU_TYPE)
            w = torch.randn(B, K, device=GPU_TYPE)
            if dynamic:
                torch._dynamo.mark_static(x, 1)
                torch._dynamo.mark_static(w, 1)
            ref = f(x, w)
            act = compiled(x, w)
            self.assertEqual(act, ref, atol=1e-2, rtol=1e-2)
        self.check_fusion()

    @parametrize("dynamic", [False, True])
    def test_shapes_pattern2(self, dynamic):
        def f(x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, correction=0)
            x_normed = (x - mean) / torch.sqrt(var + 1e-6)
            return x_normed.reshape(x.shape[0], x.shape[1] // 16, 16).abs().amax(dim=-1)

        compiled = torch.compile(f, dynamic=dynamic)
        for B in ([32, 64, 256] if dynamic else [32]):
            x = torch.randn(B, 4096, device=GPU_TYPE)
            ref = f(x)
            act = compiled(x)
            self.assertEqual(act, ref, atol=1e-2, rtol=1e-2)
        self.check_fusion()

    @parametrize("dynamic", [False, True])
    def test_shapes_varying_batch_and_dim(self, dynamic):
        """Dynamic shapes: vary both B and D at runtime."""
        import torch.nn.functional as F

        def f(x, weight):
            x = F.rms_norm(x, (x.shape[-1],), weight)
            B, D = x.shape
            return x.view(B, D // 128, 128).abs().amax(dim=-1)

        compiled = torch.compile(f, dynamic=dynamic)
        for B, D in (
            [(4, 512), (8, 1024), (16, 2048)] if dynamic else [(4, 512)]
        ):
            x = torch.randn(B, D, device=GPU_TYPE)
            w = torch.randn(D, device=GPU_TYPE)
            ref = f(x, w)
            act = compiled(x, w)
            self.assertEqual(act, ref, atol=1e-2, rtol=1e-2)
        self.check_fusion()

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

    def test_producer_consumer_rmsnorm_amax_B1(self):
        """B=1 keeps the reduced-output kernel form after index canonicalization."""
        import torch.nn.functional as F

        B, D, G = 1, 4096, 16

        def f(x, weight):
            x = F.rms_norm(x, (D,), weight)
            return x.view(B, D // G, G).abs().amax(dim=-1)

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_fusion()

    @parametrize("pointwise_kind", ["full", "row_broadcast", "col_broadcast"])
    @parametrize("epilogue_resolution", ["reduced", "full"])
    def test_reduction_fusion_pointwise_prologue_epilogue(
        self, pointwise_kind, epilogue_resolution,
    ):
        from torch._inductor.scheduler import FusedNestedReductions
        import torch.nn.functional as F

        B, D, G = 128, 4096, 128

        def f(x, weight, prologue_extra, epilogue_extra):
            x = F.rms_norm(x, (D,), weight)
            x = x.view(B, D // G, G)
            if pointwise_kind == "full":
                prologue_extra = prologue_extra.view(B, D // G, G)
            elif pointwise_kind == "row_broadcast":
                prologue_extra = prologue_extra[:, :, None]
            else:
                prologue_extra = prologue_extra.view(D // G, G)
            x = torch.ops._inductor_test.realize(x + prologue_extra)
            out = x.abs().amax(dim=-1)
            out = out + epilogue_extra
            if epilogue_resolution == "reduced":
                return out
            return (x / (out.abs() + 1e-6)[:, :, None]).view(B, D)

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        prologue_extra_shape = {
            "full": (B, D),
            "row_broadcast": (B, 1),
            "col_broadcast": (D,),
        }[pointwise_kind]
        epilogue_extra_shape = {
            "full": (B, D // G),
            "row_broadcast": (B, 1),
            "col_broadcast": (D // G,),
        }[pointwise_kind]
        prologue_extra = torch.randn(prologue_extra_shape, device=GPU_TYPE)
        epilogue_extra = torch.randn(epilogue_extra_shape, device=GPU_TYPE)
        saw_nested_reduction = False

        def check_reduction_fusion(nodes):
            nonlocal saw_nested_reduction
            fused_nodes = [n for n in nodes if isinstance(n, FusedNestedReductions)]
            self.assertEqual(len(fused_nodes), 1)
            saw_nested_reduction = True

            node2_nodes = list(fused_nodes[0].node2.get_nodes())
            reductions = [sn for sn in node2_nodes if sn.is_reduction()]
            self.assertEqual(len(reductions), 1)
            reduction = reductions[0]
            reduction_names = reduction.get_operation_names()
            _, (reduced_numel, rnumel) = reduction.group
            full_numel = reduced_numel * rnumel
            fullres_prologue_count = 0
            reduced_epilogue_count = 0
            fullres_epilogue_count = 0

            for sn in node2_nodes:
                if sn.is_reduction():
                    continue
                sn_names = sn.get_operation_names()
                is_prologue = bool(sn_names & reduction.ancestors)
                is_epilogue = bool(reduction_names & sn.ancestors)
                self.assertTrue(is_prologue or is_epilogue)
                self.assertFalse(is_prologue and is_epilogue)
                _, (sn_numel, _) = sn.group
                if is_prologue:
                    self.assertEqual(sn_numel, full_numel)
                    fullres_prologue_count += 1
                elif sn_numel == reduced_numel:
                    reduced_epilogue_count += 1
                else:
                    self.assertEqual(sn_numel, full_numel)
                    fullres_epilogue_count += 1

            self.assertGreaterEqual(fullres_prologue_count, 1)
            if epilogue_resolution == "full":
                self.assertGreaterEqual(fullres_epilogue_count, 1)
            else:
                self.assertGreaterEqual(reduced_epilogue_count, 1)
                self.assertEqual(fullres_epilogue_count, 0)
            return nodes

        with inductor_config.patch(
            _post_fusion_custom_pass=check_reduction_fusion,
            fx_graph_cache=False,
        ):
            self.check_numeric(f, (x, w, prologue_extra, epilogue_extra))
        self.assertTrue(saw_nested_reduction)
        self.check_fusion()

    def test_reduced_resolution_pointwise_prologue(self):
        from torch._inductor.scheduler import FusedNestedReductions

        B, D, G = 128, 4096, 128

        def f(x, group_extra, epilogue_extra):
            sums = (x * x).sum(dim=-1, keepdim=True)
            inv = torch.rsqrt(sums / D + 1e-6)
            group_extra = torch.ops._inductor_test.realize(group_extra + sums)
            x = (x * inv).view(B, D // G, G)
            out = (x + group_extra[:, :, None]).abs().amax(dim=-1)
            return out + epilogue_extra

        x = torch.randn(B, D, device=GPU_TYPE)
        group_extra = torch.randn(B, D // G, device=GPU_TYPE)
        epilogue_extra = torch.randn(B, D // G, device=GPU_TYPE)
        saw_reduced_prologue = False

        def check_reduction_fusion(nodes):
            nonlocal saw_reduced_prologue
            fused_nodes = [n for n in nodes if isinstance(n, FusedNestedReductions)]
            self.assertEqual(len(fused_nodes), 1)
            node2_nodes = list(fused_nodes[0].node2.get_nodes())
            reductions = [sn for sn in node2_nodes if sn.is_reduction()]
            self.assertEqual(len(reductions), 1)
            reduction = reductions[0]
            reduction_names = reduction.get_operation_names()
            _, (reduced_numel, _) = reduction.group
            for sn in node2_nodes:
                if sn.is_reduction():
                    continue
                is_prologue = bool(sn.get_operation_names() & reduction.ancestors)
                is_epilogue = bool(reduction_names & sn.ancestors)
                self.assertTrue(is_prologue or is_epilogue)
                if is_prologue:
                    _, (sn_numel, _) = sn.group
                    saw_reduced_prologue |= sn_numel == reduced_numel
            return nodes

        with inductor_config.patch(
            _post_fusion_custom_pass=check_reduction_fusion,
            fx_graph_cache=False,
        ):
            self.check_numeric(f, (x, group_extra, epilogue_extra))
        self.assertTrue(saw_reduced_prologue)
        self.check_fusion()

    # ---- Exotic indexing ----

    def test_transposed_input(self):
        """Non-contiguous (transposed) input — numerics must be correct."""

        def f(x):
            x = x.t()
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            x_norm = x / rms
            return x_norm.reshape(x.shape[0], -1, 16).abs().amax(dim=-1)

        x = torch.randn(4096, 64, device=GPU_TYPE)
        self.check_numeric(f, (x,))

    def test_strided_slice_input(self):
        """Stride-2 slice input — numerics must be correct."""

        def f(x):
            x = x[:, ::2]
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            x_norm = x / rms
            return x_norm.reshape(x.shape[0], -1, 16).abs().amax(dim=-1)

        x = torch.randn(32, 4096, device=GPU_TYPE)
        self.check_numeric(f, (x,))

    def test_multi_op_prologue_and_epilogue(self):
        """Prologue does mul+add+relu, epilogue does log1p+clamp."""
        import torch.nn.functional as F

        B, D, G = 64, 4096, 128

        def f(x, weight, bias, scale):
            x = F.rms_norm(x, (D,), weight)
            x_scaled = torch.ops._inductor_test.realize(
                torch.relu(x * scale + bias)
            )
            amax = x_scaled.view(B, D // G, G).abs().amax(dim=-1)
            return torch.clamp(torch.log1p(amax), min=0.0, max=10.0)

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        bias = torch.randn(D, device=GPU_TYPE)
        scale = torch.randn(D, device=GPU_TYPE)
        self.check_numeric(f, (x, w, bias, scale))
        self.check_fusion()

    @inductor_config.patch(emulate_precision_casts=True)
    def test_fullres_epilogue_with_multiple_outputs(self):
        """Full-res epilogue producing both FP8 output and a second derived output."""
        import torch.nn.functional as F

        B, D, G = 64, 4096, 128
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
        self.check_numeric(f, (x, w))
        self.check_fusion()

    def test_grouped_reduction_with_weight_mul(self):
        """Grouped reduction input involves element-wise weight multiply."""
        import torch.nn.functional as F

        B, D, G = 128, 4096, 32

        def f(x, weight, group_weight):
            x = F.rms_norm(x, (D,), weight)
            # Weight multiply before grouped reduction
            weighted = x * group_weight
            return weighted.view(B, D // G, G).abs().amax(dim=-1)

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        gw = torch.randn(D, device=GPU_TYPE)
        self.check_numeric(f, (x, w, gw))
        self.check_fusion()

    # ---- Producer-consumer ----

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

    @inductor_config.patch(emulate_precision_casts=True)
    def test_producer_consumer_rmsnorm_fp8_quant(self):
        """RMS norm + amax + scale + full-res quantize epilogue."""
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
        self.check_numeric(f, (x, w))
        self.check_fusion()

    @inductor_config.patch(emulate_precision_casts=True)
    def test_producer_consumer_residual_rmsnorm_fp8_quant(self):
        B, D, G = 128, 2048, 128
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        fp8_min_scale = 1.0 / (fp8_max * 512.0)

        def f(x, residual, weight):
            h = x.float() + residual.float()
            variance = h.pow(2).mean(dim=-1, keepdim=True)
            normed = h * torch.rsqrt(variance + 1e-6)
            normed_bf16 = normed.to(torch.bfloat16) * weight
            grouped = normed_bf16.view(B, D // G, G)
            absmax = grouped.abs().amax(dim=-1, keepdim=True).float()
            scales = (absmax / fp8_max).clamp(min=fp8_min_scale)
            x_scaled = (grouped / scales).clamp(-fp8_max, fp8_max)
            x_fp8 = x_scaled.to(torch.float8_e4m3fn).view(B, D)
            return x_fp8.float(), scales.squeeze(-1)

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        residual = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        w = torch.randn(D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_numeric(f, (x, residual, w))
        self.check_fusion()

    @inductor_config.patch(emulate_precision_casts=True)
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
        self.check_numeric(f, (x, w))
        self.check_fusion()

    @inductor_config.patch({"combo_kernels": True, "emulate_precision_casts": True})
    def test_combo_kernels_skip_nested_reductions(self):
        import torch.nn.functional as F

        B, D, G = 8, 512, 128
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        def quant(x, weight):
            x = F.rms_norm(x, (D,), weight)
            x_groups = x.view(B, D // G, G)
            amax = x_groups.abs().amax(dim=-1)
            scale = (amax / fp8_max).clamp(min=1e-12)
            x_fp8 = (x_groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
            return x_fp8.view(B, D).float(), scale

        def f(x0, w0, x1, w1):
            return quant(x0, w0), quant(x1, w1)

        x0 = torch.randn(B, D, device=GPU_TYPE)
        x1 = torch.randn(B, D, device=GPU_TYPE)
        w0 = torch.randn(D, device=GPU_TYPE)
        w1 = torch.randn(D, device=GPU_TYPE)
        self.check_numeric(f, (x0, w0, x1, w1))
        if inductor_config.triton.nested_reduction:
            self.assertEqual(metrics.codegen_nested_reduction, 2)
            self.assertEqual(metrics.generated_kernel_count, 2)

    @parametrize("B", [1, 128])
    def test_producer_consumer_rmsnorm_nvfp4_inline_asm(self, B):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("NVFP4 inline asm requires SM100+")

        import torch.nn.functional as F

        D, G = 4096, 16

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
                asm_str="{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}",
                constraints="=r,f,f",
                dtype=torch.int32,
                is_pure=True,
                pack=1,
            )
            return packed.to(torch.uint8).view(B, D // 2), scale.view(B, D // G)

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        w = torch.randn(D, device=GPU_TYPE, dtype=torch.bfloat16)

        with inductor_config.patch("triton.nested_reduction", False):
            ref = torch.compile(f, fullgraph=True)(x, w)
        torch._dynamo.reset()
        metrics.reset()

        act = torch.compile(f, fullgraph=True)(x, w)
        self.assertEqual(act[0], ref[0])
        self.assertEqual(act[1].float(), ref[1].float(), atol=1e-2, rtol=1e-2)
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

    def test_epilogue_rejects_intermediate_dependency(self):
        """Do not fuse a pointwise epilogue before another dependent node."""
        from torch._inductor.scheduler import FusedNestedReductions
        import torch.nn.functional as F

        B, D, G = 64, 4096, 128

        def f(x, weight):
            x = F.rms_norm(x, (D,), weight)
            amax = x.view(B, D // G, G).abs().amax(dim=-1)
            row_sum = torch.ops._inductor_test.realize(
                amax.sum(dim=-1, keepdim=True)
            )
            return amax + row_sum

        saw_nested_reduction = False

        def check_reduction_fusion(nodes):
            nonlocal saw_nested_reduction
            fused_nodes = [n for n in nodes if isinstance(n, FusedNestedReductions)]
            self.assertEqual(len(fused_nodes), 1)
            saw_nested_reduction = True
            node2_pointwise = [
                sn for sn in fused_nodes[0].node2.get_nodes() if not sn.is_reduction()
            ]
            self.assertEqual(node2_pointwise, [])
            return nodes

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        with inductor_config.patch(
            _post_fusion_custom_pass=check_reduction_fusion,
            fx_graph_cache=False,
        ):
            self.check_numeric(f, (x, w))
        self.assertTrue(saw_nested_reduction)
        self.check_fusion(expected_kernels=None)

    # ---- Fusion rejection: patterns that must NOT use nested reduction ----

    def test_reject_non_power_of_2_group_size(self):
        """group_size=17 is not power of 2 — must not fuse."""

        def f(x):
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            x_norm = x / rms
            return x_norm.reshape(4, -1, 17).abs().amax(dim=-1)

        x = torch.randn(4, 17 * 16, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        if inductor_config.triton.nested_reduction:
            self.assertEqual(metrics.codegen_nested_reduction, 0)

    def test_reject_large_group_size(self):
        """group_size=2048 exceeds MAX_SMALL_REDUCTION — must not fuse."""

        def f(x):
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            x_norm = x / rms
            return x_norm.reshape(4, -1, 2048).abs().amax(dim=-1)

        x = torch.randn(4, 4096, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        if inductor_config.triton.nested_reduction:
            self.assertEqual(metrics.codegen_nested_reduction, 0)


    def test_reject_grouped_argmax(self):
        """arg reductions need value/index tuple handling."""

        def f(x):
            x_norm = _rmsnorm(x)
            return x_norm.reshape(4, -1, 128).argmax(dim=-1)

        x = torch.randn(4, 4096, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        if inductor_config.triton.nested_reduction:
            self.assertEqual(metrics.codegen_nested_reduction, 0)

    def test_reject_grouped_var(self):
        """Welford reductions need multi-accumulator handling."""

        def f(x):
            x_norm = _rmsnorm(x)
            return x_norm.reshape(4, -1, 128).var(dim=-1, correction=0)

        x = torch.randn(4, 4096, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        if inductor_config.triton.nested_reduction:
            self.assertEqual(metrics.codegen_nested_reduction, 0)

    def test_reject_split_reduction_interaction(self):
        """B=1 with large D triggers split_reductions, breaking the nested pair."""

        def f(x, w):
            B, K, D = x.shape
            x_flat = x.reshape(B * K, D)
            rms = torch.sqrt(
                torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6
            )
            x_normed = (x_flat / rms).reshape(B, K, D)
            return (w[:, :, None] * x_normed).sum(dim=1)

        # D=16384 causes outer reduction to split -> nested fusion rejected
        x = torch.randn(1, 16, 16384, device=GPU_TYPE)
        w = torch.randn(1, 16, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        if inductor_config.triton.nested_reduction:
            self.assertEqual(metrics.codegen_nested_reduction, 0)


class NestedReductionTest(_NestedReductionBase, TestBase):
    force_persistent_outer_reduction = True


class NestedReductionNonPersistentTest(_NestedReductionBase, TestBase):
    force_persistent_outer_reduction = False


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
