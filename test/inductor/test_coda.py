# Owner(s): ["module: inductor"]

import unittest
from unittest import mock

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.source import LocalSource
from torch._dynamo.utils import counters
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.fx_passes import coda
from torch._inductor.fx_passes.joint_graph import pass_patterns
from torch._inductor.kernel.flex_gemm.epilogue import analyze_flex_gemm_epilogue
from torch._inductor.pattern_matcher import joint_fwd_bwd
from torch._inductor.utils import run_and_get_code, run_fw_bw_and_get_code
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM100OrLater, SM80OrLater
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


def rms_norm_block(x, w0, residual, gamma, w1):
    hidden = torch.mm(x, w0) + residual
    normalized = torch.nn.functional.rms_norm(hidden, (hidden.shape[-1],), gamma, 1e-5)
    return torch.mm(normalized, w1)


class TestCodaRMSNorm(TestCase):
    def setUp(self):
        super().setUp()
        counters.clear()

    def _make_args(self, device, requires_grad=False):
        m, k, hidden, output = 128, 64, 1024, 128
        return tuple(
            torch.randn(
                shape,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=requires_grad,
            )
            for shape in (
                (m, k),
                (k, hidden),
                (m, hidden),
                (hidden,),
                (hidden, output),
            )
        )

    def _trace_and_apply(
        self, fn, args, device, *, training=False, flex_available=True
    ):
        coda._coda_init(torch.device(device))
        with (
            mock.patch.object(
                coda, "_is_nvidia_sm100_or_later", return_value=flex_available
            ),
            mock.patch.object(
                coda, "ensure_cute_available", return_value=flex_available
            ),
        ):
            if training:
                graph = joint_fwd_bwd(fn, args)
            else:
                graph = make_fx(fn, select_decomp_table(), tracing_mode="fake")(*args)
            graph.graph.eliminate_dead_code()
            count = sum(patterns.apply(graph.graph) for patterns in pass_patterns)
        graph.graph.lint()
        graph.recompile()
        return graph, count

    def _assert_not_rewritten(self, fn, args, device):
        graph, count = self._trace_and_apply(fn, args, device)
        flex_nodes = graph.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.flex_gemm
        )
        self.assertEqual(count, 0)
        self.assertEqual(len(flex_nodes), 0)

    def _assert_delayed_scale(self, graph):
        for node in graph.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mul.Tensor
        ):
            for scale, projected in (node.args[:2], reversed(node.args[:2])):
                if not (
                    isinstance(scale, torch.fx.Node)
                    and scale.target is torch.ops.aten.rsqrt.default
                    and isinstance(projected, torch.fx.Node)
                    and projected.target is torch.ops.prims.convert_element_type.default
                    and isinstance(projected.args[0], torch.fx.Node)
                    and projected.args[0].target is torch.ops.aten.mm.default
                ):
                    continue
                return
        self.fail("RMS scale was not moved after the second GEMM")

    @inductor_config.patch(coda_rms_norm_rewrite=True)
    @parametrize("residual_first", (False, True))
    def test_aten_joint_pattern_rewrites_inference_graph(self, device, residual_first):
        def fn(x, w0, residual, gamma, w1):
            hidden = (
                residual + torch.mm(x, w0)
                if residual_first
                else torch.mm(x, w0) + residual
            )
            normalized = torch.nn.functional.rms_norm(
                hidden, (hidden.shape[-1],), gamma, 1e-5
            )
            return torch.mm(normalized, w1)

        graph, count = self._trace_and_apply(fn, self._make_args(device), device)
        flex_nodes = graph.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.flex_gemm
        )

        self.assertEqual(count, 1)
        self.assertEqual(len(flex_nodes), 0)
        self._assert_delayed_scale(graph)

    @inductor_config.patch(coda_rms_norm_rewrite=True)
    def test_aten_joint_pattern_rewrites_training_graph(self, device):
        graph, count = self._trace_and_apply(
            rms_norm_block,
            self._make_args(device, requires_grad=True),
            device,
            training=True,
        )
        outputs = next(node for node in graph.graph.nodes if node.op == "output")

        self.assertEqual(count, 1)
        self.assertEqual(len(outputs.args[0]), 6)
        self._assert_delayed_scale(graph)

    @inductor_config.patch(coda_rms_norm_fusion=True)
    def test_flex_gemm_joint_pattern_rewrites_inference_graph(self, device):
        graph, count = self._trace_and_apply(
            rms_norm_block, self._make_args(device), device
        )
        flex_nodes = graph.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.flex_gemm
        )
        first_body = graph.get_submodule(flex_nodes[0].args[1].target)
        local_reduce = analyze_flex_gemm_epilogue(first_body).outputs.local_reduce

        self.assertEqual(count, 2)
        self.assertEqual(len(flex_nodes), 2)
        self.assertIsNotNone(local_reduce)
        self.assertEqual(local_reduce.match.geometry.group, 16)
        self.assertEqual(local_reduce.match.geometry.axis, 1)

    @inductor_config.patch(coda_rms_norm_fusion=True)
    def test_flex_gemm_rewrites_training_joint_graph(self, device):
        graph, count = self._trace_and_apply(
            rms_norm_block,
            self._make_args(device, requires_grad=True),
            device,
            training=True,
        )
        flex_nodes = graph.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.flex_gemm
        )
        first_body = graph.get_submodule(flex_nodes[0].args[1].target)
        second_body = graph.get_submodule(flex_nodes[1].args[1].target)
        first_outputs = analyze_flex_gemm_epilogue(first_body).outputs
        second_outputs = analyze_flex_gemm_epilogue(second_body).outputs
        outputs = next(node for node in graph.graph.nodes if node.op == "output")

        self.assertEqual(count, 2)
        self.assertEqual(len(flex_nodes), 2)
        self.assertEqual(len(outputs.args[0]), 6)
        self.assertIsNotNone(first_outputs.local_reduce)
        self.assertEqual(len(first_outputs.aux_outputs), 1)
        self.assertEqual(len(second_outputs.aux_outputs), 1)
        self.assertEqual(first_outputs.aux_outputs[0].meta["val"].dtype, torch.float32)
        self.assertEqual(second_outputs.aux_outputs[0].meta["val"].dtype, torch.float32)

    @unittest.skipIf(not SM80OrLater, "BF16 GEMM requires SM80+")
    def test_replacement_numerics(self, device):
        args = self._make_args(device)
        expected = rms_norm_block(*args)
        aten_actual = coda._rms_norm_mm_reassociation(*args, 1e-5)
        flex_actual = coda._rms_norm_mm_flex_replacement(*args, 1e-5)

        for actual in (aten_actual, flex_actual):
            error = actual.float() - expected.float()
            relative_l2 = error.norm() / expected.float().norm()
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertLess(relative_l2.item(), 0.005)

    @unittest.skipIf(not SM80OrLater, "BF16 GEMM requires SM80+")
    @inductor_config.patch(
        coda_rms_norm_rewrite=True,
        fx_graph_cache=False,
        fx_graph_remote_cache=False,
    )
    def test_aten_rewrite_compiles(self, device):
        torch._dynamo.reset()
        args = self._make_args(device)
        torch._dynamo.mark_dynamic(args[0], 0, min=128, max=256)
        torch._dynamo.mark_dynamic(args[2], 0, min=128, max=256)
        expected = rms_norm_block(*args)
        compiled = torch.compile(rms_norm_block, backend="inductor", fullgraph=True)
        actual, (code,) = run_and_get_code(compiled, *args)

        error = actual.float() - expected.float()
        relative_l2 = error.norm() / expected.float().norm()
        self.assertLess(relative_l2.item(), 0.005)
        self.assertGreaterEqual(counters["inductor"]["coda_rms_norm_rewrite"], 1)
        FileCheck().check_not("flex_gemm_epilogue(").run(code)

        second_args = (
            torch.randn(256, 64, device=device, dtype=torch.bfloat16),
            args[1],
            torch.randn(256, 1024, device=device, dtype=torch.bfloat16),
            args[3],
            args[4],
        )
        second_expected = rms_norm_block(*second_args)
        second_actual = compiled(*second_args)
        second_error = second_actual.float() - second_expected.float()
        second_relative_l2 = second_error.norm() / second_expected.float().norm()
        self.assertLess(second_relative_l2.item(), 0.005)

    @unittest.skipIf(not SM80OrLater, "BF16 GEMM requires SM80+")
    @inductor_config.patch(
        coda_rms_norm_rewrite=True,
        fx_graph_cache=False,
        fx_graph_remote_cache=False,
    )
    def test_aten_training_rewrite_compiles(self, device):
        torch._dynamo.reset()
        args = self._make_args(device)
        expected_args = tuple(arg.detach().clone().requires_grad_() for arg in args)
        actual_args = tuple(arg.detach().clone().requires_grad_() for arg in args)
        tangent = torch.randn(128, 128, device=device, dtype=torch.bfloat16)
        expected = rms_norm_block(*expected_args)
        expected_grads = torch.autograd.grad(
            expected, expected_args, grad_outputs=tangent
        )

        actual = torch.compile(rms_norm_block, backend="inductor", fullgraph=True)(
            *actual_args
        )
        actual_grads = torch.autograd.grad(actual, actual_args, grad_outputs=tangent)

        output_error = actual.float() - expected.float()
        output_relative_l2 = output_error.norm() / expected.float().norm()
        self.assertLess(output_relative_l2.item(), 0.005)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            error = actual_grad.float() - expected_grad.float()
            relative_l2 = error.norm() / expected_grad.float().norm()
            self.assertLess(relative_l2.item(), 0.01)
        self.assertGreaterEqual(counters["inductor"]["coda_rms_norm_rewrite"], 1)

    @skipIfNoCuteDSL
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @inductor_config.patch(coda_rms_norm_fusion=True)
    def test_flex_gemm_rewrite_compiles(self, device):
        args = self._make_args(device)
        torch._dynamo.mark_dynamic(args[0], 0, min=128, max=256)
        torch._dynamo.mark_dynamic(args[2], 0, min=128, max=256)
        expected = rms_norm_block(*args)
        compiled = torch.compile(rms_norm_block, backend="inductor", fullgraph=True)
        actual, (code,) = run_and_get_code(compiled, *args)

        error = actual.float() - expected.float()
        relative_l2 = error.norm() / expected.float().norm()
        self.assertLess(relative_l2.item(), 0.01)
        self.assertEqual(code.count("flex_gemm_epilogue("), 2)
        FileCheck().check("local_reduce=FlexGemmRuntimeLocalReducePlan").run(code)

        second_args = (
            torch.randn(256, 64, device=device, dtype=torch.bfloat16),
            args[1],
            torch.randn(256, 1024, device=device, dtype=torch.bfloat16),
            args[3],
            args[4],
        )
        second_expected = rms_norm_block(*second_args)
        second_actual = compiled(*second_args)
        second_error = second_actual.float() - second_expected.float()
        second_relative_l2 = second_error.norm() / second_expected.float().norm()
        self.assertLess(second_relative_l2.item(), 0.01)

    @skipIfNoCuteDSL
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @inductor_config.patch(coda_rms_norm_fusion=True)
    def test_flex_gemm_training_rewrite_compiles(self, device):
        expected_args = self._make_args(device, requires_grad=True)
        actual_args = tuple(
            arg.detach().clone().requires_grad_() for arg in expected_args
        )
        expected = rms_norm_block(*expected_args)
        expected.sum().backward()

        compiled = torch.compile(rms_norm_block, backend="inductor", fullgraph=True)
        actual, code = run_fw_bw_and_get_code(lambda: compiled(*actual_args))

        output_error = actual.float() - expected.float()
        output_relative_l2 = output_error.norm() / expected.float().norm()
        self.assertLess(output_relative_l2.item(), 0.01)
        for actual_arg, expected_arg in zip(actual_args, expected_args):
            error = actual_arg.grad.float() - expected_arg.grad.float()
            relative_l2 = error.norm() / expected_arg.grad.float().norm()
            self.assertLess(relative_l2.item(), 0.02)
        self.assertEqual(sum(src.count("flex_gemm_epilogue(") for src in code), 2)

    @inductor_config.patch(coda_rms_norm_rewrite=True)
    def test_back_to_back_rms_norm_is_not_rewritten(self, device):
        def fn(x, w0, residual, gamma, w1, output_gamma):
            projected = rms_norm_block(x, w0, residual, gamma, w1)
            return torch.nn.functional.rms_norm(
                projected, (projected.shape[-1],), output_gamma, 1e-5
            )

        output_gamma = torch.randn(128, device=device, dtype=torch.bfloat16)
        args = (*self._make_args(device), output_gamma)
        self._assert_not_rewritten(fn, args, device)

    @inductor_config.patch(coda_rms_norm_rewrite=True)
    def test_fusible_reduction_consumer_is_not_rewritten(self, device):
        def fn(x, w0, residual, gamma, w1):
            output = rms_norm_block(x, w0, residual, gamma, w1)
            return output.float().square().sum(dim=-1)

        self._assert_not_rewritten(fn, self._make_args(device), device)

    @inductor_config.patch(coda_rms_norm_rewrite=True)
    @parametrize("pointwise_count,expected_count", ((8, 0), (9, 1)))
    def test_fusible_reduction_consumer_walk_is_bounded(
        self, device, pointwise_count, expected_count
    ):
        def fn(x, w0, residual, gamma, w1):
            output = rms_norm_block(x, w0, residual, gamma, w1)
            for _ in range(pointwise_count):
                output = torch.sigmoid(output)
            return output.sum(dim=-1)

        graph, count = self._trace_and_apply(fn, self._make_args(device), device)
        self.assertEqual(count, expected_count)
        if expected_count:
            self._assert_delayed_scale(graph)

    @inductor_config.patch(coda_rms_norm_rewrite=True)
    def test_pointwise_consumer_is_rewritten(self, device):
        def fn(x, w0, residual, gamma, w1):
            return torch.sigmoid(rms_norm_block(x, w0, residual, gamma, w1))

        graph, count = self._trace_and_apply(fn, self._make_args(device), device)
        self.assertEqual(count, 1)
        self._assert_delayed_scale(graph)

    @inductor_config.patch(coda_rms_norm_rewrite=True)
    def test_following_gemm_is_not_rewritten(self, device):
        def fn(x, w0, residual, gamma, w1, w2):
            projected = rms_norm_block(x, w0, residual, gamma, w1)
            return torch.mm(projected, w2)

        w2 = torch.randn(128, 64, device=device, dtype=torch.bfloat16)
        args = (*self._make_args(device), w2)
        self._assert_not_rewritten(fn, args, device)

    @inductor_config.patch(coda_rms_norm_fusion=True)
    def test_unavailable_flex_backend_keeps_aten_rewrite(self, device):
        graph, count = self._trace_and_apply(
            rms_norm_block,
            self._make_args(device),
            device,
            flex_available=False,
        )
        flex_nodes = graph.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.flex_gemm
        )

        self.assertEqual(count, 1)
        self.assertEqual(len(flex_nodes), 0)
        self._assert_delayed_scale(graph)

    @inductor_config.patch(coda_rms_norm_fusion=True)
    def test_unsupported_flex_residual_broadcast_keeps_aten_rewrite(self, device):
        args = list(self._make_args(device))
        args[2] = torch.randn(1, 1, device=device, dtype=torch.bfloat16)
        graph, count = self._trace_and_apply(rms_norm_block, tuple(args), device)
        flex_nodes = graph.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.flex_gemm
        )

        self.assertEqual(count, 1)
        self.assertEqual(len(flex_nodes), 0)
        self._assert_delayed_scale(graph)

    @inductor_config.patch(coda_rms_norm_fusion=True)
    @parametrize("residual_shape", ((1, 1024), (128, 1)))
    def test_supported_flex_residual_broadcast(self, device, residual_shape):
        args = list(self._make_args(device))
        args[2] = torch.randn(residual_shape, device=device, dtype=torch.bfloat16)
        graph, count = self._trace_and_apply(rms_norm_block, tuple(args), device)
        flex_nodes = graph.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.flex_gemm
        )

        self.assertEqual(count, 2)
        self.assertEqual(len(flex_nodes), 2)

    @parametrize("use_flex_gemm", (False, True))
    def test_dynamic_batch_adds_no_guards(self, device, use_flex_gemm):
        coda._coda_init(torch.device(device))
        real_args = self._make_args(device)
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(
            allow_non_fake_inputs=True,
            shape_env=shape_env,
        )
        x = fake_mode.from_tensor(
            real_args[0],
            source=LocalSource("x", is_input=True),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC],
                dynamic_strides=[DimDynamic.INFER_STRIDE] * 2,
            ),
        )

        static_args = []
        for index, arg in enumerate((real_args[1], real_args[3], real_args[4])):
            static_args.append(
                fake_mode.from_tensor(
                    arg,
                    source=LocalSource(f"arg{index}", is_input=True),
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[DimDynamic.STATIC] * arg.dim(),
                        dynamic_strides=[DimDynamic.INFER_STRIDE] * arg.dim(),
                    ),
                )
            )

        with fake_mode:
            residual = torch.empty(
                (x.shape[0], 1024), device=device, dtype=torch.bfloat16
            )
            graph = make_fx(rms_norm_block, select_decomp_table())(
                x, static_args[0], residual, static_args[1], static_args[2]
            )
        graph.graph.eliminate_dead_code()
        guards_before = list(shape_env.get_nontrivial_guards())

        with (
            inductor_config.patch(
                coda_rms_norm_rewrite=not use_flex_gemm,
                coda_rms_norm_fusion=use_flex_gemm,
            ),
            mock.patch.object(coda, "_is_nvidia_sm100_or_later", return_value=True),
            mock.patch.object(coda, "ensure_cute_available", return_value=True),
            fake_mode,
        ):
            count = sum(patterns.apply(graph.graph) for patterns in pass_patterns)

        graph.graph.lint()
        graph.recompile()
        self.assertEqual(count, 2 if use_flex_gemm else 1)
        self.assertEqual(
            len(
                graph.graph.find_nodes(
                    op="call_function", target=torch.ops.higher_order.flex_gemm
                )
            ),
            2 if use_flex_gemm else 0,
        )
        self.assertEqual(shape_env.get_nontrivial_guards(), guards_before)


instantiate_device_type_tests(TestCodaRMSNorm, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
