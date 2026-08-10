# Owner(s): ["module: inductor"]

from unittest import mock

import torch
from torch import fx
from torch._inductor import config
from torch._inductor.fx_passes.dedupe_graph_outputs import (
    _compute_structural_classes,
    dedupe_graph_outputs_pass,
    is_output_computation_sharing_supported,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp


def _propagate(gm: fx.GraphModule, *inputs: torch.Tensor) -> None:
    mode = FakeTensorMode()
    fake_inputs = [mode.from_tensor(value) for value in inputs]
    FakeTensorProp(gm, mode=mode).propagate(*fake_inputs)


class TestDedupeGraphOutputs(TestCase):
    @staticmethod
    def _make_sin_outputs(count: int) -> fx.GraphModule:
        graph = fx.Graph()
        x = graph.placeholder("x")
        outputs = [
            graph.call_function(torch.ops.aten.sin.default, (x,))
            for _ in range(count)
        ]
        graph.output(tuple(outputs))
        gm = fx.GraphModule({}, graph)
        _propagate(gm, torch.randn(8))
        return gm

    def test_reversed_output_order_does_not_replace_internal_users(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        output_only = graph.call_function(torch.ops.aten.sin.default, (x,))
        used_internally = graph.call_function(torch.ops.aten.sin.default, (x,))
        internal = graph.call_function(torch.ops.aten.add.Tensor, (used_internally, 1))
        siblings = [
            graph.call_function(torch.ops.aten.sin.default, (x,)) for _ in range(6)
        ]
        graph.output((used_internally, output_only, *siblings, internal))
        gm = fx.GraphModule({}, graph)
        _propagate(gm, torch.randn(8))

        dedupe_graph_outputs_pass(graph)
        graph.lint()
        gm.recompile()

        result = gm(torch.randn(8))
        self.assertEqual(len({out.data_ptr() for out in result[:8]}), 8)
        torch.testing.assert_close(result[8], result[0] + 1)
        self.assertEqual(
            len(
                [
                    node
                    for node in graph.nodes
                    if node.target is torch.ops.aten.sin.default
                ]
            ),
            1,
        )
        self.assertEqual(
            len(
                [
                    node
                    for node in graph.nodes
                    if node.target is torch.ops.aten.clone.default
                ]
            ),
            7,
        )

    def test_sparse_output_metadata_fails_closed(self):
        mode = FakeTensorMode()
        sparse = torch.sparse_coo_tensor(
            torch.tensor([[0], [1]]), torch.tensor([1.0]), (2, 2)
        )
        fake_sparse = mode.from_tensor(sparse)

        graph = fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = fake_sparse
        outputs = []
        for _ in range(8):
            output = graph.call_function(torch.ops.aten.clone.default, (x,))
            output.meta["val"] = fake_sparse
            outputs.append(output)
        graph.output(tuple(outputs))

        dedupe_graph_outputs_pass(graph)
        graph.lint()
        self.assertEqual(
            len(
                [
                    node
                    for node in graph.nodes
                    if node.target is torch.ops.aten.clone.default
                ]
            ),
            8,
        )

    def test_unrelated_sparse_metadata_does_not_disable_pass(self):
        graph = fx.Graph()
        _unused = graph.placeholder("unused")
        x = graph.placeholder("x")
        outputs = [
            graph.call_function(torch.ops.aten.sin.default, (x,)) for _ in range(8)
        ]
        graph.output(tuple(outputs))
        gm = fx.GraphModule({}, graph)

        mode = FakeTensorMode()
        sparse = torch.sparse_coo_tensor(
            torch.tensor([[0], [1]]), torch.tensor([1.0]), (2, 2)
        )
        fake_sparse = mode.from_tensor(sparse)
        fake_x = mode.from_tensor(torch.randn(8))
        FakeTensorProp(gm, mode=mode).propagate(fake_sparse, fake_x)

        dedupe_graph_outputs_pass(graph)

        sin_nodes = [
            node for node in graph.nodes if node.target is torch.ops.aten.sin.default
        ]
        self.assertEqual(len(sin_nodes), 1)

    def test_repeated_output_reference_does_not_meet_branch_threshold(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        output = graph.call_function(torch.ops.aten.sin.default, (x,))
        graph.output((output,) * 8)
        gm = fx.GraphModule({}, graph)
        _propagate(gm, torch.randn(8))

        dedupe_graph_outputs_pass(graph)

        clone_nodes = [
            node for node in graph.nodes if node.target is torch.ops.aten.clone.default
        ]
        self.assertEqual(len(clone_nodes), 0)

    def test_branch_count_bounds(self):
        for count in (7, 33):
            with self.subTest(count=count):
                gm = self._make_sin_outputs(count)
                dedupe_graph_outputs_pass(gm.graph)

                sin_nodes = [
                    node
                    for node in gm.graph.nodes
                    if node.target is torch.ops.aten.sin.default
                ]
                clone_nodes = [
                    node
                    for node in gm.graph.nodes
                    if node.target is torch.ops.aten.clone.default
                ]
                self.assertEqual(len(sin_nodes), count)
                self.assertEqual(len(clone_nodes), 0)

    def test_random_outputs_are_not_shared(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        outputs = [
            graph.call_function(torch.ops.aten.rand_like.default, (x,))
            for _ in range(8)
        ]
        graph.output(tuple(outputs))
        gm = fx.GraphModule({}, graph)
        _propagate(gm, torch.randn(8))

        dedupe_graph_outputs_pass(graph)

        random_nodes = [
            node
            for node in graph.nodes
            if node.target is torch.ops.aten.rand_like.default
        ]
        self.assertEqual(len(random_nodes), 8)

    def test_uninitialized_outputs_are_not_shared(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        outputs = [
            graph.call_function(torch.ops.aten.empty_like.default, (x,))
            for _ in range(8)
        ]
        graph.output(tuple(outputs))
        gm = fx.GraphModule({}, graph)
        _propagate(gm, torch.randn(8))

        dedupe_graph_outputs_pass(graph)

        empty_nodes = [
            node
            for node in graph.nodes
            if node.target is torch.ops.aten.empty_like.default
        ]
        self.assertEqual(len(empty_nodes), 8)

    def test_collective_outputs_are_not_shared(self):
        import torch.distributed._functional_collectives

        graph = fx.Graph()
        x = graph.placeholder("x")
        outputs = [
            graph.call_function(
                torch.ops._c10d_functional.all_reduce.default,
                (x, "sum", "0"),
            )
            for _ in range(8)
        ]
        graph.output(tuple(outputs))
        gm = fx.GraphModule({}, graph)
        _propagate(gm, torch.randn(8))

        dedupe_graph_outputs_pass(graph)

        collective_nodes = [
            node
            for node in graph.nodes
            if node.target is torch.ops._c10d_functional.all_reduce.default
        ]
        clone_nodes = [
            node for node in graph.nodes if node.target is torch.ops.aten.clone.default
        ]
        self.assertEqual(len(collective_nodes), 8)
        self.assertEqual(len(clone_nodes), 0)

    def test_input_views_keep_alias_contract(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        outputs = [
            graph.call_function(torch.ops.aten.view.default, (x, (2, 4)))
            for _ in range(8)
        ]
        graph.output(tuple(outputs))
        gm = fx.GraphModule({}, graph)
        _propagate(gm, torch.randn(8))

        dedupe_graph_outputs_pass(graph)
        gm.recompile()

        inp = torch.randn(8)
        result = gm(inp)
        self.assertTrue(all(out.data_ptr() == inp.data_ptr() for out in result))

    def test_inter_output_aliases_keep_alias_contract(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        value = graph.call_function(torch.ops.aten.sin.default, (x,))
        outputs = [
            graph.call_function(torch.ops.aten.view.default, (value, (2, 4)))
            for _ in range(8)
        ]
        graph.output(tuple(outputs))
        gm = fx.GraphModule({}, graph)
        _propagate(gm, torch.randn(8))

        dedupe_graph_outputs_pass(graph)
        gm.recompile()

        result = gm(torch.randn(8))
        self.assertEqual(len({out.data_ptr() for out in result}), 1)
        clone_nodes = [
            node for node in graph.nodes if node.target is torch.ops.aten.clone.default
        ]
        self.assertEqual(len(clone_nodes), 0)

    def test_dense_output_clones_preserve_storage_and_strides(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        outputs = []
        for _ in range(8):
            value = graph.call_function(torch.ops.aten.sin.default, (x,))
            outputs.append(
                graph.call_function(torch.ops.aten.permute.default, (value, (1, 0)))
            )
        graph.output(tuple(outputs))
        gm = fx.GraphModule({}, graph)
        _propagate(gm, torch.randn(3, 5))

        dedupe_graph_outputs_pass(graph)
        graph.lint()
        gm.recompile()

        result = gm(torch.randn(3, 5))
        self.assertEqual(len({out.data_ptr() for out in result}), 8)
        self.assertTrue(all(out.stride() == (1, 5) for out in result))

    def test_nested_node_arguments_are_structurally_matched(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        outputs = []
        for _ in range(8):
            lhs = graph.call_function(torch.ops.aten.sin.default, (x,))
            rhs = graph.call_function(torch.ops.aten.cos.default, (x,))
            outputs.append(
                graph.call_function(torch.ops.aten.cat.default, ([lhs, rhs], 0))
            )
        graph.output(tuple(outputs))
        gm = fx.GraphModule({}, graph)
        _propagate(gm, torch.randn(8))

        dedupe_graph_outputs_pass(graph)

        cat_nodes = [
            node for node in graph.nodes if node.target is torch.ops.aten.cat.default
        ]
        self.assertEqual(len(cat_nodes), 1)

    def test_constant_comparison_is_type_sensitive(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        with_int = graph.call_function(torch.ops.aten.add.Tensor, (x, 1), {"alpha": 1})
        with_bool = graph.call_function(
            torch.ops.aten.add.Tensor, (x, 1), {"alpha": True}
        )

        classes = _compute_structural_classes(graph)
        self.assertNotEqual(classes[with_int], classes[with_bool])

    def test_deep_graph_classification_is_iterative(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        outputs = []
        for _ in range(8):
            value = x
            for _ in range(1100):
                value = graph.call_function(torch.ops.aten.sin.default, (value,))
            outputs.append(value)
        graph.output(tuple(outputs))

        classes = _compute_structural_classes(graph)
        self.assertEqual(len({classes[output] for output in outputs}), 1)

    def test_prims_cast_view_outputs_are_shared(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        outputs = []
        for _ in range(8):
            value = graph.call_function(
                torch.ops.prims.convert_element_type.default,
                (x, torch.bfloat16),
            )
            value = graph.call_function(torch.ops.aten.unsqueeze.default, (value, 0))
            value = graph.call_function(torch.ops.aten.view.default, (value, (1, 2, 4)))
            outputs.append(graph.call_function(torch.ops.aten.squeeze.dim, (value, 0)))
        graph.output(tuple(outputs))
        gm = fx.GraphModule({}, graph)
        _propagate(gm, torch.randn(2, 4))

        dedupe_graph_outputs_pass(graph)

        cast_nodes = [
            node
            for node in graph.nodes
            if node.target is torch.ops.prims.convert_element_type.default
        ]
        clone_nodes = [
            node for node in graph.nodes if node.target is torch.ops.aten.clone.default
        ]
        self.assertEqual(len(cast_nodes), 1)
        self.assertEqual(len(clone_nodes), 7)

    @config.patch("cuda_backend", "triton")
    def test_production_gate_uses_graph_device(self):
        mode = FakeTensorMode()
        with mode:
            fake_cuda = torch.empty(8, device="cuda:3")

        graph = fx.Graph()
        output = graph.placeholder("output")
        output.meta["val"] = fake_cuda
        graph.output((output,))
        gm = fx.GraphModule({}, graph)

        worker = mock.Mock()
        worker.get_device_properties.return_value = mock.Mock(major=10)
        device_interface = mock.Mock(Worker=worker)
        with mock.patch(
            "torch._inductor.fx_passes.dedupe_graph_outputs.get_interface_for_device",
            return_value=device_interface,
        ):
            self.assertTrue(is_output_computation_sharing_supported(gm))

        worker.get_device_properties.assert_called_once_with(torch.device("cuda:3"))

    @config.patch("cuda_backend", "triton")
    def test_production_gate_rejects_non_sm100(self):
        mode = FakeTensorMode()
        with mode:
            fake_cuda = torch.empty(8, device="cuda:2")

        graph = fx.Graph()
        output = graph.placeholder("output")
        output.meta["val"] = fake_cuda
        graph.output((output,))
        gm = fx.GraphModule({}, graph)

        worker = mock.Mock()
        worker.get_device_properties.return_value = mock.Mock(major=9)
        device_interface = mock.Mock(Worker=worker)
        with mock.patch(
            "torch._inductor.fx_passes.dedupe_graph_outputs.get_interface_for_device",
            return_value=device_interface,
        ):
            self.assertFalse(is_output_computation_sharing_supported(gm))


if __name__ == "__main__":
    run_tests()
