# Owner(s): ["module: inductor"]
from types import SimpleNamespace

import sympy

import torch
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.loop_body import LoopBody
from torch._inductor.virtualized import ops, V
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import MockGraphHandler


class DtypeGraph(MockGraphHandler):
    def __init__(self, dtypes):
        super().__init__()
        self.dtypes = dtypes

    def get_dtype(self, buffer_name):
        return self.dtypes.get(buffer_name, torch.float32)


class TestPersistentReductionLiveness(TestCase):
    @staticmethod
    def make_features(fn):
        i = sympy.Symbol("i", integer=True)
        r = sympy.Symbol("r", integer=True)
        body = LoopBody(
            fn,
            ([i], [r]),
            {i: sympy.Integer(64), r: sympy.Integer(128)},
            [i],
            [r],
        )
        scheduler_node = type("SchedulerNodeStub", (), {"_body": body})()
        features = SIMDKernelFeatures(
            [scheduler_node],
            sympy.Integer(64),
            sympy.Integer(128),
        )
        return features

    def test_ordered_last_use_and_cache(self):
        def body(index, rindex):
            offset = index[0] * 128 + rindex[0]
            value = ops.load("input", offset)
            squared = value * value
            combined = value + squared
            result = ops.reduction(
                torch.float32, torch.bfloat16, "sum", combined
            )
            ops.store_reduction("output", index[0], result)

        with V.set_graph_handler(DtypeGraph({"input": torch.bfloat16})):
            features = self.make_features(body)
            self.assertEqual(features.persistent_reduction_live_tile_words(), 3)

            features.node_schedule.clear()
            self.assertEqual(features.persistent_reduction_live_tile_words(), 3)

    def test_reduction_state_words(self):
        def run(reduction_type, value_fn, expected):
            def body(index, rindex):
                offset = index[0] * 128 + rindex[0]
                value = ops.load("input", offset)
                result = ops.reduction(
                    torch.float32,
                    torch.float32,
                    reduction_type,
                    value_fn(value),
                )
                if isinstance(result, tuple):
                    for output_index, item in enumerate(result):
                        ops.store_reduction(
                            f"output{output_index}", index[0], item
                        )
                else:
                    ops.store_reduction("output", index[0], result)

            features = self.make_features(body)
            self.assertEqual(
                features.persistent_reduction_live_tile_words(), expected
            )

        with V.set_graph_handler(DtypeGraph({"input": torch.float32})):
            run("sum", lambda value: value, 2)
            run("welford_reduce", lambda value: value, 4)
            run("online_softmax_reduce", lambda value: (value, value), 3)

    def test_two_live_welford_states_are_high_pressure(self):
        def body(index, rindex):
            offset = index[0] * 128 + rindex[0]
            first = ops.load("first", offset)
            first_state = ops.reduction(
                torch.float32, torch.float32, "welford_reduce", first
            )
            second = ops.load("second", offset)
            second_state = ops.reduction(
                torch.float32, torch.float32, "welford_reduce", second
            )
            for state_index, item in enumerate(first_state + second_state):
                ops.store_reduction(f"output{state_index}", index[0], item)

        with V.set_graph_handler(
            DtypeGraph({"first": torch.float32, "second": torch.float32})
        ):
            features = self.make_features(body)
            self.assertEqual(features.persistent_reduction_live_tile_words(), 7)

    def test_64_bit_values_use_two_words(self):
        def body(index, rindex):
            offset = index[0] * 128 + rindex[0]
            value = ops.load("input", offset)
            result = ops.reduction(torch.float64, torch.float64, "sum", value)
            ops.store_reduction("output", index[0], result)

        with V.set_graph_handler(DtypeGraph({"input": torch.float64})):
            features = self.make_features(body)
            self.assertEqual(features.persistent_reduction_live_tile_words(), 4)

    def test_reduction_varying_index_expr_uses_dtype_width(self):
        def body(index, rindex):
            value = ops.index_expr(rindex[0], torch.int64)
            result = ops.reduction(torch.int64, torch.int64, "sum", value)
            ops.store_reduction("output", index[0], result)

        with V.set_graph_handler(DtypeGraph({})):
            features = self.make_features(body)
            self.assertEqual(features.persistent_reduction_live_tile_words(), 4)

    def test_uses_propagated_dtype_metadata(self):
        def body(index, rindex):
            offset = index[0] * 128 + rindex[0]
            value = ops.load("input", offset)
            converted = ops.identity(value)
            result = ops.reduction(
                torch.float64, torch.float64, "sum", converted
            )
            ops.store_reduction("output", index[0], result)

        with V.set_graph_handler(DtypeGraph({"input": torch.float32})):
            features = self.make_features(body)
            graph = features.node_schedule[0]._body.root_block.graph
            identity = graph.find_nodes(op="call_method", target="identity")[0]
            identity.meta["opt_ctx"] = SimpleNamespace(dtype=torch.float64)
            self.assertEqual(features.persistent_reduction_live_tile_words(), 4)

    def test_masked_subblock_fails_closed(self):
        def body(index, rindex):
            offset = index[0] * 128 + rindex[0]
            mask = ops.index_expr(rindex[0] < 64, torch.bool)
            value = ops.masked(
                mask,
                lambda: ops.load("input", offset),
                0.0,
            )
            result = ops.reduction(torch.float32, torch.float32, "sum", value)
            ops.store_reduction("output", index[0], result)

        with V.set_graph_handler(DtypeGraph({"input": torch.float32})):
            features = self.make_features(body)
            self.assertIsNone(features.persistent_reduction_live_tile_words())

    def test_indirect_and_sort_fail_closed(self):
        def indirect_body(index, rindex):
            offset = index[0] * 128 + rindex[0]
            loaded_index = ops.load("indices", offset)
            indirect_index = ops.indirect_indexing(loaded_index, 128)
            value = ops.load("input", indirect_index)
            result = ops.reduction(torch.float32, torch.float32, "sum", value)
            ops.store_reduction("output", index[0], result)

        def sort_body(index, rindex):
            offset = index[0] * 128 + rindex[0]
            value = ops.load("input", offset)
            sorted_value = ops.sort(
                (torch.float32,), (value,), False, False
            )[0]
            result = ops.reduction(
                torch.float32, torch.float32, "sum", sorted_value
            )
            ops.store_reduction("output", index[0], result)

        with V.set_graph_handler(
            DtypeGraph({"indices": torch.int64, "input": torch.float32})
        ):
            self.assertIsNone(
                self.make_features(
                    indirect_body
                ).persistent_reduction_live_tile_words()
            )
            self.assertIsNone(
                self.make_features(sort_body).persistent_reduction_live_tile_words()
            )


if __name__ == "__main__":
    run_tests()
