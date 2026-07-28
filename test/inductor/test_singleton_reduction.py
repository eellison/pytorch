# Owner(s): ["module: inductor"]

import warnings
from unittest import mock

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.fx_passes import singleton_reduction
from torch._inductor.fx_passes.singleton_reduction import (
    _has_expanding_pointwise_consumer,
    _materialization_plan,
    _MAX_MATERIALIZATION_OPS,
    _ReductionDimAnalyzer,
    _Singleton,
    _ZeroKind,
    eliminate_singleton_reductions,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    TEST_CUDA,
)


LIVE_VOCAB = 4096


class SingletonReductionTests(TestCase):
    @parametrize("low_dtype", (torch.bfloat16, torch.float16))
    @parametrize(
        "force_shape_pad,expected_fold",
        ((False, True), (True, False)),
    )
    def test_singleton_reduction_elimination(
        self, device, low_dtype, force_shape_pad, expected_fold
    ):
        vocab = LIVE_VOCAB

        def fn(target, scale):
            active = target != -100
            safe_target = torch.where(active, target, 0)
            iota = torch.arange(vocab, device=target.device).view(1, vocab)
            one_hot = torch.where(
                iota == safe_target.expand(-1, vocab),
                torch.tensor(-2.0, device=target.device),
                torch.tensor(0.0, device=target.device),
            )
            row_value = torch.where(active, scale, 0.0)
            dense = (row_value * one_hot).to(low_dtype).to(torch.float32)
            row_sum = dense.sum(dim=1, keepdim=True)
            return dense, row_sum, dense - row_sum

        target_values = [-100, -1, 0, vocab - 1, vocab, vocab + 2]
        scale_values = [
            0.0,
            -0.0,
            0.3333,
            float("inf"),
            -float("inf"),
            float("nan"),
        ]
        target = torch.tensor(target_values, device=device).repeat_interleave(
            len(scale_values)
        )[:, None]
        scale = torch.tensor(scale_values, device=device).repeat(
            len(target_values)
        )[:, None]

        with config.patch(
            singleton_reduction_elimination=False,
            pattern_matcher=False,
        ):
            expected = torch.compile(fn, fullgraph=True)(target, scale)
        counters.clear()
        with config.patch(
            force_shape_pad=force_shape_pad,
            force_disable_caches=True,
            pattern_matcher=False,
        ):
            actual, (code,) = run_and_get_code(
                torch.compile(fn, fullgraph=True), target, scale
            )
        self.assertEqual(actual, expected, equal_nan=True)
        self.assertEqual(torch.signbit(actual[1]), torch.signbit(expected[1]))
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"],
            int(expected_fold),
        )
        checker = FileCheck()
        if expected_fold:
            checker.check_not("rnumel")
        else:
            checker.check("rnumel")
        checker.run(code)

    @parametrize("miss,expected_fold", ((0.0, True), (0.5, False)))
    def test_singleton_reduction_elimination_requires_zero_miss(
        self, device, miss, expected_fold
    ):
        def fn(target, scale):
            iota = torch.arange(LIVE_VOCAB, device=target.device).reshape(
                1, LIVE_VOCAB
            )
            selected = torch.where(
                target.expand(-1, LIVE_VOCAB) == iota, 3.0, miss
            )
            dense = (selected * scale).to(torch.bfloat16).to(torch.float32)
            row_sum = dense.sum(dim=-1, keepdim=True)
            return dense, dense - row_sum

        target = torch.tensor(
            [[0], [LIVE_VOCAB - 1], [LIVE_VOCAB]], device=device
        )
        scale = torch.randn(3, 1, device=device)
        with config.patch(singleton_reduction_elimination=False):
            expected = torch.compile(fn, fullgraph=True)(target, scale)
        counters.clear()
        actual, (code,) = run_and_get_code(
            torch.compile(fn, fullgraph=True), target, scale
        )
        self.assertEqual(actual, expected)
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"],
            int(expected_fold),
        )
        checker = FileCheck()
        if expected_fold:
            checker.check_not("rnumel")
        else:
            checker.check("rnumel")
        checker.run(code)

    @parametrize("dense_dtype", (torch.float16, torch.bfloat16, torch.float64))
    @parametrize("index_dtype", (torch.int32, torch.int64))
    def test_singleton_reduction_elimination_dtypes(
        self, device, dense_dtype, index_dtype
    ):
        def fn(target, scale):
            iota = torch.arange(8, device=target.device, dtype=index_dtype).view(1, 8)
            dense = torch.where(target == iota, -scale, 0.0).to(dense_dtype)
            return dense.sum(dim=1, keepdim=True)

        target = torch.tensor([[0], [7], [8]], device=device, dtype=index_dtype)
        scale = torch.tensor([[1 / 3], [-2.0], [5.0]], device=device)
        with config.patch(singleton_reduction_elimination=False):
            expected = torch.compile(fn, fullgraph=True)(target, scale)
        torch._dynamo.reset()
        counters.clear()
        actual = torch.compile(fn, fullgraph=True)(target, scale)

        self.assertEqual(actual, expected)
        self.assertEqual(counters["inductor"]["singleton_reduction_elimination"], 1)

    def test_singleton_reduction_elimination_rejects_partial_pointwise(self, device):
        def fn(target):
            iota = torch.arange(8, device=target.device).view(1, 8)
            mask = target == iota
            denominator = torch.where(mask, 0, 1)
            quotient = torch.div(
                torch.ones_like(denominator), denominator, rounding_mode="floor"
            )
            dense = (quotient * torch.where(mask, 1, 0)).float()
            return dense.sum(dim=1, keepdim=True)

        target = torch.tensor([[-1], [8]], device=device)
        counters.clear()
        actual, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), target)

        self.assertEqual(actual, fn(target))
        self.assertEqual(counters["inductor"]["singleton_reduction_elimination"], 0)
        FileCheck().check("rnumel").run(code)

    @parametrize("dense_is_output,expected_fold", ((False, True), (True, False)))
    def test_singleton_reduction_elimination_requires_profitable_consumer(
        self, device, dense_is_output, expected_fold
    ):
        def fn(target, scale):
            iota = torch.arange(128, device=target.device).reshape(1, 128)
            dense = torch.where(target == iota, -scale, 0.0).to(
                torch.bfloat16
            ).to(torch.float32)
            row_sum = dense.sum(dim=1, keepdim=True)
            return (dense, row_sum) if dense_is_output else row_sum

        target = torch.tensor([[0], [127], [128]], device=device)
        scale = torch.randn(3, 1, device=device)
        with config.patch(singleton_reduction_elimination=False):
            expected = torch.compile(fn, fullgraph=True)(target, scale)
        counters.clear()
        actual, code = run_and_get_code(
            torch.compile(fn, fullgraph=True), target, scale
        )
        self.assertEqual(actual, expected)
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"],
            int(expected_fold),
        )
        checker = FileCheck()
        if expected_fold:
            checker.check_not("rnumel")
        else:
            checker.check("rnumel")
        checker.run("\n".join(code))

    @parametrize(
        "cast_path,expected_fold",
        (
            ("none", False),
            ("bfloat16_round_trip", True),
            ("float16_round_trip", True),
            ("float64_to_float32", False),
            ("bfloat16_output", False),
        ),
    )
    def test_singleton_reduction_elimination_live_reuse_scope(
        self, device, cast_path, expected_fold
    ):
        vocab = LIVE_VOCAB * 2 if cast_path == "bfloat16_output" else LIVE_VOCAB

        def fn(target, scale):
            iota = torch.arange(vocab, device=target.device).reshape(1, vocab)
            dense = torch.where(target == iota, -scale, 0.0)
            if cast_path == "bfloat16_round_trip":
                dense = dense.to(torch.bfloat16).to(torch.float32)
            elif cast_path == "float16_round_trip":
                dense = dense.to(torch.float16).to(torch.float32)
            elif cast_path == "float64_to_float32":
                dense = dense.to(torch.float64).to(torch.float32)
            elif cast_path == "bfloat16_output":
                dense = dense.to(torch.bfloat16)
            row_sum = dense.sum(dim=1, keepdim=True)
            return dense, dense - row_sum

        target = torch.tensor(
            [[0], [vocab - 1], [vocab]], device=device
        )
        scale = torch.randn(3, 1, device=device)
        with config.patch(singleton_reduction_elimination=False):
            expected = torch.compile(fn, fullgraph=True)(target, scale)
        counters.clear()
        actual, (code,) = run_and_get_code(
            torch.compile(fn, fullgraph=True), target, scale
        )
        self.assertEqual(actual, expected)
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"],
            int(expected_fold),
        )
        checker = FileCheck()
        if expected_fold:
            checker.check_not("rnumel")
        else:
            checker.check("rnumel")
        checker.run(code)

    @parametrize("dense_is_live,expected_fold", ((False, True), (True, False)))
    def test_singleton_reduction_elimination_hip_scope(
        self, device, dense_is_live, expected_fold
    ):
        def fn(target, scale):
            iota = torch.ops.prims.iota.default(
                LIVE_VOCAB,
                start=0,
                step=1,
                dtype=torch.int64,
                device=target.device,
                requires_grad=False,
            ).reshape(1, LIVE_VOCAB)
            dense = torch.where(target == iota, -scale, 0.0)
            dense = torch.ops.prims.convert_element_type.default(
                dense, torch.bfloat16
            )
            dense = torch.ops.prims.convert_element_type.default(
                dense, torch.float32
            )
            row_sum = dense.sum(dim=1, keepdim=True)
            return (dense, dense - row_sum) if dense_is_live else row_sum

        target = torch.tensor([[0], [LIVE_VOCAB]], device=device)
        scale = torch.randn(2, 1, device=device)
        graph = make_fx(fn)(target, scale).graph
        self.assertEqual(eliminate_singleton_reductions(graph), 1)

        graph = make_fx(fn)(target, scale).graph
        with mock.patch.object(torch.version, "hip", "test"):
            self.assertEqual(
                eliminate_singleton_reductions(graph), int(expected_fold)
            )

    @parametrize("zero_or_nan_miss", (False, True))
    def test_singleton_reduction_elimination_deep_pointwise_chain(
        self, device, zero_or_nan_miss
    ):
        def fn(target, scale):
            iota = torch.ops.prims.iota.default(
                8,
                start=0,
                step=1,
                dtype=torch.int64,
                device=target.device,
                requires_grad=False,
            ).reshape(1, 8)
            dense = torch.where(target == iota, -1.0, 0.0)
            count = _MAX_MATERIALIZATION_OPS + 1 if zero_or_nan_miss else 1200
            for _ in range(count):
                dense = dense * scale if zero_or_nan_miss else -dense
            return dense.sum(dim=1, keepdim=True)

        target = torch.tensor([[0], [8]], device=device)
        scale = torch.randn(2, 1, device=device)
        graph = make_fx(fn)(target, scale).graph
        reduction = next(
            node
            for node in graph.nodes
            if node.target is torch.ops.aten.sum.dim_IntList
        )
        dense = reduction.args[0]
        self.assertIsInstance(dense, torch.fx.Node)
        analyzer = _ReductionDimAnalyzer(reduction, 1, 8)
        singleton = analyzer.analyze_subgraph(dense)

        self.assertIsInstance(singleton, _Singleton)
        materialized_values = (
            singleton.hit
            if singleton.miss_zero_kind is _ZeroKind.EXACT
            else (singleton.hit, singleton.miss)
        )
        self.assertIsNone(
            _materialization_plan(
                materialized_values, _MAX_MATERIALIZATION_OPS
            )
        )
        self.assertEqual(eliminate_singleton_reductions(graph), 0)

    def test_singleton_reduction_elimination_analysis_budget(self, device):
        def fn(target):
            iota = torch.ops.prims.iota.default(
                8,
                start=0,
                step=1,
                dtype=torch.int64,
                device=target.device,
                requires_grad=False,
            ).reshape(1, 8)
            dense = torch.where(target == iota, -1.0, 0.0)
            for _ in range(100):
                dense = -dense
            return dense.sum(dim=1, keepdim=True)

        target = torch.tensor([[0], [8]], device=device)
        graph = make_fx(fn)(target).graph
        reduction = next(
            node
            for node in graph.nodes
            if node.target is torch.ops.aten.sum.dim_IntList
        )
        dense = reduction.args[0]
        self.assertIsInstance(dense, torch.fx.Node)
        analyzer = _ReductionDimAnalyzer(reduction, 1, 8)

        self.assertIsNone(analyzer.analyze_subgraph(dense, max_nodes=20))
        self.assertEqual(analyzer.nodes_visited, 21)

    def test_singleton_reduction_elimination_forward_analysis_budget(self, device):
        def fn(x):
            value = x.sum(dim=1, keepdim=True)
            for _ in range(100):
                value = -value
            return value.expand_as(x) + 1

        x = torch.randn(2, 8, device=device)
        graph = make_fx(fn)(x).graph
        reduction = next(
            node
            for node in graph.nodes
            if node.target is torch.ops.aten.sum.dim_IntList
        )

        result, nodes_visited = _has_expanding_pointwise_consumer(reduction, 20)
        self.assertIsNone(result)
        self.assertEqual(nodes_visited, 21)

    def test_singleton_reduction_elimination_shared_analysis_budget(self, device):
        vocab = LIVE_VOCAB

        def fn(target, scale):
            iota = torch.ops.prims.iota.default(
                vocab,
                start=0,
                step=1,
                dtype=torch.int64,
                device=target.device,
                requires_grad=False,
            ).reshape(1, vocab)
            dense_values = []
            row_sums = []
            for i in range(2):
                dense = torch.where(target == iota, -float(i + 1), 0.0) * scale[i]
                dense = torch.ops.prims.convert_element_type.default(
                    dense, torch.bfloat16
                )
                dense = torch.ops.prims.convert_element_type.default(
                    dense, torch.float32
                )
                dense_values.append(dense)
                row_sums.append(dense.sum(dim=1, keepdim=True))

            row_value = row_sums[0] + row_sums[1]
            dense_value = dense_values[0] + dense_values[1]
            for _ in range(30):
                row_value = -row_value
                dense_value = -dense_value
            return dense_value, row_value.expand(-1, vocab) + 1

        target = torch.tensor([[0], [vocab]], device=device)
        scale = torch.randn(2, 2, 1, device=device)
        graph = make_fx(fn)(target, scale).graph
        self.assertEqual(eliminate_singleton_reductions(graph), 2)

        graph = make_fx(fn)(target, scale).graph
        with mock.patch.object(singleton_reduction, "_MAX_ANALYSIS_NODES", 100):
            self.assertLess(eliminate_singleton_reductions(graph), 2)

    def test_singleton_reduction_elimination_prunes_uniform_producer(self, device):
        def fn(target, scale):
            for _ in range(100):
                scale = -scale
            scale = torch.where(scale > 0, scale, -scale)
            iota = torch.ops.prims.iota.default(
                8,
                start=0,
                step=1,
                dtype=torch.int64,
                device=target.device,
                requires_grad=False,
            ).reshape(1, 8)
            dense = torch.where(target == iota, -1.0, 0.0) * scale
            return dense.sum(dim=1, keepdim=True)

        target = torch.tensor([[0], [8]], device=device)
        scale = torch.randn(2, 1, device=device)
        graph = make_fx(fn)(target, scale).graph
        reduction = next(
            node
            for node in graph.nodes
            if node.target is torch.ops.aten.sum.dim_IntList
        )
        dense = reduction.args[0]
        self.assertIsInstance(dense, torch.fx.Node)
        analyzer = _ReductionDimAnalyzer(reduction, 1, 8)
        self.assertIsNotNone(analyzer.analyze_subgraph(dense))

        uniform_where = next(
            node
            for node in graph.nodes
            if node.target is torch.ops.aten.where.self
            and node.meta["val"].shape == scale.shape
        )
        self.assertIn(uniform_where, analyzer.memo)
        uniform_inputs = uniform_where.all_input_nodes
        self.assertTrue(
            all(input_node not in analyzer.memo for input_node in uniform_inputs)
        )

    @parametrize(
        "chain,expected_fold",
        (("shared_dag", True), ("long", False)),
    )
    def test_singleton_reduction_elimination_live_materialization_bound(
        self, device, chain, expected_fold
    ):
        vocab = LIVE_VOCAB

        def fn(target):
            iota = torch.arange(vocab, device=target.device).reshape(1, vocab)
            one_hot = torch.where(target == iota, -1.0, 0.0)
            dense = one_hot
            count = 4 if chain == "shared_dag" else 40
            for _ in range(count):
                dense = dense * one_hot if chain == "shared_dag" else -dense
            dense = dense.to(torch.bfloat16).to(torch.float32)
            row_sum = dense.sum(dim=1, keepdim=True)
            return dense, dense - row_sum

        target = torch.tensor([[0], [vocab]], device=device)
        with config.patch(singleton_reduction_elimination=False):
            expected = torch.compile(fn, fullgraph=True)(target)
        counters.clear()
        actual, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), target)
        self.assertEqual(actual, expected)
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"],
            int(expected_fold),
        )
        checker = FileCheck()
        if expected_fold:
            checker.check_not("rnumel")
        else:
            checker.check("rnumel")
        checker.run(code)

    def test_singleton_reduction_elimination_rejects_small_live_row(self, device):
        def fn(target):
            iota = torch.arange(128, device=target.device).reshape(1, 128)
            dense = (
                torch.where(target == iota, -1.0, 0.0)
                .to(torch.bfloat16)
                .to(torch.float32)
            )
            row_sum = dense.sum(dim=1, keepdim=True)
            return dense, dense - row_sum

        target = torch.tensor([[0], [128]], device=device)
        counters.clear()
        actual, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), target)
        self.assertEqual(actual, fn(target))
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"], 0
        )
        FileCheck().check("rnumel").run(code)

    def test_singleton_reduction_elimination_force_shape_pad_dense_dead(self, device):
        def fn(target):
            iota = torch.arange(8, device=target.device).reshape(1, 8)
            dense = torch.where(target == iota, -1.0, 0.0)
            return dense.sum(dim=1, keepdim=True)

        target = torch.tensor([[0], [8]], device=device)
        counters.clear()
        with config.patch(force_shape_pad=True):
            actual, (code,) = run_and_get_code(
                torch.compile(fn, fullgraph=True), target
            )
        self.assertEqual(actual, fn(target))
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"], 1
        )
        FileCheck().check_not("rnumel").run(code)

    def test_singleton_reduction_elimination_rejects_varying_value(self, device):
        def fn(target, value):
            iota = torch.arange(8, device=target.device).reshape(1, 8)
            dense = torch.where(target == iota, value, 0.0)
            return dense.sum(dim=1, keepdim=True)

        target = torch.randint(-1, 9, (3, 1), device=device)
        value = torch.randn(3, 8, device=device)
        counters.clear()
        actual, (code,) = run_and_get_code(
            torch.compile(fn, fullgraph=True), target, value
        )
        self.assertEqual(actual, fn(target, value))
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"], 0
        )
        FileCheck().check("rnumel").run(code)

    def test_singleton_reduction_elimination_rejects_varying_product(self, device):
        def fn(target, value):
            iota = torch.arange(8, device=target.device).reshape(1, 8)
            one_hot = torch.where(target == iota, 1.0, 0.0)
            dense = one_hot * value
            return dense.sum(dim=1, keepdim=True)

        target = torch.tensor([[0], [3]], device=device)
        value = torch.ones(2, 8, device=device)
        value[0, 1] = float("inf")
        counters.clear()
        actual, (code,) = run_and_get_code(
            torch.compile(fn, fullgraph=True), target, value
        )
        self.assertEqual(actual, fn(target, value), equal_nan=True)
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"], 0
        )
        FileCheck().check("rnumel").run(code)

    def test_singleton_reduction_elimination_rejects_wrapping_iota(self, device):
        def fn(target):
            iota = torch.ops.prims.iota.default(
                129,
                start=0,
                step=1,
                dtype=torch.int8,
                device=target.device,
                requires_grad=False,
            ).reshape(1, 129)
            dense = torch.where(target == iota, 1.0, 0.0)
            return dense.sum(dim=1, keepdim=True)

        target = torch.tensor([[-128], [0]], dtype=torch.int8, device=device)
        counters.clear()
        actual, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), target)
        self.assertEqual(actual, fn(target))
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"], 0
        )
        FileCheck().check("rnumel").run(code)

    @parametrize(
        "index_dtype,length,target_values",
        (
            (torch.int8, 128, (-128, -1, 0, 127)),
            (torch.uint8, 128, (0, 127, 128, 255)),
            (torch.uint8, 256, (0, 255)),
        ),
    )
    def test_singleton_reduction_elimination_index_dtype_boundary(
        self, device, index_dtype, length, target_values
    ):
        def fn(target):
            iota = torch.arange(length, device=target.device, dtype=index_dtype).view(
                1, length
            )
            dense = torch.where(target == iota, 1.0, 0.0)
            return dense.sum(dim=1, keepdim=True)

        target = torch.tensor(target_values, device=device, dtype=index_dtype)[:, None]
        with config.patch(singleton_reduction_elimination=False):
            expected = torch.compile(fn, fullgraph=True)(target)
        counters.clear()
        actual, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), target)

        self.assertEqual(actual, expected)
        self.assertEqual(counters["inductor"]["singleton_reduction_elimination"], 1)
        FileCheck().check_not("rnumel").run(code)

    def test_singleton_reduction_elimination_rejects_reindexed_iota(self, device):
        def fn(target):
            iota = torch.arange(8, device=target.device)
            reindexed = torch.as_strided(iota, (1, 8), (0, 0))
            dense = torch.where(target == reindexed, 1.0, 0.0)
            return dense.sum(dim=1, keepdim=True)

        target = torch.tensor([[0], [1]], device=device)
        counters.clear()
        actual, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), target)
        self.assertEqual(actual, fn(target))
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"], 0
        )
        FileCheck().check("rnumel").run(code)

    def test_singleton_reduction_elimination_rejects_rank_zero_sum(self, device):
        def fn(x):
            return x.sum(dim=0, keepdim=True)

        x = torch.randn((), device=device)
        counters.clear()
        actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(actual, fn(x))
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"], 0
        )

    def test_singleton_reduction_elimination_rejects_nan_to_int(self, device):
        def fn(target, scale):
            iota = torch.arange(8, device=target.device).reshape(1, 8)
            one_hot = torch.where(target == iota, -1.0, 0.0)
            dense = (one_hot * scale).to(torch.int64).to(torch.float32)
            return dense.sum(dim=1, keepdim=True)

        target = torch.tensor([[0], [8]], device=device)
        scale = torch.full((2, 1), float("inf"), device=device)
        counters.clear()
        actual, (code,) = run_and_get_code(
            torch.compile(fn, fullgraph=True), target, scale
        )
        self.assertEqual(actual, fn(target, scale))
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"], 0
        )
        FileCheck().check("rnumel").run(code)

    def test_singleton_reduction_elimination_masks_nan_to_int_hit(self, device):
        def fn(target, scale):
            iota = torch.arange(8, device=target.device).reshape(1, 8)
            dense = torch.where(target == iota, scale, 0.0)
            dense = dense.to(torch.int64).to(torch.float32)
            return dense.sum(dim=1, keepdim=True)

        target = torch.tensor([[8], [9]], device=device)
        scale = torch.tensor([[float("nan")], [1e30]], device=device)
        with config.patch(singleton_reduction_elimination=False):
            expected = torch.compile(fn, fullgraph=True)(target, scale)
        torch._dynamo.reset()
        counters.clear()
        actual, (code,) = run_and_get_code(
            torch.compile(fn, fullgraph=True), target, scale
        )

        self.assertEqual(actual, expected)
        self.assertEqual(counters["inductor"]["singleton_reduction_elimination"], 1)
        FileCheck().check_not("rnumel").run(code)

    def test_singleton_reduction_elimination_preserves_type_promotion(self, device):
        def fn(target, scale):
            iota = torch.arange(LIVE_VOCAB, device=target.device).reshape(
                1, LIVE_VOCAB
            )
            one_hot = torch.where(target == iota, -1.0, 0.0)
            dense = (one_hot * scale).to(torch.bfloat16).to(torch.float32)
            row_sum = dense.sum(dim=1, keepdim=True)
            return dense, dense - row_sum

        target = torch.tensor([[0], [LIVE_VOCAB]], device=device)
        scale = torch.tensor([[torch.iinfo(torch.int64).min], [1]], device=device)
        with config.patch(singleton_reduction_elimination=False):
            expected = torch.compile(fn, fullgraph=True)(target, scale)
        counters.clear()
        actual, (code,) = run_and_get_code(
            torch.compile(fn, fullgraph=True), target, scale
        )
        self.assertEqual(actual, expected)
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"], 1
        )
        FileCheck().check_not("rnumel").run(code)

    @parametrize("scalar_device", ("cpu", "cuda"))
    def test_singleton_reduction_elimination_preserves_scalar_promotion(
        self, device, scalar_device
    ):
        def fn(target, scale, scalar):
            iota = torch.arange(LIVE_VOCAB, device=target.device).reshape(
                1, LIVE_VOCAB
            )
            one_hot = torch.where(target == iota, 1.0, 0.0)
            dense = ((one_hot * scalar) * scale).to(torch.bfloat16).to(
                torch.float32
            )
            row_sum = dense.sum(dim=1, keepdim=True)
            return dense, dense - row_sum

        target = torch.tensor([[0], [LIVE_VOCAB]], device=device)
        scale = torch.tensor([[1 / 3], [1]], dtype=torch.float16, device=device)
        scalar = torch.tensor(3.12345, dtype=torch.float32, device=scalar_device)
        with config.patch(singleton_reduction_elimination=False):
            expected = torch.compile(fn, fullgraph=True)(target, scale, scalar)
        counters.clear()
        actual, (code,) = run_and_get_code(
            torch.compile(fn, fullgraph=True), target, scale, scalar
        )
        self.assertEqual(actual, expected)
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"], 1
        )
        FileCheck().check_not("rnumel").run(code)

    def test_singleton_reduction_elimination_preserves_uniform_view_shape(self, device):
        def fn(target, scale):
            iota = torch.arange(LIVE_VOCAB, device=target.device).reshape(
                1, 1, LIVE_VOCAB
            )
            one_hot = torch.where(target == iota, -1.0, 0.0)
            dense = (one_hot * scale.view(3, 1, 1)).to(torch.bfloat16).to(
                torch.float32
            )
            row_sum = dense.sum(dim=2, keepdim=True)
            return dense, dense - row_sum

        target = torch.tensor(
            [[[0]], [[LIVE_VOCAB - 1]], [[LIVE_VOCAB]]], device=device
        )
        scale = torch.randn(3, 1, device=device)
        with config.patch(singleton_reduction_elimination=False):
            expected = torch.compile(fn, fullgraph=True)(target, scale)
        counters.clear()
        actual, (code,) = run_and_get_code(
            torch.compile(fn, fullgraph=True), target, scale
        )
        self.assertEqual(actual, expected)
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"], 1
        )
        FileCheck().check_not("rnumel").run(code)

    def test_singleton_reduction_elimination_propagates_pointwise(self, device):
        def fn(target, scale):
            iota = torch.arange(8, device=target.device).reshape(1, 8)
            one_hot = torch.where(target == iota, -1.0, 0.0)
            dense = one_hot.abs() * one_hot * scale
            row_sum = dense.sum(dim=1, keepdim=True)
            return row_sum

        target = torch.tensor([[0], [3], [8]], device=device)
        scale = torch.randn(3, 1, device=device)
        with config.patch(singleton_reduction_elimination=False):
            expected = torch.compile(fn, fullgraph=True)(target, scale)
        counters.clear()
        actual, (code,) = run_and_get_code(
            torch.compile(fn, fullgraph=True), target, scale
        )
        self.assertEqual(actual, expected)
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"], 1
        )
        FileCheck().check_not("rnumel").run(code)

    def test_singleton_reduction_elimination_rejects_unmodeled_recompute(self, device):
        def fn(target, scale):
            iota = torch.arange(LIVE_VOCAB, device=target.device).reshape(
                1, LIVE_VOCAB
            )
            one_hot = torch.where(target == iota, -1.0, 0.0)
            dense = ((one_hot + 2) * one_hot * scale).to(torch.bfloat16).to(
                torch.float32
            )
            row_sum = dense.sum(dim=1, keepdim=True)
            return dense, dense - row_sum

        target = torch.tensor([[0], [3], [LIVE_VOCAB]], device=device)
        scale = torch.randn(3, 1, device=device)
        with config.patch(singleton_reduction_elimination=False):
            expected = torch.compile(fn, fullgraph=True)(target, scale)
        counters.clear()
        actual, (code,) = run_and_get_code(
            torch.compile(fn, fullgraph=True), target, scale
        )
        self.assertEqual(actual, expected)
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"], 0
        )
        FileCheck().check("rnumel").run(code)

    def test_singleton_reduction_elimination_rejects_different_indices(self, device):
        def fn(target_a, target_b):
            iota = torch.ops.prims.iota.default(
                8,
                start=0,
                step=1,
                dtype=torch.int64,
                device=target_a.device,
                requires_grad=False,
            ).reshape(1, 8)
            a = torch.where(target_a == iota, -1.0, 0.0)
            b = torch.where(target_b == iota, -1.0, 0.0)
            return (a + b).sum(dim=1, keepdim=True)

        def single_fn(target):
            iota = torch.ops.prims.iota.default(
                8,
                start=0,
                step=1,
                dtype=torch.int64,
                device=target.device,
                requires_grad=False,
            ).reshape(1, 8)
            return torch.where(target == iota, -1.0, 0.0).sum(
                dim=1, keepdim=True
            )

        target_a = torch.tensor([[0], [8]], device=device)
        target_b = torch.tensor([[1], [8]], device=device)
        graph = make_fx(single_fn)(target_a).graph
        self.assertEqual(eliminate_singleton_reductions(graph), 1)

        graph = make_fx(fn)(target_a, target_b).graph
        self.assertEqual(eliminate_singleton_reductions(graph), 0)
        graph.lint()

    def test_singleton_reduction_elimination_rejects_shared_dense_sums(self, device):
        def make_dense(target):
            iota = torch.ops.prims.iota.default(
                8,
                start=0,
                step=1,
                dtype=torch.int64,
                device=target.device,
                requires_grad=False,
            ).reshape(1, 8)
            return torch.where(target == iota, -1.0, 0.0)

        def fn(target):
            dense = make_dense(target)
            return (
                dense.sum(dim=1, keepdim=True),
                dense.sum(dim=1, keepdim=True),
            )

        def single_fn(target):
            return make_dense(target).sum(dim=1, keepdim=True)

        target = torch.tensor([[0], [8]], device=device)
        graph = make_fx(single_fn)(target).graph
        self.assertEqual(eliminate_singleton_reductions(graph), 1)

        graph = make_fx(fn)(target).graph
        self.assertEqual(eliminate_singleton_reductions(graph), 0)
        graph.lint()

    def test_singleton_reduction_elimination_rejects_expanded_outer_shape(self, device):
        def fn(target, scale):
            iota = torch.arange(8, device=target.device).reshape(1, 8)
            one_hot = torch.where(target == iota, -1.0, 0.0)
            dense = (one_hot * scale).to(torch.bfloat16).to(torch.float32)
            row_sum = dense.sum(dim=1, keepdim=True)
            return dense, dense - row_sum

        target = torch.tensor([[3]], device=device)
        scale = torch.randn(4, 1, device=device)
        with config.patch(singleton_reduction_elimination=False):
            expected = torch.compile(fn, fullgraph=True)(target, scale)
        counters.clear()
        actual, (code,) = run_and_get_code(
            torch.compile(fn, fullgraph=True), target, scale
        )
        self.assertEqual(actual, expected)
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"], 0
        )
        FileCheck().check("rnumel").run(code)

    def test_singleton_reduction_elimination_dynamic_batch(self, device):
        def fn(target, scale):
            iota = torch.arange(LIVE_VOCAB, device=target.device).reshape(
                1, LIVE_VOCAB, 1
            )
            dense = (
                torch.where(target == iota, -scale, 0.0)
                .to(torch.bfloat16)
                .to(torch.float32)
            )
            row_sum = dense.sum(dim=1, keepdim=True)
            return dense - row_sum

        inputs = [
            (
                torch.randint(-1, LIVE_VOCAB + 1, (batch, 1, 2), device=device),
                torch.randn(batch, 1, 2, device=device),
            )
            for batch in (3, 5)
        ]
        with config.patch(singleton_reduction_elimination=False):
            expected_fn = torch.compile(fn, fullgraph=True, dynamic=True)
            expected = [expected_fn(*args) for args in inputs]

        torch._dynamo.reset()
        counters.clear()
        compiled = torch.compile(fn, fullgraph=True, dynamic=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = [compiled(*args) for args in inputs]
        self.assertEqual(actual, expected)
        self.assertFalse(
            any("Raw SymInt value" in str(warning.message) for warning in caught)
        )
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"], 1
        )

    def test_singleton_reduction_elimination_rejects_dynamic_extent(self, device):
        def fn(target, shape_source):
            length = shape_source.shape[1]
            iota = torch.arange(length, device=target.device).view(1, length)
            dense = torch.where(target == iota, -1.0, 0.0)
            return dense.sum(dim=1, keepdim=True)

        inputs = [
            (
                torch.tensor([[0], [length]], device=device),
                torch.empty(2, length, device=device),
            )
            for length in (8, 16)
        ]
        counters.clear()
        compiled = torch.compile(fn, fullgraph=True, dynamic=True)
        actual = [compiled(*args) for args in inputs]

        self.assertEqual(actual, [fn(*args) for args in inputs])
        self.assertEqual(counters["inductor"]["singleton_reduction_elimination"], 0)

    @parametrize(
        "with_downstream_reduction,expected_fold", ((False, True), (True, False))
    )
    def test_singleton_reduction_elimination_downstream_reduction(
        self, device, with_downstream_reduction, expected_fold
    ):
        def fn(target, scale):
            iota = torch.arange(LIVE_VOCAB, device=target.device).view(
                1, LIVE_VOCAB
            )
            dense = (
                torch.where(target.expand(-1, LIVE_VOCAB) == iota, -scale, 0.0)
                .to(torch.bfloat16)
                .to(torch.float32)
            )
            row_sum = dense.sum(dim=1, keepdim=True)
            adjusted = dense - row_sum
            if with_downstream_reduction:
                other = (dense + scale).reshape(3, 512, 8)
                return dense, adjusted, other.sum(dim=0)
            return dense, adjusted

        target = torch.tensor(
            [[0], [LIVE_VOCAB - 1], [LIVE_VOCAB]], device=device
        )
        scale = torch.randn(3, 1, device=device)
        with config.patch(singleton_reduction_elimination=False):
            expected = torch.compile(fn, fullgraph=True)(target, scale)
        counters.clear()
        actual, code = run_and_get_code(
            torch.compile(fn, fullgraph=True), target, scale
        )
        self.assertEqual(actual, expected)
        self.assertEqual(
            counters["inductor"]["singleton_reduction_elimination"],
            int(expected_fold),
        )
        checker = FileCheck()
        if expected_fold:
            checker.check_not("rnumel")
        else:
            checker.check("rnumel")
        checker.run("\n".join(code))


instantiate_device_type_tests(SingletonReductionTests, globals(), only_for="cuda")


if __name__ == "__main__":
    if TEST_CUDA:
        run_tests()
