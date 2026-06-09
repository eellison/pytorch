# Owner(s): ["module: inductor"]

import contextlib
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.runtime.autotune_common import expected_autotune_queue_calls
from torch._inductor.runtime.autotune_queue import (
    _COORDESC_BENCHMARK_MODE_GROUPED,
    _COORDESC_BENCHMARK_MODE_UNGROUPED,
    _CoordinateDescentBatch,
    _CoordinateDescentCompileJob,
    _CoordinateDescentTask,
    _get_active_coordinate_descent_batch,
    coordinate_descent_batch,
)
from torch._inductor.runtime.coordinate_descent_tuner import CoordescTuner
from torch._inductor.runtime.hints import HeuristicType
from torch._inductor.runtime.triton_compat import OutOfResources
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._inductor.test_case import run_tests, TestCase


try:
    import triton  # @manual
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904


class TestAutotuneQueueCoordesc(TestCase):
    @staticmethod
    def _coordesc_batch_patch(**overrides):
        config_values = {
            "compile_threads": 2,
            "coordinate_descent_tuning": True,
            "coordinate_descent_tuning_batch": True,
        }
        config_values.update(overrides)
        return config.patch(config_values)

    def test_coordinate_descent_batch_frontier_candidate_chunks(self):
        counters.clear()
        with config.patch(
            {"coordinate_descent_tuning_batch_max_frontier_candidates": 2}
        ):
            chunks = list(
                coordinate_descent_batch()._pending_config_chunks(
                    [("task", idx) for idx in range(5)]
                )
            )

        self.assertEqual(
            chunks,
            [
                [("task", 0), ("task", 1)],
                [("task", 2), ("task", 3)],
                [("task", 4)],
            ],
        )
        self.assertGreaterEqual(
            counters["inductor"]["autotune_queue_frontier_splits"], 1
        )
        self.assertEqual(
            counters["inductor"]["autotune_queue_frontier_chunks"], 3
        )

    @staticmethod
    def _coordesc_test_config(xblock, yblock=None):
        kwargs = {"XBLOCK": xblock}
        if yblock is not None:
            kwargs["YBLOCK"] = yblock
        return triton.Config(kwargs, num_warps=4, num_stages=1)

    @staticmethod
    def _coordesc_test_key(config):
        return (
            tuple(sorted(config.kwargs.items())),
            config.num_warps,
            config.num_stages,
        )

    def _run_scalar_coordesc_test(
        self,
        baseline_config,
        timing,
        *,
        size_hints,
        inductor_meta=None,
        cached_configs=(),
        cached_benchmark_mode=None,
        compile_fails=lambda config: False,
        compile_failure_exception_factory=lambda: RuntimeError("compile failed"),
    ):
        tuner = CoordescTuner(
            name="scalar",
            size_hints=size_hints,
            inductor_meta=inductor_meta,
            frozen_fields={"num_warps"},
        )
        for cached_config in cached_configs:
            tuner.cache_benchmark_result(
                cached_config,
                timing(cached_config),
                benchmark_mode=cached_benchmark_mode,
            )

        calls = []

        def bench(config):
            calls.append(self._coordesc_test_key(config))
            if compile_fails(config):
                raise compile_failure_exception_factory()
            return timing(config)

        return tuner.autotune(bench, baseline_config), calls, tuner

    def _make_fake_coordesc_autotuner(
        self,
        name,
        baseline_config,
        timing,
        *,
        size_hints,
        inductor_meta=None,
        cached_configs=(),
        cached_benchmark_mode=None,
        compile_fails=lambda config: False,
        compile_failure_exception_factory=lambda: RuntimeError("compile failed"),
    ):
        test_case = self

        class Launcher:
            def __init__(self, config):
                self.config = config
                self.n_regs = 1
                self.n_spills = 0
                self.shared = 0
                self.cache_hash = f"{name}-{test_case._coordesc_test_key(config)}"
                self.store_cubin = False

        autotuner = object.__new__(CachingAutotuner)
        autotuner.fn = SimpleNamespace(
            __name__=name,
            src=f"def {name}():\n    pass\n",
            arg_names=[],
        )
        autotuner.lock = threading.Lock()
        autotuner.device_props = SimpleNamespace(type="cpu", index=None)
        autotuner.inductor_meta = inductor_meta or {}
        autotuner.triton_meta = {}
        autotuner.mutated_arg_names = []
        autotuner.launchers = [Launcher(baseline_config)]
        autotuner.compile_results = []
        autotuner._cached_launcher = None
        autotuner._ensure_kernel_loaded = lambda: None
        autotuner.get_device_interface = lambda: SimpleNamespace(
            is_available=lambda: False
        )
        autotuner.coordesc_tuner = CoordescTuner(
            name=name,
            size_hints=size_hints,
            inductor_meta=inductor_meta,
            frozen_fields={"num_warps"},
        )
        for cached_config in cached_configs:
            autotuner.coordesc_tuner.cache_benchmark_result(
                cached_config,
                timing(cached_config),
                benchmark_mode=cached_benchmark_mode,
            )

        class CompileResult:
            def __init__(self, config):
                self.config = config
                self.kernel = SimpleNamespace(
                    hash=f"{name}-{test_case._coordesc_test_key(config)}"
                )

            def make_launcher(self):
                return Launcher(self.config)

        def precompile_config(config):
            if compile_fails(config):
                raise compile_failure_exception_factory()
            return CompileResult(config)

        def submit_config_compile(configs):
            compiled_autotuner = SimpleNamespace(
                compile_results=[precompile_config(configs[0])],
                restore_after_unpickle=lambda old_values: None,
            )
            return SimpleNamespace(result=lambda: (compiled_autotuner, 1))

        def finish_coordesc(
            best_config, config2launcher, elapsed_ns, save_cache=True, **kwargs
        ):
            winner = config2launcher.get(best_config)
            if winner is None:
                winner = precompile_config(best_config).make_launcher()
            best_config.found_by_coordesc = True
            autotuner.launchers = [winner]
            return winner

        autotuner._precompile_config = precompile_config
        autotuner._config_compile_submitter = submit_config_compile
        autotuner._finish_coordinate_descent_tuning = finish_coordesc
        autotuner._coordesc_test_timing = timing
        autotuner._coordesc_test_calls = []
        return autotuner

    def _run_queued_coordesc_test(self, autotuners, config_overrides=None):
        task_states = []
        frontiers = []
        patch_config = {
            "compile_threads": 2,
            "coordinate_descent_tuning": True,
            "coordinate_descent_tuning_batch": True,
        }
        if config_overrides:
            patch_config.update(config_overrides)

        def benchmark_requests(requests):
            frontier = []
            timings = []
            for request in requests:
                if request.wait_ready_fn is not None:
                    request.wait_ready_fn()
                if request.setup_fn is not None:
                    request.setup_fn()
                key = self._coordesc_test_key(request.launcher.config)
                request.autotuner._coordesc_test_calls.append(key)
                frontier.append((request.autotuner.fn.__name__, key))
                timings.append(
                    request.autotuner._coordesc_test_timing(
                        request.launcher.config
                    )
                )
            frontiers.append(frontier)
            return timings

        with (
            config.patch(patch_config),
            patch(
                "torch._inductor.runtime.autotune_queue._benchmark_launcher_requests",
                side_effect=benchmark_requests,
            ),
            patch(
                "torch._inductor.runtime.triton_heuristics.red_text",
                side_effect=lambda text: text,
                create=True,
            ),
            coordinate_descent_batch(disposable_args=True) as batch,
        ):
            for autotuner in autotuners:
                queued = batch.enqueue(
                    autotuner,
                    autotuner.launchers[0],
                    (),
                    {},
                    None,
                    False,
                )
                self.assertTrue(queued)
                task = batch.tasks[-1]
                self.assertIsInstance(task, _CoordinateDescentTask)
                task_states.append(
                    (
                        autotuner.fn.__name__,
                        task.phase,
                        task.best_timing,
                    )
                )

        return frontiers, task_states

    def test_coordinate_descent_batch_matches_scalar_cached_baseline(self):
        def timing(config):
            return {
                2: 9.0,
                4: 10.0,
                8: 1.0,
                16: 2.0,
            }.get(config.kwargs["XBLOCK"], 100.0)

        scalar_baseline = self._coordesc_test_config(4)
        scalar_best, scalar_calls, _ = self._run_scalar_coordesc_test(
            scalar_baseline,
            timing,
            size_hints={"x": 16},
            cached_configs=(scalar_baseline,),
        )

        queued_baseline = self._coordesc_test_config(4)
        target = self._make_fake_coordesc_autotuner(
            "queued",
            queued_baseline,
            timing,
            size_hints={"x": 16},
            cached_configs=(queued_baseline,),
            cached_benchmark_mode=_COORDESC_BENCHMARK_MODE_GROUPED,
        )
        peer_baseline = self._coordesc_test_config(4)
        peer = self._make_fake_coordesc_autotuner(
            "peer",
            peer_baseline,
            timing,
            size_hints={"x": 16},
            cached_configs=(peer_baseline,),
            cached_benchmark_mode=_COORDESC_BENCHMARK_MODE_GROUPED,
        )

        _, task_states = self._run_queued_coordesc_test([target, peer])

        self.assertEqual(
            self._coordesc_test_key(target.launchers[0].config),
            self._coordesc_test_key(scalar_best),
        )
        self.assertEqual(target.launchers[0].config.kwargs["XBLOCK"], 8)
        self.assertNotIn(self._coordesc_test_key(scalar_baseline), scalar_calls)
        self.assertNotIn(
            self._coordesc_test_key(queued_baseline),
            target._coordesc_test_calls,
        )
        self.assertEqual(task_states[0], ("queued", "fields", 10.0))

    def test_coordinate_descent_batch_rebenchmarks_ungrouped_cache(self):
        def timing(config):
            return {
                2: 9.0,
                4: 10.0,
                8: 1.0,
            }.get(config.kwargs["XBLOCK"], 100.0)

        queued_baseline = self._coordesc_test_config(4)
        target = self._make_fake_coordesc_autotuner(
            "queued",
            queued_baseline,
            timing,
            size_hints={"x": 8},
            cached_configs=(queued_baseline,),
            cached_benchmark_mode=_COORDESC_BENCHMARK_MODE_UNGROUPED,
        )
        peer = self._make_fake_coordesc_autotuner(
            "peer",
            self._coordesc_test_config(4),
            timing,
            size_hints={"x": 8},
        )

        _, task_states = self._run_queued_coordesc_test([target, peer])

        self.assertIn(
            self._coordesc_test_key(queued_baseline),
            target._coordesc_test_calls,
        )
        self.assertEqual(task_states[0], ("queued", "baseline", None))
        self.assertEqual(
            target.coordesc_tuner.lookup_in_cache(
                queued_baseline,
                benchmark_mode=_COORDESC_BENCHMARK_MODE_GROUPED,
            ),
            10.0,
        )
        self.assertEqual(
            target.coordesc_tuner.lookup_in_cache(
                queued_baseline,
                benchmark_mode=_COORDESC_BENCHMARK_MODE_UNGROUPED,
            ),
            10.0,
        )

    def test_coordinate_descent_batch_rebenchmarks_grouped_cached_frontier_after_fallback(
        self,
    ):
        def timing(config):
            return {
                2: 9.0,
                4: 10.0,
                8: 1.0,
            }.get(config.kwargs["XBLOCK"], 100.0)

        queued_baseline = self._coordesc_test_config(4)
        queued_cached_candidate = self._coordesc_test_config(8)
        target = self._make_fake_coordesc_autotuner(
            "queued",
            queued_baseline,
            timing,
            size_hints={"x": 8},
            cached_configs=(queued_baseline, queued_cached_candidate),
            cached_benchmark_mode=_COORDESC_BENCHMARK_MODE_GROUPED,
        )
        peer = self._make_fake_coordesc_autotuner(
            "peer",
            self._coordesc_test_config(4),
            timing,
            size_hints={"x": 8},
            cached_configs=(self._coordesc_test_config(4),),
            cached_benchmark_mode=_COORDESC_BENCHMARK_MODE_GROUPED,
        )
        for autotuner in (target, peer):
            autotuner.bench = (
                lambda launcher, *args, _autotuner=autotuner, **kwargs: (
                    _autotuner._coordesc_test_calls.append(
                        self._coordesc_test_key(launcher.config)
                    ),
                    timing(launcher.config),
                )[1]
            )

        fallback_once = True

        def benchmark_requests(requests):
            nonlocal fallback_once
            if fallback_once and any(
                request.autotuner is target
                and request.launcher.config.kwargs["XBLOCK"] == 2
                for request in requests
            ):
                fallback_once = False
                raise RuntimeError("grouped benchmark failed")
            timings = []
            for request in requests:
                key = self._coordesc_test_key(request.launcher.config)
                request.autotuner._coordesc_test_calls.append(key)
                timings.append(timing(request.launcher.config))
            return timings

        with (
            self._coordesc_batch_patch(),
            patch(
                "torch._inductor.runtime.autotune_queue._benchmark_launcher_requests",
                side_effect=benchmark_requests,
            ),
            coordinate_descent_batch(disposable_args=True) as batch,
        ):
            self.assertTrue(batch.enqueue(target, target.launchers[0], (), {}, None, False))
            self.assertTrue(batch.enqueue(peer, peer.launchers[0], (), {}, None, False))

        self.assertIn(
            self._coordesc_test_key(queued_cached_candidate),
            target._coordesc_test_calls,
        )
        self.assertEqual(target.launchers[0].config.kwargs["XBLOCK"], 8)

    def test_coordinate_descent_batch_rebenchmarks_grouped_best_after_fallback(self):
        def timing(config):
            return {
                2: 12.0,
                4: 10.0,
                8: 5.0,
            }.get(config.kwargs["XBLOCK"], 100.0)

        queued_baseline = self._coordesc_test_config(4)
        target = self._make_fake_coordesc_autotuner(
            "queued",
            queued_baseline,
            timing,
            size_hints={"x": 8},
            cached_configs=(queued_baseline,),
            cached_benchmark_mode=_COORDESC_BENCHMARK_MODE_GROUPED,
        )
        target.coordesc_tuner.cache_benchmark_result(
            queued_baseline,
            1.0,
            benchmark_mode=_COORDESC_BENCHMARK_MODE_GROUPED,
        )
        peer = self._make_fake_coordesc_autotuner(
            "peer",
            self._coordesc_test_config(4),
            timing,
            size_hints={"x": 8},
            cached_configs=(self._coordesc_test_config(4),),
            cached_benchmark_mode=_COORDESC_BENCHMARK_MODE_GROUPED,
        )

        for autotuner in (target, peer):
            autotuner.bench = (
                lambda launcher, *args, _autotuner=autotuner, **kwargs: (
                    _autotuner._coordesc_test_calls.append(
                        self._coordesc_test_key(launcher.config)
                    ),
                    timing(launcher.config),
                )[1]
            )

        fallback_once = True

        def benchmark_requests(requests):
            nonlocal fallback_once
            if fallback_once and any(request.autotuner is target for request in requests):
                fallback_once = False
                raise RuntimeError("grouped benchmark failed")
            timings = []
            for request in requests:
                key = self._coordesc_test_key(request.launcher.config)
                request.autotuner._coordesc_test_calls.append(key)
                timings.append(timing(request.launcher.config))
            return timings

        with (
            config.patch(
                {
                    "compile_threads": 2,
                    "coordinate_descent_tuning": True,
                    "coordinate_descent_tuning_batch": True,
                }
            ),
            patch(
                "torch._inductor.runtime.autotune_queue._benchmark_launcher_requests",
                side_effect=benchmark_requests,
            ),
            coordinate_descent_batch(disposable_args=True) as batch,
        ):
            self.assertTrue(batch.enqueue(target, target.launchers[0], (), {}, None, False))
            self.assertTrue(batch.enqueue(peer, peer.launchers[0], (), {}, None, False))

        self.assertIn(self._coordesc_test_key(queued_baseline), target._coordesc_test_calls)
        self.assertEqual(
            target.coordesc_tuner.lookup_in_cache(
                queued_baseline,
                benchmark_mode=_COORDESC_BENCHMARK_MODE_UNGROUPED,
            ),
            10.0,
        )
        self.assertEqual(target.launchers[0].config.kwargs["XBLOCK"], 8)

    def test_coordinate_descent_batch_chunked_frontier_matches_scalar(self):
        def timing(config):
            return {
                2: 1.0,
                4: 10.0,
                8: 11.0,
            }.get(config.kwargs["XBLOCK"], 100.0)

        scalar_baseline = self._coordesc_test_config(4)
        scalar_best, _, _ = self._run_scalar_coordesc_test(
            scalar_baseline,
            timing,
            size_hints={"x": 8},
        )
        target = self._make_fake_coordesc_autotuner(
            "queued",
            self._coordesc_test_config(4),
            timing,
            size_hints={"x": 8},
        )
        peer = self._make_fake_coordesc_autotuner(
            "peer",
            self._coordesc_test_config(4),
            timing,
            size_hints={"x": 8},
        )

        counters.clear()
        frontiers, _ = self._run_queued_coordesc_test(
            [target, peer],
            config_overrides={
                "coordinate_descent_tuning_batch_max_frontier_candidates": 1
            },
        )

        self.assertEqual(
            self._coordesc_test_key(target.launchers[0].config),
            self._coordesc_test_key(scalar_best),
        )
        self.assertEqual(target.launchers[0].config.kwargs["XBLOCK"], 2)
        self.assertGreaterEqual(len(frontiers), 2)
        self.assertGreaterEqual(
            counters["inductor"]["autotune_queue_frontier_splits"], 1
        )
        self.assertGreaterEqual(
            counters["inductor"]["autotune_queue_frontier_chunks"], 2
        )

    def test_coordinate_descent_batch_matches_scalar_check_all_directions(self):
        inductor_meta = {"coordinate_descent_check_all_directions": True}

        def timing(config):
            xblock = config.kwargs["XBLOCK"]
            yblock = config.kwargs["YBLOCK"]
            if (xblock, yblock) == (8, 8):
                return 1.0
            if (xblock, yblock) == (4, 4):
                return 10.0
            return 11.0

        scalar_baseline = self._coordesc_test_config(4, 4)
        scalar_best, scalar_calls, _ = self._run_scalar_coordesc_test(
            scalar_baseline,
            timing,
            size_hints={"x": 8, "y": 8},
            inductor_meta=inductor_meta,
        )

        queued_baseline = self._coordesc_test_config(4, 4)
        target = self._make_fake_coordesc_autotuner(
            "queued",
            queued_baseline,
            timing,
            size_hints={"x": 8, "y": 8},
            inductor_meta=inductor_meta,
        )
        peer = self._make_fake_coordesc_autotuner(
            "peer",
            self._coordesc_test_config(4, 4),
            timing,
            size_hints={"x": 8, "y": 8},
            inductor_meta=inductor_meta,
        )

        self._run_queued_coordesc_test([target, peer])

        all_directions_key = self._coordesc_test_key(
            self._coordesc_test_config(8, 8)
        )
        self.assertIn(all_directions_key, scalar_calls)
        self.assertIn(all_directions_key, target._coordesc_test_calls)
        self.assertEqual(
            self._coordesc_test_key(target.launchers[0].config),
            self._coordesc_test_key(scalar_best),
        )
        self.assertEqual(target.launchers[0].config.kwargs, {"XBLOCK": 8, "YBLOCK": 8})

    def test_coordinate_descent_batch_matches_scalar_compile_failure_and_inf(self):
        def timing(config):
            if config.kwargs["XBLOCK"] == 2:
                return float("inf")
            return 1.0

        def compile_fails(config):
            return config.kwargs["XBLOCK"] == 8

        scalar_baseline = self._coordesc_test_config(4)
        scalar_best, _, _ = self._run_scalar_coordesc_test(
            scalar_baseline,
            timing,
            size_hints={"x": 8},
            compile_fails=compile_fails,
            compile_failure_exception_factory=lambda: OutOfResources(
                2, 1, "shared memory"
            ),
        )

        queued_baseline = self._coordesc_test_config(4)
        target = self._make_fake_coordesc_autotuner(
            "queued",
            queued_baseline,
            timing,
            size_hints={"x": 8},
            compile_fails=compile_fails,
            compile_failure_exception_factory=lambda: OutOfResources(
                2, 1, "shared memory"
            ),
        )
        peer = self._make_fake_coordesc_autotuner(
            "peer",
            self._coordesc_test_config(4),
            timing,
            size_hints={"x": 8},
            compile_fails=compile_fails,
            compile_failure_exception_factory=lambda: OutOfResources(
                2, 1, "shared memory"
            ),
        )

        self._run_queued_coordesc_test([target, peer])

        self.assertEqual(
            self._coordesc_test_key(target.launchers[0].config),
            self._coordesc_test_key(scalar_best),
        )
        self.assertEqual(target.launchers[0].config.kwargs["XBLOCK"], 4)
        self.assertEqual(
            target.coordesc_tuner.lookup_in_cache(
                self._coordesc_test_config(8),
                benchmark_mode=_COORDESC_BENCHMARK_MODE_GROUPED,
            ),
            float("inf"),
        )
        self.assertEqual(
            target.coordesc_tuner.lookup_in_cache(
                self._coordesc_test_config(2),
                benchmark_mode=_COORDESC_BENCHMARK_MODE_GROUPED,
            ),
            float("inf"),
        )

    def test_coordinate_descent_batch_unexpected_compile_failure_propagates(self):
        def timing(config):
            return 1.0

        def compile_fails(config):
            return config.kwargs["XBLOCK"] == 8

        target = self._make_fake_coordesc_autotuner(
            "queued",
            self._coordesc_test_config(4),
            timing,
            size_hints={"x": 8},
            compile_fails=compile_fails,
        )
        peer = self._make_fake_coordesc_autotuner(
            "peer",
            self._coordesc_test_config(4),
            timing,
            size_hints={"x": 8},
            compile_fails=compile_fails,
        )

        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            self._run_queued_coordesc_test([target, peer])

    def test_coordinate_descent_batch_groups_frontiers(self):
        class Launcher:
            def __init__(self, config):
                self.config = config
                self.n_regs = 1
                self.n_spills = 0
                self.shared = 0

        def make_autotuner(name, start):
            autotuner = object.__new__(CachingAutotuner)
            autotuner.fn = SimpleNamespace(
                __name__=name,
                src=f"def {name}():\n    pass\n",
            )
            autotuner.coordesc_tuner = CoordescTuner(
                name=name,
                size_hints={"x": 16},
                frozen_fields={"num_warps"},
            )
            autotuner.triton_meta = {"device": 0}
            autotuner.lock = threading.Lock()
            autotuner.launchers = [
                Launcher(triton.Config({"XBLOCK": start}, num_warps=4, num_stages=1))
            ]
            autotuner._ensure_kernel_loaded = lambda: None
            autotuner._precompile_config = lambda config: SimpleNamespace(
                config=config,
                kernel=SimpleNamespace(hash=f"{name}-{config.kwargs['XBLOCK']}"),
                make_launcher=lambda: Launcher(config),
            )

            def submit_config_compile(configs):
                compiled_autotuner = SimpleNamespace(
                    compile_results=[autotuner._precompile_config(configs[0])],
                    restore_after_unpickle=lambda old_values: None,
                )
                return SimpleNamespace(result=lambda: (compiled_autotuner, 1))

            autotuner._config_compile_submitter = submit_config_compile
            return autotuner

        autotuner0 = make_autotuner("kernel0", 4)
        autotuner1 = make_autotuner("kernel1", 2)
        targets = {"kernel0": 16, "kernel1": 8}
        frontiers = []
        group_states = {}

        def benchmark_requests(requests):
            frontier = []
            timings = []
            for request in requests:
                name = request.autotuner.fn.__name__
                group_states.setdefault(name, request.benchmark_group_state)
                self.assertIs(request.benchmark_group_state, group_states[name])
                xblock = request.launcher.config.kwargs["XBLOCK"]
                frontier.append((name, xblock))
                timings.append(abs(xblock - targets[name]) + 1.0)
            frontiers.append(frontier)
            return timings

        def finish_coordesc(
            autotuner,
            best_config,
            config2launcher,
            elapsed_ns,
            save_cache=True,
            **kwargs,
        ):
            winner = config2launcher[best_config]
            autotuner.launchers = [winner]
            winner.config.found_by_coordesc = True
            return winner

        with (
            config.patch(
                {
                    "compile_threads": 2,
                    "coordinate_descent_tuning": True,
                    "coordinate_descent_tuning_batch": True,
                }
            ),
            patch(
                "torch._inductor.runtime.autotune_queue._benchmark_launcher_requests",
                side_effect=benchmark_requests,
            ),
            patch.object(
                CachingAutotuner,
                "_finish_coordinate_descent_tuning",
                finish_coordesc,
            ),
            coordinate_descent_batch(disposable_args=True) as batch,
        ):
            self.assertIs(_get_active_coordinate_descent_batch(), batch)
            batch.enqueue(
                autotuner0,
                autotuner0.launchers[0],
                (),
                {},
                None,
                False,
            )
            batch.enqueue(
                autotuner1,
                autotuner1.launchers[0],
                (),
                {},
                None,
                False,
            )

        self.assertEqual(frontiers[0], [("kernel0", 4), ("kernel1", 2)])
        self.assertEqual(
            frontiers[1],
            [
                ("kernel0", 8),
                ("kernel0", 2),
                ("kernel1", 4),
                ("kernel1", 1),
            ],
        )
        self.assertEqual(autotuner0.launchers[0].config.kwargs["XBLOCK"], 16)
        self.assertEqual(autotuner1.launchers[0].config.kwargs["XBLOCK"], 8)
        self.assertEqual(set(group_states), {"kernel0", "kernel1"})
        self.assertIsNot(group_states["kernel0"], group_states["kernel1"])

    def test_coordinate_descent_batch_benchmarks_ready_subset(self):
        class Future:
            def __init__(self, result, ready):
                self._result = result
                self.ready = ready

            def done(self):
                return self.ready

            def result(self):
                self.ready = True
                return self._result

        def timing(config):
            return 1.0 if config.kwargs["XBLOCK"] == 4 else 2.0

        fast_baseline = triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1)
        slow_baseline = triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1)
        fast = self._make_fake_coordesc_autotuner(
            "fast",
            fast_baseline,
            timing,
            size_hints={"x": 8},
            cached_configs=(fast_baseline,),
            cached_benchmark_mode=_COORDESC_BENCHMARK_MODE_GROUPED,
        )
        slow = self._make_fake_coordesc_autotuner(
            "slow",
            slow_baseline,
            timing,
            size_hints={"x": 8},
            cached_configs=(slow_baseline,),
            cached_benchmark_mode=_COORDESC_BENCHMARK_MODE_GROUPED,
        )
        delayed_future = None
        frontiers = []

        def submit_compile_job(batch, task, config):
            nonlocal delayed_future
            launcher = task.autotuner._precompile_config(config).make_launcher()
            should_delay = (
                task.autotuner.fn.__name__ == "slow"
                and config.kwargs["XBLOCK"] == 8
            )
            future = Future(launcher, ready=not should_delay)
            if should_delay:
                delayed_future = future
            return _CoordinateDescentCompileJob(task, config, future)

        def benchmark_requests(requests):
            names = [request.autotuner.fn.__name__ for request in requests]
            frontiers.append(names)
            if len(frontiers) == 1:
                self.assertEqual(set(names), {"fast"})
                self.assertIsNotNone(delayed_future)
                delayed_future.ready = True
            return [
                request.autotuner._coordesc_test_timing(request.launcher.config)
                for request in requests
            ]

        counters.clear()
        with (
            self._coordesc_batch_patch(),
            patch.object(
                _CoordinateDescentBatch,
                "_submit_compile_job",
                autospec=True,
                side_effect=submit_compile_job,
            ),
            patch(
                "torch._inductor.runtime.autotune_queue._benchmark_launcher_requests",
                side_effect=benchmark_requests,
            ),
            coordinate_descent_batch(disposable_args=True) as batch,
        ):
            self.assertTrue(
                batch.enqueue(fast, fast.launchers[0], (), {}, None, False)
            )
            self.assertTrue(
                batch.enqueue(slow, slow.launchers[0], (), {}, None, False)
            )

        self.assertGreaterEqual(len(frontiers), 2)
        self.assertEqual(set(frontiers[0]), {"fast"})
        self.assertIn("slow", frontiers[-1])
        self.assertGreaterEqual(
            counters["inductor"]["autotune_queue_ready_subset_waves"], 2
        )
        self.assertEqual(fast.launchers[0].config.kwargs["XBLOCK"], 4)
        self.assertEqual(slow.launchers[0].config.kwargs["XBLOCK"], 4)

    def test_coordinate_descent_batch_reuses_compiled_launcher_without_submit(self):
        baseline_config = triton.Config(
            {"XBLOCK": 4}, num_warps=4, num_stages=1
        )
        autotuner = self._make_fake_coordesc_autotuner(
            "kernel0",
            baseline_config,
            lambda config: 1.0,
            size_hints={"x": 16},
        )
        autotuner.filename = __file__
        task = _CoordinateDescentTask(
            autotuner,
            autotuner.launchers[0],
            (),
            {},
            None,
            False,
        )
        batch = coordinate_descent_batch()

        counters.clear()
        with patch(
            "torch._inductor.async_compile.AsyncCompile.submit",
            side_effect=AssertionError("must not submit cached launcher"),
        ):
            job = batch._submit_compile_job(task, baseline_config)
            compiled = batch._resolve_compile_job(job)

        self.assertEqual(compiled, (task, baseline_config, autotuner.launchers[0]))

    def test_coordinate_descent_scalar_run_uses_effective_batch_mode_by_default(self):
        autotuner = self._make_fake_coordesc_autotuner(
            "kernel0",
            triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1),
            lambda config: 1.0,
            size_hints={"x": 16},
        )
        calls = []

        def coordinate_descent_tuning(launcher, *args, **kwargs):
            calls.append(kwargs["use_batch_benchmarking"])
            launcher.config.found_by_coordesc = True
            return launcher

        autotuner.coordinate_descent_tuning = coordinate_descent_tuning
        task = _CoordinateDescentTask(
            autotuner,
            autotuner.launchers[0],
            (),
            {},
            None,
            False,
        )

        task.run_scalar(save_kernel=False)

        self.assertEqual(calls, [None])

    def test_coordinate_descent_batch_uses_process_pool_compile_submitter(self):
        class Launcher:
            def __init__(self, config, name):
                self.config = config
                self.name = name
                self.cache_hash = name
                self.n_regs = 0
                self.n_spills = 0
                self.shared = 0

        class CompileResult:
            def __init__(self, config, name):
                self.config = config
                self.kernel = SimpleNamespace(hash=name)
                self.name = name

            def make_launcher(self):
                return Launcher(self.config, self.name)

        candidate_config = triton.Config(
            {"XBLOCK": 8}, num_warps=4, num_stages=1
        )
        candidate_compile_result = CompileResult(candidate_config, "candidate")
        restore_calls = []
        compiled_autotuner = SimpleNamespace(
            compile_results=[candidate_compile_result],
            restore_after_unpickle=lambda old_values: restore_calls.append(
                old_values
            ),
        )
        future = SimpleNamespace(result=lambda: (compiled_autotuner, 1))
        submitter = MagicMock(return_value=future)
        autotuner = self._make_fake_coordesc_autotuner(
            "kernel0",
            triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1),
            lambda config: 1.0,
            size_hints={"x": 16},
        )
        autotuner.compile_results = []
        autotuner.triton_meta = {"device": 0}
        autotuner._config_compile_submitter = submitter
        task = _CoordinateDescentTask(
            autotuner,
            autotuner.launchers[0],
            (),
            {},
            None,
            False,
        )
        batch = coordinate_descent_batch()

        counters.clear()
        with patch(
            "torch._inductor.async_compile.AsyncCompile.submit",
            side_effect=AssertionError("coordesc configs should use process pool hook"),
        ):
            job = batch._submit_compile_job(task, candidate_config)
            compiled = batch._resolve_compile_job(job)

        self.assertEqual(compiled[0], task)
        self.assertEqual(compiled[1], candidate_config)
        self.assertEqual(compiled[2].config, candidate_config)
        self.assertEqual(restore_calls, [None])
        submitter.assert_called_once_with([candidate_config])
        self.assertEqual(
            counters["inductor"]["autotune_queue_process_pool_compiles"], 1
        )
        self.assertIs(task.config2launcher[candidate_config], compiled[2])
        self.assertNotIn(candidate_compile_result, autotuner.compile_results)

    def test_coordinate_descent_batch_requires_process_pool_compile_submitter(self):
        candidate_config = triton.Config(
            {"XBLOCK": 8}, num_warps=4, num_stages=1
        )
        autotuner = self._make_fake_coordesc_autotuner(
            "kernel0",
            triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1),
            lambda config: 1.0,
            size_hints={"x": 16},
        )
        autotuner.filename = None
        autotuner._config_compile_submitter = None
        task = _CoordinateDescentTask(
            autotuner,
            autotuner.launchers[0],
            (),
            {},
            None,
            False,
        )
        batch = coordinate_descent_batch()

        counters.clear()
        with (
            config.patch({"compile_threads": 2}),
            patch(
                "torch._inductor.async_compile.AsyncCompile.submit",
                side_effect=AssertionError("must not compile configs in thread pool"),
            ) as mock_submit,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "requires a process-pool compile submitter",
            ):
                batch._submit_compile_job(task, candidate_config)

        mock_submit.assert_not_called()

    def test_coordinate_descent_batch_grouped_failure_skips_batch_cache_save(self):
        def timing(config):
            return abs(config.kwargs["XBLOCK"] - 8) + 1.0

        autotuners = [
            self._make_fake_coordesc_autotuner(
                "kernel0",
                triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1),
                timing,
                size_hints={"x": 16},
            ),
            self._make_fake_coordesc_autotuner(
                "kernel1",
                triton.Config({"XBLOCK": 2}, num_warps=4, num_stages=1),
                timing,
                size_hints={"x": 16},
            ),
        ]
        save_calls = []

        def make_finish(autotuner):
            def finish_coordesc(
                best_config, config2launcher, elapsed_ns, save_cache=True, **kwargs
            ):
                save_calls.append((autotuner.fn.__name__, save_cache, kwargs))
                winner = config2launcher.get(best_config)
                if winner is None:
                    winner = autotuner._precompile_config(best_config).make_launcher()
                winner.config.found_by_coordesc = True
                autotuner.launchers = [winner]
                return winner

            return finish_coordesc

        for autotuner in autotuners:
            autotuner.bench = (
                lambda launcher, *args, _timing=timing, **kwargs: _timing(
                    launcher.config
                )
            )
            autotuner._finish_coordinate_descent_tuning = make_finish(autotuner)

        counters.clear()
        with (
            config.patch(
                {
                    "compile_threads": 2,
                    "coordinate_descent_tuning": True,
                    "coordinate_descent_tuning_batch": True,
                }
            ),
            patch(
                "torch._inductor.runtime.autotune_queue._benchmark_launcher_requests",
                side_effect=RuntimeError("grouped benchmark failed"),
            ),
            coordinate_descent_batch(disposable_args=True) as batch,
        ):
            for autotuner in autotuners:
                self.assertTrue(
                    batch.enqueue(
                        autotuner,
                        autotuner.launchers[0],
                        (),
                        {},
                        None,
                        False,
                    )
                )

        self.assertTrue(save_calls)
        for _, save_cache, kwargs in save_calls:
            self.assertFalse(save_cache)
            self.assertFalse(kwargs["coordinate_descent_tuning_batch"])
        self.assertGreater(
            counters["inductor"]["coordesc_tuning_batch_ungrouped_cache_skips"],
            0,
        )
        for autotuner in autotuners:
            self.assertEqual(
                {
                    mode
                    for mode, _ in autotuner.coordesc_tuner.cached_benchmark_results
                },
                {_COORDESC_BENCHMARK_MODE_UNGROUPED},
            )

    def test_coordinate_descent_batch_cpu_copy_path_skips_batch_cache_save(self):
        class Launcher:
            def __init__(self, config):
                self.config = config
                self.n_regs = 1
                self.n_spills = 0
                self.shared = 0
                self.cache_hash = f"kernel-{config.kwargs['XBLOCK']}"
                self.store_cubin = False

        def timing(config):
            return abs(config.kwargs["XBLOCK"] - 8) + 1.0

        autotuner = object.__new__(CachingAutotuner)
        autotuner.fn = SimpleNamespace(
            __name__="kernel",
            src="def kernel():\n    pass\n",
            arg_names=[],
        )
        autotuner.lock = threading.Lock()
        autotuner.device_props = SimpleNamespace(type="cuda", index=0)
        autotuner.inductor_meta = {}
        autotuner.triton_meta = {}
        autotuner.heuristic_type = HeuristicType.REDUCTION
        autotuner.deterministic_mode = False
        autotuner.size_hints = {"x": 16}
        autotuner.benchmark_failure_reasons = {}
        autotuner.autotune_time_taken_ns = 0
        autotuner.coordesc_tuner = CoordescTuner(
            name="kernel",
            size_hints={"x": 16},
            frozen_fields={"num_warps"},
        )
        autotuner._ensure_kernel_loaded = lambda: None
        autotuner._skip_config_due_to_register_spilling = lambda launcher: False
        autotuner._precompile_config = lambda cfg: SimpleNamespace(
            make_launcher=lambda: Launcher(cfg)
        )
        autotuner.copy_args_to_cpu_if_needed = lambda *args, **kwargs: {
            "out": object()
        }
        autotuner._make_benchmark_call = (
            lambda launcher, cpu_copies, stream, args, kwargs, clone_args=True: lambda: timing(
                launcher.config
            )
        )
        device_interface = MagicMock()
        device_interface.is_available.return_value = True
        device_interface.current_device.return_value = 0
        device_interface.get_raw_stream.return_value = "stream"
        device_interface.device.return_value = contextlib.nullcontext()
        autotuner.get_device_interface = lambda: device_interface
        save_calls = []

        def finish_coordesc(
            best_config, config2launcher, elapsed_ns, save_cache=True, **kwargs
        ):
            save_calls.append((save_cache, kwargs))
            winner = config2launcher[best_config]
            winner.config.found_by_coordesc = True
            return winner

        autotuner._finish_coordinate_descent_tuning = finish_coordesc
        baseline = Launcher(triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1))

        counters.clear()
        with patch(
            "torch._inductor.runtime.autotune_benchmarking.benchmarker.benchmark",
            side_effect=lambda call, **kwargs: call(),
        ):
            winner = autotuner._coordinate_descent_tuning(
                baseline,
                use_batch_benchmarking=True,
            )

        self.assertEqual(winner.config.kwargs["XBLOCK"], 8)
        self.assertEqual(len(save_calls), 1)
        save_cache, kwargs = save_calls[0]
        self.assertFalse(save_cache)
        self.assertFalse(kwargs["coordinate_descent_tuning_batch"])
        self.assertGreater(
            counters["inductor"]["coordesc_tuning_batch_ungrouped_cache_skips"],
            0,
        )
        self.assertEqual(
            {mode for mode, _ in autotuner.coordesc_tuner.cached_benchmark_results},
            {_COORDESC_BENCHMARK_MODE_UNGROUPED},
        )

    def test_coordinate_descent_tuning_batch_policy_controls_grouped_path(self):
        class Launcher:
            def __init__(self, config):
                self.config = config
                self.n_regs = 1
                self.n_spills = 0
                self.shared = 0
                self.cache_hash = f"hash-{config.kwargs['XBLOCK']}"

        def run_case(
            batch_enabled,
            policy,
            heuristic_type,
            native_matmul,
            expect_grouped,
            device_type="cuda",
            coordinate_descent_tuning=True,
            use_batch_benchmarking=None,
        ):
            autotuner = object.__new__(CachingAutotuner)
            autotuner.fn = SimpleNamespace(
                __name__="kernel",
                src="def kernel():\n    pass\n",
            )
            autotuner.lock = threading.Lock()
            autotuner.coordesc_tuner = CoordescTuner(
                name="kernel",
                size_hints={"x": 16},
                frozen_fields={"num_warps"},
            )
            autotuner.size_hints = {"x": 16}
            autotuner.heuristic_type = heuristic_type
            autotuner.triton_meta = {"native_matmul": native_matmul}
            if device_type is not None:
                autotuner.triton_meta["device_type"] = device_type
            autotuner.deterministic_mode = False
            autotuner.inductor_meta = {
                "coordinate_descent_tuning": coordinate_descent_tuning,
                "coordinate_descent_tuning_batch": batch_enabled,
            }
            if policy is not None:
                autotuner.inductor_meta["coordinate_descent_tuning_batch_policy"] = (
                    policy
                )
            saved_cache_kwargs = []

            def save_cache_hook(config, time_taken_ns, **kwargs):
                saved_cache_kwargs.append(kwargs)

            autotuner.save_cache_hook = save_cache_hook
            autotuner.autotune_time_taken_ns = 0
            autotuner.benchmark_failure_reasons = {}
            autotuner._ensure_kernel_loaded = lambda: None
            autotuner._precompile_config = lambda config: SimpleNamespace(
                make_launcher=lambda: Launcher(config)
            )
            baseline = Launcher(
                triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1)
            )
            grouped_calls = []
            grouped_states = []

            def timing(launcher):
                return abs(launcher.config.kwargs["XBLOCK"] - 8) + 1.0

            autotuner.bench = lambda launcher, *args, **kwargs: timing(launcher)

            def benchmark_all_launchers(launchers, *args, **kwargs):
                grouped_calls.append(
                    [launcher.config.kwargs["XBLOCK"] for launcher in launchers]
                )
                grouped_states.append(kwargs.get("benchmark_group_state"))
                return {launcher: timing(launcher) for launcher in launchers}

            if expect_grouped:
                autotuner.benchmark_all_launchers = benchmark_all_launchers
            else:
                autotuner.benchmark_all_launchers = lambda *args, **kwargs: self.fail(
                    "batch path should be disabled"
                )

            winner = autotuner._coordinate_descent_tuning(
                baseline,
                use_batch_benchmarking=use_batch_benchmarking,
            )

            self.assertEqual(winner.config.kwargs["XBLOCK"], 8)
            self.assertEqual(len(saved_cache_kwargs), 1)
            self.assertEqual(
                saved_cache_kwargs[0]["coordinate_descent_tuning_batch"],
                expect_grouped,
            )
            if expect_grouped:
                expected_policy = (
                    policy
                    if policy is not None
                    else config._coordinate_descent_tuning_batch_default_policy
                )
                self.assertEqual(
                    saved_cache_kwargs[0]["coordinate_descent_tuning_batch_policy"],
                    expected_policy,
                )
            else:
                self.assertIsNone(
                    saved_cache_kwargs[0]["coordinate_descent_tuning_batch_policy"]
                )
            if expect_grouped:
                self.assertTrue(grouped_calls)
                self.assertEqual(grouped_calls[0], [4])
                self.assertIsNotNone(grouped_states[0])
                self.assertTrue(
                    all(
                        grouped_state is grouped_states[0]
                        for grouped_state in grouped_states
                    )
                )
            else:
                self.assertEqual(grouped_calls, [])

        cases = [
            (False, "all", HeuristicType.REDUCTION, False, False),
            (True, "none", HeuristicType.REDUCTION, False, False),
            (True, "all", HeuristicType.POINTWISE, False, True),
            (True, "auto", HeuristicType.POINTWISE, False, False),
            (True, "auto", HeuristicType.REDUCTION, False, True),
            (True, "auto", HeuristicType.PERSISTENT_REDUCTION, False, True),
            (True, "auto", HeuristicType.SPLIT_SCAN, False, True),
            (True, "auto", HeuristicType.POINTWISE, True, True),
            (True, None, HeuristicType.POINTWISE, False, False),
            (True, None, HeuristicType.REDUCTION, False, True),
            (True, None, HeuristicType.POINTWISE, True, True),
            (True, "reductions", HeuristicType.PERSISTENT_REDUCTION, False, True),
            (True, "reductions", HeuristicType.SPLIT_SCAN, False, True),
            (True, "reductions", HeuristicType.POINTWISE, True, False),
            (True, "all", HeuristicType.REDUCTION, False, True, "hip"),
            (True, "all", HeuristicType.REDUCTION, False, False, "xpu"),
            (True, "all", HeuristicType.REDUCTION, False, False, None),
            (True, "all", HeuristicType.REDUCTION, False, False, "cuda", False),
            (True, "all", HeuristicType.REDUCTION, False, False, "cuda", True, False),
        ]
        for case in cases:
            with self.subTest(case=case):
                run_case(*case)

        with (
            self.subTest(case="invalid_policy"),
            patch("torch._inductor.runtime.autotune_common.log.warning") as warning,
        ):
            run_case(True, "invalid", HeuristicType.REDUCTION, False, False)
            warning.assert_called()
        with (
            self.subTest(case="compile_threads_one"),
            config.patch({"compile_threads": 1}),
        ):
            run_case(True, "all", HeuristicType.REDUCTION, False, False)

    def test_coordinate_descent_runtime_honors_effective_batch_metadata(self):
        autotuner = object.__new__(CachingAutotuner)
        autotuner.inductor_meta = {
            "coordinate_descent_tuning": True,
            "coordinate_descent_tuning_batch_requested": True,
            "coordinate_descent_tuning_batch": False,
            "coordinate_descent_tuning_batch_policy": "all",
        }
        autotuner.triton_meta = {"device_type": "cuda"}
        autotuner.heuristic_type = HeuristicType.REDUCTION
        autotuner.deterministic_mode = False

        with config.patch({"compile_threads": 2}):
            self.assertTrue(
                autotuner.inductor_meta["coordinate_descent_tuning_batch_requested"]
            )
            self.assertFalse(autotuner._coordinate_descent_batch_enabled())
            self.assertEqual(expected_autotune_queue_calls([autotuner]), 0)


if __name__ == "__main__":
    run_tests()
