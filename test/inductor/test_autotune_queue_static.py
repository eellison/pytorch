# Owner(s): ["module: inductor"]

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.runtime.autotune_common import (
    _has_deferred_static_autotune_precompile,
    _should_defer_static_autotune_precompile,
)
from torch._inductor.runtime.autotune_queue import (
    _StaticAutotuneTask,
    autotune_queue,
    get_active_autotune_queue,
)
from torch._inductor.runtime.coordinate_descent_tuner import CoordescTuner
from torch._inductor.runtime.hints import HeuristicType
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._inductor.test_case import run_tests, TestCase


try:
    import triton  # @manual
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904

class _FakeLauncher:
    def __init__(self, config=None, name=None, timing=None, should_fail=False):
        self.config = config or triton.Config(
            {"XBLOCK": 1}, num_warps=4, num_stages=1
        )
        self.name = name
        self.timing = timing
        self.cache_hash = name if name is not None else str(self.config.kwargs["XBLOCK"])
        self.n_regs = 0
        self.n_spills = 0
        self.shared = 0
        self.store_cubin = False
        self.calls = 0
        self.should_fail = should_fail

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("invalid configuration argument")


class _FakeCompileResult:
    def __init__(self, config, name=None):
        self.config = config
        self.name = name if name is not None else str(config.kwargs["XBLOCK"])
        self.kernel = SimpleNamespace(hash=self.name)

    def make_launcher(self):
        return _FakeLauncher(self.config, self.name)


class TestAutotuneQueueStatic(TestCase):
    @staticmethod
    def _autotune_queue_patch(**overrides):
        config_values = {
            "compile_threads": 2,
            "coordinate_descent_tuning": True,
            "coordinate_descent_tuning_batch": True,
        }
        config_values.update(overrides)
        return config.patch(config_values)

    def _make_static_launcher(
        self,
        name,
        *,
        config=None,
        xblock=1,
        timing=None,
        should_fail=False,
    ):
        return _FakeLauncher(
            config
            if config is not None
            else triton.Config({"XBLOCK": xblock}, num_warps=4, num_stages=1),
            name=name,
            timing=timing,
            should_fail=should_fail,
        )

    def _make_static_compile_result(self, config, name=None):
        return _FakeCompileResult(config, name or str(config.kwargs["XBLOCK"]))

    def _make_static_autotuner(
        self,
        name="kernel",
        launchers=(),
        *,
        arg_names=("out",),
        configs=None,
        compile_results=None,
        heuristic_type=HeuristicType.REDUCTION,
        inductor_meta=None,
        triton_meta=None,
    ):
        autotuner = object.__new__(CachingAutotuner)
        autotuner.fn = SimpleNamespace(__name__=name, arg_names=list(arg_names))
        autotuner.custom_kernel = False
        autotuner.inductor_meta = (
            inductor_meta
            if inductor_meta is not None
            else {
                "coordinate_descent_tuning_batch": True,
            }
        )
        autotuner.triton_meta = (
            triton_meta if triton_meta is not None else {"device_type": "cuda"}
        )
        autotuner.heuristic_type = heuristic_type
        autotuner.deterministic_mode = False
        autotuner.size_hints = None
        autotuner.launchers = list(launchers)
        autotuner.configs = list(configs) if configs is not None else None
        autotuner.compile_results = (
            list(compile_results) if compile_results is not None else []
        )
        autotuner.precompile_time_taken_ns = 0
        autotuner.benchmark_failure_reasons = {}
        autotuner.save_cache_hook = MagicMock()
        autotuner.coordesc_tuner = CoordescTuner(
            name=name,
            size_hints={"x": 16},
            frozen_fields={"num_warps"},
        )
        autotuner._cached_launcher = None
        autotuner._cache_eligible = False
        autotuner.triton_interpret = False
        autotuner.device_props = SimpleNamespace(type="cuda", index=0)
        autotuner.get_device_interface = lambda: SimpleNamespace(
            is_available=lambda: False
        )
        autotuner._install_triton_allocator = lambda: None
        autotuner._pre_launch = lambda *args, **kwargs: None
        autotuner._post_launch = lambda: None
        autotuner.reset_to_zero_args = lambda *args, **kwargs: None
        autotuner.copy_args_to_cpu_if_needed = lambda *args, **kwargs: {}
        autotuner._make_launchers = lambda: setattr(
            autotuner,
            "launchers",
            [result.make_launcher() for result in autotuner.compile_results],
        )
        autotuner.is_statically_launchable = lambda: False
        autotuner.autotune_to_one_config = MagicMock(
            side_effect=AssertionError("static autotune should be deferred")
        )
        return autotuner

    def test_static_config_autotune_batches_across_kernels(self):
        a_launchers = [
            self._make_static_launcher(f"a{i}", xblock=i, timing=10.0 - i)
            for i in range(6)
        ]
        b_launchers = [
            self._make_static_launcher(f"b{i}", xblock=i, timing=4.0 - i)
            for i in range(4)
        ]
        autotuner_a = self._make_static_autotuner("kernel_a", a_launchers)
        autotuner_b = self._make_static_autotuner("kernel_b", b_launchers)
        benchmark_groups = []

        def benchmark_requests(requests):
            self.assertTrue(
                all(request.benchmark_group_key is not None for request in requests)
            )
            self.assertTrue(
                all(request.benchmark_group_state is not None for request in requests)
            )
            benchmark_groups.append(
                [
                    (request.autotuner.fn.__name__, request.launcher.name)
                    for request in requests
                ]
            )
            return [request.launcher.timing for request in requests]

        with (
            self._autotune_queue_patch(
                coordinate_descent_tuning=False,
                coordinate_descent_tuning_batch_min_kernels=1,
            ),
            patch(
                "torch._inductor.runtime.autotune_queue._benchmark_launcher_requests",
                side_effect=benchmark_requests,
            ),
            autotune_queue(disposable_args=True) as batch,
        ):
            self.assertIs(get_active_autotune_queue(), batch)
            autotuner_a.run("out", stream=None)
            autotuner_b.run("out", stream=None)

        self.assertEqual(
            benchmark_groups,
            [
                [("kernel_a", launcher.name) for launcher in a_launchers]
                + [("kernel_b", launcher.name) for launcher in b_launchers]
            ],
        )
        self.assertEqual(autotuner_a.launchers, [a_launchers[-1]])
        self.assertEqual(autotuner_b.launchers, [b_launchers[-1]])
        self.assertEqual(a_launchers[0].calls, 1)
        self.assertEqual(b_launchers[0].calls, 1)
        autotuner_a.save_cache_hook.assert_called_once()
        autotuner_b.save_cache_hook.assert_called_once()

    def test_static_config_autotune_falls_back_when_first_launcher_invalid(self):
        invalid = self._make_static_launcher("invalid", should_fail=True)
        winner = self._make_static_launcher("winner")
        launchers = [
            invalid,
            self._make_static_launcher("candidate1"),
            self._make_static_launcher("candidate2"),
        ]
        autotuner = self._make_static_autotuner(launchers=launchers)

        def autotune_to_one_config(*args, **kwargs):
            self.assertEqual(autotuner.launchers, launchers)
            autotuner.launchers = [winner]

        autotuner.autotune_to_one_config = MagicMock(
            side_effect=autotune_to_one_config
        )

        with (
            self._autotune_queue_patch(
                coordinate_descent_tuning=False,
                coordinate_descent_tuning_batch_min_kernels=1,
            ),
            patch(
                "torch._inductor.runtime.autotune_queue._benchmark_launcher_requests",
                side_effect=AssertionError("invalid first launch should drain scalar"),
            ),
            autotune_queue(disposable_args=True),
        ):
            autotuner.run("out", stream=None)

        self.assertEqual(invalid.calls, 1)
        self.assertEqual(winner.calls, 1)
        autotuner.autotune_to_one_config.assert_called_once()
        self.assertEqual(autotuner.launchers, [winner])

    def test_static_config_autotune_single_task_drain_restores_launchers(self):
        launchers = [
            self._make_static_launcher("first"),
            self._make_static_launcher("winner"),
            self._make_static_launcher("other"),
        ]
        autotuner = self._make_static_autotuner(launchers=launchers)

        def autotune_to_one_config(*args, **kwargs):
            self.assertEqual(autotuner.launchers, launchers)
            autotuner.launchers = [launchers[1]]

        autotuner.autotune_to_one_config = MagicMock(
            side_effect=autotune_to_one_config
        )

        with (
            self._autotune_queue_patch(
                coordinate_descent_tuning=False,
                coordinate_descent_tuning_batch_min_kernels=1,
            ),
            patch(
                "torch._inductor.runtime.autotune_queue._benchmark_launcher_requests",
                side_effect=AssertionError("single static task should drain scalar"),
            ),
            autotune_queue(disposable_args=True),
        ):
            autotuner.run("out", stream=None)

        self.assertEqual(launchers[0].calls, 1)
        autotuner.autotune_to_one_config.assert_called_once()
        self.assertEqual(autotuner.launchers, [launchers[1]])

    def test_static_config_autotune_budget_fallback_is_not_deferred(self):
        launchers = [
            self._make_static_launcher("first"),
            self._make_static_launcher("winner"),
            self._make_static_launcher("other"),
        ]
        autotuner = self._make_static_autotuner(launchers=launchers)

        def autotune_to_one_config(*args, **kwargs):
            self.assertEqual(autotuner.launchers, launchers)
            autotuner.launchers = [launchers[1]]

        autotuner.autotune_to_one_config = MagicMock(
            side_effect=autotune_to_one_config
        )

        with (
            self._autotune_queue_patch(
                coordinate_descent_tuning=False,
                coordinate_descent_tuning_batch_max_live_bytes=1,
                coordinate_descent_tuning_batch_min_kernels=1,
            ),
            autotune_queue(disposable_args=True),
        ):
            autotuner.run(torch.zeros(8), stream=None)

        autotuner.autotune_to_one_config.assert_called_once()
        self.assertEqual(launchers[1].calls, 1)
        self.assertEqual(autotuner.launchers, [launchers[1]])

    def test_static_config_process_pool_compile_job_hook(self):
        baseline_config = triton.Config(
            {"XBLOCK": 1}, num_warps=4, num_stages=1
        )
        candidate_config = triton.Config(
            {"XBLOCK": 2}, num_warps=4, num_stages=1
        )
        baseline_compile_result = self._make_static_compile_result(
            baseline_config, "baseline"
        )
        candidate_compile_result = self._make_static_compile_result(
            candidate_config, "candidate"
        )
        baseline_launcher = baseline_compile_result.make_launcher()
        restore_calls = []
        compiled_autotuner = SimpleNamespace(
            compile_results=[candidate_compile_result],
            restore_after_unpickle=lambda old_values: restore_calls.append(
                old_values
            ),
        )
        future = SimpleNamespace(result=lambda: (compiled_autotuner, 1))
        submitter = MagicMock(return_value=future)
        autotuner = self._make_static_autotuner(
            arg_names=(),
            launchers=[baseline_launcher],
            configs=[candidate_config],
            compile_results=[baseline_compile_result],
            triton_meta={"device_type": "cuda", "device": 0},
        )
        autotuner._static_config_compile_submitter = submitter
        task = _StaticAutotuneTask(
            autotuner,
            (),
            {},
            None,
            False,
        )
        batch = autotune_queue()

        counters.clear()
        with patch(
            "torch._inductor.async_compile.AsyncCompile.submit",
            side_effect=AssertionError("static configs should use process pool hook"),
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
        self.assertIn(candidate_compile_result, autotuner.compile_results)

    def test_static_config_cached_best_materializes_launcher(self):
        baseline_config = triton.Config(
            {"XBLOCK": 1}, num_warps=4, num_stages=1
        )
        candidate_config = triton.Config(
            {"XBLOCK": 2}, num_warps=4, num_stages=1
        )
        baseline_compile_result = self._make_static_compile_result(
            baseline_config, "baseline"
        )
        candidate_compile_result = self._make_static_compile_result(
            candidate_config, "candidate"
        )
        autotuner = self._make_static_autotuner(
            arg_names=(),
            launchers=[baseline_compile_result.make_launcher()],
            configs=[candidate_config],
            compile_results=[baseline_compile_result],
            triton_meta={"device_type": "cuda", "device": 0},
        )
        autotuner._precompile_config = MagicMock(return_value=candidate_compile_result)
        task = _StaticAutotuneTask(
            autotuner,
            (),
            {},
            None,
            False,
        )

        task.cache_benchmark_result(baseline_config, 10.0)
        task.cache_benchmark_result(candidate_config, 1.0)

        self.assertEqual(task.prepare_next_configs(), [])
        self.assertTrue(task.done)
        self.assertEqual(task.best_launcher.config, candidate_config)
        self.assertEqual(task.best_launcher.name, "candidate")
        autotuner._precompile_config.assert_called_once_with(candidate_config)

    def test_deferred_static_precompile_fallback_uses_process_pool_hook(self):
        baseline_config = triton.Config(
            {"XBLOCK": 1}, num_warps=4, num_stages=1
        )
        candidate_config = triton.Config(
            {"XBLOCK": 2}, num_warps=4, num_stages=1
        )
        baseline_compile_result = self._make_static_compile_result(
            baseline_config, "baseline"
        )
        candidate_compile_result = self._make_static_compile_result(
            candidate_config, "candidate"
        )
        compiled_autotuner = SimpleNamespace(
            compile_results=[candidate_compile_result],
            restore_after_unpickle=lambda old_values: None,
        )
        future = SimpleNamespace(result=lambda: (compiled_autotuner, 1))
        submitter = MagicMock(return_value=future)
        autotuner = self._make_static_autotuner(
            launchers=[baseline_compile_result.make_launcher()],
            configs=[candidate_config],
            compile_results=[baseline_compile_result],
            heuristic_type=HeuristicType.POINTWISE,
            inductor_meta={},
            triton_meta={"device_type": "cuda", "device": 0},
        )
        autotuner._static_config_compile_submitter = submitter
        autotuner._static_triton_bundle_key = "static-key"

        counters.clear()
        autotuner._finish_deferred_static_precompile()

        submitter.assert_called_once_with([candidate_config])
        self.assertEqual(
            [launcher.config for launcher in autotuner.launchers],
            [baseline_config, candidate_config],
        )
        self.assertIsNone(autotuner.configs)
        self.assertIsNone(autotuner._static_config_compile_submitter)
        self.assertIsNone(autotuner._static_triton_bundle_key)
        self.assertEqual(
            counters["inductor"]["autotune_queue_process_pool_compiles"], 1
        )

    def test_deferred_static_precompile_scalar_finish_avoids_process_pool_hook(self):
        baseline_config = triton.Config(
            {"XBLOCK": 1}, num_warps=4, num_stages=1
        )
        candidate_config = triton.Config(
            {"XBLOCK": 2}, num_warps=4, num_stages=1
        )
        baseline_compile_result = self._make_static_compile_result(baseline_config)
        candidate_compile_result = self._make_static_compile_result(candidate_config)
        autotuner = self._make_static_autotuner(
            launchers=[baseline_compile_result.make_launcher()],
            configs=[candidate_config],
            compile_results=[baseline_compile_result],
            triton_meta={"device_type": "cuda", "device": 0},
        )
        reloaded_fn = SimpleNamespace(fn=object())
        reload_kernel = Mock(return_value=SimpleNamespace(fn=reloaded_fn))
        autotuner.fn = SimpleNamespace(fn=None)
        autotuner._reload_kernel = reload_kernel
        static_submitter = Mock(
            side_effect=AssertionError("inactive queue should finish static inline")
        )
        config_submitter = Mock(
            side_effect=AssertionError("inactive queue should finish static inline")
        )
        autotuner._static_config_compile_submitter = static_submitter
        autotuner._config_compile_submitter = config_submitter
        autotuner._static_triton_bundle_key = "static-key"
        autotuner._precompile_config = Mock(return_value=candidate_compile_result)
        autotuner._dynamic_scale_rblock = Mock()

        counters.clear()
        autotuner._finish_deferred_static_precompile(use_process_pool=False)

        static_submitter.assert_not_called()
        config_submitter.assert_not_called()
        reload_kernel.assert_called_once_with()
        self.assertIs(autotuner.fn, reloaded_fn)
        autotuner._precompile_config.assert_called_once_with(candidate_config)
        self.assertEqual(
            [launcher.config for launcher in autotuner.launchers],
            [baseline_config, candidate_config],
        )
        self.assertIsNone(autotuner.configs)
        self.assertIsNone(autotuner._static_config_compile_submitter)
        self.assertIsNone(autotuner._config_compile_submitter)
        self.assertIsNone(autotuner._static_triton_bundle_key)
        self.assertEqual(
            counters["inductor"]["autotune_queue_process_pool_compiles"], 0
        )

    def test_prepare_for_benchmark_finishes_deferred_static_with_process_pool(self):
        baseline_config = triton.Config(
            {"XBLOCK": 1}, num_warps=4, num_stages=1
        )
        candidate_config = triton.Config(
            {"XBLOCK": 2}, num_warps=4, num_stages=1
        )
        baseline_compile_result = self._make_static_compile_result(
            baseline_config, "baseline"
        )
        candidate_compile_result = self._make_static_compile_result(
            candidate_config, "candidate"
        )
        compiled_autotuner = SimpleNamespace(
            compile_results=[candidate_compile_result],
            restore_after_unpickle=lambda old_values: None,
        )
        future = SimpleNamespace(result=lambda: (compiled_autotuner, 1))
        submitter = MagicMock(return_value=future)
        autotuner = self._make_static_autotuner(
            launchers=[baseline_compile_result.make_launcher()],
            configs=[candidate_config],
            compile_results=[baseline_compile_result],
            triton_meta={"device_type": "cuda", "device": 0},
        )
        autotuner._static_config_compile_submitter = submitter
        autotuner._static_triton_bundle_key = "static-key"
        autotuner._precompile_config = Mock(
            side_effect=AssertionError("deferred configs should use process pool")
        )

        def autotune_to_one_config(*args, **kwargs):
            self.assertEqual(
                [launcher.config for launcher in autotuner.launchers],
                [baseline_config, candidate_config],
            )
            autotuner.launchers = [autotuner.launchers[1]]

        autotuner.autotune_to_one_config = MagicMock(
            side_effect=autotune_to_one_config
        )

        counters.clear()
        autotuner.prepare_for_benchmark("out")

        submitter.assert_called_once_with([candidate_config])
        autotuner._precompile_config.assert_not_called()
        autotuner.autotune_to_one_config.assert_called_once()
        self.assertEqual(autotuner.launchers[0].config, candidate_config)
        self.assertIsNone(autotuner.configs)
        self.assertIsNone(autotuner._static_config_compile_submitter)
        self.assertEqual(
            counters["inductor"]["autotune_queue_process_pool_compiles"], 1
        )

    def test_static_autotune_scalar_run_clears_process_pool_state(self):
        config0 = triton.Config({"XBLOCK": 1}, num_warps=4, num_stages=1)
        launcher = self._make_static_launcher("kernel", config=config0)
        autotuner = self._make_static_autotuner(
            arg_names=(),
            launchers=[launcher],
        )
        autotuner._cached_launcher = launcher
        autotuner._static_config_compile_submitter = Mock()
        autotuner._config_compile_submitter = Mock()
        autotuner._static_triton_bundle_key = "static-key"

        def autotune_to_one_config(*args, **kwargs):
            autotuner.launchers = [launcher]

        autotuner.autotune_to_one_config = MagicMock(
            side_effect=autotune_to_one_config
        )

        task = _StaticAutotuneTask(
            autotuner,
            (),
            {},
            None,
            False,
        )
        task.run_scalar(save_kernel=False)

        autotuner.autotune_to_one_config.assert_called_once()
        self.assertIsNone(autotuner._config_compile_submitter)
        self.assertIsNone(autotuner._static_config_compile_submitter)
        self.assertIsNone(autotuner._static_triton_bundle_key)
        self.assertIsNone(autotuner._cached_launcher)

    def test_deferred_static_precompile_requires_process_pool_hook(self):
        candidate_config = triton.Config(
            {"XBLOCK": 2}, num_warps=4, num_stages=1
        )
        autotuner = self._make_static_autotuner(configs=[candidate_config])
        autotuner._static_config_compile_submitter = None
        autotuner._static_triton_bundle_key = "static-key"
        autotuner.launchers = []
        autotuner.precompile = Mock(
            side_effect=AssertionError("must not precompile static configs inline")
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "requires a process-pool compile submitter",
        ):
            autotuner._finish_deferred_static_precompile()

        autotuner.precompile.assert_not_called()
        self.assertIsNone(autotuner._static_config_compile_submitter)
        self.assertIsNone(autotuner._config_compile_submitter)
        self.assertIsNone(autotuner._static_triton_bundle_key)

    def test_static_config_deferred_precompile_detection(self):
        baseline_config = triton.Config(
            {"XBLOCK": 1}, num_warps=4, num_stages=1
        )
        candidate_config = triton.Config(
            {"XBLOCK": 2}, num_warps=4, num_stages=1
        )
        later_config = triton.Config(
            {"XBLOCK": 4}, num_warps=4, num_stages=1
        )
        autotuner = self._make_static_autotuner(
            configs=[baseline_config, candidate_config, later_config],
            heuristic_type=HeuristicType.POINTWISE,
            inductor_meta={
                "coordinate_descent_tuning": False,
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_policy": "auto",
            },
            triton_meta={"device_type": "cuda", "native_matmul": True},
        )

        with config.patch({"compile_threads": 2}):
            self.assertFalse(_should_defer_static_autotune_precompile(autotuner))

        with config.patch(
            {
                "autotune_queue_static_precompile": True,
                "compile_threads": 2,
            }
        ):
            self.assertTrue(_should_defer_static_autotune_precompile(autotuner))
            autotuner.inductor_meta["profile_bandwidth"] = True
            self.assertFalse(_should_defer_static_autotune_precompile(autotuner))
            autotuner.inductor_meta.pop("profile_bandwidth")
            autotuner.compile_results = [
                SimpleNamespace(config=baseline_config, kernel=SimpleNamespace())
            ]
            autotuner.configs = [candidate_config, later_config]
            self.assertFalse(_should_defer_static_autotune_precompile(autotuner))
            self.assertTrue(_has_deferred_static_autotune_precompile(autotuner))

    def test_async_worker_gets_static_precompile_flag(self):
        from torch._inductor.async_compile import AsyncCompile, CompiledTritonKernels

        source = "@triton.jit\ndef kernel_static_worker_config():\n    pass\n"
        pool = SimpleNamespace(submit=Mock(return_value=object()))

        try:
            with (
                config.patch(
                    {
                        "autotune_queue_static_precompile": True,
                        "compile_threads": 2,
                    }
                ),
                patch.object(AsyncCompile, "use_process_pool", return_value=True),
                patch.object(AsyncCompile, "process_pool", return_value=pool),
            ):
                AsyncCompile().triton("kernel_static_worker_config", source)

            pool.submit.assert_called_once()
            extra_config = pool.submit.call_args.args[3]
            self.assertTrue(extra_config["autotune_queue_static_precompile"])
        finally:
            CompiledTritonKernels.remove_future(source)

    def test_compile_worker_uses_static_precompile_flag(self):
        from torch._inductor.runtime.compile_tasks import _worker_compile_triton

        class FakeKernel:
            inductor_meta = {
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_policy": "auto",
            }
            triton_meta = {"device_type": "cuda"}
            heuristic_type = HeuristicType.PERSISTENT_REDUCTION
            launchers = []
            compile_results = []
            configs = [
                triton.Config({"XBLOCK": 1}, num_warps=4, num_stages=1),
                triton.Config({"XBLOCK": 2}, num_warps=4, num_stages=1),
                triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1),
            ]
            mutated_arg_names = []
            reset_to_zero_arg_names = []

            def __init__(self):
                self.precompile_calls = []
                self.prepare_for_pickle_calls = 0

            def precompile(self, **kwargs):
                self.precompile_calls.append(kwargs)

            def prepare_for_pickle(self):
                self.prepare_for_pickle_calls += 1

        kernel = FakeKernel()
        result, _elapsed_us = _worker_compile_triton(
            lambda: kernel,
            {},
            {
                "autotune_queue_static_precompile": True,
                "compile_threads": 2,
            },
        )

        self.assertIs(result, kernel)
        self.assertEqual(
            kernel.precompile_calls,
            [{"warm_cache_only": True, "max_configs": 1}],
        )
        self.assertEqual(kernel.prepare_for_pickle_calls, 1)

    def test_static_autotuner_cache_hit_gets_process_pool_compile_submitter(self):
        from torch._inductor.async_compile import AsyncCompile, CompiledTritonKernels
        from torch._inductor.codecache import StaticAutotunerFuture

        def make_autotuner(*, coordinate_descent_tuning):
            autotuner = object.__new__(CachingAutotuner)
            autotuner._config_compile_submitter = None
            autotuner.recheck_autotune_cache = lambda reload_kernel_from_src: None
            autotuner.precompile = lambda **kwargs: None
            autotuner.inductor_meta = {
                "coordinate_descent_tuning": coordinate_descent_tuning,
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_policy": "all",
            }
            autotuner.triton_meta = {"device_type": "cuda"}
            autotuner.heuristic_type = HeuristicType.REDUCTION
            autotuner.deterministic_mode = False
            autotuner.launchers = [
                SimpleNamespace(
                    config=triton.Config(
                        {"XBLOCK": 1}, num_warps=4, num_stages=1
                    )
                )
            ]
            return autotuner

        source = "@triton.jit\ndef kernel():\n    pass\n"
        ineligible_source = "@triton.jit\ndef ineligible_kernel():\n    pass\n"
        autotuner = make_autotuner(coordinate_descent_tuning=True)
        ineligible_autotuner = make_autotuner(coordinate_descent_tuning=False)
        CompiledTritonKernels.save(source, StaticAutotunerFuture(autotuner))
        CompiledTritonKernels.save(
            ineligible_source, StaticAutotunerFuture(ineligible_autotuner)
        )

        try:
            with (
                config.patch({"compile_threads": 2}),
                patch.object(AsyncCompile, "use_process_pool", return_value=False),
            ):
                result = AsyncCompile().triton("kernel", source)
                ineligible_result = AsyncCompile().triton(
                    "ineligible_kernel", ineligible_source
                )

            self.assertIs(result, autotuner)
            self.assertIsNotNone(result._config_compile_submitter)
            self.assertIs(ineligible_result, ineligible_autotuner)
            self.assertIsNone(ineligible_result._config_compile_submitter)
            self.assertIsNone(CompiledTritonKernels.get(source))
            self.assertIsNone(CompiledTritonKernels.get(ineligible_source))
        finally:
            CompiledTritonKernels.remove_future(source)
            CompiledTritonKernels.remove_future(ineligible_source)


if __name__ == "__main__":
    run_tests()
