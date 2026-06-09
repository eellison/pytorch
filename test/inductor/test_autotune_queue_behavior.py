# Owner(s): ["module: inductor"]

import contextlib
import functools
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.runtime.autotune_common import (
    apply_effective_coordinate_descent_queue_metadata,
    expected_autotune_queue_calls,
)
from torch._inductor.runtime.autotune_queue import (
    autotune_queue,
    get_active_autotune_queue,
)
from torch._inductor.runtime.hints import HeuristicType
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._inductor.test_case import run_tests, TestCase


try:
    import triton  # @manual
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904


class TestAutotuneQueueBehavior(TestCase):
    @staticmethod
    def _autotune_queue_patch(**overrides):
        config_values = {
            "compile_threads": 2,
            "coordinate_descent_tuning": True,
            "coordinate_descent_tuning_batch": True,
        }
        config_values.update(overrides)
        return config.patch(config_values)

    def _make_autotuner(
        self,
        arg_names,
        *,
        inductor_meta,
        mutated_arg_names=(),
        launcher=None,
        with_coordesc_cache=False,
    ):
        launcher = launcher or SimpleNamespace(
            config=triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1),
            store_cubin=False,
        )
        autotuner = object.__new__(CachingAutotuner)
        autotuner.fn = SimpleNamespace(arg_names=list(arg_names))
        autotuner.inductor_meta = inductor_meta
        autotuner.mutated_arg_names = list(mutated_arg_names)
        autotuner.device_props = SimpleNamespace(type="cpu", index=None)
        autotuner.launchers = [launcher]
        autotuner._cached_launcher = None
        autotuner.get_device_interface = lambda: SimpleNamespace(
            is_available=lambda: False
        )
        if with_coordesc_cache:
            autotuner.coordesc_tuner = SimpleNamespace(
                lookup_in_cache=lambda config, *, benchmark_mode=None: None,
            )
            autotuner._ensure_kernel_loaded = lambda: None
        return autotuner, launcher

    def test_live_bytes_skips_oversized_task(self):
        calls = []

        class FakeTask:
            retained_arg_bytes = 32
            queue_bytes = 32

            def run_scalar(self):
                calls.append("run_scalar")

        with (
            self._autotune_queue_patch(
                coordinate_descent_tuning_batch_max_live_bytes=16,
            ),
            patch(
                "torch._inductor.runtime.autotune_queue._CoordinateDescentTask",
                return_value=FakeTask(),
            ),
            autotune_queue(disposable_args=True) as batch,
        ):
            self.assertIs(get_active_autotune_queue(), batch)
            queued = batch.enqueue(object(), object(), (), {}, None, False)

        self.assertFalse(queued)
        self.assertEqual(calls, ["run_scalar"])
        self.assertEqual(batch.tasks, [])

    def test_live_bytes_budget_skips_before_task(self):
        autotuner, launcher = self._make_autotuner(
            ["out"],
            inductor_meta={
                "coordinate_descent_tuning": True,
                "coordinate_descent_tuning_batch": True,
            },
        )

        def coordinate_descent_tuning(launcher, *args, **kwargs):
            calls.append("scalar")
            self.assertIsNone(kwargs["use_batch_benchmarking"])
            launcher.config.found_by_coordesc = True
            return launcher

        calls = []
        autotuner.coordinate_descent_tuning = coordinate_descent_tuning

        with (
            self._autotune_queue_patch(
                coordinate_descent_tuning_batch_max_live_bytes=1,
            ),
            patch(
                "torch._inductor.runtime.autotune_queue._CoordinateDescentTask",
                side_effect=AssertionError("oversized task should not be created"),
            ),
            autotune_queue(disposable_args=True) as batch,
        ):
            self.assertIs(get_active_autotune_queue(), batch)
            queued = batch.enqueue(
                autotuner,
                launcher,
                (torch.zeros(16),),
                {},
                None,
                False,
            )

        self.assertFalse(queued)
        self.assertEqual(calls, ["scalar"])
        self.assertEqual(batch.tasks, [])

    def test_skips_tasks_requiring_arg_clones(self):
        autotuner, launcher = self._make_autotuner(
            ["inp"],
            inductor_meta={"mutated_input_arg_names": ["inp"]},
            mutated_arg_names=["inp"],
            with_coordesc_cache=True,
        )

        counters.clear()
        with (
            self._autotune_queue_patch(
                coordinate_descent_tuning_batch_max_live_bytes=0,
            ),
            autotune_queue(disposable_args=True) as batch,
        ):
            queued = batch.enqueue(
                autotuner,
                launcher,
                (torch.zeros(1),),
                {},
                None,
                False,
            )

        self.assertFalse(queued)
        self.assertEqual(batch.tasks, [])
        self.assertEqual(
            counters["inductor"]["autotune_queue_clone_arg_skips"],
            1,
        )

    def test_expected_calls_respects_min_kernels(self):
        with autotune_queue(
            expected_calls=1, disposable_args=True
        ) as batch:
            self.assertIsNone(get_active_autotune_queue())
            self.assertFalse(batch.enqueue(object(), object(), (), {}, None, False))
        with autotune_queue(expected_calls=2):
            self.assertIsNone(get_active_autotune_queue())

        with config.patch(
            {
                "coordinate_descent_tuning": True,
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_min_kernels": 2,
                "compile_threads": 2,
            }
        ):
            with autotune_queue(
                expected_calls=2, disposable_args=True
            ) as batch:
                self.assertIs(get_active_autotune_queue(), batch)

        with config.patch(
            {
                "coordinate_descent_tuning": True,
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_min_kernels": 0,
                "compile_threads": 2,
            }
        ):
            with autotune_queue(expected_calls=0):
                self.assertIsNone(get_active_autotune_queue())
            with autotune_queue(
                expected_calls=1, disposable_args=True
            ) as batch:
                self.assertIs(get_active_autotune_queue(), batch)

        with config.patch(
            {
                "coordinate_descent_tuning": False,
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_min_kernels": 1,
                "compile_threads": 2,
            }
        ):
            with autotune_queue(
                expected_calls=2, disposable_args=True
            ) as batch:
                self.assertIs(get_active_autotune_queue(), batch)

        with config.patch(
            {
                "coordinate_descent_tuning_batch": False,
                "coordinate_descent_tuning_batch_min_kernels": 1,
            }
        ):
            with autotune_queue(expected_calls=2, disposable_args=True):
                self.assertIsNone(get_active_autotune_queue())

    def test_disabled_queue_forces_scalar_coordesc_metadata(self):
        inductor_meta = {
            "coordinate_descent_tuning": True,
            "coordinate_descent_tuning_batch_requested": True,
            "coordinate_descent_tuning_batch_policy": "all",
        }
        triton_meta = {"device_type": "cuda"}

        with config.patch(
            {
                "coordinate_descent_tuning_batch": False,
                "compile_threads": 2,
            }
        ):
            enabled = apply_effective_coordinate_descent_queue_metadata(
                inductor_meta,
                triton_meta,
                HeuristicType.REDUCTION,
                False,
                keep_disabled=True,
            )

        self.assertFalse(enabled)
        self.assertTrue(inductor_meta["coordinate_descent_tuning_batch_requested"])
        self.assertFalse(inductor_meta["coordinate_descent_tuning_batch"])
        self.assertEqual(inductor_meta["coordinate_descent_tuning_batch_policy"], "all")

    def test_effective_coordesc_metadata_preserves_requested_policy(self):
        inductor_meta = {
            "coordinate_descent_tuning": True,
            "coordinate_descent_tuning_batch_requested": True,
            "coordinate_descent_tuning_batch_policy": "none",
        }
        triton_meta = {"device_type": "cuda"}

        with config.patch({"coordinate_descent_tuning_batch": True}):
            enabled = apply_effective_coordinate_descent_queue_metadata(
                inductor_meta,
                triton_meta,
                HeuristicType.REDUCTION,
                False,
            )
            enabled_after_reapply = apply_effective_coordinate_descent_queue_metadata(
                inductor_meta,
                triton_meta,
                HeuristicType.REDUCTION,
                False,
            )

        self.assertFalse(enabled)
        self.assertFalse(enabled_after_reapply)
        self.assertTrue(inductor_meta["coordinate_descent_tuning_batch_requested"])
        self.assertNotIn("coordinate_descent_tuning_batch", inductor_meta)
        self.assertEqual(
            inductor_meta["coordinate_descent_tuning_batch_policy"], "none"
        )

    def test_skips_when_compile_threads_one(self):
        counters.clear()
        with config.patch(
            {
                "compile_threads": 1,
                "coordinate_descent_tuning": True,
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_min_kernels": 1,
            }
        ):
            with autotune_queue(expected_calls=2, disposable_args=True):
                self.assertIsNone(get_active_autotune_queue())

        self.assertEqual(
            counters["inductor"]["autotune_queue_compile_thread_skips"],
            1,
        )

        counters.clear()
        with (
            config.patch(
                {
                    "compile_threads": None,
                    "coordinate_descent_tuning": True,
                    "coordinate_descent_tuning_batch": True,
                    "coordinate_descent_tuning_batch_min_kernels": 1,
                }
            ),
            patch("torch._inductor.async_compile.get_compile_threads", return_value=1),
        ):
            with autotune_queue(expected_calls=2, disposable_args=True):
                self.assertIsNone(get_active_autotune_queue())

        self.assertEqual(
            counters["inductor"]["autotune_queue_compile_thread_skips"],
            1,
        )

        counters.clear()
        with config.patch(
            {
                "compile_threads": 1,
                "coordinate_descent_tuning": True,
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_min_kernels": 1,
            }
        ):
            with autotune_queue(expected_calls=0, disposable_args=True):
                self.assertIsNone(get_active_autotune_queue())

        self.assertEqual(
            counters["inductor"]["autotune_queue_compile_thread_skips"],
            1,
        )

        with config.patch(
            {
                "compile_threads": 2,
                "coordinate_descent_tuning": True,
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_min_kernels": 1,
            }
        ):
            with autotune_queue(
                expected_calls=2, disposable_args=True
            ) as batch:
                self.assertIs(get_active_autotune_queue(), batch)

    def test_run_kernel_autotune_calls_uses_expected_calls(self):
        from torch._inductor.codegen.wrapper import PythonWrapperCodegen
        from torch._inductor.utils import IndentedBuffer
        from torch.utils._ordered_set import OrderedSet

        active = []
        kernel0 = object()
        multi_kernel_0 = object()
        wrapper = SimpleNamespace(
            kernel_autotune_calls=IndentedBuffer(),
            kernel_autotune_names=OrderedSet(["kernel0", "multi_kernel_0"]),
        )
        wrapper.kernel_autotune_calls.writeline("calls.append(bool(active))")
        wrapper._should_batch_kernel_autotune_calls = functools.partial(
            PythonWrapperCodegen._should_batch_kernel_autotune_calls,
            wrapper,
        )
        wrapper._autotune_queue_min_kernels = functools.partial(
            PythonWrapperCodegen._autotune_queue_min_kernels,
            wrapper,
        )
        wrapper._has_enough_autotune_kernels_for_queue = functools.partial(
            PythonWrapperCodegen._has_enough_autotune_kernels_for_queue,
            wrapper,
        )
        wrapper._should_defer_static_autotune_precompile = functools.partial(
            PythonWrapperCodegen._should_defer_static_autotune_precompile,
            wrapper,
        )

        with config.patch({"autotune_queue": True, "autotune_queue_min_kernels": 3}):
            self.assertFalse(wrapper._should_defer_static_autotune_precompile())
        with config.patch({"autotune_queue": True, "autotune_queue_min_kernels": 2}):
            self.assertTrue(wrapper._should_defer_static_autotune_precompile())

        @contextlib.contextmanager
        def autotune_queue_context(**kwargs):
            active.append(kwargs)
            try:
                yield
            finally:
                active.pop()

        with config.patch(
            {
                "coordinate_descent_tuning": True,
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_min_kernels": 2,
            }
        ):
            with patch(
                "torch._inductor.codegen.wrapper.autotune_common.expected_autotune_queue_calls",
                side_effect=lambda kernels: self.assertEqual(
                    kernels, [kernel0, multi_kernel_0]
                )
                or 2,
            ) as expected_calls, patch(
                "torch._inductor.codegen.wrapper.autotune_queue.autotune_queue",
                side_effect=autotune_queue_context,
            ) as autotune_queue:
                scope = {
                    "active": active,
                    "calls": [],
                    "kernel0": kernel0,
                    "multi_kernel_0": multi_kernel_0,
                }
                PythonWrapperCodegen.run_kernel_autotune_calls(wrapper, scope)

        expected_calls.assert_called_once()
        autotune_queue.assert_called_once_with(
            expected_calls=2,
            disposable_args=True,
        )
        self.assertEqual(scope["calls"], [True])

        with config.patch(
            {
                "coordinate_descent_tuning": True,
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_min_kernels": 2,
            }
        ):
            with patch(
                "torch._inductor.codegen.wrapper.autotune_common.expected_autotune_queue_calls",
                return_value=1,
            ), patch(
                "torch._inductor.codegen.wrapper.autotune_queue.autotune_queue",
                side_effect=autotune_queue_context,
            ) as autotune_queue:
                scope = {
                    "active": active,
                    "calls": [],
                    "kernel0": kernel0,
                    "multi_kernel_0": multi_kernel_0,
                }
                PythonWrapperCodegen.run_kernel_autotune_calls(wrapper, scope)
        autotune_queue.assert_not_called()
        self.assertEqual(scope["calls"], [False])

        with config.patch(
            {
                "coordinate_descent_tuning": True,
                "coordinate_descent_tuning_batch": True,
                "aot_inductor.autotune_per_kernel_alloc": True,
            }
        ):
            scope = {
                "active": active,
                "calls": [],
                "kernel0": kernel0,
                "multi_kernel_0": multi_kernel_0,
            }
            PythonWrapperCodegen.run_kernel_autotune_calls(wrapper, scope)
        self.assertEqual(scope["calls"], [False])

        with config.patch(
            {
                "coordinate_descent_tuning": True,
                "coordinate_descent_tuning_batch": True,
                "triton.autotune_with_sample_inputs": True,
            }
        ):
            with patch(
                "torch._inductor.codegen.wrapper.autotune_common.expected_autotune_queue_calls",
                return_value=2,
            ), patch(
                "torch._inductor.codegen.wrapper.autotune_queue.autotune_queue",
                side_effect=autotune_queue_context,
            ) as autotune_queue:
                scope = {
                    "active": active,
                    "calls": [],
                    "kernel0": kernel0,
                    "multi_kernel_0": multi_kernel_0,
                }
                PythonWrapperCodegen.run_kernel_autotune_calls(wrapper, scope)
        autotune_queue.assert_not_called()
        self.assertEqual(scope["calls"], [False])

    def test_expected_autotune_queue_calls(self):
        class Launcher:
            def __init__(self):
                self.config = triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1)

        def make_autotuner(
            heuristic_type,
            *,
            policy="auto",
            native_matmul=False,
            device_type="cuda",
            coordinate_descent=True,
            extra_launchers=0,
            mutated_input=False,
        ):
            autotuner = object.__new__(CachingAutotuner)
            autotuner.inductor_meta = {
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_policy": policy,
            }
            if coordinate_descent:
                autotuner.inductor_meta["coordinate_descent_tuning"] = True
            if mutated_input:
                autotuner.inductor_meta["mutated_input_arg_names"] = ["inp"]
            autotuner.triton_meta = {"native_matmul": native_matmul}
            if device_type is not None:
                autotuner.triton_meta["device_type"] = device_type
            autotuner.heuristic_type = heuristic_type
            autotuner.deterministic_mode = False
            autotuner.launchers = [Launcher() for _ in range(1 + extra_launchers)]
            return autotuner

        pointwise = make_autotuner(HeuristicType.POINTWISE)
        reduction = make_autotuner(HeuristicType.REDUCTION)
        forced_pointwise = make_autotuner(HeuristicType.POINTWISE, policy="all")
        native_matmul = make_autotuner(HeuristicType.POINTWISE, native_matmul=True)
        effective_disabled = make_autotuner(HeuristicType.REDUCTION, policy="all")
        effective_disabled.inductor_meta.update(
            {
                "coordinate_descent_tuning_batch_requested": True,
                "coordinate_descent_tuning_batch": False,
            }
        )
        static_reduction = make_autotuner(
            HeuristicType.REDUCTION,
            coordinate_descent=False,
            extra_launchers=2,
        )

        cases = [
            (pointwise, 0),
            (reduction, 1),
            (make_autotuner(HeuristicType.PERSISTENT_REDUCTION), 1),
            (make_autotuner(HeuristicType.SPLIT_SCAN), 1),
            (forced_pointwise, 1),
            (make_autotuner(HeuristicType.REDUCTION, policy="none"), 0),
            (make_autotuner(HeuristicType.POINTWISE, coordinate_descent=False), 0),
            (static_reduction, 1),
            (
                make_autotuner(
                    HeuristicType.REDUCTION,
                    coordinate_descent=False,
                    extra_launchers=1,
                    mutated_input=True,
                ),
                0,
            ),
            (
                make_autotuner(
                    HeuristicType.POINTWISE,
                    coordinate_descent=False,
                    extra_launchers=2,
                    native_matmul=True,
                ),
                1,
            ),
            (
                make_autotuner(
                    HeuristicType.POINTWISE,
                    policy="all",
                    coordinate_descent=False,
                    extra_launchers=2,
                ),
                1,
            ),
            (
                make_autotuner(
                    HeuristicType.POINTWISE,
                    policy="reductions",
                    coordinate_descent=False,
                    extra_launchers=1,
                ),
                0,
            ),
            (
                make_autotuner(
                    HeuristicType.POINTWISE,
                    policy="none",
                    coordinate_descent=False,
                    extra_launchers=1,
                ),
                0,
            ),
            (native_matmul, 1),
            (effective_disabled, 0),
            (
                make_autotuner(
                    HeuristicType.REDUCTION,
                    policy="all",
                    device_type="xpu",
                ),
                0,
            ),
            (make_autotuner(HeuristicType.REDUCTION, device_type="hip"), 1),
            (make_autotuner(HeuristicType.REDUCTION, mutated_input=True), 0),
            (SimpleNamespace(kernels=[pointwise, reduction, forced_pointwise]), 0),
            (SimpleNamespace(kernels=[pointwise, reduction]), 0),
        ]
        for kernel, expected in cases:
            self.assertEqual(expected_autotune_queue_calls([kernel]), expected)

        with config.patch({"compile_threads": 1}):
            self.assertEqual(
                expected_autotune_queue_calls(
                    [reduction, forced_pointwise, native_matmul]
                ),
                3,
            )

        reduction.launchers[0].config.found_by_coordesc = True
        self.assertEqual(expected_autotune_queue_calls([reduction]), 0)


if __name__ == "__main__":
    run_tests()
