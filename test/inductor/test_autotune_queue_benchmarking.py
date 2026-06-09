# Owner(s): ["module: inductor"]

import contextlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch
from torch._inductor.runtime.autotune_benchmarking import (
    _benchmark_launcher_requests,
    _LauncherBenchmarkRequest,
)
from torch._inductor.runtime.autotune_common import _COORDESC_UNGROUPED_BENCHMARK_KEY
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._inductor.test_case import run_tests, TestCase


try:
    import triton  # @manual
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904


class TestAutotuneQueueBenchmarking(TestCase):
    def _make_benchmark_request_autotuner(
        self,
        *,
        device_type="cpu",
        device_idx=None,
        stream="stream",
        available=None,
    ):
        autotuner = object.__new__(CachingAutotuner)
        autotuner.custom_kernel = False
        autotuner.inductor_meta = {}
        autotuner.benchmark_failure_reasons = {}
        autotuner.device_props = SimpleNamespace(type=device_type, index=device_idx)
        autotuner._skip_config_due_to_register_spilling = lambda launcher: False
        autotuner.copy_args_to_cpu_if_needed = lambda *args, **kwargs: {}
        autotuner._make_benchmark_call = (
            lambda launcher, cpu_copies, stream, args, kwargs, clone_args=True: (
                lambda: None
            )
        )

        is_available = device_type != "cpu" if available is None else available
        device_interface = MagicMock()
        device_interface.is_available.return_value = is_available
        device_interface.current_device.return_value = 0
        device_interface.get_raw_stream.return_value = stream
        device_interface.device.return_value = contextlib.nullcontext()
        autotuner.get_device_interface = lambda: device_interface
        return autotuner, device_interface

    def test_benchmark_all_launchers_uses_benchmark_many(self):
        autotuner, _device_interface = self._make_benchmark_request_autotuner(
            device_type="cuda", device_idx=0
        )

        calls = []

        def make_benchmark_call(
            launcher, cpu_copies, stream, args, kwargs, clone_args=True
        ):
            def call():
                calls.append((launcher, cpu_copies, stream, args, kwargs))

            return call

        autotuner._make_benchmark_call = make_benchmark_call

        launcher0 = MagicMock(n_spills=None)
        launcher1 = MagicMock(n_spills=None)

        def benchmark_many(callables, **kwargs):
            for call in callables:
                call()
            self.assertEqual(kwargs["device"], torch.device("cuda", 0))
            self.assertEqual(kwargs["rep"], 40)
            self.assertTrue(kwargs["is_vetted_benchmarking"])
            return [2.0, 1.0]

        with patch(
            "torch._inductor.runtime.autotune_benchmarking.benchmarker.benchmark_many",
            side_effect=benchmark_many,
        ) as mock_benchmark_many:
            timings = autotuner.benchmark_all_launchers([launcher0, launcher1], "arg")

        self.assertEqual(timings, {launcher0: 2.0, launcher1: 1.0})
        self.assertEqual([call[0] for call in calls], [launcher0, launcher1])
        mock_benchmark_many.assert_called_once()

    def test_benchmark_launcher_requests_use_current_stream(self):
        autotuner, _device_interface = self._make_benchmark_request_autotuner(
            device_type="cuda", stream="current-stream"
        )

        streams = []

        def make_benchmark_call(
            launcher, cpu_copies, stream, args, kwargs, clone_args=True
        ):
            def call():
                streams.append(stream)

            return call

        autotuner._make_benchmark_call = make_benchmark_call
        launcher = MagicMock(n_spills=None)

        def benchmark_many(callables, **kwargs):
            for call in callables:
                call()
            return [1.0]

        with patch(
            "torch._inductor.runtime.autotune_benchmarking.benchmarker.benchmark_many",
            side_effect=benchmark_many,
        ):
            timings = _benchmark_launcher_requests(
                [
                    _LauncherBenchmarkRequest(
                        autotuner=autotuner,
                        launcher=launcher,
                        args=(),
                        kwargs={},
                        device_idx=0,
                    )
                ]
            )

        self.assertEqual(timings, [1.0])
        self.assertEqual(streams, ["current-stream"])

    def test_benchmark_launcher_requests_waits_ready_before_cpu_copy(self):
        order = []
        autotuner, _device_interface = self._make_benchmark_request_autotuner()

        def copy_args_to_cpu_if_needed(*args, **kwargs):
            order.append("copy")
            return {}

        def make_benchmark_call(
            launcher, cpu_copies, stream, args, kwargs, clone_args=True
        ):
            def call():
                order.append("run")

            return call

        autotuner.copy_args_to_cpu_if_needed = copy_args_to_cpu_if_needed
        autotuner._make_benchmark_call = make_benchmark_call
        launcher = MagicMock(n_spills=None)

        def benchmark_many(callables, **kwargs):
            self.assertEqual(len(kwargs["setup_fns"]), 1)
            kwargs["setup_fns"][0]()
            callables[0]()
            return [1.0]

        with patch(
            "torch._inductor.runtime.autotune_benchmarking.benchmarker.benchmark_many",
            side_effect=benchmark_many,
        ):
            timings = _benchmark_launcher_requests(
                [
                    _LauncherBenchmarkRequest(
                        autotuner=autotuner,
                        launcher=launcher,
                        args=(),
                        kwargs={},
                        wait_ready_fn=lambda: order.append("wait"),
                        setup_fn=lambda: order.append("setup"),
                    )
                ]
            )

        self.assertEqual(timings, [1.0])
        self.assertEqual(order, ["wait", "copy", "setup", "run"])

    def test_benchmark_launcher_requests_can_skip_arg_cloning(self):
        autotuner, _device_interface = self._make_benchmark_request_autotuner()
        autotuner.copy_args_to_cpu_if_needed = Mock(
            side_effect=AssertionError("no-clone request should not copy args")
        )

        clone_arg_flags = []

        def make_benchmark_call(
            launcher, cpu_copies, stream, args, kwargs, clone_args=True
        ):
            clone_arg_flags.append(clone_args)
            self.assertEqual(cpu_copies, {})
            return lambda: None

        autotuner._make_benchmark_call = make_benchmark_call
        launcher = MagicMock(n_spills=None)

        with patch(
            "torch._inductor.runtime.autotune_benchmarking.benchmarker.benchmark_many",
            return_value=[1.0],
        ):
            timings = _benchmark_launcher_requests(
                [
                    _LauncherBenchmarkRequest(
                        autotuner=autotuner,
                        launcher=launcher,
                        args=(),
                        kwargs={},
                        clone_args=False,
                    )
                ]
            )

        self.assertEqual(timings, [1.0])
        self.assertEqual(clone_arg_flags, [False])
        autotuner.copy_args_to_cpu_if_needed.assert_not_called()

    def test_benchmark_launcher_requests_forwards_benchmark_groups(self):
        autotuner, _device_interface = self._make_benchmark_request_autotuner(
            device_type="cuda", device_idx=0
        )
        launcher0 = MagicMock(n_spills=None)
        launcher1 = MagicMock(n_spills=None)
        launcher2 = MagicMock(n_spills=None)
        group_key = object()
        group_state = {}

        benchmark_many_calls = []

        def benchmark_many(callables, **kwargs):
            benchmark_many_calls.append(kwargs)
            if "benchmark_group_keys" in kwargs:
                self.assertEqual(len(callables), 2)
                self.assertEqual(
                    kwargs["benchmark_group_keys"][0],
                    kwargs["benchmark_group_keys"][1],
                )
                self.assertIs(kwargs["benchmark_group_keys"][0], group_key)
                self.assertIs(kwargs["benchmark_group_states"][0], group_state)
                self.assertIs(kwargs["benchmark_group_states"][1], group_state)
                return [1.0, 2.0]

            self.assertEqual(len(callables), 1)
            return [3.0]

        with patch(
            "torch._inductor.runtime.autotune_benchmarking.benchmarker.benchmark_many",
            side_effect=benchmark_many,
        ), patch(
            "torch._inductor.runtime.autotune_benchmarking.benchmarker.supports_grouped_benchmark_many",
            True,
        ):
            timings = _benchmark_launcher_requests(
                [
                    _LauncherBenchmarkRequest(
                        autotuner=autotuner,
                        launcher=launcher0,
                        args=(),
                        kwargs={},
                        benchmark_group_key=group_key,
                        benchmark_group_state=group_state,
                    ),
                    _LauncherBenchmarkRequest(
                        autotuner=autotuner,
                        launcher=launcher1,
                        args=(),
                        kwargs={},
                        benchmark_group_key=group_key,
                        benchmark_group_state=group_state,
                    ),
                    _LauncherBenchmarkRequest(
                        autotuner=autotuner,
                        launcher=launcher2,
                        args=(),
                        kwargs={},
                    ),
                ]
            )

        self.assertEqual(timings, [1.0, 2.0, 3.0])
        self.assertEqual(len(benchmark_many_calls), 2)
        self.assertIn("benchmark_group_keys", benchmark_many_calls[0])
        self.assertNotIn("benchmark_group_keys", benchmark_many_calls[1])

    def test_benchmark_launcher_requests_marks_legacy_benchmark_many_ungrouped(self):
        autotuner, _device_interface = self._make_benchmark_request_autotuner(
            device_type="cuda", device_idx=0
        )
        launcher = MagicMock(n_spills=None)
        group_state = {}

        class LegacyBenchmarker:
            def __init__(self):
                self.kwargs = None

            def benchmark_many(self, callables, device=None, setup_fns=None, **kwargs):
                self.kwargs = kwargs
                return [1.0 for _ in callables]

        legacy_benchmarker = LegacyBenchmarker()
        with patch(
            "torch._inductor.runtime.autotune_benchmarking.benchmarker",
            legacy_benchmarker,
        ):
            timings = _benchmark_launcher_requests(
                [
                    _LauncherBenchmarkRequest(
                        autotuner=autotuner,
                        launcher=launcher,
                        args=(),
                        kwargs={},
                        benchmark_group_key=object(),
                        benchmark_group_state=group_state,
                    )
                ]
            )

        self.assertEqual(timings, [1.0])
        self.assertNotIn("benchmark_group_keys", legacy_benchmarker.kwargs)
        self.assertEqual(group_state[_COORDESC_UNGROUPED_BENCHMARK_KEY], 1)

    def test_benchmark_launcher_requests_normalizes_hip_benchmark_device(self):
        autotuner, _device_interface = self._make_benchmark_request_autotuner(
            device_type="hip", device_idx=0
        )
        launcher = MagicMock(n_spills=None)
        group_state = {"benchmark_iters": 4}

        def benchmark_many(callables, **kwargs):
            self.assertEqual(kwargs["device"], torch.device("cuda", 0))
            self.assertEqual(kwargs["benchmark_group_states"], [group_state])
            return [1.0]

        with patch(
            "torch._inductor.runtime.autotune_benchmarking.benchmarker.benchmark_many",
            side_effect=benchmark_many,
        ), patch(
            "torch._inductor.runtime.autotune_benchmarking.benchmarker.supports_grouped_benchmark_many",
            True,
        ):
            timings = _benchmark_launcher_requests(
                [
                    _LauncherBenchmarkRequest(
                        autotuner=autotuner,
                        launcher=launcher,
                        args=(),
                        kwargs={},
                        benchmark_group_key=object(),
                        benchmark_group_state=group_state,
                    )
                ]
            )

        self.assertEqual(timings, [1.0])

    def test_make_benchmark_call_can_skip_arg_cloning(self):
        autotuner = object.__new__(CachingAutotuner)
        autotuner.inductor_meta = {}
        autotuner.maybe_clone_args = Mock(
            side_effect=AssertionError("no-clone benchmark should not clone args")
        )
        reset_args = []
        autotuner.reset_to_zero_args = lambda *args, **kwargs: reset_args.append(
            (args, kwargs)
        )
        autotuner.restore_args_from_cpu = Mock()

        launcher = Mock()
        launcher.config = triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=1)
        arg = object()

        call = CachingAutotuner._make_benchmark_call(
            autotuner,
            launcher,
            {},
            0,
            (arg,),
            {},
            clone_args=False,
        )
        call()

        autotuner.maybe_clone_args.assert_not_called()
        self.assertEqual(reset_args, [((arg,), {})])
        launcher.assert_called_once_with(arg, stream=0)
        autotuner.restore_args_from_cpu.assert_called_once_with({})

    def test_benchmark_launcher_requests_validates_result_count(self):
        autotuner, _device_interface = self._make_benchmark_request_autotuner()
        launcher0 = MagicMock(n_spills=None)
        launcher1 = MagicMock(n_spills=None)

        with (
            patch(
                "torch._inductor.runtime.autotune_benchmarking.benchmarker.benchmark_many",
                return_value=[1.0],
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "Grouped launcher benchmark returned 1 results for 2 requests",
            ),
        ):
            _benchmark_launcher_requests(
                [
                    _LauncherBenchmarkRequest(
                        autotuner=autotuner,
                        launcher=launcher0,
                        args=(),
                        kwargs={},
                    ),
                    _LauncherBenchmarkRequest(
                        autotuner=autotuner,
                        launcher=launcher1,
                        args=(),
                        kwargs={},
                    ),
                ]
            )


if __name__ == "__main__":
    run_tests()
