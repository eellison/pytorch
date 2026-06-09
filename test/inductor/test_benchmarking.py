# Owner(s): ["module: inductor"]

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch._inductor.config import (
    inductor_default_autotune_rep,
    inductor_default_autotune_warmup,
)
from torch._inductor.runtime.benchmarking import (
    Benchmarker,
    InductorBenchmarker,
    TritonBenchmarker,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    decorateIf,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


ALL_BENCHMARKER_CLASSES = (
    Benchmarker,
    TritonBenchmarker,
)


@instantiate_parametrized_tests
class TestBenchmarker(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(12345)
        counters.clear()

    @staticmethod
    def get_counter_value(benchmarker_cls, fn_name):
        return counters["inductor"][
            f"benchmarking.{benchmarker_cls.__name__}.{fn_name}"
        ]

    @staticmethod
    def make_params(device, size=100):
        fn, fn_args, fn_kwargs = torch.sum, (torch.randn(size, device=device),), {}
        _callable = lambda: fn(*fn_args, **fn_kwargs)  # noqa: E731
        return (fn, fn_args, fn_kwargs), _callable

    class _FakeEvent:
        def __init__(self, elapsed_time=0.0, on_record=None):
            self.elapsed_time_ = elapsed_time
            self.on_record = on_record

        def record(self):
            if self.on_record is not None:
                self.on_record()

        def elapsed_time(self, end_event):
            return end_event.elapsed_time_

    class _FakeBuffer:
        def __init__(self):
            self.zero_count = 0

        def zero_(self):
            self.zero_count += 1

    @staticmethod
    def make_event_pairs(elapsed_times, get_event_pair_iters=None, on_record=None):
        def get_event_pairs(iters):
            if get_event_pair_iters is not None:
                get_event_pair_iters.append(iters)
            elapsed_time = elapsed_times.pop(0)
            return [
                (
                    TestBenchmarker._FakeEvent(on_record=on_record),
                    TestBenchmarker._FakeEvent(elapsed_time, on_record=on_record),
                )
                for _ in range(iters)
            ]

        return get_event_pairs

    @staticmethod
    def record_call(calls, name):
        def inner():
            calls.append(name)

        return inner

    @staticmethod
    @contextlib.contextmanager
    def patch_cuda_benchmarking(empty=None):
        if empty is None:
            empty = TestBenchmarker._FakeBuffer()
        with (
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.device", return_value=contextlib.nullcontext()),
            patch("torch.cuda.synchronize"),
            patch("torch.empty", return_value=empty),
        ):
            yield empty

    @staticmethod
    def benchmark_many_cuda(benchmarker, fns, **kwargs):
        benchmark_kwargs = {
            "device": "cuda",
            "estimation_iters": 3,
            "memory_warmup_iters": 0,
            "benchmark_iters": 10,
            "max_benchmark_duration": 25,
            "is_vetted_benchmarking": True,
        }
        benchmark_kwargs.update(kwargs)
        return benchmarker.benchmark_many(fns, **benchmark_kwargs)

    @unittest.skipIf(not HAS_CPU or not HAS_GPU, "requires CPU and GPU")
    @decorateIf(
        unittest.expectedFailure,
        lambda params: params["benchmarker_cls"] is Benchmarker
        and params["device"] == GPU_TYPE,
    )
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    @parametrize("device", (GPU_TYPE, "cpu"))
    def test_benchmark_smoke(self, benchmarker_cls, device):
        benchmarker = benchmarker_cls()
        (fn, fn_args, fn_kwargs), _ = self.make_params(device)
        timing = benchmarker.benchmark(fn, fn_args, fn_kwargs)
        self.assertGreater(timing, 0)
        self.assertEqual(self.get_counter_value(benchmarker_cls, "benchmark"), 1)
        self.assertEqual(
            self.get_counter_value(
                benchmarker_cls, "benchmark_cpu" if device == "cpu" else "benchmark_gpu"
            ),
            1,
        )

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmark_cpu_smoke(self, benchmarker_cls, device="cpu"):
        benchmarker = benchmarker_cls()
        _, _callable = self.make_params(device)
        timing = benchmarker.benchmark_cpu(_callable)
        self.assertGreater(timing, 0)
        self.assertEqual(self.get_counter_value(benchmarker_cls, "benchmark_cpu"), 1)

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmark_many_cpu_smoke(self, benchmarker_cls, device="cpu"):
        benchmarker = benchmarker_cls()
        _, callable_1 = self.make_params(device, size=100)
        _, callable_2 = self.make_params(device, size=200)
        timings = benchmarker.benchmark_many(
            [callable_1, callable_2],
            device=device,
            warmup=1,
            rep=1,
        )
        self.assertEqual(len(timings), 2)
        self.assertTrue(all(timing > 0 for timing in timings))
        self.assertEqual(self.get_counter_value(benchmarker_cls, "benchmark_many"), 1)

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    def test_inductor_benchmark_many_cpu_fallback(self, device="cpu"):
        benchmarker = InductorBenchmarker()
        _, callable_1 = self.make_params(device, size=100)
        _, callable_2 = self.make_params(device, size=200)
        timings = benchmarker.benchmark_many(
            [callable_1, callable_2],
            device=device,
            warmup=1,
            rep=1,
        )
        self.assertEqual(len(timings), 2)
        self.assertTrue(all(timing > 0 for timing in timings))
        self.assertEqual(
            self.get_counter_value(InductorBenchmarker, "benchmark_many"), 1
        )

    def test_inductor_benchmark_many_cuda_mocked(self):
        benchmarker = InductorBenchmarker()
        benchmarker.L2_cache_size = 4
        elapsed_times = [5.0, 2.0, 4.0, 3.0]
        calls = []
        buffer = self._FakeBuffer()

        def invalid():
            raise RuntimeError("invalid configuration")

        benchmarker.get_event_pairs = self.make_event_pairs(elapsed_times)
        with self.patch_cuda_benchmarking(buffer):
            timings = self.benchmark_many_cuda(
                benchmarker,
                [self.record_call(calls, "a"), invalid, self.record_call(calls, "b")],
                estimation_iters=1,
                memory_warmup_iters=2,
                benchmark_iters=2,
                max_benchmark_duration=100,
            )

        self.assertEqual(timings, [4.0, float("inf"), 2.0])
        self.assertEqual(calls, ["a", "b", "a", "b", "a", "a", "b", "b"])
        self.assertEqual(buffer.zero_count, 11)

    def test_inductor_benchmark_many_l2_cache_size_is_per_device(self):
        benchmarker = InductorBenchmarker()

        with (
            patch("torch.cuda.current_device", return_value=0),
            patch(
                "torch.cuda.get_device_properties",
                side_effect=[
                    SimpleNamespace(L2_cache_size=4),
                    SimpleNamespace(L2_cache_size=8),
                ],
            ) as get_device_properties,
        ):
            self.assertEqual(benchmarker.L2_cache_size, 4)
            self.assertEqual(benchmarker._get_l2_cache_size(torch.device("cuda", 1)), 8)
            self.assertEqual(benchmarker._get_l2_cache_size(torch.device("cuda", 0)), 4)

        self.assertEqual(
            [call.args[0] for call in get_device_properties.call_args_list], [0, 1]
        )

    def test_inductor_benchmark_gpu_l2_cache_size_uses_requested_device(self):
        benchmarker = InductorBenchmarker()
        elapsed_times = [5.0, 4.0]
        empty_calls = []

        def get_device_properties(device):
            return SimpleNamespace(L2_cache_size={0: 4, 1: 8}[device])

        def empty(*args, **kwargs):
            empty_calls.append((args, kwargs))
            return self._FakeBuffer()

        benchmarker.get_event_pairs = self.make_event_pairs(elapsed_times)
        with (
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.device", return_value=contextlib.nullcontext()),
            patch(
                "torch.cuda.get_device_properties",
                side_effect=get_device_properties,
            ) as get_device_properties_mock,
            patch("torch.cuda.synchronize"),
            patch("torch.empty", side_effect=empty),
        ):
            benchmarker.benchmark(
                lambda: None,
                device=torch.device("cuda", 1),
                estimation_iters=1,
                memory_warmup_iters=0,
                benchmark_iters=1,
                max_benchmark_duration=100,
                is_vetted_benchmarking=True,
            )

        self.assertEqual(
            [call.args[0] for call in get_device_properties_mock.call_args_list], [1]
        )
        self.assertEqual(empty_calls[0][0], (2,))
        self.assertEqual(empty_calls[0][1]["device"], torch.device("cuda", 1))

    def test_inductor_benchmark_many_distorts_all_invalid_results(self):
        benchmarker = InductorBenchmarker()

        def invalid():
            raise RuntimeError("invalid configuration")

        with (
            inductor_config.patch(
                {"test_configs.distort_benchmarking_result": "inverse"}
            ),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.device", return_value=contextlib.nullcontext()),
            patch("torch.cuda.synchronize"),
        ):
            timings = benchmarker.benchmark_many(
                [invalid, invalid],
                device="cuda",
                is_vetted_benchmarking=True,
            )

        self.assertEqual(timings, [0.0, 0.0])

    def test_inductor_benchmark_many_runs_setup_outside_timing(self):
        benchmarker = InductorBenchmarker()
        benchmarker.L2_cache_size = 4
        calls = []

        benchmarker.get_event_pairs = self.make_event_pairs(
            [1.0, 1.0], on_record=lambda: calls.append("record")
        )

        with self.patch_cuda_benchmarking():
            self.benchmark_many_cuda(
                benchmarker,
                [lambda: calls.append("run")],
                setup_fns=[lambda: calls.append("setup")],
                estimation_iters=1,
                benchmark_iters=1,
                max_benchmark_duration=100,
            )

        self.assertEqual(
            calls,
            [
                "setup",
                "run",
                "setup",
                "record",
                "run",
                "record",
                "setup",
                "record",
                "run",
                "record",
            ],
        )

    def test_inductor_benchmark_many_tunes_iters_per_callable(self):
        benchmarker = InductorBenchmarker()
        benchmarker.L2_cache_size = 4
        elapsed_times = [10.0, 2.0, 10.0, 2.0]
        get_event_pair_iters = []
        calls = []

        benchmarker.get_event_pairs = self.make_event_pairs(
            elapsed_times, get_event_pair_iters
        )
        with self.patch_cuda_benchmarking():
            timings = self.benchmark_many_cuda(
                benchmarker,
                [self.record_call(calls, "slow"), self.record_call(calls, "fast")],
            )

        self.assertEqual(timings, [10.0, 2.0])
        self.assertEqual(get_event_pair_iters, [3, 3, 2, 10])
        self.assertEqual(
            calls,
            ["slow", "fast"]
            + ["slow"] * 3
            + ["fast"] * 3
            + ["slow"] * 2
            + ["fast"] * 10,
        )

    def test_inductor_benchmark_many_groups_iters(self):
        benchmarker = InductorBenchmarker()
        benchmarker.L2_cache_size = 4
        elapsed_times = [10.0, 2.0, 5.0, 10.0, 2.0, 5.0]
        get_event_pair_iters = []
        calls = []

        benchmarker.get_event_pairs = self.make_event_pairs(
            elapsed_times, get_event_pair_iters
        )
        with self.patch_cuda_benchmarking():
            timings = self.benchmark_many_cuda(
                benchmarker,
                [
                    self.record_call(calls, "slow"),
                    self.record_call(calls, "fast"),
                    self.record_call(calls, "medium"),
                ],
                benchmark_group_keys=["same-kernel", "same-kernel", "other-kernel"],
            )

        self.assertEqual(timings, [10.0, 2.0, 5.0])
        self.assertEqual(get_event_pair_iters, [3, 3, 3, 10, 10, 5])
        self.assertEqual(
            calls,
            ["slow", "fast", "medium"]
            + ["slow"] * 3
            + ["fast"] * 3
            + ["medium"] * 3
            + ["slow"] * 10
            + ["fast"] * 10
            + ["medium"] * 5,
        )

    def test_inductor_benchmark_many_locks_group_iters(self):
        benchmarker = InductorBenchmarker()
        benchmarker.L2_cache_size = 4
        elapsed_times = [2.0, 2.0, 10.0, 10.0]
        get_event_pair_iters = []

        benchmarker.get_event_pairs = self.make_event_pairs(
            elapsed_times, get_event_pair_iters
        )
        group_state = {}
        with self.patch_cuda_benchmarking():
            slow_timing = self.benchmark_many_cuda(
                benchmarker,
                [lambda: None],
                benchmark_group_keys=["same-kernel"],
                benchmark_group_states=[group_state],
            )
            fast_timing = self.benchmark_many_cuda(
                benchmarker,
                [lambda: None],
                benchmark_group_keys=["same-kernel"],
                benchmark_group_states=[group_state],
            )

        self.assertEqual(slow_timing, [2.0])
        self.assertEqual(fast_timing, [10.0])
        self.assertEqual(group_state, {"benchmark_iters": 10})
        self.assertEqual(get_event_pair_iters, [3, 10, 3, 10])

    def test_inductor_benchmark_many_grouped_matches_scalar_min(self):
        benchmarker = InductorBenchmarker()
        benchmarker.L2_cache_size = 4
        elapsed_times = [1.0, 10.0]
        get_event_pair_iters = []

        benchmarker.get_event_pairs = self.make_event_pairs(
            elapsed_times, get_event_pair_iters
        )
        with self.patch_cuda_benchmarking():
            timing = self.benchmark_many_cuda(
                benchmarker,
                [lambda: None],
                benchmark_group_keys=["same-kernel"],
            )

        self.assertEqual(timing, [1.0])
        self.assertEqual(get_event_pair_iters, [3, 10])

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @decorateIf(
        unittest.expectedFailure,
        lambda params: params["benchmarker_cls"] is Benchmarker,
    )
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmark_gpu_smoke(self, benchmarker_cls, device=GPU_TYPE):
        benchmarker = benchmarker_cls()
        _, _callable = self.make_params(device)
        timing = benchmarker.benchmark_gpu(_callable)
        self.assertGreater(timing, 0)
        self.assertEqual(self.get_counter_value(benchmarker_cls, "benchmark_gpu"), 1)

    @unittest.skipIf(not HAS_CPU and not HAS_GPU, "requires CPU or GPU")
    @unittest.expectedFailure
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmark_safely_infers_device_no_devices(
        self, benchmarker_cls, device="cpu" if HAS_CPU else GPU_TYPE
    ):
        benchmarker = benchmarker_cls()
        (fn, _, _), _ = self.make_params(device)
        benchmarker.benchmark(fn, (), {})

    @unittest.skipIf(not HAS_CPU or not HAS_GPU, "requires CPU and GPU")
    @unittest.expectedFailure
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmark_safely_infers_device_many_devices(self, benchmarker_cls):
        benchmarker = benchmarker_cls()
        (fn, cpu_args, cpu_kwargs), _ = self.make_sum("cpu")
        (_, gpu_args, gpu_kwargs), _ = self.make_sum(GPU_TYPE)
        many_devices_args = cpu_args + gpu_args
        many_devices_kwargs = cpu_kwargs
        many_devices_kwargs.update(gpu_kwargs)
        benchmarker.benchmark(fn, many_devices_args, many_devices_kwargs)

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    def test_benchmark_warmup_and_rep_defaults(self):
        """Test that benchmark_gpu receives default warmup and rep values when not specified."""
        captured_kwargs = {}

        def capture_benchmark_gpu(self, _callable, **kwargs):
            captured_kwargs.update(kwargs)
            return 1.0  # Return a dummy timing

        benchmarker = TritonBenchmarker()
        (fn, fn_args, fn_kwargs), _ = self.make_params(GPU_TYPE)

        with patch.object(TritonBenchmarker, "benchmark_gpu", capture_benchmark_gpu):
            benchmarker.benchmark(fn, fn_args, fn_kwargs)

        self.assertEqual(captured_kwargs["warmup"], inductor_default_autotune_warmup)
        self.assertEqual(captured_kwargs["rep"], inductor_default_autotune_rep)

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    def test_benchmark_warmup_and_rep_custom_values(self):
        """Test that benchmark_gpu receives custom warmup and rep values when specified."""
        captured_kwargs = {}

        def capture_benchmark_gpu(self, _callable, **kwargs):
            captured_kwargs.update(kwargs)
            return 1.0  # Return a dummy timing

        benchmarker = TritonBenchmarker()
        (fn, fn_args, fn_kwargs), _ = self.make_params(GPU_TYPE)

        custom_warmup = 50
        custom_rep = 200

        with patch.object(TritonBenchmarker, "benchmark_gpu", capture_benchmark_gpu):
            benchmarker.benchmark(
                fn, fn_args, fn_kwargs, warmup=custom_warmup, rep=custom_rep
            )

        self.assertEqual(captured_kwargs["warmup"], custom_warmup)
        self.assertEqual(captured_kwargs["rep"], custom_rep)

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmarker_cpu_override_dispatch(self, benchmarker_cls, device="cpu"):
        # Registers a custom handler for 'cpu' and verifies dispatch uses it instead of the default path.
        from torch._inductor.runtime import benchmarking as _bench

        benchmarker = benchmarker_cls()

        # Snapshot registry and restore at the end to avoid cross-test pollution.
        orig = dict(_bench._BENCHMARK_DISPATCH)
        try:
            seen = {"cpu_override": 0}

            def custom_cpu(self, fn, *, warmup, rep, **kw):
                seen["cpu_override"] += 1
                return "cpu-override"

            # Override the built-in 'cpu' registration
            _bench.register_benchmarker("cpu", custom_cpu, override=True)

            # Ensure default CPU/GPU methods are NOT called if registry override works.
            with (
                patch.object(
                    benchmarker_cls,
                    "benchmark_cpu",
                    side_effect=AssertionError(
                        "benchmark_cpu should not be called when a custom 'cpu' handler is registered"
                    ),
                    create=True,
                ),
                patch.object(
                    benchmarker_cls,
                    "benchmark_gpu",
                    side_effect=AssertionError(
                        "benchmark_gpu should not be called for 'cpu' device"
                    ),
                    create=True,
                ),
            ):
                (fn, fn_args, fn_kwargs), _ = self.make_params(device)
                out = benchmarker.benchmark(fn, fn_args, fn_kwargs)
                self.assertEqual(out, "cpu-override")
                self.assertEqual(seen["cpu_override"], 1)
        finally:
            _bench._BENCHMARK_DISPATCH.clear()
            _bench._BENCHMARK_DISPATCH.update(orig)

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmarker_cpu_override_runs_callable(
        self, benchmarker_cls, device="cpu"
    ):
        from torch._inductor.runtime import benchmarking as _bench

        benchmarker = benchmarker_cls()
        orig = dict(_bench._BENCHMARK_DISPATCH)
        try:
            # Override CPU but still route to benchmark_cpu internally
            def custom_cpu(self, f, *, warmup, rep, **kw):
                # Just delegate to the original path; we want to ensure `f()` calls the user's fn.
                return self.benchmark_cpu(f, warmup=warmup, rep=rep, **kw)

            _bench.register_benchmarker("cpu", custom_cpu, override=True)
            # Define a simple op and ensure it actually runs without TypeError
            (fn, fn_args, fn_kwargs), _ = self.make_params(device)
            out = benchmarker.benchmark(fn, fn_args, fn_kwargs, warmup=1, rep=1)
            self.assertGreater(out, 0)
        finally:
            _bench._BENCHMARK_DISPATCH.clear()
            _bench._BENCHMARK_DISPATCH.update(orig)


if __name__ == "__main__":
    run_tests()
