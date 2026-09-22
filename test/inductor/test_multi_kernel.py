# Owner(s): ["module: inductor"]

import contextlib
import os
import re
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch._dynamo.testing import reset_rng_state
from torch._inductor import config, test_operators
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch._inductor.runtime.benchmarking import (
    _BENCHMARK_DISPATCH,
    _default_cuda_bench,
    InductorBenchmarker,
    set_gpu_benchmark_lock_context,
    TritonBenchmarker,
)
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.nn import functional as F
from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfRocm,
    skipIfXpu,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    IS_BIG_GPU,
    requires_triton,
)


class TransformerSnippet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(64)

    def forward(self, x1, x2):
        x1 = F.dropout(x1, 0.1)
        x2 = F.dropout(self.ln1(x2), 0.1)

        return self.ln2(x1 + x2)

    def example_inputs(self):
        return (torch.randn(2, 64).to(GPU_TYPE), torch.randn(2, 64).to(GPU_TYPE))


def _contains_multi_kernel_code(wrapper_code: str):
    return (
        re.search(r"multi_kernel_[^ ]* = async_compile.multi_kernel[(]", wrapper_code)
        is not None
    )


def _contains_size_hint_multi_kernel_code(wrapper_code: str):
    return (
        re.search(
            r"multi_kernel_[^ ]* = async_compile.size_hint_multi_kernel[(]",
            wrapper_code,
        )
        is not None
    )


def make_cpp_wrapper_test(orig_test, **extra_args):
    """
    Wrap an existing test into a new test with cpp-wrapper enabled.

    Make this as a free function rather than staticmethod in MultiKernelTest.
    Otherwise we get 'TypeError: 'staticmethod' object is not callable'
    error in py3.8. (py3.10 works)
    """

    @config.patch("cpp_wrapper", True)
    @config.patch("triton.autotune_at_compile_time", True)
    def fn(self):
        # The same kernel may have been compiled by previous tests with
        # cpp_wrapper disabled. Clear the cache so we go ahead to re-compile
        # the kernel with cpp_wrapper enabled.
        from torch._inductor import codecache

        codecache.PyCodeCache.cache_clear()
        return orig_test(self, **extra_args)

    return fn


@config.patch(
    {
        "triton.multi_kernel": int(os.environ.get("TORCHINDUCTOR_MULTI_KERNEL", "1")),
        "benchmark_kernel": True,
        "multi_kernel_hints": [64, 256, 4096],
    }
)
@instantiate_parametrized_tests
class MultiKernelTest(TestCase):
    @staticmethod
    def _benchmark_lock_call(events):
        def kernel(index):
            return SimpleNamespace(
                clone_args=lambda *args, **kwargs: (args, kwargs),
                run=lambda *args, **kwargs: events.append(f"run_{index}"),
                device_props=SimpleNamespace(type="cuda"),
            )

        multi_kernel_call = object.__new__(MultiKernelCall)
        multi_kernel_call._kernels = [kernel(0), kernel(1)]
        multi_kernel_call.arg_index = {
            0: [slice(0, 1)],
            1: [slice(0, 1)],
        }
        return multi_kernel_call

    def _benchmark_call(self, benchmarker, device_type="cuda"):
        events = []
        multi = self._benchmark_lock_call(events)
        for kernel in multi.kernels:
            kernel.device_props.type = device_type
        for patcher in (
            unittest.mock.patch.dict(_BENCHMARK_DISPATCH, {}, clear=True),
            unittest.mock.patch(
                "torch._inductor.codegen.multi_kernel.benchmarker", benchmarker
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        return multi, events

    @parametrize("failure", [None, "benchmark", "kernel"])
    @config.patch({"max_autotune": True, "autotune_cudagraph_benchmarking": True})
    def test_benchmark_sub_kernels_gpu_lock(self, failure):
        multi, events = self._benchmark_call(InductorBenchmarker())

        @contextlib.contextmanager
        def lock():
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

        previous = set_gpu_benchmark_lock_context(lock)
        self.addCleanup(set_gpu_benchmark_lock_context, previous)
        if failure == "kernel":
            multi.kernels[0].run = unittest.mock.Mock(
                side_effect=RuntimeError("failed")
            )
        timings = iter([2.0, 1.0])

        def benchmark(fn, **kwargs):
            events.append("benchmark")
            if failure == "benchmark":
                raise RuntimeError("failed")
            fn()
            return next(timings)

        with unittest.mock.patch(
            "torch._inductor.codegen.multi_kernel.benchmarker.benchmark",
            side_effect=benchmark,
        ):
            if failure:
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    multi.benchmark_sub_kernels("arg")
            else:
                self.assertEqual(multi.benchmark_sub_kernels("arg"), [2.0, 1.0])
        self.assertEqual(
            events,
            ["enter", "benchmark", "exit"]
            if failure
            else ["enter", "benchmark", "run_0", "benchmark", "run_1", "exit"],
        )

    @parametrize("device_type", ["cuda", "hip"])
    @parametrize("registered_default", [False, True])
    @config.patch({"max_autotune": True, "autotune_cudagraph_benchmarking": True})
    def test_benchmark_sub_kernels_autotunes_before_graph_benchmark(
        self, device_type, registered_default
    ):
        bench = InductorBenchmarker()
        multi, calls = self._benchmark_call(bench, device_type)
        inputs = (torch.tensor([1.0]), torch.tensor([2.0]))
        multi.arg_index = {0: [slice(0, 1)], 1: [slice(1, 2)]}

        def run(x):
            calls.append((x.item(), bench._in_cudagraph_benchmark))
            x.add_(1)

        for kernel in multi.kernels:
            kernel.clone_args = lambda x: ((x.clone(),), {})
            kernel.run = run

        def capture(fn, **kwargs):
            fn()
            fn()
            return 1.0

        with (
            unittest.mock.patch.dict(
                _BENCHMARK_DISPATCH,
                {"cuda": _default_cuda_bench} if registered_default else {},
            ),
            unittest.mock.patch.object(
                TritonBenchmarker, "benchmark_gpu_with_cuda_graph", side_effect=capture
            ),
        ):
            self.assertEqual(multi.benchmark_sub_kernels(*inputs), [1.0, 1.0])
        self.assertEqual(
            calls, [(x, guard) for x in (1.0, 2.0) for guard in (False, True, True)]
        )
        self.assertEqual(inputs, (torch.tensor([1.0]), torch.tensor([2.0])))
        self.assertFalse(bench._in_cudagraph_benchmark)

    @parametrize("invalid_configuration", [False, True])
    @config.patch({"max_autotune": True, "autotune_cudagraph_benchmarking": True})
    def test_benchmark_sub_kernels_preserves_triton_error_handling(
        self, invalid_configuration
    ):
        bench = TritonBenchmarker()
        multi, events = self._benchmark_call(bench)
        message = (
            "CUDA error: invalid configuration argument"
            if invalid_configuration
            else "unrelated kernel failure"
        )
        multi.kernels[0].run = unittest.mock.Mock(side_effect=RuntimeError(message))

        def benchmark(fn, **kwargs):
            fn()
            return 1.0

        with unittest.mock.patch.object(
            bench, "triton_do_bench", side_effect=benchmark
        ):
            if invalid_configuration:
                self.assertEqual(
                    multi.benchmark_sub_kernels("arg"), [float("inf"), 1.0]
                )
            else:
                with self.assertRaisesRegex(RuntimeError, message):
                    multi.benchmark_sub_kernels("arg")
        multi.kernels[0].run.assert_called_once()
        self.assertEqual(events, ["run_1"] if invalid_configuration else [])

    @parametrize("route", ["override", "cuda", "hip"])
    @config.patch({"max_autotune": True, "autotune_cudagraph_benchmarking": True})
    def test_benchmark_sub_kernels_preserves_dispatch(self, route):
        bench = InductorBenchmarker()
        multi, events = self._benchmark_call(bench, "hip" if route == "hip" else "cuda")
        dispatch = unittest.mock.Mock(return_value=float("inf"))
        if route == "override":
            patcher = unittest.mock.patch.object(bench, "benchmark_gpu", dispatch)
        else:
            patcher = unittest.mock.patch.dict(_BENCHMARK_DISPATCH, {route: dispatch})
        with patcher:
            self.assertEqual(multi.benchmark_sub_kernels("arg"), [float("inf")] * 2)
        self.assertEqual(dispatch.call_count, 2)
        self.assertEqual(events, [])

    @parametrize(
        "reason", ["max_autotune", "graph_disabled", "inside_graph", "cpu", "xpu"]
    )
    @config.patch({"max_autotune": True, "autotune_cudagraph_benchmarking": True})
    def test_benchmark_sub_kernels_skips_inapplicable_prewarm(self, reason):
        bench = InductorBenchmarker()
        multi, events = self._benchmark_call(
            bench, reason if reason in ("cpu", "xpu") else "cuda"
        )
        bench._in_cudagraph_benchmark = reason == "inside_graph"

        def benchmark(fn, **kwargs):
            events.append("benchmark")
            fn()
            return 1.0

        with (
            config.patch(
                {
                    "max_autotune": reason != "max_autotune",
                    "autotune_cudagraph_benchmarking": reason != "graph_disabled",
                }
            ),
            unittest.mock.patch.object(bench, "benchmark", side_effect=benchmark),
        ):
            self.assertEqual(multi.benchmark_sub_kernels("arg"), [1.0, 1.0])
        self.assertEqual(events, ["benchmark", "run_0", "benchmark", "run_1"])

    @config.patch(
        {
            "max_autotune": True,
            "autotune_cudagraph_benchmarking": True,
            "deterministic": True,
        }
    )
    def test_benchmark_sub_kernels_respects_deterministic_ban(self):
        multi, events = self._benchmark_call(InductorBenchmarker())
        with self.assertRaisesRegex(RuntimeError, "deterministic mode of Inductor"):
            multi.benchmark_sub_kernels("arg")
        self.assertEqual(events, [])

    @parametrize("has_stream", [False, True])
    @parametrize("device_type", ["cuda", "hip"])
    def test_benchmark_sub_kernels_uses_current_stream(self, has_stream, device_type):
        bench = InductorBenchmarker()
        multi, events = self._benchmark_call(bench, device_type)
        stream = [10]
        interface = SimpleNamespace(
            current_device=lambda: 0, get_raw_stream=lambda device: stream[0]
        )
        for kernel in multi.kernels:
            kernel.run = lambda *args, **kwargs: events.append(kwargs)

        def benchmark(fn, **kwargs):
            for stream[0] in (20, 30):
                fn()
            return 1.0

        kwargs = {"stream": 10} if has_stream else {}
        with (
            unittest.mock.patch.object(bench, "benchmark", side_effect=benchmark),
            unittest.mock.patch(
                "torch._dynamo.device_interface.get_interface_for_device",
                return_value=interface,
            ) as lookup,
        ):
            multi.benchmark_sub_kernels("arg", **kwargs)
        self.assertEqual(
            events,
            [{"stream": s} for s in (20, 30, 20, 30)] if has_stream else [{}] * 4,
        )
        self.assertEqual(kwargs, {"stream": 10} if has_stream else {})
        if has_stream:
            lookup.assert_has_calls([unittest.mock.call("cuda")] * 4)
            self.assertEqual(lookup.call_count, 4)
        else:
            lookup.assert_not_called()

    @skipIfXpu(msg="uses CUDA graph capture")
    def test_benchmark_sub_kernels_captures_kernel(self):
        output = torch.zeros(1, device=GPU_TYPE)

        def run(output, *, stream):
            with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
                output.add_(1)

        bench = InductorBenchmarker()
        multi, _ = self._benchmark_call(bench)
        multi._kernels = [multi.kernels[0]]
        multi.kernels[0].run = run

        def benchmark(fn, **kwargs):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                fn()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph, stream=stream, capture_error_mode="thread_local"
            ):
                fn()
            torch.cuda.synchronize()
            output.zero_()
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(output, torch.ones_like(output))
            return 1.0

        with unittest.mock.patch.object(bench, "benchmark", side_effect=benchmark):
            multi.benchmark_sub_kernels(
                output, stream=torch.cuda.current_stream().cuda_stream
            )

    def test_softmax(self, expect_multi_kernel=True):
        x = torch.rand(2, 1024).to(GPU_TYPE)
        ref = torch.softmax(x, -1)
        compiled_fn = torch.compile(torch.softmax)
        act, wrapper_code = run_and_get_code(compiled_fn, x, -1)

        # wrapper_code will contains 2 entries if cpp_wrapper=True.
        # One for the first pass and one for the second pass.
        # We mainly care about the wrapper for the final pass here.
        wrapper_code = wrapper_code[-1]
        self.assertEqual(ref, act)
        if expect_multi_kernel:
            self.assertTrue(_contains_multi_kernel_code(wrapper_code))
        else:
            self.assertFalse(_contains_multi_kernel_code(wrapper_code))

    @requires_triton()
    # TODO: bobrenjc93 to fix multi-kernel for ROCM
    @skipIfRocm
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    @skipIfXpu(msg="driver issue, torch-xpu-ops: 2295")
    def test_triton_gemm(self):
        def fn(x, y):
            return x @ y

        compiled_fn = torch.compile(
            fn,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )
        x = torch.randn(4096, 4096, device=GPU_TYPE)
        y = torch.randn(4096, 4096, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x, 0)
        act, wrapper_code = run_and_get_code(compiled_fn, x, y)
        ref = fn(x, y)

        # wrapper_code will contains 2 entries if cpp_wrapper=True.
        # One for the first pass and one for the second pass.
        # We mainly care about the wrapper for the final pass here.
        wrapper_code = wrapper_code[-1]
        self.assertEqual(ref, act)
        self.assertTrue(_contains_size_hint_multi_kernel_code(wrapper_code))

    @skipIfXpu(msg="driver issue, torch-xpu-ops: 2295")
    @requires_triton()
    # TODO: bobrenjc93 to fix multi-kernel for ROCM
    @skipIfRocm
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_triton_relu_fused_gemm(self):
        def fn(x, y):
            return (x @ y).relu()

        compiled_fn = torch.compile(
            fn,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )
        x = torch.randn(4096, 4096, device=GPU_TYPE)
        y = torch.randn(4096, 4096, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x, 0)
        act, wrapper_code = run_and_get_code(compiled_fn, x, y)
        ref = fn(x, y)

        # wrapper_code will contains 2 entries if cpp_wrapper=True.
        # One for the first pass and one for the second pass.
        # We mainly care about the wrapper for the final pass here.
        wrapper_code = wrapper_code[-1]
        self.assertEqual(ref, act)
        self.assertTrue(_contains_size_hint_multi_kernel_code(wrapper_code))

    @parametrize("force_kernel", (0, 1))
    @unittest.mock.patch.dict(
        os.environ, {"TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE": "1"}
    )
    def test_softmax_force_non_persistent_reduction(self, force_kernel):
        """
        Force a specific sub-kernel being picked by mocking the benchmark result.
        """
        x = torch.rand(2, 1024).to(GPU_TYPE)
        mock_latency = [0.2, 0.2]
        mock_latency[force_kernel] = 0.1  # this make sure force_kernel will be picked

        def f(x):
            return torch.softmax(x, -1) + force_kernel

        orig_run = MultiKernelCall.run
        picked_kernel = None

        def mock_run(self, *args, **kwargs):
            out = orig_run(self, *args, **kwargs)
            nonlocal picked_kernel
            picked_kernel = self.picked_kernel
            return out

        with (
            unittest.mock.patch.object(MultiKernelCall, "run", mock_run),
            unittest.mock.patch.object(
                MultiKernelCall,
                "benchmark_sub_kernels",
                lambda *args, **kwargs: mock_latency,
            ),
        ):
            torch.compile(f)(x)
        self.assertEqual(picked_kernel, force_kernel)

    @config.patch("warn_mix_layout", True)
    def test_softmax_warn_mixed_layout(self):
        self.test_softmax()

    test_softmax_cpp_wrapper = make_cpp_wrapper_test(
        test_softmax, expect_multi_kernel=True
    )

    def test_layernorm(self):
        ln = nn.LayerNorm(1024).to(GPU_TYPE)
        x = torch.rand(2, 1024).to(GPU_TYPE)
        ref = ln(x)
        act = torch.compile(ln)(x)
        self.assertEqual(ref, act, atol=1e-4, rtol=1e-4)

    def test_inplace_update(self):
        """
        Inductor generate inplace kernel for mul.
        """

        def f(x, y):
            return x.sum(dim=-1, keepdims=True) * (y @ y)

        x = torch.rand(1024, 1024).to(GPU_TYPE)
        y = torch.rand(1024, 1024).to(GPU_TYPE)
        ref = f(x, y)
        act = torch.compile(f)(x, y)
        self.assertEqual(ref, act)

    def test_transformer_snippet(self):
        model = TransformerSnippet().to(GPU_TYPE)
        x = model.example_inputs()

        def f(*x):
            y = model(*x)
            return y

        reset_rng_state()
        ref = f(*x)

        opt_f = torch.compile(f)
        reset_rng_state()
        act = opt_f(*x)

        # don't compare tensor if using inductor random number generator.
        # inductor random number implementation is different to eager.
        # We should fallback to eager if we want to test accuracy.
        if config.fallback_random:
            self.assertEqual(ref, act, atol=1e-4, rtol=1e-4)

    def test_transformer_snippet_with_fallback_random(self):
        """
        Same as test_transformer_snippet but fallback the random number
        generator to eager so we can check accuracy.
        """
        with config.patch("fallback_random", True):
            self.test_transformer_snippet()

    def test_batchnorm_training(self):
        """
        For training, batchnorm will tracking running mean/variance during forward pass.
        The kernel generated by inductor currently will pass in those tensors twice as arguments:
        once for input and once for output. They are ruled out as in-out argument because
        they are considered as graph inputs.

        Multi-kernel previously assumes that we never pass the same argument multi times
        for a kernel. No matter if we change inductor behavior to assure that, it's better
        to make multi-kernel being able to handle those cases.
        """
        bn = nn.BatchNorm2d(3).to(GPU_TYPE)

        @torch.compile
        def f(x):
            bn(x).sum().backward()

        _, (wrapper_code, _) = run_and_get_code(
            f, torch.randn(2, 3, 8, 8, device=GPU_TYPE)
        )
        self.assertTrue(_contains_multi_kernel_code(wrapper_code))

    def test_pass_same_arg_multi_times(self):
        """
        A super simple example that simulate how BatchNorm update the running
        stats.

        Inductor currently pass the same tensor multiple times for the generated
        kernel: once for input and once for output.

        Here is a paster for the generated kernel (without multi-kernel enabled):
        https://gist.github.com/shunting314/f0b446b4b9a28f4940e31dcd3e809cf9
        """

        def f(x, y):
            x = x.sum(dim=1, keepdim=False)
            y.copy_(y * 0.9 + x * 0.1)

        x = torch.randn(8, 16, device=GPU_TYPE)
        y = torch.randn(8, device=GPU_TYPE)
        y_ref = y.clone()

        ref = f(x, y_ref)  # noqa: F841
        act = torch.compile(f)(x, y)  # noqa: F841
        self.assertEqual(y_ref, y)

    def test_reduction_scratch_buffer(self, force_multi_kernel=1):
        """
        The explicitly realized buffer in the test function will be passed in
        as a scratch buffer for the non-persistent reduction kernel but
        can be skipped for the persistent reduction kernel.

        This causes different argument lists for non-persistent reduction kernel and
        persistent reduction kernel.

        Check documentation around torch._inductor.config.triton.multi_kernel about
        how to interpret the force_multi_kernel argument.
        """

        def f(x):
            x = x.sum(dim=-1, keepdim=True) + x
            x = test_operators.realize(x)
            x = x.sum(dim=-1, keepdim=True) + x
            return x

        x = torch.rand(16, 16, device=GPU_TYPE)
        ref = f(x)
        with config.patch("triton.multi_kernel", force_multi_kernel):
            act = torch.compile(f)(x)
        self.assertEqual(ref, act)

    def test_split_scan(self, force_multi_kernel=1):
        def f(x):
            x = x.view(-1)
            return torch.cumsum(x, 0)

        x = make_tensor(10, 3, 352, 352, low=0, dtype=torch.float32, device=GPU_TYPE)
        expect = f(x)
        with config.patch("triton.multi_kernel", force_multi_kernel):
            actual = torch.compile(f)(x)
        self.assertEqual(expect, actual)

    def test_sort_disables_multi_kernel(self, force_multi_kernel=1):
        """
        Sort currently requires a persistent kernel, so multi-kernel is not
        possible. Make sure this falls back gracefully.
        """

        def f(x):
            return x.sort(-1).values

        x = torch.rand(32, 32, device=GPU_TYPE)
        expect = f(x)
        with config.patch("triton.multi_kernel", force_multi_kernel):
            actual = torch.compile(f)(x)
        self.assertEqual(expect, actual)

    # Use benchmarking to pick the faster kernel
    test_reduction_scratch_buffer_cpp_wrapper = make_cpp_wrapper_test(
        test_reduction_scratch_buffer, force_multi_kernel=1
    )
    # force pick persistent reduction. This can be a good test since this persistent
    # reduction uses less call arguments than the corresponding non-persistent
    # reduction.
    test_reduction_scratch_buffer_cpp_wrapper_persistent_reduction = (
        make_cpp_wrapper_test(test_reduction_scratch_buffer, force_multi_kernel=2)
    )
    # force pick non-persistent reduction
    test_reduction_scratch_buffer_cpp_wrapper_non_persistent_reduction = (
        make_cpp_wrapper_test(test_reduction_scratch_buffer, force_multi_kernel=3)
    )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
