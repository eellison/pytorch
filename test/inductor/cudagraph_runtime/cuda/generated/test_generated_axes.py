"""Keep actual selected kernels across reduction extents and static GEMM reuse."""

import json
import os
from pathlib import Path
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import runtime_support as support

import torch
from torch._inductor import config
from torch._inductor.codegen.multi_kernel import MultiKernelCall, SizeHintMultiKernelCall
from torch._inductor.runtime._cudagraph.api import NativeTerminalPolicy
from torch._inductor.runtime._cudagraph.frontend import TerminalCall
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._inductor.select_algorithm import AlgorithmSelectorCache, TritonTemplateCaller
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests


def matrix_product(left, right):
    return torch.mm(left, right)


def row_softmax(value):
    return value.softmax(dim=-1)


class TestGeneratedAxes(support.MixedTestCase):
    def check_selected(self, kernel):
        self.assertIs(type(kernel), CachingAutotuner)
        self.assertEqual(len(kernel.launchers), 1)
        selected = kernel.launchers[0]
        cached = kernel._cached_launcher
        self.assertIs(cached.config, selected.config)
        self.assertIs(type(cached.__globals__["runner"]), torch._C._FastCudaLauncher)
        module = selected.__globals__["runner"].__self__
        compiled, = (result for result in kernel.compile_results
                     if result.kernel is module and result.config is selected.config)
        self.assertIs(compiled.kernel, module)
        self.assertIs(type(module.function), int)
        self.assertGreater(module.function, 0)
        self.modules[id(module)] = module
        return kernel, selected, cached, module

    @parametrize("kind", ("gemm_static", "multi_reduction"))
    def test_selected_kernel_reuse(self, device, kind):
        gemm = kind == "gemm_static"
        policy = NativeTerminalPolicy()
        self.addCleanup(policy.close)
        self.enterContext(config.patch(
            cudagraph_policy=policy, max_autotune=gemm,
            max_autotune_gemm_backends="TRITON", benchmark_epilogue_fusion=False,
            deterministic=False, multi_kernel_hints=[], **{
                "test_configs.max_mm_configs": 2, "triton.native_matmul": False,
                "triton.enable_persistent_tma_matmul": False,
                "triton.multi_kernel": 0 if gemm else 1,
            }))
        self.enterContext(mock.patch.dict(os.environ, {"TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE": "1"}))
        report = {"kind": kind, "accepted": False, "choices": [], "multi_kernel_benchmarks": []}
        select = AlgorithmSelectorCache.__call__
        benchmark = MultiKernelCall.benchmark_sub_kernels
        prepare_terminal = support.terminal_policy.prepare_terminal
        programs, selections = [], []

        def observe_choices(cache, name, choices, *args, **kwargs):
            if name == "mm":
                report["choices"].append([type(choice).__name__ for choice in choices])
            return select(cache, name, choices, *args, **kwargs)

        def observe_benchmark(carrier, *args, **kwargs):
            timings = benchmark(carrier, *args, **kwargs)
            report["multi_kernel_benchmarks"].append({"carrier": id(carrier), "timings_ms": list(timings)})
            return timings

        def observe_program(program, inputs):
            calls = tuple(event for event in program.events if type(event) is TerminalCall)
            self.assertEqual(len(calls), 1)
            carriers = tuple(view.carrier for view in program.views if type(view.carrier) is MultiKernelCall)
            self.assertEqual(bool(carriers), not gemm)
            for call in calls:
                selected = self.check_selected(call.provider)
                if gemm:
                    self.assertEqual(call.provider.heuristic_type.name, "TEMPLATE")
                else:
                    self.assertTrue(any(call.provider is carrier.kernels[carrier.picked_kernel]
                                        for carrier in carriers))
                selections.append(selected)
            programs.append(program)
            return prepare_terminal(program, inputs)

        self.enterContext(mock.patch.object(AlgorithmSelectorCache, "__call__", observe_choices))
        self.enterContext(mock.patch.object(MultiKernelCall, "benchmark_sub_kernels", observe_benchmark))
        self.enterContext(mock.patch.object(support.terminal_policy, "prepare_terminal", observe_program))
        sizes = (64,) * 5 if gemm else (1024, 128, 256, 768, 512)
        function = matrix_product if gemm else row_softmax

        def sample(size):
            if gemm:
                left = torch.randn(size, 128, device=device, dtype=torch.float16) * 0.25
                right = torch.randn(128, 128, device=device, dtype=torch.float16) * 0.25
                torch._dynamo.mark_static(left)
                torch._dynamo.mark_static(right)
                return left, right
            value = torch.randn(32, size, device=device)
            torch._dynamo.mark_static(value, 0)
            torch._dynamo.mark_dynamic(value, 1, min=128, max=1024)
            return (value,)

        try:
            compiled = torch.compile(function, fullgraph=True, dynamic=not gemm)
            graphs = support.counters["stats"]["unique_graphs"]
            algorithm_benchmarks = support.counters["inductor"]["select_algorithm_autotune"]
            cold = sample(sizes[0])
            actual = compiled(*cold)
            self.assertEqual(actual, function(*cold))
            self.assertEqual(len(policy.artifacts), 1)
            artifact, original = policy.artifacts[0]
            multi = {id(value): value for value in original.__globals__.values() if type(value) is MultiKernelCall}
            self.assertFalse(any(type(value) is SizeHintMultiKernelCall for value in original.__globals__.values()))
            if gemm:
                self.assertFalse(multi)
                self.assertTrue(report["choices"])
                for choices in report["choices"]:
                    self.assertEqual(len(choices), 2)
                    self.assertTrue(all(choice == TritonTemplateCaller.__name__ for choice in choices))
                self.assertGreater(support.counters["inductor"]["select_algorithm_autotune"] - algorithm_benchmarks, 0)
            else:
                self.assertTrue(multi)
                self.assertEqual(len(report["multi_kernel_benchmarks"]), len(multi))
                for row in report["multi_kernel_benchmarks"]:
                    carrier = multi[row["carrier"]]
                    self.assertGreaterEqual(len(carrier.kernels), 2)
                    self.assertEqual(carrier.picked_kernel, row["timings_ms"].index(min(row["timings_ms"])))
            self.assertEqual(len(policy.installations), 1)
            installation, = policy.installations
            self.assertEqual(installation.status, "ready", installation.decline)
            self.assertIsNone(installation.decline)
            entry = installation.entry
            self.assertIs(type(entry), torch._C._CUDAGraphBoxedReplay)
            self.assertIs(artifact.current_callable, entry)
            self.assertEqual(len(programs), 1)
            self.assertEqual(len(selections), 1)
            bench_count = len(report["multi_kernel_benchmarks"])
            observer = support.NativeObservation(installation)
            observer.forbidden_files.update(str(path) for path in support.IMPLEMENTATION)
            observer.forbidden_files.add(sys.modules[MultiKernelCall.__module__].__file__)
            ordinary = support.original_reference(self, compiled, artifact, original, entry, cold)
            self.assertEqual(actual, ordinary)
            held, held_inputs = [(actual, function(*cold), ordinary)], [cold]
            report["shapes"] = [list(cold[0].shape)]
            for size in sizes[1:]:
                arguments = sample(size)
                self.assertNotIn(arguments[0].data_ptr(), [values[0].data_ptr() for values in held_inputs])
                held_inputs.append(arguments)
                expected = function(*arguments)
                before = len(observer.calls)
                with observer:
                    actual = compiled(*arguments)
                self.assertEqual(actual, expected)
                ordinary = support.original_reference(self, compiled, artifact, original, entry, arguments)
                self.assertEqual(actual, ordinary)
                self.assertEqual(len(observer.calls), before + 1)
                self.assertEqual(observer.errors, [])
                self.assertEqual(observer.pending, {})
                self.assertEqual(observer.frames, {})
                self.assertIs(installation.entry, entry)
                self.assertIs(artifact.current_callable, entry)
                self.assertFalse(entry.closed)
                self.assertFalse(entry.failed)
                self.assertEqual((len(policy.artifacts), len(policy.installations), len(programs)), (1, 1, 1))
                self.assertEqual(support.counters["stats"]["unique_graphs"] - graphs, 1)
                self.assertEqual(len(report["multi_kernel_benchmarks"]), bench_count)
                for selected in selections:
                    current = self.check_selected(selected[0])
                    for previous, now in zip(selected, current, strict=True):
                        self.assertIs(previous, now)
                held.append((actual, expected, ordinary))
                report["shapes"].append(list(arguments[0].shape))
            self.assertEqual(len(observer.calls), 4)
            torch.cuda.synchronize(device)
            policy.close()
            self.assertTrue(entry.closed)
            for actual, expected, ordinary in held:
                self.assertEqual(actual, expected)
                self.assertEqual(actual, ordinary)
            report.update(accepted=True, native_hits=4, original_compiled_references=5,
                          winner_stable_across_shapes=True, outputs_survive_close=True)
        finally:
            print("GENERATED_AXES_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


instantiate_device_type_tests(TestGeneratedAxes, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
