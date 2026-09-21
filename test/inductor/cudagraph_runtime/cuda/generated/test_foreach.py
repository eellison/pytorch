"""Replay functional foreach with independently changing traced lengths."""

import json
from pathlib import Path
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import runtime_support as support

import torch
from torch._inductor.runtime._cudagraph.frontend import TerminalCall
from torch._inductor.runtime._cudagraph.api import NativeTerminalPolicy
from torch._inductor import config, metrics
from torch._inductor.runtime.triton_heuristics import CachingAutotuner, GridExpr
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests


class TestTerminalForeach(support.MixedTestCase):
    @parametrize("count", (2, 3))
    def test_distinct_dynamic_sizes(self, device, count):
        policy = NativeTerminalPolicy()
        self.addCleanup(policy.close)
        self.enterContext(config.patch(
            cudagraph_policy=policy, combo_kernel_foreach_dynamic_shapes=True,
            combo_kernels=False, combo_kernels_autotune=0, combo_kernel_per_subkernel_blocks=False,
            combo_kernel_compile_time_autotune=False, max_autotune=False, max_autotune_pointwise=False,
        ))
        prepared = []
        report = {"independent_lengths": count, "accepted": False}
        traces = []
        trace_wrapper = support.terminal_policy.trace_warmed_wrapper
        prepare_terminal = support.terminal_policy.prepare_terminal

        def observe_trace(wrapper, contract, inputs):
            self.assertNotIn("_cudagraph_call_records", wrapper.__dict__)
            self.assertNotIn("_cudagraph_frontend_attachment", wrapper.__dict__)
            trace, views = trace_wrapper(wrapper, contract, inputs)
            self.assertIs(trace.contract, contract)
            self.assertFalse(trace.shape_env.guards)
            self.assertFalse(trace.shape_env.deferred_runtime_asserts)
            traces.append(trace)
            return trace, views

        def observe_prepare(program, inputs):
            calls = tuple(event for event in program.events if type(event) is TerminalCall)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(program.integer_inputs), count)
            self.assertIs(program.contract, traces[-1].contract)
            for call in calls:
                module = call.provider.launchers[0].__globals__["runner"].__self__
                self.modules[id(module)] = module
            entry = prepare_terminal(program, inputs)
            self.assertIs(type(entry), torch._C._CUDAGraphBoxedReplay)
            prepared.append((program, entry))
            report["traced_grids"] = [repr(call.grid) for call in calls]
            return entry

        prepare = policy.prepare

        def observe_policy_prepare(wrapper, inputs, **options):
            with mock.patch.object(GridExpr, "from_meta", side_effect=AssertionError(
                    "Terminal foreach must use the actual traced grid")):
                return prepare(wrapper, inputs, **options)

        self.enterContext(mock.patch.object(policy, "prepare", observe_policy_prepare))
        self.enterContext(mock.patch.object(support.terminal_policy, "trace_warmed_wrapper", observe_trace))
        self.enterContext(mock.patch.object(support.terminal_policy, "prepare_terminal", observe_prepare))

        def functional(*values):
            return torch._foreach_add(list(values[:count]), list(values[count:]))

        def sample(lengths):
            values = tuple(torch.randn(length, device=device, dtype=torch.float32)
                           for length in (*lengths, *lengths))
            for value in values:
                torch._dynamo.mark_dynamic(value, 0)
            expected = tuple(left + right for left, right in zip(values[:count], values[count:], strict=True))
            return values, expected

        try:
            torch.manual_seed(20260915)
            graphs = support.counters["stats"]["unique_graphs"]
            kernels_before = metrics.generated_kernel_count
            compiled = torch.compile(functional, fullgraph=True, dynamic=True)
            cold_lengths = (1031, 2053, 4097)[:count]
            cold, expected = sample(cold_lengths)
            actual = compiled(*cold)
            self.assertEqual(tuple(actual), expected)
            self.assertEqual(len(policy.artifacts), 1)
            self.assertEqual(len(policy.installations), 1)
            artifact, original = policy.artifacts[0]
            installation, = policy.installations
            self.assertEqual(installation.status, "ready", installation.decline)
            entry = installation.entry
            self.assertIs(type(entry), torch._C._CUDAGraphBoxedReplay)
            self.assertIs(artifact.current_callable, entry)
            self.assertEqual(len(prepared), 1)
            self.assertIs(prepared[0][1], entry)
            self.assertFalse(artifact.fx_kwargs["is_backward"])
            kernels = {id(value): value for value in original.__globals__.values()
                       if type(value) is CachingAutotuner and "combo_grid_meta" in value.inductor_meta}
            self.assertEqual(len(kernels), 1)
            self.assertEqual(metrics.generated_kernel_count - kernels_before, 1)
            self.assertEqual(artifact.source_code.count("@triton_heuristics.foreach("), 1)
            kernel, = kernels.values()
            meta = kernel.inductor_meta["combo_grid_meta"]
            self.assertEqual(kernel.inductor_meta["grid_type"], "SequentialComboKernelGrid")
            self.assertEqual(meta["num_kernels"], count)
            self.assertIsNone(meta["min_blocks"])
            for index in range(count):
                self.assertIsNone(meta[f"xnumel_{index}"])
            terminal, = (event for event in prepared[0][0].events if type(event) is TerminalCall)
            self.assertIs(terminal.provider, kernel)
            launcher, = kernel.launchers
            cached = kernel._cached_launcher
            ordinary = support.original_reference(self, compiled, artifact, original, entry, cold)
            self.assertEqual(tuple(actual), tuple(ordinary))
            held = [(tuple(actual), expected, tuple(ordinary))]
            held_inputs = [cold]
            observer = support.NativeObservation(installation)
            observer.forbidden_files.update(str(path) for path in support.IMPLEMENTATION)
            lengths = list(cold_lengths)
            cases = []
            for index, value in enumerate((67, 4099, 127)[:count]):
                lengths[index] = value
                cases.append(tuple(lengths))
            cases.append((257, 8195, 8193)[:count])
            for lengths in cases:
                arguments, expected = sample(lengths)
                previous_pointers = {value.data_ptr() for values in held_inputs for value in values}
                self.assertTrue(previous_pointers.isdisjoint(value.data_ptr() for value in arguments))
                held_inputs.append(arguments)
                before = len(observer.calls)
                with observer:
                    actual = compiled(*arguments)
                self.assertEqual(len(observer.calls), before + 1)
                self.assertEqual(tuple(actual), expected)
                ordinary = support.original_reference(self, compiled, artifact, original, entry, arguments)
                self.assertEqual(tuple(actual), tuple(ordinary))
                self.assertIs(kernel.launchers[0], launcher)
                self.assertIs(kernel._cached_launcher, cached)
                self.assertIs(artifact.current_callable, entry)
                self.assertFalse(entry.closed)
                self.assertFalse(entry.failed)
                self.assertEqual(support.counters["stats"]["unique_graphs"] - graphs, 1)
                self.assertEqual(len(policy.artifacts), 1)
                self.assertEqual(len(policy.installations), 1)
                self.assertEqual(len(prepared), 1)
                held.append((tuple(actual), expected, tuple(ordinary)))
            self.assertEqual(observer.errors, [])
            self.assertEqual(observer.frames, {})
            self.assertEqual(observer.pending, {})
            torch.cuda.synchronize(device)
            policy.close()
            self.assertTrue(entry.closed)
            for actual, expected, ordinary in held:
                self.assertEqual(actual, expected)
                self.assertEqual(actual, ordinary)
            report.update(accepted=True, native_hits=len(observer.calls),
                          lengths=[cold_lengths, *cases], original_compiled_references=len(held),
                          generated_combo_kernels=1, outputs_survive_close=True,
                          ordinary_or_preparation_frames=0, host_record_join_or_grid_reconstruction=False)
        finally:
            print("TERMINAL_FOREACH_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


instantiate_device_type_tests(TestTerminalForeach, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
