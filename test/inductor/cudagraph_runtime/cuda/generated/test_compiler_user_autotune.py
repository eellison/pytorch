"""Trace the configuration selected by ordinary user Triton autotuning."""

import json
import math
from pathlib import Path
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import runtime_support as support


import torch
import triton
import triton.language as tl
import torch._inductor.runtime._cudagraph.policy as terminal_policy
from torch._inductor.runtime._cudagraph.frontend import TerminalCall
from torch._inductor.runtime._cudagraph.trace_views import TraceViewDeclined
from torch._inductor import config
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch._inductor.runtime.triton_heuristics import CachingAutotuner, GridExpr
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@triton.autotune(configs=[triton.Config({"BLOCK": 128}, num_warps=4),
                        triton.Config({"BLOCK": 256}, num_warps=4)], key=["M", "N"])
@triton.jit
def autotuned_dimensions(x, out, M, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M * N
    value = tl.load(x + offsets, mask, other=0)
    tl.store(out + offsets, value + 17 * M + N, mask)


def generated_user_generated(value):
    intermediate = value * 2
    out = torch.empty_like(intermediate)
    m, n = value.shape
    torch.library.wrap_triton(autotuned_dimensions)[lambda meta: (triton.cdiv(m * n, meta["BLOCK"]),)](
        intermediate, out, m, n)
    return (out * 3,)


class TestTerminalUserAutotune(TestCase):
    def test_selected_block_tracks_two_dimensions(self, device):
        fixture = support.MixedTestCase()
        self.addCleanup(lambda: self.assertTrue(fixture.doCleanups()))
        fixture.setUp()
        policy = support.POLICY
        self.assertIs(type(policy), terminal_policy.NativeTerminalPolicy)
        self.addCleanup(policy.close)
        self.enterContext(config.patch(cudagraph_policy=policy, static_launch_user_defined_triton_kernels=True,
            combo_kernels=False, max_autotune=False, max_autotune_pointwise=False))
        benchmarked, prepared, grids = [], [], []
        benchmark = CachingAutotuner.benchmark_all_configs
        prepare = policy.prepare
        prepare_terminal = terminal_policy.prepare_terminal
        report = {"accepted": False,
                   "entry": "NativeTerminalPolicy"}

        def observe_benchmark(provider, *args, **kwargs):
            if provider.inductor_meta.get("cudagraph_user_kernel") is not True:
                return benchmark(provider, *args, **kwargs)
            launchers = tuple(provider.launchers)
            self.assertEqual(len(launchers), 2)
            self.assertEqual({launcher.config.kwargs["BLOCK"] for launcher in launchers}, {128, 256})
            modules = tuple(launcher.__globals__["runner"].__self__ for launcher in launchers)
            for module in modules:
                fixture.modules[id(module)] = module
            timings = benchmark(provider, *args, **kwargs)
            self.assertEqual(set(timings), set(launchers))
            self.assertTrue(all(math.isfinite(value) for value in timings.values()))
            benchmarked.append((provider, launchers, modules, dict(timings)))
            return timings

        def observe_prepare(wrapper, inputs, **options):
            self.assertEqual(options["static_inputs"], ())
            self.assertNotIn("_cudagraph_call_records", wrapper.__dict__)
            self.assertNotIn("_cudagraph_frontend_attachment", wrapper.__dict__)
            self.assertIn("_cudagraph_terminal_attachment", wrapper.__dict__)
            with mock.patch.object(GridExpr, "from_meta", side_effect=AssertionError(
                    "The selected launcher's actual grid must be traced")):
                return prepare(wrapper, inputs, **options)

        def check_grid(program, call, box, shape, block):
            numeric = _NumericProgram(program, box)
            actual = tuple(numeric.values[numeric.add(axis)] for axis in call.grid)
            self.assertEqual(actual, (triton.cdiv(math.prod(shape), block), 1, 1))
            grids.append({"shape": shape, "grid": actual})

        def observe_program(program, inputs):
            self.assertEqual(len(program.integer_inputs), 2)
            calls = tuple(event for event in program.events if type(event) is TerminalCall)
            self.assertEqual([call.provider.inductor_meta.get("cudagraph_user_kernel") is True for call in calls],
                             [False, True, False])
            self.assertEqual(len(benchmarked), 1)
            provider, launchers, modules, timings = benchmarked[0]
            self.assertIs(calls[1].provider, provider)
            self.assertEqual(provider.inductor_meta["grid_type"], "PrecomputedGrid")
            selected, = provider.launchers
            self.assertIs(selected, min(timings, key=timings.get))
            block = selected.config.kwargs["BLOCK"]
            module = selected.__globals__["runner"].__self__
            result, = provider.compile_results
            self.assertIs(result.kernel, module)
            self.assertIs(result.config, selected.config)
            self.assertEqual([item for item in modules if item.module is not None], [module])
            for other in modules:
                if other is not module:
                    self.assertIsNone(other.function)
            formals = module.cudagraph_formal_args
            self.assertTrue(all(type(arg.attributes) is tuple for arg in formals))
            self.assertEqual(tuple(arg.formal for arg in formals), ("x", "out", "M", "N", "BLOCK"))
            block_arg, = (arg for arg in formals if arg.formal == "BLOCK")
            self.assertIsNone(block_arg.abi_index)
            self.assertEqual(block_arg.constant, block)
            with mock.patch.dict(selected.config.kwargs, {"BLOCK": 128 if block == 256 else 256}):
                with self.assertRaisesRegex(TraceViewDeclined, "Warmed provider state changed"):
                    program.check()
            program.check()
            tensor_input, = program.contract.tensor_inputs
            check_grid(program, calls[1], inputs, tuple(inputs[tensor_input.index].shape), block)
            for call in calls:
                selected_module = call.provider.launchers[0].__globals__["runner"].__self__
                fixture.modules[id(selected_module)] = selected_module
            entry = prepare_terminal(program, inputs)
            self.assertIs(type(entry), torch._C._CUDAGraphBoxedReplay)
            prepared.append((program, entry, calls[1], block))
            report.update(selected_block=block, traced_grid=repr(calls[1].grid),
                          benchmarks=[{"block": launcher.config.kwargs["BLOCK"], "milliseconds": timings[launcher]}
                                      for launcher in launchers], losing_module_unloaded=True,
                          grid_type="PrecomputedGrid", changed_selected_config_rejected=True)
            return entry

        self.enterContext(mock.patch.object(CachingAutotuner, "benchmark_all_configs", observe_benchmark))
        self.enterContext(mock.patch.object(policy, "prepare", observe_prepare))
        self.enterContext(mock.patch.object(terminal_policy, "prepare_terminal", observe_program))

        def sample(shape):
            value = torch.randint(-17, 18, shape, device=device, dtype=torch.int32)
            for dim in (0, 1):
                torch._dynamo.mark_dynamic(value, dim, min=2, max=128)
            m, n = shape
            return (value,), ((value * 2 + 17 * m + n) * 3,)

        try:
            torch.manual_seed(20260915)
            graphs = support.counters["stats"]["unique_graphs"]
            compiled = torch.compile(generated_user_generated, fullgraph=True, dynamic=True)
            cold, expected = sample((13, 19))
            actual = compiled(*cold)
            self.assertEqual(actual, expected)
            self.assertEqual(len(policy.artifacts), 1)
            self.assertEqual(len(policy.installations), 1)
            artifact, original = policy.artifacts[0]
            installation, = policy.installations
            self.assertEqual(installation.status, "ready", installation.decline)
            entry = installation.entry
            self.assertIs(type(entry), torch._C._CUDAGraphBoxedReplay)
            self.assertIs(artifact.current_callable, entry)
            self.assertEqual(len(prepared), 1)
            program, prepared_entry, user_call, block = prepared[0]
            self.assertIs(prepared_entry, entry)
            calls = tuple(event for event in program.events if type(event) is TerminalCall)
            owners = tuple((call.provider, call.provider.launchers[0], call.provider._cached_launcher) for call in calls)
            selected_module = user_call.provider.launchers[0].__globals__["runner"].__self__
            handles = selected_module.module, selected_module.function
            ordinary = support.original_reference(self, compiled, artifact, original, entry, cold)
            self.assertEqual(actual, ordinary)
            held = [(actual, expected, ordinary)]
            held_inputs = [cold]
            observer = support.NativeObservation(installation)
            observer.forbidden_files.update(str(path) for path in support.IMPLEMENTATION)
            observer.forbidden_files.discard(type(observer).__exit__.__code__.co_filename)
            observer.forbidden_files.add(sys.modules[CachingAutotuner.__module__].__file__)
            cases = ((2, 19), (7, 19), (7, 31), (35, 31), (35, 5), (11, 37), (96, 127))
            for shape in cases:
                arguments, expected = sample(shape)
                self.assertNotIn(arguments[0].data_ptr(), [values[0].data_ptr() for values in held_inputs])
                held_inputs.append(arguments)
                before = len(observer.calls)
                with observer:
                    actual = compiled(*arguments)
                self.assertEqual(len(observer.calls), before + 1)
                self.assertEqual(actual, expected)
                box = [None] * len(program.input_names)
                for index, value in observer.calls[-1]["integers"]:
                    box[index] = value
                tensor_input, = program.contract.tensor_inputs
                box[tensor_input.index] = arguments[0]
                self.assertTrue(all(value is not None for value in box))
                check_grid(program, user_call, box, shape, block)
                ordinary = support.original_reference(self, compiled, artifact, original, entry, arguments)
                self.assertEqual(actual, ordinary)
                for provider, launcher, cached in owners:
                    self.assertIs(provider.launchers[0], launcher)
                    self.assertIs(provider._cached_launcher, cached)
                self.assertEqual((selected_module.module, selected_module.function), handles)
                self.assertEqual(len(benchmarked), 1)
                self.assertEqual(len(prepared), 1)
                self.assertIs(artifact.current_callable, entry)
                self.assertFalse(entry.closed)
                self.assertFalse(entry.failed)
                self.assertEqual(support.counters["stats"]["unique_graphs"] - graphs, 1)
                self.assertEqual(len(policy.artifacts), 1)
                self.assertEqual(len(policy.installations), 1)
                held.append((actual, expected, ordinary))
            self.assertEqual(observer.errors, [])
            self.assertEqual(observer.frames, {})
            self.assertEqual(observer.pending, {})
            torch.cuda.synchronize(device)
            policy.close()
            self.assertTrue(entry.closed)
            for actual, expected, ordinary in held:
                self.assertEqual(actual, expected)
                self.assertEqual(actual, ordinary)
            report.update(accepted=True, native_hits=len(observer.calls), evaluated_grids=grids,
                          original_compiled_references=len(held), outputs_survive_close=True,
                          ordinary_or_preparation_frames=0, old_metadata_or_host_records=False)
        finally:
            print("OPAQUE_TRITON_AUTOTUNE_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


instantiate_device_type_tests(TestTerminalUserAutotune, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
