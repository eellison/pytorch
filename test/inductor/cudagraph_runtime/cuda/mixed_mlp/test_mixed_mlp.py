"""One application through the public shared-tracing entry and native replay."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from unittest import mock

import mixed_support as support

from torch._inductor.runtime._cudagraph._compiler.host_program import Normalize
from model import BLOCKS, EXPANSION, eager_reference, ResidualMLP, WIDTH
from rmsnorm_fixture import ENTRY, rmsnorm
import torch
from torch._inductor import config
from torch._inductor.runtime._cudagraph.api import IntExpr
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram, _PhysicalCall
from torch._inductor.runtime.triton_heuristics import CachingAutotuner, GridExpr
from torch._inductor.select_algorithm import AlgorithmSelectorCache, TritonTemplateCaller
from torch.testing._internal.common_cuda import tf32_off
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import recover_orig_fp32_precision, run_tests

terminal_policy, terminal_replay = support.terminal_policy, support.terminal_replay
TerminalCall, CuTeCall, CuteInvokeEvent = support.TerminalCall, support.CuTeCall, support.CuteInvokeEvent
BATCHES = (64, 2, 7, 35, 96)


class TestRepresentativeMLP(support.MixedTestCase):
    @recover_orig_fp32_precision
    @tf32_off()
    def test_residual_mlp(self, device):
        layout = support.TensorPolicy((None, WIDTH), (0, 1))
        owner = support.ObservedOrdinaryEntry(ENTRY, rmsnorm,
            policy=support.SignaturePolicy(32, 64, 16, "stream"),
            tensor_policies={"source": layout, "destination": layout})
        self.addCleanup(owner.close)
        registration = support.register_cute_entry(owner, owned_executor=True)
        self.addCleanup(registration.close)
        policy = support.POLICY
        self.addCleanup(policy.close)
        self.enterContext(config.patch(cudagraph_policy=policy, freezing=False,
            static_launch_user_defined_triton_kernels=True, combo_kernels=False,
            max_autotune=True, max_autotune_gemm_backends="TRITON",
            benchmark_epilogue_fusion=False, deterministic=False, multi_kernel_hints=[], **{
                "test_configs.max_mm_configs": 2,
                "triton.native_matmul": False,
                "triton.enable_persistent_tma_matmul": False,
                "triton.multi_kernel": 0,
            }))
        blockers = []
        preparations, traces, programs, physical_calls, launchers, captures = [], [], [], [], [], []
        prepare, trace_wrapper = policy.prepare, terminal_policy.trace_warmed_wrapper
        prepare_terminal, make_replay = terminal_policy.prepare_terminal, terminal_replay._make_replay
        choices = []
        choose = AlgorithmSelectorCache.__call__
        static_indices = ()
        report = {"accepted": False,
                  "blocks": BLOCKS, "width": WIDTH, "expansion": EXPANSION,
                  "batches": BATCHES, "dtype": "float32", "allow_tf32": False,
                  "max_mm_configs": 2}

        def observe_choices(cache, name, candidates, *args, **kwargs):
            if name == "mm":
                self.assertTrue(candidates)
                self.assertLessEqual(len(candidates), 2)
                self.assertTrue(all(isinstance(item, TritonTemplateCaller) for item in candidates))
                choices.append([{"name": item.name, "description": item.description} for item in candidates])
            return choose(cache, name, candidates, *args, **kwargs)

        def observe_prepare(wrapper, inputs, **options):
            nonlocal static_indices
            sys.setprofile(None)
            for name in ("_cudagraph_call_records", "_cudagraph_frontend_attachment",
                         "_cute_invocation_attachment", "_cute_mixed_attachment", "_cudagraph_mixed_ir"):
                self.assertNotIn(name, wrapper.__dict__)
            self.assertIn("_cudagraph_terminal_attachment", wrapper.__dict__)
            static_indices = tuple(options["static_inputs"])
            self.assertEqual({id(inputs[index]) for index in static_indices}, {id(p) for p in model.parameters()})
            preparations.append(wrapper)
            with mock.patch.object(GridExpr, "from_meta", side_effect=AssertionError("Use the traced grid")):
                return prepare(wrapper, inputs, **options)

        def observe_trace(*args):
            trace, views = trace_wrapper(*args)
            events = tuple(event for event in trace.events if type(event) is CuteInvokeEvent)
            self.assertEqual(len(events), BLOCKS)
            for event in events:
                self.assertIs(event.entry, registration)
                self.assertIs(event.call.entry, ENTRY)
            traces.append(trace)
            return trace, views

        def observe_program(program, inputs):
            ordered = tuple(event for event in program.events if type(event) in (TerminalCall, CuTeCall))
            calls = tuple(event for event in ordered if type(event) is TerminalCall)
            users = tuple(event for event in calls if event.provider.inductor_meta.get("cudagraph_user_kernel") is True)
            templates = tuple(event for event in calls if event.provider.heuristic_type.name == "TEMPLATE")
            cute = tuple(event for event in ordered if type(event) is CuTeCall)
            self.assertEqual((len(users), len(cute)), (BLOCKS, BLOCKS))
            self.assertTrue(templates)
            self.assertEqual(program.guards.expressions, ())
            normalized = {event.input_index for event in program.events if type(event) is Normalize}
            self.assertTrue(set(static_indices) <= normalized)
            dynamic, = (item for item in program.contract.tensor_inputs if type(item.size[0]) is IntExpr)
            self.assertTrue(any(type(dim) is IntExpr for item in program.allocations for dim in item.size))
            numeric = _NumericProgram(program, inputs)
            events = tuple(event for event in traces[-1].events if type(event) is CuteInvokeEvent)
            for call, event, user in zip(cute, events, users, strict=True):
                self.assertLess(ordered.index(call), ordered.index(user))
                self.assertIs(call.receipt.trace.event, event)
                self.assertIs(call.receipt.trace.terminal, traces[-1])
                self.assertEqual(call.bound.grid[0], dynamic.size[0])
                grid = tuple(numeric.values[numeric.add(axis)] for axis in call.bound.grid)
                self.assertEqual(grid, (inputs[dynamic.index].shape[0], 1, 1))
                pointers = {(field.parameter, field.byte_offset): field.source
                            for field in call.bound.fields if field.kind == "pointer"}
                self.assertEqual(set(pointers), set(call.bound.module.site.fields.pointer_descriptors))
                self.assertEqual(set(pointers.values()), set(call.pointers))
                for pointer in call.bound.module.site.fields.pointers:
                    source = dict(zip(("source", "destination"), call.pointers, strict=True))[pointer.source.formal_name]
                    self.assertEqual(pointers[pointer.parameter, pointer.byte_offset], source)
                self.assertTrue(all(formal.data_alignment == 16 for formal in call.bound.module.artifact.formals
                                    if formal.kind == "Tensor"))
            self.assertEqual(len({id(call.receipt.resources) for call in cute}), BLOCKS)
            self.assertTrue(all(call.bound.module.artifact._payload is cute[0].bound.module.artifact._payload
                                for call in cute))
            for call in calls:
                provider = call.provider
                module = provider.launchers[0].__globals__["runner"].__self__
                self.modules[id(module)] = module
                launchers.append((provider, provider.launchers[0], provider._cached_launcher, module))
                facts, = (item for item in program.effects if item.provider is provider)
                arguments = tuple(sorted((row for row in module.cudagraph_formal_args if row.abi_index is not None),
                                         key=lambda row: row.abi_index))
                self.assertEqual(facts.arguments, arguments)
                self.assertTrue(all(type(row.attributes) is tuple for row in module.cudagraph_formal_args))
                self.assertEqual({row.formal for row in facts.pointers},
                                 {row.formal for row in arguments if row.triton_type.startswith("*")})
            physical_calls.extend(cute)
            programs.append(program)
            report.update(generated_calls=len(calls) - len(users), template_calls=len(templates),
                          user_triton_calls=len(users), cute_calls=len(cute), total_kernel_calls=len(ordered),
                          symbolic_allocations=len(program.allocations), static_inputs=len(static_indices),
                          selected_gemms=[{
                              "name": call.provider.launchers[0].__globals__["runner"].__self__.name,
                              "compiled_hash": call.provider.launchers[0].__globals__["runner"].__self__.hash,
                              "config": call.provider.launchers[0].config.all_kwargs(),
                          } for call in templates])
            return prepare_terminal(program, inputs)

        def observe_capture(graph, input_count, allocations, outputs, copies, calls, launches, buffers, stream, **kwargs):
            self.assertEqual(len(calls), len(launches))
            physical = tuple((call, launch) for call, launch in zip(calls, launches, strict=True)
                             if type(call) is _PhysicalCall)
            self.assertEqual(len(physical), BLOCKS)
            for (bound, launch), call in zip(physical, physical_calls, strict=True):
                self.assertIs(bound, call.bound)
                self.assertEqual(launch.function, bound.module.function)
            captures.append(len(calls))
            return make_replay(graph, input_count, allocations, outputs, copies, calls, launches, buffers, stream, **kwargs)

        self.enterContext(mock.patch.object(AlgorithmSelectorCache, "__call__", observe_choices))
        self.enterContext(mock.patch.object(policy, "prepare", observe_prepare))
        self.enterContext(mock.patch.object(terminal_policy, "trace_warmed_wrapper", observe_trace))
        self.enterContext(mock.patch.object(terminal_policy, "prepare_terminal", observe_program))
        self.enterContext(mock.patch.object(terminal_replay, "_make_replay", observe_capture))
        phase, ordinary_calls, compiles = "cold", {"cold": 0, "reference": 0}, []
        compile_code = type(support.cute.compile).__call__.__code__

        def profile(frame, event, result):
            if event != "call":
                return
            if frame.f_code is compile_code and frame.f_locals.get("self") is support.cute.compile:
                arguments = frame.f_locals.get("args", ())
                compiles.append(arguments[0] if arguments else None)
            for _, original in policy.artifacts:
                if frame.f_code is original.__code__ and frame.f_globals is original.__globals__:
                    ordinary_calls[phase] += 1

        def sample(batch):
            value = torch.randn(batch, WIDTH, device=device, dtype=torch.float32)
            torch._dynamo.mark_dynamic(value, 0, min=2, max=128)
            torch._dynamo.mark_static(value, 1)
            return (value,), eager_reference(model, value)

        model = ResidualMLP(registration.key, device).eval()
        for parameter in model.parameters():
            torch._dynamo.mark_static(parameter)
        try:
            graphs = support.counters["stats"]["unique_graphs"]
            benchmarks = support.counters["inductor"]["select_algorithm_autotune"]
            compiled = torch.compile(model, fullgraph=True, dynamic=True)
            cold, expected = sample(BATCHES[0])
            try:
                sys.setprofile(profile)
                actual = compiled(*cold)
            finally:
                sys.setprofile(None)
            self.assertEqual(actual, expected, atol=2e-5, rtol=2e-5)
            installation, = policy.installations
            artifact, original = policy.artifacts[0]
            self.assertEqual(installation.status, "ready", installation.decline)
            published = installation.entry
            self.assertIs(type(published), torch._C._CUDAGraphBoxedReplay)
            self.assertEqual((len(programs), len(captures), len(installation.variants)), (1, 1, 1))
            self.assertEqual(ordinary_calls["cold"], 1)
            self.assertEqual(compiles, [ENTRY.target])
            self.assertTrue(choices)
            self.assertGreater(support.counters["inductor"]["select_algorithm_autotune"], benchmarks)
            artifacts = Path(self.enterContext(TemporaryDirectory()))
            (artifacts / "generated_wrapper.py.txt").write_text(artifact.source_code)
            (artifacts / "observed_cute_source.mlir").write_text(owner._capture.original_text)
            selected, selection_count = owner.selected, len(choices)
            observer = support.NativeObservation(installation)
            observer.forbidden_files.update(str(path) for path in support.IMPLEMENTATION)
            observer.forbidden_files.update((support.ordinary_module.__file__, compile_code.co_filename,
                support.JitExecutor.__call__.__code__.co_filename,
                support.JitExecutor.run_compiled_program.__code__.co_filename,
                support.ExecutionArgs.generate_execution_args.__code__.co_filename,
                sys.modules[CachingAutotuner.__module__].__file__))
            for name in ("precompile", "_precompile_config", "benchmark_all_configs"):
                blockers.append(self.enterContext(mock.patch.object(CachingAutotuner, name,
                    side_effect=AssertionError(f"Selected MLP kernels must not call {name} again"))))
            held, inputs = [], []
            for index, batch in enumerate(BATCHES):
                arguments, expected = (cold, expected) if not index else sample(batch)
                self.assertNotIn(arguments[0].data_ptr(), [item[0].data_ptr() for item in inputs])
                inputs.append(arguments)
                if index:
                    with observer:
                        actual = compiled(*arguments)
                phase = "reference"
                try:
                    sys.setprofile(profile)
                    ordinary = support.original_reference(self, compiled, artifact, original, published, arguments)
                finally:
                    sys.setprofile(None)
                self.assertEqual(actual, ordinary)
                self.assertEqual(actual, expected, atol=2e-5, rtol=2e-5)
                self.assertEqual(ordinary_calls["reference"], index + 1)
                self.assertEqual(preparations, [original])
                self.assertEqual(len(traces), 1)
                self.assertEqual(compiles, [ENTRY.target])
                self.assertIs(owner.selected, selected)
                self.assertEqual(owner._capture.calls, 1)
                self.assertEqual(len(choices), selection_count)
                for provider, launcher, cached, module in launchers:
                    self.assertIs(provider.launchers[0], launcher)
                    self.assertIs(provider._cached_launcher, cached)
                    self.assertIs(launcher.__globals__["runner"].__self__, module)
                held.append((actual, expected, ordinary))
            self.assertEqual(len(observer.calls), len(BATCHES) - 1)
            self.assertEqual((observer.errors, observer.frames, observer.pending), ([], {}, {}))
            self.assertEqual(support.counters["stats"]["unique_graphs"] - graphs, 1)
            for blocker in blockers:
                blocker.assert_not_called()
            torch.cuda.synchronize(device)
            policy.close()
            self.assertTrue(published.closed)
            self.assertTrue(all(call.receipt.closed and call.bound.module.closed for call in physical_calls))
            self.assertEqual(owner._native_borrows, set())
            registration.close()
            for actual, expected, ordinary in held:
                self.assertEqual(actual, ordinary)
                self.assertEqual(actual, expected, atol=2e-5, rtol=2e-5)
            report.update(accepted=True, native_hits=len(observer.calls), choices=choices,
                          trace_attempts=len(traces), ordinary_cute_compilations=len(compiles),
                          original_compiled_references=len(held), outputs_survive_close=True)
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            print("REPRESENTATIVE_MLP_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


instantiate_device_type_tests(TestRepresentativeMLP, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
