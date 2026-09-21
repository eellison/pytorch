"""Generated in-place foreach preserves caller storage through native replay."""

import json
from pathlib import Path
import struct
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import runtime_support as support

from torch._inductor.runtime._cudagraph._compiler.host_program import Normalize
from mutation_model import ForeachMutation, WIDTH
from torch._inductor.runtime._cudagraph.api import NativeTerminalPolicy
import torch._inductor.runtime._cudagraph.metadata as metadata
import torch
from torch._inductor import config
from torch._inductor.runtime.cudagraph_arg_mapping import InputSource, IntExpr, PointerSource
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._inductor.virtualized import V
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests


class TestInductorMutation(support.MixedTestCase):
    @parametrize("cases", (((13, 1), (2, 5), (35, 32), (13, 3)),))
    def test_foreach_input_writes(self, device, cases):
        policy = NativeTerminalPolicy()
        self.addCleanup(policy.close)
        self.enterContext(config.patch(cudagraph_policy=policy, freezing=False,
            assume_aligned_inputs=False, combo_kernels=True, max_autotune=False,
            max_autotune_pointwise=False))
        collected, traces, programs, captures, preparations = [], [], [], [], []
        collect, prepare = metadata.collect_terminal_metadata, policy.prepare
        trace_wrapper = support.terminal_policy.trace_warmed_wrapper
        prepare_terminal = support.terminal_policy.prepare_terminal
        make_replay = support.terminal_replay._make_replay
        warm_inputs = ()
        report = {"accepted": False,
                   "cases": cases,
                  "compiler_metadata": collected, "captures": captures}

        def observe_metadata(wrapper):
            result = collect(wrapper)
            collected.append({"mutated_inputs": tuple(V.graph.mutated_inputs),
                "mutated_input_idxs": tuple(V.graph.mutated_input_idxs),
                "operations": tuple(type(op).__name__ for op in V.graph.operations),
                "attached": result is not None})
            return result

        def observe_prepare(wrapper, inputs, **options):
            nonlocal warm_inputs
            sys.setprofile(None)
            warm_inputs = tuple(inputs)
            snapshots = tuple(value.clone() if isinstance(value, torch.Tensor) else value for value in inputs)
            self.assertEqual(options["static_inputs"], ())
            result = prepare(wrapper, inputs, **options)
            self.assertEqual(tuple(inputs), snapshots)
            preparations.append(wrapper)
            return result

        def observe_trace(*args):
            trace, views = trace_wrapper(*args)
            traces.append(trace)
            return trace, views

        def observe_program(program, inputs):
            calls = tuple(event for event in program.events if type(event) is support.TerminalCall)
            self.assertTrue(calls)
            self.assertFalse(any(type(event) is Normalize for event in program.events))
            self.assertEqual(program.guards.expressions, ())
            self.assertTrue(any(type(size) is IntExpr for row in program.contract.tensor_inputs for size in row.size))
            for call in calls:
                module = call.provider.launchers[0].__globals__["runner"].__self__
                self.modules[id(module)] = module
            programs.append(program)
            return prepare_terminal(program, inputs)

        def observe_capture(graph, input_count, allocations, outputs, copies, calls, launches, buffers, stream, **kwargs):
            self.assertEqual(copies, ())
            self.assertEqual(len(calls), len(launches))
            numeric = kwargs["numeric"]
            used_inputs = set()
            for call, launch in zip(calls, launches, strict=True):
                self.assertEqual(launch.function, call.module.function)
                for slot, argument in enumerate(call.arguments):
                    source = argument.source
                    root = source.root if type(source) is PointerSource else source
                    if type(root) is not InputSource:
                        continue
                    offset = numeric.values[numeric.add(source.byte_offset)] if type(source) is PointerSource else 0
                    self.assertEqual(struct.unpack("P", launch.argument_bytes[slot])[0],
                                     warm_inputs[root.index].data_ptr() + offset)
                    used_inputs.add(root.index)
            self.assertTrue(set(collected[-1]["mutated_input_idxs"]) <= used_inputs)
            captures.append({"calls": len(calls), "input_slots": sorted(used_inputs)})
            return make_replay(graph, input_count, allocations, outputs, copies, calls, launches, buffers, stream, **kwargs)

        self.enterContext(mock.patch.object(metadata, "collect_terminal_metadata", observe_metadata))
        self.enterContext(mock.patch.object(policy, "prepare", observe_prepare))
        self.enterContext(mock.patch.object(support.terminal_policy, "trace_warmed_wrapper", observe_trace))
        self.enterContext(mock.patch.object(support.terminal_policy, "prepare_terminal", observe_program))
        self.enterContext(mock.patch.object(support.terminal_replay, "_make_replay", observe_capture))
        phase, ordinary_calls = "cold", {"cold": 0, "reference": 0}

        def profile(frame, event, result):
            if event == "call":
                for _, original in policy.artifacts:
                    if frame.f_code is original.__code__ and frame.f_globals is original.__globals__:
                        ordinary_calls[phase] += 1

        def sample(batch, offset):
            roots = tuple(torch.randn(batch * WIDTH + offset, device=device, dtype=torch.float32) for _ in range(2))
            reference_roots = tuple(root.clone() for root in roots)
            inputs = tuple(root[offset:].view(batch, WIDTH) for root in roots)
            references = tuple(root[offset:].view(batch, WIDTH) for root in reference_roots)
            aliases = tuple(root.as_strided((batch, WIDTH), (WIDTH, 1), offset) for root in roots)
            before = tuple(value.clone() for value in inputs)
            prefixes = tuple(root[:offset].clone() for root in roots)
            for value in (*inputs, *references):
                torch._dynamo.mark_dynamic(value, 0, min=2, max=128)
                torch._dynamo.mark_static(value, 1)
            return roots, reference_roots, inputs, references, aliases, before, prefixes

        def check_result(actual, inputs, aliases, before):
            left, right = inputs
            self.assertIs(actual[0], left)
            self.assertIs(actual[1], right)
            self.assertEqual(inputs, tuple((value + 1.0) * 2.0 for value in before))
            self.assertEqual(aliases, inputs)
            for view, source in ((actual[2], left), (actual[3], right)):
                self.assertTrue(torch._C._is_alias_of(view, source))
                self.assertEqual(view.untyped_storage()._cdata, source.untyped_storage()._cdata)
            self.assertEqual(actual[2], left[:, ::2])
            self.assertEqual(actual[3], right[:, 1::2])
            self.assertEqual(actual[4], left + right)

        held, all_samples = [], []
        try:
            graph_count = support.counters["stats"]["unique_graphs"]
            compiled = torch.compile(ForeachMutation(), fullgraph=True, dynamic=True)
            batch, offset = cases[0]
            cold = sample(batch, offset)
            roots, reference_roots, inputs, references, aliases, before, prefixes = cold
            try:
                sys.setprofile(profile)
                actual = compiled(*inputs)
            finally:
                sys.setprofile(None)
            artifact, original = policy.artifacts[0]
            self.assertEqual(len(policy.installations), 1, repr(collected))
            installation, = policy.installations
            self.assertEqual(installation.status, "ready", installation.decline)
            entry = installation.entry
            self.assertIs(type(entry), torch._C._CUDAGraphBoxedReplay)
            self.assertIs(artifact.current_callable, entry)
            self.assertTrue(artifact.mutated_inputs)
            self.assertEqual(len(artifact.mutated_input_idxs), 2)
            self.assertEqual(tuple(artifact.inputs_to_check), ())
            self.assertIn("@triton_heuristics.foreach", artifact.source_code)
            self.assertEqual(ordinary_calls["cold"], 1)
            observer = support.NativeObservation(installation)
            observer.forbidden_files.update(str(path) for path in support.IMPLEMENTATION)
            observer.forbidden_files.add(sys.modules[CachingAutotuner.__module__].__file__)
            for index, (batch, offset) in enumerate(cases):
                current = cold if not index else sample(batch, offset)
                roots, reference_roots, inputs, references, aliases, before, prefixes = current
                self.assertTrue(all(value.data_ptr() not in {old.data_ptr() for row in all_samples for old in row[2]}
                                    for value in inputs))
                all_samples.append(current)
                if index:
                    with observer:
                        actual = compiled(*inputs)
                check_result(actual, inputs, aliases, before)
                phase = "reference"
                try:
                    sys.setprofile(profile)
                    ordinary = support.original_reference(self, compiled, artifact, original, entry, references)
                finally:
                    sys.setprofile(None)
                self.assertEqual(actual, ordinary)
                self.assertEqual(roots, reference_roots)
                self.assertEqual(tuple(root[:offset] for root in roots), prefixes)
                self.assertEqual(ordinary_calls["reference"], index + 1)
                held.append((actual, ordinary, inputs, aliases, before))
            self.assertEqual((len(preparations), len(traces), len(programs), len(captures)), (1, 1, 1, 1))
            self.assertEqual(len(installation.variants), 1)
            self.assertEqual(len(observer.calls), len(cases) - 1)
            self.assertEqual((observer.errors, observer.frames, observer.pending), ([], {}, {}))
            self.assertEqual(support.counters["stats"]["unique_graphs"] - graph_count, 1)
            torch.cuda.synchronize(device)
            policy.close()
            self.assertTrue(entry.closed)
            for actual, ordinary, inputs, aliases, before in held:
                self.assertEqual(actual, ordinary)
                check_result(actual, inputs, aliases, before)
            report.update(accepted=True, compiler_metadata=collected, captures=captures,
                native_hits=len(observer.calls), ordinary_calls=ordinary_calls,
                trace_attempts=len(traces), variants=len(installation.variants),
                capture_preserved_inputs=True, caller_aliases_preserved=True, outputs_survive_close=True)
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            print("INDUCTOR_MUTATION_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


instantiate_device_type_tests(TestInductorMutation, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
