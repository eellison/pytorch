# Owner(s): ["module: inductor"]
"""Replay actual generated rank-three and symbolic-stride allocations."""

import json
import sys
import threading
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import runtime_support as support
from model import AllocationLayouts, eager_reference

import torch
from torch._inductor import config
from torch._inductor.output_code import CompiledFxGraph
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    AllocateEvent,
)
from torch._inductor.runtime.cudagraph_arg_mapping import IntExpr, OwnedBuffer
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch._inductor.runtime.triton_heuristics import CachingAutotuner, GridExpr
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


terminal_policy, terminal_replay = support.terminal_policy, support.terminal_replay


class LayoutObservation(support.NativeObservation):
    def profile(self, frame, event, result):
        try:
            code = frame.f_code
            if event == "call" and (
                code is self.original_code or code.co_filename in self.forbidden_files
            ):
                name = code.co_filename, code.co_name
                self.frames[name] = self.frames.get(name, 0) + 1
            if (
                code is not CompiledFxGraph.__call__.__code__
                or frame.f_locals.get("self") is not self.artifact
            ):
                return
            thread = threading.get_ident()
            if event == "call":
                if thread in self.pending:
                    raise AssertionError("Nested original artifact invocation")
                self.pending[thread] = tuple(frame.f_locals["inputs"])
            elif event == "return":
                inputs = self.pending.pop(thread)
                if result is not None:
                    self.calls.append((inputs, tuple(result)))
        except BaseException as error:
            self.errors.append(f"{type(error).__name__}: {error}")


class TestAllocationLayouts(TestCase):
    def test_generated_dynamic_allocations(self, device):
        fixture = support.MixedTestCase()
        self.addCleanup(lambda: self.assertTrue(fixture.doCleanups()))
        fixture.setUp()
        policy = support.POLICY
        self.addCleanup(policy.close)
        self.enterContext(
            config.patch(
                cudagraph_policy=policy,
                freezing=False,
                allow_buffer_reuse=True,
                static_launch_user_defined_triton_kernels=True,
                combo_kernels=False,
                max_autotune=False,
                max_autotune_pointwise=False,
            )
        )
        preparations, traces, programs, captures, descriptors = [], [], [], [], []
        prepare, trace_wrapper = policy.prepare, terminal_policy.trace_warmed_wrapper
        prepare_terminal, make_replay = (
            terminal_policy.prepare_terminal,
            terminal_replay._make_replay,
        )
        report = {"accepted": False, "batch_domain": [2, 127]}

        def observe_prepare(wrapper, inputs, **kwargs):
            sys.setprofile(None)
            self.assertNotIn("_cudagraph_call_records", wrapper.__dict__)
            self.assertNotIn("_cudagraph_frontend_attachment", wrapper.__dict__)
            self.assertIn("_cudagraph_terminal_attachment", wrapper.__dict__)
            preparations.append(wrapper)
            with mock.patch.object(
                GridExpr,
                "from_meta",
                side_effect=AssertionError("Use actual traced grids"),
            ):
                return prepare(wrapper, inputs, **kwargs)

        def observe_trace(*args):
            trace, views = trace_wrapper(*args)
            allocations = tuple(
                event for event in trace.events if type(event) is AllocateEvent
            )
            self.assertTrue(
                any(
                    len(event.size) == 3
                    and any(type(dim) is torch.SymInt for dim in event.size)
                    for event in allocations
                )
            )
            self.assertTrue(
                any(
                    len(event.size) == 3
                    and any(type(dim) is torch.SymInt for dim in event.stride)
                    for event in allocations
                )
            )
            report["traced_allocations"] = [
                (str(event.size), str(event.stride)) for event in allocations
            ]
            traces.append(trace)
            return trace, views

        def observe_program(program, inputs):
            self.assertEqual(program.guards.expressions, ())
            events = tuple(
                event for event in traces[0].events if type(event) is AllocateEvent
            )
            self.assertEqual(len(program.allocations), len(events))
            self.assertTrue(
                any(
                    len(layout.size) == 3
                    and any(type(dim) is IntExpr for dim in layout.size)
                    for layout in program.allocations
                )
            )
            self.assertTrue(
                any(
                    len(layout.stride) == 3
                    and any(type(dim) is IntExpr for dim in layout.stride)
                    for layout in program.allocations
                )
            )
            users = tuple(
                event
                for event in program.events
                if type(event) is support.TerminalCall
                and event.provider.inductor_meta.get("cudagraph_user_kernel") is True
            )
            self.assertEqual(len(users), 2)
            self.assertTrue(
                any(
                    type(event) is support.TerminalCall and event not in users
                    for event in program.events
                )
            )
            programs.append(program)
            report.update(
                allocations=len(program.allocations),
                user_calls=len(users),
                kernel_calls=sum(
                    type(event) is support.TerminalCall for event in program.events
                ),
            )
            return prepare_terminal(program, inputs)

        def observe_capture(
            graph,
            input_count,
            allocations,
            outputs,
            copies,
            calls,
            launches,
            buffers,
            stream,
            **kwargs,
        ):
            self.assertEqual(len(calls), len(launches))
            numeric = kwargs["numeric"]
            layouts = []
            for layout in allocations:
                size, stride = tuple(
                    tuple(
                        numeric.values[numeric.add(value)]
                        if type(value) is IntExpr
                        else value
                        for value in values
                    )
                    for values in (layout.size, layout.stride)
                )
                tensor = buffers[layout.source]
                self.assertEqual((tuple(tensor.shape), tensor.stride()), (size, stride))
                layouts.append((size, stride, tensor.is_contiguous()))
            self.assertTrue(
                any(len(size) == 3 and contiguous for size, _, contiguous in layouts)
            )
            self.assertTrue(
                any(
                    len(size) == 3 and not contiguous for size, _, contiguous in layouts
                )
            )
            captures.append(tuple(layouts))
            native_make = torch.cuda.CUDAGraph._make_boxed_replay

            def observe_native(actual_graph, *args, **native_kwargs):
                self.assertIs(actual_graph, graph)
                actual_layouts = args[5]
                self.assertEqual(len(actual_layouts), len(allocations))
                for descriptor, layout in zip(actual_layouts, allocations, strict=True):
                    self.assertIs(descriptor[0], layout.dtype)
                    expected = tuple(
                        tuple(
                            ("value", numeric.add(value))
                            if type(value) is IntExpr
                            else value
                            for value in values
                        )
                        for values in (layout.size, layout.stride)
                    )
                    self.assertEqual(descriptor[1:], expected)
                self.assertTrue(
                    any(
                        len(row[1]) == 3
                        and any(type(value) is tuple for value in row[2])
                        and not captured[2]
                        for row, captured in zip(actual_layouts, layouts)
                    )
                )
                descriptors.append(actual_layouts)
                return native_make(actual_graph, *args, **native_kwargs)

            with mock.patch.object(
                torch.cuda.CUDAGraph, "_make_boxed_replay", observe_native
            ):
                return make_replay(
                    graph,
                    input_count,
                    allocations,
                    outputs,
                    copies,
                    calls,
                    launches,
                    buffers,
                    stream,
                    **kwargs,
                )

        self.enterContext(mock.patch.object(policy, "prepare", observe_prepare))
        self.enterContext(
            mock.patch.object(terminal_policy, "trace_warmed_wrapper", observe_trace)
        )
        self.enterContext(
            mock.patch.object(terminal_policy, "prepare_terminal", observe_program)
        )
        self.enterContext(
            mock.patch.object(terminal_replay, "_make_replay", observe_capture)
        )
        phase, ordinary_calls = "cold", {"cold": 0, "reference": 0}

        def profile(frame, event, result):
            if event == "call":
                for _, original in policy.artifacts:
                    if (
                        frame.f_code is original.__code__
                        and frame.f_globals is original.__globals__
                    ):
                        ordinary_calls[phase] += 1

        def sample(batch):
            value = torch.randn(3, batch, 128, device=device)
            torch._dynamo.mark_dynamic(value, 1, min=2, max=127)
            torch._dynamo.mark_static(value, (0, 2))
            return (value,), eager_reference(value)

        try:
            compiled = torch.compile(AllocationLayouts(), fullgraph=True, dynamic=True)
            cold, expected = sample(13)
            try:
                sys.setprofile(profile)
                actual = compiled(*cold)
            finally:
                sys.setprofile(None)
            self.assertEqual(actual, expected)
            (installation,) = policy.installations
            artifact, original = policy.artifacts[0]
            report["decline"] = installation.decline
            self.assertEqual(installation.status, "ready", installation.decline)
            self.assertEqual(ordinary_calls["cold"], 1)
            self.assertEqual(preparations, [original])
            for provider in original.__globals__.values():
                if isinstance(provider, CachingAutotuner) and provider.launchers:
                    module = provider.launchers[0].__globals__["runner"].__self__
                    fixture.modules[id(module)] = module
            published = installation.entry
            self.assertIs(type(published), torch._C._CUDAGraphBoxedReplay)
            self.assertEqual(
                (
                    len(programs),
                    len(captures),
                    len(descriptors),
                    len(installation.variants),
                ),
                (1, 1, 1, 1),
            )
            observer = LayoutObservation(installation)
            observer.forbidden_files.update(
                str(path) for path in support.IMPLEMENTATION
            )
            held, inputs = [], []
            for index, batch in enumerate((13, 2, 35, 7, 96)):
                arguments, expected = (cold, expected) if not index else sample(batch)
                self.assertNotIn(
                    arguments[0].data_ptr(), [item[0].data_ptr() for item in inputs]
                )
                inputs.append(arguments)
                if index:
                    with observer:
                        actual = compiled(*arguments)
                    native_inputs, native_outputs = observer.calls[-1]
                    self.assertEqual(len(native_outputs), len(programs[0].outputs))
                    numeric = _NumericProgram(programs[0], native_inputs)
                    owned = [
                        (output, value)
                        for output, value in zip(
                            programs[0].outputs, native_outputs, strict=True
                        )
                        if type(output) is OwnedBuffer
                    ]
                    self.assertTrue(
                        any(
                            len(output.size) == 3 and value.is_contiguous()
                            for output, value in owned
                        )
                    )
                    self.assertTrue(
                        any(
                            len(output.size) == 3 and not value.is_contiguous()
                            for output, value in owned
                        )
                    )
                    for output, value in owned:
                        size, stride = tuple(
                            tuple(
                                numeric.values[numeric.add(field)]
                                if type(field) is IntExpr
                                else field
                                for field in fields
                            )
                            for fields in (output.size, output.stride)
                        )
                        self.assertEqual(
                            (tuple(value.shape), value.stride()), (size, stride)
                        )
                self.assertEqual(actual, expected)
                self.assertEqual(
                    (tuple(actual[1].shape), actual[1].stride()),
                    ((3, batch, 128), (batch * 128, 128, 1)),
                )
                self.assertEqual(
                    (tuple(actual[2].shape), actual[2].stride()),
                    ((batch, 3, 128), (128, batch * 128, 1)),
                )
                self.assertNotIn(
                    actual[1].data_ptr(),
                    [previous[0][1].data_ptr() for previous in held],
                )
                self.assertNotIn(
                    actual[2].data_ptr(),
                    [previous[0][2].data_ptr() for previous in held],
                )
                phase = "reference"
                try:
                    sys.setprofile(profile)
                    ordinary = support.original_reference(
                        self, compiled, artifact, original, published, arguments
                    )
                finally:
                    sys.setprofile(None)
                self.assertEqual(actual, ordinary)
                self.assertEqual(ordinary_calls["reference"], index + 1)
                self.assertEqual(preparations, [original])
                held.append((actual, expected, ordinary))
            self.assertEqual(len(observer.calls), 4)
            self.assertEqual(
                (observer.errors, observer.frames, observer.pending), ([], {}, {})
            )
            torch.cuda.synchronize(device)
            policy.close()
            self.assertTrue(published.closed)
            for actual, expected, ordinary in held:
                self.assertEqual(actual, expected)
                self.assertEqual(actual, ordinary)
            report.update(
                accepted=True,
                batches=[13, 2, 35, 7, 96],
                native_hits=len(observer.calls),
                trace_attempts=len(traces),
                original_compiled_references=len(held),
                outputs_survive_close=True,
            )
        finally:
            print(
                "ALLOCATION_LAYOUT_RESULT=" + json.dumps(report, sort_keys=True),
                flush=True,
            )


instantiate_device_type_tests(TestAllocationLayouts, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
