"""Repeated Inductor, user Triton and upstream CuTe GEMM share native lifetimes."""

import json
from pathlib import Path
import struct
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parent
REPO = next(path for path in ROOT.parents if (path / "torch/__init__.py").is_file())
CUDA = REPO / "test/inductor/cudagraph_runtime/cuda"
sys.path[:0] = [str(CUDA), str(CUDA / "cute")]
import runtime_support as support
from fixture_support import load_source

from torch._inductor.runtime._cudagraph import _sdk

_sdk.activate()

import torch
from torch._inductor import config
from torch._inductor.runtime._cudagraph._compiler.compiler_cute_handoff.invocation import register_cute_entry
from torch._inductor.runtime._cudagraph.api import NativeTerminalPolicy
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall, CuteInvokeEvent
from torch._inductor.runtime._cudagraph.frontend import TerminalCall
from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource, InputSource, PointerSource
from torch._inductor.runtime.cudagraph_boxed_replay import _PhysicalCall
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch.testing._internal.common_cuda import tf32_off
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import recover_orig_fp32_precision, run_tests


prototype = load_source("_repeated_gemm_host", ROOT / "host.py")
model = load_source("_repeated_gemm_model", ROOT / "model.py")
SAMPLES = ((128, 0), (128, 8), (256, 16), (256, 24), (128, 32))


class TestRepeatedGemm(support.MixedTestCase):
    @recover_orig_fp32_precision
    @tf32_off()
    def test_generated_three_tensor_composition(self, device):
        if torch.version.hip or torch.cuda.get_device_capability(device)[0] < 8:
            self.skipTest("The upstream TensorOp GEMM requires NVIDIA SM80 or later")
        owner = prototype.make_owner(
            REPO / "third_party/cutlass/examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py"
        )
        self.addCleanup(owner.close)
        registration = register_cute_entry(owner, owned_executor=True)
        self.addCleanup(registration.close)
        self.assertEqual(registration.formals, ("a", "b", "c"))
        policy = NativeTerminalPolicy()
        self.addCleanup(policy.close)
        self.enterContext(config.patch(
            cudagraph_policy=policy,
            assume_aligned_inputs=True,
            static_launch_user_defined_triton_kernels=True,
            combo_kernels=False,
            max_autotune=False,
            **{"triton.multi_kernel": 0},
        ))
        traces, programs, captures = [], [], []
        trace_wrapper = support.terminal_policy.trace_warmed_wrapper
        prepare_terminal = support.terminal_policy.prepare_terminal
        make_replay = support.terminal_replay._make_replay

        def observe_trace(*args):
            trace, views = trace_wrapper(*args)
            events = tuple(event for event in trace.events if type(event) is CuteInvokeEvent)
            self.assertEqual(len(events), model.BLOCKS)
            for event in events:
                self.assertIs(event.entry, registration)
                self.assertEqual(len(event.operands), 3)
            traces.append(trace)
            return trace, views

        def observe_program(program, inputs):
            ordered = tuple(event for event in program.events if type(event) in (TerminalCall, CuTeCall))
            cute = tuple(event for event in ordered if type(event) is CuTeCall)
            flat = tuple(event for event in ordered if type(event) is TerminalCall)
            users = tuple(event for event in flat if event.provider.inductor_meta.get("cudagraph_user_kernel") is True)
            reductions = tuple(event for event in flat if event.provider.heuristic_type.name in
                               ("REDUCTION", "PERSISTENT_REDUCTION"))
            self.assertEqual((len(cute), len(users), len(reductions)), (model.BLOCKS,) * 3)
            self.assertGreater(len(flat), len(users))
            self.assertEqual(len({id(call.receipt.resources) for call in cute}), model.BLOCKS)
            self.assertTrue(all(call.bound.module.artifact._payload is cute[0].bound.module.artifact._payload
                                for call in cute))
            for index, (call, user) in enumerate(zip(cute, users, strict=True)):
                self.assertLess(ordered.index(call), ordered.index(user))
                if index + 1 < len(cute):
                    self.assertLess(ordered.index(user), ordered.index(cute[index + 1]))
                self.assertIs(call.receipt.invocation.compilation.selected, owner.selected)
                self.assertEqual(len(call.pointers), 3)
                left, right, destination = call.pointers
                self.assertIs(type(left.root), BufferSource)
                self.assertIs(type(right.root), InputSource)
                self.assertIs(type(destination.root), BufferSource)
                self.assertNotEqual(left.root, destination.root)
                logical = dict(zip(("a", "b", "c"), call.pointers, strict=True))
                fields = {(field.parameter, field.byte_offset): field.source
                          for field in call.bound.fields if field.kind == "pointer"}
                for pointer in call.bound.module.site.fields.pointers:
                    self.assertEqual(fields[pointer.parameter, pointer.byte_offset],
                                     logical[pointer.source.formal_name])
            self.assertEqual(len({call.pointers[1].root for call in cute}), model.BLOCKS)
            self.assertEqual(len({call.pointers[2].root for call in cute}), model.BLOCKS)
            for call in flat:
                module = call.provider.launchers[0].__globals__["runner"].__self__
                self.modules[id(module)] = module
            programs.append(program)
            return prepare_terminal(program, inputs)

        def observe_capture(graph, input_count, allocations, outputs, copies, calls, launches, buffers, stream, **kwargs):
            physical = [(call, launch) for call, launch in zip(calls, launches, strict=True)
                        if type(call) is _PhysicalCall]
            self.assertEqual(len(physical), model.BLOCKS)
            expected = tuple(event for event in programs[-1].events if type(event) is CuTeCall)
            numeric, inputs = kwargs["numeric"], kwargs["capture_inputs"]
            grids = []
            for (bound, launch), call in zip(physical, expected, strict=True):
                self.assertIs(bound, call.bound)
                self.assertEqual(launch.function, bound.module.function)
                grids.append(tuple(numeric.values[numeric.add(axis)] for axis in bound.grid))
                for field in bound.fields:
                    if type(field.source) is not PointerSource:
                        continue
                    source = field.source
                    root = inputs[source.root.index] if type(source.root) is InputSource else buffers[source.root]
                    address = root.data_ptr() + numeric.values[numeric.add(source.byte_offset)]
                    payload = launch.argument_bytes[field.parameter][field.byte_offset:field.byte_offset + 8]
                    self.assertEqual(struct.unpack("P", payload)[0], address)
            self.assertEqual(len(set(grids)), 1)
            captures.append({"calls": len(calls), "cute_grid": grids[0], "allocations": len(allocations)})
            return make_replay(graph, input_count, allocations, outputs, copies, calls, launches, buffers, stream, **kwargs)

        self.enterContext(mock.patch.object(support.terminal_policy, "trace_warmed_wrapper", observe_trace))
        self.enterContext(mock.patch.object(support.terminal_policy, "prepare_terminal", observe_program))
        self.enterContext(mock.patch.object(support.terminal_replay, "_make_replay", observe_capture))
        compiled = torch.compile(model.RepeatedGemm(registration.key).eval(), fullgraph=True, dynamic=False)
        graph_count = support.counters["stats"]["unique_graphs"]
        installations, observers, samples, held = {}, {}, [], []
        selected = None
        for rows, offset in SAMPLES:
            def tensor(shape, extra):
                count = 1
                for dimension in shape:
                    count *= dimension
                storage = torch.randn(count + extra, dtype=torch.float16, device=device) * 0.125
                return storage[extra:].view(shape)

            arguments = (tensor((rows, model.WIDTH), offset),
                         *(tensor((1, model.WIDTH, model.WIDTH), offset + 8 * index) for index in range(model.BLOCKS)))
            self.assertTrue(all(value.data_ptr() % 16 == 0 for value in arguments))
            old_pointers = {value.data_ptr() for sample in samples for value in sample}
            self.assertTrue(all(value.data_ptr() not in old_pointers for value in arguments))
            samples.append(arguments)
            before = len(programs)
            if rows in installations:
                with observers[rows]:
                    actual = compiled(*arguments)
                self.assertEqual(len(programs), before)
            else:
                actual = compiled(*arguments)
                self.assertEqual(len(programs), before + 1)
                installation = policy.installations[-1]
                self.assertEqual(installation.status, "ready", installation.decline)
                installations[rows] = installation
                observer = support.NativeObservation(installation)
                observer.forbidden_files.update(str(path) for path in support.IMPLEMENTATION)
                observer.forbidden_files.add(sys.modules[CachingAutotuner.__module__].__file__)
                observers[rows] = observer
            installation = installations[rows]
            self.assertEqual(len(installation.variants), 1)
            self.assertEqual(tuple(reason for _, reason in policy.declines), ())
            ordinary = support.original_reference(self, compiled, installation.artifact,
                                                  installation.original, installation.entry, arguments)
            expected = model.reference(*arguments)
            self.assertEqual(actual, ordinary)
            self.assertEqual(actual, expected, atol=2e-3, rtol=2e-3)
            self.assertEqual(actual[0].untyped_storage()._cdata, actual[1].untyped_storage()._cdata)
            self.assertEqual(actual[0].untyped_storage()._cdata, actual[2].untyped_storage()._cdata)
            self.assertEqual(actual[1].storage_offset() - actual[0].storage_offset(), model.WIDTH)
            self.assertEqual(actual[2].stride(), tuple(reversed(actual[0].stride())))
            self.assertEqual(owner._capture.calls, 1)
            if selected is None:
                selected = owner.selected
            self.assertIs(owner.selected, selected)
            held.append((actual, ordinary, expected))

        self.assertEqual((len(traces), len(programs), len(captures), len(policy.installations)), (2, 2, 2, 2))
        self.assertEqual(support.counters["stats"]["unique_graphs"] - graph_count, 2)
        self.assertEqual([capture["cute_grid"] for capture in captures], [(1, 1, 1), (2, 1, 1)])
        self.assertEqual(sum(len(observer.calls) for observer in observers.values()), 3)
        for observer in observers.values():
            self.assertEqual((observer.errors, observer.frames, observer.pending), ([], {}, {}))
        self.assertEqual(len({actual[0].data_ptr() for actual, _, _ in held}), len(SAMPLES))
        physical_calls = tuple(event for program in programs for event in program.events if type(event) is CuTeCall)
        self.assertEqual(len({id(call.receipt.resources) for call in physical_calls}), 2 * model.BLOCKS)
        entries = tuple(installation.entry for installation in policy.installations)
        torch.cuda.synchronize(device)
        policy.close()
        self.assertTrue(all(entry.closed for entry in entries))
        self.assertTrue(all(call.receipt.closed and call.bound.module.closed for call in physical_calls))
        self.assertEqual(owner._native_borrows, set())
        registration.close()
        for actual, ordinary, expected in held:
            self.assertEqual(actual, ordinary)
            self.assertEqual(actual, expected, atol=2e-3, rtol=2e-3)
        print("REPEATED_GEMM_RESULT=" + json.dumps({
            "accepted": True, "blocks": model.BLOCKS, "samples": SAMPLES,
            "ordinary_compilations": 1, "generated_compilations": 2,
            "native_captures": len(captures), "native_hits": 3,
            "cute_calls_per_capture": model.BLOCKS, "user_triton_calls_per_capture": model.BLOCKS,
            "ordinary_references": len(held), "held_outputs_after_close": True,
            "captures": captures,
        }, sort_keys=True), flush=True)


instantiate_device_type_tests(TestRepeatedGemm, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
