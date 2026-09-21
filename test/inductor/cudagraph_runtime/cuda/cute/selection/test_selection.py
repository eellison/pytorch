"""Ordinary CuTe configuration selection preserves independent native variants."""

import json
from pathlib import Path
import struct
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from fixture_support import load_source, REPO_ROOT

from torch._inductor.runtime._cudagraph import _sdk

_sdk.activate()

import torch
from torch._inductor.runtime._cudagraph._compiler.entry_signature import SignaturePolicy
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import InputContract, IntegerRange, TensorInput
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner import ObservedOrdinaryEntry
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall, CuteInvokeEvent
from torch._inductor.runtime._cudagraph.direct_cute import DirectCuTe
import torch._inductor.runtime._cudagraph.direct_host as direct_host
from torch._inductor.runtime._cudagraph.direct_triton import DirectTriton
import torch._inductor.runtime._cudagraph.replay as replay
from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource, ExpressionSource, InputSource, IntegerSource, IntExpr, PointerSource
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram, _PhysicalCall
from torch._inductor.runtime.cudagraph_launch_association import associate_kernel_launches
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


fixture = load_source("_cute_selection_fixture", ROOT / "fixture.py")
import shared_host_model as model


class TestCuTeSelection(TestCase):
    def test_selected_configuration(self, device):
        owners, adapters = [], []
        for tile in (1, 2):
            entry, kernel = fixture.make_fixture(tile)
            owner = ObservedOrdinaryEntry(entry, kernel, policy=SignaturePolicy(32, 64, 256, "stream"),
                                          conversion=fixture.convert_arguments)
            self.addCleanup(owner.close)
            owners.append(owner)
            adapters.append(DirectCuTe(owner))
        add = DirectTriton(model.dynamic_user_add_one)
        self.addCleanup(add.close)
        model.ADD, model.CUTE = add, fixture.make_selection(*adapters)
        self.enterContext(mock.patch.object(torch, "compile", side_effect=AssertionError("No host compiler")))
        self.enterContext(mock.patch.object(torch.func, "functionalize", side_effect=AssertionError("No functionalization")))
        n_expr, m_expr = IntExpr("boxed", 0), IntExpr("boxed", 1)
        count = IntExpr("multiply", None, (IntExpr("multiply", None, (n_expr, m_expr)), IntExpr("constant", 128)))
        length = IntExpr("multiply", None, (IntExpr("constant", 3), count))
        contract = InputContract(("integer", "integer", "tensor"),
            (TensorInput(2, torch.float32, (length,), (1,)),),
            (IntegerRange(0, 2, 127), IntegerRange(1, 2, 127)), device_index=0)
        runtime = direct_host.DirectHost(model.host, contract)
        self.addCleanup(runtime.close)
        pairs = ((5, 13), (5, 7), (7, 7), (3, 35), (13, 5), (35, 7))
        samples = []
        for index, (n, m) in enumerate(pairs):
            offset = 1 if index % 2 else 32
            value = torch.randn(3 * n * m * 128 + offset, device=device)[offset:]
            samples.append((n, m, value))
        ordinary_calls, observations, traces, captures, frames = [], [], [], [], []
        observe_direct, trace_host, make_replay = direct_host._observe_direct, direct_host.trace_host, replay._make_replay

        def count_ordinary(frame, event, result):
            if event == "call" and frame.f_code is model.host.__code__:
                ordinary_calls.append(tuple(frame.f_locals["box"][:2]))

        def observe(*args, **kwargs):
            try:
                sys.setprofile(count_ordinary)
                result = observe_direct(*args, **kwargs)
            finally:
                sys.setprofile(None)
            observations.append(result)
            return result

        def trace(*args, **kwargs):
            traces.append(tuple(args[2][:2]))
            return trace_host(*args, **kwargs)

        def capture(graph, input_count, allocations, outputs, copies, calls, launches, buffers, stream, **kwargs):
            inputs = samples[0 if not captures else 2]
            n, m = inputs[:2]
            tile = 2 if m > n else 1
            numeric = kwargs["numeric"]
            self.assertEqual(input_count, 3)
            self.assertEqual(tuple(copies), ())
            self.assertEqual(len(allocations), 2)
            self.assertEqual(len(calls), 3)
            self.assertIs(type(calls[1]), _PhysicalCall)
            nodes = tuple(launch.after[3][0][0] for launch in launches)
            associated = associate_kernel_launches(tuple(launches), graph._inspect_captured_kernel_nodes(nodes))
            self.assertEqual(associated[1].snapshot[4], ((m + tile - 1) // tile, n, 1))
            self.assertEqual(associated[1].snapshot[5], (32 * tile, 1, 1))
            for call, launch in zip(calls, launches, strict=True):
                physical = type(call) is _PhysicalCall
                for index, field in enumerate(call.fields if physical else call.arguments):
                    source = field.source
                    if type(source) is PointerSource:
                        root = inputs[source.root.index] if type(source.root) is InputSource else buffers[source.root]
                        expected, code = root.data_ptr() + numeric.values[numeric.add(source.byte_offset)], "P"
                        if physical:
                            self.assertIs(type(source.root), BufferSource)
                            self.assertEqual(expected % 256, 0)
                    else:
                        self.assertIn(type(source), (IntegerSource, ExpressionSource))
                        expected = source.value if type(source) is IntegerSource else numeric.values[numeric.add(source.expression)]
                        code = "i" if (field.kind if physical else field.triton_type) == "i32" else "q"
                    parameter, offset = (field.parameter, field.byte_offset) if physical else (index, 0)
                    payload = launch.argument_bytes[parameter][offset:offset + struct.calcsize(code)]
                    self.assertEqual(struct.unpack(code, payload)[0], expected)
            captures.append({"rows_per_cta": tile, "grid": associated[1].snapshot[4], "block": associated[1].snapshot[5]})
            return make_replay(graph, input_count, allocations, outputs, copies, calls, launches, buffers, stream, **kwargs)

        def profile(frame, event, result):
            if event == "call":
                frames.append((frame.f_code.co_filename, frame.f_code.co_name))

        held, reports, native_entry = [], [], None
        with mock.patch.object(direct_host, "_observe_direct", observe), \
             mock.patch.object(direct_host, "trace_host", trace), \
             mock.patch.object(replay, "_make_replay", capture):
            for index, inputs in enumerate(samples):
                n, m, value = inputs
                miss = index in (0, 2)
                before = len(ordinary_calls)
                box = list(inputs)
                if index == 0:
                    actual = runtime(box)
                    native_entry = runtime.entry
                elif miss:
                    actual = native_entry(box)
                else:
                    try:
                        sys.setprofile(profile)
                        actual = native_entry(box)
                    finally:
                        sys.setprofile(None)
                    self.assertEqual(frames, [])
                self.assertIs(runtime.entry, native_entry)
                self.assertEqual(box, [])
                self.assertEqual(len(ordinary_calls) - before, int(miss))
                self.assertEqual(len(runtime.variants), len(ordinary_calls))
                if miss:
                    self.assertIs(actual, observations[-1])
                ordinary_box = list(inputs)
                ordinary = model.host(ordinary_box)
                self.assertEqual(ordinary_box, [])
                self.assertNotEqual(actual[0].data_ptr(), ordinary[0].data_ptr())
                self.assertEqual(actual, ordinary, atol=2e-5, rtol=2e-5)
                self.assertEqual(actual, model.eager_reference(*inputs), atol=2e-5, rtol=2e-5)
                self.assertEqual(actual[0].untyped_storage()._cdata, actual[1].untyped_storage()._cdata)
                self.assertEqual((tuple(actual[0].shape), actual[0].stride()), ((m, n, 128), (128, m * 128, 1)))
                held.append((actual, tuple(tensor.clone() for tensor in actual)))
                reports.append({"n": n, "m": m, "tile": 2 if m > n else 1, "path": "miss" if miss else "native_hit"})
        self.assertEqual(ordinary_calls, [(5, 13), (7, 7)])
        self.assertEqual(traces, ordinary_calls)
        self.assertEqual(len(captures), 2)
        self.assertEqual([owner._capture.calls for owner in owners], [1, 1])
        self.assertIsNot(owners[0].selected, owners[1].selected)
        self.assertNotEqual(owners[0].compiled_sha256, owners[1].compiled_sha256)
        calls = []
        for variant, tile in zip(runtime.variants, (2, 1), strict=True):
            variant.program.check()
            self.assertIsNotNone(variant.guard)
            self.assertTrue(variant.guard.expressions)
            event, = (event for event in variant.program.guards.trace.events if type(event) is CuteInvokeEvent)
            self.assertIs(event.entry, adapters[tile - 1])
            call, = (event for event in variant.program.events if type(event) is CuTeCall)
            self.assertIs(call.receipt.borrow.owner, owners[tile - 1])
            self.assertIs(call.receipt.invocation.compilation.selected, owners[tile - 1].selected)
            self.assertTrue(owners[tile - 1]._native_borrows)
            self.assertEqual(call.pointers[2], call.pointers[3])
            self.assertEqual(tuple(formal.data_alignment for formal in call.bound.module.artifact.formals
                                   if formal.kind == "Tensor"), (256, 256, 256, 256))
            for n, m, value in samples:
                numeric = _NumericProgram(variant.program, (n, m, value))
                grid = tuple(numeric.values[numeric.add(axis)] for axis in call.bound.grid)
                self.assertEqual(grid, ((m + tile - 1) // tile, n, 1))
                offsets = tuple(numeric.values[numeric.add(pointer.byte_offset)] for pointer in call.pointers)
                self.assertEqual(offsets, (512 * n * m, 1024 * n * m, 0, 0))
            calls.append(call)
        runtime.close()
        self.assertTrue(native_entry.closed)
        self.assertTrue(all(call.receipt.closed for call in calls))
        self.assertTrue(all(not owner._native_borrows for owner in owners))
        for actual, expected in held:
            self.assertEqual(actual, expected)
        add.close()
        for owner in owners:
            owner.close()
        report = {"accepted": True,
                  "ordinary_miss_calls": len(ordinary_calls), "trace_calls": len(traces), "variants": 2,
                  "native_hits": 4, "ordinary_references": len(samples), "ordinary_compilations": [1, 1],
                  "captures": captures, "samples": reports, "python_frames_on_native_hits": frames,
                  "held_outputs_after_close": True}
        print("CUTE_SELECTION_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


instantiate_device_type_tests(TestCuTeSelection, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
