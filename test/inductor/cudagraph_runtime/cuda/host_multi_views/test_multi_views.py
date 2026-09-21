"""Actual Tensor-list views retain their storage through direct native replay."""

from dis import get_instructions
import json
from pathlib import Path
import sys
import struct
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "cute"))
from fixture_support import load_source

from torch._inductor.runtime._cudagraph import _sdk

_sdk.activate()

import shared_host_model
fixture = load_source("_host_multi_views_cute", ROOT.parent / "direct_autotune/cute_fixture.py")

import torch
from torch._inductor.runtime._cudagraph._compiler.entry_signature import SignaturePolicy
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import InputContract, IntegerRange, ReinterpretEvent, TensorInput
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner import ObservedOrdinaryEntry
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch._inductor.runtime._cudagraph.direct_cute import DirectCuTe
import torch._inductor.runtime._cudagraph.direct_host as direct_host
from torch._inductor.runtime._cudagraph.direct_triton import DirectTriton
from torch._inductor.runtime._cudagraph.frontend import DirectKernelCall
import torch._inductor.runtime._cudagraph.replay as replay
from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource, IntExpr, PointerSource
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

import model as guard_model


class TestHostMultiViews(TestCase):
    def test_tensor_list_variants(self, device):
        entry, kernel = fixture.make_fixture()
        owner = ObservedOrdinaryEntry(entry, kernel, policy=SignaturePolicy(32, 64, 16, "stream"),
                                      conversion=fixture.convert_arguments)
        add = DirectTriton(shared_host_model.dynamic_user_add_one)
        cute = DirectCuTe(owner)
        guard_model.OPERATION = guard_model.make_operation(add, cute)
        operation = guard_model.OPERATION
        cells = dict(zip(operation.__code__.co_freevars, (cell.cell_contents for cell in operation.__closure__), strict=True))
        self.assertIs(cells["add"], add)
        self.assertIs(cells["cute"], cute)
        globals_read = (instruction.argval for instruction in get_instructions(guard_model.host)
                        if instruction.opname == "LOAD_GLOBAL")
        self.assertFalse(any(type(guard_model.host.__globals__.get(name)) in (DirectCuTe, DirectTriton)
                             for name in globals_read))
        self.addCleanup(owner.close)
        self.addCleanup(add.close)
        self.enterContext(mock.patch.object(torch, "compile", side_effect=AssertionError("Direct host must not compile")))
        self.enterContext(mock.patch.object(torch.func, "functionalize", side_effect=AssertionError("No functionalization")))
        n_expr, m_expr = IntExpr("boxed", 0), IntExpr("boxed", 1)
        count = IntExpr("multiply", None, (IntExpr("multiply", None, (n_expr, m_expr)), IntExpr("constant", 128)))
        length = IntExpr("multiply", None, (IntExpr("constant", 3), count))
        contract = InputContract(("integer", "integer", "tensor"),
            (TensorInput(2, torch.float32, (length,), (1,)),),
            (IntegerRange(0, 2, 127), IntegerRange(1, 2, 127)), device_index=0)
        runtime = direct_host.DirectHost(guard_model.host, contract)
        self.addCleanup(runtime.close)
        self.assertIsNone(runtime.entry)
        pairs = ((5, 13), (7, 17), (13, 5), (17, 7), (3, 35), (11, 11))
        samples = []
        for index, (n, m) in enumerate(pairs):
            offset = 1 if index % 2 else 32
            backing = torch.randn(3 * n * m * 128 + offset, device=device)
            samples.append((n, m, backing[offset:]))
        observations, ordinary_calls, traces, captures, frames, view_lists = [], [], [], [], [], []
        observed_results = []
        observe_direct = direct_host._observe_direct
        trace_host = direct_host.trace_host
        make_replay = replay._make_replay
        tensor_dispatch = direct_host.HostTensorEvents.__torch_dispatch__

        def record_views(mode, func, types, args=(), kwargs=None):
            before = len(mode.state.events)
            result = tensor_dispatch(mode, func, types, args, kwargs)
            if type(result) in (tuple, list):
                self.assertEqual(len(result), 2)
                events = mode.state.events[before:]
                self.assertEqual(len(events), 2)
                for event, tensor in zip(events, result, strict=True):
                    self.assertIs(type(event), ReinterpretEvent)
                    self.assertIs(event.source, args[0])
                    self.assertIs(event.tensor, tensor)
                    self.assertTrue(torch._C._is_alias_of(event.source, tensor))
                view_lists.append((str(func), tuple(events)))
            return result

        def count_ordinary(frame, event, result):
            if event == "call" and frame.f_code is guard_model.host.__code__:
                box = frame.f_locals["box"]
                self.assertIs(type(box[0]), int)
                ordinary_calls.append(tuple(box[:2]))

        def observe(*args, **kwargs):
            box = args[-1]
            observations.append(tuple(box[:2]))
            try:
                sys.setprofile(count_ordinary)
                result = observe_direct(*args, **kwargs)
            finally:
                sys.setprofile(None)
            observed_results.append(result)
            return result

        def trace(*args, **kwargs):
            traces.append(tuple(args[2][:2]))
            return trace_host(*args, **kwargs)

        def capture(*args, **kwargs):
            calls, launches, buffers = args[5:8]
            physical, launch = calls[1], launches[1]
            numeric = kwargs["numeric"]
            pointers = []
            for field in physical.fields:
                if type(field.source) is PointerSource:
                    source = field.source
                    self.assertIs(type(source.root), BufferSource)
                    expected = buffers[source.root].data_ptr() + numeric.values[numeric.add(source.byte_offset)]
                    payload = launch.argument_bytes[field.parameter][field.byte_offset:field.byte_offset + 8]
                    self.assertEqual(struct.unpack("P", payload)[0], expected)
                    pointers.append(expected)
            self.assertEqual(len(pointers), 4)
            n, m = traces[-1]
            self.assertEqual(pointers[1] - pointers[0], n * m * 128 * 4)
            self.assertEqual(pointers[2], pointers[3])
            self.assertEqual(launches[-1].argument_bytes[0], launches[-1].argument_bytes[1])
            captures.append(tuple(calls))
            return make_replay(*args, **kwargs)

        def profile(frame, event, result):
            if event == "call":
                frames.append((frame.f_code.co_filename, frame.f_code.co_name))

        held, reports = [], []
        native_entry = None
        with mock.patch.object(direct_host, "_observe_direct", observe), \
             mock.patch.object(direct_host, "trace_host", trace), \
             mock.patch.object(replay, "_make_replay", capture), \
             mock.patch.object(direct_host.HostTensorEvents, "__torch_dispatch__", record_views):
            for index, inputs in enumerate(samples):
                n, m, value = inputs
                miss = index in (0, 2)
                previous = len(ordinary_calls)
                box = list(inputs)
                if index == 0:
                    actual = runtime(box)
                    native_entry = runtime.entry
                    self.assertIs(type(native_entry), torch._C._CUDAGraphBoxedDispatch)
                elif miss:
                    actual = runtime.entry(box)
                else:
                    try:
                        sys.setprofile(profile)
                        actual = runtime.entry(box)
                    finally:
                        sys.setprofile(None)
                    self.assertEqual(frames, [])
                self.assertEqual(box, [])
                if miss:
                    self.assertIs(actual, observed_results[-1])
                self.assertIs(runtime.entry, native_entry)
                self.assertEqual(len(ordinary_calls) - previous, int(miss))
                self.assertEqual(len(observations), len(ordinary_calls))
                self.assertEqual(len(traces), len(ordinary_calls))
                self.assertEqual(len(captures), len(ordinary_calls))
                self.assertEqual(len(runtime.variants), len(ordinary_calls))
                ordinary_box = list(inputs)
                ordinary = guard_model.host(ordinary_box)
                self.assertEqual(ordinary_box, [])
                self.assertEqual(actual, ordinary, atol=2e-5, rtol=2e-5)
                self.assertEqual(actual, shared_host_model.eager_reference(n, m, value), atol=2e-5, rtol=2e-5)
                self.assertEqual((tuple(actual[0].shape), actual[0].stride()), ((m, n, 128), (128, m * 128, 1)))
                self.assertEqual((tuple(actual[1].shape), actual[1].stride()), ((n, m, 128), (m * 128, 128, 1)))
                self.assertEqual(actual[0].untyped_storage()._cdata, actual[1].untyped_storage()._cdata)
                held.append((actual, tuple(value.clone() for value in actual)))
                reports.append({"n": n, "m": m, "path": "ordinary_miss" if miss else "native_hit",
                                "input_storage_offset": value.storage_offset()})

        self.assertEqual(observations, [(5, 13), (13, 5)])
        self.assertEqual(ordinary_calls, observations)
        self.assertEqual(traces, observations)
        self.assertEqual(len(view_lists), 2)
        self.assertEqual(len(runtime.variants), 2)
        self.assertEqual(owner._capture.calls, 1)
        triton_calls, cute_calls = [], []
        for variant, expected_block in zip(runtime.variants, (128, 256), strict=True):
            variant.program.check()
            self.assertIsNotNone(variant.guard)
            self.assertTrue(variant.guard.expressions)
            calls = tuple(event for event in variant.program.events if type(event) is DirectKernelCall)
            self.assertEqual(len(calls), 2)
            for call in calls:
                block, = (row.constant for row in call.owner.formals if row.formal == "BLOCK")
                self.assertEqual(block, expected_block)
                self.assertGreater(call.owner.module._graph_borrows, 0)
            cute_call, = (event for event in variant.program.events if type(event) is CuTeCall)
            self.assertEqual(cute_call.pointers[2], cute_call.pointers[3])
            triton_calls.extend(calls)
            cute_calls.append(cute_call)
        self.assertIsNot(triton_calls[0].owner, triton_calls[2].owner)
        self.assertEqual(tuple(item.owner for item in add._calls), tuple(call.owner for call in triton_calls[2:]))
        runtime.close()
        self.assertTrue(runtime.closed)
        self.assertTrue(native_entry.closed)
        self.assertTrue(all(call.receipt.closed for call in cute_calls))
        self.assertEqual(owner._native_borrows, set())
        self.assertTrue(all(call.owner.module._graph_borrows == 0 for call in triton_calls))
        self.assertTrue(all(not call.owner.closed for call in triton_calls))
        for actual, expected in held:
            self.assertEqual(actual, expected)
        add.close()
        owner.close()
        report = {"accepted": True,
                  "variants": 2, "ordinary_miss_calls": len(ordinary_calls), "trace_calls": len(traces),
                  "native_hits": len(samples) - len(ordinary_calls), "ordinary_references": len(samples),
                  "selected_blocks": [128, 256], "ordinary_cute_compilations": owner._capture.calls,
                  "tensor_list_operations": [name for name, _ in view_lists],
                  "recorded_tensor_list_views": sum(len(events) for _, events in view_lists),
                  "python_frames_on_native_hits": frames, "samples": reports, "held_outputs_after_close": True}
        print("HOST_MULTI_VIEWS_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


instantiate_device_type_tests(TestHostMultiViews, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
