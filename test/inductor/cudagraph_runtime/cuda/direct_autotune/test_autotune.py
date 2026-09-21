"""Actual two-config Triton selection through the direct native host cache."""

import json
from pathlib import Path
import struct
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "cute"))
from fixture_support import load_source, REPO_ROOT

from torch._inductor.runtime._cudagraph import _sdk

_sdk.activate()

import torch
from torch._inductor.runtime._cudagraph._compiler.entry_signature import SignaturePolicy
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import InputContract, IntegerRange, TensorInput
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner import ObservedOrdinaryEntry
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch._inductor.runtime._cudagraph.direct_cute import DirectCuTe
import torch._inductor.runtime._cudagraph.direct_host as direct_host
from torch._inductor.runtime._cudagraph.direct_triton import DirectTriton
from torch._inductor.runtime._cudagraph.frontend import DirectKernelCall
import torch._inductor.runtime._cudagraph.replay as replay
from torch._inductor.runtime.cudagraph_arg_mapping import ExpressionSource, InputSource, IntegerSource, IntExpr, PointerSource
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram, _PhysicalCall
from torch._inductor.runtime.cudagraph_launch_association import associate_kernel_launches
from torch.testing._internal.common_cuda import tf32_off
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import recover_orig_fp32_precision, run_tests, TestCase


import shared_host_model
import cute_fixture as fixture
model = load_source("_direct_autotune_model", ROOT / "model.py")


class TestDirectAutotune(TestCase):
    @recover_orig_fp32_precision
    @tf32_off()
    def test_gemm_composition(self, device):
        entry, kernel = fixture.make_fixture()
        owner = ObservedOrdinaryEntry(entry, kernel, policy=SignaturePolicy(32, 64, 16, "stream"),
                                      conversion=fixture.convert_arguments)
        gemm = DirectTriton(model.matmul)
        add = DirectTriton(shared_host_model.dynamic_user_add_one)
        cute = DirectCuTe(owner)
        model.OPERATION = model.make_operation(gemm, cute, add)
        self.addCleanup(owner.close)
        self.addCleanup(add.close)
        self.addCleanup(gemm.close)
        self.assertEqual(len(model.matmul.configs), 2)
        self.assertEqual(model.matmul.keys, ["COLS"])
        self.enterContext(mock.patch.object(torch, "compile", side_effect=AssertionError("No host compiler")))
        self.enterContext(mock.patch.object(torch.func, "functionalize", side_effect=AssertionError("No functionalization")))
        n_expr, m_expr = IntExpr("boxed", 0), IntExpr("boxed", 1)
        rows = IntExpr("multiply", None, (IntExpr("constant", 2), n_expr))
        columns = IntExpr("multiply", None, (IntExpr("constant", 128), m_expr))
        contract = InputContract(("integer", "integer", "tensor", "tensor"),
            (TensorInput(2, torch.float16, (rows, 128), (128, 1)),
             TensorInput(3, torch.float16, (128, columns), (columns, 1))),
            (IntegerRange(0, 2, 127), IntegerRange(1, 2, 127)), device_index=0)
        runtime = direct_host.DirectHost(model.host, contract)
        self.addCleanup(runtime.close)
        pairs = ((5, 5), (35, 5), (7, 7), (35, 7), (13, 5), (17, 7))
        samples = []
        for index, (n, m) in enumerate(pairs):
            offset = 1 if index % 2 else 32
            left = torch.randn(2 * n * 128 + offset, dtype=torch.float16, device=device)[offset:].view(2 * n, 128)
            right = torch.randn(128 * 128 * m + offset, dtype=torch.float16, device=device)[offset:].view(128, 128 * m)
            samples.append((n, m, left, right))
        observations, ordinary_calls, traces, captures, frames = [], [], [], [], []
        observed_results, selected_calls, tuning_reports = [], [], []
        observe_direct, trace_host, make_replay = direct_host._observe_direct, direct_host.trace_host, replay._make_replay

        def count_ordinary(frame, event, result):
            if event == "call" and frame.f_code is model.host.__code__:
                ordinary_calls.append(tuple(frame.f_locals["box"][:2]))

        def observe(*args, **kwargs):
            observations.append(tuple(args[-1][:2]))
            try:
                sys.setprofile(count_ordinary)
                result = observe_direct(*args, **kwargs)
            finally:
                sys.setprofile(None)
            observed_results.append(result)
            call, = gemm._calls
            selected_calls.append(call)
            self.assertIs(call.config, model.matmul.best_config)
            self.assertEqual(call.config_values, tuple(call.config.all_kwargs().items()))
            self.assertEqual(call.tuning_keys, (("COLS", observations[-1][1] * 128),))
            timings = model.matmul.configs_timings
            self.assertEqual({id(config) for config in timings}, {id(config) for config in model.matmul.configs})
            tuning_reports.append({"key_cols": observations[-1][1] * 128,
                "selected_config": dict(call.config_values),
                "config_timings_ms": [{"config": config.all_kwargs(), "timings": list(values)}
                                      for config, values in timings.items()]})
            return result

        def trace(*args, **kwargs):
            traces.append(tuple(args[2][:2]))
            return trace_host(*args, **kwargs)

        def capture(graph, input_count, allocations, outputs, copies, calls, launches, buffers, stream, **kwargs):
            inputs = samples[0 if not captures else 2]
            n, m = inputs[:2]
            numeric = kwargs["numeric"]
            selected = selected_calls[-1]
            config = dict(selected.config_values)
            self.assertEqual(input_count, 4)
            self.assertEqual(tuple(copies), ())
            self.assertEqual(len(calls), 3)
            self.assertIs(calls[0].module, selected.owner.module)
            nodes = tuple(launch.after[3][0][0] for launch in launches)
            associated = associate_kernel_launches(tuple(launches), graph._inspect_captured_kernel_nodes(nodes))
            expected_grid = ((2 * n + config["BLOCK_M"] - 1) // config["BLOCK_M"],
                             (128 * m + config["BLOCK_N"] - 1) // config["BLOCK_N"], 1)
            self.assertEqual(associated[0].snapshot[4], expected_grid)
            self.assertEqual(associated[0].snapshot[5], (config["num_warps"] * 32, 1, 1))
            for call, launch in zip(calls, launches, strict=True):
                physical = type(call) is _PhysicalCall
                for index, field in enumerate(call.fields if physical else call.arguments):
                    source = field.source
                    if type(source) is PointerSource:
                        tensor = inputs[source.root.index] if type(source.root) is InputSource else buffers[source.root]
                        expected = tensor.data_ptr() + numeric.values[numeric.add(source.byte_offset)]
                        code = "P"
                    else:
                        self.assertIn(type(source), (IntegerSource, ExpressionSource))
                        expected = source.value if type(source) is IntegerSource else numeric.values[numeric.add(source.expression)]
                        code = "i" if (field.kind if physical else field.triton_type) == "i32" else "q"
                    parameter, offset = (field.parameter, field.byte_offset) if physical else (index, 0)
                    payload = launch.argument_bytes[parameter][offset:offset + struct.calcsize(code)]
                    self.assertEqual(struct.unpack(code, payload)[0], expected)
            captures.append(expected_grid)
            return make_replay(graph, input_count, allocations, outputs, copies, calls, launches, buffers, stream, **kwargs)

        def profile(frame, event, result):
            if event == "call":
                frames.append((frame.f_code.co_filename, frame.f_code.co_name))

        held, reports, native_entry = [], [], None
        with mock.patch.object(direct_host, "_observe_direct", observe), \
             mock.patch.object(direct_host, "trace_host", trace), \
             mock.patch.object(replay, "_make_replay", capture):
            for index, inputs in enumerate(samples):
                miss = index in (0, 2)
                before = len(ordinary_calls)
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
                self.assertIs(runtime.entry, native_entry)
                if miss:
                    self.assertIs(actual, observed_results[-1])
                self.assertEqual(len(ordinary_calls) - before, int(miss))
                self.assertEqual(len(observations), len(ordinary_calls))
                self.assertEqual(len(traces), len(ordinary_calls))
                self.assertEqual(len(captures), len(ordinary_calls))
                self.assertEqual(len(runtime.variants), len(ordinary_calls))
                ordinary = model.host(list(inputs))
                self.assertEqual(actual, ordinary, atol=2e-5, rtol=2e-5)
                self.assertEqual(actual, model.torch_reference(*inputs), atol=2e-4, rtol=2e-4)
                self.assertEqual(actual[0].untyped_storage()._cdata, actual[1].untyped_storage()._cdata)
                held.append((actual, tuple(value.clone() for value in actual)))
                reports.append({"n": inputs[0], "m": inputs[1], "path": "ordinary_miss" if miss else "native_hit",
                                "input_storage_offsets": [value.storage_offset() for value in inputs[2:]]})

        self.assertEqual(observations, [(5, 5), (7, 7)])
        self.assertEqual(ordinary_calls, observations)
        self.assertEqual(traces, observations)
        self.assertEqual(len(model.matmul.cache), 2)
        self.assertEqual(owner._capture.calls, 1)
        triton_calls, cute_calls = [], []
        for variant, observed, pair in zip(runtime.variants, selected_calls, observations, strict=True):
            variant.program.check()
            self.assertIsNotNone(variant.guard)
            calls = tuple(event for event in variant.program.events if type(event) is DirectKernelCall)
            self.assertEqual(len(calls), 2)
            self.assertIs(calls[0].owner, observed.owner)
            config = dict(observed.config_values)
            for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K"):
                value, = (row.constant for row in calls[0].owner.formals if row.formal == name)
                self.assertEqual(value, config[name])
            grids = []
            for inputs in samples:
                if inputs[1] != pair[1]:
                    continue
                numeric = _NumericProgram(variant.program, inputs)
                grid = tuple(numeric.values[numeric.add(axis)] for axis in calls[0].grid)
                self.assertEqual(grid, ((2 * inputs[0] + config["BLOCK_M"] - 1) // config["BLOCK_M"],
                                       (128 * inputs[1] + config["BLOCK_N"] - 1) // config["BLOCK_N"], 1))
                grids.append(grid)
            self.assertGreater(len(set(grids)), 1)
            cute_call, = (event for event in variant.program.events if type(event) is CuTeCall)
            self.assertEqual(cute_call.pointers[0].root, cute_call.pointers[1].root)
            self.assertEqual(cute_call.pointers[2], cute_call.pointers[3])
            triton_calls.extend(calls)
            cute_calls.append(cute_call)
        self.assertTrue(all(call.owner.module._graph_borrows > 0 for call in triton_calls))
        runtime.close()
        self.assertTrue(native_entry.closed)
        self.assertTrue(all(call.receipt.closed for call in cute_calls))
        self.assertEqual(owner._native_borrows, set())
        self.assertTrue(all(call.owner.module._graph_borrows == 0 for call in triton_calls))
        self.assertTrue(all(not call.owner.closed for call in triton_calls))
        for actual, expected in held:
            self.assertEqual(actual, expected)
        gemm.close()
        add.close()
        owner.close()
        report = {"accepted": True,
                  "ordinary_miss_calls": len(ordinary_calls), "trace_calls": len(traces), "variants": 2,
                  "native_hits": 4, "ordinary_references": len(samples), "ordinary_cute_compilations": owner._capture.calls,
                  "selected_configurations": tuning_reports, "captured_gemm_grids": captures,
                  "python_frames_on_native_hits": frames, "samples": reports, "held_outputs_after_close": True}
        print("DIRECT_AUTOTUNE_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


instantiate_device_type_tests(TestDirectAutotune, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
