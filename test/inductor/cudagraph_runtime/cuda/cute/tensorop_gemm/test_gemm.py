"""Unchanged upstream CuTe tensorcore GEMM through ordinary capture and replay."""

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
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import InputContract, IntegerRange, TensorInput
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch._inductor.runtime._cudagraph.direct_cute import DirectCuTe
import torch._inductor.runtime._cudagraph.direct_host as direct_host
from torch._inductor.runtime._cudagraph.direct_triton import DirectTriton
import torch._inductor.runtime._cudagraph.replay as replay
from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource, ExpressionSource, InputSource, IntegerSource, IntExpr, PointerSource
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram, _PhysicalCall
from torch._inductor.runtime.cudagraph_launch_association import associate_kernel_launches
from torch.testing._internal.common_cuda import tf32_off
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import recover_orig_fp32_precision, run_tests, TestCase


prototype = load_source("_upstream_gemm_host", ROOT / "host_prototype.py")
import shared_host_model
model = load_source("_upstream_gemm_model", ROOT / "model.py")
UPSTREAM = (REPO_ROOT / "third_party/cutlass/examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py")


class TestUpstreamCuTeGemm(TestCase):
    @recover_orig_fp32_precision
    @tf32_off()
    def test_tensorcore_gemm(self, device):
        owner = prototype.make_owner(UPSTREAM)
        self.addCleanup(owner.close)
        self.assertIsNone(owner._cold)
        self.assertIs(owner.kernel.__self__, prototype.GEMM)
        add = DirectTriton(shared_host_model.dynamic_user_add_one)
        self.addCleanup(add.close)
        model.ADD, model.GEMM = add, DirectCuTe(owner)
        self.enterContext(mock.patch.object(torch, "compile", side_effect=AssertionError("No host compiler")))
        self.enterContext(mock.patch.object(torch.func, "functionalize", side_effect=AssertionError("No functionalization")))
        rows = IntExpr("boxed", 0)
        length = IntExpr("multiply", None, (IntExpr("constant", 2048), rows))
        contract = InputContract(("integer", "tensor", "tensor"),
            (TensorInput(1, torch.float16, (length,), (1,)), TensorInput(2, torch.float16, (32768,), (1,))),
            (IntegerRange(0, 2, 127),), device_index=0)
        runtime = direct_host.DirectHost(model.host, contract)
        self.addCleanup(runtime.close)
        samples = []
        for index, row_count in enumerate((16, 11, 35, 96, 17)):
            offset = 1 if index % 2 else 32
            a = torch.randn(2048 * row_count + offset, dtype=torch.float16, device=device)[offset:]
            b = torch.randn(32768 + offset, dtype=torch.float16, device=device)[offset:]
            samples.append((row_count, a, b))
        captures, frames, ordinary_calls = [], [], []
        make_replay, observe_direct = replay._make_replay, direct_host._observe_direct

        def ordinary_profile(frame, event, result):
            if event == "call" and frame.f_code is model.host.__code__:
                ordinary_calls.append(frame.f_locals["box"][0])

        def observe(*args, **kwargs):
            try:
                sys.setprofile(ordinary_profile)
                return observe_direct(*args, **kwargs)
            finally:
                sys.setprofile(None)

        def capture(graph, input_count, allocations, outputs, copies, calls, launches, buffers, stream, **kwargs):
            self.assertEqual(input_count, 3)
            self.assertEqual(len(allocations), 3)
            self.assertEqual(tuple(copies), ())
            self.assertEqual(len(calls), 3)
            self.assertIs(type(calls[2]), _PhysicalCall)
            numeric = kwargs["numeric"]
            nodes = tuple(launch.after[3][0][0] for launch in launches)
            associated = associate_kernel_launches(tuple(launches), graph._inspect_captured_kernel_nodes(nodes))
            self.assertEqual(associated[2].snapshot[4], (1, 1, 1))
            self.assertEqual(associated[2].snapshot[5], calls[2].module.block)
            self.assertEqual(calls[2].module.block, (prototype.GEMM.num_threads, 1, 1))
            self.assertGreater(calls[2].module.shared, 0)
            for call, launch in zip(calls, launches, strict=True):
                physical = type(call) is _PhysicalCall
                for index, field in enumerate(call.fields if physical else call.arguments):
                    source = field.source
                    if type(source) is PointerSource:
                        root = samples[0][source.root.index] if type(source.root) is InputSource else buffers[source.root]
                        expected, code = root.data_ptr() + numeric.values[numeric.add(source.byte_offset)], "P"
                        if physical:
                            self.assertIs(type(source.root), BufferSource)
                            self.assertEqual(expected % 16, 0)
                    else:
                        self.assertIn(type(source), (IntegerSource, ExpressionSource))
                        expected = source.value if type(source) is IntegerSource else numeric.values[numeric.add(source.expression)]
                        code = "i" if (field.kind if physical else field.triton_type) == "i32" else "q"
                    parameter, offset = (field.parameter, field.byte_offset) if physical else (index, 0)
                    payload = launch.argument_bytes[parameter][offset:offset + struct.calcsize(code)]
                    self.assertEqual(struct.unpack(code, payload)[0], expected)
            captures.append({"grid": associated[2].snapshot[4], "block": calls[2].module.block,
                             "shared_bytes": calls[2].module.shared})
            return make_replay(graph, input_count, allocations, outputs, copies, calls, launches, buffers, stream, **kwargs)

        def profile(frame, event, result):
            if event == "call":
                frames.append((frame.f_code.co_filename, frame.f_code.co_name))

        held = []
        with mock.patch.object(direct_host, "_observe_direct", observe), mock.patch.object(replay, "_make_replay", capture):
            for index, inputs in enumerate(samples):
                box = list(inputs)
                if index == 0:
                    actual = runtime(box)
                else:
                    try:
                        sys.setprofile(profile)
                        actual = runtime.entry(box)
                    finally:
                        sys.setprofile(None)
                    self.assertEqual(frames, [])
                self.assertEqual(box, [])
                ordinary = model.host(list(inputs))
                self.assertNotEqual(actual[0].data_ptr(), ordinary[0].data_ptr())
                self.assertEqual(actual, ordinary)
                self.assertEqual(actual, model.reference(*inputs), atol=2e-2, rtol=2e-3)
                self.assertEqual(actual[0].untyped_storage()._cdata, actual[1].untyped_storage()._cdata)
                self.assertEqual(tuple(actual[0].shape), (1, 8 * inputs[0], 128))
                held.append((actual, tuple(value.clone() for value in actual)))
        self.assertEqual(ordinary_calls, [16])
        self.assertEqual(len(runtime.variants), 1)
        self.assertEqual(len(captures), 1)
        self.assertEqual(owner._capture.calls, 1)
        self.assertTrue(hasattr(prototype.GEMM, "a_major_mode"))
        variant, = runtime.variants
        variant.program.check()
        call, = (event for event in variant.program.events if type(event) is CuTeCall)
        self.assertIs(call.receipt.invocation.compilation.selected, owner.selected)
        self.assertEqual(len(call.pointers), 3)
        for inputs in samples:
            numeric = _NumericProgram(variant.program, inputs)
            grid = tuple(numeric.values[numeric.add(axis)] for axis in call.bound.grid)
            self.assertEqual(grid, ((8 * inputs[0] + 127) // 128, 1, 1))
            offsets = tuple(numeric.values[numeric.add(pointer.byte_offset)] for pointer in call.pointers)
            self.assertEqual(offsets, (2048 * inputs[0], 32768, 0))
        runtime.close()
        self.assertTrue(call.receipt.closed)
        self.assertEqual(owner._native_borrows, set())
        for actual, expected in held:
            self.assertEqual(actual, expected)
        add.close()
        owner.close()
        print("UPSTREAM_GEMM_RESULT=" + json.dumps({"accepted": True,

            "ordinary_miss_calls": len(ordinary_calls), "ordinary_compilations": 1,
            "native_hits": 4, "ordinary_references": len(samples), "captures": captures,
            "dynamic_m": [8 * inputs[0] for inputs in samples], "python_frames_on_native_hits": frames,
            "held_outputs_after_close": True}, sort_keys=True), flush=True)


instantiate_device_type_tests(TestUpstreamCuTeGemm, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
