"""Replay unchanged TMA GEMM with independent symbolic M, N and K."""

from tempfile import TemporaryDirectory
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

import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cutlass.torch import get_leading_dim
from cuda.bindings import driver
from torch._inductor.runtime._cudagraph._compiler.entry_signature import SignaturePolicy
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import AllocateEvent, InputContract, IntegerRange, NormalizeEvent, TensorInput
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner import ObservedOrdinaryEntry
from torch._inductor.runtime._cudagraph._compiler.python_entry import PythonEntry
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch._inductor.runtime._cudagraph.direct_cute import DirectCuTe
import torch._inductor.runtime._cudagraph.direct_host as direct_host
import torch._inductor.runtime._cudagraph.replay as replay
import torch
from torch.cuda._utils import _check_cuda_bindings
from torch._inductor.runtime.cudagraph_arg_mapping import (
    ExpressionSource, InputSource, IntegerSource, IntExpr, ParameterSource, PointerSource,
)
from torch._inductor.runtime.cudagraph_boxed_replay import _ParameterProgram, _PhysicalCall
from torch._inductor.runtime.cudagraph_launch_association import associate_kernel_launches
from torch.testing._internal.common_cuda import tf32_off
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, recover_orig_fp32_precision, run_tests, TestCase


UPSTREAM = (REPO_ROOT / "third_party/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py")
upstream = load_source("_dynamic_dimensions_upstream_tma_gemm", UPSTREAM)
GEMM = None
TMA = None
copy_if_misaligned = torch._C._dynamo.guards.copy_if_misaligned
DIMENSIONS = ((128, 128, 128), (256, 128, 128), (128, 256, 128),
              (128, 128, 256), (384, 384, 384), (128, 128, 128))


@cute.jit
def invocation(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream):
    GEMM(a, b, c, stream)


def convert_arguments(a, b, c):
    return tuple(from_dlpack(value, assumed_align=16).mark_layout_dynamic(leading_dim=get_leading_dim(value))
                 for value in (a, b, c))


def host(box):
    m, n, k, a, b = box
    box.clear()
    a = copy_if_misaligned(a)
    b = copy_if_misaligned(b)
    output = torch.empty_strided((m, n, 1), (n, 1, m * n), dtype=torch.float32, device=a.device)
    TMA(a, b, output)
    return (output,)


def boxed_origins(values):
    pending, result = list(values), set()
    while pending:
        value = pending.pop()
        if type(value) is IntExpr:
            if value.op == "boxed":
                result.add(value.value)
            pending.extend(value.args)
        elif type(value) is ParameterSource:
            pending.extend((*value.args, value.value))
        elif type(value) is PointerSource:
            pending.append(value.byte_offset)
        elif type(value) is ExpressionSource:
            pending.append(value.expression)
    return result


class TestTmaDynamicDimensions(TestCase):
    @recover_orig_fp32_precision
    @tf32_off()
    @parametrize("dimensions", (DIMENSIONS,))
    def test_independent_dimensions(self, device, dimensions):
        global GEMM, TMA

        if torch.cuda.get_device_capability(device)[0] < 10:
            self.skipTest("The unchanged upstream GEMM targets Blackwell")
        artifacts = Path(self.enterContext(TemporaryDirectory()))
        report = {"accepted": False, "phase": "setup",
                  "ordinary_miss_calls": 0, "trace_calls": 0, "ordinary_references": 0,
                  "native_hits": 0, "guard_misses": 0, "python_frames_on_hits": [],
                  "samples": [], "captures": []}
        try:
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.device(device), torch.cuda.stream(stream):
                GEMM = upstream.DenseGemmKernel(cutlass.Float32, False, (128, 128), (1, 1), True)
                owner = ObservedOrdinaryEntry(PythonEntry(invocation), GEMM.kernel,
                    policy=SignaturePolicy(32, 64, 16, "stream"), conversion=convert_arguments)
                self.addCleanup(owner.close)
                TMA = DirectCuTe(owner)
                m, n, k = (IntExpr("boxed", index) for index in range(3))
                mk = IntExpr("multiply", None, (m, k))
                nk = IntExpr("multiply", None, (n, k))
                contract = InputContract(("integer", "integer", "integer", "tensor", "tensor"),
                    (TensorInput(3, torch.float16, (m, k, 1), (k, 1, mk)),
                     TensorInput(4, torch.float16, (n, k, 1), (k, 1, nk))),
                    tuple(IntegerRange(index, 128, 384) for index in range(3)),
                    device_index=torch.cuda.current_device())
                runtime = direct_host.DirectHost(host, contract)
                self.addCleanup(runtime.close)
                self.addCleanup(stream.synchronize)
                self.enterContext(mock.patch.object(torch, "compile", side_effect=AssertionError("No host compiler")))
                self.enterContext(mock.patch.object(torch.func, "functionalize", side_effect=AssertionError("No functionalization")))
                samples = [(rows, columns, reduction,
                            torch.randn((1, rows, reduction), dtype=torch.float16, device=device).permute(1, 2, 0),
                            torch.randn((1, columns, reduction), dtype=torch.float16, device=device).permute(1, 2, 0))
                           for rows, columns, reduction in dimensions for _ in range(2)]
                self.assertEqual(len({a.data_ptr() for _, _, _, a, _ in samples}), len(samples))
                self.assertEqual(len({b.data_ptr() for _, _, _, _, b in samples}), len(samples))
                make_replay, observe_direct, trace_host = replay._make_replay, direct_host._observe_direct, direct_host.trace_host
                captured_nodes = []

                def observe(*args, **kwargs):
                    report["ordinary_miss_calls"] += 1
                    return observe_direct(*args, **kwargs)

                def trace(*args, **kwargs):
                    report["trace_calls"] += 1
                    return trace_host(*args, **kwargs)

                def cluster_of(node):
                    value = _check_cuda_bindings(driver.cuGraphKernelNodeGetAttribute(
                        node, driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION))
                    return value.clusterDim.x, value.clusterDim.y, value.clusterDim.z

                def capture(graph, input_count, allocations, outputs, copies, calls, launches, buffers, capture_stream, **kwargs):
                    report["phase"] = "capture_correspondence"
                    self.assertEqual(input_count, 5)
                    self.assertEqual(len(allocations), 1)
                    self.assertEqual(tuple(copies), (3, 4))
                    call, = calls
                    launch, = launches
                    self.assertIs(type(call), _PhysicalCall)
                    self.assertEqual(call.module.cluster, (1, 1, 1))
                    self.assertEqual(call.module.site.cluster, call.module.cluster)
                    nodes = tuple(item.after[3][0][0] for item in launches)
                    associated, = associate_kernel_launches(tuple(launches), graph._inspect_captured_kernel_nodes(nodes))
                    self.assertEqual(cluster_of(nodes[0]), call.module.cluster)
                    self.assertEqual(associated.snapshot[5], call.module.block)
                    numeric, inputs = kwargs["numeric"], kwargs["capture_inputs"]
                    parameters = _ParameterProgram(numeric, input_count,
                        {layout.source: input_count + index for index, layout in enumerate(allocations)})
                    late = [(field, parameters.add(field.source)) for field in call.fields
                            if type(field.source) is ParameterSource]
                    self.assertTrue(late)
                    parameter_origins = boxed_origins(field.source for field in call.fields)
                    grid_origins = boxed_origins(call.grid)
                    self.assertEqual(parameter_origins, {0, 1, 2})
                    self.assertTrue({0, 1}.issubset(grid_origins))
                    self.assertEqual(set(numeric.integer_indices), {0, 1, 2})
                    words = parameters.evaluate(inputs, buffers)
                    for field, index in late:
                        payload = struct.pack("i" if field.source.width == 32 else "q", words[index])
                        self.assertEqual(launch.argument_bytes[field.parameter][field.byte_offset:field.byte_offset + len(payload)], payload)
                    for field in call.fields:
                        source = field.source
                        if type(source) is ParameterSource:
                            continue
                        if type(source) is PointerSource:
                            root = inputs[source.root.index] if type(source.root) is InputSource else buffers[source.root]
                            payload = struct.pack("P", root.data_ptr() + numeric.values[numeric.add(source.byte_offset)])
                        else:
                            self.assertIn(type(source), (IntegerSource, ExpressionSource))
                            value = source.value if type(source) is IntegerSource else numeric.values[numeric.add(source.expression)]
                            payload = struct.pack("i" if field.kind == "i32" else "q", value)
                        self.assertEqual(launch.argument_bytes[field.parameter][field.byte_offset:field.byte_offset + len(payload)], payload)
                    for parameter, offset, data in call.constants:
                        self.assertEqual(launch.argument_bytes[parameter][offset:offset + len(data)], data)
                    self.assertTrue(call.undefined)
                    for parameter, offset, size in call.undefined:
                        self.assertEqual(launch.argument_bytes[parameter][offset:offset + size], bytes(size))
                    self.assertEqual(set(parameters.roots.values()), {InputSource(3), InputSource(4), allocations[0].source})
                    attrs = driver.CUfunction_attribute
                    required = {name: int(_check_cuda_bindings(driver.cuFuncGetAttribute(attribute, call.module.function)))
                                for name, attribute in (("width", attrs.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH),
                                    ("height", attrs.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT),
                                    ("depth", attrs.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH),
                                    ("must_be_set", attrs.CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET))}
                    report["captures"].append({"cluster": call.module.cluster, "required_cluster": required,
                        "grid": associated.snapshot[4], "shape": tuple(inputs[:3]),
                        "grid_integer_roots": sorted(grid_origins), "parameter_integer_roots": sorted(parameter_origins),
                        "parameter_layout": call.module.parameter_layout, "late_fields": len(late),
                        "undefined_fields": len(call.undefined), "literal_fields": len(call.constants),
                        "block": call.module.block, "shared": call.module.shared})
                    captured_nodes.append(nodes[0])
                    return make_replay(graph, input_count, allocations, outputs, copies, calls, launches, buffers, capture_stream, **kwargs)

                def profile(frame, event, result):
                    if event == "call":
                        frames.append((frame.f_code.co_filename, frame.f_code.co_name))

                held = []
                frames = []
                with mock.patch.object(direct_host, "_observe_direct", observe), \
                     mock.patch.object(direct_host, "trace_host", trace), mock.patch.object(replay, "_make_replay", capture):
                    for index, inputs in enumerate(samples):
                        rows, columns, reduction, a, b = inputs
                        shape = rows, columns, reduction
                        frames = []
                        if index == 0:
                            report["phase"] = "ordinary_warm_and_prepare"
                            box = list(inputs)
                            actual = runtime(box)
                            self.assertEqual(box, [])
                            dispatch = "cold"
                        report["phase"] = f"ordinary_reference_{shape}"
                        ordinary = host(list(inputs))
                        report["ordinary_references"] += 1
                        expected = torch.einsum("mkl,nkl->mnl", a.float(), b.float())
                        self.assertEqual(ordinary[0], expected, atol=2e-3, rtol=2e-4)
                        if index:
                            report["phase"] = f"native_dispatch_{shape}"
                            before = report["ordinary_miss_calls"], report["trace_calls"]
                            box = list(inputs)
                            try:
                                sys.setprofile(profile)
                                actual = runtime.entry(box)
                            finally:
                                sys.setprofile(None)
                            after = report["ordinary_miss_calls"], report["trace_calls"]
                            if before == after:
                                dispatch = "native_hit"
                                report["native_hits"] += 1
                                report["python_frames_on_hits"].extend(frames)
                                self.assertEqual(frames, [])
                            else:
                                dispatch = "guard_miss"
                                self.assertEqual(after, (before[0] + 1, before[1] + 1))
                                report["guard_misses"] += 1
                            self.assertEqual(box, [])
                            if index % 2 or shape == dimensions[0]:
                                self.assertEqual(dispatch, "native_hit")
                        self.assertEqual(actual, ordinary, atol=0, rtol=0)
                        self.assertEqual(actual[0], expected, atol=2e-3, rtol=2e-4)
                        self.assertEqual((tuple(actual[0].shape), actual[0].stride()),
                                         ((rows, columns, 1), (columns, 1, rows * columns)))
                        self.assertNotEqual(actual[0].data_ptr(), ordinary[0].data_ptr())
                        for node in captured_nodes:
                            self.assertEqual(cluster_of(node), (1, 1, 1))
                        held.append((actual, actual[0].clone()))
                        report["samples"].append({"m": rows, "n": columns, "k": reduction,
                            "dispatch": dispatch, "python_calls": len(frames),
                            "a_pointer": a.data_ptr(), "b_pointer": b.data_ptr(),
                            "output_pointer": actual[0].data_ptr()})
                variant_count = len(runtime.variants)
                self.assertEqual(report["ordinary_miss_calls"], variant_count)
                self.assertEqual(report["trace_calls"], variant_count)
                self.assertEqual(report["guard_misses"], variant_count - 1)
                self.assertEqual(report["native_hits"], len(samples) - variant_count)
                self.assertGreaterEqual(report["native_hits"], len(dimensions))
                self.assertEqual(report["ordinary_references"], len(samples))
                self.assertEqual(len(report["captures"]), variant_count)
                self.assertEqual(owner._capture.calls, 1)
                receipts = []
                for variant in runtime.variants:
                    variant.program.check()
                    trace = variant.program.guards.trace
                    self.assertEqual({row.boxed_index for row in variant.program.integer_inputs}, {0, 1, 2})
                    self.assertEqual(sum(type(event) is AllocateEvent for event in trace.events), 1)
                    self.assertEqual(sum(type(event) is NormalizeEvent for event in trace.events), 2)
                    call, = (event for event in variant.program.events if type(event) is CuTeCall)
                    self.assertIs(call.receipt.invocation.compilation.selected, owner.selected)
                    receipts.append(call.receipt)
                self.assertEqual(len({actual[0].data_ptr() for actual, _ in held}), len(held))
                runtime.close()
                self.assertTrue(all(receipt.closed for receipt in receipts))
                self.assertEqual(owner._native_borrows, set())
                owner.close()
                for actual, expected in held:
                    self.assertEqual(actual[0], expected)
                report.update(accepted=True, phase="complete", variants=variant_count, ordinary_compilations=1,
                              held_outputs_after_close=True)
        except BaseException as error:
            report["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            (artifacts / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print("TMA_DYNAMIC_DIMENSIONS_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


instantiate_device_type_tests(TestTmaDynamicDimensions, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
