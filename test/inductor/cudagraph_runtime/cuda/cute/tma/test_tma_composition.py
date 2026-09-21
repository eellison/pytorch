"""Compose two unchanged TMA GEMMs around a user Triton cast and activation."""

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
from cuda.bindings import driver
from torch._inductor.runtime._cudagraph._compiler.entry_signature import SignaturePolicy
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import AllocateEvent, InputContract, IntegerRange, NormalizeEvent, TensorInput
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner import ObservedOrdinaryEntry
from torch._inductor.runtime._cudagraph._compiler.python_entry import PythonEntry
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch._inductor.runtime._cudagraph.direct_cute import DirectCuTe
from torch._inductor.runtime._cudagraph.direct_triton import DirectTriton
from torch._inductor.runtime._cudagraph.frontend import DirectKernelCall
import torch._inductor.runtime._cudagraph.direct_host as direct_host
import torch._inductor.runtime._cudagraph.replay as replay
import torch
from torch.cuda._utils import _check_cuda_bindings
from torch._inductor.runtime.cudagraph_arg_mapping import InputSource, IntExpr, ParameterSource, PointerSource
from torch._inductor.runtime.cudagraph_boxed_replay import _BoundCall, _ParameterProgram, _PhysicalCall
from torch._inductor.runtime.cudagraph_launch_association import associate_kernel_launches
from torch.testing._internal.common_cuda import tf32_off
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import recover_orig_fp32_precision, run_tests, TestCase
import triton
import triton.language as tl


UPSTREAM = (REPO_ROOT / "third_party/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py")
FIXTURE = ROOT / "replay_fixture.py"
upstream = load_source("_composed_upstream_tma_gemm", UPSTREAM)
first_fixture = load_source("_first_tma_composition", FIXTURE)
second_fixture = load_source("_second_tma_composition", FIXTURE)
FIRST = CAST = SECOND = None
copy_if_misaligned = torch._C._dynamo.guards.copy_if_misaligned


@triton.jit(do_not_specialize_on_alignment=["source", "destination"])
def relu_cast(source, destination, count, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + offset, offset < count, other=0)
    tl.store(destination + offset, tl.maximum(value, 0).to(tl.float16), offset < count)


def host(box):
    m, a, b = box
    box.clear()
    a = copy_if_misaligned(a)
    b = copy_if_misaligned(b)
    first = torch.empty_strided((m, 128, 1), (128, 1, 128 * m), dtype=torch.float32, device=a.device)
    FIRST(a, b, first)
    activated = torch.empty_strided((m, 128, 1), (128, 1, 128 * m), dtype=torch.float16, device=a.device)
    count = m * 128
    CAST[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](first, activated, count, BLOCK=128)
    output = torch.empty_strided((m, 128, 1), (128, 1, 128 * m), dtype=torch.float32, device=a.device)
    SECOND(activated, b, output)
    return (output,)


class TestTmaComposition(TestCase):
    @recover_orig_fp32_precision
    @tf32_off()
    def test_two_gemms_and_user_cast(self, device):
        global FIRST, CAST, SECOND

        if torch.cuda.get_device_capability(device)[0] < 10:
            self.skipTest("The unchanged upstream GEMM targets Blackwell")
        artifacts = Path(self.enterContext(TemporaryDirectory()))
        report = {"accepted": False, "phase": "setup",
                  "ordinary_miss_calls": 0, "trace_calls": 0, "ordinary_references": 0,
                  "native_hits": 0, "python_frames_on_hits": [], "samples": [], "captures": []}
        try:
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.device(device), torch.cuda.stream(stream):
                self.enterContext(mock.patch.object(torch, "compile", side_effect=AssertionError("No host compiler")))
                self.enterContext(mock.patch.object(torch.func, "functionalize", side_effect=AssertionError("No functionalization")))
                first_fixture.GEMM = upstream.DenseGemmKernel(cutlass.Float32, False, (128, 128), (1, 1), True)
                second_fixture.GEMM = upstream.DenseGemmKernel(cutlass.Float32, False, (128, 128), (1, 1), True)
                owners = tuple(ObservedOrdinaryEntry(PythonEntry(fixture.invocation), fixture.GEMM.kernel,
                    policy=SignaturePolicy(32, 64, 16, "stream"), conversion=fixture.convert_arguments)
                    for fixture in (first_fixture, second_fixture))
                for owner in owners:
                    self.addCleanup(owner.close)
                FIRST, SECOND = (DirectCuTe(owner) for owner in owners)
                CAST = DirectTriton(relu_cast)
                self.addCleanup(CAST.close)
                m = IntExpr("boxed", 0)
                batch_stride = IntExpr("multiply", None, (IntExpr("constant", 128), m))
                contract = InputContract(("integer", "tensor", "tensor"),
                    (TensorInput(1, torch.float16, (m, 128, 1), (128, 1, batch_stride)),
                     TensorInput(2, torch.float16, (128, 128, 1), (128, 1, 16384))),
                    (IntegerRange(0, 128, 384),), device_index=torch.cuda.current_device())
                runtime = direct_host.DirectHost(host, contract)
                self.addCleanup(runtime.close)
                self.addCleanup(stream.synchronize)
                samples = [(rows, (torch.randn((1, rows, 128), dtype=torch.float16, device=device) * 0.125).permute(1, 2, 0),
                            (torch.randn((1, 128, 128), dtype=torch.float16, device=device) * 0.125).permute(1, 2, 0))
                           for rows in (128, 128, 256, 384, 128)]
                self.assertEqual(len({a.data_ptr() for _, a, _ in samples}), len(samples))
                self.assertEqual(len({b.data_ptr() for _, _, b in samples}), len(samples))
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
                    self.assertEqual((input_count, len(allocations), len(calls), len(launches)), (3, 3, 3, 3))
                    self.assertEqual(tuple(copies), (1, 2))
                    self.assertEqual(tuple(layout.dtype for layout in allocations), (torch.float32, torch.float16, torch.float32))
                    first, pointwise, second = calls
                    self.assertIs(type(first), _PhysicalCall)
                    self.assertIs(type(second), _PhysicalCall)
                    self.assertIs(type(pointwise), _BoundCall)
                    roots = [layout.source for layout in allocations]
                    call_roots = [{pointer.root for field in call.fields
                                   for pointer in (field.source.pointers if type(field.source) is ParameterSource
                                                   else (field.source,) if type(field.source) is PointerSource else ())}
                                  for call in (first, second)]
                    self.assertEqual(call_roots[0], {InputSource(1), InputSource(2), roots[0]})
                    self.assertEqual(call_roots[1], {roots[1], InputSource(2), roots[2]})
                    self.assertEqual(tuple(arg.source.root for arg in pointwise.arguments[:2]), (roots[0], roots[1]))
                    nodes = tuple(launch.after[3][0][0] for launch in launches)
                    associated = associate_kernel_launches(tuple(launches), graph._inspect_captured_kernel_nodes(nodes))
                    self.assertEqual(len(associated), 3)
                    numeric, inputs = kwargs["numeric"], kwargs["capture_inputs"]
                    parameters = _ParameterProgram(numeric, input_count,
                        {layout.source: input_count + index for index, layout in enumerate(allocations)})
                    late = [(index, field, parameters.add(field.source)) for index in (0, 2)
                            for field in calls[index].fields if type(field.source) is ParameterSource]
                    self.assertTrue(late)
                    values = parameters.evaluate(inputs, buffers)
                    for index, field, output in late:
                        payload = struct.pack("i" if field.source.width == 32 else "q", values[output])
                        self.assertEqual(launches[index].argument_bytes[field.parameter][field.byte_offset:field.byte_offset + len(payload)], payload)
                    for index in (0, 2):
                        call, launch = calls[index], launches[index]
                        self.assertEqual(call.module.cluster, (1, 1, 1))
                        self.assertEqual(cluster_of(nodes[index]), call.module.cluster)
                        for parameter, offset, data in call.constants:
                            self.assertEqual(launch.argument_bytes[parameter][offset:offset + len(data)], data)
                        self.assertTrue(call.undefined)
                        for parameter, offset, size in call.undefined:
                            self.assertEqual(launch.argument_bytes[parameter][offset:offset + size], bytes(size))
                    for index, argument in enumerate(pointwise.arguments[:2]):
                        self.assertIs(type(argument.source), PointerSource)
                        pointer = buffers[argument.source.root].data_ptr() + numeric.values[numeric.add(argument.source.byte_offset)]
                        self.assertEqual(launches[1].argument_bytes[index], struct.pack("P", pointer))
                    self.assertEqual(set(parameters.roots.values()), {InputSource(1), InputSource(2), *roots})
                    report["captures"].append({"calls": ["tma", "triton_relu_cast", "tma"],
                        "late_fields": len(late), "parameter_layouts": [first.module.parameter_layout, second.module.parameter_layout],
                        "clusters": [first.module.cluster, second.module.cluster]})
                    captured_nodes.extend((nodes[0], nodes[2]))
                    return make_replay(graph, input_count, allocations, outputs, copies, calls, launches, buffers, capture_stream, **kwargs)

                def profile(frame, event, result):
                    if event == "call":
                        report["python_frames_on_hits"].append((frame.f_code.co_filename, frame.f_code.co_name))

                held = []
                with mock.patch.object(direct_host, "_observe_direct", observe), \
                     mock.patch.object(direct_host, "trace_host", trace), mock.patch.object(replay, "_make_replay", capture):
                    for index, inputs in enumerate(samples):
                        rows, a, b = inputs
                        if index == 0:
                            report["phase"] = "ordinary_warm_and_prepare"
                            box = list(inputs)
                            actual = runtime(box)
                            self.assertEqual(box, [])
                        report["phase"] = f"ordinary_reference_M{rows}"
                        ordinary = host(list(inputs))
                        report["ordinary_references"] += 1
                        first_reference = torch.einsum("mkl,nkl->mnl", a.float(), b.float())
                        activated = first_reference.relu().to(torch.float16)
                        expected = torch.einsum("mkl,nkl->mnl", activated.float(), b.float())
                        self.assertEqual(ordinary[0], expected, atol=4e-3, rtol=5e-4)
                        if index:
                            report["phase"] = f"native_hit_M{rows}"
                            box = list(inputs)
                            try:
                                sys.setprofile(profile)
                                actual = runtime.entry(box)
                            finally:
                                sys.setprofile(None)
                            report["native_hits"] += 1
                            self.assertEqual(report["python_frames_on_hits"], [])
                            self.assertEqual(box, [])
                        self.assertEqual(actual, ordinary, atol=0, rtol=0)
                        self.assertEqual(actual[0], expected, atol=4e-3, rtol=5e-4)
                        self.assertEqual((tuple(actual[0].shape), actual[0].stride()), ((rows, 128, 1), (128, 1, 128 * rows)))
                        self.assertNotEqual(actual[0].data_ptr(), ordinary[0].data_ptr())
                        self.assertEqual(tuple(cluster_of(node) for node in captured_nodes), ((1, 1, 1), (1, 1, 1)))
                        held.append((actual, actual[0].clone()))
                        report["samples"].append({"m": rows, "a_pointer": a.data_ptr(), "b_pointer": b.data_ptr(),
                                                   "output_pointer": actual[0].data_ptr()})
                self.assertEqual((report["ordinary_miss_calls"], report["trace_calls"], report["native_hits"]), (1, 1, 4))
                self.assertEqual(len(report["captures"]), 1)
                self.assertEqual(report["ordinary_references"], 5)
                variant, = runtime.variants
                program = variant.program
                program.check()
                self.assertEqual(tuple(owner._capture.calls for owner in owners), (1, 1))
                self.assertEqual(sum(type(event) is AllocateEvent for event in program.guards.trace.events), 3)
                self.assertEqual(sum(type(event) is NormalizeEvent for event in program.guards.trace.events), 2)
                calls = tuple(event for event in program.events if type(event) in (CuTeCall, DirectKernelCall))
                self.assertEqual(tuple(type(call) for call in calls), (CuTeCall, DirectKernelCall, CuTeCall))
                self.assertIs(calls[0].receipt.invocation.compilation.selected, owners[0].selected)
                self.assertIs(calls[2].receipt.invocation.compilation.selected, owners[1].selected)
                self.assertEqual(len({actual[0].data_ptr() for actual, _ in held}), len(held))
                runtime.close()
                self.assertTrue(all(call.receipt.closed for call in (calls[0], calls[2])))
                self.assertEqual(calls[1].owner.module._graph_borrows, 0)
                CAST.close()
                for owner in owners:
                    self.assertEqual(owner._native_borrows, set())
                    owner.close()
                for actual, expected in held:
                    self.assertEqual(actual[0], expected)
                report.update(accepted=True, phase="complete", variants=1, ordinary_compilations=2,
                              held_outputs_after_close=True)
        except BaseException as error:
            report["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            (artifacts / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print("TMA_COMPOSITION_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


instantiate_device_type_tests(TestTmaComposition, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
