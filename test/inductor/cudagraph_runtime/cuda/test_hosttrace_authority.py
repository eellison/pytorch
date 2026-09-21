# Owner(s): ["module: inductor"]

"""The runtime team's captured-kernel-authority properties (their
test_cuda_host_trace_authority.py and TestCudaHostTraceReconciliation) stated against
this line's evidence: the tape's launch record is read from the ORIGINAL capture's
kernel node (function symbol, parameter layout, block, shared bytes, the parameter
bytes), the kernel module the lowering builds resolves that record, and the
preparation capture is associated node by node against it."""

import gc
import struct
import weakref

import torch
from torch._inductor.runtime._cudagraph import direct_hosttrace
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.cuda._utils import _check_cuda_bindings
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


def layer_norm(x, shape, weight, bias, eps):
    return torch.ops.aten.native_layer_norm.default(x, shape, weight, bias, eps)


class TestHostTraceAuthority(TestCase):
    def _capture(self, fn, args, device):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            outputs = fn(*args)
        nodes = graph._inspect_captured_kernel_nodes(())[2]
        kernels = graph._inspect_captured_kernel_nodes(nodes)[3]
        return graph, outputs, kernels

    @dtypes(torch.float32, torch.bfloat16)
    @parametrize("rows", (1, 8))
    def test_captured_identity_and_parameter_bytes(self, device, dtype, rows):
        from cuda.bindings import driver

        from torch.cuda import _host_trace as ht

        n = 4096
        x = torch.randn(rows, n, device=device, dtype=dtype)
        weight = torch.randn(n, device=device, dtype=dtype)
        bias = torch.randn(n, device=device, dtype=dtype)
        args = (x, [n], weight, bias, 0.125)
        tape = ht.trace(layer_norm, args)
        self.assertEqual(len(tape.launches), 1)
        launch = tape.launches[0]
        layout = tuple(tuple(row) for row in launch["param_layout"])
        # the lowering's kernel module: the host symbol resolved to the driver function
        lowered = direct_hosttrace.lower_tape(tape)
        (call,) = lowered.calls
        module = call.module
        self.assertGreater(module.function, 0)
        self.assertEqual(module.parameter_layout, layout)
        self.assertEqual(module.block, tuple(launch["block"]))
        self.assertEqual(module.shared, launch["smem"])
        self.assertEqual(module.device_index, x.device.index)
        with torch.cuda.device(device):
            self.assertEqual(
                module.context, int(_check_cuda_bindings(driver.cuCtxGetCurrent()))
            )
        # the layout is the driver's parameter table of that function
        for index, entry in enumerate(layout):
            self.assertEqual(
                tuple(
                    _check_cuda_bindings(
                        driver.cuFuncGetParamInfo(module.function, index)
                    )
                ),
                entry,
            )
        end = driver.cuFuncGetParamInfo(module.function, len(layout))
        self.assertEqual(end[0], driver.CUresult.CUDA_ERROR_INVALID_VALUE)
        # the record's fields lie inside one parameter each; float constants carry
        # their exact bytes
        image = launch["hint_image"]
        for field in launch["params"]:
            matches = [
                (index, field["offset"] - offset)
                for index, (offset, size) in enumerate(layout)
                if offset <= field["offset"]
                and field["offset"] + field["size"] <= offset + size
            ]
            self.assertEqual(len(matches), 1)
            if field["kind"] in ("f32", "f64") and not isinstance(
                field["value"], (torch.SymInt, torch.SymFloat, torch.SymBool)
            ):
                kind = "f" if field["kind"] == "f32" else "d"
                self.assertEqual(
                    bytes(image[field["offset"] : field["offset"] + field["size"]]),
                    struct.pack(kind, field["value"]),
                )
        # a fresh capture of the same call: the node the recorder read the record
        # from has this function, layout, block, shared bytes and parameter bytes
        graph, outputs, kernels = self._capture(layer_norm, args, device)
        self.assertEqual(len(kernels), 1)
        captured = kernels[0]
        self.assertEqual(module.function, captured[1])
        if captured[2]:
            self.assertEqual(
                module.function,
                int(_check_cuda_bindings(driver.cuKernelGetFunction(captured[2]))),
            )
        self.assertEqual(
            layout, tuple((offset, size) for offset, size, _ in captured[8])
        )
        self.assertEqual(module.block, captured[5])
        self.assertEqual(module.shared, captured[6])
        # the non-pointer bytes of the record equal the fresh node's (pointers name
        # this capture's own allocations; the tape binds them by symbol instead)
        slots = {offset: (size, value) for offset, size, value in captured[8]}
        for field in launch["params"]:
            if field["kind"] in ("ptr", "rng"):
                continue
            start, width = field["offset"], field["size"]
            base = max(o for o in slots if o <= start)
            size, value = slots[base]
            self.assertLessEqual(start + width, base + size)
            self.assertEqual(
                bytes(image[start : start + width]),
                value[start - base : start - base + width],
            )
        del outputs, graph

    def test_prepared_entry_retains_the_tape_evidence(self, device):
        from torch.cuda import _host_trace as ht

        x = torch.randn(8, 4096, device=device)
        args = (x, [4096], None, None, 1e-5)
        tape = ht.trace(layer_norm, args)
        ref = weakref.ref(tape)
        launch = tape.launches[0]
        function, layout, image = (
            launch["func"],
            tuple(launch["param_layout"]),
            bytes(launch["hint_image"]),
        )
        lowered = direct_hosttrace.lower_tape(tape)
        entry = direct_hosttrace.prepare_hosttrace(lowered, args)
        del tape, launch
        gc.collect()
        # the entry retains the lowered tape and the tape: the record outlives the caller
        self.assertIsNotNone(ref())
        self.assertEqual(ref().launches[0]["func"], function)
        self.assertEqual(tuple(ref().launches[0]["param_layout"]), layout)
        self.assertEqual(bytes(ref().launches[0]["hint_image"]), image)
        other = ht.trace(layer_norm, args)
        self.assertEqual(other.launches[0]["func"], function)
        entry.close()

    def test_a_recorded_launch_whose_bytes_differ_from_the_node_is_refused(
        self, device
    ):
        # the association of a preparation capture: every recorded launch's argument
        # bytes must equal the captured node's, byte for byte, or the capture is refused
        from torch._inductor.runtime.cudagraph_launch_association import (
            associate_kernel_launches,
            RecordedKernelLaunch,
        )

        x = torch.randn(8, 4096, device=device)
        args = (x, [4096], None, None, 1e-5)
        graph, outputs, kernels = self._capture(layer_norm, args, device)
        nodes = graph._inspect_captured_kernel_nodes(())[2]
        self.assertEqual(len(kernels), 1)
        captured = kernels[0]
        before = (1, 1, 1, ())
        after = (1, 1, 1, ((captured[0], bytes(8)),))
        snapshot = graph._inspect_captured_kernel_nodes(nodes)
        snapshot = (1, 1, snapshot[2], snapshot[3])
        exact = tuple(value for _, _, value in captured[8])
        launch = RecordedKernelLaunch(0, before, after, captured[1], exact)
        (associated,) = associate_kernel_launches((launch,), snapshot)
        self.assertEqual(associated.snapshot[0], captured[0])
        wrong = (bytes(len(exact[0])), *exact[1:])
        launch = RecordedKernelLaunch(0, before, after, captured[1], wrong)
        with self.assertRaisesRegex(
            UnsupportedCapture, "differs from the selected launch"
        ):
            associate_kernel_launches((launch,), snapshot)
        launch = RecordedKernelLaunch(0, before, after, captured[1] + 1, exact)
        with self.assertRaisesRegex(
            UnsupportedCapture, "Captured function does not match"
        ):
            associate_kernel_launches((launch,), snapshot)
        del outputs, graph


instantiate_device_type_tests(TestHostTraceAuthority, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
