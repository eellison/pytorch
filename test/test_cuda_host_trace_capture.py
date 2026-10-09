# Owner(s): ["module: cuda graphs"]

import ctypes
import dataclasses
import unittest
from unittest import mock

import torch
import torch.cuda._host_trace_capture as host_trace_capture
from torch.cuda._host_trace import Declined
from torch.cuda._host_trace_capture import (
    _neighbors,
    _verify,
    capture_kernel_nodes,
    capture_tape,
    explicit_attributes,
    graph_node_handles,
    graph_nodes,
    instantiate_form,
    launch_images,
    launch_slots,
    MemsetNode,
)
from torch.cuda._host_trace_launch import KernelLaunch
from torch.cuda._host_trace_lower_tape import lower_tape
from torch.cuda._host_trace_opaque import OpaqueBinding, OpaqueKernel
from torch.cuda._host_trace_tape import (
    _ALLOC_SHIFT,
    _EAGER_TAG,
    _output_records,
    _symbolic_run,
    _Trace,
    current_trace,
    Tape,
    trace,
)
from torch.cuda._host_trace_triton import TritonABI, TritonArg
from torch.cuda._utils import _check_cuda_bindings
from torch.testing._internal.common_utils import (
    requires_cuda_python_bindings,
    run_tests,
    TEST_CUDA,
    TestCase,
)
from torch.utils._triton import has_triton


if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _add(x_ptr, y_ptr, n, s, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        m = i < n
        tl.store(y_ptr + i, tl.load(x_ptr + i, mask=m) + s, mask=m)

    @triton.jit
    def _add_pdl(x_ptr, y_ptr, n, s, B: tl.constexpr):
        tl.extra.cuda.gdc_wait()
        i = tl.program_id(0) * B + tl.arange(0, B)
        m = i < n
        tl.store(y_ptr + i, tl.load(x_ptr + i, mask=m) + s, mask=m)


# x, y, n, m, then the launcher's two scratch pointers
_ABI = TritonABI(
    (
        TritonArg("x", "*fp32", 0),
        TritonArg("y", "*fp32", 1),
        TritonArg("n", "i32", 2),
        TritonArg("m", "i32", 3),
    ),
    6,
    4,
    0,
)
_LAYOUT = ((0, 8), (8, 8), (16, 4), (20, 4), (24, 8), (32, 8))


def _cpu_tape(fn, *args):
    # a tape of the symbolic run on the CPU, without trace()'s capture
    tr = _Trace(torch.device("cpu"))
    positions = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor)]
    ints = [i for i, a in enumerate(args) if type(a) is int]
    out, traced = _symbolic_run(tr, fn, args, positions, ints)
    kind, outputs = _output_records(out, traced, positions)
    return Tape(tr, args, outputs, kind, None)


def _q(v):
    return v.to_bytes(8, "little", signed=True)


def _node_bytes(node, layout):
    from cuda.bindings import driver

    p = driver.cuGraphKernelNodeGetParams(node)[1]
    held = (ctypes.c_void_p * len(layout)).from_address(int(p.kernelParams))
    return tuple(ctypes.string_at(ptr, size) for ptr, (_, size) in zip(held, layout))


class TestLaunchImages(TestCase):
    def test_slot_bytes(self):
        def fn(a, n):
            out = torch.empty(n)
            slots = (a.data_ptr() + 8, out.data_ptr() + 4, n, -n, 0, 0)
            roots = (a._root, out._root)
            launch = KernelLaunch(
                "k", None, _ABI, _LAYOUT, (1, 1, 1), (128, 1, 1), 0, slots, roots
            )
            current_trace().record_launch(launch)
            return out

        a = torch.randn(8)
        lowered = lower_tape(_cpu_tape(fn, a, 5))
        values = lowered.evaluate((a, 5))
        launch = lowered.launches[0]
        images = launch_images(launch, launch_slots(launch, values, [0x10000]))
        q = lambda v: v.to_bytes(8, "little", signed=True)  # noqa: E731
        want = (
            q(a.data_ptr() + 8),
            q(0x10000 + 4),
            (5).to_bytes(4, "little"),
            (-5).to_bytes(4, "little", signed=True),
            q(0),
            q(0),
        )
        self.assertEqual(images, want)


def _capture(lowered):
    """capture_tape over buffers allocated for the tape's allocations."""
    values = lowered.evaluate(lowered.tape.args)
    buffers = [
        torch.empty_strided(
            [values[r] for r in a.sizes],
            [values[r] for r in a.strides],
            dtype=a.dtype,
            device=lowered.tape.device,
        )
        for a in lowered.allocations
    ]
    return capture_tape(lowered, [b.data_ptr() for b in buffers]), buffers


@unittest.skipIf(not TEST_CUDA or not has_triton(), "requires CUDA and Triton")
@requires_cuda_python_bindings
class TestCaptureTape(TestCase):
    def test_one_launch(self):
        def fn(x, n):
            y = torch.empty_like(x)
            _add[(triton.cdiv(n, 128),)](x, y, n, 3, B=128)
            return y

        x = torch.randn(1000, device="cuda")
        captured, buffers = _capture(lower_tape(trace(fn, (x, 1000))))
        (launch,) = captured.launches
        self.assertEqual(launch.launch.seq, captured.lowered.launches[0].seq)
        captured.segments[0].graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(buffers[0], x + 3)

    def test_chained_launches_in_order(self):
        def fn(x, n):
            y, z = torch.empty_like(x), torch.empty_like(x)
            grid = (triton.cdiv(n, 128),)
            _add[grid](x, y, n, 3, B=128)
            _add[grid](y, z, n, 4, B=128)
            return z

        x = torch.randn(1000, device="cuda")
        captured, buffers = _capture(lower_tape(trace(fn, (x, 1000))))
        self.assertEqual(len(captured.launches), 2)
        self.assertNotEqual(captured.launches[0].node, captured.launches[1].node)
        y, z = buffers
        for c, buffer in zip(captured.launches, (y, z)):
            self.assertEqual(
                _node_bytes(c.node, c.launch.launch.layout)[1],
                buffer.data_ptr().to_bytes(8, "little"),
            )
        captured.segments[0].graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(z, x + 7)

    def test_bytes_match_tritons_own_launch(self):
        def fn(x, y, n):
            _add[(triton.cdiv(n, 128),)](x, y, n, 3, B=128)
            return y

        x, y = torch.randn(1000, device="cuda"), torch.empty(1000, device="cuda")
        captured, _ = _capture(lower_tape(trace(fn, (x, y, 1000))))
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            fn(x, y, 1000)
        from cuda.bindings import runtime

        _, (node,), _ = runtime.cudaGraphGetNodes(graph.raw_cuda_graph(), numNodes=1)
        (ours,) = captured.launches
        layout = ours.launch.launch.layout
        self.assertEqual(_node_bytes(ours.node, layout), _node_bytes(int(node), layout))

    def test_a_run_after_an_eager_call(self):
        def fn(x, n):
            y = torch.empty_like(x)
            _add[(triton.cdiv(n, 128),)](x.cumsum(0), y, n, 3, B=128)
            return y

        x = torch.randn(1000, device="cuda")
        captured, buffers = _capture(lower_tape(trace(fn, (x, 1000))))
        _, run = captured.lowered.steps
        (segment,) = captured.segments
        self.assertEqual(run, range(1))
        # verified against the eager output's placeholder, which faults until
        # a replay patches it
        (c,) = segment.launches
        held = _node_bytes(c.node, c.launch.launch.layout)[0]
        self.assertEqual(held, (_EAGER_TAG | (1 << _ALLOC_SHIFT)).to_bytes(8, "little"))

    def test_an_explicit_unit_cluster_is_launched(self):
        # an explicit cluster of (1, 1, 1) is an attribute a plain launch
        # lacks: a kernel built for clusters traps without it
        from cuda.bindings import driver

        def fn(x, n):
            y = torch.empty_like(x)
            _add[(triton.cdiv(n, 128),)](x, y, n, 3, B=128)
            return y

        x = torch.randn(1000, device="cuda")
        lowered = lower_tape(trace(fn, (x, 1000)))
        lo = lowered.launches[0]
        unit = ((driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, (1, 1, 1)),)
        # a record Triton never launches (its launcher sets no cluster): no witness
        explicit = dataclasses.replace(lo, launch=dataclasses.replace(lo.launch, attributes=unit, witness=None))
        for launch, attributes in ((lo, ()), (explicit, unit)):
            captured, buffers = _capture(dataclasses.replace(lowered, launches=(launch,)))
            graph = captured.segments[0].graph.raw_cuda_graph()
            (node,) = graph_nodes(graph)
            self.assertEqual(explicit_attributes(node), attributes)
            self.assertEqual(captured.launches[0].launch.launch.cluster, (1, 1, 1))
        values = lowered.evaluate(lowered.tape.args)
        nodes, y = [c.node for c in captured.launches], buffers[0].data_ptr()
        with self.assertRaisesRegex(Declined, "has launch attributes"):
            _verify(graph, nodes, lowered.launches, values, [y], 0)

    def test_a_form_changes_the_cluster_size(self):
        # a grid of 3 that cluster 2 does not divide: the arm's cluster and
        # grid both change on the clone
        from cuda.bindings import driver

        def fn(x, n):
            y = torch.empty_like(x)
            _add[(triton.cdiv(n, 128),)](x, y, n, 3, B=128)
            return y

        x = torch.randn(384, device="cuda")
        lowered = lower_tape(trace(fn, (x, 384)))
        lo = lowered.launches[0]
        cluster = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        unit = dataclasses.replace(lo, launch=dataclasses.replace(lo.launch, attributes=((cluster, (1, 1, 1)),), witness=None))
        captured, buffers = _capture(dataclasses.replace(lowered, launches=(unit,)))
        graph = captured.segments[0].graph.raw_cuda_graph()
        exec_, clone, _ = instantiate_form(graph, [(captured.launches[0].node, {cluster: (2, 1, 1)}, (4, 1, 1))], [])
        try:
            (formed,) = graph_nodes(clone)
            self.assertEqual((formed.grid, explicit_attributes(formed)), ((4, 1, 1), ((cluster, (2, 1, 1)),)))
            buffers[0].zero_()
            stream = torch.cuda.current_stream()
            _check_cuda_bindings(driver.cuGraphLaunch(exec_, stream.cuda_stream))
            stream.synchronize()
            self.assertEqual(buffers[0], x + 3)
        finally:
            _check_cuda_bindings(driver.cuGraphExecDestroy(exec_))
            _check_cuda_bindings(driver.cuGraphDestroy(clone))

    def test_a_piece_keeps_a_programmatic_edge_behind_the_switch(self):
        from cuda.bindings import driver

        pdl = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        x, z = torch.randn(1000, device="cuda"), torch.empty(1000, device="cuda")

        def fn():
            _add_pdl[(8,)](x + 1, z, 1000, 3, B=128, launch_pdl=True)

        fn()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            fn()
        raw = graph.raw_cuda_graph()
        (handle,) = [h for h in graph_node_handles(raw) if _neighbors(driver.cuGraphNodeGetDependencies, h)[0]]
        _, n = graph_nodes(raw)
        self.assertEqual(n.attribute("PROGRAMMATIC_STREAM_SERIALIZATION"), 1)
        kernel = OpaqueKernel(n.function, n.grid, n.block, n.smem, (*explicit_attributes(n), (pdl, 1)), n.layout, n.images, ())
        for keep in (False, True):
            with mock.patch.object(host_trace_capture, "keep_programmatic_edges", keep):
                exec_, clone, _ = instantiate_form(raw, [], [([handle], OpaqueBinding((kernel,), ()))])
            try:
                _, formed = graph_nodes(clone)
                self.assertEqual(formed.attribute("PROGRAMMATIC_STREAM_SERIALIZATION"), int(keep))
                z.zero_()
                stream = torch.cuda.current_stream()
                _check_cuda_bindings(driver.cuGraphLaunch(exec_, stream.cuda_stream))
                stream.synchronize()
                self.assertEqual(z, x + 4)
            finally:
                _check_cuda_bindings(driver.cuGraphExecDestroy(exec_))
                _check_cuda_bindings(driver.cuGraphDestroy(clone))

    def test_no_launches(self):
        lowered = lower_tape(trace(lambda x: torch.empty_like(x), (torch.randn(4, device="cuda"),)))
        captured, buffers = _capture(lowered)
        self.assertEqual(captured.launches, ())
        self.assertEqual(buffers[0].shape, (4,))

    def test_declines(self):
        def fn(x, n):
            y = torch.empty_like(x)
            _add[(triton.cdiv(n, 128),)](x, y, n, 3, B=128)
            return y

        x = torch.randn(1000, device="cuda")
        lowered = lower_tape(trace(fn, (x, 1000)))
        lo = lowered.launches[0]
        wrong = dataclasses.replace(
            lo, launch=dataclasses.replace(lo.launch, grid=(9, 1, 1))
        )
        wrong_tape = dataclasses.replace(lowered, launches=(wrong,))
        msg = "launch dimension 0 is 8 at the traced call; the trace has 9"
        with self.assertRaisesRegex(AssertionError, msg):
            _capture(wrong_tape)
        with (
            mock.patch.object(host_trace_capture, "raise_trace_disagreements", False),
            self.assertLogs(host_trace_capture.log, "WARNING"),
            self.assertRaisesRegex(Declined, msg),
        ):
            _capture(wrong_tape)
        captured, buffers = _capture(lowered)
        values = lowered.evaluate(lowered.tape.args)
        nodes = [c.node for c in captured.launches]
        graph = captured.segments[0].graph.raw_cuda_graph()
        launches, y = lowered.launches, buffers[0].data_ptr()
        with self.assertRaisesRegex(Declined, "holds other bytes in slot 1"):
            _verify(graph, nodes, launches, values, [y + 4], 0)
        with self.assertRaisesRegex(Declined, "nodes the tape did not launch"):
            _verify(graph, [], launches, values, [y], 0)


@unittest.skipIf(not TEST_CUDA or not has_triton(), "requires CUDA and Triton")
@requires_cuda_python_bindings
class TestCaptureKernelNodes(TestCase):
    def test_nodes_in_stream_order(self):
        from cuda.bindings import runtime

        x, y = torch.randn(1000, device="cuda"), torch.empty(1000, device="cuda")

        def fn(stream):
            _add[(8,)](x, y, 1000, 1, B=128)
            runtime.cudaMemsetAsync(y.data_ptr(), 0, 16, stream.cuda_stream)
            _add[(4,)](y, x, 500, 2, B=128)

        first, memset, second = capture_kernel_nodes(fn)
        self.assertEqual((first.grid, second.grid), ((8, 1, 1), (4, 1, 1)))
        self.assertEqual(first.images[:3], (_q(x.data_ptr()), _q(y.data_ptr()), (1000).to_bytes(4, "little")))
        self.assertEqual(explicit_attributes(first), ())
        self.assertEqual(first.attribute("PROGRAMMATIC_STREAM_SERIALIZATION"), 0)
        self.assertIsInstance(memset, MemsetNode)
        self.assertEqual((memset.dst, memset.value), (y.data_ptr(), 0))
        self.assertEqual(memset.width * memset.element_size, 16)

    def test_skips_a_capturing_stream(self):
        streams = []
        outer = torch.cuda.CUDAGraph()
        with torch.cuda.graph(outer):
            capturing = torch.cuda.current_stream()
            with torch.cuda.stream(torch.cuda.Stream()):
                self.assertNotEqual(torch.cuda.current_stream(), capturing)
                # the pool's 32 streams wrap around onto the capturing one
                for _ in range(40):
                    capture_kernel_nodes(streams.append)
        self.assertNotIn(capturing, streams)


def setUpModule():
    from torch.cuda import _host_trace_hint_audit

    _host_trace_hint_audit.enable_for_tests()
    host_trace_capture.raise_trace_disagreements = True
    torch.cuda._host_trace.raise_unexpected = True


if __name__ == "__main__":
    run_tests()
