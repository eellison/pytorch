# Owner(s): ["module: inductor"]

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.runtime._cudagraph import direct_hosttrace
from torch._inductor.runtime.cudagraph_launch_association import (
    RecordedGraphNode,
    UnsupportedCapture,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


try:
    from cuda.bindings import runtime
except ImportError:
    runtime = None


@unittest.skipIf(runtime is None, "cuda.bindings required")
@instantiate_parametrized_tests
class TestH2DCaptureHelper(TestCase):
    def setUp(self):
        super().setUp()
        self.stream, self.source, self.destination = 71, 0x10000, 0x20000
        self.byte_count, self.node = 48, 13
        self.before = (1, 5, 7, ((11, bytes(8)),))
        self.after = (1, 5, 7, ((self.node, bytes(8)),))
        self.params = SimpleNamespace(
            srcPtr=SimpleNamespace(ptr=self.source),
            dstPtr=SimpleNamespace(ptr=self.destination),
            extent=SimpleNamespace(width=self.byte_count, height=1, depth=1),
            srcPos=SimpleNamespace(x=0, y=0, z=0),
            dstPos=SimpleNamespace(x=0, y=0, z=0),
            srcArray=0,
            dstArray=0,
            kind=runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        frontier = mock.patch.object(
            torch._C, "_cuda_get_capture_frontier", create=True
        )
        copy = mock.patch.object(
            runtime, "cudaMemcpyAsync", return_value=(runtime.cudaError_t.cudaSuccess,)
        )
        query = mock.patch.object(
            runtime,
            "cudaGraphMemcpyNodeGetParams",
            return_value=(runtime.cudaError_t.cudaSuccess, self.params),
        )
        self.frontier = frontier.start()
        self.copy = copy.start()
        self.query = query.start()
        self.addCleanup(frontier.stop)
        self.addCleanup(copy.stop)
        self.addCleanup(query.stop)

    def capture(self):
        self.frontier.side_effect = [self.before, self.after]
        return direct_hosttrace._capture_h2d_copy(
            self.destination, self.source, self.byte_count, self.stream
        )

    @parametrize("initial", (False, True))
    def test_records_exact_copy_and_ordered_frontier(self, initial):
        if initial:
            self.before = (1, 5, 7, ())
        node, event = self.capture()
        self.assertEqual(node, self.node)
        self.assertEqual(
            event, RecordedGraphNode(self.stream, self.before, self.after, "memcpy")
        )
        self.copy.assert_called_once_with(
            self.destination,
            self.source,
            self.byte_count,
            runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )
        self.query.assert_called_once_with(self.node)
        self.frontier.assert_has_calls([mock.call(self.stream)] * 2)
        self.assertEqual(self.frontier.call_count, 2)

    @parametrize(
        "field",
        (
            "srcPtr.ptr",
            "dstPtr.ptr",
            "extent.width",
            "extent.height",
            "extent.depth",
            "kind",
            "srcArray",
            "dstArray",
            "srcPos.x",
            "srcPos.y",
            "srcPos.z",
            "dstPos.x",
            "dstPos.y",
            "dstPos.z",
        ),
    )
    def test_capture_parameter_mismatch_declines(self, field):
        owner = self.params
        parts = field.split(".")
        for name in parts[:-1]:
            owner = getattr(owner, name)
        value = (
            runtime.cudaMemcpyKind.cudaMemcpyDefault
            if field == "kind"
            else getattr(owner, parts[-1]) + 1
        )
        setattr(owner, parts[-1], value)
        with self.assertRaisesRegex(
            UnsupportedCapture, "differs from its traced arguments"
        ):
            self.capture()
        self.query.assert_called_once_with(self.node)

    @parametrize("component", (0, 1, 2))
    def test_inactive_capture_declines_before_copy(self, component):
        before = list(self.before)
        before[component] = 0
        self.before = tuple(before)
        with self.assertRaisesRegex(UnsupportedCapture, "active CUDA graph capture"):
            self.capture()
        self.copy.assert_not_called()
        self.query.assert_not_called()

    @parametrize(
        "change",
        (
            "status",
            "capture",
            "graph",
            "none",
            "multiple",
            "old",
            "null",
            "before_edge",
            "after_edge",
        ),
    )
    def test_frontier_mismatch_declines_before_parameter_query(self, change):
        after = list(self.after)
        if change in ("status", "capture", "graph"):
            after[("status", "capture", "graph").index(change)] += 1
        elif change == "none":
            after[3] = ()
        elif change == "multiple":
            after[3] = ((13, bytes(8)), (17, bytes(8)))
        elif change == "old":
            after[3] = self.before[3]
        elif change == "null":
            after[3] = ((0, bytes(8)),)
        elif change == "before_edge":
            self.before = (1, 5, 7, ((11, bytes([1]) + bytes(7)),))
        else:
            after[3] = ((13, bytes([1]) + bytes(7)),)
        self.after = tuple(after)
        with self.assertRaisesRegex(UnsupportedCapture, "capture.*node"):
            self.capture()
        self.query.assert_not_called()

    @parametrize("byte_count", (0, -1))
    def test_empty_or_negative_copy_declines_before_cuda(self, byte_count):
        self.byte_count = byte_count
        with self.assertRaisesRegex(UnsupportedCapture, "positive byte count"):
            self.capture()
        self.frontier.assert_not_called()
        self.copy.assert_not_called()
        self.query.assert_not_called()

    @parametrize("stage", ("copy", "query"))
    def test_cuda_errors_propagate(self, stage):
        if stage == "copy":
            self.copy.return_value = (runtime.cudaError_t.cudaErrorInvalidValue,)
        else:
            self.query.return_value = (runtime.cudaError_t.cudaErrorNotSupported,)
        with self.assertRaises(RuntimeError):
            self.capture()
        if stage == "copy":
            self.query.assert_not_called()


if __name__ == "__main__":
    run_tests()
