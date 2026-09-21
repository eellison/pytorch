# Owner(s): ["module: inductor"]
"""Ordered capture-event completeness without CUDA execution."""

from dataclasses import replace

from torch._inductor.runtime.cudagraph_launch_association import (
    associate_kernel_launches,
    RecordedGraphNode,
    RecordedKernelLaunch,
    UnsupportedCapture,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _frontier(*nodes):
    return (1, 7, 8, tuple((node, bytes(8)) for node in nodes))


def _sequence(kinds):
    events, kernels = [], []
    before = _frontier()
    for index, kind in enumerate(kinds):
        node = 10 + index
        after = _frontier(node)
        if kind == "kernel":
            event = RecordedKernelLaunch(1, before, after, 99, (b"args",))
            kernels.append(
                (node, 99, 0, 0, (1, 1, 1), (32, 1, 1), 0, False, ((0, 4, b"args"),))
            )
        else:
            event = RecordedGraphNode(1, before, after, kind)
        events.append(event)
        before = after
    launches = tuple(event for event in events if type(event) is RecordedKernelLaunch)
    snapshot = (8, 7, tuple(range(10, 10 + len(events))), tuple(kernels))
    return launches, tuple(events), snapshot


@instantiate_parametrized_tests
class TestCaptureEvents(TestCase):
    @parametrize(
        "kinds",
        (
            ("kernel",),
            ("kernel", "kernel"),
            ("memset", "kernel"),
            ("kernel", "memset"),
            ("kernel", "memset", "kernel"),
            ("memset", "kernel", "memset", "kernel"),
            ("memcpy", "kernel", "memset", "kernel"),
            ("memset", "memset", "kernel"),
            ("memset",),
            ("memcpy",),
            ("memset", "memcpy", "memset"),
        ),
    )
    def test_complete_sequence(self, kinds):
        launches, events, snapshot = _sequence(kinds)
        actual = associate_kernel_launches(launches, snapshot, events=events)
        self.assertEqual(
            tuple(item.occurrence for item in actual), tuple(range(len(launches)))
        )
        self.assertEqual(tuple(item.snapshot for item in actual), snapshot[3])

    def test_kernel_only_api_unchanged(self):
        launches, events, snapshot = _sequence(("kernel", "kernel"))
        self.assertEqual(
            associate_kernel_launches(launches, snapshot),
            associate_kernel_launches(launches, snapshot, events=events),
        )

    def test_empty_capture(self):
        launches, events, snapshot = _sequence(())
        self.assertEqual(associate_kernel_launches(launches, snapshot), ())
        self.assertEqual(
            associate_kernel_launches(launches, snapshot, events=events), ()
        )

    @parametrize("unrecorded", ("node", "kernel", "both"))
    def test_empty_events_reject_unrecorded_graph(self, unrecorded):
        _, _, snapshot = _sequence(("kernel",))
        graph, capture, nodes, kernels = snapshot
        if unrecorded == "node":
            kernels = ()
        elif unrecorded == "kernel":
            nodes = ()
        with self.assertRaises(UnsupportedCapture):
            associate_kernel_launches((), (graph, capture, nodes, kernels), events=())

    @parametrize(
        "change",
        (
            "omitted",
            "reordered",
            "collapsed_frontier",
            "wrong_stream",
            "wrong_capture",
            "wrong_graph",
        ),
    )
    def test_interleaved_event_must_match_frontiers(self, change):
        launches, events, snapshot = _sequence(("kernel", "memset", "kernel"))
        first, middle, last = events
        if change == "omitted":
            events = (first, last)
        elif change == "reordered":
            events = (middle, first, last)
        elif change == "collapsed_frontier":
            last = replace(last, before=first.after)
            events, launches = (first, middle, last), (first, last)
        elif change == "wrong_stream":
            events = (first, replace(middle, stream=2), last)
        elif change == "wrong_capture":
            events = (first, replace(middle, after=(1, 88, 8, middle.after[3])), last)
        else:
            events = (first, replace(middle, after=(1, 7, 88, middle.after[3])), last)
        with self.assertRaises(UnsupportedCapture):
            associate_kernel_launches(launches, snapshot, events=events)

    def test_missing_prefix_event_rejected(self):
        launches, events, snapshot = _sequence(("memset", "kernel"))
        with self.assertRaisesRegex(UnsupportedCapture, "initial graph dependencies"):
            associate_kernel_launches(launches, snapshot, events=events[1:])

    @parametrize(
        "change", ("missing", "extra", "duplicate", "wrong_graph", "wrong_capture")
    )
    def test_snapshot_coverage(self, change):
        launches, events, snapshot = _sequence(("kernel", "memset", "kernel"))
        graph, capture, nodes, kernels = snapshot
        if change == "missing":
            nodes = nodes[:-1]
        elif change == "extra":
            nodes = (*nodes, 77)
        elif change == "duplicate":
            nodes = (10, 10, 12)
        elif change == "wrong_graph":
            graph = 99
        else:
            capture = 99
        with self.assertRaises(UnsupportedCapture):
            associate_kernel_launches(
                launches, (graph, capture, nodes, kernels), events=events
            )

    @parametrize(
        "change",
        (
            "empty_frontier",
            "multiple_nodes",
            "reused_node",
            "nondefault_edge",
            "unknown_kind",
            "unknown_type",
            "missing_events",
        ),
    )
    def test_invalid_event(self, change):
        launches, events, snapshot = _sequence(("kernel", "memset", "kernel"))
        first, middle, last = events
        if change == "empty_frontier":
            middle = replace(middle, after=_frontier())
        elif change == "multiple_nodes":
            middle = replace(middle, after=_frontier(11, 13))
        elif change == "reused_node":
            middle = replace(middle, after=first.after)
        elif change == "nondefault_edge":
            middle = replace(middle, after=(1, 7, 8, ((11, b"changed!"),)))
        elif change == "unknown_kind":
            middle = replace(middle, kind="event_record")
        elif change == "unknown_type":
            middle = object()
        events = () if change == "missing_events" else (first, middle, last)
        with self.assertRaises(UnsupportedCapture):
            associate_kernel_launches(launches, snapshot, events=events)

    @parametrize(
        "change",
        (
            "missing",
            "duplicate",
            "nonkernel",
            "function",
            "argument_width",
            "argument_value",
        ),
    )
    def test_kernel_correspondence_remains_exact(self, change):
        launches, events, snapshot = _sequence(("kernel", "memset", "kernel"))
        kernels = list(snapshot[3])
        if change == "missing":
            kernels.pop()
        elif change == "duplicate":
            kernels[1] = kernels[0]
        else:
            kernel = list(kernels[0])
            if change == "nonkernel":
                kernel[0] = 11
            elif change == "function":
                kernel[1] = 88
            elif change == "argument_width":
                kernel[8] = ((0, 8, b"args"),)
            else:
                kernel[8] = ((0, 4, b"nope"),)
            kernels[0] = tuple(kernel)
        with self.assertRaises(UnsupportedCapture):
            associate_kernel_launches(
                launches, (*snapshot[:3], tuple(kernels)), events=events
            )

    def test_kernel_subsequence_cannot_be_replaced(self):
        launches, events, snapshot = _sequence(("kernel", "memset", "kernel"))
        with self.assertRaisesRegex(UnsupportedCapture, "kernel sequence"):
            associate_kernel_launches(
                tuple(reversed(launches)), snapshot, events=events
            )


if __name__ == "__main__":
    run_tests()
