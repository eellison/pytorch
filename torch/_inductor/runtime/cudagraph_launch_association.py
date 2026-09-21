from dataclasses import dataclass
from typing import Literal

from torch.utils._ordered_set import OrderedSet


CaptureFrontier = tuple[int, int, int, tuple[tuple[int, bytes], ...]]
KernelSnapshot = tuple[
    int,
    int,
    int,
    int,
    tuple[int, int, int],
    tuple[int, int, int],
    int,
    bool,
    tuple[tuple[int, int, bytes], ...],
]
GraphSnapshot = tuple[int, int, tuple[int, ...], tuple[KernelSnapshot, ...]]


class UnsupportedCapture(ValueError):
    pass


@dataclass(frozen=True)
class RecordedKernelLaunch:
    stream: int
    before: CaptureFrontier
    after: CaptureFrontier
    function: int
    argument_bytes: tuple[bytes, ...]


@dataclass(frozen=True)
class RecordedGraphNode:
    """A non-kernel node the caller issued itself between launches (a memset, a
    copy), or a node of a kernel template site it launched from a registered
    template (torch._C._cuda_kernel_template_register): the caller recorded its
    capture frontiers; kernel inspection does not cover it."""

    stream: int
    before: CaptureFrontier
    after: CaptureFrontier
    kind: Literal["memset", "memcpy", "template"]


@dataclass(frozen=True)
class CapturedKernelLaunch:
    occurrence: int
    snapshot: KernelSnapshot


def associate_kernel_launches(
    launches: tuple[RecordedKernelLaunch, ...],
    snapshot: GraphSnapshot,
    *,
    events: tuple[RecordedKernelLaunch | RecordedGraphNode, ...] | None = None,
) -> tuple[CapturedKernelLaunch, ...]:
    """Every event of the capture in issue order: the graph must contain exactly
    the events' nodes, kernel inspection covers the launches. A capture of
    template (and memset / memcpy) events alone has no launches to associate."""
    if events is None:
        events = launches
    for event in events:
        if type(event) is RecordedKernelLaunch:
            continue
        if type(event) is not RecordedGraphNode or event.kind not in (
            "memset",
            "memcpy",
            "template",
        ):
            raise UnsupportedCapture("Unsupported recorded capture event")
    kernel_events = tuple(
        event for event in events if type(event) is RecordedKernelLaunch
    )
    if kernel_events != launches:
        raise UnsupportedCapture(
            "Capture events do not match the recorded kernel sequence"
        )
    if not events:
        if snapshot[2] or snapshot[3]:
            raise UnsupportedCapture("The graph contains unrecorded nodes")
        return ()
    first = events[0]
    identity = first.before[:3]
    if identity[0] != 1 or not identity[1] or not identity[2]:
        raise UnsupportedCapture("An active CUDA graph capture is required")
    if snapshot[:2] != (identity[2], identity[1]):
        raise UnsupportedCapture("Completed graph does not match the recorded capture")
    if first.before[3]:
        raise UnsupportedCapture("Capture events omit initial graph dependencies")

    nodes = []
    seen = OrderedSet()
    previous = first.before
    for event in events:
        if (
            event.stream != first.stream
            or event.before[:3] != identity
            or event.after[:3] != identity
            or event.before != previous
        ):
            raise UnsupportedCapture(
                "Events must belong to one uninterrupted stream capture"
            )
        if any(edge != bytes(8) for _, edge in (*event.before[3], *event.after[3])):
            raise UnsupportedCapture("Nondefault capture dependencies are unsupported")
        if len(event.after[3]) != 1:
            raise UnsupportedCapture(
                "Each event must produce one capture frontier node"
            )
        node = event.after[3][0][0]
        if node in seen or any(node == dependency for dependency, _ in event.before[3]):
            raise UnsupportedCapture(
                "A capture event did not produce a new capture node"
            )
        seen.add(node)
        if type(event) is RecordedKernelLaunch:
            nodes.append(node)
        previous = event.after

    if len(snapshot[2]) != len(seen) or OrderedSet(snapshot[2]) != seen:
        raise UnsupportedCapture("The graph contains unrecorded or missing nodes")
    kernels = {kernel[0]: kernel for kernel in snapshot[3]}
    if len(snapshot[3]) != len(nodes) or OrderedSet(kernels) != OrderedSet(nodes):
        raise UnsupportedCapture(
            "Kernel inspection does not cover every recorded kernel"
        )

    result = []
    for occurrence, (node, launch) in enumerate(zip(nodes, launches)):
        kernel = kernels[node]
        if not launch.function or kernel[1] != launch.function:
            raise UnsupportedCapture(
                "Captured function does not match the selected launcher"
            )
        slots = kernel[8]
        if len(slots) != len(launch.argument_bytes):
            raise UnsupportedCapture(
                "Selected launcher does not describe the complete kernel ABI"
            )
        for index, ((_, width, actual), expected) in enumerate(
            zip(slots, launch.argument_bytes)
        ):
            if len(expected) != width or actual != expected:
                raise UnsupportedCapture(
                    f"Captured argument {index} differs from the selected launch"
                )
        result.append(CapturedKernelLaunch(occurrence, kernel))
    return tuple(result)
