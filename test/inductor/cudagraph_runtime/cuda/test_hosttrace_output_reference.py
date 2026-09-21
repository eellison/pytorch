# Owner(s): ["module: inductor"]

import sys
from unittest import mock

import torch
from torch._inductor.runtime._cudagraph.direct_hosttrace import HostTraceReplay
from torch.cuda import _host_trace
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


OUTPUT_KIND = None


def operation(marker, destination, increment):
    if marker != "fixed":
        raise AssertionError(f"Expected the closed marker, got {marker!r}")
    owned = destination + increment
    destination.copy_(owned.t())
    if OUTPUT_KIND == "input":
        return destination, destination
    if OUTPUT_KIND == "owned":
        return owned, owned
    if OUTPUT_KIND == "full_views":
        return owned[:, :], owned[:, :]
    view = destination[1:]
    return view, view if OUTPUT_KIND == "view" else destination[1:]


def boxed_operation(destination, increment):
    return operation("fixed", destination, increment)


class TestHostTraceOutputReference(TestCase):
    def check_outputs(self, kind, actual, destination, owned):
        expected = (
            owned
            if kind in ("owned", "full_views")
            else destination
            if kind == "input"
            else destination[1:]
        )
        self.assertEqual(tuple(actual), (expected, expected))
        self.assertEqual(actual[0] is actual[1], kind in ("input", "owned", "view"))
        self.assertEqual(actual[0] is destination, kind == "input")

    @parametrize("kind", ("input", "view", "distinct_views", "owned", "full_views"))
    @parametrize("closed_scalar", (False, True))
    def test_shared_replay_identity_and_mutation(self, device, kind, closed_scalar):
        self.enterContext(mock.patch.dict(globals(), OUTPUT_KIND=kind))
        samples = []
        for n, dtype, offset in (
            (8, torch.float32, 1),
            (8, torch.float32, 5),
            (12, torch.float16, 1),
            (12, torch.float16, 9),
            (8, torch.float32, 9),
        ):
            backing = torch.arange(n * n + offset, device=device, dtype=dtype)
            destination = backing[offset:].view(n, n)
            samples.append((destination, torch.full_like(destination, 0.25)))
        runtime = (
            HostTraceReplay(operation, ("fixed", *samples[0]), warm_up=False)
            if closed_scalar
            else HostTraceReplay(boxed_operation)
        )
        self.addCleanup(runtime.close)
        held = []
        for step, (destination, increment) in enumerate((*samples, samples[0])):
            owned = destination + increment
            increment_before = increment.clone()
            box = [destination, increment]
            previous = runtime.misses
            miss = step == 2 or step == 0 and not closed_scalar
            frames = []

            def profile(frame, event, result):
                if event == "call":
                    frames.append(frame.f_code)

            try:
                if not miss:
                    sys.setprofile(profile)
                actual = runtime(box) if runtime.entry is None else runtime.entry(box)
            finally:
                sys.setprofile(None)
            self.assertEqual(box, [])
            self.assertEqual(runtime.misses - previous, int(miss))
            self.assertEqual(len(runtime.variants), 1 if step < 2 else 2)
            self.assertEqual(frames, [])
            self.assertEqual(destination, owned.t())
            self.assertEqual(increment, increment_before)
            self.check_outputs(kind, actual, destination, owned)
            held.append((actual, destination, owned))
        self.assertEqual(
            runtime.variants[0].lowered.symbols.positions,
            [1, 2] if closed_scalar else [0, 1],
        )
        self.assertEqual(
            len({destination.data_ptr() for destination, _ in samples}), len(samples)
        )
        runtime.close()
        for actual, destination, owned in held:
            self.check_outputs(kind, actual, destination, owned)

    @parametrize("kind", ("input", "view", "distinct_views", "owned", "full_views"))
    def test_interim_replay_identity(self, device, kind):
        self.enterContext(mock.patch.dict(globals(), OUTPUT_KIND=kind))
        # The interim build warms the ordinary host on these disposable inputs.
        build_destination = torch.zeros(65, device=device)[1:].view(8, 8)
        build_args = (
            "fixed",
            build_destination,
            torch.full_like(build_destination, 0.25),
        )
        tape = _host_trace.trace(operation, build_args, warm_up=False)
        interim = _host_trace.build(tape, operation, build_args)
        held = []
        for offset in (1, 5):
            backing = torch.arange(64 + offset, device=device, dtype=torch.float32)
            destination = backing[offset:].view(8, 8)
            increment = torch.full_like(destination, 0.25)
            owned = destination + increment
            actual = interim.replay(("fixed", destination, increment))
            self.assertEqual(destination, owned.t())
            self.check_outputs(kind, actual, destination, owned)
            held.append((actual, destination, owned))
        del interim
        for actual, destination, owned in held:
            self.check_outputs(kind, actual, destination, owned)


instantiate_device_type_tests(TestHostTraceOutputReference, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
