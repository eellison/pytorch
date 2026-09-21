# Owner(s): ["module: inductor"]

import gc
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest import mock

import torch
from torch.cuda import _host_trace as ht
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestHostTraceGC(TestCase):
    @parametrize("enabled", (False, True))
    @parametrize(
        "outcome",
        ("success", "constructor", "input", "declined", "unexpected", "finish", "end"),
    )
    def test_capture_window_and_gc_restoration(self, enabled, outcome):
        original = gc.isenabled()
        argument = SimpleNamespace(
            is_cuda=True, device=torch.device("cuda", 0), numel=lambda: 8
        )
        symbolic, result = object(), SimpleNamespace()
        events = []
        recorder = mock.Mock()
        trace = SimpleNamespace(rec=recorder, input=mock.Mock(), inputs=[])

        def construct(device, hints):
            self.assertFalse(gc.isenabled())
            if outcome == "constructor":
                raise RuntimeError("constructor failure")
            return trace

        def wrap_input(position, value):
            self.assertFalse(gc.isenabled())
            if outcome == "input":
                raise RuntimeError("input failure")
            return symbolic

        def ordinary(value):
            if value is argument:
                self.assertEqual(gc.isenabled(), enabled)
                events.append("ordinary")
            else:
                self.assertIs(value, symbolic)
                self.assertFalse(gc.isenabled())
                events.append("symbolic")
                if outcome == "declined":
                    raise ht.Declined("declined trace")
                if outcome == "unexpected":
                    raise RuntimeError("unexpected failure")
            return ()

        def finish():
            self.assertFalse(gc.isenabled())
            if outcome == "finish":
                raise RuntimeError("finish failure")

        def end():
            self.assertFalse(gc.isenabled())
            if outcome == "end":
                raise RuntimeError("end failure")

        try:
            gc.enable() if enabled else gc.disable()
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(ht._active, "trace", None, create=True)
                )
                stack.enter_context(mock.patch.object(ht, "_metadata", return_value=()))
                stack.enter_context(
                    mock.patch.object(torch._C, "_is_cow_tensor", return_value=False)
                )
                collect = stack.enter_context(mock.patch.object(gc, "collect"))
                stack.enter_context(
                    mock.patch.object(ht, "_tensor_positions", return_value=(0,))
                )
                stack.enter_context(
                    mock.patch.object(ht, "_math_bits", return_value=None)
                )
                stack.enter_context(
                    mock.patch.object(ht, "_Trace", side_effect=construct)
                )
                stack.enter_context(
                    mock.patch.object(ht, "_TraceMode", return_value=nullcontext())
                )
                stack.enter_context(mock.patch.object(ht, "_check_host_buffers"))
                stack.enter_context(mock.patch.object(ht, "Tape", return_value=result))
                stack.enter_context(
                    mock.patch.object(torch.cuda, "device", return_value=nullcontext())
                )
                stream = stack.enter_context(
                    mock.patch.object(torch.cuda, "current_stream")
                )
                trace.input.side_effect = wrap_input
                recorder.finish.side_effect = finish
                recorder.end.side_effect = end
                if outcome == "success":
                    self.assertIs(ht.trace(ordinary, (argument,)), result)
                    collect.assert_not_called()
                else:
                    error = ht.Declined if outcome == "declined" else RuntimeError
                    with self.assertRaisesRegex(error, outcome):
                        ht.trace(ordinary, (argument,))
                self.assertEqual(gc.isenabled(), enabled)
                self.assertIsNone(ht._active.trace)
                stream.return_value.synchronize.assert_called_once_with()
                expected = ["ordinary"]
                if outcome not in ("constructor", "input"):
                    expected.append("symbolic")
                self.assertEqual(events, expected)
                if outcome == "constructor":
                    recorder.end.assert_not_called()
                else:
                    recorder.end.assert_called_once_with()
        finally:
            gc.enable() if original else gc.disable()

    @parametrize("enabled", (False, True))
    def test_ordinary_error_precedes_gc_window(self, enabled):
        original = gc.isenabled()
        argument = SimpleNamespace(
            is_cuda=True, device=torch.device("cuda", 0), numel=lambda: 8
        )

        def ordinary(value):
            self.assertIs(value, argument)
            self.assertEqual(gc.isenabled(), enabled)
            raise ht.Declined("ordinary failure")

        try:
            gc.enable() if enabled else gc.disable()
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(ht._active, "trace", None, create=True)
                )
                stack.enter_context(mock.patch.object(ht, "_metadata", return_value=()))
                stack.enter_context(
                    mock.patch.object(torch._C, "_is_cow_tensor", return_value=False)
                )
                collect = stack.enter_context(mock.patch.object(gc, "collect"))
                construct = stack.enter_context(mock.patch.object(ht, "_Trace"))
                stack.enter_context(
                    mock.patch.object(ht, "_tensor_positions", return_value=(0,))
                )
                stack.enter_context(
                    mock.patch.object(ht, "_math_bits", return_value=None)
                )
                stack.enter_context(
                    mock.patch.object(torch.cuda, "device", return_value=nullcontext())
                )
                with self.assertRaisesRegex(ht.Declined, "ordinary failure"):
                    ht.trace(ordinary, (argument,))
                self.assertEqual(gc.isenabled(), enabled)
                construct.assert_not_called()
                collect.assert_not_called()
        finally:
            gc.enable() if original else gc.disable()


if __name__ == "__main__":
    run_tests()
