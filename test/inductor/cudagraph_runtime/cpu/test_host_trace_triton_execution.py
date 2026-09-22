# Owner(s): ["module: inductor"]
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import triton

import torch
from torch.cuda import _host_trace as ht, _host_trace_triton as hook
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@triton.jit
def kernel(n):
    pass


@instantiate_parametrized_tests
class TestTritonExecution(TestCase):
    @parametrize("warm_up", (False, True))
    @parametrize("failure", ("descriptor", "owner_error"))
    def test_caught_capture_refusal_cannot_omit_observed_launch(self, warm_up, failure):
        binary = object()
        calls = []
        caught = []
        records = {"launches": [], "written_roots": [], "rng_slots": []}
        recorder = SimpleNamespace(
            finish=mock.Mock(), end=mock.Mock(), records=lambda: records
        )
        stream = object()
        trace = SimpleNamespace(
            device=torch.device("cuda", 0), stream=stream, rec=recorder, inputs=[]
        )

        def run(*args, **kwargs):
            calls.append(kwargs["warmup"])
            return binary

        def host():
            try:
                kernel[(1,)](3)
            except RuntimeError as error:
                caught.append(error)
            return ()

        with (
            mock.patch.object(kernel, "run", side_effect=run),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(torch.cuda, "current_stream", return_value=stream),
            mock.patch.object(torch.cuda, "device", return_value=nullcontext()),
            hook.hooked(),
        ):
            observations = None
            if warm_up:
                with hook.observing() as observations:
                    host()
                self.assertEqual(calls, [False])
                self.assertEqual(len(observations), 1)
            owner = mock.Mock(return_value=SimpleNamespace(descriptors=(object(),)))
            if failure == "owner_error":
                owner.side_effect = RuntimeError("owner failure")
            with (
                mock.patch.object(ht, "_Trace", return_value=trace),
                mock.patch.object(ht, "_TraceMode", return_value=nullcontext()),
                mock.patch.object(
                    ht._host_trace_cute, "tracing", return_value=nullcontext()
                ),
                mock.patch.object(ht._host_trace_cute, "merge"),
                mock.patch.object(ht, "_check_host_buffers"),
                mock.patch.object(ht._PartialTrace, "of", return_value=None),
                mock.patch.object(hook, "_owner", owner),
                self.assertRaises(ht.Declined) as declined,
            ):
                ht._trace_once(host, (), [], 0, None, triton=observations)
        self.assertEqual(calls, [False, True] if warm_up else [True])
        self.assertEqual(len(caught), 1)
        self.assertIs(declined.exception.__cause__, caught[0])
        self.assertIn(str(caught[0]), str(declined.exception))
        recorder.end.assert_called_once()
        self.assertIsNone(getattr(ht._active, "trace", None))

    @parametrize("error_type", (RuntimeError, ht.Declined))
    def test_observed_execution_error_is_never_retried(self, error_type):
        from triton.runtime.jit import KernelInterface

        original_getitem = KernelInterface.__getitem__
        calls = []
        error = error_type("after ordinary execution")

        def run(*args, **kwargs):
            calls.append(kwargs["warmup"])
            raise error

        with mock.patch.object(kernel, "run", side_effect=run), hook.hooked():
            with (
                hook.observing() as observations,
                self.assertRaises(error_type) as caught,
            ):
                kernel[(1,)](3)
            self.assertIs(caught.exception, error)
            self.assertEqual(observations, [])
        self.assertEqual(calls, [False])
        self.assertIs(KernelInterface.__getitem__, original_getitem)
        self.assertIsNone(getattr(hook._state, "phase", None))
        with (
            mock.patch.object(kernel, "run", return_value=object()),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            hook.hooked(),
            hook.observing() as later,
        ):
            kernel[(1,)](3)
        self.assertEqual(len(later), 1)

    def test_error_after_run_before_observation_is_not_retried(self):
        with (
            mock.patch.object(kernel, "run", return_value=object()) as run,
            mock.patch.object(
                torch.cuda, "current_device", side_effect=RuntimeError("device query")
            ),
            hook.hooked(),
        ):
            with (
                hook.observing() as observations,
                self.assertRaisesRegex(RuntimeError, "device query"),
            ):
                kernel[(1,)](3)
        run.assert_called_once()
        self.assertEqual(observations, [])
        self.assertIsNone(getattr(hook._state, "phase", None))

    def test_pre_execution_decline_serves_ordinary_once(self):
        from torch._inductor.runtime._cudagraph.direct_hosttrace import HostTraceReplay

        ordinary = mock.Mock(return_value=())
        replay = HostTraceReplay(ordinary)
        self.addCleanup(replay.close)
        with mock.patch.object(
            ht, "trace", side_effect=ht.Declined("unsupported before execution")
        ) as trace:
            with self.assertWarnsRegex(RuntimeWarning, "unsupported before execution"):
                self.assertEqual(replay(), ())
            self.assertEqual(replay(), ())
        trace.assert_called_once()
        self.assertEqual(ordinary.call_count, 2)

    def test_native_dispatch_error_is_not_retried_and_resets_state(self):
        from torch._inductor.runtime._cudagraph.direct_hosttrace import HostTraceReplay

        ordinary = mock.Mock(return_value=())
        replay = HostTraceReplay(ordinary)
        self.addCleanup(replay.close)
        dispatch = mock.Mock(side_effect=RuntimeError("after native execution"))
        replay._hot = SimpleNamespace(
            nargs=0,
            constants_hold=None,
            arena=None,
            outputs=None,
            sequence=None,
            boxer=lambda args, arena: [],
            dispatch=dispatch,
        )
        with self.assertRaisesRegex(RuntimeError, "after native execution"):
            replay()
        dispatch.assert_called_once()
        ordinary.assert_not_called()
        self.assertFalse(replay._python_dispatch)


if __name__ == "__main__":
    run_tests()
