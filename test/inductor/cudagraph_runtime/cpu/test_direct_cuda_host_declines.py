# Owner(s): ["module: inductor"]

import gc
import weakref
from contextlib import ExitStack
from unittest import mock

import torch
from torch._inductor.runtime._cudagraph import direct_host, direct_hosttrace
from torch._inductor.runtime._cudagraph.direct_cuda_host import DirectCudaHost
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.cuda import _host_trace
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


ADAPTER = None
EVENTS = []


def converted(source):
    EVENTS.append("ordinary")
    source.add_(1)
    return source


def host(box):
    (source,) = box
    box.clear()
    output = ADAPTER(source)
    EVENTS.append("tail")
    source.add_(10)
    return (output,)


@instantiate_parametrized_tests
class TestDeclinePhases(TestCase):
    @parametrize("stage", ("output_kind", "record", "lower", "prepare"))
    def test_recognized_decline_finishes_ordinary_once(self, stage):
        events = []

        def scalar_result(source):
            converted(source)
            return 17

        with ExitStack() as stack:
            entry = scalar_result if stage == "output_kind" else converted
            stack.enter_context(
                mock.patch.dict(globals(), ADAPTER=DirectCudaHost(entry), EVENTS=events)
            )
            runtime = direct_host.DirectHost(host, object())
            stack.callback(runtime.close)
            record = stack.enter_context(
                mock.patch.object(_host_trace, "trace", return_value=object())
            )
            lower = stack.enter_context(
                mock.patch.object(direct_hosttrace, "lower_tape", return_value=object())
            )
            prepare = stack.enter_context(
                mock.patch.object(
                    direct_host,
                    "_prepare_observed",
                    wraps=direct_host._prepare_observed,
                )
            )
            if stage == "record":
                record.side_effect = _host_trace.Declined(
                    "recording deliberately unsupported"
                )
            elif stage == "lower":
                lower.side_effect = UnsupportedCapture(
                    "lowering deliberately unsupported"
                )
            elif stage == "prepare":
                prepare.side_effect = UnsupportedCapture(
                    "preparation deliberately unsupported"
                )
            source = torch.zeros(3)
            box = [source]
            (output,) = runtime(box)
            if stage == "output_kind":
                self.assertEqual(output, 17)
            else:
                self.assertIs(output, source)
            self.assertEqual(source, torch.full((3,), 11.0))
            self.assertEqual(events, ["ordinary", "tail"])
            self.assertEqual(box, [])
            self.assertEqual(runtime.variants, [])
            self.assertIsNone(runtime.entry)
            if stage == "output_kind":
                record.assert_not_called()
            else:
                record.assert_called_once()
            if stage in ("output_kind", "record"):
                lower.assert_not_called()

    @parametrize(
        "stage", ("ordinary", "ordinary_declined", "ordinary_unsupported", "record")
    )
    def test_unexpected_error_propagates_without_retry(self, stage):
        events = []
        error_type = {
            "ordinary_declined": _host_trace.Declined,
            "ordinary_unsupported": UnsupportedCapture,
        }.get(stage, RuntimeError)

        def ordinary(source):
            output = converted(source)
            if stage.startswith("ordinary"):
                raise error_type("unexpected ordinary error")
            return output

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(
                    globals(), ADAPTER=DirectCudaHost(ordinary), EVENTS=events
                )
            )
            runtime = direct_host.DirectHost(host, object())
            stack.callback(runtime.close)
            record = stack.enter_context(
                mock.patch.object(
                    _host_trace,
                    "trace",
                    side_effect=RuntimeError("unexpected recording error"),
                )
            )
            source = torch.zeros(3)
            with self.assertRaisesRegex(error_type, "unexpected"):
                runtime([source])
            self.assertEqual(source, torch.ones(3))
            self.assertEqual(events, ["ordinary"])
            self.assertEqual(runtime.variants, [])
            if stage.startswith("ordinary"):
                record.assert_not_called()
            else:
                record.assert_called_once()

    def test_unexpected_preparation_error_propagates(self):
        events = []
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(
                    globals(), ADAPTER=DirectCudaHost(converted), EVENTS=events
                )
            )
            runtime = direct_host.DirectHost(host, object())
            stack.callback(runtime.close)
            stack.enter_context(
                mock.patch.object(_host_trace, "trace", return_value=object())
            )
            stack.enter_context(
                mock.patch.object(direct_hosttrace, "lower_tape", return_value=object())
            )
            stack.enter_context(
                mock.patch.object(
                    direct_host,
                    "_prepare_observed",
                    side_effect=RuntimeError("unexpected preparation error"),
                )
            )
            source = torch.zeros(3)
            with self.assertRaisesRegex(RuntimeError, "unexpected preparation"):
                runtime([source])
            self.assertEqual(source, torch.full((3,), 11.0))
            self.assertEqual(events, ["ordinary", "tail"])
            self.assertEqual(runtime.variants, [])

    @parametrize("stage", ("ordinary", "prepare"))
    def test_failed_observation_releases_recorded_resources(self, stage):
        references = []

        def record(*args, **kwargs):
            resource = torch.ones(1)
            references.append(weakref.ref(resource))
            return resource

        def prepare(*args, **kwargs):
            raise RuntimeError("preparation failed")

        def observed_host(box):
            result = host(box)
            if stage == "ordinary":
                raise RuntimeError("ordinary tail failed")
            return result

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(globals(), ADAPTER=DirectCudaHost(converted), EVENTS=[])
            )
            runtime = direct_host.DirectHost(observed_host, object())
            stack.callback(runtime.close)
            stack.enter_context(mock.patch.object(_host_trace, "trace", new=record))
            stack.enter_context(
                mock.patch.object(
                    direct_hosttrace, "lower_tape", new=lambda resource: resource
                )
            )
            stack.enter_context(
                mock.patch.object(direct_host, "_prepare_observed", new=prepare)
            )
            source = torch.zeros(3)
            with self.assertRaisesRegex(RuntimeError, "failed"):
                runtime([source])
            gc.collect()
            self.assertEqual(len(references), 1)
            self.assertIsNone(references[0]())
            self.assertFalse(runtime._lock.locked())
            self.assertEqual(source, torch.full((3,), 11.0))


if __name__ == "__main__":
    run_tests()
