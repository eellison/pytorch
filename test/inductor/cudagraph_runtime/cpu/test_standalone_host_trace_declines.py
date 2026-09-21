# Owner(s): ["module: inductor"]

from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.runtime._cudagraph import direct_hosttrace

from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.cuda import _host_trace
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestStandaloneHostTraceDeclines(TestCase):
    def mock_preparation(self, stack, runtime, installed=False):
        def record(fn, args, *, warm_up):
            source = args[0]
            root = _host_trace._Root("arg0", 0, source.element_size())
            # Recording and native execution are mocked; pageable CPU inputs
            # cannot match this partial CUDA trace, so declines use exact classes.
            return SimpleNamespace(
                nargs=1,
                positions=[0],
                constants=(),
                inputs=[
                    _host_trace._InputRec(
                        0,
                        "arg0",
                        source.dtype,
                        list(source.shape),
                        list(source.stride()),
                        source.storage_offset(),
                        root,
                        torch.device("cuda", 0),
                        False,
                    )
                ],
                guards=(),
                device=torch.device("cuda", 0),
                outputs=(object(),),
                args=args,
            )

        lowered = SimpleNamespace(
            registration=object(),
            pinned_positions=(),
            written_positions=(),
            arena=None,
            output_arena=None,
            sequence=None,
            symbols=SimpleNamespace(positions=[0]),
            nargs=1,
            constant_positions=(),
            tensors=tuple,
            device=0,
            contract_holds=lambda args: len(args) == 1,
        )
        prepared = mock.Mock()
        dispatcher = mock.Mock(return_value=direct_hosttrace._MISSED)
        stack.enter_context(
            mock.patch.object(direct_hosttrace, "_current_raw_stream", return_value=0)
        )
        stack.enter_context(
            mock.patch.object(
                runtime, "_missed", side_effect=_host_trace.Miss("native guard miss")
            )
        )
        trace = stack.enter_context(
            mock.patch.object(_host_trace, "trace", side_effect=record)
        )
        lower = stack.enter_context(
            mock.patch.object(direct_hosttrace, "lower_tape", return_value=lowered)
        )
        prepare = stack.enter_context(
            mock.patch.object(
                direct_hosttrace, "prepare_hosttrace", return_value=prepared
            )
        )
        publish = stack.enter_context(
            mock.patch.object(
                torch._C, "_cuda_make_boxed_dispatch", return_value=dispatcher
            )
        )
        prior = None
        if installed:
            family = direct_hosttrace._Family(lowered, dispatcher, [])
            prior = direct_hosttrace._Variant(
                object(), lowered, mock.Mock(), runtime, family
            )
            family.variants.append(prior)
            runtime._families.append(family)
            runtime.variants.append(prior)
            runtime.entry = dispatcher
        return SimpleNamespace(
            record=trace,
            lower=lower,
            prepare=prepare,
            publish=publish,
            lowered=lowered,
            prepared=prepared,
            dispatcher=dispatcher,
            prior=prior,
        )

    @parametrize("installed", (False, True))
    @parametrize("stage", ("record", "lower", "lower_subclass", "prepare"))
    def test_recognized_decline_returns_ordinary_once(self, installed, stage):
        events = []

        def ordinary(source):
            events.append("ordinary")
            source.add_(1)
            return source, source.view(-1)

        runtime = direct_hosttrace.HostTraceReplay(ordinary)
        with ExitStack() as stack:
            stack.callback(runtime.close)
            mocks = self.mock_preparation(stack, runtime, installed)
            if stage == "record":
                mocks.record.side_effect = _host_trace.Declined("unsupported recording")
            elif stage.startswith("lower"):
                exception = (
                    direct_hosttrace.HostTraceLoweringDeclined
                    if stage == "lower_subclass"
                    else UnsupportedCapture
                )
                mocks.lower.side_effect = exception("unsupported lowering")
            else:
                mocks.prepare.side_effect = UnsupportedCapture(
                    "unsupported preparation"
                )
            source = torch.zeros(3)
            for expected in (1, 2):
                box = [source]
                output, view = runtime(box)
                self.assertIs(output, source)
                self.assertEqual(view, torch.full((3,), float(expected)))
                self.assertEqual(source, view)
                self.assertEqual(box, [])
                self.assertEqual(events, ["ordinary"] * expected)
                self.assertEqual(runtime.calls, expected)
                self.assertEqual(runtime.misses, expected)
                self.assertEqual(runtime.served, 0)
                self.assertFalse(runtime.lock.locked())
            self.assertEqual(runtime.variants, [mocks.prior] if installed else [])
            self.assertIs(runtime.entry, mocks.dispatcher if installed else None)
            mocks.publish.assert_not_called()
            mocks.dispatcher.append.assert_not_called()
            self.assertEqual(mocks.record.call_count, 1)
            self.assertEqual(mocks.record.call_args.kwargs, {"warm_up": False})
            self.assertEqual(mocks.lower.call_count, 0 if stage == "record" else 1)
            self.assertEqual(mocks.prepare.call_count, 1 if stage == "prepare" else 0)
            self.assertEqual(len(runtime.declined_exact), 1)

    @parametrize(
        "stage",
        (
            "ordinary",
            "ordinary_declined",
            "ordinary_unsupported",
            "record",
            "lower",
            "prepare",
        ),
    )
    def test_unexpected_error_propagates_without_retry(self, stage):
        events = []
        error_type = {
            "ordinary_declined": _host_trace.Declined,
            "ordinary_unsupported": UnsupportedCapture,
        }.get(stage, RuntimeError)
        error = error_type("injected error")

        def ordinary(source):
            events.append("ordinary")
            source.add_(1)
            raise error

        runtime = direct_hosttrace.HostTraceReplay(ordinary)
        with ExitStack() as stack:
            stack.callback(runtime.close)
            mocks = self.mock_preparation(stack, runtime)
            ordinary_stage = stage.startswith("ordinary")
            if ordinary_stage:
                mocks.record.side_effect = _host_trace.Declined("recording declines")
            else:
                {
                    "record": mocks.record,
                    "lower": mocks.lower,
                    "prepare": mocks.prepare,
                }[stage].side_effect = error
            source = torch.zeros(3)
            with self.assertRaisesRegex(error_type, "injected") as caught:
                runtime([source])
            self.assertIs(caught.exception, error)
            self.assertEqual(
                source, torch.ones(3) if ordinary_stage else torch.zeros(3)
            )
            self.assertEqual(events, ["ordinary"] if ordinary_stage else [])
            self.assertEqual(runtime.variants, [])
            self.assertIsNone(runtime.entry)
            self.assertFalse(runtime.lock.locked())
            mocks.record.assert_called_once()
            mocks.publish.assert_not_called()
            self.assertEqual(len(runtime.declined_exact), int(ordinary_stage))
            if ordinary_stage or stage == "record":
                mocks.lower.assert_not_called()
            if ordinary_stage or stage != "prepare":
                mocks.prepare.assert_not_called()

    @parametrize("installed", (False, True))
    def test_publication_error_closes_only_new_entry(self, installed):
        ordinary = mock.Mock(
            side_effect=AssertionError("publication precedes execution")
        )
        runtime = direct_hosttrace.HostTraceReplay(ordinary)
        with ExitStack() as stack:
            stack.callback(runtime.close)
            mocks = self.mock_preparation(stack, runtime, installed)
            target = mocks.dispatcher.append if installed else mocks.publish
            target.side_effect = ValueError("invalid publication")
            source = torch.zeros(3)
            with self.assertRaisesRegex(ValueError, "invalid publication"):
                runtime([source])
            self.assertEqual(source, torch.zeros(3))
            ordinary.assert_not_called()
            self.assertEqual(runtime.variants, [mocks.prior] if installed else [])
            self.assertIs(runtime.entry, mocks.dispatcher if installed else None)
            mocks.prepared.close.assert_called_once_with()
            mocks.dispatcher.close.assert_not_called()
            self.assertFalse(runtime.lock.locked())
            self.assertEqual(runtime.declined_classes, ())

    def test_decline_allows_later_successful_preparation(self):
        events = []

        def ordinary(source):
            events.append(("ordinary", source.numel()))
            source.add_(1)
            return source

        accepted = set()

        def native(box):
            source = box[0]
            if source.numel() not in accepted:
                return direct_hosttrace._MISSED
            events.append(("native", source.numel()))
            source.add_(1)
            box.clear()
            return [source]

        runtime = direct_hosttrace.HostTraceReplay(ordinary)
        with ExitStack() as stack:
            stack.callback(runtime.close)
            mocks = self.mock_preparation(stack, runtime)
            mocks.dispatcher.side_effect = native
            mocks.lower.side_effect = [
                UnsupportedCapture("first trace declines"),
                mocks.lowered,
                mocks.lowered,
            ]

            def prepare(lowered, args, **kwargs):
                accepted.add(args[0].numel())
                return mock.Mock()

            mocks.prepare.side_effect = prepare
            expected_events = []
            for size, traces, variants, ordinary_call in (
                (3, 1, 0, True),
                (3, 1, 0, True),
                (4, 2, 1, False),
                (5, 3, 2, False),
                (4, 3, 2, False),
                (3, 3, 2, True),
            ):
                source = torch.zeros(size)
                box = [source]
                (output,) = runtime(box)
                self.assertIs(output, source)
                self.assertEqual(output, torch.ones(size))
                self.assertEqual(box, [])
                expected_events.append(
                    ("ordinary" if ordinary_call else "native", size)
                )
                self.assertEqual(events, expected_events)
                self.assertEqual(runtime.traces, traces)
                self.assertEqual(len(runtime.variants), variants)
                self.assertFalse(runtime.lock.locked())
            self.assertIs(runtime.entry, mocks.dispatcher)
            self.assertEqual(
                (runtime.calls, runtime.ordinary, runtime.served), (6, 3, 3)
            )
            self.assertEqual(
                (
                    mocks.record.call_count,
                    mocks.lower.call_count,
                    mocks.prepare.call_count,
                ),
                (3, 3, 2),
            )
            first, second = (v.entry for v in runtime.variants)
            mocks.publish.assert_called_once_with(
                ((first, mocks.lowered.registration),), runtime._on_miss, ()
            )
            mocks.dispatcher.append.assert_called_once_with(
                second, mocks.lowered.registration
            )

    def test_actual_recorder_declines_cpu_input_after_ordinary_result(self):
        def ordinary(source):
            source.add_(1)
            return source

        runtime = direct_hosttrace.HostTraceReplay(ordinary)
        try:
            source = torch.zeros(3)
            box = [source]
            (output,) = runtime(box)
            self.assertIs(output, source)
            self.assertEqual(source, torch.ones(3))
            self.assertEqual(box, [])
            self.assertEqual(runtime.variants, [])
        finally:
            runtime.close()


if __name__ == "__main__":
    run_tests()
