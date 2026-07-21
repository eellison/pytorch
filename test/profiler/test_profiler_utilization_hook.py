# Owner(s): ["module: profiler"]

import json
import os
import tempfile
import unittest
from typing import Any
from unittest import mock

import torch
from torch.autograd.profiler_util import EventList, FunctionEvent
from torch.profiler import (
    kineto_available,
    profile,
    ProfilerActivity,
    register_export_chrome_trace_callback,
)
from torch.profiler.profiler import _run_export_chrome_trace_callbacks
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.hooks import RemovableHandle


class TestExportChromeTraceCallbacks(TestCase):
    def setUp(self) -> None:
        self.handles: list[RemovableHandle] = []

    def tearDown(self) -> None:
        for handle in self.handles:
            handle.remove()

    def register(self, callback) -> RemovableHandle:
        handle = register_export_chrome_trace_callback(callback)
        self.handles.append(handle)
        return handle

    def test_duplicate_registration_has_independent_handles(self) -> None:
        calls = []

        def callback(event) -> None:
            calls.append(event["name"])

        first = self.register(callback)
        self.register(callback)
        _run_export_chrome_trace_callbacks({"name": "first"})
        self.assertEqual(calls, ["first", "first"])

        first.remove()
        _run_export_chrome_trace_callbacks({"name": "second"})
        self.assertEqual(calls, ["first", "first", "second"])

    def test_callbacks_run_in_order_and_isolate_errors(self) -> None:
        event: dict[str, Any] = {"args": {}}

        def first(event) -> None:
            event["args"]["order"] = ["first"]

        def failing(event) -> None:
            event["args"]["partial"] = True
            raise RuntimeError("expected failure")

        def last(event) -> None:
            event["args"]["order"].append("last")

        self.register(first)
        self.register(failing)
        self.register(last)

        with self.assertLogs("torch.profiler.profiler", level="WARNING"):
            _run_export_chrome_trace_callbacks(event)

        self.assertEqual(event["args"]["order"], ["first", "last"])
        self.assertTrue(event["args"]["partial"])

    def test_legacy_event_list_streams_callbacks(self) -> None:
        events = EventList(
            [
                FunctionEvent(
                    id=0,
                    name="event",
                    trace_name="event",
                    thread=1,
                    start_us=2,
                    end_us=3,
                )
            ]
        )

        def annotate(event) -> None:
            event["args"]["callback_marker"] = True

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = os.path.join(tmpdir, "trace.json")
            events.export_chrome_trace(trace_path, event_callback=annotate)
            with open(trace_path) as trace:
                data = json.load(trace)

        self.assertTrue(data[0]["args"]["callback_marker"])

    @unittest.skipUnless(kineto_available(), "Kineto is required")
    def test_export_streams_events_through_callback(self) -> None:
        def annotate(event) -> None:
            self.assertNotIn("traceEvents", event)
            event.setdefault("args", {})["callback_marker"] = True

        self.register(annotate)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            trace_path = f.name

        try:
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                x = torch.randn(10, 10)
                _ = x + x

            with mock.patch(
                "json.load",
                side_effect=AssertionError("export must not read the trace back"),
            ):
                prof.export_chrome_trace(trace_path)

            with open(trace_path) as f:
                data = json.load(f)

            self.assertGreater(len(data["traceEvents"]), 0)
            self.assertTrue(
                all(
                    event.get("args", {}).get("callback_marker")
                    for event in data["traceEvents"]
                )
            )
        finally:
            os.unlink(trace_path)


if __name__ == "__main__":
    run_tests()
