# Repro for the core agent (learnfast 10-15): flipping torch._C._host_trace_section_entries after a section entry exists
# desyncs Python's form sites (_form drops the selectors) from the native arms, and a new form's zip(strict=True) raises.
from unittest import mock

import torch
import torch.cuda._host_trace_replay as _host_trace_replay  # noqa: F401
from torch.cuda._host_trace_replay import HostTraceReplay
from torch.testing._internal.common_utils import run_tests, TestCase


class TestSectionFlagDesync(TestCase):
    def test_flipping_section_entries_off_after_a_section(self):
        def fn(x):
            return (x * 2).sum(0) + 1

        f = HostTraceReplay(fn)
        with mock.patch.object(torch._C, "_host_trace_section_entries", True, create=True):
            for t in (100, 300, 5000, 6000):
                x = torch.randn(t, 64, device="cuda")
                self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        print("before flip", f.traces, f.redispatches, f.sections, dict(f.retrace_causes))
        self.assertEqual((f.traces, f.redispatches, f.sections), (1, 1, 1))
        with mock.patch.object(torch._C, "_host_trace_section_entries", False, create=True):
            for t in (100, 5000, 7000, 150, 9000):
                x = torch.randn(t, 64, device="cuda")
                self.assertEqual(f(x), fn(x), atol=0, rtol=0)

    def test_flipping_section_entries_off_before_a_new_form(self):
        # two sums, each a section at many rows: forms [0, 0], [1, 0] and [0, 1] made with the flag on; after the flip,
        # (many, many) needs the new form [1, 1] of entries that exist (no redispatch), which _form builds without the
        # selectors as sites while the native segment's arms still count them
        def fn(x, y):
            return (x * 2).sum(0) + (y * 2).sum(0)

        f = HostTraceReplay(fn)
        small, big = 100, 5000
        with mock.patch.object(torch._C, "_host_trace_section_entries", True, create=True):
            for a, b in ((small, small), (300, 300), (big, small), (6000, small), (small, big), (small, 6000)):
                x, y = torch.randn(a, 64, device="cuda"), torch.randn(b, 64, device="cuda")
                self.assertEqual(f(x, y), fn(x, y), atol=0, rtol=0)
        print("before flip", f.traces, f.redispatches, f.sections, dict(f.retrace_causes))
        with mock.patch.object(torch._C, "_host_trace_section_entries", False, create=True):
            x, y = torch.randn(big, 64, device="cuda"), torch.randn(big, 64, device="cuda")
            self.assertEqual(f(x, y), fn(x, y), atol=0, rtol=0)


if __name__ == "__main__":
    run_tests()
