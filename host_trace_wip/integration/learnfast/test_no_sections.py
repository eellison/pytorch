# test_no_sections.py: with no_sections imported, a redispatch that would be a section (sum's split reduction at many
# rows, test_a_split_reduction_is_a_section's case) is refused, the call traces again, every call is bitwise; and
# torch._C._host_trace_section_entries stays True.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.cuda._host_trace_replay import HostTraceReplay
from torch.testing._internal.common_utils import run_tests, TestCase
import no_sections


class TestNoSections(TestCase):
    def test_a_split_reduction_retraces(self):
        def fn(x):
            return (x * 2).sum(0) + 1

        before = no_sections.REFUSED["sections"]
        f = HostTraceReplay(fn)
        for t in (100, 300, 5000, 6000, 100, 5000):
            x = torch.randn(t, 64, device="cuda")
            self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        self.assertTrue(getattr(torch._C, "_host_trace_section_entries", None))
        self.assertEqual((f.traces, f.redispatches, f.sections), (2, 0, 0))  # the retrace's warm-up call counts as eager, as in test_a_section_traces_again_where_the_native_replay_takes_none
        self.assertGreaterEqual(no_sections.REFUSED["sections"] - before, 1)
        ((cause, n),) = f.retrace_causes.items()
        self.assertEqual((cause[0], cause[1], n), ("dispatch", "aten.sum.dim_IntList", 1))
        self.assertIn("section entries disabled", cause[4] or "")

    def test_a_same_chain_redispatch_is_unaffected(self):
        # the rms norm's b == 1 flip redispatches without a section (test_replay's
        # test_a_redo_agrees_with_the_variants_layout_under_its_guards)
        def fn(x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

        f = HostTraceReplay(fn)
        for b in (2, 3, 1):
            x = torch.randn(b, 64, device="cuda")
            self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        self.assertEqual((f.traces, f.redispatches), (1, 1))


if __name__ == "__main__":
    run_tests()
