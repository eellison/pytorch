# Owner(s): ["module: cuda graphs"]
# HostTraceReplay(fullgraph=True): a graph break or a call that falls back raises EagerFallback; a learning
# variant is served and counted. forbid_learners=True raises on an opaque call that does not learn at its trace.

import unittest
from unittest import mock

import torch
from torch.cuda._host_trace import EagerFallback
from torch.cuda._host_trace_replay import HostTraceReplay
from torch.cuda._host_trace_tape import EagerCall
from torch.testing._internal.common_utils import requires_cuda_python_bindings, run_tests, TEST_CUDA, TestCase
from torch.utils._triton import has_triton


def no_learn():
    # a provider whose keys never bind: a variant with its opaque call learns for good
    import test_cuda_host_trace_opaque as ot

    class NoLearn(ot.TableProvider):
        def learn(self, key, args, kwargs, operands):
            return None

    return NoLearn()


def stacked(x, o):
    # aten.stack.out has no route: one eager step; the mul is a traced launch
    return torch.stack([x, x], out=o) * 2


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
class TestFullgraph(TestCase):
    def test_a_clean_entry_passes(self):
        f = HostTraceReplay(lambda x: x * 3 + 1, fullgraph=True)
        for n in (100, 100, 100, 300, 7):
            x = torch.randn(n, device="cuda")
            self.assertEqual(f(x), x * 3 + 1)
        # the first call is the warm-up, the only eager one
        self.assertEqual((f.traces + f.replays, f.eager, f.declines, f.learners), (4, 1, [], 0))
        self.assertGreaterEqual(f.replays, 1)

    def test_an_eager_step_raises_at_the_trace(self):
        x, o = torch.randn(4, 6, device="cuda"), torch.empty(2, 4, 6, device="cuda")
        f = HostTraceReplay(stacked, opaque=())
        for _ in range(3):
            self.assertEqual(f(x, o), torch.stack([x, x]) * 2)
        (v,) = f.variants
        self.assertEqual([r.name for _, r in v.tape.launches if isinstance(r, EagerCall)], ["aten.stack.out"])
        g = HostTraceReplay(stacked, fullgraph=True, opaque=())
        self.assertEqual(g(x, o), torch.stack([x, x]) * 2)
        for _ in range(2):
            with self.assertRaisesRegex(EagerFallback, r"eager steps in the tape: aten\.stack\.out \(aten\.stack\.out's pointwise host declines"):
                g(x, o)
        self.assertEqual((g.variants, g.replays), ([], 0))

    def test_a_decline_raises(self):
        f = HostTraceReplay(lambda xs: xs[0] * 3, fullgraph=True)
        x = torch.randn(100, device="cuda")
        self.assertEqual(f([x]), x * 3)
        # every call of the class traces and raises: none is a declined class that runs eagerly
        for _ in range(2):
            with self.assertRaisesRegex(EagerFallback, "arg0 holds a tensor inside a list"):
                f([x])
        self.assertEqual((f.traces, f.eager), (2, 1))

    def test_a_whole_call_fallback_raises(self):
        g = HostTraceReplay(lambda x, *, s: x * s, fullgraph=True)
        with self.assertRaisesRegex(EagerFallback, "keyword arguments that do not bind"):
            g(torch.randn(100, device="cuda"), s=4)
        self.assertEqual(g.eager, 0)

    @unittest.skipIf(not has_triton(), "requires triton")
    def test_a_learner_is_served_and_counted(self):
        import test_cuda_host_trace_opaque as ot

        x, y = torch.randn(512, device="cuda"), torch.randn(512, device="cuda")
        for kw in ({}, {"fullgraph": True}):
            # a learning variant is not a graph break: its mul runs eagerly at every call, counted
            f = HostTraceReplay(ot.mul_chain, opaque=(no_learn(),), **kw)
            for _ in range(4):
                self.assertEqual(f(x, y), ot.mul_chain(x, y))
            self.assertEqual((f.traces, f.replays, f.eager, f.learners, f.learner_calls), (1, 2, 1, 1, 2))
            self.assertTrue(all(v.learns for v in f.variants))

    @unittest.skipIf(not has_triton(), "requires triton")
    def test_an_opaque_call_learns_at_the_trace(self):
        import test_cuda_host_trace_opaque as ot

        p = ot.TableProvider()
        f = HostTraceReplay(ot.mul_chain, opaque=(p,), fullgraph=True, forbid_learners=True)
        for n in (512, 512, 512, 640):
            x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            self.assertEqual(f(x, y), ot.mul_chain(x, y))
        # the trace learned its key out of band: no variant learns, no relower, the new key learns at its miss
        self.assertEqual((f.traces, f.relowers, f.replays, f.eager, f.learned), (1, 0, 2, 1, 2))
        self.assertEqual((f.learners, f.learner_calls), (0, 0))
        self.assertFalse(any(v.learns for v in f.variants))

    @unittest.skipIf(not has_triton(), "requires triton")
    def test_forbid_learners_raises(self):
        import test_cuda_host_trace_opaque as ot

        x, y = torch.randn(512, device="cuda"), torch.randn(512, device="cuda")
        f = HostTraceReplay(ot.mul_chain, opaque=(no_learn(),), forbid_learners=True)
        self.assertEqual(f(x, y), ot.mul_chain(x, y))
        for _ in range(2):
            with self.assertRaisesRegex(EagerFallback, r"keys neither bind nor learn at the trace: aten\.mul\.Tensor; forbid_learners=True"):
                f(x, y)
        self.assertEqual((f.variants, f.learners, f.learner_calls), ([], 0, 0))

    @unittest.skipIf(not has_triton(), "requires triton")
    @mock.patch("torch.cuda._host_trace_replay.oob_learn", False)
    def test_a_key_refused_at_serving_raises(self):
        import test_cuda_host_trace_opaque as ot

        f = HostTraceReplay(ot.mul_chain, opaque=(ot.TableProvider(),), fullgraph=True)
        x, y = torch.randn(512, device="cuda"), torch.randn(512, device="cuda")
        for _ in range(3):
            self.assertEqual(f(x, y), ot.mul_chain(x, y))
        # the provider refuses a strided operand's key at the learner's run; the trace that follows is an eager step
        y = torch.randn(1024, device="cuda")[::2]
        for _ in range(2):
            self.assertEqual(f(x, y), ot.mul_chain(x, y))
        self.assertEqual((f.traces, f.learners, f.learner_calls), (2, 2, 2))
        for _ in range(2):
            with self.assertRaisesRegex(EagerFallback, r"eager steps in the tape: aten\.mul\.Tensor \(.*TableProvider: refused\); fullgraph=True"):
                f(x, y)
        self.assertEqual((f.traces, f.learner_calls), (4, 2))
        g = HostTraceReplay(ot.mul_chain, opaque=(ot.TableProvider(),), fullgraph=True, forbid_learners=True)
        for _ in range(3):
            self.assertEqual(g(x, x), ot.mul_chain(x, x))
        with self.assertRaisesRegex(EagerFallback, r"keys neither bind nor learn at the trace: aten\.mul\.Tensor; forbid_learners=True"):
            g(x, y)


def setUpModule():
    from torch.cuda import _host_trace_hint_audit

    _host_trace_hint_audit.enable_for_tests()


if __name__ == "__main__":
    run_tests()
