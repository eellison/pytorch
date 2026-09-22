# Owner(s): ["module: cuda graphs"]
"""Control flow of the host-trace oracle (torch/testing/_internal/host_trace_oracle.py)
on the CPU, with the native entry, eager and the device RNG state faked: the runtime
team's review of the retirement's first cut (integration_r36/retirement_review,
CONTROL_PROBE.json) found `try_check` serving a hit twice and both oracle paths
erasing the served call's RNG advancement. A served call runs once, mutates the
caller's arguments once, and leaves the RNG state where it left it; eager's own
draw on the copies is undone; a miss runs nothing of eager."""

from unittest import mock

import torch
from torch.testing._internal import host_trace_oracle as oracle
from torch.testing._internal.common_utils import run_tests, TestCase


class _Rng:
    """A fake device RNG state: an int the fakes advance by 7 per served call."""

    def __init__(self):
        self.state = 0
        self.sets = []

    def get(self, device=None):
        return self.state

    def set(self, state, device=None):
        self.sets.append(state)
        self.state = state


class _Lowered:
    unwritten_outputs = ()


class _Native:
    """A native entry: increments the first argument in place, draws 7."""

    lowered = _Lowered()

    def __init__(self, rng, miss=False, mutate=True):
        self.rng = rng
        self.miss = miss
        self.mutate = mutate
        self.calls = 0
        self.miss_log = [(None, "guard failed: the fake predicate")]

    def __call__(self, *args):
        self.calls += 1
        if self.miss:
            raise RuntimeError("the call misses all 1 variants of the fake entry")
        if self.mutate:
            args[0].add_(1)
        self.rng.state += 7
        return args[0] + 1


class TestHostTraceOracleControl(TestCase):
    def setUp(self):
        super().setUp()
        self.rng = _Rng()
        self.eager_calls = 0
        patchers = [
            mock.patch.object(torch.cuda, "get_rng_state", self.rng.get),
            mock.patch.object(torch.cuda, "set_rng_state", self.rng.set),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def _eager(self, x):
        self.eager_calls += 1
        x.add_(1)
        self.rng.state += 7
        return x + 1

    def _oracle(self, native):
        o = oracle.Oracle.__new__(oracle.Oracle)
        o.fn = self._eager
        o.tape = None
        o.device = 0
        o.native = native
        o.refused = None
        return o

    def test_a_served_try_check_runs_the_native_entry_once(self):
        native = _Native(self.rng)
        x = torch.zeros(4)
        out = self._oracle(native).try_check((x,))
        self.assertEqual(native.calls, 1)
        self.assertEqual(x, torch.ones(4))  # the caller's argument mutated once
        self.assertEqual(out[0], torch.full((4,), 2.0))
        self.assertEqual(self.eager_calls, 1)

    def test_a_missed_try_check_runs_no_eager_and_serves_nothing(self):
        native = _Native(self.rng, miss=True)
        x = torch.zeros(4)
        self.assertIsNone(self._oracle(native).try_check((x,)))
        self.assertEqual(native.calls, 1)
        self.assertEqual(x, torch.zeros(4))
        # eager ran on its copies before the entry (the reference); its draw is undone
        self.assertEqual(self.rng.state, 0)

    def test_check_keeps_the_served_calls_rng_advancement(self):
        native = _Native(self.rng)
        self.rng.state = 100
        self._oracle(native).check((torch.zeros(4),))
        # eager drew 7 on the copies and was rewound; the served call's 7 stays
        self.assertEqual(self.rng.state, 107)
        self.assertEqual(self.rng.sets, [100])

    def test_check_with_a_reference_runs_no_eager(self):
        # a caller's reference stands for eager's outputs; the copies of the
        # arguments stay as made, so this is the form for a call that writes none
        native = _Native(self.rng, mutate=False)
        x = torch.zeros(4)
        self._oracle(native).check((x,), reference=torch.ones(4))
        self.assertEqual((native.calls, self.eager_calls), (1, 0))
        self.assertEqual(self.rng.state, 7)

    def test_the_variants_oracle_keeps_the_served_calls_rng_advancement(self):
        native = _Native(self.rng)
        test = self

        class Variant:
            lowered = _Lowered()
            device = 0
            fn = staticmethod(self._eager)

            def replay(self, args):
                return oracle._as_list(native(*args))

            def try_replay(self, args):
                return self.replay(args)

        Variant.native = native
        self.rng.state = 100
        x = torch.zeros(4)
        out = oracle.OracleVariant(Variant()).replay((x,))
        test.assertEqual(native.calls, 1)
        test.assertEqual(x, torch.ones(4))
        test.assertEqual(out[0], torch.full((4,), 2.0))
        test.assertEqual(self.rng.state, 107)


if __name__ == "__main__":
    run_tests()
