# Owner(s): ["module: cuda graphs"]
"""The hint audit (torch.cuda._host_trace_hint_audit) on the IR backend's
symbols alone: no trace, no device."""

from unittest import mock

from torch.cuda import _host_trace_hint_audit as audit, _host_trace_ir as ir
from torch.cuda._host_trace_tape import _guard_each, _hint
from torch.testing._internal.common_utils import run_tests, TestCase


def reads(x):
    return _hint(x) * 2


def reads_and_branches(x):
    # the branch's condition is the guard
    return 1 if _hint(x) > 3 and bool(x > 3) else 0


def reads_and_pins(x):
    h = _hint(x)
    _guard_each([x == h])
    return h


def reads_and_checks_positive(x):
    return 1 if _hint(x) > 0 and bool(x > 0) else 0


def reads_and_skips_zero(x):
    # the guard only where the hint is 0: an unguarded read elsewhere
    return 1 if _hint(x) == 0 and bool(x == 0) else 0


class TestHintAudit(TestCase):
    def setUp(self):
        super().setUp()
        audit.enable_for_tests()
        # this file's readers stand for the trace's code
        exempt = mock.patch.object(audit, "exempt", ())
        exempt.start()
        self.addCleanup(exempt.stop)
        self.env = ir.Env()
        self.a = self.env.symbol(5, "arg0.size(0)", positive=True)
        self.b = self.env.symbol(7, "arg0.size(1)", positive=True)
        self.c = self.env.symbol(9, "arg0.stride(0)")

    def raises(self, fn, *args):
        with mock.patch.object(audit, "mode", "strict"), self.assertRaises(audit.HintAuditError):
            fn(*args)

    def freeze(self):
        audit.freeze(self.env, len(self.env.records))

    def test_a_pinned_read_passes(self):
        reads_and_pins(self.c + self.b)
        self.freeze()

    def test_a_read_whose_symbols_are_pinned_passes(self):
        reads_and_pins(self.c)
        reads(self.c * 3)
        self.freeze()

    def test_a_branch_guarded_in_its_frame_passes(self):
        reads_and_branches(self.c)
        self.freeze()

    def test_a_relation_the_domains_decide_passes(self):
        # a size is positive by its declared domain (the entry checks it)
        reads_and_checks_positive(self.a)
        self.freeze()

    def test_an_unguarded_read_raises_at_the_freeze(self):
        reads(self.a * self.b)
        self.raises(self.freeze)

    def test_a_guard_outside_the_reading_frame_raises(self):
        reads(self.c)
        bool(self.c > 1)
        self.raises(self.freeze)

    def test_a_guard_only_at_one_hint_raises(self):
        reads_and_skips_zero(self.c)
        self.raises(self.freeze)

    def test_a_read_after_the_freeze_raises_at_once(self):
        self.freeze()
        self.raises(reads, self.c)

    def test_a_read_after_the_freeze_of_a_pinned_value_passes(self):
        reads_and_pins(self.c)
        self.freeze()
        reads(self.c)

    def test_a_guard_after_the_freeze_raises(self):
        self.freeze()
        self.raises(bool, self.c > 1)

    def test_a_rolled_back_read_is_dropped(self):
        n = len(self.env.records)
        reads(self.c)
        self.env.forget(n)
        self.freeze()

    def test_a_read_of_another_traces_symbol_raises(self):
        other = ir.Env().symbol(3, "arg1.size(0)")
        with mock.patch.object(ir.ACTIVE, "trace", mock.Mock(shape_env=self.env), create=True):
            self.raises(reads, other)

    def test_an_allowlisted_read_passes(self):
        key = (audit._key(reads.__code__)[0], "reads")
        with mock.patch.dict(audit.ALLOWLIST, {key: audit.Allowed("safe", "a test")}):
            reads(self.c)
            self.freeze()

    def test_a_known_offender_does_not_raise(self):
        key = (audit._key(reads.__code__)[0], "reads")
        with mock.patch.dict(audit.OFFENDERS, {key: "a test"}), mock.patch.object(audit, "mode", "strict"):
            reads(self.c)
            self.freeze()

    def test_a_warm_up_value_as_a_constant_raises(self):
        self.raises(self.env.ctx.const, audit._WarmUpInt(2))

    def test_a_constant_is_no_read(self):
        reads(self.c - self.c + 4)
        self.freeze()


if __name__ == "__main__":
    run_tests()
