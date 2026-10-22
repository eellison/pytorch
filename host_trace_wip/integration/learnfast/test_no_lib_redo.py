# test_no_lib_redo.py: a custom op whose Python body branches on a size (an op-owned guard) redispatches without the
# helper and retraces with it, bitwise either way; an aten op's redispatch (rms norm's b == 1 flip) is unaffected.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.cuda._host_trace_replay import HostTraceReplay
from torch.testing._internal.common_utils import run_tests, TestCase


@torch.library.custom_op("lf_test::branchy", mutates_args=())
def branchy(x: torch.Tensor) -> torch.Tensor:
    return x * 2 if x.shape[0] > 4 else x + 1


@branchy.register_fake
def _(x):
    return torch.empty_like(x)


def fn(x):
    return branchy(x) + 1


def run(sizes):
    f = HostTraceReplay(fn)
    out = []
    for m in sizes:
        x = torch.randn(m, 64, device="cuda")
        out.append(torch.equal(f(x), fn(x)))
    return f, out


class TestNoLibRedo(TestCase):
    def test_custom_op_redispatch_then_refused(self):
        import torch.cuda._host_trace_redispatch as rd
        f0, ok0 = run((8, 2, 9, 3))  # without the helper (installed below)
        self.assertTrue(all(ok0))
        print("baseline traces", f0.traces, "redispatches", f0.redispatches, flush=True)
        import no_lib_redo
        before = sum(no_lib_redo.REFUSED.values())
        f1, ok1 = run((8, 2, 9, 3))
        self.assertTrue(all(ok1))
        print("with helper traces", f1.traces, "redispatches", f1.redispatches, "refused", dict(no_lib_redo.REFUSED), flush=True)
        self.assertEqual(f1.redispatches, 0)
        if f0.redispatches:
            self.assertGreater(sum(no_lib_redo.REFUSED.values()) - before, 0)
            self.assertGreater(f1.traces, f0.traces)

    def test_aten_redispatch_is_unaffected(self):
        import no_lib_redo  # noqa: F401

        def g(x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

        f = HostTraceReplay(g)
        for b in (2, 3, 1):
            x = torch.randn(b, 64, device="cuda")
            self.assertEqual(f(x), g(x), atol=0, rtol=0)
        self.assertEqual((f.traces, f.redispatches), (1, 1))


if __name__ == "__main__":
    run_tests()
