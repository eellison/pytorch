# The QSA indexer top-k ports (ports/fn_ops_trace.py) under host-trace replay: _C.cooperative_topk at row counts across
# its cluster-size ladder (1..33 rows: cluster 16 / 8 / 4) and k 512 / 2048, and _C.persistent_topk's FilteredTopK path
# (> 32 rows), random logits with ragged lengths. Each
# traced call is checked against the real op on the same inputs (eager twice first: the reference must reproduce). The
# kernels write each row's selected indices in a run-dependent order (atomic slots: eager twice differs in 60-87% of the
# positions), so rows are compared as sorted index sets; the launch itself is byte-checked against the real op's by the port.
#   gpu3.sh t_ports bash python_fn.sh tests/test_fn_ops_trace.py
import collections
import os
import sys

import torch

import torch.cuda._host_trace as ht
from torch.cuda._host_trace_replay import HostTraceReplay
from torch.cuda._host_trace_tape import EagerCall
from torch.testing._internal.common_utils import run_tests, TestCase

import vllm._custom_ops  # noqa: F401  (registers the _C ops)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ports"))
import fn_ops_trace  # noqa: E402

DECLINES = collections.Counter()
_declined_init = ht.Declined.__init__


def _record_decline(exc, *args, **kwargs):
    _declined_init(exc, *args, **kwargs)
    DECLINES[str(exc)[:400]] += 1


ht.Declined.__init__ = _record_decline
ROWS = [int(x) for x in os.environ.get("FN_TEST_ROWS", "1,4,2,5,8,3,9,16,32,33,7,1").split(",")]
ROWS_P = [int(x) for x in os.environ.get("FN_TEST_ROWS_P", "33,64,100,300,64,1000,40").split(",")]
COLS = int(os.environ.get("FN_TEST_COLS", "4096"))
WS_BYTES = 64 * 2048 * 8
OP = {"coop": lambda: torch.ops._C.cooperative_topk, "persistent": lambda: torch.ops._C.persistent_topk}


def topk_step(logits, lengths, out, ws, k, which="coop"):
    OP[which]()(logits, lengths, out, ws, k, logits.shape[1])


def args_for(rows, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(rows, COLS, device="cuda", generator=g)
    lengths = torch.randint(k // 2, COLS + 1, (rows,), device="cuda", generator=g, dtype=torch.int32)
    out = torch.full((rows, k), -7, device="cuda", dtype=torch.int32)
    ws = torch.zeros(WS_BYTES, device="cuda", dtype=torch.uint8)
    return [logits, lengths, out, ws]


class TestFnOpsTrace(TestCase):
    def test_cooperative_topk_512(self):
        self._run(512, "coop", ROWS)

    def test_cooperative_topk_2048(self):
        self._run(2048, "coop", ROWS)

    def test_persistent_topk_512(self):
        self._run(512, "persistent", ROWS_P)

    def test_persistent_topk_2048(self):
        self._run(2048, "persistent", ROWS_P)

    def _run(self, k, which, rows_list):
        fn_ops_trace.install()
        DECLINES.clear()
        r = HostTraceReplay(lambda logits, lengths, out, ws: topk_step(logits, lengths, out, ws, k, which))
        for i, rows in enumerate(rows_list):
            args = args_for(rows, k, i)
            refs = []
            for _ in range(2):
                ref = [a.clone() for a in args]
                topk_step(*ref, k, which)
                refs.append(ref[2].sort(dim=1).values)
            self.assertEqual(refs[0], refs[1], atol=0, rtol=0, msg=f"eager not reproducible at {rows} rows")
            r(*args)
            torch.cuda.synchronize()
            self.assertEqual(args[2].sort(dim=1).values, refs[0], atol=0, rtol=0, msg=f"k {k} rows {rows}")
            print(f"{which} k {k} rows {rows} equal as sorted index sets", flush=True)
        print(f"traces {r.traces} replays {r.replays} redispatches {r.redispatches} eager {r.eager}", flush=True)
        steps = [rec.name for v in r.variants for _, rec in v.tape.launches if isinstance(rec, EagerCall)]
        print("eager steps:", steps, "declines:", dict(DECLINES), flush=True)
        self.assertEqual(steps, [])
        self.assertFalse(DECLINES, DECLINES)


if __name__ == "__main__":
    run_tests()
