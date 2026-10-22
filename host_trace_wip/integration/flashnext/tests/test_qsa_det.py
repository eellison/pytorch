# The QSA deterministic-order diagnosis config (scripts/qsa_det.py), checked before use (coordinator 23:3x):
# (1) at KV <= 2048 (< 512 visible blocks: every visible block selected) the stock top-k output is already ascending with a
#     -1 tail, so the reorder changes nothing there; (2) the reorder gives each row's selected set in ascending order, the
#     same bytes whatever the input order (eager twice equal); (3) top-k + reorder under host-trace replay (the ports +
#     the Triton route) is bitwise equal to eager at long and short lengths.
#   gpu3.sh t_qsadet bash python_fn.sh tests/test_qsa_det.py
import collections
import os
import sys

import torch

import torch.cuda._host_trace as ht
from torch.cuda._host_trace_replay import HostTraceReplay
from torch.cuda._host_trace_tape import EagerCall
from torch.testing._internal.common_utils import run_tests, TestCase

import vllm._custom_ops  # noqa: F401

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
sys.path.insert(0, os.path.join(HERE, "ports"))
import qsa_det  # noqa: E402

qsa_det.install()
import fn_ops_trace  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops import qsa as Q  # noqa: E402

DECLINES = collections.Counter()
_init = ht.Declined.__init__


def _rec(exc, *a, **k):
    _init(exc, *a, **k)
    DECLINES[str(exc)[:300]] += 1


ht.Declined.__init__ = _rec
K = 512
COLS = 4096  # columns at max_model_len 16384 / compress 4
WS = torch.zeros(1 << 20, device="cuda", dtype=torch.uint8)


def topk(logits, lengths, out, det, ws=None):
    op = torch.ops._C.cooperative_topk if logits.shape[0] <= 32 else torch.ops._C.persistent_topk
    if det:
        Q._fn_tie_break(logits, lengths)
    op(logits, lengths, out, WS if ws is None else ws, K, logits.shape[1])
    if det:
        Q._fn_det_order(out, logits.shape[1])


def inputs(rows, lo, hi, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(rows, COLS, device="cuda", generator=g)
    lengths = torch.randint(lo, hi + 1, (rows,), device="cuda", generator=g, dtype=torch.int32)
    # as qsa_mqa_paged leaves them: columns past a row's visible blocks are -inf
    logits.masked_fill_(torch.arange(COLS, device="cuda")[None, :] >= lengths[:, None], float("-inf"))
    return logits, lengths, torch.full((rows, K), -7, device="cuda", dtype=torch.int32)


def expected(logits, lengths, out_stock):
    # the row's selected set, ascending, -1 tail
    want = torch.full_like(out_stock, -1)
    for r in range(out_stock.shape[0]):
        s = out_stock[r][out_stock[r] >= 0].sort().values
        want[r, : s.numel()] = s
    return want


class TestQsaDet(TestCase):
    def test_zero_ties_deterministic(self):
        # the indexer's relu-sum scores: exact zeros; rows with > K visible blocks but < K positive scores tie at 0
        for i, rows in enumerate((1, 4, 8, 32, 33, 64)):
            g = torch.Generator(device="cuda").manual_seed(300 + i)
            base = torch.randn(rows, COLS, device="cuda", generator=g).clamp_min(0)  # ~half exact zeros
            lengths = torch.randint(K + 1, 2 * K - 50, (rows,), device="cuda", generator=g, dtype=torch.int32)
            base.masked_fill_(torch.arange(COLS, device="cuda")[None, :] >= lengths[:, None], float("-inf"))
            outs = []
            for _ in range(3):
                o = torch.full((rows, K), -7, device="cuda", dtype=torch.int32)
                topk(base.clone(), lengths, o, det=True)
                outs.append(o)
            self.assertEqual(outs[0], outs[1], atol=0, rtol=0, msg=f"rows {rows}")
            self.assertEqual(outs[0], outs[2], atol=0, rtol=0, msg=f"rows {rows}")
            want = torch.full((rows, K), -1, device="cuda", dtype=torch.int32)
            for r in range(rows):
                n = int(lengths[r])
                pos = (base[r, :n] > 0).nonzero().flatten()
                zeros = (base[r, :n] == 0).nonzero().flatten()
                sel = torch.cat([pos, zeros[: K - pos.numel()]]) if pos.numel() < K else base[r, :n].topk(K).indices
                want[r] = sel.sort().values.to(torch.int32)
            self.assertEqual(outs[0], want, atol=0, rtol=0, msg=f"rows {rows}: positives + lowest-index zeros")
        print("zero ties: selection deterministic (positives + lowest-index zeros), identical over 3 runs", flush=True)

    def test_short_kv_stock_order_is_ascending(self):
        for i, rows in enumerate((1, 3, 4, 8, 16, 32, 33, 64, 200)):
            logits, lengths, out = inputs(rows, 1, K - 1, i)  # fewer visible blocks than k: all selected
            topk(logits, lengths, out, det=False)
            ar = torch.arange(K, device="cuda", dtype=torch.int32)[None, :].expand(rows, K)
            want = torch.where(ar < lengths[:, None], ar, torch.full_like(ar, -1))
            self.assertEqual(out, want, atol=0, rtol=0, msg=f"stock short-KV order at {rows} rows")
            out2 = out.clone()
            Q._fn_det_order(out2, COLS)
            self.assertEqual(out2, out, atol=0, rtol=0, msg=f"reorder changed a short-KV row set at {rows} rows")
            # the full diagnosis path (tie-break + top-k + reorder) on relu-like scores (exact zeros) at short KV: same output
            z = logits.clamp_min(0).masked_fill(logits == float("-inf"), float("-inf"))
            o_stock, o_det = out.clone().fill_(-7), out.clone().fill_(-7)
            topk(z.clone(), lengths, o_stock, det=False)
            topk(z.clone(), lengths, o_det, det=True)
            self.assertEqual(o_det, o_stock, atol=0, rtol=0, msg=f"diagnosis path changed a short-KV row at {rows} rows")
        print("short KV: stock order ascending with -1 tail; reorder and tie-break + reorder are the identity (incl. exact-zero scores)", flush=True)

    def test_long_kv_reorder_deterministic(self):
        for i, rows in enumerate((1, 4, 8, 32, 33, 64)):
            logits, lengths, out = inputs(rows, K, COLS, 100 + i)
            outs = []
            for _ in range(3):
                o = out.clone()
                topk(logits, lengths, o, det=True)
                outs.append(o)
            stock = out.clone()
            topk(logits, lengths, stock, det=False)
            self.assertEqual(outs[0], expected(logits, lengths, stock), atol=0, rtol=0, msg=f"rows {rows}")
            self.assertEqual(outs[0], outs[1], atol=0, rtol=0)
            self.assertEqual(outs[0], outs[2], atol=0, rtol=0)
        print("long KV: reorder = the selected set ascending, identical over 3 runs", flush=True)

    def test_traced_topk_and_reorder_bitwise(self):
        fn_ops_trace.install()
        DECLINES.clear()
        r = HostTraceReplay(lambda logits, lengths, out, ws: topk(logits, lengths, out, True, ws))
        for i, (rows, lo, hi) in enumerate(((4, K, COLS), (8, K, COLS), (4, 1, K - 1), (16, K, COLS), (64, K, COLS),
                                            (100, 1, COLS), (4, K, COLS), (64, K, COLS))):
            logits, lengths, out = inputs(rows, lo, hi, 200 + i)
            ref = out.clone()
            topk(logits, lengths, ref, True)
            r(logits, lengths, out, WS)
            torch.cuda.synchronize()
            self.assertEqual(out, ref, atol=0, rtol=0, msg=f"rows {rows} lengths {lo}-{hi}")
        steps = [rec.name for v in r.variants for _, rec in v.tape.launches if isinstance(rec, EagerCall)]
        print(f"traced top-k + reorder bitwise; traces {r.traces} replays {r.replays} eager {r.eager} eager steps {steps} "
              f"declines {dict(DECLINES)}", flush=True)
        self.assertEqual(steps, [])
        self.assertFalse(DECLINES, DECLINES)


if __name__ == "__main__":
    run_tests()
