# Owner(s): ["module: cuda graphs"]

# memory="planned": offsets at two sizes, no overlap between live planned
# temporaries, bitwise outputs vs eager, one arena allocation per run.
import gc
import unittest

import sympy

import torch
from torch.cuda._host_trace_replay import HostTraceReplay
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TEST_CUDA, TestCase


def fn(x, w):
    # temporaries of three classes: n*4, 2n*4 and n*k*4 bytes; out is n*4
    a = x + 1
    b = a * 2
    c = torch.cat([a, b])
    d = c.relu()
    e = torch.outer(b, w)
    f = e.sum(1)
    return d[: x.numel()] + d[x.numel() :] + f


def layers(x, w1, w2, w3):
    # one weight per layer, each its own size symbol, equal at the trace
    h = x
    for w in (w1, w2, w3):
        h = torch.outer(h, w).sum(1) * 2
    return h


def halves(x):
    # temporaries of n//2 and n - n//2 elements: floors in the bytes
    n = x.shape[0]
    a = x[: n // 2] * 2
    b = x[n // 2 :] * 3
    return torch.cat([a + 1, b + 1])


def shrink(x):
    # cat and its double die before the n-element b: b fits their hole
    a = torch.cat([x, x]) * 2
    s = a.sum(0, keepdim=True)
    b = x * 3
    return b + s


def decoder(x):
    # three layers with an extend step's proportions: residual n, qkv 1.5n, gate_up 6n, act 3n
    h = x
    for _ in range(3):
        n = h.numel()
        r = h * 0.5
        qkv = torch.cat([r, r, r[: n // 2]]) * 2
        h = h + qkv[:n] + qkv[n // 2 : n // 2 + n]
        gu = torch.cat([h * 0.5] * 6) * 2
        h = h + (gu[: 3 * n] * gu[3 * n :]).view(3, n).sum(0)
    return h


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
class TestPlanned(TestCase):
    def call(self, f, n):
        x = torch.randn(n, device="cuda")
        w = torch.randn(64, device="cuda")
        return x, w, f(x, w)

    def placed(self, v, x, w):
        status, values = v.captured.lowered.compiled.evaluate_inputs((x, w))
        self.assertEqual(int(status), 0)
        return {k: (values[off], values[nb], seq, last, a) for k, (a, off, nb, seq, last) in v.memory.relocated.items()}

    @parametrize("memory", ["planned", "planned_no_reuse", "planned_no_split", "packed"])
    def test_offsets_two_sizes_no_overlap(self, memory):
        f = HostTraceReplay(fn, memory=memory)
        for n in (1000, 1000, 3000, 7000):
            self.call(f, n)
        self.assertEqual(len(f.variants), 1)
        (v,) = f.variants
        self.assertGreaterEqual(len(v.memory.relocated), 4)
        seen = []
        for n in (1000, 7000):
            x = torch.randn(n, device="cuda")
            w = torch.randn(64, device="cuda")
            placed = self.placed(v, x, w)
            seen.append({k: o for k, (o, *_) in placed.items()})
            for k, (off, nb, seq, last, a) in placed.items():
                self.assertEqual(off % 512, 0)
                for j, (off2, nb2, seq2, last2, a2) in placed.items():
                    if j <= k or a2 != a:
                        continue
                    if seq < last2 and seq2 < last:  # both live at once
                        self.assertTrue(off + nb <= off2 or off2 + nb2 <= off, (k, j, n))
        # the offsets scale with n: at least one moves between the sizes
        self.assertNotEqual(seen[0], seen[1])

    @parametrize("memory", ["planned", "planned_no_reuse", "planned_no_split", "packed"])
    @parametrize("n", [1000, 5000])
    def test_bitwise_and_one_arena(self, n, memory):
        f = HostTraceReplay(fn, memory=memory)
        g = HostTraceReplay(fn, memory="run_buffer")
        for m in (777, 777, 1234, n, 777):
            x = torch.randn(m, device="cuda")
            w = torch.randn(64, device="cuda")
            want = fn(x, w)
            self.assertEqual(f(x, w), want, atol=0, rtol=0)
            self.assertEqual(g(x, w), want, atol=0, rtol=0)
        for _ in range(3):
            self.assertEqual(f(x, w), want, atol=0, rtol=0)
        self.assertGreaterEqual(f.replays, 2)
        gc.collect()
        torch.cuda.synchronize()

        def requests(h):
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            h(x, w)
            return torch.cuda.memory_stats()["allocation.all.allocated"] - before

        # the output and one arena
        self.assertEqual(requests(f), 2)
        self.assertEqual(requests(f), requests(g))
        self.assertLess(requests(f), requests(fn))

    @parametrize("memory", ["planned", "packed"])
    def test_peak(self, memory):
        f = HostTraceReplay(fn, memory=memory)
        x = torch.randn(1 << 20, device="cuda")
        w = torch.randn(64, device="cuda")
        for _ in range(3):
            f(x, w)

        def peak(h):
            gc.collect()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            y = h(x, w)
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() - before, y.nbytes

        (p_eager, out), (p_plan, _) = peak(fn), peak(f)
        print("peak eager", p_eager, memory, p_plan)
        # every class here has one slot per simultaneously live member; the
        # arena holds them through the run, as a run buffer does, plus the output
        self.assertLessEqual(p_plan, p_eager + out + (1 << 20) * 4 * 3)


    def arena(self, v, *args):
        from torch.cuda._host_trace_memory import place

        status, values = v.captured.lowered.compiled.evaluate_inputs(args)
        self.assertEqual(int(status), 0)
        rel = v.memory.relocated
        (a,) = {r[0] for r in rel.values()}
        temps = sorted(((k, r[3], r[4]) for k, r in rel.items()), key=lambda t: t[1])
        first_fit = place(temps, [values[rel[k][2]] for k, _, _ in temps])[1]
        return values[v.captured.lowered.allocations[a].nbytes], first_fit

    def test_canonical_layers(self):
        from torch.cuda._host_trace_memory import _canonical

        f = HostTraceReplay(layers, memory="planned")
        x = torch.randn(1000, device="cuda")
        ws = [torch.randn(64, device="cuda") for _ in range(3)]
        for m in (1000, 1000, 3000):
            x = torch.randn(m, device="cuda")
            self.assertEqual(f(x, *ws), layers(x, *ws), atol=0, rtol=0)
        (v,) = f.variants
        lowered = v.captured.lowered
        exprs, renames = _canonical(lowered, list(v.memory.relocated))
        self.assertGreaterEqual(len(renames), 2)  # w2's and w3's size become w1's
        outer = [e for e in exprs.values() if len(e.free_symbols) == 2]
        self.assertGreaterEqual(len(outer), 3)
        self.assertEqual(len(set(outer)), 1)
        # the three layers' outer products share one hole, as first fit shares
        total, first_fit = self.arena(v, x, *ws)
        self.assertLessEqual(total, first_fit)
        traces = f.traces
        # a layer whose weight size differs breaks a rename: a miss, then a trace
        ws2 = [ws[0], torch.randn(32, device="cuda"), ws[2]]
        self.assertEqual(f(x, *ws2), layers(x, *ws2), atol=0, rtol=0)
        self.assertEqual(f.traces, traces + 1)
        g = HostTraceReplay(layers, memory="run_buffer")
        for m in (1000, 1000, 3000):
            g(torch.randn(m, device="cuda"), *ws)
        traces = g.traces
        g(x, *ws2)
        self.assertEqual(g.traces, traces)  # the miss is the rename's only

    @parametrize("memory", ["planned", "packed"])
    def test_floor_variables(self, memory):
        from torch.cuda._host_trace_memory import PackedArena

        f = HostTraceReplay(halves, memory=memory)
        for m in (1000, 1000, 3001, 4000):
            x = torch.randn(m, device="cuda")
            self.assertEqual(f(x), halves(x), atol=0, rtol=0)
        for m in (777, 5000, 5001):
            x = torch.randn(m, device="cuda")
            self.assertEqual(f(x), halves(x), atol=0, rtol=0)
        self.assertEqual(len(f.variants), 1)
        (v,) = f.variants
        self.assertTrue(any(isinstance(s, sympy.Function) for e in _exprs(v) for s in sympy.preorder_traversal(e)))
        # packed as sums of padded sizes, not size classes
        arenas = [a for a, _ in v.memory.arenas]
        self.assertTrue(arenas and all(isinstance(a, PackedArena) for a in arenas), arenas)

    def test_dominated_reuse(self):
        f = HostTraceReplay(shrink, memory="planned")
        for m in (1000, 1000, 3000):
            x = torch.randn(m, device="cuda")
            self.assertEqual(f(x), shrink(x), atol=0, rtol=0)
        (v,) = f.variants
        for m in (1000, 9000):
            x = torch.randn(m, device="cuda")
            total, first_fit = self.arena(v, x)
            self.assertLessEqual(total, first_fit, m)
            self.assertEqual(f(x), shrink(x), atol=0, rtol=0)
        # b's 4n bytes go where the cat's 8n were: no class of its own
        rel = v.memory.relocated
        by_bytes = {}
        status, values = v.captured.lowered.compiled.evaluate_inputs((x,))
        for k, (a, off, nb, seq, last) in rel.items():
            by_bytes.setdefault(values[nb], []).append(values[off])
        n4 = x.numel() * 4
        self.assertIn(n4, by_bytes)
        inside = [o for o in by_bytes[n4] for big in by_bytes.get(2 * n4, ()) if big <= o and o + n4 <= big + 2 * n4]
        self.assertTrue(inside, by_bytes)


    def test_greedy_by_size(self):
        # largest first reaches the most bytes live at once, where tape-order first fit does not
        f = HostTraceReplay(decoder, memory="planned")
        for m in (4096, 4096, 2048, 4096):
            x = torch.randn(m, device="cuda")
            self.assertEqual(f(x), decoder(x), atol=0, rtol=0)
        v = f.variants[0]
        total, first_fit = self.arena(v, x)
        status, values = v.captured.lowered.compiled.evaluate_inputs((x,))
        rel = v.memory.relocated
        pad = lambda n: -(-n // 512) * 512
        live = max(sum(pad(values[r[2]]) for r in rel.values() if r[3] <= s < r[4]) for *_, s, _ in rel.values())
        self.assertEqual(total, live)
        self.assertLess(total, first_fit)
        # an ordering proven only at the trace is a guard: where one fails, a miss and a trace, still bitwise
        for m in (12288, 1000, 7):
            x = torch.randn(m, device="cuda")
            self.assertEqual(f(x), decoder(x), atol=0, rtol=0)

    @parametrize("packed", [False, True])
    def test_a_tape_lowered_at_a_later_call_plans_at_its_values(self, packed):
        # the refused-key twin (_eager_sites) builds the trace's tape at a later call's arguments: its rows
        # are that call's, its leaves' hints the trace's. Planned at the hints, the arena gave the sizes
        # equal at the trace (36, 36) one symbol, whose condition the call (36, 37) fails: relocate raised
        # "an arena's conditions fail at the trace" to the caller (F11, through respec before FZ08)
        from torch.cuda._host_trace_lower_tape import lower_tape
        from torch.cuda._host_trace_memory import size_classes
        from torch.cuda._host_trace_tape import trace

        def fn(x, i):
            x.argmax(2)
            y = torch.neg(i)
            y + 1
            return y

        x = torch.randn(4, 36, 63, device="cuda", dtype=torch.float16)
        tape = trace(fn, (x, torch.randint(-9, 9, (36,), device="cuda")))
        tape.args = (x, torch.randint(-9, 9, (37,), device="cuda"))
        _, arenas = size_classes(lower_tape(tape), packed=packed)
        self.assertTrue(arenas)
        self.assertEqual([a.renames for a in arenas], [()] * len(arenas))


def _exprs(v):
    from torch.cuda._host_trace_memory import _bytes_expr

    return [_bytes_expr(v.captured.lowered, k) for k in v.memory.relocated]


instantiate_parametrized_tests(TestPlanned)

def setUpModule():
    from torch.cuda import _host_trace_hint_audit

    _host_trace_hint_audit.enable_for_tests()


if __name__ == "__main__":
    run_tests()
