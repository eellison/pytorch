# Owner(s): ["module: cuda"]

import functools
import itertools
import json
import statistics
import time
import unittest
from unittest import mock

from host_trace_testing import (
    build,
    exec_node_states,
    HostTraceTestCase,
    needs_native_replay,
    replay_backend,
)

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, skipIfRocm, TestCase


if torch.cuda.is_available():
    import host_trace_two_hint as two_hint

    from torch.cuda import _host_trace as ht

# a decode-sized projection: K = N = 4096 crosses several cuBLAS variant
# boundaries over M = 1..64 and splits K at small M (GEMM_FACTS Q1)
K = N = 4096
DTYPE = torch.bfloat16
_fresh_harvest_keys = itertools.count()


def linear(x, w, b):
    return F.linear(x, w, b)


def linear_nobias(x, w):
    return F.linear(x, w)


def _align_class(address):
    # the largest power of two dividing the address, capped: cuBLAS picks
    # kernels by operand alignment (16..256 bytes matter)
    return min(address & -address, 256) if address else 256


def _harvest_spec(tape, k, args):
    """The template key and harvest spec of region k of `tape` at `args`, as a
    replay computes them (the harvest's own contract, `_template`): the op and
    its scalars, every operand's dtype, sizes, strides and alignment class at
    the call (the out operand is an allocation: 256), the device class and the
    BLAS settings."""
    r = tape.regions[k]
    ev = ht._Evaluator()
    device = tape.device.index
    env = ht._bind_inputs(tape, ht._input_names(tape.inputs), args, device)
    metas, aligns = [], []
    for o in (*r.inputs, *r.outputs):
        metas.append(
            (
                o.dtype,
                tuple(int(ev.ev(v, env)) for v in o.sizes),
                tuple(int(ev.ev(v, env)) for v in o.strides),
            )
        )
        aligns.append(
            256 if o in r.outputs else _align_class(int(ev.ev(o.address, env)))
        )
    metas, aligns = tuple(metas), tuple(aligns)
    settings = ht._blas_settings()
    key = (device, tape.device_identity, r.op, r.scalars, metas, aligns, settings)
    return key, (r.op, r.scalars, metas, aligns)


def _harvest(tape, args, k=0):
    """The harvested template of region k of `tape` at `args` (from the
    process-wide cache when the key was harvested before), or the harvest's
    Miss by name."""
    key, spec = _harvest_spec(tape, k, args)
    return ht._template(key, spec, tape.device.index)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="closed regions are CUDA-only in this version")
class TestCudaHostTraceGemm(HostTraceTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        self.w = torch.randn(N, K, device="cuda", dtype=DTYPE) / K**0.5
        self.b = torch.randn(N, device="cuda", dtype=DTYPE)
        self.last_miss = ""
        self.last_served = None
        # the variants built from one tape, by the first one's identity
        self._pool: dict[int, list] = {}

    def _x(self, m, k=K):
        return torch.randn(m, k, device="cuda", dtype=DTYPE)

    def _templates_for(self, n=N, k=K):
        return [
            t
            for t in ht.gemm_templates()
            if f"{n}" in str(t["key"]) and f"{k}" in str(t["key"])
        ]

    def _check(self, variant, fn, args, what="", rebuild=True):
        # what an entry does with one tape's variants: the first that serves
        # the call does; a TopologyMiss names the tape, which built at these
        # inputs (no trace) is a variant with the call's node chain, kept
        # beside the others; any other Miss is the call's. With
        # `rebuild=False` only the variant given is asked.
        want = fn(*args)
        variants = self._pool.setdefault(id(variant), [variant])
        tape = None
        for v in variants if rebuild else [variant]:
            try:
                got = v.replay(args)
                break
            except ht.TopologyMiss as e:
                tape, self.last_miss = e.tape, str(e)
            except ht.Miss as e:
                self.last_miss = str(e)
        else:
            if tape is None or not rebuild:
                return False
            v = build(tape, fn, args)
            variants.append(v)
            got = v.replay(args)
        self.last_served = v
        self._assert_bitwise(got[0], want, what)
        return True

    def _variants(self, variant):
        return self._pool.get(id(variant), [variant])

    def _topology(self, variant):
        # per region of a variant, the chain its graph holds as (node kind,
        # programmatic edge) per node (the native replay's; the eager form
        # holds no graph)
        if variant.native is None:
            return None
        return [
            tuple(zip(s["kinds"], s["programmatic"]))
            for s in variant.native.region_stats()["sites"]
        ]

    def _graph_updates(self, variant):
        return variant.native.region_stats()["graph_updates"]

    def _served_kinds(self, variant, fn, args, what):
        # serves the call and returns the variant that served it with the
        # node kinds of the template it holds; the template cache is consulted
        # when the call's template differs from the one bound (a swap in the
        # chain's class, or the class miss that builds a variant): every entry
        # the call raised then names the served chain
        before = {t["key"]: t["hits"] for t in ht.gemm_templates()}
        self.assertTrue(self._check(variant, fn, args, what), what)
        hit = {
            tuple(t["kinds"])
            for t in ht.gemm_templates()
            if t["hits"] > before.get(t["key"], 0)
        }
        kinds = self._kinds(self.last_served)
        self.assertLessEqual(hit, {tuple(kinds)}, what)
        return self.last_served, kinds

    def _graph_nodes(self, variant):
        # what the driver says the variant's graph holds: (kind, enabled in
        # the exec) per node in creation order
        try:
            import cuda.bindings  # noqa: F401
        except ImportError:
            self.skipTest("cuda-python (cuda.bindings) is not installed")
        return [(kind, on) for _, kind, on in exec_node_states(variant.graph)]

    def test_linear_traces_as_a_closed_region(self):
        x = self._x(4)
        tape = ht.trace(linear, (x, self.w, self.b))
        self.assertEqual(tape.num_regions, 1)
        self.assertEqual(tape.num_launches, 0)
        r = json.loads(tape.to_json())["regions"][0]
        self.assertEqual(r["op"], "addmm")
        self.assertEqual([i["name"] for i in r["inputs"]], ["bias", "mat1", "mat2"])
        # every input dimension is a symbol on the tape; M and N differ
        (out,) = r["outputs"]
        self.assertEqual(out["name"], "out")
        self.assertIsInstance(out["sizes"][0], str)
        self.assertNotEqual(out["sizes"][0], out["sizes"][1])
        variant = build(tape, linear, (x, self.w, self.b))
        self.assertTrue(self._check(variant, linear, (self._x(4), self.w, self.b)))
        if variant.native is not None:
            self.assertEqual(len(variant.native.region_stats()["sites"]), 1)

    def test_bitwise_m_sweep_crosses_variant_boundaries(self):
        x = self._x(4)
        tape = ht.trace(linear, (x, self.w, self.b))
        variant = build(tape, linear, (x, self.w, self.b))
        before = len(self._templates_for())
        served, missed = [], []
        for m in range(1, 65):
            args = (self._x(m), self.w, self.b)
            try:
                ok = self._check(variant, linear, args, f"M={m}")
            except ht.Miss as e:
                missed.append((m, str(e)))
                continue
            (served if ok else missed).append(m)
        self.assertEqual(missed, [], missed)
        self.assertEqual(served, list(range(1, 65)))
        if variant.native is None:
            return  # the eager form selects no template: one variant serves
        used = len(self._templates_for()) - before
        # one variant per node chain the sweep selected (split-K at small M
        # adds the reduce kernel; a chain row is the kind and the programmatic
        # edge), each chain held by exactly one graph
        variants = self._variants(variant)
        chains = [self._topology(v)[0] for v in variants]
        self.assertEqual(len(set(chains)), len(chains), chains)
        self.assertGreaterEqual(len(variants), 2, chains)
        print(
            f"\n[gemm M sweep 1..64] templates used {used + 1} variants {len(variants)} chains {chains} applies {[v.native.region_stats()['applies'] for v in variants]}"
        )
        # GEMM_FACTS: 5-8 variants over M = 1..128 at this shape
        self.assertGreaterEqual(used + 1, 3)

    @needs_native_replay
    def test_split_k_boundary_builds_a_second_variant_from_the_same_tape(self):
        # M = 8 runs as two nodes (nvjet + splitKreduce), M = 12 as one. A
        # variant holds exactly the chain of the shape it was built at; the
        # other M is a TopologyMiss carrying the tape, and the tape built at
        # that M (no trace; the miss harvested the template, the build finds
        # it) is a second variant with the other chain. Each serves its
        # class bitwise; neither graph holds a node the other's chain needs.
        for m_trace, m_other in ((12, 8), (8, 12), (12, 4), (4, 24)):
            x = self._x(m_trace)
            tape = ht.trace(linear, (x, self.w, self.b))
            first = build(tape, linear, (x, self.w, self.b))
            other = (self._x(m_other), self.w, self.b)
            with self.assertRaisesRegex(ht.TopologyMiss, "graph holds") as cm:
                first.replay(other)
            self.assertIs(cm.exception.tape, tape)
            h0 = ht.gemm_harvests()
            second = build(cm.exception.tape, linear, other)
            self.assertEqual(ht.gemm_harvests(), h0)
            self.assertNotEqual(second.graph, first.graph)
            chains = (self._topology(first)[0], self._topology(second)[0])
            self.assertEqual(sorted(len(c) for c in chains), [1, 2], chains)
            for v in (first, second):
                (site,) = v.native.region_stats()["sites"]
                self.assertEqual(site["nodes"], len(self._topology(v)[0]), site)
            for m in (m_other, m_trace, m_other, m_other, m_trace):
                serving, idle = (second, first) if m == m_other else (first, second)
                args, what = (self._x(m), self.w, self.b), f"{m_trace}->{m}"
                self.assertTrue(self._check(serving, linear, args, what, rebuild=False))
                self.assertFalse(self._check(idle, linear, args, rebuild=False))
                self.assertIn("graph holds", self.last_miss)
            print(
                f"\n[split-K] traced M={m_trace}: chains {chains} applies "
                f"{[v.native.region_stats()['applies'] for v in (first, second)]}"
            )

    @needs_native_replay
    def test_cluster_change_goes_through_the_graph(self):
        # 4096 -> 11008: the M = 12 variant launches with an 8x1x1 cluster,
        # the M = 24 variant with 4x1x1 (SUPERSET.md); the exec-level
        # setter carries no attributes, so that swap goes through the graph
        w = torch.randn(11008, K, device="cuda", dtype=DTYPE) / K**0.5
        b = torch.randn(11008, device="cuda", dtype=DTYPE)
        # node and cuGraphExecUpdate, and stays bitwise. M = 8 and 16 split
        # K (two nodes), 12, 24 and 32 do not: two variants, and the cluster
        # change 8 -> 4 happens inside the one-node variant (12 -> 24)
        x = self._x(8)
        variant = build(ht.trace(linear, (x, w, b)), linear, (x, w, b))
        seen = []
        for m in (8, 12, 8, 16, 12, 24, 32, 8):
            self.assertTrue(self._check(variant, linear, (self._x(m), w, b), f"M={m}"))
            seen.append([self._graph_updates(v) for v in self._variants(variant)])
        print(
            f"\n[cluster change] graph updates per variant after each M in (8,12,8,16,12,24,32,8): {seen}"
        )
        self.assertEqual(len(seen[-1]), 2, seen)
        self.assertGreater(sum(seen[-1]), 0, seen)

    def test_template_shared_across_sites(self):
        n_sites = 32
        ws = [
            torch.randn(1024, 1024, device="cuda", dtype=DTYPE) / 32
            for _ in range(n_sites)
        ]

        def chain(x, *ws):
            for w in ws:
                x = F.linear(x, w)
            return x

        x = self._x(4, 1024)
        h0 = ht.gemm_harvests()
        tape = ht.trace(chain, (x, *ws))
        self.assertEqual(tape.num_regions, n_sites)
        variant = build(tape, chain, (x, *ws))
        # one harvest for the build key, however many sites share the shape
        self.assertLessEqual(ht.gemm_harvests() - h0, 1)
        h1 = ht.gemm_harvests()
        self.assertTrue(self._check(variant, chain, (self._x(16, 1024), *ws)))
        self.assertLessEqual(ht.gemm_harvests() - h1, 1)
        self.assertTrue(self._check(variant, chain, (self._x(4, 1024), *ws)))
        if variant.native is not None:
            self.assertEqual(len(variant.native.region_stats()["sites"]), n_sites)

    def test_per_call_cost(self):
        x = self._x(4)
        tape = ht.trace(linear, (x, self.w, self.b))
        variant = build(tape, linear, (x, self.w, self.b))
        args = (self._x(4), self.w, self.b)
        # M = 16 runs as one node where M = 4 splits K: another variant from
        # the same tape; M = 12 is another kernel of the one-node chain,
        # applied in place
        other = (self._x(16), self.w, self.b)
        same_chain = (self._x(12), self.w, self.b)
        v16 = build(tape, linear, other)
        for _ in range(5):
            variant.replay(args)
            v16.replay(other)
            v16.replay(same_chain)
        torch.cuda.synchronize()

        def timed(fn, n=100):
            xs = []
            for _ in range(n):
                t = time.perf_counter()
                fn()
                xs.append((time.perf_counter() - t) * 1e6)
            torch.cuda.synchronize()
            return statistics.median(xs)

        eager = timed(lambda: linear(*args))
        fixed = timed(lambda: variant.replay(args))
        moved = [(self._x(4), self.w, self.b) for _ in range(2)]
        i = iter(range(1000))
        rebind = timed(lambda: variant.replay(moved[next(i) % 2]))
        pairs = ((variant, args), (v16, other))
        j = iter(range(1000))

        def exec_switch():
            v, a = pairs[next(j) % 2]
            v.replay(a)

        switch = timed(exec_switch)
        k = iter(range(1000))
        apply_ = timed(lambda: v16.replay((other, same_chain)[next(k) % 2]))
        print(
            f"\n[gemm per call CPU us] eager {eager:.1f} replay same {fixed:.1f} "
            f"pointer rebind {rebind:.1f} variant switch 4<->16 {switch:.1f} "
            f"kernel switch 16<->12 in one variant {apply_:.1f}"
        )
        self.assertGreater(eager, 0)

    def test_mm_and_addmm(self):
        def f(a, b, c):
            return torch.addmm(c, a, b) + torch.mm(a, b)

        a, b, c = self._x(4), self._x(K, N), self.b
        tape = ht.trace(f, (a, b, c))
        self.assertEqual(tape.num_regions, 2)
        variant = build(tape, f, (a, b, c))
        for m in (4, 8, 12, 32):
            args = (self._x(m), b, c)
            self.assertTrue(self._check(variant, f, args, f"M={m}"))
        # a 2-D bias declines: the host copies it into the output first
        with self.assertRaisesRegex(ht.Declined, "2-D bias"):
            ht.trace(f, (a, b, self._x(4, N)))

    def test_scalars_other_than_one_decline_by_name(self):
        # with beta / alpha other than 1 the host adds the bias with a copy
        # kernel of its own before the library call: not one closed call, so
        # the trace declines by name
        def f(a, b, c):
            return torch.addmm(c, a, b, beta=0.5, alpha=2.0)

        a, b, c = self._x(4), self._x(K, N), self.b
        with self.assertRaisesRegex(ht.Declined, "beta / alpha"):
            ht.trace(f, (a, b, c))

    def test_replays_survive_host_state_churn(self):
        # some cuBLAS images carry host addresses of the harvesting call
        # (stack and heap); the replays must not depend on what is there now
        x = self._x(4)
        variant = build(
            ht.trace(linear, (x, self.w, self.b)), linear, (x, self.w, self.b)
        )

        def churn(depth):
            junk = bytearray(4096)
            return churn(depth - 1) + 1 if depth else len(junk)

        for m in (4, 8, 12, 4, 16, 8):
            churn(200)
            garbage = [bytearray(1 << 16) for _ in range(64)]
            args = (self._x(m), self.w, self.b)
            self.assertTrue(self._check(variant, linear, args, f"M={m}"))
            del garbage

    def test_batched_input_as_2d(self):
        def f(x, w):
            return F.linear(x, w)

        w = self.w
        x = torch.randn(2, 3, K, device="cuda", dtype=DTYPE)
        tape = ht.trace(f, (x, w))
        self.assertEqual(tape.num_regions, 1)
        variant = build(tape, f, (x, w))
        for s in (3, 5, 8):
            xx = torch.randn(2, s, K, device="cuda", dtype=DTYPE)
            self.assertTrue(self._check(variant, f, (xx, w), f"S={s}"))

    def test_dtype_and_layout_changes_miss_by_name(self):
        x = self._x(4)
        tape = ht.trace(linear_nobias, (x, self.w))
        variant = build(tape, linear_nobias, (x, self.w))
        with self.assertRaises(ht.Miss):
            variant.replay((x.half(), self.w.half()))
        with self.assertRaises(ht.Miss):
            variant.replay((x.float(), self.w.float()))
        # a weight stored transposed: other strides are another template
        # key (another cuBLAS kernel), served by the same variant
        wt = self.w.t().contiguous().t()
        h0 = ht.gemm_harvests()
        self.assertTrue(self._check(variant, linear_nobias, (self._x(4), wt)))
        if variant.native is not None:
            self.assertGreaterEqual(ht.gemm_harvests() - h0, 1)
        self.assertTrue(self._check(variant, linear_nobias, (self._x(8), wt)))
        self.assertTrue(self._check(variant, linear_nobias, (self._x(8), self.w)))

    def test_attribute_the_driver_does_not_report_declines_the_template(self):
        # a kernel node attribute the driver refuses to report is a miss by
        # name for that template, never a default value: a mismatch the
        # census cannot see would hang the transplant. The refusal is forced
        # with an attribute id the driver does not know.
        w = torch.randn(N + 192, K, device="cuda", dtype=DTYPE) / K**0.5
        x = self._x(4)
        read = torch._C._host_trace_harvest_nodes
        probe = functools.partial(read, probe_attr=0x7FFF)
        with mock.patch.object(torch._C, "_host_trace_harvest_nodes", probe):
            tape = ht.trace(linear_nobias, (x, w))
            with self.assertRaisesRegex(ht.Miss, "does not report kernel node"):
                _harvest(tape, (x, w))
        key = f"{N + 192}"
        misses = [t["miss"] for t in ht.gemm_templates() if key in str(t["key"])]
        self.assertEqual(len(misses), 1)
        self.assertIn("does not report kernel node attribute probe", misses[0])

    def test_blas_setting_change_is_a_new_key(self):
        w = self.w.float()
        x = self._x(4).float()
        tape = ht.trace(linear_nobias, (x, w))
        variant = build(tape, linear_nobias, (x, w))
        old = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = (
                "tf32" if old != "tf32" else "ieee"
            )
            h0 = ht.gemm_harvests()
            args = (self._x(4).float(), w)
            served = self._check(variant, linear_nobias, args)
            # on sm100 the tf32 kernels are cutlass3x with per-call TMA
            # descriptors in their parameters: a named miss, never stale
            if not served:
                self.assertIn("not rebindable", self.last_miss)
            if variant.native is not None:
                self.assertGreaterEqual(ht.gemm_harvests() - h0, 1)
            print(f"\n[fp32 after precision flip] served {served}")
        finally:
            torch.backends.cuda.matmul.fp32_precision = old
        self.assertTrue(self._check(variant, linear_nobias, (self._x(4).float(), w)))

    def test_replay_on_another_stream(self):
        # the native replay is bound to the device and stream it was prepared
        # on (O29): a call on another stream is refused by name, the bound
        # stream serves again; the eager form serves any current stream
        x = self._x(4)
        tape = ht.trace(linear, (x, self.w, self.b))
        variant = build(tape, linear, (x, self.w, self.b))
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            if replay_backend() == "native":
                with self.assertRaisesRegex(RuntimeError, "bound device and stream"):
                    variant.replay((self._x(4), self.w, self.b))
            else:
                for m in (4, 8, 16):
                    self.assertTrue(
                        self._check(
                            variant, linear, (self._x(m), self.w, self.b), f"M={m}"
                        )
                    )
        self.assertTrue(self._check(variant, linear, (self._x(4), self.w, self.b)))

    def test_two_variants_interleaved(self):
        x4, x32 = self._x(4), self._x(32)
        v4 = build(ht.trace(linear, (x4, self.w, self.b)), linear, (x4, self.w, self.b))
        v32 = build(
            ht.trace(linear, (x32, self.w, self.b)), linear, (x32, self.w, self.b)
        )
        for m in (1, 8, 12, 16, 24, 32, 48, 64):
            self.assertTrue(
                self._check(v4, linear, (self._x(m), self.w, self.b), f"v4 M={m}")
            )
            self.assertTrue(
                self._check(v32, linear, (self._x(m), self.w, self.b), f"v32 M={m}")
            )

    def test_region_records_no_real_address(self):
        # the region holds operands as expressions over the roots' symbols;
        # the library never sees a traced tensor, the harvest runs on real
        # scratch, and replay binds the slots from the real inputs. So a
        # trace whose input hints are placeholders (commit-1 patch 01b)
        # harvests and replays the same
        x = self._x(4)
        tape = ht.trace(linear, (x, self.w, self.b))
        regions = json.loads(tape.to_json())["regions"]
        real = {hex(t.data_ptr()) for t in (x, self.w, self.b)}
        for r in regions:
            for o in [*r["inputs"], *r["outputs"]]:
                self.assertIsInstance(o["address"], str)
                self.assertNotIn(
                    str(int(o["address"], 0)) if o["address"].isdigit() else "", real
                )
                self.assertTrue(any(ch.isalpha() for ch in o["address"]), o["address"])
        variant = build(tape, linear, (x, self.w, self.b))
        self.assertTrue(self._check(variant, linear, (self._x(8), self.w, self.b)))

    def test_gemm_templates_introspection(self):
        x = self._x(4)
        variant = build(
            ht.trace(linear, (x, self.w, self.b)), linear, (x, self.w, self.b)
        )
        variant.replay((self._x(4), self.w, self.b))
        if variant.native is None:
            # the eager form harvests nothing: the harvest itself fills the cache
            _harvest(variant.tape, (x, self.w, self.b))
        entries = ht.gemm_templates()
        self.assertGreaterEqual(len(entries), 1)
        for e in entries:
            self.assertEqual(
                set(e),
                {
                    "key",
                    "kernels",
                    "kinds",
                    "programmatic",
                    "node_count",
                    "scratch",
                    "harvest_us",
                    "hits",
                    "miss",
                },
            )
            self.assertEqual(len(e["kinds"]), e["node_count"])
            self.assertEqual(len(e["kernels"]), e["kinds"].count("kernel"))
            self.assertTrue(e["harvest_us"] > 0 or e["miss"])

    # ---- round 1 review fixes (hosttrace_review/gemm_round1)

    def _serve(self, variant, fn, args_of, ms, what):
        # every M served bitwise (a node chain the variant does not hold is
        # served by another exec of the same tape, _check), or refused by
        # name: a variant whose kernels are not rebindable (cutlass3x
        # descriptors)
        served, refused = [], []
        for m in ms:
            if self._check(variant, fn, args_of(m), f"{what} M={m}"):
                served.append(m)
            else:
                self.assertIn("not rebindable", self.last_miss, f"{what} M={m}")
                refused.append(m)
        return served, refused

    def _chains(self, variant, k=0):
        return [self._topology(v)[k] for v in self._variants(variant)]

    def _kinds(self, variant, k=0):
        # the node kinds of site k (None on the eager form)
        topology = self._topology(variant)
        return None if topology is None else [kind for kind, _p in topology[k]]

    def _fp32(self, m):
        return torch.randn(m, K, device="cuda", dtype=torch.float32)

    def test_fp32_ieee_linear_with_bias(self):
        # F2: the fp32 ieee bias path runs a memset and a scaling kernel
        # before the GEMM at small M; every node of the call is a node of the
        # region, memsets included
        old = torch.backends.cuda.matmul.fp32_precision
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        try:
            w, b, x = self.w.float(), self.b.float(), self._fp32(4)
            variant = build(ht.trace(linear, (x, w, b)), linear, (x, w, b))
            kinds = self._kinds(variant)
            served, refused = self._serve(
                variant,
                linear,
                lambda m: (self._fp32(m), w, b),
                (4, 1, 2, 8, 12, 16, 33, 64, 4),
                "fp32 ieee bias",
            )
            self.assertEqual(refused, [], kinds)
            print(f"\n[fp32 ieee bias] site {kinds} served {served}")
        finally:
            torch.backends.cuda.matmul.fp32_precision = old

    def test_fp32_split_at_large_m_is_another_variant(self):
        # F3: the fp32 SIMT path splits K at large M (cutlass simt + reduce)
        # where M = 4 runs one kernel; a tape traced at M = 4 serves 33 and
        # 64 through a second variant built from it at those inputs
        old = torch.backends.cuda.matmul.fp32_precision
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        try:
            w, x = self.w.float(), self._fp32(4)
            variant = build(ht.trace(linear_nobias, (x, w)), linear_nobias, (x, w))
            if variant.native is not None:
                self.assertEqual(self._kinds(variant), ["kernel"])
            served, refused = self._serve(
                variant,
                linear_nobias,
                lambda m: (self._fp32(m), w),
                (4, 33, 64, 1, 4, 64),
                "fp32 no-bias",
            )
            self.assertEqual(refused, [])
            if variant.native is not None:
                chains = self._chains(variant)
                self.assertEqual(len(chains), 2, chains)
                self.assertEqual(sorted(len(c) for c in chains), [1, 2], chains)
                print(f"\n[fp32 split at large M] chains {chains} served {served}")
        finally:
            torch.backends.cuda.matmul.fp32_precision = old

    @needs_native_replay
    def test_graph_holds_exactly_the_served_templates_nodes(self):
        # a variant's graph holds the nodes of the template its calls select,
        # one for one: no other node of any kind (no memset parked on a dummy
        # word, no kernel of another chain) and none disabled, read back from
        # the driver after kernel changes in place and the topology miss that
        # made a second variant
        x = self._x(4)
        variant = build(
            ht.trace(linear, (x, self.w, self.b)), linear, (x, self.w, self.b)
        )
        served = {}
        for m in (4, 8, 1, 12, 16, 24, 4, 12):
            args = (self._x(m), self.w, self.b)
            v, kinds = self._served_kinds(variant, linear, args, f"M={m}")
            served[id(v)] = (m, kinds)
        variants = self._variants(variant)
        self.assertGreaterEqual(len(variants), 2)
        self.assertEqual(len(served), len(variants))
        for v in variants:
            m, kinds = served[id(v)]
            nodes = self._graph_nodes(v)
            self.assertEqual([k for k, _e in nodes], kinds, f"last served M={m}")
            self.assertEqual([k for k, _e in nodes], self._kinds(v))
            self.assertEqual([e for _k, e in nodes], [True] * len(nodes), f"M={m}")

    @needs_native_replay
    def test_two_sites_and_a_launch_hold_no_extra_node(self):
        # the same one for one over a tape with two region sites and the
        # tape's own kernel between them: the graph is site, kernel, site
        w1 = torch.randn(1024, 1024, device="cuda", dtype=DTYPE) / 32
        w2 = torch.randn(1024, 1024, device="cuda", dtype=DTYPE) / 32

        def two(x, w1, w2):
            return F.linear(F.linear(x, w1) * 2, w2)

        x = self._x(4, 1024)
        tape = ht.trace(two, (x, w1, w2))
        self.assertEqual((tape.num_regions, tape.num_launches), (2, 1))
        self.assertEqual((len(tape.memsets), len(tape.memcpys)), (0, 0))
        variant = build(tape, two, (x, w1, w2))
        for m in (4, 16, 64, 4):
            args = (self._x(m, 1024), w1, w2)
            v, kinds = self._served_kinds(variant, two, args, f"M={m}")
            nodes = self._graph_nodes(v)
            self.assertEqual(
                [k for k, _e in nodes], kinds + ["kernel"] + kinds, f"M={m}"
            )
            self.assertEqual([e for _k, e in nodes], [True] * len(nodes), f"M={m}")

    def test_odd_and_narrow_shapes(self):
        # F2: N or K = 4097 runs cutlass 2.x behind a semaphore memset; the
        # narrow shapes cross the gemv / split-K variants
        for n, k in ((4097, K), (N, 4097), (3, K), (N, 3)):
            w = torch.randn(n, k, device="cuda", dtype=DTYPE) / k**0.5
            b = torch.randn(n, device="cuda", dtype=DTYPE)
            x = self._x(8, k)
            variant = build(ht.trace(linear, (x, w, b)), linear, (x, w, b))
            kinds = self._kinds(variant)
            served, refused = self._serve(
                variant,
                linear,
                lambda m, k=k, w=w, b=b: (self._x(m, k), w, b),
                (8, 1, 4, 64, 8),
                f"{k}x{n}",
            )
            self.assertEqual(refused, [], f"{k}x{n}: {kinds}")
            print(f"\n[{k}x{n}] site {kinds} served {served}")

    def test_storage_offset_of_8_bytes(self):
        # an alignment class cuBLAS serves with other kernels than the
        # aligned one
        def args(m):
            big = self._x(m, K + 4)
            return (big.view(-1)[4 : 4 + m * K].view(m, K), self.w, self.b)

        x = args(8)[0]
        self.assertEqual(x.storage_offset(), 4)
        variant = build(
            ht.trace(linear, (x, self.w, self.b)), linear, (x, self.w, self.b)
        )
        served, refused = self._serve(
            variant, linear, args, (8, 1, 4, 64, 8), "offset 8 bytes"
        )
        self.assertEqual(refused, [], self._kinds(variant))
        # the aligned x through the same tape: another key
        self.assertTrue(
            self._check(variant, linear, (self._x(8), self.w, self.b), "aligned x"),
            self.last_miss,
        )

    def test_non_contiguous_and_expanded_x(self):
        # F2, in the form of the out= rule's (c): the host clones such an
        # operand contiguous before the library call; the clone is the tape's
        # copy (a launch or one memcpy) ahead of the region, which reads the
        # clone; the layout test that made the host copy is guarded, so a
        # layout the host takes as is misses by a guard instead of replaying
        # a copy eager would not make
        def strided(m):
            return self._x(m, 2 * K)[:, ::2]

        def expanded(m):
            return self._x(1, K).expand(m, K)

        for name, x_of in (("strided", strided), ("expanded", expanded)):
            x = x_of(8)
            tape = ht.trace(linear, (x, self.w, self.b))
            self.assertEqual(tape.num_regions, 1, name)
            self.assertEqual(tape.num_launches + len(tape.memcpys), 1, name)
            variant = build(tape, linear, (x, self.w, self.b))
            kinds = self._kinds(variant)
            served, refused = self._serve(
                variant,
                linear,
                lambda m, x_of=x_of: (x_of(m), self.w, self.b),
                (8, 4, 64, 8),
                name,
            )
            self.assertEqual(refused, [], f"{name}: {kinds}")
            print(f"\n[{name} x] site {kinds} served {served}")
            # M = 1: served, or another class of the copy (a guard of the
            # clone's kernel), never a wrong answer
            if not self._check(
                variant, linear, (x_of(1), self.w, self.b), f"{name} M=1"
            ):
                self.assertIn("guard failed", self.last_miss, f"{name} M=1")
            # a contiguous x through the same variant misses by the layout
            # guard: the copy is eager's for the traced layout only
            self.assertFalse(
                self._check(
                    variant,
                    linear,
                    (self._x(8), self.w, self.b),
                    f"{name}: contiguous x",
                    rebuild=False,
                )
            )
            self.assertIn("guard failed", self.last_miss, f"{name}: contiguous x")

    def test_two_sites_and_two_variants_interleaved_on_one_stream(self):
        # F4: the two sites of a variant and two variants replayed on one
        # stream, interleaved over the split-K boundary (the scratch the
        # regions' nodes get at replay is the replay's arena: the native
        # replay's is shared per family, test_hosttrace_arena.py)
        def two(x, w1, b1, w2, b2):
            return F.linear(F.linear(x, w1, b1), w2, b2)

        w2 = torch.randn(N, K, device="cuda", dtype=DTYPE) / K**0.5
        b2 = torch.randn(N, device="cuda", dtype=DTYPE)
        x = self._x(8)
        variant = build(
            ht.trace(two, (x, self.w, self.b, w2, b2)), two, (x, self.w, self.b, w2, b2)
        )
        variant2 = build(
            ht.trace(linear, (x, self.w, self.b)), linear, (x, self.w, self.b)
        )
        for m in (8, 1, 4, 12, 64, 8, 1):
            self.assertTrue(
                self._check(
                    variant,
                    two,
                    (self._x(m), self.w, self.b, w2, b2),
                    f"two sites M={m}",
                ),
                self.last_miss,
            )
            self.assertTrue(
                self._check(
                    variant2, linear, (self._x(m), self.w, self.b), f"variant 2 M={m}"
                ),
                self.last_miss,
            )
        if variant.native is not None:
            print(f"\n[two sites] sites {[self._kinds(variant, k) for k in range(2)]}")

    # ---- batched GEMMs (aten.bmm) as closed regions

    def test_harvest_operand_buffers_live_for_one_harvest(self):
        # round 8, F4: the scratch operand sets were carved from two per-device
        # buffers grown to the largest set ever harvested and kept for the
        # process (+512 MiB after one lm_head-sized key); a buffer now lives
        # as long as its set, and a harvest leaves behind its template's own
        # graph pool only
        metas = (
            (torch.bfloat16, (8, 8256), (8256, 1)),
            (torch.bfloat16, (8256, 8192), (1, 8256)),
            (torch.bfloat16, (8, 8192), (8192, 1)),
        )
        aligns = (256, 256, 256)
        dev = torch.device("cuda", torch.cuda.current_device())
        before = torch.cuda.memory_allocated()
        first, spans = ht._harvest_operands(0, metas, aligns, dev)
        second, _ = ht._harvest_operands(1, metas, aligns, dev)
        # one buffer per set, both alive during the harvest (the allocator's
        # global counter is not compared upward: another test's garbage may
        # be freed meanwhile)
        for operands in (first, second):
            storages = {t.untyped_storage().data_ptr() for t in operands}
            self.assertEqual(len(storages), 1)
            self.assertGreaterEqual(operands[0].untyped_storage().nbytes(), sum(spans))
        self.assertNotEqual(
            first[0].untyped_storage().data_ptr(),
            second[0].untyped_storage().data_ptr(),
        )
        del first, second, operands
        self.assertLessEqual(torch.cuda.memory_allocated(), before)
        # through a harvest of a fresh key whose operands take 135 MB (the
        # harvest streams' workspaces exist from a small key's harvest first;
        # the key is fresh per run of this test: the two-hint family runs it
        # again in this process, and a harvested key is served from its
        # template); the harvest is driven directly, whichever backend
        # replays: what it leaves behind is the template's own graph pool
        x = self._x(8)
        _harvest(ht.trace(linear, (x, self.w, self.b)), (x, self.w, self.b))
        k = 8256 + 8 * next(_fresh_harvest_keys)
        w = torch.randn(8192, k, device="cuda", dtype=DTYPE) / k**0.5
        b = torch.randn(8192, device="cuda", dtype=DTYPE)
        x = self._x(8, k)
        tape = ht.trace(linear, (x, w, b))
        harvests = ht.gemm_harvests()
        before = torch.cuda.memory_allocated()
        _harvest(tape, (x, w, b))
        self.assertEqual(ht.gemm_harvests(), harvests + 1)
        self.assertLess(torch.cuda.memory_allocated() - before, 64 << 20)
        variant = build(tape, linear, (x, w, b))
        self.assertEqual(ht.gemm_harvests(), harvests + 1)
        self.assertTrue(self._check(variant, linear, (x, w, b), f"8192 x {k}"))

    def test_an_input_is_written_through_a_region_only_by_an_out_into_it(self):
        # a functional GEMM's out operand is an allocation the call made, so
        # Tape.written_inputs (commit 1) gets no region entry; the out= form
        # into an input (eager's cuBLAS host takes the given result as is, and
        # Inductor's backward writes a bmm into a donated saved activation)
        # writes that input through the region; the in-place variant still
        # declines by name
        x = self._x(8)
        tape = ht.trace(linear, (x, self.w, self.b))
        self.assertEqual(len(tape.regions), 1)
        self.assertTrue(tape.regions[0].out.root.allocation)
        self.assertEqual(tape.written_roots, [tape.regions[0].out.root.name])
        self.assertEqual(tape.written_inputs, ())
        y = torch.empty(8, N, device="cuda", dtype=DTYPE)
        wt = self.w.t().contiguous()
        tape = ht.trace(lambda a, b, o: torch.mm(a, b, out=o), (x, wt, y))
        self.assertEqual(len(tape.regions), 1)
        self.assertFalse(tape.regions[0].out.root.allocation)
        self.assertEqual(tape.written_inputs, (2,))
        with self.assertRaisesRegex(ht.Declined, "aten.addmm_.default"):
            ht.trace(lambda c, a, b: c.addmm_(a, b), (y, x, wt))

    def test_the_out_form_into_a_trace_allocation_is_the_same_region(self):
        # Inductor's generated wrapper writes every extern GEMM into a buffer it
        # allocated (extern_kernels.mm(a, b, out=buf)): the out= form of a closed
        # call is the same region, its output the given dense allocation; an out=
        # that is an input or another layout declines by name
        def f(a, b, c):
            out = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
            torch.mm(a, b, out=out)
            out2 = torch.empty_like(out)
            torch.addmm(c, a, b, out=out2)
            out3 = torch.empty(
                2, a.shape[0], b.shape[1], device=a.device, dtype=a.dtype
            )
            torch.bmm(a.expand(2, *a.shape), b.expand(2, *b.shape), out=out3)
            # out3[0] (offset 0): a select at a symbolic offset into an allocation
            # would give the add an alignment guard over the allocation's base
            # symbol, which the lowering cannot bind (a limit of the view route,
            # not of the region)
            return out + out2 + out3[0]

        a, b, c = self._x(4), self._x(K, N), self.b
        tape = ht.trace(f, (a, b, c))
        self.assertEqual(tape.num_regions, 3)
        regions = json.loads(tape.to_json())["regions"]
        self.assertEqual([r["op"] for r in regions], ["mm", "addmm", "bmm"])
        self.assertTrue(all(r.out.root.allocation for r in tape.regions))
        self.assertEqual(tape.written_inputs, ())
        variant = build(tape, f, (a, b, c))
        for m in (4, 8, 32):
            self.assertTrue(self._check(variant, f, (self._x(m), b, c), f"M={m}"))
        # an out= into an input is the same region writing that input, and an
        # out= of another layout eager's cuBLAS host takes as is (a unit stride
        # along one dimension, the other stride at least that extent: the
        # column-major result here) is the region under its own key; a result
        # without a unit stride, which the host computes into a copy of and
        # copies back, declines by name
        y = torch.empty(4, N, device="cuda", dtype=DTYPE)
        tape = ht.trace(lambda a, b, o: torch.mm(a, b, out=o), (a, b, y))
        self.assertEqual(tape.num_regions, 1)
        self.assertFalse(tape.regions[0].out.root.allocation)
        self.assertEqual(tape.written_inputs, (2,))

        def transposed(a, b):
            return torch.mm(a, b, out=torch.empty(N, 4, device="cuda", dtype=DTYPE).t())

        tape = ht.trace(transposed, (a, b))
        self.assertEqual(tape.num_regions, 1)
        self.assertTrue(self._check(build(tape, transposed, (a, b)), transposed, (a, b)))
        with self.assertRaisesRegex(ht.Declined, "compute into a copy"):
            ht.trace(
                lambda a, b: torch.mm(
                    a, b, out=torch.empty(8, 2 * N, device="cuda", dtype=DTYPE)[::2, ::2]
                ),
                (a, b),
            )

    def test_bmm_k1_rotary_product_takes_eagers_triton_override(self):
        # LlamaRotaryEmbedding.forward's inv_freq_expanded @ position_ids_expanded:
        # a (B, 32, 1) x (B, 1, 1) fp32 bmm with batch1 expanded over the batch
        # (batch stride 0), reached through matmul's reshape. Eager serves the
        # K = 1 product with torch._native's Triton outer-product kernel, not
        # cuBLAS (bmm.out would reach cuBLAS's gemmk1 GEMV kernel), so the
        # trace takes the override's own launch (torch/cuda/_host_trace_triton.py):
        # no region, eager's Triton kernel beside the position cast, bitwise
        def rotary(inv_freq, position_ids):
            inv_freq_expanded = (
                inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            )
            position_ids_expanded = position_ids[:, None, :].float()
            return inv_freq_expanded @ position_ids_expanded

        inv_freq = 1.0 / (10000 ** (torch.arange(0, 64, 2, device="cuda").float() / 64))

        def pos(b, value):
            return torch.full((b, 1), value, dtype=torch.int64, device="cuda")

        base = (inv_freq, pos(4, 16))
        tape = ht.trace(rotary, base)
        self.assertEqual((tape.num_regions, tape.num_launches), (0, 2))
        self.assertEqual(tape.launches[1]["kernel"], "_bmm_outer_product_kernel")
        h0 = ht.gemm_harvests()
        variant = build(tape, rotary, base)
        self.assertEqual(ht.gemm_harvests() - h0, 0)
        # (B = 1 misses by a guard of the position cast: a one-element
        # iterator is contiguous by numel, the sibling's own rule)
        for args in (
            base,
            (inv_freq, pos(4, 17)),
            (inv_freq, pos(2, 5)),
            (inv_freq, pos(7, 1000)),
            (inv_freq.clone(), pos(4, 3)),
        ):
            self.assertTrue(
                self._check(variant, rotary, args, f"B={args[1].shape[0]}"),
                self.last_miss,
            )
            self.assertEqual(variant.replay(args)[0].stride(), rotary(*args).stride())
        # the same product in bf16 and with a wider N, straight through bmm: the
        # override's block sizes (from M and N) and Triton's specialization of
        # every integer argument (divisible by 16 or not: M, N, M * N, B) are
        # guards, so a product in the same class rebinds and one in another
        # (M = 7: BLOCK_M 8, not 32; N = 130: BLOCK_N 128) misses
        for dtype in (torch.bfloat16, torch.float32):
            a = torch.randn(3, 40, 1, device="cuda", dtype=dtype)
            b = torch.randn(3, 1, 24, device="cuda", dtype=dtype)
            tape = ht.trace(torch.bmm, (a, b))
            self.assertEqual((tape.num_regions, tape.num_launches), (0, 1))
            variant = build(tape, torch.bmm, (a, b))
            for args in (
                (a, b),
                (
                    torch.randn(5, 36, 1, device="cuda", dtype=dtype),
                    torch.randn(5, 1, 20, device="cuda", dtype=dtype),
                ),
            ):
                self.assertTrue(
                    self._check(variant, torch.bmm, args, str(dtype)), self.last_miss
                )
            other = (
                torch.randn(5, 7, 1, device="cuda", dtype=dtype),
                torch.randn(5, 1, 130, device="cuda", dtype=dtype),
            )
            self.assertFalse(self._check(variant, torch.bmm, other, str(dtype)))
            self.assertIn("guard failed", self.last_miss)
        # bmm.out is not the eager override's route: the same product reaches
        # cuBLAS's K = 1 GEMV kernel there, which the harvest can take, but the
        # functional op the trace sees never does
        ai = inv_freq[None, :, None].expand(4, -1, 1)
        bi = pos(4, 16)[:, None, :].float()
        out = torch.empty(4, 32, 1, device="cuda")
        stream = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(stream):
            torch.ops.aten.bmm.out(ai, bi, out=out)
            stream.synchronize()
            g.capture_begin(capture_error_mode="thread_local")
            torch.ops.aten.bmm.out(ai, bi, out=out)
            g.capture_end()
        stream.synchronize()
        nodes = torch._C._host_trace_harvest_nodes(g.raw_cuda_graph())
        names = [n["name"] for n in nodes if n["kind"] == "kernel"]
        print(
            f"\n[bmm K=1] functional: eager's Triton kernel; bmm.out: {[n[:32] for n in names]}"
        )
        self.assertTrue(any("gemm" in n or "gemv" in n for n in names), names)

    def test_bmm_batched_64_cubed_bf16_and_fp32(self):
        for dtype in (torch.bfloat16, torch.float32):

            def mk(b, m, k=64, n=64):
                return (
                    torch.randn(b, m, k, device="cuda", dtype=dtype),
                    torch.randn(b, k, n, device="cuda", dtype=dtype),
                )

            base = mk(8, 64)
            tape = ht.trace(torch.bmm, base)
            self.assertEqual((tape.num_regions, tape.num_launches), (1, 0))
            variant = build(tape, torch.bmm, base)
            for args in (mk(8, 64), mk(3, 64), mk(8, 1), mk(8, 17), mk(2, 128)):
                with self.subTest(dtype=dtype, shape=tuple(args[0].shape)):
                    self.assertTrue(
                        self._check(variant, torch.bmm, args, str(dtype)),
                        self.last_miss,
                    )
            # a transposed batch2 is another layout cuBLAS takes as is (a
            # column-major batch): its own key, served without a copy
            a, _b = mk(8, 64)
            bt = torch.randn(8, 64, 64, device="cuda", dtype=dtype).transpose(1, 2)
            variant_t = build(ht.trace(torch.bmm, (a, bt)), torch.bmm, (a, bt))
            bt2 = torch.randn(8, 64, 64, device="cuda", dtype=dtype).transpose(1, 2)
            self.assertTrue(
                self._check(variant_t, torch.bmm, (a, bt2), "transposed"),
                self.last_miss,
            )
            if variant.native is not None:
                stats = variant.native.region_stats()
                print(
                    f"\n[bmm 64^3 {dtype}] nodes {stats['sites'][0]['kinds']} applies {stats['applies']}"
                )

    def test_bmm_declines_by_name(self):
        a = torch.randn(4, 16, 64, device="cuda", dtype=DTYPE)
        b = torch.randn(4, 64, 24, device="cuda", dtype=DTYPE)
        # a batch operand the host copies first (a slice-step layout): eager's
        # clone before the library call is the tape's copy (a launch or one
        # memcpy), then the region reads the clone (form (c) of the out= rule)
        tape = ht.trace(
            torch.bmm,
            (a, torch.randn(4, 64, 48, device="cuda", dtype=DTYPE)[:, :, ::2]),
        )
        self.assertEqual(tape.num_regions, 1)
        self.assertEqual(tape.num_launches + len(tape.memcpys), 1)
        # baddbmm is not a closed op: its meta copies the expanded bias into
        # the result with ATen's copy kernel, whose image carries per-call
        # host bytes no harvest reproduces
        with self.assertRaisesRegex(ht.Declined, "baddbmm"):
            ht.trace(torch.baddbmm, (torch.randn(24, device="cuda", dtype=DTYPE), a, b))
        # an fp64 batch: not a dtype the closed region records
        ad = torch.randn(2, 4, 8, device="cuda", dtype=torch.float64)
        with self.assertRaisesRegex(ht.Declined, "float64"):
            ht.trace(torch.bmm, (ad, ad.transpose(1, 2)))

    def test_mm_with_no_rows_serves_the_empty_result(self):
        # eager launches nothing for a GEMM with M = 0 and returns the empty
        # result: the region's site has no node; the native replay's template
        # registry refuses a variant without nodes by name (a documented
        # refusal, STAGE_B.md), so the eager form serves it; a shape with rows
        # misses on the empty output's stride guard, before the site
        def rows(x, w):
            return x[:0] @ w

        def tail(x, w):
            return x[8:] @ w

        w = self._x(K, N)
        for fn in (rows, tail):
            with self.subTest(fn=fn.__name__):
                x = self._x(8)
                tape = ht.trace(fn, (x, w))
                self.assertEqual((tape.num_regions, tape.num_launches), (1, 0))
                variant = build(tape, fn, (x, w))
                if replay_backend() == "native":
                    self.assertIn("needs nodes", variant.refused)
                for other in (x, self._x(8)):
                    got = variant.replay((other, w))[0]
                    self.assertEqual(tuple(got.shape), (0, N))
                    self.assertTrue(torch.equal(got, fn(other, w)))
        self.assertTrue(
            torch.equal(variant.replay((self._x(8), w))[0], tail(self._x(8), w))
        )
        with self.assertRaises(ht.Miss):
            variant.replay((self._x(12), w))

    def test_every_case_traces_the_same_program_under_other_hints(self):
        # the recorder never reads a hint: every trace this class makes, made
        # again under other hints, is the same program (host_trace_two_hint).
        # The excluded tests count harvests against the process-wide cache and
        # hold only on their first run in a process.
        cached = "counts harvests; a second run in the process finds them cached"
        two_hint.assert_family(
            self,
            exclude={
                "test_bitwise_m_sweep_crosses_variant_boundaries": cached,
                "test_blas_setting_change_is_a_new_key": cached,
                "test_dtype_and_layout_changes_miss_by_name": cached,
            },
        )

    def test_harvest_reads_the_programmatic_edge_behind_an_anchor(self):
        # a closed call captured alone leaves its first kernel without an incoming
        # edge, so the harvest launches an anchor kernel ahead of it inside the capture:
        # the call's first node then has an edge whose type says whether the library
        # launched it with programmatic stream serialization. harvest_nodes(anchored)
        # drops the anchor (the capture's one root) and flags every node from the edge
        # data; without the flag the anchor is a node like any other
        from cuda.bindings import driver

        from torch.cuda._utils import _check_cuda_bindings_driver as check

        x, w = self._x(8), self.w
        anchor = torch.empty(1, device="cuda")
        stream = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(stream):
            linear_nobias(x, w)
            stream.synchronize()
            g.capture_begin(capture_error_mode="thread_local")
            anchor.fill_(1)
            linear_nobias(x, w)
            g.capture_end()
        stream.synchronize()
        raw = g.raw_cuda_graph()
        anchored = torch._C._host_trace_harvest_nodes(raw, anchored=True)
        plain = torch._C._host_trace_harvest_nodes(raw)
        self.assertEqual(len(plain), len(anchored) + 1)
        self.assertEqual(plain[0]["kind"], "kernel")
        self.assertFalse(plain[0]["programmatic"])
        row = lambda n: (n["kind"], n.get("func"), n["programmatic"])  # noqa: E731
        self.assertEqual([row(n) for n in plain[1:]], [row(n) for n in anchored])
        # the flags are the capture's edge types
        count = check(driver.cuGraphGetNodes(raw, 0))[-1]
        nodes = [int(n) for n in check(driver.cuGraphGetNodes(raw, count))[0]]
        count = check(driver.cuGraphGetEdges(raw, 0))[-1]
        frm, to, data = check(driver.cuGraphGetEdges(raw, count))[:3]
        programmatic = {int(t) for t, d in zip(to, data) if int(d.type) == 1}
        self.assertEqual(
            [n in programmatic for n in nodes], [n["programmatic"] for n in plain]
        )
        self.assertTrue(
            all(int(d.from_port) in (0, 1) and int(d.to_port) == 0 for d in data)
        )
        print(
            f"\n[harvest anchor] {[(n['name'][:40], n['programmatic']) for n in anchored]}"
        )
        # the template the build harvests for this shape carries the same flags
        tape = ht.trace(linear_nobias, (x, w))
        variant = build(tape, linear_nobias, (x, w))
        self.assertTrue(self._check(variant, linear_nobias, (self._x(8), w)))
        entries = [t for t in self._templates_for() if t["hits"] and not t["miss"]]
        self.assertTrue(entries)
        flags = [n["programmatic"] for n in anchored]
        self.assertIn(flags, [t["programmatic"] for t in entries])
        for t in entries:
            self.assertEqual(len(t["programmatic"]), t["node_count"])

    def test_a_template_without_programmatic_launches_is_another_class(self):
        # at an 8-byte storage offset cuBLAS runs this GEMM as a legacy cutlass
        # kernel launched without programmatic stream serialization; aligned, as
        # nvjet with it. Both chains are [kernel, kernel], but a variant built at
        # the aligned x holds programmatic edges into its region nodes, and the
        # offset template's kernels would run behind them without waiting: the
        # flags are part of the class, the offset x is a topology miss served by
        # its own exec (and the reverse order splits the same way)
        def offset_x(m):
            big = self._x(m, K + 4)
            return big.view(-1)[4 : 4 + m * K].view(m, K)

        for first, second in ((self._x(8), offset_x(8)), (offset_x(8), self._x(8))):
            variant = build(
                ht.trace(linear, (first, self.w, self.b)),
                linear,
                (first, self.w, self.b),
            )
            self.assertTrue(self._check(variant, linear, (first, self.w, self.b)))
            with self.assertRaisesRegex(ht.TopologyMiss, "programmatic"):
                variant.replay((second, self.w, self.b))
            self.assertTrue(self._check(variant, linear, (second, self.w, self.b)))
            chains = self._chains(variant)
            self.assertEqual(len(chains), 2, chains)
            self.assertEqual(
                [[k for k, _p in c] for c in chains], [["kernel", "kernel"]] * 2
            )
            self.assertEqual(
                sorted(tuple(p for _k, p in c) for c in chains),
                [(False, False), (True, True)],
            )
            for v, x in zip(self._variants(variant), (first, second)):
                self.assertTrue(
                    self._check(v, linear, (x, self.w, self.b), rebuild=False)
                )

    def test_an_anchored_harvest_needs_one_root(self):
        # two roots (two anchors on forked streams joined before the call) are refused
        # by name: the harvest's capture has exactly one
        x, w = self._x(8), self.w
        a, b = torch.empty(1, device="cuda"), torch.empty(1, device="cuda")
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        g = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(s1):
            linear_nobias(x, w)
            s1.synchronize()
            g.capture_begin(capture_error_mode="thread_local")
            a.fill_(1)
            s2.wait_stream(s1)
            with torch.cuda.stream(s2):
                b.fill_(1)
            s1.wait_stream(s2)
            linear_nobias(x, w)
            g.capture_end()
        s1.synchronize()
        # a is the root; b depends on a: one root still, and b is kept as a node
        anchored = torch._C._host_trace_harvest_nodes(g.raw_cuda_graph(), anchored=True)
        plain = torch._C._host_trace_harvest_nodes(g.raw_cuda_graph())
        self.assertEqual(len(plain), len(anchored) + 1)


_WS_SCRIPT = r"""
import sys
from unittest import mock
import torch, torch.nn.functional as F
from torch.cuda import _host_trace as ht
sys.path.insert(0, sys.argv[1])
from test_cuda_host_trace_gemm import _harvest
K = N = 4096
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K**0.5
b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
lin = lambda x, w, b: F.linear(x, w, b)
for m in (1, 2, 4, 8, 12, 16, 24, 32):
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    try:
        _harvest(ht.trace(lin, (x, w, b)), (x, w, b))
    except ht.Miss as e:
        print("MISS", m, str(e)[:160])
ws = [t for t in ht._gemm_templates.values() if t.uses_ws]
print("WS_TEMPLATES", len(ws), "of", len(ht._gemm_templates))
registered = torch._C._host_trace_blas_workspaces(torch.cuda.current_stream().cuda_stream)
print("REGISTERED", len(registered))
# a stream-dependent pointer that is not a registered workspace: the harvest of a new key misses by name
x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
w2 = torch.randn(N // 2, K, device="cuda", dtype=torch.bfloat16) / K**0.5
b2 = torch.randn(N // 2, device="cuda", dtype=torch.bfloat16)
with mock.patch.object(torch._C, "_host_trace_blas_workspaces", return_value=[]):
    try:
        _harvest(ht.trace(lin, (x, w2, b2)), (x, w2, b2))
        print("NEGATIVE served")
    except ht.Miss as e:
        print("NEGATIVE", type(e).__name__, str(e)[:400])
"""


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="closed regions are CUDA-only in this version")
class TestCudaHostTraceGemmProvenance(TestCase):
    # provenance audit S2 / S3 / G2: the harvester's sanctioned classification
    # of workspace and host slots, and the address bits the two operand sets
    # vary

    def test_workspace_slot_must_equal_the_registered_workspace(self):
        # S3: with the cuBLAS workspace cache on (read once per process, so a
        # subprocess) a variant's image holds the (handle, stream) workspace
        # base, a "ws" slot only because it equals what cuBLAS registered for
        # the harvesting handle and stream; an unregistered stream-dependent
        # pointer misses by name
        import os
        import subprocess
        import sys

        env = dict(os.environ, TORCH_CUBLAS_WORKSPACE_CACHE="1")
        r = subprocess.run(
            [
                sys.executable,
                "-c",
                _WS_SCRIPT,
                os.path.dirname(os.path.abspath(__file__)),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertRegex(out, r"WS_TEMPLATES [1-9]", out[-3000:])
        self.assertRegex(out, r"REGISTERED [1-9]", out[-3000:])
        self.assertIn("NEGATIVE Miss", out, out[-3000:])
        self.assertIn("not the stream's registered cuBLAS workspace", out, out[-3000:])

    def test_host_slot_classes(self):
        # S2: a host slot is admitted as a stack or heap address, the low half
        # of a stack address in a 32-bit field, or stack-smear padding;
        # anything else has no class (the classification is the harvest's;
        # a replay keeps the template's own bytes at these slots)
        stack = (0x7F0000000000, 0x7F0000030000)
        heaps = [(0x5000000, 0x6000000)]

        def imgs(values):
            return tuple(v.to_bytes(8, "little") for v in values)

        cls = ht._host_slot_class
        self.assertEqual(
            cls(0, imgs([stack[0] + 16] * 3 + [stack[0] + 32]), stack, heaps), "stack"
        )
        self.assertEqual(
            cls(0, imgs([heaps[0][0] + 8] * 3 + [heaps[0][0] + 64]), stack, heaps),
            "heap",
        )
        nan = 0x7FC00000
        pad = [nan | (0xA5A5A5A5 << 32)] * 3 + [nan | (0x5A5A5A5A << 32)]
        self.assertEqual(cls(0, imgs(pad), stack, heaps), "padding")
        lo32 = (stack[0] + 0x8400) & 0xFFFFFFFF
        s32 = [lo32 | (1 << 32)] * 3 + [(lo32 + 0xB00) | (1 << 32)]
        self.assertEqual(cls(0, imgs(s32), stack, heaps), "stack32:0")
        self.assertIsNone(
            cls(0, imgs([0x1234567890] * 3 + [0x1234567898]), stack, heaps)
        )
        # a value that only looks like padding: the pattern differs
        self.assertIsNone(
            cls(
                0,
                imgs([nan | (0xA5A5A5A5 << 32)] * 3 + [nan | (0x5A5A5A00 << 32)]),
                stack,
                heaps,
            )
        )

    def test_unclassifiable_host_state_misses_by_name(self):
        # S2, end to end: with no stack or heap mapping admitted, the stack
        # addresses cuBLAS leaves in the split-K reduce image have no class
        # and the harvest misses by name instead of keeping them
        x = torch.randn(4, K, device="cuda", dtype=DTYPE)
        w = torch.randn(N // 4, K, device="cuda", dtype=DTYPE) / K**0.5
        b = torch.randn(N // 4, device="cuda", dtype=DTYPE)
        tape = ht.trace(linear, (x, w, b))
        with mock.patch.object(ht, "_host_mappings", return_value=((1, 2), [])):
            with self.assertRaisesRegex(ht.Miss, "neither a stack nor a heap address"):
                _harvest(tape, (x, w, b))

    def test_harvest_operand_sets_differ_above_the_alignment_class(self):
        # G2: the second operand set flips every address bit between the
        # alignment class and the 2 MiB window, from another allocation, so an
        # image field derived from finer address bits than the class cannot
        # be constant over the two captures
        metas = (
            (torch.bfloat16, (4, 4096), (4096, 1)),
            (torch.bfloat16, (4096, 4096), (1, 4096)),
            (torch.bfloat16, (4, 4096), (4096, 1)),
        )
        aligns = (256, 16, 2)
        dev = torch.device("cuda", torch.cuda.current_device())
        first, _ = ht._harvest_operands(0, metas, aligns, dev)
        second, _ = ht._harvest_operands(1, metas, aligns, dev)
        for a, b, c in zip(first, second, aligns):
            pa, pb = a.data_ptr(), b.data_ptr()
            self.assertEqual(_align_class(pa), c)
            self.assertEqual(_align_class(pb), c)
            mask = (ht._WINDOW - 1) & ~(2 * c - 1)
            self.assertEqual((pa ^ pb) & mask, mask)
            self.assertNotEqual(pa >> 21, pb >> 21)


if __name__ == "__main__":
    run_tests()
