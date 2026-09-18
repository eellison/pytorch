# Owner(s): ["module: cuda"]

import functools
import itertools
import json
import statistics
import time
import unittest
from unittest import mock

from host_trace_testing import HostTraceTestCase

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
            v = ht.build(tape, fn, args)
            variants.append(v)
            got = v.replay(args)
        self.last_served = v
        self._assert_bitwise(got[0], want, what)
        return True

    def _variants(self, variant):
        return self._pool.get(id(variant), [variant])

    def _served_kinds(self, variant, fn, args, what):
        # serves the call and returns the variant that served it with the
        # node kinds of the template the call selected, read from the
        # template cache (the entry whose hit count the call raised)
        before = {t["key"]: t["hits"] for t in ht.gemm_templates()}
        self.assertTrue(self._check(variant, fn, args, what), what)
        hit = [t for t in ht.gemm_templates() if t["hits"] > before.get(t["key"], 0)]
        self.assertEqual(len(hit), 1, what)
        return self.last_served, hit[0]["kinds"]

    def _graph_nodes(self, variant):
        # what the driver says the variant's graph holds: (kind, enabled in
        # the exec) per node in creation order
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            self.skipTest("cuda-python (cuda.bindings) is not installed")
        check = torch.cuda._utils._check_cuda_bindings
        types = cudart.cudaGraphNodeType
        names = {
            types.cudaGraphNodeTypeKernel: "kernel",
            types.cudaGraphNodeTypeMemset: "memset",
            types.cudaGraphNodeTypeMemcpy: "memcpy",
        }
        graph = variant.graph.raw_cuda_graph()
        exec_ = variant.graph.raw_cuda_graph_exec()
        _, count = check(cudart.cudaGraphGetNodes(graph, 0))
        nodes, count = check(cudart.cudaGraphGetNodes(graph, count))
        e = variant.exec
        self.assertEqual(count, e.num_nodes + e.num_memset_nodes + e.num_memcpy_nodes)
        return [
            (
                names.get(check(cudart.cudaGraphNodeGetType(n)), "other"),
                check(cudart.cudaGraphNodeGetEnabled(exec_, n)),
            )
            for n in nodes
        ]

    def test_linear_traces_as_a_closed_region(self):
        x = self._x(4)
        tape = ht.trace(linear, (x, self.w, self.b))
        self.assertEqual(tape.num_regions, 1)
        self.assertEqual(tape.num_launches, 0)
        r = json.loads(tape.to_json())["regions"][0]
        self.assertEqual(r["op"], "addmm")
        self.assertEqual([i["name"] for i in r["inputs"]], ["bias", "mat1", "mat2"])
        # every input dimension is a symbol on the tape; M and N differ
        self.assertIsInstance(r["out"]["sizes"][0], str)
        self.assertNotEqual(r["out"]["sizes"][0], r["out"]["sizes"][1])
        variant = ht.build(tape, linear, (x, self.w, self.b))
        self.assertTrue(self._check(variant, linear, (self._x(4), self.w, self.b)))
        stats = variant.region_stats()
        self.assertEqual(len(stats["sites"]), 1)

    def test_bitwise_m_sweep_crosses_variant_boundaries(self):
        x = self._x(4)
        tape = ht.trace(linear, (x, self.w, self.b))
        variant = ht.build(tape, linear, (x, self.w, self.b))
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
        used = len(self._templates_for()) - before
        # one exec per node chain the sweep selected (split-K at small M
        # adds the reduce kernel), each chain held by exactly one exec
        variants = self._variants(variant)
        chains = [v.topology[0] for v in variants]
        self.assertEqual(len(set(chains)), len(chains), chains)
        self.assertGreaterEqual(len(variants), 2, chains)
        print(
            f"\n[gemm M sweep 1..64] templates used {used + 1} execs {len(variants)} chains {chains} applies {[v.region_stats()['applies'] for v in variants]}"
        )
        # GEMM_FACTS: 5-8 variants over M = 1..128 at this shape
        self.assertGreaterEqual(used + 1, 3)

    def test_split_k_boundary_builds_a_second_exec_from_the_same_tape(self):
        # M = 8 runs as two nodes (nvjet + splitKreduce), M = 12 as one. A
        # variant holds exactly the chain of the shape it was built at; the
        # other M is a TopologyMiss carrying the tape, and the tape built at
        # that M (no trace; the miss harvested the template, the build finds
        # it) is a second variant with the other chain. Each serves its
        # class bitwise; neither exec holds a node the other's chain needs.
        for m_trace, m_other in ((12, 8), (8, 12), (12, 4), (4, 24)):
            x = self._x(m_trace)
            tape = ht.trace(linear, (x, self.w, self.b))
            first = ht.build(tape, linear, (x, self.w, self.b))
            other = (self._x(m_other), self.w, self.b)
            with self.assertRaisesRegex(ht.TopologyMiss, "graph holds") as cm:
                first.replay(other)
            self.assertIs(cm.exception.tape, tape)
            h0 = ht.gemm_harvests()
            second = ht.build(cm.exception.tape, linear, other)
            self.assertEqual(ht.gemm_harvests(), h0)
            self.assertIsNot(second.exec, first.exec)
            chains = (first.topology[0], second.topology[0])
            self.assertEqual(sorted(len(c) for c in chains), [1, 2], chains)
            for v in (first, second):
                nodes = v.region_stats()["sites"][0]["nodes"]
                self.assertEqual(len(nodes), len(v.topology[0]), nodes)
            for m in (m_other, m_trace, m_other, m_other, m_trace):
                serving, idle = (second, first) if m == m_other else (first, second)
                args, what = (self._x(m), self.w, self.b), f"{m_trace}->{m}"
                self.assertTrue(self._check(serving, linear, args, what, rebuild=False))
                self.assertFalse(self._check(idle, linear, args, rebuild=False))
                self.assertIn("graph holds", self.last_miss)
            print(
                f"\n[split-K] traced M={m_trace}: chains {chains} applies "
                f"{[v.region_stats()['applies'] for v in (first, second)]}"
            )

    def test_cluster_change_goes_through_the_graph(self):
        # 4096 -> 11008: the M = 12 variant launches with an 8x1x1 cluster,
        # the M = 24 variant with 4x1x1 (SUPERSET.md); the exec-level
        # setter carries no attributes, so that swap goes through the graph
        w = torch.randn(11008, K, device="cuda", dtype=DTYPE) / K**0.5
        b = torch.randn(11008, device="cuda", dtype=DTYPE)
        # node and cuGraphExecUpdate, and stays bitwise. M = 8 and 16 split
        # K (two nodes), 12, 24 and 32 do not: two execs, and the cluster
        # change 8 -> 4 happens inside the one-node exec (12 -> 24)
        x = self._x(8)
        variant = ht.build(ht.trace(linear, (x, w, b)), linear, (x, w, b))
        seen = []
        for m in (8, 12, 8, 16, 12, 24, 32, 8):
            self.assertTrue(self._check(variant, linear, (self._x(m), w, b), f"M={m}"))
            seen.append([v.exec.graph_updates for v in self._variants(variant)])
        print(
            f"\n[cluster change] graph updates per exec after each M in (8,12,8,16,12,24,32,8): {seen}"
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
        variant = ht.build(tape, chain, (x, *ws))
        # one harvest for the build key, however many sites share the shape
        self.assertLessEqual(ht.gemm_harvests() - h0, 1)
        h1 = ht.gemm_harvests()
        self.assertTrue(self._check(variant, chain, (self._x(16, 1024), *ws)))
        self.assertLessEqual(ht.gemm_harvests() - h1, 1)
        self.assertTrue(self._check(variant, chain, (self._x(4, 1024), *ws)))
        stats = variant.region_stats()
        self.assertEqual(len(stats["sites"]), n_sites)

    def test_per_call_cost(self):
        x = self._x(4)
        tape = ht.trace(linear, (x, self.w, self.b))
        variant = ht.build(tape, linear, (x, self.w, self.b))
        args = (self._x(4), self.w, self.b)
        # M = 16 runs as one node where M = 4 splits K: another exec from the
        # same tape; M = 12 is another kernel of the one-node chain, applied
        # in place
        other = (self._x(16), self.w, self.b)
        same_chain = (self._x(12), self.w, self.b)
        v16 = ht.build(tape, linear, other)
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
            f"pointer rebind {rebind:.1f} exec switch 4<->16 {switch:.1f} "
            f"kernel switch 16<->12 in one exec {apply_:.1f}"
        )
        self.assertGreater(eager, 0)

    def test_mm_and_addmm(self):
        def f(a, b, c):
            return torch.addmm(c, a, b) + torch.mm(a, b)

        a, b, c = self._x(4), self._x(K, N), self.b
        tape = ht.trace(f, (a, b, c))
        self.assertEqual(tape.num_regions, 2)
        variant = ht.build(tape, f, (a, b, c))
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
        variant = ht.build(
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
        variant = ht.build(tape, f, (x, w))
        for s in (3, 5, 8):
            xx = torch.randn(2, s, K, device="cuda", dtype=DTYPE)
            self.assertTrue(self._check(variant, f, (xx, w), f"S={s}"))

    def test_dtype_and_layout_changes_miss_by_name(self):
        x = self._x(4)
        tape = ht.trace(linear_nobias, (x, self.w))
        variant = ht.build(tape, linear_nobias, (x, self.w))
        with self.assertRaises(ht.Miss):
            variant.replay((x.half(), self.w.half()))
        with self.assertRaises(ht.Miss):
            variant.replay((x.float(), self.w.float()))
        # a weight stored transposed: other strides are another template
        # key (another cuBLAS kernel), served by the same variant
        wt = self.w.t().contiguous().t()
        h0 = ht.gemm_harvests()
        self.assertTrue(self._check(variant, linear_nobias, (self._x(4), wt)))
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
                ht.build(tape, linear_nobias, (x, w)).replay((x, w))
        key = f"{N + 192}"
        misses = [t["miss"] for t in ht.gemm_templates() if key in str(t["key"])]
        self.assertEqual(len(misses), 1)
        self.assertIn("does not report kernel node attribute probe", misses[0])

    def test_blas_setting_change_is_a_new_key(self):
        w = self.w.float()
        x = self._x(4).float()
        tape = ht.trace(linear_nobias, (x, w))
        variant = ht.build(tape, linear_nobias, (x, w))
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
            self.assertGreaterEqual(ht.gemm_harvests() - h0, 1)
            print(f"\n[fp32 after precision flip] served {served}")
        finally:
            torch.backends.cuda.matmul.fp32_precision = old
        self.assertTrue(self._check(variant, linear_nobias, (self._x(4).float(), w)))

    def test_replay_on_another_stream(self):
        x = self._x(4)
        tape = ht.trace(linear, (x, self.w, self.b))
        variant = ht.build(tape, linear, (x, self.w, self.b))
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for m in (4, 8, 16):
                self.assertTrue(
                    self._check(variant, linear, (self._x(m), self.w, self.b), f"M={m}")
                )

    def test_two_variants_interleaved(self):
        x4, x32 = self._x(4), self._x(32)
        v4 = ht.build(
            ht.trace(linear, (x4, self.w, self.b)), linear, (x4, self.w, self.b)
        )
        v32 = ht.build(
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
            for o in [*r["inputs"], r["out"]]:
                self.assertIsInstance(o["address"], str)
                self.assertNotIn(
                    str(int(o["address"], 0)) if o["address"].isdigit() else "", real
                )
                self.assertTrue(any(ch.isalpha() for ch in o["address"]), o["address"])
        variant = ht.build(tape, linear, (x, self.w, self.b))
        self.assertTrue(self._check(variant, linear, (self._x(8), self.w, self.b)))

    def test_gemm_templates_introspection(self):
        x = self._x(4)
        variant = ht.build(
            ht.trace(linear, (x, self.w, self.b)), linear, (x, self.w, self.b)
        )
        variant.replay((self._x(4), self.w, self.b))
        entries = ht.gemm_templates()
        self.assertGreaterEqual(len(entries), 1)
        for e in entries:
            self.assertEqual(
                set(e),
                {
                    "key",
                    "kernels",
                    "kinds",
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
        return [v.topology[k] for v in self._variants(variant)]

    def _kinds(self, variant, k=0):
        return [kind for kind, _i in variant.region_stats()["sites"][k]["nodes"]]

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
            variant = ht.build(ht.trace(linear, (x, w, b)), linear, (x, w, b))
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

    def test_fp32_split_at_large_m_is_another_exec(self):
        # F3: the fp32 SIMT path splits K at large M (cutlass simt + reduce)
        # where M = 4 runs one kernel; a tape traced at M = 4 serves 33 and
        # 64 through a second exec built from it at those inputs
        old = torch.backends.cuda.matmul.fp32_precision
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        try:
            w, x = self.w.float(), self._fp32(4)
            variant = ht.build(ht.trace(linear_nobias, (x, w)), linear_nobias, (x, w))
            self.assertEqual(self._kinds(variant), ["kernel"])
            served, refused = self._serve(
                variant,
                linear_nobias,
                lambda m: (self._fp32(m), w),
                (4, 33, 64, 1, 4, 64),
                "fp32 no-bias",
            )
            self.assertEqual(refused, [])
            chains = self._chains(variant)
            self.assertEqual(len(chains), 2, chains)
            self.assertEqual(sorted(len(c) for c in chains), [1, 2], chains)
            print(f"\n[fp32 split at large M] chains {chains} served {served}")
        finally:
            torch.backends.cuda.matmul.fp32_precision = old

    def test_exec_holds_exactly_the_served_templates_nodes(self):
        # an exec holds the nodes of the template its calls select, one for
        # one: no other node of any kind (no memset parked on a dummy word,
        # no kernel of another chain) and none disabled, read back from the
        # driver after kernel changes in place and the topology miss that
        # made a second exec
        x = self._x(4)
        variant = ht.build(
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
            self.assertEqual([k for k, _e in nodes], list(v.topology[0]))
            self.assertEqual([e for _k, e in nodes], [1] * len(nodes), f"M={m}")

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
        variant = ht.build(tape, two, (x, w1, w2))
        for m in (4, 16, 64, 4):
            args = (self._x(m, 1024), w1, w2)
            v, kinds = self._served_kinds(variant, two, args, f"M={m}")
            nodes = self._graph_nodes(v)
            self.assertEqual(
                [k for k, _e in nodes], kinds + ["kernel"] + kinds, f"M={m}"
            )
            self.assertEqual([e for _k, e in nodes], [1] * len(nodes), f"M={m}")

    def test_odd_and_narrow_shapes(self):
        # F2: N or K = 4097 runs cutlass 2.x behind a semaphore memset; the
        # narrow shapes cross the gemv / split-K variants
        for n, k in ((4097, K), (N, 4097), (3, K), (N, 3)):
            w = torch.randn(n, k, device="cuda", dtype=DTYPE) / k**0.5
            b = torch.randn(n, device="cuda", dtype=DTYPE)
            x = self._x(8, k)
            variant = ht.build(ht.trace(linear, (x, w, b)), linear, (x, w, b))
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
        variant = ht.build(
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
        # F2: the host copies such an operand into a temporary of its own
        # before the library call; the copy kernel is a node of the region
        # and its temporary is scratch of the arena
        def strided(m):
            return self._x(m, 2 * K)[:, ::2]

        def expanded(m):
            return self._x(1, K).expand(m, K)

        for name, x_of in (("strided", strided), ("expanded", expanded)):
            x = x_of(8)
            try:
                variant = ht.build(
                    ht.trace(linear, (x, self.w, self.b)), linear, (x, self.w, self.b)
                )
            except ht.Declined as e:
                self.assertIn("declined", str(e))
                print(f"\n[{name} x] declined at trace: {e}")
                continue
            kinds = self._kinds(variant)
            served, refused = self._serve(
                variant,
                linear,
                lambda m, x_of=x_of: (x_of(m), self.w, self.b),
                (8, 1, 4, 64, 8),
                name,
            )
            self.assertEqual(refused, [], f"{name}: {kinds}")
            scratch = [
                t["scratch"]
                for t in ht.gemm_templates()
                if t["hits"] and t["kinds"] == kinds
            ]
            print(f"\n[{name} x] site {kinds} served {served} scratch {scratch[:1]}")
            # a contiguous x through the same variant: another key, no copy
            self.assertTrue(
                self._check(
                    variant,
                    linear,
                    (self._x(8), self.w, self.b),
                    f"{name}: contiguous x",
                ),
                self.last_miss,
            )

    def test_arena_shared_across_sites_and_variants(self):
        # F4: one arena per replay stream; the two sites of a variant and two
        # variants replayed on one stream share its workspace and scratch,
        # split-K partials of both sites included
        def two(x, w1, b1, w2, b2):
            return F.linear(F.linear(x, w1, b1), w2, b2)

        w2 = torch.randn(N, K, device="cuda", dtype=DTYPE) / K**0.5
        b2 = torch.randn(N, device="cuda", dtype=DTYPE)
        x = self._x(8)
        variant = ht.build(
            ht.trace(two, (x, self.w, self.b, w2, b2)), two, (x, self.w, self.b, w2, b2)
        )
        allocated = torch.cuda.memory_allocated()
        variant2 = ht.build(
            ht.trace(linear, (x, self.w, self.b)), linear, (x, self.w, self.b)
        )
        # the second variant owns no workspace: its graph pool holds the output
        self.assertLess(torch.cuda.memory_allocated() - allocated, 8 << 20)
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
        # one arena for this stream (the builds' own aside), none per variant
        stream = torch.cuda.current_stream().cuda_stream
        arena = ht._arena(x.device.index, stream)
        self.assertEqual(
            [k for k in ht._arenas if k[0] == x.device.index and k[1] is not None],
            [(x.device.index, stream)],
        )
        self.assertTrue(arena.bufs)
        ws = arena.ws.numel() >> 20 if arena.ws is not None else None
        print(
            f"\n[arena] sites {[self._kinds(variant, k) for k in range(2)]} workspace {ws} MiB scratch {[b.numel() for b in arena.bufs]}"
        )

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
        # template)
        x = self._x(8)
        ht.build(ht.trace(linear, (x, self.w, self.b)), linear, (x, self.w, self.b))
        k = 8256 + 8 * next(_fresh_harvest_keys)
        w = torch.randn(8192, k, device="cuda", dtype=DTYPE) / k**0.5
        b = torch.randn(8192, device="cuda", dtype=DTYPE)
        x = self._x(8, k)
        harvests = ht.gemm_harvests()
        before = torch.cuda.memory_allocated()
        variant = ht.build(ht.trace(linear, (x, w, b)), linear, (x, w, b))
        self.assertEqual(ht.gemm_harvests(), harvests + 1)
        self.assertLess(torch.cuda.memory_allocated() - before, 64 << 20)
        self.assertTrue(self._check(variant, linear, (x, w, b), f"8192 x {k}"))

    def test_no_input_is_written_through_a_region(self):
        # every out= and in-place GEMM variant declines by name, so a closed
        # region's out operand is an allocation the call made and never an
        # input: Tape.written_inputs (commit 1) gets no region entry
        x = self._x(8)
        tape = ht.trace(linear, (x, self.w, self.b))
        self.assertEqual(len(tape.regions), 1)
        self.assertTrue(tape.regions[0].out.root.allocation)
        self.assertEqual(tape.written_roots, [tape.regions[0].out.root.name])
        self.assertEqual(tape.written_inputs, ())
        y = torch.empty(8, N, device="cuda", dtype=DTYPE)
        wt = self.w.t().contiguous()
        with self.assertRaisesRegex(ht.Declined, "aten.mm.out"):
            ht.trace(lambda a, b, o: torch.mm(a, b, out=o), (x, wt, y))
        with self.assertRaisesRegex(ht.Declined, "aten.addmm_.default"):
            ht.trace(lambda c, a, b: c.addmm_(a, b), (y, x, wt))

    def test_bmm_k1_rotary_product_takes_the_outer_product_sibling(self):
        # LlamaRotaryEmbedding.forward's inv_freq_expanded @ position_ids_expanded:
        # a (B, 32, 1) x (B, 1, 1) fp32 bmm with batch1 expanded over the batch
        # (batch stride 0), reached through matmul's reshape. Eager serves the
        # K = 1 product with torch._native's Triton outer-product kernel, not
        # cuBLAS (bmm.out would reach cuBLAS's gemmk1 GEMV kernel), so the
        # trace routes it to the broadcast-multiply sibling: no region, one
        # launch beside the position cast, bitwise
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
        h0 = ht.gemm_harvests()
        variant = ht.build(tape, rotary, base)
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
        # the same product in bf16 and with a wider N, straight through bmm
        for dtype in (torch.bfloat16, torch.float32):
            a = torch.randn(3, 40, 1, device="cuda", dtype=dtype)
            b = torch.randn(3, 1, 24, device="cuda", dtype=dtype)
            tape = ht.trace(torch.bmm, (a, b))
            self.assertEqual((tape.num_regions, tape.num_launches), (0, 1))
            variant = ht.build(tape, torch.bmm, (a, b))
            for args in (
                (a, b),
                (
                    torch.randn(5, 7, 1, device="cuda", dtype=dtype),
                    torch.randn(5, 1, 130, device="cuda", dtype=dtype),
                ),
            ):
                self.assertTrue(
                    self._check(variant, torch.bmm, args, str(dtype)), self.last_miss
                )
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
            f"\n[bmm K=1] functional: outer-product sibling; bmm.out: {[n[:32] for n in names]}"
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
            variant = ht.build(tape, torch.bmm, base)
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
            variant_t = ht.build(ht.trace(torch.bmm, (a, bt)), torch.bmm, (a, bt))
            bt2 = torch.randn(8, 64, 64, device="cuda", dtype=dtype).transpose(1, 2)
            self.assertTrue(
                self._check(variant_t, torch.bmm, (a, bt2), "transposed"),
                self.last_miss,
            )
            stats = variant.region_stats()
            print(
                f"\n[bmm 64^3 {dtype}] nodes {stats['sites'][0]['nodes']} applies {stats['applies']} rebinds {stats['rebinds']}"
            )

    def test_bmm_declines_by_name(self):
        a = torch.randn(4, 16, 64, device="cuda", dtype=DTYPE)
        b = torch.randn(4, 64, 24, device="cuda", dtype=DTYPE)
        # a batch operand the host would copy first (a slice-step layout)
        with self.assertRaisesRegex(ht.Declined, "not a cuBLAS batch operand as is"):
            ht.trace(
                torch.bmm,
                (a, torch.randn(4, 64, 48, device="cuda", dtype=DTYPE)[:, :, ::2]),
            )
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
        # result: the region's site is built with no node (nothing to anchor
        # the probe rows' nodes at) and serves that; a shape with rows misses
        # on the empty output's stride guard, before the site
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
                variant = ht.build(tape, fn, (x, w))
                self.assertEqual(variant.region_stats()["sites"][0]["nodes"], [])
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


_WS_SCRIPT = r"""
import sys
from unittest import mock
import torch, torch.nn.functional as F
from torch.cuda import _host_trace as ht
K = N = 4096
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K**0.5
b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
lin = lambda x, w, b: F.linear(x, w, b)
x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
v = ht.build(ht.trace(lin, (x, w, b)), lin, (x, w, b))
for m in (1, 2, 4, 8, 12, 16, 24, 32):
    try:
        v.replay((torch.randn(m, K, device="cuda", dtype=torch.bfloat16), w, b))
    except ht.Miss as e:
        print("MISS", m, str(e)[:160])
ws = [t for t in ht._gemm_templates.values() if t.uses_ws]
print("WS_TEMPLATES", len(ws), "of", len(ht._gemm_templates))
registered = torch._C._host_trace_blas_workspaces(torch.cuda.current_stream().cuda_stream)
print("REGISTERED", len(registered))
# a stream-dependent pointer that is not a registered workspace: the harvest of a new key misses by name
w2 = torch.randn(N // 2, K, device="cuda", dtype=torch.bfloat16) / K**0.5
b2 = torch.randn(N // 2, device="cuda", dtype=torch.bfloat16)
with mock.patch.object(torch._C, "_host_trace_blas_workspaces", return_value=[]):
    try:
        ht.build(ht.trace(lin, (x, w2, b2)), lin, (x, w2, b2)).replay((x, w2, b2))
        print("NEGATIVE served")
    except (ht.Miss, ht.TapeMismatch) as e:
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
            [sys.executable, "-c", _WS_SCRIPT],
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
        # anything else has no class
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
        maps = (stack, heaps)
        self.assertTrue(
            ht._host_slot_matches(
                "stack", (stack[0] + 4096).to_bytes(8, "little"), 0, maps
            )
        )
        self.assertFalse(
            ht._host_slot_matches("stack", heaps[0][0].to_bytes(8, "little"), 0, maps)
        )
        self.assertTrue(
            ht._host_slot_matches(
                "heap", (heaps[0][0] + 1).to_bytes(8, "little"), 0, maps
            )
        )
        self.assertTrue(
            ht._host_slot_matches(
                "stack32:0", (lo32 | (7 << 32)).to_bytes(8, "little"), 0, maps
            )
        )
        self.assertFalse(
            ht._host_slot_matches(
                "stack32:0", (0x11223344 | (7 << 32)).to_bytes(8, "little"), 0, maps
            )
        )
        self.assertTrue(ht._host_slot_matches("padding", bytes(8), 0, maps))

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
                ht.build(tape, linear, (x, w, b))

    def test_build_checks_the_captures_host_slots_by_class(self):
        # S2, build side: the capture's value at a host slot must be of the
        # template's kind (a stack slot holds a stack address of the build
        # thread), not just anything
        x = torch.randn(4, K, device="cuda", dtype=DTYPE)
        w = torch.randn(N, K, device="cuda", dtype=DTYPE) / K**0.5
        b = torch.randn(N, device="cuda", dtype=DTYPE)
        tape = ht.trace(linear, (x, w, b))
        variant = ht.build(tape, linear, (x, w, b))
        self.assertTrue(variant.region_stats()["sites"])
        stack, heaps = ht._host_mappings()
        with mock.patch.object(
            ht, "_host_mappings", return_value=((stack[1], stack[1] + 4096), heaps)
        ):
            with self.assertRaisesRegex(ht.TapeMismatch, "host slot at byte"):
                ht.build(tape, linear, (x, w, b))

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
            self.assertEqual(ht._align_class(pa), c)
            self.assertEqual(ht._align_class(pb), c)
            mask = (ht._WINDOW - 1) & ~(2 * c - 1)
            self.assertEqual((pa ^ pb) & mask, mask)
            self.assertNotEqual(pa >> 21, pb >> 21)


if __name__ == "__main__":
    run_tests()
