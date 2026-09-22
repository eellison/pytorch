# Owner(s): ["module: inductor"]
"""Closed cuBLAS GEMM regions (commit 10) lowered into the shared native replay: the
kernel nodes come from the runtime's template registry, selected per call by the
region's operand shapes and alignment; a new key misses once (harvest), then serves.
A template's node chain is part of the variant's class (E28): an exec holds exactly
the nodes of the templates its build selected, a key whose template has another chain
is a TopologyMiss the entry serves by building the same tape at the call's inputs."""

import statistics
import time

import torch
import torch.nn.functional as F
from torch.cuda._utils import _check_cuda_bindings
from torch.testing._internal.common_utils import run_tests, TestCase


# a decode-sized projection: K = N = 4096 crosses several cuBLAS variant boundaries
# over M = 1..64 and splits K at small M (GEMM_FACTS Q1)
K = N = 4096
DTYPE = torch.bfloat16


def linear(x, w, b):
    return F.linear(x, w, b)


def linear_nobias(x, w):
    return F.linear(x, w)


class TestHostTraceGemm(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from torch._inductor.runtime._cudagraph import direct_hosttrace
        from torch.cuda import _host_trace

        self.module = direct_hosttrace
        self.ht = _host_trace
        torch.manual_seed(0)
        self.w = torch.randn(N, K, device="cuda", dtype=DTYPE) / K**0.5
        self.b = torch.randn(N, device="cuda", dtype=DTYPE)

    def _x(self, m, k=K):
        return torch.randn(m, k, device="cuda", dtype=DTYPE)

    def _replay(self, fn, args):
        replay = self.module.HostTraceReplay(fn, args)
        self.addCleanup(replay.close)
        return replay

    def _check(self, replay, fn, args, what=""):
        """Bitwise against eager whether served or not; returns whether it was served."""
        want = fn(*args)
        before = replay.misses
        got = replay(*args)
        self.assertEqual(got.shape, want.shape, what)
        self.assertTrue(torch.equal(got, want), what)
        return replay.misses == before

    def _exec_states(self, replay):
        """(kind, enabled) per node of every variant's exec, in cudaGraphGetNodes
        order, read back with cudaGraphNodeGetEnabled."""
        from cuda.bindings import runtime

        kinds = {
            runtime.cudaGraphNodeType.cudaGraphNodeTypeKernel: "kernel",
            runtime.cudaGraphNodeType.cudaGraphNodeTypeMemset: "memset",
            runtime.cudaGraphNodeType.cudaGraphNodeTypeMemcpy: "memcpy",
        }
        out = []
        for variant in replay.variants:
            graph, exec_ = variant.lowered.capture_handles
            count = _check_cuda_bindings(runtime.cudaGraphGetNodes(graph, 0))[1]
            nodes = _check_cuda_bindings(runtime.cudaGraphGetNodes(graph, count))[0]
            states = []
            for node in nodes:
                kind = kinds.get(
                    _check_cuda_bindings(runtime.cudaGraphNodeGetType(node))
                )
                enabled = (
                    bool(
                        _check_cuda_bindings(
                            runtime.cudaGraphNodeGetEnabled(exec_, node)
                        )
                    )
                    if kind
                    else None
                )
                states.append((kind or "other", enabled))
            out.append(states)
        return out

    def _assert_execs_hold_their_chains(self, replay, what):
        """Every variant's exec holds exactly the tape's events in host order (a
        kernel node per launch, a memset / memcpy node each) with each site's template
        chain in place, one node for one, read back from the driver, all enabled."""
        for variant, states in zip(replay.variants, self._exec_states(replay)):
            lowered, tape = variant.lowered, variant.tape
            events = sorted(
                [(int(L["seq"]), ["kernel"]) for L in tape.launches]
                + [(seq, ["memset"]) for seq, *_ in lowered.memsets]
                + [(seq, ["memcpy"]) for seq, *_ in lowered.memcpys]
                + [
                    (r.seq, [k for k, _ in lowered.region_nodes[r.site]])
                    for r in lowered.regions
                ],
                key=lambda ev: ev[0],
            )
            expected = [kind for _, kinds in events for kind in kinds]
            self.assertEqual(
                [kind for kind, _ in states], expected, f"{what}: {states}"
            )
            self.assertTrue(
                all(enabled for _, enabled in states),
                f"{what}: a disabled node in {states}",
            )

    def _serve(self, replay, fn, args, what=""):
        """A key the site has not seen misses once (the miss harvests and registers);
        the second call must serve."""
        if self._check(replay, fn, args, what):
            return 0
        self.assertTrue(self._check(replay, fn, args, what), f"{what}: still missing")
        return 1

    def test_linear_lowers_as_a_region(self):
        replay = self._replay(linear, (self._x(4), self.w, self.b))
        self.assertEqual(len(replay.lowered.regions), 1)
        self.assertEqual(replay.lowered.calls, ())
        region = replay.lowered.regions[0]
        self.assertEqual(region.op, "addmm")
        self.assertEqual(region.ranks, (1, 2, 2, 2))
        self.assertTrue(self._check(replay, linear, (self._x(4), self.w, self.b)))
        stats = replay.region_stats()
        self.assertEqual(len(stats["sites"]), 1)
        # the exec holds exactly the M = 4 template's chain (nvjet + splitKreduce at
        # this shape), read back from the driver
        self.assertEqual(stats["sites"][0]["nodes"], len(stats["sites"][0]["kinds"]))
        self._assert_execs_hold_their_chains(replay, "after the preparation")
        self.assertEqual(stats["refused"], {})

    def test_m_sweep_crosses_variant_boundaries_bitwise(self):
        replay = self._replay(linear, (self._x(4), self.w, self.b))
        h0 = self.ht.gemm_harvests()
        misses = 0
        for m in range(1, 65):
            misses += self._serve(
                replay, linear, (self._x(m), self.w, self.b), f"M={m}"
            )
        # every new M is a new key: one miss each, except the preparation shape; an M
        # whose chain differs (K = 4096 splits K at small M) is a topology miss, served
        # by a second exec of the same tape
        stats = replay.region_stats()
        print(
            f"\n[gemm M sweep 1..64] misses {misses} harvests {self.ht.gemm_harvests() - h0} "
            f"applies {stats['applies']} graph updates {stats['graph_updates']} "
            f"execs {len(replay.variants)} chains {[s['kinds'] for s in stats['sites']]}"
        )
        self.assertLessEqual(misses, 63)
        self.assertEqual(stats["refused"], {})
        self.assertEqual(replay.ordinary, 0)
        self.assertEqual(replay.traces, 1)  # every exec is the one tape
        self.assertGreaterEqual(len(replay.variants), 2)
        classes = {
            (tuple(s["kinds"]), tuple(s["programmatic"])) for s in stats["sites"]
        }
        self.assertEqual(len(classes), len(replay.variants))
        self._assert_execs_hold_their_chains(replay, "after the sweep")
        # a second sweep serves every M at once
        for m in range(1, 65):
            self.assertTrue(
                self._check(replay, linear, (self._x(m), self.w, self.b), f"M={m}")
            )

    def test_split_k_boundary_builds_a_second_exec_from_the_same_tape(self):
        # M = 8 runs as two nodes (nvjet + splitKreduce), M = 12 as one; prepared at
        # either, the other M is a topology miss: the same tape (no re-trace) built at
        # that call's inputs is a second variant with the other chain, and each exec
        # serves its own class from then on
        for m_prepare, m_other in ((12, 8), (8, 12), (12, 4), (4, 24)):
            replay = self._replay(linear, (self._x(m_prepare), self.w, self.b))
            first = replay.region_stats()["sites"][0]["kinds"]
            h0 = self.ht.gemm_harvests()
            self.assertEqual(
                self._serve(replay, linear, (self._x(m_other), self.w, self.b)), 1
            )
            self.assertEqual(len(replay.variants), 2, f"{m_prepare}->{m_other}")
            self.assertEqual(replay.traces, 1)
            self.assertIs(replay.variants[1].tape, replay.variants[0].tape)
            self.assertEqual(replay.ordinary, 0)
            self.assertLessEqual(self.ht.gemm_harvests() - h0, 1)  # the miss harvested
            second = replay.region_stats()["sites"][1]["kinds"]
            self.assertNotEqual(first, second)
            self.assertEqual({len(first), len(second)}, {1, 2})
            for m in (m_prepare, m_other, m_other, m_prepare):
                self.assertTrue(
                    self._check(
                        replay,
                        linear,
                        (self._x(m), self.w, self.b),
                        f"{m_prepare}->{m}",
                    )
                )
            self.assertEqual(len(replay.variants), 2)
            self._assert_execs_hold_their_chains(replay, f"{m_prepare}/{m_other}")

    def test_cluster_change_goes_through_the_graph(self):
        # 4096 -> 11008: the M = 12 variant launches with an 8x1x1 cluster, the M = 8
        # split variant with 4x1x1 (SUPERSET.md); the exec-level setter carries no
        # attributes, so that swap goes through the graph node and cuGraphExecUpdate
        w = torch.randn(11008, K, device="cuda", dtype=DTYPE) / K**0.5
        b = torch.randn(11008, device="cuda", dtype=DTYPE)
        replay = self._replay(linear, (self._x(8), w, b))
        seen = []
        for m in (8, 12, 8, 16, 12, 24, 32, 8):
            self._serve(replay, linear, (self._x(m), w, b), f"M={m}")
            seen.append(replay.region_stats()["graph_updates"])
        print(
            f"\n[cluster change] graph updates after each M in (8,12,8,16,12,24,32,8): {seen}"
        )
        self.assertGreater(seen[-1], 0)

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

        h0 = self.ht.gemm_harvests()
        replay = self._replay(chain, (self._x(4, 1024), *ws))
        self.assertEqual(len(replay.lowered.regions), n_sites)
        # one harvest for the preparation key, however many sites share the shape
        self.assertLessEqual(self.ht.gemm_harvests() - h0, 1)
        h1 = self.ht.gemm_harvests()
        self._serve(replay, chain, (self._x(16, 1024), *ws))
        self.assertLessEqual(self.ht.gemm_harvests() - h1, 1)
        self.assertTrue(self._check(replay, chain, (self._x(4, 1024), *ws)))
        self.assertTrue(self._check(replay, chain, (self._x(16, 1024), *ws)))

    def test_memset_bearing_templates_are_their_own_class(self):
        # fp32 addmm 3072 -> 768: at M = 64 cuBLAS runs [memset, kernel] (a split-K
        # semaphore reset into the call's own scratch), at M = 128 [memset, kernel,
        # kernel, kernel], at M = 4 two kernels and at M = 1 one. Each chain is a class
        # of the tape: an exec per chain, built from the one tape at the first call of
        # its class; the binding drives the memset node like the kernels, and no exec
        # holds a node its template does not have
        w = torch.randn(768, 3072, device="cuda") / 3072**0.5
        b = torch.randn(768, device="cuda")

        def x(m):
            return torch.randn(m, 3072, device="cuda")

        replay = self._replay(linear, (x(64), w, b))
        kinds = replay.region_stats()["sites"][0]["kinds"]
        self.assertEqual(kinds[0], "memset")
        for m in (64, 128, 4, 64, 1, 128, 4, 64, 1):
            self._serve(replay, linear, (x(m), w, b), f"M={m}")
            self._assert_execs_hold_their_chains(replay, f"after M={m}")
        stats = replay.region_stats()
        chains = {(tuple(s["kinds"]), tuple(s["programmatic"])) for s in stats["sites"]}
        self.assertEqual(len(replay.variants), len(chains))
        self.assertGreaterEqual(len(chains), 2)
        self.assertEqual(replay.traces, 1)
        self.assertEqual(replay.ordinary, 0)
        self.assertEqual(stats["refused"], {})
        print(
            f"\n[memset chain] chains {sorted(chains, key=lambda c: len(c[0]))}, execs {len(replay.variants)}, applies {stats['applies']}"
        )
        # every exec serves its class at once now
        for m in (64, 128, 4, 1):
            self.assertTrue(self._check(replay, linear, (x(m), w, b), f"M={m} again"))

    def test_a_topology_miss_carries_the_tape_and_is_no_decline(self):
        # the typed miss the family's dispatch reports for a chain the exec does not
        # hold: the entry builds from the tape it already has (no trace, no ordinary
        # call, no declined class), and the miss log names the chain
        replay = self._replay(linear, (self._x(12), self.w, self.b))
        misses, log = replay.misses, len(replay.miss_log)
        # the first M = 8 call is a miss served natively within the call (bitwise)
        self.assertFalse(self._check(replay, linear, (self._x(8), self.w, self.b)))
        self.assertEqual(replay.misses, misses + 1)
        self.assertTrue(self._check(replay, linear, (self._x(8), self.w, self.b)))
        self.assertEqual((replay.traces, replay.ordinary, replay.declines), (1, 0, []))
        self.assertEqual(replay.declined_classes, ())
        why = " ".join(str(row) for row in replay.miss_log[log:])
        self.assertIn("build the tape at these inputs", why)
        self.assertEqual(len(replay.variants), 2)

    def test_mm_and_addmm(self):
        def f(a, b, c):
            return torch.addmm(c, a, b) + torch.mm(a, b)

        b, c = self._x(K, N), self.b
        replay = self._replay(f, (self._x(4), b, c))
        self.assertEqual([r.op for r in replay.lowered.regions], ["addmm", "mm"])
        for m in (4, 8, 12, 32):
            self._serve(replay, f, (self._x(m), b, c), f"M={m}")

    def test_scalars_other_than_one_decline_by_name(self):
        # with beta / alpha other than 1 the host adds the bias with a copy kernel of its
        # own before the library call: not one closed call, so the trace declines
        def f(a, b, c):
            return torch.addmm(c, a, b, beta=0.5, alpha=2.0)

        # after the constructor's warm-up ran, the entry is constructed in the
        # declined state (E24: the warm-up was the call), warned by name
        with self.assertWarnsRegex(RuntimeWarning, "beta / alpha"):
            replay = self._replay(f, (self._x(4), self._x(K, N), self.b))
        self.assertEqual((len(replay.variants), len(replay.declines)), (0, 1))

    def test_batched_input_folds_to_2d(self):
        w = self.w
        replay = self._replay(
            linear_nobias, (torch.randn(2, 3, K, device="cuda", dtype=DTYPE), w)
        )
        self.assertEqual(len(replay.lowered.regions), 1)
        for s in (3, 5, 8):
            xx = torch.randn(2, s, K, device="cuda", dtype=DTYPE)
            self._serve(replay, linear_nobias, (xx, w), f"S={s}")

    def test_dtype_misses_and_transposed_weight_is_a_new_key(self):
        replay = self._replay(linear_nobias, (self._x(4), self.w))
        # another dtype fails the predicate's facts: a miss that builds the fp16 variant,
        # which serves from then on
        x = self._x(4)
        self.assertFalse(self._check(replay, linear_nobias, (x.half(), self.w.half())))
        self.assertTrue(self._check(replay, linear_nobias, (x.half(), self.w.half())))
        self.assertEqual(len(replay.variants), 2)
        # a weight stored transposed: other strides are another key (another cuBLAS
        # kernel), served by the same entry after one miss
        wt = self.w.t().contiguous().t()
        h0 = self.ht.gemm_harvests()
        self.assertEqual(self._serve(replay, linear_nobias, (self._x(4), wt)), 1)
        self.assertGreaterEqual(self.ht.gemm_harvests() - h0, 1)
        self._serve(replay, linear_nobias, (self._x(8), wt))
        self.assertTrue(self._check(replay, linear_nobias, (self._x(4), self.w)))

    def test_blas_setting_change_is_a_new_key(self):
        w = self.w.float()
        replay = self._replay(linear_nobias, (self._x(4).float(), w))
        old = torch.backends.cuda.matmul.fp32_precision
        epoch = torch._C._cuda_kernel_template_settings_epoch()
        try:
            torch.backends.cuda.matmul.fp32_precision = (
                "tf32" if old != "tf32" else "ieee"
            )
            # the flip moved the settings epoch: every site's kept selection is void
            self.assertGreater(torch._C._cuda_kernel_template_settings_epoch(), epoch)
            h0 = self.ht.gemm_harvests()
            args = (self._x(4).float(), w)
            served = self._check(replay, linear_nobias, args)
            self.assertFalse(served)  # the flipped setting is another key: one miss
            served = self._check(replay, linear_nobias, args)
            # on sm100 the tf32 kernels are cutlass3x with per-call TMA descriptors in
            # their parameters: refused by name, never stale
            if not served:
                self.assertTrue(
                    any(
                        "not rebindable" in reason for reason in replay.refused.values()
                    ),
                    replay.refused,
                )
            self.assertGreaterEqual(self.ht.gemm_harvests() - h0, 1)
            print(f"\n[fp32 after precision flip] served {served}")
        finally:
            torch.backends.cuda.matmul.fp32_precision = old
        self.assertTrue(self._check(replay, linear_nobias, (self._x(4).float(), w)))

    def test_a_setting_flip_between_two_calls_reselects(self):
        # the predicate's select keeps a site's last key under the settings epoch: a
        # flipped setting is a new epoch, so the next call re-selects (a new key: one
        # miss, then served), its result is a fresh entry's under that setting, and
        # the flip back finds the first variant with no miss
        replay = self._replay(linear, (self._x(4), self.w, self.b))
        args = (self._x(4), self.w, self.b)
        self.assertTrue(self._check(replay, linear, args))
        self.assertTrue(self._check(replay, linear, args))
        matmul = torch.backends.cuda.matmul
        flag = matmul.allow_bf16_reduced_precision_reduction
        matmul.allow_bf16_reduced_precision_reduction = not flag
        try:
            misses = replay.misses
            self.assertEqual(self._serve(replay, linear, args, "flipped"), 1)
            self.assertEqual(replay.misses, misses + 1)
            got = replay(*args)
            fresh = self._replay(linear, args)
            self.assertTrue(torch.equal(got, fresh(*args)))
            self.assertTrue(torch.equal(got, linear(*args)))
        finally:
            matmul.allow_bf16_reduced_precision_reduction = flag
        misses = replay.misses
        self.assertTrue(self._check(replay, linear, args, "flipped back"))
        self.assertEqual(replay.misses, misses)

    def test_replay_on_another_stream_is_refused(self):
        # the shared runtime binds an entry to the stream it was prepared on
        replay = self._replay(linear, (self._x(4), self.w, self.b))
        with torch.cuda.stream(torch.cuda.Stream()):
            with self.assertRaisesRegex(RuntimeError, "bound device and stream"):
                replay(self._x(4), self.w, self.b)
        self.assertTrue(self._check(replay, linear, (self._x(4), self.w, self.b)))

    def test_two_entries_interleaved(self):
        r4 = self._replay(linear, (self._x(4), self.w, self.b))
        r32 = self._replay(linear, (self._x(32), self.w, self.b))
        for m in (1, 8, 12, 16, 24, 32, 48, 64):
            self._serve(r4, linear, (self._x(m), self.w, self.b), f"r4 M={m}")
            self._serve(r32, linear, (self._x(m), self.w, self.b), f"r32 M={m}")
        for m in (1, 8, 12, 16, 24, 32, 48, 64):
            self.assertTrue(
                self._check(r4, linear, (self._x(m), self.w, self.b), f"r4 M={m}")
            )
            self.assertTrue(
                self._check(r32, linear, (self._x(m), self.w, self.b), f"r32 M={m}")
            )

    def test_replays_survive_host_state_churn(self):
        # some cuBLAS images carry host addresses of the harvesting call (stack and
        # heap); the replays must not depend on what is there now
        replay = self._replay(linear, (self._x(4), self.w, self.b))

        def churn(depth):
            junk = bytearray(4096)
            return churn(depth - 1) + 1 if depth else len(junk)

        for m in (4, 8, 12, 4, 16, 8):
            churn(200)
            garbage = [bytearray(1 << 16) for _ in range(64)]
            self._serve(replay, linear, (self._x(m), self.w, self.b), f"M={m}")
            del garbage

    def test_per_call_cost(self):
        replay = self._replay(linear, (self._x(4), self.w, self.b))
        args = (self._x(4), self.w, self.b)
        other = (self._x(16), self.w, self.b)
        for _ in range(3):
            replay(*args)
            replay(*other)  # M = 16 misses once (its key is new), then serves
        torch.cuda.synchronize()
        misses = replay.misses

        def timed(fn, n=200):
            xs = []
            for _ in range(n):
                t = time.perf_counter()
                fn()
                xs.append((time.perf_counter() - t) * 1e6)
            torch.cuda.synchronize()
            return statistics.median(xs)

        eager = timed(lambda: linear(*args))
        fixed = timed(lambda: replay(*args))
        moved = [(self._x(4), self.w, self.b) for _ in range(2)]
        i = iter(range(10000))
        rebind = timed(lambda: replay(*moved[next(i) % 2]))
        j = iter(range(10000))
        switch = timed(lambda: replay(*(args, other)[next(j) % 2]))
        print(
            f"\n[gemm per call CPU us] eager {eager:.1f} native same {fixed:.1f} "
            f"pointer rebind {rebind:.1f} M switch 4<->16 {switch:.1f}"
        )
        self.assertEqual(replay.misses, misses)

    def test_another_stream_is_refused_by_name_and_the_bound_stream_serves_again(
        self,
    ):
        # retirement stage B, O29: the shared runtime binds an entry to the stream it
        # was prepared on and refuses a call on another by name (a documented
        # refusal; the eager form of the stack's suites serves any current stream)
        from torch.testing._internal.host_trace_oracle import Oracle

        oracle = Oracle(linear, (self._x(4), self.w, self.b))
        self.addCleanup(oracle.close)
        self.assertIsNone(oracle.refused)
        oracle.check((self._x(4), self.w, self.b))
        with torch.cuda.stream(torch.cuda.Stream()):
            args = (self._x(4), self.w, self.b)
            with self.assertRaisesRegex(RuntimeError, "bound device and stream"):
                oracle.native(*args)
        oracle.check((self._x(4), self.w, self.b))

    def _edges(self, graph):
        """The edges of a raw graph as (from, to, type, from_port, to_port) over
        cudaGraphGetNodes order, read with their edge data."""
        from cuda.bindings import driver

        from torch.cuda._utils import _check_cuda_bindings_driver as check

        count = check(driver.cuGraphGetNodes(graph, 0))[-1]
        nodes = check(driver.cuGraphGetNodes(graph, count))[0]
        index = {int(node): i for i, node in enumerate(nodes)}
        count = check(driver.cuGraphGetEdges(graph, 0))[-1]
        frm, to, data, _ = check(driver.cuGraphGetEdges(graph, count))
        return sorted(
            (
                index[int(f)],
                index[int(t)],
                int(d.type),
                int(d.from_port),
                int(d.to_port),
            )
            for f, t, d in zip(frm, to, data)
        )

    def _plain_capture(self, fn, args):
        """The plain stream capture of fn at args on a side stream (warmed there), kept."""
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(stream):
            fn(*args)
            stream.synchronize()
            graph.capture_begin(capture_error_mode="thread_local")
            fn(*args)
            graph.capture_end()
        stream.synchronize()
        self.addCleanup(lambda g=graph: None)
        return graph

    def _assert_edges_as_the_plain_capture(self, fn, args, what):
        replay = self._replay(fn, args)
        self.assertTrue(self._check(replay, fn, args, what))
        graph, _ = replay.variants[0].lowered.capture_handles
        native = self._edges(graph)
        self.assertEqual(
            native, self._edges(self._plain_capture(fn, args).raw_cuda_graph()), what
        )
        programmatic = [e for e in native if e[2] != 0]
        self.assertTrue(all(e[2:] == (1, 1, 0) for e in programmatic), programmatic)
        stats = replay.region_stats()
        flagged = sum(sum(s["programmatic"]) for s in stats["sites"])
        print(
            f"\n[{what}] {len(programmatic)} programmatic of {len(native)} edges; sites {[(s['kinds'], s['programmatic']) for s in stats['sites']]}"
        )
        return replay, graph, native, programmatic, flagged

    def test_region_nodes_keep_the_librarys_programmatic_edges(self):
        # cuBLAS launches its kernels with programmatic stream serialization, so a
        # stream capture records a programmatic dependent launch edge (type 1, from the
        # programmatic port) into each. The harvest reads that per template node behind
        # its anchor kernel and the preparation launches the node the same way: the
        # prepared graph carries exactly the edges the plain capture of the function
        # carries, a variant swap within the exec leaves them, and every served call is
        # bitwise
        def block(x, w, b):
            return F.linear(F.layer_norm(x, (K,)), w, b)

        args = (self._x(8), self.w, self.b)
        replay, graph, native, programmatic, flagged = (
            self._assert_edges_as_the_plain_capture(
                block, args, "layer norm then linear"
            )
        )
        # every node the harvest flagged sits behind the kernel before it here, so the
        # flags are the edges
        self.assertEqual(len(programmatic), flagged)
        for m in (16, 8, 16):
            self._serve(replay, block, (self._x(m), self.w, self.b), f"M={m}")
        self.assertEqual(self._edges(graph), native)
        self._assert_execs_hold_their_chains(replay, "after the swaps")

    def test_the_lowered_tape_lists_its_programmatic_nodes(self):
        # what a sequence planner reads (the composition rule: a block freed at node i
        # is reusable by node j only past a full-completion edge): the tape seq of every
        # node in issue order and the indices of the nodes whose incoming edge is
        # programmatic, both from the prepared graph; the flagged nodes here are the
        # region's, behind the layer norm's kernel and each other
        def block(x, w, b):
            return F.linear(F.layer_norm(x, (K,)), w, b)

        replay, _, native, programmatic, flagged = (
            self._assert_edges_as_the_plain_capture(
                block, (self._x(8), self.w, self.b), "listed"
            )
        )
        lowered = replay.variants[0].lowered
        self.assertEqual(
            lowered.programmatic_nodes, tuple(sorted(e[1] for e in programmatic))
        )
        self.assertEqual(len(lowered.programmatic_nodes), flagged)
        self.assertEqual(len(lowered.node_seqs), len(native) + 1)
        self.assertEqual(list(lowered.node_seqs), sorted(lowered.node_seqs))
        (region,) = lowered.regions
        self.assertEqual(
            {lowered.node_seqs[i] for i in lowered.programmatic_nodes}, {region.seq}
        )

    def test_region_behind_a_memset_records_the_edge_the_plain_capture_records(self):
        # the region's first node behind a memset node: the preparation launches it as
        # the library does, and the driver records for it whatever it records for the
        # library's own launch behind the same memset
        def zeros_then_linear(x, w, b):
            return F.linear(torch.zeros_like(x), w, b)

        self._assert_edges_as_the_plain_capture(
            zeros_then_linear, (self._x(8), self.w, self.b), "memset then linear"
        )

    def test_a_template_without_programmatic_launches_is_another_class(self):
        # at an 8-byte storage offset cuBLAS runs this GEMM as a legacy cutlass kernel
        # launched without programmatic stream serialization, aligned as nvjet with it:
        # the same [kernel, kernel] chain, other edges. A variant prepared at the aligned
        # x holds programmatic edges into its region nodes; the offset template's kernels
        # would run behind them without waiting, so the flags are part of the class and
        # the offset x is a topology miss served by its own exec (either order)
        def offset_x(m):
            big = self._x(m, K + 4)
            return big.view(-1)[4 : 4 + m * K].view(m, K)

        for first, second in ((self._x(8), offset_x(8)), (offset_x(8), self._x(8))):
            replay = self._replay(linear, (first, self.w, self.b))
            self.assertTrue(self._check(replay, linear, (first, self.w, self.b)))
            self.assertEqual(self._serve(replay, linear, (second, self.w, self.b)), 1)
            self.assertEqual(len(replay.variants), 2)
            self.assertIn("programmatic", " ".join(str(row) for row in replay.miss_log))
            sites = replay.region_stats()["sites"]
            self.assertEqual([s["kinds"] for s in sites], [["kernel", "kernel"]] * 2)
            self.assertEqual(
                sorted(tuple(s["programmatic"]) for s in sites),
                [(False, False), (True, True)],
            )
            self._assert_execs_hold_their_chains(replay, "two classes")
            for x in (first, second, first):
                self.assertTrue(self._check(replay, linear, (x, self.w, self.b)))

    def test_a_region_first_in_the_graph_has_no_incoming_edge(self):
        # the harvest flags the first node (behind its anchor); the prepared graph has no
        # node before it, so no edge, as the plain capture
        replay, _, native, programmatic, flagged = (
            self._assert_edges_as_the_plain_capture(
                linear, (self._x(8), self.w, self.b), "linear alone"
            )
        )
        self.assertGreaterEqual(flagged, len(programmatic))
        self.assertEqual(
            len(native),
            len(replay.variants[0].lowered.region_nodes[replay.lowered.regions[0].site])
            - 1,
        )

    def test_out_forms_the_host_takes_as_is(self):
        # Inductor's extern GEMMs write into buffers of its own: a row-major buffer
        # padded beyond N (the host passes its leading dimension) and a buffer that is a
        # view of an input (a donated buffer the backward reuses); eager's cuBLAS host
        # takes both as they are, so the region does, with the out's strides in its key
        # and the input among the tape's written roots
        wt = self.w.t()

        def padded(x, w):
            buf = torch.empty(x.shape[0], N + 64, device="cuda", dtype=DTYPE)
            out = buf[:, :N]
            torch.mm(x, w, out=out)
            return out

        replay = self._replay(padded, (self._x(4), wt))
        (region,) = replay.lowered.regions
        self.assertEqual(region.op, "mm")
        self.assertEqual(tuple(int(v) for v in region.metas[-1][2]), (N + 64, 1))
        for m in (4, 8, 12, 32):
            self._serve(replay, padded, (self._x(m), wt), f"padded M={m}")

        def into_input(x, w, acc):
            torch.mm(x, w, out=acc[1])
            return acc

        def bmm_into_input(x, w, acc):
            torch.bmm(x, w, out=acc)
            return acc

        w3 = torch.randn(2, K, 256, device="cuda", dtype=DTYPE) / K**0.5
        for fn, make in (
            (
                into_input,
                lambda m: (
                    self._x(m),
                    wt,
                    torch.zeros(2, m, N, device="cuda", dtype=DTYPE),
                ),
            ),
            (
                bmm_into_input,
                lambda m: (
                    torch.randn(2, m, K, device="cuda", dtype=DTYPE),
                    w3,
                    torch.zeros(2, m, 256, device="cuda", dtype=DTYPE),
                ),
            ),
        ):
            replay = self._replay(fn, make(8))
            self.assertEqual(replay.tape.written_inputs, (2,))
            (region,) = replay.lowered.regions
            self.assertIsInstance(region.sources[-1].root, self.module.InputSource)
            for m in (8, 4, 16):
                x, w, acc = args = make(m)
                want = torch.bmm(x, w) if fn is bmm_into_input else torch.mm(x, w)
                got = replay(*args)
                self.assertEqual(got.data_ptr(), acc.data_ptr(), f"{fn.__name__} M={m}")
                self.assertTrue(
                    torch.equal(acc if fn is bmm_into_input else acc[1], want),
                    f"{fn.__name__} M={m}",
                )
                if fn is into_input:
                    self.assertEqual(acc[0].abs().max().item(), 0)

    def test_out_forms_the_host_would_copy_decline_by_name(self):
        # a result cuBLAS cannot take (no unit stride) eager computes into a copy and
        # copies back; a 1-D bias into a non-contiguous result eager copies in with a
        # kernel of its own before the GEMM: neither is one closed call
        def strided(x, w):
            buf = torch.empty(2 * x.shape[0], 2 * N, device="cuda", dtype=DTYPE)
            out = buf[::2, ::2]
            torch.mm(x, w, out=out)
            return out

        with self.assertWarnsRegex(RuntimeWarning, "compute into a copy"):
            replay = self._replay(strided, (self._x(4), self.w.t()))
        self.assertEqual((len(replay.variants), len(replay.declines)), (0, 1))

        def bias_into_padded(x, w, b):
            buf = torch.empty(x.shape[0], N + 64, device="cuda", dtype=DTYPE)
            out = buf[:, :N]
            torch.addmm(b, x, w, out=out)
            return out

        with self.assertWarnsRegex(RuntimeWarning, "copies the bias"):
            replay = self._replay(bias_into_padded, (self._x(4), self.w.t(), self.b))
        self.assertEqual((len(replay.variants), len(replay.declines)), (0, 1))

    def test_a_template_launched_with_a_cluster_of_one_keeps_the_attribute(self):
        # cuBLAS launches its sm100 nvjet "1x1" kernels (the HF attention's bmm shapes in
        # bf16) with an explicit cluster dimension of (1, 1, 1); the harvest's census keeps
        # it (0 is a node launched without one), and the transplanted node must carry it:
        # without the attribute the kernel raises Warp Illegal Instruction
        from cuda.bindings import driver, runtime

        def f(a, b):
            return torch.bmm(a, b)

        a = torch.randn(48, 128, 64, device="cuda", dtype=DTYPE)
        b = torch.randn(48, 64, 128, device="cuda", dtype=DTYPE)
        replay = self._replay(f, (a, b))
        kernels = [
            n
            for t in self.ht._gemm_templates.values()
            for n in t.nodes
            if n["kind"] == "kernel"
        ]
        clusters = {tuple(n["attrs"][:3]) for n in kernels}
        if (1, 1, 1) not in clusters:
            self.skipTest(
                f"cuBLAS selected no kernel launched with a cluster of one here: {clusters}"
            )
        for _ in range(2):
            self.assertTrue(self._check(replay, f, (a, b), "cluster of one"))
        (variant,) = replay.variants
        graph, _ = variant.lowered.capture_handles
        count = _check_cuda_bindings(runtime.cudaGraphGetNodes(graph, 0))[1]
        nodes = _check_cuda_bindings(runtime.cudaGraphGetNodes(graph, count))[0]
        dims = []
        for node in nodes:
            if (
                _check_cuda_bindings(runtime.cudaGraphNodeGetType(node))
                != runtime.cudaGraphNodeType.cudaGraphNodeTypeKernel
            ):
                continue
            value = _check_cuda_bindings(
                driver.cuGraphKernelNodeGetAttribute(
                    int(node),
                    driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
                )
            )
            dims.append((value.clusterDim.x, value.clusterDim.y, value.clusterDim.z))
        self.assertIn((1, 1, 1), dims, dims)

    def test_an_operand_the_host_clones_is_the_tapes_copy_then_the_region(self):
        # an operand cuBLAS cannot take as is (TinyLlama under Inductor's math attention
        # hands bmm a (128, 1, 64) operand with a zero stride on its size-1 dimension; a
        # 2-D operand with no unit stride) eager's host clones contiguous before the
        # library call: the tape records eager's copy as a launch and the region reads the
        # clone, an allocation of the trace
        base = torch.randn(128, 64, device="cuda", dtype=DTYPE)
        b3 = torch.randn(128, 64, 128, device="cuda", dtype=DTYPE)

        def zero_stride(base, b):
            return torch.bmm(base.as_strided((128, 1, 64), (64, 0, 1)), b)

        def no_unit_stride(x, w):
            return torch.mm(x[:, ::2], w)

        for fn, args, what in (
            (zero_stride, (base, b3), "bmm mat1 (128, 1, 64) strides (64, 0, 1)"),
            (
                no_unit_stride,
                (self._x(8, 2 * K), self.w.t()),
                "mm mat1 strides (2K, 2)",
            ),
        ):
            replay = self._replay(fn, args)
            tape = replay.tape
            # eager's clone of a source that is contiguous in memory is one cudaMemcpyAsync
            # (Copy.cu copy_device_to_device), of any other a copy kernel
            self.assertEqual(tape.num_regions, 1, what)
            self.assertEqual(
                (tape.num_launches, len(tape.memcpys)) in ((1, 0), (0, 1)),
                True,
                f"{what}: {tape.num_launches} launches, {len(tape.memcpys)} memcpys",
            )
            (region,) = replay.lowered.regions
            # the region's first operand is the clone, an allocation of the trace (planned
            # into the arena: a source past the call's two inputs), not the input
            self.assertNotIn(
                getattr(region.sources[0].root, "index", None), (0, 1), what
            )
            self.assertTrue(self._check(replay, fn, args, what))
            self.assertTrue(self._check(replay, fn, args, what))


if __name__ == "__main__":
    run_tests()
