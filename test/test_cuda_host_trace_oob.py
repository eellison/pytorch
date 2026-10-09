# Owner(s): ["module: cuda graphs"]

import unittest
from unittest import mock

import torch
from torch.testing._internal.common_cuda import tf32_off
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUDA,
    TestCase,
)


if TEST_CUDA:
    import torch.cuda._host_trace_harvest as harvest
    import torch.cuda._host_trace_replay as replay
    from torch.cuda._host_trace_harvest import HarvestProvider
    from torch.cuda._host_trace_replay import HostTraceReplay


def fn(x, w):
    y = x @ w
    # a traced launch: a trace of only eager steps runs the call eagerly
    return y, y * w


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
@instantiate_parametrized_tests
class TestHostTraceOobLearn(TestCase):
    def test_a_drawing_op_learns_on_inputs_in_its_range(self):
        # a drawing op's out-of-band learn fills its probabilities in [0, 1]:
        # bernoulli on uniform(-1, 1) is a device-side assert
        f = HostTraceReplay(lambda p: torch.bernoulli(p), opaque=(HarvestProvider(families=("rng",)),))
        with mock.patch.object(replay, "oob_learn", True):
            for _ in range(3):
                p = torch.ones(63, 26, device="cuda", dtype=torch.bfloat16)
                self.assertEqual(f(p), torch.ones_like(p), atol=0, rtol=0)
        torch.cuda.synchronize()
        self.assertEqual((f.traces, f.oob_binds), (1, 1))

    @parametrize("oob", [False, True])
    def test_trace_binds_its_keys(self, oob):
        # with oob_learn a trace's opaque calls bind at keys learned out of
        # band, so no variant learns: no learner run, relower or second trace
        p = HarvestProvider()
        f = HostTraceReplay(fn, opaque=(p,))
        with tf32_off(), mock.patch.object(replay, "oob_learn", oob):
            for n in (128, 128, 256, 256, 192, 128):
                a, b = (torch.randn(n, n, device="cuda") for _ in range(2))
                self.assertEqual(f(a, b), fn(a, b), atol=0, rtol=0)
        if oob:
            self.assertEqual((f.traces, f.relowers, f.oob_binds), (1, 0, 1))
            self.assertFalse(any(v.learns for v in f.variants))
        else:
            self.assertEqual((f.traces, f.relowers, f.oob_binds), (1, 1, 0))

    def test_refused_key_stays_eager(self):
        # a key the provider refuses does not bind out of band either: its
        # call stays an eager step
        harvest_ = HarvestProvider._harvest

        def refusing(self, key, *args, **kwargs):
            if key.sizes[0] == (192, 192):
                raise harvest._Refused("refused by the test")
            return harvest_(self, key, *args, **kwargs)

        p = HarvestProvider()
        f = HostTraceReplay(fn, opaque=(p,))
        with tf32_off(), mock.patch.object(HarvestProvider, "_harvest", refusing), mock.patch.object(replay, "oob_learn", True):
            for n in (192, 192, 128, 128, 192):
                a, b = (torch.randn(n, n, device="cuda") for _ in range(2))
                self.assertEqual(f(a, b), fn(a, b), atol=0, rtol=0)
        self.assertEqual(len(p.refused), 1)
        self.assertEqual(f.oob_binds, 0)

    def test_a_learn_lays_aliased_operands_out_as_at_the_call(self):
        # blas keys do not key their operands' aliasing, but a learn lays them
        # out on one buffer: at the call's distance (B 1920 bytes up from A
        # at the third call), never the trace's (20), which contradicts B's
        # alignment (a kernel for 16-byte aligned B runs on B at +20)
        def mm(a):
            return a @ a.split(a.shape[0] // 2 + 1)[-1].t()

        learn, keys = HostTraceReplay._learn, []

        def checked(self_, op, provider, call, key, *args, **kwargs):
            keys.append(key)
            self.assertTrue(all((key.align[lead] + d) % 256 == key.align[i] for i, lead, d in key.alias), key)
            return learn(self_, op, provider, call, key, *args, **kwargs)

        f = HostTraceReplay(mm)
        with mock.patch.object(HostTraceReplay, "_learn", checked), mock.patch.object(replay, "oob_learn", True):
            for a in (torch.randn(18, 96), torch.randn(96, 18).t(), torch.randn(18, 96)):
                a = a.to(device="cuda", dtype=torch.float16)
                self.assertEqual(f(a), mm(a), atol=0, rtol=0)
        self.assertEqual([k.alias for k in keys], [((1, 0, 20),), ((1, 0, 1920),)])
        self.assertEqual((f.traces, f.replays), (1, 1))

    def test_a_refused_aliased_key_twin_guards_the_calls_key(self):
        # the GEMM of two views of one storage refused at a call: its twin (_eager_sites) guards
        # the call's key, B 268 bytes up from A, not the trace's 768, which the call's tape fails
        def fn(a):
            n = a.shape[0]
            v = torch.split(a, [n - n // 3, n // 3])[1]
            return (v @ a.t()) + 1

        harvest_ = HarvestProvider._harvest

        def refusing(self, key, *args, **kwargs):
            if key.sizes[0] == (66, 24):
                raise harvest._Refused("refused by the test")
            return harvest_(self, key, *args, **kwargs)

        p = HarvestProvider()
        f = HostTraceReplay(fn, opaque=(p,))
        with mock.patch.object(HarvestProvider, "_harvest", refusing), mock.patch.object(replay, "oob_learn", True):
            for a in (torch.randn(96, 13), torch.randn(12, 48), torch.randn(24, 200).t(), torch.randn(24, 200).t()):
                a = a.to(device="cuda", dtype=torch.float16)
                self.assertEqual(f(a), fn(a), atol=0, rtol=0)
        self.assertEqual(list(p.refused.values()), ["refused by the test"])
        self.assertEqual((f.traces, f.eager_sites, f.eager_sites_refusals), (1, 1, {}))


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
class TestHostTraceStaleBinding(TestCase):
    def test_a_gemm_on_an_eager_size_1_output_binds_at_eagers_stride(self):
        # sort is an eager step whose fake kernel predicts a contiguous output; eager's keeps the
        # input's stride 27 on the size-1 dim. The trace bound the GEMM at the key with the fake's
        # stride (24, learned by g), then the warm-up gave the output eager's stride: the site's row
        # key was another than its binding's, a _StaleBinding raised to the caller. It binds after the
        # trace, at the key with eager's stride
        def fn(a):
            v = a.sort(1).values
            return v.t() @ v, a * 2

        p = HarvestProvider()
        g, f = HostTraceReplay(fn, opaque=(p,)), HostTraceReplay(fn, opaque=(p,))
        x = torch.randn(1, 24, device="cuda", dtype=torch.bfloat16)
        for _ in range(2):
            self.assertEqual(g(x), fn(x), atol=0, rtol=0)
        y = torch.randn(1, 27, device="cuda", dtype=torch.bfloat16)[:, :24]
        for _ in range(3):
            self.assertEqual(f(y), fn(y), atol=0, rtol=0)
        self.assertEqual((f.traces, f.replays, f.eager, f.learned, p.harvests), (1, 1, 1, 1, 2))
        ((site,),) = [v.captured.lowered.sites for v in f.variants]
        self.assertEqual(site.site.key.strides[1], (27, 1))

    def test_a_gemm_on_an_eager_size_1_output_keys_eagers_stride_at_each_call(self):
        # eager's sort keeps its input's stride on the size-1 dim (27, then 30), its fake kernel
        # predicts a contiguous output. The warm-up check gives the output eager's rule there, the
        # input's stride where the size is 1, not the warm-up's 27: a call at stride 30 keys the GEMM
        # at 30, as eager's call does (with 27 it would bind the stride-27 key's kernels, silently)
        def fn(a):
            v = a.sort(1).values
            return v.t() @ v, a * 2

        p = HarvestProvider()
        f = HostTraceReplay(fn, opaque=(p,))
        for width in (27, 27, 30, 30):
            y = torch.randn(1, width, device="cuda", dtype=torch.bfloat16)[:, :24]
            self.assertEqual(f(y), fn(y), atol=0, rtol=0)
        self.assertEqual(f.traces, 1)
        self.assertEqual({k[4][1] for k in p.bindings}, {(27, 1), (30, 1)})


def setUpModule():
    from torch.cuda import _host_trace_hint_audit

    _host_trace_hint_audit.enable_for_tests()


if __name__ == "__main__":
    run_tests()
