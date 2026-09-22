# Owner(s): ["module: cuda"]

import json
import os
import threading
import unittest
from unittest import mock

from host_trace_testing import (
    build,
    capture_graph,
    graph_functions,
    graph_nodes,
    HostTraceTestCase,
    replay_backend,
)

import torch
import torch.nn.functional as F
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_CUDNN_ATTENTION
from torch.testing._internal.common_utils import run_tests, skipIfRocm


if torch.cuda.is_available():
    import host_trace_two_hint as two_hint

    from torch.cuda import _host_trace as ht

DTYPE = torch.bfloat16
CUDNN = int(torch.nn.attention.SDPBackend.CUDNN_ATTENTION.value)
H, DH = 8, 64


def sdpa(q, k, v, mask):
    return F.scaled_dot_product_attention(q, k, v, mask)


def sdpa_causal(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def sdpa_gqa(q, k, v, mask):
    return F.scaled_dot_product_attention(q, k, v, mask, enable_gqa=True)


def sdpa_scaled(q, k, v, mask):
    return F.scaled_dot_product_attention(q, k, v, mask, scale=0.5)


def cudnn_op(q, k, v, bias):
    # the op itself, outside the selector
    return torch.ops.aten._scaled_dot_product_cudnn_attention(
        q, k, v, bias, False, 0.0, False, False
    )[0]


def sdpa_grads(q, k, v, mask, g):
    # torch.autograd.grad on the main thread: the forward region, then the
    # engine runs the backward node on its device worker thread, where the
    # trace mode brings the recorder along
    leaves = [t.detach().requires_grad_(True) for t in (q, k, v)]
    out = F.scaled_dot_product_attention(*leaves, mask)
    return torch.autograd.grad(out, leaves, g)


def _choice(q, k, v, mask, causal=False, gqa=False):
    return torch._fused_sdp_choice(
        q, k, v, mask, 0.0, causal, scale=None, enable_gqa=gqa
    )


def _padding_mask(B, Lq, Lk, dtype=DTYPE):
    # a left-padded batch's additive mask (B, 1, Lq, Lk): row b hides its
    # first b keys, never the last
    m = torch.zeros(B, 1, Lq, Lk, device="cuda", dtype=dtype)
    for b in range(B):
        m[b, :, :, : min(b, Lk - 1)] = float("-inf")
    return m


def _qkv(B, Lq, Lk, dh=DH, h=H, hkv=None, dtype=DTYPE):
    q = torch.randn(B, h, Lq, dh, device="cuda", dtype=dtype)
    k = torch.randn(B, hkv or h, Lk, dh, device="cuda", dtype=dtype)
    v = torch.randn(B, hkv or h, Lk, dh, device="cuda", dtype=dtype)
    return q, k, v


def _align_class(address):
    return min(address & -address, 256) if address else 256


def _harvest(tape, args, k=0):
    """The harvested template of region k of `tape` at `args`: the key and spec
    as a replay computes them (every operand's dtype, sizes, strides and
    alignment class; an output is an allocation: 256; the op, its scalars, the
    device class and the BLAS settings)."""
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
        # an allocation of the tape (an output, or the backward's saved
        # forward outputs) is 256-byte aligned by the allocator
        aligns.append(
            256 if o.root.allocation else _align_class(int(ev.ev(o.address, env)))
        )
    spec = (r.op, r.scalars, tuple(metas), tuple(aligns))
    key = (device, tape.device_identity, *spec, ht._blas_settings())
    return ht._template(key, spec, device)


def _cudnn_templates():
    return [t for t in ht.gemm_templates() if str(t["key"]).count("cudnn_sdpa")]


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN attention not supported")
@skipIfRocm(msg="closed regions are CUDA-only in this version")
class TestCudaHostTraceCudnnAttention(HostTraceTestCase):
    """aten._scaled_dot_product_cudnn_attention as a closed region: eager's own
    selection on a GPU that prefers cuDNN (sm90 and up), no pin."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        q, k, v = _qkv(4, 1, 32)
        if _choice(q, k, v, _padding_mask(4, 1, 32)) != CUDNN:
            self.skipTest("eager's SDPA selector does not pick cuDNN attention here")
        self.last_miss = ""
        # the variants built from one tape, by the first one's identity
        self._pool: dict[int, list] = {}

    def _check(self, variant, fn, args, *, stride=False, outputs=1):
        # an entry's rule over one tape's variants: the first that serves the
        # call does; a TopologyMiss names the tape, which built at these
        # inputs (no trace) is a variant with the call's node chain
        want = fn(*args)
        want = want if isinstance(want, (tuple, list)) else (want,)
        torch.cuda.synchronize()
        variants = self._pool.setdefault(id(variant), [variant])
        tape = None
        for v in variants:
            try:
                got = v.replay(args)
                break
            except ht.TopologyMiss as e:
                tape, self.last_miss = e.tape, str(e)
            except ht.Miss as e:
                self.last_miss = str(e)
        else:
            if tape is None:
                return False
            v = build(tape, fn, args)
            variants.append(v)
            got = v.replay(args)
        torch.cuda.synchronize()
        for g, w in zip(got[:outputs], want[:outputs]):
            self._assert_bitwise(g, w, "outputs differ bitwise", stride=stride)
        return True

    def _trace(self, fn, args):
        tape = ht.trace(fn, args)
        return tape, build(tape, fn, args)

    def test_a_masked_decode_call_is_one_region_under_eagers_selection(self):
        args = (*_qkv(4, 1, 32), _padding_mask(4, 1, 32))
        self.assertEqual(_choice(*args), CUDNN)
        tape, variant = self._trace(sdpa, args)
        self.assertEqual(tape.num_regions, 1)
        self.assertEqual(tape.num_launches, 0)
        self.assertEqual(len(tape.memsets), 0)
        r = json.loads(tape.to_json())["regions"][0]
        self.assertEqual(r["op"], "cudnn_sdpa")
        self.assertEqual(
            [i["name"] for i in r["inputs"]], ["query", "key", "value", "attn_bias"]
        )
        self.assertEqual([o["name"] for o in r["outputs"]], ["output"])
        # the batch and the key length are symbols of the tape
        self.assertIsInstance(r["inputs"][0]["sizes"][0], str)
        self.assertIsInstance(r["inputs"][1]["sizes"][2], str)
        self.assertEqual(tape.regions[0].scalars, (False, False, None, True))
        self.assertTrue(self._check(variant, sdpa, args))
        # other padded lengths and batches: the region re-keys and harvests,
        # one tape, no re-trace, one variant (the forward is one kernel at
        # every key: no TopologyMiss)
        cases = [(4, 48), (4, 64), (4, 16), (2, 32), (8, 32), (6, 40), (4, 17), (3, 25)]
        for B, Lk in cases:
            other = (*_qkv(B, 1, Lk), _padding_mask(B, 1, Lk))
            self.assertTrue(self._check(variant, sdpa, other), (B, Lk, self.last_miss))
        self.assertEqual(len(self._pool[id(variant)]), 1)
        if replay_backend() == "native":
            # the process-wide template cache holds a cuDNN template per key
            # served (the family rerun finds them there already)
            self.assertGreaterEqual(len(_cudnn_templates()), len(cases))

    def test_a_square_prefill_mask(self):
        args = (*_qkv(4, 32, 32), _padding_mask(4, 32, 32))
        self.assertEqual(_choice(*args), CUDNN)
        tape, variant = self._trace(sdpa, args)
        self.assertEqual((tape.num_regions, tape.num_launches), (1, 0))
        for B, L in ((4, 64), (4, 48), (2, 16), (4, 33)):
            other = (*_qkv(B, L, L), _padding_mask(B, L, L))
            self.assertTrue(self._check(variant, sdpa, other), (B, L, self.last_miss))

    def test_causal_without_a_mask(self):
        args = _qkv(4, 32, 32)
        self.assertEqual(_choice(*args, None, causal=True), CUDNN)
        tape, variant = self._trace(sdpa_causal, args)
        self.assertEqual((tape.num_regions, tape.num_launches), (1, 0))
        self.assertEqual(
            [i.name for i in tape.regions[0].inputs], ["query", "key", "value"]
        )
        self.assertEqual(tape.regions[0].scalars, (False, True, None, False))
        for B, L in ((4, 64), (2, 48)):
            self.assertTrue(
                self._check(variant, sdpa_causal, _qkv(B, L, L)), (B, L, self.last_miss)
            )

    def test_head_dim_128_fp16_and_gqa(self):
        args = (*_qkv(4, 1, 32, dh=128), _padding_mask(4, 1, 32))
        self.assertEqual(_choice(*args), CUDNN)
        tape, variant = self._trace(sdpa, args)
        self.assertEqual((tape.num_regions, tape.num_launches), (1, 0))
        self.assertTrue(
            self._check(
                variant, sdpa, (*_qkv(2, 1, 48, dh=128), _padding_mask(2, 1, 48))
            )
        )
        args = (
            *_qkv(4, 1, 32, dtype=torch.float16),
            _padding_mask(4, 1, 32, torch.float16),
        )
        self.assertEqual(_choice(*args), CUDNN)
        tape, variant = self._trace(sdpa, args)
        self.assertEqual((tape.num_regions, tape.num_launches), (1, 0))
        self.assertTrue(
            self._check(
                variant,
                sdpa,
                (
                    *_qkv(4, 1, 40, dtype=torch.float16),
                    _padding_mask(4, 1, 40, torch.float16),
                ),
            )
        )
        args = (*_qkv(4, 1, 32, hkv=2), _padding_mask(4, 1, 32))
        self.assertEqual(_choice(*args, gqa=True), CUDNN)
        tape, variant = self._trace(sdpa_gqa, args)
        self.assertEqual((tape.num_regions, tape.num_launches), (1, 0))
        self.assertTrue(
            self._check(
                variant, sdpa_gqa, (*_qkv(4, 1, 48, hkv=2), _padding_mask(4, 1, 48))
            )
        )

    def test_the_output_follows_the_querys_layout(self):
        # a projection's (B, S, H, D) view transposed: the output keeps q's
        # stride order (alloc_with_matching_layout), which the tape allocates
        # the same way; the mask of a later step (a sliced row of a square
        # mask) has another alignment class, a key term
        B, L = 4, 18
        x = torch.randn(B, 1, H, DH, device="cuda", dtype=DTYPE)
        q = x.transpose(1, 2)
        k = torch.randn(B, L, H, DH, device="cuda", dtype=DTYPE).transpose(1, 2)
        v = torch.randn(B, L, H, DH, device="cuda", dtype=DTYPE).transpose(1, 2)
        square = _padding_mask(B, L, L)
        args = (q, k, v, square[:, :, -1:, :])
        self.assertEqual(_choice(*args), CUDNN)
        self.assertNotEqual(args[3].data_ptr() % 16, 0)
        tape, variant = self._trace(sdpa, args)
        self.assertTrue(self._check(variant, sdpa, args, stride=True))
        self.assertEqual(
            tape.regions[0].out.strides[2], tape.regions[0].inputs[0].strides[2]
        )

    def test_the_template_holds_eagers_kernel_and_the_operands_slots(self):
        args = (*_qkv(4, 1, 32), _padding_mask(4, 1, 32))
        tape = ht.trace(sdpa, args)
        t = _harvest(tape, args)
        self.assertEqual(t.kinds, ("kernel",))
        node = t.nodes[0]
        self.assertIn("sdpa", node["name"])
        # the five pointer slots: query, key, value, the bias, the output; no
        # workspace, the scale a constant of the image; each of the five
        # 200-byte tensor descriptors of the parameter struct carries
        # addresses into the harvesting thread's stack (the frontend's host
        # scaffolding, dead at kernel time): host slots by mapping class,
        # the template keeps its bytes as it keeps cuBLAS's
        self.assertEqual(sorted(s[1] for s in node["slots"]), [0, 1, 2, 3, 4])
        self.assertEqual(node["ws_slots"], [])
        self.assertEqual(node["scratch_slots"], [])
        self.assertGreaterEqual(len(node["host_slots"]), 1)
        self.assertLessEqual(
            {cls for _off, cls in node["host_slots"]}, {"stack", "dead", "padding"}
        )
        self.assertEqual(t.layout, ("scratch", "scratch", "out"))
        self.assertEqual(t.scratch, [8, 8])
        self.assertFalse(t.uses_ws)
        # E36: eager's own function object on this thread (the harvest's, whose
        # plan cache the template's handle belongs to; the cache is thread
        # local and leaked, so the handle outlives the thread). Another
        # thread's plan holds the same kernel by name and launch shape under
        # another function handle (its own load of the kernel)
        eager_graph = capture_graph(lambda: sdpa(*args))
        eager = graph_functions(eager_graph)
        self.assertEqual(len(eager), 1)
        self.assertIn(node["func"], eager[0])
        other: list = []

        def on_thread():
            g = capture_graph(lambda: sdpa(*args))
            other.append((graph_functions(g), graph_nodes(g)[0]))

        th = threading.Thread(target=on_thread)
        th.start()
        th.join()
        mine = graph_nodes(eager_graph)[0]
        self.assertEqual([k[:4] for k in other[0][1]], [k[:4] for k in mine])
        self.assertEqual(len(other[0][1][0][4]), len(mine[0][4]))
        self.assertEqual(len(other[0][0]), 1)

    def test_the_scale_and_the_flags_are_key_terms(self):
        args = (*_qkv(4, 1, 32), _padding_mask(4, 1, 32))
        tape = ht.trace(sdpa, args)
        tape2 = ht.trace(sdpa_scaled, args)
        self.assertEqual(tape2.regions[0].scalars, (False, False, 0.5, True))
        t1, t2 = _harvest(tape, args), _harvest(tape2, args)
        self.assertIsNot(t1, t2)
        self.assertNotEqual(t1.key, t2.key)
        variant = build(tape2, sdpa_scaled, args)
        self.assertTrue(
            self._check(
                variant, sdpa_scaled, (*_qkv(2, 1, 48), _padding_mask(2, 1, 48))
            )
        )

    def test_forward_and_backward_in_one_trace(self):
        def inputs(B, L):
            return (
                *_qkv(B, L, L),
                _padding_mask(B, L, L),
                torch.randn(B, H, L, DH, device="cuda", dtype=DTYPE),
            )

        args = inputs(4, 32)
        tape = ht.trace(sdpa_grads, args)
        self.assertEqual(
            [r.op for r in tape.regions], ["cudnn_sdpa", "cudnn_sdpa_backward"]
        )
        self.assertEqual(tape.regions[0].scalars, (True, False, None, True))
        self.assertEqual(
            [o.name for o in tape.regions[0].outputs], ["output", "logsumexp"]
        )
        self.assertEqual(
            [i.name for i in tape.regions[1].inputs],
            ["grad_out", "query", "key", "value", "out", "logsumexp", "attn_bias"],
        )
        self.assertEqual(tape.num_launches, 0)
        variant = build(tape, sdpa_grads, args)
        self.assertTrue(self._check(variant, sdpa_grads, args, outputs=3))
        for B, L in ((4, 64), (2, 48), (4, 40)):
            self.assertTrue(
                self._check(variant, sdpa_grads, inputs(B, L), outputs=3),
                (B, L, self.last_miss),
            )
        # the backward's template: a memset into the workspace, three kernels,
        # dq / dk / dv the returns, the workspace the scratch
        t = _harvest(tape, args, 1)
        self.assertEqual(t.kinds, ("memset", "kernel", "kernel", "kernel"))
        self.assertEqual(t.layout, ("out", "out", "out", "scratch"))
        self.assertEqual(t.nodes[0]["dst_role"][0], "scratch")

    def test_dropout_and_the_ragged_env_decline_by_name(self):
        args = (*_qkv(4, 32, 32), _padding_mask(4, 32, 32))

        def dropped(q, k, v, mask):
            return F.scaled_dot_product_attention(q, k, v, mask, dropout_p=0.5)

        with self.assertRaisesRegex(ht.Declined, "dropout"):
            ht.trace(dropped, args)
        with mock.patch.dict(os.environ, {"TORCH_CUDNN_SDPA_AVOID_RECOMPILE": "1"}):
            with self.assertRaisesRegex(ht.Declined, "AVOID_RECOMPILE"):
                ht.trace(sdpa, args, warm_up=False)

    def test_misaligned_operands_decline_by_name(self):
        # cuDNN faults (a sticky misaligned-address error) on a q / k / v base
        # below 16 bytes and on a bias below 4: the trace declines naming it,
        # without a warm-up (eager would fault)
        B, Lk = 4, 32
        q, k, v = _qkv(B, 1, Lk)
        mask = _padding_mask(B, 1, Lk)
        buf = torch.randn(B * H * DH + 8, device="cuda", dtype=DTYPE)
        q_mis = buf[4 : 4 + B * H * DH].view(B, H, 1, DH)
        self.assertEqual(q_mis.data_ptr() % 16, 8)
        with self.assertRaisesRegex(ht.Declined, "query's base is not 16-byte aligned"):
            ht.trace(sdpa, (q_mis, k, v, mask), warm_up=False)
        mbuf = torch.zeros(B * Lk + 1, device="cuda", dtype=DTYPE)
        m_mis = mbuf[1:].view(B, 1, 1, Lk)
        self.assertEqual(m_mis.data_ptr() % 4, 2)
        with self.assertRaisesRegex(
            ht.Declined, "attn_bias's base is not 4-byte aligned"
        ):
            ht.trace(sdpa, (q, k, v, m_mis), warm_up=False)

    def test_a_bias_eager_cannot_expand_raises_eagers_error(self):
        # the op with a 3-D bias (the selector never sends one to cuDNN): the
        # host's expand raises, and the region's expand raises the same
        q, k, v = _qkv(4, 1, 32)
        bias = torch.zeros(4, 1, 32, device="cuda", dtype=DTYPE)
        with self.assertRaisesRegex(RuntimeError, "expanded size"):
            cudnn_op(q, k, v, bias)
        with self.assertRaisesRegex(RuntimeError, "expand"):
            ht.trace(cudnn_op, (q, k, v, bias), warm_up=False)
        # a 2-D bias expands to (B, 1, S_q, S_kv): the region's operand
        bias2 = _padding_mask(1, 1, 32)[0, 0]
        tape = ht.trace(cudnn_op, (q, k, v, bias2))
        self.assertEqual(
            tape.regions[0].inputs[3].sizes[0], tape.regions[0].inputs[0].sizes[0]
        )
        variant = build(tape, cudnn_op, (q, k, v, bias2))
        self.assertTrue(self._check(variant, cudnn_op, (*_qkv(2, 1, 32), bias2)))

    def test_every_case_traces_the_same_program_under_other_hints(self):
        # the recorder never reads a hint: every trace this class makes, made
        # again under other hints, is the same program (host_trace_two_hint)
        two_hint.assert_family(self)


if __name__ == "__main__":
    run_tests()
