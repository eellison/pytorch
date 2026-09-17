# Owner(s): ["module: cuda"]

import json
import unittest

import torch
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    parametrize,
    run_tests,
    skipIfRocm,
    TestCase,
)


if torch.cuda.is_available():
    import host_trace_two_hint as two_hint

    from torch.cuda import _host_trace as ht

# the op whose CUDA host is traced: the dense flash forward (mha_fwd)
flash_forward = torch.ops.aten._flash_attention_forward.default


def flash(q, k, v, causal, scale):
    # the dense path ignores max_q / max_k; the returned rng_state and the
    # unused tensor are uninitialized when there is no dropout
    return flash_forward(q, k, v, None, None, 0, 0, 0.0, causal, False, scale=scale)


def flash_dropout(q, k, v, p):
    return flash_forward(q, k, v, None, None, 0, 0, p, False, False)


# the dense flash backward host (mha_bwd): the dot(dO, O) preprocessing kernel,
# the seqk-parallel dq/dk/dv kernel and the dq accumulator convert, per call
flash_backward = torch.ops.aten._flash_attention_backward.default


def flash_bwd(dout, q, k, v, out, lse, rng_state, unused, causal, p, scale):
    # the dense path ignores max_q / max_k; rng_state is the forward's, read by
    # the kernel when p > 0
    return flash_backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        None,
        None,
        0,
        0,
        p,
        causal,
        rng_state,
        unused,
        scale=scale,
    )


def num_splits_heuristic(batch_nheads_mblocks, num_SMs, num_n_blocks, max_splits):
    # flash_api.cpp's heuristic in its float32 arithmetic
    import numpy as np

    f = np.float32
    if f(batch_nheads_mblocks) >= f(0.8) * f(num_SMs):
        return 1
    max_splits = min(max_splits, num_SMs, num_n_blocks)

    def ceildiv(a, b):
        return (a + b - 1) // b

    def eligible(s):
        return s == 1 or ceildiv(num_n_blocks, s) != ceildiv(num_n_blocks, s - 1)

    eff = []
    best = f(0)
    for s in range(1, max_splits + 1):
        if not eligible(s):
            eff.append(f(0))
            continue
        n_waves = f(batch_nheads_mblocks * s) / f(num_SMs)
        e = n_waves / np.ceil(n_waves)
        if e > best:
            best = e
        eff.append(e)
    for s in range(1, max_splits + 1):
        if eligible(s) and float(eff[s - 1]) >= 0.85 * float(best):
            return s
    return 1


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "flash attention not supported")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceFlash(TestCase):
    # 32 heads and 512 query rows keep batch * heads * m_blocks above the
    # split heuristic's threshold on any current GPU, so the shape sweeps
    # below stay on the plain kernel; the split path has its own tests
    H = 32
    D = 128

    def _qkv(self, B, Sq, Sk, H=None, Hk=None, dtype=torch.bfloat16, offset=0):
        H = H or self.H
        Hk = Hk or H

        def make(S, heads):
            flat = torch.randn(
                B * S * heads * self.D + offset, device="cuda", dtype=dtype
            )
            return flat[offset:].view(B, S, heads, self.D)

        return make(Sq, H), make(Sk, Hk), make(Sk, Hk)

    def _args(self, q, k, v, causal=False, scale=0.125):
        return (q, k, v, causal, scale)

    def _trace(self, *shape, **kw):
        causal = kw.pop("causal", False)
        scale = kw.pop("scale", 0.125)
        args = self._args(*self._qkv(*shape, **kw), causal, scale)
        return ht.trace(flash, args), args

    def _check_served(self, variant, args):
        out = variant.replay(args)
        ref = flash(*args)
        # the same kernels with the same launch configuration: bitwise equal
        self.assertEqual(out[0], ref[0], atol=0, rtol=0)
        self.assertEqual(out[1], ref[1], atol=0, rtol=0)
        return out

    def test_trace_records_the_forward(self):
        tape, _ = self._trace(4, 512, 512)
        parsed = json.loads(tape.to_json())
        # one kernel, or the split-KV kernel and its combine when the
        # heuristic splits at this shape on this device
        splits = parsed["opaque"][0]["expected"]
        self.assertEqual(tape.num_launches, 1 if splits == 1 else 2)
        launch = parsed["launches"][0]
        # the grid's batch and head dims are the input's size symbols (the
        # split kernel folds them into its third dim)
        sizes = parsed["inputs"][0]["sizes"]
        if splits == 1:
            self.assertEqual(launch["grid"][1], sizes[0])
            self.assertEqual(launch["grid"][2], sizes[2])
        else:
            self.assertIn(sizes[0], launch["grid"][2])
            self.assertIn(sizes[2], launch["grid"][2])
        # one by-value struct: its pointer fields are symbolic, named
        names = {
            p["name"] for p in launch["params"] if p["kind"] == "ptr" and not p["const"]
        }
        self.assertTrue(
            {"q_ptr", "k_ptr", "v_ptr", "o_ptr", "softmax_lse_ptr"} <= names
        )

    def test_replay_matches_eager_at_other_shapes(self):
        tape, args = self._trace(4, 512, 512)
        variant = ht.build(tape, flash, args)
        # sequence lengths that keep the traced kernel's even-MN branch, and
        # batch 1 and 8 from a batch-4 trace
        for B, Sq, Sk in [
            (4, 512, 512),
            (1, 512, 512),
            (8, 512, 512),
            (2, 1024, 256),
            (4, 384, 1024),
            (3, 512, 64),
        ]:
            self._check_served(variant, self._args(*self._qkv(B, Sq, Sk)))

    def test_branch_change_is_a_named_miss(self):
        tape, args = self._trace(4, 512, 512)
        variant = ht.build(tape, flash, args)
        # seqlen_q 500 is not a multiple of the kernel's block: the even-MN
        # branch flips, a guard on the tape
        with self.assertRaisesRegex(ht.Miss, "guard failed"):
            variant.replay(self._args(*self._qkv(4, 500, 512)))
        # a constant of the variant
        self.assertIsNone(
            variant.try_replay(self._args(*self._qkv(4, 512, 512), causal=True))
        )

    def test_causal_traces_separately(self):
        tape, args = self._trace(4, 512, 512, causal=True)
        variant = ht.build(tape, flash, args)
        for B, Sq, Sk in [(2, 512, 512), (1, 1024, 1024)]:
            self._check_served(variant, self._args(*self._qkv(B, Sq, Sk), causal=True))

    def test_scale_from_the_head_dim(self):
        # scale=None: the host derives 1/sqrt(d) from the symbolic head dim
        tape, args = self._trace(4, 512, 512, scale=None)
        variant = ht.build(tape, flash, args)
        self._check_served(variant, self._args(*self._qkv(2, 512, 1024), scale=None))
        self.assertEqual(variant.replay(args)[0], flash(*args)[0], atol=0, rtol=0)

    def test_the_scale_narrows_like_the_float_formal(self):
        # set_params_fprop took a float softmax_scale before the conversion:
        # the scale narrowed at the formal and scale_softmax_log2 = scale *
        # log2(e) was computed from that float. The host narrows it the same
        # way (ht::round_float32: Float32 of the symbolic scale on the tape),
        # so the constants a replay writes are eager's at the head dims where
        # the once- and twice-narrowed products differ in the last bit
        D = self.D
        try:
            for d in (48, 120, 200):
                self.D = d
                tape, args = self._trace(2, 128, 128, scale=None)
                self.assertIn("Float32(", tape.to_json())
                variant = ht.build(tape, flash, args)
                self._check_served(variant, self._args(*self._qkv(2, 128, 128), scale=None))
        finally:
            self.D = D

    def test_gqa_ratio_change(self):
        tape, args = self._trace(4, 512, 512, H=32, Hk=32)
        variant = ht.build(tape, flash, args)
        # the head ratio is an expression of the two head counts
        self._check_served(variant, self._args(*self._qkv(4, 512, 512, H=32, Hk=8)))
        self._check_served(variant, self._args(*self._qkv(2, 512, 512, H=32, Hk=4)))

    def test_decode_gqa_swap_path(self):
        # seqlen_q 1 with grouped heads takes the host's transpose path: views
        # inside the host and a batch stride the host rescales
        # batch 32 keeps the swapped problem (32 x 8 kv heads) on the plain kernel
        tape, args = self._trace(32, 1, 1024, H=32, Hk=8)
        variant = ht.build(tape, flash, args)
        for B, Sk in [(32, 1024), (64, 2048), (40, 512)]:
            self._check_served(variant, self._args(*self._qkv(B, 1, Sk, H=32, Hk=8)))
        # at batch 1 the heuristic splits the cache: a miss named after it
        with self.assertRaisesRegex(ht.Miss, "num_splits_heuristic"):
            variant.replay(self._args(*self._qkv(1, 1, 1024, H=32, Hk=8)))

    def test_inputs_with_storage_offsets(self):
        tape, args = self._trace(4, 512, 512)
        variant = ht.build(tape, flash, args)
        # flash reads its inputs 16 bytes at a time; offsets that keep that
        # alignment replay (the addresses are expressions of the offsets)
        for offset in (8, 64):
            self._check_served(
                variant, self._args(*self._qkv(2, 512, 512, offset=offset))
            )

    def test_fp16(self):
        tape, args = self._trace(2, 512, 512, dtype=torch.float16)
        variant = ht.build(tape, flash, args)
        self._check_served(
            variant, self._args(*self._qkv(4, 512, 256, dtype=torch.float16))
        )
        # dtype is part of the input record
        self.assertIsNone(variant.try_replay(self._args(*self._qkv(2, 512, 512))))

    def _num_sms(self):
        return (
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            * 2
        )

    def test_split_kv_and_the_heuristic(self):
        # decode: one query row, few heads, a long cache; the split count comes
        # from num_splits_heuristic, an opaque call on the tape
        Sk = 8192
        tape, args = self._trace(1, 1, Sk, H=8, Hk=8)
        parsed = json.loads(tape.to_json())
        self.assertEqual([o["fn"] for o in parsed["opaque"]], ["num_splits_heuristic"])
        splits = parsed["opaque"][0]["expected"]
        self.assertGreater(splits, 1)
        self.assertEqual(tape.num_launches, 2)  # split kernel and combine
        variant = ht.build(tape, flash, args)
        self._check_served(variant, self._args(*self._qkv(1, 1, Sk, H=8, Hk=8)))
        # the same heuristic at replay: a different split count is a miss
        # named after it
        num_n_blocks = (Sk + 127) // 128
        traced = num_splits_heuristic(8, self._num_sms(), num_n_blocks, 128)
        self.assertEqual(traced, splits)
        for B in range(1, 33):
            expect = num_splits_heuristic(B * 8, self._num_sms(), num_n_blocks, 128)
            new_args = self._args(*self._qkv(B, 1, Sk, H=8, Hk=8))
            if expect == splits:
                self._check_served(variant, new_args)
            else:
                with self.assertRaisesRegex(ht.Miss, "num_splits_heuristic"):
                    variant.replay(new_args)

    def test_split_count_ties(self):
        # shapes where a wave count is an exact integer are where a float
        # re-evaluation could disagree with the host; the opaque call runs
        # the host's own function, so served and missed follow it exactly
        Sk = 4096
        num_n_blocks = (Sk + 127) // 128
        sms = self._num_sms()
        ties = [
            B
            for B in range(1, 65)
            if any((B * 8 * s) % sms == 0 for s in range(1, min(128, num_n_blocks) + 1))
        ]
        self.assertTrue(ties)
        tape, args = self._trace(ties[0], 1, Sk, H=8, Hk=8)
        splits = json.loads(tape.to_json())["opaque"][0]["expected"]
        variant = ht.build(tape, flash, args)
        for B in ties:
            expect = num_splits_heuristic(B * 8, sms, num_n_blocks, 128)
            new_args = self._args(*self._qkv(B, 1, Sk, H=8, Hk=8))
            if expect == splits:
                self._check_served(variant, new_args)
            else:
                self.assertIsNone(variant.try_replay(new_args))

    def test_dropout_declines_by_name(self):
        q, k, v = self._qkv(2, 128, 128, H=8)
        with self.assertRaisesRegex(RuntimeError, "dropout is not traceable"):
            ht.trace(flash_dropout, (q, k, v, 0.1))
        self.assertFalse(torch._C._host_trace_tracing())
        # the ordinary path is unchanged
        out = flash_dropout(q, k, v, 0.1)
        self.assertEqual(out[0].shape, q.shape)

    def test_a_copy_on_write_input_stays_lazy(self):
        # q, k, v are read through the const accessor: a lazy clone stays
        # copy-on-write through an ordinary flash call, the trace's warm-up
        # and the build; out and softmax_lse go through the mutable form.
        q, k, v = self._qkv(2, 128, 128)
        lazy_q, lazy_k = torch._lazy_clone(q), torch._lazy_clone(k)
        self.assertTrue(torch._C._is_cow_tensor(lazy_q))
        ref = flash(*self._args(q, k, v))
        out = flash(*self._args(lazy_q, lazy_k, v))
        self.assertTrue(torch._C._is_cow_tensor(lazy_q))
        self.assertTrue(torch._C._is_cow_tensor(lazy_k))
        self.assertEqual(out[0], ref[0])
        args = self._args(lazy_q, lazy_k, v)
        tape = ht.trace(flash, args)
        variant = ht.build(tape, flash, args)
        self.assertTrue(torch._C._is_cow_tensor(lazy_q))
        self.assertTrue(torch._C._is_cow_tensor(lazy_k))
        self.assertEqual(variant.replay(args)[0], ref[0])

    def test_ordinary_mixed_dtype_still_raises(self):
        q, k, v = self._qkv(2, 128, 128, H=8)
        with self.assertRaisesRegex(RuntimeError, "same dtype"):
            flash(q, k.float(), v, False, 0.125)

    def test_every_case_traces_the_same_program_under_other_hints(self):
        # the recorder never reads a hint: every trace this class makes, made
        # again under other hints, is the same program (host_trace_two_hint)
        two_hint.assert_family(self)


def _eager_kernels(fn, args):
    # the kernels one eager call launches, in order, by the profiler's name
    from torch.profiler import profile, ProfilerActivity

    fn(*args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*args)
        torch.cuda.synchronize()
    return [
        e.name
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
        and not e.name.startswith(("Memset", "Memcpy"))
    ]


def _tape_kernels(tape):
    return [torch._C._demangle(L["kernel"]) for L in tape.launches]


@unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "flash attention not supported")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceFlashBackward(TestCase):
    # head dim 64: the seqk-parallel kernel's blocks are 128 x 128 on a device
    # with 144 KB of shared memory per block, so lengths that are multiples of
    # 128 keep the even-MN branch. With one or two key blocks every dq element
    # receives at most two atomic adds onto zero, which is order-independent,
    # so eager itself is bitwise reproducible at these lengths; longer keys
    # need the deterministic flag for a bitwise comparison (its own test).
    H = 8
    D = 64

    def _qkv(self, device, B, Sq, Sk, H=None, Hk=None, dtype=torch.bfloat16, offset=0):
        H = H or self.H
        Hk = Hk or H

        def make(S, heads):
            flat = torch.randn(
                B * S * heads * self.D + offset, device=device, dtype=dtype
            )
            return flat[offset:].view(B, S, heads, self.D)

        return make(Sq, H), make(Sk, Hk), make(Sk, Hk)

    def _case(self, device, B, Sq, Sk, causal=False, p=0.0, scale=0.125, seed=0, **kw):
        # the backward's inputs: the forward's outputs on fresh q, k, v and a
        # random cotangent; with dropout the forward's rng_state carries the
        # seed and offset the backward kernel re-derives the mask from
        torch.manual_seed(seed)
        q, k, v = self._qkv(device, B, Sq, Sk, **kw)
        out, lse, rng_state, unused, _ = flash_forward(
            q, k, v, None, None, 0, 0, p, causal, False, scale=scale
        )
        dout = torch.randn_like(out)
        return (dout, q, k, v, out, lse, rng_state, unused, causal, p, scale)

    def _trace(self, device, *shape, **kw):
        args = self._case(device, *shape, **kw)
        # a CUDAGraph collected during the trace's capture would invalidate it
        return ht.trace(flash_bwd, args), args

    def _check_served(self, variant, args):
        got = variant.replay(args)
        want = flash_bwd(*args)
        torch.cuda.synchronize()
        for g, w in zip(got, want):
            self.assertEqual(g.shape, w.shape)
            self.assertEqual(g.stride(), w.stride())
            self.assertEqual(g.dtype, w.dtype)
            # the same kernels with the same launch configuration and, with
            # dropout, the same mask from the forward's rng_state: bitwise
            self.assertTrue(torch.equal(g, w))
        return got

    def _gen(self, device):
        return torch.cuda.default_generators[torch.device(device).index or 0]

    def test_trace_records_the_three_launches(self, device):
        tape, _ = self._trace(device, 2, 128, 128)
        parsed = json.loads(tape.to_json())
        names = _tape_kernels(tape)
        self.assertEqual(tape.num_launches, 3)
        self.assertIn("flash_bwd_dot_do_o_kernel", names[0])
        self.assertIn("flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel", names[1])
        self.assertIn("flash_bwd_convert_dq_kernel", names[2])
        # dout: (b, seqlen_q, h, d), the first input; the host's shape checks
        # unify q's, k's and v's batch and head dims with dout's, and the
        # guard pass reads the earliest symbol into every use (E27), so the
        # grids name dout's symbols
        sizes = parsed["inputs"][0]["sizes"]
        for launch in parsed["launches"]:
            # every grid is (blocks, b, h) over the input's size symbols
            self.assertEqual(launch["grid"][1], sizes[0])
            self.assertEqual(launch["grid"][2], sizes[2])
        # the by-value struct: every pointer the kernels read or write is a
        # symbolic field, the forward's fields and the backward's alike
        symbolic = {
            p["name"]
            for p in parsed["launches"][1]["params"]
            if p["kind"] == "ptr" and not p["const"]
        }
        self.assertTrue(
            {
                "q_ptr",
                "k_ptr",
                "v_ptr",
                "o_ptr",
                "softmax_lse_ptr",
                "do_ptr",
                "dq_ptr",
                "dk_ptr",
                "dv_ptr",
                "dq_accum_ptr",
                "dsoftmax_sum",
            }
            <= symbolic
        )
        # the convert kernel takes the split count beside the struct: the typed
        # launch records it positionally, a constant 1 without the flag
        splits = [p for p in parsed["launches"][2]["params"] if p["name"] == ""]
        self.assertEqual(
            [(p["kind"], p["expr"], p["const"]) for p in splits], [("i32", 1, True)]
        )
        # the backward draws no randomness: no declaration on the tape
        self.assertIsNone(tape.rng_increment)

    @parametrize("causal", [False, True])
    @parametrize("p", [0.0, 0.1])
    def test_replay_matches_eager_at_other_shapes(self, device, causal, p):
        tape, args = self._trace(device, 2, 128, 128, causal=causal, p=p)
        variant = ht.build(tape, flash_bwd, args)
        # other batch sizes, lengths and addresses from one trace: the same
        # kernel variant (even-MN, causal and dropout are constants of it)
        for B, Sq, Sk, offset, seed in [
            (2, 128, 128, 0, 1),
            (1, 256, 128, 0, 2),
            (4, 128, 256, 0, 3),
            (3, 256, 256, 64, 4),
        ]:
            self._check_served(
                variant,
                self._case(
                    device, B, Sq, Sk, causal=causal, p=p, offset=offset, seed=seed
                ),
            )
        # a length off the block grid flips the even-MN branch: a guard
        with self.assertRaisesRegex(ht.Miss, "guard failed"):
            variant.replay(self._case(device, 2, 100, 128, causal=causal, p=p))
        # the flags are constants of the variant
        self.assertIsNone(
            variant.try_replay(self._case(device, 2, 128, 128, causal=not causal, p=p))
        )

    def test_dropout_backward_draws_nothing(self, device):
        # the mask comes from the forward's rng_state; neither eager nor the
        # replay touches the generator
        tape, args = self._trace(device, 2, 128, 128, p=0.1)
        variant = ht.build(tape, flash_bwd, args)
        gen = self._gen(device)
        before = gen.get_offset()
        flash_bwd(*args)
        self.assertEqual(gen.get_offset(), before)
        self._check_served(variant, args)
        self.assertEqual(gen.get_offset(), before)
        # the same forward under the same seed gives the same rng_state, and
        # the backward of it the same gradients at another address
        again = self._case(device, 2, 128, 128, p=0.1, offset=8)
        self.assertTrue(torch.equal(again[6], args[6]))
        self._check_served(variant, again)

    def test_scale_from_the_head_dim(self, device):
        # scale=None: the host derives 1/sqrt(d) from the symbolic head dim
        tape, args = self._trace(device, 2, 128, 128, scale=None)
        variant = ht.build(tape, flash_bwd, args)
        self._check_served(variant, self._case(device, 1, 256, 256, scale=None, seed=5))

    def test_gqa_sums_the_head_groups(self, device):
        # fewer key heads than query heads: the kernel writes per-query-head
        # dk / dv into expanded buffers the host allocates and sums over the
        # groups into the outputs (at::sum_out on the reductions sibling)
        tape, args = self._trace(device, 2, 128, 128, H=8, Hk=2)
        self.assertEqual(tape.num_launches, 5)
        names = _tape_kernels(tape)
        self.assertEqual(sum("reduce_kernel" in n for n in names), 2)
        variant = ht.build(tape, flash_bwd, args)
        for B, Sq, Sk, Hk, seed in [
            (3, 256, 128, 2, 1),
            (4, 128, 256, 4, 2),
            (2, 256, 256, 2, 3),
        ]:
            self._check_served(
                variant, self._case(device, B, Sq, Sk, H=8, Hk=Hk, seed=seed)
            )
        # the sibling's iterator coalesces a size-1 dim differently: batch 1
        # and a single key head (MQA) are guards of the reduction
        for B, Hk in [(1, 2), (2, 1)]:
            with self.assertRaisesRegex(ht.Miss, "guard failed"):
                variant.replay(self._case(device, B, 128, 128, H=8, Hk=Hk, seed=4))

    def test_deterministic_flag(self, device):
        # torch.use_deterministic_algorithms(True): the host allocates one dq
        # accumulator slice per key-block group (a zeros allocation, its count
        # from the SM count and b * h), the kernel grid and the convert's split
        # count follow it; eager is then bitwise reproducible at any length.
        # The flag's fill of uninitialized memory is off: eager's empty would
        # launch a fill kernel per allocation the trace does not record
        # (a TapeMismatch at the build, see STAGE2B.md)
        with DeterministicGuard(True, fill_uninitialized_memory=False):
            tape, args = self._trace(device, 2, 128, 512)
            parsed = json.loads(tape.to_json())
            splits = [p for p in parsed["launches"][2]["params"] if p["name"] == ""]
            self.assertEqual(len(splits), 1)
            self.assertFalse(splits[0]["const"])
            self.assertEqual(parsed["launches"][1]["grid"][0], splits[0]["expr"])
            variant = ht.build(tape, flash_bwd, args)
            for B, Sq, Sk, seed in [
                (1, 128, 512, 1),
                (4, 256, 384, 2),
                (2, 128, 512, 3),
            ]:
                self._check_served(variant, self._case(device, B, Sq, Sk, seed=seed))

    def test_tape_holds_the_kernels_eager_launches(self, device):
        # fidelity: the tape's launches are the profiler's kernels of one
        # eager call, by name and in order, for each variant of the host
        cases = {
            "plain": dict(),
            "causal": dict(causal=True),
            "dropout": dict(p=0.1),
            "gqa": dict(H=8, Hk=2),
        }
        for label, kw in cases.items():
            with self.subTest(label):
                tape, args = self._trace(device, 2, 128, 128, **kw)
                want = _eager_kernels(flash_bwd, args)
                got = _tape_kernels(tape)
                self.assertEqual(len(got), len(want))
                for g, w in zip(got, want):
                    if "reduce_kernel" in w:
                        # the sibling's reduction names its functor; eager's is
                        # a lambda: the template and its integer arguments agree
                        self.assertEqual(g.split("<")[0], w.split("<")[0])
                    else:
                        self.assertEqual(g, w)
        with DeterministicGuard(True, fill_uninitialized_memory=False):
            tape, args = self._trace(device, 2, 128, 512)
            self.assertEqual(_tape_kernels(tape), _eager_kernels(flash_bwd, args))

    def test_the_varlen_hosts_decline_by_name(self, device):
        # mha_varlen_fwd / mha_varlen_bwd sit in the traced scope unconverted:
        # a call with cu_seqlens declines at their entry, by name, before any
        # raw read of the lengths
        B, S, H, D = 2, 64, 4, 64
        q, k, v = (
            torch.randn(B * S, H, D, device=device, dtype=torch.bfloat16)
            for _ in range(3)
        )
        cu = torch.arange(0, (B + 1) * S, S, device=device, dtype=torch.int32)

        def fwd(q, k, v, cu_q, cu_k):
            return flash_forward(q, k, v, cu_q, cu_k, S, S, 0.0, False, False)

        with self.assertRaisesRegex(ht.Declined, "cu_seqlens .mha_varlen_fwd."):
            ht.trace(fwd, (q, k, v, cu, cu))
        out, lse, rng_state, unused, _ = fwd(q, k, v, cu, cu)
        dout = torch.randn_like(out)

        def bwd(dout, q, k, v, out, lse, cu_q, cu_k, rng_state, unused):
            return flash_backward(
                dout, q, k, v, out, lse, cu_q, cu_k, S, S, 0.0, False, rng_state, unused
            )

        with self.assertRaisesRegex(ht.Declined, "cu_seqlens .mha_varlen_bwd."):
            ht.trace(bwd, (dout, q, k, v, out, lse, cu, cu, rng_state, unused))

    def test_a_lazily_cloned_out_materializes_as_eagers_data_ptr_did(self, device):
        # the forward's `out` reaches the backward through the forward helper's
        # mutable accessor: the unconverted host read it with data_ptr(), which
        # materializes a copy-on-write tensor, and the converted host does the
        # same in ordinary mode, at the warm-up and at the build (A98:
        # materialize where eager did, never elsewhere); the values are bitwise
        args = self._case(device, 2, 128, 128)
        want = flash_bwd(*args)
        lazy = torch._lazy_clone(args[4])
        self.assertTrue(torch._C._is_cow_tensor(lazy))
        lazy_args = args[:4] + (lazy,) + args[5:]
        got = flash_bwd(*lazy_args)
        self.assertFalse(torch._C._is_cow_tensor(lazy))
        for g, w in zip(got, want):
            self.assertTrue(torch.equal(g, w))
        lazy = torch._lazy_clone(args[4])
        lazy_args = args[:4] + (lazy,) + args[5:]
        tape = ht.trace(flash_bwd, lazy_args)
        self.assertFalse(torch._C._is_cow_tensor(lazy))
        variant = ht.build(tape, flash_bwd, lazy_args)
        for g, w in zip(variant.replay(lazy_args), want):
            self.assertTrue(torch.equal(g, w))

    def test_the_varlen_hosts_decline_by_name(self, device):
        # mha_varlen_fwd / mha_varlen_bwd sit in the traced scope unconverted:
        # a call with cu_seqlens declines at their entry, by name, before any
        # raw read of the lengths
        B, S, H, D = 2, 64, 4, 64
        q, k, v = (
            torch.randn(B * S, H, D, device=device, dtype=torch.bfloat16)
            for _ in range(3)
        )
        cu = torch.arange(0, (B + 1) * S, S, device=device, dtype=torch.int32)

        def fwd(q, k, v, cu_q, cu_k):
            return flash_forward(q, k, v, cu_q, cu_k, S, S, 0.0, False, False)

        with self.assertRaisesRegex(ht.Declined, "cu_seqlens .mha_varlen_fwd."):
            ht.trace(fwd, (q, k, v, cu, cu))
        out, lse, rng_state, unused, _ = fwd(q, k, v, cu, cu)
        dout = torch.randn_like(out)

        def bwd(dout, q, k, v, out, lse, cu_q, cu_k, rng_state, unused):
            return flash_backward(
                dout, q, k, v, out, lse, cu_q, cu_k, S, S, 0.0, False, rng_state, unused
            )

        with self.assertRaisesRegex(ht.Declined, "cu_seqlens .mha_varlen_bwd."):
            ht.trace(bwd, (dout, q, k, v, out, lse, cu, cu, rng_state, unused))

    def test_a_lazily_cloned_out_materializes_as_eagers_data_ptr_did(self, device):
        # the forward's `out` reaches the backward through the forward helper's
        # mutable accessor: the unconverted host read it with data_ptr(), which
        # materializes a copy-on-write tensor, and the converted host does the
        # same in ordinary mode, at the warm-up and at the build (A98:
        # materialize where eager did, never elsewhere); the values are bitwise
        args = self._case(device, 2, 128, 128)
        want = flash_bwd(*args)
        lazy = torch._lazy_clone(args[4])
        self.assertTrue(torch._C._is_cow_tensor(lazy))
        lazy_args = args[:4] + (lazy,) + args[5:]
        got = flash_bwd(*lazy_args)
        self.assertFalse(torch._C._is_cow_tensor(lazy))
        for g, w in zip(got, want):
            self.assertTrue(torch.equal(g, w))
        lazy = torch._lazy_clone(args[4])
        lazy_args = args[:4] + (lazy,) + args[5:]
        tape = ht.trace(flash_bwd, lazy_args)
        self.assertFalse(torch._C._is_cow_tensor(lazy))
        variant = ht.build(tape, flash_bwd, lazy_args)
        for g, w in zip(variant.replay(lazy_args), want):
            self.assertTrue(torch.equal(g, w))
        # the mutable read names out's root among the tape's written roots, so
        # a replay materializes a fresh lazy clone before it runs, as eager's
        # call does; the values equal. The six const inputs stay lazy
        # (test_a_copy_on_write_input_stays_lazy)
        out_root = next(i for i in tape.inputs if i.position == 4).root.name
        self.assertIn(out_root, tape.written_roots)
        lazy = torch._lazy_clone(args[4])
        got = variant.replay(args[:4] + (lazy,) + args[5:])
        self.assertFalse(torch._C._is_cow_tensor(lazy))
        for g, w in zip(got, want):
            self.assertTrue(torch.equal(g, w))

    def test_a_copy_on_write_input_stays_lazy(self, device):
        # q, k, v, dout, the logsumexp and rng_state are read through the
        # const accessor: lazy clones stay copy-on-write through an ordinary
        # call, the trace's warm-up and the build (out goes through the
        # forward helper's mutable form, as its data_ptr() did before)
        args = self._case(device, 2, 128, 128, p=0.1)
        dout, q, k, v, out, lse, rng_state, unused = args[:8]
        lazy = [torch._lazy_clone(t) for t in (dout, q, k, v, lse, rng_state)]
        for t in lazy:
            self.assertTrue(torch._C._is_cow_tensor(t))
        lazy_args = (
            lazy[0],
            lazy[1],
            lazy[2],
            lazy[3],
            out,
            lazy[4],
            lazy[5],
            unused,
        ) + args[8:]
        want = flash_bwd(*args)
        got = flash_bwd(*lazy_args)
        for t in lazy:
            self.assertTrue(torch._C._is_cow_tensor(t))
        for g, w in zip(got, want):
            self.assertTrue(torch.equal(g, w))
        tape = ht.trace(flash_bwd, lazy_args)
        variant = ht.build(tape, flash_bwd, lazy_args)
        for t in lazy:
            self.assertTrue(torch._C._is_cow_tensor(t))
        for g, w in zip(variant.replay(lazy_args), want):
            self.assertTrue(torch.equal(g, w))

    def test_every_case_traces_the_same_program_under_other_hints(self, device):
        # the recorder never reads a hint: every trace this class makes, made
        # again under other hints, is the same program (host_trace_two_hint)
        two_hint.assert_family(
            self, exclude={"test_exclusions": "a device-generic class attribute"}
        )


instantiate_device_type_tests(
    TestCudaHostTraceFlashBackward, globals(), only_for="cuda"
)

if __name__ == "__main__":
    run_tests()
