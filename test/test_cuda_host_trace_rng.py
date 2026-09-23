# Owner(s): ["module: cuda"]

import gc
import unittest

from host_trace_testing import assert_eager_function_handles, build

import torch
import torch.nn.functional as F
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfRocm,
    TEST_CUDA_PYTHON_BINDINGS,
    TestCase,
)


if torch.cuda.is_available():
    import host_trace_two_hint as two_hint

    from torch.cuda import _host_trace as ht

flash_forward = torch.ops.aten._flash_attention_forward.default
H, DH = 8, 64
SEED = 1234


def flash_dropout(q, k, v, p):
    # (out, softmax_lse, philox_seed, philox_offset, debug_mask): the kernel
    # writes the seed and offset it drew from into the two state tensors
    return flash_forward(q, k, v, None, None, 0, 0, p, False, False)


def dropout(x, p):
    return torch.native_dropout(x, p, True)


def step(x, w):
    # a decode-shaped composition with a random stage in the middle
    h = F.layer_norm(x, (x.shape[-1],), w, None)
    d = torch.native_dropout(h, 0.1, True)[0]
    y = F.silu(d + x)
    return (y * w).sum(-1)


def step2(x, w):
    # two random stages, each with its own increment
    h = F.layer_norm(x, (x.shape[-1],), w, None)
    d = torch.native_dropout(h, 0.1, True)[0]
    y = torch.native_dropout(F.silu(d + x), 0.2, True)[0]
    return (y * w).sum(-1)


def two_dropouts(x):
    # two random launches of different sizes in one trace: the second draws
    # from where the first stopped
    d1 = torch.native_dropout(x, 0.1, True)[0]
    d2 = torch.native_dropout(d1[:, :1024].contiguous(), 0.2, True)[0]
    return d1, d2


def _offset():
    torch.cuda.synchronize()
    return _gen().get_offset()


def _gen():
    return torch.cuda.default_generators[torch.cuda.current_device()]


def _replayer(variant):
    return lambda *a: variant.replay(a)


def _sequence(fn, args, n):
    # n consecutive calls from a fixed generator state: the values and the
    # generator offset after each call
    torch.cuda.manual_seed(SEED)
    outs, offsets = [], []
    for _ in range(n):
        out = fn(*args)
        if isinstance(out, torch.Tensor):
            out = [out]
        outs.append([t.clone() for t in out])
        torch.cuda.synchronize()
        offsets.append(_gen().get_offset())
    return outs, offsets


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceRng(TestCase):
    def _equal_sequences(self, got, ref, offsets, ref_offsets):
        self.assertEqual(offsets, ref_offsets)
        for g, r in zip(got, ref):
            self.assertEqual(len(g), len(r))
            for a, b in zip(g, r):
                self.assertEqual(a, b, atol=0, rtol=0)

    def _trace(self, fn, args):
        tape = ht.trace(fn, args)
        self.assertIsNotNone(tape.rng_increment)
        # the trace's warm-up drew one call's randomness before the sequences
        # start; the build draws nothing
        return build(tape, fn, args)

    def test_the_graph_setter_exists(self):
        # upstream patch 11: the per-replay generator increment
        self.assertTrue(hasattr(torch.cuda.CUDAGraph, "set_generator_increment"))

    def test_native_dropout_sequence_matches_eager(self):
        x = torch.randn(4, 4096, device="cuda")
        args = (x, 0.1)
        tape = self._trace(dropout, args)
        ref, ref_offsets = _sequence(dropout, args, 5)
        got, offsets = _sequence(_replayer(tape), args, 5)
        self._equal_sequences(got, ref, offsets, ref_offsets)
        # the mask is a real dropout mask, not all ones
        self.assertLess(got[0][1].float().mean().item(), 0.95)

    def test_native_dropout_other_shapes_advance_by_their_own_increment(self):
        x = torch.randn(4, 4096, device="cuda")
        tape = self._trace(dropout, (x, 0.1))
        for shape in [(3, 4096), (8, 4096), (1, 4096), (16, 1024)]:
            y = torch.randn(*shape, device="cuda")
            ref, ref_offsets = _sequence(dropout, (y, 0.1), 3)
            got, offsets = _sequence(_replayer(tape), (y, 0.1), 3)
            self._equal_sequences(got, ref, offsets, ref_offsets)
        # the vector width is a guard: a count that leaves a remainder misses
        with self.assertRaisesRegex(ht.Miss, "guard failed"):
            tape.replay((torch.randn(3, 4097, device="cuda"), 0.1))

    def test_native_dropout_bf16_and_fp16(self):
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.randn(2, 8192, device="cuda", dtype=dtype)
            tape = self._trace(dropout, (x, 0.25))
            ref, ref_offsets = _sequence(dropout, (x, 0.25), 3)
            got, offsets = _sequence(_replayer(tape), (x, 0.25), 3)
            self._equal_sequences(got, ref, offsets, ref_offsets)

    def test_ordinary_sibling_matches_the_real_dropout(self):
        # outside a trace the sibling runs the same kernel with the same
        # launch and the same increment: identical sequences from one seed
        C = torch._C
        for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.float64):
            for numel in (
                1,
                2,
                3,
                7,
                8,
                15,
                16,
                17,
                255,
                256,
                257,
                1023,
                1024,
                1025,
                65537,
                2**20 + 1,
            ):
                x = torch.randn(numel, device="cuda", dtype=dtype)
                ref, ref_offsets = _sequence(dropout, (x, 0.3), 2)
                got, offsets = _sequence(
                    lambda t, p: C._host_trace_ti_native_dropout(t, p, True),
                    (x, 0.3),
                    2,
                )
                self._equal_sequences(got, ref, offsets, ref_offsets)

    @unittest.skipIf(not TEST_CUDA_PYTHON_BINDINGS, "cuda.bindings reads the nodes")
    def test_native_dropout_replays_eager_function_handles(self):
        # the entry is compiled into Dropout.cu and launches that file's
        # fused_dropout_kernel_vec / fused_dropout_kernel instantiations: the
        # entry's node, the tape's launch and the replay's node hold the
        # function handle eager's capture holds (E36); the vector width follows
        # the input's alignment and element count as in eager
        def real(t):
            return torch.native_dropout(t, 0.1, True)

        def entry(t):
            return torch._C._host_trace_ti_native_dropout(t, 0.1, True)

        x = torch.randn(4, 4096, device="cuda")
        cases = {
            "f32 (vec 4)": x,
            "bf16 (vec 8)": x.to(torch.bfloat16),
            "f16 misaligned (vec 1)": x.to(torch.float16).flatten()[1:],
            "f32 odd count (vec 1)": x.flatten()[:4097],
        }
        for name, t in cases.items():
            with self.subTest(case=name):
                eager = assert_eager_function_handles(
                    self, real, (t,), entry, launches=1
                )
                self.assertEqual(len(eager), 1)

    def test_native_dropout_declines(self):
        x = torch.randn(4, 4096, device="cuda")
        with self.assertRaisesRegex(ht.Declined, "train=False"):
            ht.trace(lambda t: torch.native_dropout(t, 0.1, False), (x,))
        with self.assertRaisesRegex(ht.Declined, "non-contiguous"):
            ht.trace(lambda t: torch.native_dropout(t.t(), 0.1, True), (x,))
        self.assertFalse(torch._C._host_trace_tracing())

    def test_undeclared_randomness_declines_by_name(self):
        # an op the tracer does not know that would draw from the generator
        x = torch.randn(4, 4096, device="cuda")
        with self.assertRaisesRegex(ht.Declined, "poisson"):
            ht.trace(lambda t: t + torch.poisson(t.abs()), (x,))
        self.assertFalse(torch._C._host_trace_tracing())

    def test_two_random_launches_in_one_trace(self):
        # each launch declares its own increment; the tape carries both slots
        # and the second kernel starts at the first's increment
        x = torch.randn(4, 4096, device="cuda")
        tape = self._trace(two_dropouts, (x,))
        slots = tape.tape.rng_slots
        self.assertEqual(len(slots), 2)
        # the slice's contiguous copy is a launch of its own between the two
        # dropout kernels; the slots name the dropout launches, in order
        self.assertLess(slots[0]["launch"], slots[1]["launch"])
        for r in slots:
            self.assertIn("Dropout", tape.tape.launches[r["launch"]]["kernel"])
        ref, ref_offsets = _sequence(two_dropouts, (x,), 5)
        got, offsets = _sequence(_replayer(tape), (x,), 5)
        self._equal_sequences(got, ref, offsets, ref_offsets)
        # the generator advanced by both increments
        torch.cuda.manual_seed(SEED)
        o0 = _offset()
        torch.native_dropout(x, 0.1, True)
        o1 = _offset()
        torch.native_dropout(x[:, :1024].contiguous(), 0.2, True)
        o2 = _offset()
        self.assertEqual(ref_offsets[1] - ref_offsets[0], (o1 - o0) + (o2 - o1))

    def test_two_random_launches_replay_at_another_batch(self):
        # both increments change with the batch; the prefix sum follows
        x = torch.randn(4, 4096, device="cuda")
        tape = self._trace(two_dropouts, (x,))
        for B in (8, 2, 16):
            xb = torch.randn(B, 4096, device="cuda")
            ref, ref_offsets = _sequence(two_dropouts, (xb,), 3)
            got, offsets = _sequence(_replayer(tape), (xb,), 3)
            self._equal_sequences(got, ref, offsets, ref_offsets)

    def test_composition_with_two_random_stages(self):
        w = 1 + 0.1 * torch.randn(1024, device="cuda")
        x = torch.randn(4, 1024, device="cuda")
        tape = self._trace(step2, (x, w))
        self.assertEqual(len(tape.tape.rng_slots), 2)
        ref, ref_offsets = _sequence(step2, (x, w), 10)
        got, offsets = _sequence(_replayer(tape), (x, w), 10)
        self._equal_sequences(got, ref, offsets, ref_offsets)

    def test_trace_and_build_consumption_is_exact(self):
        # the trace draws randomness of its own: one call's increment for the
        # warm-up, none without it; the build draws nothing (it runs nothing
        # of the function); a drift in these counts changes the stream a
        # program sees after tracing
        x = torch.randn(4, 4096, device="cuda")
        torch.cuda.manual_seed(SEED)
        o0 = _offset()
        dropout(x, 0.1)
        one = _offset() - o0
        self.assertGreater(one, 0)
        for warm_up, trace_calls in ((True, 1), (False, 0)):
            torch.cuda.manual_seed(SEED)
            o0 = _offset()
            tape = ht.trace(dropout, (x, 0.1), warm_up=warm_up)
            o1 = _offset()
            build(tape, dropout, (x, 0.1))
            o2 = _offset()
            self.assertEqual(o1 - o0, trace_calls * one)
            self.assertEqual(o2 - o1, 0)

    def test_a_call_executes_the_user_function_exactly_once(self):
        # an entry (a trace, hits, a miss traced again) executes the user's
        # function once per call: the ordinary call whose outputs the caller
        # receives is the warm-up, the trace follows with warm_up=False and
        # draws nothing, and the build draws nothing; a function that mutates
        # its input and draws shows an extra execution in the tensor and in
        # the generator's advance (test_cuda_host_trace counts the executions
        # of the mutation-only case)
        def fn(x):
            return x.add_(torch.native_dropout(x, 0.1, True)[0])

        def run(entry):
            torch.cuda.manual_seed(SEED)
            xs = [torch.randn(4, 4096, device="cuda") for _ in range(2)]
            xs.append(torch.randn(2, 8, 4096, device="cuda"))
            outs, advances = [], []
            for x in xs:
                before = _offset()
                outs.append(entry(x).clone())
                advances.append(_offset() - before)
            return outs, advances

        variants = []

        def entry(x):
            for v in variants:
                out = v.try_replay((x,))
                if out is not None:
                    return out[0]
            out = fn(x)
            tape = ht.trace(fn, (x,), warm_up=False)
            variants.append(build(tape, fn, (x,)))
            return out

        want, want_advances = run(fn)
        got, got_advances = run(entry)
        self.assertEqual(len(variants), 2)
        # per call (the trace, the hit, the miss): one eager call's advance of
        # the generator, and the mutated input bitwise
        self.assertGreater(min(want_advances), 0)
        self.assertEqual(got_advances, want_advances)
        for g, w in zip(got, want):
            self.assertEqual(g, w, atol=0, rtol=0)

    def test_tape_is_bound_to_its_device_class(self):
        # the SM count is folded into dropout's grid cap and increment: a tape
        # replayed on a device with another count misses before any GPU work
        x = torch.randn(4, 4096, device="cuda")
        tape = ht.trace(dropout, (x, 0.1))
        props = torch.cuda.get_device_properties(x.device)
        ident = dict(tape.device_identity)
        self.assertEqual(ident["multi_processor_count"], props.multi_processor_count)
        ident["multi_processor_count"] += 1
        tape.device_identity = tuple(ident.items())
        with self.assertRaisesRegex(ht.Miss, "multi_processor_count"):
            build(tape, dropout, (x, 0.1))
        ident["multi_processor_count"] -= 1
        tape.device_identity = tuple(ident.items())
        build(tape, dropout, (x, 0.1))

    def test_composition_with_a_random_stage(self):
        w = 1 + 0.1 * torch.randn(1024, device="cuda")
        x = torch.randn(4, 1024, device="cuda")
        tape = self._trace(step, (x, w))
        ref, ref_offsets = _sequence(step, (x, w), 10)
        got, offsets = _sequence(_replayer(tape), (x, w), 10)
        self._equal_sequences(got, ref, offsets, ref_offsets)
        # two batch sizes interleaved from one tape
        x2 = torch.randn(2, 1024, device="cuda")
        torch.cuda.manual_seed(SEED)
        ref = [step(x, w).clone(), step(x2, w).clone(), step(x, w).clone()]
        torch.cuda.synchronize()
        ref_off = _gen().get_offset()
        torch.cuda.manual_seed(SEED)
        got = [
            tape.replay((x, w))[0].clone(),
            tape.replay((x2, w))[0].clone(),
            tape.replay((x, w))[0].clone(),
        ]
        torch.cuda.synchronize()
        self.assertEqual(_gen().get_offset(), ref_off)
        for a, b in zip(got, ref):
            self.assertEqual(a, b, atol=0, rtol=0)

    def test_a_graph_freed_between_the_trace_and_the_build_keeps_eagers_sequences(
        self,
    ):
        # a launch's philox seed / offset pointers are a replay's per-capture
        # generator words; the tape's are the trace capture's, freed when the
        # trace ended, and only the allocator's free list made them the same
        # block again. A graph freed between the trace and the build puts its
        # own words on top of that list, so the replay's words are elsewhere:
        # an image a replay pushes (another batch allocates afresh) must still
        # name its own capture's words, and the values stay eager's (a replay
        # naming the trace's freed words draws from another state)
        x = torch.randn(4, 4096, device="cuda")
        stale = build(ht.trace(dropout, (x, 0.1)), dropout, (x, 0.1))
        tape = ht.trace(dropout, (x, 0.1))
        del stale
        gc.collect()
        variant = build(tape, dropout, (x, 0.1))
        j = tape.rng_slots[0]["launch"]
        rng = [p for p in tape.launches[j]["params"] if p["kind"] == "rng"]
        self.assertTrue(rng)
        for B in (4, 8, 2):
            xb = torch.randn(B, 4096, device="cuda")
            ref, ref_offsets = _sequence(dropout, (xb, 0.1), 3)
            got, offsets = _sequence(_replayer(variant), (xb, 0.1), 3)
            self._equal_sequences(got, ref, offsets, ref_offsets)

    def test_generator_capture_pointers_binding(self):
        gen = torch.cuda.default_generators[torch.cuda.current_device()]
        with self.assertRaisesRegex(RuntimeError, "no stream capture is active"):
            torch._C._host_trace_generator_capture_pointers(gen)
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
            seed_ptr, offset_ptr, intragraph = (
                torch._C._host_trace_generator_capture_pointers(gen)
            )
        self.assertNotEqual(seed_ptr, 0)
        self.assertNotEqual(offset_ptr, 0)
        self.assertEqual(intragraph, 0)

    def test_every_case_traces_the_same_program_under_other_hints(self):
        # the recorder never reads a hint: every trace this class makes, made
        # again under other hints, is the same program (host_trace_two_hint)
        two_hint.assert_family(self)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "flash attention not supported")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceFlashDropout(TestCase):
    def _qkv(self, B, S, dtype=torch.bfloat16):
        return tuple(
            torch.randn(B, S, H, DH, device="cuda", dtype=dtype) for _ in range(3)
        )

    def _outputs(self, out):
        # out, softmax_lse and the (seed, offset) pair the kernel drew from;
        # the fourth output is the host's uninitialized placeholder tensor
        return [out[0], out[1], out[2]]

    def _sequence(self, fn, args, n):
        torch.cuda.manual_seed(SEED)
        outs, offsets = [], []
        for _ in range(n):
            outs.append([t.clone() for t in self._outputs(fn(*args))])
            torch.cuda.synchronize()
            offsets.append(_gen().get_offset())
        return outs, offsets

    def _equal(self, got, ref, offsets, ref_offsets):
        self.assertEqual(offsets, ref_offsets)
        for g, r in zip(got, ref):
            for a, b in zip(g, r):
                self.assertEqual(a, b, atol=0, rtol=0)

    def test_flash_dropout_sequence_matches_eager(self):
        q, k, v = self._qkv(4, 128)
        args = (q, k, v, 0.1)
        t = ht.trace(flash_dropout, args)
        self.assertIsNotNone(t.rng_increment)
        tape = build(t, flash_dropout, args)
        ref, ref_offsets = self._sequence(flash_dropout, args, 5)
        got, offsets = self._sequence(_replayer(tape), args, 5)
        self._equal(got, ref, offsets, ref_offsets)
        # b * h * 32 offsets per call
        self.assertEqual(ref_offsets[1] - ref_offsets[0], 4 * H * 32)

    def test_flash_dropout_other_batches_advance_by_b_h_32(self):
        q, k, v = self._qkv(4, 128)
        tape = build(
            ht.trace(flash_dropout, (q, k, v, 0.1)), flash_dropout, (q, k, v, 0.1)
        )
        for B in (2, 8, 1):
            args = (*self._qkv(B, 128), 0.1)
            ref, ref_offsets = self._sequence(flash_dropout, args, 3)
            got, offsets = self._sequence(_replayer(tape), args, 3)
            self._equal(got, ref, offsets, ref_offsets)
            self.assertEqual(ref_offsets[1] - ref_offsets[0], B * H * 32)

    def test_dropout_then_flash_dropout_in_one_trace(self):
        # two random launches of different kinds: the sibling's inline philox
        # state and flash's state inside its params struct; flash starts at
        # the dropout's increment
        q, k, v = self._qkv(4, 128)

        def fn(q, k, v, p):
            h = torch.native_dropout(q, 0.1, True)[0]
            return flash_dropout(h, k, v, p)

        args = (q, k, v, 0.1)
        t = ht.trace(fn, args)
        self.assertEqual(len(t.rng_slots), 2)
        tape = build(t, fn, args)
        ref, ref_offsets = self._sequence(fn, args, 5)
        got, offsets = self._sequence(_replayer(tape), args, 5)
        self._equal(got, ref, offsets, ref_offsets)
        for B in (2, 8):
            args_b = (*self._qkv(B, 128), 0.1)
            ref, ref_offsets = self._sequence(fn, args_b, 3)
            got, offsets = self._sequence(_replayer(tape), args_b, 3)
            self._equal(got, ref, offsets, ref_offsets)

    def test_flash_dropout_ordinary_path_unchanged(self):
        q, k, v = self._qkv(2, 128)
        torch.cuda.manual_seed(SEED)
        a = flash_dropout(q, k, v, 0.1)
        torch.cuda.manual_seed(SEED)
        b = flash_dropout(q, k, v, 0.1)
        self.assertEqual(a[0], b[0], atol=0, rtol=0)
        self.assertFalse(torch._C._host_trace_tracing())

    def test_every_case_traces_the_same_program_under_other_hints(self):
        # the recorder never reads a hint: every trace this class makes, made
        # again under other hints, is the same program (host_trace_two_hint)
        two_hint.assert_family(self)


_TWO_STATES_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxCudaState.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <ATen/cuda/host_trace/Launch.h>
#include <ATen/cuda/host_trace/Philox.h>
#include <ATen/cuda/host_trace/Recorder.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <mutex>
namespace ht = at::cuda::host_trace;
// the philox state as a kernel argument, recorded as an `rng` field
using ht::TracedPhilox;

__global__ void write_offset(int64_t* out, at::PhiloxCudaState st) {
  // the philox offset the kernel would draw from
  auto seeds = at::cuda::philox::unpack(st);
  if (threadIdx.x == 0) {
    out[0] = static_cast<int64_t>(std::get<1>(seeds));
  }
}

// Two rng_increment declarations before any launch, one random kernel per
// state. In order: K1 takes state A, K2 state B. Swapped: K2 (state B)
// launches first, which host order would pair with declaration A.
at::Tensor two_states(const at::Tensor& out, bool swapped) {
  c10::cuda::CUDAGuard guard(out.device());
  auto* gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState a, b;
  const int64_t inc_a = ht::rng_increment(c10::SymInt(4));
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    a = gen->philox_cuda_state(inc_a);
  }
  const int64_t inc_b = ht::rng_increment(c10::SymInt(8));
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    b = gen->philox_cuda_state(inc_b);
  }
  const c10::SymInt p = ht::sym_mutable_data_ptr(out);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  TracedPhilox pa(a), pb(b);
  if (swapped) {
    ht::launch(write_offset, 1, 1, 0, stream, p + 8, pb);
    ht::launch(write_offset, 1, 1, 0, stream, p, pa);
  } else {
    ht::launch(write_offset, 1, 1, 0, stream, p, pa);
    ht::launch(write_offset, 1, 1, 0, stream, p + 8, pb);
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("two_states", &two_states);
}
"""


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceRngPairing(TestCase):
    # provenance audit S1: an rng_increment declaration is paired with the
    # launch that carries the state it produced, checked against the
    # generator's own intragraph offset, not only by host order
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import os

        from torch.utils.cpp_extension import CUDA_HOME, load_inline

        nvcc = CUDA_HOME and os.path.join(CUDA_HOME, "bin", "nvcc")
        if not nvcc or not os.path.isfile(nvcc):
            raise unittest.SkipTest("requires nvcc for the test extension")
        cls.extension = load_inline(
            "hosttrace_rng_two_states",
            cpp_sources="",
            cuda_sources=_TWO_STATES_SOURCE,
            functions=None,
            with_cuda=True,
            extra_cflags=["-std=c++20"],
            extra_cuda_cflags=["-std=c++20"],
        )

    def test_a_launch_carrying_another_declarations_state_declines(self):
        def host(out):
            return self.extension.two_states(out, True)

        out = torch.zeros(2, dtype=torch.int64, device="cuda")
        with self.assertRaisesRegex(
            ht.Declined, "philox state of another rng_increment declaration"
        ):
            ht.trace(host, (out,))
        self.assertFalse(torch._C._host_trace_tracing())

    def test_declarations_paired_in_order_replay_eager_offsets(self):
        def host(out):
            return self.extension.two_states(out, False)

        out = torch.zeros(2, dtype=torch.int64, device="cuda")
        tape = ht.trace(host, (out,))
        for j in range(2):
            params = tape.launches[j]["params"]
            kinds = {p["name"]: (p["kind"], p["size"]) for p in params if p["name"]}
            self.assertEqual(kinds["philox_offset_intragraph"], ("u64", 8))
            self.assertEqual(kinds["philox_captured"], ("u8", 1))
            # the exempt `rng` bytes: the seed / offset pointers and the tail
            # after the flag; the intragraph offset is byte-checked
            self.assertEqual(
                sorted(p["size"] for p in params if p["kind"] == "rng"), [7, 16]
            )
        variant = build(tape, host, (out,))
        ref, ref_offsets = _sequence(host, (out,), 3)
        got, offsets = _sequence(_replayer(variant), (out,), 3)
        self.assertEqual(offsets, ref_offsets)
        for g, r in zip(got, ref):
            self.assertEqual(g[0].tolist(), r[0].tolist())
        # the second kernel draws where the first declaration stopped
        self.assertEqual(got[0][0][1].item() - got[0][0][0].item(), 4)


# ---- the ten distributions of ATen's distribution template (E43): the entry on the same
# tensor as the op, per case, and the dtypes that reach each kernel route
def _distribution_cases():
    C = torch._C
    f = (torch.float32, torch.float16, torch.bfloat16, torch.float64)
    return {
        # random_: the 64-bit route for int64 / double, the 32-bit one otherwise
        "random_": (
            lambda t: t.random_(),
            lambda t: C._host_trace_ti_random(t, None),
            (torch.int64, torch.int32, torch.float32, torch.bool),
        ),
        # random_.from with a range below 2^28: the 32-bit route
        "random_from_to": (
            lambda t: t.random_(3, 100),
            lambda t: C._host_trace_ti_random_from_to(t, 3, 100, None),
            (torch.int64, torch.int32, torch.float32, torch.bfloat16),
        ),
        # random_.from with a range of 2^41: the 64-bit route
        "random_from_to_64": (
            lambda t: t.random_(-(2**40), 2**40),
            lambda t: C._host_trace_ti_random_from_to(t, -(2**40), 2**40, None),
            (torch.int64, torch.float32, torch.float64),
        ),
        # random_.to: random_(self, 0, to)
        "random_to": (
            lambda t: t.random_(7),
            lambda t: C._host_trace_ti_random_from_to(t, 0, 7, None),
            (torch.int64, torch.int32),
        ),
        # random_.from at int64's lowest with no `to`: the full 64-bit kernel
        "random_full_64": (
            lambda t: t.random_(-(2**63), None),
            lambda t: C._host_trace_ti_random_from_to(t, -(2**63), None, None),
            (torch.int64, torch.float32, torch.float64, torch.bfloat16),
        ),
        "uniform_": (
            lambda t: t.uniform_(-1.0, 2.0),
            lambda t: C._host_trace_ti_uniform(t, -1.0, 2.0, None),
            f,
        ),
        "normal_": (
            lambda t: t.normal_(0.5, 2.0),
            lambda t: C._host_trace_ti_normal(t, 0.5, 2.0, None),
            f,
        ),
        "bernoulli_scalar": (
            lambda t: t.bernoulli_(0.3),
            lambda t: C._host_trace_ti_bernoulli_scalar(t, 0.3, None),
            (torch.float32, torch.bool, torch.int64, torch.bfloat16),
        ),
        "exponential_": (
            lambda t: t.exponential_(1.5),
            lambda t: C._host_trace_ti_exponential(t, 1.5, None),
            f,
        ),
        "geometric_": (
            lambda t: t.geometric_(0.3),
            lambda t: C._host_trace_ti_geometric(t, 0.3, None),
            (torch.float32, torch.float16, torch.int64, torch.float64),
        ),
        "cauchy_": (
            lambda t: t.cauchy_(0.0, 1.0),
            lambda t: C._host_trace_ti_cauchy(t, 0.0, 1.0, None),
            f,
        ),
        "log_normal_": (
            lambda t: t.log_normal_(1.0, 2.0),
            lambda t: C._host_trace_ti_log_normal(t, 1.0, 2.0, None),
            f,
        ),
    }


def _bernoulli_tensor_case(dtype, shape=(4, 4096)):
    # bernoulli_ with a probability tensor of the type the host casts to (float, double for a
    # double self): CUDA_tensor_apply2's kernel, the philox state a member of its functor
    p = torch.rand(
        *shape,
        device="cuda",
        dtype=torch.float64 if dtype is torch.float64 else torch.float32,
    )
    return (
        lambda t, p: t.bernoulli_(p),
        lambda t, p: torch._C._host_trace_ti_bernoulli_tensor(t, p, None),
        p,
    )


def _filled(shape, dtype):
    if dtype is torch.bool:
        return torch.zeros(*shape, device="cuda", dtype=dtype)
    return torch.randn(*shape, device="cuda").to(dtype)


def inductor_seeds(x):
    # Inductor's seed op (torch/_inductor/ir.py RandomSeeds): aten.randint.low_out over int64's
    # limits into a fresh int64 buffer of one seed per random op
    buf = torch.empty([x.shape[0]], dtype=torch.int64, device="cuda")
    return torch.ops.aten.randint.low_out(-(2**63), 2**63 - 1, [x.shape[0]], out=buf)


def two_fills(x):
    # two distributions in one trace: the second draws from where the first stopped
    a = torch.rand_like(x)
    b = torch.randn_like(x[:, :1024])
    return a, b


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceDistributions(TestCase):
    def _equal_sequences(self, got, ref, offsets, ref_offsets):
        self.assertEqual(offsets, ref_offsets)
        for g, r in zip(got, ref):
            self.assertEqual(len(g), len(r))
            for a, b in zip(g, r):
                self.assertEqual(a, b, atol=0, rtol=0)

    def _trace(self, fn, args):
        tape = ht.trace(fn, args)
        self.assertIsNotNone(tape.rng_increment)
        return tape, build(tape, fn, args)

    @unittest.skipIf(not TEST_CUDA_PYTHON_BINDINGS, "cuda.bindings reads the nodes")
    def test_distribution_entries_replay_eager_function_handles(self):
        # each entry is compiled into eager's Distribution*.cu and launches the
        # template's kernel over the hoisted functors: the entry's node, the
        # tape's launch and the replay's node hold the function handle eager's
        # capture holds (E36), on every dtype route of the ten distributions
        for name, (real, entry, dtypes) in _distribution_cases().items():
            for dtype in dtypes:
                with self.subTest(case=name, dtype=dtype):
                    x = _filled((4, 4096), dtype)
                    eager = assert_eager_function_handles(
                        self, real, (x,), entry, launches=1
                    )
                    self.assertEqual(len(eager), 1)
        for dtype in (torch.float32, torch.bool, torch.float64):
            real, entry, p = _bernoulli_tensor_case(dtype)
            with self.subTest(case="bernoulli_tensor", dtype=dtype):
                x = _filled((4, 4096), dtype)
                eager = assert_eager_function_handles(
                    self, real, (x, p), entry, launches=1
                )
                self.assertEqual(len(eager), 1)

    @unittest.skipIf(not TEST_CUDA_PYTHON_BINDINGS, "cuda.bindings reads the nodes")
    def test_strided_outputs_replay_eager_function_handles(self):
        # a sliced 2-D view takes the strided store (the offset calculator as
        # slots); a transposed dense view coalesces to one dim, a 1-D slice
        # or a column is the trivial store with its own byte stride
        real, entry, _ = _distribution_cases()["uniform_"]
        x = torch.randn(64, 96, device="cuda")
        cases = {
            "sliced 2-D (strided store)": x[:, :48],
            "transposed (coalesced to 1-D)": x.t(),
            "1-D slice (stride 8)": x.flatten()[::2],
            "column (stride 384)": x[:, 3],
        }
        for name, t in cases.items():
            with self.subTest(case=name):
                assert_eager_function_handles(self, real, (t,), entry, launches=1)

    def test_distribution_sequences_match_eager(self):
        # a replay draws what eager draws from the same generator state and
        # leaves the generator where eager leaves it, at the traced shape and
        # at another (the increment follows the element count)
        for name, (real, _, dtypes) in _distribution_cases().items():
            with self.subTest(case=name):
                x = _filled((4, 4096), dtypes[0])
                _, variant = self._trace(real, (x,))
                ref, ref_offsets = _sequence(real, (x,), 4)
                got, offsets = _sequence(_replayer(variant), (x,), 4)
                self._equal_sequences(got, ref, offsets, ref_offsets)
                for shape in ((3, 5000), (1, 8192), (16, 1024)):
                    y = _filled(shape, dtypes[0])
                    ref, ref_offsets = _sequence(real, (y,), 3)
                    got, offsets = _sequence(_replayer(variant), (y,), 3)
                    self._equal_sequences(got, ref, offsets, ref_offsets)

    def test_bernoulli_tensor_sequence_matches_eager(self):
        for dtype in (torch.float32, torch.float64):
            real, _, p = _bernoulli_tensor_case(dtype)
            with self.subTest(dtype=dtype):
                x = _filled((4, 4096), dtype)
                tape, variant = self._trace(real, (x, p))
                self.assertEqual(len(tape.rng_slots), 1)
                ref, ref_offsets = _sequence(real, (x, p), 4)
                got, offsets = _sequence(_replayer(variant), (x, p), 4)
                self._equal_sequences(got, ref, offsets, ref_offsets)
                # the mask is a real mask, not all ones or all zeros
                mean = got[0][0].float().mean().item()
                self.assertTrue(0.05 < mean < 0.95)
                # another shape: the same fixed increment, the grid from the new count
                for shape in ((3, 5000), (16, 1024)):
                    y = _filled(shape, dtype)
                    _, _, q = _bernoulli_tensor_case(dtype, shape)
                    ref, ref_offsets = _sequence(real, (y, q), 3)
                    got, offsets = _sequence(_replayer(variant), (y, q), 3)
                    self._equal_sequences(got, ref, offsets, ref_offsets)

    def test_strided_output_sequences_match_eager(self):
        real, _, _ = _distribution_cases()["normal_"]
        x = torch.randn(64, 96, device="cuda")
        views = {
            "sliced 2-D": lambda t: t[:, : t.shape[1] // 2],
            "slice": lambda t: t.flatten()[::2],
        }
        for name, view in views.items():
            with self.subTest(case=name):
                t = view(x)
                _, variant = self._trace(real, (t,))
                ref, ref_offsets = _sequence(real, (t,), 3)
                got, offsets = _sequence(_replayer(variant), (t,), 3)
                self._equal_sequences(got, ref, offsets, ref_offsets)
                u = view(torch.randn(48, 80, device="cuda"))
                ref, ref_offsets = _sequence(real, (u,), 3)
                got, offsets = _sequence(_replayer(variant), (u,), 3)
                self._equal_sequences(got, ref, offsets, ref_offsets)

    def test_inductor_seed_op_traces_as_eager(self):
        # E43: aten.randint.low_out into a fresh int64 buffer (randint_out's
        # same-size resize_ and random_.from) is one launch of the template's
        # kernel with an rng slot; a replay's seeds are eager's from the same
        # generator state, at the traced count and at another
        x = torch.zeros(8, device="cuda")
        tape, variant = self._trace(inductor_seeds, (x,))
        self.assertEqual(tape.num_launches, 1)
        self.assertEqual(len(tape.rng_slots), 1)
        self.assertIn(
            "distribution_elementwise_grid_stride_kernel", tape.launches[0]["kernel"]
        )
        for n in (8, 16, 3):
            y = torch.zeros(n, device="cuda")
            ref, ref_offsets = _sequence(inductor_seeds, (y,), 3)
            got, offsets = _sequence(_replayer(variant), (y,), 3)
            self._equal_sequences(got, ref, offsets, ref_offsets)
        # the seeds are int64 values over the whole range, not a truncated one
        seeds = got[0][0]
        self.assertEqual(seeds.dtype, torch.int64)
        self.assertTrue((seeds.abs() > 2**32).any().item())

    @unittest.skipIf(not TEST_CUDA_PYTHON_BINDINGS, "cuda.bindings reads the nodes")
    def test_inductor_seed_op_replays_eager_function_handle(self):
        x = torch.zeros(8, device="cuda")
        assert_eager_function_handles(self, inductor_seeds, (x,), launches=1)

    def test_factories_reach_the_entries(self):
        # eager's own CompositeExplicit bodies (rand, randn, randint, the _like
        # forms, bernoulli, normal(float, float, size)) allocate under the
        # trace and end in an entry: one launch each, the sequences eager's
        n = 4096

        def factories(x):
            return (
                torch.rand(x.shape[0], n, device="cuda"),
                torch.randn(x.shape[0], n, device="cuda", dtype=torch.bfloat16),
                torch.randint(0, 10, (x.shape[0], n), device="cuda"),
                torch.rand_like(x),
                torch.randn_like(x),
                torch.randint_like(x, 5),
                torch.bernoulli(torch.full_like(x, 0.25)),
                torch.normal(0.0, 1.0, (x.shape[0], n), device="cuda"),
            )

        x = torch.randn(4, n, device="cuda")
        tape, variant = self._trace(factories, (x,))
        self.assertEqual(len(tape.rng_slots), 8)
        ref, ref_offsets = _sequence(factories, (x,), 3)
        got, offsets = _sequence(_replayer(variant), (x,), 3)
        self._equal_sequences(got, ref, offsets, ref_offsets)
        y = torch.randn(2, n, device="cuda")
        ref, ref_offsets = _sequence(factories, (y,), 3)
        got, offsets = _sequence(_replayer(variant), (y,), 3)
        self._equal_sequences(got, ref, offsets, ref_offsets)

    def test_two_distributions_in_one_trace(self):
        # each launch declares its own increment; the second kernel starts at
        # the first's increment, at the traced batch and at another
        x = torch.randn(4, 4096, device="cuda")
        tape, variant = self._trace(two_fills, (x,))
        self.assertEqual(len(tape.rng_slots), 2)
        for B in (4, 8, 2):
            xb = torch.randn(B, 4096, device="cuda")
            ref, ref_offsets = _sequence(two_fills, (xb,), 3)
            got, offsets = _sequence(_replayer(variant), (xb,), 3)
            self._equal_sequences(got, ref, offsets, ref_offsets)

    def test_the_default_generator_passed_explicitly_traces(self):
        x = torch.randn(4, 4096, device="cuda")
        gen = _gen()

        def fn(t):
            return t.uniform_(0.0, 1.0, generator=gen)

        _, variant = self._trace(fn, (x,))
        ref, ref_offsets = _sequence(fn, (x,), 3)
        got, offsets = _sequence(_replayer(variant), (x,), 3)
        self._equal_sequences(got, ref, offsets, ref_offsets)

    def test_another_generator_declines_by_name(self):
        x = torch.randn(4, 4096, device="cuda")
        g = torch.Generator(device="cuda")
        with self.assertRaisesRegex(ht.Declined, "generator other than the default"):
            ht.trace(lambda t: t.normal_(generator=g), (x,))
        self.assertFalse(torch._C._host_trace_tracing())

    def test_distribution_declines_and_refusals(self):
        x = torch.randn(4, 4096, device="cuda")
        with self.assertRaisesRegex(ht.Declined, "complex"):
            ht.trace(lambda t: t.uniform_(), (x.to(torch.complex64),))
        p_cpu = torch.rand(4, 4096)
        with self.assertRaisesRegex(ht.Declined, "cpu tensor operand"):
            ht.trace(lambda t: t.bernoulli_(p_cpu), (x,))
        # eager's own refusals, with eager's texts
        with self.assertRaisesRegex(
            RuntimeError, "expects 'from' to be less than 'to'"
        ):
            ht.trace(lambda t: t.random_(5, 5), (x,))
        with self.assertRaisesRegex(RuntimeError, "expects p to be in"):
            ht.trace(lambda t: t.bernoulli_(1.5), (x,))
        with self.assertRaisesRegex(
            RuntimeError, "more than one element of the written-to tensor"
        ):
            ht.trace(lambda t: t.uniform_(), (x[:1].expand(4, 4096),))
        self.assertFalse(torch._C._host_trace_tracing())

    def test_ordinary_entries_match_the_real_ops(self):
        # outside a trace an entry runs the same kernel with the same launch
        # and the same increment: identical sequences from one seed
        for name, (real, entry, dtypes) in _distribution_cases().items():
            for numel in (1, 3, 256, 257, 4097, 65537):
                with self.subTest(case=name, numel=numel):
                    x = _filled((numel,), dtypes[0])
                    ref, ref_offsets = _sequence(real, (x,), 2)
                    got, offsets = _sequence(entry, (x,), 2)
                    self._equal_sequences(got, ref, offsets, ref_offsets)

    def test_every_case_traces_the_same_program_under_other_hints(self):
        # the recorder never reads a hint: every trace this class makes, made
        # again under other hints, is the same program (host_trace_two_hint)
        two_hint.assert_family(self)


if __name__ == "__main__":
    run_tests()
