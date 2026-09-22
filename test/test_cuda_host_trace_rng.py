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
        with self.assertRaisesRegex(ht.Declined, "rand"):
            ht.trace(lambda t: t + torch.rand_like(t), (x,))
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
#include <ATen/cuda/host_trace/Field.h>
#include <ATen/cuda/host_trace/Launch.h>
#include <ATen/cuda/host_trace/Recorder.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <mutex>
namespace ht = at::cuda::host_trace;

// the philox state as a kernel argument, recorded as an `rng` field (the
// sibling's TracedPhilox in Dropout.cu)
struct TracedPhilox : ht::TracedBase {
  at::PhiloxCudaState pod;
  ht::BytesField<0, sizeof(at::PhiloxCudaState)> bytes{this, "philox_args"};
  explicit TracedPhilox(const at::PhiloxCudaState& st)
      : TracedBase(&pod, sizeof(at::PhiloxCudaState)), pod() {
    new (static_cast<void*>(bytes)) at::PhiloxCudaState(st);
  }
};

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


if __name__ == "__main__":
    run_tests()
