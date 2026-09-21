# Owner(s): ["module: inductor"]

import gc
import unittest
from unittest import mock

import triton
import triton.language as tl
from direct_example import example

import torch
from torch._inductor.runtime._cudagraph import direct_host, replay
from torch._inductor.runtime._cudagraph.api import (
    DirectCuTe,
    DirectHost,
    DirectTriton,
    InputContract,
    IntegerRange,
    IntExpr,
    ObservedOrdinaryEntry,
    PythonEntry,
    SignaturePolicy,
    TensorInput,
)
from torch._inductor.runtime._cudagraph.direct_cuda_host import DirectCudaHost
from torch._inductor.runtime._cudagraph.frontend import DirectPhysicalCall
from torch._inductor.runtime.cudagraph_arg_mapping import ParameterSource
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@triton.jit(do_not_specialize=["N"], do_not_specialize_on_alignment=["N"])
def add_one(X, Y, N, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + index, index < N, other=0)
    tl.store(Y + index, value + 1, index < N)


@triton.jit(
    do_not_specialize=["input_address", "owned_bits", "count"],
    do_not_specialize_on_alignment=["input_address", "owned_bits", "count"],
)
def store_words(output, input_address, owned_bits, count):
    tl.store(output, input_address.to(tl.int64))
    tl.store(output + 1, owned_bits.to(tl.int64))
    tl.store(output + 2, count.to(tl.int64))


ADD = RANDOM = CUTE = STORE = None


def one_random(source):
    return torch.native_dropout(source, 0.25, True)


def two_random(source):
    first, mask1 = torch.native_dropout(source, 0.25, True)
    second, mask2 = torch.native_dropout(first, 0.125, True)
    return second, mask1, mask2


def maybe_random(source):
    if source.shape[0] == 7:
        return source.view(source.shape), source.view(source.shape)
    return one_random(source)


def host(box):
    rows, source = box
    box.clear()
    count = rows * 128
    backing = torch.empty_strided(
        (count + 4,), (1,), dtype=source.dtype, device=source.device
    )
    view = backing[2 : count + 2].view(rows, 128)
    block = 128 if rows < 16 else 256
    ADD[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](
        source, view, count, BLOCK=block
    )
    first, *first_masks = RANDOM(view)
    middle = torch.empty_like(first)
    if CUTE is None:
        ADD[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](
            first, middle, count, BLOCK=block
        )
    else:
        CUTE(first, middle, rows)
    second, *second_masks = RANDOM(middle)
    output = torch.empty_like(second)
    ADD[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](
        second, output, count, BLOCK=block
    )
    return (output, *first_masks, *second_masks)


def host_late(box):
    rows, source = box
    result = host(box)
    owned = torch.empty((rows * 128,), dtype=source.dtype, device=source.device)
    words = torch.empty((3,), dtype=torch.int64, device=source.device)
    STORE[(1,)](words, source.data_ptr(), owned.data_ptr() % (1 << 31), rows)
    return (*result, owned, words)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.hip is None
    and _get_torch_cuda_version() >= (12, 8),
    "requires NVIDIA CUDA >= 12.8",
)
class TestMixedRng(TestCase):
    def setup_host(self, device, entry=one_random, cute=False):
        self.enterContext(torch.cuda.device(device))
        self.enterContext(torch.random.fork_rng(devices=[torch.cuda.current_device()]))
        add = DirectTriton(add_one)
        self.addCleanup(add.close)
        cute_call = None
        if cute:
            owner = ObservedOrdinaryEntry(
                PythonEntry(example.launch_affine),
                example.affine,
                policy=SignaturePolicy(32, 64, 16, "stream"),
                conversion=example.convert_arguments,
            )
            self.addCleanup(owner.close)
            cute_call = DirectCuTe(owner)
        self.enterContext(
            mock.patch.dict(
                globals(), ADD=add, RANDOM=DirectCudaHost(entry), CUTE=cute_call
            )
        )
        rows = IntExpr("boxed", 0)
        return InputContract(
            ("integer", "tensor"),
            (TensorInput(1, torch.float32, (rows, 128), (128, 1)),),
            (IntegerRange(0, 2, 127),),
            torch.cuda.current_device(),
        )

    def sequence(self, device, entry=one_random, cute=False):
        contract = self.setup_host(device, entry, cute)
        runner = DirectHost(host, contract)
        self.addCleanup(runner.close)
        sequence = (5, 5, 7, 7, 35, 35, 5)
        samples = [torch.randn(rows, 128, device=device) for rows in sequence]
        self.assertEqual(len({value.data_ptr() for value in samples}), len(samples))
        generator = torch.cuda.default_generators[torch.cuda.current_device()]
        generator.manual_seed(1234)
        generator.set_offset(64)
        initial = generator.get_state()
        expected, offsets = [], []
        for rows, source in zip(sequence, samples, strict=True):
            expected.append(host([rows, source]))
            offsets.append(generator.get_offset())
        final = generator.get_state()
        generator.set_state(initial)
        actual = []
        for index, rows in enumerate(sequence):
            box = [rows, samples[index]]
            if index in (1, 3, 5, 6):
                with mock.patch.object(
                    direct_host,
                    "_observe_direct",
                    side_effect=AssertionError("native hit retraced"),
                ):
                    actual.append(runner.entry(box))
            else:
                actual.append(runner(box))
            self.assertEqual(box, [])
            self.assertEqual(generator.get_offset(), offsets[index])
            self.assertIsNotNone(runner.entry)
            samples[index] = None
            gc.collect()
            torch.empty((rows * 128 + 4,), device=device).fill_(99)
        self.assertEqual(generator.get_state(), final)
        self.assertGreaterEqual(len(runner.variants), 2)
        torch.cuda.synchronize(device)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        for variant in runner.variants:
            calls = [
                event
                for event in variant.program.events
                if type(event) is DirectPhysicalCall
            ]
            if calls:
                self.assertIsNotNone(variant.program.rng)
        runner.close()
        self.assertEqual(actual, expected, atol=0, rtol=0)

    @parametrize("two_stage", (False, True))
    def test_sequence_matches_ordinary_on_misses_and_hits(self, device, two_stage):
        self.sequence(device, two_random if two_stage else one_random)

    def test_zero_launch_branch_preserves_rng_position(self, device):
        self.sequence(device, maybe_random)

    def test_cute_between_random_invocations(self, device):
        self.sequence(device, cute=True)

    def test_rng_precedes_late_address_parameters(self, device):
        contract = self.setup_host(device)
        store = DirectTriton(store_words)
        self.addCleanup(store.close)
        self.enterContext(mock.patch.dict(globals(), STORE=store))
        runner = DirectHost(host_late, contract)
        self.addCleanup(runner.close)
        prepared = []
        original_prepare = replay._make_replay

        def inspect(*args, **kwargs):
            self.assertIsNotNone(kwargs["rng"])
            late = [
                argument.source
                for call in args[5]
                for argument in getattr(call, "arguments", ())
                if type(argument.source) is ParameterSource
            ]
            self.assertEqual({value.width for value in late}, {32, 64})
            self.assertEqual(len(late), 2)
            entry = original_prepare(*args, **kwargs)
            prepared.append(entry)
            return entry

        self.enterContext(
            mock.patch.object(replay, "_make_replay", side_effect=inspect)
        )
        sequence = (5, 5, 35, 35, 5)
        samples = [torch.randn(rows, 128, device=device) for rows in sequence]
        generator = torch.cuda.default_generators[torch.cuda.current_device()]
        generator.manual_seed(2718)
        generator.set_offset(64)
        initial = generator.get_state()
        expected = [
            host([rows, source]) for rows, source in zip(sequence, samples, strict=True)
        ]
        final = generator.get_state()
        generator.set_state(initial)
        outputs = []
        for index, (rows, source) in enumerate(zip(sequence, samples, strict=True)):
            box = [rows, source]
            if index in (1, 3, 4):
                with mock.patch.object(
                    direct_host,
                    "_observe_direct",
                    side_effect=AssertionError("native hit retraced"),
                ):
                    *result, owned, words = runner.entry(box)
            else:
                *result, owned, words = runner(box)
            self.assertEqual(box, [])
            self.assertEqual(
                words,
                torch.tensor(
                    (source.data_ptr(), owned.data_ptr() % (1 << 31), rows),
                    dtype=torch.int64,
                    device=device,
                ),
            )
            outputs.append(tuple(result))
        self.assertEqual(generator.get_state(), final)
        self.assertEqual(outputs, expected, atol=0, rtol=0)
        self.assertEqual(len(prepared), 2)

    def test_two_preparations_retain_independent_capture_state(self, device):
        contract = self.setup_host(device, two_random)
        example = torch.randn(5, 128, device=device)
        variant = direct_host.prepare_direct(host, contract, [5, example])
        self.addCleanup(variant.program.close)
        self.addCleanup(variant.entry.close)
        originals = tuple(
            (
                event.owner,
                event.owner.lowered.calls,
                tuple(call.constants for call in event.owner.lowered.calls),
                tuple(
                    bytes(launch["hint_image"]) for launch in event.owner.tape.launches
                ),
            )
            for event in variant.program.events
            if type(event) is DirectPhysicalCall
        )
        second = replay.prepare_terminal(variant.program, [5, example])
        self.addCleanup(second.close)
        generator = torch.cuda.default_generators[torch.cuda.current_device()]
        samples = [torch.randn_like(example) for _ in range(4)]
        generator.manual_seed(909)
        generator.set_offset(128)
        initial = generator.get_state()
        expected = [host([5, sample]) for sample in samples]
        final = generator.get_state()
        generator.set_state(initial)
        actual = [variant.entry([5, samples[0]]), second([5, samples[1]])]
        variant.entry.close()
        actual.extend(second([5, sample]) for sample in samples[2:])
        self.assertEqual(generator.get_state(), final)
        torch.cuda.synchronize(device)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        for owner, calls, constants, images in originals:
            self.assertIs(owner.lowered.calls, calls)
            self.assertEqual(tuple(call.constants for call in calls), constants)
            self.assertEqual(
                tuple(bytes(launch["hint_image"]) for launch in owner.tape.launches),
                images,
            )


instantiate_device_type_tests(TestMixedRng, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
