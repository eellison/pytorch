# Owner(s): ["module: inductor"]

import unittest

import torch
from torch._inductor.runtime._cudagraph.host_trace import HostTraceReplay

from torch._inductor.runtime.cudagraph_arg_mapping import ExpressionSource, IntExpr
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


_HOST_TRACE_SUPPORTED = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and _get_torch_cuda_version() >= (12, 8)
    and hasattr(torch._C, "_CUDAGraphCompiledEvaluation")
    and hasattr(torch._C, "_host_trace_ti_sum")
)


def _sum_last(tensor):
    return torch.sum(tensor, -1)


def _fixed_and_dynamic(tensor):
    fixed = torch.empty((4, 4), device=tensor.device, dtype=tensor.dtype)
    fixed.copy_(tensor[:4, :4])
    return torch.sum(tensor, -1), torch.sum(fixed, -1)


@unittest.skipUnless(
    _HOST_TRACE_SUPPORTED,
    "requires NVIDIA CUDA >= 12.8 and reduction host tracing",
)
class TestCudaHostTraceNonMemsetReductionReplay(TestCase):
    def _input(self, device, shape, dtype=torch.bfloat16):
        return torch.randn(shape, device=device, dtype=dtype)

    def _start(self, fn, args):
        replay = HostTraceReplay(fn)
        self.addCleanup(replay.close)
        self._check(replay, args)
        self.assertEqual(len(replay.variants), 1)
        self.assertIsInstance(replay.entry, torch._C._CUDAGraphBoxedDispatch)
        return replay

    def _check(self, replay, args):
        expected = replay.fn(*args)
        box = list(args)
        actual = replay(box)
        self.assertEqual(box, [])
        expected = expected if isinstance(expected, tuple) else (expected,)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        for variant in replay.variants:
            self.assertEqual(variant.program.tape.memsets, [])
        for output, reference in zip(actual, expected, strict=True):
            self.assertEqual(output.shape, reference.shape)
            self.assertEqual(output.stride(), reference.stride())
        return actual

    def _values(self, variant, args, expressions):
        numeric = _NumericProgram(variant.program.records, variant.family.box(args))
        return tuple(
            numeric.values[numeric.add(expression)]
            if type(expression) is IntExpr
            else expression
            for expression in expressions
        )

    @parametrize("dtype", (torch.bfloat16, torch.float32))
    def test_sum_serves_new_shapes_and_addresses(self, device, dtype):
        original = self._input(device, (64, 4096), dtype)
        replay = self._start(_sum_last, (original,))
        program = replay.variants[0].program
        self.assertEqual(len(program.calls), 1)
        self.assertEqual(program.tape.memsets, [])
        served = 0
        for shape in (
            (64, 4096),
            (32, 4096),
            (128, 4096),
            (48, 3000),
            (64, 2048),
            (1024, 512),
            (16, 4096),
        ):
            tensor = self._input(device, shape, dtype)
            self.assertNotEqual(tensor.data_ptr(), original.data_ptr())
            before = len(replay.variants)
            self._check(replay, (tensor,))
            served += len(replay.variants) == before
            after = len(replay.variants)
            self._check(replay, (self._input(device, shape, dtype),))
            self.assertEqual(len(replay.variants), after)
        self.assertGreaterEqual(served, 5)

    def test_nonconstant_block_changes_within_one_variant(self, device):
        original = self._input(device, (8, 4096))
        replay = self._start(_sum_last, (original,))
        program = replay.variants[0].program
        (call,) = (call for call in program.calls)
        self.assertIsNotNone(call.block)
        self.assertTrue(any(axis.op != "constant" for axis in call.block))
        before = self._values(replay.variants[0], (original,), call.block)
        self.assertEqual(before, call.module.block)
        changed = self._input(device, (4, 4096))
        after = self._values(replay.variants[0], (changed,), call.block)
        self.assertNotEqual(before, after)
        self._check(replay, (changed,))
        self.assertEqual(len(replay.variants), 1)
        self._check(replay, (original,))
        self.assertEqual(len(replay.variants), 1)

    @parametrize("operation", ("mean", "amax", "sum_first"))
    @parametrize("shape", ((48, 3000), (64, 2048), (16, 4096)))
    def test_other_reductions(self, device, operation, shape):
        fn = {
            "mean": lambda tensor: torch.mean(tensor, -1),
            "amax": lambda tensor: torch.amax(tensor, -1),
            "sum_first": lambda tensor: torch.sum(tensor, 0),
        }[operation]
        replay = self._start(fn, (self._input(device, (64, 4096)),))
        self._check(replay, (self._input(device, shape),))
        variants = len(replay.variants)
        self._check(replay, (self._input(device, shape),))
        self.assertEqual(len(replay.variants), variants)

    def test_dtype_and_rank_misses_create_reusable_variants(self, device):
        original = self._input(device, (64, 4096))
        replay = self._start(_sum_last, (original,))
        changed = (
            self._input(device, (64, 4096), torch.float16),
            self._input(device, (4, 64, 4096)),
        )
        for variants, tensor in enumerate(changed, 2):
            self._check(replay, (tensor,))
            self.assertEqual(len(replay.variants), variants)
            self._check(replay, (tensor,))
            self._check(replay, (original,))
            self.assertEqual(len(replay.variants), variants)

    @parametrize("shape", ((32, 4096), (48, 3000), (16, 4096)))
    def test_agrees_with_interim_replay_and_eager(self, device, shape):
        from torch.cuda import _host_trace

        original = self._input(device, (64, 4096))
        replay = self._start(_sum_last, (original,))
        tape = replay.variants[0].program.tape
        interim = _host_trace.build(tape, _sum_last, (original,))
        tensor = self._input(device, shape)
        (actual,) = self._check(replay, (tensor,))
        self.assertEqual(actual, interim.replay((tensor,))[0], atol=0, rtol=0)

    def test_reduction_mutation_occurs_once_on_miss_and_reuse(self, device):
        def update(dst, src):
            reduced = torch.mean(src, -1)
            dst.copy_((dst + reduced[:, None]).t())
            return dst

        replay = HostTraceReplay(update)
        self.addCleanup(replay.close)
        original = (
            self._input(device, (8, 8), torch.float32),
            torch.full((8, 4096), 0.25, device=device, dtype=torch.float32),
        )
        changed = (
            self._input(device, (12, 12), torch.float16),
            torch.full((12, 4096), 0.5, device=device, dtype=torch.float16),
        )
        for args, variants in (
            (original, 1),
            (original, 1),
            (changed, 2),
            (changed, 2),
            (original, 2),
        ):
            dst, src = args
            expected = (dst + torch.mean(src, -1)[:, None]).t().contiguous()
            source_before = src.clone()
            pointer = dst.data_ptr()
            box = list(args)
            (actual,) = replay(box)
            self.assertEqual(box, [])
            self.assertEqual(actual.data_ptr(), pointer)
            self.assertEqual(actual, expected, atol=0, rtol=0)
            self.assertEqual(dst, expected, atol=0, rtol=0)
            self.assertEqual(src, source_before, atol=0, rtol=0)
            self.assertEqual(len(replay.variants), variants)

    @parametrize("shape", ((4, 2048), (12, 3072)))
    def test_scalar_fields_and_fixed_intermediates(self, device, shape):
        original = self._input(device, (8, 4096), torch.float32)
        replay = self._start(_fixed_and_dynamic, (original,))
        program = replay.variants[0].program
        calls = program.calls
        self.assertEqual(len(calls), 3)
        fixed_sizes = [
            tuple(size if type(size) is int else size.value for size in allocation.size)
            for allocation in program.allocations
            if all(
                type(size) is int or (type(size) is IntExpr and size.op == "constant")
                for size in allocation.size
            )
        ]
        if program.arena is None:
            self.assertIn((4, 4), fixed_sizes)
        else:
            fixed = [
                rec
                for rec in program.tape.allocs
                if tuple(
                    size.node.expr if isinstance(size, torch.SymInt) else size
                    for size in rec.sizes
                )
                == (4, 4)
            ]
            self.assertEqual(len(fixed), 1)
            block = program.arena.blocks[fixed[0].name]
            self.assertEqual(block.size, 16 * original.element_size())
        self.assertIn((4,), fixed_sizes)
        self.assertTrue(
            any(
                any(
                    type(size) is IntExpr and size.op != "constant"
                    for size in allocation.size
                )
                for allocation in program.allocations
            )
        )
        scalar_expressions = {
            id(field.source.expression): field.source.expression
            for call in calls
            for field in call.fields
            if field.kind != "pointer" and type(field.source) is ExpressionSource
        }
        expressions = tuple(scalar_expressions.values())
        changed = self._input(device, shape, torch.float32)
        self.assertNotEqual(changed.data_ptr(), original.data_ptr())
        before = self._values(replay.variants[0], (original,), expressions)
        after = self._values(replay.variants[0], (changed,), expressions)
        self.assertGreaterEqual(
            sum(a != b for a, b in zip(before, after, strict=True)), 2
        )
        self._check(replay, (changed,))
        variants = len(replay.variants)
        inputs = [self._input(device, shape, torch.float32) for _ in range(4)]
        expected = [_fixed_and_dynamic(tensor) for tensor in inputs]
        outputs = [replay([tensor]) for tensor in inputs]
        self.assertEqual(outputs, expected, atol=0, rtol=0)
        self.assertEqual(len(replay.variants), variants)
        pointers = [tensor.data_ptr() for result in outputs for tensor in result]
        self.assertEqual(len(set(pointers)), len(pointers))
        replay.close()
        self.assertEqual(outputs, expected, atol=0, rtol=0)


instantiate_device_type_tests(
    TestCudaHostTraceNonMemsetReductionReplay, globals(), only_for="cuda"
)

if __name__ == "__main__":
    run_tests()
