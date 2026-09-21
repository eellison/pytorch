# Owner(s): ["module: inductor"]

import unittest

import torch
import torch.nn.functional as F
from torch._inductor.runtime._cudagraph.host_trace import HostTraceReplay
from torch._inductor.runtime.cudagraph_arg_mapping import (
    ExpressionSource,
    IntExpr,
    ParameterSource,
)
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


_HOST_TRACE_SUPPORTED = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and _get_torch_cuda_version() >= (12, 8)
    and hasattr(torch._C, "_CUDAGraphBoxedDispatch")
    and hasattr(torch._C, "_CUDAGraphCompiledEvaluation")
    and hasattr(torch._C, "_host_trace_ti_add")
)


def _walk(expression):
    if isinstance(expression, ExpressionSource):
        expression = expression.expression
    yield expression
    if isinstance(expression, ParameterSource) and isinstance(
        expression.value, IntExpr
    ):
        yield from _walk(expression.value)
    for argument in expression.args:
        yield from _walk(argument)


@unittest.skipUnless(
    _HOST_TRACE_SUPPORTED,
    "requires NVIDIA CUDA >= 12.8 and TensorIterator host tracing",
)
class TestCudaHostTraceTIReplay(TestCase):
    def _pair(self, device, shape, *, dtype=torch.bfloat16, offset=0):
        rows, columns = shape
        tensors = []
        for _ in range(2):
            storage = torch.randn(rows * columns + offset, device=device, dtype=dtype)
            tensors.append(storage[offset:].view(rows, columns))
        return tuple(tensors)

    def _start(self, fn, args, *, arena=None):
        replay = HostTraceReplay(fn, arena=arena)
        self.addCleanup(replay.close)
        self._check(replay, args)
        self.assertEqual(len(replay.variants), 1)
        self.assertIsInstance(replay.entry, torch._C._CUDAGraphBoxedDispatch)
        return replay

    def _check(self, replay, args):
        expected = replay.fn(*args)
        box = list(args)
        (actual,) = replay(box)
        self.assertEqual(box, [])
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(actual.stride(), expected.stride())
        return actual

    @parametrize("shape", ((48, 4096), (64, 2048), (7, 1000), (1024, 1024), (3, 8)))
    def test_add_serves_new_shapes_and_addresses(self, device, shape):
        original = self._pair(device, (64, 4096))
        replay = self._start(torch.add, original)
        program = replay.variants[0].program
        self.assertEqual(len(program.calls), 1)
        self.assertEqual(len(program.allocations), 1)
        args = self._pair(device, shape)
        self.assertTrue(
            all(a.data_ptr() != b.data_ptr() for a, b in zip(args, original))
        )
        self._check(replay, args)
        self._check(replay, self._pair(device, shape))
        self.assertEqual(len(replay.variants), 1)

    @parametrize(
        "shape",
        ((32, 4096), (128, 4096), (48, 3000), (33, 2049), (97, 1009), (64, 128)),
    )
    def test_broadcast_divider_rebinds(self, device, shape):
        x, _ = self._pair(device, (64, 4096))
        b = torch.randn(4096, device=device, dtype=x.dtype)
        replay = self._start(torch.add, (x, b))
        program = replay.variants[0].program
        opaque = program.tape.opaque
        self.assertTrue(opaque)
        self.assertEqual({record["kind"] for record in opaque}, {"rebind"})
        self.assertEqual(
            {record["fn"] for record in opaque}, {"intdivider_m1", "intdivider_shift"}
        )
        operations = {
            expression.op
            for call in program.calls
            for field in call.fields
            if field.kind != "pointer"
            for expression in _walk(field.source)
        }
        # the host's opaque functions are plan calls (`pcall`, the pointer ABI; `call`
        # is the vector ABI a frontend's callbacks keep)
        self.assertTrue({"call", "pcall"} & operations)
        x, _ = self._pair(device, shape)
        b = torch.randn(shape[1], device=device, dtype=x.dtype)
        self._check(replay, (x, b))
        self._check(replay, (x, b))
        self.assertEqual(len(replay.variants), 1)

    def test_alignment_miss_creates_reusable_variant(self, device):
        original = self._pair(device, (64, 4096), offset=8)
        replay = self._start(torch.add, original)
        self._check(replay, self._pair(device, (64, 4096), offset=16))
        self.assertEqual(len(replay.variants), 1)
        self._check(replay, self._pair(device, (64, 4096), offset=4))
        self.assertEqual(len(replay.variants), 2)
        self._check(replay, self._pair(device, (64, 4096), offset=12))
        self._check(replay, original)
        self.assertEqual(len(replay.variants), 2)

    def test_integer_alpha_rounds_once(self, device):
        alpha = 2**62 + 2**38 + 1

        def fn(a, b):
            return torch.add(a, b, alpha=alpha)

        x = torch.zeros(64, 64, device=device)
        y = torch.ones_like(x)
        self.assertFalse(torch.equal(fn(x, y), torch.add(x, y, alpha=float(alpha))))
        replay = self._start(fn, (x, y))
        for rows in (64, 96):
            a = torch.zeros(rows, 64, device=device)
            b = torch.ones_like(a)
            self.assertNotEqual(a.data_ptr(), x.data_ptr())
            actual = self._check(replay, (a, b))
            self.assertEqual(actual.view(torch.int32), fn(a, b).view(torch.int32))
            self.assertEqual(len(replay.variants), 1)

    @parametrize("operation", ("silu", "gelu", "gelu_tanh", "square"))
    @parametrize("shape", ((48, 3000), (5, 7)))
    def test_unary_operations(self, device, operation, shape):
        functions = {
            "silu": F.silu,
            "gelu": F.gelu,
            "gelu_tanh": lambda tensor: F.gelu(tensor, approximate="tanh"),
            "square": lambda tensor: torch.mul(tensor, tensor),
        }
        replay = self._start(functions[operation], (self._pair(device, (64, 4096))[0],))
        self._check(replay, (self._pair(device, shape)[0],))
        self._check(replay, (self._pair(device, shape)[0],))
        self.assertEqual(len(replay.variants), 1)

    @parametrize("arena", (False, True))
    def test_two_operations_compose(self, device, arena):
        def fn(x, y):
            return F.silu(torch.add(x, y))

        replay = self._start(fn, self._pair(device, (32, 4096)), arena=arena)
        program = replay.variants[0].program
        self.assertEqual(len(program.calls), 2)
        blocks = len(program.arena.blocks) if program.arena is not None else 0
        self.assertEqual(len(program.allocations) + blocks, 2)
        self.assertEqual(blocks, 1 if arena else 0)
        self._check(replay, self._pair(device, (48, 4096)))
        self._check(replay, self._pair(device, (9, 1000)))
        self.assertEqual(len(replay.variants), 1)

    def test_dtype_and_broadcast_misses_retrace_once(self, device):
        replay = self._start(torch.add, self._pair(device, (64, 4096)))
        args = self._pair(device, (64, 4096), dtype=torch.float16)
        self._check(replay, args)
        self.assertEqual(len(replay.variants), 2)
        self._check(replay, args)
        self.assertEqual(len(replay.variants), 2)
        x, _ = self._pair(device, (64, 4096))
        self._check(replay, (x, x[:1]))
        self.assertEqual(len(replay.variants), 3)
        self._check(replay, (x, x[:1]))
        self.assertEqual(len(replay.variants), 3)

    @parametrize("shape", ((48, 4096), (7, 1000), (3, 8)))
    def test_agrees_with_interim_replay(self, device, shape):
        from torch.cuda import _host_trace

        original = self._pair(device, (64, 4096))
        replay = self._start(torch.add, original)
        tape = replay.variants[0].program.tape
        interim = _host_trace.build(tape, torch.add, original)
        args = self._pair(device, shape)
        actual = self._check(replay, args)
        self.assertEqual(actual, interim.replay(args)[0], atol=0, rtol=0)
        self.assertEqual(len(replay.variants), 1)

    @parametrize("next_offset", (4, 8))
    def test_mutation_occurs_once_on_misses_and_reuse(self, device, next_offset):
        def update(dst, src):
            dst.copy_((dst + src).t())
            return dst

        replay = HostTraceReplay(update)
        self.addCleanup(replay.close)
        original = self._pair(device, (64, 64), offset=8)

        def check(args):
            dst, src = args
            expected = (dst + src).t().contiguous()
            src_before = src.clone()
            pointer = dst.data_ptr()
            box = list(args)
            (actual,) = replay(box)
            self.assertEqual(box, [])
            self.assertEqual(actual, expected, atol=0, rtol=0)
            self.assertEqual(dst, expected, atol=0, rtol=0)
            self.assertEqual(src, src_before, atol=0, rtol=0)
            self.assertEqual(actual.data_ptr(), pointer)

        check(original)
        self.assertEqual(len(replay.variants), 1)
        check(original)
        self.assertEqual(len(replay.variants), 1)
        changed = self._pair(device, (96, 96), offset=next_offset)
        check(changed)
        variants = 2 if next_offset == 4 else 1
        self.assertEqual(len(replay.variants), variants)
        check(changed)
        check(original)
        self.assertEqual(len(replay.variants), variants)


instantiate_device_type_tests(TestCudaHostTraceTIReplay, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
