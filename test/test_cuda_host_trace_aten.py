# Owner(s): ["module: cuda graphs"]

import unittest

import torch
import torch.nn.functional as F
from torch.cuda._host_trace_capture import capture_kernel_nodes, KernelNode
from torch.cuda._host_trace_launch import KernelLaunch
from torch.cuda._host_trace_replay import HostTraceReplay
from torch.cuda._host_trace_tape import _hint, EagerCall, trace
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda_python_bindings,
    run_tests,
    TEST_CUDA,
    TestCase,
)


def silu_mul(x, y):
    return F.silu(x) * y


def _inputs(m, h, dtype, broadcast):
    x = torch.randn(m, h, device="cuda", dtype=dtype)
    y = torch.randn(h if broadcast else (m, h), device="cuda", dtype=dtype)
    return x, y


def _unread(node: KernelNode) -> set[tuple[int, int]]:
    # (param, byte) eager leaves uninitialized and the kernel never reads:
    # StridedOp<f, 3>'s OffsetCalculator entries past dims, and its empty
    # functor with the tail padding
    if "vectorized" in node.name or "unrolled" in node.name:
        return set()
    image = node.images[1]
    dims = int.from_bytes(image[24:28], "little")
    sizes, strides = 28 + 12 * dims, 328 + 12 * dims
    skipped = [*range(sizes, 328), *range(strides, 628), *range(628, len(image))]
    return {(1, b) for b in skipped}


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
@unittest.skipIf(not hasattr(torch._C, "_cuda_hostTraceAten"), "needs traced hosts")
class TestHostTraceAten(TestCase):
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("broadcast", [False, True])
    def test_replays_new_shapes(self, dtype, broadcast):
        entry = HostTraceReplay(silu_mul)
        for h in (4096, 768):
            for m in (64, 200, 7, 1):
                x, y = _inputs(m, h, dtype, broadcast)
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                out = entry(x, y)
                torch.cuda.synchronize()
                replay_peak = torch.cuda.max_memory_allocated() - base
                self.assertEqual(out, silu_mul(x, y), atol=0, rtol=0)
                del out
                torch.cuda.reset_peak_memory_stats()
                silu_mul(x, y)
                torch.cuda.synchronize()
                self.assertEqual(replay_peak, torch.cuda.max_memory_allocated() - base)
        # one variant for m > 1, one for m == 1
        self.assertEqual(entry.traces, 2)
        self.assertEqual(entry.eager, 0)
        for v in entry.variants:
            self.assertTrue(all(isinstance(s, range) for s in v.captured.lowered.steps))

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("broadcast", [False, True])
    @parametrize("m", [64, 7, 1])
    def test_records_match_eager(self, dtype, broadcast, m):
        x, y = _inputs(m, 4096, dtype, broadcast)
        tape = trace(silu_mul, (x, y))
        launches = [c for _, c in tape.launches]
        self.assertTrue(all(isinstance(c, KernelLaunch) for c in launches))
        nodes = capture_kernel_nodes(lambda s: silu_mul(x, y))
        self.assertEqual(len(launches), len(nodes))
        for launch, node in zip(launches, nodes):
            self.assertEqual(launch.function, node.function)
            self.assertEqual(tuple(int(_hint(g)) for g in launch.grid), node.grid)
            self.assertEqual((launch.block, launch.smem), (node.block, node.smem))
            self.assertEqual(tuple(launch.layout), tuple(node.layout))
            declared = _unread(node)
            fields = zip(launch.fields, launch.slots)
            for i, ((param, at, width), v) in enumerate(fields):
                declared |= {(param, b) for b in range(at, at + width)}
                if i not in launch.pointers:
                    eager = node.images[param][at : at + width]
                    eager = int.from_bytes(eager, "little", signed=True)
                    self.assertEqual(int(_hint(v)), eager)
            for p, images in enumerate(zip(launch.images, node.images)):
                pairs = enumerate(zip(*images))
                diff = [b for b, (o, e) in pairs if o != e and (p, b) not in declared]
                self.assertEqual(diff, [], msg=f"{launch.name} param {p}")

    def test_decline_is_an_eager_call(self):
        x = torch.randn(64, 4096, device="cuda", dtype=torch.float16)
        y = torch.randn(4096, device="cuda", dtype=torch.float32)
        tape = trace(silu_mul, (x, y))
        calls = [c for _, c in tape.launches]
        self.assertIsInstance(calls[0], KernelLaunch)
        self.assertIsInstance(calls[1], EagerCall)
        entry = HostTraceReplay(silu_mul)
        entry(x, y)
        self.assertEqual(entry(x, y), silu_mul(x, y), atol=0, rtol=0)


instantiate_parametrized_tests(TestHostTraceAten)

if __name__ == "__main__":
    run_tests()
