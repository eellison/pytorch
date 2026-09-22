# Owner(s): ["module: inductor"]

import torch
from torch._inductor.runtime._cudagraph._sdk import activate


activate()

import cutlass

from cuda.bindings import driver
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from torch._inductor.runtime._cudagraph.api import (
    DirectCuTe,
    ObservedOrdinaryEntry,
    PythonEntry,
    SignaturePolicy,
)
from torch._inductor.runtime._cudagraph.direct_hosttrace import HostTraceReplay
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@cute.kernel
def affine_grid(source: cute.Tensor, destination: cute.Tensor):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    row = block * 4 + thread // 128
    column = thread % 128
    if row < source.shape[0]:
        destination[row, column] = source[row, column] * 2.0 + 1.0


@cute.jit
def launch_affine_grid(
    source: cute.Tensor, destination: cute.Tensor, stream: driver.CUstream
):
    rows = cutlass.Int32(source.shape[0])
    affine_grid(source, destination).launch(
        grid=((rows + 4 - 1) // 4, 1, 1),
        block=(512, 1, 1),
        smem=0,
        stream=stream,
    )


def convert_arguments(source, destination):
    values = tuple(
        from_dlpack(tensor, assumed_align=16, use_32bit_stride=False)
        for tensor in (source, destination)
    )
    for value in values:
        value.mark_compact_shape_dynamic(0, stride_order=(0, 1), divisibility=1)
    return values


class TestCuTeGridExpressions(TestCase):
    def test_rounded_grid_rebinds_across_boundaries(self, device):
        owner = ObservedOrdinaryEntry(
            PythonEntry(launch_affine_grid),
            affine_grid,
            policy=SignaturePolicy(32, 64, 16, "stream"),
            conversion=convert_arguments,
        )
        self.addCleanup(owner.close)
        entry = DirectCuTe(owner)

        def host(source):
            destination = torch.empty_like(source)
            entry(source, destination)
            return destination

        cases = ((5, 0), (3, 4), (4, 8), (9, 12), (8, 4), (1, 8), (5, 12))
        inputs = [
            torch.randn(rows * 128 + offset, device=device)[offset:].view(rows, 128)
            for rows, offset in cases
        ]
        replay = HostTraceReplay(host, (inputs[0],))
        self.addCleanup(replay.close)
        held = []
        for source in inputs:
            output, expected = replay(source), source * 2.0 + 1.0
            self.assertEqual(output, expected, atol=0, rtol=0)
            self.assertEqual(
                (len(replay.variants), replay.misses, replay.ordinary, replay.declines),
                (1, 0, 0, []),
            )
            held.append((output, expected))
        replay.close()
        for output, expected in held:
            self.assertEqual(output, expected, atol=0, rtol=0)


instantiate_device_type_tests(TestCuTeGridExpressions, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
