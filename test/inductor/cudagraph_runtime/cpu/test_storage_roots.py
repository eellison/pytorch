"""Displaced address values share storage liveness without losing their offsets."""

from types import SimpleNamespace


import torch
from torch._inductor.runtime._cudagraph._compiler.host_program import Allocate
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch._inductor.runtime._cudagraph.frontend import DirectKernelCall
from torch._inductor.runtime._cudagraph.replay import _CaptureCall, _release_steps
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource, CallArgument, InputSource, IntegerSource, IntExpr, OwnedBuffer,
    ParameterSource, PointerSource, storage_roots,
)
from torch._inductor.runtime.cudagraph_boxed_replay import _BoundCall, _PhysicalCall, _PhysicalField
from torch.testing._internal.common_utils import run_tests, TestCase


class TestStorageRoots(TestCase):
    def test_displaced_addresses_preserve_one_final_storage_use(self):
        owned = OwnedBuffer(BufferSource("owned"), torch.float32, (8,), (1,))
        output = OwnedBuffer(BufferSource("output"), torch.int64, (1,), (1,))
        first = PointerSource(InputSource(0), IntExpr("constant", 0))
        displaced = PointerSource(InputSource(0), IntExpr("constant", 12))
        independent = PointerSource(owned.source, IntExpr("constant", 4))
        difference = ParameterSource("sub", 64, args=(
            ParameterSource("pointer", 64, displaced),
            ParameterSource("pointer", 64, first),
        ))
        address = ParameterSource("add", 64, args=(
            ParameterSource("pointer", 64, independent), difference,
        ))
        scalar = ParameterSource("constant", 64, 17)
        self.assertEqual(address.pointers, (independent, displaced, first))
        self.assertEqual(storage_roots(address), (owned.source, InputSource(0)))
        self.assertEqual(storage_roots(scalar), ())
        self.assertEqual(storage_roots(IntegerSource(17)), ())
        self.assertEqual(address.pointers, (independent, displaced, first))
        self.assertIs(difference.args[0].value, displaced)
        self.assertIs(difference.args[1].value, first)

        grid = (IntExpr("constant", 1),) * 3
        arguments = (
            CallArgument("input", 0, 0, "*fp32", first),
            CallArgument("owned", 1, 1, "*fp32", owned.source),
        )
        physical = _PhysicalCall((
            _PhysicalField(0, 0, "i64", address),
            _PhysicalField(1, 0, "i64", scalar),
            _PhysicalField(2, 0, "pointer", output.source),
        ), None, grid, ())
        calls = tuple(_CaptureCall(bound, 1, (1, 1, 1), (), ()) for bound in (
            _BoundCall(arguments, None, grid), physical,
        ))
        program = SimpleNamespace(
            saved_input_indices=(0, 1), allocations=(owned, output), outputs=(output,),
            events=(
                Allocate("owned", torch.float32, (8,), (1,)), DirectKernelCall(None, arguments, grid),
                Allocate("output", torch.int64, (1,), (1,)), CuTeCall(physical, (), (), None),
            ),
        )
        self.assertEqual(_release_steps(program, calls), (
            ("drop", 1), ("allocate", 0), ("kernel", 0),
            ("allocate", 1), ("kernel", 1), ("drop", 0),
        ))


if __name__ == "__main__":
    run_tests()
