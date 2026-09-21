# Owner(s): ["module: inductor"]

"""Build saved-input drops from validated terminal events and bound roots."""

from types import SimpleNamespace

import torch
from torch._inductor.runtime._cudagraph._compiler.host_program import (
    Allocate,
    Normalize,
)
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch._inductor.runtime._cudagraph.frontend import DirectKernelCall, TerminalCall
from torch._inductor.runtime._cudagraph.replay import _CaptureCall, _release_steps
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BorrowedInputOutput,
    BufferSource,
    CallArgument,
    InputSource,
    IntExpr,
    OutputReference,
    OwnedBuffer,
    ParameterSource,
    PointerSource,
    TensorViewOutput,
)
from torch._inductor.runtime.cudagraph_boxed_replay import (
    _BoundCall,
    _PhysicalCall,
    _PhysicalField,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


ZERO = IntExpr("constant", 0)
GRID = (IntExpr("constant", 1),) * 3


@instantiate_parametrized_tests
class TestReleaseSteps(TestCase):
    def test_mixed_calls_keep_late_roots_until_final_use(self):
        first = OwnedBuffer(BufferSource("first"), torch.float32, (8,), (1,))
        second = OwnedBuffer(BufferSource("second"), torch.float32, (8,), (1,))
        args0 = (
            CallArgument("input0", 0, 0, "*fp32", PointerSource(InputSource(0), ZERO)),
            CallArgument(
                "input1",
                1,
                1,
                "*fp32",
                PointerSource(InputSource(1), IntExpr("constant", 4)),
            ),
            CallArgument("input4", 2, 2, "*fp32", InputSource(4)),
            CallArgument("output", 3, 3, "*fp32", first.source),
        )
        encoded0 = ParameterSource(
            "add",
            64,
            args=(
                ParameterSource(
                    "pointer", 64, PointerSource(InputSource(0), IntExpr("constant", 8))
                ),
                ParameterSource("constant", 64, 1),
            ),
        )
        physical = _PhysicalCall(
            (
                _PhysicalField(0, 0, "pointer", PointerSource(InputSource(2), ZERO)),
                _PhysicalField(1, 0, "i64", encoded0),
                _PhysicalField(2, 0, "pointer", PointerSource(second.source, ZERO)),
            ),
            None,
            GRID,
            (),
        )
        encoded1 = ParameterSource(
            "xor",
            64,
            args=(
                ParameterSource("pointer", 64, PointerSource(InputSource(1), ZERO)),
                ParameterSource("constant", 64, 16),
            ),
        )
        args2 = (
            CallArgument("encoded", 0, 0, "i64", encoded1),
            CallArgument("first", 1, 1, "*fp32", first.source),
            CallArgument("second", 2, 2, "*fp32", second.source),
        )
        calls = tuple(
            _CaptureCall(bound, 1, (1, 1, 1), (), ())
            for bound in (
                _BoundCall(args0, None, GRID),
                physical,
                _BoundCall(args2, None, GRID),
            )
        )
        events = (
            Normalize(1, 0),
            Allocate("first", torch.float32, (8,), (1,)),
            DirectKernelCall(None, args0, GRID),
            Allocate("second", torch.float32, (8,), (1,)),
            CuTeCall(physical, (), (), None),
            TerminalCall(None, args2, GRID),
        )
        program = SimpleNamespace(
            saved_input_indices=(0, 1, 2, 3, 4),
            outputs=(second,),
            allocations=(first, second),
            events=events,
        )
        self.assertEqual(
            _release_steps(program, calls),
            (
                ("drop", 3),
                ("allocate", 0),
                ("kernel", 0),
                ("drop", 4),
                ("allocate", 1),
                ("kernel", 1),
                ("drop", 0),
                ("drop", 2),
                ("kernel", 2),
                ("drop", 1),
            ),
        )

    def test_returned_input_and_view_roots_remain_live(self):
        layout = OwnedBuffer(BufferSource("output"), torch.float32, (8,), (1,))
        arguments = tuple(
            CallArgument(
                f"input{index}",
                index,
                index,
                "*fp32",
                PointerSource(InputSource(index), ZERO),
            )
            for index in range(3)
        )
        calls = (_CaptureCall(_BoundCall(arguments, None, GRID), 1, (1, 1, 1), (), ()),)
        program = SimpleNamespace(
            saved_input_indices=(0, 1, 2),
            allocations=(layout,),
            outputs=(
                BorrowedInputOutput(InputSource(0)),
                TensorViewOutput(InputSource(1), (4,), (1,), 1),
                layout,
                OutputReference(0),
                OutputReference(1),
                OutputReference(2),
            ),
            events=(
                Allocate("output", torch.float32, (8,), (1,)),
                DirectKernelCall(None, arguments, GRID),
            ),
        )
        self.assertEqual(
            _release_steps(program, calls),
            (("allocate", 0), ("kernel", 0), ("drop", 2)),
        )

    @parametrize(
        "reason", ("no_candidates", "no_allocations", "no_calls", "only_returned")
    )
    def test_unavailable_release_keeps_existing_path(self, reason):
        layout = OwnedBuffer(BufferSource("output"), torch.float32, (8,), (1,))
        arguments = (
            CallArgument("input", 0, 0, "*fp32", PointerSource(InputSource(0), ZERO)),
        )
        calls = (_CaptureCall(_BoundCall(arguments, None, GRID), 1, (1, 1, 1), (), ()),)
        program = SimpleNamespace(
            saved_input_indices=(0,),
            allocations=(layout,),
            outputs=(layout,),
            events=(
                Allocate("output", torch.float32, (8,), (1,)),
                DirectKernelCall(None, arguments, GRID),
            ),
        )
        if reason == "no_candidates":
            program.saved_input_indices = ()
        elif reason == "no_allocations":
            program.allocations = ()
        elif reason == "no_calls":
            calls = ()
        else:
            program.outputs = (BorrowedInputOutput(InputSource(0)),)
        self.assertIsNone(_release_steps(program, calls))


if __name__ == "__main__":
    run_tests()
