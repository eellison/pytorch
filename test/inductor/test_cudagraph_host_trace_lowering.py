# Owner(s): ["module: inductor"]

from types import SimpleNamespace

import sympy

import torch
from torch._inductor.runtime._cudagraph.host_trace import _HostIntegers
from torch._inductor.runtime.cudagraph_arg_mapping import (
    bind_output_slots,
    grid_expression_inputs,
    InputSource,
    IntExpr,
    TensorViewOutput,
)
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch._inductor.runtime.cudagraph_host_trace_mapping import HostTraceInputMetadata
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import FloorDiv, PythonMod


@instantiate_parametrized_tests
class TestHostTraceLowering(TestCase):
    @parametrize("rows, columns, row_stride", ((4, 8, 8), (12, 16, 20), (8, 32, 32)))
    def test_symbolic_metadata_reaches_numeric_output_layout(
        self, rows, columns, row_stride
    ):
        m, n, stride = sympy.symbols("m n stride", integer=True, positive=True)
        mapping = SimpleNamespace(
            metadata_symbols={
                m: HostTraceInputMetadata(0, "size", 0),
                n: HostTraceInputMetadata(0, "size", 1),
                stride: HostTraceInputMetadata(0, "stride", 0),
            }
        )
        lower = _HostIntegers(mapping)
        expressions = (m * n, FloorDiv(m * n, m), 2 * stride, PythonMod(m + 1, 4))
        tensor = torch.empty_strided((rows, columns), (row_stride, 1))
        numeric = _NumericProgram(
            SimpleNamespace(input_names=("x",), integer_inputs=()), (tensor,)
        )
        values = tuple(
            numeric.values[numeric.add(lower(value))] for value in expressions
        )
        self.assertEqual(
            values, (rows * columns, columns, 2 * row_stride, (rows + 1) % 4)
        )
        output = TensorViewOutput(
            InputSource(0), (lower(FloorDiv(m * n, m)),), (lower(stride),), 0
        )
        self.assertEqual(
            bind_output_slots((output,), (), ("x",), set(), symbolic=True), (output,)
        )

    def test_tensor_metadata_cannot_load_a_boxed_integer(self):
        expression = IntExpr("size", 0, (IntExpr("constant", 0),))
        self.assertIsNone(
            grid_expression_inputs(expression, boxed_indices={0}, tensor_indices=set())
        )
        self.assertEqual(
            grid_expression_inputs(expression, boxed_indices=set(), tensor_indices={0}),
            (0,),
        )

    @parametrize("dimension", (-1, 2))
    def test_preparation_rejects_invalid_tensor_dimensions(self, dimension):
        numeric = _NumericProgram(
            SimpleNamespace(input_names=("x",), integer_inputs=()), (torch.empty(3, 4),)
        )
        with self.assertRaisesRegex(UnsupportedCapture, "original input dimension"):
            numeric.add(IntExpr("stride", 0, (IntExpr("constant", dimension),)))

    def test_division_obligation_precedes_result_range(self):
        n, divisor = sympy.symbols("n divisor", integer=True)
        lower = _HostIntegers(
            SimpleNamespace(
                metadata_symbols={
                    n: HostTraceInputMetadata(0, "size", 0),
                    divisor: HostTraceInputMetadata(0, "stride", 0),
                }
            )
        )
        result = FloorDiv(n, divisor)
        lower(result)
        self.assertLess(
            lower.guards.index(sympy.Gt(divisor, 0)),
            lower.guards.index(sympy.Lt(result, 2**63)),
        )


if __name__ == "__main__":
    run_tests()
