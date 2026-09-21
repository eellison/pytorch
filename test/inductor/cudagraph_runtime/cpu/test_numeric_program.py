"""Cold numeric admission follows signed scalar and operation domains."""

from types import SimpleNamespace


from torch._inductor.runtime.cudagraph_arg_mapping import IntegerInput, IntExpr
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


RECORDS = SimpleNamespace(input_names=("scalar",), integer_inputs=(IntegerInput("scalar", 0),))
BOXED = IntExpr("boxed", 0)


@instantiate_parametrized_tests
class TestNumericProgram(TestCase):
    @parametrize("value", (-(1 << 63), -1, 0, 1, (1 << 63) - 1))
    def test_signed_boxed_value_and_identity_arithmetic(self, value):
        program = _NumericProgram(RECORDS, [value])
        expression = IntExpr("multiply", args=(
            IntExpr("add", args=(BOXED, IntExpr("constant", 0))), IntExpr("constant", 1)))
        result = program.add(expression)
        self.assertEqual(program.values[result], value)
        self.assertEqual(program.instructions[0], ("boxed", 0))


    @parametrize("value", (True, 1.0, -(1 << 63) - 1, 1 << 63))
    def test_boxed_type_and_width_remain_checked(self, value):
        with self.assertRaisesRegex(UnsupportedCapture, "signed int64"):
            _NumericProgram(RECORDS, [value])

    @parametrize("operation", ("ceildiv", "floordiv"))
    def test_negative_dividend_remains_unsupported(self, operation):
        program = _NumericProgram(RECORDS, [-1])
        with self.assertRaisesRegex(UnsupportedCapture, "division domain"):
            program.add(IntExpr(operation, args=(BOXED, IntExpr("constant", 2))))

    @parametrize("operation,left,right", (
        ("add", -(1 << 63), -1),
        ("multiply", (1 << 63) - 1, 2),
    ))
    def test_signed_intermediate_overflow_remains_checked(self, operation, left, right):
        program = _NumericProgram(RECORDS, [left])
        with self.assertRaisesRegex(UnsupportedCapture, "exceeds int64"):
            program.add(IntExpr(operation, args=(BOXED, IntExpr("constant", right))))


if __name__ == "__main__":
    run_tests()
