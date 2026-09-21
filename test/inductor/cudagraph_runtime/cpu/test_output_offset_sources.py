"""Output expression leaves retain their declared integer or Tensor source kind."""


from torch._inductor.runtime.cudagraph_arg_mapping import (
    bind_output_slots, BorrowedInputOutput, InputSource, IntegerOutput, IntExpr, TensorViewOutput,
)
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


@instantiate_parametrized_tests
class TestOutputOffsetSources(TestCase):
    def test_mixed_integer_and_metadata_expression(self):
        expression = IntExpr("add", args=(IntExpr("boxed", 0),
            IntExpr("multiply", args=(IntExpr("constant", -1), IntExpr("storage_offset", 1)))))
        outputs = (IntegerOutput(expression), TensorViewOutput(InputSource(1), (1,), (1,), expression))
        self.assertEqual(bind_output_slots(outputs, (), ("count", "tensor"), {0}, symbolic=True), outputs)


    @parametrize("op,index", (("boxed", 1), ("storage_offset", 0), ("boxed", 2), ("storage_offset", 2)))
    def test_wrong_kind_or_missing_input_declines_recursively(self, op, index):
        expression = IntExpr("add", args=(IntExpr("constant", 1),
            IntExpr("multiply", args=(IntExpr("constant", -2), IntExpr(op, index)))))
        for output in (IntegerOutput(expression), TensorViewOutput(InputSource(1), (1,), (1,), expression)):
            outputs = (BorrowedInputOutput(InputSource(1)), output)
            self.assertIsNone(bind_output_slots(outputs, (), ("count", "tensor"), {0}, symbolic=True))


if __name__ == "__main__":
    run_tests()
