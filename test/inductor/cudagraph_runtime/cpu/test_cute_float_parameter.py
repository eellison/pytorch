# Owner(s): ["module: inductor"]

import struct
from types import SimpleNamespace

from torch._inductor.runtime._cudagraph._sdk import activate
from torch._inductor.runtime.cudagraph_boxed_replay import (
    _NumericProgram,
    _ParameterProgram,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


activate()

from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.artifact import (
    ParameterExpression,
)
from torch._inductor.runtime._cudagraph._compiler.cute_bridge.parameter_program import (
    lower_parameter,
)
from torch._inductor.runtime._cudagraph._compiler.user_triton.linking import (
    LinkDeclined,
)


@instantiate_parametrized_tests
class TestCuTeFloatParameter(TestCase):
    @parametrize(
        "bits",
        (
            0x00000000,
            0x80000000,
            0x00000001,
            0x80000001,
            0x3F800000,
            0xBF800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
        ),
    )
    def test_literal_preserves_bits_through_signed_parameter_transport(self, bits):
        expression = ParameterExpression(
            "constant", "f32", None, (), f"0x{bits:08X} : f32", (), ()
        )
        source, obligations = lower_parameter(expression, None)
        self.assertEqual(obligations, ())
        numeric = _NumericProgram(
            SimpleNamespace(input_names=(), integer_inputs=()), ()
        )
        parameters = _ParameterProgram(numeric, 0, {})
        index = parameters.add(source)
        value = parameters.evaluate((), {})[index]
        self.assertEqual(struct.pack("<i", value), bits.to_bytes(4, "little"))

    @parametrize("bits", (0x7F800001, 0xFF800001, 0x7FC00001, 0xFFC00001))
    def test_nan_literal_declines_before_lossy_transport(self, bits):
        expression = ParameterExpression(
            "constant", "f32", None, (), f"0x{bits:08X} : f32", (), ()
        )
        with self.assertRaisesRegex(LinkDeclined, "NaN f32 parameter literals"):
            lower_parameter(expression, None)


if __name__ == "__main__":
    run_tests()
