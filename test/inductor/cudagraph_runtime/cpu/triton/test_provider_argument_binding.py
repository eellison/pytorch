from dataclasses import replace

from torch._inductor.runtime._cudagraph._compiler.user_triton.contract import Invocation
from torch._inductor.runtime.cudagraph_arg_mapping import (
    bind_launcher_arguments, bind_wrapper_arguments, BufferSource, CallArgument,
    ExpressionSource, InputSource, IntegerSource, IntExpr, KernelArgument,
)
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


@instantiate_parametrized_tests
class TestProviderArgumentBinding(TestCase):
    def setUp(self):
        super().setUp()
        self.formals = ("input", "output", "xnumel", "rnumel", "XBLOCK")
        self.kernel_arguments = (
            KernelArgument("input", 0, "*fp32", 0, None),
            KernelArgument("output", 1, "*fp32", 1, None),
            KernelArgument("xnumel", 2, "i32", 2, None),
            KernelArgument("rnumel", 3, "constexpr", None, 128),
            KernelArgument("XBLOCK", 4, "constexpr", None, 256),
        )
        extent = IntExpr("multiply", args=(IntExpr("boxed", 0), IntExpr("constant", 128)))
        self.call = Invocation("generated", self.formals, (
            CallArgument("input", 0, 0, "*fp32", InputSource(1)),
            CallArgument("output", 1, 1, "*fp32", BufferSource("output")),
            CallArgument("xnumel", 2, 2, "i32", ExpressionSource(extent)),
            CallArgument("rnumel", 3, 3, "constexpr", IntegerSource(128)),
        ), (IntExpr("ceildiv", args=(extent, IntExpr("constant", 256))),
            IntExpr("constant", 1), IntExpr("constant", 1)))
        self.selected = bind_launcher_arguments(
            self.kernel_arguments, list(self.formals), list(self.formals[:4]), list(self.formals[:3]),
        )
        self.assertIsNotNone(self.selected)

    def test_provider_invocation_preserves_physical_sources(self):
        bound = bind_wrapper_arguments(self.call, self.selected)
        self.assertEqual(bound, self.call.arguments[:3])
        for actual, original in zip(bound, self.call.arguments[:3], strict=True):
            self.assertIs(actual, original)
        self.assertEqual([row.call_arg_index for row in self.selected], [0, 1, 2, 3, None])
        self.assertEqual([row.abi_index for row in self.selected], [0, 1, 2, None, None])

    @parametrize("damage", ("formal", "type"))
    def test_provider_invocation_rejects_launcher_mismatch(self, damage):
        call, selected = self.call, self.selected
        if damage == "formal":
            call = replace(call, formals=("other", *self.formals[1:]))
        else:
            arguments = (*self.kernel_arguments[:2], replace(self.kernel_arguments[2], triton_type="i64"),
                         *self.kernel_arguments[3:])
            selected = bind_launcher_arguments(
                arguments, list(self.formals), list(self.formals[:4]), list(self.formals[:3]),
            )
            self.assertIsNotNone(selected)
        self.assertIsNone(bind_wrapper_arguments(call, selected))


if __name__ == "__main__":
    run_tests()
