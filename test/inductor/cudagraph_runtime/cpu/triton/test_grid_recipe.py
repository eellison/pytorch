from types import SimpleNamespace

from torch._inductor.runtime.cudagraph_arg_mapping import (
    bind_grid_recipe, CallArgument, ExpressionSource, IntegerInput, IntExpr,
    KernelCallRecord, LauncherArgument,
)
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch._inductor.runtime.triton_heuristics import GridExpr
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


@instantiate_parametrized_tests
class TestGeneratedGridRecipe(TestCase):
    @parametrize("block", (32, 128))
    @parametrize("literal_second", (False, True))
    def test_selected_combo_grid_matches_emitted_launcher(self, block, literal_second):
        meta = {"grid_type": "SequentialComboKernelGrid", "combo_grid_meta": {
            "default_config": {"XBLOCK": 64}, "num_kernels": 2, "min_blocks": None,
            "xnumel_0": None, "xnumel_1": 2053 if literal_second else None,
            "no_x_dim_0": False, "no_x_dim_1": False,
        }}
        grid = GridExpr.from_meta(meta, {"XBLOCK": block})
        arguments = tuple(CallArgument(f"xnumel_{index}", index, index, "i32",
                                      ExpressionSource(IntExpr("boxed", index))) for index in range(2))
        call = KernelCallRecord(0, "combo", tuple(row.formal for row in arguments),
                                arguments, "SequentialComboKernelGrid")
        selected = tuple(LauncherArgument(row.formal, index, index, "i32", index, None)
                         for index, row in enumerate(arguments))
        recipe = bind_grid_recipe(call, selected, grid.recipe)
        self.assertIsNotNone(recipe)
        records = SimpleNamespace(input_names=("left", "right"),
                                  integer_inputs=(IntegerInput("left", 0), IntegerInput("right", 1)))
        for left, right in ((1031, 2053), (67, 4099), (2051, 103)):
            numeric = _NumericProgram(records, (left, right))
            actual = tuple(numeric.values[numeric.add(axis)] for axis in recipe)
            emitted = grid.eval_slow({"xnumel_0": left, "xnumel_1": right})
            self.assertEqual(actual, emitted)
            self.assertEqual(actual, ((left + block - 1) // block
                + ((2053 if literal_second else right) + block - 1) // block, 1, 1))
        changed = (selected[0], LauncherArgument("unrelated", 1, 1, "i32", 1, None))
        self.assertIsNone(bind_grid_recipe(call, changed, grid.recipe))


if __name__ == "__main__":
    run_tests()
