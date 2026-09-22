# Owner(s): ["module: inductor"]

from torch._inductor.runtime._cudagraph._sdk import activate


activate()

from cutlass._mlir import ir

from torch._inductor.runtime._cudagraph._compiler.accessors import _snapshot
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _module():
    return ir.Module.parse(
        """
module {
  func.func @sibling() -> i64 {
    %c = arith.constant 9 : i64
    return %c : i64
  }
  func.func @getter(%arg0: i64) -> i64 attributes {callee = @sibling} {
    %c = arith.constant 3 : i64
    return %c : i64
  }
}
"""
    )


@instantiate_parametrized_tests
class TestOperationSnapshot(TestCase):
    @parametrize("mutation", ("constant", "symbol", "type", "location"))
    def test_function_snapshot_detects_relevant_mutations(self, mutation):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = _module()
            function = tuple(module.body.operations)[1].operation
            block = function.regions[0].blocks[0]
            constant = next(iter(block.operations)).operation
            before = _snapshot(function)
            if mutation == "constant":
                constant.attributes["value"] = ir.IntegerAttr.get(
                    ir.IntegerType.get_signless(64), 4
                )
            elif mutation == "symbol":
                function.attributes["callee"] = ir.FlatSymbolRefAttr.get("getter")
            elif mutation == "type":
                argument_type = ir.IntegerType.get_signless(32)
                block.arguments[0].set_type(argument_type)
                function.attributes["function_type"] = ir.TypeAttr.get(
                    ir.FunctionType.get(
                        [argument_type], [ir.IntegerType.get_signless(64)]
                    )
                )
            else:
                constant.location = ir.Location.file("changed.mlir", 9, 3)
            self.assertTrue(module.operation.verify())
            after = _snapshot(function)
            self.assertNotEqual(after, before)
            self.assertNotEqual(after[1], before[1])
            if mutation == "location":
                self.assertEqual(after[0], before[0])

    @parametrize("mutation", ("constant", "attribute"))
    def test_module_snapshot_detects_sibling_mutations(self, mutation):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = _module()
            sibling, function = (view.operation for view in module.body.operations)
            before_module = _snapshot(module.operation)
            before_function = _snapshot(function)
            if mutation == "constant":
                constant = next(iter(sibling.regions[0].blocks[0].operations)).operation
                constant.attributes["value"] = ir.IntegerAttr.get(
                    ir.IntegerType.get_signless(64), 10
                )
            else:
                sibling.attributes["custom"] = ir.StringAttr.get("changed")
            self.assertTrue(module.operation.verify())
            self.assertNotEqual(_snapshot(module.operation), before_module)
            self.assertEqual(_snapshot(function), before_function)

    def test_alias_rich_signature_attributes_and_locations_survive(self):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = ir.Module.parse(
                """
!tensor = !cute.memref<f32, gmem, align<16>, "(?,128):(?{i64 div=8},1)">
#map = affine_map<(d0) -> (d0 + 1)>
#loc = loc(fused["source", "lowered"])
module {
  func.func @getter(%arg0: !tensor) -> !tensor
      attributes {mapping = #map, payload = dense<[1, 2]> : tensor<2xi32>} {
    return %arg0 : !tensor loc(#loc)
  } loc(#loc)
}
"""
            )
            function = next(iter(module.body.operations)).operation
            self.assertTrue(module.operation.verify())
            text, data = _snapshot(function)
            self.assertIn(str(function.regions[0].blocks[0].arguments[0].type), text)
            self.assertIn(str(function.attributes["mapping"]), text)
            self.assertIn(str(function.attributes["payload"]), text)
            restored_module = ir.Module.parse(data)
            restored = next(iter(restored_module.body.operations)).operation
            self.assertTrue(restored.verify())
            self.assertEqual(_snapshot(restored), (text, data))
            self.assertEqual(str(restored.location), str(function.location))


if __name__ == "__main__":
    run_tests()
