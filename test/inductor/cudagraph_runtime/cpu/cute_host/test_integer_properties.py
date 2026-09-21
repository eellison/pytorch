"""Keep compiler overflow properties and exactness attributes in source provenance."""

from torch._inductor.runtime._cudagraph._sdk import activate

activate()

from torch._inductor.runtime._cudagraph._compiler.argument_flow import _Values
from cutlass._mlir import ir
from cutlass._mlir.dialects.llvm import IntegerOverflowFlags
from torch._inductor.runtime._cudagraph._compiler.overflow_properties import _flags
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


@instantiate_parametrized_tests
class TestValueFlowProperties(TestCase):
    @parametrize("operation,flag_names", (
        ("add", ("nsw",)), ("sub", ("nuw",)), ("mul", ("nsw", "nuw")),
        ("shl", ("nsw", "nuw")), ("trunc", ("nsw", "nuw")),
    ))
    @parametrize("flagged", (False, True))
    def test_exact_compiler_overflow_property(self, operation, flag_names, flagged):
        suffix = " overflow<" + ", ".join(flag_names) + ">" if flagged else ""
        arguments = "%a: i64" if operation == "trunc" else "%a: i32, %b: i32"
        instruction = (f"llvm.trunc %a{suffix} : i64 to i32" if operation == "trunc"
                       else f"llvm.{operation} %a, %b{suffix} : i32")
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = ir.Module.parse(f"""module {{ llvm.func @property({arguments}) -> i32 {{
              %value = {instruction}
              llvm.return %value : i32
            }} }}""")
            self.assertTrue(module.operation.verify())
            before = str(module)
            function, = module.body.operations
            block = function.regions[0].blocks[0]
            operation = tuple(block.operations)[0].operation
            expected = 0
            if flagged:
                for name in flag_names:
                    expected |= int(getattr(IntegerOverflowFlags, name))
            self.assertEqual(_flags(operation), expected)
            source = _Values(block.arguments).read(operation.results[0])
            attributes = dict(source.attributes)
            if expected:
                self.assertEqual(set(attributes), {"overflowFlags"})
                retained = ir.Attribute.parse(attributes["overflowFlags"])
                self.assertIsInstance(retained, ir.IntegerAttr)
                self.assertEqual(retained.type, ir.IntegerType.get_signless(32))
                self.assertEqual(retained.value, expected)
            else:
                self.assertEqual(source.attributes, ())
            self.assertEqual(str(module), before)

    @parametrize("operation", ("udiv", "sdiv", "lshr", "ashr"))
    @parametrize("exact", (False, True))
    def test_exact_source_unit_attribute(self, operation, exact):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = ir.Module.parse(f"""module {{ llvm.func @exact(%a: i32, %b: i32) -> i32 {{
              %value = llvm.{operation} %a, %b : i32
              llvm.return %value : i32
            }} }}""")
            function, = module.body.operations
            block = function.regions[0].blocks[0]
            operation = tuple(block.operations)[0].operation
            operation.opview.isExact = exact
            self.assertTrue(module.operation.verify())
            self.assertEqual(operation.opview.isExact, exact)
            self.assertEqual("isExact" in operation.attributes, exact)
            before = str(module)
            source = _Values(block.arguments).read(operation.results[0])
            if exact:
                self.assertEqual(set(dict(source.attributes)), {"isExact"})
                retained = ir.Attribute.parse(dict(source.attributes)["isExact"])
                self.assertIsInstance(retained, ir.UnitAttr)
                self.assertEqual(retained, operation.attributes["isExact"])
            else:
                self.assertEqual(source.attributes, ())
            self.assertEqual(str(module), before)


if __name__ == "__main__":
    run_tests()
