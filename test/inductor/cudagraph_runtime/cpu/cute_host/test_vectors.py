"""Project typed fixed-vector lanes and query their actual compiler layout."""

import hashlib
import json
from pathlib import Path

from torch._inductor.runtime._cudagraph._sdk import activate

activate()

ROOT = Path(__file__).resolve().parent / "fixtures"

from torch._inductor.runtime._cudagraph._compiler.aggregate_plan import _scalar_leaves
from torch._inductor.runtime._cudagraph._compiler.argument_flow import _Values, parameter_leaves, ValueFlow
from torch._inductor.runtime._cudagraph._compiler.compiler_type_layout import compile_type_layouts
from cutlass._mlir import ir
from torch._inductor.runtime._cudagraph._compiler.llvm_query import _type
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


SLOT = "!llvm.struct<(i1, i1, i1, vector<4xi32>)>"


@instantiate_parametrized_tests
class TestFixedIntegerVectors(TestCase):
    def test_actual_gemm_parameter_zero_vector(self):
        saved = ROOT / "observation_gemm_attempt1_artifacts/lowered_host.mlir"
        data = saved.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         "040b7da2198aca0aa225855fbe9bdb9c875615d824f25caad7a2a441d314186b")
        lines = data.decode().splitlines()
        starts = [index for index, line in enumerate(lines) if line.startswith("llvm.func @cutlass___call__")]
        self.assertEqual(len(starts), 1)
        start = starts[0]
        end = lines.index("}", start)
        names = {"%9", "%15", "%35", "%36", "%37", "%38", "%39"}
        statements = [line for line in lines[start:end] if line.strip().split(" = ", 1)[0] in names]
        self.assertEqual(len(statements), len(names))
        text = f"module {{ llvm.func @slot() -> {SLOT} {{\n" + "\n".join(statements)
        text += f"\nllvm.return %39 : {SLOT}\n}} }}"
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = ir.Module.parse(text)
            self.assertTrue(module.operation.verify())
            before = str(module)
            function, = module.body.operations
            block = function.regions[0].blocks[0]
            result = tuple(block.operations)[-1].operands[0]
            flow = _Values(block.arguments).read_parameter(result)
            leaves = parameter_leaves(flow, result.type)
            expected = tuple(((index,), "i1") for index in range(3))
            expected += tuple(((3, index), "i32") for index in range(4))
            self.assertEqual(tuple((path, str(typ)) for path, typ, _ in leaves),
                             expected)
            for _, _, value in leaves[:3]:
                self.assertEqual(value.kind, "constant")
                attribute = ir.Attribute.parse(value.value)
                self.assertIsInstance(attribute, ir.BoolAttr)
                self.assertFalse(attribute.value)
            for _, _, value in leaves[3:]:
                self.assertEqual(value.kind, "constant")
                attribute = ir.Attribute.parse(value.value)
                self.assertIsInstance(attribute, ir.IntegerAttr)
                self.assertEqual(attribute.value, 0)
            self.assertEqual(str(module), before)

    def test_actual_slot_layout_and_packed_padding(self):
        actual = json.loads((ROOT / "observation_gemm_attempt1_artifacts/cuda_parameter_layout.json").read_text())
        with ir.Context():
            types = (ir.Type.parse(SLOT), ir.Type.parse("!llvm.struct<packed (i1, i1, i1, vector<4xi32>)>"))
            self.assertEqual(_type(types[0]), "{ i1, i1, i1, <4 x i32> }")
            layouts = compile_type_layouts(types, device_target="sm_100a")
        ordinary, packed = layouts.layouts
        self.assertEqual((ordinary.size, ordinary.alignment), (actual["parameters"][0]["size"], 16))
        self.assertEqual(tuple(field.offset for field in ordinary.fields), (0, 1, 2, 16, 20, 24, 28))
        self.assertEqual((packed.size, packed.alignment), (19, 1))
        self.assertEqual(tuple(field.offset for field in packed.fields), (0, 1, 2, 3, 7, 11, 15))

    @parametrize("width,values", (
        (1, (0, 1, 1, 0)), (8, (-128, -1, 0, 127)), (16, (-32768, -1, 0, 32767)),
        (32, (-(2**31), -1, 0, 2**31 - 1)), (64, (-(2**63), -1, 0, 2**63 - 1)),
    ))
    def test_typed_dense_lanes_preserve_signed_bits(self, width, values):
        literals = tuple(("true" if value else "false") if width == 1 else str(value) for value in values)
        vector = f"vector<{len(values)}xi{width}>"
        text = f"""module {{ llvm.func @lanes() -> {vector} {{
          %value = llvm.mlir.constant(dense<[{', '.join(literals)}]> : {vector}) : {vector}
          llvm.return %value : {vector}
        }} }}"""
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = ir.Module.parse(text)
            self.assertTrue(module.operation.verify())
            function, = module.body.operations
            block = function.regions[0].blocks[0]
            result = tuple(block.operations)[-1].operands[0]
            flow = _Values(block.arguments).read_parameter(result)
            leaves = parameter_leaves(flow, result.type)
            self.assertEqual(tuple(path for path, _, _ in leaves), tuple((index,) for index in range(len(values))))
            mask = (1 << width) - 1
            self.assertEqual(tuple(int(ir.Attribute.parse(value.value).value) & mask
                                   for _, _, value in leaves), tuple(value & mask for value in values))

    @parametrize("width,value", ((32, -1), (64, -(2**63))))
    def test_splat_lanes_use_typed_attribute(self, width, value):
        vector = f"vector<3xi{width}>"
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = ir.Module.parse(f"""module {{ llvm.func @splat() -> {vector} {{
              %value = llvm.mlir.constant(dense<{value}> : {vector}) : {vector}
              llvm.return %value : {vector}
            }} }}""")
            self.assertTrue(module.operation.verify())
            function, = module.body.operations
            block = function.regions[0].blocks[0]
            result = tuple(block.operations)[-1].operands[0]
            leaves = parameter_leaves(_Values(block.arguments).read_parameter(result), result.type)
            self.assertEqual(tuple(ir.IntegerAttr(ir.Attribute.parse(source.value)).value
                                   for _, _, source in leaves), (value,) * 3)

    def test_vector_formal_lanes_keep_original_paths(self):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = ir.Module.parse("""module {
              llvm.func @formal(%value: vector<3xi32>) -> vector<3xi32> {
                llvm.return %value : vector<3xi32>
              }
            }""")
            self.assertTrue(module.operation.verify())
            function, = module.body.operations
            argument, = function.regions[0].blocks[0].arguments
            flow = _Values((argument,)).read_parameter(argument)
            leaves = parameter_leaves(flow, argument.type)
            self.assertEqual(tuple(source for _, _, source in leaves),
                             tuple(ValueFlow("argument", "i32", 0, (index,)) for index in range(3)))

    @parametrize("kind", ("zero", "undef", "poison"))
    def test_vector_undefined_semantics_remain_exact(self, kind):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = ir.Module.parse(f"""module {{ llvm.func @seed() -> vector<4xi32> {{
              %value = llvm.mlir.{kind} : vector<4xi32>
              llvm.return %value : vector<4xi32>
            }} }}""")
            self.assertTrue(module.operation.verify())
            function, = module.body.operations
            block = function.regions[0].blocks[0]
            result = tuple(block.operations)[-1].operands[0]
            reader = _Values(block.arguments)
            leaves = parameter_leaves(reader._read(result), result.type)
            self.assertEqual(tuple(source.kind for _, _, source in leaves), (kind,) * 4)
            if kind == "zero":
                reader.read(result)
            else:
                with self.assertRaisesRegex(ValueError, f"Live uninitialized parameter provenance: {kind}"):
                    reader.read(result)
            if kind == "poison":
                with self.assertRaisesRegex(ValueError, "Live uninitialized parameter provenance: poison"):
                    reader.read_parameter(result)
            else:
                reader.read_parameter(result)

    @parametrize("spelling", ("vector<2x2xi32>", "vector<[4]xi32>", "vector<4xf32>", "vector<4xi7>"))
    def test_unsupported_vectors_decline_consistently(self, spelling):
        with ir.Context():
            typ = ir.Type.parse(spelling)
            for inspect in (_type, _scalar_leaves, _Values._children):
                with self.assertRaisesRegex(ValueError, "fixed rank-one vector"):
                    inspect(typ)


if __name__ == "__main__":
    run_tests()
