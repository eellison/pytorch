"""Check compiled physical fields, undefined leaves, and padding independently."""

import hashlib
from pathlib import Path

from torch._inductor.runtime._cudagraph._sdk import activate

activate()

ROOT = Path(__file__).resolve().parent / "fixtures"

from torch._inductor.runtime._cudagraph._compiler.argument_flow import ParameterFlow, ValueFlow, _Values
from torch._inductor.runtime._cudagraph._compiler.compiler_type_layout import compile_type_layouts
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from torch._inductor.runtime._cudagraph._compiler.physical_parameters import project_parameters
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests, parametrize, run_tests, TestCase,
)


@instantiate_parametrized_tests
class TestPhysicalParameters(TestCase):
    def test_actual_descriptor(self):
        text = (ROOT / "tma_descriptor_prefix.mlir").read_bytes()
        self.assertEqual(hashlib.sha256(text).hexdigest(),
                         "be29df0069ccc5e0b315e1dd959b555ae04c0549d31d7ea2b7d7e936eb9d32f2")
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = ir.Module.parse(text.decode())
            host, = module.body.operations
            block = host.regions[0].blocks[0]
            returned = tuple(block.operations)[-1].operands[0]
            before = str(module)
            pointer, = compile_type_layouts((llvm.PointerType.get(),), device_target="sm_100a").layouts
            reader = _Values(block.arguments, local_pointer_width=pointer.size * 8)
            flow = reader.read_parameter(returned)
            parameter = ParameterFlow(0, str(returned.type), flow)
            layout = compile_type_layouts((returned.type,), device_target="sm_100a")
            result, = project_parameters((parameter,), layout)
            self.assertIs(result.parameter, parameter)
            self.assertEqual((result.size, result.alignment, result.padding), (128, 8, ()))
            self.assertEqual(tuple((leaf.byte_offset, leaf.byte_size) for leaf in result.fields),
                             tuple((8 * index, 8) for index in range(8)))
            self.assertEqual(tuple((leaf.byte_offset, leaf.byte_size, leaf.source.kind)
                                   for leaf in result.undefined),
                             tuple((8 * index, 8, "undef") for index in range(8, 16)))
            pending, inputs = [result.fields[0].source], set()
            while pending:
                value = pending.pop()
                if value.kind == "argument":
                    inputs.add((value.argument, value.path, value.llvm_type))
                pending.extend(value.operands)
            self.assertEqual(inputs, {(0, (0,), "!llvm.ptr<1>")})
            self.assertEqual(str(module), before)

    def test_padding_is_distinct_from_undefined_integer(self):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = ir.Module.parse("""module {
              llvm.func @fields(%arg0: i32) -> !llvm.struct<(i32, i64)> {
                %empty = llvm.mlir.undef : !llvm.struct<(i32, i64)>
                %result = llvm.insertvalue %arg0, %empty[0] : !llvm.struct<(i32, i64)>
                llvm.return %result : !llvm.struct<(i32, i64)>
              }
            }""")
            block = tuple(module.body.operations)[0].regions[0].blocks[0]
            returned = tuple(block.operations)[-1].operands[0]
            flow = _Values(block.arguments).read_parameter(returned)
            parameter = ParameterFlow(0, str(returned.type), flow)
            layout = compile_type_layouts((returned.type,), device_target="sm_100a")
            result, = project_parameters((parameter,), layout)
            self.assertEqual(result.padding, ((4, 4),))
            self.assertEqual(tuple((leaf.byte_offset, leaf.byte_size) for leaf in result.undefined), ((8, 8),))
            self.assertEqual(result.fields[0].source, ValueFlow("argument", "i32", 0))

    @parametrize("width,literal,data", ((1, "true", b"\x01"), (8, "-3 : i8", b"\xfd"),
                                      (16, "-2 : i16", b"\xfe\xff")))
    def test_exact_narrow_literal_bytes(self, width, literal, data):
        with ir.Context():
            typ = ir.IntegerType.get_signless(width)
            value = ValueFlow("constant", str(typ), value=literal)
            parameter = ParameterFlow(0, str(typ), value)
            layout = compile_type_layouts((typ,), device_target="sm_100a")
            result, = project_parameters((parameter,), layout)
            self.assertEqual((result.fields, result.undefined, result.padding), ((), (), ()))
            constant, = result.constants
            self.assertEqual(constant.data, data)
            self.assertEqual(constant.field.source, value)
            self.assertEqual(constant.field.byte_size, len(data))

    def test_narrow_argument_requires_mutable_transport(self):
        with ir.Context():
            typ = ir.IntegerType.get_signless(16)
            parameter = ParameterFlow(0, str(typ), ValueFlow("argument", "i16", 0))
            layout = compile_type_layouts((typ,), device_target="sm_100a")
            with self.assertRaisesRegex(ValueError, "Unsupported physical parameter leaf"):
                project_parameters((parameter,), layout)

    def test_compiler_layout_cannot_be_substituted(self):
        with ir.Context():
            i32, i64 = ir.IntegerType.get_signless(32), ir.IntegerType.get_signless(64)
            parameter = ParameterFlow(0, str(i32), ValueFlow("argument", "i32", 0))
            layout = compile_type_layouts((i64,), device_target="sm_100a")
            with self.assertRaisesRegex(ValueError, "Physical layout differs"):
                project_parameters((parameter,), layout)


if __name__ == "__main__":
    run_tests()
