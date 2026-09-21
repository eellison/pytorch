"""TMA stride alignment follows descriptor basis modes and compiler projections."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from torch._inductor.runtime._cudagraph._sdk import activate

activate()

from cutlass._mlir import ir, passmanager
from cutlass._mlir.dialects import cute, cute_nvgpu, func
from cutlass.cute import core
from torch._inductor.runtime._cudagraph._compiler.cfg_values import read_cfg_function
from torch._inductor.runtime._cudagraph._compiler.cute_bridge.numeric import lower_numeric, NumericSource
from torch._inductor.runtime._cudagraph._compiler.decoded_values import prepare_decodings
from torch._inductor.runtime._cudagraph._compiler.overflow_properties import bind_properties
from torch._inductor.runtime._cudagraph._compiler.owned_numeric import freeze_numeric
from torch._inductor.runtime._cudagraph._compiler.tma_requirements import (
    _atom_layout, _basis_groups, project_tma_property, read_tma_requirements,
)
from torch._inductor.runtime.cudagraph_arg_mapping import IntegerInput, IntExpr
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


WORKTREE = next(parent for parent in Path(__file__).resolve().parents if (parent / "torch/_inductor").is_dir())
SOURCE = WORKTREE / "test/inductor/cudagraph_runtime/cpu/cute_host/fixtures/observation_gemm_attempt1_artifacts/source_host.mlir"


def source_module():
    lines = SOURCE.read_text().splitlines(True)
    cut = next(index for index, line in enumerate(lines) if " = cute.kernel_smem_size " in line)
    return ir.Module.parse("module {\n" + "".join(lines[:cut]) + "  return %126 : i32\n}\n}\n")


def make_constructor(template, typ, format_name, *, nested=False):
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        function = func.FuncOp("constructor", ([typ], []))
    block = function.add_entry_block()
    with ir.InsertionPoint(block):
        smem = cute.StaticOp(template.operands[1].type).result
        cta_type = template.operands[2].type
        if nested:
            cta_type = ir.Type.parse(str(cta_type).replace("1@0", "1@1@0"))
        cta = cute.StaticOp(cta_type).result
        constructor = cute_nvgpu.AtomCopyMakeNonExecTiledTmaLoadOp(
            block.arguments[0], smem, cta, template.attributes["kind"],
            num_multicast=1, tma_format=cute_nvgpu.TmaDataFormat[format_name],
        ).operation
        func.ReturnOp([])
    if not module.operation.verify():
        raise AssertionError("Expected a valid compiler TMA constructor")
    return module, constructor


def compile_projection(requirement, *, derived=False, property="stride"):
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        function = func.FuncOp("probe", ([requirement.tensor.type], [ir.IntegerType.get_signless(64)]))
    block = function.add_entry_block()
    with ir.InsertionPoint(block):
        tensor = block.arguments[0]
        if derived:
            layout = cute.GetLayoutOp(tensor).result
            shape = core._unpack_x_tuple(cute.GetShapeOp(layout).result)
            strides = core._unpack_x_tuple(cute.GetStrideOp(layout).result)
            first = ((strides[0][0], strides[0][1] + 8) if len(requirement.path) == 2
                     else strides[0] + 8)
            adjusted = core.make_layout(shape, stride=(first, *strides[1:]))
            pointer = cute.GetIterOp(tensor).result
            tensor = cute.MakeViewOp(requirement.tensor.type, pointer, layout=adjusted).result
        projected = project_tma_property(requirement, tensor, property)
        func.ReturnOp([projected])
    if not module.operation.verify():
        raise AssertionError("TMA projection did not preserve valid typed host IR")
    before = str(module)
    passmanager.PassManager.parse("builtin.module(cute-to-nvvm{enable-cuda-dialect cubin-chip=sm_100a})").run(module.operation)
    cfg = read_cfg_function(module, "probe")
    return before, freeze_numeric(bind_properties(cfg), prepare_decodings(cfg), (0,))


@instantiate_parametrized_tests
class TestTmaStrideRequirements(TestCase):
    @parametrize("basis,expected", (
        ('(64,128,1):(1@1,1@0,1@2)', ((0,), (2,))),
        ('(64,(128,2)):(1@1,(1@0,1@2))', ((0,), (2,))),
        ('(64,(128,2)):(1@1,(1@0@1,1@2@0))', ((1, 0), (0, 2))),
        ('(64,128):(1@1,1@0)', ((0,),)),
        ('(64,128):(1@1,4@0)', ((0,),)),
        ('64:1@2', ()),
        ('(64,(128,2)):(1@1,(1@0,2@0))', ((0,),)),
    ))
    def test_only_outer_descriptor_modes(self, basis, expected):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            text = ('!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 131072, '
                    f'tma_gbasis = <"{basis}">, tma_format = F16_RN>')
            layout, bits = _atom_layout(ir.Type.parse(text))
            self.assertEqual(bits, 16)
            paths = tuple(dict.fromkeys(path for group in tuple(tuple(dict.fromkeys(paths)) for paths in _basis_groups(layout)[1:]) for path in group))
            self.assertEqual(paths, expected)

    @parametrize("change", ("missing", "duplicate", "unknown", "packed"))
    def test_incomplete_or_unsupported_type_contract_declines(self, change):
        text = ('!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 131072, '
                'tma_gbasis = <"(64,128):(1@1,1@0)">, tma_format = F16_RN>')
        replacements = {
            "missing": text.replace('tma_gbasis = <"(64,128):(1@1,1@0)">, ', ""),
            "duplicate": text.replace("tma_format =", 'tma_gbasis = <"128:1@0">, tma_format ='),
            "unknown": text[:-1] + ", unexpected = 1>",
            "packed": text.replace("F16_RN", "U4_UNPACK_U8"),
        }
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            with self.assertRaises((ValueError, ir.MLIRError)):
                _atom_layout(ir.Type.parse(replacements[change]))

    def test_original_constructor_authority_is_immutable(self):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = source_module()
            host, = module.body.operations
            constructors = [view.operation for view in host.regions[0].blocks[0].operations
                            if view.operation.name == "cute_nvgpu.atom.make_non_exec_tiled_tma_load"]
            before = str(module)
            requirement, other = read_tma_requirements(constructors[0])[0]
            self.assertEqual((requirement.path, other.path), ((0,), (2,)))
            self.assertEqual((requirement.old_bits, requirement.new_bits, requirement.divisor), (16, 16, 8))
            self.assertEqual(requirement.tensor, constructors[0].operands[0])
            self.assertEqual(requirement.constructor, constructors[0])
            self.assertNotEqual(requirement.tensor, read_tma_requirements(constructors[1])[0][0].tensor)
            with self.assertRaises(FrozenInstanceError):
                requirement.divisor = 1
            self.assertEqual(str(module), before)

    def test_source_recheck_without_active_location(self):
        with ir.Context(), ir.raw_values():
            with ir.Location.unknown():
                module = source_module()
                host, = module.body.operations
                constructor = next(view.operation for view in host.regions[0].blocks[0].operations
                                   if view.operation.name == "cute_nvgpu.atom.make_non_exec_tiled_tma_load")
            before = str(module)
            requirements = read_tma_requirements(constructor)[0]
            self.assertEqual(tuple(row.path for row in requirements), ((0,), (2,)))
            self.assertEqual(str(module), before)

    @parametrize("nested", (False, True))
    def test_derived_view_projection_keeps_original_stride_expression(self, nested):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = source_module()
            host, = module.body.operations
            constructor = next(view.operation for view in host.regions[0].blocks[0].operations
                               if view.operation.name == "cute_nvgpu.atom.make_non_exec_tiled_tma_load")
            if nested:
                typ = ir.Type.parse('!cute.memref<f16, gmem, align<16>, "((1,128),64,1):((?{i64},?{i64}),1,?{i64})">')
                nested_module, constructor = make_constructor(constructor, typ, "F16_RN", nested=True)
                self.assertTrue(nested_module.operation.verify())
            requirement = read_tma_requirements(constructor)[0][0]
            self.assertEqual(requirement.path, (0, 1) if nested else (0,))
            before, numeric = compile_projection(requirement, derived=True)
            self.assertIn("cute.make_view", before)
            self.assertIn("cute.recast_layout", before)
            resolve = mock.Mock(return_value=NumericSource(IntExpr("boxed", 0), 128, 256))
            lowered = lower_numeric(numeric, resolve)
            if nested:
                self.assertIn("mode = [0, 1]", before)
                resolve.assert_called_once_with(0, (1, 1, 1))
            else:
                resolve.assert_called_once_with(0, (1, 1, 0))
            records = SimpleNamespace(input_names=("stride",), integer_inputs=(IntegerInput("stride", 0),))
            for stride in (128, 136, 137, 144):
                tape = _NumericProgram(records, (stride,))
                actual = tape.values[tape.add(lowered.values[0].expression)]
                self.assertEqual(actual, stride + 8)
                self.assertEqual(actual % requirement.divisor == 0, stride % 8 == 0)

    @parametrize("format_name,bits,divisor", (("F16_RN", 16, 8), ("F32_RN", 32, 4)))
    def test_static_stride_and_recast_width(self, format_name, bits, divisor):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            original = source_module()
            host, = original.body.operations
            template = next(view.operation for view in host.regions[0].blocks[0].operations
                            if view.operation.name == "cute_nvgpu.atom.make_non_exec_tiled_tma_load")
            typ = ir.Type.parse('!cute.memref<f16, gmem, align<16>, "(128,64,1):(144,1,18432)">')
            module, constructor = make_constructor(template, typ, format_name)
            self.assertTrue(module.operation.verify())
            requirements = read_tma_requirements(constructor)[0]
            requirement = next(row for row in requirements if row.path == (0,) and row.new_bits == bits)
            self.assertEqual((requirement.old_bits, requirement.new_bits, requirement.divisor), (16, bits, divisor))
            _, numeric = compile_projection(requirement)
            resolve = mock.Mock(side_effect=AssertionError("Static stride requested a runtime source"))
            lowered = lower_numeric(numeric, resolve)
            self.assertEqual(lowered.values[0].expression, IntExpr("constant", 144 * 16 // bits))
            resolve.assert_not_called()

    def test_original_byte_alignment_precedes_truncating_recast(self):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            original = source_module()
            host, = original.body.operations
            template = next(view.operation for view in host.regions[0].blocks[0].operations
                            if view.operation.name == "cute_nvgpu.atom.make_non_exec_tiled_tma_load")
            module, constructor = make_constructor(template, template.operands[0].type, "F32_RN")
            self.assertTrue(module.operation.verify())
            requirements = tuple(row for row in read_tma_requirements(constructor)[0] if row.path == (0,))
            self.assertEqual(tuple((row.old_bits, row.new_bits, row.divisor) for row in requirements),
                             ((16, 16, 8), (16, 32, 4)))
            expressions = []
            for requirement in requirements:
                _, numeric = compile_projection(requirement)
                resolve = mock.Mock(return_value=NumericSource(IntExpr("boxed", 0), 128, 256))
                lowered = lower_numeric(numeric, resolve)
                resolve.assert_called_once_with(0, (1, 1, 0))
                expressions.append(lowered.values[0].expression)
            records = SimpleNamespace(input_names=("stride",), integer_inputs=(IntegerInput("stride", 0),))
            for stride, expected in ((144, (True, True)), (145, (False, True)), (152, (True, True))):
                tape = _NumericProgram(records, (stride,))
                values = tuple(tape.values[tape.add(expression)] for expression in expressions)
                self.assertEqual(values, (stride, stride // 2))
                self.assertEqual(tuple(value % row.divisor == 0 for value, row in zip(values, requirements)), expected)

    @parametrize("basis", (
        '(64,((128,2),)):(1@1,((1@0,1@2),))',
        '(64,((128,),2)):(1@1,((1@0,),1@2))',
    ))
    def test_nested_descriptor_contributions_are_not_flattened(self, basis):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            text = ('!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 131072, '
                    f'tma_gbasis = <"{basis}">, tma_format = F16_RN>')
            layout, _ = _atom_layout(ir.Type.parse(text))
            with self.assertRaisesRegex(ValueError, "scalar basis contributions"):
                _basis_groups(layout)

    def test_rank_one_basis_wrappers_preserve_nested_source_mode(self):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            text = ('!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 131072, '
                    'tma_gbasis = <"(64,((128,),)):(1@1,((1@0@1,),))">, tma_format = F16_RN>')
            layout, _ = _atom_layout(ir.Type.parse(text))
            self.assertEqual(_basis_groups(layout), (((1,),), ((1, 0),)))

    @parametrize("format_name,bits", (("F16_RN", 16), ("F32_RN", 32)))
    def test_recast_dimension_projection_uses_compiler_shape(self, format_name, bits):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            original = source_module()
            host, = original.body.operations
            template = next(view.operation for view in host.regions[0].blocks[0].operations
                            if view.operation.name == "cute_nvgpu.atom.make_non_exec_tiled_tma_load")
            typ = ir.Type.parse('!cute.memref<f16, gmem, align<16>, "(128,64,1):(144,1,18432)">')
            module, constructor = make_constructor(template, typ, format_name)
            self.assertTrue(module.operation.verify())
            _, dimensions = read_tma_requirements(constructor)
            self.assertEqual(tuple(row.path for row in dimensions), ((1,), (0,)))
            for row, expected in zip(dimensions, (64 * 16 // bits, 128), strict=True):
                self.assertEqual((row.old_bits, row.new_bits, row.grouped), (16, bits, False))
                _, numeric = compile_projection(row, property="shape")
                resolve = mock.Mock(side_effect=AssertionError("Static shape requested a runtime source"))
                lowered = lower_numeric(numeric, resolve)
                self.assertEqual(lowered.values[0].expression, IntExpr("constant", expected))
                resolve.assert_not_called()

    def test_grouped_dimension_sources_keep_shape_and_stride_order(self):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            original = source_module()
            host, = original.body.operations
            template = next(view.operation for view in host.regions[0].blocks[0].operations
                            if view.operation.name == "cute_nvgpu.atom.make_non_exec_tiled_tma_load")
            typ = ir.Type.parse('!cute.memref<f16, gmem, align<16>, "(?{i64},?{i64},?{i64},?{i64},?{i64},?{i64}):(?{i64},1,?{i64},?{i64},?{i64},?{i64})">')
            module, constructor = make_constructor(template, typ, "F16_RN")
            self.assertTrue(module.operation.verify())
            _, rows = read_tma_requirements(constructor)
            self.assertEqual(tuple((row.path, row.group, row.grouped) for row in rows),
                             (((1,), 0, False), ((0,), 1, False), ((2,), 2, False),
                              ((3,), 3, False), ((4,), 4, True), ((5,), 4, True)))
            for row in rows:
                _, numeric = compile_projection(row, property="shape")
                resolve = mock.Mock(return_value=NumericSource(IntExpr("boxed", 0), 0, 1 << 40))
                lower_numeric(numeric, resolve)
                resolve.assert_called_once_with(0, (1, 0, row.path[0]))
                if row.grouped:
                    _, numeric = compile_projection(row, property="stride")
                    resolve.reset_mock()
                    lower_numeric(numeric, resolve)
                    resolve.assert_called_once_with(0, (1, 1, row.path[0] - 1))


if __name__ == "__main__":
    run_tests()
