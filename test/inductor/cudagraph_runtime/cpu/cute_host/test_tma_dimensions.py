"""Typed CuTe descriptor dimensions must retain the CUDA globalDim domain."""

from contextlib import nullcontext
import ctypes
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

from torch._inductor.runtime._cudagraph._sdk import activate

activate()

from cutlass._mlir import ir, passmanager
from cutlass._mlir.dialects import cute, func
from cutlass.cute import core
import torch
from torch._inductor.runtime._cudagraph.api import InputContract, IntegerRange, TensorInput
from torch._inductor.runtime._cudagraph.cute_adapter import _integer_requirement_guards, _tma_dimension_guards, _tma_stride_guards
from torch._inductor.runtime._cudagraph.direct_host import _direct_origin
from torch._inductor.runtime._cudagraph.extraction import trace_host
from torch._inductor.runtime._cudagraph.frontend import lower_terminal
from torch._inductor.runtime._cudagraph.guard_export import prepare_guard
from torch._inductor.runtime._cudagraph._compiler.cfg_values import read_cfg_function
from torch._inductor.runtime._cudagraph._compiler.cute_bridge.numeric import lower_numeric, NumericSource
from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.artifact import TmaDimensionDomain, TmaStrideDomain
from torch._inductor.runtime._cudagraph._compiler.decoded_values import prepare_decodings
from torch._inductor.runtime._cudagraph._compiler.entry_signature import RuntimeRequirement
from torch._inductor.runtime._cudagraph._compiler.overflow_properties import bind_properties
from torch._inductor.runtime._cudagraph._compiler.owned_numeric import freeze_numeric
from torch._inductor.runtime._cudagraph._compiler.tma_requirements import _atom_layout, read_tma_requirements
from torch._inductor.runtime.cudagraph_arg_mapping import IntExpr
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


WORKTREE = next(parent for parent in Path(__file__).resolve().parents if (parent / "torch/_inductor").is_dir())
sys.path.insert(0, str(WORKTREE / "test/inductor/cudagraph_runtime/cpu/cute_host"))
from test_tma_requirements import compile_projection, make_constructor, source_module


I64_MAX = (1 << 63) - 1


def identity(box):
    dimension, tensor = box
    box.clear()
    return tensor, dimension


def compile_shape(constructor, descriptor_axis):
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        function = func.FuncOp("shape_probe", ([constructor.operands[0].type], [ir.IntegerType.get_signless(64)]))
    block = function.add_entry_block()
    with ir.InsertionPoint(block):
        basis_type, new_bits = _atom_layout(constructor.results[0].type)
        basis = core._unpack_x_tuple(cute.GetStrideOp(cute.StaticOp(basis_type).result).result)
        mode = basis[descriptor_axis]
        if not isinstance(mode, core.ScaledBasis):
            raise AssertionError("Expected one compiler source mode for the descriptor dimension")
        path = tuple(mode.mode)
        old_bits = core.Numeric.from_mlir_type(constructor.operands[0].type.value_type).width
        layout = cute.GetLayoutOp(block.arguments[0]).result
        recast = cute.RecastLayoutOp(new_bits, old_bits, layout).dst
        shape = cute.GetShapeOp(recast).result
        selected_type = shape.type.get_op_res_type(mode=list(path))
        selected = cute.GetOp(selected_type, shape, mode=list(path)).result
        value, = cute.GetScalarsOp(selected).results
        if str(value.type) != "i64":
            raise AssertionError("Expected the exact i64 recast shape projection")
        func.ReturnOp([value])
    if not module.operation.verify():
        raise AssertionError("Expected valid typed compiler shape projection")
    passmanager.PassManager.parse("builtin.module(cute-to-nvvm{enable-cuda-dialect cubin-chip=sm_100a})").run(module.operation)
    cfg = read_cfg_function(module, "shape_probe")
    return path, freeze_numeric(bind_properties(cfg), prepare_decodings(cfg), (0,))


@instantiate_parametrized_tests
class TestTmaDimensions(TestCase):
    @parametrize("descriptor_axis", (0, 1))
    @parametrize("dimension, expected", ((0, 0), (1 << 32, 1), ((1 << 32) + 1, 0)))
    def test_encoded_dimension_domain(self, descriptor_axis, dimension, expected):
        source_axis = 1 - descriptor_axis
        shape = [128, 64, 1]
        example = torch.empty_strided(shape, (64, 1, 8192), dtype=torch.float16)
        example_dimension = shape[source_axis]
        shape[source_axis] = IntExpr("boxed", 0)
        contract = InputContract(
            ("integer", "tensor"),
            (TensorInput(1, torch.float16, tuple(shape), (64, 1, 8192)),),
            (IntegerRange(0, 0, I64_MAX),), device_index=0,
        )
        origin, _ = _direct_origin(identity, contract)
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            trace = trace_host(identity, contract, [example_dimension, example], (), None,
                               direct=True, context_factory=lambda state: nullcontext())
        trace = replace(trace, compiler_binding=origin)
        symbol, = trace.symbol_sources
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            original = source_module()
            host, = original.body.operations
            template = next(view.operation for view in host.regions[0].blocks[0].operations
                            if view.operation.name == "cute_nvgpu.atom.make_non_exec_tiled_tma_load")
            dimensions = ["128", "64", "1"]
            dimensions[source_axis] = "?{i64}"
            typ = ir.Type.parse('!cute.memref<f16, gmem, align<16>, "(' + ",".join(dimensions) + '):(64,1,8192)">')
            module, constructor = make_constructor(template, typ, "F16_RN")
            self.assertTrue(module.operation.verify())
            path, projected_shape = compile_shape(constructor, descriptor_axis)
            self.assertEqual(path, (source_axis,))
            resolve_shape = mock.Mock(return_value=NumericSource(IntExpr("boxed", 0), 0, I64_MAX))
            lowered = lower_numeric(projected_shape, resolve_shape)
            resolve_shape.assert_called_once_with(0, (1, 0))
            self.assertEqual(lowered.values[0].expression, IntExpr("boxed", 0))
            self.assertEqual(lowered.obligations, ())
            requirements, dimensions = read_tma_requirements(constructor)
            self.assertTrue(all(row.constructor == constructor and row.tensor == constructor.operands[0]
                                for row in requirements))
            consumers = []
            for index, requirement in enumerate(requirements):
                _, numeric = compile_projection(requirement)
                consumers.append(SimpleNamespace(site_id=0, role="tma_stride", index=index,
                                                  result_type="i64", numeric=numeric))
            groups = {}
            for index, requirement in enumerate(requirements):
                if requirement.group is not None:
                    groups.setdefault((requirement.group, requirement.new_bits // 8), []).append(index)
            site = SimpleNamespace(site_id=0, tma_stride_divisors=tuple(row.divisor for row in requirements),
                                   tma_stride_domains=tuple(TmaStrideDomain(tuple(indices), key[1])
                                                            for key, indices in groups.items()))
            dimension_groups = {}
            for index, requirement in enumerate(dimensions):
                dimension_groups.setdefault((requirement.group, requirement.grouped), []).append(index)
                _, numeric = compile_projection(requirement, property="shape")
                consumers.append(SimpleNamespace(site_id=0, role="tma_shape", index=index,
                                                  result_type="i64", numeric=numeric))
                self.assertFalse(requirement.grouped)
            site.tma_dimension_domains = tuple(TmaDimensionDomain(tuple(indices), key[1])
                                                for key, indices in dimension_groups.items())
            resolve_stride = mock.Mock(side_effect=AssertionError("Static stride requested a runtime source"))
            guards, obligations = _tma_stride_guards(SimpleNamespace(consumers=tuple(consumers)), site,
                                                     SimpleNamespace(numeric=resolve_stride), {0: symbol})
            resolve_stride.assert_not_called()
            self.assertEqual(obligations, ())
            resolve_shape.reset_mock()
            dimension_guards, dimension_obligations = _tma_dimension_guards(
                SimpleNamespace(consumers=tuple(consumers)), site,
                SimpleNamespace(numeric=resolve_shape), {0: symbol})
            resolve_shape.assert_called_once_with(0, (1, 0))
            self.assertEqual(dimension_obligations, ())
            guards += dimension_guards
            guards += _integer_requirement_guards(RuntimeRequirement("integer_range", (), 0, I64_MAX), symbol)
        program = lower_terminal(trace, (), extra_guards=guards)
        self.addCleanup(program.close)
        guard = prepare_guard(program, [example_dimension, example], required=True)
        self.assertTrue(set(guard.boxed_integer_indices).issubset({0}))
        predicate = ctypes.CFUNCTYPE(ctypes.c_int8, ctypes.POINTER(ctypes.c_int64),
                                    ctypes.POINTER(ctypes.c_double))(guard.function_address)
        def accepts(value):
            inputs = [value, example]
            values = [inputs[index] for index in guard.boxed_integer_indices]
            values.extend(inputs[index].data_ptr() for index in guard.boxed_pointer_indices)
            values.extend(inputs[index].storage_offset() for index in guard.boxed_storage_offset_indices)
            return predicate((ctypes.c_int64 * len(values))(*values), None)

        self.assertEqual(accepts(example_dimension), 1)
        # Only a compiled predicate receives invalid dimensions; no descriptor is encoded or launched.
        self.assertEqual(accepts(dimension), expected)


if __name__ == "__main__":
    run_tests()
