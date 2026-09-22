"""TMA outer-stride bounds survive an independently symbolic singleton stride."""

from contextlib import nullcontext
import ctypes
from dataclasses import replace
from functools import partial
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

from torch._inductor.runtime._cudagraph._sdk import activate

activate()

from cutlass._mlir import ir
import sympy
import torch
from torch._inductor.runtime._cudagraph.api import InputContract, IntegerRange, TensorInput
from torch._inductor.runtime._cudagraph.cute_adapter import _integer_requirement_guards, _symbolic, _tma_stride_guards
from torch._inductor.runtime._cudagraph.direct_host import _direct_origin
from torch._inductor.runtime._cudagraph.extraction import trace_host
from torch._inductor.runtime._cudagraph.frontend import lower_terminal
from torch._inductor.runtime._cudagraph.guard_export import prepare_guard
from torch._inductor.runtime._cudagraph._compiler.cute_bridge.numeric import NumericSource
from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.artifact import TmaStrideDomain
from torch._inductor.runtime._cudagraph._compiler.entry_signature import RuntimeRequirement
from torch._inductor.runtime._cudagraph._compiler.tma_requirements import (
    _atom_layout, _basis_groups, read_tma_requirements,
)
from torch._inductor.runtime.cudagraph_arg_mapping import IntExpr
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


WORKTREE = next(parent for parent in Path(__file__).resolve().parents if (parent / "torch/_inductor").is_dir())
sys.path.insert(0, str(WORKTREE / "test/inductor/cudagraph_runtime/cpu/cute_host"))
from test_tma_requirements import compile_projection, make_constructor, source_module


I64_MAX = (1 << 63) - 1


def identity(box):
    stride, tensor = box
    box.clear()
    return tensor, stride


def grouped_identity(box):
    left, right, tensor = box
    box.clear()
    return tensor, left, right


@instantiate_parametrized_tests
class TestTmaStrideDomain(TestCase):
    @parametrize("stride, expected", (((1 << 39) - 8, 1), (1 << 39, 0), (1 << 40, 0)))
    def test_singleton_outer_stride_has_cuda_byte_range(self, stride, expected):
        backing = torch.empty(128 * 64, dtype=torch.float16)
        example = backing.as_strided((128, 64, 1), (64, 1, 8192))
        probe = backing.as_strided(example.shape, (64, 1, stride))
        self.assertEqual(probe.data_ptr(), example.data_ptr())
        self.assertEqual(probe.untyped_storage().nbytes(), example.untyped_storage().nbytes())
        self.assertEqual(probe.stride()[2], stride)
        self.assertEqual(stride * probe.element_size() % 16, 0)
        self.assertEqual(expected, int(stride * probe.element_size() < 1 << 40))
        contract = InputContract(
            ("integer", "tensor"),
            (TensorInput(1, torch.float16, (128, 64, 1), (64, 1, IntExpr("boxed", 0))),),
            (IntegerRange(0, 0, I64_MAX),), device_index=0,
        )
        origin, _ = _direct_origin(identity, contract)
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            trace = trace_host(identity, contract, [8192, example], (), None,
                               direct=True, context_factory=lambda state: nullcontext())
        trace = replace(trace, compiler_binding=origin)
        symbol, = trace.symbol_sources

        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = source_module()
            host, = module.body.operations
            constructor = next(view.operation for view in host.regions[0].blocks[0].operations
                               if view.operation.name == "cute_nvgpu.atom.make_non_exec_tiled_tma_load")
            requirements = read_tma_requirements(constructor)[0]
            self.assertEqual(tuple(row.path for row in requirements), ((0,), (2,)))
            self.assertTrue(all(row.old_bits == row.new_bits == 16 for row in requirements))
            consumers = []
            for index, requirement in enumerate(requirements):
                _, numeric = compile_projection(requirement)
                consumers.append(SimpleNamespace(site_id=0, role="tma_stride", index=index,
                                                  result_type="i64", numeric=numeric))

            def resolve(source_index, path):
                self.assertEqual(source_index, 0)
                self.assertIn(path, ((1, 1, 0), (1, 1, 1)))
                if path == (1, 1, 0):
                    return NumericSource(IntExpr("constant", 64), 64, 64)
                return NumericSource(IntExpr("boxed", 0), 0, I64_MAX)

            self.assertEqual(tuple(row.group for row in requirements), (0, 1))
            site = SimpleNamespace(site_id=0, tma_stride_divisors=tuple(row.divisor for row in requirements),
                                   tma_stride_domains=(TmaStrideDomain((0,), 2), TmaStrideDomain((1,), 2)))
            guards, obligations = _tma_stride_guards(SimpleNamespace(consumers=tuple(consumers)), site,
                                                     SimpleNamespace(numeric=resolve), partial(_symbolic, symbols={0: symbol}))
            guards += _integer_requirement_guards(RuntimeRequirement("integer_range", (), 0, I64_MAX), symbol)
            for obligation in obligations:
                self.assertEqual(obligation.expression, IntExpr("boxed", 0))
                guards += (sympy.Ge(symbol, obligation.lower), sympy.Le(symbol, obligation.upper))
        program = lower_terminal(trace, (), extra_guards=guards)
        self.addCleanup(program.close)
        guard = prepare_guard(program, [8192, example], required=True)
        self.assertEqual(guard.boxed_integer_indices, (0,))
        self.assertEqual(guard.boxed_pointer_indices, ())
        self.assertEqual(guard.boxed_storage_offset_indices, ())
        predicate = ctypes.CFUNCTYPE(ctypes.c_int8, ctypes.POINTER(ctypes.c_int64),
                                    ctypes.POINTER(ctypes.c_double))(guard.function_address)
        self.assertEqual(predicate((ctypes.c_int64 * 1)(8192), None), 1)
        # The predicate alone receives the out-of-range descriptor metadata.
        self.assertEqual(predicate((ctypes.c_int64 * 1)(stride), None), expected)

    @parametrize("left, right, expected", (
        ((1 << 39) + 8, (1 << 39) + 16, 1),
        (0, 8192, 1),
        (0, 0, 1),
        (0, 1 << 39, 0),
        (1 << 39, 1 << 40, 0),
    ))
    def test_grouped_outer_stride_uses_encoded_gcd(self, left, right, expected):
        example = torch.empty(128 * 64, dtype=torch.float16).as_strided(
            (128, 64, 1, 1, 1, 1), (64, 1, 8192, 8192, 8192, 8192))
        probe = example.as_strided(example.shape, (64, 1, 8192, 8192, left, right))
        self.assertEqual(probe.data_ptr(), example.data_ptr())
        self.assertEqual(probe.untyped_storage().nbytes(), example.untyped_storage().nbytes())
        contract = InputContract(
            ("integer", "integer", "tensor"),
            (TensorInput(2, torch.float16, tuple(example.shape),
                         (64, 1, 8192, 8192, IntExpr("boxed", 0), IntExpr("boxed", 1))),),
            (IntegerRange(0, 0, I64_MAX), IntegerRange(1, 0, I64_MAX)), device_index=0,
        )
        origin, _ = _direct_origin(grouped_identity, contract)
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            trace = trace_host(grouped_identity, contract, [8192, 8192, example], (), None,
                               direct=True, context_factory=lambda state: nullcontext())
        trace = replace(trace, compiler_binding=origin)
        symbols = {source.value: symbol for symbol, source in trace.symbol_sources.items()}
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            original = source_module()
            host, = original.body.operations
            template = next(view.operation for view in host.regions[0].blocks[0].operations
                            if view.operation.name == "cute_nvgpu.atom.make_non_exec_tiled_tma_load")
            typ = ir.Type.parse('!cute.memref<f16, gmem, align<16>, "(?,?,?,?,?,?):(?{i64},1,?{i64},?{i64},?{i64},?{i64})">')
            module, constructor = make_constructor(template, typ, "F16_RN")
            self.assertTrue(module.operation.verify())
            requirements = read_tma_requirements(constructor)[0]
            self.assertEqual(tuple((row.path, row.group) for row in requirements),
                             (((0,), 0), ((2,), 1), ((3,), 2), ((4,), 3), ((5,), 3)))
            consumers = []
            for index, requirement in enumerate(requirements):
                _, numeric = compile_projection(requirement)
                consumers.append(SimpleNamespace(site_id=0, role="tma_stride", index=index,
                                                  result_type="i64", numeric=numeric))

            def resolve(source_index, path):
                self.assertEqual(source_index, 0)
                self.assertIn(path, tuple((1, 1, index) for index in range(5)))
                index = path[-1]
                if index < 3:
                    value = 64 if index == 0 else 8192
                    return NumericSource(IntExpr("constant", value), value, value)
                return NumericSource(IntExpr("boxed", index - 3), 0, I64_MAX)

            site = SimpleNamespace(site_id=0, tma_stride_divisors=(8,) * 5,
                                   tma_stride_domains=tuple(TmaStrideDomain(indices, 2)
                                                            for indices in ((0,), (1,), (2,), (3, 4))))
            guards, obligations = _tma_stride_guards(SimpleNamespace(consumers=tuple(consumers)), site,
                                                     SimpleNamespace(numeric=resolve), partial(_symbolic, symbols=symbols))
            for obligation in obligations:
                self.assertEqual(obligation.expression.op, "boxed")
                symbol = symbols[obligation.expression.value]
                guards += (sympy.Ge(symbol, obligation.lower), sympy.Le(symbol, obligation.upper))
        program = lower_terminal(trace, (), extra_guards=guards)
        self.addCleanup(program.close)
        guard = prepare_guard(program, [8192, 8192, example], required=True)
        self.assertEqual(set(guard.boxed_integer_indices), {0, 1})
        self.assertEqual(guard.boxed_pointer_indices, ())
        self.assertEqual(guard.boxed_storage_offset_indices, ())
        values = [left, right]
        packed = (ctypes.c_int64 * 2)(*(values[index] for index in guard.boxed_integer_indices))
        predicate = ctypes.CFUNCTYPE(ctypes.c_int8, ctypes.POINTER(ctypes.c_int64),
                                    ctypes.POINTER(ctypes.c_double))(guard.function_address)
        self.assertEqual(predicate(packed, None), expected)

    @parametrize("basis, expected", (
        ('(64,(128,2)):(1@1,(1@0,2@0))', (((0,),),)),
        ('(64,(128,2)):(1@1,(1@0@1,1@2@0))', (((1, 0), (0, 2)),)),
        ('(64,128,2):(1@1,1@0,2@0)', (((0,),), ((0,),))),
    ))
    def test_basis_group_identity_preserves_duplicate_and_nested_modes(self, basis, expected):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            text = ('!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f16, copy_bits = 131072, '
                    f'tma_gbasis = <"{basis}">, tma_format = F16_RN>')
            layout, bits = _atom_layout(ir.Type.parse(text))
            self.assertEqual(bits, 16)
            self.assertEqual(tuple(tuple(dict.fromkeys(paths)) for paths in _basis_groups(layout)[1:]), expected)


if __name__ == "__main__":
    run_tests()
