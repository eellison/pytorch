"""Preserve constructor-specific TMA requirements through source helper projection."""

from dataclasses import replace
import hashlib
from pathlib import Path
import re

from torch._inductor.runtime._cudagraph._sdk import activate

activate()

from cutlass._mlir import ir
from cutlass._mlir.dialects import cute
from torch._inductor.runtime._cudagraph._compiler.accessors import _snapshot
from torch._inductor.runtime._cudagraph._compiler.continuation import _uses
from torch._inductor.runtime._cudagraph._compiler.entry_signature import MetadataSnapshot, ParameterMetadata
from torch._inductor.runtime._cudagraph._compiler.helpers import _slice, emit_dispatch_helpers, ScalarRequest
from torch._inductor.runtime._cudagraph._compiler.source_dispatch import _tma_requirements, check_dispatch_source
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


WORKTREE = next(parent for parent in Path(__file__).resolve().parents if (parent / "torch/_inductor").is_dir())
FIXTURE = WORKTREE / "test/inductor/cudagraph_runtime/cpu/cute_host/fixtures/observation_gemm_attempt1_artifacts/source_host.mlir"


def _source():
    data = FIXTURE.read_bytes()
    if hashlib.sha256(data).hexdigest() != "c481a895eb3155bfe961dd4581b5132336a30375d6fee7d2e8ffe91acba278e0":
        raise AssertionError("Expected the unchanged compiler host fixture")
    text = data.decode()
    launch, = (line for line in text.splitlines() if " = cuda.launch_ex " in line)
    callee = re.search(r"@kernels::@([A-Za-z_0-9]+)", launch).group(1)
    argument_types = launch.split(": !cuda.launch_cfg<max_attrs = 17>, (", 1)[1].rsplit(") -> !cuda.result", 1)[0]
    module = ir.Module.parse(f"module {{\n{text}\ngpu.module @kernels {{\n  cuda.kernel @{callee}({argument_types})\n}}\n}}")
    host = next(iter(module.body.operations)).operation
    name = host.attributes["sym_name"].value
    params = tuple(ParameterMetadata("Stream" if index == 3 else "Tensor", f"arg{index}", index, index)
                   for index in range(4))
    metadata = MetadataSnapshot(name, name, "Abi.Tbd", (), params, ParameterMetadata("Unit", "", None, None))
    return module, host, metadata


def _derived_source():
    module, host, metadata = _source()
    block, = host.regions[0].blocks
    constructor = next(view.operation for view in block.operations
                       if view.operation.name == "cute_nvgpu.atom.make_non_exec_tiled_tma_load")
    with ir.InsertionPoint(constructor):
        pointer = cute.GetIterOp(block.arguments[0]).result
        layout = cute.GetLayoutOp(block.arguments[1]).result
        tensor = cute.MakeViewOp(block.arguments[0].type, pointer, layout=layout).result
    constructor.operands[0] = tensor
    source = check_dispatch_source(module, host.attributes["sym_name"].value, metadata)
    return source, constructor, tensor


@instantiate_parametrized_tests
class TestTmaStrideTransport(TestCase):
    def test_constructor_dependencies_are_scoped_and_deduplicated(self):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module, host, metadata = _source()
            source = check_dispatch_source(module, host.attributes["sym_name"].value, metadata)
            site, = source.sites
            constructors = tuple(view.operation for view in site.block.operations
                                 if view.operation.name in {"cute_nvgpu.atom.make_non_exec_tiled_tma_load",
                                                            "cute_nvgpu.atom.make_non_exec_tiled_tma_store"})
            self.assertEqual(len(constructors), 3)
            self.assertEqual(len(site.tma_strides), 6)
            for constructor, divisor in zip(constructors, (8, 8, 4), strict=True):
                requirements = _tma_requirements(tuple(constructor.results))[0]
                self.assertEqual(tuple(row.path for row in requirements), ((0,), (2,)))
                self.assertEqual(tuple(row.divisor for row in requirements), (divisor, divisor))
                self.assertTrue(all(row.constructor == constructor and row.tensor == constructor.operands[0]
                                    for row in requirements))
            self.assertEqual(_tma_requirements(source.arguments)[0], ())

    def test_dimension_consumers_preserve_constructor_axes(self):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module, host, metadata = _source()
            source = check_dispatch_source(module, host.attributes["sym_name"].value, metadata)
            site, = source.sites
            self.assertEqual(len(site.tma_dimensions), 9)
            for start in (0, 3, 6):
                rows = site.tma_dimensions[start:start + 3]
                self.assertEqual(tuple((row.path, row.group, row.grouped) for row in rows),
                                 (((1,), 0, False), ((0,), 1, False), ((2,), 2, False)))
                self.assertTrue(all(row.constructor == rows[0].constructor for row in rows))
            uses = tuple(row for row in _uses(source) if row[1] in ("tma_shape", "tma_dimension_stride"))
            self.assertEqual(tuple((row[1], row[2]) for row in uses), tuple(("tma_shape", i) for i in range(9)))
            self.assertTrue(all(value == site.tma_dimensions[index].tensor for _, _, index, value in uses))

    @parametrize("role", ("tma_stride", "tma_shape"))
    def test_helper_preserves_derived_view_layout_and_original_ids(self, role):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            source, constructor, tensor = _derived_source()
            site, = source.sites
            self.assertEqual(site.tma_strides[0].constructor, constructor)
            self.assertEqual(site.tma_strides[0].tensor, tensor)
            before = _snapshot(source.host)
            request = ScalarRequest("project_derived_property", role, 0, tensor, site)
            helpers = emit_dispatch_helpers(source, (request,))
            helper, = helpers.helpers
            helper.check()
            self.assertEqual(helper.source_ids, (0, 1, 2, 3))
            self.assertEqual(helper.result_type, "i64")
            self.assertEqual(helper.output_value, tensor)
            self.assertEqual(_snapshot(source.host), before)
            block, = helper.operation.regions[0].blocks
            views = tuple(view.operation for view in block.operations if view.operation.name == "cute.make_view")
            view, = views
            self.assertEqual(view.operands[0].owner.operands[0], block.arguments[0])
            self.assertEqual(view.operands[1].owner.operands[0], block.arguments[1])
            self.assertTrue(any(op.operation.name == "cute.get_layout" and op.operation.operands[0] == view.results[0]
                                for op in block.operations))
            self.assertNotIn(constructor.name, {op.operation.name for op in block.operations})

    @parametrize("fault", ("tensor", "site"))
    @parametrize("role", ("tma_stride", "tma_shape"))
    def test_projection_rejects_changed_source_identity(self, fault, role):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            source, _, tensor = _derived_source()
            site, = source.sites
            request = ScalarRequest("wrong_property", role, 0, tensor, site)
            if fault == "tensor":
                request = replace(request, value=source.arguments[0])
            else:
                request = replace(request, site=replace(site))
            with self.assertRaisesRegex(ValueError, "original constructor operand|not owned"):
                _slice(source, request)


if __name__ == "__main__":
    run_tests()
