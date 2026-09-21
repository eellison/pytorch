"""Continuation preserves sparse stride consumers for actual grouped dimensions."""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

from torch._inductor.runtime._cudagraph._sdk import activate

activate()

from cutlass._mlir import ir
from torch._inductor.runtime._cudagraph._compiler.continuation import _uses
from torch._inductor.runtime._cudagraph._compiler.source_dispatch import check_dispatch_source
from torch._inductor.runtime._cudagraph._compiler.tma_requirements import read_tma_requirements
from torch.testing._internal.common_utils import run_tests, TestCase


WORKTREE = next(parent for parent in Path(__file__).resolve().parents if (parent / "torch/_inductor").is_dir())
sys.path.insert(0, str(WORKTREE / "test/inductor/cudagraph_runtime/cpu/cute_host"))
from test_tma_requirements import make_constructor
from test_tma_stride_transport import _source


class TestTmaDimensionConsumers(TestCase):
    def test_grouped_dimensions_have_exact_sparse_stride_indices(self):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            original, host, metadata = _source()
            source = check_dispatch_source(original, host.attributes["sym_name"].value, metadata)
            site, = source.sites
            template = site.tma_dimensions[0].constructor
            typ = ir.Type.parse('!cute.memref<f16, gmem, align<16>, "(?{i64},?{i64},?{i64},?{i64},?{i64},?{i64}):(?{i64},1,?{i64},?{i64},?{i64},?{i64})">')
            module, constructor = make_constructor(template, typ, "F16_RN")
            self.assertTrue(module.operation.verify())
            _, dimensions = read_tma_requirements(constructor)
            self.assertEqual(tuple((row.group, row.grouped) for row in dimensions),
                             ((0, False), (1, False), (2, False), (3, False), (4, True), (4, True)))
            # Exercise enumeration only; helper/source ownership is tested with genuine dispatch sites separately.
            enumeration = SimpleNamespace(predicate=None, sites=(replace(site, tma_dimensions=dimensions),))
            uses = tuple(row for row in _uses(enumeration) if row[1] in ("tma_shape", "tma_dimension_stride"))
            self.assertEqual(tuple((role, index) for _, role, index, _ in uses),
                             (("tma_shape", 0), ("tma_shape", 1), ("tma_shape", 2), ("tma_shape", 3),
                              ("tma_shape", 4), ("tma_dimension_stride", 4),
                              ("tma_shape", 5), ("tma_dimension_stride", 5)))
            self.assertTrue(all(value == dimensions[index].tensor for _, _, index, value in uses))


if __name__ == "__main__":
    run_tests()
