"""Read TMA host metadata and preserve fixed launch attributes."""

from contextlib import ExitStack, nullcontext
import ctypes
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from torch._inductor.runtime._cudagraph._sdk import activate

activate()
ROOT = Path(__file__).resolve().parent / "fixtures"

from cuda.bindings import driver
from torch._inductor.runtime._cudagraph._compiler.cute_bridge.provider import CuTeKernelOwner
from cutlass._mlir import ir
from torch._inductor.runtime._cudagraph._compiler import emitter_v2
from torch._inductor.runtime._cudagraph._compiler.emitter_v2 import _snapshot, _validate_tree, emit_scalar_helper
from torch._inductor.runtime._cudagraph._compiler.source_dispatch import _pure
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


ACTUAL_HOST_OPS = {
    "cute.coalesce", "cute.cosize", "cute.deref_arith_tuple_iter", "cute.dice", "cute.get",
    "cute.make_coord", "cute.make_identity_layout", "cute.mma.make_fragment", "cute.recast_iter",
    "cute.recast_layout", "cute.size", "cute.slice", "cute.tiled.mma.partition_shape",
    "cute.tiled_divide", "cute.tuple_sub", "cute_nvgpu.atom.make_non_exec_tiled_tma_load",
    "cute_nvgpu.atom.make_non_exec_tiled_tma_store", "vector.from_elements",
}


def _host_prefix():
    data = (ROOT / "observation_gemm_attempt1_artifacts/source_host.mlir").read_bytes()
    if hashlib.sha256(data).hexdigest() != "c481a895eb3155bfe961dd4581b5132336a30375d6fee7d2e8ffe91acba278e0":
        raise RuntimeError("The observed upstream host source changed")
    lines = data.decode().splitlines(True)
    cut = next(index for index, line in enumerate(lines) if " = cute.kernel_smem_size " in line)
    return "module {\n" + "".join(lines[:cut]) + "  return %126 : i32\n}\n}\n"


@instantiate_parametrized_tests
class TestTmaHostOps(TestCase):
    def test_actual_metadata_prefix_and_scalar_grid_slice(self):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = ir.Module.parse(_host_prefix())
            self.assertTrue(module.operation.verify())
            before = _snapshot(module.operation)
            host = next(iter(module.body.operations)).operation
            operations = tuple(view.operation for view in host.regions[0].blocks[0].operations)
            seen = set()
            for op in operations[:-1]:
                _pure(module, op, {})
                seen.add(op.name)
            self.assertEqual(seen & ACTUAL_HOST_OPS, ACTUAL_HOST_OPS)
            self.assertEqual(_snapshot(module.operation), before)
            helper = emit_scalar_helper(module, host, tuple(operations[-1].operands), "actual_grid_slice")
            helper.check()
            copied = tuple(view.operation for view in helper.operation.regions[0].blocks[0].operations)
            self.assertIn("cute.tuple_sub", {op.name for op in copied})
            self.assertNotIn("cute_nvgpu.atom.make_non_exec_tiled_tma_load", {op.name for op in copied})
            self.assertEqual(helper.source_ids, (0, 1, 2, 3))
            self.assertEqual(tuple(value.type for value in helper.operation.regions[0].blocks[0].arguments),
                             tuple(value.type for value in host.regions[0].blocks[0].arguments))
            self.assertTrue(module.operation.verify())

    def test_unrecognized_constructor_attribute_is_not_discarded(self):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = ir.Module.parse(_host_prefix())
            self.assertTrue(module.operation.verify())
            host = next(iter(module.body.operations)).operation
            constructor = next(view.operation for view in host.regions[0].blocks[0].operations
                               if view.operation.name == "cute_nvgpu.atom.make_non_exec_tiled_tma_load")
            constructor.attributes["unexpected"] = ir.UnitAttr.get()
            with self.assertRaisesRegex(ValueError, "Unsupported source dependency or effect"):
                _validate_tree(module, constructor, {})

    @parametrize("cluster", (None, (2, 3, 1)))
    def test_capture_launch_transports_exact_fixed_cluster(self, cluster):
        owner = object.__new__(CuTeKernelOwner)
        owner._phase = "ready"
        owner._block, owner._shared, owner._cluster = (128, 1, 1), 196736, cluster
        owner._site = object()
        owner._loader = SimpleNamespace(_graphs=(object(),), context=7,
                                        validate=Mock(return_value=SimpleNamespace(function=123)))
        grid, stream, images = (4, 6, 1), 456, (b"abcdefgh", bytes(range(16)))
        success = driver.CUresult.CUDA_SUCCESS
        observed = []

        def launch_ex(config, function, parameters, extra):
            self.assertEqual(function, 123)
            self.assertEqual((config.gridDimX, config.gridDimY, config.gridDimZ), grid)
            self.assertEqual((config.blockDimX, config.blockDimY, config.blockDimZ), owner._block)
            self.assertEqual((config.sharedMemBytes, int(config.hStream), extra), (owner._shared, stream, 0))
            self.assertEqual(config.numAttrs, 1)
            attribute, = config.attrs
            self.assertEqual(attribute.id, driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION)
            self.assertEqual((attribute.value.clusterDim.x, attribute.value.clusterDim.y, attribute.value.clusterDim.z), cluster)
            pointers = ctypes.cast(parameters, ctypes.POINTER(ctypes.c_void_p))
            observed.append(tuple(ctypes.string_at(pointers[index], len(value)) for index, value in enumerate(images)))
            return (success,)

        with patch.object(CuTeKernelOwner, "_locked", lambda self: nullcontext()), \
             patch.object(driver, "cuStreamGetCtx", return_value=(success, 7)), \
             patch.object(driver, "cuStreamIsCapturing", return_value=(success, driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE)), \
             patch.object(driver, "cuLaunchKernel", return_value=(success,)) as plain, \
             patch.object(driver, "cuLaunchKernelEx", side_effect=launch_ex) as extended:
            owner.launch(images, grid, stream)
            if cluster is None:
                plain.assert_called_once()
                extended.assert_not_called()
                self.assertEqual(plain.call_args.args[:9], (123, *grid, *owner._block, owner._shared, stream))
            else:
                plain.assert_not_called()
                extended.assert_called_once()
                self.assertEqual(observed, [images])


class TestSharedMemoryQuerySnapshots(TestCase):
    def setUp(self):
        super().setUp()
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        contexts.enter_context(ir.Context())
        contexts.enter_context(ir.Location.unknown())
        contexts.enter_context(ir.raw_values())
        self.module = ir.Module.parse("""module {
          func.func @host() -> i64 {
            %size = cute.kernel_smem_size @kernels::@body : i64
            return %size : i64
          }
          gpu.module @kernels {
            cuda.kernel @body()
          }
        }""")
        self.host = next(iter(self.module.body.operations)).operation
        gpu_module = tuple(self.module.body.operations)[1].operation
        self.kernel = next(iter(gpu_module.regions[0].blocks[0].operations)).operation
        self.query = next(iter(self.host.regions[0].blocks[0].operations)).operation

    def test_dispatch_validation_only_needs_kernel_identity(self):
        before = _snapshot(self.module.operation)
        queried = {}
        with patch.object(emitter_v2, "_snapshot", wraps=emitter_v2._snapshot) as snapshot:
            for _ in range(16):
                _pure(self.module, self.query, queried)
        self.assertEqual(tuple(queried), (self.kernel,))
        self.assertEqual(snapshot.call_count, 0)
        self.assertEqual(_snapshot(self.module.operation), before)

    def test_dispatch_validation_rejects_unknown_kernel(self):
        self.query.attributes["kernel_name"] = ir.SymbolRefAttr.get(["kernels", "missing"])
        with self.assertRaisesRegex(ValueError, "does not identify an existing cuda.kernel"):
            _pure(self.module, self.query, {})

    def test_owned_helper_still_checks_original_kernel(self):
        helper = emit_scalar_helper(self.module, self.host, (self.query.results[0],), "smem")
        helper.check()
        self.kernel.attributes["test_marker"] = ir.UnitAttr.get()
        with self.assertRaisesRegex(RuntimeError, "changed"):
            helper.check()


if __name__ == "__main__":
    run_tests()
