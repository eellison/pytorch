# Owner(s): ["module: inductor"]

import gc
import hashlib
import weakref
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from torch._inductor.runtime._cudagraph._sdk import activate


activate()

from cuda.bindings import driver

from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime import loading
from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.artifact import (
    ArtifactSite,
    BinaryImage,
    DispatchArtifact,
    NodeFields,
    Parameter,
    Registration,
)
from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.loading import (
    ArtifactKernels,
    CapturedKernel,
)
from torch._inductor.runtime._cudagraph._compiler.cute_bridge import provider
from torch._inductor.runtime._cudagraph._compiler.cute_bridge.provider import (
    CuTeKernelOwner,
    ProviderDeclined,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class _EagerOwner:
    pass


@instantiate_parametrized_tests
class TestCuTeCapturedKernel(TestCase):
    def setUp(self):
        super().setUp()
        self.context, self.device, self.stream, self.function = 7, 2, 29, 123
        self.layout = ((0, 8), (16, 16))
        self.symbol = b"captured_kernel"
        attrs = driver.CUfunction_attribute
        self.attributes = {
            attrs.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: 0,
            attrs.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH: 0,
            attrs.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT: 0,
            attrs.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH: 0,
            attrs.CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET: 0,
            attrs.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: 1024,
            attrs.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: 4096,
        }
        self.success = driver.CUresult.CUDA_SUCCESS
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.open_loaders = tuple(loading._OPEN_LOADERS)
        self.open_owners = tuple(provider._OPEN_OWNERS)
        self.calls = {}
        replies = {
            "cuCtxGetCurrent": lambda: (self.success, self.context),
            "cuCtxGetDevice": lambda: (self.success, self.device),
            "cuStreamGetCtx": lambda stream: (self.success, self.context),
            "cuDeviceGetAttribute": lambda attr, device: (self.success, 65536),
            "cuFuncGetName": lambda function: (self.success, self.symbol),
            "cuFuncGetParamInfo": self.parameter_info,
            "cuFuncGetAttribute": lambda attr, function: (
                self.success,
                self.attributes[attr],
            ),
            "cuStreamSynchronize": lambda stream: (self.success,),
        }
        for name, reply in replies.items():
            self.calls[name] = self.stack.enter_context(
                patch.object(driver, name, side_effect=reply)
            )
        self.forbidden = (
            "cuLibraryLoadData",
            "cuLibraryGetKernel",
            "cuKernelGetFunction",
            "cuLibraryUnload",
            "cuFuncSetAttribute",
        )
        for name in self.forbidden:
            self.calls[name] = self.stack.enter_context(
                patch.object(driver, name, side_effect=AssertionError(name))
            )
        # Compiler provenance is supplied by the fixture; loader checks stay real.
        self.stack.enter_context(
            patch.object(DispatchArtifact, "check", lambda self: None)
        )

        def constant_consumer(artifact, site, role, index):
            return (128, 1, 1)[index] if role == "block" else 64

        self.stack.enter_context(
            patch.object(provider, "_constant_consumer", constant_consumer)
        )

    def tearDown(self):
        for name in self.forbidden:
            self.calls[name].assert_not_called()
        self.assertEqual(tuple(loading._OPEN_LOADERS), self.open_loaders)
        self.assertEqual(tuple(provider._OPEN_OWNERS), self.open_owners)
        super().tearDown()

    def parameter_info(self, function, index):
        self.assertEqual(function, self.function)
        if index == len(self.layout):
            return (driver.CUresult.CUDA_ERROR_INVALID_VALUE,)
        return (self.success, *self.layout[index])

    def artifact(self, eager, cluster=None):
        binary = b"compiler artifact bytes"
        digest = hashlib.sha256(binary).hexdigest()
        site = ArtifactSite(
            0,
            None,
            0,
            ("module", "captured_kernel"),
            Registration("captured_kernel", "handle", 0, "binary", digest),
            (
                Parameter(0, "!llvm.ptr", 0, 0, 8, 8),
                Parameter(1, "!llvm.struct<(i64, i64)>", 1, 1, 16, 8),
            ),
            NodeFields(0, "captured_kernel", (8, 16), (), (), (), ()),
            (),
            (),
            2,
            (),
            cluster,
        )
        payload = SimpleNamespace(
            binaries=(BinaryImage(0, "binary", digest, binary),), sites=(site,)
        )
        binding = SimpleNamespace(kernel_owner=eager, kernel_payload=payload)
        artifact = object.__new__(DispatchArtifact)
        object.__setattr__(artifact, "_payload", payload)
        object.__setattr__(artifact, "_guards", SimpleNamespace(binding=binding))
        return artifact, CapturedKernel(
            site, self.function, eager, self.context, self.device
        )

    def owner(self, artifact, captured):
        return CuTeKernelOwner(
            artifact,
            captured.site,
            stream=self.stream,
            block=(128, 1, 1),
            shared=64,
            captured=captured,
        )

    @parametrize("kind", ("loader", "provider"))
    @parametrize("cluster", (None, (2, 1, 1)))
    def test_exact_function_without_reload_or_mutation(self, kind, cluster):
        if cluster is not None:
            attrs = driver.CUfunction_attribute
            self.attributes[attrs.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH] = 2
            self.attributes[attrs.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT] = 1
            self.attributes[attrs.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH] = 1
            self.attributes[attrs.CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET] = 1
        artifact, captured = self.artifact(_EagerOwner(), cluster)
        if kind == "loader":
            owner = ArtifactKernels(artifact, stream=self.stream, captured=captured)
            kernel = owner.validate(
                captured.site, (bytes(8), bytes(16)), (3, 1, 1), (128, 1, 1), 64
            )
        else:
            owner = self.owner(artifact, captured)
            kernel = owner
        try:
            self.assertEqual(kernel.function, captured.function)
            self.assertEqual(kernel.parameter_layout, self.layout)
            self.assertEqual(kernel.max_dynamic_shared, 4096)
            self.calls["cuFuncGetName"].assert_called_once_with(captured.function)
        finally:
            owner.close()
        self.calls["cuStreamSynchronize"].assert_called_once_with(self.stream)

    @parametrize(
        "mismatch", ("owner", "payload", "context", "device", "site", "function")
    )
    def test_captured_identity_mismatch_rejected(self, mismatch):
        artifact, captured = self.artifact(_EagerOwner())
        if mismatch == "payload":
            artifact._guards.binding.kernel_payload = SimpleNamespace(
                **vars(artifact._payload)
            )
        else:
            replacements = {
                "owner": _EagerOwner(),
                "context": self.context + 1,
                "device": self.device + 1,
                "site": captured.site._replace(),
                "function": 0,
            }
            captured = captured._replace(**{mismatch: replacements[mismatch]})
        reason = "another artifact" if mismatch == "site" else "exact compiler owner"
        with self.assertRaisesRegex(ValueError, reason):
            self.owner(artifact, captured)
        self.calls["cuFuncGetName"].assert_not_called()

    @parametrize(
        "mismatch",
        (
            "symbol",
            "size",
            "overlap",
            "extra_slot",
            "static_shared",
            "cluster",
            "partial_cluster",
            "fixed_cluster",
        ),
    )
    def test_driver_contract_mismatch_rejected(self, mismatch):
        cluster = (2, 1, 1) if mismatch == "fixed_cluster" else None
        artifact, captured = self.artifact(_EagerOwner(), cluster)
        attrs = driver.CUfunction_attribute
        if mismatch == "symbol":
            self.symbol = b"another_kernel"
            reason = "kernel symbol"
        elif mismatch == "size":
            self.layout = ((0, 4), (16, 16))
            reason = "parameter layout"
        elif mismatch == "overlap":
            self.layout = ((0, 8), (4, 16))
            reason = "parameter layout"
        elif mismatch == "extra_slot":
            self.layout += ((32, 8),)
            reason = "additional parameter slots"
        elif mismatch == "static_shared":
            self.attributes[attrs.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES] = 4
            reason = "Static shared memory"
        elif mismatch == "cluster":
            self.attributes[attrs.CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET] = 1
            reason = "explicit cluster"
        elif mismatch == "partial_cluster":
            self.attributes[attrs.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH] = 2
            reason = "invalid cluster requirements"
        else:
            self.attributes[attrs.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH] = 3
            self.attributes[attrs.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT] = 1
            self.attributes[attrs.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH] = 1
            reason = "fixed source cluster"
        with self.assertRaisesRegex(ValueError, reason):
            self.owner(artifact, captured)

    @parametrize("resource", ("threads", "shared"))
    def test_existing_function_resource_limits_are_not_raised(self, resource):
        attrs = driver.CUfunction_attribute
        attr = (
            attrs.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
            if resource == "threads"
            else attrs.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        )
        self.attributes[attr] = 32
        artifact, captured = self.artifact(_EagerOwner())
        with self.assertRaisesRegex(
            ProviderDeclined, "exceed the loaded device resources"
        ):
            self.owner(artifact, captured)

    def test_changed_context_rejected_before_validation(self):
        artifact, captured = self.artifact(_EagerOwner())
        owner = self.owner(artifact, captured)
        self.context += 1
        try:
            with self.assertRaisesRegex(RuntimeError, "must remain current"):
                owner.check()
        finally:
            self.context -= 1
            owner.close()

    def borrowed_owner(self):
        eager = _EagerOwner()
        artifact, captured = self.artifact(eager)
        owner = self.owner(artifact, captured)
        return owner, weakref.ref(eager)

    def test_graph_borrow_retains_eager_owner_and_prevents_close(self):
        owner, eager = self.borrowed_owner()
        borrow = owner._borrow_for_cudagraph()
        try:
            gc.collect()
            self.assertIsNotNone(eager())
            self.assertIs(borrow._owner, owner)
            with self.assertRaisesRegex(
                RuntimeError, "borrowed by a prepared CUDA graph"
            ):
                owner.close()
            with self.assertRaisesRegex(RuntimeError, "live graph tokens"):
                owner._loader.close()
            self.calls["cuStreamSynchronize"].assert_not_called()
        finally:
            del borrow
            gc.collect()
            owner.close()
        del owner
        gc.collect()
        self.assertIsNone(eager())

    def test_failed_cleanup_retains_eager_owner_until_retry(self):
        owner, eager = self.borrowed_owner()
        self.calls["cuStreamSynchronize"].side_effect = RuntimeError("sync failed")
        try:
            with self.assertRaisesRegex(RuntimeError, "sync failed"):
                owner.close()
            gc.collect()
            self.assertIsNotNone(eager())
            self.assertFalse(owner.closed)
        finally:
            self.calls["cuStreamSynchronize"].side_effect = None
            self.calls["cuStreamSynchronize"].return_value = (self.success,)
            owner.close()
        del owner
        gc.collect()
        self.assertIsNone(eager())


if __name__ == "__main__":
    run_tests()
