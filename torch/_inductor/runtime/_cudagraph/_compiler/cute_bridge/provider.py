"""Cold ownership of one exact CuTe site and its aggregate CUDA parameters."""

from __future__ import annotations

import ctypes
import math
from contextlib import contextmanager
from threading import Lock

from cuda.bindings import driver
from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.artifact import ArtifactSite, DispatchArtifact
from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.loading import ArtifactKernels
from torch._inductor.runtime.cudagraph_boxed_replay import _KernelModule
from torch.cuda._utils import _check_cuda_bindings


class ProviderDeclined(ValueError):
    pass


def _constant_consumer(artifact, site, role, index):
    from torch._inductor.runtime._cudagraph._compiler.values import ScalarValue

    matches = [artifact.consumers[number] for number in site.consumer_ids
               if (artifact.consumers[number].role, artifact.consumers[number].index) == (role, index)]
    if len(matches) != 1:
        raise ProviderDeclined("The selected site lacks one exact launch consumer")
    consumer, = matches
    numeric = consumer.numeric
    numeric.check()
    cfg = numeric._cfg
    if (consumer.site_id != site.site_id or consumer.source_order != numeric.source_order
            or cfg.result_types != (consumer.result_type,) or len(cfg.blocks) != 1):
        raise ProviderDeclined("Fixed launch fields require a single constant-return consumer")
    block, = cfg.blocks
    if block.terminator.kind != "return" or len(block.terminator.values) != 1 or block.terminator.edges:
        raise ProviderDeclined("Fixed launch fields cannot depend on control flow")
    constants = {}
    for instruction in block.instructions:
        flow = instruction.expression
        if (flow.kind != "constant" or flow.argument is not None or flow.path or flow.operands
                or flow.constant is None or flow.predicate is not None):
            raise ProviderDeclined("Fixed launch fields must be compiler literals, independent of inputs")
        typ, data, size = flow.constant
        if typ not in ("i1", "i32", "i64") or typ != flow.llvm_type:
            raise ProviderDeclined("Unsupported fixed launch constant type")
        value = ScalarValue(typ, int.from_bytes(data, "little"), size)
        if value.data() != data:
            raise ProviderDeclined("Fixed launch constant lost its exact compiler bytes")
        constants[instruction.result] = value
    value = constants.get(block.terminator.values[0])
    if value is None or value.llvm_type != consumer.result_type:
        raise ProviderDeclined("Fixed launch return is not a compiler constant")
    return value.integer(signed=consumer.result_type != "i1")


# Checked close is explicit; failed releases retain both the token and its code.
_OPEN_OWNERS: list[CuTeKernelOwner] = []
_FAILED_BORROWS: list[_CuTeGraphBorrow] = []


class _CuTeGraphBorrow:
    __slots__ = ("_owner", "_token")

    def __init__(self, owner, token):
        self._owner, self._token = owner, token

    def __del__(self):
        owner = getattr(self, "_owner", None)
        if owner is None:
            return
        try:
            owner._release_graph(self._token)
        except BaseException:
            _FAILED_BORROWS.append(self)
        else:
            self._owner = None
            self._token = None

    def __reduce_ex__(self, protocol):
        raise TypeError("CuTe graph borrows cannot be copied or serialized")


class CuTeKernelOwner(_KernelModule):
    def __init__(self, artifact: DispatchArtifact, site: ArtifactSite, *, stream: int,
                 block: tuple[int, int, int], shared: int | None):
        if type(artifact) is not DispatchArtifact:
            raise ProviderDeclined("CuTe ownership requires a factory-produced dispatch artifact")
        artifact.check_site(site)
        if (type(block) is not tuple or len(block) != 3 or any(type(value) is not int or value <= 0 for value in block)
                or shared is not None and (type(shared) is not int or not 0 <= shared < 2**32)):
            raise ProviderDeclined("Expected a fixed block and an optional nonnegative fixed shared request")
        if tuple(_constant_consumer(artifact, site, "block", axis) for axis in range(3)) != block:
            raise ProviderDeclined("Fixed block differs from the exact compiler consumers")
        if shared is not None:
            if _constant_consumer(artifact, site, "shared", 0) != shared:
                raise ProviderDeclined("Fixed shared bytes differ from the exact compiler consumer")
            if not 0 <= _constant_consumer(artifact, site, "kernel_smem", 0) <= shared:
                raise ProviderDeclined("Fixed shared bytes do not cover the compiler kernel requirement")
            for index, diagnostic in enumerate(site.diagnostics):
                if _constant_consumer(artifact, site, "diagnostic", index) != int(diagnostic.expected):
                    raise ProviderDeclined("The compiler shared-memory diagnostic rejects this launch")
        self._artifact, self._site = artifact, site
        self._block, self._shared = block, shared
        self._cluster = site.cluster
        self._lock = Lock()
        self._phase = "loading"
        self._kernel = None
        self._loader = ArtifactKernels(artifact, stream=stream)
        self._owners = self._state()
        _OPEN_OWNERS.append(self)
        try:
            self._kernel, = (kernel for original, kernel in self._loader._kernels if original is site)
            limits = self._loader.limits
            if (any(value > limit for value, limit in zip(block, limits.block))
                    or math.prod(block) > min(self._kernel.max_threads, limits.threads)
                    or shared is not None and shared > self._kernel.max_dynamic_shared):
                raise ProviderDeclined("Fixed launch fields exceed the loaded device resources")
            self._phase = "ready"
            self._owners = self._state()
            self.check()
        except BaseException as error:
            self._phase = "failed"
            self._owners = self._state()
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note(f"CuTe owner retained after cleanup failed: {cleanup}")
            raise

    @property
    def artifact(self):
        return self._artifact

    @property
    def site(self):
        return self._site

    @property
    def function(self) -> int:
        if self._phase != "ready":
            raise RuntimeError("CuTe kernel owner is closed or failed")
        return self._kernel.function

    @property
    def parameter_sizes(self):
        return self._site.fields.parameter_sizes

    @property
    def parameter_layout(self):
        return self._kernel.parameter_layout

    @property
    def block(self):
        return self._block

    @property
    def shared(self):
        return self._shared

    @property
    def max_dynamic_shared(self):
        self._ready()
        return self._kernel.max_dynamic_shared

    @property
    def cluster(self):
        return self._cluster

    @property
    def closed(self):
        return self._phase == "closed"

    def _state(self):
        return self._artifact, self._site, self._block, self._shared, self._cluster, self._lock, self._loader, self._kernel

    def _check_owners(self):
        state = self._state()
        if len(state) != len(self._owners) or any(actual is not old for actual, old in zip(state, self._owners)):
            raise RuntimeError("CuTe kernel owner identity changed")

    @contextmanager
    def _locked(self):
        lock = self._lock
        if not lock.acquire(blocking=False):
            raise RuntimeError("CuTe kernel owner is busy or reentrant")
        try:
            self._check_owners()
            yield
        finally:
            lock.release()

    def _ready(self):
        if self._phase != "ready":
            raise RuntimeError("CuTe kernel owner is closed or failed")

    def check(self):
        with self._locked():
            self._ready()
            self._loader.check()

    def _borrow_for_cudagraph(self):
        with self._locked():
            self._ready()
            token = self._loader.acquire_graph()
            try:
                return _CuTeGraphBorrow(self, token)
            except BaseException as error:
                try:
                    self._loader.release_graph(token)
                except BaseException as cleanup:
                    error.add_note(f"CuTe graph token retained after cleanup failed: {cleanup}")
                raise

    def _release_graph(self, token):
        with self._locked():
            self._loader.release_graph(token)

    def launch(self, parameter_images: tuple[bytes, ...], grid: tuple[int, int, int], stream: int,
               *, shared: int | None = None):
        with self._locked():
            self._ready()
            if type(stream) is not int or not 0 <= stream < 2**64 or not self._loader._graphs:
                raise ProviderDeclined("CuTe capture launch requires a raw stream and a held graph borrow")
            if self._shared is None:
                if shared is None:
                    raise ProviderDeclined("Dynamic shared bytes require an evaluated launch request")
            else:
                if shared is not None:
                    raise ProviderDeclined("A fixed shared request cannot be overridden")
                shared = self._shared
            kernel = self._loader.validate(self._site, parameter_images, grid, self._block, shared)
            if int(_check_cuda_bindings(driver.cuStreamGetCtx(stream))) != self._loader.context:
                raise ProviderDeclined("CuTe capture stream belongs to another CUDA context")
            if (_check_cuda_bindings(driver.cuStreamIsCapturing(stream))
                    != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE):
                raise ProviderDeclined("CuTe provider launches are supported only during stream capture")
            storage = tuple(ctypes.create_string_buffer(value, len(value)) for value in parameter_images)
            arguments = (ctypes.c_void_p * len(storage))(*(ctypes.addressof(value) for value in storage))
            if self._cluster is None:
                _check_cuda_bindings(driver.cuLaunchKernel(
                    kernel.function, *grid, *self._block, shared, stream, ctypes.addressof(arguments), 0))
            else:
                attribute = driver.CUlaunchAttribute()
                attribute.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
                attribute.value.clusterDim.x, attribute.value.clusterDim.y, attribute.value.clusterDim.z = self._cluster
                config = driver.CUlaunchConfig()
                config.gridDimX, config.gridDimY, config.gridDimZ = grid
                config.blockDimX, config.blockDimY, config.blockDimZ = self._block
                config.sharedMemBytes, config.hStream = shared, stream
                config.attrs, config.numAttrs = [attribute], 1
                _check_cuda_bindings(driver.cuLaunchKernelEx(config, kernel.function, ctypes.addressof(arguments), 0))

    def close(self):
        with self._locked():
            if self.closed:
                return
            if self._loader._graphs:
                raise RuntimeError("Cannot close a CuTe kernel borrowed by a prepared CUDA graph")
            try:
                self._loader.close()
            except BaseException:
                self._phase = "failed"
                raise
            self._phase = "closed"
            for index, owner in enumerate(_OPEN_OWNERS):
                if owner is self:
                    _OPEN_OWNERS.pop(index)
                    break

    def __reduce_ex__(self, protocol):
        raise TypeError("CuTe kernel owners cannot be copied or serialized")
