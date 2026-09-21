from __future__ import annotations

import ctypes
import hashlib
import math
from typing import NamedTuple

from cuda.bindings import driver
from torch.cuda._utils import _check_cuda_bindings

from .artifact import DispatchArtifact


class LoadedKernel(NamedTuple):
    function: int
    parameter_layout: tuple[tuple[int, int], ...]
    max_threads: int
    max_dynamic_shared: int
    required_cluster: tuple[int, int, int] = (0, 0, 0)
    cluster_size_must_be_set: bool = False


class DeviceLimits(NamedTuple):
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    threads: int


# Retain buffers and code until explicit successful cleanup, including failures.
_OPEN_LOADERS: list[ArtifactKernels] = []


class ArtifactKernels:
    def __init__(self, artifact: DispatchArtifact, *, stream: int) -> None:
        if type(artifact) is not DispatchArtifact:
            raise TypeError("Loading requires the factory-owned dispatch artifact")
        artifact.check()
        if type(stream) is not int or not 0 <= stream < 2**64:
            raise ValueError("Loading requires a nonnegative raw CUDA stream")
        self.artifact = artifact
        self.context = int(_check_cuda_bindings(driver.cuCtxGetCurrent()))
        if not self.context:
            raise ValueError("Loading requires a current CUDA context")
        self.device = int(_check_cuda_bindings(driver.cuCtxGetDevice()))
        self.stream = stream
        if int(_check_cuda_bindings(driver.cuStreamGetCtx(stream))) != self.context:
            raise ValueError("The loading stream belongs to another CUDA context")
        self._buffers = ()
        self._libraries = ()
        self._kernels = ()
        self._owned_libraries: tuple[int, ...] = ()
        self._graphs: tuple[object, ...] = ()
        self._phase = "loading"
        self._owners = ()
        attrs = driver.CUdevice_attribute
        grid_attrs = (attrs.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, attrs.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                      attrs.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
        block_attrs = (attrs.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, attrs.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                       attrs.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)
        self.limits = DeviceLimits(
            tuple(int(_check_cuda_bindings(driver.cuDeviceGetAttribute(attr, self.device))) for attr in grid_attrs),
            tuple(int(_check_cuda_bindings(driver.cuDeviceGetAttribute(attr, self.device))) for attr in block_attrs),
            int(_check_cuda_bindings(driver.cuDeviceGetAttribute(attrs.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, self.device))),
        )
        if any(value <= 0 for value in (*self.limits.grid, *self.limits.block, self.limits.threads)):
            raise ValueError("The device reported invalid launch limits")
        _OPEN_LOADERS.append(self)
        try:
            libraries, buffers, kernels = {}, [], {}
            for image in artifact.binaries:
                if (type(image.data) is not bytes or not image.data
                        or hashlib.sha256(image.data).hexdigest() != image.sha256
                        or image.library_slot in libraries):
                    raise ValueError("Artifact binary bytes or library slots differ")
                buffer = ctypes.create_string_buffer(image.data)
                buffers.append(buffer)
                self._buffers = tuple(buffers)
                library = int(_check_cuda_bindings(driver.cuLibraryLoadData(
                    ctypes.addressof(buffer), [], [], 0, [], [], 0)))
                self._owned_libraries += (library,)
                libraries[image.library_slot] = (image, library)
                self._libraries = tuple(libraries.items())
            for site in artifact.sites:
                artifact.check_site(site)
                registration = site.registration
                image, library = libraries[registration.library_slot]
                if (registration.binary_global != image.global_name or registration.binary_sha256 != image.sha256
                        or site.callee[1] != registration.kernel_symbol):
                    raise ValueError("The site registration lost its exact original binary")
                key = tuple(registration)
                sizes = tuple(parameter.size for parameter in site.parameters)
                if sizes != site.fields.parameter_sizes:
                    raise ValueError("Copied parameter and native field sizes differ")
                if key not in kernels:
                    symbol = registration.kernel_symbol
                    if type(symbol) is not str or not symbol or "\0" in symbol:
                        raise ValueError("The exact registered kernel symbol is invalid")
                    handle = _check_cuda_bindings(driver.cuLibraryGetKernel(library, symbol.encode()))
                    function = int(_check_cuda_bindings(driver.cuKernelGetFunction(handle)))
                    layout = tuple(tuple(int(value) for value in _check_cuda_bindings(
                        driver.cuFuncGetParamInfo(function, index))) for index in range(len(sizes)))
                    tail = driver.cuFuncGetParamInfo(function, len(sizes))
                    if tail[0] != driver.CUresult.CUDA_ERROR_INVALID_VALUE:
                        _check_cuda_bindings(tail)
                        raise ValueError("The loaded kernel has additional parameter slots")
                    end = 0
                    for (offset, size), expected in zip(layout, sizes):
                        if offset < end or size != expected or size <= 0:
                            raise ValueError("Driver parameter layout differs from copied compiler sizes")
                        end = offset + size
                    attributes = driver.CUfunction_attribute
                    if _check_cuda_bindings(driver.cuFuncGetAttribute(attributes.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function)):
                        raise ValueError("Static shared memory needs additional handling")
                    required_cluster = tuple(int(_check_cuda_bindings(driver.cuFuncGetAttribute(attr, function))) for attr in (
                        attributes.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH,
                        attributes.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT,
                        attributes.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH,
                    ))
                    cluster_required = int(_check_cuda_bindings(driver.cuFuncGetAttribute(
                        attributes.CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET, function)))
                    if (cluster_required not in (0, 1)
                            or not (required_cluster == (0, 0, 0) or all(value > 0 for value in required_cluster))):
                        raise ValueError("The loaded kernel reported invalid cluster requirements")
                    shared_limit = int(_check_cuda_bindings(driver.cuDeviceGetAttribute(
                        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, self.device)))
                    _check_cuda_bindings(driver.cuFuncSetAttribute(
                        function, attributes.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_limit))
                    kernel = LoadedKernel(
                        function, layout,
                        int(_check_cuda_bindings(driver.cuFuncGetAttribute(
                            attributes.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function))),
                        int(_check_cuda_bindings(driver.cuFuncGetAttribute(
                            attributes.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function))),
                        required_cluster, bool(cluster_required),
                    )
                    if kernel.max_threads <= 0 or kernel.max_dynamic_shared < 0:
                        raise ValueError("The loaded kernel reported invalid resource limits")
                    kernels[key] = kernel
                elif tuple(size for _, size in kernels[key].parameter_layout) != sizes:
                    raise ValueError("Sites sharing one registration disagree on its parameter layout")
                kernel = kernels[key]
                if site.cluster is None:
                    if kernel.cluster_size_must_be_set or any(kernel.required_cluster):
                        raise ValueError("The loaded kernel requires an explicit cluster configuration")
                elif (type(site.cluster) is not tuple or len(site.cluster) != 3
                        or any(type(value) is not int or not 0 < value < 2**32 for value in site.cluster)
                        or (any(kernel.required_cluster) and site.cluster != kernel.required_cluster)):
                    raise ValueError("The fixed source cluster differs from the loaded kernel requirements")
            self._kernels = tuple((site, kernels[tuple(site.registration)]) for site in artifact.sites)
            artifact.check()
            self._phase = "ready"
            self._owners = self._state()
            self.check()
        except BaseException as error:
            self._phase = "failed"
            self._owners = self._state()
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note(f"The loaded artifact remains retained after cleanup failed: {cleanup}")
            raise

    def _state(self):
        return (self.artifact, self.context, self.device, self.stream, self.limits,
                self._buffers, self._libraries, self._kernels, self._owned_libraries, self._graphs, self._phase)

    def _check_owners(self) -> None:
        state = self._state()
        if len(state) != len(self._owners) or any(value is not owner for value, owner in zip(state, self._owners)):
            raise RuntimeError("Loaded artifact ownership changed")
        handles = tuple(library for _, (_, library) in self._libraries)
        if self._owned_libraries != handles[:len(self._owned_libraries)]:
            raise RuntimeError("Loaded library cleanup ownership changed")

    def _check_context(self) -> None:
        if (int(_check_cuda_bindings(driver.cuCtxGetCurrent())) != self.context
                or int(_check_cuda_bindings(driver.cuCtxGetDevice())) != self.device
                or int(_check_cuda_bindings(driver.cuStreamGetCtx(self.stream))) != self.context):
            raise RuntimeError("The original loaded CUDA device, context and stream must remain current")

    def check(self) -> None:
        if self._phase != "ready":
            raise RuntimeError("Loaded artifact is closed, active or failed")
        self._check_owners()
        if len(self._owned_libraries) != len(self._libraries):
            raise RuntimeError("A loaded artifact library has already been released")
        self.artifact.check()
        self._check_owners()
        self._check_context()

    def validate(self, site, arguments: tuple[bytes, ...], grid: tuple[int, int, int],
                 block: tuple[int, int, int], shared: int) -> LoadedKernel:
        self.check()
        self.artifact.check_site(site)
        kernel = next(kernel for original, kernel in self._kernels if original is site)
        if (type(arguments) is not tuple or any(type(value) is not bytes for value in arguments)
                or tuple(len(value) for value in arguments) != tuple(size for _, size in kernel.parameter_layout)):
            raise ValueError("Packed arguments disagree with the loaded device signature")
        if (type(grid) is not tuple or type(block) is not tuple or len(grid) != 3 or len(block) != 3
                or any(type(value) is not int or value <= 0 for value in (*grid, *block))
                or any(value > limit for value, limit in zip(grid, self.limits.grid))
                or any(value > limit for value, limit in zip(block, self.limits.block))
                or math.prod(block) > min(kernel.max_threads, self.limits.threads)
                or type(shared) is not int or not 0 <= shared <= kernel.max_dynamic_shared):
            raise ValueError("Launch configuration exceeds the actual loaded device limits")
        return kernel

    def acquire_graph(self) -> object:
        self.check()
        token = object()
        self._graphs += (token,)
        self._owners = self._state()
        return token

    def check_graph(self, token: object) -> None:
        self.check()
        if type(token) is not object or not any(token is original for original in self._graphs):
            raise RuntimeError("Graph token is foreign or already released")

    def release_graph(self, token: object) -> None:
        self._check_owners()
        if type(token) is not object or not any(token is original for original in self._graphs):
            raise RuntimeError("Graph token is foreign or already released")
        self._graphs = tuple(original for original in self._graphs if original is not token)
        self._owners = self._state()

    def poison(self) -> None:
        self._check_owners()
        if self._phase != "closed":
            self._phase = "failed"
            self._owners = self._state()

    def close(self) -> None:
        if self._phase == "closed":
            return
        self._check_owners()
        if self._graphs:
            raise RuntimeError("Loaded artifact still owns live graph tokens")
        self._check_context()
        self._phase = "closing"
        self._owners = self._state()
        try:
            _check_cuda_bindings(driver.cuStreamSynchronize(self.stream))
            while self._owned_libraries:
                remaining = self._owned_libraries[:-1]
                _check_cuda_bindings(driver.cuLibraryUnload(self._owned_libraries[-1]))
                self._owned_libraries = remaining
                self._owners = self._state()
        except BaseException:
            self._phase = "failed"
            self._owners = self._state()
            raise
        self._buffers = ()
        self._libraries = ()
        self._kernels = ()
        self._phase = "closed"
        self._owners = self._state()
        for index, owner in enumerate(_OPEN_LOADERS):
            if owner is self:
                _OPEN_LOADERS.pop(index)
                break
