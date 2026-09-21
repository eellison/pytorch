# Owner(s): ["module: cuda"]

import ctypes
import unittest

import torch
from torch._inductor.runtime._cudagraph.host_trace import prepare_host_trace
from torch.cuda import _host_trace
from torch.cuda._utils import _check_cuda_bindings
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    requires_cuda_python_bindings,
    run_tests,
    TestCase,
)


class _OffsetPool:
    def __init__(self, offset):
        from cuda.bindings import runtime

        self.runtime = runtime
        self.capacity = 8 * 1024 * 1024
        self.base = _check_cuda_bindings(runtime.cudaMalloc(self.capacity))
        self.cursor = 0
        self.allocated = []
        self.freed = []

        def allocate(size, device, stream):
            address = self.base + self.cursor + offset
            self.cursor += size + 256
            if self.cursor > self.capacity:
                return None
            self.allocated.append(address)
            return address

        def free(address, size, device, stream):
            self.freed.append(address)

        self.allocate = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
        )(allocate)
        self.free = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
        )(free)
        self.allocator = torch._C._cuda_customAllocator(
            ctypes.cast(self.allocate, ctypes.c_void_p).value,
            ctypes.cast(self.free, ctypes.c_void_p).value,
        )
        self.pool = torch.cuda.MemPool(self.allocator)

    def close(self):
        self.pool = None
        torch.cuda.empty_cache()
        if sorted(self.allocated) != sorted(self.freed):
            raise AssertionError("Replay retained a custom segment after close")
        _check_cuda_bindings(self.runtime.cudaFree(self.base))


@requires_cuda_python_bindings
@unittest.skipIf(
    not hasattr(torch._C, "_HostTraceKernel")
    or torch.version.cuda is None
    or tuple(map(int, torch.version.cuda.split("."))) < (12, 8),
    "requires NVIDIA CUDA 12.8 or later with host tracing",
)
class TestCudaGraphOwnedAlignment(TestCase):
    def _prepare(self, device, empty=False):
        def fn(x, marker):
            marker.copy_(x)
            return torch.empty((0 if empty else 64,), device=device)

        x = torch.ones(64, device=device)
        marker = torch.zeros(128, device=device)[::2]
        tape = _host_trace.trace(fn, (x, marker))
        variant = prepare_host_trace(tape, (x, marker))
        marker.zero_()
        return variant, x, marker

    def test_misaligned_owned_allocation_rejects_before_launch(self, device):
        if torch.cuda.get_allocator_backend() != "native":
            self.skipTest("custom MemPool requires the native caching allocator")
        variant, x, marker = self._prepare(device)
        pool = _OffsetPool(16)
        try:
            with torch.cuda.use_mem_pool(pool.pool):
                with self.assertRaisesRegex(ValueError, "256-byte-aligned"):
                    variant.entry([x, marker])
            self.assertTrue(variant.entry.failed)
            self.assertTrue(pool.allocated)
            self.assertTrue(all(address % 256 == 16 for address in pool.allocated))
            self.assertEqual(marker, torch.zeros_like(marker))
            with self.assertRaisesRegex(RuntimeError, "closed or failed"):
                variant.entry([x, marker])
        finally:
            variant.close()
            pool.close()
        healthy, x, marker = self._prepare(device)
        try:
            healthy.entry([x, marker])
            self.assertEqual(marker, x)
        finally:
            healthy.close()

    @parametrize("empty", (False, True))
    def test_aligned_owned_allocation_and_zero_bytes(self, device, empty):
        if torch.cuda.get_allocator_backend() != "native":
            self.skipTest("custom MemPool requires the native caching allocator")
        variant, x, marker = self._prepare(device, empty)
        pool = _OffsetPool(0)
        outputs = None
        try:
            with torch.cuda.use_mem_pool(pool.pool):
                outputs = variant.entry([x, marker])
            self.assertEqual(marker, x)
            self.assertEqual(outputs[0].shape, (0 if empty else 64,))
            self.assertEqual(outputs[0].data_ptr() % 256, 0)
            if empty:
                self.assertEqual(outputs[0].data_ptr(), 0)
                self.assertEqual(pool.allocated, [])
            else:
                self.assertTrue(pool.allocated)
        finally:
            outputs = None
            variant.close()
            pool.close()


instantiate_device_type_tests(TestCudaGraphOwnedAlignment, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
