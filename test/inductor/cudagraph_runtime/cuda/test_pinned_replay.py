# Owner(s): ["module: inductor"]
"""Pinned replay inputs retain bound copy sources and unfinished submissions."""

import gc
import mmap
import sys
import threading
import unittest
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    requires_cuda_python_bindings,
    run_tests,
    skipIfRocm,
    TestCase,
)


_SOURCE = r"""
__global__ void observe_copy(const long long* src, long long* dst) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 512) dst[i] = src[i];
}
"""


@requires_cuda_python_bindings
@unittest.skipIf(_get_torch_cuda_version() < (12, 8), "requires CUDA 12.8 or newer")
@skipIfRocm
class TestPinnedReplay(TestCase):
    def capture(self, device):
        from torch.cuda import _compile_kernel

        sys.path.insert(0, str(Path(torch.__file__).resolve().parents[1] / "test"))
        try:
            from test_cuda import get_wait_for_cpu_kernel
        finally:
            sys.path.pop(0)

        gate = get_wait_for_cpu_kernel()
        kernel = _compile_kernel(_SOURCE, "observe_copy")
        source = torch.arange(512, dtype=torch.int64).pin_memory()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            temporary = torch.empty(512, dtype=torch.int64, device=device)
            output = torch.empty_like(temporary)
            temporary.copy_(source, non_blocking=True)
            kernel(grid=(16, 1, 1), block=(32, 1, 1), args=[temporary, output])
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                temporary.copy_(source, non_blocking=True)
                memcpy_node = torch._C._cuda_get_capture_frontier(stream.cuda_stream)[
                    3
                ][0][0]
                kernel(grid=(16, 1, 1), block=(32, 1, 1), args=[temporary, output])
                kernel_node = torch._C._cuda_get_capture_frontier(stream.cuda_stream)[
                    3
                ][0][0]
            graph.instantiate()
        self.addCleanup(stream.synchronize)
        return SimpleNamespace(
            graph=graph,
            stream=stream,
            gate=gate,
            kernel=kernel,
            source=source,
            temporary=temporary,
            output=output,
            memcpy_node=memcpy_node,
            kernel_node=kernel_node,
        )

    def make_entry(self, c, pinned_positions=(0,), input_count=1):
        batch = c.graph._prepare_kernel_replay_updates(
            ((c.kernel_node, 0, input_count), (c.kernel_node, 1, input_count + 1)),
            input_count + 2,
            (),
            (),
            1,
            memcpy_bindings=((c.memcpy_node, None, 0, None, input_count, None, 0),),
        )
        entry = c.graph._make_boxed_replay(
            batch,
            c.stream,
            (c.kernel, c.source, c.temporary, c.output),
            input_count,
            (0,),
            ((torch.int64, (512,), (1,)),) * 2,
            None,
            (1,),
            ((), (("constant", 4096),)),
            pinned_positions=pinned_positions,
        )
        self.addCleanup(entry.close)
        with torch.cuda.stream(c.stream):
            warm_outputs = [entry([c.source]) for _ in range(4)]
        c.stream.synchronize()
        del warm_outputs
        return entry

    @contextmanager
    def block_stream(self, c):
        flag = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        with torch.cuda.stream(c.stream):
            c.gate(grid=(1, 1, 1), block=(1, 1, 1), args=[flag])
        try:
            yield
        finally:
            flag[0] = 1
            c.stream.synchronize()

    @contextmanager
    def registered_storage(self, stream, entry):
        from cuda.bindings import runtime

        from torch.cuda._utils import _check_cuda_bindings

        storage = mmap.mmap(-1, 8192)
        tensor = torch.frombuffer(storage, dtype=torch.int64)
        pointer = tensor.data_ptr()
        _check_cuda_bindings(runtime.cudaHostRegister(pointer, len(storage), 0))
        self.assertTrue(tensor.is_pinned())
        self.assertFalse(
            torch._C._host_trace_record_host_event(tensor, stream.cuda_stream)
        )
        del tensor
        try:
            yield storage
        finally:
            entry.close()
            _check_cuda_bindings(runtime.cudaHostUnregister(pointer))
            storage.close()

    def source_from_storage(self, storage, offset):
        source = torch.frombuffer(storage, dtype=torch.int64)[offset : offset + 512]
        source.copy_(torch.arange(512, dtype=torch.int64) + offset)
        return source

    def test_capture_copies_the_pinned_source(self, device):
        c = self.capture(device)
        with torch.cuda.stream(c.stream):
            c.graph.replay()
        c.stream.synchronize()
        self.assertEqual(c.output.cpu(), c.source)

    @parametrize("offset", (0, 3))
    def test_managed_bound_source_retains_its_block(self, device, offset):
        c = self.capture(device)
        entry = self.make_entry(c)
        source = torch.arange(512 + offset, dtype=torch.int64).pin_memory()[offset:]
        expected = source.clone()
        pointer = source.untyped_storage().data_ptr()
        ref = weakref.ref(source)
        storage = StorageWeakRef(source.untyped_storage())
        reserve = [
            torch.empty(512 + offset, dtype=torch.int64, pin_memory=True)
            for _ in range(8)
        ]
        del reserve
        with self.block_stream(c), torch.cuda.stream(c.stream):
            box = [source]
            (output,) = entry(box)
            self.assertEqual(box, [])
            del source
            gc.collect()
            self.assertIsNone(ref())
            self.assertFalse(storage.expired())
            pressure = [
                torch.full((512 + offset,), -1, dtype=torch.int64, pin_memory=True)
                for _ in range(8)
            ]
            self.assertNotIn(
                pointer, [t.untyped_storage().data_ptr() for t in pressure]
            )
        entry.wait_for_h2d()
        self.assertEqual(output.cpu(), expected)
        gc.collect()
        self.assertIsNone(ref())
        self.assertFalse(storage.expired())
        entry.close()
        gc.collect()
        self.assertIsNone(ref())
        self.assertTrue(storage.expired())

    @parametrize("offset", (0, 3))
    def test_external_bound_source_is_retained_after_completion(self, device, offset):
        c = self.capture(device)
        entry = self.make_entry(c)
        with self.registered_storage(c.stream, entry) as storage:
            source = self.source_from_storage(storage, offset)
            expected = source.clone()
            ref = weakref.ref(source)
            allocation = StorageWeakRef(source.untyped_storage())
            with self.block_stream(c), torch.cuda.stream(c.stream):
                box = [source]
                (output,) = entry(box)
                self.assertEqual(box, [])
                del source
                gc.collect()
                self.assertIsNotNone(ref())
                self.assertFalse(allocation.expired())
            entry.wait_for_h2d()
            gc.collect()
            self.assertIsNone(ref())
            self.assertFalse(allocation.expired())
            self.assertEqual(output.cpu(), expected)
            entry.close()
            gc.collect()
            self.assertIsNone(ref())
            self.assertTrue(allocation.expired())

    def test_external_submissions_retire_before_later_work(self, device):
        c = self.capture(device)
        entry = self.make_entry(c)
        with self.registered_storage(c.stream, entry) as first_storage:
            with self.registered_storage(c.stream, entry) as second_storage:
                first = self.source_from_storage(first_storage, 0)
                second = self.source_from_storage(second_storage, 3)
                first_ref, second_ref = weakref.ref(first), weakref.ref(second)
                first_allocation = StorageWeakRef(first.untyped_storage())
                second_allocation = StorageWeakRef(second.untyped_storage())
                first_expected, second_expected = first.clone(), second.clone()
                with torch.cuda.stream(c.stream):
                    (first_output,) = entry([first])
                    completed = c.stream.record_event()
                del first
                completed.synchronize()
                with self.block_stream(c), torch.cuda.stream(c.stream):
                    (second_output,) = entry([second])
                    del second
                    gc.collect()
                    self.assertIsNone(first_ref())
                    self.assertIsNotNone(second_ref())
                    self.assertTrue(first_allocation.expired())
                    self.assertFalse(second_allocation.expired())
                entry.wait_for_h2d()
                gc.collect()
                self.assertIsNone(second_ref())
                self.assertFalse(second_allocation.expired())
                self.assertEqual(first_output.cpu(), first_expected)
                self.assertEqual(second_output.cpu(), second_expected)
                entry.close()
                gc.collect()
                self.assertIsNone(second_ref())
                self.assertTrue(second_allocation.expired())

    def test_external_submission_retains_storage_after_set_and_rebind(self, device):
        c = self.capture(device)
        entry = self.make_entry(c)
        with self.registered_storage(c.stream, entry) as memory:
            first = self.source_from_storage(memory, 0).detach()
            first_expected = first.clone()
            first_ref = weakref.ref(first)
            allocation = StorageWeakRef(first.untyped_storage())
            second = (torch.arange(512, dtype=torch.int64) + 7).pin_memory()
            with self.block_stream(c), torch.cuda.stream(c.stream):
                (first_output,) = entry([first])
                first.set_(torch.empty(512, dtype=torch.int64))
                (second_output,) = entry([second])
                del first
                gc.collect()
                self.assertIsNotNone(first_ref())
                self.assertFalse(
                    allocation.expired(), "set_ released an unfinished copy storage"
                )
            entry.wait_for_h2d()
            gc.collect()
            self.assertIsNone(first_ref())
            self.assertTrue(allocation.expired())
            self.assertEqual(first_output.cpu(), first_expected)
            self.assertEqual(second_output.cpu(), second)

    def test_wait_ignores_unrelated_later_stream_work(self, device):
        c = self.capture(device)
        entry = self.make_entry(c)
        source = torch.arange(512, dtype=torch.int64).pin_memory()
        with torch.cuda.stream(c.stream):
            (output,) = entry([source])
        finished = threading.Event()
        errors = []

        def wait():
            try:
                entry.wait_for_h2d()
            except Exception as error:
                errors.append(error)
            finally:
                finished.set()

        with self.block_stream(c):
            worker = threading.Thread(target=wait)
            worker.start()
            try:
                self.assertTrue(
                    finished.wait(10), "wait synchronized unrelated later work"
                )
            finally:
                if finished.is_set():
                    worker.join()
        worker.join(10)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(output.cpu(), source)

    @parametrize("positions", ((0, 0), (2,), (1,)))
    def test_invalid_pinned_indices_decline_at_preparation(self, device, positions):
        c = self.capture(device)
        with self.assertRaises((ValueError, IndexError)):
            self.make_entry(c, positions, input_count=2)


instantiate_device_type_tests(TestPinnedReplay, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
