# Owner(s): ["module: inductor"]

import faulthandler
import multiprocessing
import os
import struct
import threading
import unittest

import torch
from torch._inductor.runtime._cudagraph.host_trace import HostTraceReplay
from torch.cuda._utils import _check_cuda_bindings, _HAS_CUDA_BINDINGS
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


def _threaded_replay(device, detach_context):
    from cuda.bindings import driver

    if os.name == "posix":
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    faulthandler.enable()
    check = TestCase()

    def fn(x):
        return torch.ops.aten.native_layer_norm(x, [128], None, None, 1e-5)[0]

    inputs = [torch.randn(8, 128, device=device) for _ in range(3)]
    expected = [fn(value) for value in inputs]
    replay = HostTraceReplay(fn)
    retained = [replay([inputs[0]]), replay([inputs[0]])]
    torch.cuda.synchronize(device)
    check.assertEqual(len(replay.variants), 1)
    check.assertEqual(len({value.data_ptr() for value in inputs}), 3)
    errors = []

    def worker():
        try:
            for index in (1, 2) if detach_context else (1,):
                check.assertEqual(
                    int(_check_cuda_bindings(driver.cuCtxGetCurrent())), 0
                )
                torch.cuda.current_stream(device)
                check.assertEqual(
                    int(_check_cuda_bindings(driver.cuCtxGetCurrent())), 0
                )
                (result,) = replay([inputs[index]])
                torch.cuda.synchronize(device)
                check.assertEqual(result, expected[index])
                check.assertEqual(len(replay.variants), 1)
                retained.append(result)
                if detach_context:
                    _check_cuda_bindings(driver.cuCtxSetCurrent(0))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(90)
    check.assertFalse(thread.is_alive(), "Replay worker did not finish")
    if errors:
        raise errors[0]
    replay.close()


def _threaded_parameter_batch(device, api):
    from cuda.bindings import driver, runtime

    if os.name == "posix":
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    faulthandler.enable()
    check = TestCase()
    torch.cuda.set_device(device)
    kernel = torch.cuda._compile_kernel(
        """
        __global__ void copy_bias(const int* input, int* output) {
            output[threadIdx.x] = input[threadIdx.x] + 7;
        }
        """,
        "copy_bias",
    )
    original = torch.arange(32, dtype=torch.int32, device=device)
    replacement = original + 17
    output = torch.empty_like(original)
    expected = replacement + 7
    kernel(block=(32, 1, 1), args=[original, output])
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.graph(graph, stream=stream):
        kernel(block=(32, 1, 1), args=[original, output])
        info = _check_cuda_bindings(
            runtime.cudaStreamGetCaptureInfo(stream.cuda_stream)
        )
        check.assertEqual(info[-1], 1)
        node = int(info[3][0])
    graph.replay()
    torch.cuda.synchronize(device)
    if api == "public":
        graph.update_kernel_params({node: {0: struct.pack("P", original.data_ptr())}})
    else:
        batch = graph._prepare_kernel_pointer_updates(((node, 0, 0),), 1)
    check.assertNotEqual(original.data_ptr(), replacement.data_ptr())
    errors = []

    def worker():
        try:
            check.assertEqual(int(_check_cuda_bindings(driver.cuCtxGetCurrent())), 0)
            if api == "public":
                graph.update_kernel_params(
                    {node: {0: struct.pack("P", replacement.data_ptr())}}
                )
                graph.replay()
            else:
                graph._replay_kernel_pointer_updates(batch, (replacement.data_ptr(),))
            torch.cuda.synchronize(device)
            check.assertEqual(output, expected)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(90)
    check.assertFalse(thread.is_alive(), "Parameter-update worker did not finish")
    if errors:
        raise errors[0]
    graph.reset()


@unittest.skipUnless(
    _HAS_CUDA_BINDINGS
    and torch.version.cuda is not None
    and tuple(map(int, torch.version.cuda.split("."))) >= (12, 8)
    and hasattr(torch._C, "_CUDAGraphBoxedDispatch"),
    "requires CUDA host tracing and cuda-python driver bindings",
)
class TestCudaHostTraceThreads(TestCase):
    @parametrize("detach_context", (False, True))
    def test_first_pointer_patch_establishes_thread_context(
        self, device, detach_context
    ):
        process = multiprocessing.get_context("spawn").Process(
            target=_threaded_replay, args=(device, detach_context)
        )
        process.start()
        process.join(120)
        if process.is_alive():
            process.terminate()
            process.join(10)
            self.fail("Isolated replay process did not finish")
        self.assertEqual(process.exitcode, 0)

    @parametrize("api", ("public", "pointer_batch"))
    def test_parameter_batch_establishes_thread_context(self, device, api):
        process = multiprocessing.get_context("spawn").Process(
            target=_threaded_parameter_batch, args=(device, api)
        )
        process.start()
        process.join(120)
        if process.is_alive():
            process.terminate()
            process.join(10)
            self.fail("Isolated parameter-update process did not finish")
        self.assertEqual(process.exitcode, 0)


instantiate_device_type_tests(TestCudaHostTraceThreads, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
