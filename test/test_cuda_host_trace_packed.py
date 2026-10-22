# Owner(s): ["module: cuda graphs"]

import ctypes
import struct
import unittest
from unittest import mock

import torch
from torch.cuda import _host_trace_replay
from torch.cuda._host_trace_capture import capture_kernel_nodes
from torch.cuda._host_trace_launch import KernelLaunch
from torch.cuda._host_trace_tape import current_trace, EagerCall, trace
from torch.cuda._utils import _check_cuda, _cuda_load_module, _get_gpu_runtime_library, _nvrtc_compile
from torch.testing._internal.common_utils import requires_cuda_python_bindings, run_tests, TEST_CUDA, TestCase


class HostTraceReplay(_host_trace_replay.HostTraceReplay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._called = True


_SOURCE = r"""
struct P { const float* x; long long n; float* y; float s; };
extern "C" __global__ void scale(P p) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < p.n) p.y[i] = p.x[i] * p.s;
}
"""
# the kernel's one parameter, P, as two pieces: x and n, then y and s (and padding)
_PIECES = ((0, 16), (16, 16))
_FIELDS = ((0, 0, 8), (0, 8, 8), (1, 0, 8))
_IMAGES = (bytes(16), bytes(8) + struct.pack("<f", 2.0) + bytes(4))
_KERNEL = []


def _function():
    if not _KERNEL:
        ptx, _ = _nvrtc_compile(_SOURCE, "scale")
        _KERNEL.append(_cuda_load_module(ptx, ["scale"])["scale"])
    return _KERNEL[0].func.value


def _launch(x, y, n):
    image = ctypes.create_string_buffer(struct.pack("<QqQf4x", x.data_ptr(), n, y.data_ptr(), 2.0), 32)
    params = (ctypes.c_void_p * 1)(ctypes.addressof(image))
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    _check_cuda(_get_gpu_runtime_library().cuLaunchKernel(ctypes.c_void_p(_function()), (n + 127) // 128, 1, 1, 128, 1, 1, 0, stream, params, None))


def scale(x):
    y = torch.empty_like(x)
    n = x.numel()
    tr = current_trace()
    if tr is None:
        _launch(x, y, n)
        return y
    tr.record_launch(
        KernelLaunch("scale", _function(), None, _PIECES, ((n + 127) // 128, 1, 1), (128, 1, 1), 0, (x.data_ptr(), n, y.data_ptr()),
                     (x._root, y._root), None, _FIELDS, (), (), frozenset((0, 2)), _IMAGES, packed=True)
    )
    return y


@torch.library.custom_op("host_trace_test::packed_scale", mutates_args=(), device_types="cuda")
def _packed_scale(x: torch.Tensor) -> torch.Tensor:
    # an extension op: its launch is opaque to the tracer
    y = torch.empty_like(x)
    _launch(x, y, x.numel())
    return y


_packed_scale.register_fake(lambda x: torch.empty_like(x))


def eager_calls(f):
    return [r for v in f.variants for _, r in v.tape.launches if isinstance(r, EagerCall)]


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
class TestPackedLaunch(TestCase):
    def test_the_kernel_takes_one_parameter(self):
        x = torch.randn(1000, device="cuda")
        y = torch.empty_like(x)
        (node,) = capture_kernel_nodes(lambda _: _launch(x, y, 1000))
        self.assertEqual(node.layout, ((0, 32),))

    def test_numerics_across_sizes(self):
        # the pieces are contiguous in a native replay's images (each 8-byte
        # aligned), so the kernel's one parameter reads them all; inputs kept
        # alive so each call patches other addresses as well as n
        f = HostTraceReplay(scale)
        sizes = [1000, 3000, 1, 129, 4096, 1000]
        xs = [torch.randn(n, device="cuda") for n in sizes]
        for x in xs:
            self.assertEqual(f(x), x * 2, atol=0, rtol=0)
        self.assertEqual((f.traces, f.replays, f.eager), (1, len(sizes) - 1, 0))
        self.assertEqual(eager_calls(f), [])


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
class TestTracedImpl(TestCase):
    op = torch.ops.host_trace_test.packed_scale.default

    def tearDown(self):
        torch.cuda._host_trace._TRACED_IMPLS.pop(self.op, None)
        super().tearDown()

    def run_sizes(self, f):
        for n in (1000, 3000, 129, 1000):
            x = torch.randn(n, device="cuda")
            self.assertEqual(f(x), x * 2, atol=0, rtol=0)

    def eager_call(self):
        ((_, call),) = trace(_packed_scale, (torch.randn(1000, device="cuda"),)).launches
        self.assertIsInstance(call, EagerCall)
        return call

    @mock.patch.object(torch.cuda._host_trace, "traced_impls", False)
    def test_a_registered_launcher_is_unused_with_the_switch_off(self):
        torch.cuda._host_trace.register_traced_impl(self.op, scale)
        self.eager_call()
        f = HostTraceReplay(_packed_scale)
        self.run_sizes(f)
        self.assertEqual(f.replays, 0)

    @mock.patch.object(torch.cuda._host_trace, "traced_impls", True)
    def test_a_registered_packed_launcher_replays_with_no_eager_call(self):
        # as the trtllm fork records trtllm-gen's KernelParams: one struct parameter in pieces
        torch.cuda._host_trace.register_traced_impl(self.op, scale)
        f = HostTraceReplay(_packed_scale)
        self.run_sizes(f)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 3, 0))
        self.assertEqual(eager_calls(f), [])

    @mock.patch.object(torch.cuda._host_trace, "traced_impls", True)
    def test_a_declining_launcher_leaves_an_eager_call(self):
        def declines(x):
            raise current_trace().decline("the port has no launch")

        torch.cuda._host_trace.register_traced_impl(self.op, declines)
        self.assertIn("the port has no launch", self.eager_call().reason)
        f = HostTraceReplay(_packed_scale)
        self.run_sizes(f)
        self.assertEqual(f.replays, 0)


def setUpModule():
    from torch.cuda import _host_trace_hint_audit

    _host_trace_hint_audit.enable_for_tests()


if __name__ == "__main__":
    run_tests()
