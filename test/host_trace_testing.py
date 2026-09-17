# Owner(s): ["module: cuda"]
"""Shared plumbing of the host-tracing suites (test_cuda_host_trace*.py): the
capture that reads the nodes one call launches, the bitwise comparison, the
trace / build / replay driver and the skips. A rule of one family (its kernel
name pattern, its byte masks, the reduce image check) stays in its suite."""

import os
import unittest
from typing import NamedTuple
from unittest import mock

import torch
from torch.testing._internal.common_utils import TEST_CUDA_PYTHON_BINDINGS, TestCase


if torch.cuda.is_available():
    from torch.cuda import _host_trace as ht

C = torch._C

# the source tree, for the tests that read the recorder's sources and run its
# lint (torch.__file__ is site-packages in CI); a wheel-only environment skips them
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

needs_two_gpus = unittest.skipIf(torch.cuda.device_count() < 2, "needs two GPUs")


def require_nvcc():
    """A SkipTest where nvcc is not available: a test host is built as an
    extension, in this process or a subprocess."""
    import shutil

    from torch.utils.cpp_extension import CUDA_HOME

    nvcc = shutil.which("nvcc") or (
        CUDA_HOME and os.path.join(CUDA_HOME, "bin", "nvcc")
    )
    if not nvcc or not os.path.isfile(nvcc):
        raise unittest.SkipTest("requires nvcc for the test extension")


def load_test_extension(name, cuda_source):
    """A test host built as an extension against the installed headers
    (torch.utils.cpp_extension.load_inline: compiled once per name and source
    under TORCH_EXTENSIONS_DIR, every later process reuses the build), or a
    SkipTest where nvcc is not available."""
    from torch.utils.cpp_extension import load_inline

    require_nvcc()
    return load_inline(
        name,
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=None,
        with_cuda=True,
        extra_cflags=["-std=c++20"],
        extra_cuda_cflags=["-std=c++20"],
    )


def bits(t):
    # the bytes as integers: NaN payloads and signed zeros compare by value
    return t.contiguous().view(
        torch.int8
        if t.element_size() == 1
        else {2: torch.int16, 4: torch.int32, 8: torch.int64, 16: torch.int64}[
            t.element_size()
        ]
    )


class Case(NamedTuple):
    # one replay of a built variant: its outputs, or the miss that refused it
    out: list | None
    miss: str | None


def exec_node_states(graph):
    """(index, kind, enabled) over a built variant's graph in cudaGraphGetNodes
    order, the state read back from the exec with cudaGraphNodeGetEnabled;
    kernel, memset and memcpy nodes carry one, any other kind None."""
    from cuda.bindings import runtime as cudart

    from torch.cuda._utils import _check_cuda_bindings as check

    raw, exe = graph.raw_cuda_graph(), graph.raw_cuda_graph_exec()
    count = check(cudart.cudaGraphGetNodes(raw, 0))[1]
    nodes = check(cudart.cudaGraphGetNodes(raw, count))[0] if count else []
    kinds = {
        cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel: "kernel",
        cudart.cudaGraphNodeType.cudaGraphNodeTypeMemset: "memset",
        cudart.cudaGraphNodeType.cudaGraphNodeTypeMemcpy: "memcpy",
    }
    states = []
    for i, node in enumerate(nodes):
        kind = kinds.get(check(cudart.cudaGraphNodeGetType(node)))
        enabled = (
            bool(check(cudart.cudaGraphNodeGetEnabled(exe, node))) if kind else None
        )
        states.append((i, kind or "other", enabled))
    return states


def capture_graph(fn):
    """fn once eagerly, then under stream capture on a side stream that waits
    for the current one (where the caller made the inputs); the graph kept so
    its nodes can be read."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        g = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(g, stream=stream, capture_error_mode="relaxed"):
            fn()
    stream.synchronize()
    return g


def graph_functions(graph):
    """The device function of every kernel node of a graph, the (CUfunction,
    CUkernel) handles of cuGraphKernelNodeGetParams, in cuGraphGetNodes order;
    a memcpy node as the string "memcpy", a memset node as "memset". The driver
    API: the runtime bindings are another cudart instance, where torch's
    kernels are not registered."""
    from cuda.bindings import driver as drv

    from torch.cuda._utils import _check_cuda_bindings as check

    raw = graph.raw_cuda_graph()
    count = check(drv.cuGraphGetNodes(raw, 0))[1]
    nodes = check(drv.cuGraphGetNodes(raw, count))[0] if count else []
    kinds = {
        drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY: "memcpy",
        drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET: "memset",
    }
    out = []
    for node in nodes:
        kind = check(drv.cuGraphNodeGetType(node))
        if kind == drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            params = check(drv.cuGraphKernelNodeGetParams(node))
            out.append((int(params.func), int(params.kern)))
        else:
            out.append(kinds.get(kind, str(kind)))
    return out


def assert_eager_function_handles(test, real, args, entry=None, launches=None):
    """E36's gate for one call: the variant built from the tape of `real(*args)`
    (and the entry's own capture, when `entry` is given) holds at every node
    what eager's capture of the same call holds: the kind, and for a kernel
    node the function handle, so the replay launches eager's own kernel, not
    a twin. Returns eager's node list."""
    eager = graph_functions(capture_graph(lambda: real(*args)))
    if entry is not None:
        test.assertEqual(graph_functions(capture_graph(lambda: entry(*args))), eager)
    tape = ht.trace(real, args)
    variant = ht.build(tape, real, args)
    test.assertEqual(graph_functions(variant.graph), eager)
    if launches is not None:
        test.assertEqual(tape.num_launches, launches)
    return eager


def assert_no_disabled_memset(test, variant, what=""):
    """No memset node of a built exec is disabled: on driver 580.126.20 a kernel
    node behind a disabled memset node launches before the stream's prior work
    completes (a programmatic-dependent-launch pair in front makes it
    deterministic). Nothing in the stack disables a node (an exec holds exactly
    the capture's nodes); this reads the driver's view back. Returns the states."""
    states = exec_node_states(variant.graph)
    off = [i for i, kind, enabled in states if kind == "memset" and enabled is False]
    test.assertEqual(off, [], f"{what}: disabled memset node(s) {off}; nodes {states}")
    return states


class HostTraceTestCase(TestCase):
    def setUp(self):
        super().setUp()
        # every exec a suite builds is read back once: a build must never
        # leave a memset node disabled (assert_no_disabled_memset)
        if TEST_CUDA_PYTHON_BINDINGS and torch.cuda.is_available():
            build = ht.Variant._build

            def checked_build(variant, args):
                build(variant, args)
                assert_no_disabled_memset(self, variant, "after the build")

            patcher = mock.patch.object(ht.Variant, "_build", checked_build)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _capture_nodes(self, fn, memsets):
        # fn once eagerly, then under stream capture; the capture stream waits
        # for the current one, where the caller made the inputs
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
            g = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(g, stream=stream, capture_error_mode="relaxed"):
                out = fn()
        stream.synchronize()
        e = C._HostTraceExec(g, torch.cuda.current_device())
        kernels = [
            (
                e.kernel_name(j),
                tuple(e.grid(j)),
                tuple(e.block(j)),
                e.smem(j),
                e.image(j),
            )
            for j in range(e.num_nodes)
        ]
        # the memset accessors exist only on an exec that records memset nodes;
        # read only when asked
        memsets = (
            [
                (e.memset_dst(j), e.memset_bytes(j), e.memset_value(j))
                for j in range(e.num_memset_nodes)
            ]
            if memsets
            else None
        )
        return kernels, memsets, out

    def _capture(self, fn):
        # the kernel nodes one call produces: (name, grid, block, smem, image)
        kernels, _, out = self._capture_nodes(fn, memsets=False)
        return kernels, out

    def _capture_with_memsets(self, fn):
        # the kernel nodes and the memset nodes as (dst, bytes, value)
        return self._capture_nodes(fn, memsets=True)

    def _assert_bitwise(self, got, want, msg="outputs differ bitwise", *, stride=False):
        self.assertEqual(got.shape, want.shape)
        if stride:
            self.assertEqual(got.stride(), want.stride())
        self.assertEqual(got.dtype, want.dtype)
        self.assertTrue(torch.equal(bits(got), bits(want)), msg)

    def _replay_cases(
        self,
        fn,
        base_args,
        new_args_list,
        msg=None,
        atol=0,
        rtol=0,
        stride=False,
        **build_kw,
    ):
        # trace and build fn at base_args, then replay every case in turn: a
        # served replay's first output is compared with eager (bitwise unless a
        # tolerance is given; msg(args) names the case), a miss keeps its text
        tape = ht.trace(fn, base_args)
        variant = ht.build(tape, fn, base_args, **build_kw)
        cases = []
        for args in new_args_list:
            try:
                out = variant.replay(args)
            except ht.Miss as e:
                cases.append(Case(None, str(e)))
                continue
            want = fn(*args)
            torch.cuda.synchronize()
            if atol or rtol:
                self.assertEqual(out[0].shape, want.shape)
                self.assertEqual(out[0], want, atol=atol, rtol=rtol)
            else:
                what = msg(args) if msg else "outputs differ bitwise"
                self._assert_bitwise(out[0], want, what, stride=stride)
            cases.append(Case(out, None))
        return tape, variant, cases
