# Owner(s): ["module: cuda graphs"]

# Tests that import SGLang or vLLM, in their own process: importing them changes
# process state (SGLang's common.py replaces triton.next_power_of_2)

import importlib.util
import unittest

import torch
from torch.cuda import _host_trace_replay
from torch.cuda._host_trace_launch import KernelLaunch
from torch.cuda._host_trace_tape import trace
from torch.testing._internal.common_utils import (
    requires_cuda_python_bindings,
    run_tests,
    TEST_CUDA,
    TestCase,
)


class HostTraceReplay(_host_trace_replay.HostTraceReplay):
    # traces at its first call, as in test_cuda_host_trace_cute
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._called = True


try:
    import cutlass  # noqa: F401

    HAS_CUTE = True
except ImportError:
    HAS_CUTE = False

HAS_SGLANG = importlib.util.find_spec("sglang") is not None

if HAS_CUTE:
    from torch.cuda import _host_trace_cute as htc

    # torch.cuda._host_trace_replay was imported before cutlass: compiles
    # from here on are described
    htc.install()


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@unittest.skipIf(not HAS_CUTE, "requires the CuTe DSL")
@requires_cuda_python_bindings
class TestHostTraceSGLang(TestCase):
    @unittest.skipIf(
        not TEST_CUDA or torch.cuda.get_device_capability() < (10, 0),
        "requires Blackwell",
    )
    @unittest.skipIf(not HAS_SGLANG, "requires SGLang")
    def test_sglang_tgv_gemm(self):
        # cute.experimental, without TVM-FFI: its own passes add the TMA
        # descriptors to the launch; 2-CTA clusters, PDL
        try:
            from sglang.kernels.ops.gemm import cutedsl_bf16_gemm as tgv
        except ImportError:
            raise unittest.SkipTest("requires SGLang") from None

        def bf16(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

        def f(x, w):
            return tgv._tgv_bf16_gemm_run(x, w, None)

        w = bf16(4096, 4096)
        tactics = {tgv._pick_tactic(m, 4096, 4096) for m in (1, 3, 8, 13, 16)}
        self.assertEqual(len(tactics), 1)
        ((_, launch),) = trace(f, (bf16(8, 4096), w)).launches
        self.assertIsInstance(launch, KernelLaunch)
        self.assertEqual(launch.cluster, (2, 1, 1))
        self.assertTrue(launch.programmatic)
        self.assertEqual(len(launch.descriptors), 2)
        r = HostTraceReplay(f)
        for m in (8, 1, 13, 16, 3, 8):
            x = bf16(m, 4096)
            self.assertEqual(r(x, w), f(x, w), atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager), (1, 5, 0))


def setUpModule():
    from torch.cuda import _host_trace_hint_audit
    import torch.cuda._host_trace_capture as capture

    _host_trace_hint_audit.enable_for_tests()
    capture.raise_trace_disagreements = True
    torch.cuda._host_trace.raise_unexpected = True


if __name__ == "__main__":
    run_tests()
