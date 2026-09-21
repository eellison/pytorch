"""Generated autotuned kernels through the input-only terminal policy."""

import json
import os
from pathlib import Path
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import runtime_support as support

import torch
from torch._inductor.runtime._cudagraph.frontend import TerminalCall
from torch._inductor.runtime._cudagraph.api import NativeTerminalPolicy
from torch._inductor import config
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests


class TestTerminalPolicyGenerated(support.MixedTestCase):
    @parametrize("kind", ("gemm", "multi"))
    def test_autotuned_kernel(self, device, kind):
        policy = NativeTerminalPolicy()
        self.addCleanup(policy.close)
        gemm = kind == "gemm"
        self.enterContext(config.patch(
            cudagraph_policy=policy, max_autotune=gemm, max_autotune_gemm_backends="TRITON",
            benchmark_epilogue_fusion=False, deterministic=False, multi_kernel_hints=[],
            **{"test_configs.max_mm_configs": 2, "triton.native_matmul": False,
               "triton.enable_persistent_tma_matmul": False, "triton.multi_kernel": 0 if gemm else 1},
        ))
        self.enterContext(mock.patch.dict(os.environ, {"TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE": "1"}))
        programs = []
        prepare_terminal = support.terminal_policy.prepare_terminal

        def observe(program, inputs):
            calls = tuple(event for event in program.events if type(event) is TerminalCall)
            self.assertEqual(len(calls), 1)
            self.assertEqual(any(type(view.carrier) is MultiKernelCall for view in program.views), not gemm)
            for call in calls:
                module = call.provider.launchers[0].__globals__["runner"].__self__
                self.modules[id(module)] = module
            programs.append(program)
            return prepare_terminal(program, inputs)

        self.enterContext(mock.patch.object(support.terminal_policy, "prepare_terminal", observe))

        def function(*args):
            return (torch.mm(*args),) if gemm else (args[0].softmax(-1),)

        def sample(rows):
            left = torch.randn(rows, 128 if gemm else 1024, device=device,
                               dtype=torch.float16 if gemm else torch.float32) * 0.25
            torch._dynamo.mark_dynamic(left, 0, min=2, max=128)
            torch._dynamo.mark_static(left, 1)
            if not gemm:
                return (left,)
            right = torch.randn(128, 128, device=device, dtype=torch.float16) * 0.25
            torch._dynamo.mark_static(right)
            return left, right

        report = {"kind": kind, "accepted": False}
        try:
            compiled = torch.compile(function, fullgraph=True, dynamic=True)
            before = support.counters["stats"]["unique_graphs"]
            inputs = sample(64)
            actual = compiled(*inputs)
            self.assertEqual(actual, function(*inputs))
            self.assertEqual(len(policy.artifacts), 1)
            installation, = policy.installations
            self.assertEqual(installation.status, "ready", installation.decline)
            artifact, original = policy.artifacts[0]
            entry = installation.entry
            self.assertIs(type(entry), torch._C._CUDAGraphBoxedReplay)
            self.assertNotIn("_cudagraph_call_records", original.__dict__)
            reference = support.original_reference(self, compiled, artifact, original, entry, inputs)
            self.assertEqual(actual, reference)
            held, held_inputs = [(actual, reference)], [inputs]
            observer = support.NativeObservation(installation)
            observer.forbidden_files.update(str(path) for path in support.IMPLEMENTATION)
            for rows in (2, 7, 35, 96):
                inputs = sample(rows)
                self.assertNotIn(inputs[0].data_ptr(), [values[0].data_ptr() for values in held_inputs])
                held_inputs.append(inputs)
                with observer:
                    actual = compiled(*inputs)
                reference = support.original_reference(self, compiled, artifact, original, entry, inputs)
                self.assertEqual(actual, reference)
                held.append((actual, reference))
            self.assertEqual(len(programs), 1)
            self.assertEqual(len(observer.calls), 4)
            self.assertEqual(observer.errors, [])
            self.assertEqual(observer.frames, {})
            self.assertEqual(observer.pending, {})
            self.assertEqual(support.counters["stats"]["unique_graphs"] - before, 1)
            torch.cuda.synchronize(device)
            policy.close()
            self.assertTrue(entry.closed)
            for actual, reference in held:
                self.assertEqual(actual, reference)
            report.update(accepted=True, native_hits=4, original_compiled_references=5,
                          old_metadata_or_host_records=False, outputs_survive_close=True)
        finally:
            print("TERMINAL_GENERATED_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


instantiate_device_type_tests(TestTerminalPolicyGenerated, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
