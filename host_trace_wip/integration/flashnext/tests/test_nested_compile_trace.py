# Regression test for DESIGN_nested_compile.md: a torch.compile'd function called inside an opaque custom op (vLLM's
# CustomOp.maybe_compile'd forward_native inside vllm.moe_forward_shared under compile mode) must run as in stock in every
# call of a HostTraceReplay entry, the trace call included. Today the trace call returns its warm-up's result, the warm-up
# runs under the witness TorchDispatchMode, and dynamo skips compiling under a non-infra dispatch mode
# (torch/_dynamo/convert_frame.py "non-infra torch dispatch mode present"), so the nested function runs uncompiled there:
# test_trace_call_runs_stock_compiled_code fails until the correctness fix (option (b)) lands. Replays are expected to match
# already (the op is an eager step that reruns the compiled function).
#   gpu3.sh t_nc bash python_fn.sh tests/test_nested_compile_trace.py
import torch
import torch.nn.functional as F

import torch.cuda._host_trace_replay as R

if __import__("os").environ.get("FN_WITNESS_INFRA") == "1":  # the fix (b) prototype (scripts/witness_infra.py)
    import sys as _sys

    _sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))), "scripts"))
    import witness_infra

    witness_infra.install()
from torch.testing._internal.common_utils import run_tests, TestCase

H, I = 256, 512


def silu_mul_native(x):
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


silu_mul_compiled = torch.compile(silu_mul_native, dynamic=True)


@torch.library.custom_op("fntest::shared_mlp", mutates_args=())
def shared_mlp(x: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    return silu_mul_compiled(x @ w_up) @ w_down


@shared_mlp.register_fake
def _(x, w_up, w_down):
    return x.new_empty(x.shape[0], w_down.shape[1])


def step(x, w_up, w_down, bias):
    return (torch.ops.fntest.shared_mlp(x, w_up, w_down) * 2 + bias,)


class TestNestedCompileTrace(TestCase):
    def _calls(self):
        torch.manual_seed(0)
        w_up = torch.randn(H, 2 * I, device="cuda", dtype=torch.bfloat16) * 0.05
        w_down = torch.randn(I, H, device="cuda", dtype=torch.bfloat16) * 0.05
        bias = torch.randn(H, device="cuda", dtype=torch.bfloat16)
        for n in (3, 17):  # the compiled function warmed up eagerly first, as vLLM's warm-up does
            step(torch.randn(n, H, device="cuda", dtype=torch.bfloat16), w_up, w_down, bias)
        entry = R.HostTraceReplay(step)
        out = []
        for n in (4, 8, 4, 8, 64, 4, 64):
            x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16)
            want = step(x, w_up, w_down, bias)[0]
            t0, r0 = entry.traces, entry.replays
            got = entry(x, w_up, w_down, bias)[0]
            torch.cuda.synchronize()
            path = "trace" if entry.traces > t0 else "replay" if entry.replays > r0 else "eager"
            out.append((n, path, got, want))
        return out

    def test_the_probe_distinguishes_compiled_from_uncompiled(self):
        x = torch.randn(64, 2 * I, device="cuda", dtype=torch.bfloat16)
        self.assertFalse(torch.equal(silu_mul_compiled(x), silu_mul_native(x)), "pick shapes where Inductor's fused bf16 differs")

    def test_replays_run_stock_compiled_code(self):
        for n, path, got, want in self._calls():
            if path == "replay":
                self.assertEqual(got, want, atol=0, rtol=0, msg=f"replay at {n}")

    def test_trace_call_runs_stock_compiled_code(self):
        for n, path, got, want in self._calls():
            if path == "trace":
                self.assertEqual(got, want, atol=0, rtol=0, msg=f"trace call at {n}")


if __name__ == "__main__":
    run_tests()
