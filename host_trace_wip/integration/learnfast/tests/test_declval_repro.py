# learnfast 10-18: repro for the core agent (GDN prefill template keys per layer). A layer's op whose output size mixes its
# weight's length (a static-prefix layout symbol) with the call's rows: core's _reads keys the mixed size by the variant's
# node, so each layer has its own template key and every layer is redone in full. With the key fixed: one redo for the
# first layer (it reads an argument), one for the second (it reads the first's output), the rest instances.
# TPL_DECLVAL=1 runs it with learnfast's adapter prototype (tpl_layer) installed.
import os
import sys

import torch
import triton
import triton.language as tl
from torch.cuda._host_trace_replay import HostTraceReplay
from torch.testing._internal.common_utils import run_tests, TestCase

if os.environ.get("TPL_DECLVAL") == "1":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import tpl_layer  # noqa: F401


@triton.jit
def _mul_cols(x_ptr, w_ptr, y_ptr, n, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < n
    tl.store(y_ptr + i, tl.load(x_ptr + i, mask=m) * tl.load(w_ptr + i % N, mask=m), mask=m)


@torch.library.custom_op("lf_repro::flat_mul", mutates_args=())
def flat_mul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    B = 128 if x.shape[0] <= 16 else 256
    y = torch.empty(x.shape[0] * w.shape[0], device=x.device, dtype=x.dtype)
    _mul_cols[(triton.cdiv(y.numel(), B),)](x, w, y, y.numel(), w.numel(), B=B)
    return y.view(x.shape[0], w.shape[0])


@flat_mul.register_fake
def _(x, w):
    return x.new_empty(x.shape[0], w.shape[0])


class TestDeclvalRepro(TestCase):
    def test_layers_share_a_template_with_a_mixed_static_prefix_size(self):
        layers = 4
        ws = [torch.randn(64, device="cuda") for _ in range(layers)]

        def fn(*a):
            *weights, x = a
            for w in weights:
                x = torch.ops.lf_repro.flat_mul(x, w)
            return x

        f = HostTraceReplay(fn, static_prefix=layers)
        for m in (8, 24, 8, 24):
            x = torch.randn(m, 64, device="cuda")
            self.assertEqual(f(*ws, x), fn(*ws, x), atol=0, rtol=0)
        print("traces, redispatches, redos, instances =", (f.traces, f.redispatches, f.redos, f.redo_instances), flush=True)
        self.assertEqual((f.traces, f.redispatches, f.redos, f.redo_instances), (1, 1, 2, layers - 2))


if __name__ == "__main__":
    run_tests()
