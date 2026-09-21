# Owner(s): ["module: inductor"]

import sys
from unittest import mock

import triton
import triton.language as tl

import torch
from torch._inductor.runtime._cudagraph import direct_host
from torch._inductor.runtime._cudagraph.api import (
    DirectHost,
    DirectTriton,
    InputContract,
    IntegerRange,
    IntExpr,
    TensorInput,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@triton.jit(
    do_not_specialize=["N", "STRIDE"],
    do_not_specialize_on_alignment=["N", "STRIDE"],
)
def add_bias(SOURCE, OUTPUT, N, STRIDE, BIAS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(SOURCE + index * STRIDE, index < N, other=0)
    tl.store(OUTPUT + index, values + BIAS, index < N)


ADD = None
KIND = None


def host(box):
    n, stride, source = box
    box.clear()
    output = torch.empty_strided((n,), (1,), dtype=source.dtype, device=source.device)
    bias = 1 if n < 256 else 2
    ADD[(triton.cdiv(n, 128),)](source, output, n, stride, BIAS=bias, BLOCK=128)
    if KIND == "owned":
        first, second = output, output
    elif KIND == "input":
        first, second = source, source
    else:
        first = output[1:]
        if KIND == "view" or KIND == "branch_views" and n < 256:
            second = first
        else:
            second = output[1:]
    return n, None, first, second, n


class TestOutputReference(TestCase):
    @parametrize("kind", ("owned", "view", "input", "distinct_views", "branch_views"))
    def test_output_identity_across_variants(self, device, kind):
        with torch.cuda.device(device):
            adapter = DirectTriton(add_bias)
            self.addCleanup(adapter.close)
            self.enterContext(mock.patch.dict(globals(), ADD=adapter, KIND=kind))
            count, stride = IntExpr("boxed", 0), IntExpr("boxed", 1)
            contract = InputContract(
                ("integer", "integer", "tensor"),
                (TensorInput(2, torch.float32, (count,), (stride,)),),
                (IntegerRange(0, 32, 1024), IntegerRange(1, 1, 4)),
                device_index=torch.cuda.current_device(),
            )
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            observations = self.enterContext(
                mock.patch.object(
                    direct_host, "_observe_direct", wraps=direct_host._observe_direct
                )
            )
            samples = []
            for n, stride, offset in (
                (128, 1, 1),
                (256, 2, 5),
                (512, 3, 1),
                (128, 2, 9),
                (384, 1, 13),
            ):
                backing = torch.randn(n * stride + offset, device=device)
                samples.append(backing[offset::stride])
            identities, held = [], []
            for step, source in enumerate(samples):
                n, stride = source.numel(), source.stride(0)
                box = [n, stride, source]
                previous = observations.call_count
                frames = []

                def profile(frame, event, result):
                    if event == "call":
                        frames.append(frame.f_code)

                try:
                    if step >= 2:
                        sys.setprofile(profile)
                    actual = (
                        runtime(box) if runtime.entry is None else runtime.entry(box)
                    )
                finally:
                    sys.setprofile(None)
                self.assertEqual(box, [])
                self.assertEqual(observations.call_count - previous, int(step < 2))
                self.assertEqual(len(runtime.variants), min(step + 1, 2))
                self.assertEqual(frames, [])
                expected = source if kind == "input" else source + (1 if n < 256 else 2)
                if kind in ("view", "distinct_views", "branch_views"):
                    expected = expected[1:]
                self.assertEqual(actual, (n, None, expected, expected, n))
                same = (
                    kind in ("owned", "view", "input")
                    or kind == "branch_views"
                    and n < 256
                )
                identities.append((actual[2] is actual[3], same))
                if kind == "input":
                    self.assertIs(actual[2], source)
                held.append((actual, (n, None, expected.clone(), expected.clone(), n)))
            self.assertEqual(
                len({source.data_ptr() for source in samples}), len(samples)
            )
            self.assertEqual(
                identities, [(expected, expected) for _, expected in identities]
            )
            runtime.close()
            for actual, expected in held:
                self.assertEqual(actual, expected)


instantiate_device_type_tests(TestOutputReference, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
