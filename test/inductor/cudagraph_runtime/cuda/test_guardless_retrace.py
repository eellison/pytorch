"""Actual specialization misses may append an unspecialized native variant."""

import json
import sys


import torch
import triton
import triton.language as tl
from torch._inductor.runtime._cudagraph import direct_host, replay
from torch._inductor.runtime._cudagraph.api import (
    DirectHost, DirectTriton, InputContract, IntegerRange, IntExpr,
    TensorInput,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


@triton.jit(do_not_specialize_on_alignment=["src", "dst"])
def k_add1(src, dst, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(dst + i, tl.load(src + i, m, other=0) + 1, m)


ADD1 = None


def host(box):
    n, x = box
    box.clear()
    out = torch.empty_strided((n,), (1,), dtype=x.dtype, device=x.device)
    ADD1[(triton.cdiv(n, 128),)](x, out, n, BLOCK=128)
    return (out,)


class TestGuardlessRetrace(TestCase):
    @parametrize("specialized_first", (True, False))
    def test_direct(self, device, specialized_first):
        global ADD1

        sizes = (128, 100, 128, 103, 100) if specialized_first else (100, 128, 103)
        expected_variants = 2 if specialized_first else 1
        with torch.cuda.device(device):
            ADD1 = DirectTriton(k_add1)
            self.addCleanup(ADD1.close)
            count = IntExpr("boxed", 0)
            contract = InputContract(
                ("integer", "tensor"), (TensorInput(1, torch.float32, (count,), (1,)),),
                (IntegerRange(0, 1, 1 << 20),), device_index=torch.cuda.current_device(),
            )
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            samples = [(n, torch.randn(n, device=device)) for n in sizes]
            self.assertEqual(len({value.data_ptr() for _, value in samples}), len(samples))
            counts = {"ordinary": 0, "trace": 0, "capture": 0, "python": 0}

            def profile(frame, event, result):
                if event != "call":
                    return
                counts["python"] += 1
                if frame.f_code is host.__code__ and type(frame.f_locals["box"][0]) is int:
                    counts["ordinary"] += 1
                if frame.f_code is direct_host.trace_host.__code__:
                    counts["trace"] += 1
                if frame.f_code is replay._make_replay.__code__:
                    counts["capture"] += 1

            held, reports = [], []
            entry = None
            for index, (n, value) in enumerate(samples):
                miss = index < expected_variants
                before = counts.copy()
                box = [n, value]
                try:
                    sys.setprofile(profile)
                    actual = runtime(box) if index == 0 else runtime.entry(box)
                finally:
                    sys.setprofile(None)
                self.assertEqual(box, [])
                self.assertEqual(counts["ordinary"] - before["ordinary"], int(miss))
                self.assertEqual(counts["trace"] - before["trace"], int(miss))
                self.assertEqual(counts["capture"] - before["capture"], int(miss))
                if not miss:
                    self.assertEqual(counts["python"], before["python"])
                if entry is None:
                    entry = runtime.entry
                    expected_type = (torch._C._CUDAGraphBoxedDispatch if specialized_first
                                     else torch._C._CUDAGraphBoxedReplay)
                    self.assertIs(type(entry), expected_type)
                self.assertIs(runtime.entry, entry)
                self.assertEqual(len(runtime.variants), min(index + 1, expected_variants))
                ordinary = host([n, value])
                self.assertEqual(actual, ordinary)
                self.assertEqual(actual, (value + 1,))
                self.assertNotEqual(actual[0].data_ptr(), ordinary[0].data_ptr())
                held.append((actual, tuple(output.clone() for output in actual)))
                reports.append({"n": n, "path": "ordinary_miss" if miss else "native_hit"})

            self.assertEqual(counts["ordinary"], expected_variants)
            self.assertEqual(counts["trace"], expected_variants)
            self.assertEqual(counts["capture"], expected_variants)
            if specialized_first:
                first, generic = runtime.variants
                self.assertTrue(first.guard.expressions)
                self.assertEqual(generic.guard.expressions, ())
                self.assertIn("return true;", generic.guard.cpp_source)
            else:
                self.assertIsNone(runtime.variants[0].guard)
            for variant in runtime.variants:
                variant.program.check()
            runtime.close()
            self.assertTrue(entry.closed)
            ADD1.close()
            for actual, expected in held:
                self.assertEqual(actual, expected)
            print("GUARDLESS_RETRACE_RESULT=" + json.dumps({
                "route": "direct", "accepted": True,
                "samples": reports, "variants": expected_variants,
                "ordinary_misses": counts["ordinary"], "traces": counts["trace"],
                "captures": counts["capture"], "native_hits": len(samples) - expected_variants,
                "ordinary_references": len(samples), "python_calls_on_native_hits": 0,
                "held_outputs_after_close": True,
            }, sort_keys=True), flush=True)


instantiate_device_type_tests(TestGuardlessRetrace, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
