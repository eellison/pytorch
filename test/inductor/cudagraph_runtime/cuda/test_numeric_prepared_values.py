# Owner(s): ["module: inductor"]

from unittest import mock

import triton
import triton.language as tl

import torch
from torch._inductor.runtime._cudagraph.api import (
    DirectHost,
    DirectTriton,
    InputContract,
    IntegerRange,
    TensorInput,
)
from torch._inductor.runtime.cudagraph_arg_mapping import IntExpr
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@triton.jit(do_not_specialize_on_alignment=["source", "output"])
def add_one(source, output, count, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(source + offsets, offsets < count, other=0)
    tl.store(output + offsets, values + 1, offsets < count)


ADD = None


def host(box):
    count, source = box
    box.clear()
    output = torch.empty_strided(
        (2 * count,), (1,), dtype=source.dtype, device=source.device
    )
    tail = output[count:]
    ADD[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](
        source[count:], tail, count, BLOCK=128
    )
    return tail, count + 7


class TestNumericPreparedValues(TestCase):
    def test_capture_expression_offsets_and_symbolic_outputs(self, device):
        with torch.cuda.device(device):
            adapter = DirectTriton(add_one)
            self.addCleanup(adapter.close)
            self.enterContext(mock.patch.dict(globals(), {"ADD": adapter}))
            count = IntExpr("boxed", 0)
            length = IntExpr("multiply", args=(IntExpr("constant", 2), count))
            contract = InputContract(
                ("integer", "tensor"),
                (TensorInput(1, torch.float32, (length,), (1,)),),
                (IntegerRange(0, 2, 127),),
                device_index=torch.cuda.current_device(),
            )
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            held = []
            for size in (8, 16, 8):
                source = torch.randn(2 * size, device=device)
                output, scalar = runtime([size, source])
                expected = source[size:] + 1
                self.assertEqual(output, expected)
                self.assertEqual(scalar, size + 7)
                self.assertEqual(output.storage_offset(), size)
                held.append((output, expected))
            self.assertEqual(len(runtime.variants), 1)
            for output, expected in held:
                self.assertEqual(output, expected)


instantiate_device_type_tests(TestNumericPreparedValues, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
