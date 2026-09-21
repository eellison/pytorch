# Owner(s): ["module: inductor"]

from contextlib import nullcontext
from dataclasses import replace
from unittest import mock

import torch
from torch._inductor.runtime._cudagraph.api import (
    InputContract,
    IntegerRange,
    TensorInput,
)
from torch._inductor.runtime._cudagraph.direct_host import _direct_origin
from torch._inductor.runtime._cudagraph.extraction import trace_host
from torch._inductor.runtime._cudagraph.frontend import lower_terminal
from torch._inductor.runtime.cudagraph_arg_mapping import (
    bind_output_slots,
    BorrowedInputOutput,
    BufferSource,
    InputSource,
    IntegerOutput,
    IntExpr,
    OutputReference,
    OwnedBuffer,
    TensorViewOutput,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def host(box):
    n, source = box
    box.clear()
    output = torch.empty_strided((n,), (1,), dtype=source.dtype, device=source.device)
    view = output[1:]
    distinct = output[1:]
    return None, n, output, output, view, view, distinct, source, source, n


@instantiate_parametrized_tests
class TestOutputReference(TestCase):
    def test_symbolic_lowering_preserves_tensor_object_identity(self):
        contract = InputContract(
            ("integer", "tensor"),
            (TensorInput(1, torch.float32, (IntExpr("boxed", 0),), (1,)),),
            (IntegerRange(0, 2, 1024),),
            device_index=0,
        )
        origin, _ = _direct_origin(host, contract)
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            trace = trace_host(
                host,
                contract,
                [8, torch.empty(8)],
                (),
                None,
                direct=True,
                context_factory=lambda state: nullcontext(),
            )
        program = lower_terminal(replace(trace, compiler_binding=origin), ())
        self.addCleanup(program.close)
        self.assertEqual(program.outputs[3], OutputReference(2))
        self.assertEqual(program.outputs[5], OutputReference(4))
        self.assertEqual(program.outputs[8], OutputReference(7))
        self.assertIs(type(program.outputs[6]), TensorViewOutput)
        self.assertIs(type(program.outputs[1]), IntegerOutput)
        self.assertIs(type(program.outputs[9]), IntegerOutput)
        self.assertIsNone(program.outputs[0])

    def test_reference_bindings_preserve_previous_tensor_slots(self):
        allocation = OwnedBuffer(BufferSource("output"), torch.float32, (8,), (1,))
        outputs = (
            BorrowedInputOutput(InputSource(0)),
            OutputReference(0),
            OutputReference(1),
            allocation,
            OutputReference(3),
            TensorViewOutput(allocation.source, (7,), (1,), 1),
            OutputReference(5),
        )
        slots = bind_output_slots(
            outputs, (allocation,), ("source",), set(), symbolic=True
        )
        self.assertEqual(slots, (*outputs[:3], 0, *outputs[4:]))

    @parametrize("index", (-1, 0, 1, 2, 4, True))
    def test_reference_requires_a_preceding_tensor(self, index):
        outputs = (
            None,
            IntegerOutput(3),
            OutputReference(index),
            BorrowedInputOutput(InputSource(0)),
        )
        self.assertIsNone(
            bind_output_slots(outputs, (), ("source",), set(), symbolic=True)
        )


if __name__ == "__main__":
    run_tests()
