# Owner(s): ["module: inductor"]

from contextlib import nullcontext
from dataclasses import replace
from unittest import mock

import sympy

import torch
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    IntegerRange,
)
from torch._inductor.runtime._cudagraph.api import InputContract, TensorInput
from torch._inductor.runtime._cudagraph.direct_host import _direct_origin
from torch._inductor.runtime._cudagraph.extraction import trace_host
from torch._inductor.runtime._cudagraph.guard_export import (
    export_guards,
    GuardExportDeclined,
)
from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource, IntExpr
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import FloorDiv, PythonMod


def allocated_view(box):
    n, tensor = box
    box.clear()
    storage = torch.empty_strided(
        (n + 4,), (1,), dtype=tensor.dtype, device=tensor.device
    )
    view = storage.as_strided((n,), (1,), 2)
    return view, view.data_ptr()


def extract():
    n = IntExpr("boxed", 0)
    contract = InputContract(
        ("integer", "tensor"),
        (TensorInput(1, torch.float32, (n,), (1,)),),
        (IntegerRange(0, 1, 8192),),
        device_index=0,
    )
    origin, _ = _direct_origin(allocated_view, contract)
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        trace = trace_host(
            allocated_view,
            contract,
            [8, torch.empty(8)],
            (),
            None,
            direct=True,
            context_factory=lambda state: nullcontext(),
        )
    return replace(trace, compiler_binding=origin)


@instantiate_parametrized_tests
class TestOwnedAddressGuards(TestCase):
    @parametrize("modulus", (8, 16, 32))
    def test_intermediate_view_alignment_uses_symbolic_root(self, modulus):
        trace = extract()
        view, address = trace.outputs
        (binding,) = (
            value
            for value in trace.address_bindings
            if type(value.root) is BufferSource
        )
        self.assertIsNone(binding.value.node.hint)
        self.assertEqual(binding.alignment, 256)
        self.assertEqual(trace.tensor_roots(view).byte_offset, 8)
        self.assertEqual(sympy.simplify(address.node._expr - binding.symbol), 8)
        relation = sympy.Eq if modulus == 8 else sympy.Ne
        guard = relation(PythonMod(address.node._expr, modulus), 0, evaluate=False)
        result = export_guards(trace, (guard,))
        self.assertIn(guard, result.discharged)
        self.assertFalse(
            any(str(binding.symbol) in value for value in result.expressions)
        )

    def test_incompatible_view_alignment_declines(self):
        trace = extract()
        address = trace.outputs[1].node._expr
        guard = sympy.Eq(PythonMod(address, 16), 0, evaluate=False)
        with self.assertRaisesRegex(
            GuardExportDeclined, "not proven before allocation"
        ):
            export_guards(trace, (guard,))

    def test_missing_allocator_contract_declines(self):
        trace = extract()
        trace = replace(
            trace,
            address_bindings=tuple(
                replace(binding, alignment=16)
                if type(binding.root) is BufferSource
                else binding
                for binding in trace.address_bindings
            ),
        )
        with self.assertRaisesRegex(GuardExportDeclined, "allocator alignment"):
            export_guards(trace)

    @parametrize("operation", (FloorDiv, PythonMod))
    def test_alignment_does_not_prove_partial_operation_domain(self, operation):
        trace = extract()
        (binding,) = (
            value
            for value in trace.address_bindings
            if type(value.root) is BufferSource
        )
        divisor = PythonMod(binding.symbol, 512, evaluate=False)
        value = operation(12, divisor, evaluate=False)
        guard = sympy.Ge(value, 0, evaluate=False)
        with self.assertRaisesRegex(GuardExportDeclined, "operation domain"):
            export_guards(trace, (guard,))


if __name__ == "__main__":
    run_tests()
