# Owner(s): ["module: inductor"]
"""Integer min/max payloads remain dynamic through preparation and compiled evaluation."""

from types import SimpleNamespace

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.runtime._cudagraph.host_trace import _HostIntegers
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch._inductor.runtime.cudagraph_compiled_evaluation import (
    compile_numeric,
    EarlyStatus,
    integer_payload_contract,
)
from torch._inductor.runtime.cudagraph_host_trace_mapping import HostTraceInputMetadata
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import Max, Min


@instantiate_parametrized_tests
class TestHostMinMaxPayload(TestCase):
    @parametrize("kind", ("torch_min", "torch_max", "sympy_min", "sympy_max"))
    @parametrize("arity", (2, 4))
    @parametrize("with_contract", (False, True))
    def test_unsimplified_dynamic_payload(self, kind, arity, with_contract):
        env = ShapeEnv(duck_shape=False, specialize_zero_one=False)
        symbols = tuple(
            env.create_unspecified_symbol(hint, LocalSource(name), DimDynamic.DYNAMIC)
            for name, hint in zip(("a", "b", "c"), (4, 8, 16))
        )
        mapping = SimpleNamespace(
            metadata_symbols={
                symbol: HostTraceInputMetadata(0, "size", dimension)
                for dimension, symbol in enumerate(symbols)
            },
            tape=SimpleNamespace(shape_env=env),
            translate=lambda value: value,
        )
        constructors = {
            "torch_min": Min,
            "torch_max": Max,
            "sympy_min": sympy.Min,
            "sympy_max": sympy.Max,
        }
        a, b, c = symbols
        terms = (a - 10, b + 3, c - 20, sympy.Integer(7))[:arity]
        expression = constructors[kind](*terms)
        self.assertEqual(len(expression.args), arity)
        contract = integer_payload_contract(env) if with_contract else None
        lower = _HostIntegers(mapping, contract)
        result = lower(expression)
        numeric = _NumericProgram(
            SimpleNamespace(input_names=("x",), integer_inputs=()),
            (torch.empty(4, 8, 16),),
        )
        output = numeric.add(result)
        compiled = compile_numeric(numeric)
        operation = min if kind.endswith("min") else max
        self.assertEqual(numeric.values[output], operation((-6, 11, -4, 7)[:arity]))
        winners = set()
        for shape in ((4, 8, 16), (40, 2, 3), (40, 20, 40), (3, 4, 3), (10, 4, 27)):
            values = (shape[0] - 10, shape[1] + 3, shape[2] - 20, 7)[:arity]
            expected = operation(values)
            winners.add(values.index(expected))
            status, actual = compiled.evaluate_leaves(
                compiled.bind_inputs((torch.empty(shape),))
            )
            self.assertEqual(status, EarlyStatus.SUCCESS)
            self.assertEqual(actual[output], expected)
        self.assertGreater(len(winners), 1)
        self.assertEqual(env.guards, [])


if __name__ == "__main__":
    run_tests()
