# Owner(s): ["module: inductor"]

from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.runtime._cudagraph import direct_hosttrace, hosttrace_partition
from torch.cuda import _host_trace
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import FloorDiv


@instantiate_parametrized_tests
class TestHostTracePartitionPreflight(TestCase):
    def _environment(self):
        environment = _host_trace._TraceShapeEnv()
        rows = environment.create_symbol(8, LocalSource("rows"))
        return environment, rows

    def _record(self, environment, expression, row):
        environment.attribute = lambda env, expr: row
        environment.evaluate_expr(expression, hint=True)

    def _partition(self, environment):
        ops = []
        for index, parent, depth, sequence in (
            (0, -1, 0, 0),
            (1, 0, 1, 0),
            (2, -1, 0, 2),
        ):
            op = _host_trace._OpRec(
                index, torch.ops.aten.add_.Tensor, parent, depth, (), {}, sequence, 0
            )
            op.seq[1] = sequence + 1
            op.guard_range[1] = len(environment.guards)
            op.route = "entry"
            ops.append(op)
        guards, _, _, kept_raw = environment.tape_guards()
        tape = SimpleNamespace(
            shape_env=environment,
            device=torch.device("cuda", 0),
            device_identity=(),
            nargs=0,
            positions=(),
            constants={},
            inputs=[],
            allocs=[],
            launches=[],
            opaque=[],
            rng_increment=None,
            all_on_capture_stream=True,
            written_roots=[],
            written_inputs=(),
            memsets=[],
            host_buffers=[],
            memcpys=[],
            regions=[],
            outputs=[],
            root_facts=[
                ("domain", str(s), 1, None)
                for s in environment.backed_var_to_val
                if s.is_positive
            ],
            ops=ops,
            guards=guards,
            kept_raw=kept_raw,
            guard_rows=list(environment.guard_rows),
            guard_also=dict(environment.guard_also),
            raw_guards=lambda: tuple(g.expr for g in environment.guards),
        )
        lowered = SimpleNamespace(
            device=0, symbols=SimpleNamespace(opaque={}, by_symbol={})
        )
        variant = SimpleNamespace(tape=tape, lowered=lowered)
        return hosttrace_partition.Partition(SimpleNamespace(), variant, (0,))

    @parametrize(
        "rows,retained",
        (
            (((0, 0, "metadata"),), True),
            (((-1, 0, "python"),), True),
            (((1, 1, "kernel"),), False),
            (((1, 1, "kernel"), (0, 0, "metadata")), True),
            (((1, 1, "kernel"), (-1, 0, "python")), True),
            (((1, 1, "kernel"), (2, 1, "other kernel")), True),
            (((0, 1, "kernel"), (1, 1, "nested kernel")), False),
            (((2, 1, "other kernel"),), True),
        ),
    )
    def test_every_guard_raiser_must_belong_to_a_cut(self, rows, retained):
        environment, symbol = self._environment()
        guard = sympy.Le(symbol, 16)
        for row in rows:
            self._record(environment, guard, row)
        self.assertEqual(len(environment.guards), 1)
        if len(rows) > 1:
            self.assertEqual(environment.guard_also[0], list(rows[1:]))
        partition = self._partition(environment)
        self.assertEqual(partition.guard_tape.guards, [guard] if retained else [])
        self.assertEqual(partition.guard_tape.root_facts, partition.tape.root_facts)
        self.assertIs(partition.guard_tape.inputs, partition.tape.inputs)
        self.assertTrue(partition.guard_tape.empty())

    def test_raw_metadata_guard_survives_a_removed_kernel_pin(self):
        environment, symbol = self._environment()
        kernel_guard = sympy.Eq(symbol, 8)
        metadata_guard = sympy.Le(symbol, 16)
        self._record(environment, kernel_guard, (0, 1, "kernel"))
        self._record(environment, metadata_guard, (0, 0, "metadata"))
        self.assertEqual(environment.tape_guards()[0], [kernel_guard])
        partition = self._partition(environment)
        self.assertEqual(partition.guard_tape.guards, [metadata_guard])
        (guard,) = partition.guard_tape.guards
        self.assertIs(guard.xreplace({symbol: sympy.Integer(12)}), sympy.true)
        self.assertIs(guard.xreplace({symbol: sympy.Integer(32)}), sympy.false)

    def test_original_partial_operation_guard_order_is_retained(self):
        environment, symbol = self._environment()
        divisor = environment.create_symbol(2, LocalSource("divisor"), positive=None)
        nonzero = sympy.Ne(divisor, 0)
        quotient = sympy.Le(FloorDiv(symbol, divisor, evaluate=False), 16)
        self._record(environment, nonzero, (0, 0, "domain"))
        self._record(environment, quotient, (0, 0, "metadata"))
        partition = self._partition(environment)
        self.assertEqual(partition.guard_tape.guards, [nonzero, quotient])
        self.assertIs(partition.guard_tape.guards[0], nonzero)
        self.assertIs(partition.guard_tape.guards[1], quotient)

    def test_cached_preflight_rejects_before_running_a_cut(self):
        environment, symbol = self._environment()
        self._record(environment, sympy.Le(symbol, 16), (0, 0, "metadata"))
        partition = self._partition(environment)
        with (
            mock.patch.object(direct_hosttrace, "lower_tape") as lower,
            mock.patch.object(
                direct_hosttrace, "check_predicate", return_value=False
            ) as check,
            mock.patch.object(partition, "_run_cut") as cut,
        ):
            self.assertIsNone(partition.serve((), []))
            self.assertIsNone(partition.serve((), []))
        lower.assert_called_once_with(partition.guard_tape, (), device=0)
        self.assertEqual(check.call_count, 2)
        check.assert_has_calls(
            [mock.call(lower.return_value, [], regions=False)] * 2,
        )
        cut.assert_not_called()
        self.assertFalse(partition.built)
        self.assertEqual(partition.serves, 0)


if __name__ == "__main__":
    run_tests()
