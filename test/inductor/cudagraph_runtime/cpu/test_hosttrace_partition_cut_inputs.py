# Owner(s): ["module: inductor"]

from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._inductor.runtime._cudagraph import hosttrace_partition as hp
from torch.cuda import _host_trace as ht
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._pytree import _deregister_pytree_node, register_pytree_node
from torch.utils._sympy.functions import FloorDiv


@instantiate_parametrized_tests
class TestPartitionCutInputs(TestCase):
    def _plan(self, operator, args=(), kwargs=None, outputs=None, produced=()):
        # Exercise the actual planner and cut runner without a native segment.
        partition = object.__new__(hp.Partition)
        partition.tape = SimpleNamespace(outputs=[])
        partition.uses = {name: [] for name in produced}
        partition.out_plan = []
        partition.evaluator = ht._Evaluator()
        partition._input_index = {"p0": 0}
        partition._alloc_nbytes = dict.fromkeys(produced, 16)
        op = ht._OpRec(0, operator, -1, 0, args, {} if kwargs is None else kwargs, 0, 0)
        op.seq[1] = 1
        op.outputs = outputs
        cut = hp._Cut(op, set(produced))
        partition._plan_cut(0, cut)
        return partition, cut

    @parametrize("kind", ("empty", "nested"))
    def test_nested_and_empty_inputs(self, kind):
        seen = []

        def operator(*args, **kwargs):
            seen.append((args, kwargs))

        symbol = sympy.Symbol("n", integer=True)
        if kind == "empty":
            args, kwargs = ([], (), {}), {"options": {"empty": [], "pair": ()}}
            expected = (args, kwargs)
        else:
            args = ([{"number": symbol, "empty": ()}], {"values": [None, True, 2.5]})
            kwargs = {"options": {"n": symbol + 1, "empty": []}}
            expected = (
                ([{"number": 3, "empty": ()}], {"values": [None, True, 2.5]}),
                {"options": {"n": 4, "empty": []}},
            )
        partition, cut = self._plan(operator, args, kwargs)
        self.assertEqual(partition._run_cut(0, cut, {"n": 3}, [], {}, set()), [None])
        self.assertEqual(seen, [expected])
        if kind == "nested":
            self.assertIs(type(seen[0][0][1]["values"][1]), bool)
            self.assertIs(type(seen[0][0][1]["values"][2]), float)

    def test_symbolic_values_and_views_rebind_to_fresh_storage(self):
        size, stride, offset = sympy.symbols("size stride offset", integer=True)
        reference = hp._Ref("p0", (size,), (stride,), offset, torch.float32)
        seen = []

        def operator(values, *, number):
            seen.append((values[0], number))
            return values[0]

        partition, cut = self._plan(
            operator, ([reference],), {"number": size + offset}, reference
        )
        storage_a = torch.arange(32, dtype=torch.float32)
        storage_b = torch.arange(32, dtype=torch.float32) + 100
        for storage, rows, step, displacement in (
            (storage_a, 3, 2, 1),
            (storage_b, 4, 3, 2),
            (storage_a, 2, 4, 3),
        ):
            env = {"size": rows, "stride": step, "offset": displacement}
            known = set()
            (actual,) = partition._run_cut(0, cut, env, [storage], {}, known)
            expected = storage.as_strided((rows,), (step,), displacement)
            self.assertEqual(actual, expected)
            self.assertEqual(actual.stride(), (step,))
            self.assertEqual(actual.storage_offset(), displacement)
            self.assertEqual(
                actual.untyped_storage().data_ptr(),
                storage.untyped_storage().data_ptr(),
            )
            self.assertEqual(known, {storage.untyped_storage().data_ptr()})
            self.assertEqual(seen[-1][1], rows + displacement)
            self.assertIs(type(seen[-1][1]), int)
        self.assertIsNot(seen[0][0], seen[2][0])

    @parametrize("failed_index", (0, 1, 2))
    def test_materialization_error_preserves_order_before_operator(self, failed_index):
        symbols = sympy.symbols("first second third", integer=True)
        expressions = tuple(FloorDiv(1, value, evaluate=False) for value in symbols)
        operator = mock.Mock(return_value=None)
        partition, cut = self._plan(
            operator,
            ([expressions[0]], {"nested": expressions[1]}),
            {"last": expressions[2]},
        )
        env = {
            str(symbol): int(index != failed_index)
            for index, symbol in enumerate(symbols)
        }
        with mock.patch.object(
            partition, "_materialize", wraps=partition._materialize
        ) as materialize:
            with self.assertRaises(ZeroDivisionError):
                partition._run_cut(0, cut, env, [], {}, set())
        self.assertEqual(
            [call.args[0] for call in materialize.call_args_list],
            list(expressions[: failed_index + 1]),
        )
        operator.assert_not_called()

    def test_operator_mutation_gets_fresh_argument_containers(self):
        symbol = sympy.Symbol("n", integer=True)
        seen = []

        def operator(values, *, options):
            self.assertEqual(values, [{"number": options["number"]}, []])
            self.assertEqual(options["empty"], [])
            seen.append((values, values[0], options, options["empty"]))
            values[0].clear()
            values.append("changed")
            options["empty"].append("changed")
            options["number"] = -1

        partition, cut = self._plan(
            operator,
            ([{"number": symbol}, []],),
            {"options": {"number": symbol, "empty": []}},
        )
        for value in (3, 7, 3):
            partition._run_cut(0, cut, {"n": value}, [], {}, set())
        for index in range(4):
            self.assertEqual(len({id(call[index]) for call in seen}), 3)

    def test_prepared_inputs_skip_discovery_but_outputs_are_flattened(self):
        symbol = sympy.Symbol("n", integer=True)
        reference = hp._Ref("p0", (4,), (1,), 0, torch.float32)
        outputs = []

        def operator(values, *, options):
            result = {"result": [values[0], {"number": options["number"]}]}
            outputs.append(result)
            return result

        partition, cut = self._plan(
            operator,
            ([reference],),
            {"options": {"number": symbol}},
            {"result": [reference, {"number": symbol}]},
        )
        tensor = torch.arange(4, dtype=torch.float32)
        with (
            mock.patch.object(
                hp, "tree_map", side_effect=AssertionError("input rediscovery")
            ),
            mock.patch.object(hp, "tree_flatten", wraps=hp.tree_flatten) as flatten,
        ):
            for value in (3, 7):
                actual = partition._run_cut(0, cut, {"n": value}, [tensor], {}, set())
                self.assertEqual(actual, [tensor, value])
        self.assertEqual(flatten.call_count, 2)
        for call, output in zip(flatten.call_args_list, outputs):
            self.assertIs(call.args[0], output)

    @parametrize("alias", (False, True))
    def test_input_storage_is_known_before_fresh_output_validation(self, alias):
        reference = hp._Ref("p0", (4,), (1,), 0, torch.float32)
        fresh = hp._Ref("fresh", (4,), (1,), 0, torch.float32)
        calls = []

        def operator(values, *, options):
            calls.append(1)
            return options["tensor"] if alias else options["tensor"].clone()

        partition, cut = self._plan(
            operator,
            ([reference],),
            {"options": {"tensor": reference}},
            fresh,
            ("fresh",),
        )
        tensor = torch.arange(4, dtype=torch.float32)
        known, boundary = set(), {}
        if alias:
            with self.assertRaisesRegex(hp.SwapMismatch, "aliases an existing storage"):
                partition._run_cut(0, cut, {}, [tensor], boundary, known)
            self.assertEqual(boundary, {})
        else:
            (actual,) = partition._run_cut(0, cut, {}, [tensor], boundary, known)
            self.assertEqual(actual, tensor)
            self.assertIs(boundary["fresh"], actual)
            self.assertIn(actual.untyped_storage().data_ptr(), known)
        self.assertEqual(calls, [1])
        self.assertIn(tensor.untyped_storage().data_ptr(), known)

    def test_custom_pytree_retains_context_and_storage_scan(self):
        class Bundle:
            def __init__(self, values, label):
                self.values = values
                self.label = label

        register_pytree_node(
            Bundle,
            lambda bundle: (bundle.values, bundle.label),
            lambda values, label: Bundle(list(values), label),
        )
        self.addCleanup(_deregister_pytree_node, Bundle)
        symbol = sympy.Symbol("n", integer=True)
        reference = hp._Ref("p0", (4,), (1,), 0, torch.float32)
        seen = []

        def operator(bundle):
            seen.append(bundle)
            self.assertEqual(bundle.label, "context")
            return bundle.values[0]

        partition, cut = self._plan(
            operator, (Bundle([reference, symbol], "context"),), outputs=reference
        )
        for value in (3, 7):
            tensor = torch.full((4,), value, dtype=torch.float32)
            known = set()
            (actual,) = partition._run_cut(0, cut, {"n": value}, [tensor], {}, known)
            self.assertEqual(actual, tensor)
            self.assertEqual(seen[-1].values[1], value)
            self.assertIn(tensor.untyped_storage().data_ptr(), known)
        self.assertIsNot(seen[0], seen[1])


if __name__ == "__main__":
    run_tests()
