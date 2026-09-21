# Owner(s): ["module: inductor"]

from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.runtime._cudagraph._compiler.host_program import Allocate
from torch._inductor.runtime._cudagraph.frontend import DirectKernelCall
from torch._inductor.runtime._cudagraph.replay import _CaptureCall, _release_steps
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource,
    CallArgument,
    ExpressionSource,
    InputSource,
    IntegerSource,
    IntExpr,
    OwnedBuffer,
    ParameterSource,
    PointerSource,
    storage_roots,
)
from torch._inductor.runtime.cudagraph_boxed_replay import _BoundCall
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestStorageRootDAG(TestCase):
    @parametrize("pointer", (False, True))
    def test_shared_nodes_are_visited_once_without_hashing_offsets(self, pointer):
        offset = IntExpr("boxed", 0)
        for _ in range(64):
            offset = IntExpr("multiply", args=(offset, offset))
        root = InputSource(2)
        leaf = (
            ParameterSource("pointer", 64, PointerSource(root, offset))
            if pointer
            else ParameterSource("constant", 64, 0)
        )
        expression = leaf
        for _ in range(64):
            expression = ParameterSource("add", 64, args=(expression, expression))
        accesses = 0
        getattribute = ParameterSource.__getattribute__

        def read(value, name):
            nonlocal accesses
            if name == "op":
                accesses += 1
                if accesses > 65:
                    raise AssertionError("shared parameter node was visited repeatedly")
            return getattribute(value, name)

        with (
            mock.patch.object(ParameterSource, "__getattribute__", read),
            mock.patch.object(
                ParameterSource,
                "__hash__",
                side_effect=AssertionError("parameter hash"),
            ),
            mock.patch.object(
                PointerSource, "__hash__", side_effect=AssertionError("pointer hash")
            ),
            mock.patch.object(
                IntExpr, "__hash__", side_effect=AssertionError("offset hash")
            ),
        ):
            self.assertEqual(storage_roots(expression), (root,) if pointer else ())
        self.assertEqual(accesses, 65)

    def test_first_use_order_matches_pointer_projection(self):
        first, second, third = InputSource(3), BufferSource("owned"), InputSource(0)
        a = ParameterSource("pointer", 64, PointerSource(first, IntExpr("constant", 4)))
        b = ParameterSource(
            "pointer", 64, PointerSource(second, IntExpr("constant", 8))
        )
        c = ParameterSource(
            "pointer", 64, PointerSource(third, IntExpr("constant", 12))
        )
        again = ParameterSource(
            "pointer", 64, PointerSource(first, IntExpr("constant", 16))
        )
        shared = ParameterSource("add", 64, args=(b, a))
        graph = ParameterSource(
            "select",
            64,
            args=(
                ParameterSource("constant", 1, 0),
                shared,
                ParameterSource(
                    "add",
                    64,
                    args=(c, ParameterSource("add", 64, args=(shared, again))),
                ),
            ),
        )
        self.assertEqual(graph.pointers, (b.value, a.value, c.value, again.value))
        self.assertEqual(storage_roots(graph), (second, first, third))
        self.assertEqual(
            storage_roots(graph), tuple(dict.fromkeys(p.root for p in graph.pointers))
        )

    def test_pointer_and_root_deduplication_remain_distinct(self):
        root = InputSource(0)
        first = PointerSource(root, IntExpr("constant", 4))
        equal = PointerSource(InputSource(0), IntExpr("constant", 4))
        different = PointerSource(root, IntExpr("constant", 12))
        leaves = tuple(
            ParameterSource("pointer", 64, pointer)
            for pointer in (first, equal, different)
        )
        graph = ParameterSource(
            "add",
            64,
            args=(
                ParameterSource("add", 64, args=leaves[:2]),
                leaves[2],
            ),
        )
        self.assertEqual(graph.pointers, (first, different))
        self.assertEqual(storage_roots(graph), (root,))
        self.assertIs(leaves[0].value, first)
        self.assertIs(leaves[1].value, equal)
        self.assertIs(leaves[2].value, different)

    def test_offsets_are_not_evaluated_or_simplified(self):
        calls = []

        def callback(args):
            calls.append(args)
            raise AssertionError("offset evaluation during root discovery")

        offset = IntExpr("call", (16, callback), (IntExpr("constant", 7),))
        pointer = PointerSource(BufferSource("owned"), offset)
        leaf = ParameterSource("pointer", 64, pointer)
        cancelled = ParameterSource("sub", 64, args=(leaf, leaf))
        self.assertEqual(storage_roots(cancelled), (pointer.root,))
        self.assertEqual(calls, [])
        self.assertIs(pointer.byte_offset, offset)

    @parametrize(
        "kind", ("input", "buffer", "pointer", "integer", "expression", "constant")
    )
    def test_nonparameter_sources_keep_their_contract(self, kind):
        root = InputSource(1)
        sources = {
            "input": root,
            "buffer": BufferSource("owned"),
            "pointer": PointerSource(root, IntExpr("constant", 4)),
            "integer": IntegerSource(7),
            "expression": ExpressionSource(IntExpr("boxed", 0)),
            "constant": ParameterSource("constant", 64, 9),
        }
        expected = (
            (sources[kind],)
            if kind in ("input", "buffer")
            else (root,)
            if kind == "pointer"
            else ()
        )
        self.assertEqual(storage_roots(sources[kind]), expected)

    def test_shared_addresses_preserve_last_use_and_all_select_branches(self):
        zero = IntExpr("constant", 0)
        first = ParameterSource("pointer", 64, PointerSource(InputSource(0), zero))
        second = ParameterSource("pointer", 64, PointerSource(InputSource(2), zero))
        cancelled = ParameterSource("sub", 64, args=(first, first))
        selected = ParameterSource(
            "select", 64, args=(ParameterSource("constant", 1, 0), first, second)
        )
        shared = selected
        for _ in range(32):
            shared = ParameterSource("add", 64, args=(shared, shared))
        grid = (IntExpr("constant", 1),) * 3
        output = OwnedBuffer(BufferSource("output"), torch.int64, (1,), (1,))
        arguments = [
            (
                CallArgument("address", 0, 0, "i64", source),
                CallArgument("output", 1, 1, "*i64", output.source),
            )
            for source in (cancelled, second, shared)
        ]
        calls = tuple(
            _CaptureCall(_BoundCall(args, None, grid), 1, (1, 1, 1), (), ())
            for args in arguments
        )
        program = SimpleNamespace(
            saved_input_indices=(0, 1, 2),
            allocations=(output,),
            outputs=(output,),
            events=(
                Allocate("output", torch.int64, (1,), (1,)),
                *tuple(DirectKernelCall(None, args, grid) for args in arguments),
            ),
        )
        self.assertEqual(
            _release_steps(program, calls),
            (
                ("drop", 1),
                ("allocate", 0),
                ("kernel", 0),
                ("kernel", 1),
                ("kernel", 2),
                ("drop", 0),
                ("drop", 2),
            ),
        )


if __name__ == "__main__":
    run_tests()
