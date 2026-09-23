# Owner(s): ["module: inductor"]

from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._inductor.runtime._cudagraph import (
    direct_hosttrace as dh,
    hosttrace_partition as hp,
)
from torch._inductor.runtime.cudagraph_arg_mapping import IntegerInput, IntExpr
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch._inductor.runtime.cudagraph_compiled_evaluation import compile_evaluation
from torch.cuda import _host_trace as ht
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import CeilToInt, FloorDiv, FloorToInt, IntTrueDiv


def _boxed_batch(expressions, examples):
    names = tuple(f"arg{index}" for index in range(len(examples)))
    records = SimpleNamespace(
        input_names=names,
        integer_inputs=tuple(
            IntegerInput(name, index)
            for index, (name, value) in enumerate(zip(names, examples))
            if type(value) is int
        ),
    )
    numeric = _NumericProgram(records, examples)
    outputs = tuple(numeric.add(expression) for expression in expressions)
    compiled = compile_evaluation(numeric, pointer_count=len(examples))
    return torch._C._CUDAGraphBoxedNumeric(
        input_count=len(examples),
        numeric_plan=(numeric.integer_indices, tuple(numeric.instructions)),
        compiled_evaluation=torch._C._CUDAGraphCompiledEvaluation(
            **compiled.registration_kwargs
        ),
        output_indices=outputs,
    )


@instantiate_parametrized_tests
class TestBoxedPartitionNumeric(TestCase):
    @parametrize("parameter", (False, True))
    def test_metadata_projection_rebinds_the_same_tensor(self, parameter):
        tensor = torch.empty((3, 4))
        if parameter:
            tensor = torch.nn.Parameter(tensor, requires_grad=False)
        size = IntExpr("size", 0, (IntExpr("constant", 0),))
        stride = IntExpr("stride", 0, (IntExpr("constant", 0),))
        offset = IntExpr("storage_offset", 0)
        extent = IntExpr("add", args=(IntExpr("multiply", args=(size, stride)), offset))
        expressions = (extent, offset, size, stride, extent)
        batch = _boxed_batch(expressions, (tensor,))
        for rows, columns, row_stride, column_stride, storage_offset in (
            (3, 4, 4, 1, 0),
            (5, 3, 11, 2, 7),
            (2, 6, 7, 1, 1),
        ):
            storage = torch.empty(rows * row_stride + storage_offset)
            tensor.set_(
                storage.untyped_storage(),
                storage_offset,
                (rows, columns),
                (row_stride, column_stride),
            )
            box = [tensor, object()]
            result = batch(box)
            expected_extent = rows * row_stride + storage_offset
            self.assertEqual(
                result,
                (expected_extent, storage_offset, rows, row_stride, expected_extent),
            )
            self.assertTrue(all(type(value) is int for value in result))
            self.assertIs(box[0], tensor)
            self.assertEqual(len(box), 2)

    @parametrize(
        "operation,inputs",
        (
            ("add", ((1 << 63) - 1, 1)),
            ("multiply", ((1 << 63) - 1, 2)),
            ("floordiv", (1, 0)),
            ("and", (2, 1)),
        ),
    )
    def test_native_failure_returns_no_partial_results(self, operation, inputs):
        expression = IntExpr(operation, args=(IntExpr("boxed", 0), IntExpr("boxed", 1)))
        batch = _boxed_batch((IntExpr("constant", 7), expression), (1, 1))
        box = list(inputs)
        self.assertIsNone(batch(box))
        self.assertEqual(box, list(inputs))
        expected = 2 if operation == "add" else 1
        self.assertEqual(batch([1, 1]), (7, expected))

    @parametrize("value", (True, 1.0, "1", 1 << 63))
    def test_boxed_integer_requires_exact_int64(self, value):
        batch = _boxed_batch((IntExpr("boxed", 0),), (1,))
        exception = OverflowError if type(value) is int else TypeError
        with self.assertRaises(exception):
            batch([value])
        self.assertEqual(batch([7]), (7,))

    @parametrize("kind", ("subclass", "rank", "layout"))
    def test_metadata_rejection_does_not_poison_the_batch(self, kind):
        class TensorSubclass(torch.Tensor):
            pass

        tensor = torch.empty((2, 3))
        expression = IntExpr("size", 0, (IntExpr("constant", 1),))
        batch = _boxed_batch((expression,), (tensor,))
        if kind == "subclass":
            invalid = tensor.as_subclass(TensorSubclass)
            exception = TypeError
        elif kind == "rank":
            invalid = torch.empty(2)
            exception = ValueError
        else:
            invalid = tensor.to_sparse()
            exception = ValueError
        with self.assertRaises(exception):
            batch([invalid])
        self.assertEqual(batch([tensor]), (3,))

    @parametrize("kind", ("pointer", "fconst"))
    def test_native_batch_rejects_noninteger_programs(self, kind):
        expression = IntExpr(kind, 0)
        examples = (torch.empty(1),) if kind == "pointer" else ()
        with self.assertRaisesRegex(ValueError, "pure integer"):
            _boxed_batch((expression,), examples)


@instantiate_parametrized_tests
class TestPartitionBoundaryFallback(TestCase):
    def setUp(self):
        super().setUp()
        self.size, self.stride, self.offset = sympy.symbols(
            "size stride offset", integer=True
        )

    def _partition(self, expressions, operator, *, expected=None):
        symbols = dh._Symbols(SimpleNamespace(inputs=[]))
        symbols.by_symbol = {
            self.size: dh._Property("size", 0, 0),
            self.stride: dh._Property("stride", 0, 0),
            self.offset: dh._Property("offset", 0),
        }
        reference = hp._Ref(
            "p0", (self.size,), (self.stride,), self.offset, torch.float32
        )
        op = ht._OpRec(0, operator, -1, 0, (reference, *expressions), {}, 0, 0)
        op.seq[1] = 1
        op.guard_range[1] = 0
        op.outputs = reference if expected is None else expected
        tape = SimpleNamespace(
            shape_env=ht._TraceShapeEnv(),
            device=torch.device("cuda", 0),
            device_identity=(),
            nargs=1,
            positions=(0,),
            constants=(),
            inputs=[
                SimpleNamespace(
                    root=SimpleNamespace(name="p0"),
                    device=torch.device("cuda", 0),
                    pinned=False,
                )
            ],
            allocs=[],
            launches=[],
            opaque=[],
            rng_increment=None,
            all_on_capture_stream=True,
            written_roots=["p0"],
            written_inputs=(0,),
            memsets=[],
            host_buffers=[],
            memcpys=[],
            regions=[],
            outputs=[],
            root_facts=[],
            ops=[op],
            guards=[],
            kept_raw=[],
            guard_rows=[],
            guard_also={},
            raw_guards=lambda: (),
        )
        variant = SimpleNamespace(
            tape=tape, lowered=SimpleNamespace(device=0, symbols=symbols)
        )
        partition = hp.Partition(SimpleNamespace(), variant, (0,))
        partition.preflight = object()
        return partition

    def _serve(self, partition, tensor):
        # These CPU fixtures isolate boundary evaluation after the retained guard.
        with mock.patch.object(dh, "check_predicate", return_value=True):
            return partition.serve((tensor,), [tensor])

    def test_supported_values_rebind_without_python_evaluation(self):
        seen = []

        def operator(tensor, value):
            seen.append(value)
            return tensor

        partition = self._partition((self.size + self.stride + self.offset,), operator)
        tensor = torch.empty(32).as_strided((3,), (2,), 1)
        self._serve(partition, tensor)
        self.assertIsNotNone(partition._boundary)
        self.assertEqual(seen, [6])
        with mock.patch.object(partition.evaluator, "ev") as evaluate:
            tensor.set_(torch.empty(32).untyped_storage(), 3, (4,), (3,))
            self._serve(partition, tensor)
        evaluate.assert_not_called()
        self.assertEqual(seen, [6, 10])
        self.assertTrue(all(type(value) is int for value in seen))

    def test_native_binding_error_uses_python_before_one_cut(self):
        class TensorSubclass(torch.Tensor):
            pass

        seen = []

        def operator(tensor, value):
            seen.append(value)
            return tensor

        partition = self._partition((self.size + self.stride,), operator)
        tensor = torch.empty(3)
        self._serve(partition, tensor)
        self.assertIsNotNone(partition._boundary)
        seen.clear()
        subclass = tensor.as_subclass(TensorSubclass)
        with self.assertRaises(TypeError):
            partition._boundary([subclass])
        with mock.patch.object(partition, "_env", wraps=partition._env) as fallback:
            self._serve(partition, subclass)
        fallback.assert_called_once()
        self.assertEqual(seen, [4])

    @parametrize("kind", ("floor_float", "ceil_float", "boolean", "absolute"))
    def test_unsupported_expression_keeps_python_value_and_type(self, kind):
        if kind == "floor_float":
            expression = FloorToInt(IntTrueDiv(self.stride, 1, evaluate=False))
            stride, expected = (1 << 53) + 1, 1 << 53
        elif kind == "ceil_float":
            expression = CeilToInt(IntTrueDiv(self.stride, 1, evaluate=False))
            stride, expected = (1 << 53) + 1, 1 << 53
        elif kind == "boolean":
            expression = sympy.Gt(self.stride, 1)
            stride, expected = 3, True
        else:
            expression = sympy.Abs(self.stride - 4)
            stride, expected = 3, 1
        seen = []

        def operator(tensor, value):
            seen.append(value)
            return tensor

        partition = self._partition((expression,), operator)
        tensor = torch.empty(1).as_strided((1,), (stride,))
        self._serve(partition, tensor)
        self.assertIsNone(partition._boundary)
        self.assertEqual(seen, [expected])
        self.assertIs(type(seen[0]), type(expected))

    def test_wide_cancellation_falls_back_before_the_cut(self):
        square = sympy.Mul(self.stride, self.stride, evaluate=False)
        negative_square = sympy.Mul(-1, self.stride, self.stride, evaluate=False)
        expression = sympy.Add(square, negative_square, self.stride, evaluate=False)
        seen = []

        def operator(tensor, value):
            seen.append(value)
            return tensor

        partition = self._partition((expression,), operator)
        tensor = torch.empty(1).as_strided((1,), (2,))
        self._serve(partition, tensor)
        self.assertIsNotNone(partition._boundary)
        tensor.as_strided_((1,), (1 << 32,))
        self.assertIsNone(partition._boundary([tensor]))
        with mock.patch.object(partition, "_env", wraps=partition._env) as fallback:
            self._serve(partition, tensor)
        fallback.assert_called_once()
        self.assertEqual(seen, [2, 1 << 32])
        self.assertEqual(partition.serves, 2)

    @parametrize("result_kind", ("tensor", "integer"))
    def test_speculative_division_failure_keeps_lazy_error_timing(self, result_kind):
        expression = FloorDiv(1, self.stride - 1, evaluate=False)
        calls = []

        def operator(tensor):
            calls.append(1)
            tensor.add_(1)
            return tensor if result_kind == "tensor" else 0

        partition = self._partition((), operator, expected=expression)
        tensor = torch.zeros(1).as_strided((1,), (2,))
        partition._boundary = partition._compile_boundary([tensor])
        partition._boundary_ready = True
        self.assertIsNotNone(partition._boundary)
        tensor.as_strided_((1,), (1,))
        self.assertIsNone(partition._boundary([tensor]))
        error = hp.SwapMismatch if result_kind == "tensor" else ZeroDivisionError
        with self.assertRaises(error):
            self._serve(partition, tensor)
        self.assertEqual(calls, [1])
        self.assertEqual(tensor, torch.ones_like(tensor))

    def test_metadata_snapshot_precedes_cut_mutation(self):
        tensor = torch.ones(1)
        calls = []

        def operator(view):
            calls.append(1)
            tensor.resize_(2)
            return tensor

        partition = self._partition((), operator)
        with self.assertRaisesRegex(RuntimeError, "wrote or aliased an input"):
            self._serve(partition, tensor)
        self.assertIsNotNone(partition._boundary)
        self.assertEqual(calls, [1])
        self.assertEqual(tensor.shape, (2,))


if __name__ == "__main__":
    run_tests()
