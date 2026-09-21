# Owner(s): ["module: inductor"]
"""Compiled integer payloads and combined evaluator library ownership."""

import gc
from types import SimpleNamespace

import torch
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource,
    IntegerInput,
    IntExpr,
    ParameterSource,
    PointerSource,
)
from torch._inductor.runtime.cudagraph_boxed_replay import (
    _NumericProgram,
    _ParameterProgram,
)
from torch._inductor.runtime.cudagraph_compiled_evaluation import (
    compile_evaluation,
    compile_numeric,
    EarlyStatus,
    LateStatus,
    MAX_I64,
    MIN_I64,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _numeric_program(examples):
    names = tuple(f"arg{index}" for index in range(len(examples)))
    records = SimpleNamespace(
        input_names=names,
        integer_inputs=tuple(
            IntegerInput(names[index], index)
            for index, value in enumerate(examples)
            if type(value) is int
        ),
    )
    return _NumericProgram(records, examples)


@instantiate_parametrized_tests
class TestCompiledNumeric(TestCase):
    @parametrize("op", ("add", "multiply"))
    @parametrize(
        "operands",
        (
            (0, 0),
            (MIN_I64, 1),
            (MAX_I64, 1),
            (MIN_I64, -1),
            (MAX_I64, -1),
            (MAX_I64, 2),
            (MIN_I64, 2),
            (MIN_I64, MIN_I64),
            (MAX_I64, MAX_I64),
            (-3037000499, 3037000499),
            (3037000500, 3037000500),
        ),
    )
    def test_checked_arithmetic(self, op, operands):
        expression = IntExpr(op, args=(IntExpr("boxed", 0), IntExpr("boxed", 1)))
        program = _numeric_program((2, 3))
        output = program.add(expression)
        compiled = compile_numeric(program)
        status, values = compiled.evaluate_leaves(compiled.bind_inputs(operands))
        left, right = operands
        expected = left + right if op == "add" else left * right
        if MIN_I64 <= expected <= MAX_I64:
            self.assertEqual(status, EarlyStatus.SUCCESS)
            self.assertEqual(values[output], expected)
            reference = _numeric_program(operands)
            reference.add(expression)
            self.assertEqual(values, tuple(reference.values))
        else:
            self.assertEqual(
                status,
                EarlyStatus.ADD_OVERFLOW
                if op == "add"
                else EarlyStatus.MULTIPLY_OVERFLOW,
            )

    @parametrize("op", ("ceildiv", "floordiv"))
    @parametrize(
        "operands",
        (
            (0, 1),
            (1, MAX_I64),
            (MAX_I64, 1),
            (MAX_I64, 2),
            (MAX_I64, MAX_I64),
            (0, 0),
            (1, 0),
            (-1, 1),
            (1, -1),
            (MIN_I64, -1),
        ),
    )
    def test_division_domain_and_boundaries(self, op, operands):
        expression = IntExpr(op, args=(IntExpr("boxed", 0), IntExpr("boxed", 1)))
        program = _numeric_program((7, 3))
        output = program.add(expression)
        compiled = compile_numeric(program)
        status, values = compiled.evaluate_leaves(compiled.bind_inputs(operands))
        left, right = operands
        if left < 0 or right <= 0:
            self.assertEqual(status, EarlyStatus.DIVISION_DOMAIN)
        else:
            self.assertEqual(status, EarlyStatus.SUCCESS)
            expected = left // right + (op == "ceildiv" and left % right != 0)
            self.assertEqual(values[output], expected)

    @parametrize("operands", ((MIN_I64, MAX_I64), (MAX_I64, MIN_I64), (0, 0), (-1, 1)))
    def test_comparisons_share_inputs(self, operands):
        program = _numeric_program((1, 2))
        roots = tuple(
            program.add(IntExpr(op, args=(IntExpr("boxed", 0), IntExpr("boxed", 1))))
            for op in ("eq", "ne", "lt", "le", "gt", "ge")
        )
        compiled = compile_numeric(program)
        status, values = compiled.evaluate_leaves(compiled.bind_inputs(operands))
        left, right = operands
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(
            tuple(values[index] for index in roots),
            (
                int(left == right),
                int(left != right),
                int(left < right),
                int(left <= right),
                int(left > right),
                int(left >= right),
            ),
        )
        self.assertEqual(compiled.leaf_bindings, (("boxed", 0), ("boxed", 1)))

    @parametrize("operands", ((0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (-1, 0), (1, 2)))
    def test_boolean_and_checks_both_operands(self, operands):
        program = _numeric_program((1, 1))
        output = program.add(
            IntExpr("and", args=(IntExpr("boxed", 0), IntExpr("boxed", 1)))
        )
        compiled = compile_numeric(program)
        status, values = compiled.evaluate_leaves(compiled.bind_inputs(operands))
        if all(value in (0, 1) for value in operands):
            self.assertEqual(
                (status, values[output]),
                (EarlyStatus.SUCCESS, operands[0] & operands[1]),
            )
        else:
            self.assertEqual(status, EarlyStatus.BOOLEAN_DOMAIN)

    @parametrize("condition", (0, 1, -1, 2))
    def test_select_requires_boolean_condition(self, condition):
        program = _numeric_program((1,))
        output = program.add(
            IntExpr(
                "select",
                args=(
                    IntExpr("boxed", 0),
                    IntExpr("constant", MIN_I64),
                    IntExpr("constant", MAX_I64),
                ),
            )
        )
        compiled = compile_numeric(program)
        status, values = compiled.evaluate_leaves(compiled.bind_inputs((condition,)))
        if condition in (0, 1):
            self.assertEqual(
                (status, values[output]),
                (EarlyStatus.SUCCESS, MIN_I64 if condition else MAX_I64),
            )
        else:
            self.assertEqual(status, EarlyStatus.BOOLEAN_DOMAIN)

    def test_select_keeps_unselected_branch_errors_in_instruction_order(self):
        program = _numeric_program((1, 2))
        division = IntExpr(
            "floordiv", args=(IntExpr("constant", 4), IntExpr("boxed", 1))
        )
        program.add(
            IntExpr(
                "select", args=(IntExpr("boxed", 0), IntExpr("constant", 7), division)
            )
        )
        compiled = compile_numeric(program)
        self.assertEqual(
            compiled.evaluate_leaves(compiled.bind_inputs((1, 0)))[0],
            EarlyStatus.DIVISION_DOMAIN,
        )
        self.assertEqual(
            compiled.evaluate_leaves(compiled.bind_inputs((2, 0)))[0],
            EarlyStatus.DIVISION_DOMAIN,
        )

    def test_metadata_bindings_and_shared_subexpressions(self):
        tensor = torch.arange(64).as_strided((3, 4), (8, 1), 5)
        program = _numeric_program((tensor, 2))
        size = IntExpr("size", 0, (IntExpr("constant", 0),))
        stride = IntExpr("stride", 0, (IntExpr("constant", 0),))
        shared = IntExpr("multiply", args=(size, stride))
        offset = IntExpr("add", args=(shared, IntExpr("storage_offset", 0)))
        output = program.add(IntExpr("add", args=(offset, IntExpr("boxed", 1))))
        repeated = program.add(IntExpr("multiply", args=(shared, shared)))
        compiled = compile_numeric(program)
        del program
        gc.collect()
        changed = torch.arange(128).as_strided((5, 3), (12, 2), 7)
        leaves = compiled.bind_inputs((changed, 9))
        self.assertEqual(leaves, (5, 12, 7, 9))
        status, values = compiled.evaluate_leaves(leaves)
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual((values[output], values[repeated]), (76, 3600))
        self.assertEqual(compiled.cpp_source.count("__builtin_mul_overflow"), 2)
        self.assertIsNotNone(compiled.library_owner)
        self.assertNotIn("switch", compiled.cpp_source)
        self.assertNotIn("for (", compiled.cpp_source)

    def test_compilation_does_not_read_preparation_values(self):
        program = _numeric_program((3,))
        output = program.add(
            IntExpr("add", args=(IntExpr("boxed", 0), IntExpr("constant", 2)))
        )
        program.values = None
        compiled = compile_numeric(program)
        status, values = compiled.evaluate_leaves(compiled.bind_inputs((100,)))
        self.assertEqual((status, values[output]), (EarlyStatus.SUCCESS, 102))

    @parametrize("value", (True, 1.0, MIN_I64 - 1, MAX_I64 + 1))
    def test_binding_rejects_non_int64_values(self, value):
        program = _numeric_program((1,))
        program.add(IntExpr("boxed", 0))
        compiled = compile_numeric(program)
        with self.assertRaisesRegex(ValueError, "exact int64"):
            compiled.bind_inputs((value,))
        with self.assertRaisesRegex(ValueError, "prepared int64 ABI"):
            compiled.evaluate_leaves((value,))

    @parametrize(
        "instructions",
        (
            [("boxed", 0), ("add", 0, 1)],
            [("boxed", 0), ("mystery", 0, 0)],
            [("boxed", 0), ("constant", MAX_I64 + 1)],
            [("size", 0, 0)],
            [("boxed", 0), ("boxed", 0)],
            [("constant", 0)],
        ),
    )
    def test_invalid_program_is_rejected_before_compilation(self, instructions):
        program = _numeric_program((1,))
        program.instructions = instructions
        with self.assertRaises(ValueError):
            compile_numeric(program)


@instantiate_parametrized_tests
class TestCompiledEvaluation(TestCase):
    def test_combined_library_counts_values_and_root_metadata(self):
        early = _numeric_program((torch.empty(3, 4), 2))
        size = IntExpr("size", 0, (IntExpr("constant", 0),))
        product = IntExpr("multiply", args=(size, IntExpr("boxed", 1)))
        owner = BufferSource("owned")
        late = _ParameterProgram(early, 2, {owner: 2})
        late.add(ParameterSource("value", 64, product))
        pointer = ParameterSource(
            "pointer", 64, PointerSource(owner, IntExpr("constant", 8))
        )
        late.add(pointer)
        late.add(ParameterSource("trunc", 32, args=(pointer,)))
        compiled = compile_evaluation(early, late, pointer_count=3)
        self.assertIs(compiled.library_owner, compiled.early.library_owner)
        self.assertIs(compiled.library_owner, compiled.late.library_owner)
        self.assertEqual(compiled.early_count, len(early.instructions))
        self.assertEqual(
            (compiled.leaf_count, compiled.late_count, compiled.pointer_count),
            (2, 3, 3),
        )
        self.assertEqual(compiled.late.output_roots, ((), (2,), (2,)))
        self.assertEqual(compiled.late.pointer_inputs, (2,))
        self.assertEqual(
            set(compiled.registration_kwargs),
            {
                "early_address",
                "late_address",
                "leaf_count",
                "early_count",
                "late_count",
                "pointer_count",
                "library_owner",
            },
        )
        self.assertEqual(
            compiled.registration_kwargs["early_address"],
            compiled.early.function_address,
        )
        self.assertEqual(
            compiled.registration_kwargs["late_address"], compiled.late.function_address
        )
        leaves = compiled.early.bind_inputs((torch.empty(7, 4), 3))
        status, values = compiled.early.evaluate_leaves(leaves)
        self.assertEqual(status, EarlyStatus.SUCCESS)
        roots = (4096, 0, (1 << 63) + 256)
        status, outputs = compiled.late.evaluate(values, roots)
        self.assertEqual(status, LateStatus.SUCCESS)
        self.assertEqual(outputs, (21, -(1 << 63) + 264, 264))
        self.assertEqual(
            outputs,
            torch._C._cuda_evaluate_parameter_program(late.plan(), values, roots),
        )
        self.assertEqual(
            compiled.cpp_source.count('extern "C" int32_t evaluate_early'), 1
        )
        self.assertEqual(
            compiled.cpp_source.count('extern "C" int32_t evaluate_late'), 1
        )
        self.assertNotIn("switch", compiled.cpp_source)
        self.assertNotIn("for (", compiled.cpp_source)

    def test_early_only_registration_has_no_late_entry(self):
        early = _numeric_program((3,))
        output = early.add(
            IntExpr("add", args=(IntExpr("boxed", 0), IntExpr("constant", 1)))
        )
        compiled = compile_evaluation(early, pointer_count=3)
        self.assertIsNone(compiled.late)
        self.assertEqual(
            (compiled.late_address, compiled.late_count, compiled.pointer_count),
            (0, 0, 3),
        )
        self.assertGreater(compiled.early_address, 0)
        self.assertEqual(compiled.early.evaluate_leaves((9,))[1][output], 10)

    def test_empty_programs_keep_valid_early_and_late_abis(self):
        early = _numeric_program(())
        late = _ParameterProgram(early, 0, {})
        compiled = compile_evaluation(early, late, pointer_count=0)
        self.assertEqual(
            (
                compiled.leaf_count,
                compiled.early_count,
                compiled.late_count,
                compiled.pointer_count,
            ),
            (0, 0, 0, 0),
        )
        self.assertGreater(compiled.early_address, 0)
        self.assertGreater(compiled.late_address, 0)
        self.assertEqual(compiled.early.evaluate_leaves(()), (EarlyStatus.SUCCESS, ()))
        self.assertEqual(compiled.late.evaluate((), ()), (LateStatus.SUCCESS, ()))

    @parametrize("divisor", (0, 2))
    def test_nonempty_zero_output_late_program_keeps_errors(self, divisor):
        early = _numeric_program((2,))
        value = early.add(IntExpr("boxed", 0))
        late = _ParameterProgram(early, 1, {})
        late.rows = [("constant", 64, 7), ("value", 64, value), ("udiv", 64, 0, 1, ())]
        compiled = compile_evaluation(early, late, pointer_count=1)
        self.assertEqual(compiled.late_count, 0)
        self.assertGreater(compiled.late_address, 0)
        status, values = compiled.early.evaluate_leaves((divisor,))
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(
            compiled.late.evaluate(values, (0,)),
            (LateStatus.DIVISION_DOMAIN if divisor == 0 else LateStatus.SUCCESS, ()),
        )

    @parametrize("count", (True, -1, 0))
    def test_invalid_pointer_counts_decline(self, count):
        early = _numeric_program((1,))
        early.add(IntExpr("boxed", 0))
        with self.assertRaisesRegex(ValueError, "Pointer count"):
            compile_evaluation(early, pointer_count=count)

    def test_late_program_requires_shared_dag_and_root_count(self):
        early = _numeric_program((1,))
        early.add(IntExpr("boxed", 0))
        other = _numeric_program((1,))
        with self.assertRaisesRegex(ValueError, "same numeric DAG"):
            compile_evaluation(early, _ParameterProgram(other, 1, {}), pointer_count=1)
        with self.assertRaisesRegex(ValueError, "counts differ"):
            compile_evaluation(early, _ParameterProgram(early, 1, {}), pointer_count=2)


if __name__ == "__main__":
    run_tests()
