# Owner(s): ["module: inductor"]
"""Fixed-width compiled parameters agree with the native evaluator."""

import gc
import random
from types import SimpleNamespace

import torch
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource,
    InputSource,
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
    compile_parameter,
    LateStatus,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _make_program(rows, outputs, early_count=2, root_count=0):
    records = SimpleNamespace(
        input_names=tuple(f"arg{index}" for index in range(early_count)),
        integer_inputs=tuple(
            IntegerInput(f"arg{index}", index) for index in range(early_count)
        ),
    )
    numeric = _NumericProgram(records, (0,) * early_count)
    for index in range(early_count):
        numeric.add(IntExpr("boxed", index))
    program = _ParameterProgram(numeric, root_count, {})
    program.rows = list(rows)
    program.outputs = list(outputs)
    return program


def _binary_program(op, width, flags=()):
    rows = [("value", 64, 0), ("value", 64, 1)]
    first, second = 0, 1
    if width != 64:
        rows.extend((("trunc", width, 0), ("trunc", width, 1)))
        first, second = 2, 3
    row = (op, width, first, second)
    if op in ("add", "sub", "mul", "shl", "udiv", "sdiv", "lshr", "ashr"):
        row += (flags,)
    rows.append(row)
    return _make_program(rows, (len(rows) - 1,))


def _sample_pairs(width):
    boundary = [
        -(1 << 63),
        -(1 << 31),
        -65,
        -33,
        -2,
        -1,
        0,
        1,
        2,
        3,
        31,
        32,
        63,
        64,
        (1 << 31) - 1,
        (1 << 32) - 1,
        (1 << 63) - 1,
    ]
    pairs = [(a, b) for a in boundary for b in (-1, 0, 1, 2, width - 1, width)]
    generator = random.Random(1729 + width)
    pairs.extend(
        (
            generator.randrange(-(1 << 63), 1 << 63),
            generator.randrange(-(1 << 63), 1 << 63),
        )
        for _ in range(24)
    )
    return pairs


@instantiate_parametrized_tests
class TestCompiledParameter(TestCase):
    def compile(self, program):
        compiled = compile_parameter(program)
        source = compiled.cpp_source
        self.assertNotIn("switch", source)
        self.assertNotIn("for (", source)
        self.assertNotIn("while (", source)
        return compiled

    def compare(self, program, compiled, early, roots=()):
        status, values = compiled.evaluate(early, roots)
        try:
            expected = torch._C._cuda_evaluate_parameter_program(
                program.plan(), tuple(early), tuple(roots)
            )
        except (ValueError, RuntimeError) as error:
            message = str(error)
            if "Early value cannot" in message:
                expected_status = LateStatus.EARLY_VALUE_RANGE
            elif (
                "Pointer addition overflow" in message
                or "Pointer subtraction underflow" in message
            ):
                expected_status = LateStatus.POINTER_RANGE
            elif "division by zero" in message or "Signed division overflow" in message:
                expected_status = LateStatus.DIVISION_DOMAIN
            elif "Live poison" in message:
                expected_status = LateStatus.LIVE_POISON
            else:
                self.fail(f"Unexpected native error: {message}")
            self.assertEqual(
                status, expected_status, (program.plan(), early, roots, message)
            )
        else:
            self.assertEqual(status, LateStatus.SUCCESS, (program.plan(), early, roots))
            self.assertEqual(values, expected, (program.plan(), early, roots))
        return status, values

    @parametrize(
        "op",
        (
            "add",
            "sub",
            "mul",
            "udiv",
            "sdiv",
            "urem",
            "srem",
            "and",
            "or",
            "xor",
            "shl",
            "lshr",
            "ashr",
        ),
    )
    @parametrize("width", (32, 64))
    def test_binary_operations(self, op, width):
        program = _binary_program(op, width)
        compiled = self.compile(program)
        for values in _sample_pairs(width):
            self.compare(program, compiled, values)

    @parametrize("op", ("add", "sub", "mul", "shl"))
    @parametrize("flags", (("nuw",), ("nsw",), ("nuw", "nsw")))
    @parametrize("width", (32, 64))
    def test_overflow_flags(self, op, flags, width):
        program = _binary_program(op, width, flags)
        compiled = self.compile(program)
        for values in _sample_pairs(width):
            self.compare(program, compiled, values)

    @parametrize("op", ("udiv", "sdiv", "lshr", "ashr"))
    @parametrize("width", (32, 64))
    def test_exact_flags(self, op, width):
        program = _binary_program(op, width, ("exact",))
        compiled = self.compile(program)
        for values in _sample_pairs(width):
            self.compare(program, compiled, values)

    @parametrize(
        "predicate",
        ("eq", "ne", "ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge"),
    )
    @parametrize("width", (1, 32, 64))
    def test_comparisons(self, predicate, width):
        rows = [("value", 64, 0), ("value", 64, 1)]
        first, second = 0, 1
        if width != 64:
            rows.extend((("trunc", width, 0), ("trunc", width, 1)))
            first, second = 2, 3
        comparison = len(rows)
        rows.extend((("icmp", predicate, first, second), ("zext", 32, comparison)))
        program = _make_program(rows, (len(rows) - 1,))
        compiled = self.compile(program)
        for values in _sample_pairs(width):
            self.compare(program, compiled, values)

    @parametrize("width", (1, 32))
    @parametrize("value", (-(1 << 63), -1, 0, 1, (1 << 31) - 1, 1 << 31, (1 << 63) - 1))
    def test_casts_and_signed_transport(self, width, value):
        program = _make_program(
            (("value", 64, 0), ("trunc", width, 0), ("sext", 64, 1), ("zext", 64, 1)),
            (2, 3),
            early_count=1,
        )
        compiled = self.compile(program)
        self.compare(program, compiled, (value,))
        self.assertEqual(
            compiled.evaluate((-1,), ()), (LateStatus.SUCCESS, (-1, (1 << width) - 1))
        )

    @parametrize(
        "value", (-(1 << 63), -(1 << 31) - 1, -(1 << 31), -1, 0, (1 << 32) - 1, 1 << 32)
    )
    def test_early_i32_domain_without_implicit_truncation(self, value):
        program = _make_program((("value", 32, 0),), (0,), early_count=1)
        compiled = self.compile(program)
        self.compare(program, compiled, (value,))

    @parametrize(
        "root,offset,status",
        (
            (0, 0, LateStatus.SUCCESS),
            (0, -1, LateStatus.POINTER_RANGE),
            ((1 << 64) - 1, 1, LateStatus.POINTER_RANGE),
            ((1 << 64) - 1, 0, LateStatus.SUCCESS),
            (1 << 63, -(1 << 63), LateStatus.SUCCESS),
            (0, -(1 << 63), LateStatus.POINTER_RANGE),
            ((1 << 64) - 1, -(1 << 63), LateStatus.SUCCESS),
        ),
    )
    def test_pointer_boundaries(self, root, offset, status):
        program = _make_program(
            (("pointer", 64, 0, 0),), (0,), early_count=1, root_count=1
        )
        compiled = self.compile(program)
        actual, _ = self.compare(program, compiled, (offset,), (root,))
        self.assertEqual(actual, status)

    @parametrize("condition", (0, 1))
    @parametrize("poison_kind", ("nuw", "nsw", "shift", "exact"))
    def test_select_masks_only_unselected_poison(self, condition, poison_kind):
        rows = [
            ("constant", 1, condition),
            ("constant", 32, (1 << 32) - 1),
            ("constant", 32, 1),
            ("constant", 32, 32),
            ("constant", 32, (1 << 31) - 1),
            ("constant", 32, 3),
            ("constant", 32, 2),
        ]
        rows.append(
            {
                "nuw": ("add", 32, 1, 2, ("nuw",)),
                "nsw": ("add", 32, 4, 2, ("nsw",)),
                "shift": ("shl", 32, 2, 3, ()),
                "exact": ("udiv", 32, 5, 6, ("exact",)),
            }[poison_kind]
        )
        rows.append(("select", 32, 0, 7, 2))
        program = _make_program(rows, (8,), early_count=0)
        compiled = self.compile(program)
        status, result = self.compare(program, compiled, ())
        self.assertEqual(
            status, LateStatus.LIVE_POISON if condition else LateStatus.SUCCESS
        )
        if not condition:
            self.assertEqual(result, (1,))

    @parametrize("fault", ("zero_division", "signed_division_overflow"))
    def test_unselected_undefined_operation_still_fails(self, fault):
        rows = [
            ("constant", 1, 0),
            ("constant", 64, 1),
            ("constant", 64, 0),
            ("constant", 64, 1 << 63),
            ("constant", 64, (1 << 64) - 1),
        ]
        rows.append(
            ("udiv", 64, 1, 2, ())
            if fault == "zero_division"
            else ("sdiv", 64, 3, 4, ())
        )
        rows.append(("select", 64, 0, 5, 1))
        program = _make_program(rows, (6,), early_count=0)
        compiled = self.compile(program)
        self.assertEqual(
            self.compare(program, compiled, ())[0], LateStatus.DIVISION_DOMAIN
        )

    @parametrize("output", (4, 6))
    def test_poison_condition_and_poison_operand_division(self, output):
        rows = (
            ("constant", 32, (1 << 32) - 1),
            ("constant", 32, 1),
            ("add", 32, 0, 1, ("nuw",)),
            ("constant", 32, 0),
            ("udiv", 32, 2, 3, ()),
            ("icmp", "eq", 2, 1),
            ("select", 32, 5, 1, 3),
        )
        program = _make_program(rows, (output,), early_count=0)
        compiled = self.compile(program)
        self.assertEqual(self.compare(program, compiled, ())[0], LateStatus.LIVE_POISON)

    def test_original_program_and_conservative_root_metadata(self):
        records = SimpleNamespace(input_names=("a",), integer_inputs=())
        numeric = _NumericProgram(records, (torch.empty(1),))
        owner = BufferSource("owned")
        program = _ParameterProgram(numeric, 1, {owner: 1})
        first = ParameterSource(
            "pointer", 64, PointerSource(InputSource(0), IntExpr("constant", -4))
        )
        second = ParameterSource(
            "pointer", 64, PointerSource(owner, IntExpr("constant", 8))
        )
        condition = ParameterSource("constant", 1, 0)
        selected = ParameterSource("select", 64, args=(condition, first, second))
        program.add(selected)
        program.add(ParameterSource("trunc", 32, args=(selected,)))
        compiled = self.compile(program)
        self.assertEqual(compiled.output_roots, ((0, 1), (0, 1)))
        self.assertEqual(compiled.pointer_inputs, (0, 1))
        self.assertEqual(compiled.output_widths, (64, 32))
        self.compare(program, compiled, tuple(numeric.values), (4096, (1 << 63) + 256))
        numeric.values = object()
        rebuilt = self.compile(program)
        self.assertEqual(rebuilt.cpp_source, compiled.cpp_source)
        gc.collect()
        self.assertEqual(
            compiled.evaluate((-4, 8), (4096, 8192)), (LateStatus.SUCCESS, (8200, 8200))
        )

    @parametrize(
        "case",
        (
            "width",
            "operand",
            "flags",
            "repeat_flag",
            "constant",
            "cast",
            "comparison",
            "output",
            "early",
            "pointer",
        ),
    )
    def test_malformed_program_declines(self, case):
        rows = [("constant", 32, 1)]
        outputs = (0,)
        if case == "width":
            rows = [("constant", 16, 1)]
        elif case == "operand":
            rows.append(("add", 32, 0, 1, ()))
        elif case == "flags":
            rows.append(("add", 32, 0, 0, ("exact",)))
        elif case == "repeat_flag":
            rows.append(("add", 32, 0, 0, ("nuw", "nuw")))
        elif case == "constant":
            rows = [("constant", 32, 1 << 32)]
        elif case == "cast":
            rows.append(("zext", 32, 0))
        elif case == "comparison":
            rows.append(("icmp", "bad", 0, 0))
        elif case == "output":
            outputs = (1,)
        elif case == "early":
            rows = [("value", 32, 2)]
        elif case == "pointer":
            rows = [("pointer", 64, 0, 0)]
        with self.assertRaises((ValueError, TypeError)):
            compile_parameter(_make_program(rows, outputs))


@instantiate_parametrized_tests
class TestCompiledParameterZeroOutput(TestCase):
    @parametrize("divisor", (0, 2))
    def test_nonempty_program_preserves_eager_errors(self, divisor):
        program = _make_program(
            (("constant", 64, 7), ("value", 64, 0), ("udiv", 64, 0, 1, ())),
            (),
            early_count=1,
        )
        compiled = compile_parameter(program)
        self.assertEqual(compiled.output_widths, ())
        self.assertEqual(len(program.rows), 3)
        status, outputs = compiled.evaluate((divisor,), ())
        self.assertEqual(outputs, ())
        if divisor == 0:
            with self.assertRaisesRegex(RuntimeError, "division by zero"):
                torch._C._cuda_evaluate_parameter_program(
                    program.plan(), (divisor,), ()
                )
            self.assertEqual(status, LateStatus.DIVISION_DOMAIN)
        else:
            self.assertEqual(
                torch._C._cuda_evaluate_parameter_program(
                    program.plan(), (divisor,), ()
                ),
                (),
            )
            self.assertEqual(status, LateStatus.SUCCESS)

    def test_empty_program_is_noop(self):
        program = _make_program((), (), early_count=0)
        compiled = compile_parameter(program)
        self.assertEqual(program.rows, [])
        self.assertEqual(compiled.output_widths, ())
        self.assertEqual(
            torch._C._cuda_evaluate_parameter_program(program.plan(), (), ()), ()
        )
        self.assertEqual(compiled.evaluate((), ()), (LateStatus.SUCCESS, ()))


if __name__ == "__main__":
    run_tests()
