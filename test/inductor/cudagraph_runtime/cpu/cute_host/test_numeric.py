"""SSA numeric lowering checked against the owned compiler evaluator."""

import json
from pathlib import Path
from types import SimpleNamespace

from torch._inductor.runtime._cudagraph._sdk import activate

activate()

ROOT = Path(__file__).resolve().parent / "fixtures"

from unittest import mock

from torch._inductor.runtime._cudagraph._compiler.cfg_values import read_cfg_function
from torch._inductor.runtime._cudagraph._compiler.cute_bridge.numeric import Comparison, lower_numeric, NumericDeclined, NumericSource
from cutlass._mlir import ir
from torch._inductor.runtime._cudagraph._compiler.decoded_values import prepare_decodings
from torch._inductor.runtime._cudagraph._compiler.overflow_properties import bind_properties
from torch._inductor.runtime._cudagraph._compiler.owned_numeric import evaluate_owned, freeze_numeric
from torch._inductor.runtime.cudagraph_arg_mapping import IntegerInput, IntExpr
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch._inductor.runtime._cudagraph._compiler.metadata_regions import MetadataAggregate, UnavailablePointer
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase
from torch._inductor.runtime._cudagraph._compiler.values import scalar


@instantiate_parametrized_tests
class TestCuTeNumericLowering(TestCase):
    def freeze(self, source, source_order):
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = ir.Module.parse(source)
            self.assertTrue(module.operation.verify())
            cfg = read_cfg_function(module, "probe")
            return freeze_numeric(bind_properties(cfg), prepare_decodings(cfg), source_order)

    @parametrize("kind", ("positive", "negative", "boolean"))
    def test_literal_keeps_exact_type_and_bits(self, kind):
        typ, literal, expected = {
            "positive": ("i32", "7 : i32", 7),
            "negative": ("i64", "-9 : i64", -9),
            "boolean": ("i1", "true", 1),
        }[kind]
        numeric = self.freeze(f'''module {{ llvm.func @probe(%unused: !llvm.ptr) -> {typ} {{
          %value = llvm.mlir.constant({literal}) : {typ}
          llvm.return %value : {typ}
        }} }}''', (19,))
        resolve = mock.Mock(side_effect=AssertionError("Unused source was requested"))
        lowered = lower_numeric(numeric, resolve)
        self.assertEqual(lowered.values[0].llvm_type, typ)
        self.assertEqual(lowered.values[0].expression, IntExpr("constant", expected))
        self.assertEqual(lowered.obligations, ())
        resolve.assert_not_called()

    def test_original_order_and_nested_leaf(self):
        typ = "!llvm.struct<(ptr<1>, struct<(i32, i64)>)>"
        numeric = self.freeze(f'''module {{ llvm.func @probe(%unused: i64, %arg: {typ}) -> i32 {{
          %extent = llvm.extractvalue %arg[1, 0] : {typ}
          llvm.return %extent : i32
        }} }}''', (11, 3))
        resolve = mock.Mock(return_value=NumericSource(IntExpr("boxed", 7), 2, 513))
        lowered = lower_numeric(numeric, resolve)
        resolve.assert_called_once_with(3, (1, 0))
        self.assertEqual(lowered.values[0].expression, IntExpr("boxed", 7))
        self.assertEqual((lowered.values[0].lower, lowered.values[0].upper), (2, 513))
        self.assertEqual(lowered.obligations, ())

    def test_unbounded_source_returns_explicit_width_obligation(self):
        numeric = self.freeze('''module { llvm.func @probe(%n: i32) -> i32 {
          llvm.return %n : i32
        } }''', (5,))
        resolve = mock.Mock(return_value=NumericSource(IntExpr("boxed", 2), 2, None))
        with mock.patch.object(ir.Module, "parse", side_effect=AssertionError("Lowering reparsed IR")):
            lowered = lower_numeric(numeric, resolve)
        self.assertEqual(len(lowered.obligations), 1)
        obligation, = lowered.obligations
        self.assertEqual((obligation.expression, obligation.lower, obligation.upper),
                         (IntExpr("boxed", 2), -(1 << 31), (1 << 31) - 1))
        self.assertEqual((lowered.values[0].lower, lowered.values[0].upper), (2, (1 << 31) - 1))
        resolve.assert_called_once_with(5, ())

    def test_equal_ranges_keep_distinct_original_sources(self):
        numeric = self.freeze('''module { llvm.func @probe(%a: i32, %b: i32) -> i1 {
          %same = llvm.icmp "eq" %a, %b : i32
          llvm.return %same : i1
        } }''', (9, 2))
        sources = {9: NumericSource(IntExpr("boxed", 4), 2, 513),
                   2: NumericSource(IntExpr("boxed", 0), 2, 513)}
        resolve = mock.Mock(side_effect=lambda index, path: sources[index])
        lowered = lower_numeric(numeric, resolve)
        self.assertEqual([(call.args, call.kwargs) for call in resolve.call_args_list],
                         [((9, ()), {}), ((2, ()), {})])
        self.assertEqual(lowered.values[0].expression,
                         Comparison("eq", "i32", IntExpr("boxed", 4), IntExpr("boxed", 0)))

    @parametrize("cast", ("sext", "zext"))
    def test_safe_extension_preserves_source(self, cast):
        numeric = self.freeze(f'''module {{ llvm.func @probe(%n: i32) -> i64 {{
          %wide = llvm.{cast} %n : i32 to i64
          llvm.return %wide : i64
        }} }}''', (5,))
        lower = -17 if cast == "sext" else 0
        lowered = lower_numeric(numeric, lambda index, path: NumericSource(IntExpr("boxed", 2), lower, 513))
        self.assertEqual(lowered.values[0].expression, IntExpr("boxed", 2))
        self.assertEqual(lowered.values[0].llvm_type, "i64")
        self.assertEqual((lowered.values[0].lower, lowered.values[0].upper), (lower, 513))
        self.assertEqual(lowered.obligations, ())

    @parametrize("literal", (False, True))
    def test_zext_does_not_erase_unsigned_bit_semantics(self, literal):
        numeric = self.freeze('''module { llvm.func @probe(%n: i32) -> i64 {
          %wide = llvm.zext %n : i32 to i64
          llvm.return %wide : i64
        } }''', (0,))
        expression = IntExpr("constant", -1) if literal else IntExpr("boxed", 0)

        def resolve(index, path):
            return NumericSource(expression, -1, -1)

        if literal:
            lowered = lower_numeric(numeric, resolve)
            self.assertEqual(lowered.values[0].expression, IntExpr("constant", (1 << 32) - 1))
            self.assertEqual(evaluate_owned(numeric, (scalar("i32", -1),))[0].integer(), (1 << 32) - 1)
        else:
            with self.assertRaisesRegex(NumericDeclined, "dynamic bit reinterpretation"):
                lower_numeric(numeric, resolve)

    @parametrize("predicate", ("slt", "uge"))
    def test_comparison_keeps_decoded_predicate(self, predicate):
        numeric = self.freeze(f'''module {{ llvm.func @probe(%n: i32) -> i1 {{
          %four = llvm.mlir.constant(4 : i32) : i32
          %selected = llvm.icmp "{predicate}" %n, %four : i32
          llvm.return %selected : i1
        }} }}''', (2,))
        lowered = lower_numeric(numeric, lambda index, path: NumericSource(IntExpr("boxed", 5), 2, 17))
        self.assertEqual(lowered.values[0].expression,
                         Comparison(predicate, "i32", IntExpr("boxed", 5), IntExpr("constant", 4)))
        for length in (2, 4, 17):
            expected = length < 4 if predicate == "slt" else length >= 4
            self.assertEqual(evaluate_owned(numeric, (scalar("i32", length),))[0].integer(signed=False), int(expected))

    @parametrize("flag", ("none", "nsw", "nuw", "both"))
    def test_proven_multiplication_uses_existing_tape(self, flag):
        suffix = "" if flag == "none" else " overflow<" + ("nsw, nuw" if flag == "both" else flag) + ">"
        numeric = self.freeze(f'''module {{ llvm.func @probe(%n: i32) -> i32 {{
          %width = llvm.mlir.constant(128 : i32) : i32
          %size = llvm.mul %n, %width{suffix} : i32
          llvm.return %size : i32
        }} }}''', (3,))
        lowered = lower_numeric(numeric, lambda index, path: NumericSource(IntExpr("boxed", 0), 2, 513))
        self.assertEqual(lowered.values[0].expression,
                         IntExpr("multiply", args=(IntExpr("boxed", 0), IntExpr("constant", 128))))
        self.assertEqual((lowered.values[0].lower, lowered.values[0].upper), (256, 65664))
        self.assertEqual(lowered.obligations, ())
        self.assertEqual(evaluate_owned(numeric, (scalar("i32", 513),))[0].integer(), 65664)

    @parametrize("constant", (0, 7))
    def test_addition_uses_numeric_tape(self, constant):
        numeric = self.freeze(f'''module {{ llvm.func @probe(%n: i32) -> i32 {{
          %c = llvm.mlir.constant({constant} : i32) : i32
          %sum = llvm.add %n, %c : i32
          llvm.return %sum : i32
        }} }}''', (3,))

        def resolve(index, path):
            return NumericSource(IntExpr("boxed", 0), 2, 513)

        if constant == 0:
            self.assertEqual(lower_numeric(numeric, resolve).values[0].expression, IntExpr("boxed", 0))
        else:
            self.assertEqual(lower_numeric(numeric, resolve).values[0].expression,
                             IntExpr("add", args=(IntExpr("boxed", 0), IntExpr("constant", constant))))

    @parametrize("fault", ("signed_overflow", "nuw"))
    def test_arithmetic_properties_are_not_assumed_true(self, fault):
        suffix = " overflow<nuw>" if fault == "nuw" else ""
        numeric = self.freeze(f'''module {{ llvm.func @probe(%n: i8) -> i8 {{
          %two = llvm.mlir.constant(2 : i8) : i8
          %result = llvm.mul %n, %two{suffix} : i8
          llvm.return %result : i8
        }} }}''', (0,))
        lower, upper, message = (-1, -1, "nuw") if fault == "nuw" else (1, 127, "signed overflow or wrapping")
        with self.assertRaisesRegex(NumericDeclined, message):
            lower_numeric(numeric, lambda index, path: NumericSource(IntExpr("boxed", 0), lower, upper))

    @parametrize("fault", ("trunc", "division", "control_flow"))
    def test_unsupported_semantics_decline(self, fault):
        bodies = {
            "trunc": "%value = llvm.trunc %n : i64 to i32\nllvm.return %value : i32",
            "division": "%two = llvm.mlir.constant(2 : i64) : i64\n"
                        "%value = llvm.sdiv %n, %two : i64\nllvm.return %value : i64",
            "control_flow": "llvm.br ^next(%n : i64)\n^next(%value: i64):\nllvm.return %value : i64",
        }
        result = "i32" if fault == "trunc" else "i64"
        numeric = self.freeze(f"module {{ llvm.func @probe(%n: i64) -> {result} {{ {bodies[fault]} }} }}", (0,))
        message = {"control_flow": "one return block", "division": "nonnegative dividend",
                   "trunc": "Unsupported numeric instruction"}[fault]
        with self.assertRaisesRegex(NumericDeclined, message):
            lower_numeric(numeric, lambda index, path: NumericSource(IntExpr("boxed", 0), -17, 513))

    @parametrize("predicate", ("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"))
    def test_consumed_comparison_uses_numeric_boolean(self, predicate):
        numeric = self.freeze(f'''module {{ llvm.func @probe(%n: i32) -> i32 {{
          %four = llvm.mlir.constant(4 : i32) : i32
          %same = llvm.icmp "{predicate}" %n, %four : i32
          %value = llvm.zext %same : i1 to i32
          llvm.return %value : i32
        }} }}''', (0,))
        lowered = lower_numeric(numeric, lambda index, path: NumericSource(IntExpr("boxed", 0), 2, 9))
        expected_tag = predicate if predicate in ("eq", "ne") else predicate[1:]
        self.assertEqual(lowered.values[0].expression,
                         IntExpr(expected_tag, args=(IntExpr("boxed", 0), IntExpr("constant", 4))))
        records = SimpleNamespace(input_names=("n",), integer_inputs=(IntegerInput("n", 0),))
        for value in (2, 4, 9):
            tape = _NumericProgram(records, (value,))
            actual = tape.values[tape.add(lowered.values[0].expression)]
            self.assertEqual(actual, evaluate_owned(numeric, (scalar("i32", value),))[0].integer())

    def test_division_addition_have_independent_ssa_consumers(self):
        numeric = self.freeze('''module { llvm.func @probe(%n: i32) -> i32 {
          %seven = llvm.mlir.constant(7 : i32) : i32
          %three = llvm.mlir.constant(3 : i32) : i32
          %sum = llvm.add %n, %seven overflow<nsw> : i32
          %q = llvm.sdiv %sum, %three : i32
          %result = llvm.add %q, %sum : i32
          llvm.return %result : i32
        } }''', (0,))
        lowered = lower_numeric(numeric, lambda index, path: NumericSource(IntExpr("boxed", 0), 2, 17))
        total = IntExpr("add", args=(IntExpr("boxed", 0), IntExpr("constant", 7)))
        self.assertEqual(lowered.values[0].expression,
            IntExpr("add", args=(IntExpr("floordiv", args=(total, IntExpr("constant", 3))), total)))
        records = SimpleNamespace(input_names=("n",), integer_inputs=(IntegerInput("n", 0),))
        for value in (2, 3, 7, 17):
            tape = _NumericProgram(records, (value,))
            self.assertEqual(tape.values[tape.add(lowered.values[0].expression)],
                             evaluate_owned(numeric, (scalar("i32", value),))[0].integer())

    def test_add_nsw_preserves_leaf_range_obligation(self):
        numeric = self.freeze('''module { llvm.func @probe(%n: i8) -> i8 {
          %seven = llvm.mlir.constant(7 : i8) : i8
          %result = llvm.add %n, %seven overflow<nsw> : i8
          llvm.return %result : i8
        } }''', (0,))
        lowered = lower_numeric(numeric, lambda index, path: NumericSource(IntExpr("boxed", 0), 0, 127))
        obligation, = lowered.obligations
        self.assertEqual((obligation.expression, obligation.lower, obligation.upper), (IntExpr("boxed", 0), 0, 120))
        self.assertEqual((lowered.values[0].lower, lowered.values[0].upper), (7, 127))
        self.assertEqual(evaluate_owned(numeric, (scalar("i8", 120),))[0].integer(), 127)
        with self.assertRaises(ValueError):
            evaluate_owned(numeric, (scalar("i8", 121),))

    def test_select_does_not_hide_invalid_ssa_arm(self):
        numeric = self.freeze('''module { llvm.func @probe(%n: i32) -> i32 {
          %false = llvm.mlir.constant(false) : i1
          %zero = llvm.mlir.constant(0 : i32) : i32
          %invalid = llvm.sdiv %n, %zero : i32
          %result = llvm.select %false, %invalid, %n : i1, i32
          llvm.return %result : i32
        } }''', (0,))
        with self.assertRaisesRegex(NumericDeclined, "positive constant divisor"):
            lower_numeric(numeric, lambda index, path: NumericSource(IntExpr("boxed", 0), 2, 17))

    def test_actual_cute_grid_ssa_matches_owned_evaluator(self):
        typ = "!llvm.struct<(ptr<1>, struct<(struct<(i32, i32)>, i64)>)>"
        numeric = self.freeze(f'''module {{ llvm.func @probe(%a: {typ}, %b: {typ}, %c: {typ},
            %d: {typ}, %bias: i32, %stream: !llvm.ptr) -> i32 {{
          %false = llvm.mlir.constant(false) : i1
          %one = llvm.mlir.constant(1 : i32) : i32
          %zero = llvm.mlir.constant(0 : i32) : i32
          %two = llvm.mlir.constant(2 : i32) : i32
          %x = llvm.extractvalue %a[1, 0, 0] : {typ}
          %q = llvm.sdiv %x, %two : i32
          %product = llvm.mul %q, %two : i32
          %different = llvm.icmp "ne" %x, %product : i32
          %negative = llvm.icmp "slt" %x, %zero : i32
          %positive = llvm.icmp "eq" %negative, %false : i1
          %condition = llvm.and %different, %positive : i1
          %increment = llvm.add %q, %one : i32
          %result = llvm.select %condition, %increment, %q {{fastmathFlags = #llvm.fastmath<none>}} : i1, i32
          llvm.return %result : i32
        }} }}''', tuple(range(6)))
        observed = json.loads((ROOT / "diagnostic_numeric.json").read_text())
        self.assertEqual(json.loads(json.dumps(numeric._cfg, default=str)), observed["cfg"])
        self.assertEqual([list(item) for item in numeric._flags], observed["flags"])
        resolve = mock.Mock(return_value=NumericSource(IntExpr("boxed", 1), 2, 127))
        lowered = lower_numeric(numeric, resolve)
        resolve.assert_called_once_with(0, (1, 0, 0))
        self.assertEqual(lowered.obligations, ())
        expression = lowered.values[0].expression
        self.assertEqual(expression.op, "select")
        self.assertEqual(expression.args[0].op, "and")
        self.assertEqual(expression.args[2].op, "floordiv")
        records = SimpleNamespace(input_names=("unused", "m"), integer_inputs=(IntegerInput("m", 1),))
        for value in (2, 3, 4, 7, 35, 126, 127):
            aggregate = MetadataAggregate(typ, (((0,), UnavailablePointer("!llvm.ptr<1>", 8)),
                ((1, 0, 0), scalar("i32", value)), ((1, 0, 1), scalar("i32", 5)),
                ((1, 1), scalar("i64", 128 * value))))
            arguments = (aggregate, aggregate, aggregate, aggregate, scalar("i32", 0), UnavailablePointer("!llvm.ptr", 8))
            tape = _NumericProgram(records, (2, value))
            actual = tape.values[tape.add(expression)]
            self.assertEqual(actual, evaluate_owned(numeric, arguments)[0].integer())
            self.assertEqual(actual, (value + 1) // 2)

    @parametrize("width, flag", ((8, "none"), (8, "both"), (32, "nsw"), (64, "nuw")))
    def test_subtraction_preserves_width_order_and_flags(self, width, flag):
        typ = f"i{width}"
        maximum = (1 << (width - 1)) - 1
        suffix = (
            ""
            if flag == "none"
            else " overflow<" + ("nsw, nuw" if flag == "both" else flag) + ">"
        )
        numeric = self.freeze(
            f"""module {{ llvm.func @probe(%a: {typ}, %b: {typ}) -> {typ} {{
          %result = llvm.sub %a, %b{suffix} : {typ}
          llvm.return %result : {typ}
        }} }}""",
            (0, 1),
        )
        sources = (
            NumericSource(IntExpr("boxed", 0), 4, maximum),
            NumericSource(IntExpr("boxed", 1), 0, 3),
        )
        lowered = lower_numeric(numeric, lambda index, path: sources[index])
        (result,) = lowered.values
        self.assertEqual(result.llvm_type, typ)
        self.assertEqual((result.lower, result.upper), (1, maximum))
        self.assertEqual(lowered.obligations, ())
        records = SimpleNamespace(
            input_names=("a", "b"),
            integer_inputs=(IntegerInput("a", 0), IntegerInput("b", 1)),
        )
        for left, right in ((4, 0), (4, 3), (maximum, 3), (maximum, 0)):
            tape = _NumericProgram(records, (left, right))
            actual = tape.values[tape.add(result.expression)]
            expected = evaluate_owned(numeric, (scalar(typ, left), scalar(typ, right)))[
                0
            ].integer()
            self.assertEqual(actual, expected)
            self.assertEqual(actual, left - right)

    @parametrize("width", (8, 64))
    def test_subtraction_keeps_negative_results(self, width):
        typ = f"i{width}"
        maximum = (1 << (width - 1)) - 1
        numeric = self.freeze(
            f"""module {{ llvm.func @probe(%a: {typ}, %b: {typ}) -> {typ} {{
          %result = llvm.sub %a, %b overflow<nsw> : {typ}
          llvm.return %result : {typ}
        }} }}""",
            (0, 1),
        )
        sources = (
            NumericSource(IntExpr("boxed", 0), 0, 3),
            NumericSource(IntExpr("boxed", 1), 4, maximum),
        )
        lowered = lower_numeric(numeric, lambda index, path: sources[index])
        (result,) = lowered.values
        self.assertEqual((result.lower, result.upper), (-maximum, -1))
        self.assertEqual(lowered.obligations, ())
        records = SimpleNamespace(
            input_names=("a", "b"),
            integer_inputs=(IntegerInput("a", 0), IntegerInput("b", 1)),
        )
        for left, right in ((0, maximum), (3, 4), (0, 4)):
            tape = _NumericProgram(records, (left, right))
            actual = tape.values[tape.add(result.expression)]
            self.assertEqual(
                actual,
                evaluate_owned(numeric, (scalar(typ, left), scalar(typ, right)))[
                    0
                ].integer(),
            )
            self.assertEqual(actual, left - right)

    @parametrize(
        "width, left, right, flag",
        (
            (8, -1, -2, "nuw"),
            (64, -(1 << 63), -(1 << 63), "none"),
            (8, -2, 0, "none"),
        ),
    )
    def test_subtraction_folds_exact_signed_literals(self, width, left, right, flag):
        typ = f"i{width}"
        suffix = "" if flag == "none" else f" overflow<{flag}>"
        numeric = self.freeze(
            f"""module {{ llvm.func @probe(%unused: i32) -> {typ} {{
          %a = llvm.mlir.constant({left} : {typ}) : {typ}
          %b = llvm.mlir.constant({right} : {typ}) : {typ}
          %result = llvm.sub %a, %b{suffix} : {typ}
          llvm.return %result : {typ}
        }} }}""",
            (0,),
        )
        resolve = mock.Mock(side_effect=AssertionError("Unused source was requested"))
        lowered = lower_numeric(numeric, resolve)
        (result,) = lowered.values
        self.assertEqual(result.expression, IntExpr("constant", left - right))
        self.assertEqual(
            (result.llvm_type, result.lower, result.upper),
            (typ, left - right, left - right),
        )
        self.assertEqual(lowered.obligations, ())
        self.assertEqual(
            evaluate_owned(numeric, (scalar("i32", 0),))[0].integer(), left - right
        )
        resolve.assert_not_called()

    @parametrize(
        "left, right, flag, wrapped",
        (
            (127, -1, "none", -128),
            (-128, 1, "none", 127),
            (127, -1, "nsw", None),
            (0, 1, "nuw", None),
            (-2, -1, "nuw", None),
        ),
    )
    def test_subtraction_rejects_wrap_and_unsigned_borrow(
        self, left, right, flag, wrapped
    ):
        suffix = "" if flag == "none" else f" overflow<{flag}>"
        numeric = self.freeze(
            f"""module {{ llvm.func @probe(%unused: i32) -> i8 {{
          %a = llvm.mlir.constant({left} : i8) : i8
          %b = llvm.mlir.constant({right} : i8) : i8
          %result = llvm.sub %a, %b{suffix} : i8
          llvm.return %result : i8
        }} }}""",
            (0,),
        )
        message = "nuw property" if flag == "nuw" else "signed overflow or wrapping"
        with self.assertRaisesRegex(NumericDeclined, message):
            lower_numeric(
                numeric,
                mock.Mock(side_effect=AssertionError("Unused source was requested")),
            )
        if wrapped is None:
            with self.assertRaisesRegex(ValueError, flag):
                evaluate_owned(numeric, (scalar("i32", 0),))
        else:
            self.assertEqual(
                evaluate_owned(numeric, (scalar("i32", 0),))[0].integer(), wrapped
            )

    def test_subtraction_keeps_dynamic_negative_domain_decline(self):
        numeric = self.freeze(
            """module { llvm.func @probe(%a: i64, %b: i64) -> i64 {
          %result = llvm.sub %a, %b : i64
          llvm.return %result : i64
        } }""",
            (0, 1),
        )
        values = (-1, -(1 << 63))
        sources = tuple(
            NumericSource(IntExpr("boxed", index), value, value)
            for index, value in enumerate(values)
        )
        with self.assertRaisesRegex(NumericDeclined, "proven nonnegative operands"):
            lower_numeric(numeric, lambda index, path: sources[index])
        self.assertEqual(
            evaluate_owned(numeric, tuple(scalar("i64", value) for value in values))[
                0
            ].integer(),
            (1 << 63) - 1,
        )

    def test_subtraction_does_not_treat_i1_as_signed_arithmetic(self):
        numeric = self.freeze(
            """module { llvm.func @probe(%a: i1, %b: i1) -> i1 {
          %result = llvm.sub %a, %b : i1
          llvm.return %result : i1
        } }""",
            (0, 1),
        )
        with self.assertRaisesRegex(
            NumericDeclined, "exact operand widths or properties"
        ):
            lower_numeric(
                numeric,
                lambda index, path: NumericSource(IntExpr("boxed", index), 0, 1),
            )
        self.assertEqual(
            evaluate_owned(numeric, (scalar("i1", 0), scalar("i1", 1)))[0].integer(
                signed=False
            ),
            1,
        )

    def compile_range_guards(self, obligations, symbol, maximum, *, terminal=True):
        import ctypes

        import sympy

        from torch._dynamo.source import LocalSource
        from torch._inductor.codecache import CppCodeCache
        from torch._inductor.runtime._cudagraph.cute_adapter import _symbolic
        from torch._inductor.runtime._cudagraph.direct_hosttrace import (
            _Lowering,
            _PREDICATE_PREAMBLE,
        )
        from torch._inductor.runtime._cudagraph.guard_export import _TerminalPrinter
        from torch.utils._sympy.value_ranges import ValueRanges

        guards = []
        for obligation in obligations:
            value = _symbolic(obligation.expression, {("size", 0, 0): symbol})
            guards.extend(
                (sympy.Ge(value, obligation.lower), sympy.Le(value, obligation.upper))
            )
        lowering = _Lowering(None)
        checks = [
            f"if (!({lowering.cpp(guard, {symbol: 'value'})}) || bad) return 0;"
            for guard in guards
        ]
        cpp = [
            *_PREDICATE_PREAMBLE,
            'extern "C" int8_t host_guard(int64_t value) {',
            "bool bad = false;",
            *checks,
            "return 1;",
            "}",
        ]
        if terminal:
            source = LocalSource("value")
            mapping = {symbol: [source]}
            printer = _TerminalPrinter(
                mapping,
                lambda source: source.name,
                mapping,
                sources=(source,),
                source_domains={id(source): ValueRanges(0, maximum)},
                symbols=frozenset((symbol,)),
            )
            printed = [printer.doprint(guard) for guard in guards]
            name = printer.source_to_symbol[source].name
            cpp.extend(
                (
                    'extern "C" int8_t terminal_guard(int64_t value) {',
                    f"const int64_t {name} = value;",
                    *(f"if (!({guard})) return 0;" for guard in printed),
                    "return 1;",
                    "}",
                )
            )
        library = CppCodeCache.load("\n".join(cpp))
        for name in ("host_guard", "terminal_guard") if terminal else ("host_guard",):
            function = getattr(library, name)
            function.argtypes = [ctypes.c_int64]
            function.restype = ctypes.c_int8
        return library

    @parametrize("operation", ("add", "mul"))
    @parametrize("source_kind", ("size", "floordiv"))
    def test_metadata_arithmetic_domain_uses_compiled_guards(
        self, operation, source_kind
    ):
        import sympy

        import torch

        numeric = self.freeze(
            f"""module {{ llvm.func @probe(%n: i8) -> i8 {{
          %seven = llvm.mlir.constant(7 : i8) : i8
          %result = llvm.{operation} %n, %seven overflow<nsw> : i8
          llvm.return %result : i8
        }} }}""",
            (0,),
        )
        expression = IntExpr("size", 0, (IntExpr("constant", 0),))
        divisor = 1 if source_kind == "size" else 8
        if divisor != 1:
            expression = IntExpr(
                "floordiv", args=(expression, IntExpr("constant", divisor))
            )
        lowered = lower_numeric(
            numeric, lambda index, path: NumericSource(expression, 0, None)
        )
        upper = 120 if operation == "add" else 18
        self.assertEqual(len(lowered.obligations), 2 if source_kind == "size" else 4)
        width, arithmetic = lowered.obligations[0], lowered.obligations[-1]
        self.assertEqual(
            (width.expression, width.lower, width.upper), (expression, -128, 127)
        )
        self.assertEqual(
            (arithmetic.expression, arithmetic.lower, arithmetic.upper),
            (expression, 0, upper),
        )
        if source_kind == "floordiv":
            self.assertEqual(
                [
                    (row.expression, row.lower, row.upper)
                    for row in lowered.obligations[1:-1]
                ],
                [
                    (expression.args[0], 0, (1 << 63) - 1),
                    (expression.args[1], 1, (1 << 63) - 1),
                ],
            )
        symbol = sympy.Symbol("size", integer=True)
        maximum = 128 * divisor
        library = self.compile_range_guards(lowered.obligations, symbol, maximum)
        records = SimpleNamespace(input_names=("tensor",), integer_inputs=())
        for size in (
            0,
            divisor - 1,
            divisor,
            (upper + 1) * divisor - 1,
            (upper + 1) * divisor,
            maximum,
        ):
            value = size // divisor
            expected = int(value <= upper)
            self.assertEqual(library.host_guard(size), expected)
            self.assertEqual(library.terminal_guard(size), expected)
            if expected:
                tape = _NumericProgram(records, (torch.empty(size),))
                actual = tape.values[tape.add(lowered.values[0].expression)]
                self.assertEqual(
                    actual, evaluate_owned(numeric, (scalar("i8", value),))[0].integer()
                )
            else:
                self.assertGreater(value + 7 if operation == "add" else value * 7, 127)

    def test_derived_guard_overflow_is_rejected(self):
        import sympy

        from torch._inductor.runtime._cudagraph.address_guard_printer import (
            GuardExportDeclined,
        )

        numeric = self.freeze(
            """module { llvm.func @probe(%n: i8) -> i8 {
          %seven = llvm.mlir.constant(7 : i8) : i8
          %result = llvm.add %n, %seven overflow<nsw> : i8
          llvm.return %result : i8
        } }""",
            (0,),
        )
        expression = IntExpr(
            "add",
            args=(
                IntExpr("size", 0, (IntExpr("constant", 0),)),
                IntExpr("constant", 1),
            ),
        )
        lowered = lower_numeric(
            numeric, lambda index, path: NumericSource(expression, 1, None)
        )
        symbol = sympy.Symbol("size", integer=True)
        maximum = (1 << 63) - 1
        with self.assertRaisesRegex(GuardExportDeclined, "fit signed int64"):
            self.compile_range_guards(lowered.obligations, symbol, maximum)
        library = self.compile_range_guards(
            lowered.obligations, symbol, maximum, terminal=False
        )
        for size in (0, 119, 120, 127, maximum - 1, maximum):
            self.assertEqual(library.host_guard(size), int(size <= 119))

    @parametrize(
        "fault", ("overflow", "underflow", "add_overflow", "division", "select", "and")
    )
    def test_cancelled_metadata_keeps_raw_arithmetic_domains(self, fault):
        import ctypes

        import sympy

        import torch
        from torch._inductor.codecache import CppCodeCache
        from torch._inductor.runtime._cudagraph._compiler.cute_bridge.lowering import (
            _Operands,
        )
        from torch._inductor.runtime._cudagraph.cute_adapter import _symbolic
        from torch._inductor.runtime._cudagraph.direct_hosttrace import (
            _Lowering,
            _PREDICATE_PREAMBLE,
        )

        numeric = self.freeze(
            """module { llvm.func @probe(%n: i8) -> i8 {
          %seven = llvm.mlir.constant(7 : i8) : i8
          %result = llvm.add %n, %seven overflow<nsw> : i8
          llvm.return %result : i8
        } }""",
            (0,),
        )
        zero = IntExpr("constant", 0)
        stride = IntExpr("stride", 0, (zero,))
        size = IntExpr("size", 1, (zero,))
        if fault in ("overflow", "underflow"):
            factor = 2 if fault == "overflow" else -2
            hidden = IntExpr("multiply", args=(stride, IntExpr("constant", factor)))
            valid = ((1 << 63) - 1) // 2 if factor > 0 else 1 << 62
            invalid = valid + 1
        elif fault == "add_overflow":
            hidden = IntExpr("add", args=(stride, IntExpr("constant", 1)))
            valid, invalid = (1 << 63) - 2, (1 << 63) - 1
        elif fault == "division":
            dividend = IntExpr("add", args=(stride, IntExpr("constant", -1)))
            hidden = IntExpr("floordiv", args=(dividend, IntExpr("constant", 8)))
            valid, invalid = 1, 0
        else:
            args = (stride, zero, zero) if fault == "select" else (stride, zero)
            hidden = IntExpr(fault, args=args)
            valid, invalid = 1, 2
        expression = IntExpr(
            "add", args=(IntExpr("multiply", args=(hidden, zero)), size)
        )
        operands = _Operands(
            SimpleNamespace(input_contract=SimpleNamespace(integer_ranges=())),
            SimpleNamespace(formals=()),
            {},
            None,
            storage_offset_indices=(0, 1),
        )
        source = operands.numeric_value(expression)
        self.assertEqual((source.lower, source.upper), (0, (1 << 63) - 1))
        lowered = lower_numeric(numeric, lambda index, path: source)
        x, y = sympy.symbols("stride size", integer=True)
        symbols = {("stride", 0, 0): x, ("size", 1, 0): y}
        self.assertEqual(_symbolic(expression, symbols), y)
        guards = []
        for row in lowered.obligations:
            value = _symbolic(row.expression, symbols)
            guards.extend((sympy.Ge(value, row.lower), sympy.Le(value, row.upper)))
        lowering = _Lowering(None)
        checks = [
            f"if (!({lowering.cpp(guard, {x: 'x', y: 'y'})}) || bad) return 0;"
            for guard in guards
        ]
        cpp = "\n".join(
            (
                *_PREDICATE_PREAMBLE,
                'extern "C" int8_t guard(int64_t x, int64_t y) {',
                "bool bad = false;",
                *checks,
                "return 1;",
                "}",
            )
        )
        library = CppCodeCache.load(cpp)
        library.guard.argtypes = [ctypes.c_int64, ctypes.c_int64]
        library.guard.restype = ctypes.c_int8
        records = SimpleNamespace(input_names=("strided", "sized"), integer_inputs=())
        for stride_value, size_value, expected in (
            (valid, 4, 1),
            (invalid, 4, 0),
            (valid, 121, 0),
            (valid, 4, 1),
        ):
            self.assertEqual(library.guard(stride_value, size_value), expected)
            if expected:
                args = (
                    torch.empty_strided(
                        (1,), (stride_value,), device="meta", dtype=torch.uint8
                    ),
                    torch.empty(size_value, device="meta", dtype=torch.uint8),
                )
                tape = _NumericProgram(records, args)
                self.assertEqual(
                    tape.values[tape.add(lowered.values[0].expression)],
                    evaluate_owned(numeric, (scalar("i8", size_value),))[0].integer(),
                )

    @parametrize("kind", ("rational", "float", "to_float", "identity", "sqrt"))
    def test_guard_arithmetic_retains_floating_values(self, kind):
        import ctypes
        import math

        import sympy

        from torch._inductor.codecache import CppCodeCache
        from torch._inductor.runtime._cudagraph.direct_hosttrace import (
            _Lowering,
            _PREDICATE_PREAMBLE,
        )
        from torch.utils._sympy.functions import Identity, ToFloat

        symbol = sympy.Symbol("value", integer=True)
        coefficient = {
            "rational": sympy.Rational(1, 2),
            "float": sympy.Float("1.25"),
            "to_float": ToFloat(symbol),
            "identity": Identity(sympy.Rational(1, 2)),
            "sqrt": sympy.Pow(symbol, sympy.Rational(1, 2), evaluate=False),
        }[kind]
        expression = sympy.Add(
            sympy.Mul(coefficient, symbol, evaluate=False), 3, evaluate=False
        )
        code = _Lowering(None).cpp(expression, {symbol: "value"})
        cpp = "\n".join(
            (
                *_PREDICATE_PREAMBLE,
                'extern "C" int probe(int64_t value, double* result) {',
                "bool bad = false;",
                f"*result = {code};",
                "return bad;",
                "}",
            )
        )
        library = CppCodeCache.load(cpp)
        library.probe.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_double)]
        library.probe.restype = ctypes.c_int
        for value in (1, 7, 1 << 40):
            result = ctypes.c_double()
            self.assertEqual(library.probe(value, ctypes.byref(result)), 0)
            factor = {
                "rational": 0.5,
                "float": 1.25,
                "to_float": float(value),
                "identity": 0.5,
                "sqrt": math.sqrt(value),
            }[kind]
            self.assertEqual(result.value, factor * value + 3.0)

    def test_integer_conversion_keeps_checked_parent_arithmetic(self):
        import ctypes

        import sympy

        from torch._inductor.codecache import CppCodeCache
        from torch._inductor.runtime._cudagraph.direct_hosttrace import (
            _Lowering,
            _PREDICATE_PREAMBLE,
        )
        from torch.utils._sympy.functions import FloorToInt, IntTrueDiv

        symbol = sympy.Symbol("value", integer=True, nonnegative=True)
        converted = FloorToInt(IntTrueDiv(symbol, 2), evaluate=False)
        maximum = (1 << 63) - 1
        expression = sympy.Add(converted, maximum, evaluate=False)
        code = _Lowering(None).cpp(expression, {symbol: "value"})
        cpp = "\n".join(
            (
                *_PREDICATE_PREAMBLE,
                'extern "C" int probe(int64_t value, int64_t* result) {',
                "bool bad = false;",
                f"*result = {code};",
                "return bad;",
                "}",
            )
        )
        library = CppCodeCache.load(cpp)
        library.probe.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_int64)]
        library.probe.restype = ctypes.c_int
        for value in (0, 1, 2, 3):
            result = ctypes.c_int64()
            self.assertEqual(
                library.probe(value, ctypes.byref(result)), int(value >= 2)
            )
            if value < 2:
                self.assertEqual(result.value, maximum)


if __name__ == "__main__":
    run_tests()
