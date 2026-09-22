# Owner(s): ["module: inductor"]
"""Native host callbacks and ordered floating-point values in compiled replay."""

import ctypes
import gc
import math
import struct
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import sympy

import torch
from torch._inductor.codecache import CppCodeCache
from torch._inductor.runtime.cudagraph_arg_mapping import (
    grid_expression_inputs,
    IntegerInput,
    IntExpr,
    ParameterSource,
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
    STRICT_FLOAT_FLAGS,
)
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.cuda._host_trace import _Evaluator, _round_float32, Float32
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _bits(value):
    return struct.unpack("q", struct.pack("d", value))[0]


def _program(examples=()):
    names = tuple(f"arg{index}" for index in range(len(examples)))
    records = SimpleNamespace(
        input_names=names,
        integer_inputs=tuple(
            IntegerInput(name, index) for index, name in enumerate(names)
        ),
    )
    return _NumericProgram(records, examples)


_REFERENCE = r"""#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
extern "C" int64_t host_rebind(const std::vector<int64_t>& args) {
  if (args.size() != 2 || args[1] == 0) throw std::invalid_argument("invalid divisor");
  return args[0] / args[1] + 7;
}
extern "C" int64_t host_empty(const std::vector<int64_t>& args) {
  if (!args.empty()) throw std::invalid_argument("expected no arguments");
  return 13;
}
static std::vector<std::vector<int64_t>> calls;
static int64_t fail_at;
extern "C" void reset_calls(int64_t failure) {
  calls.clear();
  fail_at = failure;
}
extern "C" int64_t call_count() { return calls.size(); }
extern "C" int64_t call_size(int64_t index) { return calls.at(index).size(); }
extern "C" int64_t call_argument(int64_t index, int64_t argument) {
  return calls.at(index).at(argument);
}
extern "C" int64_t host_sequence(const std::vector<int64_t>& args) {
  calls.push_back(args);
  if (static_cast<int64_t>(calls.size()) - 1 == fail_at) {
    throw std::invalid_argument("callback failed");
  }
  int64_t result = 100 * calls.size();
  for (int64_t value : args) result += value;
  return result;
}
extern "C" int64_t host_rebind_ptr(const int64_t* args, size_t count) {
  if (count != 2 || args[1] == 0) throw std::invalid_argument("invalid divisor");
  return args[0] / args[1] + 7;
}
extern "C" int64_t host_empty_ptr(const int64_t* args, size_t count) {
  if (count != 0 || args != nullptr) throw std::invalid_argument("expected no arguments");
  return 13;
}
extern "C" int64_t reference(int op, int64_t x, int64_t y) {
  double a, b;
  std::memcpy(&a, &x, sizeof(a));
  std::memcpy(&b, &y, sizeof(b));
  volatile double left = a, right = b;
  double value;
  switch (op) {
    case 0: value = -left; break;
    case 1: value = std::sqrt(left); break;
    case 2: value = left + right; break;
    case 3: value = left - right; break;
    case 4: value = left * right; break;
    case 5: value = left / right; break;
    case 6: value = std::pow(left, right); break;
    case 8: value = static_cast<double>(static_cast<float>(left)); break;
    default: {
      const float narrowed = static_cast<float>(left);
      int32_t bits;
      std::memcpy(&bits, &narrowed, sizeof(bits));
      return bits;
    }
  }
  int64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
static float original_float_formal(float scale) {
  return scale * M_LOG2E;
}
extern "C" int32_t original_flash_scale(double scale) {
  const float result = original_float_formal(scale);
  int32_t bits;
  std::memcpy(&bits, &result, sizeof(bits));
  return bits;
}
"""


class _CallOwner:
    def __init__(self, empty=False, pointer_abi=False):
        self.library = CppCodeCache.load(_REFERENCE, extra_flags=STRICT_FLOAT_FLAGS)
        name = ("host_empty" if empty else "host_rebind") + (
            "_ptr" if pointer_abi else ""
        )
        function = getattr(self.library, name)
        self.address = ctypes.cast(function, ctypes.c_void_p).value
        self.preparation_calls = 0

    def __call__(self, arguments):
        self.preparation_calls += 1
        return 13 if not arguments else arguments[0] // arguments[1] + 7


@instantiate_parametrized_tests
class TestHostEvaluation(TestCase):
    def sequence_library(self):
        library = CppCodeCache.load(_REFERENCE, extra_flags=STRICT_FLOAT_FLAGS)
        library.reset_calls.argtypes = (ctypes.c_int64,)
        library.reset_calls.restype = None
        for name, arity in (("call_count", 0), ("call_size", 1), ("call_argument", 2)):
            function = getattr(library, name)
            function.argtypes = (ctypes.c_int64,) * arity
            function.restype = ctypes.c_int64
        return library

    @parametrize("failure", (-1, 0, 2, 4))
    def test_callback_argument_sequence_and_failure(self, failure):
        library = self.sequence_library()
        owner = library.host_sequence
        address = ctypes.cast(owner, ctypes.c_void_p).value
        program = _program((3, 5, 7, 11))
        program.instructions = [("boxed", index) for index in range(4)]
        arguments = ((), (0, 1), (3, 2, 1, 0), (2,), ())
        program.instructions.extend(
            ("call", address, owner, *args) for args in arguments
        )
        compiled = compile_numeric(program)
        for inputs in ((3, 5, 7, 11), (13, 17, 19, 23)):
            library.reset_calls(failure)
            status, values = compiled.evaluate_leaves(inputs)
            self.assertEqual(
                status,
                EarlyStatus.SUCCESS if failure == -1 else EarlyStatus.CALL_FAILED,
            )
            count = len(arguments) if failure == -1 else failure + 1
            self.assertEqual(library.call_count(), count)
            for index, args in enumerate(arguments[:count]):
                self.assertEqual(library.call_size(index), len(args))
                actual = tuple(
                    library.call_argument(index, slot) for slot in range(len(args))
                )
                expected = tuple(inputs[slot] for slot in args)
                self.assertEqual(actual, expected)
                if index != failure:
                    self.assertEqual(
                        values[4 + index], 100 * (index + 1) + sum(expected)
                    )

    @parametrize("before_first_call", (True, False))
    def test_arithmetic_failure_preserves_callback_prefix(self, before_first_call):
        library = self.sequence_library()
        owner = library.host_sequence
        address = ctypes.cast(owner, ctypes.c_void_p).value
        program = _program((2,))
        program.instructions = [("boxed", 0)]
        if not before_first_call:
            program.instructions.append(("call", address, owner))
        numerator = len(program.instructions)
        program.instructions.extend(
            (("constant", 8), ("floordiv", numerator, 0), ("call", address, owner, 0))
        )
        compiled = compile_numeric(program)
        for divisor in (0, 2):
            library.reset_calls(-1)
            status, values = compiled.evaluate_leaves((divisor,))
            self.assertEqual(
                status,
                EarlyStatus.DIVISION_DOMAIN if divisor == 0 else EarlyStatus.SUCCESS,
            )
            count = int(not before_first_call) + int(divisor != 0)
            self.assertEqual(library.call_count(), count)
            if divisor != 0:
                self.assertEqual(values[-1], count * 100 + divisor)

    @parametrize(
        "op",
        (
            "fneg",
            "fsqrt",
            "fadd",
            "fsub",
            "fmul",
            "fdiv",
            "fpow",
            "ftobits32",
            "fround32",
        ),
    )
    @parametrize(
        "operands",
        (
            (1.5, 2.0),
            (-0.0, 0.0),
            (0.0, -0.0),
            (-4.0, 0.5),
            (math.inf, -math.inf),
            (math.nan, 2.0),
            (1e300, 1e-300),
        ),
    )
    def test_native_float_differential(self, op, operands):
        unary = op in ("fneg", "fsqrt", "ftobits32", "fround32")
        count = 1 if unary else 2
        examples = (_bits(4.0), _bits(2.0))[:count]
        arguments = tuple(IntExpr("boxed", index) for index in range(count))
        program = _program(examples)
        output = program.add(IntExpr(op, args=arguments))
        compiled = compile_numeric(program)
        inputs = tuple(_bits(value) for value in operands[:count])
        status, values = compiled.evaluate_leaves(inputs)
        library = CppCodeCache.load(_REFERENCE, extra_flags=STRICT_FLOAT_FLAGS)
        reference = library.reference
        reference.argtypes = (ctypes.c_int, ctypes.c_int64, ctypes.c_int64)
        reference.restype = ctypes.c_int64
        ops = (
            "fneg",
            "fsqrt",
            "fadd",
            "fsub",
            "fmul",
            "fdiv",
            "fpow",
            "ftobits32",
            "fround32",
        )
        expected = reference(ops.index(op), inputs[0], 0 if unary else inputs[1])
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(values[output], expected)
        prepared = _program(inputs)
        prepared.add(IntExpr(op, args=arguments))
        self.assertEqual(tuple(prepared.values), values)

    @parametrize("value", (-(1 << 63), -(1 << 53) - 1, 0, (1 << 53) + 1, (1 << 63) - 1))
    def test_integer_conversion(self, value):
        program = _program((1,))
        output = program.add(IntExpr("ffromint", args=(IntExpr("boxed", 0),)))
        status, values = compile_numeric(program).evaluate_leaves((value,))
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(values[output], _bits(float(value)))

    @parametrize("value", (-0.0, math.inf, -math.inf, math.nan))
    def test_constant_bits(self, value):
        program = _program()
        output = program.add(IntExpr("fconst", _bits(value)))
        status, values = compile_numeric(program).evaluate_leaves(())
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(values[output], _bits(value))

    @parametrize("head_dim", (8, 16, 32, 48, 64, 80, 96, 120, 128, 160, 200, 256))
    def test_original_flash_float_formal(self, head_dim):
        scale = 1.0 / math.sqrt(head_dim)
        rounded = IntExpr("fround32", args=(IntExpr("boxed", 0),))
        log2_scale = IntExpr(
            "fmul", args=(rounded, IntExpr("fconst", _bits(math.log2(math.e))))
        )
        program = _program((_bits(scale),))
        output = program.add(IntExpr("ftobits32", args=(log2_scale,)))
        status, values = compile_numeric(program).evaluate_leaves((_bits(scale),))
        reference = CppCodeCache.load(
            _REFERENCE, extra_flags=STRICT_FLOAT_FLAGS
        ).original_flash_scale
        reference.argtypes = (ctypes.c_double,)
        reference.restype = ctypes.c_int32
        expected = reference(scale)
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(values[output], expected)
        self.assertEqual(program.values[output], expected)
        golden = {48: 0x3E553B94, 120: 0x3E06DC37, 200: 0x3DD0ECB1}
        if head_dim in golden:
            self.assertEqual(expected, golden[head_dim])

    @parametrize("value", (0.0, -0.0, 1.0 + 2**-24, 1e300, -1e300, math.inf, math.nan))
    def test_float32_hint_and_python_execution(self, value):
        reference = CppCodeCache.load(
            _REFERENCE, extra_flags=STRICT_FLOAT_FLAGS
        ).reference
        reference.argtypes = (ctypes.c_int, ctypes.c_int64, ctypes.c_int64)
        reference.restype = ctypes.c_int64
        expected = reference(8, _bits(value), 0)
        self.assertEqual(_bits(_round_float32(value)), expected)
        env = ShapeEnv()
        symbol = sympy.Symbol("scale", real=True)
        original = torch.SymFloat(SymNode(symbol, env, float, value))
        rounded = _round_float32(original)
        self.assertIs(rounded.node.shape_env, env)
        self.assertEqual(rounded.node._expr, Float32(symbol))
        self.assertEqual(_bits(rounded.node.hint), expected)
        self.assertEqual(env.guards, [])
        program = _Evaluator()
        self.assertEqual(_bits(program.ev(rounded, {"scale": value})), expected)

    def test_float32_keeps_original_symbol_and_fx_provenance(self):
        env = ShapeEnv()
        symbol = sympy.Symbol("scale", real=True)
        original = torch.SymFloat(SymNode(symbol, env, float, 1.1))
        env._set_replacement(symbol, sympy.Float(1.1), "test replacement")
        with patch.object(
            env, "_create_fx_call_function", return_value=(None, True)
        ) as fx:
            rounded = _round_float32(original)
        self.assertEqual(rounded.node._expr, Float32(symbol))
        fx.assert_called_once_with(_round_float32, (original.node.fx_node,))
        self.assertEqual(env.guards, [])

    def test_addition_is_not_reassociated(self):
        program = _program(tuple(map(_bits, (1e16, -1e16, 1.0))))
        a, b, c = (IntExpr("boxed", index) for index in range(3))
        left = program.add(IntExpr("fadd", args=(IntExpr("fadd", args=(a, b)), c)))
        right = program.add(IntExpr("fadd", args=(a, IntExpr("fadd", args=(b, c)))))
        status, values = compile_numeric(program).evaluate_leaves(
            tuple(program.inputs.values())
        )
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(values[left], _bits(1.0))
        self.assertEqual(values[right], _bits(0.0))

    def test_multiply_add_is_not_contracted(self):
        inputs = tuple(map(_bits, (1.0 + 2**-27, 1.0 - 2**-27, -1.0)))
        program = _program(inputs)
        a, b, c = (IntExpr("boxed", index) for index in range(3))
        output = program.add(IntExpr("fadd", args=(IntExpr("fmul", args=(a, b)), c)))
        status, values = compile_numeric(program).evaluate_leaves(inputs)
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(values[output], _bits(0.0))

    def test_callback_rebinding_and_exception(self):
        owner = _CallOwner()
        expression = IntExpr(
            "call", (owner.address, owner), (IntExpr("boxed", 0), IntExpr("boxed", 1))
        )
        program = _program((24, 8))
        output = program.add(expression)
        compiled = compile_numeric(program)
        self.assertEqual(owner.preparation_calls, 1)
        status, values = compiled.evaluate_leaves((1024, 32))
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(values[output], 39)
        self.assertEqual(owner.preparation_calls, 1)
        status, _ = compiled.evaluate_leaves((1024, 0))
        self.assertEqual(status, EarlyStatus.CALL_FAILED)

    def test_pointer_abi_callback_rebinding_and_exception(self):
        # `pcall`: the same call through (const int64_t*, size_t), the arguments a stack
        # array of the evaluator; a throwing callee fails the evaluation the same way
        owner = _CallOwner(pointer_abi=True)
        expression = IntExpr(
            "pcall", (owner.address, owner), (IntExpr("boxed", 0), IntExpr("boxed", 1))
        )
        program = _program((24, 8))
        output = program.add(expression)
        self.assertEqual(program.instructions[output][0], "pcall")
        compiled = compile_numeric(program)
        self.assertIn("const int64_t arguments[] = {", compiled.cpp_source)
        self.assertNotIn("std::vector<int64_t> arguments", compiled.cpp_source)
        self.assertIn(owner, compiled.library_owner)
        status, values = compiled.evaluate_leaves((1024, 32))
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(values[output], 39)
        status, _ = compiled.evaluate_leaves((1024, 0))
        self.assertEqual(status, EarlyStatus.CALL_FAILED)
        empty = _CallOwner(empty=True, pointer_abi=True)
        program = _program()
        output = program.add(IntExpr("pcall", (empty.address, empty)))
        compiled = compile_evaluation(program, pointer_count=0)
        status, values = compiled.early.evaluate_leaves(())
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(values[output], 13)

    def test_callback_owner_survives_program(self):
        owner = _CallOwner(empty=True)
        reference = weakref.ref(owner)
        program = _program()
        output = program.add(IntExpr("call", (owner.address, owner)))
        compiled = compile_evaluation(program, pointer_count=0)
        del owner, program
        gc.collect()
        self.assertIsNotNone(reference())
        status, values = compiled.early.evaluate_leaves(())
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(values[output], 13)
        del compiled
        gc.collect()
        self.assertIsNone(reference())

    def test_callback_float_and_physical_parameter_roundtrip(self):
        owner = _CallOwner()
        call = IntExpr(
            "call", (owner.address, owner), (IntExpr("boxed", 0), IntExpr("boxed", 1))
        )
        scale = IntExpr(
            "fdiv",
            args=(
                IntExpr("fconst", _bits(1.0)),
                IntExpr("fsqrt", args=(IntExpr("ffromint", args=(call,)),)),
            ),
        )
        narrowed = IntExpr("ftobits32", args=(scale,))
        numeric = _program((24, 8))
        parameters = _ParameterProgram(numeric, 2, {})
        parameters.add(
            ParameterSource("trunc", 32, args=(ParameterSource("value", 64, narrowed),))
        )
        compiled = compile_evaluation(numeric, parameters, pointer_count=2)
        status, early = compiled.early.evaluate_leaves((1024, 32))
        self.assertEqual(status, EarlyStatus.SUCCESS)
        status, late = compiled.late.evaluate(early, (0, 0))
        self.assertEqual(status, LateStatus.SUCCESS)
        expected = struct.unpack("i", struct.pack("f", 1.0 / math.sqrt(39.0)))[0]
        self.assertEqual(late, (expected,))

    def test_call_grid_origins_are_checked_without_float_shapes(self):
        owner = _CallOwner()
        size = IntExpr("size", 1, (IntExpr("constant", 0),))
        call = IntExpr("call", (owner.address, owner), (size, IntExpr("boxed", 0)))
        self.assertEqual(
            grid_expression_inputs(call, boxed_indices={0}, tensor_indices={1}), (0, 1)
        )
        self.assertIsNone(
            grid_expression_inputs(call, boxed_indices={0}, tensor_indices={2})
        )
        self.assertIsNone(grid_expression_inputs(IntExpr("ffromint", args=(call,))))

    @parametrize(
        "bad",
        (
            (0, lambda values: 1),
            (-1, lambda values: 1),
            (1 << 64, lambda values: 1),
            (1, None),
        ),
    )
    def test_callback_admission(self, bad):
        with self.assertRaisesRegex(UnsupportedCapture, "address and callable owner"):
            _program().add(IntExpr("call", bad))

    def test_float_arity_rejected(self):
        with self.assertRaisesRegex(UnsupportedCapture, "number of operands"):
            _program().add(IntExpr("fadd"))

    def test_invalid_callback_instruction_rejected(self):
        program = _program()
        program.instructions = [("call", 1, lambda values: 1, 0)]
        with self.assertRaisesRegex(ValueError, "operands are invalid"):
            compile_numeric(program)


if __name__ == "__main__":
    run_tests()
