# Owner(s): ["module: cuda graphs"]

import gc
import math
import os
import random
import subprocess
import sys

import torch
from torch.cuda._host_trace_program import (
    _read_leaf,
    _step,
    compile_program,
    f32_bits,
    f64_bits,
    IntegerProgram,
    LEAVES,
    MAX_I64,
    MIN_I64,
    OutOfDomain,
    Status,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


S = Status
# (op, operands, the value or the status the evaluator returns)
CASES = [
    ("add", (MAX_I64 - 1, 1), MAX_I64),
    ("add", (MAX_I64, 1), S.ADD_OVERFLOW),
    ("add", (MIN_I64, -1), S.ADD_OVERFLOW),
    ("add", (MIN_I64, MAX_I64), -1),
    ("multiply", (-(1 << 62), 2), MIN_I64),
    ("multiply", (1 << 62, 2), S.MULTIPLY_OVERFLOW),
    ("multiply", (MIN_I64, -1), S.MULTIPLY_OVERFLOW),
    ("floordiv", (7, 2), 3),
    ("floordiv", (0, 5), 0),
    ("floordiv", (MAX_I64, 1), MAX_I64),
    ("floordiv", (7, 0), S.DIVISION_DOMAIN),
    ("floordiv", (7, -2), S.DIVISION_DOMAIN),
    ("floordiv", (-7, 2), S.DIVISION_DOMAIN),
    ("floordiv", (MIN_I64, -1), S.DIVISION_DOMAIN),
    ("ceildiv", (7, 2), 4),
    ("ceildiv", (8, 2), 4),
    ("ceildiv", (0, 3), 0),
    ("ceildiv", (MAX_I64, 2), 1 << 62),
    ("ceildiv", (-1, 2), S.DIVISION_DOMAIN),
    ("ceildiv", (5, 0), S.DIVISION_DOMAIN),
    ("eq", (3, 3), 1),
    ("ne", (3, 3), 0),
    ("lt", (MIN_I64, MAX_I64), 1),
    ("le", (4, 3), 0),
    ("gt", (-1, -2), 1),
    ("ge", (3, 3), 1),
    ("and", (1, 1), 1),
    ("and", (1, 0), 0),
    ("and", (2, 1), S.BOOLEAN_DOMAIN),
    ("and", (1, -1), S.BOOLEAN_DOMAIN),
    ("select", (1, 5, 6), 5),
    ("select", (0, 5, 6), 6),
    ("select", (2, 5, 6), S.BOOLEAN_DOMAIN),
    ("bitand", (-8, 12), 8),
    ("bitor", (5, 3), 7),
    ("bitor", (MIN_I64, MAX_I64), -1),
    ("bitxor", (-1, MAX_I64), MIN_I64),
    ("bitlength", (0,), 0),
    ("bitlength", (5,), 3),
    ("bitlength", (-5,), 3),
    ("bitlength", (MAX_I64,), 63),
    ("bitlength", (MIN_I64,), 64),
    ("lshift", (3, 2), 12),
    ("lshift", (-3, 0), -3),
    ("lshift", (0, 1 << 40), 0),
    ("lshift", (1, 62), 1 << 62),
    ("lshift", (-1, 63), MIN_I64),
    ("lshift", (1, 63), S.MULTIPLY_OVERFLOW),
    ("lshift", (-2, 63), S.MULTIPLY_OVERFLOW),
    ("lshift", (1, 64), S.MULTIPLY_OVERFLOW),
    ("lshift", (1 << 31, 32), S.MULTIPLY_OVERFLOW),
    ("lshift", (1, -1), S.SHIFT_DOMAIN),
    ("lshift", (0, -1), S.SHIFT_DOMAIN),
    ("f32div", (1, 3), f32_bits(1, 3)),
    ("f32div", (MAX_I64, 7), f32_bits(MAX_I64, 7)),
    ("f32div", (0, 5), 0),
    ("f32div", (1, 0), S.DIVISION_DOMAIN),
    ("f32div", (-1, 2), S.DIVISION_DOMAIN),
    ("tofloat", (3,), f64_bits(3.0)),
    ("tofloat", (MAX_I64,), f64_bits(2.0**63)),
    ("tofloat", ((1 << 53) + 1,), f64_bits(2.0**53)),
    ("fsqrt", (f64_bits(64.0),), f64_bits(8.0)),
    ("fsqrt", (f64_bits(2921.0),), f64_bits(math.sqrt(2921.0))),
    ("fsqrt", (f64_bits(-0.0),), f64_bits(-0.0)),
    ("fsqrt", (f64_bits(-1.0),), S.FLOAT_DOMAIN),
    ("fsqrt", (f64_bits(math.inf),), S.FLOAT_DOMAIN),
    ("fsqrt", (f64_bits(math.nan),), S.FLOAT_DOMAIN),
    ("fdiv", (f64_bits(1.0), f64_bits(math.sqrt(80.0))), f64_bits(1.0 / math.sqrt(80.0))),
    ("fdiv", (f64_bits(1.0), f64_bits(0.0)), S.FLOAT_DOMAIN),
    ("fdiv", (f64_bits(0.0), f64_bits(-0.0)), S.FLOAT_DOMAIN),
    ("fdiv", (f64_bits(1e308), f64_bits(1e-308)), S.FLOAT_DOMAIN),
    ("feq", (f64_bits(0.125), f64_bits(0.125)), 1),
    ("feq", (f64_bits(0.0), f64_bits(-0.0)), 1),
    ("feq", (f64_bits(math.nan), f64_bits(math.nan)), 0),
    ("flt", (f64_bits(-3.0), f64_bits(0.5)), 1),
    ("flt", (f64_bits(0.5), f64_bits(-3.0)), 0),
    ("flt", (f64_bits(-0.0), f64_bits(0.0)), 0),
    ("flt", (f64_bits(-2.0), f64_bits(-1.0)), 1),
    ("min", (3, MIN_I64, 4), MIN_I64),
    ("max", (3, MAX_I64, 4), MAX_I64),
]


EDGES = [0, 1, -1, 2, -2, 3, 7, 62, 63, 64, 1 << 31, 1 << 62, -(1 << 62)]
EDGES += [MIN_I64, MIN_I64 + 1, MAX_I64, MAX_I64 - 1]
OPS = ["add", "multiply", "floordiv", "ceildiv", "eq", "ne", "lt", "le", "gt", "ge"]
OPS += ["and", "bitand", "bitor", "bitxor", "bitlength", "lshift", "f32div", "min", "max", "select"]
OPS += ["tofloat", "fsqrt", "fdiv", "feq", "flt"]


def _reference(instructions, leaves):
    # the evaluator's status and values by `_step`, row by row
    values, reads = [], iter(leaves)
    for op, *args in instructions:
        if op == "constant":
            v = args[0]
        elif op in LEAVES:
            v = next(reads)
        else:
            v = _step(op, [values[x] for x in args])
            if isinstance(v, Status):
                return v, None
        values.append(v)
    return Status.SUCCESS, tuple(values)


def _random_view(rng):
    sizes = [rng.randint(0, 5) for _ in range(rng.choice([1, 2, 2, 2]))]
    strides = [rng.choice([0, 1, 2, 5]) for _ in sizes]
    return torch.empty(64).as_strided(sizes, strides, rng.randint(0, 8))


def _over_boxed(op, n, hint=1):
    # op applied to n int inputs
    p = IntegerProgram([hint] * n)
    p.emit(op, *[p.emit("boxed", i) for i in range(n)])
    return p


class TestIntegerProgram(TestCase):
    @parametrize("case", CASES, name_fn=lambda c: f"{c[0]}_{CASES.index(c)}")
    def test_evaluator_matches_the_emission_rules(self, case):
        op, args, want = case
        compiled = compile_program(_over_boxed(op, len(args)))
        status, out = compiled.evaluate_inputs(args)
        if isinstance(want, Status):
            self.assertEqual(status, want)
            p = IntegerProgram(list(args))
            with self.assertRaises(OutOfDomain) as e:
                p.emit(op, *[p.emit("boxed", i) for i in range(len(args))])
            self.assertEqual(e.exception.status, want)
        else:
            self.assertEqual(status, S.SUCCESS)
            self.assertEqual(out, (*args, want))
            p = IntegerProgram(list(args))
            p.emit(op, *[p.emit("boxed", i) for i in range(len(args))])
            self.assertEqual(p.values, list(out))

    @parametrize("op", ["min", "max"])
    @parametrize("count", [2, 3, 9])
    def test_min_max_read_every_operand(self, op, count):
        compiled = compile_program(_over_boxed(op, count))
        extreme = -5 if op == "min" else 5
        for at in range(count):
            args = [0] * count
            args[at] = extreme
            self.assertEqual(compiled.evaluate_inputs(args), (S.SUCCESS, (*args, extreme)))

    def test_first_failing_row_decides_the_status(self):
        # evaluation is in row order: an overflowing prefix fails even when
        # the final value fits, and an unselected branch still fails
        p = IntegerProgram([1, 1])
        a, b = p.emit("boxed", 0), p.emit("boxed", 1)
        total = p.emit("add", p.emit("add", a, b), p.emit("constant", -MAX_I64))
        quotient = p.emit("floordiv", a, b)
        p.emit("select", p.emit("constant", 1), total, quotient)
        compiled = compile_program(p)
        status, out = compiled.evaluate_inputs((MAX_I64 - 1, 1))
        self.assertEqual(status, S.SUCCESS)
        self.assertEqual(out[total], 0)
        self.assertEqual(compiled.evaluate_inputs((MAX_I64, 1))[0], S.ADD_OVERFLOW)
        self.assertEqual(compiled.evaluate_inputs((5, 0))[0], S.DIVISION_DOMAIN)

    def test_domains_rows_fail_at_no_call(self):
        # emitted while `domains` is a list, a division or shift reads
        # admitted operands and records where its own were admitted
        p = IntegerProgram([7, 2])
        a, b = p.emit("boxed", 0), p.emit("boxed", 1)
        p.domains = []
        rows = [p.emit(op, a, b) for op in ("floordiv", "ceildiv", "lshift")]
        domains, p.domains = p.domains, None
        compiled = compile_program(p)
        calls = (((7, 2), [3, 4, 28], [1, 1, 1]), ((7, 0), [0, 0, 7], [0, 0, 1]), ((-7, -1), [0, 0, -7], [0, 0, 0]))
        for args, values, admitted in calls:
            status, out = compiled.evaluate_inputs(args)
            self.assertEqual(status, S.SUCCESS)
            self.assertEqual([out[r] for r in rows], values)
            self.assertEqual([out[r] for r in domains], admitted)

    def test_float_domains_rows_fail_at_no_call(self):
        p = IntegerProgram([4, 2])
        x, y = (p.emit("tofloat", p.emit("boxed", i)) for i in range(2))
        p.domains = []
        rows = [p.emit("fsqrt", x), p.emit("fdiv", x, y)]
        # a quotient by a double of magnitude below 1 may overflow: not admitted
        with self.assertRaises(OutOfDomain):
            p.emit("fdiv", x, p.emit("constant", f64_bits(0.5)))
        domains, p.domains = p.domains, None
        compiled = compile_program(p)
        calls = (((4, 2), [2.0, 2.0], [1, 1]), ((4, -2), [2.0, -2.0], [1, 1]), ((-4, 0), [0.0, 0.0], [0, 0]))
        for args, values, admitted in calls:
            status, out = compiled.evaluate_inputs(args)
            self.assertEqual(status, S.SUCCESS)
            self.assertEqual([out[r] for r in rows], [f64_bits(v) for v in values])
            self.assertEqual([out[r] for r in domains], admitted)

    def test_identical_rows_are_one_row(self):
        p = IntegerProgram([3, torch.zeros(2, 3)])
        a = p.emit("boxed", 0)
        s = p.emit("size", 1, 1)
        self.assertEqual(p.emit("boxed", 0), a)
        self.assertEqual(p.emit("size", 1, 1), s)
        self.assertNotEqual(p.emit("size", 1, 0), s)
        m = p.emit("multiply", a, s)
        self.assertEqual(p.emit("multiply", a, s), m)
        # no canonicalization: the operands' order is part of the row
        self.assertNotEqual(p.emit("multiply", s, a), m)
        self.assertEqual(p.emit("max", a, s, a), p.emit("max", a, s, a))
        self.assertEqual(len(p.instructions), 6)
        self.assertEqual(p.values, [3, 3, 2, 9, 9, 3])

    def test_metadata_leaves(self):
        def view(base):
            return base.t()[1:]

        base = torch.arange(24).reshape(4, 6)
        p = IntegerProgram([view(base), 10])
        rows = [
            p.emit("pointer", 0),
            p.emit("storage_offset", 0),
            p.emit("size", 0, 0),
            p.emit("size", 0, 1),
            p.emit("stride", 0, 0),
            p.emit("stride", 0, 1),
            p.emit("boxed", 1),
        ]
        last = p.emit("add", p.emit("multiply", rows[2], rows[6]), rows[1])
        compiled = compile_program(p)

        other = torch.arange(40).reshape(5, 8)
        x = view(other)
        status, out = compiled.evaluate_inputs([x, 3])
        self.assertEqual(status, S.SUCCESS)
        self.assertEqual(out[: len(rows)], (x.data_ptr(), 1, 7, 5, 1, 8, 3))
        self.assertEqual(out[last], 7 * 3 + 1)
        self.assertEqual(compiled.evaluate_inputs(p.inputs), (S.SUCCESS, tuple(p.values)))

    def test_evaluate_inputs_checks_each_leaf(self):
        p = IntegerProgram([torch.zeros(2, 3), 4])
        p.emit("stride", 0, 1)
        p.emit("boxed", 1)
        compiled = compile_program(p)
        t = torch.zeros(2, 3)
        self.assertEqual(compiled.evaluate_inputs([t, 5]), (S.SUCCESS, (1, 5)))
        bad = [
            [t],
            [t, 5, 6],
            [t, True],
            [t, 5.0],
            [t, 1 << 63],
            [5, 5],
            [torch.zeros(3), 5],
            [torch.zeros(2, 3).to_sparse(), 5],
        ]
        for inputs in bad:
            self.assertIsNone(compiled.evaluate_inputs(inputs), msg=str(inputs))

    def test_bind_reads_exactly_tensor_or_parameter(self):
        # a subclass could report other metadata, or run Python that clears
        # `inputs` while bind reads it
        inputs = []

        class Lying(torch.Tensor):
            def size(self, *dim):
                return 99 if dim else (99, 99)

            def data_ptr(self):
                return 12345

        class Clearing(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                policy = "sizes"
                return cls._make_wrapper_subclass(
                    cls, elem.shape, dispatch_sizes_strides_policy=policy
                )

            def __init__(self, elem):
                self.elem = elem

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                inputs.clear()
                gc.collect()
                name = func.overloadpacket.__name__.removeprefix("sym_")
                return getattr(args[0].elem, name)()

        p = IntegerProgram([torch.zeros(3, 4), torch.zeros(5, 6)])
        p.emit("size", 0, 0)
        p.emit("pointer", 0)
        p.emit("size", 1, 0)
        compiled = compile_program(p)
        t = torch.zeros(5, 6)[1:]
        param = torch.nn.Parameter(torch.zeros(3, 4))
        self.assertEqual(compiled.evaluate_inputs([param, t]), (S.SUCCESS, (3, param.data_ptr(), 4)))
        for x in [torch.zeros(3, 4).as_subclass(Lying), Clearing(torch.zeros(3, 4))]:
            for _ in range(20):
                inputs[:] = [x, torch.zeros(5, 6)[1:]]
                self.assertIsNone(compiled.evaluate_inputs(inputs))
                inputs[:] = [x, torch.zeros(5, 6)[1:]]
                self.assertIsNone(compiled.evaluate_inputs(tuple(inputs)))

    def test_constructor_rejects_malformed_rows(self):
        # the interpreter checks the rows itself: evaluation has no bounds checks
        bad = [
            ([("constant", 1), ("add", 0, 2)], 0),
            ([("constant", 1), ("add", 0, 1)], 0),
            ([("constant", 1), ("add", 0, -1)], 0),
            ([("constant", 1), ("add", 0, 1 << 40)], 0),
            ([("constant", 1), ("add", 0, 0.5)], 0),
            ([("constant", 1), ("min", 0)], 0),
            ([("constant", 1), ("min",)], 0),
            ([("constant", 1), ("select", 0, 0)], 0),
            ([("constant", 1), ("bitlength", 0, 0)], 0),
            ([("constant", 1), ("lshift", 0)], 0),
            ([("constant", 1), ("f32div", 0)], 0),
            ([("constant", 1), ("tofloat", 0, 0)], 0),
            ([("constant", 1), ("fsqrt", 0, 0)], 0),
            ([("constant", 1), ("fdiv", 0)], 0),
            ([("constant", 1), ("feq", 0, 0, 0)], 0),
            ([("constant", 1), ("flt", 0)], 0),
            ([("boxed", 3)], 2),
            ([("pointer", -1)], 2),
            ([("size", 0, -1)], 1),
            ([("size", 0)], 1),
            ([("boxed", 0, 0)], 1),
            ([("constant", 1, 2)], 0),
            ([("constant",)], 0),
            ([("constant", 1 << 63)], 0),
            ([()], 0),
            ([("mod", 0, 0)], 0),
            ([(3, 1)], 0),
            ([], -1),
        ]
        for rows, input_count in bad:
            with self.assertRaises((ValueError, RuntimeError), msg=str(rows)):
                torch._C._HostTraceProgram(rows, input_count)

    def test_malformed_rows_are_rejected(self):
        p = IntegerProgram([4, torch.zeros(2)])
        a = p.emit("boxed", 0)
        bad = [
            ("constant", 1 << 63),
            ("constant", MIN_I64 - 1),
            ("constant",),
            ("boxed", 1),
            ("boxed", 2),
            ("size", 0, 0),
            ("size", 1, 1),
            ("size", 1),
            ("pointer", 0),
            ("add", a, a + 1),
            ("add", a),
            ("select", a, a),
            ("min", a),
            ("mod", a, a),
        ]
        for row in bad:
            with self.assertRaises(ValueError, msg=str(row)):
                p.emit(*row)
        with self.assertRaises(TypeError):
            p.emit("add", a, True)
        self.assertEqual(p.instructions, [("boxed", 0)])

    def test_int64_bounds_as_constants(self):
        p = IntegerProgram([])
        lo, hi = p.emit("constant", MIN_I64), p.emit("constant", MAX_I64)
        p.emit("add", lo, hi)
        self.assertEqual(compile_program(p).evaluate_inputs(()), (S.SUCCESS, (MIN_I64, MAX_I64, -1)))

    @parametrize("seed", range(4))
    def test_matches_the_reference_on_random_programs(self, seed):
        # random well-formed programs over int and tensor inputs, evaluated at
        # the int64 edges: the same leaves, status and values as `_step`
        rng = random.Random(seed)

        doubles = [f64_bits(x) for x in (0.0, -0.0, 0.5, 2.0, 64.0, -3.0, 1e308, 1e-308, math.inf, math.nan)]

        def value():
            r = rng.random()
            return rng.choice(EDGES) if r < 0.4 else rng.choice(doubles) if r < 0.6 else rng.randint(-9, 9)

        for _ in range(50):
            p = IntegerProgram([value(), value(), _random_view(rng), value()])
            for i in (0, 1, 3):
                p.emit("boxed", i)
            if p.inputs[2].dim() == 2:
                for leaf in ("size", "stride"):
                    p.emit(leaf, 2, rng.randint(0, 1))
                p.emit("pointer", 2)
                p.emit("storage_offset", 2)
            for _ in range(rng.randint(1, 30)):
                op = rng.choice([*OPS, "constant"])
                if op == "constant":
                    operands = [value()]
                else:
                    k = rng.randint(2, 4)
                    arity = {"select": 3, "bitlength": 1, "tofloat": 1, "fsqrt": 1, "min": k, "max": k}.get(op, 2)
                    rows = len(p.instructions)
                    operands = [rng.randrange(rows) for _ in range(arity)]
                try:
                    p.emit(op, *operands)
                except OutOfDomain:
                    pass
            compiled = compile_program(p)
            leaves = [row for row in p.instructions if row[0] in LEAVES]
            for _ in range(20):
                inputs = [value(), value(), _random_view(rng), value()]
                read = [_read_leaf(row, inputs[row[1]]) for row in leaves]
                if None in read:
                    self.assertIsNone(compiled.evaluate_inputs(inputs))
                    continue
                self.assertEqual(compiled.evaluate_inputs(inputs), _reference(p.instructions, read))

    def test_no_compiler_at_runtime(self):
        # the evaluator is in libtorch: no host compiler, no Inductor import
        script = """if True:
            import sys
            import torch
            from torch.cuda._host_trace_program import compile_program, IntegerProgram
            p = IntegerProgram([3, torch.zeros(2, 5)])
            p.emit("multiply", p.emit("boxed", 0), p.emit("size", 1, 1))
            compiled = compile_program(p)
            assert compiled.evaluate_inputs([4, torch.zeros(1, 6)]) == (0, (4, 6, 24))
            assert not any(m.startswith("torch._inductor") for m in sys.modules), "inductor"
        """
        env = {**os.environ, "CC": "/nonexistent/cc", "CXX": "/nonexistent/c++"}
        subprocess.check_call([sys.executable, "-c", script], env=env)

    def test_empty_program(self):
        compiled = compile_program(IntegerProgram([1]))
        self.assertEqual(compiled.evaluate_inputs([2]), (S.SUCCESS, ()))


instantiate_parametrized_tests(TestIntegerProgram)


def setUpModule():
    from torch.cuda import _host_trace_hint_audit
    import torch.cuda._host_trace as host_trace

    _host_trace_hint_audit.enable_for_tests()
    host_trace.raise_unexpected = True


if __name__ == "__main__":
    run_tests()
