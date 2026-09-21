# Owner(s): ["module: inductor"]

from types import SimpleNamespace

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    AllocateEvent,
    ReinterpretEvent,
)
from torch._inductor.runtime._cudagraph.address_trace import TensorAddressRoots
from torch._inductor.runtime._cudagraph.cuda_tape_import import expression, TapeSources
from torch._inductor.runtime._cudagraph.host_trace import _HostIntegers
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource,
    ExpressionSource,
    InputSource,
    IntExpr,
    PointerSource,
    storage_roots,
)
from torch._inductor.runtime.cudagraph_boxed_replay import (
    _NumericProgram,
    _PhysicalCall,
    _PhysicalField,
)
from torch._inductor.runtime.cudagraph_compiled_evaluation import (
    compile_numeric,
    EarlyStatus,
)
from torch._inductor.runtime.cudagraph_host_trace_mapping import HostTraceInputMetadata
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import FloorDiv, PythonMod


def symbol(env, name, hint):
    source = LocalSource(name)
    term = env.create_unspecified_symbol(hint, source, DimDynamic.DYNAMIC)
    return env.create_symintnode(term, hint=hint, source=source)


@instantiate_parametrized_tests
class TestTapeImport(TestCase):
    def setUp(self):
        super().setUp()
        self.env = ShapeEnv(duck_shape=False, specialize_zero_one=False)
        self.n, self.m = (symbol(self.env, name, 8) for name in ("outer_n", "outer_m"))
        self.mode = FakeTensorMode(shape_env=self.env)
        self.roots = TensorAddressRoots(self.mode)
        self.addresses = {}
        self.base = self.allocate((self.n + 4,))
        with self.mode:
            self.view = self.base.as_strided((self.n,), (1,), 2)
        self.roots.record(ReinterpretEvent(self.base, self.view, (self.n,), (1,), 2))
        self.other = self.allocate((self.m,))
        self.local = ShapeEnv(duck_shape=False, specialize_zero_one=False)
        inputs = []
        for index, offset in ((0, 2), (1, 0)):

            def term(key, hint):
                return symbol(self.local, f"local_{index}_{key}", hint)

            inputs.append(
                SimpleNamespace(
                    position=index,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                    sizes=[term("size", 8)],
                    strides=[term("stride", 1)],
                    offset=term("offset", offset),
                    root=SimpleNamespace(
                        name=f"p{index}", itemsize=4, sym=term("base", 4096)
                    ),
                )
            )
        self.tape = SimpleNamespace(
            shape_env=self.local,
            inputs=inputs,
            allocs=[],
            opaque=[],
            host_buffers=[],
            guards=[],
            nargs=2,
        )
        outer_mapping = SimpleNamespace(
            metadata_symbols={
                expression(self.n): HostTraceInputMetadata(0, "size", 0),
                expression(self.m): HostTraceInputMetadata(1, "size", 0),
            },
            tape=SimpleNamespace(shape_env=self.env),
            translate=lambda value: value,
        )
        self.lower = _HostIntegers(outer_mapping)
        self.imported = self.make_importer()

    def allocate(self, size):
        with self.mode:
            tensor = torch.empty_strided(size, (1,), dtype=torch.float32)
        self.roots.record(
            AllocateEvent(tensor, tuple(size), (1,), torch.float32, tensor.device)
        )
        return tensor

    def address(self, resolution):
        if resolution.root not in self.addresses:
            self.addresses[resolution.root] = self.env.create_unbacked_symint()
        return self.addresses[resolution.root]

    def make_importer(self):
        return TapeSources(
            self.tape,
            {0: self.view, 1: self.other},
            self.roots,
            self.address,
            lambda value: self.lower(expression(value)),
        )

    def test_equal_hints_keep_distinct_formals_and_outer_buffer_roots(self):
        left, right = self.tape.inputs
        self.assertEqual(self.imported.translate(left.sizes[0]), expression(self.n))
        self.assertEqual(self.imported.translate(right.sizes[0]), expression(self.m))
        self.assertNotEqual(expression(self.n), expression(self.m))
        self.assertEqual(
            self.imported.source(InputSource(0)).root, self.roots(self.base).root
        )
        self.assertEqual(
            self.imported.source(InputSource(1)).root, self.roots(self.other).root
        )
        self.assertNotEqual(
            self.imported.source(InputSource(0)).root,
            self.imported.source(InputSource(1)).root,
        )

    def test_storage_base_and_view_pointer_rebase_differ_by_offset(self):
        record = self.tape.inputs[0]
        root = expression(self.address(self.roots(self.base)))
        base = self.imported.translate(record.root.sym)
        pointer = self.imported.translate(record.root.sym + 4 * record.offset)
        self.assertEqual(sympy.simplify(base - root), 0)
        self.assertEqual(sympy.simplify(pointer - root), 8)
        self.assertEqual(
            self.imported.source(InputSource(0)).byte_offset, IntExpr("constant", 8)
        )

    def test_partial_guard_survives_exact_pointer_rebasing(self):
        record = self.tape.inputs[0]
        address = expression(record.root.sym + 4 * record.offset)
        guard = sympy.Ge(FloorDiv(12, FloorDiv(address, 509)), 0)
        self.tape.guards = [guard]
        rebased = self.imported.guards()[-1]
        self.assertTrue(rebased.has(FloorDiv))
        root = expression(self.address(self.roots(self.base)))
        self.assertEqual(rebased.free_symbols, {root})
        self.assertNotEqual(rebased, sympy.true)
        self.assertEqual(rebased.xreplace({root: sympy.Integer(512)}), sympy.true)
        with self.assertRaises(ZeroDivisionError):
            rebased.xreplace({root: sympy.Integer(256)})

    def test_local_allocation_gets_a_fresh_outer_root(self):
        q = symbol(self.local, "allocation_q", 0)
        record = SimpleNamespace(
            name="alloc0",
            q=q,
            root=SimpleNamespace(name="a0", sym=256 * q, itemsize=4),
            dtype=torch.float32,
            sizes=[self.tape.inputs[0].sizes[0] + 1],
            strides=[1],
        )
        self.tape.allocs.append(record)
        imported = self.make_importer()
        output = self.allocate((self.n + 1,))
        imported.bind_allocation(record, output)
        self.assertEqual(
            imported.source(BufferSource("alloc0")).root, self.roots(output).root
        )
        self.assertNotEqual(
            imported.source(BufferSource("alloc0")).root, self.roots(self.base).root
        )
        with self.assertRaisesRegex(UnsupportedCapture, "already imported"):
            imported.bind_allocation(record, output)

    def test_physical_fields_and_grid_share_outer_numeric_sources(self):
        n = IntExpr("size", 0, (IntExpr("constant", 0),))
        owner = object()
        constant = ((2, 0, b"\x81\x02"),)
        call = _PhysicalCall(
            (
                _PhysicalField(
                    0,
                    0,
                    "pointer",
                    PointerSource(InputSource(0), IntExpr("constant", 4)),
                ),
                _PhysicalField(1, 0, "i64", ExpressionSource(n)),
            ),
            owner,
            (n, IntExpr("constant", 1), IntExpr("constant", 1)),
            (),
            constants=constant,
        )
        imported = self.imported.call(call)
        self.assertIs(imported.module, owner)
        self.assertIs(imported.constants, constant)
        self.assertEqual(imported.fields[0].source.root, self.roots(self.base).root)
        self.assertIs(imported.fields[1].source.expression, imported.grid[0])
        program = _NumericProgram(
            SimpleNamespace(input_names=("x", "y"), integer_inputs=()),
            (torch.empty(8), torch.empty(8)),
        )
        offset = program.add(imported.fields[0].source.byte_offset)
        count = program.add(imported.grid[0])
        compiled = compile_numeric(program)
        for length in (3, 8, 17):
            status, values = compiled.evaluate_leaves(
                compiled.bind_inputs((torch.empty(length), torch.empty(8)))
            )
            self.assertEqual(status, EarlyStatus.SUCCESS)
            self.assertEqual(values[offset], 12)
            self.assertEqual(values[count], length)

    def test_unbound_allocations_and_opaque_events_decline(self):
        unknown = sympy.Symbol("unbound", integer=True)
        with self.assertRaisesRegex(UnsupportedCapture, "unbound symbolic source"):
            self.imported.translate(unknown)
        self.tape.opaque.append(object())
        with self.assertRaisesRegex(UnsupportedCapture, "opaque"):
            self.make_importer()

    def test_extra_guards_retain_canonical_mapping_ownership(self):
        mapping = self.imported.mapping
        pointer = next(
            symbol
            for symbol, root in mapping.address_symbols.items()
            if root == InputSource(0)
        )
        guard = sympy.Eq(PythonMod(pointer, 8), 0, evaluate=False)
        self.assertEqual(
            self.imported.guards((guard,))[-1].free_symbols,
            {expression(self.address(self.roots(self.base)))},
        )
        other = self.make_importer()
        with self.assertRaisesRegex(UnsupportedCapture, "unbound local source"):
            other.guards((guard,))

    def test_rng_slots_require_canonical_records(self):
        self.tape.rng_slots = [object()]
        with self.assertRaisesRegex(UnsupportedCapture, "canonical records"):
            self.make_importer()

    def test_rng_request_without_slots_declines(self):
        self.tape.rng_increment = 4
        self.tape.rng_slots = []
        with self.assertRaisesRegex(UnsupportedCapture, "recorded RNG slots"):
            self.make_importer()

    def test_closed_scalar_formals_are_not_silently_rebound(self):
        self.tape.constants = ((1, int, 2),)
        with self.assertRaisesRegex(UnsupportedCapture, "only accepts tensor formals"):
            self.make_importer()


def local_tape(inputs, *, allocate=False):
    env = ShapeEnv(duck_shape=False, specialize_zero_one=False)
    records = []
    for index, (size, offset) in enumerate(inputs):
        records.append(
            SimpleNamespace(
                position=index,
                dtype=torch.float32,
                device=torch.device("cpu"),
                sizes=[symbol(env, f"arg{index}.size", size)],
                strides=[symbol(env, f"arg{index}.stride", 1)],
                offset=symbol(env, f"arg{index}.offset", offset),
                root=SimpleNamespace(
                    name=f"p{index}",
                    itemsize=4,
                    sym=symbol(env, f"arg{index}.base", 4096),
                ),
            )
        )
    allocs = []
    if allocate:
        quotient = symbol(env, "alloc0.base/256", 0)
        allocs.append(
            SimpleNamespace(
                name="alloc0",
                seq=0,
                q=quotient,
                dtype=torch.float32,
                root=SimpleNamespace(name="a0", sym=256 * quotient, itemsize=4),
                sizes=[records[0].sizes[0] + 3],
                strides=[1],
            )
        )
    return SimpleNamespace(
        shape_env=env,
        inputs=records,
        allocs=allocs,
        opaque=[],
        host_buffers=[],
        guards=[],
        constants=(),
        nargs=len(inputs),
    )


@instantiate_parametrized_tests
class TestTapeImportComposition(TestCase):
    def setUp(self):
        super().setUp()
        self.env = ShapeEnv(duck_shape=False, specialize_zero_one=False)
        self.n = symbol(self.env, "outer_n", 8)
        self.m = symbol(self.env, "outer_m", 8)
        self.mode = FakeTensorMode(shape_env=self.env)
        self.roots = TensorAddressRoots(self.mode)
        self.addresses = {}
        self.base = self.allocate(self.n + 4)
        self.first = self.view(self.base, self.n, 2)
        self.other = self.allocate(self.m)
        mapping = SimpleNamespace(
            metadata_symbols={
                expression(self.n): HostTraceInputMetadata(0, "size", 0),
                expression(self.m): HostTraceInputMetadata(1, "size", 0),
            },
            tape=SimpleNamespace(shape_env=self.env),
            translate=lambda value: value,
        )
        self.lower = _HostIntegers(mapping)

    def allocate(self, size):
        with self.mode:
            tensor = torch.empty_strided((size,), (1,), dtype=torch.float32)
        self.roots.record(
            AllocateEvent(tensor, (size,), (1,), tensor.dtype, tensor.device)
        )
        return tensor

    def view(self, source, size, offset):
        with self.mode:
            tensor = source.as_strided((size,), (1,), source.storage_offset() + offset)
        self.roots.record(ReinterpretEvent(source, tensor, (size,), (1,), offset))
        return tensor

    def address(self, resolution):
        if resolution.root not in self.addresses:
            self.addresses[resolution.root] = self.env.create_unbacked_symint()
        return self.addresses[resolution.root]

    def importer(self, tape, *arguments):
        return TapeSources(
            tape,
            dict(enumerate(arguments)),
            self.roots,
            self.address,
            lambda value: self.lower(expression(value)),
        )

    def evaluate(self, expressions, *lengths):
        program = _NumericProgram(
            SimpleNamespace(input_names=("x", "y"), integer_inputs=()),
            (torch.empty(8), torch.empty(8)),
        )
        slots = tuple(program.add(value) for value in expressions)
        compiled = compile_numeric(program)
        status, values = compiled.evaluate_leaves(
            compiled.bind_inputs((torch.empty(lengths[0]), torch.empty(lengths[1])))
        )
        self.assertEqual(status, EarlyStatus.SUCCESS)
        return tuple(values[index] for index in slots)

    @parametrize("reuse_tape", (False, True))
    def test_same_adapter_allocations_have_distinct_outer_identities(self, reuse_tape):
        first_tape = local_tape(((8, 2),), allocate=True)
        second_tape = first_tape if reuse_tape else local_tape(((8, 0),), allocate=True)
        owner = object()
        local_size = IntExpr("size", 0, (IntExpr("constant", 0),))
        call = _PhysicalCall(
            (
                _PhysicalField(0, 0, "pointer", InputSource(0)),
                _PhysicalField(1, 0, "pointer", BufferSource("alloc0")),
                _PhysicalField(2, 0, "i64", ExpressionSource(local_size)),
            ),
            owner,
            (local_size, IntExpr("constant", 1), IntExpr("constant", 1)),
            (),
        )
        imported_calls, output_roots = [], []
        for tape, argument, size in (
            (first_tape, self.first, self.n),
            (second_tape, self.other, self.m),
        ):
            imported = self.importer(tape, argument)
            output = self.allocate(size + 3)
            imported.bind_allocation(tape.allocs[0], output)
            imported_calls.append(imported.call(call))
            output_roots.append(self.roots(output).root)
        self.assertNotEqual(output_roots[0], output_roots[1])
        for call, root, argument in zip(
            imported_calls, output_roots, (self.first, self.other), strict=True
        ):
            self.assertIs(call.module, owner)
            self.assertEqual(
                storage_roots(call.fields[0].source), (self.roots(argument).root,)
            )
            self.assertEqual(storage_roots(call.fields[1].source), (root,))
        self.assertEqual(
            self.evaluate(tuple(call.grid[0] for call in imported_calls), 17, 23),
            (17, 23),
        )

    @parametrize("relationship", ("repeated", "view", "distinct"))
    def test_formal_identity_tracks_outer_storage_and_view_displacement(
        self, relationship
    ):
        second = (
            self.first
            if relationship == "repeated"
            else self.view(self.base, self.n, 1)
            if relationship == "view"
            else self.other
        )
        offset = 2 if relationship == "repeated" else 1 if relationship == "view" else 0
        tape = local_tape(((8, 2), (8, offset)))
        imported = self.importer(tape, self.first, second)
        left, right = (imported.source(InputSource(index)) for index in (0, 1))
        if relationship == "distinct":
            self.assertNotEqual(left.root, right.root)
        else:
            self.assertEqual(left.root, right.root)
        self.assertEqual(
            self.evaluate((left.byte_offset, right.byte_offset), 17, 23),
            (8, offset * 4),
        )
        sizes = tuple(
            imported.integer(IntExpr("size", index, (IntExpr("constant", 0),)))
            for index in (0, 1)
        )
        self.assertEqual(
            self.evaluate(sizes, 17, 23), (17, 23 if relationship == "distinct" else 17)
        )
        pointers = tuple(
            imported.translate(record.root.sym + 4 * record.offset)
            for record in tape.inputs
        )
        if relationship != "distinct":
            self.assertEqual(sympy.simplify(pointers[1] - pointers[0]), offset * 4 - 8)
        else:
            self.assertNotEqual(pointers[0].free_symbols, pointers[1].free_symbols)

    def test_output_view_feeds_next_call_without_losing_root_or_guards(self):
        first_tape = local_tape(((8, 2),), allocate=True)
        first = self.importer(first_tape, self.first)
        allocation = self.allocate(self.n + 3)
        first.bind_allocation(first_tape.allocs[0], allocation)
        output_view = self.view(allocation, self.n + 1, 1)
        second_tape = local_tape(((9, 1),), allocate=True)
        record = second_tape.inputs[0]
        guard = sympy.Eq(
            PythonMod(expression(record.root.sym + 4 * record.offset), 8),
            4,
            evaluate=False,
        )
        second_tape.guards.append(guard)
        second = self.importer(second_tape, output_view)
        final_output = self.allocate(self.n + 4)
        second.bind_allocation(second_tape.allocs[0], final_output)
        owner = object()
        local_size = IntExpr("size", 0, (IntExpr("constant", 0),))
        call = _PhysicalCall(
            (
                _PhysicalField(
                    0,
                    0,
                    "pointer",
                    PointerSource(InputSource(0), IntExpr("constant", 8)),
                ),
                _PhysicalField(1, 0, "pointer", BufferSource("alloc0")),
            ),
            owner,
            (local_size, IntExpr("constant", 1), IntExpr("constant", 1)),
            (),
        )
        imported_call = second.call(call)
        self.assertEqual(
            imported_call.fields[0].source.root, self.roots(allocation).root
        )
        self.assertEqual(
            imported_call.fields[1].source.root, self.roots(final_output).root
        )
        self.assertNotEqual(
            imported_call.fields[0].source.root, imported_call.fields[1].source.root
        )
        self.assertEqual(
            self.evaluate(
                (imported_call.fields[0].source.byte_offset, imported_call.grid[0]),
                17,
                8,
            ),
            (12, 18),
        )
        root = expression(self.address(self.roots(allocation)))
        imported_guard = second.guards()[-1]
        self.assertEqual(imported_guard.free_symbols, {root})
        self.assertEqual(
            imported_guard.xreplace({root: sympy.Integer(256)}), sympy.true
        )
        self.assertEqual(
            imported_guard.xreplace({root: sympy.Integer(258)}), sympy.false
        )
        self.assertEqual(sympy.simplify(second.translate(record.root.sym) - root), 0)


if __name__ == "__main__":
    run_tests()
