# Owner(s): ["module: inductor"]
"""Live symbolic storage roots retain provenance across the host-trace bridge."""

from types import SimpleNamespace
from unittest.mock import Mock

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource,
    InputSource,
    IntExpr,
    PointerSource,
)
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch._inductor.runtime.cudagraph_host_trace_mapping import (
    HostTraceInputMetadata,
    HostTraceSymbolMapping,
)
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import FloorDiv, Identity


def symbol(environment, name, hint):
    source = LocalSource(name)
    expression = environment.create_unspecified_symbol(hint, source, DimDynamic.DYNAMIC)
    return environment.create_symintnode(expression, hint=hint, source=source)


def input_record(environment, position=0, offset=3, dtype=torch.float32):
    prefix = f"arg{position}"
    root = SimpleNamespace(
        name=f"p{position}",
        sym=symbol(environment, f"{prefix}_base", 1 << (40 + position)),
        itemsize=dtype.itemsize,
    )
    return SimpleNamespace(
        position=position,
        name=prefix,
        dtype=dtype,
        sizes=[symbol(environment, f"{prefix}_size", 9)],
        strides=[symbol(environment, f"{prefix}_stride", 1)],
        offset=symbol(environment, f"{prefix}_offset", offset),
        root=root,
    )


@instantiate_parametrized_tests
class TestHostTraceSymbolMapping(TestCase):
    def setUp(self):
        super().setUp()
        self.environment = ShapeEnv(duck_shape=False, specialize_zero_one=False)
        self.input = input_record(self.environment)
        self.tape = SimpleNamespace(
            shape_env=self.environment,
            inputs=[self.input],
            allocs=[],
            opaque=[],
            nargs=1,
        )

    @parametrize("offset", (0, 3, 11))
    def test_input_root_uses_current_data_pointer_and_storage_offset(self, offset):
        self.input = input_record(self.environment, offset=offset)
        self.tape.inputs = [self.input]
        mapping = HostTraceSymbolMapping(self.tape)
        (address,) = mapping.address_symbols
        original_offset = self.input.offset.node._expr
        expression = mapping.translate(self.input.root.sym)
        self.assertEqual(expression, address - original_offset * 4)
        self.assertEqual(mapping.address_symbols[address], InputSource(0))
        self.assertEqual(mapping.root_alignments, {InputSource(0): 1})
        self.assertIs(mapping.tape, self.tape)
        values = {address: (1 << 46) + 20, original_offset: offset + 2}
        self.assertEqual(
            int(expression.xreplace(values)), values[address] - 4 * (offset + 2)
        )
        view = mapping.view_pointer(
            self.input.root, self.input.offset + 2, torch.float32, int
        )
        self.assertEqual(view, PointerSource(InputSource(0), IntExpr("constant", 8)))

    @parametrize("offset", (0, 3, 11))
    def test_absolute_view_keeps_dynamic_input_offset(self, offset):
        mapping = HostTraceSymbolMapping(self.tape)
        numeric_offset = IntExpr(
            "multiply", args=(IntExpr("constant", -4), IntExpr("storage_offset", 0))
        )
        lower = Mock(return_value=numeric_offset)
        pointer = mapping.view_pointer(self.input.root, 0, torch.float32, lower)
        lower.assert_called_once_with(-4 * self.input.offset.node._expr)
        tensor = torch.empty(64)[offset : offset + 9]
        program = _NumericProgram(
            SimpleNamespace(input_names=("x",), integer_inputs=()), [tensor]
        )
        slot = program.add(pointer.byte_offset)
        self.assertEqual(program.values[slot], -4 * offset)
        self.assertEqual(
            tensor.data_ptr() + program.values[slot],
            tensor.untyped_storage().data_ptr(),
        )
        self.assertIn(("storage_offset", 0), program.instructions)

    @parametrize(
        "dtype,view_dtype,scale",
        ((torch.complex64, torch.float32, 2), (torch.float32, torch.uint8, 4)),
    )
    def test_dtype_changing_view_uses_its_own_element_width(
        self, dtype, view_dtype, scale
    ):
        self.input = input_record(self.environment, dtype=dtype)
        self.tape.inputs = [self.input]
        mapping = HostTraceSymbolMapping(self.tape)
        offset = self.input.offset * scale + 1
        pointer = mapping.view_pointer(self.input.root, offset, view_dtype, int)
        self.assertEqual(
            pointer,
            PointerSource(InputSource(0), IntExpr("constant", view_dtype.itemsize)),
        )
        (address,) = mapping.address_symbols
        absolute = (
            self.input.root.sym.node.expr + offset.node.expr * view_dtype.itemsize
        )
        self.assertEqual(mapping.translate(absolute), address + view_dtype.itemsize)

    def test_metadata_bindings_keep_original_symbols_after_specialization(self):
        size = self.input.sizes[0]
        original = size.node._expr
        self.assertTrue(size == 9)
        self.assertEqual(size.node.expr, 9)
        mapping = HostTraceSymbolMapping(self.tape)
        self.assertEqual(
            mapping.metadata_symbols[original], HostTraceInputMetadata(0, "size", 0)
        )
        self.assertEqual(
            mapping.metadata_symbols[self.input.strides[0].node._expr],
            HostTraceInputMetadata(0, "stride", 0),
        )
        self.assertEqual(
            mapping.metadata_symbols[self.input.offset.node._expr],
            HostTraceInputMetadata(0, "storage_offset"),
        )
        self.assertEqual(
            mapping.translate(sympy.Eq(original, 9)), sympy.Eq(original, 9)
        )

    def test_tensor_positions_can_be_compacted_around_constant_arguments(self):
        self.input = input_record(self.environment, position=2)
        self.tape.inputs = [self.input]
        self.tape.nargs = 4
        mapping = HostTraceSymbolMapping(self.tape, input_indices={2: 0})
        self.assertEqual(mapping.input_indices, {2: 0})
        self.assertEqual(mapping.root_source(self.input.root), InputSource(0))
        self.assertEqual(
            mapping.metadata_symbols[self.input.sizes[0].node._expr],
            HostTraceInputMetadata(0, "size", 0),
        )
        pointer = mapping.view_pointer(
            self.input.root, self.input.offset + 1, torch.float32, int
        )
        self.assertEqual(pointer, PointerSource(InputSource(0), IntExpr("constant", 4)))
        with self.assertRaisesRegex(
            UnsupportedCapture, "exact runtime argument indices"
        ):
            HostTraceSymbolMapping(self.tape, input_indices={1: 0})

    def test_guards_and_derived_scalars_share_root_substitution(self):
        mapping = HostTraceSymbolMapping(self.tape)
        (address,) = mapping.address_symbols
        base = self.input.root.sym.node.expr
        offset = self.input.offset.node.expr
        guard = sympy.Eq(sympy.Mod(base + 4 * offset, 16), 0)
        translated = mapping.translate(guard)
        self.assertEqual(translated, sympy.Eq(sympy.Mod(address, 16), 0))
        scalar = mapping.translate(FloorDiv(base + 4 * offset, 16))
        self.assertEqual(scalar, FloorDiv(address, 16))
        self.assertTrue(bool(translated.xreplace({address: 1 << 44})))
        self.assertFalse(bool(translated.xreplace({address: (1 << 44) + 4})))
        self.assertEqual(
            mapping.translate(self.input.root.sym.node), address - 4 * offset
        )

    def test_translation_preserves_float_identity_barriers(self):
        mapping = HostTraceSymbolMapping(self.tape)
        base = self.input.root.sym.node.expr
        (address,) = mapping.address_symbols
        expected_base = address - 4 * self.input.offset.node.expr
        value = Identity(Identity(sympy.Float(0.1) * base) + sympy.Float(0.2))
        translated = mapping.translate(value)
        self.assertEqual(
            translated,
            Identity(Identity(sympy.Float(0.1) * expected_base) + sympy.Float(0.2)),
        )
        self.assertEqual(len(translated.atoms(Identity)), 2)

    def test_owned_intermediate_has_allocator_alignment_and_dynamic_layout(self):
        quotient = symbol(self.environment, "owned_quotient", 0)
        root = SimpleNamespace(name="a0", sym=256 * quotient, itemsize=4)
        allocation = SimpleNamespace(
            name="alloc0",
            seq=0,
            root=root,
            q=quotient,
            dtype=torch.float32,
            sizes=[self.input.sizes[0]],
            strides=[1],
        )
        self.tape.allocs = [allocation]
        mapping = HostTraceSymbolMapping(self.tape)
        source = BufferSource("alloc0")
        pointer_symbol = next(
            symbol
            for symbol, value in mapping.address_symbols.items()
            if value == source
        )
        self.assertEqual(mapping.root_source(root), source)
        self.assertEqual(mapping.root_alignments[source], 256)
        self.assertEqual(mapping.translate(root.sym), pointer_symbol)
        self.assertEqual(mapping.translate(quotient), pointer_symbol / 256)
        self.assertEqual(
            mapping.translate(allocation.sizes[0]), self.input.sizes[0].node.expr
        )
        self.assertEqual(
            mapping.view_pointer(root, 3, torch.float16, int),
            PointerSource(source, IntExpr("constant", 6)),
        )
        address = (1 << 45) + 256 * 17
        self.assertEqual(
            int(mapping.translate(quotient + 2).xreplace({pointer_symbol: address})),
            address // 256 + 2,
        )

    @parametrize("failure", ("alignment", "quotient", "width"))
    def test_allocation_requires_exact_alignment_provenance(self, failure):
        quotient = symbol(self.environment, "owned_quotient", 0)
        root = SimpleNamespace(name="a0", sym=256 * quotient, itemsize=4)
        allocation = SimpleNamespace(
            name="alloc0", root=root, q=quotient, dtype=torch.float32
        )
        if failure == "alignment":
            root.sym = 128 * quotient
        elif failure == "quotient":
            allocation.q = quotient + 1
        else:
            root.itemsize = 8
        self.tape.allocs = [allocation]
        with self.assertRaisesRegex(UnsupportedCapture, "alignment fact|element width"):
            HostTraceSymbolMapping(self.tape)

    @parametrize("ambiguity", ("root", "symbol", "metadata", "position"))
    def test_ambiguous_input_roots_decline(self, ambiguity):
        other = input_record(self.environment, position=1)
        if ambiguity == "root":
            other.root = self.input.root
        elif ambiguity == "symbol":
            other.root.sym = self.input.root.sym
        elif ambiguity == "metadata":
            other.offset = self.input.offset
        else:
            other.position = self.input.position
        self.tape.inputs.append(other)
        self.tape.nargs = 2
        with self.assertRaisesRegex(UnsupportedCapture, "ambiguous"):
            HostTraceSymbolMapping(self.tape)

    def test_pointer_with_multiple_or_nonlinear_roots_declines(self):
        other = input_record(self.environment, position=1)
        self.tape.inputs.append(other)
        self.tape.nargs = 2
        mapping = HostTraceSymbolMapping(self.tape)
        with self.assertRaisesRegex(UnsupportedCapture, "exactly one"):
            mapping.pointer(self.input.root.sym + other.root.sym, int)
        with self.assertRaisesRegex(UnsupportedCapture, "unit storage-root"):
            mapping.pointer(2 * self.input.root.sym, int)
        with self.assertRaisesRegex(UnsupportedCapture, "exactly one"):
            mapping.pointer(1 << 40, int)
        with self.assertRaisesRegex(UnsupportedCapture, "integer address"):
            mapping.pointer(self.input.root.sym.node.expr + sympy.Rational(1, 2), int)

    def test_unrecorded_symbol_and_root_decline(self):
        mapping = HostTraceSymbolMapping(self.tape)
        with self.assertRaisesRegex(UnsupportedCapture, "unbound symbolic"):
            mapping.translate(sympy.Symbol("unknown", integer=True))
        clone = SimpleNamespace(**vars(self.input.root))
        with self.assertRaisesRegex(UnsupportedCapture, "no recorded storage root"):
            mapping.view_pointer(clone, 0, torch.float32, int)
        foreign = ShapeEnv(duck_shape=False, specialize_zero_one=False)
        with self.assertRaisesRegex(UnsupportedCapture, "another ShapeEnv"):
            mapping.translate(symbol(foreign, "foreign_size", 9))


if __name__ == "__main__":
    run_tests()
