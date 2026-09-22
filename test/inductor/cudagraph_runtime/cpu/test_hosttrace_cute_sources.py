# Owner(s): ["module: inductor"]

import contextlib
import struct
from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._inductor.runtime._cudagraph import direct_hosttrace
from torch._inductor.runtime._cudagraph._sdk import activate
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch._inductor.runtime._cudagraph.hosttrace_cute import CuTeLowering
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource,
    InputSource,
    IntegerSource,
    IntExpr,
    ParameterSource,
    PointerSource,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


activate()

from cutlass._mlir import ir

from torch._inductor.runtime._cudagraph._compiler.cfg_values import read_cfg_function
from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.artifact import (
    Consumer,
    FieldSource,
    IntegerField,
    NodeFields,
    ParameterExpression,
)
from torch._inductor.runtime._cudagraph._compiler.cute_bridge.lowering import (
    _lower_cute_call,
    _Operands,
)
from torch._inductor.runtime._cudagraph._compiler.decoded_values import (
    prepare_decodings,
)
from torch._inductor.runtime._cudagraph._compiler.overflow_properties import (
    bind_properties,
)
from torch._inductor.runtime._cudagraph._compiler.owned_numeric import freeze_numeric


class _ReachedLaunch(Exception):
    pass


class _Module:
    def __init__(self, width):
        self.parameter_sizes = (width // 8,)
        self.image = None

    def launch(self, images, grid, stream, shared=None, block=None):
        self.image = images[0]
        raise _ReachedLaunch


@instantiate_parametrized_tests
class TestHostTraceCuTeSources(TestCase):
    def test_original_operand_does_not_add_a_native_pointer_binding(self):
        from test_block_bindings import _Graph, _Module as BindingModule

        from torch._inductor.runtime.cudagraph_boxed_replay import (
            _make_replay,
            _NumericProgram,
        )
        from torch._inductor.runtime.cudagraph_launch_association import (
            RecordedKernelLaunch,
        )

        image = struct.pack("q", 7)
        call = direct_hosttrace._PhysicalCall(
            (direct_hosttrace._PhysicalField(0, 0, "i64", IntegerSource(7)),),
            BindingModule(16),
            tuple(IntExpr("constant", value) for value in (2, 1, 1)),
            (),
            storage_sources=(PointerSource(InputSource(0), IntExpr("constant", 0)),),
        )
        kernel = (40, 17, 0, 0, (2, 1, 1), (8, 4, 1), 16, False, ((0, 8, image),))
        launch = RecordedKernelLaunch(
            10, (1, 20, 30, ()), (1, 20, 30, ((40, bytes(8)),)), 17, (image,)
        )
        numeric = _NumericProgram(
            SimpleNamespace(input_names=("x",), integer_inputs=()), (torch.empty(4),)
        )
        arguments, _ = _make_replay(
            _Graph(kernel), 1, (), (), (), (call,), (launch,), {}, None, numeric=numeric
        )
        self.assertEqual(arguments[3], 1)
        self.assertEqual(arguments[4], ())

    def call(self, kind, width, value):
        typ = f"i{width}"
        if kind == "literal":
            source = FieldSource(
                "compiler_constant", None, None, (), "value", (), typ, (), value
            )
        else:
            expression = ParameterExpression(
                "constant", typ, None, (), f"{value} : {typ}", (), ()
            )
            source = FieldSource(
                "compiler_expression",
                None,
                None,
                (),
                "value",
                (),
                typ,
                (),
                expression=expression,
            )
        fields = NodeFields(
            0, "probe", (width // 8,), (), (IntegerField(0, 0, typ, source),), (), ()
        )
        site = SimpleNamespace(site_id=0, arm=None, fields=fields)
        artifact = SimpleNamespace(sites=(site,), formals=())
        module = _Module(width)
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            owned = ir.Module.parse("""module {
              llvm.func @one() -> i64 {
                %value = llvm.mlir.constant(1 : i64) : i64
                llvm.return %value : i64
              }
            }""")
            cfg = read_cfg_function(owned, "one")
            numeric = freeze_numeric(bind_properties(cfg), prepare_decodings(cfg), ())
            artifact.consumers = tuple(
                Consumer(axis, "one", 0, "grid", axis, "i64", (), numeric)
                for axis in range(3)
            )
            operands = _Operands(
                SimpleNamespace(input_contract=SimpleNamespace(integer_ranges=())),
                artifact,
                {},
                lambda value: IntExpr("constant", value),
            )
            call, predicate, obligations = _lower_cute_call(
                artifact, site, module, operands
            )
        self.assertIsNone(predicate)
        self.assertEqual(obligations, ())
        self.assertIsInstance(
            call.fields[0].source,
            IntegerSource if kind == "literal" else ParameterSource,
        )
        return call, module

    def prepare_to_launch(self, call):
        lowered = SimpleNamespace(
            tape=SimpleNamespace(device_identity=(), launches=({"seq": 0},)),
            contract_holds=lambda args: True,
            box=lambda args, arena, outputs, sequence: list(args),
            written_positions=(),
            device=0,
            records=SimpleNamespace(input_names=(), integer_inputs=()),
            outputs=(),
            allocations=(),
            memsets=(),
            host_tables=(),
            memcpys=(),
            rng=None,
            rng_fields=(),
            regions=(),
            calls=(call,),
        )
        from torch.cuda import _host_trace

        with (
            mock.patch.object(_host_trace, "_device_identity", return_value=()),
            mock.patch.object(
                direct_hosttrace,
                "_register_preparation_regions",
                return_value=((), None),
            ),
            mock.patch.object(direct_hosttrace, "check_predicate", return_value=True),
            mock.patch.object(
                direct_hosttrace, "_capture", return_value=contextlib.nullcontext()
            ),
            mock.patch.object(direct_hosttrace._Preparation, "abort"),
            mock.patch.object(
                torch.cuda, "device", return_value=contextlib.nullcontext()
            ),
            mock.patch.object(torch.cuda, "current_stream"),
            mock.patch.object(torch.cuda, "Stream"),
            mock.patch.object(torch.cuda, "CUDAGraph"),
            mock.patch.object(torch._C, "_cuda_get_capture_frontier"),
        ):
            with self.assertRaises(_ReachedLaunch):
                direct_hosttrace.prepare_hosttrace(lowered, ())

    @parametrize("kind", ("literal", "parameter"))
    @parametrize("width,value", ((32, -17), (64, -(1 << 40) + 3)))
    def test_compiler_integer_sources_reach_preparation(self, kind, width, value):
        call, module = self.call(kind, width, value)
        self.prepare_to_launch(call)
        self.assertEqual(module.image, struct.pack("i" if width == 32 else "q", value))

    def test_parameter_pointer_extends_allocation_use(self):
        root = BufferSource("temporary")
        pointer = PointerSource(root, IntExpr("constant", 12))
        source = ParameterSource("pointer", 64, pointer)
        field = direct_hosttrace._PhysicalField(0, 0, "pointer", source)
        call = direct_hosttrace._PhysicalCall((field,), None, (), ())
        uses, excluded = direct_hosttrace._buffer_uses(
            SimpleNamespace(launches=({"seq": 9},)), (call,), (), (), (), ()
        )
        self.assertEqual(uses, {"temporary": [9]})
        self.assertEqual(tuple(excluded), ())

    def test_original_operand_use_survives_physical_lowering(self):
        pointer = PointerSource(BufferSource("operand"), IntExpr("constant", 4))
        bound = direct_hosttrace._PhysicalCall((), None, (), ())
        receipt = SimpleNamespace(owner=None)
        call = CuTeCall(bound, (pointer,), (), receipt)
        invocation = object()
        launch = {
            "cute": SimpleNamespace(invocation=invocation, site_index=0),
            "kernel": "probe",
        }
        symbols = SimpleNamespace(mapping=SimpleNamespace(root_alignments={}))
        with mock.patch.object(torch.cuda, "current_stream"):
            lowering = CuTeLowering(None, None, symbols, 0)
        with mock.patch.object(lowering, "_lower_invocation", return_value=(call,)):
            physical = lowering.lower(launch)
        uses, _ = direct_hosttrace._buffer_uses(
            SimpleNamespace(launches=({"seq": 7},)), (physical,), (), (), (), ()
        )
        self.assertEqual(uses, {"operand": [7]})
        self.assertEqual(physical.storage_sources, call.pointers)

    @parametrize("allocator", ("arena", "sequence"))
    def test_encoded_and_original_pointers_rebase_together(self, allocator):
        root = BufferSource("temporary")
        pointer = PointerSource(root, IntExpr("constant", 12))
        leaf = ParameterSource("pointer", 64, pointer)
        encoded = ParameterSource("add", 64, args=(leaf, leaf))
        field = direct_hosttrace._PhysicalField(0, 0, "i64", encoded)
        call = direct_hosttrace._PhysicalCall(
            (field,), None, (), (), storage_sources=(pointer,)
        )
        record = SimpleNamespace(
            name="temporary",
            sizes=(32,),
            strides=(1,),
            root=SimpleNamespace(itemsize=4),
            seq=0,
        )
        tape = SimpleNamespace(launches=({"seq": 9},), allocs=(record,))
        layout = SimpleNamespace(source=root)
        symbols = SimpleNamespace(by_symbol={})
        lowering = SimpleNamespace(subst=sympy.sympify)
        args = (
            tape,
            symbols,
            lowering,
            (layout,),
            (call,),
            (),
            (),
            (),
            (),
            (),
            {"temporary"},
            {},
            3,
        )
        if allocator == "arena":
            result = direct_hosttrace._arena_pass(*args, sympy.Integer(4096), {})
        else:
            result = direct_hosttrace._sequence_pass(*args)
        plan, allocations, calls = result[:3]
        self.assertTrue(plan.covers("temporary"))
        self.assertEqual(allocations, ())
        rebased = calls[0]
        expected = PointerSource(InputSource(3), IntExpr("constant", 12))
        self.assertEqual(rebased.storage_sources, (expected,))
        self.assertEqual(rebased.fields[0].source.pointers, (expected,))
        self.assertIs(
            rebased.fields[0].source.args[0], rebased.fields[0].source.args[1]
        )
        self.assertEqual(call.fields[0].source.pointers, (pointer,))

    def test_deep_shared_parameter_rebase(self):
        pointer = PointerSource(BufferSource("temporary"), IntExpr("constant", 12))
        leaf = ParameterSource("pointer", 64, pointer)
        encoded = leaf
        depth = 2048
        for _ in range(depth):
            encoded = ParameterSource("add", 64, args=(encoded, leaf), flags=("nuw",))
        call = direct_hosttrace._PhysicalCall(
            (
                direct_hosttrace._PhysicalField(0, 0, "i64", encoded),
                direct_hosttrace._PhysicalField(1, 0, "i64", encoded),
            ),
            None,
            (),
            (),
            storage_sources=(pointer,),
        )
        expected = PointerSource(InputSource(3), IntExpr("constant", 28))
        rebased = direct_hosttrace._rebase_call(
            call, lambda value: expected if type(value) is PointerSource else value
        )
        self.assertEqual(rebased.storage_sources, (expected,))
        node = rebased.fields[0].source
        self.assertIs(node, rebased.fields[1].source)
        rebased_leaf = node.args[1]
        for _ in range(depth):
            self.assertEqual((node.op, node.width, node.flags), ("add", 64, ("nuw",)))
            self.assertIs(node.args[1], rebased_leaf)
            node = node.args[0]
        self.assertIs(node, rebased_leaf)
        self.assertEqual(node.value, expected)
        self.assertEqual(leaf.value, pointer)
        self.assertIs(call.fields[0].source, encoded)


if __name__ == "__main__":
    run_tests()
