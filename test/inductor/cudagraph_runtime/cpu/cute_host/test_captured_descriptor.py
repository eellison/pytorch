# Owner(s): ["module: inductor"]

import dataclasses
import itertools
import struct
from types import SimpleNamespace

import torch
from torch._inductor.runtime._cudagraph._sdk import activate
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


activate()

import cutlass
import cutlass.cute as cute

from cuda.bindings import driver

from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.factory import (
    bind_captured_artifact,
)
from torch._inductor.runtime._cudagraph._compiler.cute_bridge.lowering import (
    _lower_cute_call,
    _Operands,
)
from torch._inductor.runtime._cudagraph._compiler.cute_bridge.numeric import (
    NumericDeclined,
)
from torch._inductor.runtime._cudagraph._compiler.entry_signature import (
    build_entry_signature,
)
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_reuse.binding import (
    bind_captured_metadata,
)
from torch._inductor.runtime._cudagraph._compiler.python_entry import EntryCall, Operand
from torch._inductor.runtime.cudagraph_arg_mapping import (
    InputSource,
    IntExpr,
    PointerSource,
)
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.cuda import _host_trace_cute_dsl as dsl
from torch.cuda._host_trace_cute import _RECORDER_NODE, _RecorderInvocation
from torch.cuda._host_trace_cute_desc import (
    _arguments,
    _captured_launches,
    Descriptor,
    Program,
    record,
    Unexpressed,
)
from torch.fx.experimental.symbolic_shapes import ShapeEnv


@cute.kernel
def _kernel(a: cute.Tensor, out: cute.Tensor, eps: cutlass.Float32):
    out[0] = a[0] + eps


@cute.jit
def _host(
    a: cute.Tensor,
    optional: cute.Tensor | None,
    out: cute.Tensor,
    eps: cutlass.Float32,
    stream: driver.CUstream,
):
    _kernel(a, out, eps).launch(
        grid=((cute.size(a.shape) + 3) // 4, 1, 1), block=(32, 1, 1), stream=stream
    )
    _kernel(a, out, eps).launch(
        grid=((cute.size(a.shape) + 3) // 4, 1, 1), block=(32, 1, 1), stream=stream
    )


@cute.jit
def _conditional_then_outer(
    a: cute.Tensor, out: cute.Tensor, eps: cutlass.Float32, stream: driver.CUstream
):
    if cute.size(a.shape) > 8:
        _kernel(a, out, eps).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    _kernel(a, out, eps).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


@cute.jit
def _middle_stream_host(
    a: cute.Tensor,
    stream: driver.CUstream,
    optional: cute.Tensor | None,
    out: cute.Tensor,
    eps: cutlass.Float32,
):
    _kernel(a, out, eps).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


@cute.jit
def _first_stream_host(
    stream: driver.CUstream,
    a: cute.Tensor,
    optional: cute.Tensor | None,
    out: cute.Tensor,
    eps: cutlass.Float32,
):
    _kernel(a, out, eps).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


@instantiate_parametrized_tests
class TestCapturedDescriptor(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        dsl.install()
        n = cute.sym_int(32)
        fake = cute.runtime.make_fake_tensor(
            cutlass.Float32, (n,), (1,), assumed_align=16
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        cls.compiled = cute.compile(
            _host,
            fake,
            None,
            fake,
            cutlass.Float32(0),
            stream,
            options="--gpu-arch sm_90a --enable-tvm-ffi",
            no_jit_engine=True,
        )
        cls.descriptor = dsl._compiles[cls.compiled].descriptor
        cls.descriptor.check()
        cls.serialized = cls.descriptor.to_json()

    def bind(
        self,
        rows,
        eps,
        *,
        loaded=False,
        descriptor=None,
        owner=None,
        stride=1,
        lower=True,
    ):
        descriptor = self.descriptor if descriptor is None else descriptor
        owner = self.compiled if owner is None else owner
        if loaded:
            owner = dsl.loaded_program(lambda *args: None, object(), self.serialized)
            descriptor = owner.descriptor
        program = Program(descriptor, "test", __name__, owner, (), "cuda:0", 1)
        mode = FakeTensorMode(shape_env=ShapeEnv())
        with mode:
            x = torch.empty_strided((rows,), (stride,), device="cuda")
            y = torch.empty_strided((rows,), (stride,), device="cuda")
        values = (x, y, eps)
        call = EntryCall(
            0,
            program.entry,
            program.entry.target,
            (),
            _RECORDER_NODE,
            values,
            (),
            tuple(Operand(("args", i), value, value) for i, value in enumerate(values)),
        )
        local = _RecorderInvocation(
            program, program, call, values, values, None, mode.shape_env, mode
        )
        signature = build_entry_signature(
            local, call, policy=program.policy, metadata=descriptor.metadata
        )
        binding = bind_captured_metadata(signature, program)
        artifact = bind_captured_artifact(binding)
        tensors = {
            id(value): PointerSource(InputSource(i), IntExpr("constant", 0))
            for i, value in enumerate((x, y))
        }
        operands = _Operands(
            local, artifact, tensors, lambda value: IntExpr("constant", int(value))
        )
        bound, obligations = None, ()
        if lower:
            bound, _, obligations = _lower_cute_call(
                artifact, artifact.sites[0], object(), operands
            )
        return program, artifact, bound, obligations, x, y

    def test_real_tvm_payload_roundtrip(self):
        loaded = Descriptor.from_json(self.serialized)
        self.assertEqual(loaded.to_json(), self.serialized)
        self.assertEqual(loaded.metadata.abi, "Abi.Tbd")
        self.assertEqual(
            [
                (row.name, row.ir_arg_index, row.abi_arg_index)
                for row in loaded.metadata.params
            ],
            [("a", 0, 0), ("out", 1, 1), ("eps", 2, 2), ("stream", 3, None)],
        )
        self.assertEqual(
            [formal.source_arg_index for formal in loaded.payload.formals], [0, 1, 2, 3]
        )
        self.assertEqual(
            [name for name, *_ in loaded.parameters], ["a", "optional", "out", "eps"]
        )

    @parametrize("loaded", [False, True])
    @parametrize("eps", [0.0, -0.0, 0.03125])
    def test_actual_call_float_bits_and_typed_grid(self, loaded, eps):
        program, artifact, bound, _, x, y = self.bind(17, eps, loaded=loaded)
        field = next(
            row
            for row in artifact.sites[0].fields.integers
            if row.source.formal_name == "eps"
        )
        value = next(
            row
            for row in bound.fields
            if row.parameter == field.parameter and row.byte_offset == field.byte_offset
        )
        self.assertEqual(value.kind, "i32")
        self.assertEqual(value.source.op, "constant")
        self.assertEqual(
            value.source.value, struct.unpack("<I", struct.pack("<f", eps))[0]
        )
        self.assertIs(artifact._guards.binding.kernel_owner, program.kernel_owner)
        self.assertIs(artifact._payload, program.descriptor.payload)
        arguments = _arguments(program.descriptor, (x, None, y, eps), {})
        self.assertIs(arguments["a"], x)
        self.assertIs(arguments["out"], y)
        self.assertIsNone(arguments["optional"])
        self.assertEqual(arguments["eps"], eps)
        from torch._inductor.runtime._cudagraph.address_scalars import symbolic_integer

        self.assertEqual(
            tuple(int(symbolic_integer(expr, {}, ValueError)) for expr in bound.grid),
            (5, 1, 1),
        )

    def test_elided_unit_is_still_a_runtime_argument(self):
        _, _, _, _, x, y = self.bind(8, 0.125)
        with self.assertRaisesRegex(Unexpressed, "Unit operand must remain None"):
            _arguments(self.descriptor, (x, y, y, 0.125), {})
        with self.assertRaises(TypeError):
            _arguments(self.descriptor, (x, None, y, 0.125), {"eps": 0.5})

    def test_bound_constant_change_is_rejected(self):
        program, artifact, _, _, _, _ = self.bind(8, 0.125)
        use = artifact.signature.operands[2].scalar.use
        object.__setattr__(use, "value", 0.25)
        with self.assertRaisesRegex(RuntimeError, "changed|lost"):
            artifact.check()
        self.assertIs(artifact._guards.binding.kernel_owner, program.kernel_owner)

    def test_nested_launch_cannot_be_omitted(self):
        fake = cute.runtime.make_fake_tensor(
            cutlass.Float32, (cute.sym_int(32),), (1,), assumed_align=16
        )
        compiled = cute.compile(
            _conditional_then_outer,
            fake,
            fake,
            cutlass.Float32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--gpu-arch sm_90a --enable-tvm-ffi",
            no_jit_engine=True,
        )
        descriptor = dsl._compiles[compiled].descriptor
        self.assertIsNone(descriptor.payload)
        self.assertRegex(
            descriptor.declined, "source.*Unsupported shared-memory diagnostic effects"
        )
        with self.assertRaisesRegex(
            Unexpressed, "Unsupported shared-memory diagnostic effects"
        ):
            descriptor.check()

    @parametrize("stream_index", [0, 1])
    def test_stream_position_preserves_compiler_indices(self, stream_index):
        from torch.cuda import _host_trace as ht
        from torch.cuda._host_trace_cute import CuteTrace

        fake = cute.runtime.make_fake_tensor(
            cutlass.Float32, (cute.sym_int(32),), (1,), assumed_align=16
        )
        arguments = [fake, None, fake, cutlass.Float32(0)]
        arguments.insert(
            stream_index, cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        )
        compiled = cute.compile(
            _first_stream_host if stream_index == 0 else _middle_stream_host,
            *arguments,
            options="--gpu-arch sm_90a --enable-tvm-ffi",
            no_jit_engine=True,
        )
        descriptor = dsl._compiles[compiled].descriptor
        self.assertEqual(descriptor.payload.stream.source_index, stream_index)
        self.assertEqual(descriptor.payload.stream.llvm_arg_index, stream_index)
        self.assertEqual(
            [row.name for row in descriptor.metadata.params],
            ["stream", "a", "out", "eps"]
            if stream_index == 0
            else ["a", "stream", "out", "eps"],
        )
        shape_env = ht._TraceShapeEnv()
        trace = SimpleNamespace(
            device=torch.device("cuda:0"),
            shape_env=shape_env,
            fake_mode=FakeTensorMode(shape_env=shape_env),
            cute=CuteTrace(SimpleNamespace(cuda_stream=0)),
            rec=SimpleNamespace(
                register_root=lambda *args: None, next_seq=itertools.count().__next__
            ),
            tensors=[],
        )
        x, out = (
            ht._TracedTensor(
                trace, ht._Root(name, address, 4), [17], [1], 0, torch.float32
            )
            for name, address in (("input", 4096), ("output", 8192))
        )
        nodes = (
            {
                "name": descriptor.payload.sites[0].registration.kernel_symbol,
                "func": 101,
            },
        )
        program = Program(descriptor, "test", __name__, compiled, nodes, "cuda:0", 1)
        record(
            trace,
            program,
            (x, None, out, 0.125),
            {},
            lambda value: (value, value is x),
            "test",
        )
        self.assertEqual(len(trace.cute.launches), 1)
        invocation = trace.cute.launches[0]["cute"].invocation
        self.assertEqual(len(invocation.call.arguments), stream_index)
        self.assertEqual(
            tuple(name for name, _ in invocation.call.keyword_arguments),
            ("a", "out", "eps") if stream_index == 0 else ("out", "eps"),
        )
        self.assertEqual(
            invocation.signature.operands[stream_index].origin, "environment_stream"
        )
        self.assertEqual(invocation.signature.operands[-1].scalar.use.value, 0.125)
        pointers = {
            row["name"]: row["value"]
            for row in trace.cute.launches[0]["params"]
            if row["kind"] == "ptr"
        }
        self.assertEqual(pointers, {"a.data_ptr": 4096, "out.data_ptr": 8192})
        self.assertEqual(trace.cute.written_roots, ["output"])
        self.assertFalse(torch.cuda.is_initialized())

    @parametrize("dimensions", ["static", "shared", "mismatch"])
    def test_record_uses_shared_binding_and_typed_launches(self, dimensions):
        import sympy

        from torch.cuda import _host_trace as ht
        from torch.cuda._host_trace_cute import CuteTrace

        shape_env = ht._TraceShapeEnv()
        mode = FakeTensorMode(shape_env=shape_env)
        trace = SimpleNamespace(
            device=torch.device("cuda:0"),
            shape_env=shape_env,
            fake_mode=mode,
            cute=CuteTrace(SimpleNamespace(cuda_stream=0)),
            rec=SimpleNamespace(
                register_root=lambda *args: None,
                next_seq=itertools.count().__next__,
            ),
            tensors=[],
            hints=None,
        )
        sizes = (17, 17)
        if dimensions != "static":
            rows = ht._Trace.symbol(trace, 17, "rows", positive=True)
            factor = ht._Trace.symbol(
                trace, 4 if dimensions == "shared" else 8, "factor", positive=True
            )
            sizes = (rows, rows * factor // 4)
        x, out = (
            ht._TracedTensor(
                trace, ht._Root(name, address, 4), [size], [1], 0, torch.float32
            )
            for (name, address), size in zip((("input", 4096), ("output", 8192)), sizes)
        )
        nodes = tuple(
            {"name": site.registration.kernel_symbol, "func": 101 + index}
            for index, site in enumerate(self.descriptor.payload.sites)
        )
        program = Program(
            self.descriptor, "test", __name__, self.compiled, nodes, "cuda:0", 1
        )
        if dimensions == "mismatch":
            with self.assertRaisesRegex(ht.Declined, "compiler symbol"):
                record(
                    trace,
                    program,
                    (x, None, out, 0.03125),
                    {},
                    lambda value: (value, value is x),
                    "test",
                )
            self.assertEqual(trace.cute.launches, [])
            self.assertEqual(trace.cute.written_roots, [])
            return
        record(
            trace,
            program,
            (x, None, out, 0.03125),
            {},
            lambda value: (value, value is x),
            "test",
        )
        self.assertEqual(len(trace.cute.launches), 2)
        self.assertEqual(trace.cute.written_roots, ["output"])
        for index, launch in enumerate(trace.cute.launches):
            self.assertEqual(launch["seq"], index)
            self.assertEqual(launch["grid"], (5, 1, 1))
            self.assertIs(launch["cute"].invocation.owner, program)
            self.assertIsNotNone(launch["cute"].invocation.owner_provider)
            self.assertEqual(launch["cute"].invocation.read_only, frozenset({"a"}))
            scalar = next(row for row in launch["params"] if row["name"] == "eps")
            begin = scalar["offset"]
            self.assertEqual(
                launch["hint_image"][begin : begin + 4], struct.pack("<f", 0.03125)
            )
        if dimensions == "shared":
            self.assertIn(
                sympy.Eq(sizes[1].node.expr, sizes[0].node.expr),
                [guard.expr for guard in shape_env.guards],
            )
            self.assertEqual(out._fake.shape[0].node.expr, sizes[1].node.expr)
            invocation = trace.cute.launches[0]["cute"].invocation
            self.assertEqual(
                invocation.operands[1].shape[0].node.expr, sizes[0].node.expr
            )
            self.assertIs(invocation.origins[id(invocation.operands[1])][0], out._root)
        self.assertFalse(torch.cuda.is_initialized())

    @parametrize("widths", [(32, 64), (64, 32)])
    def test_independent_shape_stride_widths(self, widths):
        shape_bits, stride_bits = widths
        fake = cute.runtime.make_fake_tensor(
            cutlass.Float32,
            (cute.sym_int(shape_bits),),
            (cute.sym_int(stride_bits, divisibility=8),),
            assumed_align=16,
        )
        compiled = cute.compile(
            _host,
            fake,
            None,
            fake,
            cutlass.Float32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--gpu-arch sm_90a --enable-tvm-ffi",
            no_jit_engine=True,
        )
        descriptor = dsl._compiles[compiled].descriptor
        program, artifact, _, _, _, _ = self.bind(
            17,
            0.125,
            descriptor=descriptor,
            owner=compiled,
            stride=8,
            lower=shape_bits == 32,
        )
        self.assertEqual(program.policy.shape_bits, shape_bits)
        self.assertEqual(program.policy.stride_bits, stride_bits)
        self.assertEqual(
            tuple(
                (row.bits, row.divisibility) for row in artifact._guards.binding.symbols
            ),
            ((shape_bits, None), (stride_bits, 8)),
        )
        with self.assertRaisesRegex(ValueError, "divisibility"):
            self.bind(17, 0.125, descriptor=descriptor, owner=compiled, stride=3)
        if shape_bits == 64:
            with self.assertRaisesRegex(NumericDeclined, "llvm.trunc"):
                self.bind(17, 0.125, descriptor=descriptor, owner=compiled, stride=8)

    @parametrize("same_registration", [False, True])
    def test_repeated_symbol_uses_ordered_occurrences(self, same_registration):
        first, second = self.descriptor.payload.sites
        registration = (
            first.registration
            if same_registration
            else second.registration._replace(
                kernel_symbol=first.registration.kernel_symbol
            )
        )
        sites = (
            first,
            second._replace(
                registration=registration,
                callee=(second.callee[0], registration.kernel_symbol),
                fields=second.fields._replace(kernel_symbol=registration.kernel_symbol),
            ),
        )
        descriptor = dataclasses.replace(
            self.descriptor, payload=self.descriptor.payload._replace(sites=sites)
        )
        loaded = dsl.loaded_program(lambda *args: None, object(), descriptor.to_json())
        sites = loaded.descriptor.payload.sites
        nodes = tuple(
            {"name": site.registration.kernel_symbol, "func": 101 + i}
            for i, site in enumerate(sites)
        )
        program = Program(
            loaded.descriptor, "test", __name__, loaded, nodes, "cuda:0", 1
        )
        captured = _captured_launches(program, sites)
        self.assertEqual(tuple(row.function for row in captured), (101, 102))
        for row, site in zip(captured, sites):
            self.assertIs(row.site, site)

    def test_conditional_equal_symbol_different_registration_declines(self):
        first, second = self.descriptor.payload.sites
        other = second.registration._replace(
            library_slot=second.registration.library_slot + 1,
            kernel_symbol=first.registration.kernel_symbol,
        )
        sites = (
            first._replace(arm=True),
            second._replace(arm=False, registration=other),
        )
        descriptor = dataclasses.replace(
            self.descriptor, payload=self.descriptor.payload._replace(sites=sites)
        )
        with self.assertRaisesRegex(Unexpressed, "different registrations"):
            descriptor.check()

    def test_opposite_observed_arm_is_rejected(self):
        first, second = self.descriptor.payload.sites
        program = Program(
            self.descriptor,
            "test",
            __name__,
            self.compiled,
            ({"name": first.registration.kernel_symbol, "func": 101},),
            "cuda:0",
            1,
        )
        with self.assertRaisesRegex(Unexpressed, "launch order"):
            _captured_launches(program, (second,))


if __name__ == "__main__":
    run_tests()
