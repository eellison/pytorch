# Owner(s): ["module: inductor"]

from dataclasses import replace
from types import SimpleNamespace

from torch._inductor.runtime._cudagraph._sdk import activate


activate()

from torch._inductor.runtime._cudagraph._compiler.argument_flow import (
    ParameterFlow,
    ValueFlow,
)
from torch._inductor.runtime._cudagraph._compiler.dispatch_join import BoundDispatchSite
from torch._inductor.runtime._cudagraph._compiler.fields_dispatch import _records
from torch._inductor.runtime._cudagraph._compiler.lowering import (
    FormalLowering,
    MetadataSlot,
)
from torch._inductor.runtime._cudagraph._compiler.mapping import ParameterBinding
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestCuTeScalarFields(TestCase):
    def fixture(self, dtype):
        formals = (
            FormalLowering(MetadataSlot((3,), "eps", "Var", 2, 2), 2, 1, dtype, dtype),
            FormalLowering(
                MetadataSlot((0,), "count", "Var", 0, 0), 0, 0, "i64", "i64"
            ),
        )
        parameters = tuple(
            ParameterFlow(
                index,
                formal.llvm_type,
                ValueFlow("argument", formal.llvm_type, formal.llvm_arg_index),
            )
            for index, formal in enumerate(formals)
        )
        bindings = tuple(
            ParameterBinding(index, index, None, formal, parameter)
            for index, (formal, parameter) in enumerate(zip(formals, parameters))
        )
        source = SimpleNamespace(callee=("module", "kernel"), launch=object())
        compiled = SimpleNamespace(
            callee=source.callee,
            source_launch=source.launch,
            parameters=bindings,
            launch=SimpleNamespace(
                index=4,
                parameters=parameters,
                registration=SimpleNamespace(kernel_symbol="kernel"),
            ),
        )
        site = BoundDispatchSite(source, compiled)
        components = SimpleNamespace(
            plan=SimpleNamespace(formals=()),
            dispatch=SimpleNamespace(
                joined=SimpleNamespace(
                    sites=(site,),
                    mapping=SimpleNamespace(formals=SimpleNamespace(formals=formals)),
                )
            ),
        )
        return components, SimpleNamespace(formals=()), site

    @parametrize("dtype", ("f32", "i32", "i64"))
    def test_scalar_fields_keep_declared_type_width_and_original_source(self, dtype):
        components, layout, site = self.fixture(dtype)
        nodes, static = _records(components, layout, site)
        (node,) = nodes
        self.assertEqual(static, ())
        self.assertEqual(node.parameter_sizes, (int(dtype[1:]) // 8, 8))
        self.assertEqual((node.launch, node.kernel_symbol), (4, "kernel"))
        self.assertEqual(
            (node.pointers, node.padding, node.constants, node.fixed), ((), (), (), ())
        )
        self.assertEqual(node.scalar_descriptors, ((0, 0, dtype), (1, 0, "i64")))
        for field, binding in zip(node.integers, site.compiled.parameters):
            self.assertEqual(field.source.kind, "scalar_formal")
            self.assertEqual(field.source.ir_arg_index, binding.formal.ir_arg_index)
            self.assertEqual(field.source.formal_name, binding.formal.metadata.name)
            self.assertEqual(field.source.metadata_path, binding.formal.metadata.path)
            self.assertEqual(
                (field.source.property, field.source.property_path), ("value", ())
            )
            self.assertIs(field.source.value, binding.parameter.source)

    @parametrize("dtype", ("f16", "f64", "i1", "i16"))
    def test_other_scalar_widths_remain_unsupported(self, dtype):
        components, layout, site = self.fixture(dtype)
        with self.assertRaisesRegex(ValueError, "Unsupported mutable scalar formal"):
            _records(components, layout, site)

    @parametrize(
        "change",
        (
            "source_index",
            "source_path",
            "arithmetic",
            "foreign_formal",
            "parameter_type",
            "parameter_identity",
            "device_index",
        ),
    )
    def test_f32_does_not_relax_exact_formal_binding(self, change):
        components, layout, site = self.fixture("f32")
        first, second = site.compiled.parameters
        parameter = first.parameter
        if change in ("source_index", "source_path", "arithmetic"):
            changes = {
                "source_index": {"argument": 0},
                "source_path": {"path": (0,)},
                "arithmetic": {
                    "kind": "fadd",
                    "operands": (parameter.source, parameter.source),
                },
            }
            parameter = replace(
                parameter, source=replace(parameter.source, **changes[change])
            )
            first = replace(first, parameter=parameter)
        elif change == "foreign_formal":
            first = replace(first, formal=replace(first.formal))
        elif change == "parameter_type":
            parameter = replace(parameter, llvm_type="i32")
            first = replace(first, parameter=parameter)
        elif change == "parameter_identity":
            first = replace(first, parameter=replace(parameter))
        else:
            first = replace(first, device_parameter_index=1)
        site.compiled.parameters = (first, second)
        site.compiled.launch.parameters = (parameter, second.parameter)
        with self.assertRaisesRegex(
            ValueError,
            "exact compiler parameter order or type|unchanged whole-formal parameters",
        ):
            _records(components, layout, site)


if __name__ == "__main__":
    run_tests()
