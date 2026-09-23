from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from torch._inductor.runtime._cudagraph._compiler.components import DispatchComponents
from torch._inductor.runtime._cudagraph._compiler.dispatch_join import BoundDispatchSite
from torch._inductor.runtime._cudagraph._compiler.fields import ConstantField, FieldSource, IntegerField, NodeFields, Padding, PointerField, StaticProperty, UndefinedField
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_adapter.invocation import ArtifactInvocation
from torch._inductor.runtime._cudagraph._compiler.target_layout import TargetLayout

if TYPE_CHECKING:
    from torch._inductor.runtime._cudagraph._compiler.lowering import FormalLoweringSet


def _dispatch(components):
    return components if type(components) is ArtifactInvocation else components.dispatch


def _check_owners(components: DispatchComponents | ArtifactInvocation, layout: TargetLayout) -> None:
    if type(components) not in (DispatchComponents, ArtifactInvocation) or type(layout) is not TargetLayout:
        raise TypeError("Expected actual dispatch components and their compiler-owned target layout")
    components.check()
    layout.check()
    mapping = _dispatch(components).joined.mapping
    if (layout.query.plan is not components.plan or components.plan.mapping.program is not mapping.program
            or layout.query.host_target != "aarch64-unknown-linux-gnu"):
        raise ValueError("Dispatch fields and target layout require the same compiler program and aggregate plan")
    by_llvm = {item.llvm_arg_index: item for item in mapping.formals.formals}
    tensors = {item.source_arg_index: item for item in components.plan.formals}
    targets = {item.source_arg_index: item for item in layout.formals}
    if (len(by_llvm) != len(mapping.formals.formals) or set(by_llvm) != set(range(len(mapping.flow.host_types)))
            or any(by_llvm[index].llvm_type != typ for index, typ in enumerate(mapping.flow.host_types))
            or len(tensors) != len(components.plan.formals) or len(targets) != len(layout.formals)
            or set(tensors) != set(targets)):
        raise ValueError("Dispatch formal and target layout coverage disagree")


def _records(components: DispatchComponents | ArtifactInvocation, layout: TargetLayout, site: BoundDispatchSite, physical=()):
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm

    dispatch = _dispatch(components)
    joined = dispatch.joined
    if type(site) is not BoundDispatchSite or not any(site is item for item in joined.sites):
        raise ValueError("Field descriptors require an exact owned dispatch site")
    tensors = {item.source_arg_index: item for item in components.plan.formals}
    targets = {item.source_arg_index: item for item in layout.formals}
    compiled = site.compiled
    if (compiled.callee != site.source.callee or compiled.source_launch != site.source.launch
            or compiled.launch.registration.kernel_symbol != site.source.callee[1]
            or len(compiled.parameters) != len(compiled.launch.parameters)):
        raise ValueError("Dispatch field site lost its actual registered launch")
    pointers, integers, padding, sizes, static, seen_static = [], [], [], [], [], set()
    undefined, constants = [], []
    for binding, parameter in zip(compiled.parameters, compiled.launch.parameters):
        formal, value = binding.formal, parameter.source
        if (binding.parameter is not parameter or parameter.index != len(sizes)
                or binding.device_parameter_index != parameter.index or parameter.llvm_type != value.llvm_type):
            raise ValueError("Dispatch fields lost their exact compiler parameter order or type")
        if formal is None and not (value.kind in ("constant", "zero") and value.llvm_type in ("i32", "i64")):
            record = physical[parameter.index]
            if record.parameter is not parameter:
                raise ValueError("Constructed parameter lost its exact compiled source")
            for leaf in record.fields:
                source = FieldSource("compiler_expression", None, None, (), "value", leaf.path, leaf.source)
                if leaf.kind == "pointer":
                    pointers.append(PointerField(parameter.index, leaf.byte_offset, source))
                else:
                    integers.append(IntegerField(parameter.index, leaf.byte_offset, leaf.llvm_type, source))
            padding.extend(Padding(parameter.index, offset, size) for offset, size in record.padding)
            undefined.extend(UndefinedField(parameter.index, leaf.byte_offset, leaf.byte_size, leaf.source)
                             for leaf in record.undefined)
            constants.extend(ConstantField(parameter.index, item.field.byte_offset, item.data, item.field.source)
                             for item in record.constants)
            sizes.append(record.size)
            continue
        if formal is None:
            if (value.kind not in ("constant", "zero") or value.llvm_type not in ("i32", "i64")
                    or value.argument is not None or value.path or value.operands or value.attributes
                    or value.kind == "zero" and value.value is not None):
                raise ValueError("Dispatch literal fields require exact scalar compiler constants")
            source = FieldSource("compiler_constant", None, None, (), "value", (), value)
            integers.append(IntegerField(parameter.index, 0, value.llvm_type, source))
            sizes.append(int(value.llvm_type[1:]) // 8)
            continue
        if (not any(formal is item for item in joined.mapping.formals.formals)
                or value.kind != "argument" or value.path or value.operands or value.attributes
                or value.value is not None or value.argument != formal.llvm_arg_index
                or parameter.llvm_type != formal.llvm_type or value.llvm_type != formal.llvm_type):
            raise ValueError("Dispatch fields require unchanged whole-formal parameters in actual ABI order")
        source_index = formal.ir_arg_index
        if source_index not in tensors:
            if formal.metadata.kind != "Var" or value.llvm_type not in ("i32", "i64", "f32"):
                raise ValueError("Unsupported mutable scalar formal; only exact i32/i64/f32 fields are admitted")
            source = FieldSource("scalar_formal", source_index, formal.metadata.name, formal.metadata.path,
                                 "value", (), value)
            integers.append(IntegerField(parameter.index, 0, value.llvm_type, source))
            sizes.append(int(value.llvm_type[1:]) // 8)
            continue
        tensor, target = tensors[source_index], targets[source_index]
        leaves = {item.path: item for item in tensor.leaves}
        fields = {item.path: item for item in target.fields}
        if (formal.metadata.kind != "Tensor" or tensor.llvm_arg_index != formal.llvm_arg_index
                or target.llvm_arg_index != formal.llvm_arg_index or tensor.llvm_type != formal.llvm_type
                or target.llvm_type != formal.llvm_type or len(leaves) != len(tensor.leaves)
                or len(fields) != len(target.fields) or set(leaves) != set(fields)):
            raise ValueError("Tensor fields do not cover this exact forwarded compiler aggregate")
        intervals = []
        for target_field in target.fields:
            leaf = leaves[target_field.path]
            spec = leaf.component.spec
            if (target_field.llvm_type != leaf.llvm_type or spec.source_arg_index != source_index
                    or leaf.component.source.argument != source_index or leaf.component.source.path != leaf.path):
                raise ValueError("Physical leaf lost its exact source property or LLVM type")
            source = FieldSource("tensor_property", source_index, formal.metadata.name, formal.metadata.path,
                                 spec.property, spec.property_path, leaf.component.source)
            if spec.property == "pointer":
                with dispatch.program.context:
                    pointer_type = ir.Type.parse(leaf.llvm_type)
                if spec.property_path or not isinstance(pointer_type, llvm.PointerType) or target_field.size != 8:
                    raise ValueError("Native pointer fields require an exact compiler-sized 8-byte effective pointer")
                pointers.append(PointerField(parameter.index, target_field.offset, source))
            elif spec.property in ("shape", "stride") and leaf.llvm_type in ("i32", "i64"):
                if target_field.size != int(leaf.llvm_type[1:]) // 8:
                    raise ValueError("Native integer width differs from the compiler field size")
                integers.append(IntegerField(parameter.index, target_field.offset, leaf.llvm_type, source))
            else:
                raise ValueError("Unsupported mutable tensor property or integer width")
            intervals.append((target_field.offset, target_field.offset + target_field.size))
        cursor = 0
        for begin, end in sorted(intervals):
            if begin < cursor or end > target.size:
                raise ValueError("Native fields overlap or exceed the actual parameter")
            if begin > cursor:
                padding.append(Padding(parameter.index, cursor, begin - cursor))
            cursor = end
        if cursor < target.size:
            padding.append(Padding(parameter.index, cursor, target.size - cursor))
        sizes.append(target.size)
        if source_index not in seen_static:
            for component in tensor.constants:
                spec = component.spec
                static.append(StaticProperty(source_index, formal.metadata.name, spec.property,
                                             spec.property_path, component.source))
            seen_static.add(source_index)
    node = NodeFields(compiled.launch.index, compiled.launch.registration.kernel_symbol, tuple(sizes),
                      tuple(pointers), tuple(integers), (), tuple(padding), tuple(undefined), tuple(constants))
    return (node,), tuple(static)


@dataclass(frozen=True)
class DispatchSiteFields:
    components: DispatchComponents | ArtifactInvocation = field(repr=False)
    site: BoundDispatchSite = field(repr=False)
    formals: FormalLoweringSet = field(repr=False)
    layout: TargetLayout = field(repr=False)
    physical: tuple
    nodes: tuple[NodeFields, ...]
    static_properties: tuple[StaticProperty, ...]
    _owners: tuple[Any, ...] = field(repr=False)

    def check(self) -> None:
        owned = self.components, self.site, self.formals, self.layout, self.physical, self.nodes, self.static_properties
        if len(owned) != len(self._owners) or any(value is not owner for value, owner in zip(owned, self._owners)):
            raise RuntimeError("Dispatch field descriptor ownership changed")
        _check_owners(self.components, self.layout)
        if self.formals is not _dispatch(self.components).joined.mapping.formals:
            raise ValueError("Dispatch fields lost their actual formal owner")
        nodes, static = _records(self.components, self.layout, self.site, self.physical)
        if (nodes, static) != (self.nodes, self.static_properties):
            raise RuntimeError("Dispatch fields changed their exact compiler-derived source, offset or padding")
        actual_sources = tuple(item.source.value for node in self.nodes for item in (*node.pointers, *node.integers))
        expected_sources = tuple(item.source.value for node in nodes for item in (*node.pointers, *node.integers))
        if (any(actual is not expected for actual, expected in zip(actual_sources, expected_sources))
                or any(actual.value is not expected.value for actual, expected in zip(self.static_properties, static))):
            raise RuntimeError("Dispatch fields lost the original component expression identity")


@dataclass(frozen=True)
class DispatchFieldPlan:
    components: DispatchComponents | ArtifactInvocation = field(repr=False)
    layout: TargetLayout = field(repr=False)
    sites: tuple[DispatchSiteFields, ...]
    _owners: tuple[Any, ...] = field(repr=False)

    def check(self) -> None:
        owned = self.components, self.layout, self.sites
        if len(owned) != len(self._owners) or any(value is not owner for value, owner in zip(owned, self._owners)):
            raise RuntimeError("Dispatch field plan ownership changed")
        if len(self.sites) != len(_dispatch(self.components).joined.sites):
            raise ValueError("Dispatch field plan does not cover every actual site")
        for fields, site in zip(self.sites, _dispatch(self.components).joined.sites):
            if fields.components is not self.components or fields.layout is not self.layout or fields.site is not site:
                raise ValueError("Dispatch field plans lost their exact admitted sites")
            fields.check()


def derive_dispatch_fields(components: DispatchComponents | ArtifactInvocation, layout: TargetLayout) -> DispatchFieldPlan:
    """Describe exact parameter updates; capture bytes and runtime eligibility remain external."""
    _check_owners(components, layout)
    sites = []
    dispatch = _dispatch(components)
    for site in dispatch.joined.sites:
        physical = ()
        if any(binding.formal is None and not (binding.parameter.source.kind in ("constant", "zero")
                   and binding.parameter.llvm_type in ("i32", "i64")) for binding in site.compiled.parameters):
            from torch._inductor.runtime._cudagraph._compiler.compiler_type_layout import compile_type_layouts
            from cutlass._mlir import ir
            from torch._inductor.runtime._cudagraph._compiler.physical_parameters import project_parameters

            parameters = site.compiled.launch.parameters
            with dispatch.program.context:
                types = tuple(ir.Type.parse(parameter.llvm_type) for parameter in parameters)
                compiled = compile_type_layouts(types, device_target=dispatch.program.arch)
                physical = project_parameters(parameters, compiled)
        nodes, static = _records(components, layout, site, physical)
        formals = dispatch.joined.mapping.formals
        owners = components, site, formals, layout, physical, nodes, static
        sites.append(DispatchSiteFields(*owners, owners))
    sites = tuple(sites)
    return DispatchFieldPlan(components, layout, sites, (components, layout, sites))
