"""Join an ordinary compilation to traced properties using the existing IR readers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from torch._inductor.runtime._cudagraph._compiler.accessors import analyze_accessors
from torch._inductor.runtime._cudagraph._compiler.aggregate_plan import build_aggregate_plan
from torch._inductor.runtime._cudagraph._compiler.continuation import BoundDispatchHelper, bind_dispatch_consumers, check_dispatch_consumers
from torch._inductor.runtime._cudagraph._compiler.dispatch_join import join_dispatch
from torch._inductor.runtime._cudagraph._compiler.join_components import SymbolicComponent
from torch._inductor.runtime._cudagraph._compiler.joined import _compose
from torch._inductor.runtime._cudagraph._compiler.lowering import bind_host_formals
from torch._inductor.runtime._cudagraph._compiler.mapping import bind_dispatch_sites
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner import OrdinaryCompilation
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_reuse.binding import OrdinaryBinding


@dataclass(frozen=True, eq=False)
class _OrdinaryProgram:
    compilation: OrdinaryCompilation = field(repr=False)
    _owner: OrdinaryCompilation = field(repr=False)

    @property
    def source_metadata(self):
        return (self.compilation.function_metadata,)

    @property
    def source_module(self):
        return self.compilation.source_module

    @property
    def source_context(self):
        return self.compilation.source_context

    @property
    def module(self):
        return self.compilation.module

    @property
    def context(self):
        return self.compilation.context

    @property
    def function_name(self):
        return self.compilation.function_name

    @property
    def arch(self):
        return self.compilation.helpers.source.sites[0].diagnostics[-1].arch

    def check(self):
        if type(self.compilation) is not OrdinaryCompilation or self.compilation is not self._owner:
            raise RuntimeError("Ordinary reader program lost its exact compilation owner")
        self.compilation.check()
        if type(self.arch) is not str or any(
                site.diagnostics[-1].arch != self.arch for site in self.compilation.helpers.source.sites):
            raise ValueError("Ordinary source sites disagree on their compiler target")


def _properties(bound, mapping):
    by_source = {row.metadata.ir_arg_index: row.source for row in bound.params}
    result = []
    for component in mapping.components:
        spec = component.spec
        source = by_source.get(spec.source_arg_index)
        if source is None or source.tensor is None:
            raise ValueError("Ordinary component has no original Tensor operand")
        if spec.property == "pointer" and not spec.property_path:
            use = None
        elif spec.property in ("shape", "stride") and len(spec.property_path) == 1:
            uses = source.tensor.shape if spec.property == "shape" else source.tensor.strides
            index, = spec.property_path
            if type(index) is not int or not 0 <= index < len(uses):
                raise ValueError("Ordinary component is outside its original Tensor property")
            use = uses[index]
        else:
            raise ValueError("Unsupported ordinary Tensor component property")
        result.append(SymbolicComponent(component, source, use))
    return tuple(result)


@dataclass(frozen=True, eq=False)
class OrdinaryCorrespondence:
    bound: OrdinaryBinding
    program: _OrdinaryProgram = field(repr=False)
    formals: object
    joined: object
    mapping: object
    plan: object
    consumers: tuple
    launches: tuple
    properties: tuple
    _owners: tuple = field(repr=False)

    def _state(self):
        return (self.bound, self.program, self.formals, self.joined, self.mapping, self.plan,
                self.consumers, self.launches, self.properties)

    def check(self):
        from torch._inductor.runtime._cudagraph._compiler.validation_snapshots import validation_snapshots

        state = self._state()
        owners = self._owners
        if len(state) != len(owners) or any(value is not old for value, old in zip(state, owners)):
            raise RuntimeError("Ordinary correspondence owners changed")
        compilation = self.bound.compilation
        with validation_snapshots(compilation.source_module, compilation.module):
            self._check(state, owners)

    def _check(self, state, owners):
        self.bound.check()
        self.program.check()
        self.formals.check()
        self.joined.check()
        self.mapping.check()
        self.plan.check()
        compilation = self.bound.compilation
        if (self.program.compilation is not compilation or self.formals.program is not self.program
                or self.formals.metadata is not compilation.function_metadata
                or self.joined.mapping.program is not self.program
                or self.joined.mapping.formals is not self.formals
                or self.mapping.program is not self.program or self.mapping.specs is not compilation.accessors
                or self.plan.mapping is not self.mapping
                or len(self.consumers) != len(compilation.helpers.helpers)):
            raise RuntimeError("Ordinary correspondence mixed compiler generations")
        consumers, helpers, joined = self.consumers, compilation.helpers.helpers, self.joined
        if type(consumers) is not tuple or any(type(consumer) is not BoundDispatchHelper for consumer in consumers):
            raise TypeError("Expected the exact ordinary dispatch consumer tuple")
        for consumer, helper in zip(consumers, helpers):
            if consumer.joined is not joined or consumer.helper is not helper:
                raise RuntimeError("Ordinary launch consumer lost its emitted helper")
        check_dispatch_consumers(joined, consumers)
        if _compose(self.bound, self.formals, self.joined.mapping.flow) != self.launches:
            raise RuntimeError("Ordinary kernel parameters lost their original operands")
        actual = _properties(self.bound, self.mapping)
        if len(actual) != len(self.properties) or any(
                row.component is not old.component or row.source is not old.source or row.use is not old.use
                for row, old in zip(actual, self.properties)):
            raise RuntimeError("Ordinary aggregate fields lost their original symbolic properties")
        if (self._owners is not owners or any(value is not old for value, old in zip(self._state(), state))
                or compilation.helpers.helpers is not helpers):
            raise RuntimeError("Ordinary correspondence owners changed")


def join_ordinary_correspondence(bound: OrdinaryBinding) -> OrdinaryCorrespondence:
    if type(bound) is not OrdinaryBinding:
        raise TypeError("Expected an exact ordinary metadata binding")
    bound.check()
    compilation = bound.compilation
    program = _OrdinaryProgram(compilation, compilation)
    formals = bind_host_formals(program, compilation.function_metadata, compilation.helpers.source.source_types)
    joined = join_dispatch(bind_dispatch_sites(program, formals))
    mapping = analyze_accessors(program, compilation.accessors)
    plan = build_aggregate_plan(mapping)
    consumers = bind_dispatch_consumers(joined, compilation.helpers)
    launches = _compose(bound, formals, joined.mapping.flow)
    result = OrdinaryCorrespondence(bound, program, formals, joined, mapping, plan, consumers,
                                    launches, _properties(bound, mapping), ())
    result = replace(result, _owners=result._state())
    result.check()
    return result
