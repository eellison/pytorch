"""The existing typed reader applied to an intercepted SDK compilation."""

from dataclasses import dataclass, field, replace
from hashlib import sha256

from ..accessors import analyze_accessors
from ..aggregate_plan import build_aggregate_plan
from ..compiler_boundary import _snapshot
from ..continuation import bind_dispatch_consumers, check_dispatch_consumers
from ..dispatch_join import join_dispatch
from ..entry_signature import snapshot_metadata
from ..lowering import bind_host_formals
from ..mapping import bind_dispatch_sites
from ..validation_snapshots import validation_snapshots
from .owner import _OrdinaryCapture


@dataclass(frozen=True, eq=False)
class CapturedProgram:
    selected: object = field(repr=False)
    capture: _OrdinaryCapture = field(repr=False)
    module: object = field(repr=False)
    module_snapshot: tuple = field(repr=False)

    @property
    def source_metadata(self):
        return (self.capture.function_metadata,)

    @property
    def source_module(self):
        return self.capture.module

    @property
    def source_context(self):
        return self.capture.context

    @property
    def context(self):
        return self.module.context

    @property
    def function_name(self):
        return self.capture.name

    @property
    def arch(self):
        return self.capture.helpers.source.sites[0].diagnostics[-1].arch

    def check(self):
        capture = self.capture
        if (
            type(capture) is not _OrdinaryCapture
            or capture.calls != 1
            or capture.decline is not None
            or capture.module is None
            or self.selected.ir_module is not self.module
            or self.selected.function_name != capture.name
            or snapshot_metadata(capture.function_metadata) != capture.metadata
            or _snapshot(capture.module) != (capture.text, capture.bytecode)
            or _snapshot(self.module) != self.module_snapshot
        ):
            raise RuntimeError(
                "Captured compilation changed its compiler owner or snapshots"
            )
        capture.helpers.check()
        for accessor in capture.accessors:
            accessor.check()
        if any(
            site.diagnostics[-1].arch != self.arch
            for site in capture.helpers.source.sites
        ):
            raise ValueError("Captured sites disagree on the compiler target")


@dataclass(frozen=True, eq=False)
class CapturedCorrespondence:
    program: CapturedProgram
    formals: object
    joined: object
    mapping: object
    plan: object
    consumers: tuple
    _owners: tuple = field(repr=False)

    @property
    def metadata(self):
        return self.program.capture.metadata

    @property
    def source_sha256(self):
        return sha256(self.program.capture.bytecode).hexdigest()

    @property
    def compiled_sha256(self):
        return sha256(self.program.module_snapshot[1]).hexdigest()

    def _state(self):
        return (
            self.program,
            self.formals,
            self.joined,
            self.mapping,
            self.plan,
            self.consumers,
        )

    def check(self):
        if len(self._owners) != 6 or any(
            a is not b for a, b in zip(self._state(), self._owners)
        ):
            raise RuntimeError("Captured correspondence owners changed")
        program = self.program
        with validation_snapshots(program.source_module, program.module):
            program.check()
            self.formals.check()
            self.joined.check()
            self.mapping.check()
            self.plan.check()
            if (
                self.formals.program is not program
                or self.formals.metadata is not program.capture.function_metadata
                or self.joined.mapping.program is not program
                or self.joined.mapping.formals is not self.formals
                or self.mapping.program is not program
                or self.mapping.specs is not program.capture.accessors
                or self.plan.mapping is not self.mapping
                or len(self.consumers) != len(program.capture.helpers.helpers)
            ):
                raise RuntimeError("Captured correspondence mixed compiler generations")
            for consumer, helper in zip(
                self.consumers, program.capture.helpers.helpers
            ):
                if consumer.joined is not self.joined or consumer.helper is not helper:
                    raise RuntimeError("Captured consumer lost its emitted helper")
            check_dispatch_consumers(self.joined, self.consumers)


def read_captured_compilation(selected, capture):
    if type(capture) is not _OrdinaryCapture or capture.decline is not None:
        raise ValueError(
            f"Compiler source capture declined: {getattr(capture, 'decline', None)}"
        )
    program = CapturedProgram(
        selected, capture, selected.ir_module, _snapshot(selected.ir_module)
    )
    program.check()
    formals = bind_host_formals(
        program, capture.function_metadata, capture.helpers.source.source_types
    )
    joined = join_dispatch(bind_dispatch_sites(program, formals))
    mapping = analyze_accessors(program, capture.accessors)
    plan = build_aggregate_plan(mapping)
    consumers = bind_dispatch_consumers(joined, capture.helpers)
    result = CapturedCorrespondence(
        program, formals, joined, mapping, plan, consumers, ()
    )
    result = replace(result, _owners=result._state())
    result.check()
    return result


def capture_dispatch_payload(selected, capture):
    from ..cudagraph_cute_runtime.factory import prepare_dispatch_payload
    from ..ordinary_artifact_adapter.invocation import normalize_artifact_invocation
    from ..stream_layout import compile_stream_layout
    from ..target_layout import compile_target_layout

    invocation = normalize_artifact_invocation(
        read_captured_compilation(selected, capture)
    )
    layout = compile_target_layout(invocation.plan)
    stream_layout = compile_stream_layout(
        invocation.formals, invocation.stream_source_index
    )
    return prepare_dispatch_payload(invocation, layout, stream_layout)
