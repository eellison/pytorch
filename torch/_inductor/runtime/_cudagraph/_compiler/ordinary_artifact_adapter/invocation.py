"""Checked normalization of the two supported CuTe compilation authorities."""

from hashlib import sha256
from typing import Any, NamedTuple


class _Invocation(NamedTuple):
    authority: Any
    signature: Any
    bound: Any
    program: Any
    metadata: Any
    formals: Any
    joined: Any
    plan: Any
    properties: tuple
    launches: tuple
    consumers: tuple
    kind: str
    source_sha256: str
    compiled_sha256: str
    stream_source_index: int


def _project(authority):
    from torch._inductor.runtime._cudagraph._compiler.entry import DispatchEntry
    from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_reuse.correspondence import OrdinaryCorrespondence

    if type(authority) is DispatchEntry:
        authority.check()
        original = authority.joined.invocation
        dispatch = authority.components.dispatch
        signature, bound, program = original.signature, original.bound, original.program
        formals, joined, plan = original.lowering, dispatch.joined, authority.components.plan
        properties, launches, consumers = authority.joined.properties, original.launches, dispatch.consumers
        kind, stream_kind = "retained", "EnvStream"
        source_hash, compiled_hash = program.source_sha256, program.compiled_sha256
        if program.bundle.original_host is not signature.target:
            raise ValueError("Retained artifact lost its original Python target")
        actual = signature.bind_metadata(program.source_metadata[0])
        if (actual.metadata != bound.metadata or len(actual.params) != len(bound.params)
                or any(row.source is not old.source or row.metadata != old.metadata
                       for row, old in zip(actual.params, bound.params))):
            raise ValueError("Retained artifact changed its exact metadata binding")
    elif type(authority) is OrdinaryCorrespondence:
        authority.check()
        bound, program = authority.bound, authority.program
        signature = bound.signature
        formals, joined, plan = authority.formals, authority.joined, authority.plan
        properties, launches, consumers = authority.properties, authority.launches, authority.consumers
        kind, stream_kind = "ordinary", "Stream"
        compilation = bound.compilation
        source_hash = sha256(compilation.source_bytecode).hexdigest()
        compiled_hash = sha256(compilation.module_bytecode).hexdigest()
    else:
        raise TypeError("Artifact invocation requires an exact DispatchEntry or OrdinaryCorrespondence")
    metadata = bound.metadata
    if (formals.program is not program or joined.mapping.program is not program
            or joined.mapping.formals is not formals or plan.mapping.program is not program
            or joined.mapping.flow.module is not program.module
            or metadata != joined.admission.source.metadata
            or len(bound.params) != len(signature.operands)
            or any(row.source is not operand for row, operand in zip(bound.params, signature.operands))):
        raise ValueError("Artifact invocation mixed source, operand or compiler owners")
    streams = [row for row in metadata.params if row.kind == stream_kind]
    if (len(streams) != 1 or type(streams[0].ir_arg_index) is not int
            or any(site.source.stream_source_arg_index != streams[0].ir_arg_index for site in joined.sites)):
        raise ValueError("Artifact invocation lacks one exact original stream formal")
    return _Invocation(authority, signature, bound, program, metadata, formals, joined, plan,
                       properties, launches, consumers, kind, source_hash, compiled_hash, streams[0].ir_arg_index)


class ArtifactInvocation:
    __slots__ = ("_record", "_seal")

    def __new__(cls):
        raise TypeError("ArtifactInvocation must be created by normalize_artifact_invocation")

    def __setattr__(self, name, value):
        raise AttributeError("Artifact invocation ownership is immutable")

    @property
    def authority(self):
        return self._record.authority

    @property
    def signature(self):
        return self._record.signature

    @property
    def bound(self):
        return self._record.bound

    @property
    def program(self):
        return self._record.program

    @property
    def metadata(self):
        return self._record.metadata

    @property
    def formals(self):
        return self._record.formals

    @property
    def joined(self):
        return self._record.joined

    @property
    def plan(self):
        return self._record.plan

    @property
    def properties(self):
        return self._record.properties

    @property
    def launches(self):
        return self._record.launches

    @property
    def consumers(self):
        return self._record.consumers

    @property
    def kind(self):
        return self._record.kind

    @property
    def source_sha256(self):
        return self._record.source_sha256

    @property
    def compiled_sha256(self):
        return self._record.compiled_sha256

    @property
    def stream_source_index(self):
        return self._record.stream_source_index

    def check(self):
        if (type(self) is not ArtifactInvocation or type(self._record) is not _Invocation
                or type(self._seal) is not tuple or len(self._seal) != 2
                or self._seal[0] is not self or self._seal[1] is not self._record):
            raise RuntimeError("Artifact invocation ownership changed")
        actual = _project(self.authority)
        if (any(value is not old for value, old in zip(actual[:11], self._record[:11]))
                or actual[11:] != self._record[11:]):
            raise RuntimeError("Artifact invocation fields changed their original authority")

    def check_layout(self, layout, stream_layout):
        from torch._inductor.runtime._cudagraph._compiler.stream_layout import StreamLayout
        from torch._inductor.runtime._cudagraph._compiler.target_layout import TargetLayout

        self.check()
        if type(layout) is not TargetLayout or type(stream_layout) is not StreamLayout:
            raise TypeError("Expected the exact compiler target and stream layouts")
        layout.check()
        stream_layout.check()
        if (layout.query.plan is not self.plan or stream_layout.formals is not self.formals
                or stream_layout.source_index != self.stream_source_index
                or layout.query.host_target != stream_layout.host_target):
            raise ValueError("Artifact layouts belong to another original compiler invocation")


def normalize_artifact_invocation(authority) -> ArtifactInvocation:
    if type(authority) is ArtifactInvocation:
        authority.check()
        return authority
    record = _project(authority)
    result = object.__new__(ArtifactInvocation)
    object.__setattr__(result, "_record", record)
    object.__setattr__(result, "_seal", (result, record))
    result.check()
    return result
