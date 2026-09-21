from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from torch._inductor.runtime._cudagraph._compiler.entry_signature import EntrySignature


@dataclass(frozen=True)
class JoinedValue:
    kind: str
    llvm_type: str
    source: Any = field(default=None, repr=False)
    path: tuple[int, ...] = ()
    value: str | None = None
    operands: tuple[JoinedValue, ...] = ()
    attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class JoinedLaunch:
    index: int
    registration: Any
    parameters: tuple[JoinedValue, ...]


def _compose(bound: Any, lowering: Any, flow: Any) -> tuple[JoinedLaunch, ...]:
    by_source = {parameter.metadata.ir_arg_index: parameter.source for parameter in bound.params
                 if parameter.metadata.ir_arg_index is not None}
    by_llvm = {formal.llvm_arg_index: by_source[formal.ir_arg_index] for formal in lowering.formals}

    def translate(value: Any) -> JoinedValue:
        if value.kind == "argument":
            if value.argument not in by_llvm:
                raise ValueError("LLVM argument has no original Python entry source")
            return JoinedValue("source", value.llvm_type, by_llvm[value.argument], path=value.path)
        return JoinedValue(value.kind, value.llvm_type, path=value.path, value=value.value,
                           operands=tuple(translate(item) for item in value.operands), attributes=value.attributes)

    return tuple(JoinedLaunch(launch.index, launch.registration,
                              tuple(translate(parameter.source) for parameter in launch.parameters))
                 for launch in flow.launches)


@dataclass(frozen=True)
class JoinedInvocation:
    signature: EntrySignature = field(repr=False)
    program: Any = field(repr=False)
    bound: Any = field(repr=False)
    lowering: Any = field(repr=False)
    flow: Any = field(repr=False)
    launches: tuple[JoinedLaunch, ...]
    _owners: tuple[Any, ...] = field(repr=False)

    def check(self) -> None:
        values = self.signature, self.program, self.bound, self.lowering, self.flow
        if any(value is not owner for value, owner in zip(values, self._owners)):
            raise RuntimeError("Joined source ownership changed")
        self.signature.check()
        self.program.check()
        self.lowering.check()
        self.flow.check()
        actual = self.signature.bind_metadata(self.program.source_metadata[0])
        if (self.program.bundle.original_host is not self.signature.target
                or self.lowering.program is not self.program
                or self.flow.module is not self.program.module
                or self.flow.function_name != self.program.function_name
                or actual.metadata != self.bound.metadata
                or _compose(self.bound, self.lowering, self.flow) != self.launches):
            raise RuntimeError("Joined Python-to-kernel argument correspondence changed")
