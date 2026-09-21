from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from torch._inductor.runtime._cudagraph._compiler.entry_signature import snapshot_metadata


if TYPE_CHECKING:
    from torch._inductor.runtime._cudagraph._compiler.compiler_owner import TaggedProgram


@dataclass(frozen=True)
class ComponentCompilation:
    program: TaggedProgram
    accessors: tuple[Any, ...]
    metadata: Any = field(repr=False)
    _owners: tuple[Any, ...] = field(repr=False)

    def check(self) -> None:
        if any(value is not owner for value, owner in zip((self.program, self.accessors, self.metadata), self._owners)):
            raise RuntimeError("Component compilation association changed")
        self.program.check()
        expected = snapshot_metadata(self.metadata)
        actual = snapshot_metadata(self.program.source_metadata[0])
        if expected.params != actual.params or expected.symbols != actual.symbols or expected.ret != actual.ret:
            raise RuntimeError("Accessor signature differs from the original compiler metadata")
