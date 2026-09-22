from __future__ import annotations

import io
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from torch._inductor.runtime._cudagraph._compiler.launch_events import CloneBundle


def _take_snapshot(module: Any) -> tuple[str, bytes]:
    buffer = io.BytesIO()
    module.operation.write_bytecode(buffer)
    return str(module), buffer.getvalue()


def _snapshot(module: Any) -> tuple[str, bytes]:
    from torch._inductor.runtime._cudagraph._compiler.validation_snapshots import read_snapshot

    return read_snapshot(module.operation, _take_snapshot, module)


@dataclass(frozen=True)
class CompiledProgram:
    bundle: CloneBundle = field(repr=False)
    source_module: Any = field(repr=False)
    source_context: Any = field(repr=False)
    source_artifact: Any = field(repr=False)
    source_text: str = field(repr=False)
    source_bytecode: bytes = field(repr=False)
    artifact: Any = field(repr=False)
    module: Any = field(repr=False)
    context: Any = field(repr=False)
    compiled_bytecode: bytes = field(repr=False)
    module_text: str = field(repr=False)
    module_bytecode: bytes = field(repr=False)
    function_name: str
    arch: str
    source_sha256: str
    compiled_sha256: str

    def check(self) -> None:
        import cutlass.compiler as compiler

        self.bundle.check_originals()
        if not self.source_artifact.is_consumed or self.artifact.is_consumed:
            raise RuntimeError("Unexpected compiler artifact lifecycle")
        if (type(self.artifact) is not compiler.CompiledMlirArtifact
                or self.artifact.get_bitcode() != self.compiled_bytecode
                or sha256(self.compiled_bytecode).hexdigest() != self.compiled_sha256
                or sha256(self.source_bytecode).hexdigest() != self.source_sha256):
            raise RuntimeError("Original compiled payload changed")
        for module, context, expected in (
            (self.source_module, self.source_context, (self.source_text, self.source_bytecode)),
            (self.module, self.context, (self.module_text, self.module_bytecode)),
        ):
            if module.context != context:
                raise RuntimeError("Original Module context changed")
            with context:
                if _snapshot(module) != expected:
                    raise RuntimeError("Original compiler Module changed")
