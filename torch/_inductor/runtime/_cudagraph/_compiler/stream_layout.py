from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from torch._inductor.runtime._cudagraph._compiler.elf_table import ELFDATA2LSB, ELFCLASS64, EM_AARCH64, read_i64_table
from torch._inductor.runtime._cudagraph._compiler.llvm_query import _type
from torch._inductor.runtime._cudagraph._compiler.lowering import FormalLoweringSet


_HOST_TARGET = "aarch64-unknown-linux-gnu"
_TABLE = "cudagraph_stream_layout"


def _stream_type(formals: FormalLoweringSet, source_index: int) -> tuple[str, str]:
    from torch._inductor.runtime._cudagraph._compiler.accessors import _function, _tagged_arguments
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm

    if type(formals) is not FormalLoweringSet or type(source_index) is not int or source_index < 0:
        raise TypeError("Expected original bound formals and a nonnegative source index")
    formals.check()
    found = [item for item in formals.formals if item.ir_arg_index == source_index]
    if (len(found) != 1 or found[0].metadata.kind not in ("Stream", "EnvStream")
            or found[0].source_type != "!cuda.stream"):
        raise ValueError("Requested source index is not the original stream formal")
    formal = found[0]
    program = formals.program
    with program.context, ir.raw_values():
        host = _function(program.module, program.function_name, "llvm.func")
        arguments = _tagged_arguments(host)
        if (formal.llvm_arg_index >= len(arguments) or tuple(arguments)[formal.llvm_arg_index] != source_index
                or source_index not in arguments or str(arguments[source_index].type) != formal.llvm_type):
            raise RuntimeError("Stream layout lost its actual compiler-preserved source formal")
        typ = arguments[source_index].type
        if not isinstance(typ, llvm.PointerType):
            raise ValueError("The original stream formal must have a pointer LLVM type")
        result = str(typ), _type(typ)
    formals.check()
    return result


def _query_text(llvm_type: str) -> str:
    return (f'target triple = "{_HOST_TARGET}"\n'
            f"@{_TABLE} = constant [2 x i64] [\n"
            f"  i64 ptrtoint (ptr getelementptr ({llvm_type}, ptr null, i64 1) to i64),\n"
            f"  i64 ptrtoint (ptr getelementptr ({{ i8, {llvm_type} }}, ptr null, i64 0, i32 1) to i64)\n]\n")


def _read(data: bytes) -> tuple[int, int]:
    result = read_i64_table(data, _TABLE, 2, expected_machine=EM_AARCH64,
                           expected_class=ELFCLASS64, expected_endianness=ELFDATA2LSB).values
    size, alignment = result
    if size <= 0 or alignment <= 0 or alignment & (alignment - 1) or size % alignment:
        raise ValueError("Invalid compiler stream-pointer size or ABI alignment")
    return size, alignment


@dataclass(frozen=True)
class StreamLayout:
    formals: FormalLoweringSet = field(repr=False)
    source_index: int
    llvm_type: str
    size: int
    alignment: int
    host_target: str
    source: Any = field(repr=False)
    artifact: Any = field(repr=False)
    llvm_text: str = field(repr=False)
    object_bytes: bytes = field(repr=False)
    object_sha256: str
    _owners: tuple[Any, ...] = field(repr=False)

    def check(self) -> None:
        import cutlass.compiler as compiler

        owned = (self.formals, self.source_index, self.llvm_type, self.source, self.artifact,
                 self.llvm_text, self.object_bytes)
        if len(owned) != len(self._owners) or any(value is not owner for value, owner in zip(owned, self._owners)):
            raise RuntimeError("Stream layout source or query artifact ownership changed")
        actual_type, spelling = _stream_type(self.formals, self.source_index)
        if (actual_type != self.llvm_type or self.host_target != _HOST_TARGET
                or _query_text(spelling) != self.llvm_text or type(self.source) is not compiler.LlvmIrArtifact
                or not self.source.is_consumed or type(self.artifact) is not compiler.ObjectArtifact
                or self.artifact.is_consumed or len(self.artifact.metadata)
                or self.artifact.get_data() != self.object_bytes
                or sha256(self.object_bytes).hexdigest() != self.object_sha256):
            raise RuntimeError("Stream layout query differs from the exact original formal or object")
        if type(self.size) is not int or type(self.alignment) is not int or _read(self.object_bytes) != (self.size, self.alignment):
            raise RuntimeError("Stream layout fields differ from the compiler table")


def compile_stream_layout(bound_formals: FormalLoweringSet, stream_source_index: int) -> StreamLayout:
    import cutlass.compiler as compiler

    llvm_type, spelling = _stream_type(bound_formals, stream_source_index)
    if compiler.Compiler.detect_host_triple() != _HOST_TARGET:
        raise ValueError("Stream layout currently requires the inspected AArch64 Linux host target")
    llvm_text = _query_text(spelling)
    source = compiler.LlvmIrArtifact.from_textual_form(llvm_text.encode())
    if type(source) is not compiler.LlvmIrArtifact or source.is_consumed or len(source.metadata):
        raise RuntimeError("Expected a data-only LLVM stream layout artifact")
    native = compiler.CuteCompiler()
    native.set_device_target(bound_formals.program.arch)
    native.set_host_target(_HOST_TARGET)
    native.set_abi(compiler.Abi.Tbd)
    artifact = native.compile_to(source, compiler.ArtifactType.Object)
    if not source.is_consumed or type(artifact) is not compiler.ObjectArtifact or artifact.is_consumed:
        raise RuntimeError("Expected the exact compiler-owned stream layout object")
    data = artifact.get_data()
    size, alignment = _read(data)
    owners = bound_formals, stream_source_index, llvm_type, source, artifact, llvm_text, data
    result = StreamLayout(bound_formals, stream_source_index, llvm_type, size, alignment, _HOST_TARGET,
                          source, artifact, llvm_text, data, sha256(data).hexdigest(), owners)
    result.check()
    return result
