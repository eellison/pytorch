from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from torch._inductor.runtime._cudagraph._compiler.elf_table import ELFDATA2LSB, ELFCLASS64, EM_AARCH64, read_i64_table
from torch._inductor.runtime._cudagraph._compiler.layout_ir import emit_layout_ir, LayoutIR
from torch._inductor.runtime._cudagraph._compiler.llvm_query import emit_llvm_query


@dataclass(frozen=True)
class FieldLayout:
    path: tuple[int, ...]
    llvm_type: str
    offset: int
    size: int
    alignment: int


@dataclass(frozen=True)
class FormalLayout:
    source_arg_index: int
    llvm_arg_index: int
    llvm_type: str
    size: int
    alignment: int
    fields: tuple[FieldLayout, ...]


def _layouts(query: LayoutIR, values: tuple[int, ...]) -> tuple[FormalLayout, ...]:
    if len(values) != len(query.rows):
        raise ValueError("Compiler layout table has the wrong row count")
    table = {(row.source_arg_index, row.kind, row.path): value for row, value in zip(query.rows, values)}
    if len(table) != len(values) or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Compiler layout table has duplicate or negative entries")
    result = []
    for formal in query.plan.formals:
        index = formal.source_arg_index
        size, alignment = table[index, "allocation_size", ()], table[index, "abi_alignment", ()]
        if size <= 0 or alignment <= 0 or alignment & (alignment - 1) or size % alignment:
            raise ValueError("Unsupported compiler aggregate size or alignment")
        fields = tuple(FieldLayout(
            leaf.path, leaf.llvm_type, table[index, "offset", leaf.path],
            table[index, "leaf_allocation_size", leaf.path], table[index, "leaf_abi_alignment", leaf.path],
        ) for leaf in formal.leaves)
        end = 0
        for item in sorted(fields, key=lambda item: item.offset):
            if (item.size <= 0 or item.alignment <= 0 or item.alignment & (item.alignment - 1)
                    or item.offset < end or item.offset + item.size > size):
                raise ValueError("Compiler component layout overlaps or exceeds its aggregate")
            end = item.offset + item.size
        result.append(FormalLayout(index, formal.llvm_arg_index, formal.llvm_type, size, alignment, fields))
    return tuple(result)


@dataclass(frozen=True)
class TargetLayout:
    query: LayoutIR = field(repr=False)
    source: Any = field(repr=False)
    artifact: Any = field(repr=False)
    llvm_text: str = field(repr=False)
    object_bytes: bytes = field(repr=False)
    object_sha256: str
    values: tuple[int, ...]
    formals: tuple[FormalLayout, ...]
    _owners: tuple[Any, ...] = field(repr=False)

    def check(self) -> None:
        import cutlass.compiler as compiler

        owned = self.query, self.source, self.artifact, self.llvm_text, self.object_bytes, self.values, self.formals
        if len(owned) != len(self._owners) or any(value is not owner for value, owner in zip(owned, self._owners)):
            raise RuntimeError("Compiler layout ownership changed")
        self.query.check()
        if (emit_llvm_query(self.query) != self.llvm_text
                or not self.source.is_consumed or type(self.artifact) is not compiler.ObjectArtifact
                or self.artifact.is_consumed or len(self.artifact.metadata)
                or self.artifact.get_data() != self.object_bytes
                or sha256(self.object_bytes).hexdigest() != self.object_sha256):
            raise RuntimeError("Compiler layout object changed")
        table = read_i64_table(self.object_bytes, self.query.table_symbol, len(self.query.rows),
                              expected_machine=EM_AARCH64, expected_class=ELFCLASS64,
                              expected_endianness=ELFDATA2LSB)
        if table.values != self.values or _layouts(self.query, table.values) != self.formals:
            raise RuntimeError("Compiler layout table no longer matches its typed requests")


def compile_target_layout(plan: Any) -> TargetLayout:
    import cutlass.compiler as compiler

    host_target = compiler.Compiler.detect_host_triple()
    if host_target != "aarch64-unknown-linux-gnu":
        raise ValueError("The object reader currently supports the inspected AArch64 Linux target")
    query = emit_layout_ir(plan, host_target)
    llvm_text = emit_llvm_query(query)
    source = compiler.LlvmIrArtifact.from_textual_form(llvm_text.encode())
    if type(source) is not compiler.LlvmIrArtifact or source.is_consumed or len(source.metadata):
        raise RuntimeError("Expected the exact data-only layout source artifact")
    native = compiler.CuteCompiler()
    native.set_device_target(plan.mapping.program.arch)
    native.set_host_target(host_target)
    # Tbd explicitly requests no wrapper for a module containing only constant data.
    native.set_abi(compiler.Abi.Tbd)
    artifact = native.compile_to(source, compiler.ArtifactType.Object)
    if not source.is_consumed or type(artifact) is not compiler.ObjectArtifact or artifact.is_consumed:
        raise RuntimeError("Expected the compiler-owned layout object continuation")
    data = artifact.get_data()
    table = read_i64_table(data, query.table_symbol, len(query.rows), expected_machine=EM_AARCH64,
                          expected_class=ELFCLASS64, expected_endianness=ELFDATA2LSB)
    formals = _layouts(query, table.values)
    result = TargetLayout(query, source, artifact, llvm_text, data, sha256(data).hexdigest(), table.values, formals,
                          (query, source, artifact, llvm_text, data, table.values, formals))
    result.check()
    return result
