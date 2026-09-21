"""Ask the host compiler for sizes, alignments, and aggregate field offsets."""

from dataclasses import dataclass

from torch._inductor.runtime._cudagraph._compiler.aggregate_plan import _scalar_leaves
from cutlass._mlir import ir
from torch._inductor.runtime._cudagraph._compiler.elf_table import ELFDATA2LSB, ELFCLASS64, EM_AARCH64, read_i64_table
from torch._inductor.runtime._cudagraph._compiler.llvm_query import _type
from torch._inductor.runtime._cudagraph._compiler.target_layout import FieldLayout


@dataclass(frozen=True)
class TypeLayout:
    llvm_type: str
    size: int
    alignment: int
    fields: tuple[FieldLayout, ...]


@dataclass(frozen=True)
class CompiledTypeLayouts:
    host_target: str
    types: tuple
    llvm_text: str
    object_bytes: bytes
    layouts: tuple[TypeLayout, ...]


def compile_type_layouts(types, *, device_target):
    import cutlass.compiler as compiler

    if type(types) is not tuple or not types or any(not isinstance(typ, ir.Type) for typ in types):
        raise TypeError("Expected actual compiler types in request order")
    context = types[0].context
    if any(typ.context != context for typ in types):
        raise ValueError("Type layout requests must share their original compiler context")
    host_target = compiler.Compiler.detect_host_triple()
    if host_target != "aarch64-unknown-linux-gnu":
        raise ValueError("The object reader currently supports AArch64 Linux")
    expressions, requests = [], []
    with context:
        for typ in types:
            spelling = _type(typ)
            leaves = _scalar_leaves(typ)
            questions = [
                f"getelementptr ({spelling}, ptr null, i64 1)",
                f"getelementptr ({{ i8, {spelling} }}, ptr null, i64 0, i32 1)",
            ]
            for path, leaf_name in leaves:
                leaf = _type(ir.Type.parse(leaf_name))
                indices = ", ".join(("i64 0", *(f"i32 {index}" for index in path)))
                questions.extend((
                    f"getelementptr ({spelling}, ptr null, {indices})",
                    f"getelementptr ({leaf}, ptr null, i64 1)",
                    f"getelementptr ({{ i8, {leaf} }}, ptr null, i64 0, i32 1)",
                ))
            requests.append((str(typ), leaves, len(expressions)))
            expressions.extend(f"i64 ptrtoint (ptr {question} to i64)" for question in questions)
    symbol = "__cudagraph_type_layouts"
    text = (f'target triple = "{host_target}"\n'
            f'@{symbol} = constant [{len(expressions)} x i64] [\n'
            + ",\n".join(expressions) + "\n]\n")
    source = compiler.LlvmIrArtifact.from_textual_form(text.encode())
    if type(source) is not compiler.LlvmIrArtifact or source.is_consumed or len(source.metadata):
        raise ValueError("Expected a compiler-owned data-only layout source")
    native = compiler.CuteCompiler()
    native.set_device_target(device_target)
    native.set_host_target(host_target)
    native.set_abi(compiler.Abi.Tbd)
    artifact = native.compile_to(source, compiler.ArtifactType.Object)
    if not source.is_consumed or type(artifact) is not compiler.ObjectArtifact or artifact.is_consumed:
        raise ValueError("Expected the compiler-owned layout object")
    data = artifact.get_data()
    table = read_i64_table(data, symbol, len(expressions), expected_machine=EM_AARCH64,
                          expected_class=ELFCLASS64, expected_endianness=ELFDATA2LSB)
    layouts = []
    for spelling, leaves, first in requests:
        size, alignment = table.values[first:first + 2]
        if size < 0 or alignment <= 0 or alignment & (alignment - 1) or size % alignment:
            raise ValueError("Compiler returned an invalid type size or alignment")
        fields = []
        for index, (path, leaf) in enumerate(leaves):
            position = first + 2 + 3 * index
            offset, field_size, field_alignment = table.values[position:position + 3]
            if (offset < 0 or field_size <= 0 or field_alignment <= 0
                    or field_alignment & (field_alignment - 1) or offset + field_size > size):
                raise ValueError("Compiler field layout exceeds its actual aggregate")
            fields.append(FieldLayout(path, leaf, offset, field_size, field_alignment))
        end = 0
        for field in sorted(fields, key=lambda field: field.offset):
            if field.offset < end:
                raise ValueError("Compiler aggregate fields overlap")
            end = field.offset + field.size
        layouts.append(TypeLayout(spelling, size, alignment, tuple(fields)))
    return CompiledTypeLayouts(host_target, types, text, data, tuple(layouts))
