from __future__ import annotations

import struct
from dataclasses import dataclass


ELFCLASS64 = 2
ELFDATA2LSB = 1
EM_AARCH64 = 183

_HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
_SECTION = struct.Struct("<IIQQQQIIQQ")
_SYMBOL = struct.Struct("<IBBHQQ")
_REL = struct.Struct("<QQ")
_RELA = struct.Struct("<QQq")
_SECTION_TYPES = frozenset({0, 1, 2, 3, 4, 7, 9, 0x6FFF4C03})


@dataclass(frozen=True)
class ELFTable:
    symbol: str
    values: tuple[int, ...]
    data: bytes
    machine: int
    elf_class: int
    endianness: int
    section_index: int
    section_name: str
    section_alignment: int
    symbol_value: int
    object_offset: int


@dataclass(frozen=True)
class _Section:
    name: int
    kind: int
    flags: int
    address: int
    offset: int
    size: int
    link: int
    info: int
    alignment: int
    entry_size: int


def _span(data: bytes, offset: int, size: int, label: str) -> bytes:
    if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
        raise ValueError(f"{label} is outside object bounds")
    return data[offset:offset + size]


def _string(strings: bytes, offset: int, label: str) -> bytes:
    if offset < 0 or offset >= len(strings):
        raise ValueError(f"{label} string offset is outside table bounds")
    end = strings.find(b"\0", offset)
    if end < 0:
        raise ValueError(f"{label} string is not terminated")
    return strings[offset:end]


def read_i64_table(
    object_bytes: bytes,
    symbol: str,
    element_count: int,
    *,
    expected_machine: int,
    expected_class: int,
    expected_endianness: int,
) -> ELFTable:
    """Read one global i64 table from a strict AArch64 ELF64 relocatable object."""
    if type(object_bytes) is not bytes or type(symbol) is not str or not symbol or "\0" in symbol:
        raise TypeError("Expected immutable object bytes and a nonempty symbol name")
    if type(element_count) is not int or element_count <= 0:
        raise ValueError("Expected a positive table element count")
    requested = (expected_machine, expected_class, expected_endianness)
    if any(type(value) is not int for value in requested) or requested != (EM_AARCH64, ELFCLASS64, ELFDATA2LSB):
        raise ValueError("Only explicit little-endian ELF64 AArch64 is supported")
    try:
        symbol_bytes = symbol.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("The table symbol must be ASCII") from error
    header = _HEADER.unpack(_span(object_bytes, 0, _HEADER.size, "ELF header"))
    (ident, kind, machine, version, entry, ph_offset, sh_offset, flags, header_size,
     ph_entry_size, ph_count, sh_entry_size, sh_count, sh_names) = header
    if ident[:4] != b"\x7fELF" or ident[4:7] != bytes((expected_class, expected_endianness, 1)):
        raise ValueError("ELF magic, class, endianness, or version differs")
    if ident[7] not in (0, 3) or any(ident[8:]):
        raise ValueError("Unsupported ELF ABI or identification padding")
    if kind != 1 or machine != expected_machine or version != 1 or flags or entry:
        raise ValueError("Expected an ordinary AArch64 relocatable object")
    if ph_offset or ph_count or ph_entry_size:
        raise ValueError("Program headers are unsupported in the table object")
    if header_size != _HEADER.size or sh_entry_size != _SECTION.size:
        raise ValueError("Unexpected ELF header or section entry size")
    if not 0 < sh_count < 0xFF00 or not 0 < sh_names < sh_count:
        raise ValueError("Missing or extended ELF section numbering is unsupported")
    if sh_offset < _HEADER.size or sh_offset % 8:
        raise ValueError("Invalid section header table offset")
    raw_sections = _span(object_bytes, sh_offset, sh_count * _SECTION.size, "Section header table")
    sections = tuple(_Section(*fields) for fields in _SECTION.iter_unpack(raw_sections))
    if any(raw_sections[:_SECTION.size]):
        raise ValueError("Section zero must be empty; extended numbering is unsupported")
    occupied = [(0, _HEADER.size), (sh_offset, sh_offset + len(raw_sections))]
    for section in sections[1:]:
        if section.kind not in _SECTION_TYPES or section.kind == 0:
            raise ValueError("Unsupported ELF section type, including NOBITS or extended indices")
        if section.flags & 0x800:
            raise ValueError("Compressed sections are unsupported")
        if section.address:
            raise ValueError("Relocatable sections must not have assigned addresses")
        alignment = section.alignment
        if alignment and (alignment & (alignment - 1) or section.offset % alignment):
            raise ValueError("Invalid section alignment")
        _span(object_bytes, section.offset, section.size, "Section")
        if section.size:
            occupied.append((section.offset, section.offset + section.size))
    occupied.sort()
    if any(left[1] > right[0] for left, right in zip(occupied, occupied[1:])):
        raise ValueError("Overlapping ELF file regions are unsupported")
    names_section = sections[sh_names]
    if names_section.kind != 3:
        raise ValueError("Section names must use a string table")
    names = _span(object_bytes, names_section.offset, names_section.size, "Section names")
    if not names or names[0] or names[-1]:
        raise ValueError("Invalid section-name string table")
    section_names = tuple(_string(names, item.name, "Section name") for item in sections)
    symbol_sections = [(index, section) for index, section in enumerate(sections) if section.kind == 2]
    if len(symbol_sections) != 1:
        raise ValueError("Expected exactly one ordinary ELF symbol table")
    symbol_index, symtab = symbol_sections[0]
    if symtab.entry_size != _SYMBOL.size or symtab.size % _SYMBOL.size or not symtab.size:
        raise ValueError("Invalid symbol table entry size")
    if not 0 < symtab.link < sh_count or sections[symtab.link].kind != 3:
        raise ValueError("Symbol table lacks its linked string table")
    strings_section = sections[symtab.link]
    strings = _span(object_bytes, strings_section.offset, strings_section.size, "Symbol strings")
    if not strings or strings[0] or strings[-1]:
        raise ValueError("Invalid symbol string table")
    symbols = tuple(_SYMBOL.iter_unpack(_span(object_bytes, symtab.offset, symtab.size, "Symbol table")))
    if any(symbols[0]) or not 1 <= symtab.info <= len(symbols):
        raise ValueError("Invalid null symbol or local-symbol boundary")
    matches = []
    for index, record in enumerate(symbols):
        name, info, other, section_index, value, size = record
        if section_index == 0xFFFF or other & ~3:
            raise ValueError("Extended symbol indices or attributes are unsupported")
        if section_index >= sh_count and section_index not in (0xFFF1, 0xFFF2):
            raise ValueError("Symbol section index is outside table bounds")
        if (info >> 4 == 0) != (index < symtab.info):
            raise ValueError("Symbol binding disagrees with the local-symbol boundary")
        if _string(strings, name, "Symbol name") == symbol_bytes:
            matches.append(record)
    if len(matches) != 1:
        raise ValueError("Requested table symbol is missing or duplicated")
    _, info, _, target_index, value, size = matches[0]
    if info != 0x11 or not 0 < target_index < sh_count:
        raise ValueError("Table symbol must be a defined global STT_OBJECT")
    target = sections[target_index]
    if target.kind != 1 or target.flags != 2:
        raise ValueError("Table must occupy an ordinary read-only allocated PROGBITS section")
    if size != element_count * 8:
        raise ValueError("Table symbol size differs from the expected i64 count")
    if value > target.size or size > target.size - value:
        raise ValueError("Table symbol is outside section bounds")
    for section in sections:
        if section.kind not in (4, 9):
            continue
        decoder = _RELA if section.kind == 4 else _REL
        if section.entry_size != decoder.size or section.size % decoder.size:
            raise ValueError("Invalid relocation entry size")
        if section.link != symbol_index or not 0 < section.info < sh_count:
            raise ValueError("Invalid relocation symbol or target section")
        relocation_target = sections[section.info]
        if relocation_target.kind != 1:
            raise ValueError("Unsupported relocation target section")
        entries = decoder.iter_unpack(_span(object_bytes, section.offset, section.size, "Relocations"))
        for relocation in entries:
            offset, info = relocation[:2]
            if offset >= relocation_target.size or info >> 32 >= len(symbols):
                raise ValueError("Relocation offset or symbol is outside bounds")
            if section.info == target_index:
                raise ValueError("Relocations targeting the table section are unsupported")
    try:
        section_name = section_names[target_index].decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("The table section name must be ASCII") from error
    object_offset = target.offset + value
    table_bytes = _span(object_bytes, object_offset, size, "Table contents")
    values = tuple(item[0] for item in struct.iter_unpack("<q", table_bytes))
    return ELFTable(symbol, values, table_bytes, machine, ident[4], ident[5], target_index,
                    section_name, target.alignment, value, object_offset)
