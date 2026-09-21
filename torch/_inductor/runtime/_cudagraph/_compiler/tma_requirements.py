"""Preserve host TMA requirements from the compiler's descriptor basis."""

from dataclasses import dataclass
from math import gcd
import re
from typing import Any


_ATOM = re.compile(
    r'!cute_nvgpu\.atom\.non_exec_tiled_tma_(load|store)<'
    r'(?:(sm_[0-9]+), )?([^,<>]+), copy_bits = ([1-9][0-9]*), '
    r'tma_gbasis = (<"[^"\\]+">), tma_format = ([A-Z0-9_]+)>'
)
_FORMAT_BITS = {
    "U8": 8, "U16": 16, "U32": 32, "S32": 32, "U64": 64, "S64": 64,
    "F16_RN": 16, "F32_RN": 32, "F32_FTZ_RN": 32, "F64_RN": 64,
    "BF16_RN": 16, "TF32_RN": 32, "TF32_FTZ_RN": 32,
}


@dataclass(frozen=True)
class TmaStrideRequirement:
    constructor: Any
    tensor: Any
    path: tuple[int, ...]
    old_bits: int
    new_bits: int
    divisor: int
    group: int | None = None


@dataclass(frozen=True)
class TmaDimensionRequirement:
    constructor: Any
    tensor: Any
    path: tuple[int, ...]
    old_bits: int
    new_bits: int
    group: int
    grouped: bool


def _atom_layout(atom_type):
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import cute, cute_nvgpu

    text = str(atom_type)
    match = _ATOM.fullmatch(text)
    if match is None or ir.Type.parse(text) != atom_type:
        raise ValueError("Unsupported complete TMA atom type")
    kind, arch, element, _, basis, format_name = match.groups()
    cls = (cute_nvgpu.CopyAtomNonExecTiledTmaLoadType if kind == "load"
           else cute_nvgpu.CopyAtomNonExecTiledTmaStoreType)
    if (not cls.isinstance(atom_type) or (arch is not None) != (kind == "load")
            or str(cls(atom_type).value_type) != element):
        raise ValueError("TMA atom fields disagree with its compiler type")
    if format_name not in _FORMAT_BITS or format_name not in cute_nvgpu.TmaDataFormat.__members__:
        raise ValueError("TMA stride requirements support only unpacked byte-addressable formats")
    spelling = "!cute.layout" + basis
    layout = cute.LayoutType(ir.Type.parse(spelling))
    if str(layout) != spelling:
        raise ValueError("TMA basis did not round-trip through the compiler layout parser")
    return layout, _FORMAT_BITS[format_name]


def _basis_groups(layout_type):
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import cute, func
    from cutlass.cute import core

    with ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp("tma_basis", ([], []))
        with ir.InsertionPoint(function.add_entry_block()):
            layout = cute.StaticOp(layout_type).result
            basis = core._unpack_x_tuple(cute.GetStrideOp(layout).result)
            func.ReturnOp([])
    groups = basis if isinstance(basis, tuple) else (basis,)
    result = []
    for group in groups:
        if isinstance(group, tuple) and len(group) > 1:
            terms = group
        else:
            # CUTLASS unwraps rank-one groups, but does not flatten grouped contributions.
            while isinstance(group, tuple) and len(group) == 1:
                group = group[0]
            terms = (group,)
        paths = []
        for term in terms:
            if not isinstance(term, core.ScaledBasis):
                raise ValueError("TMA descriptor groups require scalar basis contributions")
            path = tuple(term.mode)
            if any(type(index) is not int or index < 0 for index in path):
                raise ValueError("TMA basis contains an invalid source mode")
            paths.append(path)
        result.append(tuple(paths))
    return tuple(result)


def read_tma_requirements(constructor):
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import cute, cute_nvgpu
    from cutlass.cute import core

    constructor = constructor.operation if isinstance(constructor, ir.OpView) else constructor
    if not isinstance(constructor, ir.Operation):
        raise TypeError("Expected an original host TMA constructor")
    views = {
        "cute_nvgpu.atom.make_non_exec_tiled_tma_load": cute_nvgpu.AtomCopyMakeNonExecTiledTmaLoadOp,
        "cute_nvgpu.atom.make_non_exec_tiled_tma_store": cute_nvgpu.AtomCopyMakeNonExecTiledTmaStoreOp,
    }
    cls = views.get(constructor.name)
    attrs = {"kind", "num_multicast", "tma_format"} if constructor.name.endswith("_load") else {"tma_format"}
    if (cls is None or not isinstance(constructor.opview, cls) or len(constructor.operands) != 3
            or len(constructor.results) != 2 or constructor.regions or constructor.successors
            or set(constructor.attributes) - attrs or not constructor.verify()):
        raise ValueError("Expected an exact supported host TMA constructor")
    tensor = constructor.operands[0]
    if not isinstance(tensor.type, cute.MemRefType):
        raise ValueError("TMA constructor requires a compiler MemRef operand")
    old_bits = core.Numeric.from_mlir_type(tensor.type.value_type).width
    if old_bits not in (8, 16, 32, 64):
        raise ValueError("TMA stride requirements support only byte-addressable source elements")
    layout_type, new_bits = _atom_layout(constructor.results[0].type)
    groups = _basis_groups(layout_type)
    widths = (new_bits,) if old_bits == new_bits else (old_bits, new_bits)
    # Preserve original byte alignment before a wider recast divides dynamic strides.
    strides = tuple(TmaStrideRequirement(constructor, tensor, path, old_bits, bits,
                                        16 // gcd(16, bits // 8), group if bits == new_bits else None)
                    for group, paths in enumerate(groups[1:]) for path in dict.fromkeys(paths) for bits in widths)
    dimensions = tuple(TmaDimensionRequirement(constructor, tensor, path, old_bits, new_bits, group, len(paths) > 1)
                       for group, paths in enumerate(groups) for path in paths)
    return strides, dimensions


def project_tma_property(requirement, cloned_tensor, property):
    """Return the selected recast shape or element stride as an i64 SSA value."""
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import arith, cute

    if (type(requirement) not in (TmaStrideRequirement, TmaDimensionRequirement)
            or property not in ("shape", "stride")
            or not isinstance(cloned_tensor, ir.Value)
            or cloned_tensor.type != requirement.tensor.type
            or requirement.constructor.operands[0] != requirement.tensor):
        raise ValueError("TMA projection lost its original tensor source")
    layout = cute.GetLayoutOp(cloned_tensor).result
    recast = cute.RecastLayoutOp(requirement.new_bits, requirement.old_bits, layout).dst
    values = (cute.GetShapeOp(recast) if property == "shape" else cute.GetStrideOp(recast)).result
    selected_type = values.type.get_op_res_type(mode=list(requirement.path))
    selected = cute.GetOp(selected_type, values, mode=list(requirement.path)).result
    scalars = tuple(cute.GetScalarsOp(selected).results)
    if (len(scalars) != 1 or not isinstance(scalars[0].type, ir.IntegerType)
            or scalars[0].type.width not in (32, 64)):
        raise ValueError("TMA basis mode must select one i32 or i64 property")
    value = scalars[0]
    return arith.ExtSIOp(ir.IntegerType.get_signless(64), value).result if value.type.width == 32 else value
