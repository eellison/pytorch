"""Translate compiler-owned scalar SSA into the native typed parameter program."""

import math
import struct

from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.artifact import ParameterExpression
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource, InputSource, IntExpr, ParameterSource, PointerSource,
)
from torch._inductor.runtime._cudagraph._compiler.user_triton.linking import LinkDeclined

from .numeric import RangeObligation


def lower_parameter(expression, operands):
    obligations = []
    cache = {}

    def read(value):
        if type(value) is not ParameterExpression:
            raise LinkDeclined("Parameter expression lost its immutable compiler provenance")
        if value in cache:
            return cache[value]
        typ = ir.Type.parse(value.llvm_type)
        pointer = isinstance(typ, llvm.PointerType)
        floating = isinstance(typ, ir.F32Type)
        width = 64 if pointer else 32 if floating else typ.width if isinstance(typ, ir.IntegerType) else None
        if width not in (1, 32, 64) or not pointer and not floating and not typ.is_signless:
            raise LinkDeclined("Parameter expression requires a pointer, an f32 or signless i1/i32/i64")
        attrs = {name: ir.Attribute.parse(text) for name, text in value.attributes}
        if len(attrs) != len(value.attributes):
            raise LinkDeclined("Parameter expression has duplicate compiler attributes")
        if value.kind == "argument":
            if attrs or value.operands or value.value is not None:
                raise LinkDeclined("Parameter argument has unexpected compiler payload")
            formal = operands.formals.get(value.source_arg_index)
            leaves = [] if formal is None else [leaf for leaf in formal.leaves if leaf.path == value.path]
            if len(leaves) != 1 or leaves[0].llvm_type != value.llvm_type:
                raise LinkDeclined("Parameter argument lacks one exact original compiler leaf")
            leaf, = leaves
            source = operands.property(formal, leaf.property, leaf.property_path)
            if pointer:
                if leaf.property != "pointer" or leaf.size != 8:
                    raise LinkDeclined("Parameter pointer lost its compiler pointer leaf")
                if type(source) in (InputSource, BufferSource):
                    source = PointerSource(source, IntExpr("constant", 0))
                if type(source) is not PointerSource:
                    raise LinkDeclined("Parameter pointer lacks its traced storage root")
                result = ParameterSource("pointer", 64, source)
            elif type(source) is ParameterSource:
                if (formal.kind != "Var" or leaf.property != "value" or leaf.property_path
                        or width not in (32, 64) or source.width != width):
                    raise LinkDeclined("Late scalar import differs from its original compiler integer leaf")
                result = source
            else:
                if width == 1:
                    raise LinkDeclined("Dynamic i1 parameter imports are not supported")
                numeric = operands.numeric_value(source)
                minimum, maximum = -(1 << (width - 1)), (1 << (width - 1)) - 1
                if (numeric.lower is None or numeric.lower < minimum
                        or numeric.upper is None or numeric.upper > maximum):
                    obligations.append(RangeObligation(numeric.expression, minimum, maximum,
                                                       "CuTe parameter source integer width"))
                result = ParameterSource("value", width, numeric.expression)
        elif value.kind in ("constant", "zero"):
            if pointer or attrs or value.operands or value.source_arg_index is not None or value.path:
                raise LinkDeclined("Parameter constant has unsupported typed provenance")
            if value.kind == "zero":
                if value.value is not None:
                    raise LinkDeclined("Compiler zero has an unexpected literal")
                integer = 0
            else:
                attr = ir.Attribute.parse(value.value)
                if isinstance(attr, ir.BoolAttr) and width == 1:
                    integer = int(attr.value)
                elif isinstance(attr, ir.IntegerAttr) and attr.type == typ:
                    integer = attr.value
                elif floating and isinstance(attr, ir.FloatAttr) and attr.type == typ:
                    literal = attr.value
                    if math.isnan(literal):
                        raise LinkDeclined("NaN f32 parameter literals are unsupported")
                    # an f32 literal (a kernel's scalar constant) as its bit pattern
                    integer = struct.unpack("<I", struct.pack("<f", literal))[0]
                else:
                    raise LinkDeclined("Parameter literal differs from its compiler integer type")
            result = ParameterSource("constant", width, integer & ((1 << width) - 1))
        else:
            if (not value.kind.startswith("llvm.") or value.source_arg_index is not None
                    or value.path or value.value is not None):
                raise LinkDeclined("Computed parameter has unexpected source metadata")
            args = tuple(read(arg) for arg in value.operands)
            types = tuple(ir.Type.parse(arg.llvm_type) for arg in value.operands)
            op = value.kind.removeprefix("llvm.")
            flags = ()
            if "overflowFlags" in attrs:
                attr = attrs.pop("overflowFlags")
                mask = int(llvm.IntegerOverflowFlags.nsw | llvm.IntegerOverflowFlags.nuw)
                if (not isinstance(attr, ir.IntegerAttr) or str(attr.type) != "i32"
                        or attr.value < 0 or attr.value & ~mask):
                    raise LinkDeclined("Parameter arithmetic has invalid LLVM overflow flags")
                flags = tuple(name for name in ("nuw", "nsw")
                              if attr.value & int(getattr(llvm.IntegerOverflowFlags, name)))
                if op not in ("add", "sub", "mul", "shl"):
                    raise LinkDeclined("Parameter operation cannot retain LLVM overflow flags")
            if "isExact" in attrs:
                attr = attrs.pop("isExact")
                if flags or not isinstance(attr, ir.UnitAttr) or op not in ("udiv", "sdiv", "lshr", "ashr"):
                    raise LinkDeclined("Parameter operation cannot retain LLVM exactness")
                flags = ("exact",)
            if op == "icmp":
                predicate = attrs.pop("predicate", None)
                if (attrs or width != 1 or pointer or len(args) != 2 or types[0] != types[1]
                        or not isinstance(types[0], ir.IntegerType) or not isinstance(predicate, ir.IntegerAttr)
                        or predicate.type != ir.IntegerType.get_signless(64)):
                    raise LinkDeclined("Parameter comparison lacks its exact integer predicate")
                result = ParameterSource(op, 1, str(llvm.ICmpPredicate(predicate.value)), args)
            elif op in ("ptrtoint", "inttoptr", "bitcast"):
                if attrs or flags or len(args) != 1:
                    raise LinkDeclined("Parameter cast has unsupported attributes or operands")
                source_pointer = isinstance(types[0], llvm.PointerType)
                if op == "ptrtoint" and source_pointer and not pointer:
                    result = args[0] if width == 64 else ParameterSource("trunc", width, args=args)
                elif op == "inttoptr" and pointer and not source_pointer and args[0].width == 64:
                    result = args[0]
                elif op == "bitcast" and typ == types[0]:
                    result = args[0]
                else:
                    raise LinkDeclined("Parameter cast does not preserve supported pointer/integer bits")
            elif op in ("trunc", "zext", "sext"):
                if attrs or flags or pointer or len(args) != 1 or not isinstance(types[0], ir.IntegerType):
                    raise LinkDeclined("Parameter integer cast has unsupported typed operands")
                if (op == "trunc" and width >= args[0].width or op != "trunc" and width <= args[0].width):
                    raise LinkDeclined("Parameter integer cast has invalid widths")
                result = ParameterSource(op, width, args=args)
            elif op == "select":
                if (attrs or flags or pointer or len(args) != 3 or args[0].width != 1
                        or types[1] != typ or types[2] != typ):
                    raise LinkDeclined("Parameter selection has incompatible compiler types")
                result = ParameterSource(op, width, args=args)
            elif op in ("add", "sub", "mul", "udiv", "sdiv", "urem", "srem",
                        "shl", "lshr", "ashr", "and", "or", "xor"):
                if (attrs or pointer or len(args) != 2 or types != (typ, typ)
                        or flags and op not in ("add", "sub", "mul", "shl", "udiv", "sdiv", "lshr", "ashr")):
                    raise LinkDeclined("Parameter arithmetic has incompatible compiler types or flags")
                result = ParameterSource(op, width, args=args, flags=flags)
            else:
                raise LinkDeclined(f"Unsupported compiler parameter operation: {value.kind}")
        cache[value] = result
        return result

    with ir.Context(), ir.Location.unknown():
        result = read(expression)
    return result, tuple(dict.fromkeys(obligations))
