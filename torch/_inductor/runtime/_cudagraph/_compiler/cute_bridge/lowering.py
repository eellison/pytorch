"""Join recorded Python operands to compiler-owned CuTe parameter fields."""

from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource, ExpressionSource, grid_expression_inputs, InputSource, IntegerSource,
    IntExpr, ParameterSource, PointerSource,
)
from torch._inductor.runtime.cudagraph_boxed_replay import _PhysicalCall, _PhysicalField
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import SymPyValueRangeAnalysis, ValueRanges
from torch._inductor.runtime._cudagraph._compiler.user_triton.linking import LinkDeclined

from .numeric import lower_numeric, NumericSource, RangeObligation
from .parameter_program import lower_parameter


class _Operands:
    def __init__(self, trace, artifact, tensors, expressions, *,
                 parameter_expression=None, storage_offset_indices=()):
        self.trace = trace
        self.artifact = artifact
        self.tensors = tensors
        self.expressions = expressions
        self.parameter_expression = parameter_expression
        self.storage_offset_indices = set(storage_offset_indices)
        self.formals = {formal.source_arg_index: formal for formal in artifact.formals}
        self.ranges = {row.index: row for row in trace.input_contract.integer_ranges}
        if len(self.formals) != len(artifact.formals):
            raise LinkDeclined("CuTe compiler formals have duplicate original indices")

    def tensor(self, formal):
        operand = self.artifact.signature.operands[formal.operand_index]
        if (formal.kind != "Tensor" or operand.name != formal.name or operand.tensor is None
                or id(operand.tensor.value) not in self.tensors):
            raise LinkDeclined("CuTe formal has no recorded Python Tensor identity")
        return operand.tensor

    def property(self, formal, name, path):
        if formal.kind == "Var":
            operand = self.artifact.signature.operands[formal.operand_index]
            if (operand.name != formal.name or operand.scalar is None or operand.tensor is not None
                    or operand.scalar.kind != "integer" or name != "value" or path):
                raise LinkDeclined("CuTe integer formal lost its original symbolic value")
            if self.parameter_expression is not None:
                bits = operand.scalar.bits
                source = self.parameter_expression(operand.scalar.use.value, f"i{bits}")
                if source is not None:
                    if type(source) is not ParameterSource or bits not in (32, 64) or source.width != bits:
                        raise LinkDeclined("CuTe late scalar differs from its original integer formal width")
                    return source
            return self.expressions(operand.scalar.use.value)
        tensor = self.tensor(formal)
        if name == "pointer" and path == ():
            return self.tensors[id(tensor.value)]
        if name not in ("shape", "stride") or len(path) != 1:
            raise LinkDeclined("Unsupported CuTe Tensor property path")
        uses = tensor.shape if name == "shape" else tensor.strides
        if type(path[0]) is not int or not 0 <= path[0] < len(uses):
            raise LinkDeclined("CuTe Tensor property path exceeds its original operand")
        return self.expressions(uses[path[0]].value)

    def field(self, source):
        if source.kind == "compiler_constant":
            bits = {"i32": 32, "i64": 64}.get(source.llvm_type)
            if (bits is None or type(source.constant) is not int
                    or source.ir_arg_index is not None or source.formal_name is not None
                    or source.metadata_path or source.property != "value" or source.property_path
                    or source.component_path or not -(2 ** (bits - 1)) <= source.constant < 2 ** (bits - 1)):
                raise LinkDeclined("CuTe literal field lost its exact typed compiler value")
            return source.constant
        formal = self.formals.get(source.ir_arg_index)
        if (formal is None or source.kind != ("scalar_formal" if formal.kind == "Var" else "tensor_property")
                or source.formal_name != formal.name
                or source.metadata_path != formal.metadata_path):
            raise LinkDeclined("CuTe field lost its exact original formal")
        leaves = [leaf for leaf in formal.leaves if leaf.path == source.component_path]
        if len(leaves) != 1:
            raise LinkDeclined("CuTe field lacks one compiler-identified aggregate leaf")
        leaf, = leaves
        if (source.llvm_type, source.property, source.property_path) != (
                leaf.llvm_type, leaf.property, leaf.property_path):
            raise LinkDeclined("CuTe field property differs from its compiler leaf")
        return self.property(formal, source.property, source.property_path)

    def numeric(self, source_index, path):
        formal = self.formals.get(source_index)
        if formal is None:
            raise LinkDeclined("CuTe computation has no original formal")
        leaves = [leaf for leaf in formal.leaves if leaf.path == path]
        if len(leaves) != 1 or leaves[0].property == "pointer":
            raise LinkDeclined("CuTe computation is not a recorded integer metadata leaf")
        leaf, = leaves
        value = self.property(formal, leaf.property, leaf.property_path)
        return self.numeric_value(value)

    def numeric_value(self, value):
        if type(value) is ParameterSource:
            raise LinkDeclined("CuTe grid, dispatch and allocation metadata require early integer sources")
        if type(value) is int:
            return NumericSource(IntExpr("constant", value), value, value)
        inputs = grid_expression_inputs(value, boxed_indices=set(self.ranges),
                                        tensor_indices=self.storage_offset_indices)
        if inputs is None:
            raise LinkDeclined("CuTe metadata expression lacks supported recorded sources")

        def bounds(expression):
            if expression.op == "constant":
                return ValueRanges(expression.value, expression.value)
            if expression.op == "storage_offset":
                return ValueRanges(0, (1 << 63) - 1)
            if expression.op == "boxed":
                row = self.ranges[expression.value]
                return ValueRanges(row.lower, int_oo if row.upper is None else row.upper)
            if expression.op == "select":
                _, when_true, when_false = (bounds(arg) for arg in expression.args)
                return ValueRanges(min(when_true.lower, when_false.lower), max(when_true.upper, when_false.upper))
            left, right = (bounds(arg) for arg in expression.args)
            if expression.op == "add":
                return SymPyValueRangeAnalysis.add(left, right)
            if expression.op == "multiply":
                return SymPyValueRangeAnalysis.mul(left, right)
            if expression.op == "ceildiv":
                numerator = SymPyValueRangeAnalysis.add(left, expression.args[1].value - 1)
                return SymPyValueRangeAnalysis.floordiv(numerator, right)
            if expression.op == "floordiv":
                return SymPyValueRangeAnalysis.floordiv(left, right)
            if expression.op in ("eq", "ne", "lt", "le", "gt", "ge", "and"):
                return ValueRanges(0, 1)
            raise LinkDeclined("CuTe metadata expression has no interval semantics")

        interval = bounds(value)
        if not interval.is_int:
            raise LinkDeclined("CuTe metadata expression has no inherited integer interval")
        lower, upper = (None if bound in (-int_oo, int_oo) else int(bound)
                        for bound in (interval.lower, interval.upper))
        return NumericSource(value, lower, upper)


def _lower_cute_call(artifact, site, owner, operands):
    obligations = []
    predicate = None
    fields = []
    for field in site.fields.pointers:
        if field.source.kind == "compiler_expression":
            source, required = lower_parameter(field.source.expression, operands)
            obligations.extend(required)
            if source.width != 64:
                raise LinkDeclined("CuTe computed pointer has an unsupported physical width")
        else:
            source = operands.field(field.source)
        if type(source) not in (BufferSource, InputSource, PointerSource, ParameterSource):
            raise LinkDeclined("CuTe pointers must name recorded buffers or normalized inputs")
        fields.append(_PhysicalField(field.parameter, field.byte_offset, "pointer", source))
    for field in site.fields.integers:
        if field.source.kind == "compiler_expression":
            source, required = lower_parameter(field.source.expression, operands)
            if source.width != {"i32": 32, "i64": 64}.get(field.dtype):
                raise LinkDeclined("CuTe computed integer differs from its physical ABI width")
            fields.append(_PhysicalField(field.parameter, field.byte_offset, field.dtype, source))
            obligations.extend(required)
            continue
        value = operands.field(field.source)
        if type(value) is ParameterSource:
            if value.width != {"i32": 32, "i64": 64}.get(field.dtype):
                raise LinkDeclined("CuTe late scalar differs from its physical ABI width")
            fields.append(_PhysicalField(field.parameter, field.byte_offset, field.dtype, value))
            continue
        source = ExpressionSource(value) if type(value) is IntExpr else IntegerSource(value)
        fields.append(_PhysicalField(field.parameter, field.byte_offset, field.dtype, source))
        numeric = operands.numeric_value(value)
        bits = {"i32": 32, "i64": 64}.get(field.dtype)
        if bits is None:
            raise LinkDeclined("CuTe integer field has an unsupported physical width")
        minimum, maximum = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        if (numeric.lower is None or numeric.lower < minimum
                or numeric.upper is None or numeric.upper > maximum):
            obligations.append(RangeObligation(numeric.expression, minimum, maximum,
                                              "CuTe physical integer field"))
    grids = {}
    for consumer in artifact.consumers:
        if consumer.site_id == site.site_id and consumer.role == "grid":
            value = lower_numeric(consumer.numeric, operands.numeric)
            if len(value.values) != 1 or type(value.values[0].expression) is not IntExpr:
                raise LinkDeclined("CuTe grid did not lower to one native integer recipe")
            grids[consumer.index] = value.values[0].expression
            obligations.extend(value.obligations)
        elif consumer.site_id is None and consumer.role == "predicate":
            value = lower_numeric(consumer.numeric, operands.numeric)
            if predicate is not None or len(value.values) != 1:
                raise LinkDeclined("CuTe artifact lacks one exact dispatch predicate")
            predicate = value.values[0]
            obligations.extend(value.obligations)
    unconditional = (all(item.arm is None for item in artifact.sites)
                     and any(site is item for item in artifact.sites))
    if set(grids) != {0, 1, 2} or (predicate is None) != unconditional:
        raise LinkDeclined("CuTe launch lacks complete grid or dispatch computations")
    bound = _PhysicalCall(tuple(fields), owner, tuple(grids[index] for index in range(3)),
                          tuple(tuple(row) for row in site.fields.padding),
                          tuple((row.parameter, row.byte_offset, row.data) for row in site.fields.constants),
                          tuple((row.parameter, row.byte_offset, row.byte_size) for row in site.fields.undefined))
    return bound, predicate, tuple(obligations)
