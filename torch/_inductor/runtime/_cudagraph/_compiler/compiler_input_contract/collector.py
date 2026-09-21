"""Collect boxed input facts while the final Inductor wrapper IR is alive."""

import sympy
import torch
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import InputContract, IntegerRange, TensorInput
from torch._inductor import ir
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.runtime.cudagraph_arg_mapping import IntExpr
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._sympy.numbers import int_oo


class CompilerInputDeclined(ValueError):
    pass


def compiler_inputs(wrapper: PythonWrapperCodegen) -> InputContract:
    if type(wrapper) is not PythonWrapperCodegen:
        raise CompilerInputDeclined("Expected the final Python wrapper IR")
    sizevars = getattr(V.graph, "sizevars", None)
    if not isinstance(sizevars, SizeVarAllocator):
        raise CompilerInputDeclined("Compiler inputs require the live graph size variables")
    environment = sizevars.shape_env
    guards = tuple(environment.guards)
    try:
        if environment.deferred_runtime_asserts:
            raise CompilerInputDeclined("Deferred compiler assertions need an explicit availability contract")
        names = tuple(wrapper.get_graph_input_names())
        inputs = wrapper.get_graph_inputs()
        if (not names or any(type(name) is not str for name in names)
                or len(names) != len(set(names)) or any(name not in inputs for name in names)):
            raise CompilerInputDeclined("Expected unique compiler input names in boxed order")
        bindings = wrapper._cudagraph_integer_bindings()
        if bindings is None:
            raise CompilerInputDeclined("Boxed integers lost their compiler symbol bindings")
        _, expressions = bindings
        kinds, symbols, tensors, ranges = [], {}, [], []
        for index, name in enumerate(names):
            value = inputs[name]
            if isinstance(value, sympy.Expr):
                if value.is_integer is not True:
                    raise CompilerInputDeclined("Boxed scalar is not an integer compiler symbol")
                if free_unbacked_symbols(value):
                    raise CompilerInputDeclined("Unbacked compiler inputs need an explicit availability contract")
                value = sizevars.simplify(value)
                if (not isinstance(value, sympy.Symbol) or value.is_integer is not True
                        or free_unbacked_symbols(value)):
                    raise CompilerInputDeclined("Each boxed integer requires an integer compiler symbol")
                domain = environment.var_to_range.get(value)
                if (domain is None or not isinstance(domain.lower, sympy.Integer)
                        or domain.lower < 1 or not sizevars.statically_known_gt(value, 0)):
                    raise CompilerInputDeclined("Boxed integer lacks an inherited positive range")
                if domain.upper == int_oo:
                    upper = None
                elif isinstance(domain.upper, sympy.Integer) and domain.upper >= domain.lower:
                    upper = int(domain.upper)
                else:
                    raise CompilerInputDeclined("Boxed integer has an unsupported inherited upper bound")
                symbols[value] = expressions[value]
                ranges.append(IntegerRange(index, int(domain.lower), upper))
                kinds.append("integer")
            else:
                while type(value) in (ir.TensorBox, ir.StorageBox):
                    value = value.data
                if type(value) not in (ir.InputBuffer, ir.DonatedBuffer):
                    raise CompilerInputDeclined("Tensor input requires an original compiler InputBuffer")
                layout = value.get_layout()
                if (type(layout) is not ir.FixedLayout or layout.offset != 0
                        or type(layout.dtype) is not torch.dtype or layout.device.type != "cuda"
                        or type(layout.device.index) is not int or layout.device.index < 0):
                    raise CompilerInputDeclined("Tensor input requires a fixed CUDA layout with zero offset")
                tensors.append((index, layout))
                kinds.append("tensor")

        def dimension(value):
            if type(value) is not int and not isinstance(value, sympy.Expr):
                raise CompilerInputDeclined("Input layout dimension is not a compiler integer expression")
            value = sizevars.simplify(value)
            if isinstance(value, sympy.Integer):
                return int(value)
            expression = wrapper._cudagraph_user_grid(value, symbols)
            if type(expression) is not IntExpr:
                raise CompilerInputDeclined("Input layout exceeds supported compiler integer expressions")
            return expression

        devices = {layout.device.index for _, layout in tensors}
        if len(devices) != 1:
            raise CompilerInputDeclined("Compiler Tensor inputs require one CUDA device")
        tensor_inputs = tuple(
            TensorInput(index, layout.dtype, tuple(map(dimension, layout.size)),
                        tuple(map(dimension, layout.stride))) for index, layout in tensors
        )
        return InputContract(tuple(kinds), tensor_inputs, tuple(ranges), next(iter(devices)))
    finally:
        if tuple(environment.guards) != guards:
            raise RuntimeError("Compiler input collection changed ShapeEnv guards")
