import math
from typing import Any, Dict, List, Sequence

import sympy
from sympy import S

import torch
from .virtualized import V

# Creating a SymPy zero

from .loop_body import LoopBody
from .utils import dominated_nodes
from torch._inductor.index_propagation import SymPyOps, TypedExpr


class PreservesZeros(SymPyOps):

    def __init__(self):
        
        self.count = 0
        self.output_value = None

    @staticmethod
    def load(name: str, index: sympy.Expr):
        # any load gets broadcasted.. only in prologues ! not generally applicable
        # todo - mask can be non zero!
        # breakpoint()
        return TypedExpr(S.Zero, torch.float)

    @staticmethod
    def store(*args, **kwargs):
        breakpoint()
        pass


    @staticmethod
    def indirect_indexing(*args, **kwargs):
        return sympy.S.Zero

    def __getattr__(self, name: str):
        from torch._inductor.codegen.common import OpDecompositions

        # just need to add support for bitwise shifting ops..
        count = self.count
        self.count += 1

        def inner(*args: Any, **kwargs: Any):
            
            print("Name")
            breakpoint()
            if hasattr(PreservesZeros, name):
                # how do I get the inner here ?
                m = getattr(PreservesZeros, name)
                out = m(*args, **kwargs)
                print(name, out)
                return out

            if hasattr(OpDecompositions, name):
                out = getattr(OpDecompositions, name)(*args, **kwargs)
                print(name, out)
                return out.value

            nonlocal count
            # need to use dtype propagation here
            out =  TypedExpr(sympy.Symbol(f"unknown_{count}"), torch.float)
            print(name, out)
            return out 
            # var_arguments = [
            #     a
            #     for a in itertools.chain(args, kwargs.values())
            #     if isinstance(a, IndexPropVar)
            # ]
            # if not all(v.is_symbolic for v in var_arguments):
            #     return self.fallback(name, args, kwargs)

            # return self.propagate_sympy(name, args, kwargs)

        return inner




def preserves_zero_mask(node: torch._inductor.scheduler.SchedulerNode, index_vars: Sequence[Sequence[sympy.Expr]]) -> bool:
    """
    """
    with V.set_ops_handler(PreservesZeros()):
        node._body(*index_vars)
        
    return
# breakpoint()
    try:
        with V.set_ops_handler(
            (V.get_ops_handler(), var_ranges)
        ), V.kernel.set_current_node(self):
            self._body(*index_vars)
    except Exception:
        log.fatal("Error in codegen for %s", self.node)
        raise

    # int64_dtype_nodes = [
    #     node
    #     for node in loop_body.get_nodes()
    #     if (
    #         node.target == "to_dtype"
    #         and node.args[2] == torch.int64
    #         and node not in bv.unbounded_vars
    #     )
    # ]
    # if not int64_dtype_nodes:
    #     return

    # bounds = bv.get_bounds()

    # # TODO - if dominated node of one to_dtype is not expressible in int32,
    # # we should short circuit another to_dtype node if that node also dominates
    # for node in int64_dtype_nodes:
    #     try_to_reduce_precision(
    #         node,
    #         bounds,
    #         loop_body.indirect_vars,
    #         loop_body.indexing_exprs,
    #         bv.replacement_vals,
    #     )
