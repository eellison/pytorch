from typing import Callable

from torch._inductor.runtime._cudagraph._compiler.continuation import BoundDispatchHelper
from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_dispatch_evaluation.selection import DispatchInputs, DispatchSelection
from torch._inductor.runtime._cudagraph._compiler.values import ScalarValue


def select_with_policy(inputs: DispatchInputs, evaluate: Callable[[BoundDispatchHelper], ScalarValue]) -> DispatchSelection:
    dispatch = inputs.components.dispatch
    evaluated, values = [], []

    def compute(bound):
        result = evaluate(bound)
        if type(result) is not ScalarValue or result.llvm_type != bound.helper.result_type:
            raise ValueError("Dispatch helper changed its exact scalar result type")
        evaluated.append(bound.helper.symbol)
        values.append((bound.role, bound.index, result))
        return result

    predicates = [bound for bound in dispatch.consumers if bound.site is None]
    if len(predicates) != 1 or predicates[0].role != "predicate":
        raise ValueError("Dispatch requires its exact original root predicate")
    predicate = compute(predicates[0])
    if predicate.llvm_type != "i1":
        raise ValueError("Dispatch predicate must return its original i1 value")
    arm = bool(predicate.integer(signed=False))
    sites = [site for site in dispatch.joined.sites if site.source.arm is arm]
    if len(sites) != 1:
        raise ValueError("Dispatch predicate does not select exactly one source site")
    site = sites[0]
    selected = {}
    for bound in dispatch.consumers:
        if bound.site is site.source:
            selected[bound.role, bound.index] = compute(bound)

    def integer(role, index, llvm_type):
        value = selected[role, index]
        if value.llvm_type != llvm_type:
            raise ValueError("Dispatch launch field changed its exact compiler width")
        return value.integer()

    grid = tuple(integer("grid", axis, "i32") for axis in range(3))
    block = tuple(integer("block", axis, "i32") for axis in range(3))
    shared = integer("shared", 0, "i64")
    kernel_smem = integer("kernel_smem", 0, "i64")
    for index, diagnostic in enumerate(site.source.diagnostics):
        value = selected["diagnostic", index]
        if value.llvm_type != "i1":
            raise ValueError("Dispatch diagnostic changed its original predicate width")
        if bool(value.integer(signed=False)) is not diagnostic.expected:
            raise ValueError(f"Original compiler shared-memory diagnostic rejected dispatch: {diagnostic.kind}")
    if any(value <= 0 for value in (*grid, *block)) or not 0 <= shared < 2**32 or kernel_smem < 0:
        raise ValueError("Dispatch launch dimensions or shared-memory values are invalid")
    owners = inputs, site, grid, block, shared, kernel_smem, tuple(evaluated), tuple(values)
    result = DispatchSelection(*owners, owners)
    result.check()
    return result
