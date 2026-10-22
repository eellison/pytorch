# Diagnostic: run a driver with HarvestProvider's budget_unit default set to "group", the overlay's budget semantics
# (1024 groups; int6's default is 1024 keys, cuBLAS harvests counted). Edits the default in place so the adapter's
# co_varnames probes still see the real signature. Usage: group_budget.py <driver.py> [args]
import inspect, runpy, sys

from torch.cuda._host_trace_harvest import HarvestProvider

f = HarvestProvider.__init__
params = list(inspect.signature(f).parameters.values())
positional = [p for p in params if p.kind is p.POSITIONAL_OR_KEYWORD and p.default is not p.empty]
if f.__kwdefaults__ and "budget_unit" in f.__kwdefaults__:
    f.__kwdefaults__["budget_unit"] = "group"
else:
    f.__defaults__ = tuple("group" if p.name == "budget_unit" else d for p, d in zip(positional, f.__defaults__))
print("group_budget: budget_unit default", inspect.signature(f).parameters["budget_unit"].default, flush=True)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
