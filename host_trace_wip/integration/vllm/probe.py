import sys, torch
import os; sys.path.insert(0, os.environ["INTEG_ARMV"])
import torch.cuda._host_trace_replay as R
from torch.cuda._host_trace_harvest import HarvestProvider
print("torch", torch.__file__, "R", R.__file__)
hv = R.HostTraceReplay.__init__.__code__.co_varnames
for k in ("static_prefix", "static_shapes", "learn_in_learning_variants", "redispatch_learners", "oob_learn"):
    print("HostTraceReplay", k, k in hv)
pv = HarvestProvider.__init__.__code__.co_varnames
for k in ("lendable", "lend_alias_groups"):
    print("HarvestProvider", k, k in pv)
print("_HostTraceBound", hasattr(torch._C, "_HostTraceBound"))
import inspect
print("HostTraceReplay sig", inspect.signature(R.HostTraceReplay.__init__))
print("HarvestProvider sig", inspect.signature(HarvestProvider.__init__))
import vllm; print("vllm", vllm.__version__, vllm.__file__)
import adapter, adapter_cpp
print("adapter STATIC_PREFIX", adapter.STATIC_PREFIX, "BOUND", adapter_cpp.BOUND)
if os.environ.get("CAND_FORK") == "1":
    import flashinfer.trtllm_trace, vllm_ops_trace
    print("fork", flashinfer.trtllm_trace.__file__, "ops", vllm_ops_trace.__file__)
print("HarvestProvider save/load", hasattr(HarvestProvider, "save"), hasattr(HarvestProvider, "load"))
