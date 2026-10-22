# q8p diagnosis: run drive_fn.py (arm V, ARMV_SOUND + ARMV_PORTS) with torch.cuda._host_trace._TRACED_IMPLS lookups logged for the
# vLLM _C ops, plus the trace's in_aten / forced state at each lookup site, to see why the registered ports are not consulted.
import collections
import runpy
import sys

import torch
import torch.cuda._host_trace as ht
import torch.cuda._host_trace_tape as T

LOG = collections.Counter()


class _Impls(dict):
    def get(self, k, d=None):
        v = super().get(k, d)
        if "_C" in str(k):
            LOG[(str(k), v is not None)] += 1
        return v


ht._TRACED_IMPLS = _Impls(ht._TRACED_IMPLS)
_orig_reg = ht.register_traced_impl


def _reg(op, f):
    LOG[("registered", str(op))] += 1
    ht._TRACED_IMPLS[op] = f


ht.register_traced_impl = _reg
_orig_op = T._Trace.__torch_dispatch__ if hasattr(T._Trace, "__torch_dispatch__") else None
import atexit


@atexit.register
def _dump():
    print("PORTS_DBG traced_impls flag", ht.traced_impls, flush=True)
    for k, n in sorted(LOG.items(), key=str):
        print("PORTS_DBG", k, n, flush=True)


sys.argv = ["drive_fn.py"] + sys.argv[1:]
runpy.run_path("/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/vllm/bench/drive_fn.py", run_name="__main__")
