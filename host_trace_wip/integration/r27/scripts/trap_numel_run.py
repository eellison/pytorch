# run drive_fn.py with torch.numel and Tensor.numel trapped: print the Python stack of the first call on a symbolic tensor
import runpy, sys, traceback
import torch
_seen = [0]
def wrap(f):
    def g(*a, **k):
        try:
            return f(*a, **k)
        except RuntimeError as e:
            if "symbolic" in str(e) and _seen[0] < 3:
                _seen[0] += 1
                print("TRAP_NUMEL2 stack:\n" + "".join(traceback.format_stack(limit=60)), flush=True)
            raise
    return g
torch.numel = wrap(torch.numel)
torch.Tensor.numel = wrap(torch.Tensor.numel)
torch.Tensor.nelement = wrap(torch.Tensor.nelement)
import torch.cuda._host_trace as _ht
_ht.cpp_entry = False
import faulthandler
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
