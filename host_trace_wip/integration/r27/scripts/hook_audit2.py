import triton
from triton import knobs
h = knobs.runtime.launch_enter_hook
print("triton", triton.__version__, "bare import: hook", type(h).__name__, vars(h) if hasattr(h, "__dict__") else h)
import inspect
print(inspect.getsource(type(h))[:1500])
import torch  # noqa: F401
h = knobs.runtime.launch_enter_hook
print("after torch:", vars(h) if hasattr(h, "__dict__") else h)
