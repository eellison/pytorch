# armF/mixed/driveM.py (mixed chunk + decode batches, decode-through-extend) on the stock adapter: with MIXM_F=1,
# sglang_stock/adapter/adapter_cpp.py stands in for mixed/adapter_m.py.   launcher mixed_stock.py <driveM args...>
import os
import runpy
import sys

A = "/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/armF"
if os.environ.get("MIXM_F") == "1":
    sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapter"), A]
    import adapter_cpp  # before sglang: arms the CuTe route before any CuTe compile

    sys.modules["adapter_m"] = adapter_cpp
sys.argv = [f"{A}/mixed/driveM.py"] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
