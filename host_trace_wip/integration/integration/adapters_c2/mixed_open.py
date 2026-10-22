# Mixed driver (armF/mixed/driveM.py) on the open GEMM: SGLang's bf16 GEMM backend initialized (armF/tgv/drive_bf16.py)
# and, with MIXM_F=1, the TGV-capable adapter (ARMF_ADAPTER_DIR, adapters_c2/tgv) standing in for mixed/adapter_m.py.
#   launcher adapters_c2/mixed_open.py <driveM args...>
import os
import runpy
import sys

S = "/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/armF"
if os.environ.get("MIXM_F") == "1":
    sys.path.insert(0, os.environ["ARMF_ADAPTER_DIR"])
    import adapter  # before sglang: arms the CuTe route before any CuTe compile

    sys.modules["adapter_m"] = adapter
    print("mixed_open: adapter_m =", adapter.__file__, flush=True)
sys.argv = [f"{S}/tgv/drive_bf16.py", f"{S}/mixed/driveM.py"] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
