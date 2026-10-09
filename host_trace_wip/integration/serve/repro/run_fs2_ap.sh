#!/usr/bin/env bash
# after the 10-09 reboot: fs2 (Flash-Next V-stock, host metadata tensors as meta tensors, top-k _C externs) then the
# torch-level repros (aten_ple_repro.py, core_gaps_repro.py) on the same step280 + FZ12 tree
bash serve/repro/run_fs2.sh
bash serve/python_vllm_s280.sh serve/repro/aten_ple_repro.py; echo "aten_ple rc=$?"
bash serve/python_vllm_s280.sh serve/repro/core_gaps_repro.py; echo "core rc=$?"
