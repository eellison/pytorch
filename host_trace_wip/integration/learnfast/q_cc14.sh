#!/usr/bin/env bash
# 10-14 GPU 3, lowest priority (gpulock.sh yields to any other lane's waiter). Closure (storage form) + redo_context with
# stand-ins only for the per-step arguments (LAZY static prefix), CuTe intercept resolve:
#   cc9c (9B --check smoke, 6 prefill lengths) -> ccm_chk (MiMo --check, 11 decode-curve points) -> cc27a (27B learn 1..400) -> ccm_dc (MiMo decode curve, timing of warm-up)
cd $(dirname $0)
CC="DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_FNRESOLVE=1 DIAG27_OPSPAN=1 DIAG27_SET=redo_resolves_arguments=1 LF_GPU=3 LF_CPUS=108-143"
M9="ARMV_SECTIONS=1 MODEL=/data/eellison/models/Qwen3.5-9B MODEL_EXTRA=--trust-remote-code KV_GIB=4 INTEG_GPU_UTIL=0.08"
./gpulock.sh 3 36000 env $M9 $CC CHECK_TIMEOUT=900 ./run27checkm.sh cc9c "60 61 64 65 70 80 128 129" > logs/cc9c.log 2>&1
echo done >> logs/cc9c.log
# the smoke gates the rest: run27checkm's rc=0, no non-bitwise row, no redo_context error
python3 - <<'PY' || { echo "cc9c smoke failed; stopping" >> logs/q_cc14.out; exit 1; }
import json, re, sys
log = open("logs/cc9c.log", errors="replace").read()
rows = re.findall(r"^\s+\(('(?:decode|prefill|mixed)'.*?)\) \d+$", log, re.M)
bad = [r for r in rows if "False" in r]
errs = [l for l in open("out/cc9c.diag.jsonl") if "redo_context_error" in l]
print("cc9c rows", len(rows), "bad", bad[:3], "redo_context_errors", len(errs))
sys.exit(0 if "rc=0" in log and rows and not bad and not errs else 1)
PY
./gpulock.sh 3 36000 env $CC REDOCTX_LAZY=1 RUN_TIMEOUT=1120 ./mimorc.sh ccm_chk chk > logs/ccm_chk.log 2>&1
echo done >> logs/ccm_chk.log
./gpulock.sh 3 36000 env ARMV_SECTIONS=1 $CC LEARN_TIMEOUT=1100 ./run27.sh cc27a 1 400 > logs/cc27a.log 2>&1
echo done >> logs/cc27a.log
./gpulock.sh 3 36000 env $CC REDOCTX_LAZY=1 RUN_TIMEOUT=1120 ./mimorc.sh ccm_dc dc > logs/ccm_dc.log 2>&1
echo done >> logs/ccm_dc.log
