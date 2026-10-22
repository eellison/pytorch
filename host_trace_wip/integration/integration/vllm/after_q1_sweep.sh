#!/usr/bin/env bash
# After q1 (gpu0_chain.sh prints Q1_DONE or "q1 skipped"): the InferenceX-method sweep (GPU 0, one hold per point), with an in-server
# check burst after each V-stock point (vllm/ix_run_check.py --check-burst 2).
# Line: candidate 4 (CAND_LINE=c4: step290 + build_cpp9, has F24k = the F26 fix) if armV_stock_fast is bitwise there
# (stockfast_c4_check, GPU 3); else attn+lc (c8l install, lacks F24k: F26 possible, the check bursts watch for it) with armV_stock_fast
# if bitwise there (stockfast_check_gpu0_c2), else with the serve lane's checked serve/armV_stock.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while ! grep -q "Q1_DONE\|q1 skipped" vllm/logs/gpu0_chain.out 2>/dev/null; do sleep 120; done
while ! grep -q ALLDONE vllm/logs/q3_stockfast_check_gpu0_c2.log 2>/dev/null; do sleep 60; done
while ! grep -q ALLDONE vllm/logs/q3_stockfast_c4_check.log 2>/dev/null; do sleep 60; done
bitwise() {  # bitwise <drive check json> <range json>: prints 1 if both exist and every check is bitwise
  python3 - "$1" "$2" <<'PY'
import json, os, sys
ok = True
d, r = sys.argv[1], sys.argv[2]
if os.path.exists(d):
    ch = json.load(open(d))["V"]["checks"]
    ok &= len(ch) > 0 and all(c["bitwise"] for c in ch)
else:
    ok = False
if os.path.exists(r):
    c = json.load(open(r))["checks"]
    ok &= c["n"] > 0 and c["bitwise"] == c["n"]
else:
    ok = False
print(int(ok))
PY
}
M=vllm/ix_master_attnlc.yaml
if [ "$(bitwise vllm/out/c2_stockfast_c4_Vcheck.json vllm/out/c2_stockfast_c4_range.json)" = 1 ]; then
  SKUS="qwen38bp4-bf16-gb300-vllm qwen38bnc4-bf16-gb300-vllm qwen38bvs4-bf16-gb300-vllm"
  echo "line c4 (armV_stock_fast bitwise on candidate 4)"
else
  SKUS="qwen38bpl-bf16-gb300-vllm qwen38bncl-bf16-gb300-vllm qwen38bvsl-bf16-gb300-vllm"
  echo "c4 check not bitwise: line attnlc"
  if [ "$(bitwise vllm/out/c2_stockfast_attnlc_Vcheck.json vllm/out/c2_stockfast_attnlc_range.json)" != 1 ]; then
    sed 's#integration/vllm/armV_stock_fast#integration/serve/armV_stock#' vllm/ix_master_attnlc.yaml > vllm/ix_master_attnlc_checked.yaml
    M=vllm/ix_master_attnlc_checked.yaml
    echo "armV_stock_fast not bitwise on attnlc: V-stock uses serve/armV_stock"
  fi
fi
sed -i "s#--master vllm/ix_master_attnlc[a-z_]*.yaml#--master $M#" vllm/ix_sweep_attnlc.sh
echo "sweep master $M skus $SKUS"
SKUS="$SKUS" POINTS="1k1k:1 1k1k:4 1k1k:16 1k1k:64 8k1k:1 8k1k:4 8k1k:16 8k1k:64" bash vllm/ix_sweep_attnlc.sh vl1 > vllm/logs/ix_vl1.log 2>&1
echo SWEEP_DONE
