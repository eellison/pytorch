#!/usr/bin/env bash
# After diag_open_c2.sh: the combined open variant with PDL tracing / programmatic edges off (does a PDL edge between
# a traced-impl launch and the fork's kernels explain the NaNs?).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
until grep -q "^ALLDONE" vllm/logs/diag_open_c2.log; do sleep 60; done
CUDA_VISIBLE_DEVICES=1 flock -x /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock env CAND_LINE=cpp CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_TRACED_IMPLS=1 \
  ARMV_HT_SET="torch.cuda._host_trace.trace_pdl=0,torch.cuda._host_trace_capture.keep_programmatic_edges=0" bash -c '
  read used total < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 1 | wc -l; echo); u=0.16; echo "util $u"
  INTEG_GPU_UTIL=$u exec timeout 2400 taskset -c 72-107 bash python_vllm_cand2.sh vllm/bench/drive.py --arm V --check --batch-size 5 8 64 --prefill-lens 64 --mixed 0 --out vllm/out/diag_both_nopdl.json' > vllm/logs/diag_both_nopdl.log 2>&1
echo "$(date +%T) diag_both_nopdl rc=$?"
echo ALLDONE
