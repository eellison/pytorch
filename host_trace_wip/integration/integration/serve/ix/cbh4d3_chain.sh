#!/usr/bin/env bash
# dev-box GPU 1 (inside dev3_run.sh: CUDA_VISIBLE_DEVICES=1, HT_GPU=1, CPUs 36-71): the unsynced churn set cbh4d3 as one job:
# the bindings learn pass (this host, same build and config), then default / fullnc / bnc / vcsl / vrs x 3 runs, arm order rotated per
# round; no syncs in the measured windows (async scheduling as shipped), wall ms/token; ours with freeze, both learnfast helpers,
# strict windows, actual max_seq_len. Own-core gate at 25% busy on 36-71. dev-box rows are reported on their own.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export CB_NOSYNC=1 HT_CPUS=36-71 OWN_BUSY_FRAC=0.25 STOCK_LINE=c5h SET=cbh4d3
[ -s vllm/out/bindings_cbh4d3.pkl ] || ARMS=learn R=0 bash vllm/round_cb_line.sh
ALL=(default fullnc bnc vcsl vrs)
for r in 1 2 3; do
  k=$(( (r - 1) % ${#ALL[@]} )); order=("${ALL[@]:$k}" "${ALL[@]:0:$k}")
  ARMS="${order[*]}" R=$r bash vllm/round_cb_line.sh
done
echo "CBH4D3_DONE $(ls vllm/out/t_cbh4d3_*_r[123].json 2>/dev/null | wc -l)/15"
