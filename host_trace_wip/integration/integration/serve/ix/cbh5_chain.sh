#!/usr/bin/env bash
# cbh5 (main 10-15, launch only on main's go): default, vcsl_h7 (head ms7, split off), vcsl_h7s (head ms7, executor-level split threshold 64)
# x 3 rounds interleaved, truly unsynced churn C = 16/64/128, same rules as cbh4b/c (PSI start gate, host-noise flag, 0 harness syncs,
# strict windows), torch.cuda.empty_cache() once after the warm-up in every arm; one lock hold per run.
#   GPU=0 CPUS=0-35 bash serve/ix/cbh5_chain.sh      (or GPU=2 with its CPU range)
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
: ${GPU:?} ${CPUS:?}
CB_NOSYNC=1 DRIVE_EMPTY_CACHE=1 OWN_BUSY_FRAC=0.30 SET=cbh5 ROUND=vllm/round_cb_line.sh GPU=$GPU CPUS=$CPUS LINE=c5h ARMS="default vcsl_h7 vcsl_h7s" bash serve/ix/set_chain.sh
