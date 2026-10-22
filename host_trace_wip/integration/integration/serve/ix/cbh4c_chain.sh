#!/usr/bin/env bash
# gb300-box GPU 0 (main 10-14): cbh4c = default, bnc, vcsl_fh, vrs_fh (new head + form_held on) x 3 rounds interleaved, truly unsynced churn,
# same rules (PSI start gate, host-noise flag, 0 harness syncs, strict windows), torch.cuda.empty_cache() once after the warm-up + freeze in
# every arm (DRIVE_EMPTY_CACHE, reserved before / after recorded). One gpu0.lock hold per run (interleaves with Flash-Next's chain13).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
CB_NOSYNC=1 DRIVE_EMPTY_CACHE=1 OWN_BUSY_FRAC=0.30 SET=cbh4c ROUND=vllm/round_cb_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="default bnc vcsl_fh vrs_fh" bash serve/ix/set_chain.sh
