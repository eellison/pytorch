#!/usr/bin/env bash
# GPU 2 (10-13 night): FlashInfer-GDN timed sets on vLLM defaults: n27wf (cold-cache startup runs, offline learns, 3 rounds x 5 arms), then
# KV pool 64 GiB: vLLM's default max_num_seqs 1024 needs one Mamba cache block per sequence (40 GiB gave 835 blocks and every vLLM arm refused to start).
# n27cbf (learns, 3 rounds x 5 arms). Our arms: GDN traced (ARMV_GDN_WS), section entries refused at creation (no_sections), gc.freeze.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
COLD=1 R27_KV_GIB=64 R27_TIMEOUT=7200 R27_ARMS="default fullnc vcvp breakable vsvp" bash timing_gpuN3.sh round_27wf_line.sh n27wf 2 36-71 >> logs/timing_n27wf.out 2>&1
R27_KV_GIB=64 R27_TIMEOUT=7200 R27_ARMS="default fullnc vcvp breakable vsvp" bash timing_gpuN3.sh round_27cbf_line.sh n27cbf 2 36-71 >> logs/timing_n27cbf.out 2>&1
