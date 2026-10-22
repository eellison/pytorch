#!/usr/bin/env bash
# chain_ag2.sh: atengaps recheck (untimed, correctness only): int6 snap/step337_ag + build_cpp19 (cat of one part + bs-1 bool->bf16
# copy fixes, on the step334_ss line). V arm, diagnosis config (Triton GDN + FN_QSA_DET), shared-expert stream ON with the
# side-streams port overlay, decode (bs 8 / 1 / 64) / prefill / mixed, 2 rounds, every call checked; gb300-box GPU 3.
# Queued behind learnfast's Flash-Next ladder and the memory lane's jobs: starts only once logs/mon/ag2.go exists.
F=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
cd $F
echo $$ > $F/logs/mon/chain_ag2.pid
while [ ! -f $F/logs/mon/ag2.go ]; do sleep 60; done
echo "$(date +%T) chain_ag2 go"
export FN_WT=$I/snap/step337_ag FN_INSTALL=$I/build_cpp19/install TORCH_EXTENSIONS_DIR=$F/cache/torch_ext_c19
FN_QSA_DET=1 SGL_OPS_TRACE_DIR=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/sidestreams/port_overlay CMD_TIMEOUT=1200 \
  bash gpu3.sh fn_v7 bash run_fn_arm.sh fn_v7 V --rounds 2 --kinds decode prefill mixed --llm-kw '{"gdn_prefill_backend": "triton"}'
echo "$(date +%T) chain_ag2 done"
