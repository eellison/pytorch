#!/usr/bin/env bash
# chain_ag.sh: the atengaps lane's index_select > 2**31-span fix (int6 snap/step335_ag + build_cpp17): repro_ple_conv.py, then the
# V arm (diagnosis config: Triton GDN + deterministic QSA; step335_ag was recreated 06:25 on step334_ss, so the shared-expert
# stream is ON (vLLM default) with the side-streams port overlay; decode / prefill / mixed, 2 rounds, every call checked), on
# gb300-box GPU 3 (gpu3.lock per job).
F=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
cd $F
export FN_WT=$I/snap/step335_ag FN_INSTALL=$I/build_cpp17/install TORCH_EXTENSIONS_DIR=$F/cache/torch_ext_c17
CMD_TIMEOUT=600 bash gpu3.sh r_ple6 bash python_fn.sh tests/repro_ple_conv.py
sleep 60
FN_QSA_DET=1 SGL_OPS_TRACE_DIR=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/sidestreams/port_overlay CMD_TIMEOUT=1200 bash gpu3.sh fn_v6 bash run_fn_arm.sh fn_v6 V --rounds 2 \
  --kinds decode prefill mixed --llm-kw '{"gdn_prefill_backend": "triton"}'
echo "$(date +%T) chain_ag done"
