#!/usr/bin/env bash
# chain_p.sh: the nested-compile fix (b) prototype (FN_WITNESS_INFRA=1, scripts/witness_infra.py): (1) the regression test and
# the repro; (2) Flash-Next Vc decode (diagnosis config, as fn_vc2) with the prototype; (3) Qwen3-8B Vc decode without and
# with it (no-change check). One gpu3.lock hold per job; 60 s gaps for the higher-priority lanes.
F=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
cd $F
FN_WITNESS_INFRA=1 HOSTTRACE_TORCH_DISABLE_CACHES=0 CMD_TIMEOUT=600 bash gpu3.sh t_nc2 bash python_fn.sh tests/test_nested_compile_trace.py
sleep 60
FN_WITNESS_INFRA=1 HOSTTRACE_TORCH_DISABLE_CACHES=0 CMD_TIMEOUT=600 bash gpu3.sh r_nc2 bash python_fn.sh tests/repro_nested_compile.py
sleep 60
FN_WITNESS_INFRA=1 FN_QSA_DET=1 VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 CMD_TIMEOUT=1200 bash gpu3.sh fn_vc5 bash run_fn_arm.sh fn_vc5 Vc \
  --rounds 1 --kinds decode --llm-kw '{"gdn_prefill_backend": "triton"}'
for t in q8_vc_base q8_vc_wi; do
  sleep 60
  wi=0; [ $t = q8_vc_wi ] && wi=1
  FN_WITNESS_INFRA=$wi FN_MODEL=/data/eellison/models/Qwen3-8B INTEG_KV_BYTES=$(( 16 << 30 )) CMD_TIMEOUT=1200 bash gpu3.sh $t \
    bash run_fn_arm.sh $t Vc --rounds 2 --kinds decode
done
echo "$(date +%T) chain_p done"
