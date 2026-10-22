#!/usr/bin/env bash
# run_fn_arm.sh TAG ARM [fn_interleave args...]: Qwen3.8-Flash-Next NVFP4 under this lane's armV adapter with every traced
# call checked against a fresh eager reference, shapes interleaved (scripts/fn_interleave.py). Line: python_fn.sh (int6
# snap/step331 + build_cpp14, the trtllm_moe_cpp overlay and extensions). Flags as the MiMo lane's run_mimo_arm.sh
# (ARMV_SOUND=1, VP + VP_STOCK, CuTe observation, ports, gc.freeze after warm-up), without MiMo's backport / DeepGEMM pin.
# torch._C._host_trace_section_entries is left at its default; the adapter refuses sections at creation and reports
# sections_added (a run with sections > 0 is invalid). ARM: Vc (compiled forward + VP), V (eager forward + VP_STOCK).
set -u
TAG=$1; ARM=$2; shift 2
F=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
export ARMV_GC_FREEZE=1 INTEG_ARMV=$F/armV
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_CHECK_TWICE=${ARMV_CHECK_TWICE:-1}
export ARMV_VP=${ARMV_VP:-1} ARMV_VP_UVA=0 ARMV_VP_STOCK=1
# check state rows saved in host memory (r27 knob): three device copies of 64 requests' GDN state OOMed the GPU (fn_v1)
export ARMV_CHECK_STATE_CPU=${ARMV_CHECK_STATE_CPU:-1}
if [ "$ARM" = Vc ]; then
  export ARMV_INDUCTOR_SEAMS=1 HOSTTRACE_TORCH_DISABLE_CACHES=0
fi
export ARMV_DUMP_VARIANTS=${ARMV_DUMP_VARIANTS:-$F/out/${TAG}_variants.json}
export INTEG_GPU_UTIL=${INTEG_GPU_UTIL:-0.88} INTEG_KV_BYTES=${INTEG_KV_BYTES:-$(( 40 << 30 ))}
cd $F
# FN_MODEL: another model through the same driver (e.g. Qwen3-8B for a no-change check; no --language-model-only then)
MODEL=${FN_MODEL:-/data/eellison/models/Qwen3.8-Flash-Next-NVFP4}
LMO=--language-model-only; [ -n "${FN_MODEL:-}" ] && LMO=
bash $F/python_fn.sh $F/scripts/fn_interleave.py --arm $ARM --check --model $MODEL \
  --trust-remote-code $LMO "$@" --out $F/out/$TAG.json
rc=$?
echo "$TAG rc=$rc"
exit $rc
