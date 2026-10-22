#!/usr/bin/env bash
# vLLM-level repro of the cold-cache retry loop: Qwen3-8B Vc (armV_stock_fast_m3, sound + ports + Inductor seams) with fresh vLLM and
# Inductor caches, small workload; DIAG_AUTOTUNE logs every CachingAutotuner tuning / run (launcher count, under a trace or not).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
C=$(mktemp -d /tmp/vc_cold_XXXX)
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.5 INTEG_KV_BYTES=96636764160
env CAND_LINE=${VC_LINE:-c5g} CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
    ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m3 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 DIAG_AUTOTUNE=1 \
    VLLM_CACHE_ROOT=$C/vllm TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton \
    timeout 1500 bash python_vllm_cand2.sh vllm/bench/drive_fn.py --arm Vc --model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 64 --prefill-lens 64 --mixed 0 --ix-mixed 1024x64@1024-2048 2048x128@6000-8000 --ix-reps 4 --out serve/out/vc_cold.json > serve/logs/vc_cold_run.log 2>&1
echo "vc_cold rc=$?"
grep "DIAG_AT" serve/logs/vc_cold_run.log | head -40
python3 -c "
import json; V=json.load(open('serve/out/vc_cold.json'))['V']; print('VCCOLD', {k: V[k] for k in ('traces','replays','eager','variants')}); [print('VCCOLD_ENTRY', k[:12], e['traces'], e['replays'], e['declines'][:2]) for k,e in V['entries'].items()]"
rm -rf $C
