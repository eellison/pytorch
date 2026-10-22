#!/usr/bin/env bash
# gb300-box GPU 2: section-refusal validation (one forward), as the dev-box checks: on Triton GDN with vLLM-default max_num_seqs, both forwards (Vc+VP,
# V+VP), every hybrid kind (prefill 64/512/2048, mixed 1032/1088/2112, decode bs 4-20 walk), eager twice, sections counted per entry.
# Outputs out/secval_{c,e}.json; summary by scripts/nan_summary.py. The first r27 validation on that box.
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R/build_c12 TORCH_EXTENSIONS_DIR=$R/torch_ext_c12 DRIVE_STEP_WITH_SAMPLE=1
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=
export HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_CHECK_STATE_CPU=0 ARMV_CHECK_TWICE=1 ARMV_GC_FREEZE=1
export INTEG_GPU_UTIL=0.5 INTEG_KV_BYTES=$(( 40 << 30 ))
for fwd in c; do
  if [ $fwd = c ]; then arm=Vc; extra="ARMV_INDUCTOR_SEAMS=1"; kw='{"kv_cache_dtype": "fp8", "gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "NONE"}}'
  else arm=V; extra=""; kw='{"kv_cache_dtype": "fp8", "gdn_prefill_backend": "triton"}'; fi
  env $extra timeout 2400 bash $R/python_r27.sh $R/bench/drive_sweep27.py --arm $arm --check --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only \
    --max-model-len 16384 --batch-size $(seq -s ' ' 4 20) --decode-tokens 12 --prefill-lens 64 512 2048 --mixed 0 \
    --ix-mixed 1024x8@1024-2048 1024x64@1024-2048 2048x64@1024-2048 --ix-reps 4 --llm-kw "$kw" --out $R/out/secval_$fwd.json > $R/logs/secval_$fwd.log 2>&1
  echo "dev3ck $fwd rc=$?"
done
python3 $R/scripts/nan_summary.py $R/out/secval_c.json; grep -c "ARMV_SECTION " $R/logs/secval_c.log; grep -c "refused by the adapter" $R/logs/secval_c.log; grep -c "ARMV_SECTIONS_ADDED" $R/logs/secval_c.log
for f in $R/out/secval_c.json; do python3 -c "
import json; d=json.load(open('$f')); print('$f', 'sections:', {k:e.get('sections') for k,e in d['V']['entries'].items() if e.get('sections')}, 'unsound', d['V'].get('unsound_learned'))"; done
