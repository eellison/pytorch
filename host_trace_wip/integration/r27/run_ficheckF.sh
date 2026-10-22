#!/usr/bin/env bash
# run_ficheckF.sh FWD N (DIAGNOSIS: flashinfer#5334 fenced GDN kernel in eager and ours; as run_ficheck2.sh, with ARMV_CHECK_TWICE: eager run twice, so a row tells the reference's own nondeterminism): FlashInfer GDN prefill (vLLM's GB300 default) correctness of ours, FWD = c (Vc+VP, compiled forward) or e (V+VP,
# eager forward), N fresh processes. Every hybrid kind checked (prefill 64/512/2048, mixed 1032/2112, decode bs 4-12 walk); vLLM
# defaults otherwise (max_num_seqs default). Per process: out/ficheck_<FWD>_<k>.json; summary by scripts/nan_summary.py.
FWD=$1 N=${2:-4}
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R/build_c12 TORCH_EXTENSIONS_DIR=$R/torch_ext_c12 DRIVE_STEP_WITH_SAMPLE=1
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=
export HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_CHECK_STATE_CPU=0 ARMV_CHECK_TWICE=1 FI_FENCE=1
export INTEG_GPU_UTIL=0.5 INTEG_KV_BYTES=$(( 40 << 30 ))
if [ $FWD = c ]; then arm=Vc; export ARMV_INDUCTOR_SEAMS=1; kw='{"kv_cache_dtype": "fp8", "compilation_config": {"cudagraph_mode": "NONE"}}'
else arm=V; kw='{"kv_cache_dtype": "fp8"}'; fi
for k in $(seq 1 $N); do
  timeout 1200 bash $R/python_r27.sh $R/bench/drive_sweep27.py --arm $arm --check --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only \
    --max-model-len 16384 --batch-size $(seq -s ' ' 4 12) --decode-tokens 12 --prefill-lens 64 512 2048 --mixed 0 --ix-mixed 1024x8@1024-2048 2048x64@1024-2048 --ix-reps 4 \
    --llm-kw "$kw" --out $R/out/ficheckF_${FWD}_$k.json > $R/logs/ficheckF_${FWD}_$k.log 2>&1
  echo "ficheck $FWD $k rc=$?"
done
python3 $R/scripts/nan_summary.py $R/out/ficheckF_${FWD}_*.json
