#!/usr/bin/env bash
# ours-uncompiled check rerun with both helpers (no_sections, no_lib_redo), 10-13
# decode bs 4 -> 20 (one variant across sizes), prefill 64, mixed 1024x8 / 1024x64 at KV 1-2K; outputs, GDN state, sampler tokens compared
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R/build_c12 TORCH_EXTENSIONS_DIR=$R/torch_ext_c12 DRIVE_STEP_WITH_SAMPLE=1
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=
export HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_CHECK_KINDS=decode,mixed ARMV_CHECK_STATE_CPU=0
export INTEG_GPU_UTIL=0.5 INTEG_KV_BYTES=$(( 40 << 30 ))
bash $R/python_r27.sh $R/bench/drive_sweep27.py --arm V --check --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only \
  --max-model-len 16384 --batch-size $(seq -s ' ' 4 20) --decode-tokens 20 --prefill-lens 64 --mixed 0 --ix-mixed 1024x8@1024-2048 1024x64@1024-2048 --ix-reps 4 \
  --llm-kw '{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton"}' --out $R/out/vsvp_check2_c12.json > $R/logs/vsvp_check2_c12.log 2>&1
echo "vsvp_check rc=$?"
python3 $R/b200_bundle/tools/verdict.py $R/out/vsvp_check2_c12.json | head -20
python3 - <<'P'
import json, collections
d = json.load(open("/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/out/vsvp_check2_c12.json"))
print(collections.Counter((r["kind"], r["bitwise"], r.get("state_bitwise"), r.get("vp_tokens_equal")) for r in d["V"]["checks"]))
print({k: d["V"].get(k) for k in ("traces", "replays", "eager", "variants", "unsound_learned")})
P
