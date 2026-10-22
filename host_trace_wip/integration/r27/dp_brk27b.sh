#!/usr/bin/env bash
# dp_brk27b.sh (10-15): the same decode probe for vLLM eager (enforce_eager: the reference), vLLM default and ours (vcvp, vsvp),
# compared with dp_brk27's default-nc / breakable. From dp_brk27.sh, after Flash-Next's dp_brk.sh): 27B breakable decode check before the n27dg timed rounds. Greedy decode of seeded
# prompts (24 new tokens): bs 31 (padded to 32) / 32 / 64 at 1K prompts and bs 8 (unpadded) / 9 (padded to 16) at 8K prompts; n27d
# config (max_num_seqs 128, Triton GDN prefill, fp8 KV). default-nc twice (its run-to-run spread) and breakable; token ids per sequence
# to out/dp27/, compared at the end. Untimed; GPU 2, one gpu2.lock hold per run (LANES.md), CPUs 36-71.
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
G=2
mkdir -p $R27/out/dp27
export DRIVE_STEP_WITH_SAMPLE=1 R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R27/build_c12 TORCH_EXTENSIONS_DIR=$R27/torch_ext_c12
export R27_OVERLAY=$S DRIVE_DECODE_PROBE="31:1024 32:1024 64:1024 8:8192 9:8192" DRIVE_LOGIT_PROBE_DIR=$R27/out/dp27
KW='{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton"}'
one() {  # tag arm env...
  local tag=$1 arm=$2; shift 2
  exec 9>$E/gpu$G.lock
  if ! flock -x 9; then echo "$(date +%T) flock gpu$G.lock failed; aborting"; exit 1; fi
  while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 20480 ]; do sleep 30; done
  echo "$(date +%T) start $tag"
  env "$@" DRIVE_LOGIT_PROBE_TAG=$tag CUDA_VISIBLE_DEVICES=$G INTEG_GPU_UTIL=0.9 INTEG_KV_BYTES=$((72 << 30)) timeout 1800 taskset -c 36-71 \
    bash $R27/python_r27.sh $R27/bench/drive_sweep27.py --arm $arm --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only \
    --max-model-len 16384 --batch-size 1 --decode-tokens 12 --prefill-lens 64 --mixed 0 --llm-kw "$KW" --out $R27/out/dp27/$tag.json > $R27/logs/dp27_$tag.log 2>&1
  echo "$(date +%T) end $tag rc=$?"
  exec 9>&-
}
one eager eager
one default default HOSTTRACE_TORCH_DISABLE_CACHES=0
one vsvp V ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0 ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=prefill,mixed HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_REFREEZE=1 ARMV_BINDINGS_LOAD=$R27/bindings/n27d_v.pkl
KW='{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "NONE"}}'
one vcvp Vc ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0 ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=prefill,mixed HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_REFREEZE=1 ARMV_INDUCTOR_SEAMS=1 ARMV_BINDINGS_LOAD=$R27/bindings/n27d.pkl
python3 - <<'PY' | tee $R27/out/dp27/CMP_vs_eager.txt
import json, os
D = "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/out/dp27"
for bs, pl in ((31, 1024), (32, 1024), (64, 1024), (8, 8192), (9, 8192)):
    pa = f"{D}/eager_dec{bs}_{pl}.json"
    if not os.path.exists(pa):
        print(f"bs {bs} @{pl}: no eager"); continue
    a = json.load(open(pa))
    for other in ("fullnc_a", "fullnc_b", "breakable", "default", "vcvp", "vsvp"):
        p = f"{D}/{other}_dec{bs}_{pl}.json"
        if not os.path.exists(p):
            print(f"bs {bs} @{pl}: no {other}"); continue
        b = json.load(open(p))
        same = sum(x == y for x, y in zip(a, b))
        first = [next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None) for x, y in zip(a, b)]
        print(f"bs {bs} @{pl}: eager vs {other}: {same}/{len(a)} sequences identical; first divergences {sorted(f for f in first if f is not None)[:8]}; "
              f"mean distinct tokens per sequence {sum(len(set(x)) for x in b) / len(b):.1f} (eager {sum(len(set(x)) for x in a) / len(a):.1f})")
PY
echo "$(date +%T) dp_brk27b done"
