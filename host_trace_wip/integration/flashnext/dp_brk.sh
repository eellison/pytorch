#!/usr/bin/env bash
# dp_brk.sh: breakable decode check (10-15): greedy decode of seeded 1024-token prompts at bs 31 (padded) / 32 / 64 (unpadded capture sizes),
# tokens per sequence; token ids to out/dp/<tag>_dec<BS>.json for fullnc (twice: the eager-graph spread) and breakable; compared at the end.
# Untimed; gb300-box GPU 2, gpu2.lock per run (slots between chain_g2c's arms), CPUs 36-71, the timed sets' build / settings.
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
L=/data/eellison/src/pytorch/agent_space/paramgraph/land
echo $$ > $D/logs/mon/dp_brk.pid
mkdir -p $D/out/dp
export FN_WT=$L/int6/snap/step334_ss FN_INSTALL=$L/int6/build_cpp15/install TORCH_EXTENSIONS_DIR=$D/cache/torch_ext_c15
export SGL_OPS_TRACE_DIR=$L/core/sidestreams/port_overlay FN_FI_AUTOTUNE_FILE=$D/cache/fi_autotune/f2_autotune.json HOSTTRACE_TORCH_DISABLE_CACHES=0
export FN_MOE_OVERLAY=$L/scratch/sglang/site DRIVE_STEP_WITH_SAMPLE=1 DRIVE_KID=0 INTEG_GPU_UTIL=0.9 INTEG_KV_BYTES=$((40 << 30)) CUDA_VISIBLE_DEVICES=2
export DRIVE_DECODE_PROBE="31 32 64" DRIVE_LOGIT_PROBE_DIR=$D/out/dp
one() {  # tag arm env...
  local tag=$1 arm=$2; shift 2
  ( flock -x 9 || { echo "$(date +%T) $1: flock failed, not running"; exit 1; }; echo "$(date +%T) $tag holding gpu2.lock"
    env "$@" DRIVE_LOGIT_PROBE_TAG=$tag timeout --foreground 1800 taskset -c 36-71 bash $D/python_fn.sh $D/scripts/with_fi_autotune.py $D/bench/drive_sweep27.py \
      --arm $arm --model /data/eellison/models/Qwen3.8-Flash-Next-NVFP4 --trust-remote-code --language-model-only --max-model-len 16384 \
      --llm-kw '{"gdn_prefill_backend": "triton"}' --batch-size 1 --decode-tokens 12 --prefill-lens 64 --mixed 0 --out $D/out/dp/$tag.json \
      > $D/logs/dp_$tag.log 2>&1; echo "$(date +%T) $tag rc=$?" ) 9>$L/scratch/eager/gpu2.lock
}
one fullnc_a fullnc
one breakable fullnc_full VLLM_USE_BREAKABLE_CUDAGRAPH=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
one fullnc_b fullnc
python3 - <<'PY'
import json, os
D = "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext/out/dp"
for bs in (31, 32, 64):
    a = json.load(open(f"{D}/fullnc_a_dec{bs}.json"))
    for other in ("fullnc_b", "breakable"):
        p = f"{D}/{other}_dec{bs}.json"
        if not os.path.exists(p):
            continue
        b = json.load(open(p))
        same = sum(x == y for x, y in zip(a, b))
        first = [next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None) for x, y in zip(a, b)]
        distinct = sum(len(set(x)) for x in b) / len(b)
        print(f"bs {bs}: fullnc_a vs {other}: {same}/{len(a)} sequences identical; first divergence {sorted(f for f in first if f is not None)[:8]}; mean distinct tokens per sequence {distinct:.1f} (fullnc_a {sum(len(set(x)) for x in a) / len(a):.1f})")
PY
echo "$(date +%T) dp_brk done"
