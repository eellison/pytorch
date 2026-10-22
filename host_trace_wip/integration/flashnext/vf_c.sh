#!/usr/bin/env bash
# vf_c.sh: the head crash (coordinator 10-16): (A) ix mixed 512x16@1024-2048 + neighbours on step_ms14 + build_cpp_ms9 with the MoE
# overlay rebuilt against build_cpp_ms9 (core/flashnext/moe_ms9), every call checked bitwise; (B) the original crashing full w run with that
# overlay. After vf_b ($1) and the build. gb300-box GPU 1, gpu1.lock, CPUs 72-107, untimed.
#
# per call at the alt / steady sizes, the patch counters and (first 4 calls per size) the variant that ran.
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
L=/data/eellison/src/pytorch/agent_space/paramgraph/land
echo $$ > $D/logs/mon/vf_c.pid

mkdir -p $D/out/vf
OURS=(ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
      ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=
      ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1)
one() {  # tag snap build extdir [extra env...] -- [driver args]
  local tag=$1 snap=$2 build=$3 ext=$4; shift 4
  local extra=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do extra+=("$1"); shift; done; [ "${1:-}" = "--" ] && shift
  local wl=("$@"); [ ${#wl[@]} = 0 ] && wl=(--batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0 --alt-decode 97:100@1024-2048 37:50@1024-2048 63:65@1024-2048 --ix-reps 6 --dcurve-ctx 1024 --dcurve-bs 97 100 --dcurve-steps 32 --dcurve-warm 8)
  set -- $tag $snap $build $ext
  ( flock -x 9 || { echo "$(date +%T) $1: flock failed, not running"; exit 1; }; echo "$(date +%T) $1 holding gpu1.lock"
    env "${OURS[@]}" "${extra[@]}" FN_WT=$L/int6/snap/$2 FN_INSTALL=$L/int6/$3/install TORCH_EXTENSIONS_DIR=$D/cache/$4 SGL_OPS_TRACE_DIR=$L/core/sidestreams/port_overlay \
      FN_FI_AUTOTUNE_FILE=$D/cache/fi_autotune/f1_autotune.json HOSTTRACE_TORCH_DISABLE_CACHES=0 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_KID=0 DRIVE_REQUIRE_WARM=0 \
      INTEG_GPU_UTIL=0.9 INTEG_KV_BYTES=$((40 << 30)) CUDA_VISIBLE_DEVICES=1 \
      timeout --foreground 1500 taskset -c 72-107 bash $D/python_fn.sh $D/scripts/with_fi_autotune.py $D/bench/drive_sweep27.py --arm V \
      --model /data/eellison/models/Qwen3.8-Flash-Next-NVFP4 --trust-remote-code --language-model-only --max-model-len 16384 \
      --llm-kw '{"gdn_prefill_backend": "triton"}' "${wl[@]}" \
      --out $D/out/vf/$1.json > $D/logs/vf_$1.log 2>&1; echo "$(date +%T) $1 rc=$?" ) 9>$L/scratch/eager/gpu1.lock
}
while [ -n "$(ps -o args= -p $1 2>/dev/null | grep vf_b.sh)" ] || [ ! -f $D/moe_ms9/fi_htm_moe.so ]; do sleep 30; done
echo "$(date +%T) vf_c: vf_b done, moe_ms9 built: $(ldd $D/moe_ms9/fi_htm_moe.so | grep -o 'int6/[^/]*/install' | sort -u | tr '\n' ' ')"
MOE=(FI_HTM_MOE_BUILD=$D/moe_ms9 FN_MOE_BUILD=$D/moe_ms9)
CHK=(FN_QSA_DET=1 ARMV_CHECK_STATE_CPU=1 ARMV_CHECK_TWICE=1)
# (A) the crashing shape and its neighbours on the head with the ABI-clean overlay, every call checked
one c_mixed_ms14 step_ms14 build_cpp_ms9 torch_ext_ms9 DRIVE_FWD_SYNC=0 "${MOE[@]}" "${CHK[@]}" -- \
  --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0 --ix-mixed 256x64@1024-2048 512x16@1024-2048 512x64@1024-2048 1024x16@1024-2048 --ix-reps 3 --check
# (B) the original crashing run (full w workload, ours on the head) with the ABI-clean overlay, unchecked
MIX=(); for p in 256 512 1024 2048 4096; do for nd in 16 64; do MIX+=(${p}x${nd}@1024-2048); done; done
one c_w_ms14 step_ms14 build_cpp_ms9 torch_ext_ms9 DRIVE_FWD_SYNC=0 "${MOE[@]}" -- --batch-size 1 --decode-tokens 8 \
  --prefill-lens 64 128 256 512 1024 2048 4096 8192 16000 --mixed 0 --ix-mixed "${MIX[@]}" \
  --alt-decode 37:50@1024-2048 63:65@1024-2048 97:100@1024-2048 --ix-reps 6 --dcurve-ctx 1024 --dcurve-bs 97 100 --dcurve-steps 32 --dcurve-warm 8
for t in c_mixed_ms14 c_w_ms14; do echo "== $t: $(grep -c 'illegal' $D/logs/vf_$t.log) illegal-instruction lines; json $(ls $D/out/vf/$t.json 2>/dev/null)"; done
python3 - <<'PY3'
import json
D = "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext/out/vf/"
try:
    c = json.load(open(D + "c_mixed_ms14.json"))["V"].get("checks") or []
    print(f"c_mixed_ms14: checks {len(c)}, bitwise {sum(1 for x in c if x.get('bitwise'))}, eager-twice bitwise {sum(1 for x in c if x.get('eager_twice_bitwise'))}")
except Exception as e:
    print("c_mixed_ms14 no result", e)
PY3
echo "$(date +%T) vf_c done"
