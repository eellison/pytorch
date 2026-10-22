#!/usr/bin/env bash
# vf_alt.sh: untimed check (coordinator 10-15): does the alternating-decode variant-switch cost (alt 97<->100, ours 0.78x of fullnc on
# step334_ss) improve on the current head with valid-first evaluation (int6 snap/step_ms14 on build_cpp_ms9)? Ours (V) on both lines,
# back to back: alt 97<->100, 37<->50, 63<->65 and steady decode bs 97 / 100 at 1K; gb300-box GPU 1 after chain_g1x (PID $1) stops,
# gpu1.lock per run, CPUs 72-107. Launcher: this lane's python_fn.sh with the head's tree / install (python_int6_build_ms9.sh carries
# SGLang's PYTHONPATH, not vLLM's).
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
L=/data/eellison/src/pytorch/agent_space/paramgraph/land
echo $$ > $D/logs/mon/vf_alt.pid
while [ -n "$(ps -o args= -p $1 2>/dev/null | grep chain_g1x)" ]; do sleep 30; done
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
one v_step334ss step334_ss build_cpp15 torch_ext_c15
one v_step_ms14 step_ms14 build_cpp_ms9 torch_ext_ms9
# ABI check of the head run (coordinator): every traced forward of ours vs a fresh eager reference, decode bs 1 / 8 / 64 + prefill 64 / 512,
# diagnosis config (deterministic QSA order), eager twice as control; the alt timing counts only if this is all bitwise
one vchk_step_ms14 step_ms14 build_cpp_ms9 torch_ext_ms9 FN_QSA_DET=1 ARMV_CHECK_STATE_CPU=1 ARMV_CHECK_TWICE=1 -- \
  --batch-size 1 8 64 --decode-tokens 8 --prefill-lens 64 512 --mixed 0 --check
python3 - <<'PY'
import json
D = "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext/out/vf/"
for t in ("v_step334ss", "v_step_ms14"):
    try:
        j = json.load(open(D + t + ".json"))
    except Exception as e:
        print(t, "no result", e); continue
    alt = {k: round(v["step_ms"], 2) for k, v in j["ix"].items() if k.startswith("alt")}
    dc = {k: round(v["step_ms"], 2) for k, v in j["dcurve"].items()}
    print(t, "alt", alt, "steady", dc, "variants", j["V"].get("variants"), "window_violations", j.get("window_violations"))
try:
    c = json.load(open(D + "vchk_step_ms14.json"))["V"].get("checks") or []
    bw = sum(1 for x in c if x.get("bitwise")); tw = sum(1 for x in c if x.get("eager_twice_bitwise"))
    print(f"vchk_step_ms14: checks {len(c)}, bitwise {bw}, eager-twice bitwise {tw}, paths {sorted(set(x.get('path') for x in c if x.get('path')))}")
except Exception as e:
    print("vchk_step_ms14 no result", e)
PY
echo "$(date +%T) vf_alt done"
