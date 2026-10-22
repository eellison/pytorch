#!/usr/bin/env bash
# vf_b.sh: untimed (coordinator 10-16): the full w workload of ours on step334_ss + build_cpp15 WITH the fast cuBLAS bindings
# preloaded (ARMV_BINDINGS_LOAD, as the timed v arm), per-call probe on: do 97 and 100 share a decode variant there (the 0.80x alt cost)?
# gb300-box GPU 1, gpu1.lock, CPUs 72-107.
#
# per call at the alt / steady sizes, the patch counters and (first 4 calls per size) the variant that ran.
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
L=/data/eellison/src/pytorch/agent_space/paramgraph/land
echo $$ > $D/logs/mon/vf_b.pid

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
MIX=(); for p in 256 512 1024 2048 4096; do for nd in 16 64; do MIX+=(${p}x${nd}@1024-2048); done; done
W=(--batch-size 1 --decode-tokens 8 --prefill-lens 64 128 256 512 1024 2048 4096 8192 16000 --mixed 0 --ix-mixed "${MIX[@]}"
   --alt-decode 37:50@1024-2048 63:65@1024-2048 97:100@1024-2048 --ix-reps 6 --dcurve-ctx 1024 --dcurve-bs 97 100 --dcurve-steps 32 --dcurve-warm 8)
one wb_step334ss step334_ss build_cpp15 torch_ext_c15 DRIVE_FWD_SYNC=0 FN_CALL_PROBE=1 ARMV_BINDINGS_LOAD=$D/bindings/fn_v.fast.pkl -- "${W[@]}"
python3 - <<'PY'
import json
D = "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext/out/vf/"
for t in ("wb_step334ss",):
    try:
        j = json.load(open(D + t + ".json"))
    except Exception as e:
        print(t, "no result", e); continue
    alt = {k: round(v["step_ms"], 2) for k, v in j["ix"].items() if k.startswith("alt")}
    dc = {k: round(v["step_ms"], 2) for k, v in j["dcurve"].items()}
    dec = {k: e.get("variants") for k, e in j["V"]["entries"].items() if "decode" in k}
    print(t, "alt", alt, "steady", dc, "variants total", j["V"].get("variants"), "decode entry", dec, "window_violations", j.get("window_violations"))
PY
python3 - <<'PY2'
import collections, re
L = "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext/logs/"
for t in ("wb_step334ss",):
    per = collections.defaultdict(list)
    for line in open(L + f"vf_{t}.log"):
        m = re.match(r"CALLPROBE ntok=(\d+) call=(\d+) replayed=(\w+) variants=(\d+) ran=(\S+) packs=(-?\d+) node_sets=(-?\d+) mem_sets=(-?\d+)", line)
        if m:
            per[int(m.group(1))].append((int(m.group(2)), m.group(5), int(m.group(6)), int(m.group(7)), int(m.group(8)), m.group(4)))
    for n in sorted(per):
        rows = per[n]
        ran = [r[1] for r in rows if r[1] != "None"]
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
        print(f"{t} bs {n}: calls {len(rows)}, entry variants {rows[-1][5]}, variant(s) seen {sorted(set(ran))}, median per call: packs {med([r[2] for r in rows])}, node sets {med([r[3] for r in rows])}, mem sets {med([r[4] for r in rows])}")
PY2
echo "$(date +%T) vf_b done"
