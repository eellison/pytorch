#!/usr/bin/env bash
# cold8.sh TAG ARM KEYS NPROC: one cold start of our arm on Qwen3-8B (c5h), GPU $LF_GPU (caller holds its lock), CPUs $CPUS.
# Cold: fresh empty VLLM_CACHE_ROOT (torch.compile + FlashInfer autotune caches), TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR;
# FlashInfer JIT modules / TVM-FFI / our extension stay built (as in the vLLM lane's cold definition). No preloaded bindings
# from earlier runs: at t0 NPROC side processes learn KEYS (batchlearn.py, the warm-up range's keys) while the engine starts;
# the adapter's ARMV_BINDINGS_LOAD waits for their merged file. ARM: Vc (compiled forward) or VR (eager forward).
LF=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
TAG=$1 ARM=$2 K=$3 N=$4
C=$LF/cold/$TAG; rm -rf $C; mkdir -p $C/vllm $C/inductor $C/triton
G=${LF_GPU:-2}; CPU0=${CPU0:-36}
export COLD_T0=$(date +%s.%N) COLD_STAMPS=$C/stamps.jsonl
B=$C/bindings.pkl
if [ -n "${PRE_BINDINGS:-}" ]; then cp $PRE_BINDINGS $B; touch $B.done; fi
[ -n "${PRE_BINDINGS:-}" ] || ( t=$(date +%s.%N)
  for i in $(seq 0 $((N-1))); do
    c0=$((CPU0 + i * 24 / N)); c1=$((CPU0 + (i+1) * 24 / N - 1)); [ $c1 -lt $c0 ] && c1=$c0
    OMP_NUM_THREADS=1 LF_GPU=$G LF_CPUS=$c0-$c1 $LF/py.sh $LF/batchlearn.py --keys $K --out $C/b_$i.pkl --json $C/b_$i.json --shard $i/$N --order nk --deal 64 ${LEARN_FLAGS:-} > $C/b_$i.log 2>&1 &
  done; wait
  LF_GPU=$G LF_CPUS=$CPU0-$((CPU0+3)) $LF/py.sh $LF/merge.py $B $(for i in $(seq 0 $((N-1))); do echo $C/b_$i.pkl; done) > $C/merge.log 2>&1
  touch $B.done
  echo "{\"ev\": \"learn_done\", \"t\": $(python3 -c "print(round($(date +%s.%N) - $COLD_T0, 3))")}" >> $COLD_STAMPS ) &
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
BASE=(CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1)
case $ARM in
  Vc) A=("${BASE[@]}" INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 DRIVE_PHASES=1 ARMV_VPS=1); DA=Vc ;;
  VR) A=("${BASE[@]}" INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 DRIVE_PHASES=1 ARMV_VPS=1); DA=V ;;
esac
env "${A[@]}" DRIVE_STEP_WITH_SAMPLE=1 ARMV_BINDINGS_LOAD=$B VLLM_CACHE_ROOT=$C/vllm TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton \
  INTEG_GPU_UTIL=${UTIL:-0.5} CUDA_VISIBLE_DEVICES=$G timeout ${COLD_TIMEOUT:-1500} taskset -c $((CPU0+24))-$((CPU0+35)) bash python_vllm_cand2.sh $LF/coldwrap.py vllm/bench/drive_sweep.py --arm $DA --max-model-len 16384 \
  --batch-size ${COLD_BS:-1 64} --decode-tokens 3 --prefill-lens ${COLD_PL:-64 1000} --prefill-reps 2 --mixed 1 --out $C/run.json > $C/run.log 2>&1
echo "rc=$?"
wait
python3 - $C <<'PY'
import json, sys
C = sys.argv[1]
st = [json.loads(l) for l in open(C + "/stamps.jsonl")]
print({s["ev"]: s["t"] for s in st})
try:
    d = json.load(open(C + "/run.json"))
    V = d.get("V", {})
    print("init_s", round(d.get("init_s", 0), 1), "total_wall_s", round(d.get("total_wall_s", 0), 1), "traces", V.get("traces"), "harvests", V.get("harvests"), "bindings_loaded", V.get("bindings_loaded"), "variants", V.get("variants"))
    for k, rows in d.get("rows", {}).items():
        print(" ", k, [(r[0], r[1], round(r[2], 3)) for r in rows[:3]])
except Exception as e:
    print("no run.json", e)
PY
