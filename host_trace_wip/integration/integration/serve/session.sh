#!/usr/bin/env bash
# One `vllm serve` session (run it under the GPU's lock; the caller holds it): bash session.sh ARM JOB...
# ARM: eager | default (vLLM shipped: compile + FULL_AND_PIECEWISE graphs) | V (enforce_eager + arm V installed over
# POST /armv/install) | eagerm (enforce_eager + V's max_seq_len policy only: V's bitwise reference).
# All arms on the candidate 2 C++ line (python_vllm_cand2.sh, CAND_LINE=cpp). V/eagerm load plugin/armv_serve.py through
# vLLM's extension points (--worker-extension-cls + the armv_endpoints endpoint plugin); the server itself is stock.
# JOBs, in order:
#   check1[:S] | burst8[:S] serve_check.py, fixed prompt set at concurrency 1 / 8 -> out/${TAG}_${ARM}_<job><S>.json
#   checkon | checkoff      V: adapter check (every replayed step vs the model's eager forward) on / off
#   warm                    serve_warm2.py range warm-up (decode bs 1-256, prefill lengths, mixed)
#   ctr:NAME                GET /armv/counters?first_calls=1 -> out/${TAG}_${ARM}_ctr_NAME.json (V/eagerm)
#   summary:NAME            GET /armv/summary -> out/${TAG}_${ARM}_summary_NAME.json
#   sweep:NAME:ISL:OSL:RR:C1,C2,..   vllm bench serve, random dataset, per concurrency C: 10*C prompts, 2*C warm-ups,
#                           request rate inf, ignore-eos, temperature 0; V counters after each point
# Env: GPU (1), TAG, ATTN (FLASHINFER; empty = vLLM picks), M (Qwen3-8B), MAXLEN (10240), PORT, SCPU / CCPU (server / client CPUs), SERVER_EXTRA (more serve args),
# ARMV env (ARMV_*) passes through to the worker.
set -u
SV=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve
G=$(dirname $SV)
cd $G
ARM=$1; shift
ATTN=${ATTN-FLASHINFER} GPU=${GPU:-1} TAG=${TAG:-a1} M=${M:-/data/eellison/models/Qwen3-8B} MAXLEN=${MAXLEN:-10240} PORT=${PORT:-$((18800 + GPU))}
if [ $GPU = 0 ]; then SCPU=${SCPU:-0-63} CCPU=${CCPU:-64-71}; else SCPU=${SCPU:-72-101} CCPU=${CCPU:-102-107}; fi
OUT=$SV/out LOGS=$SV/logs
export CUDA_VISIBLE_DEVICES=$GPU CAND_LINE=cpp
read used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | tr -d ,)
UTIL=${UTIL:-$(python3 -c "print(min(0.6, round(($total - $used - 4096) / $total, 3)))")}
X=(--enforce-eager)
case $ARM in
  eager) ;;
  default) X=(); export HOSTTRACE_TORCH_DISABLE_CACHES=0 ;;
  V|eagerm) X+=(--worker-extension-cls armv_serve.ArmVWorkerExt)
     export VLLM_EXTRA_PYTHONPATH=$SV/plugin VLLM_PLUGINS=armv_endpoints,lora_filesystem_resolver,lora_hf_hub_resolver ;;
  *) echo "ARM=$ARM" >&2; exit 2 ;;
esac
pfx=${TAG}_${ARM}
mem() { nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits; }
client() { taskset -c $CCPU bash python_vllm_cand2.sh "$@"; }
t0=$(date +%s.%N)
echo "$(date +%T) $pfx server start gpu $GPU util $UTIL (gpu used $used MiB) env: $(env | grep -E '^ARMV_' | tr '\n' ' ')"
taskset -c $SCPU bash python_vllm_cand2.sh -m vllm.entrypoints.cli.main serve $M --port $PORT ${ATTN:+--attention-backend $ATTN} \
  --max-model-len $MAXLEN --gpu-memory-utilization $UTIL --no-enable-prefix-caching --seed 0 "${X[@]}" ${SERVER_EXTRA:-} \
  > $LOGS/${pfx}_server.log 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null' EXIT
for i in $(seq 1800); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
if ! curl -sf localhost:$PORT/health >/dev/null; then echo "$pfx server failed; tail:"; tail -30 $LOGS/${pfx}_server.log; exit 1; fi
echo "$(date +%T) $pfx ready_s $(python3 -c "import time; print(round(time.time()-$t0,1))") mem_mib $(mem)"
case $ARM in
  V) echo "install $(curl -s -X POST localhost:$PORT/armv/install -H 'Content-Type: application/json' -d '{"mode":"trace","first_calls":true}')" ;;
  eagerm) echo "install $(curl -s -X POST localhost:$PORT/armv/install -H 'Content-Type: application/json' -d '{"mode":"policy"}')" ;;
esac
ctr() { curl -s "localhost:$PORT/armv/counters?first_calls=${2:-0}" > $OUT/${pfx}_ctr_$1.json; python3 - $OUT/${pfx}_ctr_$1.json <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("ctr", {k: d.get(k) for k in ("traces", "replays", "variants", "boundaries", "segments", "learner_variants", "eager", "relowers", "keys", "exact_keys", "bad_keys", "harvests", "bound", "checks", "reserved_mib", "calls")})
EOF
}
for job in "$@"; do
  IFS=: read -r kind a1 a2 a3 a4 a5 <<< "$job"
  ts=$(date +%s)
  case $kind in
    check1) client $SV/serve_check.py --port $PORT --model $M --conc 1 --out $OUT/${pfx}_check1${a1}.json ;;
    burst8) client $SV/serve_check.py --port $PORT --model $M --conc 8 --n 48 --out $OUT/${pfx}_burst8${a1}.json ;;
    checkon) echo "check $(curl -s -X POST localhost:$PORT/armv/check -H 'Content-Type: application/json' -d '{"on":true}')" ;;
    checkoff) echo "check $(curl -s -X POST localhost:$PORT/armv/check -H 'Content-Type: application/json' -d '{"on":false}')" ;;
    warm) client $SV/serve_warm2.py --port $PORT --model $M > $LOGS/${pfx}_warm.log 2>&1; echo "warm rc=$? $(tail -1 $LOGS/${pfx}_warm.log)" ;;
    ctr) ctr $a1 1 ;;
    summary) curl -s localhost:$PORT/armv/summary > $OUT/${pfx}_summary_$a1.json ;;
    sweep)
      for c in ${a5//,/ }; do
        client -m vllm.entrypoints.cli.main bench serve --backend vllm --base-url http://127.0.0.1:$PORT --model $M \
          --dataset-name random --random-input-len $a2 --random-output-len $a3 --random-range-ratio "{\"input\": $a4, \"output\": $a4}" \
          --num-prompts $((10 * c)) --num-warmups $((2 * c)) --max-concurrency $c --request-rate inf --ignore-eos --temperature 0 --seed 0 \
          --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --save-result --result-dir $OUT --result-filename ${pfx}_${a1}_c$c.json \
          > $LOGS/${pfx}_${a1}_c$c.log 2>&1
        echo "$(date +%T) $a1 c$c rc=$? $(grep -E 'Output token throughput|Median TPOT' $LOGS/${pfx}_${a1}_c$c.log | tr -s ' ' | tr '\n' ' ')"
        [ $ARM = V ] && ctr ${a1}_c$c 1
      done ;;
    *) echo "job $job?" ;;
  esac
  echo "$(date +%T) $pfx job $job done in $(( $(date +%s) - ts )) s mem_mib $(mem)"
done
echo "$(date +%T) $pfx server log: $(grep -E 'KV cache size|Graph capturing finished|Maximum concurrency' $LOGS/${pfx}_server.log | tr -s ' ' | cut -c1-200 | tr '\n' '|')"
