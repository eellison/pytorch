#!/usr/bin/env bash
# bench/serve_range.sh ARM (eagerm|default|V): phase 6 step 5, one `vllm serve` session on GPU 2 (run under gpu2.lock by
# run_serve_range.sh). Engine settings as rangesweep.py (FLASHINFER, max_model_len 16384, 8192 batched tokens, 256 seqs,
# gpu_memory_utilization 0.6, prefix caching off, seed 0); vLLM's shipped scheduler otherwise (async scheduling on).
# Range warm-up over HTTP (bench/serve_warm.py), then the local ShareGPT file at concurrency 1, 8 and 64.
# V: ARMV_STATIC_SHAPES=1 ARMV_WIDE_SLOTS=1 (rs5), counters from ARMV_SERVE_LOG (rewritten every 2 s) before and after.
# Server on CPUs 108-135, client on 136-143.
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
ARM=$1 PORT=${PORT:-18741} M=/data/eellison/models/Qwen3-8B OUT=bench/out/serve TAG=${TAG:-sr1}
DS=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/armF/e2e/sharegpt_local.json
mkdir -p $OUT
LOG=$V/$OUT/${TAG}_${ARM}_armv.json
case $ARM in
  eagerm) L=python_vllm.sh; X=--enforce-eager; export VLLM_EXTRA_PYTHONPATH=$V/armV/serve_plugin ARMV_SERVE_MODE=policy;;
  default) L=python_vllm.sh; X=; export HOSTTRACE_TORCH_DISABLE_CACHES=0;;
  V) L=python_vllm_cpp.sh; X=--enforce-eager
     export VLLM_EXTRA_PYTHONPATH=$V/armV/serve_plugin ARMV_SERVE_LOG=$LOG ARMV_SERVE_LOG_EVERY=2 ARMV_STATIC_SHAPES=1 ARMV_WIDE_SLOTS=1;;
esac
mem() { nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits; }
snap() {  # copy the counters (V) after two idle dump periods
  sleep 5
  [ -f $LOG ] && cp $LOG $V/$OUT/${TAG}_${ARM}_armv_$1.json
  echo "$1 mem_mib $(mem) $([ -f $LOG ] && python3 -c "import json; d=json.load(open('$LOG')); print({k: d.get(k) for k in ('forward_calls','keys','traces','relowers','variants','learned','harvests','eager','reserved_mib')})")"
}
t0=$(date +%s.%N)
taskset -c 108-135 bash $L -m vllm.entrypoints.cli.main serve $M --port $PORT --attention-backend FLASHINFER --max-model-len 16384 \
  --max-num-batched-tokens 8192 --max-num-seqs 256 --gpu-memory-utilization 0.6 --no-enable-prefix-caching --seed 0 $X \
  > bench/logs/${TAG}_${ARM}_server.log 2>&1 &
SP=$!
for i in $(seq 900); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
echo "ready_s $(python3 -c "import time; print(round(time.time()-$t0,1))") mem_mib $(mem)"
taskset -c 136-143 timeout 3000 bash python_vllm.sh bench/serve_warm.py --port $PORT > bench/logs/${TAG}_${ARM}_warm.log 2>&1
echo "warm rc=$? $(tail -1 bench/logs/${TAG}_${ARM}_warm.log)"
snap warm
run() {  # name conc prompts
  taskset -c 136-143 timeout 2400 bash python_vllm.sh -m vllm.entrypoints.cli.main bench serve --backend vllm --base-url http://127.0.0.1:$PORT \
    --model $M --dataset-name sharegpt --dataset-path $DS --num-prompts $3 --max-concurrency $2 --ignore-eos --temperature 0 --seed 0 \
    --num-warmups 0 --percentile-metrics ttft,tpot,itl,e2el --save-result --save-detailed --result-dir $OUT \
    --result-filename ${TAG}_${ARM}_$1.json > bench/logs/${TAG}_${ARM}_$1.log 2>&1
  echo "$1 rc=$? $(date +%T)"
}
run c1 1 32
run c8 8 128
run c64 64 384
snap served
kill $SP; wait $SP
echo "server rc=$?"
