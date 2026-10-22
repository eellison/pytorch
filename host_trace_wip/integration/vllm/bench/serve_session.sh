#!/usr/bin/env bash
# bench/serve_session.sh ARM (eager|eagerm|default|V|Vd; Vd: V tracing decode keys only; eagerm: eager plus V's max_seq_len policy, V's bitwise reference): one `vllm serve` session on GPU 2 (run under gpu2.lock by run_serve.sh).
# Server on CPUs 108-135, client on 136-143. Same engine settings as bench/drive.py (FLASHINFER, max_model_len 4096,
# gpu_memory_utilization 0.6, prefix caching off, seed 0), and vLLM's shipped scheduler settings otherwise (async
# scheduling and chunked prefill on by default). Each workload runs twice with the same prompts: pass 1 (cold for V),
# then pass 2.
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
ARM=$1 PORT=${PORT:-18731} M=/data/eellison/models/Qwen3-8B OUT=bench/out/serve TAG=${TAG:-s1}
mkdir -p $OUT
case $ARM in
  eager) L=python_vllm.sh; X=--enforce-eager;;
  eagerm) L=python_vllm.sh; X=--enforce-eager; export VLLM_EXTRA_PYTHONPATH=$V/armV/serve_plugin ARMV_SERVE_MODE=policy;;
  default) L=python_vllm.sh; X=; export HOSTTRACE_TORCH_DISABLE_CACHES=0;;
  V|Vd) L=python_vllm_cpp.sh; X=--enforce-eager; export VLLM_EXTRA_PYTHONPATH=$V/armV/serve_plugin ARMV_SERVE_LOG=$V/$OUT/${TAG}_${ARM}_armv.json
     [ $ARM = Vd ] && export ARMV_SERVE_TRACE=decode;;
esac
t0=$(date +%s.%N)
taskset -c 108-135 bash $L -m vllm.entrypoints.cli.main serve $M --port $PORT --attention-backend FLASHINFER --max-model-len 4096 \
  --gpu-memory-utilization 0.6 --no-enable-prefix-caching --seed 0 $X > bench/logs/${TAG}_${ARM}_server.log 2>&1 &
SP=$!
for i in $(seq 900); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
echo "ready_s $(python3 -c "import time; print(round(time.time()-$t0,1))") mem_mib $(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)"
run() {  # name conc prompts in out ratio
  taskset -c 136-143 timeout 1500 bash python_vllm.sh -m vllm.entrypoints.cli.main bench serve --backend vllm --base-url http://127.0.0.1:$PORT \
    --model $M --dataset-name random --random-input-len $4 --random-output-len $5 --random-range-ratio $6 --num-prompts $3 \
    --max-concurrency $2 --ignore-eos --temperature 0 --seed 0 --num-warmups 0 --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --save-detailed --result-dir $OUT --result-filename ${TAG}_${ARM}_$1.json > bench/logs/${TAG}_${ARM}_$1.log 2>&1
  echo "$1 rc=$? $(date +%T)"
}
for pass in 1 2; do
  for w in "r1 1 16 512 128 0" "r8 8 64 512 128 0" "r64 64 256 512 128 0" "m8 8 128 512 256 0.9" "m64 64 256 512 256 0.9"; do
    set -- $w; run p${pass}_$1 $2 $3 $4 $5 $6
  done
  echo "pass $pass mem_mib $(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)"
done
kill $SP; wait $SP
echo "server rc=$?"
