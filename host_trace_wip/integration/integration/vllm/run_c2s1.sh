#!/usr/bin/env bash
# Candidate 2, step 1 (correctness + counts), GPU 1 under gpu1.lock per command. Both lines x {closed, open}:
# drive.py --check (decode 1/8/64, prefill 64/512/2048, 4x128 + 8x128, mixed 8x128 + specs) and rangesweep --check (rs4 settings).
# util: 0.6 on an empty GPU 1, else 0.16 (recorded in each log; counts do not depend on the KV pool size). SKIP: tags to skip.
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock
cd $G
run() {  # tag line open script args...
  local tag=$1 line=$2 open=$3; shift 3
  local fork=0; local envs=()
  [ "$open" = open ] && fork=1 && envs=(ARMV_TRACED_IMPLS=1 ARMV_TRTLLM_FORK=1)
  case " $SKIP " in *" $tag "*) echo "skip $tag"; return ;; esac
  echo "$(date +%T) start $tag"
  CUDA_VISIBLE_DEVICES=1 flock -x $LOCK env "${envs[@]}" CAND_LINE=$line CAND_FORK=$fork bash -c '
    read used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 1 | tr -d ,)
    u=$(python3 -c "print(0.6 if $used < 2048 else 0.16)")  # a foreign job on GPU 1 swings 8-226 GB without the lock; 0.1 leaves no KV at 16384 batched tokens
    echo "gpu1 used $used total $total util $u"
    INTEG_GPU_UTIL=$u exec timeout 3600 taskset -c 72-107 bash python_vllm_cand2.sh "$@"' _ "$@" > vllm/logs/$tag.log 2>&1
  echo "$(date +%T) end $tag rc=$?"
}
[ -z "${NOPAD:-}" ] && ARMV_PAD=1 run c2_cpp_closed_pad_Vcheck cpp closed vllm/bench/drive.py --arm V --check --batch-size 1 5 8 64 100 --prefill-lens 64 100 300 512 2048 --prefill-reqs 4 8 --mixed-spec 32x512 4x2048 64x64 128x16 --out vllm/out/c2_cpp_closed_pad_Vcheck.json
for line in ${LINES:-cpp py}; do
  for v in closed open; do
    run c2_${line}_${v}_Vcheck $line $v vllm/bench/drive.py --arm V --check --prefill-reqs 4 8 --mixed-spec 32x512 4x2048 64x64 128x16 --out vllm/out/c2_${line}_${v}_Vcheck.json
    ARMV_STATIC_SHAPES=1 run c2_${line}_${v}_range $line $v vllm/bench/rangesweep.py --check --out vllm/out/c2_${line}_${v}_range.json
  done
done
echo ALLDONE
