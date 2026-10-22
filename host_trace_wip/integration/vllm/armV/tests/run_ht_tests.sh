#!/usr/bin/env bash
# run_ht_tests.sh <tag> <test files...> [-- pytest args]: host-trace tests from src_vllm1/test on GPU 2 (gpu2.lock, CPUs 72-107).
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
TAG=$1; shift
files=(); while [ $# -gt 0 ] && [ "$1" != -- ]; do files+=("$1"); shift; done; [ "$1" = -- ] && shift
cd $V/src_vllm1/test
for f in "${files[@]}"; do
  log=$V/armV/tests/logs/${TAG}_$(basename $f .py).log
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=2 flock /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock \
    taskset -c 72-107 timeout 3600 bash $V/python_vllm_v.sh -m pytest -p no:cacheprovider -q -rfE $f "$@" > $log 2>&1 || true
  echo "$f: $(tail -1 $log)"
done
echo DONE
