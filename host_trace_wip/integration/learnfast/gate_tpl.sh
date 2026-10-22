#!/usr/bin/env bash
# gate_tpl.sh WT TAG [names]: host-trace test files from WT/test on build_cpp_ms8/install, one gpulock.sh hold per file (GPU 3).
set -u
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
W=$1 tag=$2; shift 2
names=("$@"); [ ${#names[@]} -eq 0 ] && names=(replay harvest python_entry cute triton aten opaque memory guards inductor/test_cudagraph_host_trace.py)
export CUTE_DSL_EXAMPLES=/data/eellison/src/pytorch/agent_space/paramgraph/wt-land-int5b/third_party/cutlass/examples/python/CuTeDSL
export LF_GPU=${GATE_GPU:-3} LF_CPUS=${GATE_CPUS:-108-143} LF_WT=$W LF_INSTALL=${GATE_INSTALL:-/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/build_cpp_ms8/install}
export TORCH_EXTENSIONS_DIR=${GATE_EXT:-$L/torch_ext_ms8} HOLD_S=1150 HOSTTRACE_TORCH_DISABLE_CACHES=0
mkdir -p $L/logs/$tag; cd $W/test; fail=0
for n in "${names[@]}"; do
  case $n in *.py) f=$n; b=$(basename $n .py);; *) f=test_cuda_host_trace_$n.py; b=$n;; esac
  log=$L/logs/$tag/$b.log
  $L/gpulock.sh ${GATE_GPU:-3} 1800 $L/py.sh "$f" > $log 2>&1; rc=$?
  [ $rc -eq 75 ] && { echo "$b: ABORT (no lock)"; exit 4; }
  ran=$(grep -E '^Ran [0-9]+ tests?' $log | tail -1)
  v=PASS; { [ $rc -ne 0 ] || [ -z "$ran" ] || ! grep -qE '^OK' $log; } && { v=FAIL; fail=1; }
  echo "$b: $v rc=$rc $ran | $(grep -E '^(OK|FAILED)' $log | tail -1)"
done
exit $fail
