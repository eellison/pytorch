#!/usr/bin/env bash
# 10-14 Flash-Next decode ladder at 1K, untimed, after the 27B lane's gpu3_kid.sh queue (coordinator priority); subsets of
# the ladder per run (a 32-bs chunk with fills and --check does ~4 points in a 20 min hold): criterion + closure vs baseline
cd $(dirname $0)
# the 27B lane's queue: its launcher and its kidx_line.sh runs (anchored: shells quoting the name do not count); quiet for 10 min
quiet=0
while [ $quiet -lt 10 ]; do
  # also the Flash-Next lane's cbsmoke13 (coordinator order 10-14: ahead of this ladder), queued or running
  if pgrep -f "^bash gpu3_kid.sh|^bash kidx_line.sh|^bash -c bash kidx_line.sh|^bash [^ ]*/scripts/cbsmoke" > /dev/null; then quiet=0; else quiet=$((quiet + 1)); fi
  sleep 60
done
G="LF_GPU=3 LF_CPUS=108-143"
PC="DIAG27_PROV=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 $G"
run() { tag=$1; shift; ./gpulock.sh 3 72000 env "$@" > logs/$tag.log 2>&1; echo done >> logs/$tag.log; }
run fnp_s1 $PC BS="1 2 3 4 5 8 16 32" RUN_TIMEOUT=1120 ./fnrc.sh fnp_s1 chk
run fnb_s1 $G BS="1 2 3 4 5 8 16 32" RUN_TIMEOUT=1120 ./fnrc.sh fnb_s1 chk
# s2 dropped (fnp_s1: low_latency_gemm's redo at a new M changes its chain / keyed sites -> refused; the ladder cannot collapse under refused sections)
