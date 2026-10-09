# sourced by the GPU 0 inner scripts
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
cd $G
idle_check() {  # waits (<= IDLE_MAX s, default 120) until GPU 0 has no compute process and <= 5% util over 10 samples; prints the verdict
  local t0=$(date +%s) apps util busy
  while :; do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | wc -l)
    busy=0
    for i in $(seq 10); do util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0); [ "$util" -gt 5 ] && busy=1; sleep 1; done
    if [ "$apps" = 0 ] && [ $busy = 0 ]; then echo "idle_check ok ($(( $(date +%s) - t0 )) s wait)"; return 0; fi
    if [ $(( $(date +%s) - t0 )) -gt ${IDLE_MAX:-300} ]; then
      local who=$(for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0); do ps -o pid=,user=,args= -p $p | cut -c1-120; done | tr '\n' ';')
      echo "idle_check CONTENDED apps=$apps busy=$busy owners: $who"; return 1; fi
    sleep 20
  done
}
load_gate() {  # waits (<= GATE_MAX s, default 7200) until load1 < LOAD_MAX (90) and busy cores on 0-71 (10 s window) < BUSY_MAX (24); prints the verdict
  local t0=$(date +%s) l1 busy
  while :; do
    l1=$(cut -d' ' -f1 /proc/loadavg); busy=$(python3 vllm/cpubusy.py 10 | cut -d' ' -f1)
    if python3 -c "import sys; sys.exit(not ($l1 < ${LOAD_MAX:-90} and $busy < ${BUSY_MAX:-24}))"; then
      echo "load_gate ok load1 $l1 busy_cores0-71 $busy ($(( $(date +%s) - t0 )) s wait)"; return 0; fi
    if [ $(( $(date +%s) - t0 )) -gt ${GATE_MAX:-7200} ]; then echo "load_gate TIMEOUT load1 $l1 busy_cores0-71 $busy"; return 1; fi
    sleep 30
  done
}
util_now() {  # UTIL_CAP (default 0.6) unless GPU 0 holds foreign memory
  read used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 0 | tr -d ,)
  python3 -c "print(min(${UTIL_CAP:-0.6}, round(($total - $used - 4096) / $total, 3)))"
}
trun() {  # tag ENV=.. -- script args
  local tag=$1; shift; local envs=()
  if [ -f vllm/out/$tag.json ] && [ -z "${RERUN:-}" ]; then echo "$(date +%T) have $tag"; return; fi  # jobs re-queue to fill contended skips
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local verdict="" try
  for try in 1; do  # a foreign GPU 0 job (gpu0_job.sh waited for idle before taking the lock): skip the run, no contended rows
    verdict=$(idle_check | tail -1)
    case "$verdict" in "idle_check ok"*) break ;; esac
  done
  echo "$(date +%T) start $tag $verdict load $(cut -d' ' -f1-3 /proc/loadavg) busy_cores0-71 $(python3 vllm/cpubusy.py 5 | cut -d' ' -f1)" | tee -a vllm/logs/timing_idle.log
  case "$verdict" in "idle_check ok"*) ;; *) echo "$(date +%T) SKIPPED_CONTENDED $tag" | tee -a vllm/logs/timing_idle.log; return ;; esac
  local u=$(util_now)
  env "${envs[@]}" INTEG_GPU_UTIL=$u timeout 3000 bash python_vllm_cand2.sh "$@" > vllm/logs/$tag.log 2>&1
  echo "$(date +%T) end $tag rc=$? util=$u"
}
OPEN=(CAND_FORK=1 ARMV_TRACED_IMPLS=1 ARMV_TRTLLM_FORK=1)
