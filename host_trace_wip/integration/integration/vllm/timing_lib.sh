# sourced by the GPU 0 inner scripts
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
cd $G
idle_check() {  # waits (<= IDLE_MAX s, default 120) until GPU 0 has no compute process and <= 5% util over 10 samples; prints the verdict
  local t0=$(date +%s) apps util busy
  while :; do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i ${HT_GPU:-0} | wc -l)
    busy=0
    for i in $(seq 10); do util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i ${HT_GPU:-0}); [ "$util" -gt 5 ] && busy=1; sleep 1; done
    if [ "$apps" = 0 ] && [ $busy = 0 ]; then echo "idle_check ok ($(( $(date +%s) - t0 )) s wait)"; return 0; fi
    if [ $(( $(date +%s) - t0 )) -gt ${IDLE_MAX:-300} ]; then
      local who=$(for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i ${HT_GPU:-0}); do ps -o pid=,user=,args= -p $p | cut -c1-120; done | tr '\n' ';')
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
  read used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i ${HT_GPU:-0} | tr -d ,)
  python3 -c "print(min(${UTIL_CAP:-0.6}, round(($total - $used - 4096) / $total, 3)))"
}
trun() {  # tag ENV=.. -- script args
  local tag=$1; shift; local envs=()
  if [ -f vllm/out/$tag.json ] && [ -z "${RERUN:-}" ]; then echo "$(date +%T) have $tag"; return; fi  # jobs re-queue to fill contended skips
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local verdict="" try
  if [ "${TRUN_SHARED:-0}" = 1 ]; then
    # TRUN_SHARED=1 (coordinator 10-22, untimed work: bitwise checks, warm-up validations): the GPU is shared with other users'
    # processes, no idle / own-cores / PSI gate; the foreign processes present are recorded for the audit of the result
    verdict="idle_check ok (shared, untimed; foreign on GPU ${HT_GPU:-0}: $(for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i ${HT_GPU:-0}); do ps -o pid=,user= -p $p | tr -s ' ' | tr '\n' ','; done))"
  else
  for try in 1; do  # a foreign GPU 0 job (gpu0_job.sh waited for idle before taking the lock): skip the run, no contended rows
    verdict=$(idle_check | tail -1)
    case "$verdict" in "idle_check ok"*) break ;; esac
  done
  fi
  local own=${HT_CPUS:-0-71} lw=0 lo hi n
  lo=${own%-*}; hi=${own#*-}; n=$((hi - lo + 1))
  # gate (10-13, coordinator): own pinned cores quiet before the run: mean busy fraction on them over a 10 s sample < OWN_BUSY_FRAC (default
  # 0.10), waiting up to 30 min; the GPU's foreign-process check is idle_check's. Global load is recorded at start, not gated on.
  while [ "${TRUN_SHARED:-0}" != 1 ] && python3 -c "import sys; sys.exit(not ($(python3 vllm/cpubusy.py 10 $lo $hi | cut -d' ' -f1) / $n >= ${OWN_BUSY_FRAC:-0.10}))" && [ $lw -lt 1800 ]; do sleep 20; lw=$((lw + 30)); done
  [ $lw -ge 1800 ] && echo "$(date +%T) OWN_CORES_GATE_TIMEOUT $tag (flag this run)" | tee -a vllm/logs/timing_idle.log
  # 10-14 (main, all lanes): host memory pressure gate: /proc/pressure/memory "full" avg60 < 2% before the run, waiting up to 60 min
  local pw=0 p60
  p60=$(awk '/^full/{split($3,a,"="); print a[2]}' /proc/pressure/memory)
  while [ "${TRUN_SHARED:-0}" != 1 ] && python3 -c "import sys; sys.exit(not ($p60 >= ${PSI_GATE:-2}))" && [ $pw -lt 3600 ]; do sleep 30; pw=$((pw + 30)); p60=$(awk '/^full/{split($3,a,"="); print a[2]}' /proc/pressure/memory); done
  echo "$(date +%T) PSI_GATE $tag waited ${pw}s full_avg60 $p60$( [ $pw -ge 3600 ] && echo ' TIMEOUT (flag this run)')" | tee -a vllm/logs/timing_idle.log
  echo "$(date +%T) start $tag $verdict load $(cut -d' ' -f1-3 /proc/loadavg) busy_cores0-71 $(python3 vllm/cpubusy.py 5 | cut -d' ' -f1) busy_cores_own_${own} $(python3 vllm/cpubusy.py 5 ${own%-*} ${own#*-} | cut -d' ' -f1)" | tee -a vllm/logs/timing_idle.log
  case "$verdict" in "idle_check ok"*) ;; *) echo "$(date +%T) SKIPPED_CONTENDED $tag" | tee -a vllm/logs/timing_idle.log; return ;; esac
  local u=$(util_now)
  env "${envs[@]}" INTEG_GPU_UTIL=$u timeout ${TRUN_TIMEOUT:-3000} bash python_vllm_cand2.sh "$@" > vllm/logs/$tag.log 2>&1
  echo "$(date +%T) end $tag rc=$? util=$u"
}
OPEN=(CAND_FORK=1 ARMV_TRACED_IMPLS=1 ARMV_TRTLLM_FORK=1)
