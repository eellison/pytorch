#!/usr/bin/env bash
# q_snap.sh: name the decode-then-prefill allocation: ours V run_buffer, decode phase (bs edges to 128) then single-request prefills
# at 1024 / 4096 / 16376 tokens (per-T peaks), allocator snapshot around the 16376 one; then the same with decode bs edges to 8 only
# (does it scale with sequences?). GPU 0 untimed, load gate before the lock.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
export UTIL=0.6 HEAD=2 HEAD_WT=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/snap/wip14 LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150
export RANGE=1 DRIVE_RANGE_WARM=1 DRIVE_RANGE_PHASES=decode,psweep DRIVE_STARTUP_STEPS=5 SECTIONS=1 MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1
for v in 128 8; do
  w=0; while [ "$(cut -d. -f1 /proc/loadavg)" -ge 60 ] && [ $w -lt 1800 ]; do sleep 30; w=$((w + 30)); done
  echo "$(date +%T) seqs $v"
  DRIVE_RANGE_MAX_SEQS=$v DRIVE_MEM_SNAPSHOT=$L/out/memsnap_decode${v}_p16k.pickle DRIVE_MEM_SNAPSHOT_PHASE=psweep_last ./gpulock.sh 0 3600 ./startup27.sh run vt snap$v > logs/snap$v.log 2>&1
  echo "$(date +%T) rc=$? $(grep PSWEEP logs/snap$v.log)"
done
