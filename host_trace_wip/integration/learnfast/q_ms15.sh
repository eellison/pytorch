#!/usr/bin/env bash
# q_ms15.sh: step_ms15 / build_cpp_ms10, ARMV_MEMORY=run_buffer with sections on (the core fix), GPU 0 untimed, gate before lock:
#  snap15: decode phase then single prefills at 1024 / 4096 / 16376 (per-T peak allocated; the earlier 199 GiB case)
#  ms15:   rs5-style cold start (edge phases, 800-step stream): ready, ready-to-steady, stream events, peak allocated per phase/stream
# Any CUDA error stops the chain.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
export UTIL=0.6 HEAD=3 LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150 RANGE=1 SECTIONS=1 ARMV_MEMORY=run_buffer MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1
gate() { w=0; while [ "$(cut -d. -f1 /proc/loadavg)" -ge 60 ] && [ $w -lt 1800 ]; do sleep 30; w=$((w + 30)); done; }
cudaerr() { grep -qiE "CUDA error|illegal instruction|illegal memory|Xid|cudaErrorIllegal" $1; }
gate; echo "$(date +%T) snap15"
DRIVE_RANGE_PHASES=decode,psweep DRIVE_STARTUP_STEPS=5 ./gpulock.sh 0 3600 ./startup27.sh run vt snap15 > logs/snap15.log 2>&1
echo "$(date +%T) rc=$? $(grep PSWEEP logs/snap15.log)"
cudaerr logs/snap15.log && { echo "CUDA ERROR in snap15: stop"; exit 1; }
gate; echo "$(date +%T) ms15"
DRIVE_STARTUP_STEPS=800 ./gpulock.sh 0 3600 ./startup27.sh run vt ms15_r1 > logs/rw_vtms15_r1.log 2>&1
echo "$(date +%T) rc=$?"
cudaerr logs/rw_vtms15_r1.log && echo "CUDA ERROR in ms15_r1"
