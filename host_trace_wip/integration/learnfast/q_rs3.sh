#!/usr/bin/env bash
# q_rs3.sh: pin-origin test, then rs3 = rs2 + pin rule refinement (PROV_PIN_ATEN=1) + GDN nums host tensor as a pinned argument
# (ARMV_PIN_UNPINNED=1: its shape no longer keys the entry form; r27's existing knob, ~5.5 us host copy). GPU 3 untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
./tests/run_pin_where.sh > logs/pin_where.log 2>&1
echo "$(date +%T) pin test: $(grep -E '^(OK|FAILED)' logs/pin_where.log)"
export UTIL=0.6 HEAD=1 LF_GPU=3 LF_CPUS=108-143 HOLD_S=1150 RANGE=1 SECTIONS=1 MEMO=1 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 DRIVE_ENTRY_FORMS=1
echo "$(date +%T) rs3"; DRIVE_STARTUP_STEPS=800 ./gpulock.sh 3 3600 ./startup27.sh run vt rs3 > logs/rw_vtrs3.log 2>&1
echo "$(date +%T) rs3 rc=$?"
