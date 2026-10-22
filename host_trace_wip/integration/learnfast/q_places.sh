#!/usr/bin/env bash
# q_places.sh: section_places cache + bisect (core_places.patch on step_ms15 = learnfast/places_snap): unit tests, then the
# workload equality check (every call vs the scan) in eager and run_buffer memory, then the gate. GPU 0, untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
cd $L
export LF_GPU=0 LF_CPUS=0-35 LF_WT=$L/places_snap LF_INSTALL=$I/build_cpp_ms10/install TORCH_EXTENSIONS_DIR=$L/torch_ext_ms10 HOLD_S=900 HOSTTRACE_TORCH_DISABLE_CACHES=0
./gpulock.sh 0 3600 ./py.sh places_snap/test/test_cuda_host_trace_memory.py -v -k SectionPlaces > logs/places_unit.log 2>&1
echo "$(date +%T) unit: $(grep -E '^(OK|FAILED)' logs/places_unit.log)"
unset LF_WT LF_INSTALL
export UTIL=0.6 HEAD=3 HEAD_WT=$L/places_snap LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150 RANGE=1 SECTIONS=1 MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1
export DRIVE_RANGE_PHASES=decode,prefill1,mixed DRIVE_STARTUP_STEPS=50 DIAG27_PLACES_CHECK=1
for m in eager run_buffer; do
  ARMV_MEMORY=$m ./gpulock.sh 0 3600 ./startup27.sh run vt places_$m > logs/places_$m.log 2>&1
  echo "$(date +%T) $m rc=$? $(grep PLACES_CHECK logs/places_$m.log)"
done
sed -i 's/LF_GPU=3 LF_CPUS=108-143/LF_GPU=${GATE_GPU:-3} LF_CPUS=${GATE_CPUS:-108-143}/; s|LF_INSTALL=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/build_cpp_ms8/install|LF_INSTALL=${GATE_INSTALL:-/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/build_cpp_ms8/install}|; s|TORCH_EXTENSIONS_DIR=$L/torch_ext_ms8|TORCH_EXTENSIONS_DIR=${GATE_EXT:-$L/torch_ext_ms8}|; s|$L/gpulock.sh 3 1800|$L/gpulock.sh ${GATE_GPU:-3} 1800|' gate_tpl.sh
GATE_GPU=0 GATE_CPUS=0-35 GATE_INSTALL=$I/build_cpp_ms10/install GATE_EXT=$L/torch_ext_ms10 ./gate_tpl.sh $L/places_snap placesgate > logs/placesgate.out 2>&1
echo "$(date +%T) gate rc=$?"
