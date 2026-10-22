#!/usr/bin/env bash
# IMA: GPU core dump at the fault (no memory contents), then cuda-gdb names the faulting kernel
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
cd $R/out/coredump && rm -f core*.cuda*
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_GENERATION_FLAGS='skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' CUDA_COREDUMP_FILE=$R/out/coredump/core_%p.cuda
IMA_BS="5 6" IMA_TAG=ima_core bash $R/run_ima_c12.sh > $R/logs/ima_core.log 2>&1; grep -E "^decode|illegal|rc=" $R/logs/ima_core.log
ls -la $R/out/coredump
f=$(ls $R/out/coredump/core_*.cuda 2>/dev/null | head -1)
[ -n "$f" ] && /usr/local/cuda-13.0/bin/cuda-gdb -batch -ex "target cudacore $f" -ex "info cuda kernels" -ex "bt" -ex "info cuda devices" 2>&1 | head -60
