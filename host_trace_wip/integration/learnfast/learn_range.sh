#!/usr/bin/env bash
# learn_range.sh SEED.pkl OUT.pkl MAX_TOKENS [MAX_SEQS VOCAB_N] : the warm-up range's cuBLAS bindings from a short run's harvest.
#   SEED: a bindings file saved by the adapter (ARMV_BINDINGS_SAVE) after a short run that traced every family once.
#   OUT: the merged bindings file for ARMV_BINDINGS_LOAD.
# Env: PY = the launcher that runs python in the SAME build and environment as the serving process (the bindings' meta
# must match: GPU, driver, torch build, CUDA libraries); N = processes (default 8); CPU0 = first CPU of the 36-CPU block;
# LF_GPU = the GPU (the launcher must honor it or set CUDA_VISIBLE_DEVICES itself).
set -euo pipefail
cd $(dirname $0)
SEED=$1 OUT=$2 T=$3 S=${4:-0} VN=${5:-}
TAG=lr_$(basename $OUT .pkl)
${PY:-./py.sh} enum_keys.py --seed $SEED --out out/${TAG}_keys.pkl --max-tokens $T ${S:+--max-seqs $S} ${VN:+--seqs-n $VN}
SCRIPT=batchlearn.py PY=${PY:-./py.sh} ./parf.sh $TAG ${N:-8} out/${TAG}_keys.pkl --order nk --deal 64 --one-fill
cp out/$TAG.pkl $OUT
echo "wrote $OUT"
