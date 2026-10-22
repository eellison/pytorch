#!/usr/bin/env bash
# run.sh <triton|cute> <stock212|stockdev>: run one notebook's cells on gb300-box GPU 3 under the share rule
# (shared gpu3.share.lock, >= 30 GB free, short run).
#   stock212: the isolated venv venv_t212 (released torch 2.12.1 from PyPI, triton 3.7.1, cutlass-dsl 4.6.2, tvm-ffi 0.1.11)
#   stockdev: the default env's torch (a 2.15 dev build without host-trace code), IPython from pydeps/
set -uo pipefail
L=/data/eellison/src/pytorch/agent_space/paramgraph/land
C=$L/scratch/colab
which=$1 mode=$2
export CUDA_VISIBLE_DEVICES=3 PYTHONDONTWRITEBYTECODE=1 TMPDIR=$C/tmp/$which-$mode
export TORCH_EXTENSIONS_DIR=$C/cache/torch_ext TVM_FFI_CACHE_DIR=$C/cache/tvm-ffi-$mode
rm -rf "$TMPDIR" $C/run_${which}_${mode}
mkdir -p "$TMPDIR" $TORCH_EXTENSIONS_DIR $TVM_FFI_CACHE_DIR
exec 9>$L/scratch/eager/gpu3.share.lock
flock -s 9
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 3)
if (( total - used < 30000 )); then echo "GPU 3 has $((total - used)) MiB free (< 30 GB); not running"; exit 3; fi
date; nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader -i 3
cd $C
if [[ $mode == stock212 ]]; then
  env -u PYTHONPATH PATH=/usr/local/cuda-13.0/bin:/usr/bin:/bin timeout 900 taskset -c 126-143 $C/venv_t212/bin/python -I build_colab.py run $which $mode
else
  source /data/eellison/src/pytorch/agent_space/env.sh
  PYTHONPATH=$C/pydeps timeout 900 taskset -c 126-143 python build_colab.py run $which $mode
fi
echo "rc $?"; date
