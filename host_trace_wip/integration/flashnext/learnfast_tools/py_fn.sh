#!/usr/bin/env bash
# the learnfast tools' launcher (PY) for Flash-Next: the timed ours arm's exact build and environment (python_fn.sh with the
# caller's FN_WT / FN_INSTALL / SGL_OPS_TRACE_DIR / TORCH_EXTENSIONS_DIR, i.e. round_fn_line.sh's COMMON), GPU LF_GPU, CPUs LF_CPUS
export CUDA_VISIBLE_DEVICES=${LF_GPU:-0}
exec taskset -c ${LF_CPUS:-0-35} bash /data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext/python_fn.sh "$@"
