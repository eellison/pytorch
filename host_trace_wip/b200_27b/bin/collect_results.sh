#!/usr/bin/env bash
# results tarball for the return trip: outputs, summaries, verdicts, timing table, logs (tail-trimmed), environment report.
set -uo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; source "$B/env.sh"
R=$WORK/results; rm -rf "$R"; mkdir -p "$R/logs"
cp -r "$WORK/out" "$R/" 2>/dev/null; cp -r "$WORK/timing" "$R/" 2>/dev/null
rm -f "$R"/timing/*.log
for f in "$WORK"/logs/*.log "$WORK"/timing/*.log; do [ -f "$f" ] && tail -n 400 "$f" > "$R/logs/$(basename "$f")"; done
{
  nvidia-smi; nvidia-smi topo -m; uname -a; lscpu | head -20
  python -c "import torch, sys; print(sys.version); print('torch', torch.__version__, torch.version.cuda, torch.__file__); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
  python -c "import importlib.metadata as m
for p in ['vllm','flashinfer-python','nvidia-cutlass-dsl','cuda-python','apache-tvm-ffi','triton','transformers']:
    try: print(p, m.version(p))
    except Exception: print(p, '-')"
  (cd "$TORCH_SRC" 2>/dev/null && git log -1 --format='torch %H %s') ; (cd "$VLLM_SRC" 2>/dev/null && git log -1 --format='vllm %H %s')
  "$CUDA_HOME/bin/nvcc" --version | tail -2
  ls -la "$FI_HT_BUILD"; md5sum "$FI_HT_BUILD/fi_ht_trtllm.so"
  python "$B/bin/check_flags.py" 2>&1
} > "$R/ENV.txt" 2>&1
cp "$WORK/NOTES.md" "$R/" 2>/dev/null || echo "WARNING: write $WORK/NOTES.md (see RUNBOOK step 9)"
tar -czf "$WORK/r27_b200_results.tgz" -C "$WORK" results && echo "$WORK/r27_b200_results.tgz"
