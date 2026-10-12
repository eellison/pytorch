#!/usr/bin/env bash
# The traced trtllm-gen extension for this GPU's arch: checks upstream/src against the installed FlashInfer, writes src/ from it
# (ht/make_patch.py), builds $FI_HT_BUILD/fi_ht_trtllm.so against the installed torch (ht/build.py), and prints our nvcc flags
# against FlashInfer's own fmha_gen build.ninja (bin/check_flags.py; run a stock trtllm-gen call first so it exists).
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
SITE=${FI_SITE_PKG:-$(python -c "import importlib.util,os; print(os.path.dirname(importlib.util.find_spec('flashinfer').origin))")}
( cd "$SITE/data" && sha256sum -c "$FI_HT_CPP_ROOT/src/UPSTREAM.sha256" )
( cd "$FI_HT_CPP_ROOT/upstream/src" && sha256sum -c "$FI_HT_CPP_ROOT/src/UPSTREAM.sha256" )
python "$FI_HT_CPP_ROOT/ht/make_patch.py"
python "$FI_HT_CPP_ROOT/ht/build.py"
python "$BUNDLE/bin/check_flags.py" || true
