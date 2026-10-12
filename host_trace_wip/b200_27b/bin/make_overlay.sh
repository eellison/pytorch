#!/usr/bin/env bash
# Builds $OVERLAY/flashinfer: a package directory of symlinks to the INSTALLED flashinfer 0.6.18, except decode.py and prefill.py
# (stock files + a 2-line hook at each trtllm-gen binding call) and trtllm_cpp.py (the glue to the traced extension), which are
# the bundle's real files. Checks the installed decode.py / prefill.py are the 0.6.18 files the hooked copies come from.
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
SITE=${FI_SITE_PKG:-$(python -c "import importlib.util,os; print(os.path.dirname(importlib.util.find_spec('flashinfer').origin))")}
echo "installed flashinfer: $SITE"
python -c "import importlib.metadata as m; v=m.version('flashinfer-python'); print('flashinfer-python', v); assert v == '0.6.18', v"
( cd "$SITE" && echo "d82d107a644596a9349780b839b34e690c50169ea4cea3d0ee02b78e9dc88c1a  decode.py
743953a8d301d89c6943dc48a8f6b393c0d636b71b165ae1842c2c65c7defaab  prefill.py" | sha256sum -c - )
rm -rf "$OVERLAY/flashinfer"; mkdir -p "$OVERLAY/flashinfer"
for e in "$SITE"/* "$SITE"/.[!.]*; do
  [ -e "$e" ] || continue
  n=$(basename "$e")
  case $n in decode.py|prefill.py|trtllm_cpp.py|__pycache__) continue ;; esac
  ln -s "$e" "$OVERLAY/flashinfer/$n"
done
cp "$BUNDLE"/overlay_src/flashinfer/{decode.py,prefill.py,trtllm_cpp.py} "$OVERLAY/flashinfer/"
echo "overlay at $OVERLAY/flashinfer"
