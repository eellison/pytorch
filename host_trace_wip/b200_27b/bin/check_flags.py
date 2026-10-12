# Our traced-extension nvcc flags (trtllm_cpp/ht/build_flags.py) against FlashInfer's own fmha_gen build.ninja cuda_cflags on this
# machine: the arch, defines and include set must match (the drift test and the KernelParams layout assume FlashInfer's build).
import os
import re
import shlex
import sys

sys.path.insert(0, os.path.join(os.environ["FI_HT_CPP_ROOT"], "ht"))
import build_flags as bf  # noqa: E402

ninja = os.path.join(bf.STOCK_BUILD, "build.ninja")
print("arch", bf.ARCH, "stock build", ninja, "exists", os.path.exists(ninja))
if not os.path.exists(ninja):
    sys.exit("no stock fmha_gen build yet: run bin/warm.sh first (one stock trtllm-gen attention call builds it)")
text = re.sub(r"\$\n\s*", " ", open(ninja).read())  # ninja line continuations
var = {}
for line in text.splitlines():
    m = re.match(r"^([A-Za-z_][\w.]*)\s*=\s*(.*)$", line)  # top-level bindings only (not indented ones of build edges)
    if m:
        var[m.group(1)] = m.group(2)
expand = lambda s: re.sub(r"\$(\w+)|\$\{(\w+)\}", lambda m: expand(var.get(m.group(1) or m.group(2), "")), s.replace("$$", "\0")).replace("\0", "$")
stock = shlex.split(expand(var.get("cuda_cflags", "")))
ours = bf.stock_cuda_flags()
norm = lambda fs: [f.replace('\\"', '"') for f in fs]
s, o = norm(stock), norm(ours)
pair = lambda fs: {(fs[i], fs[i + 1]) if fs[i] == "-isystem" else fs[i] for i in range(len(fs)) if not (i and fs[i - 1] == "-isystem")}
# nvcc packaging / build-speed flags that do not change the code
BENIGN = {"--compress-mode=size", "--threads=1", "-Xfatbin=-compress-all"}
ps, po = pair(s) - BENIGN, pair(o) - BENIGN
print("only in stock:", sorted(map(str, ps - po)))
print("only in ours:", sorted(map(str, po - ps)))
print("MATCH" if ps == po else "DIFFER (see above)")
