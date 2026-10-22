# pyspy_top.py RAW [STEPS]: py-spy --format raw folded stacks -> share of samples by leaf function (self) and by function anywhere on
# the stack (inclusive), main thread only; with STEPS (steps in the sampled window) also ms per step at the sampling rate (250 Hz).
import collections, re, sys

raw, steps = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else None
self_c, incl_c, total = collections.Counter(), collections.Counter(), 0
for line in open(raw):
    line = line.rstrip()
    if not line:
        continue
    stack, _, n = line.rpartition(" ")
    n = int(n)
    frames = stack.split(";")
    if not any("MainThread" in f or "process" in f for f in frames[:1]) and "drive_sweep27" not in stack:
        continue  # other threads (tqdm monitor, etc.)
    total += n
    fr = [re.sub(r" \(.*?/(site-packages/|site/|torch/|vllm/)?", " (", f) for f in frames]
    self_c[fr[-1]] += n
    for f in set(fr):
        incl_c[f] += n
ms = (lambda n: f"{n / 250 * 1e3 / steps:7.2f} ms/step") if steps else (lambda n: "")
print(f"{raw}: {total} samples")
print("TOP SELF (leaf frames)")
for f, n in self_c.most_common(25):
    print(f"  {100 * n / total:5.1f}% {ms(n)}  {f[:150]}")
print("TOP INCLUSIVE")
for f, n in incl_c.most_common(40):
    print(f"  {100 * n / total:5.1f}% {ms(n)}  {f[:150]}")
syncs = {k: n for k, n in incl_c.items() if re.search(r"\b(tolist|item|synchronize|cpu|_local_scalar_dense|nonzero)\b", k)}
print("SYNC-LIKE frames (inclusive):", sorted(((round(100 * n / total, 1), k[:120]) for k, n in syncs.items()), reverse=True)[:12])
