# perf script (-F tid,ip,sym,dso) of drive_ix2's DRIVE_PERF_DECODE samples -> share of samples whose stack contains each marker
# (innermost marker wins), and the top leaf symbols under a chosen marker.   python3 perf_fold.py SCRIPT [MARKER]
import collections, sys

MARKERS = [("cudaDeviceSynchronize", "sync"), ("cuCtxSynchronize", "sync"), ("cuGraphExecKernelNodeSetParams", "node set (kernel)"),
           ("cuGraphExecMemcpyNodeSetParams", "node set (memcpy)"), ("cuGraphExecMemsetNodeSetParams", "node set (memset)"),
           ("cudaGraphExecMemcpyNodeSetParams", "node set (memcpy)"), ("cudaGraphLaunch", "graph launch"), ("cuGraphLaunch", "graph launch"),
           ("cuLaunchKernel", "kernel launch"), ("cudaLaunchKernel", "kernel launch"), ("patch_and_replay", "entry: patch+replay (other)"),
           ("evaluate_rows", "entry: evaluate"), ("HostTraceVariant::commit", "entry: commit (other)"), ("entry_call", "entry (other)"),
           ("cudaMemcpyAsync", "memcpy"), ("cuMemcpy", "memcpy"), ("_PyEval_EvalFrameDefault", "python interp")]
samples, cur = [], None
for line in open(sys.argv[1]):
    if line and not line[0].isspace():
        cur = []
        samples.append(cur)
    elif line.strip() and cur is not None:
        parts = line.strip().split(" ", 1)
        cur.append(parts[1] if len(parts) > 1 else "")
cnt, leaf = collections.Counter(), collections.Counter()
want = sys.argv[2] if len(sys.argv) > 2 else None
for st in samples:
    tag = "other"
    for fr in st:  # innermost first
        hit = next((t for m, t in MARKERS if m in fr), None)
        if hit:
            tag = hit
            break
    cnt[tag] += 1
    if want and tag == want and st:
        leaf[st[0][:110]] += 1
n = len(samples)
print(f"{n} samples")
for k, v in cnt.most_common():
    print(f"{v / n * 100:6.1f}%  {k}")
for k, v in leaf.most_common(15):
    print(f"   {v:5d}  {k}")
