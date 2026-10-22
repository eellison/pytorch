"""logit_cmp.py DIR A B: compare the probe logits DIR/A_<L>.pt vs DIR/B_<L>.pt per length (NaN counts as unequal)."""
import glob, os, sys
try:
    import torch
except ImportError:  # the queue scripts call a python without torch: rerun through r27's launcher (once)
    if os.environ.get("LOGIT_CMP_RELAUNCHED"):
        raise
    os.environ["LOGIT_CMP_RELAUNCHED"] = "1"
    r27 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.execvp("bash", ["bash", os.path.join(r27, "python_r27.sh"), os.path.abspath(__file__), *sys.argv[1:]])
D, A, B = sys.argv[1:4]
for pa in sorted(glob.glob(os.path.join(D, f"{A}_*.pt")), key=lambda p: int(p.rsplit("_", 1)[1][:-3])):
    L = pa.rsplit("_", 1)[1][:-3]
    pb = os.path.join(D, f"{B}_{L}.pt")
    if not os.path.exists(pb):
        continue
    x, y = torch.load(pa), torch.load(pb)
    nan = (int(x.isnan().sum()), int(y.isnan().sum()))
    eq = bool(torch.equal(x, y)) and nan == (0, 0)
    d = (x - y).abs()
    print(f"L={L:>5s} {'BITWISE' if eq else 'DIFFER'}: NaN {nan}, max|d| {float(d[~d.isnan()].max()) if (~d.isnan()).any() else float('nan'):.4g}, "
          f"argmax {int(x.argmax())} vs {int(y.argmax())}, top-5 overlap {len(set(x.topk(5).indices.tolist()) & set(y.topk(5).indices.tolist()))}/5")
