# python3 armV/dumpcmp.py A.pt B.pt: drive.py --dump-decode files (compute_logits inputs of each pure decode step, in
# order); per batch size, the steps bitwise equal, and the first differing step with its max abs difference.
import collections, sys
import torch

a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
print(len(a), len(b), "steps")
res = collections.defaultdict(lambda: [0, 0, None])
for i, (x, y) in enumerate(zip(a, b)):
    r = res[x.shape[0]]
    if x.shape != y.shape:
        print("shape mismatch at", i, x.shape, y.shape)
        break
    eq = torch.equal(x, y)
    r[0] += eq
    r[1] += 1
    if not eq and r[2] is None:
        r[2] = (i, (x.float() - y.float()).abs().max().item())
for bs, (n_eq, n, first) in sorted(res.items()):
    print(f"bs {bs}: {n_eq}/{n} bitwise, first diff {first}")
