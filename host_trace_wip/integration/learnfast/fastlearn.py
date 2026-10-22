# Lean offline cuBLAS learn from a key set: HarvestProvider.learn on operands carved from buffers reused per operand slot
# (one max-size buffer per (operand index, dtype), filled once), the eager call into the out= buffer first (verify digests
# eager's output there), keys grouped by (N, K) so each group's templates are warm. Same provider code path as the in-engine
# learn: every binding is still harvested and checked by cuBLAS's own launch.
#   bash py.sh fastlearn.py --keys KEYS.pkl --out out/x.pkl [--busy-poll] [--order file|nk] [--shard I/N] [--threads K]
import argparse
import collections
import json
import pickle
import threading
import time

p = argparse.ArgumentParser()
p.add_argument("--keys", required=True)
p.add_argument("--out", required=True)
p.add_argument("--json", default="")
p.add_argument("--limit", type=int, default=0)
p.add_argument("--order", default="file", choices=["file", "nk"])
p.add_argument("--shard", default="")
p.add_argument("--busy-poll", action="store_true", help="verify's launch waits poll the event without time.sleep(1e-4)")
p.add_argument("--fresh-fill", action="store_true", help="refill the operand buffers' used span per key (as _learn)")
p.add_argument("--set", action="append", default=[])
a = p.parse_args()

import torch  # noqa: E402
import torch.cuda._host_trace_harvest as H  # noqa: E402
from torch.cuda._host_trace_opaque import fill_uniform, library_state_as, OpaqueKey, span_bytes  # noqa: E402

for s in a.set:
    lhs, v = s.split("=", 1)
    setattr(H, lhs.split(".", 1)[1], eval(v))

if a.busy_poll:
    import types

    _t = H.time
    H.time = types.SimpleNamespace(monotonic=_t.monotonic, perf_counter=_t.perf_counter, time=_t.time, sleep=lambda s: None if s <= 1e-4 else _t.sleep(s))

t_start = time.perf_counter()
data = pickle.load(open(a.keys, "rb"))
entries = [k for k, _ in data["entries"]]
if a.order == "nk":
    entries.sort(key=lambda k: (tuple(k[3][1]), k[3][0][0]))
if a.shard:
    i, n = map(int, a.shard.split("/"))
    m = -(-len(entries) // n)
    entries = entries[i * m : (i + 1) * m]
if a.limit:
    entries = entries[: a.limit]

OPS = {"aten::mm": torch.ops.aten.mm, "aten::addmm": torch.ops.aten.addmm, "aten::bmm": torch.ops.aten.bmm}
NIN = {"aten::mm": 2, "aten::addmm": 3, "aten::bmm": 2}
dev = torch.device("cuda", 0)
gen = torch.Generator(dev).manual_seed(0)
# per (operand index, dtype): the largest span the keys need, one buffer each (+256 for an alignment class)
need: dict = collections.defaultdict(int)
for k in entries:
    _, opn, dtypes, sizes, strides, classes, *_ = H._local(k)
    for i, (dt, z, st, c) in enumerate(zip(dtypes, sizes, strides, classes)):
        need[i, dt] = max(need[i, dt], span_bytes(z, st, dt.itemsize) + (0 if c == 256 else c))
bufs = {}
for (i, dt), n in need.items():
    b = torch.empty(-(-n // 512) * 512 + 512, dtype=torch.uint8, device=dev)
    if dt.is_floating_point:
        fill_uniform(b[: b.numel() // dt.itemsize * dt.itemsize].view(dt), -1, 1, gen)
    bufs[i, dt] = b
torch.cuda.synchronize()
t_bufs = time.perf_counter() - t_start

provider = H.HarvestProvider(("blas",))


def view(i, dt, sizes, strides, cls):
    off = 0 if cls == 256 else cls
    n = span_bytes(sizes, strides, dt.itemsize)
    flat = bufs[i, dt][off : off + -(-n // dt.itemsize) * dt.itemsize].view(dt)
    if a.fresh_fill and i < 2:
        fill_uniform(flat, -1, 1, gen)
    return flat.as_strided(sizes, strides)


def one(k):
    alias, opn, dtypes, sizes, strides, classes, scalars, device, state = H._local(k)
    packet = OPS[opn]
    key = OpaqueKey(packet.default, dtypes, sizes, strides, tuple(0 if c == 256 else c for c in classes), scalars, device, state, alias)
    ops = [view(i, dt, z, st, c) for i, (dt, z, st, c) in enumerate(zip(dtypes, sizes, strides, classes))]
    ins, out = ops[: NIN[opn]], ops[NIN[opn]]
    with library_state_as(state), torch._C._AutoDispatchBelowADInplaceOrView():
        packet.out(*ins, out=out)  # eager's result in the output: verify digests it there
    return provider.learn(key, tuple(ins), {}, ins + [out])


t0 = time.perf_counter()
cpu0 = time.process_time()
per, failed = [], 0
for k in entries:
    t = time.perf_counter()
    failed += one(k) is None
    per.append(time.perf_counter() - t)
    if a.limit and a.limit <= 64:
        torch.cuda.synchronize()
        print("key", k[3], "ms %.2f" % (1e3 * (time.perf_counter() - t)), flush=True)
torch.cuda.synchronize()
t_learn = time.perf_counter() - t0
cpu_learn = time.process_time() - cpu0
t1 = time.perf_counter()
saved = provider.save(a.out)
per.sort()
R = {
    "keys": len(entries), "failed": failed, "harvests": provider.harvests, "full_templates": sum(len(v) for v in provider.templates.values()),
    "buffers_s": round(t_bufs, 2), "buffer_gb": round(sum(b.numel() for b in bufs.values()) / 2**30, 2),
    "learn_wall_s": round(t_learn, 2), "learn_cpu_s": round(cpu_learn, 2), "per_key_ms_median": round(1e3 * per[len(per) // 2], 3),
    "per_key_ms_mean": round(1e3 * t_learn / max(len(entries), 1), 3), "save_s": round(time.perf_counter() - t1, 2), "saved": saved,
    "total_s": round(time.perf_counter() - t_start, 2), "loadavg": open("/proc/loadavg").read().split()[:3], "flags": vars(a),
}
print(json.dumps(R))
if a.json:
    json.dump(R, open(a.json, "w"), indent=1)
