# Offline cuBLAS learn with the sibling checks batched (prototype; existing harvest code, no core change):
#  - a key with a harvested template of its kernels: HarvestProvider._harvest's sibling route (one capture, adopt) with
#    _VERIFY_SIBLINGS off, then the provider's own in-place GEMM check enqueued, not waited on: eager into the output,
#    digests (_cuda_hostTraceDigests) of every operand, then per scratch fill 0x00 / 0xFF the output filled 0xFF, the
#    binding launched (_launch) on scratch filled so, digests again. B keys enqueue back to back; one sync; every key's
#    digest buffers compared on the device. A key that fails any comparison, holds a constant CUDA address (verify's
#    mirror route), or takes no sibling is learned by provider.learn as before (full harvest, standard verify, memset /
#    owned-slot repair, or refusal). So every binding kept passed the same bitwise comparisons as the serial learn.
#  - everything else: provider.learn.
#   bash py.sh batchlearn.py --keys KEYS.pkl --out out/x.pkl [--batch 64] [--shard I/N]
import argparse
import collections
import dataclasses
import json
import pickle
import time
import types

p = argparse.ArgumentParser()
p.add_argument("--keys", required=True)
p.add_argument("--out", required=True)
p.add_argument("--json", default="")
p.add_argument("--limit", type=int, default=0)
p.add_argument("--shard", default="")
p.add_argument("--batch", type=int, default=64)
p.add_argument("--order", default="file", choices=["file", "nk"])
p.add_argument("--one-fill", action="store_true", help="a binding with no scratch slot and no memset into scratch launches once (its scratch is unreachable: the 0xFF launch repeats the 0x00 one)")
p.add_argument("--deal", type=int, default=0, help="shard by dealing chunks of this many keys (in --order) round robin")
a = p.parse_args()

import torch  # noqa: E402
import torch.cuda._host_trace_harvest as H  # noqa: E402
from torch.cuda._host_trace_opaque import fill_uniform, library_state_as, OpaqueKey, span_bytes  # noqa: E402
from torch.cuda._host_trace_tape import _gc_hold  # noqa: E402

_t = H.time
H.time = types.SimpleNamespace(monotonic=_t.monotonic, perf_counter=_t.perf_counter, time=_t.time, sleep=lambda s: None if s <= 1e-4 else _t.sleep(s))

t_start = time.perf_counter()
data = pickle.load(open(a.keys, "rb"))
entries = [k for k, _ in data["entries"]]
if a.order == "nk":
    entries.sort(key=lambda k: (tuple(k[3][1]), k[3][0][0]))
if a.shard:
    i, n = map(int, a.shard.split("/"))
    if a.deal:
        entries = [k for j, k in enumerate(entries) if (j // a.deal) % n == i]
    else:
        m = -(-len(entries) // n)
        entries = entries[i * m : (i + 1) * m]
if a.limit:
    entries = entries[: a.limit]

OPS = {"aten::mm": torch.ops.aten.mm, "aten::addmm": torch.ops.aten.addmm, "aten::bmm": torch.ops.aten.bmm}
NIN = {"aten::mm": 2, "aten::addmm": 3, "aten::bmm": 2}
dev = torch.device("cuda", 0)
gen = torch.Generator(dev).manual_seed(0)
need: dict = collections.defaultdict(int)
for k in entries:
    _, opn, dtypes, sizes, strides, classes, *_ = H._local(k)
    if not isinstance(opn, str):
        continue
    for i, (dt, z, st, c) in enumerate(zip(dtypes, sizes, strides, classes)):
        need[i, dt] = max(need[i, dt], span_bytes(z, st, dt.itemsize) + (0 if c == 256 else c))
bufs = {}
for (i, dt), n in need.items():
    b = torch.empty(-(-n // 512) * 512 + 512, dtype=torch.uint8, device=dev)
    if dt.is_floating_point:
        fill_uniform(b[: b.numel() // dt.itemsize * dt.itemsize].view(dt), -1, 1, gen)
    bufs[i, dt] = b
SCRATCH = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
torch.cuda.synchronize()
t_bufs = time.perf_counter() - t_start

provider = H.HarvestProvider(("blas",))
stream = torch.cuda.current_stream()
C = collections.Counter()


def view(i, dt, sizes, strides, cls):
    off = 0 if cls == 256 else cls
    n = span_bytes(sizes, strides, dt.itemsize)
    return bufs[i, dt][off : off + -(-n // dt.itemsize) * dt.itemsize].view(dt).as_strided(sizes, strides)


class _MmDtype:
    """aten::mm with an output dtype other than its operands' (a router GEMM into fp32): keyed as aten.mm.dtype in
    builds whose harvest takes it (_OUT_DTYPE), its eager call mm.dtype_out with out_dtype the output's"""

    default = None

    @staticmethod
    def out(a, b, out):
        return torch.ops.aten.mm.dtype_out(a, b, out.dtype, out=out)


if hasattr(torch.ops.aten.mm, "dtype") and getattr(H, "_OUT_DTYPE", None):
    _MmDtype.default = torch.ops.aten.mm.dtype


def prepared(k):
    alias, opn, dtypes, sizes, strides, classes, scalars, device, state = H._local(k)
    packet = OPS[opn]
    if opn == "aten::mm" and dtypes[-1] != dtypes[0]:
        if _MmDtype.default is None:
            raise RuntimeError(f"a mixed-dtype mm key {dtypes} in a build whose harvest has no mm.dtype")
        packet = _MmDtype
    key = OpaqueKey(packet.default, dtypes, sizes, strides, tuple(0 if c == 256 else c for c in classes), scalars, device, state, alias)
    ops = [view(i, dt, z, st, c) for i, (dt, z, st, c) in enumerate(zip(dtypes, sizes, strides, classes))]
    return key, packet, ops[: NIN[opn]], ops[NIN[opn]], state


def call_args(packet, ins, out):
    # the traced call's args as the provider sees them: mm.dtype takes its out_dtype after the operands
    return (*ins, out.dtype) if packet is _MmDtype else tuple(ins)


def generic(k):
    """a key whose op is not one of the out= GEMMs (aten.mm.dtype_out, say): HostTraceReplay._learn on synthetic
    operands at the key's metadata, its call spec from the op's schema (tensor arguments in order, the key's scalars
    in the others' places), as the warm-up path learns any key"""
    import types
    import torch.utils._pytree as pytree
    from torch.cuda._host_trace_replay import HostTraceReplay
    alias, op, dtypes, sizes, strides, classes, scalars, device, state = H._local(k)
    key = OpaqueKey(op, dtypes, sizes, strides, tuple(0 if c == 256 else c for c in classes), scalars, device, state, alias)
    ntens = len(dtypes) - sum(1 for r in op._schema.returns if r.alias_info is None)  # inputs: operands less fresh outputs
    leaves, positions, sc = [], [], iter(scalars)
    args, kwargs = [], {}
    for arg in op._schema.arguments:
        is_t = isinstance(arg.type, torch.TensorType) or str(arg.type).startswith("Tensor")
        if is_t and len(positions) < ntens:
            positions.append(len(leaves)); leaves.append(object()); v = leaves[-1]
        elif is_t:
            continue
        else:
            try:
                v = next(sc)
            except StopIteration:
                continue
            leaves.append(v)
        (kwargs.__setitem__(arg.name, v) if arg.kwarg_only else args.append(v))
    spec = pytree.tree_flatten((tuple(args), kwargs))[1]
    C["generic"] += 1
    return HostTraceReplay._learn(types.SimpleNamespace(learned=0, learn_s=0.0), op, provider, (spec, frozenset(positions)), key)


def standard(k):
    if not isinstance(H._local(k)[1], str):
        return generic(k)
    key, packet, ins, out, state = prepared(k)
    with library_state_as(state), torch._C._AutoDispatchBelowADInplaceOrView():
        packet.out(*ins, out=out)
    C["standard"] += 1
    return provider.learn(key, call_args(packet, ins, out), {}, ins + [out])


def sibling(k):
    """the sibling route's binding, unchecked; None where the key takes no sibling (or holds a constant address)"""
    if not isinstance(H._local(k)[1], str):
        return None  # not an out= GEMM: the standard (generic) path
    key, packet, ins, out, state = prepared(k)
    nk = provider._normal(key)
    if nk in provider.bindings or provider.refusal(key) is not None:
        return None
    group = (H._OUT.get(key.op, key.op), key.dtypes, key.device, key.state)
    siblings = provider.templates.get(group)
    if not siblings:
        return None
    H._VERIFY_SIBLINGS = False
    try:
        with torch.cuda.device(key.device), _gc_hold, H._LOCK, library_state_as(state):
            b = provider._harvest(key, call_args(packet, ins, out), {}, ins + [out], siblings, True)
    except (H._NoSibling, H._Refused):
        return None
    finally:
        H._VERIFY_SIBLINGS = True
    provider.harvests += 1
    hits, misaligned = H._address_hits(b.nodes, torch._C._cuda_hostTraceStackBounds(), {})
    if hits or misaligned:
        C["address_fallback"] += 1
        return None  # a constant CUDA address: verify's mirror route and its records, in provider.learn
    return key, nk, packet, ins, out, state, b


def enqueue(item):
    """the in-place check of the binding, enqueued: eager's digests, then per fill the binding's"""
    key, nk, packet, ins, out, state, b = item
    ops = ins + [out]
    columns = [False] * len(ins) + [True]
    with library_state_as(state), torch._C._AutoDispatchBelowADInplaceOrView():
        packet.out(*ins, out=out)
    want, _ = torch._C._cuda_hostTraceDigests(ops, columns)
    ptrs = [t.data_ptr() for t in ops]
    gots = []
    reach = any(t.kind == "scratch" for n in b.nodes for t in H._slots(n))
    if not reach:
        C["unreachable_scratch"] += 1
    for fill in (0, 0xFF) if reach or not a.one_fill else (0,):
        out.view(H._BITS[out.element_size()]).fill_(-1)
        at, scratch = 0, []
        for n in b.scratch:
            SCRATCH[at : at + n].fill_(fill)
            scratch.append(SCRATCH.data_ptr() + at)
            at += -(-n // 512) * 512
        H._launch(b, ptrs, scratch, stream.cuda_stream)
        got, _ = torch._C._cuda_hostTraceDigests(ops, columns)
        gots.append(got)
    return want, gots


def flush(pending):
    if not pending:
        return
    checks = [enqueue(it) for it in pending]
    done = torch.cuda.Event()
    done.record()
    deadline = time.monotonic() + H._HANG
    while not done.query():
        if time.monotonic() > deadline + 1.0 * len(pending):
            raise RuntimeError("batched check did not finish (a hang: rerun with the standard verify)")
    bad = torch.stack([torch.stack([(w != g).any() for g in gs]).any() for w, gs in checks]).tolist()
    for it, wrong in zip(pending, bad):
        key, nk, *_, b = it
        if wrong:
            C["batch_failed"] += 1
            provider.harvests -= 1
            standard_keys.append(it)
            continue
        b = provider._compact(b)
        provider.bindings[nk] = b
        C["batched"] += 1
    pending.clear()


standard_keys: list = []
t0 = time.perf_counter()
cpu0 = time.process_time()
pending = []
by_key = {}
for k in entries:
    it = sibling(k)
    if it is None:
        flush(pending)  # a full harvest's template must not be ahead of checks it would change nothing for; keep order simple
        standard(k)
        continue
    by_key[id(it)] = k
    pending.append(it)
    if len(pending) >= a.batch:
        flush(pending)
flush(pending)
for it in standard_keys:
    standard(by_key[id(it)])
torch.cuda.synchronize()
t_learn = time.perf_counter() - t0
cpu_learn = time.process_time() - cpu0
t1 = time.perf_counter()
saved = provider.save(a.out)
R = {
    "keys": len(entries), "bindings": len(provider.bindings), "refused": len(provider.refused), "harvests": provider.harvests,
    "counts": dict(C), "full_templates": sum(len(v) for v in provider.templates.values()), "buffers_s": round(t_bufs, 2),
    "learn_wall_s": round(t_learn, 2), "learn_cpu_s": round(cpu_learn, 2), "per_key_ms_mean": round(1e3 * t_learn / max(len(entries), 1), 3),
    "save_s": round(time.perf_counter() - t1, 2), "saved": saved, "total_s": round(time.perf_counter() - t_start, 2),
    "loadavg": open("/proc/loadavg").read().split()[:3], "flags": vars(a),
}
print(json.dumps(R))
if a.json:
    json.dump(R, open(a.json, "w"), indent=1)
