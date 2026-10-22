"""DIAGNOSTIC ONLY (withdrawn 2026-10-13: these captures are the ports' per-launch witness checks; skipping them weakens a soundness check). capmemo (learnfast prototype; COPY, the shared callers stay untouched): memoize the witness captures that the trtllm-gen
attention port (fork_c5 flashinfer/trtllm_trace.record) and the vLLM _C launcher ports (vllm_ports/vllm_ops_trace
_Params.launch) take once per layer, by wrapping torch.cuda._host_trace_capture.capture_kernel_nodes for those two
callers only (recognized by their run closure).

Key: the caller's code, the binding / op, and each argument's metadata (tensors: dtype, sizes, strides, storage offset,
address % 256; others: their value). The first two captures of a key at different addresses classify every 8-byte word
of every node's parameters: equal in both (a constant), or one tensor's address + the same displacement in both (moves
with that tensor); anything else, or any other field differing, and the key is never memoized. From the third call on,
the nodes are the memo with each moving word re-based to this call's tensor. CAPMEMO_CHECK=1 also captures every call
and compares the predicted nodes field by field and byte by byte (counts in STATS; a mismatch uses the capture)."""
import dataclasses
import os

import torch
import torch.cuda._host_trace_capture as _cap

STATS = {"captures": 0, "memo_hits": 0, "checked": 0, "mismatch": 0, "unmemoizable_keys": 0, "passthrough": 0}
CHECK = os.environ.get("CAPMEMO_CHECK") == "1"
_orig = _cap.capture_kernel_nodes
_MEMO: dict = {}  # key -> ("first", nodes, addrs) | ("memo", template) | ("no",)


def _closure(fn):
    names = fn.__code__.co_freevars
    return {n: c.cell_contents for n, c in zip(names, fn.__closure__ or ())}


def _tensors_and_key(fn):
    mod = fn.__module__ or ""
    qn = fn.__qualname__
    cl = _closure(fn)
    if mod.endswith("trtllm_trace") and qn.endswith("record.<locals>.run"):
        leaves, owner = list(cl.get("stand_ins", ())), cl.get("binding")
    elif mod.endswith("vllm_ops_trace") and qn.endswith("launch.<locals>.run"):
        leaves, owner = list(cl.get("args", ())) + [cl["kwargs"][k] for k in sorted(cl.get("kwargs", {}))], getattr(cl.get("self"), "op", None)
    else:
        return None
    anchor = cl.get("anchor")
    tensors, sig = [], [fn.__code__, id(owner)]
    for x in [anchor, *leaves]:
        if isinstance(x, torch.Tensor):
            tensors.append(x)
            sig.append(("T", x.dtype, tuple(x.shape), x.stride(), x.storage_offset(), x.data_ptr() % 256))
        elif x is None or isinstance(x, (int, float, bool, str)):
            sig.append(("V", type(x).__name__, x))
        else:
            return None  # an argument this memo does not key on
    return tensors, tuple(sig)


def _span(t):
    n = 1 + sum((s - 1) * st for s, st in zip(t.shape, t.stride())) if all(t.shape) else 0
    return max(n, 1) * t.element_size()


def _fields(n):
    if isinstance(n, _cap.KernelNode):
        return (n.name, n.function, n.layout, n.grid, n.block, n.smem, n.attributes)
    return None


def _classify(n1, n2, a1, a2, spans):
    """per node: the image words that move with a tensor ((param, offset, tensor, delta)), or None if the pair is
    not explained by constants and moving words. Node 0 is the run's anchor fill, which both callers discard
    (`_, *nodes = capture_kernel_nodes(run)`): it is kept as captured, unclassified."""
    if len(n1) != len(n2) or not n1:
        return None
    plan = [("keep", None)]
    for x, y in zip(n1[1:], n2[1:]):
        if type(x) is not type(y):
            return None
        if isinstance(x, _cap.MemsetNode):
            fx, fy = dataclasses.astuple(x)[1:], dataclasses.astuple(y)[1:]
            if fx != fy:
                return None
            hit = [(i, x.dst - a) for i, a in enumerate(a1) if 0 <= x.dst - a < spans[i] and y.dst - a2[i] == x.dst - a]
            if x.dst == y.dst:
                plan.append(("memset", None))
            elif len({d for _, d in hit}) >= 1 and hit:
                plan.append(("memset", hit[0]))
            else:
                return None
            continue
        if not isinstance(x, _cap.KernelNode) or _fields(x) != _fields(y) or len(x.images) != len(y.images):
            return None
        moves = []
        for p, (ix, iy) in enumerate(zip(x.images, y.images)):
            if len(ix) != len(iy):
                return None
            if ix == iy:
                continue
            for off in range(0, len(ix), 8):
                wx, wy = ix[off : off + 8], iy[off : off + 8]
                if wx == wy:
                    continue
                if len(wx) < 8:
                    return None
                qx, qy = int.from_bytes(wx, "little"), int.from_bytes(wy, "little")
                hit = [(i, qx - a) for i, a in enumerate(a1) if 0 <= qx - a <= spans[i] and qy - a2[i] == qx - a]
                if not hit:
                    return None
                moves.append((p, off, *hit[0]))
        plan.append(("kernel", moves))
    return plan


def _predict(template, addrs):
    nodes, plan = template
    out = []
    for n, (kind, mv) in zip(nodes, plan):
        if kind == "keep":
            out.append(n)
            continue
        if kind == "memset":
            out.append(n if mv is None else dataclasses.replace(n, dst=addrs[mv[0]] + mv[1]))
            continue
        if not mv:
            out.append(n)
            continue
        images = [bytearray(i) for i in n.images]
        for p, off, i, d in mv:
            images[p][off : off + 8] = (addrs[i] + d).to_bytes(8, "little")
        out.append(dataclasses.replace(n, images=tuple(bytes(i) for i in images)))
    return out


def capture_kernel_nodes(fn, *a, **k):
    got = _tensors_and_key(fn) if not a and not k else None
    if got is None:
        STATS["passthrough"] += 1
        return _orig(fn, *a, **k)
    tensors, key = got
    addrs, spans = [t.data_ptr() for t in tensors], [_span(t) for t in tensors]
    ent = _MEMO.get(key)
    if ent is not None and ent[0] == "memo":
        pred = _predict(ent[1], addrs)
        STATS["memo_hits"] += 1
        if not CHECK:
            return pred
        real = _orig(fn)
        STATS["captures"] += 1
        STATS["checked"] += 1
        if real[1:] != pred[1:]:
            STATS["mismatch"] += 1
            return real
        return pred
    real = _orig(fn)
    STATS["captures"] += 1
    if ent is None:
        _MEMO[key] = ("first", real, addrs)
    elif ent[0] == "first" and all(x != y for x, y in zip(ent[2][1:], addrs[1:])):
        plan = _classify(ent[1], real, ent[2], addrs, spans)
        if plan is None:
            _MEMO[key] = ("no",)
            STATS["unmemoizable_keys"] += 1
        else:
            _MEMO[key] = ("memo", (real, plan))
            # re-base the template to this capture's addresses: _predict patches from these nodes
            _MEMO[key] = ("memo", (real, [(kind, ([(p, off, i, d) for p, off, i, d in mv] if kind == "kernel" else mv)) for kind, mv in plan]))
    return real


_cap.capture_kernel_nodes = capture_kernel_nodes
