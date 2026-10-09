# trtllm-gen paged attention under CUDA graph host tracing, through FlashInfer's own C++ launcher: under a trace
# the two bindings run as flashinfer_ht_cpp custom ops whose kernel calls the host-trace build of the launcher
# (land/core/trtllm_cpp: the patched csrc/trtllm_fmha_kernel_launcher.cu and include/flashinfer/trtllm/fmha with
# its sizes and addresses as c10::SymInt), which records the FMHA launch (and the LSE launch) with every field an
# expression of the trace's values and every branch on one a guard of the op. Outside a trace the bindings run
# unchanged (FlashInfer's stock module).
from __future__ import annotations

import ctypes
import functools
import os
import sys
from typing import Any

import torch

_ROOT = os.environ.get("FI_HT_CPP_ROOT", "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/trtllm_cpp")
# check each recorded launch byte for byte against the stock binding's, captured on stand-ins at the hints
CHECK = os.environ.get("FI_HT_CPP_CHECK", "1") == "1"
DECODE, PREFILL = "decode", "context"

# trtllm Data_type
_DTYPES = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2, torch.int8: 3, torch.int32: 4,
           torch.float8_e4m3fn: 5, torch.float8_e5m2: 6, torch.uint8: 7}


@functools.cache
def ext() -> Any:
    """The traced build (build/fi_ht_trtllm.so), its cubin loader FlashInfer's own callback."""
    from flashinfer.jit.cubin_loader import setup_cubin_loader

    build = os.environ.get("FI_HT_BUILD", os.path.join(_ROOT, "build"))
    sys.path.insert(0, build)
    try:
        import fi_ht_trtllm
    finally:
        sys.path.remove(build)
    setup_cubin_loader(fi_ht_trtllm.__file__)
    # FlashInfer's own module, loaded (its binding is what calls here): found among this process's mappings, not
    # through FlashInfer's JIT (which takes the shared cache's file locks)
    with open("/proc/self/maps") as f:
        paths = {line.split()[-1] for line in f if line.rstrip().endswith("/fmha_gen.so")}
    if len(paths) != 1:
        raise RuntimeError(f"FlashInfer's fmha_gen module is not loaded once: {sorted(paths)}")
    path = paths.pop()
    # its kernels' host stubs are local symbols: their values from its symbol table
    import subprocess

    out = subprocess.run(["nm", "--defined-only", path], capture_output=True, text=True, check=True).stdout
    symbols = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in "tT" and parts[2].startswith("_ZN10flashinfer") and "Kernel" in parts[2]:
            symbols[parts[2]] = int(parts[0], 16)
    fi_ht_trtllm.set_stock_library(path, symbols)
    return fi_ht_trtllm


# each binding argument's kind, in the binding's order: T tensor (! written, ? optional), S SymInt, i int,
# f float, b bool, V Variant<double, Tensor> (two op arguments)
_SPECS = {
    DECODE: "out:T!,out_sf:T!?,q:T,k:T,v:T,ws:T!,counter:T!,pages:T,seq_lens:T,max_q:S,max_kv:S,bmm1:V,bmm2:V,o_sf_scale:f,"
    "o_sf_vec:i,o_sf_start:i,batch:S,window_left:i,sparse_top_k:i,sm_count:i,pdl:b,ws_size:S,sinks:T?,cum_q:T?,k_sf:T?,"
    "v_sf:T?,skip:f?,shared:b?,lse:T!?,lse_st_tok:S,lse_st_heads:S,block_sparse:b,top_k_lens:T?,transform:i,fp16:b?",
    PREFILL: "out:T!,out_sf:T!?,q:T,k:T,v:T,ws:T!,counter:T,pages:T,seq_lens:T,max_q:S,max_kv:S,bmm1:V,bmm2:V,o_sf_scale:f,"
    "o_sf_vec:i,o_sf_start:i,batch:S,window_left:i,cum_q:T,cum_kv:T,sm_count:i,pdl:b,ws_size:S,sinks:T?,k_sf:T?,v_sf:T?,"
    "skip:f?,shared:b?,fp16:b?,spcompress:b?,causal:b,lse:T!?,lse_st_tok:S,lse_st_heads:S",
}
_TYPES = {"T": "Tensor", "T?": "Tensor?", "S": "SymInt", "i": "SymInt", "f": "float", "f?": "float?", "b": "bool", "b?": "bool?"}
_BINDINGS: dict[str, Any] = {}
# tests: each ordinary call's (kind, binding, args) while a list
RECORD: list | None = None
_OPS: dict[str, Any] = {}


def _spec(kind: str) -> list[tuple[str, str]]:
    return [tuple(a.split(":")) for a in _SPECS[kind].split(",")]  # type: ignore[misc]


def _to_binding(kind: str, flat: tuple) -> tuple:
    out, it = [], iter(flat)
    for _, ty in _spec(kind):
        if ty == "V":
            value, tensor = next(it), next(it)
            out.append(value if tensor is None else tensor)
        else:
            out.append(next(it))
    return tuple(out)


def _to_op(kind: str, args: tuple) -> tuple:
    flat: list[Any] = []
    for (_, ty), a in zip(_spec(kind), args, strict=True):
        if ty == "V":
            flat += [1.0, a] if isinstance(a, torch.Tensor) else [float(a), None]
        else:
            flat.append(a)
    return tuple(flat)


def _hint(v: Any) -> Any:
    from torch.cuda._host_trace_tape import _hint as hint

    return hint(v) if isinstance(v, torch.SymInt) else v


def _pieces(size: int, tmas: list) -> tuple[tuple[int, int], ...]:
    """The struct parameter split so each CUtensorMap is a whole piece (KernelLaunch.packed)."""
    cuts = sorted({0, size, *(t[1] for t in tmas), *(t[1] + 128 for t in tmas)})
    return tuple((a, b - a) for a, b in zip(cuts, cuts[1:]))


def _cluster_attributes(cluster_x: int, policy: int) -> tuple[tuple[Any, Any], ...]:
    """The launch attributes run() sets for a cluster width, as explicit_attributes reads them off its node."""
    from cuda.bindings import driver

    from torch.cuda._host_trace_capture import plain_attributes

    a = driver.CUlaunchAttributeID
    pairs = [(a.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, (cluster_x, 1, 1))]
    if policy:
        pairs.append((a.CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, policy))
    plain = plain_attributes(torch.cuda.current_device())
    return tuple((k, v) for k, v in pairs if v != plain.get(k))


def launches(rec: dict, roots: tuple) -> Any:
    """A record of the traced build as a KernelLaunch."""
    from torch.cuda._host_trace_launch import KernelLaunch, TmaDescriptor

    if tuple(rec["cluster"][1:]) != (1, 1):
        raise ValueError(f"a cluster shape {rec['cluster']}")
    attributes = _cluster_attributes(rec["cluster"][0], rec["policy"])
    slots: list[Any] = []
    fields: list[tuple[int, int, int]] = []
    pointers: set[int] = set()
    if rec["packed"]:
        (image,) = rec["params"]
        layout = _pieces(len(image), rec["tmas"])
        starts = [a for a, _ in layout]

        def piece(at: int) -> tuple[int, int]:
            k = max(i for i, a in enumerate(starts) if a <= at)
            return k, at - starts[k]

        for _, at, width, value, pointer in rec["fields"]:
            k, off = piece(at)
            if pointer:
                pointers.add(len(slots))
            fields.append((k, off, width))
            slots.append(value)
        descriptors = []
        for _, at, dtype, address, shape, strides, box, swizzle, fill in rec["tmas"]:
            k, off = piece(at)
            assert off == 0
            descriptors.append(TmaDescriptor(k, len(slots), dtype, tuple(box), swizzle, fill, edits=False))
            pointers.add(len(slots))
            slots += [address, *shape, *strides]
        images = tuple(bytes(image[a : a + n]) for a, n in layout)
        return KernelLaunch(rec["name"], rec["function"], None, layout, tuple(rec["grid"]), tuple(rec["block"]), rec["smem"],
                            tuple(slots), roots, ext(), tuple(fields), tuple(descriptors), attributes, frozenset(pointers),
                            images, programmatic=rec["pdl"], packed=True)
    layout = tuple((off, len(p)) for off, p in zip(rec["offsets"], rec["params"]))
    for param, at, width, value, pointer in rec["fields"]:
        if pointer:
            pointers.add(len(slots))
        fields.append((param, at, width))
        slots.append(value)
    # a runtime launch (LSE) sets no cluster: its node has none
    return KernelLaunch(rec["name"], rec["function"], None, layout, tuple(rec["grid"]), tuple(rec["block"]), rec["smem"],
                        tuple(slots), roots, ext(), tuple(fields), (), (), frozenset(pointers),
                        tuple(bytes(p) for p in rec["params"]), programmatic=rec["pdl"])


def record(kind: str, binding: Any, args: tuple) -> None:
    """The binding's launches on the current trace's traced tensors, recorded by the traced build of FlashInfer's
    launcher. A decline (a case the build does not reproduce) leaves the call eager."""
    from torch.cuda._host_trace_cute import _stand_in
    from torch.cuda._host_trace_tape import _PLACEHOLDER_LOW, _TracedTensor, current_trace

    tr = current_trace()
    if torch.cuda.current_device() != tr.device.index or torch.cuda.current_stream() != tr.stream:
        raise tr.decline(f"trtllm-gen {kind}: called off the trace's device or stream")
    if any(isinstance(a, torch.Tensor) and not isinstance(a, _TracedTensor) for a in args):
        raise tr.decline(f"trtllm-gen {kind}: a tensor the trace does not track")
    try:
        recs = (ext().decode if kind == DECODE else ext().context)(*args)
    except NotImplementedError as e:
        if "fi_ht:" not in str(e):
            raise
        raise tr.decline(f"trtllm-gen {kind}: {str(e).split('fi_ht: ', 1)[1].splitlines()[0]}") from None
    roots: list[Any] = []
    for a in args:
        if isinstance(a, _TracedTensor) and not any(r is a._root for r in roots):
            roots.append(a._root)
    records = [launches(r, tuple(roots)) for r in recs]
    if CHECK:
        stand_ins = [_stand_in(a) if isinstance(a, _TracedTensor) else _hint(a) for a in args]
        bad = compare(records, binding, stand_ins, tr.device, lambda v, pointer: _hint(v) & _PLACEHOLDER_LOW if pointer else _hint(v))
        if bad:
            raise tr.decline(f"trtllm-gen {kind}: the traced build's launches are not the binding's: {bad[:4]}")
    for r in records:
        tr.record_launch(r)


def captured(binding: Any, args: list, device: Any) -> list:
    """The binding's own launches on `args`, captured (never run)."""
    from torch.cuda._host_trace_capture import capture_kernel_nodes
    from torch.utils._python_dispatch import _disable_current_modes

    with _disable_current_modes():
        anchor = torch.empty(1, device=device)

    def run(_: torch.cuda.Stream) -> None:
        with _disable_current_modes():
            anchor.fill_(0)
            binding(*args)

    _, *nodes = capture_kernel_nodes(run)
    return nodes


def compare(records: list, binding: Any, args: list, device: Any, value: Any, nodes: list | None = None) -> list[str]:
    """Each difference between the launches `records` evaluate to (value(slot, is_pointer) per slot) and the
    binding's own launches on `args` (`nodes`, else captured): kernel, grid, block, smem, launch attributes and every
    parameter byte."""
    from torch.cuda._host_trace_capture import explicit_attributes, KernelNode, pack_params

    if nodes is None:
        nodes = captured(binding, args, device)
    if len(nodes) != len(records) or any(not isinstance(n, KernelNode) for n in nodes):
        return [f"the binding launched {[getattr(n, 'name', n) for n in nodes]}, the traced build {[r.name for r in records]}"]
    bad = []
    for launch, node in zip(records, nodes):
        pointers = [i in launch.pointers for i in range(len(launch.slots))]
        mine = b"".join(pack_params(launch, [value(v, q) for v, q in zip(launch.slots, pointers)], pointers))
        real = b"".join(node.images)
        for what, a, b in (("name", launch.name, node.name), ("grid", tuple(value(v, False) for v in launch.grid), tuple(node.grid)),
                           ("block", tuple(value(v, False) for v in launch.block), tuple(node.block)), ("smem", launch.smem, node.smem),
                           ("attributes", launch.attributes, explicit_attributes(node))):
            if a != b:
                bad.append(f"{node.name} {what}: traced {a}, binding {b}")
        if mine != real:
            diff = [i for i in range(max(len(mine), len(real))) if i >= len(mine) or i >= len(real) or mine[i] != real[i]]
            bad.append(f"{node.name}: {len(diff)} param bytes differ from byte {diff[:8]}")
    return bad


def parity(kind: str, binding: Any, args: tuple) -> list[str]:
    """The traced build at one concrete call (real tensors, no trace) against the binding's own launch on them;
    ["the binding launched []"] where the stock launcher launches nothing (its cubin is missing: it only logs)."""
    nodes = captured(binding, list(args), args[0].device)
    if not nodes:
        return ["the binding launched []"]
    recs = (ext().decode if kind == DECODE else ext().context)(*args)
    return compare([launches(r, ()) for r in recs], binding, list(args), args[0].device, lambda v, pointer: int(v), nodes)


def _dtype(t: torch.Tensor) -> int:
    return _DTYPES[t.dtype]


def preload(kind: str, args: Any) -> None:
    """At a host trace's warm-up: load every cubin of the call's class (its dtypes, head dims, page size and phase)
    that the cubin cache has, which a trace or a redispatch cannot load under its capture."""
    out, q, k = args[0], args[2], args[3]
    names = ext().candidates(_dtype(q), _dtype(k), _dtype(out), kind == DECODE, q.shape[-1], out.shape[-1], k.shape[-2])
    from flashinfer.artifacts import ArtifactPath
    from flashinfer.jit import env as jit_env

    have = set(os.listdir(os.path.join(jit_env.FLASHINFER_CUBIN_DIR, ArtifactPath.TRTLLM_GEN_FMHA)))
    for name in names:
        if f"{name}.cubin" in have:
            ext().load(_dtype(q), _dtype(k), _dtype(out), name)


def _op(kind: str) -> Any:
    if kind in _OPS:
        return _OPS[kind]
    params, written = [], []
    for name, ty in _spec(kind):
        if ty == "V":
            params += [f"float {name}", f"Tensor? {name}_t"]
        elif ty.startswith("T!"):
            params.append(f"Tensor({chr(ord('a') + len(written))}!){ty[2:]} {name}")
            written.append(name)
        else:
            params.append(f"{_TYPES[ty]} {name}")

    def kernel(*flat: Any) -> None:
        from torch.cuda._host_trace_tape import _TracedTensor, current_trace

        args = _to_binding(kind, flat)
        if current_trace() is not None and any(isinstance(a, _TracedTensor) for a in args):
            try:
                record(kind, _BINDINGS[kind], args)
            except BaseException:
                if os.environ.get("FI_HT_CPP_DEBUG"):
                    import traceback

                    traceback.print_exc()
                raise
            return
        _BINDINGS[kind](*args)

    op = torch.library.custom_op(f"flashinfer_ht_cpp::trtllm_paged_attention_{kind}", kernel, mutates_args=written,
                                 device_types="cuda", schema=f"({', '.join(params)}) -> ()")
    op.register_fake(lambda *flat: None)
    _OPS[kind] = op
    return op


def traced(binding: Any, kind: str) -> Any:
    """binding, run through the custom op under a dispatch mode (a host trace's, and its warm-up's witness), so the
    trace records the traced build's launches; else the binding itself."""
    _BINDINGS[kind] = binding
    op = _op(kind)

    def call(*args: Any) -> Any:
        if torch._C._len_torch_dispatch_stack():
            from torch.cuda._host_trace_tape import _TracedTensor

            if not any(isinstance(a, _TracedTensor) for a in args):
                preload(kind, args)
            return op(*_to_op(kind, args))
        if RECORD is not None:
            RECORD.append((kind, binding, args))
        return binding(*args)

    return call
