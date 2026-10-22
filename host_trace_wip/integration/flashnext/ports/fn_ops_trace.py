# Python ports of the host launchers of the vLLM 0.29 _C ops Qwen3.8-Flash-Next's QSA indexer calls
# (vllm/models/qwen4_exp/nvidia/ops/qsa.py qsa_select_paged_tokens): _C.cooperative_topk (csrc/libtorch_stable/
# cooperative_topk.cu, <= 32 rows) and _C.persistent_topk (topk.cu). Same form as the shared ports
# (integration/vllm/vllm_ports/vllm_ops_trace.py): each launch is recorded with its fields as expressions of the trace's
# values, every branch on a symbolic value a guard, and checked byte for byte (function, name, grid, block, dynamic smem,
# launch attributes, PDL, parameter layout and bytes) against the real op's launch captured on stand-ins at the traced
# call's values; any difference declines. These kernels take one struct parameter and launch with a dynamic smem size
# and (cooperative_topk) an explicit cluster dimension, which the shared _Params.launch does not cover: _StructCall.launch
# below is that launch with a struct, smem and attributes. Compile-time constants (block size, tie workspace, smem sizes)
# are printed from the real header by ports/consts/topk_consts.cu (consts/topk_consts.json), not re-derived.
from __future__ import annotations

import functools
import json
import os
import struct
import sys
from typing import Any

import torch

sys.path.insert(0, os.environ.get("VLLM_PORTS_DIR", "/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/vllm/vllm_ports"))
import vllm_ops_trace as V  # noqa: E402

_C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "consts", "topk_consts.json")))


class _StructCall(V._Params):
    def launch_struct(self, kernel: tuple[int, int, str], size: int, members: list[tuple[int, str, Any]], grid: tuple, block: tuple,
                      smem: int, attributes: tuple = (), pdl: bool = False) -> None:
        """Record the launch of `kernel` with one struct parameter of `size` bytes, members (offset, format, value) with
        format P (a traced tensor's address), q, i, I or f; dynamic smem `smem` (an int); launch `attributes`
        ((CUlaunchAttributeID, value) as explicit_attributes has them). Checked against the real op's launch first."""
        from torch.cuda._host_trace_capture import capture_kernel_nodes, KernelNode, pack_params
        from torch.cuda._host_trace_launch import _GRID_LIMITS, KernelLaunch
        from torch.cuda._host_trace_tape import _PLACEHOLDER_LOW
        from torch.utils._python_dispatch import _disable_current_modes

        func, _, name = kernel
        image, mask = bytearray(size), bytearray(size)
        slots, fields, pointers = [], [], []
        for at, fmt, v in members:
            width = V._WIDTH[fmt]
            mask[at : at + width] = b"\xff" * width
            if fmt == "P":
                if not isinstance(v, torch.SymInt):
                    raise self.tr.decline(f"{self.what}: a pointer that is not a traced tensor's")
                slots.append(v)
                fields.append((0, at, 8))
                pointers.append(True)
            elif isinstance(v, torch.SymInt):
                lo, hi = V._LIMITS[fmt]
                self.require(v >= lo, f"a {fmt} member >= {lo}")
                self.require(v <= hi, f"a {fmt} member <= {hi}")
                slots.append(v)
                fields.append((0, at, width))
                pointers.append(False)
            else:
                struct.pack_into("<" + fmt, image, at, v)
        for axis, (n, limit) in enumerate(zip(grid, _GRID_LIMITS)):
            if isinstance(n, torch.SymInt):
                self.require(n >= 1, f"grid axis {axis} >= 1")
                self.require(n <= limit, f"grid axis {axis} <= {limit}")
            else:
                self.require(1 <= n <= limit, f"grid axis {axis} in [1, {limit}]")
        record = KernelLaunch(name=name, function=func, abi=None, layout=((0, size),), grid=grid, block=block, smem=smem,
                              slots=tuple(slots), roots=tuple(self.roots), fields=tuple(fields), attributes=tuple(attributes),
                              pointers=frozenset(i for i, p in enumerate(pointers) if p), images=(bytes(image),),
                              programmatic=pdl)
        value, pointer_value, stand = V._call_values(self.tr)
        args = [stand(a) for a in self.args]
        kwargs = {k: stand(v) for k, v in self.kwargs.items()}
        with _disable_current_modes():
            anchor = torch.empty(1, device=self.tr.device)

        def run(_: torch.cuda.Stream) -> None:
            with _disable_current_modes():
                anchor.fill_(0)
                self.op(*args, **kwargs)

        _, *nodes = capture_kernel_nodes(run)
        if len(nodes) != 1 or not isinstance(nodes[0], KernelNode):
            raise self.tr.decline(f"{self.what}: the op launched {nodes}")
        node = nodes[0]
        from cuda.bindings import driver

        pdl_id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        plain = V._plain(self.tr.device.index)
        explicit = tuple((k, v) for k, v in node.attributes if k != pdl_id and v != plain.get(k))
        at_call = [pointer_value(v) & _PLACEHOLDER_LOW if p else value(v) for v, p in zip(slots, pointers)]
        mine = pack_params(record, at_call, pointers)[0]
        real = node.images[0] if len(node.images) == 1 else b""
        evaluated = (func, name, tuple(map(value, grid)), tuple(map(value, block)), smem, tuple(attributes), pdl, ((0, size),))
        launched = (node.function, node.name, node.grid, node.block, node.smem, explicit,
                    bool(node.attribute("PROGRAMMATIC_STREAM_SERIALIZATION")), node.layout)
        bad = [i for i in range(size) if mask[i] and (i >= len(real) or mine[i] != real[i])]
        if evaluated != launched or bad:
            raise self.tr.decline(f"{self.what}: the op launched {launched}, the port {evaluated}; param bytes {bad[:8]} differ")
        self.tr.record_launch(record)


    def launch_params(self, kernel: tuple[int, int, str], params: list[tuple[str, Any]], grid: tuple, block: tuple, smem: int) -> None:
        """launch_struct for a kernel of plain parameters (one per (format, value), natural alignment), as
        cudaLaunchKernel(args) passes them: each parameter its own piece."""
        from torch.cuda._host_trace_capture import capture_kernel_nodes, KernelNode, pack_params
        from torch.cuda._host_trace_launch import _GRID_LIMITS, KernelLaunch
        from torch.cuda._host_trace_tape import _PLACEHOLDER_LOW
        from torch.utils._python_dispatch import _disable_current_modes

        func, _, name = kernel
        layout, images, slots, fields, pointers = [], [], [], [], []
        at = 0
        for p, (fmt, v) in enumerate(params):
            width = V._WIDTH[fmt]
            at = (at + width - 1) // width * width
            layout.append((at, width))
            at += width
            image = bytearray(width)
            if fmt == "P":
                if not isinstance(v, torch.SymInt):
                    raise self.tr.decline(f"{self.what}: a pointer that is not a traced tensor's")
                slots.append(v)
                fields.append((p, 0, 8))
                pointers.append(True)
            elif isinstance(v, torch.SymInt):
                lo, hi = V._LIMITS[fmt]
                self.require(v >= lo, f"a {fmt} parameter >= {lo}")
                self.require(v <= hi, f"a {fmt} parameter <= {hi}")
                slots.append(v)
                fields.append((p, 0, width))
                pointers.append(False)
            else:
                struct.pack_into("<" + fmt, image, 0, v)
            images.append(bytes(image))
        for axis, (n, limit) in enumerate(zip(grid, _GRID_LIMITS)):
            if isinstance(n, torch.SymInt):
                self.require(n >= 1, f"grid axis {axis} >= 1")
                self.require(n <= limit, f"grid axis {axis} <= {limit}")
            else:
                self.require(1 <= n <= limit, f"grid axis {axis} in [1, {limit}]")
        record = KernelLaunch(name=name, function=func, abi=None, layout=tuple(layout), grid=grid, block=block, smem=smem,
                              slots=tuple(slots), roots=tuple(self.roots), fields=tuple(fields), attributes=(),
                              pointers=frozenset(i for i, p in enumerate(pointers) if p), images=tuple(images))
        value, pointer_value, stand = V._call_values(self.tr)
        args = [stand(a) for a in self.args]
        kwargs = {k: stand(v) for k, v in self.kwargs.items()}
        with _disable_current_modes():
            anchor = torch.empty(1, device=self.tr.device)

        def run(_: torch.cuda.Stream) -> None:
            with _disable_current_modes():
                anchor.fill_(0)
                self.op(*args, **kwargs)

        _, *nodes = capture_kernel_nodes(run)
        if len(nodes) != 1 or not isinstance(nodes[0], KernelNode):
            raise self.tr.decline(f"{self.what}: the op launched {nodes}")
        node = nodes[0]
        from cuda.bindings import driver

        pdl_id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        plain = V._plain(self.tr.device.index)
        explicit = tuple((k, v) for k, v in node.attributes if k != pdl_id and v != plain.get(k))
        at_call = [pointer_value(v) & _PLACEHOLDER_LOW if p else value(v) for v, p in zip(slots, pointers)]
        mine = pack_params(record, at_call, pointers)
        evaluated = (func, name, tuple(map(value, grid)), tuple(map(value, block)), smem, (), False, tuple(layout))
        launched = (node.function, node.name, node.grid, node.block, node.smem, explicit,
                    bool(node.attribute("PROGRAMMATIC_STREAM_SERIALIZATION")), node.layout)
        bad = [p for p, (m, r) in enumerate(zip(mine, node.images)) if bytes(m) != r]
        if evaluated != launched or bad or len(node.images) != len(mine):
            raise self.tr.decline(f"{self.what}: the op launched {launched}, the port {evaluated}; params {bad[:8]} differ")
        self.tr.record_launch(record)


@functools.cache
def kernel_where(fragment: str, demangled_suffix: str) -> tuple[int, int, str]:
    """V.kernel of the one vllm:: kernel whose demangled name contains fragment and ends with demangled_suffix."""
    names = [V._demangle(n) for n in V._symtab(V._loaded(V.MODULE)) if n.startswith("_ZN4vllm") and fragment in n]
    hits = sorted({d for d in names if d and fragment in d and d.endswith(demangled_suffix)})
    if len(hits) != 1:
        raise LookupError(f"{fragment} ... {demangled_suffix}: {hits[:6]}")
    return V.kernel(hits[0])


def _cluster(cs: int) -> tuple:
    from cuda.bindings import driver

    return ((driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, (1, cs, 1)),)


@functools.cache
def _major(device: int) -> int:
    return torch.cuda.get_device_properties(device).major


# cooperative_topk(Tensor logits, Tensor lengths, Tensor! output, Tensor! workspace, int k, int max_seq_len) -> ()
def cooperative_topk(*args: Any, **kwargs: Any) -> None:
    op = torch.ops._C.cooperative_topk.default
    c = _StructCall(op, args, kwargs)
    a = V._bind(op, args, kwargs)
    logits, lengths, out, ws, k = a["logits"], a["lengths"], a["output"], a["workspace"], a["k"]
    c.require(logits.dtype == torch.float32 and logits.dim() == 2, "2-D float32 logits")
    c.require(lengths.dtype == torch.int32 and lengths.dim() in (1, 2) and V._dense(lengths), "contiguous 1-D / 2-D int32 lengths")
    c.require(out.dtype == torch.int32 and out.dim() == 2, "2-D int32 output")
    num_rows = logits.shape[0]
    c.require(lengths.numel() == num_rows, "lengths numel == rows")
    c.require(out.shape[0] == num_rows and out.shape[1] == k, "output [rows, k]")
    c.require(k in (512, 1024, 2048), "k 512 / 1024 / 2048")
    stride = logits.stride(0)
    c.require(num_rows <= 64, "rows <= 64")
    c.require(stride % 4 == 0, "stride % 4 == 0")
    c.require(ws.dtype == torch.uint8, "uint8 workspace")
    ties = _C["kMaxTies"] if k <= _C["kBlockSize"] else k
    c.require(ws.shape[0] >= num_rows * ties * _C["sizeof_Tie"], "workspace large enough")
    # launch_cooperative_topk_impl's cluster-size ladder (each a guard on the row count)
    if num_rows <= 4 and _major(c.tr.device.index) >= 10:
        cs, smem = 16, _C["kSmemSize8"]
    elif num_rows <= 8:
        cs, smem = 8, _C["kSmemSize8"]
    elif num_rows <= 33:
        cs, smem = 4, _C["kSmemSize4"]
    else:
        cs, smem = 2, _C["kSmemSize2"]
    fn = kernel_where(f"cooperative_topk_cs{cs}", f"cooperative_topk_cs{cs}<{k}u>(vllm::cooperative::CooperativeTopKParams<{k}u>)")
    members = [(0, "P", logits.data_ptr()), (8, "P", out.data_ptr()), (16, "P", lengths.data_ptr()), (24, "P", ws.data_ptr()),
               (32, "I", num_rows), (36, "I", stride)]
    c.launch_struct(fn, _C["sizeof_Params"], members, (num_rows, cs, 1), (_C["kBlockSize"], 1, 1), smem, _cluster(cs))


# persistent_topk(Tensor logits, Tensor lengths, Tensor! output, Tensor! workspace, int k, int max_seq_len) -> (): only its
# FilteredTopKRaggedTransform path (num_rows > 32 with >= 128 KiB opt-in smem, every GB300 call vLLM's QSA makes: it picks
# cooperative_topk at <= 32 rows); the persistent radix path (memset + cooperative kernel, occupancy-sized grid) declines.
def persistent_topk(*args: Any, **kwargs: Any) -> None:
    op = torch.ops._C.persistent_topk.default
    c = _StructCall(op, args, kwargs)
    a = V._bind(op, args, kwargs)
    logits, lengths, out, k = a["logits"], a["lengths"], a["output"], a["k"]
    c.require(logits.dtype == torch.float32 and logits.dim() == 2, "2-D float32 logits")
    c.require(lengths.dtype == torch.int32 and lengths.dim() in (1, 2) and V._dense(lengths), "contiguous 1-D / 2-D int32 lengths")
    c.require(out.dtype == torch.int32 and out.dim() == 2, "2-D int32 output")
    num_rows = logits.shape[0]
    c.require(lengths.numel() == num_rows, "lengths numel == rows")
    c.require(out.shape[0] == num_rows and out.shape[1] == k, "output [rows, k]")
    c.require(k in (512, 1024, 2048), "k 512 / 1024 / 2048")
    c.require(torch.cuda.get_device_properties(c.tr.device.index).shared_memory_per_block_optin >= 128 * 1024, "128 KiB opt-in smem")
    c.require(num_rows > 32, "rows > 32 (the FilteredTopK path; the persistent radix path is not ported)")
    stride = logits.stride(0)
    # ComputeFilteredTopKVecSize<float>(max_len = stride): gcd(stride, 4); each a guard
    vs = 4 if stride % 4 == 0 else 2 if stride % 2 == 0 else 1
    fn = kernel_where("FilteredTopKUnifiedKernel", f"FilteredTopKUnifiedKernel<float, int, {vs}, {k}u, {'false' if vs == 4 else 'true'}>"
                      "(float const*, int*, int const*, unsigned int, unsigned int, unsigned int)")
    params = [("P", logits.data_ptr()), ("P", out.data_ptr()), ("P", lengths.data_ptr()), ("I", num_rows), ("I", k), ("I", stride)]
    c.launch_params(fn, params, (num_rows, 1, 1), (_C["FILTERED_TOPK_BLOCK_THREADS"], 1, 1), _C["FILTERED_TOPK_SMEM_DYNAMIC"])


LAUNCHERS = {("_C", "cooperative_topk"): cooperative_topk, ("_C", "persistent_topk"): persistent_topk}


def install() -> list[Any]:
    import torch.cuda._host_trace as ht

    done = []
    for (ns, n), f in LAUNCHERS.items():
        o = getattr(getattr(torch.ops, ns), n).default
        ht.register_traced_impl(o, V._loud(o, f))
        done.append(o)
    ht.traced_impls = True
    return done
