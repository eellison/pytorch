# Python ports of the host launchers of five vLLM _C ops (csrc/libtorch_stable layernorm_kernels.cu rms_norm and
# fused_add_rms_norm, pos_encoding_kernels.cu rotary_embedding, activation_kernels.cu silu_and_mul, cache_kernels.cu
# reshape_and_cache_flash), registered with torch.cuda._host_trace.register_traced_impl (src_vllmcpp CHANGES item 21).
# Each call records its kernel launch as a KernelLaunch whose fields are expressions of the trace's values, checked
# byte for byte (function, name, grid, block, smem, attributes, PDL, parameter layout and bytes) against the real op's
# launch captured on stand-ins at the hints; any difference declines and the call takes the extern harvest. The helpers
# are sglang/armF/externport/sgl_ops_trace.py's (imported, not copied). The kernels take plain parameters (no struct)
# and launch with <<<>>> (no PDL).
from __future__ import annotations

import ctypes
import functools
import math
import os
import struct
import sys
from typing import Any

import torch

sys.path.insert(0, "/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/armF/externport")
from sgl_ops_trace import _bind, _Call, _cudart_check, _demangle, _loaded, _plain, _select, _symtab  # noqa: E402

MODULE = "_C_stable_libtorch.abi3"
_EXPORT = "PyInit__C_stable_libtorch"
_LIMITS = {"I": (0, 2**32 - 1), "i": (-(2**31), 2**31 - 1), "q": (-(2**63), 2**63 - 1)}
_WIDTH = {"P": 8, "q": 8, "I": 4, "i": 4, "f": 4, "?": 1}
# vllm_is_batch_invariant(): VLLM_BATCH_INVARIANT, read once by the C++ (a static)
BATCH_INVARIANT = bool(int(os.environ.get("VLLM_BATCH_INVARIANT", "0") or 0))
_BF16 = torch.bfloat16


@functools.cache
def kernel(demangled: str) -> tuple[int, int, str]:
    """sgl_ops_trace.kernel for vllm::: the static __global__ stub found from the .symtab, cudaGetFuncBySymbol
    asked of each mapped libcudart copy."""
    path = _loaded(MODULE)
    symbols = _symtab(path)
    mangled = [n for n in symbols if n.startswith("_ZN4vllm") and _demangle(n) == demangled]
    if len(mangled) != 1:
        raise LookupError(f"{demangled} in {path}: {mangled}")
    lib = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
    stub = ctypes.cast(getattr(lib, _EXPORT), ctypes.c_void_p).value - symbols[_EXPORT] + symbols[mangled[0]]
    with open("/proc/self/maps") as maps:
        copies = sorted({line.split()[-1] for line in maps if "/libcudart" in line})
    errors = []
    for copy in copies:
        func = ctypes.c_void_p()
        err = ctypes.CDLL(copy, mode=os.RTLD_NOLOAD).cudaGetFuncBySymbol(ctypes.byref(func), ctypes.c_void_p(stub))
        if err == 0:
            break
        errors.append(err)
    else:
        _cudart_check(errors[-1] if errors else -1, f"cudaGetFuncBySymbol({mangled[0]}) in {copies}")
    from cuda.bindings import driver

    from torch.cuda._utils import _check_cuda_bindings

    # the device function's own name: a stub's mangling can differ from it beyond the internal-linkage L (an enable_if
    # return type)
    return func.value, stub, _check_cuda_bindings(driver.cuFuncGetName(func.value)).decode()


class _Params(_Call):
    def launch(self, kernel: tuple[int, int, str], params: list[tuple[str, Any]], grid: tuple, block: tuple) -> None:  # type: ignore[override]
        """Record the launch of `kernel` with one kernel parameter per (format, value), format P (a pointer), q, i, I,
        f or ? (bool), at natural alignment, after checking it against the real op's launch."""
        from torch.cuda._host_trace_capture import capture_kernel_nodes, KernelNode, pack_params
        from torch.cuda._host_trace_cute import _stand_in
        from torch.cuda._host_trace_launch import _GRID_LIMITS, KernelLaunch
        from torch.cuda._host_trace_tape import _hint, _PLACEHOLDER_LOW, _TracedTensor
        from torch.utils._python_dispatch import _disable_current_modes

        func, _, name = kernel
        layout, images, slots, fields, pointers = [], [], [], [], []
        at = 0
        for p, (fmt, v) in enumerate(params):
            width = _WIDTH[fmt]
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
                lo, hi = _LIMITS[fmt]
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
        record = KernelLaunch(name=name, function=func, abi=None, layout=tuple(layout), grid=grid, block=block, smem=0,
                              slots=tuple(slots), roots=tuple(self.roots), fields=tuple(fields), attributes=(),
                              pointers=frozenset(i for i, p in enumerate(pointers) if p), images=tuple(images),
                              programmatic=False)

        def stand(a: Any) -> Any:
            return _stand_in(a) if isinstance(a, _TracedTensor) else _hint(a)

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

        pdl = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        plain = _plain(self.tr.device.index)
        explicit = tuple((k, v) for k, v in node.attributes if k != pdl and v != plain.get(k))
        hints = [_hint(v) & _PLACEHOLDER_LOW if p else _hint(v) for v, p in zip(slots, pointers)]
        mine = pack_params(record, hints, pointers)
        evaluated = (func, name, tuple(map(_hint, grid)), tuple(map(_hint, block)), 0, (), False, tuple(layout))
        launched = (node.function, node.name, node.grid, node.block, node.smem, explicit,
                    bool(node.attribute("PROGRAMMATIC_STREAM_SERIALIZATION")), node.layout)
        bad = [p for p, (m, r) in enumerate(zip(mine, node.images)) if bytes(m) != r]
        if evaluated != launched or bad or len(node.images) != len(mine):
            raise self.tr.decline(f"{self.what}: the op launched {launched}, the port {evaluated}; params {bad[:8]} differ")
        self.tr.record_launch(record)


def _rows(t: torch.Tensor) -> Any:
    return math.prod(t.shape[:-1])


def _dense(t: torch.Tensor) -> bool:
    # is_contiguous without its size-1 exemption (no guard on T == 1): a decline there instead
    return all(bool(t.stride(i) == math.prod(t.shape[i + 1 :])) for i in range(t.dim()))


def _max_block(num_tokens: Any) -> Any:
    # max_block_size: 1024 batch-invariant, else 1024 below 256 tokens, 256 from there
    return 1024 if BATCH_INVARIANT else _select(num_tokens < 256, 1024, 256)


def _min(a: int, b: Any) -> Any:
    """min(a, b) for an int a and a select b of ints: a select of the mins, no guard."""
    from torch.cuda._host_trace import select

    if not isinstance(b, torch.SymInt):
        return min(a, b)
    return select(b <= a, b, a)


# ---- rms_norm(Tensor! result, Tensor input, Tensor? weight, float epsilon)


def rms_norm(*args: Any, **kwargs: Any) -> None:
    op = torch.ops._C.rms_norm.default
    c = _Params(op, args, kwargs)
    a = _bind(op, args, kwargs)
    out, x, w, eps = a["result"], a["input"], a["weight"], a["epsilon"]
    c.require(x.dtype == out.dtype == _BF16 and (w is None or w.dtype == _BF16), "bf16 result, input and weight (cross-checked)")
    c.require(x.dim() in (2, 3), "rank 2 or 3 input (the bf16 instantiations)")
    c.require(_dense(out), "a contiguous result")
    c.require(x.stride(-1) == 1, "input inner stride 1 (else the op copies it first)")
    hidden = int(x.shape[-1])  # the vector width and kernel follow it: a guard
    if w is not None:
        c.require(w.dim() == 1 and w.shape[0] == hidden and w.stride(0) == 1, "a 1-d contiguous weight of the hidden size")
    rank = x.dim()
    num_tokens = _rows(x)
    vec = math.gcd(8, hidden)
    block = _min(hidden // vec, _max_block(num_tokens))
    fn = kernel(f"void vllm::rms_norm_kernel<c10::BFloat16, {vec}, {rank}, {str(w is not None).lower()}>(c10::BFloat16*, "
                "c10::BFloat16 const*, long, long, long, long, long, c10::BFloat16 const*, long, float, int, int)")
    params = [("P", out.data_ptr()), ("P", x.data_ptr()), ("q", x.stride(-2)), ("q", x.stride(-3) if rank >= 3 else 0), ("q", 0),
              ("q", x.shape[-2] if rank >= 3 else 0), ("q", 0), ("P", w.data_ptr()) if w is not None else ("q", 0), ("q", 0),
              ("f", eps), ("i", num_tokens), ("i", hidden)]
    c.launch(fn, params, (num_tokens, 1, 1), (block, 1, 1))


# ---- fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, float epsilon)

# the host stubs of both widths demangle as the plain overload's; the device function of width 8 is the enable_if one
# (named by cuFuncGetName in kernel())
_FUSED_ADD = ("std::enable_if<(({w})==(0))||(!vllm::_typeConvert<c10::BFloat16>::exists), void>::type "
              "vllm::fused_add_rms_norm_kernel<c10::BFloat16, {w}, {h}>(c10::BFloat16*, long, c10::BFloat16*, "
              "c10::BFloat16 const*, float, int, int, long)")


def fused_add_rms_norm(*args: Any, **kwargs: Any) -> None:
    op = torch.ops._C.fused_add_rms_norm.default
    c = _Params(op, args, kwargs)
    a = _bind(op, args, kwargs)
    x, res, w, eps = a["input"], a["residual"], a["weight"], a["epsilon"]
    c.require(x.dtype == res.dtype == _BF16 and (w is None or w.dtype == _BF16), "bf16 input, residual and weight (cross-checked)")
    c.require(res.stride(-1) == 1, "residual inner stride 1")
    c.require(x.dim() >= 2 and res.dim() >= 2, "input and residual of rank >= 2")
    hidden = int(x.shape[-1])
    if w is not None:
        c.require(_dense(w), "a contiguous weight")
    num_tokens = _rows(x)
    in_stride, res_stride = x.stride(-2), res.stride(-2)
    block = _min(hidden, _max_block(num_tokens))
    # the 8-wide kernel: 16-byte aligned pointers and offsets that are multiples of 8 elements (a kernel switch: a guard)
    ptrs = [x.data_ptr(), res.data_ptr(), *([w.data_ptr()] if w is not None else [])]
    wide = not BATCH_INVARIANT and hidden % 8 == 0 and all(bool(p % 16 == 0) for p in ptrs) and bool(in_stride % 8 == 0) and bool(res_stride % 8 == 0)
    fn = kernel(_FUSED_ADD.format(w=8 if wide else 0, h=str(w is not None).lower()))
    params = [("P", x.data_ptr()), ("q", in_stride), ("P", res.data_ptr()), ("P", w.data_ptr()) if w is not None else ("q", 0),
              ("f", eps), ("i", num_tokens), ("i", hidden), ("q", res_stride)]
    c.launch(fn, params, (num_tokens, 1, 1), (block, 1, 1))


# ---- rotary_embedding(Tensor positions, Tensor! query, Tensor!? key, int head_size, Tensor cos_sin_cache, bool is_neox,
#      int rope_dim_offset=0, bool inverse=False)


def rotary_embedding(*args: Any, **kwargs: Any) -> None:
    op = torch.ops._C.rotary_embedding.default
    c = _Params(op, args, kwargs)
    a = _bind(op, args, kwargs)
    pos, q, k, hs, cache = a["positions"], a["query"], a["key"], a["head_size"], a["cos_sin_cache"]
    neox, offset, inverse = a["is_neox"], a["rope_dim_offset"], a["inverse"]
    c.require(pos.dtype == torch.int64, "int64 positions")
    c.require(q.dtype == cache.dtype == _BF16 and (k is None or k.dtype == _BF16), "bf16 query, key and cos_sin_cache (cross-checked)")
    pd = pos.dim()
    c.require(pd in (1, 2) and cache.dim() == 2, "positions [T] or [B, S]; cos_sin_cache 2-d")
    c.require(q.dim() > pd and (k is None or k.dim() > pd), "query and key of more dims than positions")
    for t in (q, k) if k is not None else (q,):
        for i in range(pd):
            c.require(t.shape[i] == pos.shape[i], "query, key and positions of one token shape")
    num_tokens = pos.numel()
    qh = int(math.prod(q.shape[pd:]))
    kh = int(math.prod(k.shape[pd:])) if k is not None else 0
    c.require(qh % hs == 0 and kh % hs == 0, "query and key widths multiples of head_size")
    nh = qh // hs
    nkv = kh // hs if k is not None else nh
    c.require(nh % nkv == 0, "heads a multiple of kv heads")
    rot = int(cache.shape[1])
    c.require(rot + offset <= hs, "rot_dim + rope_dim_offset <= head_size")
    head_stride = q.stride(-2) if q.dim() == pd + 2 else hs
    fn = kernel(f"void vllm::rotary_embedding_kernel<c10::BFloat16, c10::BFloat16, {str(bool(neox)).lower()}>(long const*, "
                "c10::BFloat16*, c10::BFloat16*, c10::BFloat16 const*, int, long, long, long, int, int, int, long, bool)")
    params = [("P", pos.data_ptr()), ("P", q.data_ptr()), ("P", k.data_ptr()) if k is not None else ("q", 0), ("P", cache.data_ptr()),
              ("i", rot), ("q", q.stride(pd - 1)), ("q", k.stride(pd - 1) if k is not None else 0), ("q", head_stride),
              ("i", nh), ("i", nkv), ("i", hs), ("q", offset), ("?", bool(inverse))]
    c.launch(fn, params, (num_tokens, 1, 1), (min(nh * rot // 2, 512), 1, 1))


# ---- silu_and_mul(Tensor! result, Tensor input)

_ACT = ("void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat16>"
        "(c10::BFloat16 const&, float)), &(__nv_bfloat162 vllm::packed_silu_kernel<__nv_bfloat162>(__nv_bfloat162 const&, "
        "float)), true, {vec}, false, {wide}>(c10::BFloat16*, c10::BFloat16 const*, int, float, float, float)")


def silu_and_mul(*args: Any, **kwargs: Any) -> None:
    op = torch.ops._C.silu_and_mul.default
    c = _Params(op, args, kwargs)
    a = _bind(op, args, kwargs)
    out, x = a["result"], a["input"]
    c.require(x.dtype == out.dtype == _BF16, "bf16 result and input (cross-checked)")
    c.require(torch.cuda.get_device_capability(x.device)[0] >= 10, "cc >= 10 (built with CUDA >= 12.9)")
    c.require(_dense(x) and _dense(out), "contiguous input and result (the kernel's row offsets)")
    d = int(x.shape[-1]) // 2
    num_tokens = _rows(x)
    c.require(num_tokens >= 1, "T >= 1 (else no launch)")
    # 256-bit vectors above 128 tokens: a kernel switch (a guard)
    wide = bool(num_tokens > 128)
    vec = (32 if wide else 16) // 2
    use_vec = d % vec == 0
    block = min(d // vec, 1024) if use_vec else min(d, 1024)
    fn = kernel(_ACT.format(vec=str(use_vec).lower(), wide=str(use_vec and wide).lower()))
    params = [("P", out.data_ptr()), ("P", x.data_ptr()), ("i", d), ("f", 0.0), ("f", 1.0), ("f", 0.0)]
    c.launch(fn, params, (num_tokens, 1, 1), (block, 1, 1))


# ---- reshape_and_cache_flash(Tensor key, Tensor value, Tensor! key_cache, Tensor! value_cache, Tensor slot_mapping,
#      str kv_cache_dtype, Tensor k_scale, Tensor v_scale)


def reshape_and_cache_flash(*args: Any, **kwargs: Any) -> None:
    op = torch.ops._C_cache_ops.reshape_and_cache_flash.default
    c = _Params(op, args, kwargs)
    a = _bind(op, args, kwargs)
    k, v, kc, vc, slots = a["key"], a["value"], a["key_cache"], a["value_cache"], a["slot_mapping"]
    ks, vs = a["k_scale"], a["v_scale"]
    c.require(a["kv_cache_dtype"] == "auto", "kv_cache_dtype auto")
    c.require(k.dtype == v.dtype == kc.dtype == vc.dtype == _BF16, "bf16 key, value and caches (cross-checked)")
    c.require(slots.dtype == torch.int64 and slots.dim() == 1, "1-d int64 slot_mapping")
    c.require(k.dim() == 3 and v.dim() == 3 and kc.dim() == 4 and vc.dim() == 4, "key, value [T, H, D]; caches [N, B, H, D]")
    num_tokens = slots.shape[0]
    heads, hs, block_size = int(k.shape[1]), int(k.shape[2]), int(kc.shape[1])
    c.require(kc.stride(0) == vc.stride(0), "one block stride for both caches")
    scales = int(ks.numel())
    c.require(tuple(ks.shape) == tuple(vs.shape) and scales in (1, heads), "k_scale and v_scale of shape [1] or [H]")
    fn = kernel("void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0>"
                "(__nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*, __nv_bfloat16*, long const*, long, long, long, "
                "long, long, int, int, int, float const*, float const*, int)")
    params = [("P", k.data_ptr()), ("P", v.data_ptr()), ("P", kc.data_ptr()), ("P", vc.data_ptr()), ("P", slots.data_ptr()),
              ("q", kc.stride(0)), ("q", kc.stride(1)), ("q", kc.stride(2)), ("q", k.stride(0)), ("q", v.stride(0)),
              ("i", heads), ("i", hs), ("i", block_size), ("P", ks.data_ptr()), ("P", vs.data_ptr()), ("i", 1 if scales > 1 else 0)]
    c.launch(fn, params, (num_tokens, 1, 1), (min(heads * hs, 512), 1, 1))


LAUNCHERS = {
    ("_C", "rms_norm"): rms_norm,
    ("_C", "fused_add_rms_norm"): fused_add_rms_norm,
    ("_C", "rotary_embedding"): rotary_embedding,
    ("_C", "silu_and_mul"): silu_and_mul,
    ("_C_cache_ops", "reshape_and_cache_flash"): reshape_and_cache_flash,
}


def ops() -> list[Any]:
    return [getattr(getattr(torch.ops, ns), n).default for ns, n in LAUNCHERS]


def install(names: Any = None) -> list[Any]:
    """Register the launchers of `names` (op names; all five by default), switch the registry on, and return the
    registered OpOverloads. vllm._custom_ops must be imported (the ops registered, _C loaded)."""
    import torch.cuda._host_trace as ht

    done = []
    for (ns, n), f in LAUNCHERS.items():
        if names is None or n in names:
            o = getattr(getattr(torch.ops, ns), n).default
            ht.register_traced_impl(o, f)
            done.append(o)
    ht.traced_impls = True
    return done
