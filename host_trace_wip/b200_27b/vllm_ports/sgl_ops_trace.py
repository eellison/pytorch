# Python ports of the host launchers of four SGLang JIT ops (csrc/elementwise
# qknorm.cuh, rope.cuh, activation.cuh, kvcache.cuh), registered with
# torch.cuda._host_trace.register_traced_impl. Under a trace with traced_impls on,
# each call records its kernel launch as a KernelLaunch whose every field is an
# expression of the trace's values, checked byte for byte against the real op's
# launch captured on stand-ins at the hints. A decline leaves the call to the
# trace's other routes (the extern harvest). See PLAN.md.
from __future__ import annotations

import ctypes
import functools
import os
import re
import struct
from typing import Any

import torch

PDL = True  # kUsePDL: the modules' "_true" (is_arch_support_pdl on sm103)
_LIMITS = {"I": (0, 2**31 - 1), "i": (-(2**31), 2**31 - 1), "q": (-(2**63), 2**63 - 1)}
_WIDTH = {"P": 8, "q": 8, "I": 4, "i": 4, "f": 4}


# ---- kernel handles from the JIT module's symbol table


def _loaded(module: str) -> str:
    with open("/proc/self/maps") as f:
        for line in f:
            path = line.rstrip("\n").split(maxsplit=5)[-1]
            if path.endswith(f"/{module}.so"):
                return path
    raise LookupError(f"{module}.so is not loaded")


@functools.cache
def _symtab(path: str) -> dict[str, int]:
    """name -> st_value of every named .symtab symbol of an ELF64 file."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:5] != b"\x7fELF\x02":
        raise ValueError(f"{path} is not ELF64")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    out: dict[str, int] = {}
    for _, kind, _, _, off, size, link, _, _, entsize in sections:
        if kind != 2:  # SHT_SYMTAB
            continue
        stroff = sections[link][4]
        for at in range(off, off + size, entsize):
            name, _, _, _, value, _ = struct.unpack_from("<IBBHQQ", data, at)
            if name:
                end = data.index(b"\0", stroff + name)
                out[data[stroff + name : end].decode()] = value
    return out


@functools.cache
def _demangler() -> Any:
    lib = ctypes.CDLL("libstdc++.so.6")
    fn = lib.__cxa_demangle
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    return fn


def _demangle(name: str) -> str | None:
    status = ctypes.c_int()
    p = _demangler()(name.encode(), None, None, ctypes.byref(status))
    if status.value != 0 or not p:
        return None
    try:
        return ctypes.string_at(p).decode()
    finally:
        ctypes.CDLL(None).free(ctypes.c_void_p(p))


def _cudart_check(err: int, what: str) -> None:
    if err != 0:
        raise RuntimeError(f"{what} returned cudaError {err}")


@functools.cache
def kernel(module: str, export: str, demangled: str) -> tuple[int, int, str]:
    """(cudaFunction_t, host stub address, device function name) of the __global__
    function `demangled` in the loaded JIT module, without its launcher: the
    stub's st_value plus the load base, found from the exported `export`
    (dlsym) less its st_value, then cudaGetFuncBySymbol (absent from
    cuda.bindings). A process can map several libcudart copies and the module's
    registration binds through the global scope, not its own dlsym scope, so
    ask each mapped copy; only the one holding the registration answers."""
    path = _loaded(module)
    symbols = _symtab(path)
    mangled = [n for n in symbols if n.startswith("_ZN6sglang") and _demangle(n) == demangled]
    if len(mangled) != 1:
        raise LookupError(f"{demangled} in {path}: {mangled}")
    lib = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
    stub = ctypes.cast(getattr(lib, export), ctypes.c_void_p).value - symbols[export] + symbols[mangled[0]]
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
    # the host stub of a static __global__ carries internal linkage (L); the device function's name does not
    return func.value, stub, re.sub(r"^_ZN6sglangL(?=\d)", "_ZN6sglang", mangled[0])


@functools.cache
def blocks_per_sm(func: int, block: int) -> int:
    from cuda.bindings import driver

    from torch.cuda._utils import _check_cuda_bindings

    return int(_check_cuda_bindings(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(func, block, 0)))


@functools.cache
def sm_count(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.cache
def _plain(device: int) -> dict[Any, Any]:
    from torch.cuda._host_trace_capture import plain_attributes

    return plain_attributes(device)


# ---- the launch record and its witness


def _bind(op: Any, args: tuple, kwargs: dict) -> dict[str, Any]:
    out = {}
    for i, a in enumerate(op._schema.arguments):
        out[a.name] = args[i] if i < len(args) else kwargs.get(a.name, a.default_value)
    return out


class _Call:
    """One call's port: requires (guards where symbolic, a decline where
    false), the struct, and the record, checked against the real op."""

    def __init__(self, op: Any, args: tuple, kwargs: dict) -> None:
        from torch.cuda._host_trace_tape import _TracedTensor, current_trace

        self.op, self.args, self.kwargs = op, args, kwargs
        self.tr = tr = current_trace()
        self.what = op._schema.name
        if tr is None:
            raise RuntimeError(f"{self.what}: a traced impl outside a trace")
        if torch.cuda.current_device() != tr.device.index or torch.cuda.current_stream() != tr.stream:
            raise tr.decline(f"{self.what}: called off the trace's device or stream")
        leaves = [*args, *kwargs.values()]
        if any(isinstance(a, torch.Tensor) and not isinstance(a, _TracedTensor) for a in leaves):
            raise tr.decline(f"{self.what}: a tensor the trace does not track")
        self.roots: list[Any] = []
        for a in leaves:
            if isinstance(a, _TracedTensor) and not any(r is a._root for r in self.roots):
                self.roots.append(a._root)

    def require(self, cond: Any, what: str) -> None:
        if not bool(cond):
            raise self.tr.decline(f"{self.what}: {what} does not hold")

    def launch(self, kernel: tuple[int, int, str], size: int, members: list[tuple[int, str, Any]], grid: tuple, block: tuple) -> None:
        """Record the launch of `kernel` with the one struct parameter of
        `size` bytes, members (offset, format, value) with format P (a
        pointer), q, i, I or f, after checking it against the real op's."""
        from torch.cuda._host_trace_capture import capture_kernel_nodes, KernelNode, pack_params
        from torch.cuda._host_trace_cute import _stand_in
        from torch.cuda._host_trace_launch import _GRID_LIMITS, KernelLaunch
        from torch.cuda._host_trace_tape import _hint, _PLACEHOLDER_LOW, _TracedTensor
        from torch.utils._python_dispatch import _disable_current_modes

        func, _, name = kernel
        image = bytearray(size)
        mask = bytearray(size)
        slots, fields, pointers = [], [], []
        for at, fmt, v in members:
            width = _WIDTH[fmt]
            mask[at : at + width] = b"\xff" * width
            if fmt == "P":
                if not isinstance(v, torch.SymInt):
                    raise self.tr.decline(f"{self.what}: a pointer that is not a traced tensor's")
                slots.append(v)
                fields.append((0, at, 8))
                pointers.append(True)
            elif isinstance(v, torch.SymInt):
                lo, hi = _LIMITS[fmt]
                self.require(v >= lo, f"a {fmt} member >= {lo}")
                self.require(v <= hi, f"a {fmt} member <= {hi}")
                slots.append(v)
                fields.append((0, at, width))
                pointers.append(False)
            else:
                struct.pack_into("<" + fmt, image, at, v)
        for axis, (n, limit) in enumerate(zip(grid, _GRID_LIMITS)):
            if not isinstance(n, torch.SymInt):
                self.require(1 <= n <= limit, f"grid axis {axis} in [1, {limit}]")
        record = KernelLaunch(name=name, function=func, abi=None, layout=((0, size),), grid=grid, block=block, smem=0,
                              slots=tuple(slots), roots=tuple(self.roots), fields=tuple(fields), attributes=(),
                              pointers=frozenset(i for i, p in enumerate(pointers) if p), images=(bytes(image),),
                              programmatic=PDL)

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
        mine = pack_params(record, hints, pointers)[0]
        real = node.images[0] if len(node.images) == 1 else b""
        evaluated = (func, name, tuple(map(_hint, grid)), tuple(map(_hint, block)), 0, (), PDL, ((0, size),))
        launched = (node.function, node.name, node.grid, node.block, node.smem, explicit, bool(node.attribute("PROGRAMMATIC_STREAM_SERIALIZATION")), node.layout)
        bad = [i for i in range(size) if mask[i] and (i >= len(real) or mine[i] != real[i])]
        if evaluated != launched or bad:
            raise self.tr.decline(f"{self.what}: the op launched {launched}, the port {evaluated}; param bytes {bad[:8]} differ")
        self.tr.record_launch(record)


def _cap(need: Any, cap: int) -> Any:
    """min(need, cap) without a guard: a select row."""
    from torch.cuda._host_trace import select

    return select(need <= cap, need, cap) if isinstance(need, torch.SymInt) else min(need, cap)


def _select(c: Any, a: Any, b: Any) -> Any:
    from torch.cuda._host_trace import select

    return select(c, a, b) if isinstance(c, torch.SymBool) else (a if c else b)


_BF16 = torch.bfloat16
_DTYPE_ARG = {torch.bfloat16: "bf16_t", torch.float16: "fp16_t", torch.float32: "fp32_t"}
_DTYPE_CXX = {torch.bfloat16: "__nv_bfloat16"}


# ---- fused_inplace_qknorm(q, k, q_weight, k_weight, eps, *, head_dim)


def qknorm(*args: Any, **kwargs: Any) -> None:
    op = torch.ops.sglang.fused_inplace_qknorm.default
    c = _Call(op, args, kwargs)
    a = _bind(op, args, kwargs)
    q, k, qw, kw, eps = a["q"], a["k"], a["q_weight"], a["k_weight"], a["eps"]
    hd = a["head_dim"] or q.shape[-1]
    c.require(hd == 128 and q.dtype == _BF16, "head_dim 128, bf16 (cross-checked)")
    c.require(q.dim() == 3 and k.dim() == 3 and qw.dim() == 1 and kw.dim() == 1, "q, k [N, H, D], weights [D]")
    c.require(k.dtype == qw.dtype == kw.dtype == _BF16, "bf16 k and weights")
    n, nq, d = q.shape
    c.require(k.shape[0] == n and k.shape[2] == d and d == hd and qw.shape[0] == d and kw.shape[0] == d, "shapes [N, Q, D], [N, K, D], [D]")
    c.require(q.stride(1) == d and q.stride(2) == 1 and k.stride(1) == d and k.stride(2) == 1, "head strides (D, 1)")
    nk = k.shape[1]
    c.require(n >= 1, "N >= 1")
    works = (nq + nk) * n
    c.require(works <= 2**32 - 1, "(Q + K) * N fits uint32")
    fn = kernel("sgl_kernel_jit_qknorm_128_true_bf16_t", "__tvm_ffi_qknorm",
                "void sglang::fused_qknorm_warp<128l, true, __nv_bfloat16>(sglang::QKNormParams)")
    cap = sm_count(c.tr.device.index) * blocks_per_sm(fn[0], 128)
    members = [(0, "P", q.data_ptr()), (8, "P", k.data_ptr() - 2 * nq * 128), (16, "q", q.stride(0)), (24, "q", k.stride(0)),
               (32, "I", nq), (36, "I", nk), (40, "f", eps), (48, "P", qw.data_ptr()), (56, "P", kw.data_ptr()), (64, "I", n)]
    c.launch(fn, 72, members, (_cap((works + 3) // 4, cap), 1, 1), (128, 1, 1))


# ---- apply_rope_inplace(q, k, cos_sin_cache, positions, *, is_neox, rope_dim)


def rope(*args: Any, **kwargs: Any) -> None:
    op = torch.ops.sglang.apply_rope_inplace.default
    c = _Call(op, args, kwargs)
    a = _bind(op, args, kwargs)
    q, k, cache, pos = a["q"], a["k"], a["cos_sin_cache"], a["positions"]
    rd = a["rope_dim"] or cache.shape[-1]
    c.require(a["is_neox"] is True and rd == 128, "is_neox, rope_dim 128 (cross-checked)")
    c.require(q.dtype == k.dtype == _BF16 and cache.dtype == torch.float32, "bf16 q, k and fp32 cos_sin_cache")
    c.require(pos.dtype in (torch.int32, torch.int64), "int32 or int64 positions")
    c.require(q.dim() == 3 and k.dim() == 3 and cache.dim() == 2 and pos.dim() == 1, "q, k [N, H, D], cache [-1, D], positions [N]")
    n, nq, d = q.shape
    nk = k.shape[1]
    c.require(d == rd and k.shape[0] == n and k.shape[2] == d and cache.shape[1] == d and pos.shape[0] == n, "shapes")
    dd = q.stride(1)
    c.require(k.stride(1) == dd and q.stride(2) == 1 and k.stride(2) == 1, "one head stride, inner stride 1")
    c.require(n >= 1, "N >= 1")
    works = (nq + nk) * n
    c.require(works <= 2**32 - 1, "(Q + K) * N fits uint32")
    idx = "int" if pos.dtype == torch.int32 else "long"
    fn = kernel("sgl_kernel_jit_fused_rope_true_128_true_bf16_t_bf16_t", "__tvm_ffi_run_rope",
                f"void sglang::fused_rope_kernel<true, 128l, true, __nv_bfloat16, __nv_bfloat16, {idx}, 16u>(sglang::FusedRopeParams)")
    cap = sm_count(c.tr.device.index) * blocks_per_sm(fn[0], 128)
    members = [(0, "P", q.data_ptr()), (8, "P", k.data_ptr() - nq * dd * 2), (16, "P", cache.data_ptr()), (24, "P", pos.data_ptr()),
               (32, "q", q.stride(0) * 2), (40, "q", k.stride(0) * 2), (48, "q", dd * 2), (56, "I", nq), (60, "I", nk), (64, "I", n)]
    c.launch(fn, 72, members, (_cap((works + 7) // 8, cap), 1, 1), (128, 1, 1))


# ---- _run_activation_inplace(op_name, input, out)

_ACT = {"silu": 0, "gelu": 1, "gelu_tanh": 2}


def activation(*args: Any, **kwargs: Any) -> None:
    op = torch.ops.sglang._run_activation_inplace.default
    c = _Call(op, args, kwargs)
    a = _bind(op, args, kwargs)
    name, x, out = a["op_name"], a["input"], a["out"]
    c.require(name in _ACT, f"activation {name!r} one of {sorted(_ACT)}")
    c.require(x.dtype == out.dtype == _BF16, "bf16 input and out (cross-checked)")
    c.require(x.dim() == 2 and out.dim() == 2, "2-d input and out")
    n, h2 = x.shape
    h = out.shape[1]
    # kMaxVecBytes: 32 on Blackwell and later, else 16
    vec = 16 if torch.cuda.get_device_capability(x.device)[0] >= 10 else 8
    c.require(h2 == 2 * h and out.shape[0] == n and h % vec == 0, f"input [N, 2H], out [N, H], H % {vec} == 0")
    c.require(x.stride(1) == 1 and x.stride(0) == h2 and out.stride(1) == 1 and out.stride(0) == h, "contiguous input and out")
    c.require(n >= 1, "N >= 1")
    items = n * (h // vec)
    c.require(items <= 2**32 - 1, "the items fit uint32")
    fn = kernel("sgl_kernel_jit_activation_bf16_t_true", "__tvm_ffi_run_activation",
                f"void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::ActivationKind){_ACT[name]}, true, false, false, false>(sglang::ActivationParams)")
    members = [(0, "P", x.data_ptr()), (8, "P", out.data_ptr()), (16, "I", h), (20, "I", n), (24, "q", 0), (32, "I", 1)]
    c.launch(fn, 40, members, ((items + 255) // 256, 1, 1), (256, 1, 1))


# ---- store_cache(k, v, k_cache, v_cache, indices, *, row_bytes, v_row_bytes, num_split, size_limit, reserved_skip_index)


def _threads(k_row_bytes: int, v_row_bytes: int) -> int:
    # kvcache.py _jit_kvcache_module's heuristic for num_threads 0
    n = 32
    if k_row_bytes % n or v_row_bytes % n:
        return n
    kb, vb = k_row_bytes / n, v_row_bytes / n
    while kb % 8 == 0 and vb % 8 == 0 and kb + vb >= 64:
        n, kb, vb = n * 2, kb / 2, vb / 2
    return n


def store_cache(*args: Any, **kwargs: Any) -> None:
    op = torch.ops.sglang.store_cache.default
    c = _Call(op, args, kwargs)
    a = _bind(op, args, kwargs)
    k, v, kc, vc, idx = a["k"], a["v"], a["k_cache"], a["v_cache"], a["indices"]
    c.require(k.dtype == v.dtype == kc.dtype == vc.dtype == _BF16, "bf16 k, v and caches (cross-checked)")
    c.require(idx.dtype in (torch.int32, torch.int64), "int32 or int64 indices")
    c.require(k.dim() == v.dim() == kc.dim() == vc.dim() == 2 and idx.dim() == 1, "2-d k, v, caches; 1-d indices")
    rk = a["row_bytes"] or k.shape[-1] * 2
    rv = a["v_row_bytes"] or v.shape[-1] * 2
    c.require(a["num_split"] == 0 and rk == rv == 2048 and _threads(rk, rv) == 128, "rows of 2048 bytes, 4 warps each (cross-checked)")
    b = k.shape[0]
    c.require(v.shape[0] == b and idx.shape[0] == b and k.shape[1] * 2 == rk and v.shape[1] * 2 == rv, "k [B, DK], v [B, DV], indices [B]")
    c.require(kc.shape[1] * 2 == rk and vc.shape[1] * 2 == rv, "cache rows of DK and DV")
    c.require(k.stride(1) == 1 and v.stride(1) == 1 and kc.stride(1) == 1 and vc.stride(1) == 1, "inner stride 1")
    for t in (k, v, kc, vc):
        # WARP_UNIFORM_16B on 512-byte splits: 16-byte aligned rows
        c.require(t.stride(0) * 2 % 16 == 0, "16-byte row strides")
        c.require(t.data_ptr() % 16 == 0, "16-byte aligned rows")
    c.require(b >= 1, "B >= 1")
    c.require(b <= 2**30, "B * 4 fits uint32")
    limit = a["size_limit"] if a["size_limit"] > 0 else kc.shape[0]
    name = "int" if idx.dtype == torch.int32 else "long"
    fn = kernel("sgl_kernel_jit_kvcache_2048_2048_128_true", "__tvm_ffi_store_cache",
                f"void sglang::store_kvcache_kernel<2048l, 2048l, 128u, true, {name}>(sglang::StoreKVCacheParams)")
    sm = sm_count(c.tr.device.index)
    warps = b * 4
    one, two, four = warps <= sm, warps <= 2 * sm, warps <= 4 * sm
    per_block = _select(one, 1, _select(two, 2, _select(four, 4, 8)))
    grid = _select(one, warps, _select(two, (warps + 1) // 2, _select(four, (warps + 3) // 4, (warps + 7) // 8)))
    members = [(0, "P", k.data_ptr()), (8, "P", v.data_ptr()), (16, "P", kc.data_ptr()), (24, "P", vc.data_ptr()), (32, "P", idx.data_ptr()),
               (40, "q", k.stride(0) * 2), (48, "q", v.stride(0) * 2), (56, "q", kc.stride(0) * 2), (64, "q", vc.stride(0) * 2),
               (72, "q", idx.stride(0)), (80, "I", b), (88, "q", limit), (96, "q", a["reserved_skip_index"])]
    c.launch(fn, 104, members, (grid, 1, 1), (32, per_block, 1))


LAUNCHERS = {
    "fused_inplace_qknorm": qknorm,
    "apply_rope_inplace": rope,
    "_run_activation_inplace": activation,
    "store_cache": store_cache,
}


def install(names: Any = None) -> None:
    """Register the launchers of `names` (all four by default) and switch the
    registry on. The ops must be registered (sglang imported)."""
    import torch.cuda._host_trace as ht

    for n in names or LAUNCHERS:
        ht.register_traced_impl(getattr(torch.ops.sglang, n).default, LAUNCHERS[n])
    ht.traced_impls = True
