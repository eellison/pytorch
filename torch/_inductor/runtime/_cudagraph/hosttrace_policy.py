"""Inductor's compiled artifact under the host trace: the tape as the CUDA-graph mechanism.

A CUDAGraphPolicy at the seam where cudagraph trees enters (cudagraph_post_compile ->
policy.cudagraphify), selected by ``torch._inductor.config.triton.cudagraph_host_trace``
where no ``config.cudagraph_policy`` instance is set. Per artifact (one per dynamo guard
set) the generated wrapper is traced once by the recorder (torch.cuda._host_trace) into a
tape of Inductor's own kernels, lowered and served by the native replay
(direct_hosttrace.HostTraceReplay); a call the tape's guards reject is a re-trace of the
same artifact (a second variant of the same entry); dynamo's recompile stays dynamo's.

Inputs. The artifact's graph inputs are the tape's inputs. The tensors dynamo marks static
(parameters, buffers, mark_static_address) and the artifact's constants are bound once as
the dispatch's hidden trailing inputs (E32: the caller upholds that their addresses hold;
a miss re-reads them); the other tensors are symbolic addresses read from the call's box
per call; a SymInt input is rebound from the size (or stride) of the tensor input that
carries its symbol, so a dynamic length enters the tape as the tape's own symbol.

Seams. The wrapper reaches the GPU through helpers the recorder's dispatch mode never sees;
the wrapper (and every function of its generated module) is cloned with those globals
replaced, the module's own globals untouched: empty_strided_cuda -> a traced allocation,
reinterpret_tensor -> a traced view, assert_size_stride / assert_alignment /
copy_if_misaligned / assert_tensor_metadata -> symbolic checks (each comparison a guard of
the tape), each CachingAutotuner -> a recording proxy that runs the selected static
launcher's own grid arithmetic on the traced values and records the ABI-ordered launch
through the loaded StaticallyLaunchedCudaKernel (E36: the tape's node holds the CUfunction
Inductor loaded), MultiKernelCall -> a decline by name. The extern GEMMs (their out= forms)
and the fallback ops go through the dispatcher and reach the recorder as they are.
Inductor's ShapeEnv guards over the placeholders (produce_guards_expression) are
re-evaluated under the trace, each comparison a guard, so the tape stands on its own
behind dynamo's front door.
"""

from __future__ import annotations

import copy
import threading
import time
import types
import weakref
from dataclasses import dataclass

import torch
from torch._dynamo.utils import counters, get_static_address_type
from torch._inductor.cudagraph_utils import (
    CUDAGraphPolicy,
    cudagraphs_log,
    log_cudagraph_skip_and_bump_counter,
)
from torch._logging import trace_structured
from torch.cuda._utils import _check_cuda_bindings
from torch.utils._ordered_set import OrderedSet

from .direct_hosttrace import _MISSED


_SEAM_NAMES = (
    "empty_strided_cuda",
    "reinterpret_tensor",
    "assert_size_stride",
    "assert_size_stride_grouped",
    "assert_alignment",
    "assert_tensor_metadata",
    "copy_if_misaligned",
)


def _ht():
    from torch.cuda import _host_trace

    return _host_trace


def _active_trace():
    return getattr(_ht()._active, "trace", None)


def _declined(what):
    return _ht().Declined(f"host_trace(inductor): {what} (declined)")


def _guard(cond, what):
    # under a trace the comparison records a guard of the tape and returns the hint's
    # truth; a false one is a class the artifact never serves natively
    if not bool(cond):
        raise _declined(what)


def _probe_address(t):
    # an allocation's base is a multiple of 256 by construction; only its offset decides
    offset_bytes = t._sym_offset * t.element_size()
    return offset_bytes if t._root.allocation else t._root.sym + offset_bytes


def _clone(function, namespace):
    """`function` with `namespace` as its globals (the same dict object, so the clones of
    one module resolve each other through it)."""
    result = types.FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    result.__kwdefaults__ = (
        None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
    )
    result.__dict__.update(function.__dict__)
    return result


class _Seams:
    """The wrapper's C helper globals under a trace; outside one, each is the original."""

    def __init__(self, guards):
        self.o_empty = guards._empty_strided_cuda
        self.o_reinterpret = guards._reinterpret_tensor
        self.o_assert = guards.assert_size_stride
        self.o_assert_grouped = guards.assert_size_stride_grouped
        self.o_align = guards.assert_alignment
        self.o_copy_if_misaligned = getattr(guards, "copy_if_misaligned", None)

    def empty_strided_cuda(self, size, stride, dtype):
        tr = _active_trace()
        if tr is None:
            return self.o_empty(size, stride, dtype)
        return torch.empty_strided(
            tuple(size), tuple(stride), dtype=dtype, device=tr.device
        )

    def reinterpret_tensor(self, t, size, stride, offset_increment=0):
        if _active_trace() is None or type(t) is not _ht()._TracedTensor:
            return self.o_reinterpret(t, size, stride, offset_increment)
        return torch.as_strided(
            t, tuple(size), tuple(stride), t._sym_offset + offset_increment
        )

    def assert_size_stride(self, t, size, stride, op_name=None):
        if _active_trace() is None or type(t) is not _ht()._TracedTensor:
            if op_name is None:
                return self.o_assert(t, size, stride)
            return self.o_assert(t, size, stride, op_name)
        if t.dim() != len(size):
            raise _declined(
                f"assert_size_stride: rank {t.dim()} vs {len(size)} ({op_name})"
            )
        for d in range(t.dim()):
            _guard(
                t.shape[d] == size[d],
                f"assert_size_stride: size({d}) {t.shape[d]} != {size[d]} ({op_name})",
            )
            # eager's check ignores the stride of a size-1 dimension
            if bool(t.shape[d] > 1):
                _guard(
                    t._sym_strides[d] == stride[d],
                    f"assert_size_stride: stride({d}) {t._sym_strides[d]} != {stride[d]} ({op_name})",
                )
        return True

    def assert_size_stride_grouped(self, items, sizes, strides, op_name=None):
        for t, s, st in zip(items, sizes, strides):
            self.assert_size_stride(t, s, st, op_name)
        return True

    def assert_alignment(self, t, alignment, op_name=None):
        if _active_trace() is None or type(t) is not _ht()._TracedTensor:
            if op_name is None:
                return self.o_align(t, alignment)
            return self.o_align(t, alignment, op_name)
        _guard(
            _probe_address(t) % alignment == 0,
            f"assert_alignment: {op_name} is not {alignment}-byte aligned at the trace",
        )
        return True

    def assert_tensor_metadata(self, t, size, stride, dtype, op_name=None):
        # runtime_utils.assert_tensor_metadata (a fallback op's output under dynamic
        # shapes) reaches the C++ assert_size_stride, which has no HANDLE_TH_ERRORS and
        # aborts the process on a tensor with symbolic sizes: routed here
        if _active_trace() is None or type(t) is not _ht()._TracedTensor:
            from torch._inductor.runtime.runtime_utils import assert_tensor_metadata

            return assert_tensor_metadata(t, size, stride, dtype, op_name)
        if t.dtype != dtype:
            raise _declined(
                f"assert_tensor_metadata: {op_name} produced {t.dtype}, expected {dtype}"
            )
        return self.assert_size_stride(t, size, stride, op_name)

    def copy_if_misaligned(self, t):
        # eager clones an input whose address is not 16-byte aligned; the tape guards the
        # alignment class instead, and a misaligned class is declined by name
        if _active_trace() is None or type(t) is not _ht()._TracedTensor:
            return self.o_copy_if_misaligned(t)
        from torch._inductor.utils import ALIGNMENT

        _guard(
            _probe_address(t) % ALIGNMENT == 0,
            f"copy_if_misaligned: an input is not {ALIGNMENT}-byte aligned at the trace (Inductor would copy it)",
        )
        return t


_SEAMS = None


def _seams():
    global _SEAMS
    if _SEAMS is None:
        _SEAMS = _Seams(torch._C._dynamo.guards)
    return _SEAMS


class StaticKernelOwner:
    """What _HostTraceTritonModule reads of a launch's owner, over Inductor's loaded
    StaticallyLaunchedCudaKernel: `module` (num_warps, shared, _borrow_for_cudagraph),
    `device_index`, `abi_layout`, `check()`."""

    def __init__(self, kernel, device, layout):
        self.module = kernel
        self.device_index = device
        self.abi_layout = tuple(layout)
        self._loaded = (kernel.module, kernel.function)

    def check(self):
        from torch._inductor.runtime._cudagraph.direct_triton import (
            DirectTritonDeclined,
        )

        if (self.module.module, self.module.function) != self._loaded:
            raise DirectTritonDeclined(
                "the Inductor kernel was unloaded or reloaded since the trace"
            )


_LAYOUTS: dict = {}
_LAYOUTS_LOCK = threading.Lock()


def _layout(kernel, n_slots):
    key = (id(kernel), kernel.function, n_slots)
    with _LAYOUTS_LOCK:
        got = _LAYOUTS.get(key)
    if got is None:
        from cuda.bindings import driver

        got = []
        for i in range(n_slots):
            got.append(
                tuple(
                    _check_cuda_bindings(driver.cuFuncGetParamInfo(kernel.function, i))
                )
            )
        result = driver.cuFuncGetParamInfo(kernel.function, n_slots)
        if result[0] != driver.CUresult.CUDA_ERROR_INVALID_VALUE:
            raise _declined(
                f"kernel {kernel.name} has more parameters than its ABI ({n_slots})"
            )
        with _LAYOUTS_LOCK:
            _LAYOUTS[key] = got
    return got


class Recording:
    """Stands in for a CachingAutotuner global of the cloned wrapper: outside a trace the
    autotuner's own run; under one, the launch recorded on the tape."""

    def __init__(self, autotuner, name):
        self.autotuner, self.name = autotuner, name

    def __getattr__(self, item):
        return getattr(self.autotuner, item)

    def run(self, *args, stream=None, **kwargs):
        tr = _active_trace()
        if tr is None:
            return self.autotuner.run(*args, stream=stream, **kwargs)
        try:
            return _record_launch(self.autotuner, self.name, tr, args, stream, kwargs)
        except BaseException as error:
            if tr.triton.failure is None:
                tr.triton.failure = error
            raise


def _record_launch(prov, name, tr, args, stream, kwargs):
    from torch._inductor.runtime.static_triton_launcher import (
        StaticallyLaunchedCudaKernel,
    )

    def decline(why):
        raise _declined(f"Triton kernel {name}: {why}")

    if kwargs:
        decline(f"launched with keyword arguments {sorted(kwargs)}")
    if len(prov.launchers) != 1:
        decline(
            f"the autotuner holds {len(prov.launchers)} launchers (not warmed to one)"
        )
    launcher = prov.launchers[0]
    if prov.inductor_meta.get("host_tma_descriptor_args"):
        decline("host-side TMA descriptors are built in the launcher")
    if stream != tr.stream.cuda_stream:
        decline("launched on a stream other than the trace's")
    if getattr(launcher, "_is_static", False) is not True:
        return _record_python_launch(prov, name, tr, args, stream, launcher, decline)
    runner = launcher.__globals__.get("runner")
    kernel = getattr(runner, "__self__", None)
    if type(kernel) is not StaticallyLaunchedCudaKernel or getattr(
        kernel, "device_agnostic", False
    ):
        decline("the launcher's runner is not a loaded StaticallyLaunchedCudaKernel")
    if kernel.function is None or kernel.module is None:
        decline("the kernel is not loaded")
    device = tr.device.index
    formals = kernel.cudagraph_formal_args
    if formals is None:
        decline("the kernel has no compiler ABI correspondence (cudagraph_formal_args)")
    dims, call_args = _run_launcher(launcher, args, stream, decline)
    abi_rows = sorted(
        (r for r in formals if r.abi_index is not None), key=lambda r: r.abi_index
    )
    arg_tys = kernel.arg_tys
    if len(abi_rows) != len(call_args) or len(arg_tys) != len(call_args):
        decline(
            f"ABI mismatch: {len(abi_rows)} rows, {len(call_args)} call args, arg_tys {arg_tys!r}"
        )
    n_scratch = int(bool(kernel.has_global_scratch)) + int(
        bool(kernel.has_profile_scratch)
    )
    if kernel.has_global_scratch and kernel.global_scratch_size:
        decline(
            f"a global scratch allocation of {kernel.global_scratch_size} bytes per launch"
        )
    layout = _layout(kernel, len(arg_tys) + n_scratch)
    rows = [
        (row, val, ty, layout[i])
        for i, (row, val, ty) in enumerate(zip(abi_rows, call_args, arg_tys))
    ]
    kinds = []
    params, written = _abi_params(prov, rows, kinds, decline)
    _scratch_params(params, layout, range(len(arg_tys), len(arg_tys) + n_scratch))
    owner = StaticKernelOwner(kernel, device, layout)
    _append_launch(
        tr,
        decline,
        dims,
        layout,
        params,
        written,
        htt_launch(owner, None, kernel, kinds),
        kernel.name,
        int(kernel.function),
        int(kernel.num_warps),
        int(kernel.shared),
    )


def htt_launch(owner, jit, binary, kinds):
    from torch.cuda import _host_trace_triton as htt

    return htt.TritonLaunch(owner, jit, binary, tuple(kinds))


def _run_launcher(launcher, args, stream, decline):
    """The launcher's own grid arithmetic on the traced values, its runner replaced:
    the grid and the arguments it would hand the runner (the static runner takes
    `(grid, stream, *slots)`, Triton's `(grid, stream, function, metadata, launch
    metadata, hooks, *formals)`: the trailing arguments are returned as given)."""
    got = []

    def record(gx, gy, gz, stream_, *rest):
        got.append((gx, gy, gz, rest))

    _clone(launcher, {**launcher.__globals__, "runner": record})(*args, stream=stream)
    if len(got) != 1:
        decline(f"the launcher called its runner {len(got)} times")
    gx, gy, gz, rest = got[0]
    return (gx, gy, gz), rest


def _abi_params(prov, rows, kinds, decline):
    """The ABI slots of a launch over the traced values, `(row, value, ABI type,
    (offset, size))` per slot in ABI order, as the tape's params: the compile-time facts
    (`tt.divisibility`, the int32 slots) as guards, a pointer read-only when Inductor
    named it an input and its kernel does not mutate it, written otherwise."""
    from torch._inductor.runtime._cudagraph.direct_triton import _POINTER_DTYPES

    ht = _ht()
    mutated = OrderedSet(getattr(prov, "mutated_arg_names", ()) or ())
    params, written = [], []
    for row, val, ty, (offset, size) in rows:
        # the compiled ABI's specialization attributes: the divisibility Inductor
        # proved at compile time is the only kind this proxy turns into a guard;
        # another kind is an axis it does not read (declined by name, as the JIT
        # hook's owner declines an unsupported specialization)
        other = sorted(
            OrderedSet([n for n, _ in (row.attributes or ()) if n != "tt.divisibility"])
        )
        if other:
            decline(
                f"argument {row.formal} carries specialization attributes {other} this proxy does not read"
            )
        divisors = [v for n, v in (row.attributes or ()) if n == "tt.divisibility"]
        div = max([1, *divisors])
        if ty == "O":
            if type(val) is not ht._TracedTensor:
                decline(
                    f"pointer argument {row.formal} is a {type(val).__name__}, not a tensor of the trace"
                )
            dtype = _POINTER_DTYPES.get(row.triton_type)
            if dtype is not None and val.dtype != dtype:
                decline(
                    f"pointer argument {row.formal} is {val.dtype}; the kernel was compiled for {row.triton_type}"
                )
            if size != 8:
                decline(f"pointer argument {row.formal} has a {size}-byte slot")
            address = val._root.sym + val._sym_offset * val.element_size()
            if div >= 16 and not bool(_probe_address(val) % 16 == 0):
                decline(
                    f"pointer argument {row.formal} is not 16-byte aligned at the trace; the kernel was compiled aligned"
                )
            access = (
                "r"
                if row.formal.startswith("in_ptr") and row.formal not in mutated
                else "rw"
            )
            params.append(
                {
                    "offset": offset,
                    "size": size,
                    "kind": "ptr",
                    "value": address,
                    "name": row.formal,
                    "access": access,
                }
            )
            kinds.append((row.formal, "ptr", access))
            if access == "rw":
                written.append(val._root.name)
            continue
        kind = {"i": "i32", "l": "i64"}.get(ty)
        if kind is None:
            decline(
                f"argument {row.formal} of ABI type {ty!r} has no slot kind on the tape"
            )
        if size != (4 if kind == "i32" else 8):
            decline(f"integer argument {row.formal} has a {size}-byte slot for {kind}")
        if type(val) is not int and type(val) is not torch.SymInt:
            decline(f"integer argument {row.formal} is a {type(val).__name__}")
        if div > 1 and not bool(val % div == 0):
            decline(
                f"integer argument {row.formal} is not divisible by {div} at the trace; the kernel was compiled divisible"
            )
        if kind == "i32" and not (bool(val >= -(2**31)) and bool(val <= 2**31 - 1)):
            decline(f"integer argument {row.formal} is outside the int32 range")
        params.append(
            {
                "offset": offset,
                "size": size,
                "kind": kind,
                "value": val,
                "name": row.formal,
                "access": "",
            }
        )
        kinds.append((row.formal, kind, ""))
    return params, written


def _scratch_params(params, layout, indices):
    # an advertised zero-size scratch slot: the null pointer, a constant of the launch
    for k, i in enumerate(indices):
        offset, size = layout[i]
        params.append(
            {
                "offset": offset,
                "size": size,
                "kind": "ptr",
                "value": 0,
                "name": f"__scratch_{k}",
                "access": "",
            }
        )


def _append_launch(
    tr,
    decline,
    dims,
    layout,
    params,
    written,
    launch,
    kernel_name,
    function,
    num_warps,
    smem,
):
    ht = _ht()
    for axis, (extent, limit) in enumerate(zip(dims, (2**31 - 1, 65535, 65535))):
        if type(extent) is not int and type(extent) is not torch.SymInt:
            decline(f"grid axis {axis} is a {type(extent).__name__}")
        if not (bool(extent >= 1) and bool(extent <= limit)):
            decline(f"grid axis {axis} is outside the launch bounds at the traced call")
    total = max((o + s for o, s in layout), default=0)
    image = bytearray(total)
    for p in params:
        image[p["offset"] : p["offset"] + p["size"]] = ht._pack(
            p["kind"], ht._hint(p["value"])
        )
    block = (num_warps * 32, 1, 1)
    tt = tr.triton
    tt.launches.append(
        {
            "seq": tr.rec.next_seq(),
            "kernel": kernel_name,
            "func": function,
            "param_layout": list(layout),
            "grid": dims,
            "block": block,
            "block_expr": block,
            "smem": smem,
            "params": params,
            "hint_image": bytes(image),
            "triton": launch,
        }
    )
    for root in written:
        if root not in tt.written_roots:
            tt.written_roots.append(root)
    return None


class CompiledKernelOwner:
    """The owner of a compilation Inductor launches through Triton's own launcher (no
    static kernel behind the launcher: a kernel name longer than the 150 characters
    Triton keeps for its cache file names, so Inductor found no cubin path; a launch
    attribute; a user-defined kernel; the static launcher off), in the shape
    _HostTraceTritonModule reads. The ABI is read as the runtime's DirectTritonOwner
    reads it, from a StaticallyLaunchedCudaKernel over the compilation's cubin, without
    that owner's JITFunction contract: Inductor's launch never reselects, and a kernel
    compiled in a worker arrives with its JITFunction stripped. The tape's node holds the
    function Triton loaded for eager's launch (`binary.function`); the module here is
    borrowed for the graph's lifetime as the runtime's is."""

    def __init__(self, binary, device):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace

        from cuda.bindings import driver

        from torch._inductor.runtime._cudagraph.direct_triton import (
            DirectTritonDeclined,
        )
        from torch._inductor.runtime._cudagraph.triton_scratch import (
            scratch_specs,
            TritonScratchDeclined,
        )
        from torch._inductor.runtime.static_triton_launcher import (
            StaticallyLaunchedCudaKernel,
        )

        if type(binary.kernel) is not bytes or binary.kernel != binary.asm.get("cubin"):
            raise DirectTritonDeclined(
                "the compilation has no cubin to read the ABI from"
            )
        if getattr(binary.metadata, "num_ctas", 1) != 1:
            raise DirectTritonDeclined(
                "the compilation launches more than one CTA per program"
            )
        self.binary, self.device_index = binary, device
        self.module = None
        self.closed = False
        with TemporaryDirectory(prefix="hosttrace_inductor_") as directory:
            cubin = Path(directory) / "selected.cubin"
            cubin.write_bytes(binary.kernel)
            selected = SimpleNamespace(
                src=binary.src,
                metadata=binary.metadata,
                hash=binary.hash,
                asm={"cubin": binary.kernel},
                _cubin_path=str(cubin),
            )
            try:
                module = StaticallyLaunchedCudaKernel(selected)
            except NotImplementedError as error:
                raise DirectTritonDeclined(str(error)) from error
            try:
                if module.cudagraph_formal_args is None:
                    raise DirectTritonDeclined(
                        "the compilation has no compiler ABI correspondence"
                    )
                if "M" in module.arg_tys or module.tensordesc_arg_names:
                    raise DirectTritonDeclined(
                        "TMA descriptor arguments are not traced"
                    )
                try:
                    self.scratch = scratch_specs(module)
                except TritonScratchDeclined as error:
                    raise DirectTritonDeclined(str(error)) from error
                if any(spec.size for spec in self.scratch):
                    raise DirectTritonDeclined(
                        "a nonzero launcher scratch allocation is not traced"
                    )
                self.formals = module.cudagraph_formal_args
                self.scratch_abi_indices = tuple(
                    range(len(module.arg_tys), len(module.arg_tys) + len(self.scratch))
                )
                module.load_kernel(device)
                self.module = module
                layout = []
                for index in range(len(module.arg_tys) + len(self.scratch)):
                    layout.append(
                        tuple(
                            _check_cuda_bindings(
                                driver.cuFuncGetParamInfo(module.function, index)
                            )
                        )
                    )
                result = driver.cuFuncGetParamInfo(module.function, len(layout))
                if result[0] != driver.CUresult.CUDA_ERROR_INVALID_VALUE:
                    _check_cuda_bindings(result)
                    raise DirectTritonDeclined(
                        "the loaded kernel has unexpected trailing parameters"
                    )
                self.abi_layout = tuple(layout)
                self._loaded = (module.module, module.function)
                self._eager = (binary.function, binary.module, binary.kernel)
            except BaseException:
                module.close()
                self.closed = True
                raise

    def check(self):
        from torch._inductor.runtime._cudagraph.direct_triton import (
            DirectTritonDeclined,
        )

        module, binary = self.module, self.binary
        if (
            self.closed
            or module is None
            or (module.module, module.function) != self._loaded
            or (binary.function, binary.module, binary.kernel) != self._eager
        ):
            raise DirectTritonDeclined(
                "the Inductor compilation or its loaded kernel changed since the trace"
            )


_OWNERS: dict = {}
_OWNERS_LOCK = threading.Lock()


def _compiled_owner(binary, device, decline):
    from torch._inductor.runtime._cudagraph.direct_triton import DirectTritonDeclined

    key = (id(binary), device)
    with _OWNERS_LOCK:
        owner = _OWNERS.get(key)
        try:
            if owner is None:
                owner = CompiledKernelOwner(binary, device)
                _OWNERS[key] = owner
            else:
                owner.check()
        except DirectTritonDeclined as error:
            _OWNERS.pop(key, None)
            decline(str(error))
    return owner


def _record_python_launch(prov, name, tr, args, stream, launcher, decline):
    """A launcher over Triton's own compilation: `runner(grid, stream, function,
    metadata, launch metadata, hooks, *formals)` with `bin.run`, Triton's C launcher,
    which reads data_ptr(). The compilation is the one Inductor selected at compile time
    (its launch never respecializes), so the record is the static route's with the ABI
    read through CompiledKernelOwner, the guards the compile-time facts, and the tape's
    node eager's own loaded function (E36)."""
    from cuda.bindings import driver

    binary = launcher.__globals__.get("bin")
    jit = getattr(getattr(binary, "src", None), "fn", None)
    if jit is None or jit.arg_names is None:
        decline("the launcher holds no compilation with a kernel signature")
    device = tr.device.index
    names = tuple(jit.arg_names)
    dims, rest = _run_launcher(launcher, args, stream, decline)
    if len(rest) < len(names):
        decline(f"ABI mismatch: {len(rest)} runner arguments for {len(names)} formals")
    values = dict(zip(names, rest[len(rest) - len(names) :]))
    owner = _compiled_owner(binary, device, decline)
    layout, arg_tys = tuple(owner.abi_layout), owner.module.arg_tys
    kinds, rows = [], []
    for row in owner.formals:
        val = values[row.formal]
        if row.abi_index is None:
            # a constexpr (a config value the launcher passes as a literal) or a value
            # Inductor pinned at compile time: the compiled constant, a guard on a
            # symbolic value
            const = row.constant
            if type(val) is torch.SymInt:
                same = type(const) is int and bool(val == const)
            else:
                same = type(val) is type(const) and val == const
            if not same:
                decline(
                    f"constant argument {row.formal} is {val!r} at the trace; the kernel was compiled for {const!r}"
                )
            kinds.append((row.formal, "constexpr", ""))
            continue
        rows.append((row, val, arg_tys[row.abi_index], layout[row.abi_index]))
    rows.sort(key=lambda r: r[0].abi_index)
    params, written = _abi_params(prov, rows, kinds, decline)
    _scratch_params(params, layout, owner.scratch_abi_indices)
    with torch.cuda.device(device):
        binary._init_handles()
    function = int(binary.function)
    for index, (offset, size) in enumerate(layout):
        # the layout read from the owner's load holds for Triton's load of the same
        # cubin; read back rather than assumed
        got = _check_cuda_bindings(driver.cuFuncGetParamInfo(function, index))
        if tuple(got) != (offset, size):
            decline(
                "the parameter layout of eager's loaded function differs from the compilation's"
            )
    num_warps, smem = int(binary.metadata.num_warps), int(binary.metadata.shared)
    if owner.module.num_warps != num_warps or owner.module.shared != smem:
        decline(
            "the launch configuration of the compilation differs from its loaded module's"
        )
    _append_launch(
        tr,
        decline,
        dims,
        layout,
        params,
        written,
        htt_launch(owner, jit, binary, kinds),
        binary.name,
        function,
        num_warps,
        smem,
    )


class _NoMultiKernel:
    def __init__(self, multi, name):
        self.multi, self.name = multi, name

    def __getattr__(self, item):
        return getattr(self.multi, item)

    def run(self, *args, **kwargs):
        if _active_trace() is None:
            return self.multi.run(*args, **kwargs)
        raise _declined(
            f"{self.name} is a MultiKernelCall (a runtime choice among kernels)"
        )


def _clone_wrapper(model, constants):
    """The generated wrapper cloned into its own namespace: every function of the
    generated module (the partitions, the subgraphs, `call`) re-created over one copy of
    the module's globals with the seams substituted; under graph_partition the Runner is
    copied with the cloned partitions. Returns (callable, namespace, constant names)."""
    from torch._inductor.codegen.multi_kernel import MultiKernelCall
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    fn = model.__func__ if isinstance(model, types.MethodType) else model
    if not isinstance(fn, types.FunctionType):
        raise _declined(
            f"the compiled callable is a {type(model).__name__}, not the generated wrapper"
        )
    g = fn.__globals__
    names = []
    for c in constants:
        name = next((k for k, v in g.items() if v is c), None)
        if name is None:
            raise _declined("a constant of the artifact is not a global of its module")
        names.append(name)
    namespace = dict(g)
    for key, val in g.items():
        if isinstance(val, types.FunctionType) and val.__globals__ is g:
            namespace[key] = _clone(val, namespace)
    seams = _seams()
    for key, val in list(namespace.items()):
        if key in _SEAM_NAMES:
            namespace[key] = getattr(seams, key)
        elif isinstance(val, CachingAutotuner):
            namespace[key] = Recording(val, key)
        elif isinstance(val, MultiKernelCall):
            namespace[key] = _NoMultiKernel(val, key)
    if isinstance(model, types.MethodType):
        runner = copy.copy(model.__self__)
        partitions = []
        for p in getattr(runner, "partitions", ()):
            if not (isinstance(p, types.FunctionType) and p.__globals__ is g):
                raise _declined(
                    "a partition of the generated wrapper is not a function of its module"
                )
            partitions.append(namespace[p.__name__])
        runner.partitions = partitions
        call = types.MethodType(_clone(fn, namespace), runner)
    else:
        call = namespace.get(fn.__name__)
        if not (isinstance(call, types.FunctionType) and call.__code__ is fn.__code__):
            call = _clone(fn, namespace)
    return call, namespace, tuple(names)


@dataclass(frozen=True)
class _Plan:
    """How the artifact's graph inputs become the traced function's arguments: the live
    tensor inputs (roots read from the box per call), the hidden ones (static inputs, bound
    once), the SymInt inputs' sources (input position, source input, size|stride, dim)."""

    n_inputs: int
    live: tuple
    hidden: tuple
    ints: tuple


def _int_sources(example_inputs):
    """Per SymInt example input, the tensor input and dimension carrying its symbol."""
    sources = []
    for k, x in enumerate(example_inputs):
        if isinstance(x, torch.Tensor):
            continue
        if not isinstance(x, torch.SymInt):
            raise _declined(
                f"graph input {k} is a {type(x).__name__}; only tensors and symbolic integers are bound"
            )
        expr = x.node.expr
        found = None
        for j, t in enumerate(example_inputs):
            if not isinstance(t, torch.Tensor):
                continue
            for kind, values in (("size", t.size()), ("stride", t.stride())):
                for d, v in enumerate(values):
                    if isinstance(v, torch.SymInt) and v.node.expr == expr:
                        found = (k, j, kind, d)
                        break
                if found:
                    break
            if found:
                break
        if found is None:
            raise _declined(
                f"symbolic integer input {k} ({expr}) is not the size or stride of any tensor input"
            )
        sources.append(found)
    return tuple(sources)


def _guards_expression(example_inputs):
    env = None
    for x in example_inputs:
        if isinstance(x, torch.SymInt):
            env = x.node.shape_env
            break
        mode = getattr(x, "fake_mode", None)
        if mode is not None and mode.shape_env is not None:
            env = mode.shape_env
            break
    if env is None:
        return None, None
    try:
        return env, env.produce_guards_expression(list(example_inputs))
    except Exception as e:
        raise _declined(
            f"the artifact's guards are unavailable: {type(e).__name__}: {e}"
        ) from e


def _read_int(inputs, source):
    j, kind, d = source
    return inputs[j].size(d) if kind == "size" else inputs[j].stride(d)


def _int_function(expr, symbols):
    """`expr` (sympy, over `symbols`) as a Python function of the symbols' values, as
    dynamo evaluates its guards (PythonPrinter's source under SYMPY_INTERP)."""
    from torch.fx.experimental.symbolic_shapes import SYMPY_INTERP
    from torch.utils._sympy.printers import PythonPrinter

    names = ", ".join(str(s) for s in symbols)
    return eval(f"lambda {names}: {PythonPrinter().doprint(expr)}", dict(SYMPY_INTERP))


class _Retry(Exception):
    """The first call cannot be traced as made but a later one may (a pageable CPU input
    the previous artifact produced outside its tape): the installation stays armed."""


class Installation:
    """One artifact's callable under the policy: armed until its first real call, which
    runs the ordinary wrapper for its result and builds the entry (the trace, the
    lowering, the preparation); ready from then on, with the family's native dispatch as
    the hit path; declined by name when the trace, the lowering or the preparation
    declined (the ordinary wrapper serves); closed by the policy."""

    def __init__(
        self,
        policy,
        model,
        example_inputs,
        static_input_idxs,
        constants,
        *,
        device_index,
        is_backward,
        mutated_input_idxs=(),
    ):
        self.policy = policy
        self.model = model
        self.static_input_idxs = tuple(static_input_idxs)
        self.constants = tuple(constants)
        self.device_index = int(device_index)
        self.is_backward = bool(is_backward)
        self.mutated_input_idxs = tuple(mutated_input_idxs)
        self.n_inputs = len(example_inputs)
        self.artifact = None
        self.status = "armed"
        self.decline = None
        self.replay = None
        self.plan = None
        self.traced = None
        self.build_s = None
        self.static_rebinds = 0
        self.retries = 0
        self._lock = threading.RLock()
        self._family = None
        self._boxer = None
        self._hidden = ()
        self._bound = ()
        # per output: "tensor", "int" (a SymInt the wrapper returns: a size or stride of
        # an input, or an expression over them, read per call) or "none"; None when
        # every output is a tensor; `_non_tensor` lists the others' positions in order
        self._output_kinds = None
        self._non_tensor = ()
        self._int_outputs = {}
        try:
            self.shape_env, self.guards_code = _guards_expression(example_inputs)
            self.int_sources = _int_sources(example_inputs)
        except _ht().Declined as e:
            self.decline = str(e)

    # ---- the states -----------------------------------------------------------------

    def __call__(self, new_inputs):
        family = self._family
        if family is not None:
            box = self._boxer(tuple(new_inputs))
            replay = self.replay
            replay._python_dispatch = True
            try:
                outputs = family.dispatch(box)
            finally:
                replay._python_dispatch = False
            if outputs is not _MISSED:
                replay.calls += 1
                if self._output_kinds is not None:
                    outputs = self._assemble(outputs, new_inputs)
                new_inputs.clear()
                return outputs
            return self._cold(new_inputs)
        return self._first(new_inputs)

    def _first(self, new_inputs):
        with self._lock:
            if self._family is not None:
                return self(new_inputs)
            if self.status != "armed":
                return self.model(new_inputs)
            inputs = tuple(new_inputs)
            result = self.model(new_inputs)
            if self.decline is not None:
                self._declined(self.decline)
                return result
            t0 = time.perf_counter()
            try:
                self._build(inputs, result)
            except _Retry as e:
                self.retries += 1
                if self.retries >= 3:
                    self._declined(f"{e} (after {self.retries} calls)")
                else:
                    cudagraphs_log.info("host trace: %s; retried at the next call", e)
                return result
            except _ht().Declined as e:
                self._declined(str(e))
                return result
            except BaseException:
                self.status = "failed"
                raise
            self.build_s = time.perf_counter() - t0
            self.status = "ready"
            cudagraphs_log.info(
                "host trace: artifact traced in %.1f s: %s",
                self.build_s,
                self.summary(),
            )
            return result

    def _cold(self, new_inputs):
        inputs = tuple(new_inputs)
        new_inputs.clear()
        replay = self.replay
        plan = self.plan
        if len(replay.variants) >= replay.max_variants:
            self._declined(
                f"the call misses all {replay.max_variants} variants of the entry"
            )
            return list(self.model(list(inputs)))
        with self._lock:
            hidden = tuple(inputs[i] for i in plan.hidden)
            if any(a is not b for a, b in zip(hidden, self._hidden)):
                # a static input replaced (E32: not checked on the hit path): the miss
                # re-reads it, and the family's bound inputs follow
                self._hidden = hidden
                self.static_rebinds += 1
            args = (*(inputs[i] for i in plan.live), *self._hidden, *self.constants)
            try:
                outputs = replay._serve_cold(args)
            finally:
                if self._family is not None:
                    try:
                        self._after_cold()
                    except _ht().Declined as e:
                        self._declined(str(e))
        kinds = self._output_kinds
        if kinds is not None and len(outputs) != len(kinds):
            # served by a variant (the tensor outputs alone); the ordinary wrapper
            # returns the whole tuple
            return self._assemble(outputs, inputs)
        return list(outputs)

    # ---- the build ------------------------------------------------------------------

    def _build(self, inputs, result):
        from .direct_hosttrace import HostTraceReplay

        if len(inputs) != self.n_inputs:
            raise _declined(
                f"{len(inputs)} inputs at the call, {self.n_inputs} at the compile"
            )
        if not isinstance(result, (list, tuple)):
            raise _declined(
                f"the artifact returned a {type(result).__name__}, not a tuple"
            )
        kinds = []
        for o in result:
            if isinstance(o, torch.Tensor):
                kinds.append("tensor")
            elif o is None:
                kinds.append("none")
            elif type(o) is int:
                kinds.append("int")
            else:
                raise _declined(f"an output of the artifact is a {type(o).__name__}")
        self._output_kinds = None if all(k == "tensor" for k in kinds) else tuple(kinds)
        self._non_tensor = tuple((i, k) for i, k in enumerate(kinds) if k != "tensor")
        live, hidden = [], []
        static = OrderedSet(self.static_input_idxs)
        for i, x in enumerate(inputs):
            if isinstance(x, torch.Tensor):
                if x.device.type == "cpu" and not x.is_pinned():
                    # a saved tensor the forward produced outside its tape (an attention's
                    # philox state is a CPU tensor outside a capture, a device tensor from
                    # the forward's replay): traced once the producer replays
                    raise _Retry(
                        f"graph input {i} is a pageable CPU tensor at this call"
                    )
                if i in static and get_static_address_type(x) is not None:
                    hidden.append(i)
                else:
                    live.append(i)
        if not live and not hidden:
            raise _declined("the artifact has no tensor input")
        plan = _Plan(self.n_inputs, tuple(live), tuple(hidden), self.int_sources)
        traced, self._namespace, names = _clone_wrapper(self.model, self.constants)
        self.traced = self._traced_function(traced, plan, names)
        self._hidden = tuple(inputs[i] for i in hidden)
        args = (*(inputs[i] for i in live), *self._hidden, *self.constants)
        replay = HostTraceReplay(self.traced, args, warm_up=False)
        try:
            family = replay._hot
            if family is None or not replay.variants:
                raise _declined("the entry built no variant")
            if family.pinned:
                raise _declined(
                    "pinned CPU inputs are not served by the policy's hit path"
                )
            if family.outputs is not None or family.sequence is not None:
                raise _declined(
                    "the output arena and the allocation sequence box per-call entries after the tape's tensors; the hidden-input binding needs the tape's tensors last"
                )
            self.replay = replay
            self.plan = plan
            self._family = family
            self._after_cold()
        except BaseException:
            self.replay = None
            self._family = None
            replay.close()
            raise

    def _traced_function(self, call, plan, constant_names):
        n_inputs, order, ints = plan.n_inputs, plan.live + plan.hidden, plan.ints
        n_tensors, namespace = len(order), self._namespace
        env, code = self.shape_env, self.guards_code

        def inductor_call(*roots):
            args = [None] * n_inputs
            for k in range(n_tensors):
                args[order[k]] = roots[k]
            for k, name in enumerate(constant_names):
                namespace[name] = roots[n_tensors + k]
            for pos, j, kind, d in ints:
                t = args[j]
                args[pos] = t.size(d) if kind == "size" else t.stride(d)
            tracing = _active_trace() is not None
            if code is not None and tracing:
                # the artifact's own guards under the trace: each comparison a guard
                if not bool(env.evaluate_guards_expression(code, args)):
                    raise _declined(
                        f"the artifact's guards do not hold at the trace: {code[:300]}"
                    )
            out = call(args)
            if kinds is None or not tracing:
                return out
            # the recorder takes tensors: the integer outputs (sizes the wrapper returns
            # for the backward) are read from the inputs at the replay
            self._record_int_outputs(out, roots, order)
            return tuple(o for o, k in zip(out, kinds) if k == "tensor")

        kinds = self._output_kinds
        return inductor_call

    def _record_int_outputs(self, out, roots, order):
        kinds = self._output_kinds
        if len(out) != len(kinds):
            raise _declined(
                f"the artifact returned {len(out)} outputs at the trace, {len(kinds)} at the first call"
            )
        sources = {}
        for k, root in enumerate(roots):
            for kind, values in (("size", root.shape), ("stride", root._sym_strides)):
                for d, v in enumerate(values):
                    if isinstance(v, torch.SymInt):
                        sources.setdefault(v.node.expr, (order[k], kind, d))
        found = {}
        for i, (o, kind) in enumerate(zip(out, kinds)):
            if kind != "int":
                continue
            if type(o) is int:
                found[i] = ("const", o)
                continue
            if not isinstance(o, torch.SymInt):
                raise _declined(
                    f"output {i} is a {type(o).__name__} at the trace, an int at the first call"
                )
            expr = o.node.expr
            if expr.is_Integer:
                found[i] = ("const", int(expr))
            elif expr in sources:
                found[i] = ("input", *sources[expr])
            else:
                # an expression over the inputs' sizes (a DynamicCache's seen-token
                # count is the cache length + 1): a Python function of its sources
                symbols = sorted(expr.free_symbols, key=str)
                missing = [str(s) for s in symbols if s not in sources]
                if missing:
                    raise _declined(
                        f"integer output {i} is {expr}; {', '.join(missing)} is not a size or stride of an input"
                    )
                fn = _int_function(expr, symbols)
                found[i] = ("expr", fn, tuple(sources[s] for s in symbols))
        self._int_outputs = found

    def _assemble(self, outputs, inputs):
        # the tensors in order, the non-tensor outputs inserted at their positions
        # (ascending, so each insert lands at its final index)
        result = list(outputs)
        for i, kind in self._non_tensor:
            if kind == "none":
                result.insert(i, None)
                continue
            source = self._int_outputs[i]
            if source[0] == "const":
                result.insert(i, source[1])
            elif source[0] == "input":
                result.insert(i, _read_int(inputs, source[1:]))
            else:
                result.insert(i, source[1](*(_read_int(inputs, s) for s in source[2])))
        return result

    def _after_cold(self):
        """The family's box layout and bound inputs after any build or miss: the box of a
        hit is the live roots alone (my boxer over the graph inputs), the hidden roots
        and the arena are the dispatch's bound trailing inputs (a growth rebinds the
        arena alone, so every cold path rebinds the whole tuple)."""
        family = self._family
        n_live = len(self.plan.live)
        for w in family.written:
            if w >= n_live and w < n_live + len(self._hidden):
                t = self._hidden[w - n_live]
                if torch._C._is_cow_tensor(t):
                    raise _declined(
                        "a static input written by the artifact is copy-on-write"
                    )
        written = [w for w in family.written if w < n_live]
        self._boxer = torch._C._HostTraceBoxer(list(self.plan.live), written)
        bound = (*self._hidden, *self.constants)
        if family.arena is not None:
            bound = (*bound, family.arena.tensor)
        family.dispatch.bind(bound)
        self._bound = bound

    # ---- the outcomes ---------------------------------------------------------------

    def _declined(self, reason):
        if "aten.random_.from" in reason and "fallback_random" not in reason:
            # Inductor's own dropout / rand draw their per-call seeds with
            # aten.randint.low_out -> aten.random_.from, which no host of the line
            # traces; with config.fallback_random the random ops are ATen's (a traced
            # native_dropout with its rng slot), so the log names the configuration
            reason += (
                " [Inductor's own random seeds: torch._inductor.config.fallback_random"
                " = True routes dropout and rand through ATen's traced hosts]"
            )
        self.status = "declined"
        self.decline = reason
        self._family = None
        # the artifact drops a declined installation; the policy keeps it for its report
        self.policy._declines.append((self, reason))
        artifact = self.artifact() if self.artifact is not None else None
        if artifact is not None and artifact.current_callable is self:
            artifact.current_callable = self.model
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "cudagraph_host_trace_decline",
                "encoding": "string",
            },
            payload_fn=lambda: reason,
        )
        log_cudagraph_skip_and_bump_counter(
            f"skipping cudagraphs due to a host trace decline: {reason}"
        )

    def summary(self):
        replay = self.replay
        if replay is None or not replay.variants:
            return {"status": self.status, "decline": self.decline}
        tape = replay.tape
        return {
            "status": self.status,
            "is_backward": self.is_backward,
            "inputs": self.n_inputs,
            "live": len(self.plan.live),
            "hidden": len(self.plan.hidden),
            "constants": len(self.constants),
            "ints": len(self.plan.ints),
            "launches": tape.num_launches,
            "regions": tape.num_regions,
            "guards": tape.num_guards,
            "outputs": len(tape.outputs),
            "non_tensor_outputs": 0
            if self._output_kinds is None
            else sum(k != "tensor" for k in self._output_kinds),
            "allocations": len(replay.lowered.allocations),
            "variants": len(replay.variants),
            "calls": replay.calls,
            "misses": replay.misses,
            "traces": replay.traces,
            "ordinary": replay.ordinary,
        }

    def close(self):
        with self._lock:
            if self.status == "closed":
                return
            artifact = self.artifact() if self.artifact is not None else None
            if artifact is not None and artifact.current_callable is self:
                artifact.current_callable = self.model
            self._family = None
            if self.replay is not None:
                self.replay.close()
            self.status = "closed"


class HostTracePolicy(CUDAGraphPolicy):
    """The tape policy: one Installation per artifact at cudagraphify. Set explicitly as
    config.cudagraph_policy (tests, inspection) or selected by
    config.triton.cudagraph_host_trace (the process-wide default instance)."""

    def __init__(self):
        self._installations = []
        self._declines = []
        self._lock = threading.Lock()

    def __deepcopy__(self, memo):
        # config snapshots share the policy and its live installations
        return self

    @property
    def installations(self):
        return tuple(i for i in (r() for r in self._installations) if i is not None)

    @property
    def declines(self):
        return tuple(self._declines)

    def cudagraphify(
        self,
        model,
        example_inputs,
        static_input_idxs,
        *,
        device_index,
        is_backward,
        is_inference,
        constants=(),
        mutated_input_idxs=(),
        **kwargs,
    ):
        installation = Installation(
            self,
            model,
            example_inputs,
            static_input_idxs,
            constants,
            device_index=device_index,
            is_backward=is_backward,
            mutated_input_idxs=mutated_input_idxs,
        )
        with self._lock:
            self._installations.append(weakref.ref(installation))
        counters["inductor"]["cudagraph_host_trace_installations"] += 1
        return installation

    def wrap_output(self, output_code):
        callable_ = getattr(output_code, "current_callable", None)
        if isinstance(callable_, Installation):
            callable_.artifact = weakref.ref(output_code)
        return output_code

    def close(self):
        for installation in self.installations:
            installation.close()


_DEFAULT = None
_DEFAULT_LOCK = threading.Lock()


def host_trace_policy():
    """The process-wide policy config.triton.cudagraph_host_trace selects."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = HostTracePolicy()
        return _DEFAULT
