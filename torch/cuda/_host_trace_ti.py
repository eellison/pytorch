"""The CUDA ops with a traced TensorIterator sibling (private): elementwise
add, sub, rsub, mul, div (their in-place, out= and .Scalar forms too), silu,
gelu, sin, cos, exp, rsqrt, neg, reciprocal, tanh, sqrt, pow, copy_ (incl.
casts, and the contiguous memcpy), fill_ / zero_ (eager's memset over a dense
tensor) and the fill factories (full, zeros, ones, *_like, new_*), arange, the
comparisons eq / ne / lt / le / gt / ge, masked_fill, clamp / clamp_min /
clamp_max / relu, the reductions sum, mean, amax / max, index_copy_ and
index_put_; plus the converted softmax / log_softmax host, whose entry
allocates the output, and the eager hosts with a sibling beside them
(index_select, embedding, cat, triu / tril).

Each entry is what the op's CUDA kernel host does around gpu_kernel or
gpu_reduce_kernel, on the SymInt-typed sibling iterator in
aten/src/ATen/cuda/host_trace/ti. Under a trace the mode calls it in place of
the op; at a replay's build it runs in ordinary mode so the captured graph
holds the launches the tape describes. Operands must be CUDA tensors of one
dtype; a binary op may take one CPU scalar operand, the Python number the
dispatcher unwrapped from its wrapped tensor, whose value is a constant of
the tape as the real kernel bakes it into its functor. A 0-dim CPU tensor
the caller made is an implicit CPU scalar input and declines by name. Type
promotion of a CUDA operand and the 64-bit indexing split decline by name
inside the entry.

Entry contract: an entry takes the op's arguments as the op's schema orders
them, runs the sibling (torch._C._host_trace_ti_*) or raises Declined, and
performs the op's own argument checks first with the op's error texts so a
call that eager refuses is refused identically. It must not dispatch the op
it stands for (the mode declines that re-entry). register_traced_entry
refuses a second registration of an op unless replace=True.
"""

from __future__ import annotations

import math

import torch
from torch.cuda._host_trace import _TRACED_ENTRIES, Declined, register_traced_entry


aten = torch.ops.aten
_C = torch._C


def _cuda_operands(op, *operands):
    for t in operands:
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            what = (
                "a scalar"
                if not isinstance(t, torch.Tensor)
                else f"a {t.device} tensor"
            )
            raise Declined(
                f"host_trace: {op} with {what} operand is not traced (declined)"
            )


def _scalar_operand(op, t):
    # a CPU scalar as the real binary ops accept one beside a CUDA tensor
    # (TensorIteratorConfig allow_cpu_scalars): the Python number the
    # dispatcher unwrapped from its wrapped tensor (the binding wraps it
    # again). A symbolic number is pinned, as the wrapping pins it in eager.
    # A 0-dim CPU tensor the caller made (a closure, a list entry, a pinned
    # argument) is a host value with no record on the tape, read once at the
    # trace: an implicit CPU scalar input, declined by name
    if isinstance(t, torch.SymInt):
        return int(t)
    if isinstance(t, torch.SymFloat):
        return float(t)
    if isinstance(t, torch.SymBool):
        return bool(t)
    if isinstance(t, torch.Tensor):
        if t.is_cpu and t.dim() == 0:
            raise Declined(
                f"host_trace: {op} with a 0-dim CPU tensor operand: an implicit CPU scalar input, read on the host at the trace and never again; pass a Python number or a CUDA tensor (declined)"
            )
        return None
    return t if isinstance(t, (bool, int, float, complex)) else None


def _operands(op, *tensors):
    # the tensor operands with at most one of them a CPU scalar (the kernel
    # host reads its value; the iterator refuses two); anything else declines
    operands = []
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            operands.append(t)
            continue
        scalar = _scalar_operand(op, t)
        if scalar is None:
            _cuda_operands(op, t)
        operands.append(scalar)
    if not any(isinstance(t, torch.Tensor) and t.is_cuda for t in operands):
        _cuda_operands(op, tensors[0])
    return operands


def _binary_operands(op, self, other):
    # (self, other) with at most one of them a CPU scalar; anything else declines
    return tuple(_operands(op, self, other))


def _is_bool(t):
    return t.dtype is torch.bool if isinstance(t, torch.Tensor) else isinstance(t, bool)


def _sub_check(self, other):
    # BinaryOps.h sub_check (a NotImplementedError in eager, as
    # TORCH_CHECK_NOT_IMPLEMENTED raises)
    if _is_bool(self) and _is_bool(other):
        raise NotImplementedError(
            "Subtraction, the `-` operator, with two bool tensors is not supported. "
            "Use the `^` or `logical_xor()` operator instead."
        )
    if _is_bool(self) or _is_bool(other):
        raise NotImplementedError(
            "Subtraction, the `-` operator, with a bool tensor is not supported. "
            "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead."
        )


def _alpha_check(dtype, alpha):
    # BinaryOps.h alpha_check, before anything else as in the real add
    if isinstance(alpha, bool) and dtype is not torch.bool:
        raise RuntimeError("Boolean alpha only supported for Boolean results.")
    if isinstance(alpha, float) and not (dtype.is_floating_point or dtype.is_complex):
        raise RuntimeError(
            "For integral input tensors, argument alpha must not be a floating point number."
        )
    if isinstance(alpha, complex) and not dtype.is_complex:
        raise RuntimeError(
            "For non-complex input tensors, argument alpha must not be a complex number."
        )


def _category(t):
    # TypeProperties.cpp's promotion categories of a scalar operand
    if isinstance(t, torch.Tensor):
        dtype = t.dtype
    else:
        dtype = {
            bool: torch.bool,
            int: torch.int64,
            float: torch.float64,
            complex: torch.complex128,
        }[type(t)]
    if dtype.is_complex:
        return 3
    if dtype.is_floating_point:
        return 2
    return 0 if dtype is torch.bool else 1


def _promoted(self, other):
    # the result dtype the real add checks alpha against, without dispatching
    # result_type under the mode; a Python number or a CPU scalar tensor keeps
    # the device tensor's dtype unless it is of a higher category
    # (TypeProperties.cpp result_type: the sibling declines that promotion
    # later, the check only needs the category)
    if all(isinstance(t, torch.Tensor) and t.is_cuda for t in (self, other)):
        return torch.promote_types(self.dtype, other.dtype)
    tensor, scalar = (
        (self, other)
        if isinstance(self, torch.Tensor) and self.is_cuda
        else (other, self)
    )
    if not isinstance(tensor, torch.Tensor):
        return torch.get_default_dtype()
    if _category(scalar) <= _category(tensor):
        return tensor.dtype
    if _category(scalar) == 3:
        return torch.promote_types(tensor.dtype, torch.complex64)
    return torch.get_default_dtype() if _category(scalar) == 2 else torch.int64


# `out`: the structured kernel's out= or in-place destination (self for the
# in-place form, which writes the input's storage), None for the allocating
# form. The .Scalar overloads are the same kernels over a wrapped number
# (BinaryOps.cpp add(Tensor, Scalar) and friends). alpha crosses as the Python
# number it is: an int stays an int64 Scalar that add_kernel's
# alpha.to<opmath_t>() rounds once (2**62 + 2**38 + 1 differs from its float),
# never a double first.


def _add(self, other, alpha=1, *, out=None):
    _alpha_check(_promoted(self, other), alpha)
    a, b = _binary_operands(aten.add.Tensor, self, other)
    return _C._host_trace_ti_add(a, b, alpha, out)


def _sub(self, other, alpha=1, *, out=None):
    # BinaryOps.cpp sub_out: sub_check, then the add kernel with -alpha
    _sub_check(self, other)
    _alpha_check(_promoted(self, other), alpha)
    a, b = _binary_operands(aten.sub.Tensor, self, other)
    return _C._host_trace_ti_add(a, b, -alpha, out)


def _rsub(self, other, alpha=1):
    # BinaryOps.cpp rsub(Tensor, Scalar): sub(wrapped_scalar_tensor(other),
    # self, alpha), so the add kernel over (other, self) with -alpha
    _sub_check(other, self)
    _alpha_check(_promoted(self, other), alpha)
    a, b = _binary_operands(aten.rsub.Scalar, other, self)
    return _C._host_trace_ti_add(a, b, -alpha)


def _mul(self, other, *, out=None):
    a, b = _binary_operands(aten.mul.Tensor, self, other)
    return _C._host_trace_ti_mul(a, b, out)


def _div(self, other, *, out=None):
    a, b = _binary_operands(aten.div.Tensor, self, other)
    return _C._host_trace_ti_div(a, b, out)


def _inplace(op, fn):
    # the in-place overload: self is the destination
    def entry(self, *args, **kwargs):
        _cuda_operands(op, self)
        return fn(self, *args, out=self, **kwargs)

    return entry


def _out(op, fn):
    # the out= overload: `out` is the destination and must already have the
    # broadcast shape (a resize inside the trace declines by name)
    def entry(*args, out, **kwargs):
        _cuda_operands(op, out)
        return fn(*args, out=out, **kwargs)

    return entry


def _unary(op, binding):
    def entry(self):
        _cuda_operands(op, self)
        return binding(self)

    return entry


def _gelu(self, approximate="none"):
    _cuda_operands(aten.gelu.default, self)
    return _C._host_trace_ti_gelu(self, approximate)


def _pow_tensor_scalar(self, exponent):
    # Pow.cpp's meta and impl for pow.Tensor_Scalar: the numpy check, the
    # result dtype (a cast of the base declines as promotion), then the
    # exponents 0 and 1 are a fill and a copy (not traced), the rest the kernel
    _cuda_operands(aten.pow.Tensor_Scalar, self)
    if isinstance(exponent, torch.SymInt):
        exponent = int(exponent)
    elif isinstance(exponent, torch.SymFloat):
        exponent = float(exponent)
    integral = not (self.dtype.is_floating_point or self.dtype.is_complex)
    if integral and isinstance(exponent, (bool, int)) and exponent < 0:
        raise RuntimeError("Integers to negative integer powers are not allowed.")
    if _promoted(self, exponent) != self.dtype:
        raise Declined(
            f"host_trace: pow of a {self.dtype} base with exponent {exponent} promotes the base, which is not traced (declined)"
        )
    if exponent == 0 or exponent == 1:
        raise Declined(
            f"host_trace: pow with exponent {exponent} is a fill or a copy in the real op, which is not traced (declined)"
        )
    return _C._host_trace_ti_pow_tensor_scalar(self, exponent)


def _copy_(self, src, non_blocking=False):
    _cuda_operands(aten.copy_.default, self, src)
    return _C._host_trace_ti_copy_(self, src)


def _clone(self, memory_format=None):
    # at::native::clone: the allocation empty_like makes for the format, then
    # copy_ (a contiguous pair is a memcpy in the real op and declines there)
    _cuda_operands(aten.clone.default, self)
    mf = torch.preserve_format if memory_format is None else memory_format
    out = torch.empty_like(self, memory_format=mf)
    return _C._host_trace_ti_copy_(out, self)


register_traced_entry(aten.add.Tensor, _add)
register_traced_entry(aten.sub.Tensor, _sub)
register_traced_entry(aten.rsub.Scalar, _rsub)
register_traced_entry(aten.mul.Tensor, _mul)
register_traced_entry(aten.div.Tensor, _div)
for _name, _fn in (("add", _add), ("sub", _sub), ("mul", _mul), ("div", _div)):
    _op, _op_ = getattr(aten, _name), getattr(aten, _name + "_")
    register_traced_entry(_op.Scalar, _fn)
    register_traced_entry(_op.out, _out(_op.out, _fn))
    register_traced_entry(_op_.Tensor, _inplace(_op_.Tensor, _fn))
    register_traced_entry(_op_.Scalar, _inplace(_op_.Scalar, _fn))
register_traced_entry(
    aten.silu.default, _unary(aten.silu.default, _C._host_trace_ti_silu)
)
register_traced_entry(aten.gelu.default, _gelu)
register_traced_entry(aten.copy_.default, _copy_)
# reciprocal is what `2.0 / x` dispatches before its scalar multiply
register_traced_entry(
    aten.reciprocal.default,
    _unary(aten.reciprocal.default, _C._host_trace_ti_reciprocal),
)
register_traced_entry(
    aten.tanh.default, _unary(aten.tanh.default, _C._host_trace_ti_tanh)
)
register_traced_entry(
    aten.sqrt.default, _unary(aten.sqrt.default, _C._host_trace_ti_sqrt)
)
register_traced_entry(aten.pow.Tensor_Scalar, _pow_tensor_scalar)


def _native_dropout(self, p, train):
    # Dropout.cu's fused kernel through the sibling (train mode only): the
    # philox increment is an expression of the element count on the tape and
    # the replay sets it on the graph before each call (commit 9)
    _cuda_operands(aten.native_dropout.default, self)
    return _C._host_trace_ti_native_dropout(self, float(p), train)


register_traced_entry(aten.clone.default, _clone)
register_traced_entry(aten.native_dropout.default, _native_dropout)


# ---- fills (FillKernel.cu's entry): fill_ and zero_ on a tensor, and the factories
# as eager composes them (TensorFactories.cpp, Fill.cpp): the allocation,
# through the trace mode (a traced root), then fill_. The value is a constant
# of the variant; a symbolic value is pinned, as eager's Scalar pins it. The
# factories keep their SymInt sizes (registered with symint=True), so a size
# derived from a traced shape stays a value of the tape.


def _plain_scalar(op, v):
    # a Scalar argument that is a value of the op (a fill value, a bound):
    # the Python number, a symbolic number pinned (as _scalar_operand pins an
    # operand), a 0-dim CPU tensor declined there
    s = _scalar_operand(op, v)
    if s is None:
        raise Declined(f"host_trace: {op} with a tensor value is not traced (declined)")
    return s


def _factory_device(op, layout, device, pin_memory):
    # the options the factory would allocate with: strided, not pinned, on
    # the trace's device (an allocation elsewhere is not traced)
    if layout not in (None, torch.strided):
        raise Declined(
            f"host_trace: {op} with layout {layout} is not traced (declined)"
        )
    if pin_memory:
        raise Declined(f"host_trace: {op} into pinned memory is not traced (declined)")
    dev = torch.device(device) if device is not None else torch.get_default_device()
    if dev.type != "cuda":
        raise Declined(f"host_trace: {op} on {dev} is not traced (declined)")
    return dev


def _full_dtype(fill_value, dtype):
    # TensorFactories.cpp infer_full_options
    if dtype is not None:
        return dtype
    if isinstance(fill_value, bool):
        return torch.bool
    if isinstance(fill_value, int):
        return torch.int64
    if isinstance(fill_value, complex):
        double = torch.get_default_dtype() is torch.float64
        return torch.complex128 if double else torch.complex64
    return torch.get_default_dtype()


def _fill_(self, value):
    _cuda_operands(aten.fill_.Scalar, self)
    return _C._host_trace_ti_fill_(self, _plain_scalar(aten.fill_.Scalar, value))


def _zero_(self):
    # TensorFactories.cu zero_cuda_: a memset over a dense tensor (a memset
    # record of the tape, no launch), fill_(0) over a strided one
    _cuda_operands(aten.zero_.default, self)
    return _C._host_trace_ti_zero_(self)


def _fill(self, value):
    # Fill.cpp fill: empty_like then fill_
    _cuda_operands(aten.fill.Scalar, self)
    value = _plain_scalar(aten.fill.Scalar, value)
    return _C._host_trace_ti_fill_(torch.empty_like(self), value)


def _full(size, fill_value, dtype=None, layout=None, device=None, pin_memory=None):
    value = _plain_scalar(aten.full.default, fill_value)
    dev = _factory_device(aten.full.default, layout, device, pin_memory)
    out = torch.empty(list(size), dtype=_full_dtype(value, dtype), device=dev)
    return _C._host_trace_ti_fill_(out, value)


def _zeros(size, dtype=None, layout=None, device=None, pin_memory=None):
    dev = _factory_device(aten.zeros.default, layout, device, pin_memory)
    dtype = torch.get_default_dtype() if dtype is None else dtype
    return _C._host_trace_ti_zero_(torch.empty(list(size), dtype=dtype, device=dev))


def _ones(size, dtype=None, layout=None, device=None, pin_memory=None):
    # TensorFactories.cpp ones: full(size, 1.)
    return _full(size, 1.0, dtype, layout, device, pin_memory)


def _like(op, self, dtype, layout, device, pin_memory, memory_format):
    # at::empty_like with self's options merged with the given ones
    _cuda_operands(op, self)
    if layout not in (None, torch.strided) or pin_memory:
        raise Declined(
            f"host_trace: {op} with layout {layout} or pinned is not traced (declined)"
        )
    if device is not None and torch.device(device).type != "cuda":
        raise Declined(f"host_trace: {op} on {device} is not traced (declined)")
    return torch.empty_like(
        self, dtype=dtype, device=device, memory_format=memory_format
    )


def _zeros_like(
    self, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    out = _like(
        aten.zeros_like.default, self, dtype, layout, device, pin_memory, memory_format
    )
    return _C._host_trace_ti_zero_(out)


def _ones_like(
    self, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    out = _like(
        aten.ones_like.default, self, dtype, layout, device, pin_memory, memory_format
    )
    return _C._host_trace_ti_fill_(out, 1.0)


def _full_like(
    self,
    fill_value,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    value = _plain_scalar(aten.full_like.default, fill_value)
    out = _like(
        aten.full_like.default, self, dtype, layout, device, pin_memory, memory_format
    )
    return _C._host_trace_ti_fill_(out, value)


def _new(op, self, size, dtype, layout, device, pin_memory):
    # self.new_empty(size, ...): self's dtype and device unless given
    _cuda_operands(op, self)
    return self.new_empty(
        list(size), dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


def _new_zeros(self, size, dtype=None, layout=None, device=None, pin_memory=None):
    out = _new(aten.new_zeros.default, self, size, dtype, layout, device, pin_memory)
    return _C._host_trace_ti_zero_(out)


def _new_ones(self, size, dtype=None, layout=None, device=None, pin_memory=None):
    out = _new(aten.new_ones.default, self, size, dtype, layout, device, pin_memory)
    return _C._host_trace_ti_fill_(out, 1.0)


def _new_full(
    self, size, fill_value, dtype=None, layout=None, device=None, pin_memory=None
):
    value = _plain_scalar(aten.new_full.default, fill_value)
    out = _new(aten.new_full.default, self, size, dtype, layout, device, pin_memory)
    return _C._host_trace_ti_fill_(out, value)


register_traced_entry(aten.fill_.Scalar, _fill_)
register_traced_entry(aten.zero_.default, _zero_)
register_traced_entry(aten.fill.Scalar, _fill)
register_traced_entry(aten.full.default, _full, symint=True)
register_traced_entry(aten.zeros.default, _zeros, symint=True)
register_traced_entry(aten.ones.default, _ones, symint=True)
register_traced_entry(aten.zeros_like.default, _zeros_like)
register_traced_entry(aten.ones_like.default, _ones_like)
register_traced_entry(aten.full_like.default, _full_like)
register_traced_entry(aten.new_zeros.default, _new_zeros, symint=True)
register_traced_entry(aten.new_ones.default, _new_ones, symint=True)
register_traced_entry(aten.new_full.default, _new_full, symint=True)


# ---- arange (RangeFactories.cu's entry): TensorFactories.cpp arange's dtype rule and
# RangeFactories.cu arange_cuda_out's host (RangeUtils.h's checks and size),
# in SymInt arithmetic where a bound is symbolic, so the model-derived
# `torch.arange(past, past + q)` keeps the cache length as a value: its
# checks are guards, the length a size, start and step fields of the launch.


def _sym_number(op, v):
    if isinstance(v, (torch.SymInt, torch.SymFloat)):
        return v
    if isinstance(v, torch.SymBool):
        return bool(v)
    if isinstance(v, (bool, int, float)):
        return v
    _cuda_operands(op, v)
    raise Declined(f"host_trace: {op} with a tensor bound is not traced (declined)")


def _arange_check_bounds(start, end, step):
    # RangeUtils.h arange_check_bounds, its texts; a symbolic comparison is a
    # guard of the tape
    if not (step > 0 or step < 0):
        raise RuntimeError("step must be nonzero")
    for v in (start, end):
        if isinstance(v, float) and not math.isfinite(v):
            raise RuntimeError(f"unsupported range: {start} -> {end}")
    if not ((step > 0 and end >= start) or (step < 0 and end <= start)):
        raise RuntimeError("upper bound and lower bound inconsistent with step sign")


def _arange_size(dtype, start, end, step):
    # RangeUtils.h compute_arange_size: the exact integer formula for int64
    # over integral bounds, a double ceil otherwise (an integer division of
    # non-negative operands, as the checks above leave them, floors)
    integral = all(isinstance(v, (bool, int, torch.SymInt)) for v in (start, end, step))
    if dtype is torch.int64 and integral:
        sgn = 1 if step > 0 else -1
        return (end - start + step - sgn) // step
    return math.ceil((end - start) / step)


def _arange_entry(op):
    def entry(*bounds, dtype=None, layout=None, device=None, pin_memory=None):
        if op is aten.arange.default:
            start, end, step = 0, bounds[0], 1
        elif op is aten.arange.start:
            (start, end), step = bounds, 1
        else:
            start, end, step = bounds
        start, end, step = (_sym_number(op, v) for v in (start, end, step))
        dev = _factory_device(op, layout, device, pin_memory)
        integral = all(
            isinstance(v, (bool, int, torch.SymInt)) for v in (start, end, step)
        )
        if dtype is None:
            dtype = torch.int64 if integral else torch.get_default_dtype()
        _arange_check_bounds(start, end, step)
        size = _arange_size(dtype, start, end, step)
        out = torch.empty([size], dtype=dtype, device=dev)
        if dtype.is_floating_point:
            # a plain int bound crosses as the int it is and converts to the
            # accumulate type once in the kernel host, as eager's
            # start.to<accscalar_t>() does; a symbolic one is its float
            # expression, evaluated per call
            start, step = (
                torch.sym_float(v) if isinstance(v, torch.SymInt) else v
                for v in (start, step)
            )
        elif not integral:
            raise Declined(
                f"host_trace: {op} with a floating bound into {dtype} (a truncating conversion) is not traced (declined)"
            )
        return _C._host_trace_ti_arange(start, step, out)

    return entry


for _op in (aten.arange.default, aten.arange.start, aten.arange.start_step):
    register_traced_entry(_op, _arange_entry(_op), symint=True)


# ---- comparisons (CompareEQKernel.cu / CompareKernels.cu entries): a bool result over operands of one
# dtype, one of them may be a CPU scalar (a scalar that would promote the
# CUDA operand declines inside the entry, as for add)


def _compare(op, name):
    def entry(self, other):
        a, b = _binary_operands(op, self, other)
        return _C._host_trace_ti_compare(a, b, name)

    return entry


for _name in ("eq", "ne", "lt", "le", "gt", "ge"):
    for _overload in ("Scalar", "Tensor"):
        _op = getattr(getattr(aten, _name), _overload)
        register_traced_entry(_op, _compare(_op, _name))


# ---- masked_fill (Indexing.cu's entry)


def _masked_fill_(self, mask, value):
    _cuda_operands(aten.masked_fill_.Scalar, self, mask)
    value = _plain_scalar(aten.masked_fill_.Scalar, value)
    return _C._host_trace_ti_masked_fill_(self, mask, value)


def _masked_fill(self, mask, value):
    # TensorAdvancedIndexing.cpp masked_fill: expand_outplace, a contiguous
    # clone of self, then masked_fill_ (the clone of a contiguous self is a
    # memcpy in eager and declines there)
    _cuda_operands(aten.masked_fill.Scalar, self, mask)
    value = _plain_scalar(aten.masked_fill.Scalar, value)
    _, expanded = torch.broadcast_tensors(mask, self)
    result = expanded.clone(memory_format=torch.contiguous_format)
    return _C._host_trace_ti_masked_fill_(result, mask, value)


register_traced_entry(aten.masked_fill_.Scalar, _masked_fill_)
register_traced_entry(aten.masked_fill.Scalar, _masked_fill)


# ---- clamp / clamp_min / clamp_max with scalar bounds (TensorCompare.cu's entry):
# the structured meta's rules (TensorCompare.cpp): at least one bound, no
# complex, a non-floating self promoted by its bound is a cast and declines


def _clamp_bound(op, self, b):
    if b is None:
        return None
    if isinstance(b, torch.Tensor):
        raise Declined(f"host_trace: {op} with a tensor bound is not traced (declined)")
    b = _plain_scalar(op, b)
    if isinstance(b, complex):
        raise NotImplementedError("clamp is not supported for complex types")
    if not self.dtype.is_floating_point and _promoted(self, b) != self.dtype:
        raise Declined(
            f"host_trace: {op} of a {self.dtype} tensor with bound {b} promotes it, which is not traced (declined)"
        )
    return b


def _clamp(self, min=None, max=None, *, out=None, op=aten.clamp.default):
    if min is None and max is None:
        raise RuntimeError(
            "torch.clamp: At least one of 'min' or 'max' must not be None"
        )
    _cuda_operands(op, self)
    if self.dtype.is_complex:
        raise NotImplementedError("clamp is not supported for complex types")
    lo, hi = _clamp_bound(op, self, min), _clamp_bound(op, self, max)
    return _C._host_trace_ti_clamp(self, lo, hi, out)


def _clamp_min(self, min, *, out=None):
    return _clamp(self, min, None, out=out, op=aten.clamp_min.default)


def _clamp_max(self, max, *, out=None):
    return _clamp(self, None, max, out=out, op=aten.clamp_max.default)


def _relu(self, *, out=None):
    # Activation.cpp relu / relu_: clamp_min(self, 0) after the bool check
    if isinstance(self, torch.Tensor) and self.dtype is torch.bool:
        raise NotImplementedError("Boolean inputs not supported for relu")
    return _clamp(self, 0, None, out=out, op=aten.relu.default)


def _hardtanh(self, min_val=-1, max_val=1, *, out=None):
    # Activation.cpp hardtanh / hardtanh_: clamp(self, min_val, max_val)
    return _clamp(self, min_val, max_val, out=out, op=aten.hardtanh.default)


for _name, _fn in (
    ("clamp", _clamp),
    ("clamp_min", _clamp_min),
    ("clamp_max", _clamp_max),
    ("relu", _relu),
    ("hardtanh", _hardtanh),
):
    register_traced_entry(getattr(aten, _name).default, _fn)
    _op_ = getattr(aten, _name + "_").default
    register_traced_entry(_op_, _inplace(_op_, _fn))


def _to_copy(
    self,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    non_blocking=False,
    memory_format=None,
):
    # at::native::_to_copy for a CUDA tensor changing dtype only: the
    # allocation empty_like makes (preserve_format unless given), then copy_
    # through the sibling (a cast copy). A device, layout or pinning change
    # is a different host and declines.
    _cuda_operands(aten._to_copy.default, self)
    if (
        (device is not None and torch.device(device) != self.device)
        or (layout is not None and layout != self.layout)
        or pin_memory
    ):
        raise Declined(
            "host_trace: _to_copy across devices, layouts or into pinned memory is not traced (declined)"
        )
    mf = torch.preserve_format if memory_format is None else memory_format
    out = torch.empty_like(self, dtype=dtype, memory_format=mf)
    return _C._host_trace_ti_copy_(out, self)


register_traced_entry(aten._to_copy.default, _to_copy)
register_traced_entry(aten.sin.default, _unary(aten.sin.default, _C._host_trace_ti_sin))
register_traced_entry(aten.cos.default, _unary(aten.cos.default, _C._host_trace_ti_cos))
register_traced_entry(aten.exp.default, _unary(aten.exp.default, _C._host_trace_ti_exp))
register_traced_entry(
    aten.rsqrt.default, _unary(aten.rsqrt.default, _C._host_trace_ti_rsqrt)
)
register_traced_entry(aten.neg.default, _unary(aten.neg.default, _C._host_trace_ti_neg))


# ---- softmax / log_softmax (SoftMax.cu, the converted host): the structured
# kernels allocate their output outside the dispatcher, so the entry allocates
# it here (through the trace mode: a traced root) and calls the host with it.
# The output is contiguous with the input's dtype, or float32 for
# half_to_float, as the structured meta sets it.


def _softmax_entry(log_softmax):
    op = aten._log_softmax.default if log_softmax else aten._softmax.default

    def entry(self, dim, half_to_float):
        _cuda_operands(op, self)
        if half_to_float and self.dtype is not torch.float16:
            raise RuntimeError("conversion is supported for Half type only")
        dtype = torch.float32 if half_to_float else self.dtype
        out = torch.empty(self.shape, dtype=dtype, device=self.device)
        return _C._host_trace_softmax_out(
            self, int(dim), bool(half_to_float), log_softmax, out
        )

    return entry


register_traced_entry(aten._softmax.default, _softmax_entry(False))
register_traced_entry(aten._log_softmax.default, _softmax_entry(True))


def _softmax_backward_entry(log_softmax):
    op = (
        aten._log_softmax_backward_data.default
        if log_softmax
        else aten._softmax_backward_data.default
    )

    def entry(grad_output, output, dim, input_dtype):
        # the structured meta: grad_input has grad's sizes, contiguous, grad's
        # dtype, or Half for the (Float grad, Half input) pair
        _cuda_operands(op, grad_output, output)
        half_pair = grad_output.dtype is torch.float32 and input_dtype is torch.float16
        dtype = torch.float16 if half_pair else grad_output.dtype
        out = torch.empty(grad_output.shape, dtype=dtype, device=grad_output.device)
        return _C._host_trace_softmax_backward_out(
            grad_output, output, int(dim), input_dtype, log_softmax, out
        )

    return entry


register_traced_entry(
    aten._softmax_backward_data.default, _softmax_backward_entry(False)
)
register_traced_entry(
    aten._log_softmax_backward_data.default, _softmax_backward_entry(True)
)


# ---- nll_loss forward / backward (Loss.cu, the converted hosts): structured
# kernels whose outputs the entry allocates as the metas (LossNLL.cpp) shape
# them: a {batch} loss for reduction none over a 2-D input, else a scalar;
# total_weight a scalar; grad_input the input's shape, contiguous; all in the
# input's dtype. The host reads no tensor value (the ignored-index count and
# total_weight live on the device).


def _nll_loss_forward(self, target, weight, reduction, ignore_index):
    op = aten.nll_loss_forward.default
    _cuda_operands(op, self, target, *(() if weight is None else (weight,)))
    shape = (self.shape[0],) if reduction == 0 and self.dim() == 2 else ()
    output = torch.empty(shape, dtype=self.dtype, device=self.device)
    total_weight = torch.empty((), dtype=self.dtype, device=self.device)
    return _C._host_trace_nll_loss_forward_out(
        self, target, weight, int(reduction), int(ignore_index), output, total_weight
    )


def _nll_loss_backward(
    grad_output, self, target, weight, reduction, ignore_index, total_weight
):
    op = aten.nll_loss_backward.default
    _cuda_operands(
        op,
        grad_output,
        self,
        target,
        total_weight,
        *(() if weight is None else (weight,)),
    )
    grad_input = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    return _C._host_trace_nll_loss_backward_out(
        grad_output,
        self,
        target,
        weight,
        int(reduction),
        int(ignore_index),
        total_weight,
        grad_input,
    )


register_traced_entry(aten.nll_loss_forward.default, _nll_loss_forward)
register_traced_entry(aten.nll_loss_backward.default, _nll_loss_backward)


# ---- reductions (the entries in ReduceSumProdKernel.cu, ReduceMomentKernel.cu,
# ReduceMaxValuesKernel.cu): one input of one dtype
# reduced into an output of the same dtype; dim=None and dim=[] reduce every
# dim. A dtype argument that differs from the input is the real op's type
# promotion and declines.


def _dims(dim) -> list[int]:
    if dim is None:
        return []
    if isinstance(dim, int):
        return [dim]
    return [int(d) for d in dim]


def _same_dtype(op, self, dtype) -> None:
    if dtype is not None and dtype != self.dtype:
        raise Declined(
            f"host_trace: {op} with dtype={dtype} on a {self.dtype} input promotes, which is not traced (declined)"
        )


def _sum_dim(self, dim=None, keepdim=False, dtype=None):
    _cuda_operands(aten.sum.dim_IntList, self)
    _same_dtype(aten.sum.dim_IntList, self, dtype)
    return _C._host_trace_ti_sum(self, _dims(dim), bool(keepdim))


def _sum(self, dtype=None):
    _cuda_operands(aten.sum.default, self)
    _same_dtype(aten.sum.default, self, dtype)
    return _C._host_trace_ti_sum(self, [], False)


def _sum_out(self, dim=None, keepdim=False, dtype=None, *, out):
    # at::sum_out: the result written into the caller's tensor (flash's
    # backward sums the GQA head groups of dk / dv into the outputs)
    _cuda_operands(aten.sum.IntList_out, self, out)
    _same_dtype(aten.sum.IntList_out, self, dtype)
    return _C._host_trace_ti_sum(self, _dims(dim), bool(keepdim), out)


def _mean_dim(self, dim=None, keepdim=False, dtype=None):
    _cuda_operands(aten.mean.dim, self)
    _same_dtype(aten.mean.dim, self, dtype)
    return _C._host_trace_ti_mean(self, _dims(dim), bool(keepdim))


def _mean(self, dtype=None):
    _cuda_operands(aten.mean.default, self)
    _same_dtype(aten.mean.default, self, dtype)
    return _C._host_trace_ti_mean(self, [], False)


def _amax(self, dim=(), keepdim=False):
    _cuda_operands(aten.amax.default, self)
    return _C._host_trace_ti_amax(self, _dims(dim), bool(keepdim))


def _max(self):
    # max over every dim: max_all_launch_kernel, the same kernel as amax
    _cuda_operands(aten.max.default, self)
    return _C._host_trace_ti_amax(self, [], False)


register_traced_entry(aten.sum.dim_IntList, _sum_dim)
register_traced_entry(aten.sum.default, _sum)
register_traced_entry(aten.sum.IntList_out, _sum_out)
register_traced_entry(aten.mean.dim, _mean_dim)
register_traced_entry(aten.mean.default, _mean)
register_traced_entry(aten.amax.default, _amax)
register_traced_entry(aten.max.default, _max)


# ---- eager hosts with a traced sibling beside the real host (EagerOps.h):
# index_select (Indexing.cu), cat (Shape.cu) and embedding_dense_backward
# (Embedding.cu). embedding's forward is Embedding.cpp's composite over
# index_select, written out here so the mode routes the index_select to its
# sibling.


def _index_select(self, dim, index):
    _cuda_operands(aten.index_select.default, self, index)
    return _C._host_trace_ti_index_select(self, int(dim), index)


def _embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    # at::native::embedding_symint: padding_idx, scale_grad_by_freq and sparse
    # only shape the backward; the forward is an index_select over the rows
    _cuda_operands(aten.embedding.default, weight, indices)
    if indices.dtype not in (torch.long, torch.int):
        raise RuntimeError(
            "Expected tensor for argument #2 'indices' to have one of the following "
            f"scalar types: Long, Int; but got {indices.dtype} instead "
            "(while checking arguments for embedding)"
        )
    if indices.dim() == 1:
        return torch.index_select(weight, 0, indices)
    size = list(indices.shape) + [weight.shape[-1]]
    return torch.index_select(weight, 0, indices.reshape(-1)).view(size)


def _cat(tensors, dim=0):
    for t in tensors:
        _cuda_operands(aten.cat.default, t)
    return _C._host_trace_ti_cat(list(tensors), int(dim))


def _embedding_dense_backward(
    grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
):
    # Embedding.cu embedding_dense_backward_cuda: the zeroed table (a memset
    # record) and the feature kernel when the index count is at most 3072
    # without frequency scaling, the real host's route as a guard; the
    # sort-based route (cub) declines by name in the sibling. num_weights and
    # padding_idx stay values (registered with symint=True): the table's row
    # count is the allocation's size, padding_idx a field of the launch
    _cuda_operands(aten.embedding_dense_backward.default, grad_output, indices)
    return _C._host_trace_ti_embedding_dense_backward(
        grad_output, indices, num_weights, padding_idx, bool(scale_grad_by_freq)
    )


register_traced_entry(aten.index_select.default, _index_select)
register_traced_entry(aten.embedding.default, _embedding)
register_traced_entry(aten.cat.default, _cat)
register_traced_entry(
    aten.embedding_dense_backward.default, _embedding_dense_backward, symint=True
)


# ---- index_copy_ / index_put_ (TensorAdvancedIndexing.cpp; the kernels of
# IndexKernel.cu through the siblings compiled beside them): the StaticCache
# writes k_out.index_copy_(2, cache_position, k) and k_out[:, :, pos] = k. The
# hosts' view arithmetic (the meta's checks, index_copy_out's restrides,
# make_info's broadcast / transpose / restride_src / reshape_indexer) is
# written out here over the traced tensors; the sibling builds the iterator
# and launches. The functional forms fill a fresh result from self first, as
# the real ops do (a memcpy record for a contiguous self).


def _index_copy_checks(self, dim, index, source):
    # TORCH_PRECOMPUTE_META_FUNC(index_copy): the checks in the meta's order
    # with its texts; returns the wrapped dim
    nd = self.dim()
    span = max(nd, 1)
    if dim < -span or dim >= span:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-span}, {span - 1}], but got {dim})"
        )
    dim = dim + nd if dim < 0 else dim
    if index.dim() >= 2:
        raise IndexError(
            f"index_copy_(): Index should have dimension 1 or 0 (got {index.dim()})"
        )
    num = index.numel()
    if source.dim() == 0:
        if not bool(num == 1):
            raise IndexError(
                f"index_copy_(): When source is scalar, index should have one element (got {num})"
            )
    elif source.dim() != nd and nd != 0:
        raise IndexError(
            f"index_copy_(): When source and destination are not scalars, their dimensionality must match. Source dimensionality ({source.dim()}), destination dimensionality ({nd})"
        )
    if index.dtype is not torch.int64:
        raise RuntimeError(
            f"index_copy_(): Expected a long tensor for index, but got {index.dtype}"
        )
    if self.dtype is not source.dtype:
        raise RuntimeError(
            f"index_copy_(): self and source expected to have the same dtype, but got (self) {self.dtype} and (source) {source.dtype}"
        )
    self_sliced = [s for d, s in enumerate(self.shape) if d != dim]
    source_sliced = [s for d, s in enumerate(source.shape) if d != dim]
    if len(self_sliced) != len(source_sliced) or not all(
        bool(a == b) for a, b in zip(self_sliced, source_sliced)
    ):
        raise RuntimeError(
            f"index_copy_(): Source/destination tensor must have same slice shapes. Destination slice shape: {self_sliced} at dimension {dim} and source slice shape: {source_sliced} at dimension 0."
        )
    if source.dim() != 0 and not bool(num == source.shape[dim]):
        raise IndexError(
            f"index_copy_(): Number of indices ({num}) should be equal to source.size(dim) ({source.shape[dim]})"
        )
    return dim


def _index_copy_(self, dim, index, source):
    _cuda_operands(aten.index_copy_.default, self, index, source)
    dim = _index_copy_checks(self, dim, index, source)
    return _C._host_trace_ti_index_copy_(self, dim, index, source)


def _index_copy(self, dim, index, source):
    # the structured kernel's contiguous result, filled from self by copy_,
    # then the kernel on it
    _cuda_operands(aten.index_copy.default, self, index, source)
    dim = _index_copy_checks(self, dim, index, source)
    out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    out.copy_(self)
    return _C._host_trace_ti_index_copy_(out, dim, index, source)


def _expandable_to(shape, desired):
    # ExpandUtils.h is_expandable_to, each size comparison a guard
    if len(shape) > len(desired):
        return False
    lead = len(desired) - len(shape)
    return all(
        bool(sz == 1) or bool(sz == desired[lead + i]) for i, sz in enumerate(shape)
    )


def _index_put_impl(op, self, indices, value, accumulate):
    # _index_put_impl_ and make_info over the traced tensors: bool / byte
    # masks (a nonzero, a synchronizing read), a CPU index or value (copied
    # to the device by the real op) and the sort-based accumulate /
    # deterministic path decline by name
    indices = list(indices)
    if len(indices) > self.dim():
        raise IndexError(
            f"too many indices for tensor of dimension {self.dim()} (got {len(indices)})"
        )
    for idx in indices:
        if idx is None:
            continue
        if idx.dtype in (torch.bool, torch.uint8):
            raise Declined(
                f"host_trace: {op} with a boolean / byte mask index: the mask's nonzero is a synchronizing host read (declined)"
            )
        if idx.dtype not in (torch.int64, torch.int32):
            raise IndexError(
                "tensors used as indices must be long, int, byte or bool tensors"
            )
        _cuda_operands(op, idx)
    # _index_put_impl_'s assert_no_overlap on self against the value and each
    # index, before make_info restrides them (an overlap eager rejects declines
    # by name; between two inputs it is an address guard of the tape)
    _C._host_trace_ti_assert_no_overlap(self, value)
    for idx in indices:
        if idx is not None:
            _C._host_trace_ti_assert_no_overlap(self, idx)
    if accumulate or torch.are_deterministic_algorithms_enabled():
        raise Declined(
            f"host_trace: {op} with accumulate=True or under deterministic algorithms takes the sort-based kernel (index_put_with_sort), which is not traced (declined)"
        )
    defined = [i for i in indices if i is not None]
    if not defined:
        raise Declined(
            f"host_trace: {op} with no index tensor is not traced (declined)"
        )
    if len(defined) > 1:
        defined = list(torch.broadcast_tensors(*defined))
    it = iter(defined)
    indices = [None if i is None else next(it) for i in indices]
    indices += [None] * (self.dim() - len(indices))
    src = self
    # transposeToFront: the defined indices made adjacent at the front
    pos = [d for d, i in enumerate(indices) if i is not None]
    if pos != list(range(pos[0], pos[0] + len(pos))):
        dims = pos + [d for d, i in enumerate(indices) if i is None]
        src = self.permute(dims)
        indices = [indices[d] for d in dims]
    indices = [
        None if i is None else (i if i.dtype is torch.int64 else i.long())
        for i in indices
    ]

    # AdvancedIndex: the indexed dims' sizes and byte strides, self restrided
    # with the broadcast index shape at stride 0 in their place, the indices
    # reshaped to broadcast over it
    element_size = src.element_size()
    src_sizes, src_strides = list(src.shape), list(src.stride())
    dims_before = dims_after = dims_indexed = 0
    replacement: list = []
    indexed_sizes: list = []
    indexed_strides: list = []
    for d, i in enumerate(indices):
        if i is None:
            if dims_indexed == 0:
                dims_before += 1
            else:
                dims_after += 1
        else:
            dims_indexed += 1
            replacement = list(i.shape)
            indexed_sizes.append(src_sizes[d])
            indexed_strides.append(src_strides[d] * element_size)
    if any(bool(s == 0) for s in indexed_sizes) and not any(
        bool(s == 0) for s in replacement
    ):
        raise IndexError("index is out of bounds for dimension with size 0")
    end = dims_before + dims_indexed
    src_r = src.as_strided(
        src_sizes[:dims_before] + replacement + src_sizes[end:],
        src_strides[:dims_before] + [0] * len(replacement) + src_strides[end:],
    )
    idx_r = [
        i.reshape([1] * dims_before + list(i.shape) + [1] * dims_after)
        for i in indices
        if i is not None
    ]
    if len(idx_r) >= 2 and not all(
        all(bool(a == b) for a, b in zip(x.stride(), idx_r[0].stride()))
        for x in idx_r[1:]
    ):
        idx_r = [i.contiguous() for i in idx_r]
    # make_index_put_iterator's checks
    if not _expandable_to(list(value.shape), list(src_r.shape)):
        raise RuntimeError(
            f"shape mismatch: value tensor of shape {tuple(value.shape)} cannot be broadcast to indexing result of shape {tuple(src_r.shape)}"
        )
    if value.dtype is not src_r.dtype:
        raise RuntimeError(
            f"Index put requires the source and destination dtypes match, got {src_r.dtype} for the destination and {value.dtype} for the source."
        )
    _C._host_trace_ti_index_put_(src_r, value, idx_r, indexed_sizes, indexed_strides)


def _index_put_(self, indices, values, accumulate=False):
    _cuda_operands(aten.index_put_.default, self, values)
    _index_put_impl(aten.index_put_.default, self, indices, values, accumulate)
    return self


def _index_put(self, indices, values, accumulate=False):
    # TensorAdvancedIndexing.cpp index_put: a preserve-format clone, written
    # in place
    _cuda_operands(aten.index_put.default, self, values)
    out = self.clone(memory_format=torch.preserve_format)
    _index_put_impl(aten.index_put.default, out, indices, values, accumulate)
    return out


register_traced_entry(aten.index_copy_.default, _index_copy_)
register_traced_entry(aten.index_copy.default, _index_copy)
register_traced_entry(aten.index_put_.default, _index_put_)
register_traced_entry(aten.index_put.default, _index_put)


# ---- triu / tril (TriangularOps.cu, the sibling appended to the real host):
# the structured meta's contiguous result (self for the in-place ops), the
# diagonal a value of the launch (a SymInt is not pinned: _SYMINT_KERNELS)


def _triu_tril(op, upper, inplace):
    def entry(self, diagonal=0):
        _cuda_operands(op, self)
        out = (
            self
            if inplace
            else torch.empty(self.shape, dtype=self.dtype, device=self.device)
        )
        return _C._host_trace_ti_triu_tril(self, diagonal, upper, out)

    return entry


register_traced_entry(aten.triu.default, _triu_tril(aten.triu.default, True, False))
register_traced_entry(aten.tril.default, _triu_tril(aten.tril.default, False, False))
register_traced_entry(aten.triu_.default, _triu_tril(aten.triu_.default, True, True))
register_traced_entry(aten.tril_.default, _triu_tril(aten.tril_.default, False, True))


# ---- the K = 1 bmm: torch._native's eager override of aten::bmm on CUDA
# (ops/bmm_outer_product) runs a Triton outer-product kernel for a
# (B, M, 1) x (B, 1, N) product, one multiply per element in the inputs'
# dtype (fp32 opmath for the half types, as ATen's mul), so the sibling is
# the broadcast multiply: bitwise the eager result. The trace routes bmm
# here only when the override's condition holds (_host_trace._outer_product_bmm);
# every other bmm is a closed cuBLAS region.


def _bmm_outer_product(a, b):
    _cuda_operands(aten.bmm.default, a, b)
    if a.dtype is not b.dtype:
        raise RuntimeError(f"expected scalar type {a.dtype} but found {b.dtype}")
    return _C._host_trace_ti_mul(a, b)


register_traced_entry(aten.bmm.default, _bmm_outer_product)


# ---- the generated siblings (torchgen/dest/ufunc.py over ti/siblings.yaml
# and native_functions.yaml's ufunc_inner_loop; HostTraceSiblingBindings.h):
# one generic entry per functional overload, bound to the op's schema, and the
# same entry behind the in-place form (self is the destination), the out=
# form and the .Scalar twins. An op with a hand entry keeps it for the whole
# group (add: its alpha checks); its generated binding stays available for the
# parity test. A spec marked `entry: hand` names its hand entry below.


def _scalar_value(op, v):
    # a non-tensor argument of the op as the binding takes it: a Python number
    # (a symbolic one pinned, as eager's Scalar pins it) or a string selector;
    # a tensor in a Scalar position declines
    if isinstance(v, torch.Tensor):
        raise Declined(f"host_trace: {op} with a tensor value is not traced (declined)")
    s = _scalar_operand(op, v)
    return v if s is None else s


def _generated_entry(op, binding, ntensors):
    # the op's arguments in schema order (the mode passes kwarg-only ones by
    # name and omits defaults): the tensors, then the values, then `out`
    schema = op._schema
    names = [a.name for a in schema.arguments if not a.is_out]
    defaults = {
        a.name: a.default_value for a in schema.arguments if a.has_default_value()
    }

    def entry(*args, out=None, **kwargs):
        bound = dict(zip(names, args))
        bound.update(kwargs)
        values = [bound[n] if n in bound else defaults[n] for n in names]
        tensors = _operands(op, *values[:ntensors])
        scalars = [_scalar_value(op, v) for v in values[ntensors:]]
        return binding(*tensors, *scalars, out)

    return entry


def _out_entry(op, fn):
    # the out= overload names its destination as the schema does (`out`,
    # `grad_input`); the entry receives it as a keyword
    (destination,) = [a.name for a in op._schema.arguments if a.is_out]

    def entry(*args, **kwargs):
        out = kwargs.pop(destination)
        _cuda_operands(op, out)
        return fn(*args, out=out, **kwargs)

    return entry


def _overload(name):
    base, _, overload = name.partition(".")
    return getattr(getattr(aten, base), overload or "default")


# the hand entries of specs marked `entry: hand` (the op's own checks and the
# host work around the generated launch)


def _native_dropout_backward(grad_output, mask, scale):
    # Dropout.cu native_dropout_backward_cuda's mask check (its text spells
    # the dtype as c10::ScalarType prints it)
    _cuda_operands(aten.native_dropout_backward.default, grad_output, mask)
    if mask.dtype is not torch.bool:
        name = str(mask.dtype).removeprefix("torch.")
        name = {
            "float32": "Float",
            "float64": "Double",
            "float16": "Half",
            "int64": "Long",
            "int32": "Int",
            "int16": "Short",
            "int8": "Char",
            "uint8": "Byte",
            "bfloat16": "BFloat16",
        }.get(name, name)
        raise RuntimeError(f"Mask should be Bool Scalar Type{name}")
    return _C._host_trace_ti_gen_native_dropout_backward(
        grad_output, mask, float(scale), None
    )


def _mse_loss(self, target, reduction=1):
    # Loss.cpp mse_loss_out: the elementwise kernel, then mean / sum over
    # every dim for the reduced forms (the reduction siblings)
    _cuda_operands(aten.mse_loss.default, self, target)
    a, b = _binary_operands(aten.mse_loss.default, self, target)
    loss = _C._host_trace_ti_gen_mse_loss(a, b, None)
    if reduction == 0:
        return loss
    return torch.mean(loss) if reduction == 1 else torch.sum(loss)


_HAND_GENERATED = {
    "native_dropout_backward": _native_dropout_backward,
    "mse_loss": _mse_loss,
}


for _name, _binding, _overloads, _ntensors, _hand in _C._host_trace_ti_gen_siblings():
    _op = _overload(_name)
    if _hand:
        register_traced_entry(_op, _HAND_GENERATED[_name])
        continue
    if _op in _TRACED_ENTRIES:
        continue
    _entry = _generated_entry(_op, getattr(_C, _binding), _ntensors)
    register_traced_entry(_op, _entry)
    for _other in _overloads:
        _op2 = _overload(_other)
        if _op2._schema.arguments[0].alias_info is not None:
            register_traced_entry(_op2, _inplace(_op2, _entry))
        elif any(a.is_out for a in _op2._schema.arguments):
            register_traced_entry(_op2, _out_entry(_op2, _entry))
        else:
            register_traced_entry(_op2, _entry)


# ---- foreach / fused optimizers (ti/ForeachOps.h): the multi_tensor_apply
# host over symbolic numels. The lists are CUDA tensors of one dtype per list;
# the scalars (the .Scalar value, alpha, the hyperparameters) are constants of
# the tape as eager bakes them into the launch. A fast-route refusal takes the
# op's own slow path (the per-tensor op, traced through its sibling) inside
# the entry, as eager does.


def _foreach_add_scalar_(self, scalar):
    op = aten._foreach_add_.Scalar
    _cuda_operands(op, *self)
    _C._host_trace_foreach_add_scalar_(list(self), _plain_scalar(op, scalar))


def _foreach_add_list_(self, other, alpha=1):
    op = aten._foreach_add_.List
    _cuda_operands(op, *self, *other)
    _C._host_trace_foreach_add_list_(list(self), list(other), _plain_scalar(op, alpha))


def _fused_adamw_(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale=None,
    found_inf=None,
):
    # the tensor_lr overload reads a CPU lr on the host (lr.item()) at the
    # trace and never again: declined by name; a CUDA lr is a pointer of the
    # launch and `lr` the unused 1.0 the eager host passes
    lr_tensor = None
    if isinstance(lr, torch.Tensor):
        op = aten._fused_adamw_.tensor_lr
        if not lr.is_cuda:
            raise Declined(
                f"host_trace: {op} with a {lr.device} tensor lr: read on the host at the trace and never again; pass a float or a CUDA tensor (declined)"
            )
        lr_tensor, lr = lr, 1.0
    else:
        op = aten._fused_adamw_.default
    optional = [t for t in (grad_scale, found_inf) if t is not None]
    _cuda_operands(
        op,
        *self,
        *grads,
        *exp_avgs,
        *exp_avg_sqs,
        *max_exp_avg_sqs,
        *state_steps,
        *optional,
    )
    _C._host_trace_fused_adamw_(
        list(self),
        list(grads),
        list(exp_avgs),
        list(exp_avg_sqs),
        list(max_exp_avg_sqs),
        list(state_steps),
        lr_tensor,
        float(lr),
        float(beta1),
        float(beta2),
        float(weight_decay),
        float(eps),
        bool(amsgrad),
        bool(maximize),
        grad_scale,
        found_inf,
    )


register_traced_entry(aten._foreach_add_.Scalar, _foreach_add_scalar_)
register_traced_entry(aten._foreach_add_.List, _foreach_add_list_)
register_traced_entry(aten._fused_adamw_.default, _fused_adamw_)
register_traced_entry(aten._fused_adamw_.tensor_lr, _fused_adamw_)
