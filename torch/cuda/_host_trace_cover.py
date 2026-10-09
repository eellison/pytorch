"""Host tracing (private): proving a HostTraceReplay covers a box of argument sizes, solved, not swept.

The entry's arguments are affine in the box's axes (ArgsAt: T, or (nd, T)). Each traced variant's guards are
sympy over its symbols; a symbol read from the entry's arguments (sizes and int arguments affine in the axes, a
changed tensor fresh and contiguous) becomes a polynomial in them, every other symbol (addresses, allocation
bases) its traced hint. Per guard, with M the lcm of the floor divisors on T, each residue class T = M k + r turns
the floors into polynomials in k; a guard's truth then changes only at the real roots of its critical
polynomials (relation sides, Min/Max and Piecewise branches), so it is evaluated at those roots' neighbours and
once between them. A guard this cannot reduce (or whose M exceeds the range) is evaluated at every T and counted
as a fallback: vectorized over T in exact integer arithmetic, else sympy at each T. `cover` calls the entry where the solver finds a point it does not hold, until none is left.
"""

import collections
import functools
import hashlib
import itertools
import math
import operator
import re
import time
from fractions import Fraction

import numpy as np
import sympy

import torch
import torch.utils._sympy.functions as F
from torch.cuda import _host_trace_ir as _ir

_SIZE = re.compile(r"arg(\d+)\.size\((\d+)\)$")
_STRIDE = re.compile(r"arg(\d+)\.stride\((\d+)\)$")
_OFFSET = re.compile(r"arg(\d+)\.storage_offset\(\)$")
_INT = re.compile(r"arg(\d+)$")
T = sympy.Symbol("T", integer=True, positive=True)
ND = sympy.Symbol("nd", integer=True, nonnegative=True)
# the symbols of ArgsAt's axes by its number of axes, unless named (decode: ("bs", "S"))
AXES = {1: (T,), 2: (ND, T)}
_K = sympy.Symbol("k", integer=True)
# an axis up to this long is evaluated vectorized first (exact as evaluate; about 7 ms per guard over 8192 T against
# up to 0.3 s for the critical points of a CuTe tile-count guard); 0: the solver first
vector_first = 16384
# a guard is evaluated at the box's points as data: an IR trace's from its record (no sympy export), a sympy one
# (an entry's) at its symbols' values; False: substituted (xreplace) and solved per guard
ir_rows = True


def axis(name):
    return {"T": T, "nd": ND}.get(name) or sympy.Symbol(name, integer=True, nonnegative=True)


class _Fallback(Exception):
    pass


def _sympy(e):
    """e rebuilt with sympy's evaluating constructors, torch's integer functions as floor arithmetic."""
    if not e.args:
        return e
    a = [_sympy(x) for x in e.args]
    f = e.func
    if f is F.FloorDiv:
        return sympy.floor(a[0] / a[1])
    if f is F.CeilDiv:
        return sympy.ceiling(a[0] / a[1])
    if f in (F.Mod, F.PythonMod):
        return a[0] - a[1] * sympy.floor(a[0] / a[1])
    if f is F.ModularIndexing:
        q = sympy.floor(a[0] / a[1])
        return q - a[2] * sympy.floor(q / a[2])
    if f in (F.CleanDiv, F.IntTrueDiv):
        return a[0] / a[1]
    if f is F.PowByNatural:
        return a[0] ** a[1]
    if f is F.Min:
        return sympy.Min(*a)
    if f is F.Max:
        return sympy.Max(*a)
    if f is F.Where:
        # an undecided select (a CuTe integer if/else) is evaluated vectorized: sympy's Piecewise constructor
        # simplifies its conditions as sets, minutes per guard
        if a[0] not in (sympy.true, sympy.false):
            raise _Fallback("Where")
        return a[1] if a[0] is sympy.true else a[2]
    if f is F.Identity:
        return a[0]
    if f in (F.FloorToInt, F.TruncToInt):
        return sympy.floor(a[0])
    if f is F.CeilToInt:
        return sympy.ceiling(a[0])
    if isinstance(e, sympy.Function) and f.__module__.startswith("torch"):
        raise _Fallback(f"{f.__name__}")
    return f(*a)


def _real_roots(p, lo, hi):
    """The real roots of the polynomial p in T inside [lo, hi] (floats)."""
    poly = sympy.Poly(p, T)
    if poly.degree() < 1:
        return []
    return [float(sympy.re(x)) for x in poly.nroots(n=15) if abs(sympy.im(x)) < 1e-9 and lo <= float(sympy.re(x)) <= hi]


def _const_floors(e, lo, hi):
    """e with each floor/ceiling of a rational function of T that has no pole in [lo, hi] and keeps one value
    there replaced by that value (Mod(T - 1, T)'s floor((T - 1)/T) is 0): the function's range on [lo, hi] lies
    between its values at the ends and at its derivative's real roots."""

    def const(x):
        arg = x.args[0]
        if arg.free_symbols != {T} or arg.has(sympy.floor, sympy.ceiling):
            return x
        num, den = sympy.fraction(sympy.together(arg))
        if not den.has(T) or _real_roots(den, lo - 1e-9, hi + 1e-9):
            return x
        f = math.floor if isinstance(x, sympy.floor) else math.ceil
        vals = [Fraction(int(v.p), int(v.q)) for v in (arg.xreplace({T: sympy.Integer(lo)}), arg.xreplace({T: sympy.Integer(hi)}))]
        dnum, _ = sympy.fraction(sympy.together(sympy.diff(arg, T)))
        for c in _real_roots(sympy.expand(dnum), lo, hi):
            v = float(arg.xreplace({T: sympy.Float(c, 30)}))
            vals += [Fraction(v) - Fraction(1, 10**9), Fraction(v) + Fraction(1, 10**9)]
        a, b = f(min(vals)), f(max(vals))
        return sympy.Integer(a) if a == b else x

    return e.replace(lambda x: isinstance(x, (sympy.floor, sympy.ceiling)), const)


def _modulus(e):
    """The lcm of the denominators of T's coefficients in e's floor arguments (nested floors multiply)."""
    m = 1
    for fl in e.atoms(sympy.floor, sympy.ceiling):
        arg = fl.args[0]
        inner = _modulus(arg) if arg.has(sympy.floor, sympy.ceiling) else 1
        p = sympy.Poly(arg.replace(lambda x: isinstance(x, (sympy.floor, sympy.ceiling)), lambda x: sympy.Integer(0)), T)
        d = math.lcm(1, *(sympy.Rational(c).q for c in p.coeffs()))
        m = math.lcm(m, inner * d)
    return m


def _branches(e, depth=0):
    """e with its Min/Max/Piecewise resolved to each branch combination (critical differences on the side)."""
    nodes = [x for x in sympy.preorder_traversal(e) if isinstance(x, (sympy.Min, sympy.Max, sympy.Piecewise))]
    if not nodes:
        return [e], []
    if depth > 6:
        raise _Fallback("Min/Max nesting")
    n = nodes[0]
    crit = []
    if isinstance(n, sympy.Piecewise):
        options = [b for b, _ in n.args]
        for _, c in n.args:
            if isinstance(c, sympy.core.relational.Relational):
                crit.append(c.lhs - c.rhs)
    else:
        options = list(n.args)
        crit += [x - y for x, y in itertools.combinations(options, 2)]
    out = []
    for o in options:
        got, more = _branches(e.xreplace({n: o}), depth + 1)
        out += got
        crit += more
    if len(out) > 64:
        raise _Fallback("Min/Max branches")
    return out, crit


def _critical(g):
    """Polynomials in k whose sign changes are the only places g's truth can change."""
    crit = []
    rels = [x for x in sympy.preorder_traversal(g) if isinstance(x, sympy.core.relational.Relational)]
    if not rels and g not in (sympy.true, sympy.false):
        rels = [sympy.Ne(g, 0)]
    for r in rels:
        got, more = _branches(r.lhs - r.rhs)
        crit += got + more
    return crit


def _roots(p):
    p = sympy.expand(p)
    if not p.has(_K):
        return []
    if p.has(sympy.floor, sympy.ceiling, sympy.Min, sympy.Max, sympy.Piecewise, sympy.Abs):
        raise _Fallback(f"not polynomial in k: {p}")
    poly = sympy.Poly(p, _K)
    if poly.degree() == 1:
        a, b = poly.all_coeffs()
        return [float(-b / a)]
    return [float(sympy.re(x)) for x in poly.nroots(n=15) if abs(sympy.im(x)) < 1e-9]


def _truth(g, v):
    try:
        r = g.xreplace({T: sympy.Integer(v)})
    except ZeroDivisionError:
        return False  # undefined at v: the integer program's status fails there, so evaluate misses
    if r not in (sympy.true, sympy.false):
        r = sympy.simplify(r)
    if r not in (sympy.true, sympy.false):
        raise _Fallback(f"undecided at T={v}: {r}")
    return r is sympy.true


def _solve_guard(g, lo, hi, stats):
    """Boolean array of g's truth over T in [lo, hi]."""
    n = hi - lo + 1
    m = _modulus(g)
    if m > n // 4:
        raise _Fallback(f"modulus {m}")
    out = np.zeros(n, dtype=bool)
    for r in range(m):
        kmin, kmax = -(-(lo - r) // m), (hi - r) // m
        if kmin > kmax:
            continue
        gk = g.xreplace({T: m * _K + r})
        if gk.has(sympy.floor, sympy.ceiling):
            gk = gk.replace(lambda x: isinstance(x, (sympy.floor, sympy.ceiling)), lambda x: x.func(sympy.expand(x.args[0])))
        if gk.has(sympy.floor, sympy.ceiling):
            raise _Fallback("floor left after the residue substitution")
        points = {kmin, kmax}
        for p in _critical(gk):
            for x in _roots(p):
                if kmin - 1 <= x <= kmax + 1:
                    points.update(k for k in (math.floor(x), math.ceil(x)) if kmin <= k <= kmax)
        points = sorted(points)
        # truth is constant on each open gap between consecutive points
        prev = None
        for k in points:
            if prev is not None and k > prev + 1:
                stats["evals"] += 1
                v = _truth(g, m * (prev + 1) + r)
                out[m * (prev + 1) + r - lo : m * (k - 1) + r - lo + 1 : m] = v
            stats["evals"] += 1
            out[m * k + r - lo] = _truth(g, m * k + r)
            prev = k
    return out


def _map(fn):
    ufunc = np.frompyfunc(fn, 1, 1)
    return lambda x: ufunc(x) if isinstance(x, np.ndarray) else fn(x)


def _fold(op):
    return lambda *a: functools.reduce(op, a)


def _pow(b, x):
    return Fraction(1) / b ** -x if isinstance(x, int) and x < 0 else b ** x


# torch's integer functions by Python's integer operators (sizes are nonnegative, so Mod is PythonMod)
_VEC = {
    sympy.Add: _fold(operator.add), sympy.Mul: _fold(operator.mul), sympy.Pow: _pow,
    F.FloorDiv: operator.floordiv, F.CleanDiv: operator.floordiv, F.CeilDiv: lambda x, y: -(-x // y),
    F.Mod: operator.mod, F.PythonMod: operator.mod, F.ModularIndexing: lambda x, d, m: x // d % m,
    F.IntTrueDiv: operator.truediv, F.FloatTrueDiv: operator.truediv, F.PowByNatural: operator.pow, F.FloatPow: operator.pow,
    F.Min: _fold(np.minimum), F.Max: _fold(np.maximum), sympy.Min: _fold(np.minimum), sympy.Max: _fold(np.maximum),
    F.Where: lambda c, x, y: np.where(np.asarray(c, dtype=bool), x, y), F.Identity: lambda x: x,
    F.LShift: operator.lshift, F.RShift: operator.rshift,
    F.FloorToInt: _map(math.floor), F.CeilToInt: _map(math.ceil), F.TruncToInt: _map(math.trunc), F.RoundToInt: _map(round),
    sympy.floor: _map(math.floor), sympy.ceiling: _map(math.ceil), F.ToFloat: _map(float), F.TruncToFloat: _map(lambda v: float(math.trunc(v))),
    sympy.Eq: operator.eq, sympy.Ne: operator.ne, sympy.Lt: operator.lt, sympy.Le: operator.le, sympy.Gt: operator.gt, sympy.Ge: operator.ge,
    sympy.And: _fold(np.logical_and), sympy.Or: _fold(np.logical_or), sympy.Not: np.logical_not,
}
_BITWISE = {"bitwise_and": operator.and_, "bitwise_or": operator.or_, "bitwise_xor": operator.xor}

# The same over int64 arrays, exact while every value stays below 2**62: the ops that can grow a value check
# their operands' bounds first (OverflowError: evaluate over Python ints instead); no division by zero or float.
_LIMIT = 1 << 62


def _bound(*xs):
    return [int(np.abs(x).max(initial=0)) if isinstance(x, np.ndarray) else abs(int(x)) for x in xs]


def _grows(op, bound):
    def f(*a):
        if bound(*_bound(*a)) >= _LIMIT:
            raise OverflowError("int64")
        return op(*a)
    return f


def _divides(op):
    def f(x, *d):
        if any((np.asarray(y) == 0).any() for y in d):
            raise ZeroDivisionError("int64")
        return op(x, *d)
    return f


def _natural(op):
    def f(b, x):
        if (np.asarray(x) < 0).any():
            raise ValueError("negative exponent")
        return op(b, x)
    return f


def _pow_bound(b, x):
    return 1 if b <= 1 else _LIMIT if x >= 62 else b ** x


def _bit_length(v):
    if not isinstance(v, np.ndarray):
        return abs(int(v)).bit_length()
    if _bound(v)[0] >= 1 << 53:
        raise OverflowError("int64")
    return np.frexp(np.abs(v).astype(np.float64))[1].astype(np.int64)


def _identity(x):
    return x


_KEEP = (F.Min, F.Max, sympy.Min, sympy.Max, F.Where, F.Identity, sympy.And, sympy.Or, sympy.Not)
_VEC64 = {
    **{k: v for k, v in _VEC.items() if k in _KEEP or issubclass(k, sympy.core.relational.Relational)},
    sympy.Add: _grows(_fold(operator.add), lambda *b: sum(b)), sympy.Mul: _grows(_fold(operator.mul), lambda *b: math.prod(b)),
    sympy.Pow: _grows(_natural(operator.pow), _pow_bound), F.PowByNatural: _grows(_natural(operator.pow), _pow_bound),
    F.LShift: _grows(_natural(operator.lshift), lambda b, x: b << min(x, 62)), F.RShift: _natural(operator.rshift),
    F.FloorDiv: _divides(operator.floordiv), F.CleanDiv: _divides(operator.floordiv), F.CeilDiv: _divides(lambda x, y: -(-x // y)),
    F.Mod: _divides(operator.mod), F.PythonMod: _divides(operator.mod), F.ModularIndexing: _divides(lambda x, d, m: x // d % m),
    F.FloorToInt: _identity, F.CeilToInt: _identity, F.TruncToInt: _identity, F.RoundToInt: _identity, sympy.floor: _identity, sympy.ceiling: _identity,
}


def _elementwise(func, n):
    """A function class that evaluates itself at integers (the host trace's BitLength, F32Div), per element."""
    def one(*v):
        r = func(*(sympy.Integer(x) if isinstance(x, int) else x for x in v))
        if not r.is_Integer:
            raise _Fallback(f"{func.__name__} undecided at {v}")
        return int(r)
    ufunc = np.frompyfunc(one, n, 1)
    return lambda *a: ufunc(*a) if any(isinstance(x, np.ndarray) for x in a) else one(*a)


def _vector(e, ts, memo=None):
    """e at each T of ts (an object array of Python ints), or at {axis: array} broadcast together: exact, as the
    replay's evaluate computes it. memo maps subexpressions already evaluated at ts (guards repeat whole subtrees).
    At int64 arrays by _VEC64, raising where a value could leave int64 or is not an integer."""
    i64 = (next(iter(ts.values())) if isinstance(ts, dict) else ts).dtype == np.int64
    if isinstance(e, sympy.logic.boolalg.BooleanAtom):
        return bool(e)
    if e.is_Integer:
        if i64 and abs(int(e)) >= _LIMIT:
            raise OverflowError("int64")
        return int(e)
    if i64 and e.is_number:
        raise _Fallback(f"{e} in int64")
    if e.is_Rational:
        return Fraction(int(e.p), int(e.q))
    if e.is_Float:
        return float(e)
    if isinstance(ts, dict):
        if e in ts:
            return ts[e]
    elif e == T:
        return ts
    if memo is None:
        memo = {}
    elif e in memo:
        return memo[e]
    if isinstance(e, sympy.Piecewise):
        out = None
        for x, c in reversed(e.args):
            v = _vector(x, ts, memo)
            out = v if out is None else np.where(np.asarray(_vector(c, ts, memo), dtype=bool), v, out)
        memo[e] = out
        return out
    if not e.args:
        raise _Fallback(f"symbol {e}")
    name = getattr(e.func, "__name__", "")
    op = (_VEC64 if i64 else _VEC).get(e.func)
    if op is None and name.startswith("BitwiseFn_"):
        op = _BITWISE.get(name[len("BitwiseFn_"):])
    if op is None and name == "BitLength":
        op = _bit_length if i64 else _map(lambda v: abs(v).bit_length())
    if op is None and i64:
        raise _Fallback(f"no int64 form for {name}")
    if op is None and name.startswith("OpaqueUnaryFn_"):
        op = _map(getattr(math, name[len("OpaqueUnaryFn_"):]))
    if op is None and isinstance(e, sympy.Function) and "eval" in vars(e.func):
        op = _elementwise(e.func, len(e.args))
    if op is None:
        raise _Fallback(f"no vector form for {name}")
    out = memo[e] = op(*(_vector(x, ts, memo) for x in e.args))
    return out


def _exact(e, arrays):
    """e by _vector at int64 arrays where that is exact, else at Python ints (arrays(dtype) builds the points)."""
    try:
        return _vector(e, arrays(np.int64))
    except (_Fallback, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return _vector(e, arrays(object))


_EVAL_ERRORS = (_Fallback, TypeError, ValueError, ZeroDivisionError, OverflowError, KeyError)
_REL = {"Eq": operator.eq, "Ne": operator.ne, "Lt": operator.lt, "Le": operator.le, "Gt": operator.gt, "Ge": operator.ge}
_IR_BITWISE = {"bitand": operator.and_, "bitor": operator.or_, "bitxor": operator.xor}
# an IR value up to this many points is kept for the variant's other guards (they repeat subtrees); a box-sized one
# is recomputed
_MEMO_SIZE = 1 << 16
_DECLINED = object()


def _ir_vector(n, leaves, memo, i64, large):
    """The IR node n at the box's points, as _vector evaluates its sympy export: leaves maps a symbol's name to its
    value there (an array broadcast over the box, int64 where i64, else of Python ints; or a Python int); memo and
    large hold the nodes already evaluated (by id), large the values over _MEMO_SIZE points, kept for one guard
    only. Raises as _vector does where int64 is not exact; _Fallback on the float lane."""
    out = memo.get(n.id, large.get(n.id))
    if out is not None:
        return out
    op, a = n.op, n.args

    def ev(x):
        return _ir_vector(x, leaves, memo, i64, large)

    if op == "const":
        if i64 and abs(a[0]) >= _LIMIT:
            raise OverflowError("int64")
        out = a[0]
    elif op == "sym":
        out = leaves[a[0]]
    elif op == "add":
        vs = [ev(t) for t, _ in a[1]]
        if i64 and abs(a[0]) + sum(abs(c) * b for (_, c), b in zip(a[1], _bound(*vs))) >= _LIMIT:
            raise OverflowError("int64")
        out = a[0]
        for (_, c), v in zip(a[1], vs):
            out = out + c * v
    elif op == "mul":
        if any(e < 0 for _, e in a[1]):
            raise _Fallback("a negative power")
        vs = [ev(f) for f, _ in a[1]]
        if i64:
            b = abs(a[0])
            for x, (_, e) in zip(_bound(*vs), a[1]):
                b = min(b * _pow_bound(max(x, 1), e), _LIMIT)
            if b >= _LIMIT:
                raise OverflowError("int64")
        out = a[0]
        for v, (_, e) in zip(vs, a[1]):
            out = out * v**e
    elif op in ("floordiv", "ceildiv", "mod"):
        x, y = ev(a[0]), ev(a[1])
        if (np.asarray(y) == 0).any():
            raise ZeroDivisionError(op)
        out = x // y if op == "floordiv" else -(-x // y) if op == "ceildiv" else x % y
    elif op in ("min", "max"):
        vs = [ev(x) for x in a]
        if any(isinstance(v, np.ndarray) for v in vs):
            out = functools.reduce(np.minimum if op == "min" else np.maximum, vs)
        else:
            out = min(vs) if op == "min" else max(vs)
    elif op == "nod":
        vs = [ev(x) for x in a]
        k = len(vs) // 2

        def one(*v):
            return int(_ir._dense(list(v[:k]), list(v[k:])))

        if any(isinstance(v, np.ndarray) for v in vs):
            out = np.frompyfunc(one, len(vs), 1)(*vs)
            out = out.astype(np.int64) if i64 else out
        else:
            out = one(*vs)
    elif op == "pbn":
        b, x = ev(a[0]), ev(a[1])
        if (np.asarray(x) < 0).any():
            raise ValueError("negative exponent")
        if i64 and _pow_bound(*_bound(b, x)) >= _LIMIT:
            raise OverflowError("int64")
        out = b**x
    elif op == "bitlen":
        v = ev(a[0])
        out = _bit_length(v) if i64 or not isinstance(v, np.ndarray) else _map(lambda u: abs(u).bit_length())(v)
    elif op in _IR_BITWISE:
        out = _IR_BITWISE[op](ev(a[0]), ev(a[1]))
    elif op == "where":
        c = ev(a[0])
        if not isinstance(c, np.ndarray):
            out = ev(a[1]) if c else ev(a[2])
        else:
            x, y = ev(a[1]), ev(a[2])
            out = np.where(c, x, y) if i64 else np.where(c, np.asarray(x, dtype=object), np.asarray(y, dtype=object))
    elif op in ("eq", "ne", "lt", "le"):
        out = _REL[op.capitalize()](ev(a[0]), 0)
    elif op in ("true", "false"):
        out = op == "true"
    elif op in ("and", "or"):
        vs = [ev(x) for x in a]
        if any(isinstance(v, np.ndarray) for v in vs):
            out = functools.reduce(np.logical_and if op == "and" else np.logical_or, vs)
        else:
            out = all(vs) if op == "and" else any(vs)
    elif op == "not":
        v = ev(a[0])
        out = np.logical_not(v) if isinstance(v, np.ndarray) else not v
    else:
        # the float lane and f32div: the guard's sympy export
        raise _Fallback(f"no vector form for the IR kind {op}")
    (memo if not isinstance(out, np.ndarray) or out.size <= _MEMO_SIZE else large)[n.id] = out
    return out


def _ir_axes(n, masks, memo):
    """The bitmask of the box's axes the IR node n reads, by its symbols' (masks: name -> mask)."""
    out = memo.get(n.id)
    if out is None:
        out = masks.get(n.args[0], 0) if n.op == "sym" else 0
        for x in _ir.children(n):
            out |= _ir_axes(x, masks, memo)
        memo[n.id] = out
    return out


def _ir_key(x, leaves, memo):
    """A hashable form of the IR node x with each symbol replaced by its substitution (leaves: name -> key):
    equal keys across traces evaluate equally over the box."""
    if not isinstance(x, _ir.Node):
        return tuple(_ir_key(a, leaves, memo) for a in x) if isinstance(x, tuple) else (type(x), x)
    out = memo.get(x.id)
    if out is None:
        if x.op in ("sym", "fsym"):
            out = leaves.get(x.args[0], ("free", x.args[0]))
        elif x.op in ("add", "mul"):
            out = (x.op, (type(x.args[0]), x.args[0]), tuple((_ir_key(t, leaves, memo), (type(c), c)) for t, c in x.args[1]))
        else:
            out = (x.op, _ir_key(x.args, leaves, memo))
        memo[x.id] = out
    return out


class _Leaves:
    """A variant's symbols at the box's points: per name the axes it reads (a bitmask) and its value over the box
    per dtype, each axis broadcast along its own dim (a symbol on no axis: a Python int), and along the (nd, T)
    sum for the symbols that read the axes only through it. `sympy(where, dtype)` keys the same by symbol for
    _vector (every value an array)."""

    def __init__(self, cover, subs, dep):
        axes, nd = cover.axes, len(cover.axes)
        dims = [[-1 if j == k else 1 for j in range(nd)] for k in range(nd)]
        grid = {d: {x: np.arange(a, b + 1).astype(d).reshape(s) for x, a, b, s in zip(axes, cover.lo, cover.hi, dims)} for d in (np.int64, object)}
        summed = axes == (ND, T)
        line = {d: np.arange(cover.lo[0] + cover.lo[1], cover.hi[0] + cover.hi[1] + 1).astype(d) for d in (np.int64, object)} if summed else None
        self.nd, self.symbols = nd, subs
        self.masks, self.axes_memo, self.key_memo = {}, {}, {}
        self.keys = {s.name: (type(e), e) for s, e in subs.items()}
        self.values = {"box": {np.int64: {}, object: {}}, "line": {np.int64: {}, object: {}}}
        for s, e in subs.items():
            if s in dep:
                self.masks[s.name] = sum(1 << k for k, x in enumerate(axes) if e.has(x))
                u = sympy.expand(e.xreplace({T: T - ND})) if summed else None
                for d in (np.int64, object):
                    for where, ts, x in (("box", grid[d], e), ("line", line and line[d], u)):
                        if x is None or (where == "line" and x.has(ND)):
                            continue
                        try:
                            self.values[where][d][s.name] = _vector(x, ts)
                        except (_Fallback, TypeError, ValueError, ZeroDivisionError, OverflowError):
                            pass
            else:
                v = int(e) if getattr(e, "is_Integer", False) else float(e) if getattr(e, "is_Float", False) else e
                for where in ("box", "line"):
                    self.values[where][object][s.name] = v
                    if isinstance(v, int) and abs(v) < _LIMIT:
                        self.values[where][np.int64][s.name] = v
        self.line_names = set(self.values["line"][object])
        self._sympy = {}

    def sympy(self, where, dtype):
        key = (where, dtype)
        if key not in self._sympy:
            shape = (1,) * (self.nd if where == "box" else 1)
            out = {}
            for s in self.symbols:
                v = self.values[where][dtype].get(s.name)
                if v is not None:
                    out[s] = v if isinstance(v, np.ndarray) else np.full(shape, v, dtype=dtype)
            # _vector reads the dtype off the first value
            out = {None: np.zeros(shape, dtype=dtype), **out}
            self._sympy[key] = out
        return self._sympy[key]


def _guard_row(g, lo, hi, counts, stats, fallbacks, where):
    """Boolean array of the substituted guard g's truth over T in [lo, hi]; a guard the solver cannot reduce is
    evaluated at every T (counted in fallbacks)."""
    if hi - lo + 1 <= vector_first:
        t = time.perf_counter()
        try:
            row = np.broadcast_to(np.asarray(_exact(g, np.arange(lo, hi + 1).astype), dtype=bool), (hi - lo + 1,)).copy()
            counts["guards_vector"] += 1
            stats["vector_s"] += time.perf_counter() - t
            return row
        except (_Fallback, TypeError, ValueError, ZeroDivisionError, OverflowError):
            pass
    try:
        e = _const_floors(_sympy(g), lo, hi)
        row = np.full(hi - lo + 1, e is sympy.true) if e in (sympy.true, sympy.false) else _solve_guard(e, lo, hi, stats)
        counts["guards_solved"] += 1
        return row
    except (_Fallback, TypeError, ValueError, ZeroDivisionError, sympy.PolynomialError) as err:
        counts["guards_fallback"] += 1
        stats["fallback_evals"] += hi - lo + 1
        t = time.perf_counter()
        route = "per_T"
        try:
            ts = np.arange(lo, hi + 1).astype(object)
            row = np.broadcast_to(np.asarray(_vector(g, ts), dtype=bool), ts.shape).copy()
            route = "vector"
        except (_Fallback, TypeError, ValueError, ZeroDivisionError, OverflowError):
            pass
        if route == "per_T":
            # torch's integer functions evaluate at integer arguments
            row = np.array([_truth(g, x) for x in range(lo, hi + 1)])
        stats[f"fallback_{route}"] += 1
        stats["fallback_s"] += time.perf_counter() - t
        fallbacks.append((*where, route, f"{err!r}"[:120] + " | " + str(g)[:160]))
        return row


def _symbols(tape, at):
    """{symbol: polynomial in T or traced hint} over the tape's symbols, and per class how many."""
    env = tape.shape_env
    sources, values, _ = getattr(env, "export", env).symbol_table()
    plan, base, axes = at.plan, at.base, at.axes
    subs, kinds = {}, collections.Counter()

    def lin(a, b):
        return sum((ai * x for ai, x in zip(a, axes)), sympy.Integer(b)), any(a)

    def size(p, d):
        q = plan[p]
        if q is None:
            return sympy.Integer(base[p].size(d)), False
        return (q[1][d], q[1][d].has(*axes)) if isinstance(q[1][d], sympy.Expr) else lin(*q[1][d])

    for s, src in sources.items():
        nm = src[0].name if hasattr(src[0], "name") else str(src[0])
        nm = getattr(src[0], "nm", nm)
        hit = None
        if m := _SIZE.match(nm):
            p, d = int(m[1]), int(m[2])
            hit = size(p, d)
        elif m := _STRIDE.match(nm):
            p, d = int(m[1]), int(m[2])
            if plan[p] is None:
                hit = sympy.Integer(base[p].stride(d)), False
            else:
                prod = sympy.Integer(1)
                for e in range(d + 1, base[p].dim()):
                    prod *= size(p, e)[0]
                hit = sympy.expand(prod), prod.has(*axes)
        elif m := _OFFSET.match(nm):
            p = int(m[1])
            hit = sympy.Integer(0 if plan[p] is not None else base[p].storage_offset()), False
        elif (m := _INT.match(nm)) and int(m[1]) < len(base) and isinstance(base[int(m[1])], int):
            q = plan[int(m[1])]
            hit = (sympy.Integer(base[int(m[1])]), False) if q is None else lin(q[1], q[2])
        if hit is None:
            subs[s] = values[s]
            kinds["hint"] += 1
        else:
            subs[s] = hit[0]
            kinds["T" if hit[1] else "fixed"] += 1
    return subs, kinds


def _runs(mask, lo):
    """The maximal runs [a, b] of T (T = lo + index) where mask holds."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return [(lo + int(a), lo + int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]


class Cover:
    """Which points of a box (T in [lo, hi], or (nd, T) in [lo, hi] per axis) the traced variants' guards accept,
    each guard solved for its truth over the box once and cached: a guard on one axis by its critical points along
    it, a guard on both axes per nd slice along T. `guards` (variant -> guard indices) picks the subset a query
    covers: the variant's graph guards (Tape.graph) by default, one op's guards for its selector."""

    def __init__(self, at, lo, hi, guards=lambda v: v.tape.graph):
        self.at, self.guards = at, guards
        self.lo, self.hi = (lo, hi) if isinstance(lo, tuple) else ((lo,), (hi,))
        self.axes = at.axes
        self.shape = tuple(b - a + 1 for a, b in zip(self.lo, self.hi))
        self.counts, self.stats, self.fallbacks = collections.Counter(), collections.Counter(), []
        self.queries = []  # per uncovered() its seconds
        self.splits = {}  # (variant id, guard) -> the axes the guard's truth changes along
        self.split_guards = {}  # substituted guard -> those axes, for the guards that split the box
        self._subs, self._rows, self._solved, self._sums = {}, {}, {}, set()
        self._selected = {}  # variant id -> (variant, lowered, entry guards seen, its selected): entry guards only grow
        self._sites = {}  # (variant id, site) -> (variant, lowered, the site's entry guards seen, its selected)
        self._interned = {}  # (shape, digest) -> one copy of a large row: sites' rows repeat across sites and variants
        self._reduced = {}  # id of an evaluated guard -> (it, its row, its split axes)
        self._anded = {}  # ids of rows -> (the rows, their AND)
        self._ored = {}  # ids of a site's own AND and its entries' -> (those, their OR)
        self._by_site = {}  # lowering id -> [lowering, entry guards indexed, site -> its (index, guards)]
        self._substituted = {}  # (guard, its symbols' substitution) -> the substituted guard, shared across variants
        self._leaves = {}  # variant id -> its _Leaves
        self._ir_memo = {}  # (variant id, box or line, dtype) -> IR node id -> its value there (up to _MEMO_SIZE points)
        self._evaluated = {}  # a guard's substituted form (sympy: guard and bindings; IR: _ir_key) -> (row, along the sum)

    def _solve(self, e, where):
        """e's truth over the box, broadcastable to self.shape."""
        free = [k for k, x in enumerate(self.axes) if e.has(x)]
        if not free:
            return None
        if e in self._solved:
            self.counts["guards_shared"] += 1
            return self._solved[e]
        self._solved[e] = self._solve_new(e, free, where)
        return self._solved[e]

    def _solve_new(self, e, free, where):
        # T is positive, so an axis from lo is solved as T + lo - 1 over [1, hi - lo + 1] (an nd axis starts at 0)
        def along(u, lo, hi, x=T):
            return _guard_row(u.xreplace({x: T + (lo - 1)}), 1, hi - lo + 1, self.counts, self.stats, self.fallbacks, where)

        if len(free) == 1:
            k = free[0]
            row = along(e, self.lo[k], self.hi[k], self.axes[k])
            return row.reshape([-1 if j == k else 1 for j in range(len(self.axes))])
        # both axes through nd + T only (the token count of a mixed step): one solve along the sum
        u = e.xreplace({T: T - ND})
        if self.axes == (ND, T) and not sympy.expand(u).has(ND):
            row = along(u, self.lo[0] + self.lo[1], self.hi[0] + self.hi[1])
            self.stats["sums"] += 1
            self._sums.add(e)
            return row[np.add.outer(np.arange(self.shape[0]), np.arange(self.shape[1]))]
        # otherwise over the box at once, each axis broadcast along its own dim (a term on one axis costs that axis)
        t = time.perf_counter()
        try:
            dims = [[-1 if j == k else 1 for j in range(len(self.axes))] for k in range(len(self.axes))]

            def grid(dtype):
                return {x: np.arange(a, b + 1).astype(dtype).reshape(d) for x, a, b, d in zip(self.axes, self.lo, self.hi, dims)}

            out = np.broadcast_to(np.asarray(_exact(e, grid), dtype=bool), self.shape).copy()
            self.counts["guards_vector"] += 1
            self.stats["boxes"] += 1
            self.stats["vector_s"] += time.perf_counter() - t
            return out
        except (_Fallback, TypeError, ValueError, ZeroDivisionError, OverflowError):
            pass
        # or one solve along the last axis per value of the first
        out = np.empty(self.shape, dtype=bool)
        for j, n in enumerate(range(self.lo[0], self.hi[0] + 1)):
            out[j] = along(e.xreplace({self.axes[0]: n}), self.lo[1], self.hi[1], self.axes[1])
        self.stats["slices"] += self.shape[0]
        return out

    def _symbols(self, variant):
        vi = id(variant)
        if vi not in self._subs:
            t = time.perf_counter()
            subs, kinds = _symbols(variant.tape, self.at)
            self.stats["symbols_s"] += time.perf_counter() - t
            self.counts.update({f"symbols_{k}": n for k, n in kinds.items()})
            # the variant is kept: its id keys the cache
            self._subs[vi] = subs, {s for s, e in subs.items() if isinstance(e, sympy.Expr) and e.has(*self.axes)}, variant
        return self._subs[vi]

    def expr(self, variant, i):
        """The variant's guard i over the box's axes (other symbols at their traced hints)."""
        return variant.tape.guards[i].xreplace(self._symbols(variant)[0])

    def _variant_leaves(self, variant):
        vi = id(variant)
        if vi not in self._leaves:
            subs, dep, _ = self._symbols(variant)
            t = time.perf_counter()
            self._leaves[vi] = _Leaves(self, subs, dep)
            self.stats["leaves_s"] += time.perf_counter() - t
        return self._leaves[vi]

    def _evaluate(self, mask, line, evaluate):
        """(truth over the box, along the sum) of a guard on the axes `mask` by evaluate(where, dtype) at the box's
        points ("box") or along the (nd, T) sum ("line", where `line`: the guard reads the axes only through it),
        in int64 where exact; (None, False) on no axis where it holds; _DECLINED where it cannot (47a's routes)."""
        free = [k for k in range(len(self.axes)) if mask >> k & 1]
        line = line and len(free) == 2
        n = sum(self.hi[:2]) - sum(self.lo[:2]) + 1 if line else self.shape[free[0]] if len(free) == 1 else 0
        if n > vector_first:
            return _DECLINED
        t = time.perf_counter()
        for dtype in (np.int64, object):
            try:
                v = evaluate("line" if line else "box", dtype)
                break
            except _EVAL_ERRORS:
                continue
        else:
            return _DECLINED
        if not free:
            if np.asarray(v).all():
                return None, False
            self.counts["guards_fixed_false"] += 1
            return np.zeros((1,) * len(self.axes), dtype=bool), False
        self.counts["guards_axes"] += 1
        self.counts["guards_vector"] += 1
        self.stats["vector_s"] += time.perf_counter() - t
        if line:
            self.stats["sums"] += 1
            row = np.broadcast_to(np.asarray(v, dtype=bool), (n,))
            return row[np.add.outer(np.arange(self.shape[0]), np.arange(self.shape[1]))], True
        self.stats["boxes"] += len(free) > 1
        return np.broadcast_to(np.asarray(v, dtype=bool), tuple(self.shape[k] if k in free else 1 for k in range(len(self.axes)))).copy(), False

    def _ir_row(self, variant, record):
        """A guard of the variant's IR trace from its record (an Env record in its context: its own, or an entry's), as
        _evaluate; shared across variants by its substituted form (_ir_key)."""
        node, written = record
        lv = self._variant_leaves(variant)
        tops = (written[1], written[2]) if written is not None else (node,)
        key = ("ir", written and written[0], tuple(_ir_key(x, lv.keys, lv.key_memo) for x in tops))
        hit = self._evaluated.get(key)
        if hit is not None:
            self.counts["guards_shared"] += 1
            return hit
        mask = 0
        for x in tops:
            mask |= _ir_axes(x, lv.masks, lv.axes_memo)
        names = set()
        if mask and len(self.axes) == 2:
            for x in tops:
                names |= x.free_symbols
        line = self.axes == (ND, T) and all(s in lv.line_names for s in names if s in lv.masks)
        vi = id(variant)

        def evaluate(where, dtype):
            memo = self._ir_memo.setdefault((vi, where, dtype), {})
            leaves, large = lv.values[where][dtype], {}
            if written is None:
                return _ir_vector(node, leaves, memo, dtype is np.int64, large)
            return _REL[written[0]](*(_ir_vector(x, leaves, memo, dtype is np.int64, large) for x in tops))

        self.counts["guards_ir"] += 1
        out = self._evaluate(mask, line, evaluate)
        if out is not _DECLINED:
            self._evaluated[key] = out
        return out

    def _sympy_row(self, variant, g, subs, dep):
        """The sympy guard g (an entry's, or a sympy trace's) at its symbols' values, as _evaluate; shared across
        variants by (g, its symbols' substitution)."""
        free = g.free_symbols
        key = (g, frozenset((s, subs[s]) for s in free if s in subs))
        hit = self._evaluated.get(key)
        if hit is not None:
            self.counts["guards_shared"] += 1
            return hit
        lv = self._variant_leaves(variant)
        mask = 0
        for s in free & dep:
            mask |= lv.masks[s.name]
        line = self.axes == (ND, T) and all(s.name in lv.line_names for s in free & dep)
        out = self._evaluate(mask, line, lambda where, dtype: _vector(g, lv.sympy(where, dtype)))
        if out is not _DECLINED:
            self._evaluated[key] = out
        return out

    def _record_guard(self, variant, i):
        # one record's sympy export (Tape.guards exports every record); i a record index, an entry's sympy guard or
        # an entry's record
        if isinstance(i, sympy.Basic):
            return i
        env = getattr(variant.tape, "shape_env", None)
        if not isinstance(env, _ir.Env):
            return variant.tape.guards[i]
        node, written = env.records[i] if isinstance(i, int) else i
        return env.export.relation(written) if written is not None else env.export.expr(node)

    def _row(self, variant, i, g=None):
        """The truth over the box of the variant's guard i (or of g, an entry's guard keyed i), None where it holds
        over the whole box."""
        vi = id(variant)
        if (vi, i) not in self._rows:
            subs, dep, _ = self._symbols(variant)
            self.counts["guards"] += 1
            got, label = _DECLINED, g
            env = getattr(variant.tape, "shape_env", None)
            if ir_rows and isinstance(env, _ir.Env) and not isinstance(g, sympy.Basic):
                label = env.records[i] if g is None else g
                got = self._ir_row(variant, label)
            if got is _DECLINED:
                g = label = self._record_guard(variant, i if g is None else g)
                if ir_rows:
                    got = self._sympy_row(variant, g, subs, dep)
            if got is _DECLINED:
                got = self._solved_row(g, subs, dep, i)
                label = got[2]
            # a guard shared across variants (_evaluated) is reduced once: its row is then one object, as _and keys it
            hit = self._reduced.get(id(got))
            if hit is not None and hit[0] is got:
                row, split = hit[1:]
            else:
                row, split = self._reduce(got, label)
                self._reduced[id(got)] = (got, row, split)
            if split is not None:
                self.splits[vi, i] = split
            self._rows[vi, i] = row
        return self._rows[vi, i]

    def _reduce(self, got, label):
        """(row, its split axes) of an evaluated guard: None where it holds over the whole box, else kept only
        along the axes it varies on (a sum's row is box-sized): accepts ANDs the small rows first."""
        row, summed = got[:2]
        if row is None:
            return None, None
        vary = [k for k in range(len(self.axes)) if row.shape[k] > 1 and (np.diff(row, axis=k) != 0).any()]
        if not vary and row.flat[0]:
            return None, None
        split = tuple(str(self.axes[k]) for k in vary)
        split = ("nd+T",) if split and summed else split
        if split:
            key = label if isinstance(label, sympy.Basic) else _ir.render(label[0])[:160]
            self.split_guards[key] = split
        return np.ascontiguousarray(row[tuple(slice(None) if k in vary else slice(0, 1) for k in range(len(self.axes)))]), split

    def _solved_row(self, g, subs, dep, i):
        """(row, along the sum, the substituted guard) by substituting g and solving it (47a's route)."""
        key = (g, frozenset((s, subs[s]) for s in g.free_symbols if s in subs))
        e = self._substituted.get(key)
        if e is None:
            e = self._substituted[key] = g.xreplace(subs)
        row = None
        if g.free_symbols & dep:
            self.counts["guards_axes"] += 1
            row = self._solve(e, (i,))
        if row is None and e == sympy.false:
            # on no axis but false at the recorded arguments: a size fixed over the box differs from the trace's
            self.counts["guards_fixed_false"] += 1
            row = np.zeros((1,) * len(self.axes), dtype=bool)
        return row, row is not None and e in self._sums, e

    def site_selected(self, variant, site):
        """Where the selector at native site `site` selects an entry: its traced launches (the op's own guards)
        or a fold's or redispatch's (their guards, LoweredTape.lowering.entry_guards); a twin's refused op's
        key guards are Tape.twin_guards."""
        return np.broadcast_to(self._site_selected(variant, site), self.shape)

    def _site_selected(self, variant, site):
        lowered = variant.captured.lowered
        entries = self._entries(lowered.lowering, site)
        hit = self._sites.get((id(variant), site))
        if hit is not None and hit[0] is variant and hit[1] is lowered and hit[2] == len(entries):
            return hit[3]
        k = lowered.selectors[site - len(lowered.sites)].op
        if k in variant.tape.twin_guards:
            ok = self._and(self._row(variant, ("twin", k, n), g) for n, g in enumerate(variant.tape.twin_guards[k]))
        else:
            ok = self._accepts(variant, variant.tape.ops[k].guards)
        parts = [ok, *(self._and(self._row(variant, ("entry", id(lowered.lowering), j, n), g) for n, g in enumerate(guards)) for j, guards in entries)]
        hit = self._ored.get(tuple(map(id, parts)))
        if hit is not None:
            self._sites[id(variant), site] = (variant, lowered, len(entries), hit[1])
            return hit[1]
        for part in parts[1:]:
            ok = ok | part
        if ok.size > 4096:
            ok = np.ascontiguousarray(ok)
            hit = self._interned.setdefault((ok.shape, hashlib.blake2b(ok.data, digest_size=16).digest()), ok)
            ok = hit if np.array_equal(hit, ok) else ok
        self._ored[tuple(map(id, parts))] = (parts, ok)
        self._sites[id(variant), site] = (variant, lowered, len(entries), ok)
        return ok

    def _entries(self, lowering, site):
        """The (index, guards) of the lowering's entry guards at the site; entry guards only grow."""
        hit = self._by_site.get(id(lowering))
        if hit is None or hit[0] is not lowering:
            hit = self._by_site[id(lowering)] = [lowering, 0, collections.defaultdict(list)]
        for j in range(hit[1], len(lowering.entry_guards)):
            s, guards = lowering.entry_guards[j]
            hit[2][s].append((j, guards))
        hit[1] = len(lowering.entry_guards)
        return hit[2][site]

    def selected(self, variant):
        """Where the variant's graph guards hold and every op's selector selects an entry."""
        lowered = variant.captured.lowered
        n = len(lowered.lowering.entry_guards) if lowered.selectors else 0
        hit = self._selected.get(id(variant))
        if hit is not None and hit[0] is variant and hit[1] is lowered and hit[2] == n:
            return hit[3]
        rows = [self._site_selected(variant, len(lowered.sites) + k) for k in range(len(lowered.selectors))]
        out = np.broadcast_to(self._and([self._accepts(variant), *rows]), self.shape)
        self._selected[id(variant)] = (variant, lowered, n, out)
        return out

    def _and(self, rows):
        """The AND of rows (None holds), broadcastable over the box and only as large as its rows need; one object
        per set of rows (the memo holds them, so their ids stay theirs)."""
        rows = {id(row): row for row in rows if row is not None}
        key = frozenset(rows)
        hit = self._anded.get(key)
        if hit is not None:
            return hit[1]
        by_shape = {}
        for row in rows.values():
            by_shape[row.shape] = row if row.shape not in by_shape else by_shape[row.shape] & row
        out = np.ones((1,) * len(self.shape), dtype=bool)
        for row in sorted(by_shape.values(), key=np.size):
            out = out & row
        self._anded[key] = (list(rows.values()), out)
        return out

    def _accepts(self, variant, guards=None):
        return self._and(self._row(variant, i) for i in (self.guards(variant) if guards is None else guards))

    def accepts(self, variant, guards=None):
        """Boolean array over the box (a read-only view): where every guard of the subset holds."""
        return np.broadcast_to(self._accepts(variant, guards), self.shape)

    def uncovered(self, entry, guards=None, selectors=False):
        """The runs no non-learning variant of the entry accepts (with `selectors`, also selects an entry at every
        op), along the last axis: (lo, hi) of T, or (nd, lo, hi) per nd; [] proves the box covered."""
        t = time.perf_counter()
        covered = np.zeros(self.shape, dtype=bool)
        for v in entry.variants:
            if not v.learns:
                covered |= self.selected(v) if selectors else self.accepts(v, None if guards is None else guards(v))
        runs = self.runs(~covered)
        self.queries.append(time.perf_counter() - t)
        return runs

    def runs(self, mask):
        """The maximal runs of the box's points where mask holds, along the last axis: (lo, hi) of T, or
        (nd, lo, hi) per nd."""
        if len(self.shape) == 1:
            return _runs(mask, self.lo[0])
        return [(self.lo[0] + j, a, b) for j in range(self.shape[0]) for a, b in _runs(mask[j], self.lo[1])]

    def regions(self):
        """The number of distinct truth vectors of the solved guards over the box, and per split-axes tuple the
        number of guards that split those axes."""
        return len(np.unique(self.hashes())), dict(collections.Counter(self.splits.values()))

    def hashes(self):
        """Per point of the box a hash of the solved guards' truth vector there."""
        h = np.zeros(self.shape, dtype=np.int64)
        for row in self._rows.values():
            if row is not None:
                h = h * 1000003 + np.broadcast_to(row, self.shape)
        return h


def _affine(points, values):
    """(coefficients per axis, constant) of the integer affine function through (points, values): fitted at the
    first affinely independent points, checked at the rest."""
    d = len(points[0])
    basis = []
    for p in points:
        rows = [[*q, 1] for q in basis + [p]]
        if sympy.Matrix(rows).rank() == len(rows):
            basis.append(p)
        if len(basis) == d + 1:
            break
    if len(basis) < d + 1:
        raise ValueError(f"{len(points)} points do not span {d} axes")
    sol = sympy.Matrix([[*q, 1] for q in basis]).LUsolve(sympy.Matrix([values[points.index(q)] for q in basis]))
    if not all(x.is_integer for x in sol):
        raise ValueError(f"not integer affine: {list(zip(points, values))}")
    *a, c = (int(x) for x in sol)
    for p, v in zip(points, values):
        if sum(ai * pi for ai, pi in zip(a, p)) + c != v:
            raise ValueError(f"not affine: {v} at {p}")
    return tuple(a), c


class ArgsAt:
    """The entry's arguments at a point (T, or a tuple of sizes such as (nd, T)), from recorded calls at d + 1
    affinely independent points and more: each argument whose shape (or int value) changes across them is integer
    affine in the point, checked at every recorded point; a changed tensor is a fresh contiguous buffer. `axes`
    names the point's axes (default T, or (nd, T)). `sizes` gives the dims that are not affine as sympy expressions
    over the axes, {(argument, dim): expr} (a page table's width ceil(S / 64) is CeilDiv(S, 64)), checked the same
    way."""

    def __init__(self, got, axes=None, sizes=None):
        pts = sorted(got)
        self.dims = len(pts[0]) if isinstance(pts[0], tuple) else 1
        self.axes = AXES[self.dims] if axes is None else tuple(map(axis, axes))
        points = [p if isinstance(p, tuple) else (p,) for p in pts]
        calls = [got[p] for p in pts]
        self.base, self.T0 = calls[0], pts[0]
        self.plan = []  # per argument: None (as at T0), ("int", a, b) or ("tensor", [(a, b) or expr per dim], dtype, device); a per axis
        for i, x0 in enumerate(self.base):
            xs = [c[i] for c in calls]
            if isinstance(x0, torch.Tensor):
                if all(x.shape == x0.shape for x in xs):
                    self.plan.append(None)
                    continue
                if not all(x.is_contiguous() for x in xs):
                    raise ValueError("a changed argument that is not contiguous")
                dims = []
                for d in range(x0.dim()):
                    if (i, d) not in (sizes or {}):
                        dims.append(_affine(points, [x.shape[d] for x in xs]))
                        continue
                    e = sympy.sympify(sizes[i, d])
                    for p, x in zip(points, xs):
                        if int(e.xreplace(dict(zip(self.axes, p)))) != x.shape[d]:
                            raise ValueError(f"argument {i} dim {d} is {x.shape[d]} at {p}, not {e}")
                    dims.append(e)
                self.plan.append(("tensor", dims, x0.dtype, x0.device))
            elif isinstance(x0, int) and any(x != x0 for x in xs):
                self.plan.append(("int", *_affine(points, xs)))
            else:
                self.plan.append(None)

    def __call__(self, *point):
        point = point[0] if len(point) == 1 and isinstance(point[0], tuple) else point
        out = []
        for x, p in zip(self.base, self.plan):
            if p is None:
                out.append(x)
            elif p[0] == "int":
                out.append(sum(a * v for a, v in zip(p[1], point)) + p[2])
            else:
                at = dict(zip(self.axes, point))
                out.append(torch.empty([int(q.xreplace(at)) if isinstance(q, sympy.Expr) else sum(a * v for a, v in zip(q[0], point)) + q[1] for q in p[1]], dtype=p[2], device=p[3]))
        return tuple(out)


def _inside(point, runs):
    if isinstance(point, tuple):
        return any(r[0] == point[0] and r[1] <= point[1] <= r[2] for r in runs)
    return any(r[0] <= point <= r[1] for r in runs)


def _middle(runs):
    """The middle of the widest run: T, or (nd, T)."""
    r = max(runs, key=lambda r: r[-1] - r[-2])
    x = (*r[:-2], r[-2] + (r[-1] - r[-2]) // 2)
    return x if len(x) > 1 else x[0]


def _counters(entry):
    return {k: getattr(entry, k) for k in ("traces", "relowers", "folds", "redispatches", "eager")}


def cover(entry, at, call, lo, hi, mode="graph", log=print, max_calls=64, count_regions=True):
    """Calls the entry until the solver (Cover) proves every point of the box [lo, hi] (T, or (nd, T) with tuple
    bounds) covered: first by the variants' graph guards, calling the middle of the widest run they leave (a
    trace); then by every op's selector. mode "graph" calls the middle of the widest run where some selector
    selects nothing (the call dispatches each failing op again, or folds a trace); mode "op" covers one op at a
    time: per region of its own guards' truth (ops with identical guards solved and dispatched together) where
    it selects nothing, its op alone is dispatched again at one point (_host_trace_redispatch), the function not
    called; a refused one calls the entry there. No run left proves the cover. Returns the stats; count_regions=False
    skips counting the regions of the guards' truth (a diagnostic the cover does not need: the startup path)."""
    if mode not in ("graph", "op"):
        raise ValueError(f"cover: mode {mode!r} is not 'graph' or 'op'")
    solver = Cover(at, lo, hi)
    before, t0 = _counters(entry), time.perf_counter()
    called = {"graph": [], "selectors": [], "ops": []}
    stuck, dispatched, refused, groups = None, 0, collections.Counter(), []
    runs = solver.uncovered(entry)
    while runs and len(called["graph"]) < max_calls:
        x = _middle(runs)
        called["graph"].append(x)
        # a variant that learns keys is relowered at a later call
        for _ in range(3):
            call(x)
            runs = solver.uncovered(entry)
            if not _inside(x, runs):
                break
        if _inside(x, runs):
            stuck = ("graph", x)
            break
    if not runs and stuck is None and mode == "graph":
        runs = solver.uncovered(entry, selectors=True)
        while runs and len(called["selectors"]) < max_calls:
            x = _middle(runs)
            called["selectors"].append(x)
            call(x)
            runs = solver.uncovered(entry, selectors=True)
            if _inside(x, runs):
                stuck = ("selectors", x)
                break
    elif not runs and stuck is None:

        def elsewhere(v):
            # the points another variant selects: a refused redispatch's call traces one (the replay tries them all)
            out = np.zeros(solver.shape, dtype=bool)
            for w in entry.variants:
                if w is not v and not w.learns:
                    out |= solver.selected(w)
            return out

        held, seen = np.zeros(solver.shape, dtype=bool), set()
        while stuck is None and (todo := [v for v in entry.variants if not v.learns and id(v) not in seen]):
            v = todo[0]
            seen.add(id(v))
            lowered = v.captured.lowered
            n = len(lowered.sites)
            # the points no earlier variant's graph guards hold, as the replay's search order has them
            mine = solver.accepts(v) & ~held
            held |= mine
            same = collections.defaultdict(list)
            for k, sel in enumerate(lowered.selectors):
                key = ("twin", sel.op) if sel.op in v.tape.twin_guards else tuple(solver.expr(v, i) for i in v.tape.ops[sel.op].guards)
                same[key].append(n + k)
            for sites in same.values():
                group = {"sites": len(sites), "dispatched_at": []}
                groups.append(group)
                while len(group["dispatched_at"]) < max_calls:
                    ok = mine & ~elsewhere(v)
                    runs = solver.runs(ok & ~np.logical_and.reduce([solver.site_selected(v, s) for s in sites]))
                    if not runs:
                        break
                    x = _middle(runs)
                    group["dispatched_at"].append(x)
                    args = at(x)
                    with entry._lock:
                        ev = v.native.evaluate(args)
                        unselected = [] if ev is None else [s for s in sites if s in ev[3]]
                        done = bool(unselected) and entry._redispatch(v, args, list(ev[0]), unselected) is None
                    if done:
                        dispatched += 1
                    else:
                        refused[str(list(entry.redispatch_refusals)[-1:])[:200]] += 1
                        called["ops"].append(x)
                        call(x)
                    ok = mine & ~elsewhere(v)
                    if _inside(x, solver.runs(ok & ~np.logical_and.reduce([solver.site_selected(v, s) for s in sites]))):
                        stuck = ("op", x)
                        break
                if stuck:
                    break
        runs = [] if stuck else solver.uncovered(entry, selectors=True)
    after = _counters(entry)
    regions, splits = solver.regions() if count_regions else (None, dict(collections.Counter(solver.splits.values())))
    c = {
        "mode": mode,
        "proven": not runs and stuck is None,
        "uncovered": runs[:8],
        "stuck_at": stuck,
        "variants": sum(not v.learns for v in entry.variants),
        "called": called,
        "dispatched": dispatched,
        "refused": dict(refused),
        "groups": [(g["sites"], len(g["dispatched_at"])) for g in groups],
        "delta": {k: after[k] - before[k] for k in after},
        "s": round(time.perf_counter() - t0, 3),
        "queries": len(solver.queries),
        "query_s": round(sum(solver.queries), 4),
        "regions": regions,
        "splits": splits,
        "guards": dict(solver.counts, **solver.stats),
        "fallbacks": solver.fallbacks[:8],
    }
    log(f"cover {mode}: proven {c['proven']} variants {c['variants']} called {called} dispatched {dispatched} refused {c['refused']} delta {c['delta']} in {c['s']}s, regions {regions}, groups {c['groups'][:16]}")
    return c


def points(solver, per_region=3):
    """Sample points of every region of the solved guards' truth (Cover.regions): per maximal run along the
    last axis of one truth vector, its ends and middle."""
    h = solver.hashes()
    out = []
    rows = [h] if h.ndim == 1 else list(h)
    for j, row in enumerate(rows):
        cuts = np.flatnonzero(row[1:] != row[:-1]) + 1
        for a, b in zip([0, *cuts.tolist()], [*(cuts - 1).tolist(), len(row) - 1]):
            for t in sorted({a, a + (b - a) // 2, b})[:per_region]:
                x = solver.lo[-1] + t
                out.append(x if h.ndim == 1 else (solver.lo[0] + j, x))
    return out
