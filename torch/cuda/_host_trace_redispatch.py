"""Host tracing (private): one op of a variant dispatched again, alone.

A call that a variant's graph guards hold has every op's input metadata at
its formula in the variant's symbols; where an op's own guards (its
selector) fail, its dispatch (OpRec.redo: its traced host, or its Triton or
CuTe DSL launcher) runs again on a fresh trace, over tensors at those
formulas' values at the call and the planned roots. Its launches become an
entry of the selector, lowered as a fold's are, with every guard the run
recorded as its predicate. No Python of the traced function runs, and no
other op. FoldRefused where the op's outputs, allocations or launch topology
differ: the call traces again.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import replace
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensorConverter, FakeTensorMode
from torch.cuda import _host_trace_hint_audit as _hint_audit, _host_trace_ir as _ir
from torch.cuda._host_trace import _drop_tracebacks, Declined, declined
from torch.cuda._host_trace_launch import compiled_cluster, KernelLaunch
from torch.cuda._host_trace_lower_tape import FoldRefused, HintsFail, IRRename, lower_entries
from torch.cuda._host_trace_program import compile_program
from torch.cuda._host_trace_tape import (
    _active,
    _ALLOC_ALIGNMENT,
    _capture,
    _CAPTURE_ERRORS,
    _InputRec,
    _ir_census,
    _PLACEHOLDER_LOW,
    _PLACEHOLDER_TAG,
    _placeholder,
    _Root,
    _sym_expr,
    _sym_key,
    _Trace,
    _TracedTensor,
    _TraceMode,
    CallValues,
    fx_config,
    Memcpy,
)
from torch.cuda._utils import _check_cuda_bindings
from torch.utils import _pytree as pytree


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from torch.cuda._host_trace_lower_tape import LoweredLaunch, LoweredMemset, LoweredSelector, LoweredTape
    from torch.cuda._host_trace_tape import OpRec


def redispatch(
    lowered: LoweredTape, args: Sequence[Any], values: Sequence[int], unselected: Sequence[int], dispatches: dict | None = None
) -> tuple[torch._C._HostTraceProgram, list[tuple[int, int, tuple[LoweredLaunch | LoweredMemset, ...]]]]:
    """Selectors `unselected` (native site indices) of the variant, whose
    graph guards hold the call `args` (at rows `values`), as entries from
    their ops dispatched again: fold's result, with no trace."""
    program = lowered.lowering.program
    rows = len(program.values)
    # the ops' input formulas are evaluated at the call, each new row too
    program.inputs, program.values = tuple(args), list(values)
    try:
        selected = []
        # one that refused before goes first: refusing again costs no other op's dispatch
        with _hosting(lowered.tape.device):
            for i in sorted(unselected, key=lambda i: i not in lowered.lowering.refused):
                selector = lowered.selectors[i - len(lowered.sites)]
                try:
                    try:
                        selected.append((i, *_redo(lowered, selector, hosted=True, dispatches=dispatches)))
                    except HintsFail:
                        selected.append((i, *_redo(lowered, selector, ir=False, hosted=True)))
                except FoldRefused as e:
                    lowered.lowering.refused.add(i)
                    e.op = selector.op
                    raise
        selected.sort(key=lambda s: unselected.index(s[0]))
        values = program.values
    except (Declined, FoldRefused) as e:
        compiled = compile_program(program) if len(program.values) > rows else None
        if isinstance(e, Declined):
            raise FoldRefused(f"a lowering declined: {e}", compiled) from e
        if e.program is None:
            e.program = compiled
        raise
    finally:
        program.inputs = ()
    return lower_entries(lowered, args, values, selected)


_redo_fakes = threading.local()


@contextlib.contextmanager
def _hosting(device: torch.device) -> Iterator[None]:
    # as trace() runs a host: Triton and CuTe DSL launches intercepted, any other launch refused
    from torch.cuda._host_trace_cute import intercepting as cute_intercepting
    from torch.cuda._host_trace_triton_launch import intercepting

    with torch.cuda.device(device), _capture(device), intercepting(), cute_intercepting():
        yield


def _redo(
    lowered: LoweredTape,
    selector: LoweredSelector,
    ir: bool = True,
    hosted: bool = False,
    dispatches: dict[Any, list[tuple[_Trace, list[torch.SymInt], list[_Root]]]] | None = None,
) -> tuple[list[Any], list[tuple[int, Any]]]:
    # ir off renames in sympy
    # dispatches: the family's dispatches by what they read of their call, each with its fresh symbols and roots
    old, lo = lowered.tape, lowered.lowering
    op: OpRec = old.ops[selector.op]
    if op.kind not in ("traced", "eager") or op.redo is None or torch.cuda._host_trace.symbolic != "ir":
        raise FoldRefused(f"{op.func} has no traced dispatch to run again")
    # the fresh symbols' env, as _Trace makes one: only a redo that dispatches builds the _Trace and its tensors
    env = _ir.Env()
    # no fresh symbol shares a name with the variant's: one the dispatch did
    # not get from its arguments (a closure's) is refused, not renamed
    env.unique_ids |= old.shape_env.unique_ids
    env._id_skips.update(getattr(old.shape_env, "_id_skips", {}))
    # an IR variant's entry takes the fresh trace's nodes renamed in the variant's context
    ir = ir and lo.ir
    key = _sym_key if ir else _sym_expr
    renamer = IRRename(old.shape_env, env, f"{op.func} dispatched on a symbol of no argument")
    # the fresh trace's symbol -> its formula in the variant's symbols (by name, a node, in ir)
    mapping: dict[Any, Any] = renamer.mapping if ir else {}
    fresh: dict[Any, torch.SymInt] = {}
    roots: dict[int, _Root] = {}
    back: dict[int, _Root] = {}  # by the fresh root's id
    # the call's tensors (by the variant's tensor's id), the inputs and the outer tensors, built where it dispatches
    tensors: dict[int, tuple] = {}
    inputs: list[_InputRec] = []
    outer: dict[int, tuple] = {}
    made: list[torch.SymInt] = []
    at_call: list[int] = []  # each of `made`'s value: its formula's at the call, or a root's placeholder
    # what the dispatch reads of its call, in order: constants, the fresh symbols, roots and tensors
    shape: list[Any] = []

    def bind(sym: torch.SymInt, to: Any) -> None:
        if ir:
            n = _sym_key(to)
            mapping[sym.node.node.args[0]] = old.shape_env.ctx.const(n) if isinstance(n, int) else n
        else:
            mapping[_sym_expr(sym)] = _sym_expr(to)

    def value(v: Any) -> Any:
        if isinstance(v, (torch.SymBool, torch.SymFloat)):
            # a heuristic's constexpr, say: its formula is not an input row
            raise FoldRefused(f"{op.func} takes a symbolic {type(v).__name__}")
        if not isinstance(v, torch.SymInt):
            # 1, 1.0 and True are equal, as are 0.0 and -0.0
            shape.append((type(v), repr(v) if isinstance(v, float) else v))
            return v
        k = key(v)
        if k not in fresh:
            at_call.append(lo.program.values[lo.row(k)])
            fresh[k] = env.symbol(at_call[-1], f"r{len(fresh)}", positive=bool(_sym_expr(v).is_positive))
            made.append(fresh[k])
            bind(fresh[k], v)
        shape.append(fresh[k])
        return fresh[k]

    def root(r: _Root) -> _Root:
        if id(r) not in roots:
            if r.kind == "argument":
                base = lo.program.values[lo.row(r.sym)]
                at_call.append(_PLACEHOLDER_TAG | (base & _PLACEHOLDER_LOW))
                sym = env.symbol(at_call[-1], f"{r.name}.base")
                made.append(sym)
                bind(sym, r.sym)
            else:
                at_call.append(_placeholder(r) // _ALLOC_ALIGNMENT)
                fq = env.symbol(at_call[-1], f"{r.name}.base/{_ALLOC_ALIGNMENT}")
                made.append(fq)
                if ir:
                    (q,) = _sym_key(r.sym).free_symbols
                    mapping[fq.node.node.args[0]] = old.shape_env.ctx.symbols[q]
                else:
                    (q,) = _sym_expr(r.sym).free_symbols
                    mapping[_sym_expr(fq)] = q
                sym = _ALLOC_ALIGNMENT * fq
            roots[id(r)] = _Root(r.name, sym, r.kind, r.index, r.host)
            back[id(roots[id(r)])] = r
            if r.kind == "argument":
                # the input's record, as trace() has it: an as_strided bound, an overlap, copy-on-write
                rec = next(i for i in old.inputs if i.root is r)
                layout = [value(n) for n in rec.sizes], [value(s) for s in rec.strides], value(rec.offset)
                rec = replace(rec, sizes=layout[0], strides=layout[1], offset=layout[2], root=roots[id(r)])
                inputs.append(rec)
                shape.append((rec.dtype, rec.extent, rec.cow))
            else:
                # the storage, as trace() has it: an as_strided bound
                if r.kind == "allocation":
                    a = next(a for a in old.allocs if a.root is r)
                    sizes, strides, offset, dtype = a.sizes, a.strides, 0, a.dtype
                else:
                    o = next(o for o in old.eager_outputs if o._root is r)
                    sizes, strides, offset, dtype = o.shape, o._sym_strides, o._sym_offset, o.dtype
                layout = [value(n) for n in sizes], [value(s) for s in strides], value(offset)
                outer[id(roots[id(r)])] = (roots[id(r)], *layout, dtype, old.device)
                shape.append(dtype)
        shape.append(roots[id(r)])
        return roots[id(r)]

    def tensor(t: _TracedTensor) -> _TracedTensor:
        if id(t) not in tensors:
            sizes, strides = [value(n) for n in t.shape], [value(s) for s in t._sym_strides]
            tensors[id(t)] = (root(t._root), sizes, strides, value(t._sym_offset), t.dtype, t.device)
            shape.append((t.dtype, t.device))
        shape.append(t)
        return t

    # the hints: the formulas at the call, rows that fail at no other call
    with lo.symbolic.total():
        call = pytree.tree_map(lambda a: tensor(a) if isinstance(a, _TracedTensor) else value(a), op.call)
    # a dispatch that read the same of its call, whose guards hold at this one, is this one's: the op's records are a
    # function of what it reads (an entry's are taken wherever its predicate holds)
    sig, cached, nodes = None, None, 0
    if dispatches is not None and ir and not any(isinstance(a, torch.Tensor) and not isinstance(a, _TracedTensor) for a in pytree.tree_leaves(op.call)):
        index = {k: i for i, k in enumerate([*map(id, (*made, *roots.values())), *tensors])}
        read = tuple((index[id(x)], x.node.node.args[0] in env.ctx.positive) if isinstance(x, torch.SymInt) else (index[id(x)],) if isinstance(x, (_Root, _TracedTensor)) else x for x in shape)
        sig = (op.func if isinstance(op.func, OpOverload) else op.redo, op.kind, repr(pytree.tree_structure(op.call)), read)
        try:
            hash(sig)
        except TypeError:
            sig = None
    if sig in (dispatches or {}):
        consts = _ir.Ctx()
        hints = [consts.const(x) for x in at_call]
        # the latest first: a layer takes the one before's
        for c in reversed(dispatches[sig]):
            at = {s.node.node.args[0]: h for s, h in zip(c[1], hints)}
            with contextlib.suppress(KeyError, ArithmeticError, ValueError):
                if all(consts.transfer(g, c[0].shape_env.ctx, at, {}).op == "true" for g, _ in c[0].shape_env.records):
                    cached = c
                    break
    if cached is not None:
        tr, made_then, roots_then = cached
        env = tr.shape_env
        renamer = IRRename(old.shape_env, env, renamer.refusal)
        renamer.mapping.update({s.node.node.args[0]: mapping[m.node.node.args[0]] for s, m in zip(made_then, made)})
        mapping = renamer.mapping
        back = {id(s): back[id(m)] for s, m in zip(roots_then, roots.values())}
    else:
        # FakeTensorMode() extracts the whole stack: a thread's redos reuse one, each with a fresh converter (a nested
        # redo makes its own)
        fake = vars(_redo_fakes).pop("mode", None) or FakeTensorMode(allow_fallback_kernels=False)
        fake.fake_tensor_converter = FakeTensorConverter(export=False)
        tr = _Trace(old.device, fake_mode=fake, shape_env=env)
        tr.values = CallValues(env, {m.node.node.args[0]: x for m, x in zip(made, at_call)})
        tr.inputs, tr.arguments = inputs, {id(i.root): i for i in inputs}
        tr.outer = {k: _TracedTensor(*o) for k, o in outer.items()}
        built = {k: _TracedTensor(*t) for k, t in tensors.items()}
        call = pytree.tree_map_only(_TracedTensor, lambda a: built[id(a)], call)
        if len(op.allocs):
            tr.first_alloc = old.allocs[op.allocs.start].root.index
        prior = getattr(_active, "trace", None)
        _active.trace = tr
        # an operator's dispatch runs where _TraceMode dispatched it, a launcher where the host called it
        mode = contextlib.nullcontext() if isinstance(op.func, OpOverload) else _TraceMode(tr)
        try:
            # hosted: inside the caller's _hosting, one capture for all its redos
            with contextlib.nullcontext() if hosted else _hosting(old.device):
                tr.stream = torch.cuda.current_stream(old.device)
                with mode, fx_config.patch(backed_size_oblivious=False):  # type: ignore[attr-defined]
                    op.redo(*call)
                from cuda.bindings import runtime

                graph = _check_cuda_bindings(runtime.cudaStreamGetCaptureInfo(tr.stream.cuda_stream))[2]
                # less the capture's own node (tape._capture's _sleep(0))
                nodes = _check_cuda_bindings(runtime.cudaGraphGetNodes(graph, numNodes=0))[1] - 1
                if nodes and hosted:
                    raise declined(f"the host enqueued {nodes} operations the trace does not record")
        except Exception as e:
            refused = isinstance(e, Declined) or (hosted and getattr(e, "error_code", None) in _CAPTURE_ERRORS)
            if not refused and torch.cuda._host_trace.raise_unexpected:
                raise
            raise FoldRefused(f"{op.func} declines again: {type(e).__name__}: {e}") from e
        finally:
            _active.trace = prior
            _redo_fakes.mode = fake
    if tr.declined is not None or _ir_census(tr):
        _drop_tracebacks(tr.declined)
        raise FoldRefused(f"{op.func} declines again: {tr.declined or _ir_census(tr)}")
    # its records are the entry's predicate
    _hint_audit.freeze(env, len(env.records))
    eager = op.kind == "eager"
    if len(tr.ops) != 1 or tr.sites or (tr.eager_outputs and not eager) or tr.ops[0].kind != op.kind:
        raise FoldRefused(f"{op.func} dispatches otherwise")
    # one whose guards read a symbol the dispatch made itself is never taken
    names = {s.node.node.args[0] for s in made}
    if sig is not None and cached is None and not nodes and all(g.free_symbols <= names for g, _ in env.records):
        dispatches.setdefault(sig, []).append((tr, made, list(roots.values())))
    new = tr.ops[0]
    if eager:
        # the call's fresh outputs are the variant's, one to one
        (_, call), (_, own) = tr.launches[0], old.launches[op.launches.start]
        for a, b in zip(call.outputs, own.outputs):
            if a._root.kind == "eager" and b._root.kind == "eager":
                if back.setdefault(id(a._root), b._root) is not b._root or sum(r is b._root for r in back.values()) > 1:
                    raise FoldRefused(f"{op.func} returned other metadata")
                if ir:
                    ((fq,), (q,)) = _sym_key(a._root.sym).free_symbols, _sym_key(b._root.sym).free_symbols
                    mapping[fq] = old.shape_env.ctx.symbols[q]
                else:
                    ((fq,), (q,)) = _sym_expr(a._root.sym).free_symbols, _sym_expr(b._root.sym).free_symbols
                    mapping[fq] = q
    if len(new.launches) != len(op.launches) or len(new.allocs) != len(op.allocs):
        raise FoldRefused(f"{op.func} dispatches otherwise")
    olds = old.allocs[op.allocs.start : op.allocs.stop]
    for a, b in zip(tr.allocs, olds):
        bind(a.q, b.q)
        back[id(a.root)] = b.root

    def rename(v: Any) -> Any:
        if ir:
            return renamer(v)
        e = _sym_expr(v)
        if not e.free_symbols <= mapping.keys():
            raise FoldRefused(f"{op.func} dispatched on a symbol of no argument")
        e = e.xreplace(mapping)
        return int(e) if e.is_Integer else e

    if ir:
        # Env records, as the variant's own guards, and its checks of the call's values (Tape.checks)
        records = [*env.records, *((c.node.node, c.node.written) for c in tr.checks)]
        guards = [r for r in map(renamer.record, records) if r[0].op != "true"]
        false = any(r[0].op == "false" for r in guards)
    else:
        exprs = [*(g.expr for g in env.guards), *(c.node.expr for c in tr.checks)]
        guards = [g for g in map(rename, exprs) if g is not sympy.true]
        false = sympy.false in guards
    if false:
        raise AssertionError(f"host_trace: {op.func}'s dispatch again guards a condition the call fails")
    # a symbol the redo's guards pin (s == 1 at a size-1 call): its layouts and the variant's agree under them
    ctx, pins = old.shape_env.ctx if ir else None, {}
    for g, _ in guards if ir else ():
        c, ts = ctx.as_terms(g.args[0]) if g.op == "eq" else (0, {})
        if len(ts) == 1 and (t := next(iter(ts))).op == "sym" and abs(ts[t]) == 1:
            pins[t.args[0]] = ctx.const(-c * ts[t])
    subs = {**ctx.symbols, **pins} if pins else {}

    def pinned(v: Any) -> Any:
        v = ctx.transfer(v, ctx, subs, {}) if isinstance(v, _ir.Node) else v
        return v.args[0] if isinstance(v, _ir.Node) and v.op == "const" else v

    def same(x: Any, y: Any) -> bool:
        if isinstance(x, _TracedTensor) != isinstance(y, _TracedTensor):
            return False
        if not isinstance(x, _TracedTensor):
            if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
                return x is y
            return rename(x) == key(y) if isinstance(x, torch.SymInt) or isinstance(y, torch.SymInt) else x == y
        layout = [*map(rename, x.shape), *map(rename, x._sym_strides), rename(x._sym_offset)]
        want = [*map(key, y.shape), *map(key, y._sym_strides), key(y._sym_offset)]
        agree = layout == want or (pins and list(map(pinned, layout)) == list(map(pinned, want)))
        return back.get(id(x._root)) is y._root and x.dtype == y.dtype and agree

    for a, b in zip(tr.allocs, olds):
        # as same() compares an output: under the redo's pins ((1, n) against (s, n) at s == 1)
        layout, want = [*map(rename, a.sizes), *map(rename, a.strides)], [*map(key, b.sizes), *map(key, b.strides)]
        if a.dtype != b.dtype or (layout != want and not (pins and list(map(pinned, layout)) == list(map(pinned, want)))):
            raise FoldRefused(f"{op.func} allocates otherwise")
    if len(new.outputs) != len(op.outputs) or not all(map(same, new.outputs, op.outputs)):
        raise FoldRefused(f"{op.func} returned other metadata")
    if eager:
        # a Triton or CuTe DSL launch: its function, then its grid and options
        # or streams, compared as its arguments are (as fold compares them)
        split = [(c.target[:2], c.target[2:]) if isinstance(c.target, tuple) else (c.target, ()) for c in (call, own)]
        new_leaves, new_spec = pytree.tree_flatten((split[0][1], call.args, call.kwargs))
        old_leaves, old_spec = pytree.tree_flatten((split[1][1], own.args, own.kwargs))
        if split[0][0] != split[1][0] or call.generator is not own.generator or new_spec != old_spec or not all(map(same, new_leaves, old_leaves)):
            raise FoldRefused(f"another eager call of {op.func}")
        # the variant's own call stands; only the guards that select it are new
        return guards, []
    launches = []
    olds = [rec for _, rec in old.launches[op.launches.start : op.launches.stop]]
    descriptors = getattr(torch._C, "_host_trace_entry_descriptors_enabled", bool)()
    for n, (_, x), y in zip(selector.nodes, tr.launches, olds, strict=True):
        if isinstance(x, KernelLaunch) and (x.rng or x.rng_increment or x.cpu_scalars):
            raise FoldRefused(f"an RNG or CPU scalar launch in {op.func}")
        if isinstance(x, KernelLaunch) and x.descriptors and not descriptors:
            # an entry holds no TMA descriptor without entry descriptors
            raise FoldRefused(f"a TMA launch in {op.func}")
        if isinstance(x, Memcpy):
            raise FoldRefused(f"a memcpy in {op.func}")
        if any(id(r) not in back for r in x.roots):
            raise FoldRefused(f"a launch in {op.func} of a root of no role")
        owned = tuple(back[id(r)] for r in x.roots)
        topology = type(x) is type(y) and {id(r) for r in owned} == {id(r) for r in y.roots}
        if topology and isinstance(x, KernelLaunch):
            topology = (x.attributes, x.programmatic) == (y.attributes, y.programmatic)
            topology = topology and (x.function == y.function or compiled_cluster(x.function) == compiled_cluster(y.function))
        elif topology:
            topology = x.element_size == y.element_size and rename(x.height) == key(y.height)
        if not topology:
            raise FoldRefused(f"another launch topology in {op.func}")
        launches.append((lowered.launches[n].seq, _renamed(x, owned, rename)))
    return guards, launches


def _renamed(x: Any, roots: tuple[_Root, ...], rename: Callable[[Any], Any]) -> Any:
    # a kernel launch or memset with its values renamed and its roots `roots`
    if isinstance(x, KernelLaunch):
        geometry = (tuple(map(rename, x.grid)), tuple(map(rename, x.block)), rename(x.smem))
        return replace(x, slots=tuple(map(rename, x.slots)), roots=roots, grid=geometry[0], block=geometry[1], smem=geometry[2])
    return replace(x, slots=(rename(x.slots[0]),), roots=roots, width=rename(x.width), height=rename(x.height), pitch=rename(x.pitch))
