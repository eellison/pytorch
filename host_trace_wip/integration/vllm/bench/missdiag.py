# Phase 6 step 4b diagnostic: why a family that already has variants traces again. Runs bench/newkey.py (same args)
# with HostTraceReplay._miss / _relower wrapped, then prints, per entry, each variant's guards that variant 0 lacks
# (and the reverse), and the misses with the relower outcome before them.
#   CUDA_VISIBLE_DEVICES=2 bash python_vllm_cpp.sh bench/missdiag.py --out bench/out/newkey/X.json [newkey args]
import os, runpy, sys

import traceback

import torch.cuda._host_trace_ir as IR
import torch.cuda._host_trace_replay as R

STACKS = {}  # new guard records made during a relower: stack -> (count, example node)
IN_RELOWER = [False]
_record = IR.Env._record


def record(self, g, written):
    n = len(self.records)
    _record(self, g, written)
    if IN_RELOWER[0] and len(self.records) > n:
        st = "".join(traceback.format_list(traceback.extract_stack()[-14:-1]))
        c, ex = STACKS.get(st, (0, None))
        STACKS[st] = (c + 1, ex or f"{g.op} {str(g)[:200]}")


IR.Env._record = record

LOG = []
_miss, _relower = R.HostTraceReplay._miss, R.HostTraceReplay._relower


def miss(self, contract, args, candidates=(), before=None):
    n = sum(len(f) for f in self._families.values())
    ints = [a for a in args if isinstance(a, int)][:8]
    shapes = [tuple(a.shape) for a in args[:3] if hasattr(a, "shape")]
    LOG.append(("miss", id(self), n, len(candidates), before is not None, ints, shapes))
    return _miss(self, contract, args, candidates, before)


WATCH = {"bind_opaque", "remap", "_relower", "_build"}


def relower(self, family, args, variant, values):
    rets = []

    def local(frame, event, arg):
        if event == "line" and frame.f_code.co_name == "bind_opaque":
            rets.append(("line", frame.f_lineno))
        elif event == "return" and arg is None:
            rets.append((frame.f_code.co_name, frame.f_lineno))
        elif event == "exception":
            rets.append((frame.f_code.co_name, frame.f_lineno, arg[0].__name__, str(arg[1])[:300]))
        return local

    def tracer(frame, event, arg):
        return local if frame.f_code.co_name in WATCH and "_host_trace" in frame.f_code.co_filename else None

    sys.settrace(tracer)
    IN_RELOWER[0] = True
    try:
        out = _relower(self, family, args, variant, values)
    finally:
        sys.settrace(None)
        IN_RELOWER[0] = False
    LOG.append(("relower", id(self), out is not None, [r for r in rets if r[0] != "line"][-6:], [r[1] for r in rets if r[0] == "line"][-12:]))
    return out


R.HostTraceReplay._miss, R.HostTraceReplay._relower = miss, relower
sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "newkey.py"), *sys.argv[1:]]
g = runpy.run_path(sys.argv[0], run_name="__main__")
V = g["V"]
names = {id(e): k for k, e in V.entries.items()}
print("=== misses (entry, variants before, candidates, after relower fail, ints, shapes)")
for rec in LOG:
    print(rec[0], str(names.get(rec[1], rec[1]))[:60], *rec[2:])


print("=== new guards during relower:", len(STACKS), "stacks")
for st, (c, ex) in sorted(STACKS.items(), key=lambda kv: -kv[1][0])[:4]:
    print("--- count", c, "example", ex)
    print(st)


def gset(tape):
    return [str(x.expr) for x in tape.shape_env.guards]


def srcs(tape, exprs):
    env = tape.shape_env
    names = sorted({str(f) for x in exprs for f in getattr(x, "free_symbols", ())})
    return [(n, env.names.get(n), getattr(env.ctx.symbols.get(n), "hint", None)) for n in names]


for k, e in V.entries.items():
    vs = e.variants
    if len(vs) < 2:
        continue
    print("=== entry", str(k)[:80], "variants", len(vs))
    base = gset(vs[-1].tape)
    for i, v in enumerate(vs):
        cur = gset(v.tape)
        add, lack = [x for x in cur if x not in base], [x for x in base if x not in cur]
        print(f"  v{i} guards {len(cur)} learns {v.learns} +{len(add)} -{len(lack)}")
        for x in add[:25]:
            print("     +", x[:200])
        for x in lack[:25]:
            print("     -", x[:200])
        diff = [x.expr for x in v.tape.shape_env.guards if str(x.expr) in add or str(x.expr) in lack]
        for row in srcs(v.tape, diff)[:30]:
            print("     sym", row)
