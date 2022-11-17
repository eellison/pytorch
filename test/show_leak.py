import torch
import torch._dynamo
import torch._inductor
import refcycle
import weakref

foo_dies = False

def test_closure():
    @torch._dynamo.optimize()
    def foo(x):
        return x + 1 + 2

    def finalize():
        global foo_dies
        foo_dies = True

    weakref.finalize(foo, finalize)

    return foo(torch.rand([4], device="cuda"))

out = test_closure()
snapshot = refcycle.snapshot()
debug_objs = [t for t in snapshot if isinstance(t, torch._inductor.compile_fx.DebugWrapper)]
out2 = test_closure()
snapshot2 = refcycle.snapshot()
debug_objs2 = [t for t in snapshot if isinstance(t, torch._inductor.compile_fx.DebugWrapper)]
assert debug_objs[0] is debug_objs2[0]

assert foo_dies == True
assert len(debug_objs) == 1
debug_obj = debug_objs[0]
ancestors = snapshot.ancestors(debug_obj, generations=3)