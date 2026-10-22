# Validation shim (r27): patches/cute_arith/cute_arith_tuple_iter.patch's interpreter change applied to the snapshot's
# torch.cuda._host_trace_cute by wrapping _Eval.op, before the patch is merged (checks; labelled timing runs).
from torch.cuda import _host_trace_cute as htc

_orig_op = htc._Eval.op


def _op(self, op):
    if op.name in ("cute.make_arith_tuple_iter", "cute.deref_arith_tuple_iter"):
        a = self.operand(op, 0)
        if not all(type(x) is int for x in htc._flat(a)):
            raise htc._Undescribed(f"{op.name} of the coordinate {a}")
        return a
    if op.name == "cute.make_identity_tensor":
        a = self.operand(op, 0)
        pat = self.program.described(op.types[0], "layout")
        shape, fixed = htc._flat(a), htc._flat(pat[0])
        if len(shape) != len(fixed):
            raise htc._Undescribed(f"{op.name} of {a} into {op.types[0]}")
        leaves = [s if htc._dynamic(x) else x for s, x in zip(shape, fixed)]
        return {"ptr": htc._unflatten(pat[0], [0] * len(fixed)), "shape": htc._unflatten(pat[0], leaves), "stride": pat[1]}
    return _orig_op(self, op)


htc._Eval.op = _op
