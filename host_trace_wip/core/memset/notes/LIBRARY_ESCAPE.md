# A library body's Python decision escaping into Python state (open, for the user)

Status 10-20: decided C (the coordinator, deferred to by the user): redo_policy "inputs_only" (default) states the contract in
its docstring; P is the opt-in redo_policy "strict" (snap/wip43, for ms34). A verify-mode shadow retrace is a possible
follow-up.

## The hole
A library body (non-ATen operator or dispatch unit) runs Python at the trace. If that Python decides on a symbol
(`int(x.shape[0])`, `if s > 16:`) and writes the outcome into Python state that code outside the op reads, the decision's
guard is the body's own op guard. When it fails, a redispatch redoes that op alone; the readers keep the trace's value.
- coldstart/repro/escape_through_owned_guard.py: X stashes int(x.shape[0]), op Y reads it. wip42: wrong at sizes 16, 32.
- coldstart/repro/escape_to_user_code.py: the user's code reads it (`y + STASH["n"]`, a plain-int branch). wip42: wrong.
- Decisions in user code between ops are graph guards (they retrace): not affected.

## Options
- (P) refuse a redispatch whose failing own guard a library body's Python decided on (recorded or deduped there, outside an
  ATen op's C++ host). Evaluation snap wip44 (Env.python_decided, TORCH_HOST_TRACE_ESCAPE_RULE=P). Fixes all repros.
  - Suites: 40 unique tests break (replay 28, harvest 3, cute 4, triton 5; python_entry repeats replay/harvest; inductor,
    aten, guards, lower, opaque, memory 0): every Python kernel / tactic choice in a body (logs/escapeP_wip44).
  - 27B vt with P enforced (diag27 DIAG27_ESCAPE_P=enforce, partial 717 s): 537 of 537 redispatches refused (GDN core op
    420, flashinfer_mm_fp4 117; first marked guard a pin 292, a branch 245); decode range warm-up 311.8 s, 22 traces.
  - 8B: 0 (the adapter refuses library redos today).
- (T) redo every later library body too: dropped, misses the user-code escapes (escape_to_user_code.py; an ATen op's baked
  scalar and a plain-int branch are not re-run).
- (C) contract: a library body leaks no decision into Python state read outside the op; P an opt-in strict mode.
  - torch.library docs (repo d98f9246473) forbid global state only for register_autograd's setup_context / backward
    (torch/_library/custom_ops.py:681-682, torch/library.py:1365-1366); mutates_args "MUST be accurate" (tensors only,
    custom_ops.py:94-95); `_register_effectful_op` exists for "side effects that is not capturable by the schema"
    (library.py:590-591, 1290-1292).
  - Stock torch.compile (coldstart/repro/escape_torch_compile.py): op to op correct (bodies run every call); op to user code
    wrong / stale (Dynamo guards the global at trace time).
- Ppin (refuse only failing concretization pins): not sound for branch escapes; at least 292 of 537 27B refusals remain.
