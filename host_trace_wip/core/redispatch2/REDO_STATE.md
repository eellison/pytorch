# A redo's custom-op body reading module state (vLLM core task i): design note, for review

## The case
The adapter swaps the step's traced tensors into modules and the forward context while the traced function runs
(layer.kv_cache, the trtllm workspace, block tables), so the trace sees those reads as the step's arguments. A
redispatch re-runs one op's body alone, outside the step. The body then reads the real tensor, declines ("aten.slice.Tensor
of an untraced tensor with symbolic arguments"), and the call retraces. Repro: integration/vllm/probe/redo_module_state_repro.py
(module state: 2 traces, 0 redispatches; control with kv as an argument: 1 trace, 1 redispatch).

## Pick: option 1, resolve by storage identity (no new API)
At a redo, the fresh trace gets a resolver for the call's own tensor arguments. When the body hands an op a plain CUDA
tensor (not a traced one), the resolver replaces it with that argument's fresh stand-in (the argument's root and its
symbolic sizes, strides and offset, built as `_redo` builds any argument it reads) if exactly one argument matches:
- it has the same storage, as an object: `t.untyped_storage()._cdata`, the StorageImpl, so no address comparison;
- it has the same view: dtype, sizes, strides and storage offset equal to that argument's at the call.

Anything else still declines, and the call retraces as today (counted in retrace_causes). That covers no match, a
different view of the storage (kv_cache[layer] where the argument is the whole cache), and two arguments that both
match.

## Why it is sound
The variant's tape already rests on the fact this checks: during the trace, the swapped attribute was the traced
argument itself, so every launch the body recorded reads that argument's root at that argument's layout. The resolver
hands the redo exactly that tensor, symbolically, and only when the real tensor is the argument at this call: the same
storage object and the same view. So the redo sees what the trace saw, and its entry carries the same assumptions as
the trace's own launches, no more. The entry's guards come from the redo as usual, in the argument's symbols. A tensor
that is not provably the argument is never resolved, so nothing is guessed from addresses or shapes.

## Why not option 2 (the traced function registers attribute bindings)
- It needs a new API in the adapter (owner, attribute, argument index) and a hook in the redo to set and restore module
  attributes around the body.
- It is state the redo would mutate on user objects, under the replay's lock, while another thread may read them.
- It covers only the attributes registered, so a forward-context field the adapter forgot still declines.

Option 1 needs neither the adapter nor any mutation, and works for any route by which the body reaches the argument's
storage.

## Scope and implementation sketch
- `_redo`: map each tensor argument position of the call to its storage identity and view, and set
  `tr.resolve = lambda t: ...` on the fresh trace. A match builds the argument's fresh root and stand-in through the
  existing `root()`/`value()` (an argument root the op itself did not read becomes an input of the redo too).
- `_TraceMode.__torch_dispatch__` (one place, before an op's route): plain CUDA tensor leaves go through
  `current_trace().resolve` when it is set (only redo traces set it), else they are left as today.
- The dispatch memo (81) does not store a redo that resolved a tensor: the resolution is lazy, so the memo's root
  mapping by order would not hold.
- Test: the repro as a core test (module state: 1 trace + 1 redispatch, bitwise), plus a decline case (the state is a
  different view of the argument's storage: still retraces, counted).
