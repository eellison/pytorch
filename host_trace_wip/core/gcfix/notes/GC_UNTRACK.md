# GC footprint of host traces (gcfix lane)

Tree: land/core/gcfix/tree (= snap/step_ms10 + patches/gc_untrack.py.diff; tree_ms8 = the same on step_ms8). C++: land/core/gcfix/build_cpp_gc1 (= build_cpp_ms8/src +
patches/gc_untrack.csrc.diff). Launcher: python_gc1.sh; GPU wrapper g3gc.sh / q.sh (gpu3.share.lock).
Repro: repro/gc_retain.py (4-layer bf16 decoder through HostTraceReplay; --static: one trace per size).

## What a trace left behind (step_ms8, before)

- About 27.5k GC-tracked objects alive per trace of the 4-layer decoder (6.9k per layer; 8B's 240k for 36 layers fits).
- 100% reachable from entry._families -> _Variant. About 62% is the IR Env (ctx.table hash-cons: Nodes and their arg
  tuples; ctx.memo with 3.5k keys); the rest is the tapes (unbound and bound), lowered and captured records, traced
  tensors.
- Most of it cannot be dropped: folds, redispatches, relowers, _eager_sites and _retrace_cause read and extend the
  lowering, the Env and the tapes after build; OpRec.call keeps traced tensors for redo.
- The Env was cyclic (Env._export -> _SympyExport.env): a dropped entry left every node to the cyclic collector.
- Cyclic garbage: 0 per trace in the repro; 28 per harvest learn, all from torch.utils._pytree.tree_flatten's
  recursive closure `helper` (function <-> its own cell, plus its evaluated annotations): 7 objects per call.

## The change

1. _SympyExport holds its Env by weakref (no Env cycle).
2. Ctx.drop_memos(): a slow call's touched variants drop the add/mul/compare memo and bounds memo (pure caches of
   interned nodes; `written` is setdefault, so recomputation is identical).
3. OpRec.redo of operator calls, jiterator launches and dispatch units are frozen dataclasses (eq=False keeps the
   redo-template key by identity), not closures.
4. Env.owner_sets (IR and sympy backends) hold frozensets; a record replaces its set instead of adding to it.
5. (dropped by the coordinator: pytree.tree_flatten's recursive closure; not reached on step_ms10's learn path.)
6. torch._C._host_trace_gc_untrack(roots, kinds, classify) (C++, PyObject_GC_UnTrack / PyObject_GC_IsTracked, the
   documented C API) and _host_trace_replay._gc_untrack / _gc_flags. At the end of each slow call (_settle, inside
   _gc_hold) the variants it built (_build) or extended (_add_entries: folds and redispatches) and the redo caches
   (self._dispatches) are walked; switch _host_trace_replay.gc_untrack (default True).

### Which objects are untracked (the audit)

An object is untracked only when it is *sealed*: its type is immutable or final (or it is a final container) and
every GC referent is sealed, an object already untracked, or of a clean type. So nothing mutable is reachable from an
untracked object: it cannot become part of a reference cycle and refcounting frees it as before.

- Immutable: tuple, frozenset, _host_trace_ir.Node (hash-consed, slots set once), IRSymNode (slots set once),
  torch.SymInt/SymFloat/SymBool, pytree TreeSpec (frozen dataclass), host-trace frozen dataclasses without a
  functools.cached_property (cached ones, LoweredTape and OpaqueBinding, write their __dict__ later: excluded).
- Final (nothing writes them once the slow call that made them ends): _Root, _InputRec, _IntInputRec, _AllocRec,
  _OutputRec, _IntOutputRec, OpRec (_host_trace_tape._GC_RECORDS; the only writes are inside the trace and
  Tape.__init__; bind_opaque and _eager_sites replace records and copy lists before writing). _TracedTensor is
  final but never untracked (GC_KEEP); its symbolic sizes/strides/offset/storage nbytes nodes are walked from C++.
- Containers (list, dict, set): untracked only when every reference to them (the reference count, less the walk's
  own) comes from immutable or final objects or other final containers: a record's sizes list, a call's kwargs.
  The containers folds, redispatches and relowers extend are held by the mutable hubs (Tape, _TapeLowering, Env,
  IntegerProgram, _Variant) and stay tracked: Tape.ops/allocs/launches, lowering.entry_guards/templates/reads/refused,
  program values/instructions, Env records/owners/owner_sets lists, ctx.table/nodes/memo, variant arms/folded/
  eager_sites/tried, self._dispatches.
- Clean (may be held by a sealed object; never untracked): types, modules, OpOverload(Packet), sympy expressions,
  and the Env (its reachable state is its IR, guard records and sympy export, which refer to nothing outside it).
- Everything else (functions, cells, foreign objects: CUDAGraph, CompiledKernel owners, cuda-bindings handles) is
  neither walked nor clean: whatever holds one stays tracked.

Types are allowed only when every class in the MRO is a heap type or object, so deallocation goes through
subtype_dealloc's PyObject_GC_UnTrack (never the _PyObject_GC_UNTRACK macro that asserts a tracked object).
The walk holds strong references to what it visits (classify runs Python and may switch threads).

7. Folded entries (variant.folded) keep their launches without the witness closure: it checks a launch at its
   capture only, and held the folded trace's whole Env (a fold left 860 objects, now 0-3).

## Numbers (gb300-box GPU 3, untimed share; host loaded, so gen-2 times are relative)

repro/gc_retain.py --variants 50 --static (4-layer decoder, 50 traced variants), on step_ms10:

| | tracked per trace | tracked after 50 | gen-2 (baseline without variants) | gen-2 due to the variants |
|---|---|---|---|---|
| step_ms10 | 27,537 | 1,790,670 | 677 ms (99 ms) | +578 ms |
| this patch, gc_untrack off | 23,309 | 1,579,370 | 633 ms (115 ms) | +518 ms |
| this patch | 1,016 | 464,691 | 197 ms (114 ms) | +83 ms |

Cyclic garbage per trace: 0 before and after (repro and TestGcFootprint); per harvest learn 28 on step_ms8 (the pytree closure, a fix that was dropped)
and 0 on step_ms10 either way (its learn path no longer flattens there). A dropped entry: before,
about 90k objects of 3 variants needed the cyclic collector (the Env cycle); after, refcounting frees them.
Trace call time unchanged (0.32 s both). What stays tracked per variant (about 1k for 4 layers): traced tensors
(284; never untracked), the mutable hubs (Tape, lowering, Env, program, variant and their lists/dicts), Triton
witness closures and foreign objects (CUDAGraph, CompiledKernel owners). Of the remaining +83 ms, the hubs' large
containers (ctx.table/nodes about 2.9k entries, program instructions/values about 2.2k, per 4-layer variant) are
about half: each gen-2 visits their elements. See "Next step" below.

Pre-existing, not from this patch: each run of 4 entries (8 variants) leaves 8-36 KB of Python allocations
(converging over 10 runs: caches filling), identical on step_ms8 without the patch; the drop test compares
untracking on vs off for that reason.

## Next step (not now; coordinator 2026-10-15)

What stays tracked per variant is O(model size) through two things: the mutable hubs' large containers, and the
traced tensors.

- Hubs: Env ctx.table / ctx.nodes / ctx.symbols / unique_ids / records, IRLowering._memo / guard_rows, IntegerProgram
  instructions / values, Tape lists. Each gen-2 visits every element (about 12k per 4-layer variant), so this is about
  half the remaining +83 ms. Option A: freeze them after the build (tuples, frozen dicts as sorted tuples or a C++
  table) and copy them back to mutable (re-tracked) on the first fold/redispatch/relower that extends the variant.
  Folds and redispatches are rare after warm-up, so most variants stay frozen. The cost is a copy at the first
  extension, plus an "is frozen" check at each mutation site (Ctx.mk, IntegerProgram.emit, Env._record,
  lowering.entry_guards); it needs a single choke point per hub. Option B: keep the IR table and the program in C++
  (the C++ program already exists, _HostTraceProgram). Then the Python side holds one handle per variant. That is the
  "C++ state + bounded Python" end state, but it means porting Ctx/IRLowering.
- Traced tensors (about 70 per layer): never untracked. Gen-2 traverses each one (THPVariable_traverse). They are
  needed by redispatch's redo (OpRec.call) and fold comparisons. Replacing them after the build with a frozen
  metadata record (root, sizes, strides, offset, dtype), rebuilt into a _TracedTensor only when a redo needs one,
  would make OpRec and its tuples fully untrackable.
- Small: CapturedSegment holds a CUDAGraph (foreign; about 20 tracked objects with its hook dicts), and Triton
  witness closures on unfolded variants are kept because a relower captures again.
- Gen-1 outliers (MiMo lane, 2026-10-15, logged here at the coordinator's request; GC lane stood down until after the talk
  numbers): ours V at bs-1 decode on MiMo (gb300-box GPU 0, untimed, mem_eager's diag adapter): 2 gen-1 collections per ~1000
  replayed calls at ~0.7 s each (1.41 s over 1012 calls; a gc.collect(1) probe measured 1.01 s), i.e. ~1.4 ms/step on the
  mean and two ~0.7 s step stalls; gen-0 was 14 collections, 2.5 ms total. With ARMV_GC_FREEZE_AFTER_WARM the heap is
  frozen after warm-up, so the survivors are per-step garbage that outlives gen 0 (a large young generation between gen-1
  passes). Data: land/core/trtllm_moe_cpp/out/prof_bs1/gcon.hyb.gc.json, gcon.hyb.micro.json (gc on) vs gcoff.* (gc off).
