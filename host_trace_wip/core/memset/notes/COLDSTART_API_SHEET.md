# Cold-start API sheet (approved by the user 10-20; ms34): op-local redo over lifted per-step state

The concepts are the user's (op-owned guards with a local redo; lifting global state into op inputs; closure "not relaxing
any existing safety"). This sheet names the API that carries them into core. Every item but 1's default is opt-in. The only
callers are the vLLM integration adapters (r27 armV_r27, MiMo armV, integration armV_stock_fast_m4). Landed in
snap/step_ms34 (10-20), tests in test_cuda_host_trace_replay.py and test_cuda_host_trace_cute.py.

## 1. `HostTraceReplay(fn, ..., redo_policy="inputs_only")`
Approved 10-20; escape decision C (the coordinator, deferred to by the user). Values "inputs_only" (default), "strict",
"refuse". `"all"` is gone: no value restores step_ms32's redo of every library body, which is unsound.
Contract (in the redo_policy docstring): a library body must not leak decisions into Python state that code outside the op
reads (another op's body, fn's own code); its effects flow through its outputs only.
- `"inputs_only"` (default; core since step_ms33 without the kwarg): a library body (a non-ATen operator or a dispatch
  unit, whose Python body runs again in a redo) is redone only if
  (a) statically, every free symbol of its records (launches, allocations, the guards it owns) is one of its own call's,
      its own allocations', a tensor's (tracked: resolved or declined), or a `lifted` position's when `redo_context` is
      given (item 2); and
  (b) on its first redispatch, a confirmation (the op redone once at its own trace call: arguments rebuilt from the
      trace's, rows evaluated by a Python pass, no hint reads; inside `redo_context` where it applies) reproduces its
      recorded launches launch for launch, and records no guard beyond the trace's except ones over its own symbols (and
      the static prefix's layouts and bases; see "Not API").
  Cached per traced op (variant x op index), checked before another op's redo template is taken. Within one variant, an op
  whose redo template key (_reads) is a confirmed op's, and whose records (every launch field, allocation and owned guard,
  pins included; constants exactly) are that op's with the read-walk roots renamed by position (a bijection) and the static
  prefix's layouts at their values, shares that confirmation (the property redo templates rest on); never across variants
  or processes. Counters: _LIBRARY_COUNTS "confirmations" (own) and "shared".
  Labelled refusals: "its records read <s>, which its call does not carry", "..., lifted state, with no redo_context",
  "confirmation: <first differing field>".
- `"refuse"`: a redispatch never redoes a library body (the call retraces); what the vLLM adapters do today through
  learnfast's no_lib_redo.
- `"all"` (step_ms32's behaviour, redo every library body): removed, unsound (repros coldstart/repro/lifted_redo*.py).
- `"strict"` (opt-in): `"inputs_only"` plus: a redispatch whose failing own guard a library body's Python decided on (recorded
  or deduped there, outside an ATen op's C++ host; Env.python_decided) is refused ("redo_policy strict: ..."), so a body that
  breaks the contract retraces instead of leaving stale values. Sound without the contract; costs every Python kernel /
  tactic choice in a body (40 core tests' redos; 537 of 537 redispatches on 27B: GDN 420, FP4 GEMM 117).
  Test: test_a_decision_a_library_body_leaks_into_python_state (stash read by another op, by fn's scalar, by fn's branch:
  stale under "inputs_only", bitwise under "strict"). notes/LIBRARY_ESCAPE.md.
- Possible follow-up (not now): a verify mode that shadow-retraces after a library redo and compares, to catch contract
  breaks in tests.

## 2. `HostTraceReplay(fn, ..., lifted=None)`
- `None` (default): no lifted positions.
- A collection of positions in fn's flat arguments that carry per-step state the integration lifted out of its holders
  (vLLM: the step's attention / GDN metadata tensors and lifted ints). Under `inputs_only` with `redo_context`, their
  symbols count as an op's inputs. Declared by the caller, never inferred.

## 3. `HostTraceReplay(fn, ..., redo_context=None)`
- `None` (default): an op whose records read a lifted symbol refuses ("lifted state, with no redo_context").
- `Callable[[RedoCall], ContextManager]`. Core calls it only around the redo (and the confirmation) of a library body whose
  records read a lifted symbol (lazy), with `RedoCall(args, func, rank)`:
  - `args`: fn's flat arguments, the redo's symbolic stand-ins at the `lifted` positions, the call's own objects elsewhere;
  - `func`: the op's operator;
  - `rank`: the op's index among the tape's ops of the same `func` (vLLM's MoE ops find their layer by such a counter).
  The context manager makes the integration's holders hold `args`' stand-ins while the redo runs and restores them on exit
  (vLLM: `_md_rebuild` + `override_forward_context`, and the MoE counter at `rank`). A body that reads a copy instead of
  the holder fails (a) or (b) and refuses. Such a redo's stand-ins are part of what it read: another op takes its dispatch
  (the family dispatch cache) only where it read the same, stand-ins included.
- `rank` is tracked: a context that reads it makes the op's redo its own (no redo template, family dispatch or shared
  confirmation from or to it); ops whose context does not read it share as any op does.
- Assumption (in the docstring, next to item 1's contract): a context takes the same path for the same operator and argument
  structure, apart from what it reads through tracked fields such as `rank`.

## 4. `torch.cuda._host_trace.close_over(modules=(), tensors=()) -> Closure`
- Declares fixed every parameter and buffer of `modules` (and submodules), every tensor held as one of their attributes or
  in a list / tuple attribute (KV caches, state, pre-made views), and `tensors`. `Closure.release()` undeclares. Opt-in
  process state while a Closure is held.
- In a redo, a declared tensor on the storage of a static-prefix argument (the same object every call) is read at its own
  layout (constants) on that argument's root; where several static-prefix arguments share the storage, the one the op's
  recorded launches read. The confirmation compares values with the static prefix's layout symbols at their values (the
  static prefix's layouts are the same every call, as redo templates already key them). Any other plain tensor declines.
- A declared tensor that is itself a static-prefix argument resolves to that argument by identity, with or without
  `redo_resolves_arguments` (item 5). A redo that resolved any tensor (declared or by item 5) is never stored in the family
  dispatch cache (core's existing rule for resolved reads).
- Contract as the static prefix's: changing a declared tensor (set_, .data =, a layout change, a reassigned attribute) needs
  release() and statics_changed(). (Changed from the earlier draft: no invalidation hooks, no constant-address roots.)
- Default: nothing declared.

## 5. `torch.cuda._host_trace.redo_resolves_arguments` (exists, RD5; narrowed and extended)
- Resolution by object identity only: a plain tensor a redo's body reaches that IS one of the call's tensor arguments
  becomes that argument's stand-in at its input record's layout (no storage / view equality, no tie-break). It applies at
  the torch-function layer (before autograd), at the Triton launch intercept, in record_launch and at the CuTe DSL launch
  intercept.
- Default unchanged (off).

## Not API (internal, with tests), changes a reviewer should know
- CuTe `_roots_of` also searches a redo's outer roots.
- A redispatch runs under the trace's builtin min / max handling (trace_builtin_minmax), so a body's `min(s, 24)` redoes as
  the trace recorded it (sym_min), not as a branch (test_a_library_body_redoes_with_the_traces_builtin_min; step_ms33
  refuses it).
- With a static prefix, every library-body confirmation (not only closure reads) compares launch values with the static
  prefix's layout symbols at their values, and accepts extra guards over the static prefix's layouts and bases: those are
  the same every call under the static prefix's contract (redo templates already key them by value).
- `_reads_lifted` (which ops get redo_context) counts an op's allocations' sizes too, so a body whose only lifted read is a
  workspace size gets its context rather than a refused confirmation.
- The redo's resolver runs at the torch-function layer through a TorchFunctionMode active only during a redo that has one
  (redo_resolves_arguments on, or a Closure held): no cost on traces or replays.
- Env.python_decided (the mark "strict" reads) is recorded on every IR trace: one set insert per guard a library body's Python records or dedupes; nothing on replays.

## Stays in the integration (labelled overlays, not core)
- fla_symlen (flash-linear-attention chunked ops use `.shape[0]` instead of `len()`; `len()` must return a Python int).
- The adapters turning 1-5 on, and sections on with eager-order memory.

Cross-layer template key-out (tpl_layer): CORRECTION 10-19: the "dropped" decision rested on an invalid ablation. KEYOUT=0 set
TPL_LAYER_KEYOUT, which only tpl_closure reads; on step_ms32 diag27 loads tpl_layer, which always keys out. The "no key-out" run
(146.0 s ready) was the vt configuration; vt r2 gives 151.5 s, and vt r1's 179.2 s was a cold first round (init 37.4 s vs 25-26 s).
Key-out's effect is unmeasured; a valid ablation is DIAG27_TPL=0 (no tpl_layer). Held for the user: the unread unpinned CPU
argument kind (X1, ~30 s of warm-up and ~55 s to steady on 27B).

## User decisions 2026-10-22 (via the coordinator)
- APPROVED: X1, the unread unpinned CPU argument kind ("a CPU tensor passed in explicitly that the trace never reads, accepted at any
  shape"): no entry split on its shape; a CPU tensor the trace reads (.item(), indexing) still splits or pins as today. Lands as ms39
  after ms38 (the ms14 draft). Adapters: ARMV_PIN_UNPINNED replaced by the core kind (27B and Flash-Next); the Flash-Next range
  warm-up's request-count sweep shrinks accordingly. Measure: Flash-Next entries and warm-up per window (expect ~17 -> 3 entries,
  ~11 min less), 27B warm-up / ready-to-steady.
- ms40, CHANGED by the user (later on 10-22): "LayerName by role" is DROPPED. Instead the layer lookup moves out of the op: the caller
  resolves what the body reads through the name (KV cache, GDN state, per-layer metadata) and passes them as explicit tensor
  arguments, so the ops are identical across layers by construction and no name token is keyed (the user's lift-global-state-into-
  inputs idea applied to the per-layer selection). Steps: (1) read-only assessment: for the GDN op (and the trtllm decode op), list
  everything the body reaches through layer_name (tensors, metadata objects, scalars) and whether it is per-step or per-layer;
  (2) a sibling op in our arm (labelled overlay, as the Flash-Next ones: same kernels, same outputs) taking those as explicit
  arguments, the model call site patched in our arm only; (3) keep the bundled (a) static-layout-by-role key change (per-layer state
  offsets still differ); (4) measure on 27B: GDN confirmations own / shared, warm-up, the full gate. Upstream note (STATUS.md): the
  clean vLLM form is explicit tensors instead of the name lookup; vLLM switched to names for piecewise-compile graph size, which host
  tracing does not need. Order: ms38, ms39 (X1), then this.
- OUT for now: trace persistence across processes (memset/notes/TRACE_PERSISTENCE.md stays a note; no work on it).
- Each lands with the full gate (suites, parity8b, fn_gate, 27B churn 1245 + run27wf, 8B churn 9022).
- ms39 (X1) read semantics, coordinator 10-22: (b). The unread kind applies only while the trace never reads the tensor's bytes; on a
  read the kind is declined for that trace and today's handling applies (shape key; CPU-scalar per-call records where the read feeds a
  launch; pins / guards otherwise). No eager-step path (a graph break per replay and a second mechanism for what the per-call CPU-scalar
  records and op-owned guards cover). Diagnostics name the decision: "pN read at op K: unread kind declined". Tests: the draft's
  eager-step test becomes a decline test; plus an unread tensor at 5 shapes -> 1 entry, 0 retraces.
- ms39 LANDED 10-22 15:4x: snap/step_ms39 (X1, the unread unpinned CPU argument kind, read semantics (b)); Flash-Next entries 44 -> 3,
  warm-up -8.8 min; gate green (parity8b in shared mode, 1.028x; an exclusive run owed in the next quiet window); 27B timing pending.
