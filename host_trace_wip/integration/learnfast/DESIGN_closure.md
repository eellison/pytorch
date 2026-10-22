# Design note (learnfast -> coordinator): closing over declared fixed storages by address (opt-in, default off)

Status: DRAFT for the coordinator. It replaces rules 1, 2, 4a and 4b of DESIGN_redo_resolve.md. Rule 3 (CuTe
_roots_of includes a redo's outer roots) stays as a separate one-line fix.

User's decision: "this is basically what they already do. we are not relaxing any existing safety". vLLM's CUDA graphs
bake weight and KV pointers in and assume they never move after warm-up. The vLLM integration replaces those graphs, so
for vLLM, closing over the same storages adds no new assumption. In core the switch is an option, off by default; the
vLLM adapters turn it on.

## API (one opt-in; default off)
- `torch.cuda._host_trace.close_over(modules=(), tensors=())` (name open), called by the integration before the
  first trace. It declares storages fixed by address:
  - every parameter's and buffer's storage of `modules`;
  - every tensor held as a module attribute, including lists of them (vLLM: each layer's kv_cache entries, the mamba
    conv/ssm state, pre-made views such as a conv1d weight view);
  - `tensors`: persistent runner buffers vLLM allocates once and its own graphs bake in (input buffers, metadata
    buffers such as causal_conv1d's batch_ptr).
- With the switch off, behaviour is unchanged.

## Semantics
1. Base, not layout. A tensor on a declared storage, read inside a trace or a redo, gets a constant base pointer: the
   storage's address. Its offset, sizes and strides are traced as for any tensor.
   - Inside the traced code, a view made from a declared tensor is an ordinary traced view: its metadata is symbolic
     where it derives from step inputs, guarded otherwise.
   - A tensor reached from state that was made outside the trace: if it is a declared object (a pre-made view held
     in module state), it traces as constant base + its layout, and the layout is guarded at the trace. The hooks
     below keep that layout fixed.
   - Any other plain tensor on a declared storage is a per-step object that state handed over: for example a
     metadata slice `buf[:num_reqs + 1]` made by the runner each step. It has no fixed layout. It must arrive as a
     traced argument, or the read declines (retrace). The trace must never bake its layout as constants.
   - So the match is: declared object -> constant base + guarded layout; plain tensor that IS one of the call's
     tensor arguments (object identity, e.g. per-step metadata the body reads from the forward context) -> that
     argument's stand-in (traced, symbolic); anything else -> decline.
2. Invalidation with no per-call check (the user's rule):
   - Hooked on declared modules: Module.__setattr__ and register_parameter / register_buffer.
   - Hooked on declared tensors: `.data =`, `set_`, `resize_`, and in-place metadata ops (as_strided_, transpose_).
   - Any of these marks every variant that closed over that storage or tensor stale; its next call retraces.
   - In-place value updates (optimizer steps, in-place weight sync, KV writes) stay valid by design, as in vLLM.
3. Out of contract: raw storage pointer swaps from C++ that bypass the hooks. vLLM's own graphs carry the same
   exposure. A debug mode (off) re-checks recorded bases and layouts per call, for tests.

## Required tests (core)
- A per-step view over a declared storage with a varying length, reached by an op's body from state (not an argument).
  At a different length it must decline and retrace; never replay a stale baked layout.
- The same view passed as an argument replays bitwise at the other length.
- A declared pre-made view held in module state replays bitwise. Reassigning that attribute invalidates the variant.
- `resize_` / `set_` / `.data =` on a declared tensor, and parameter reassignment: each invalidates; the next call
  retraces and stays bitwise.
- In-place weight update: still valid, and outputs follow the new values bitwise.

## Prototype results (27B, GPU 2, monkeypatches in diag27.py; no core change)
- ccl: closure by object over the entry's static prefix, plus rule 3; --check over sizes 2..257: all rows bitwise
  (output and state, eager-twice clean). 8 traces vs 9 for rules 1-4b: the mm_fp4 pow-2 flips redispatch. The GDN
  classes still trace, because the GDN section path is a separate track (crash past 64 tokens, OPSPAN rejected).
- dcl3 (declared set walked from the runner and model by object): the conv1d weight now resolves. The GDN redo then
  declines on `_causal_conv1d_fwd_kernel: argument cache_indices_ptr is a tensor the trace does not track`.
  cache_indices is per-step mamba metadata: the op's body reads it from the forward context, not from its arguments.
  Closure cannot and must not cover it, since its layout varies per step. The call does pass that same object in its
  argument pytree (ARMV_METADATA=eager). So the design keeps one narrow resolution: a plain tensor that IS (by object)
  one of the current call's tensor arguments becomes that argument's stand-in, with symbolic layout. There is no
  storage/view equality and no tie-break. Anything else that is not declared declines.
- dcl: the conv1d weight reached the Triton launch as an object the static-prefix match did not hold. The adapter lifts
  `.data`-style aliases, not the module's own Parameter objects. A real declaration registers the module's own
  objects. dcl3 (declared set = tensors walked from the runner and model by object) is queued.
- The storage-only prototype (dcl2, any view of a declared storage taken as constant layout) is withdrawn: it would bake
  the layout of per-step views (rule 1 above forbids that).

## What closure cannot cover
- Per-step tensors that op bodies read from forward-context state instead of their arguments. They decline, which is
  correct but not a fix. The integration can pass them as op arguments instead.
- Buffers grown in place: batch_ptr.resize_ in causal_conv1d's grid callback. Covered, because resize_ invalidates.
  The cost is a retrace when vLLM grows it, which happens rarely, at new maxima.

## Expected wins
- Redispatch of library-body custom ops without RD5's storage/view rule or any tie-break.
- The static prefix (1916 lifted arguments on 27B, 473 on 8B) drops out of per-call argument binding. Not measured
  yet; measure it in the core prototype.
