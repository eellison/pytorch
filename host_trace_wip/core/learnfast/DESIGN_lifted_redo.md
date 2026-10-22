# Design delta (learnfast -> coordinator, 1 page): op-local redo over lifted per-step state

The user's design (DECISIONS / memory 10-14/15): per-step state an op body reads becomes an explicit op input, by patching
its holder to hold a SymInt at trace time; then the inputs-only criterion admits the op and the redo binds those symbols
per step. Credit: the user.

## What already holds at trace time (no change)
The vLLM adapters (r27 armV, MiMo armV) already patch the holders: `_md_rebuild` rebuilds every attention / GDN metadata
object with its CUDA tensors -> the step's traced tensor inputs and its lifted ints (`_LIFT`) -> the step's SymInt inputs,
and installs it with `override_forward_context` (plus slot_mapping). opdeps' "reads_outside" rows (MiMo
unified_attention int_input 1115-1125 / tensor_input 1095-1097, 27B GDN int_input 1959/1968/1977, tensor_input
1921-1925) are exactly those lifted positions: every per-step value those ops read is already a SymInt / traced tensor
in the trace. No field needs to be added for MiMo attention or 27B GDN (any other read would be a constant of the
entry's key, the trace contract as today).

## Delta 1 (prov_redo + the adapter's own helpers; no core change)
- opdeps: an extra symbol whose origin is a lifted position (the step's metadata tensors and ints: positions
  [n_p + n_a + 2, len - n_vp) of the adapter's flat step) classifies as "lifted"; verdict `args_or_lifted`.
- prov_redo (PROV_LIFT=1): admits args_only, declared (closure on), and args_or_lifted ops; the confirmation at the
  trace's values runs with the binding below.
- Binding per step: the redo runs under the adapter's forward context rebuilt by `_md_rebuild` from the redo's stand-ins
  for the lifted positions (fresh symbols bound to the call's values), i.e. the holders hold this call's symbols. No
  parameter / attribute swap (closure resolves declared storages), no step-function call. Prototype hook: the existing
  `redo_resolves_arguments` resolver (as in my redo_context prototype).
- Escaped reads: (a) a body that reads a copy instead of the holder reads either the original trace's SymInt (the redo
  refuses: a symbol of no argument) or a concrete int (the confirmation sees a constant where the trace has a symbol:
  the op is refused); (b) my reading of "any guard on a lifted symbol that isn't owned by the op must still refuse":
  a failing guard on a lifted symbol that is graph-level (model code outside op bodies read it) keeps retracing (as
  today), and a failing guard that *pins* a lifted symbol (s == c: a concretized value, which can escape into a dict /
  cache another op reads) refuses the redispatch even when the op owns it. Residual (stated, not checked): an op body
  whose Python branches on a lifted value and writes Python state another op reads; the same caveat as every library-
  op redo. Please correct (b) if you meant something else.
- Sections stay refused in reported configs (no_sections / ARMV_REFUSE_SECTIONS), as today.

## Delta 2 (core, NOT coded until you approve): keyed-site creation inside a redo (Flash-Next low_latency_gemm)
- Today `_redo` raises FoldRefused("another keyed site") when the redo makes an opaque call no site of the op binds.
- Change: an opaque call in a redo becomes a new keyed site of the variant (the provider's key over the call's rows),
  the entry's launches exclude its nodes as for existing sites, and the site's rows are added to the program
  (lower_entry + bind_opaque paths). ~150-250 LOC in _host_trace_redispatch / lower_tape + test (an op whose untaken
  branch calls mm). Needs your OK (new core behaviour).

## Delta 3 (core, NOT coded until you approve): exact-overlap section check (MiMo moe_forward liveness)
- check_section refuses a section whose base is an allocation made inside the op and "not live over its op's
  launches". Replace that liveness test by an exact one: the section's temporaries and reads overlap no live allocation
  over the section's launch span (interval test on the plan), no OPSPAN widening (+42% plan peak rejected earlier).
  ~100 LOC in _host_trace_memory + test. Needs your OK; it also only matters once sections are allowed in a reported
  config (the RD2 cause, baked forward-context ints, is gone under Delta 1).

## Order / measurement
Delta 1 first: MiMo (decode-curve --check, then the cb prewarm pass: 346 traces baseline), 27B (n27wf: 32 traces),
traces + warm-up seconds + bitwise on every row, then the 8B-lane startup harness comparison. Untimed GPU only.

## Refinement of rule (b), 10-15 (coordinator-approved, positive form)
- A failing pin on a lifted symbol is exempt only if every record of that guard (new or deduplicated) was made inside a
  dispatched ATen op's C++ symbolic evaluation (torch._C._cuda_hostTraceRedispatch / _cuda_hostTraceRedispatchOp on the
  stack), and the guard is the redone op's own. Such a guard is a kernel-internal decision (a broadcast's size-1 check, a
  CompositeExplicitAutograd body's host); the redo re-evaluates it at the call and check_section checks the op's outputs.
  A record from Python (library code reading the int, or the step's own code) keeps the refusal.
- Found from the 27B GDN pin (s == 1 on a lifted per-step count): its stack is torch/_ops redispatch -> guard_bool inside
  an ATen op in the GDN body, no library frame. Prototype: diag27 PROV_PIN_ATEN=1 (GUARD_WHERE marks), prov_redo
  _pinned_lifted. Test: learnfast/tests/test_pin_where.py.
