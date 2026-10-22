# Landing bundle for the core agent (learnfast, 10-15): splicing fix + redispatch cost

Order: the core agent's section_entries desync fix first (repro: learnfast/core_sec/test_section_flag_desync.py,
test_flipping_section_entries_off_before_a_new_form: zip ValueError at _host_trace_replay.py:1044 on step_ms10), then:

## 1. Delta 3: exact-overlap check_section  (core, ready)
- Patch: learnfast/core_sec.patch (base step_ms10; `patch -p1`): _host_trace_memory.check_section + new _lifetimes;
  test_cuda_host_trace_replay.py: split_op_late_output op, 2 new parametrized tests, one regex update.
- Rule: each base the section reaches (allocation, freed argument, its own temporaries) has a (made, freed) lifetime in
  seqs from the plan, per HostTraceVariant::allocate's three forms (eager order / run buffer / tensors-only run). A base
  whose lifetime does not cover the op's span is refused only if a non-dropped base made after its free, or freed
  before it is made, falls in the span. No OPSPAN widening. Bases outside the run keep the old liveness test; held /
  relocated allocations unchanged (always live).
- Tests: accepted where the old rule refused (input freed inside the op; bitwise, 1 trace / 1 redispatch / 1 section,
  eager and run_buffer); refused for overlap (output made after the input's free: "shares bytes with allocation ...",
  retraces, both memory forms). 5 of these fail on step_ms10 unpatched.
- Gate on step_ms10 + patch (learnfast/sec_snap, GPU 3, logs/secgate): replay 332, harvest 254, python_entry 640,
  cute 77, triton 67, aten 1761, opaque 53, memory 24, guards 13, inductor/test_cudagraph_host_trace 98: all PASS.

## 2. Refusal memo  (core, ready)
- Patch: learnfast/core_memo.patch (base step_ms10 + Delta 3; `patch -p1`): _Variant.refused_sections, recorded in
  _add_entries on a redispatch's check_section refusal as (site, entry predicate row) -> message; _redispatch refuses at
  once (same message, refusals count, _last_refusal) when the call fails that site's selector and holds that predicate;
  HostTraceReplay.refused_again counts them. Exact: the predicate is the op's own guards from the refused redo, so the
  redo would take the same branch to the same section.
- Test: test_a_refused_section_is_refused_again_without_a_redo (two dynamic ops; the first variant refuses at once with
  the same refusal and no redo; the second variant takes y's section): PASS on step_ms10 + Delta 3 + memo.
- Equality on the workload (prototype refusal_memo.py, 27B vt RANGE=1 decode + prefill2, memo off / on): traces 10 = 10
  (same token sequence), the same 16 successful redispatches in order, the same 3 refusal reasons; prefill2 262.7 ->
  231.1 s.
- Gate on step_ms10 + Delta 3 + memo (learnfast/memo_snap, logs/memogate): running.

## 3. RD2 ("section entries disabled" on mm_fp4 / attention): no core change
- RD2 was a redo baking per-step forward-context ints into section sizes (DESIGN_section_sizing.md). With Delta 1 (lifted
  per-step state bound per redo; prov_redo / diag27 prototype) nothing is baked for the GDN op; mm_fp4 and the attention
  decode op read only their arguments. The refusal itself was no_sections (adapter side), not a core bug.
- Evidence, sections on (Delta 3 copy), ours V, lifted state + fla overlay + cross-layer templates + memo, Triton GDN,
  every forward checked against eager twice: n27wf prefill part 97/97 bitwise (state bitwise), 208 sections, 3 traces;
  mixed part 210/210 bitwise, 272 sections, 14 traces. 9B 60..62 memcheck: 0 errors, but that run took no redispatch
  (mm pins per token count without bindings), so it does not exercise sections.
- To turn sections on in the reported config: drop no_sections from the adapters (r27 armV_r27/adapter.py import,
  MiMo's ARMV_REFUSE_SECTIONS) once 1 + the desync fix land. That is an adapter change, not core.

## 4. Cross-layer templates  (prototype; core port needs a design OK)
- Prototype: learnfast/tpl_layer.py on core's templates (step_ms10). Key-outs: a vLLM layer name (LayerName / str naming a
  no_compile_layers key) and a static-prefix argument's plain offset symbol (layers' views into one cache buffer); pointer
  slots shifted by the offsets' difference; one-time confirmation per (key, layer) with _assert_same_entry, mismatch ->
  full redos for that layer. GDN keys 48 -> 3 per class; 0 confirmation mismatches over 1,677 confirmations.
- Core port would make "an opaque value-type constant of the call" and "a plain declared offset" keyable with the
  confirmation; that is new behaviour in _reads / _instantiate (about 120 LOC + tests). Not written until approved.

## 5. Redispatch overhead (for the core agent)
- learnfast/notes/REDISPATCH_OVERHEAD.md: _for_instance 87 s of 370 s profiled, 18 s of it a per-instance frozenset over
  every input's root key (cache per lowering), transfer + conjunction 35 s; lower_entry 22 s; guard_sloc 11 s.
