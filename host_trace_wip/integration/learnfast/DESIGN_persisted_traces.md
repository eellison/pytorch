# Design note (learnfast -> coordinator): persisted traces, so a cold start loads variants instead of tracing

Status: design only, nothing built. It needs the user's review (new core API and file format).

## Why
- 8B cold start (repeat 2, GPU 2, load ~80): our CUDA-graph warm-up is the 3 traces plus the cuBLAS learn.
  - Traces: 2.2 / 1.8 / 1.8 s on the compiled forward and 3.6 / 3.2 / 3.0 s on the eager forward.
  - Builds: 0.27-0.36 s each.
  - Persisting the traces leaves the builds: about 1 s instead of 5.7 s (compiled) or 9.8 s (eager).
- 27B (d27a): about 22 s per trace vs 1.4 s per build, and 3 to 77+ traces depending on the range.
- A cold start after the first-ever run then needs neither a trace nor a learn. Bindings already persist (0.1 s load
  for 5,120 keys, 1.5 s for 66,560).

## What a persisted variant holds (everything the build reads; nothing the trace recomputes)
- The family key:
  - the argument contract (argument_contract: kinds, dtypes, devices, the checked global state);
  - the entry's static-prefix layout, or under closure the declared storages' identities. Addresses are not
    persisted: they are re-bound at load (see checks).
- The tape:
  - ops, launches and allocations;
  - each KernelLaunch's function identity. Not a handle; one of: (library file, demangled symbol) for ATen/vLLM _C
    kernels; Triton (cache key, cubin sha256); CuTe (compiled object sha256); cuBLAS as today's binding names. All
    are resolved to CUfunctions at load.
  - grid, block, smem, attributes, PDL;
  - the parameter layout and every slot as an IR expression over the trace's symbols;
  - memcpy and memset records; host steps; eager steps with their reasons.
- The guards:
  - the ShapeEnv IR records (graph guards, op-owned guards with owner sets, selectors' rows);
  - the redispatch entries and templates the variant gained (fold or redispatch), with their predicates.
- Keyed sites: key formulas, roles, scratch. The bindings themselves stay in the bindings file, keyed as today.
- The memory plan inputs (allocation sizes as rows). The plan itself is recomputed at load by plan_memory: it is cheap
  and deterministic.
- Not persisted: CUDA graphs and execs (rebuilt by _build at load), arenas, addresses, the Python callable.

## Load-time checks (the bindings' _meta plus more); any mismatch -> ignore the file and trace
- Exactly as for bindings: GPU name/SM/SM count, driver, torch version and git hash, CUDA library files (name + size),
  and the caller's `versions`.
- Code: sha256 of every Python source file whose frames recorded a launch or a guard (the trace knows its SLocs), plus
  the installed versions of vLLM, FlashInfer, Triton and the CuTe DSL. The trace froze Python-level decisions (config
  flags, branches on Python state) that no guard records. Hashing the code that made them is the only sound key
  short of re-tracing.
- Config: a hash of the integration's declared config (vLLM config + model config + adapter env flags), supplied by the
  caller as `versions`, as the bindings do.
- Kernels: each function identity must resolve. Triton/CuTe objects must be found by hash in their caches. ATen/vLLM
  symbols must resolve in the loaded libraries. Otherwise the file is ignored.
- Library state: library_state() and the torch backend flags captured in the contract must equal the current ones.
- First call after load:
  - the native evaluation must hold the call, i.e. the guards are re-evaluated; nothing is trusted from the file but
    the formulas;
  - an optional verify-on-load mode (default off, used by tests and the first deployment) checks the first replay of
    each variant against eager bitwise, like --check.

## Soundness argument
- A persisted variant is a function of the traced code, its inputs' contract, library state and kernel binaries.
- The checks pin all of these. Within them, re-running the trace would produce the same tape: tracing reads no hint
  (the no-hint-read rule), so the tape is a function of those inputs.
- Guards are still evaluated per call as today, so a call outside the persisted variants' guards traces as usual. A
  stale file can only be ignored, never replayed.

## Interaction with closure (DESIGN_closure.md)
- Declared storages' addresses differ between processes. The persisted form records declared tensors by
  declaration identity (module path + parameter/buffer/attribute name, or the integration's tensor key), never by
  address.
- At load the integration's close_over() re-declares them, and the build binds the current addresses as constants.

## Estimated win
- 8B, 3 variants: compiled 5.7 -> ~1.0 s, eager 9.8 -> ~0.9 s, plus deserialization (expected well under 1 s;
  unmeasured).
- 27B: per variant ~22 s -> ~1.4 s.
- Combined with persisted bindings, a later cold start's CUDA-graph warm-up for 8B is about 1-2 s, vs vLLM's capture
  passes from the warm logs: default 7 + 18 = 25 s, no-compile 1 + 6 = 7 s, breakable 49 + 55 = 104 s.
