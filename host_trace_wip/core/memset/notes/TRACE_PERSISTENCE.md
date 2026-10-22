# Persisting traces across processes (design note, core agent 10-21; no code; for the user via the coordinator)

Caches are a bonus, never relied on: the no-cache path stays the tested path, a cache is used all-or-nothing, a load failure is
logged and never fatal, and a loaded trace is treated exactly as a fresh trace's first replay.

## Why it is the biggest cold-start lever left
- 27B (quiet host, ms37): engine ready 249-267 s = init 20-29 s + warm-up 215-223 s; inside the warm-up: traces 37 s, redispatches
  116-121 s (library confirmations 32-37 s + entry builds), the rest replays and engine steps. The prototypes' 149-152 s is the bar.
- Flash-Next (IFX_FN.md): the range warm-up costs 1092 s (C=16) and 943 s (C=64) per process, ~1500 s of it tracing (95 + 75
  traces at 8-16 s each; a 2500-launch step traces slowly), the untimed passes' first replays are cheap (no step over 0.15 s that
  is not a trace in t_ifxattr2_v's pass rows), so a persisted trace set turns the warm-up into a load.
- It does not depend on the held decisions (X1, LayerName by role, cross-layer key-out); it composes with them.

## What a persisted trace must contain (per adapter entry key, every fully traced variant)
1. The tape: ops (operator by schema name / overload, call structure and constants, kind, outputs), inputs (position, layout as
   IR nodes, extent length, cow, dtype, device), allocations (sizes / strides nodes, dtype, q), launches (KernelLaunch: name, the
   kernel's persistent identity (3), parameter layout, grid / block / smem / slots as IR nodes, roots by name, fields, TMA
   descriptors, launch attributes, pointer slots, parameter images, programmatic, rng fields, CPU-scalar sources), Memcpy /
   Memset records, the IR Env (the interned node table, symbols with their hints, records = guards with their written forms,
   python_decided marks), owner sets (op-owned guards), keyed sites (op, provider, key, call), checks. Addresses are already
   process-independent (p / a / e roots with symbolic bases, placeholders patched at replay; pinned host roots likewise).
2. Derived state, persisted to save time but recomputable from (1) + the body code: the lowered program (rows, compiled
   IntegerProgram, entry guards, section placements), redo entries and templates (Template rows / roles / sections / pins),
   library-body confirmations per traced op (confirmed / refused with the reason), the restore index of bindings. 27B's build
   accumulators put a variant's lowering at ~2 s; persisting it is worth it, its validity is tied to the code identity in (4).
3. Kernel identities and images. A CUfunction handle is per process; the persisted identity is (origin, symbol, arch) with origin
   one of: a shared library file (path, size, sha256: vLLM _C ops, libtorch_cuda's ATen kernels, FlashInfer's JIT .so files), a
   compiled artifact (Triton / CuTe DSL / Inductor: the cache file path and the cubin or .so sha256; the KernelLaunch.owner knows
   it), or a harvest binding image (already saved with the bindings). On load the handle is re-resolved (module load + symbol).
   DESIGN ITEM: the native C++ host records the handle and the op name only; recording the origin needs a registration hook
   (fatbin -> library) or cuFuncGetName + cuFuncGetModule with a module registry, at trace time, once per function.
4. Bindings (opaque providers): HarvestProvider.save / load exist (meta: GPU, driver, torch, loaded CUDA libraries by name and
   size, the caller's versions, format); the same mechanism and directory.
5. Never persisted: the native exec graph (cudaGraphExec): rebuilt lazily on the first hit from the tape, as the lazy upload
   does today; variants with eager regions or unbound opaque calls (not fully traced: not saved); anything holding process
   objects (a redo context's objects are rebuilt by the integration per process; close_over'd tensors enter by static-prefix
   position, not identity).

## What must be re-validated on load (the key; any mismatch ignores the whole cache)
- Build identity: host-trace core format version (bumped with any tape / IR / program change), torch (version, git, cuda).
- Kernel binaries: every origin's sha256 (libraries by path + size + sha, artifacts by hash), the driver version, the CUDA runtime.
- Library versions whose Python decides records: vLLM, FlashInfer, Triton, flash-linear-attention, our overlays (plan_range,
  fla_symlen, qsa_symrows, ported.py by file hash), the adapter version: these are the caller's `versions` today; the contract is
  that the key changes when the step function's code or the bodies it calls change (for vLLM the model config hash + the
  versions above suffice; the general user gets a documented obligation, as torch.compile's cache keys on source).
- Device: name, SM major / minor, SM count (as the bindings meta). Not the pool layout: addresses are symbolic.
- Model identity: the model config hash (shapes, dtypes, quantization, parallelism); weight values are irrelevant to a trace.

## How it stays sound
- A loaded variant is a fresh trace's first replay: its own guards are evaluated at every call (nothing is carried as "known"
  from the saving process), the exec is instantiated on the first hit, bindings bind by key with the meta check, the static
  prefix's contract applies from load (statics_changed semantics: the first call checks in full), confirmations are valid only
  under the same code identity and are recomputed otherwise (cheap next to a trace).
- Bitwise option: `verify_first_hit` (opt-in) runs the first hit of each loaded variant against eager and compares bitwise
  (the --check machinery), one eager step per variant; off by default, on in our checks.
- All-or-nothing: a cache under a different key is not opened; a partially unloadable file (a missing library, a failed symbol
  lookup) discards the whole set for that entry; the no-cache path is what runs then, and is what the gates test.
- Nothing is relaxed: no guard is dropped, no record rewritten; persistence only replaces "trace now" by "load what an identical
  process traced before", under an identity stronger than the bindings' (sha256 of the binaries, not size).

## Expected warm-restart time (estimates; the instantiate cost per variant is the number to measure first)
- 27B vp: warm-up 215-223 s -> load of ~100 variants (tapes of ~2500 launches: ~2-3 MB each, ~250 MB; a few seconds) + lazy exec
  instantiate on first hit (unmeasured; the lazy-upload counters give it; if ~0.2-0.5 s per variant, 20-50 s spread over the
  warm-up's steps) + the warm-up's own GPU work (~20 s): engine ready ~60-110 s against 249-267 s today and vt's 149-152 s.
- Flash-Next: 1092 + 943 s of warm-up per process -> ~1-3 min (170 variants loaded, first hits cheap: the pass rows show no
  first-replay step over 0.15 s), with the untimed passes and windows unchanged.
- Both keep the bindings preload (1-3 s) and the model load as today.

## API (two knobs, nothing else)
- `torch.cuda._host_trace.persist(dir: str, key: str) -> Persistence`: process-level opt-in (as close_over returns a handle).
  While held, every HostTraceReplay loads its entry's variants from `dir` when a set under the composed key exists, and
  `Persistence.save()` (explicit, e.g. at engine ready) writes the fully traced variants of every replay. The core composes
  the full key from `key` (the integration's: model config hash + its versions) and its own parts (format version, torch,
  device, driver, library and artifact hashes). `Persistence.release()` stops loading and saving.
- `verify_first_hit=False` on persist(): the bitwise option above.
- Default: nothing persisted, nothing loaded (global state opt-in, never default).

## Sequencing (if approved)
1. Measure the exec instantiate / first-hit cost per variant on 27B and Flash-Next (the lazy-upload counters; one run each).
2. Kernel origin recording at trace time (C++ host hook + the launcher owners' artifact paths); a trace census of origins.
3. Tape + Env + program serialization with a format version; the load path treating a variant as untouched (guards, lazy exec).
4. Gate: the suites with a save/load round trip (same process: save, drop, load, replay bitwise; then a fresh process), the
   usual gate, parity8b / fn_gate, the churn checks with a loaded set, and the no-cache path unchanged.
