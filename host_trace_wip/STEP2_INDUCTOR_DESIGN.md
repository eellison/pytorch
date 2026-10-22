# Step 2 implementation design: Inductor-generated Triton under host tracing

Replaces the step-2 half of `STEP2_3_INDUCTOR_CUBLAS.md`. That design split `call`
with an upstream `Scheduler.should_partition` clause. The new stack does the split at
the tape level (step 1b): the whole `call` is one tape, including `Runner.call` across
Inductor's own partitions, and every extern op is an eager-call record. None of the
prototype's ungated changes (C1-C5 in the old doc) are ported.

Code references: wt-land (main + T4/T5/R5). Reference implementation: the `_Seams` in
`wt-integration-14@baef109ee1c:torch/_inductor/runtime/_cudagraph/hosttrace_policy.py`.

## 1. I0: flag-off parity

Rule: the flag never changes codegen. Every host-trace behavior is installed at
post-compile time, on the already generated module.

- `triton.cudagraph_host_trace: bool = False` goes in `config.py` beside
  `triton.cudagraph_trees` (2080). It is added to `_save_config_ignore` and
  `_cache_config_ignore_prefix` (~3170-3215), the same as `cudagraph_policy`. Why: the
  flag chooses a runtime wrapper, so an FX-cache entry is valid for both values.
- No `torch.cuda._host_trace*` import happens at module import time. The policy module
  imports them lazily, inside `cudagraphify`. Why: zero import cost and no cuda.bindings
  requirement when the flag is off.
- Every added line in `output_code.py` is `if config.triton.cudagraph_host_trace:`. With
  the flag off, the only extra cost is one bool read per post_compile, and nothing is
  added per call.

Tests, in `test/inductor/test_cudagraph_host_trace.py`, class `TestFlagOffParity`:
- `run_and_get_code(fn)` gives identical source with the flag on and off, parametrized
  over pointwise, reduction, extern mm, dynamic=True and a graph_partition=True case.
  Also, `FxGraphCache` key equality between the two configs.
- With the flag off, `compiled_graph.current_callable` has the same type and
  `__qualname__` as today's trees wrapper (`cudagraphify` from compile_fx), and
  `counters["inductor"]["cudagraph_host_trace_*"]` stays empty.
- A subprocess runs a compiled model with the flag off and asserts that no module whose
  name starts with `torch.cuda._host_trace` is in `sys.modules`.

## 2. Flag and seam

- **Scope.** The flag takes effect only when `triton.cudagraphs` is True, the existing
  master switch. Why: it keeps `mode="reduce-overhead"` as the single user entry point.
- **The policy.** New file `torch/_inductor/cudagraph_host_trace.py` holds
  `class HostTracePolicy(CUDAGraphPolicy)`, which overrides only `cudagraphify`. Why: the
  seam already exists, bypasses `cudagraph_partition_post_compile` (so the whole `call`
  reaches us), and is in both cache-ignore lists. It does not override `should_wrap` or
  `wrap_output`, which avoids POL-B8 (double wrapping of RegionalOutputCode).
- **Resolution.** One helper, `_active_cudagraph_policy()` in `output_code.py`, returns
  `config.cudagraph_policy`, or else `HostTracePolicy()` when the flag is on, or else
  None. It replaces the three direct reads: `cudagraph_post_compile` (297) and
  `post_compile` (889, 922). Why: one decision point.
- **Precedence.**
  - An explicit `cudagraph_policy` together with the flag raises a `RuntimeError` at
    post_compile. Why: two owners of the wrapper would silently disagree.
  - With the flag on, `cudagraph_trees` is ignored for that compile: host trace wins.
  - Compiles without the flag keep trees, unchanged ("don't replace trees, yet").
- **Gated one-liners in `output_code.py`:**
  - `prepare_cudagraph_post_compile` (213) skips the `int(t)` SymInt specialization of
    example inputs. Why: the tape treats sizes symbolically, and specialization would
    only serve trees.
  - `maybe_handle_backward_generation` (181) returns early. Why: it asserts a trees
    manager exists whenever `is_backward and cudagraph_trees`, which fails for a
    host-trace forward (the POL-B assert).
  - `maybe_realign_inputs` (431) is called with `ran_cudagraphs=False`. Why: the
    realigning wrapper then sits outside the installation, so the tape only ever sees
    aligned mutated inputs. Non-mutated ones are handled by the generated
    `copy_if_misaligned` (§4).
- **cpp_wrapper / AOTI / FXIR.** `HostTracePolicy.cudagraphify` returns `model`
  unchanged, bumps the counter `cudagraph_host_trace_skip_cpp_wrapper`, and logs once.
  Why: there is no Python module namespace to clone.

## 3. Launches

### Recording onto T4/T5 without duplicating the ABI

Refactor T5 first. The tail of `_intercept` in
`torch/cuda/_host_trace_triton_launch.py` (everything after the binder) becomes:

```python
def record_launch(tr, name, function: int, owner, abi: TritonABI,
                  values: Sequence[Any], dims: Sequence[Any]) -> None
```

It covers:
- the per-arg slot fill (the constexpr hint check, pointer roots, `_float_bits`, the int
  range guard) plus the `tt.divisibility` guard for each `TritonArg.divisibility > 1`;
- the grid checks (1-3 dims, `_GRID_LIMITS`, empty-grid skip, volume <= 2**31-1);
- `param_layout`, `block = (abi.num_warps*32, 1, 1)`, `tr.record_launch`.

`_intercept` calls it with `binary.function` and `binary` as owner.

Also, `TritonLaunch.binary` becomes `function: int` plus `owner: Any` (a keepalive). Why:
capture and replay already use only `int(t.binary.function)` (capture.py:135,178;
replay.py:82), and Inductor's static launcher has no `CompiledKernel`.

The Inductor side goes in `torch/_inductor/cudagraph_host_trace.py`,
`class _InductorKernel`, a proxy placed in the cloned namespace (§4) in place of each
module-global `CachingAutotuner`. There is no class monkeypatch. Why: the proxy touches
only the installed graph, and other compiles and eager Triton are untouched.

`_InductorKernel.run(*args, stream, **kw)`:
1. Outside a trace, it forwards to `autotuner.run` (warm-up and fallback calls).
2. It declines on extra kwargs, `type(autotuner) is not CachingAutotuner`
   (DebugAutotuner, MultiKernelCall), non-empty `_plugins`, `config.triton_interpret`
   or a dump/profiler flag, and `host_tma_descriptor_args`.
3. It raises `Declined(retry=True)` when `len(autotuner.launchers) != 1`, coordinate
   descent is on but has not run, or a combo-kernel tuning group is pending. Why: T5's
   rule is that the first call tunes and a later call traces.
4. It finds the winning result with `r.config is launcher.config` over
   `autotuner.compile_results`, which is pruned to the winner by
   `_prune_compile_results_to_launcher` (1111). It then builds `(src, metadata,
   function)`:
   - `TritonCompileResult`: `kernel.src`, `kernel.metadata`, `kernel.function`.
   - `StaticTritonCompileResult`: `ASTSource(fn, compile_meta["signature"],
     compile_meta["constants"], compile_meta["configs"][0])`, exactly as
     `_precompile_config` (1374) builds it. The metadata is `SimpleNamespace(num_warps,
     shared, num_ctas=1, global_scratch_size or 0, profile_scratch_size or 0,
     launch_cooperative_grid=False, launch_pdl=False, tensordesc_meta)` from the
     `StaticallyLaunchedTritonKernel`, and the function is `function` or
     `functions[device]`.

   `triton_abi` then gives the ABI, and `param_layout` cross-checks it against the
   loaded function. Why: T4 stays the one owner of the ABI. A reconstruction mismatch
   surfaces as a decline, never as a wrong launch.
5. The grid comes from Inductor's launcher itself. The proxy re-execs the launcher's
   source (`_gen_launcher_code`, 2911) with `scope["runner"]` replaced by a recorder, so
   Inductor's own grid expressions run over SymInts. The recorder gets the runner's
   positional layout: static `(gx, gy, gz, stream, *args)` or the Python launcher's
   `(gx, gy, gz, stream, function, metadata..., *args)` (3350-3385). It maps `call_args`
   by `abi.args` slot order and calls `record_launch`. Why: grid code is never copied,
   and a change in Inductor's grid codegen flows through.

Tests: `TestInductorLaunches`: pointwise and persistent reduction; `dynamic=True`, 3
shapes, one capture; `triton.use_static_cuda_launcher` on/off; DebugAutotuner, TMA and
kwargs declines; first call warms up, second traces.

### Autotuning and coordinate descent

Both happen in the warm-up call, the first miss that R5 runs untraced. Both are sticky
per kernel: the autotuner prunes to one launcher and never re-tunes on shape. So there is
no autotune-key guard, unlike T5's handling of Triton `Autotuner`. A pending tune is
`Declined(retry=True)` and is not memoized by R5, so the next call traces. Why: the
winner is fixed for the kernel object's lifetime, and the recorded ABI is the winner's.

## 4. Wrapper host code

Installation clones each module-level function of `compiled_graph` (`call`, `Runner`
methods, the partition functions from `codegen_partition_call`, 5005) into one fresh
globals dict. Only names the header (wrapper.py:1842-1851) or `add_import_once`
imported are rebound. Why: the generated module stays shared and unmodified, which keeps
flag-off parity and cache sharing.

The seams are rebound names in that dict. Each checks `torch.cuda._host_trace_tape.
current_trace()`, and when it is None it calls the original C++
(`torch._C._dynamo.guards`, guards.cpp:1313,1327). The reference semantics are kept:

| name | under a trace |
|---|---|
| `empty_strided_cuda` | `torch.empty_strided(..., device=tr.device)` becomes a tape allocate |
| `reinterpret_tensor` | `torch.as_strided(t, size, stride, t._sym_offset + off)`, checked by `_Trace._check_as_strided` |
| `empty_strided_cpu`, `empty_strided_cpu_pinned`, `alloc_from_pool` | decline (a host buffer, or a pool alias the tape cannot bound) |
| `assert_size_stride`, `assert_size_stride_grouped` | no-op (trusted) |
| `assert_alignment` | no-op (trusted) |
| `assert_tensor_metadata` | no-op (trusted) |
| `copy_if_misaligned` | a real runtime branch, so a dispatch guard: one alignment guard per live input; a misaligned input is traced as a clone and is its own variant |
| any other `torch._C._dynamo.guards` helper | declines by name ("X is not traced"); a test checks every helper is seamed |

- **Workspaces.** `generate_workspace_allocation` (4086-4134) writes
  `empty_strided_cuda` and then `.zero_()`. `zero_` becomes a memset record. That means
  a new tape record kind, lowered to `cuMemsetD8Async` at capture and patched with
  `cuGraphExecMemsetNodeSetParams`. Why: it is the only in-graph op Inductor emits that
  is not a Triton launch or an extern op, and a memset is cheaper than an eager segment.
  The fallback is an eager-call record via 1b.
- **Closure tensors.** Constants and parameters captured by `call` become constant
  roots (baked address plus a strong reference in the tape). Why: Inductor already
  assumes these are fixed. The contract is documented: in-place `.data` swaps are not
  supported.
- **Int outputs.** A SymInt returned by `call` (dynamic shapes return sizes) gets an
  output record. Today `_output_records` declines non-tensor outputs.
- **Guards: none of our own.** (user, 2026-09-25: "we should expect, maximal information
  given to us, clean input ... as inductor does it today"; "we dont need to add extra
  guards, we can assume the symints given to us is valid".)
  - Dynamo's guards and Inductor's own assumptions are preconditions of the call. The
    tape adds no guard that restates them.
  - `assert_size_stride`, `assert_size_stride_grouped`, `assert_alignment` and the
    generated runtime asserts are skipped by the seams, not turned into guards.
  - The kernel specializations `_InductorKernel` checks today (constexpr values and
    `tt.divisibility` of ints and pointers) are facts Inductor proved when it compiled the
    kernel, so they are read at the hint and never guarded.
  - Runtime dispatch stays (user: "we will still need our own runtime dispatch etc ...
    but we should assume the inputs that are described to us are valid"). Guards from
    real decisions are kept as today and select the variant at replay: host branches in
    the generated code, size-based kernel or multi-kernel choice, per-shape library
    choices (item 4), and 1b boundaries. They are counted for the sweep, not flagged.
  - Net effect: a graph with no such decision has one trace for every call Dynamo admits.
    In practice every live input carries one alignment guard (`copy_if_misaligned`). Sizes still flow through the integer program
    (allocation sizes, grids, offsets); they just aren't checked.
  - `.item()` and unbacked SymInts decline until 1b supports data-dependent sizes.
- **Extern kernels** (`extern_kernels.mm`, fallback aten ops, collectives) reach
  `_TraceMode.__torch_dispatch__` and become eager-call records through step 1b. Before
  1b they decline, so pure-Triton graphs work first.

Tests: `TestWrapperSeams`: `reinterpret_tensor` offsets/views; untraced seams call the
C++ original; asserts become guards that miss correctly; split-K workspace at two sizes.

## 5. Installation and policy

`_Installation.__call__(new_inputs)` does `args = tuple(new_inputs); new_inputs.clear()`
and calls `HostTraceReplay(lambda *xs: cloned_call(list(xs)))(*args)`.

- **When to trace.** Tracing starts on the first call. R5's first miss is the untraced
  warm-up, which also autotunes. Why: this matches trees' warm-up-then-record contract.
- **Variants.** `max_variants=16` is R5's default, overridable through
  `triton.cudagraph_host_trace_max_variants`. Past the cap, calls run the plain `call`.
- **Fallback.** Any `Declined` runs the plain (realigned) callable. Declines count in
  `counters["inductor"]["cudagraph_host_trace_declined"]` and are logged once per reason
  through `trace_structured` (`name: cudagraph_host_trace_decline`).
- **Inputs.** Generator, CustomClass or other non-Tensor/SymInt inputs make
  `cudagraphify` return the plain callable. Why: `cudagraph_tests` (692-733) already
  excludes most of these, and the user asked for plain Tensor inputs first.
- **Fail reasons.** v1 honors every existing `cudagraph_fail_reasons` entry, including
  "mutated inputs" and "input storage mutation". Lifting "mutated inputs" is a follow-up:
  the tape handles it naturally, but trees' reasons are shared code.
- **Inputs: as Inductor sees them.**
  - `static_input_idxs` (parameters, buffers, saved activations in backward) keep constant
    layouts but still get an address slot. The index means "address-stable under trees,
    which re-checks the address on every replay", not "the same tensor every call":
    backward marks saved activations static, a second module instance shares the compiled
    graph, and `p.data = ...` rebinds a parameter (REVIEW_22 F1/F2, wrong grads and stale
    weights when the address was baked in). A pointer row costs about 3.5 ns.
  - Only the dims that are symbolic in Inductor's graph are symbolic in the tape; every
    other size and stride is a constant. Live inputs keep an address slot.
  - No per-call input contract check (dtype, device, rank, strides): Dynamo's guards and
    Inductor's input handling already establish it.
- **Graphs.** Forward, backward and inference graphs are each installed independently;
  `is_backward` needs no manager lookup. The lazily compiled backward reads the flag
  from the config at its compile, and forward/backward agreement is checked (see §7).
- **Memory.** R7's eager-like memory owns tape allocations; no trees pool, no
  `mark_step_begin`.

Tests: `TestInstallation`: fwd+bwd step equal to eager with no trees assert; variant
cap then plain fallback; a declined graph matching eager; trees and host trace mixed.

## 6. PR slicing (in order)

| PR | contents | tests | needs 1b |
|---|---|---|---|
| 2.0 | flag, cache-ignore entries, `_active_cudagraph_policy`, `HostTracePolicy` returning `model` unchanged, the three gated one-liners | `TestFlagOffParity`; trees assert gone | no |
| 2.1 | `record_launch` factored out of T5, `TritonLaunch.function/owner`, `_InductorKernel` proxy and launcher re-exec | `TestInductorLaunches` plus existing T5 tests unchanged | no |
| 2.2 | namespace clone, seams (all size/stride/alignment asserts skipped), `_Installation` over HostTraceReplay, Inductor's inputs (static inputs keep constant layouts and an address slot, only Inductor's symbolic dims symbolic, no input check, no specialization guards) | `TestWrapperSeams` (no workspace, two-input graphs), `TestInstallation` on pure-Triton graphs, one trace over many sizes with zero guards | no for pure Triton; **yes** for any graph with an extern op |
| 2.3 | memset record kind (tape, lowering, capture, patch), constant roots, int outputs | workspace and split-K graphs, dynamic-shape SymInt returns | no (memset); the eager-segment fallback needs 1b |

2.0 lands alone as the parity gate. 2.1 is reviewable against T4/T5 alone. 2.2 is the
first PR where the flag does anything, and realistic models (mm) need 1b merged first.

## 7. Risks and open questions

1. **Memset width patching.** `cuGraphExecMemsetNodeSetParams` may reject a change in
   width or element count. Recommendation: verify with a microtest in 2.3 first. If it
   fails, key the variant on workspace size (a guard) or use the 1b eager segment.
2. **Config consistency across the lazy backward.** The backward compiles later, under
   whatever config is active then. Recommendation: record the flag in the forward's
   `CompiledFxGraph` metadata. If the backward disagrees, it falls back to plain with a
   logged reason, never mixing with trees.
3. **`release_benchmark_artifacts`** (718) clears `launchers`/`compile_results`.
   Recommendation: the proxy snapshots `(src, metadata, function, owner)` at trace time.
   `owner` keeps the winning result alive, so replay never re-reads the autotuner.
4. **device_agnostic static kernels** (`functions` per device). Recommendation: select
   `functions[tr.device.index]` and decline if it is missing. Test on a single device
   only.
5. (Dropped: the "kernel names over 150 chars" case; REVIEW_21 found no such limit.)
6. **Per-call validation cost for graphs with many parameters.** Constant roots remove
   parameters from the per-replay patch set, and 2.2 drops the input check. Measure in
   the model sweep.
7. **Lifting "mutated inputs".** Recommendation: keep it for after 2.3, gated on a test
   showing the tape reproduces eager's in-place writes, including aliasing views.
8. **Constant-root soundness** (a swapped `.data`). Recommendation: document the
   contract in the policy docstring, and add an address guard only if it shows up in
   practice.
