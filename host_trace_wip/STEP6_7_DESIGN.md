# Items 6-7 design: CuTe DSL launches and training under host tracing

Read-only design, 2026-09-25. Code references are to snapshot `20260925-1bdone`
(extracted to `land/scratch/step67/src/torch/cuda/`), to STEP2_INDUCTOR_DESIGN.md and to
the frozen reference `wt-integration-14@baef109ee1c`. The reference has 2579 lines of CuTe
recorder (`_host_trace_cute.py`, `_host_trace_cute_dsl.py`, `_host_trace_cute_desc.py`)
with three routes. This design ports one route, and only after the EagerCall route is in.

Summary. Both items are mostly already correct through step 1b: a CuTe call or a training
op the tape cannot describe runs eagerly, in order, and its memory is eager's. The new
work is:
- (6) turning CuTe calls into graph launches through the same record as Triton;
- (7) lifting "mutated inputs" and freeing backward inputs at last use.

## 1. What must exist first

| needs | from | why |
|---|---|---|
| 1b EagerCall with non-aten targets (the `("triton", jit, grid, options)` tuple, `_host_trace_tape.py:124`) and REVIEW_1B fixes 1-2 | item 1 | 6.0 records CuTe calls this way; `_fused_adamw_`, `_foreach_*_` and `_amp_foreach_non_finite_check_and_unscale_` return None |
| `record_launch(tr, name, function, owner, abi, values, dims)` and `TritonLaunch.function/owner` | 2.1 | 6.1 is a second producer for the same record |
| namespace clone and seams, `_Installation`, fwd/bwd/inference installed independently | 2.2 | Inductor's CuteDSL `run` seam; every item-7 graph |
| memset record (`buf.zero_()` before atomic-add backward kernels) | 2.3 | embedding/index_put backward; otherwise an EagerCall per zero_ |
| static inputs by identity and address | 2.4 | parameters and optimizer state as constant roots |
| C++ replay | 3 | only 6.2 (per-replay CUtensorMap re-encode); nothing else |
| GEMM regions | 4 | perf only: until then every mm is an EagerCall and splits the tape |
| attention/philox regions | 5 | perf only: SDPA fwd/bwd are EagerCalls until then |

Items 3-5 are perf-only here. Inductor dropout needs no item 5: `inductor_seeds` is an
EagerCall that advances the generator every replay, as eager does.

## 2. Item 6: CuTe DSL

### 2.1 Already covered by 1b
- **torch._native CuTe overrides** (QuACK rmsnorm, topk, scatter_add) sit below the aten
  op that `_TraceMode` sees. They are EagerCalls of that aten op today, with no CuTe code.
  Why keep it that way: recording them as launches needs the override's
  `const_data_ptr()` alignment test to run outside the mode, on a symbolic address with an
  alignment guard. That is 6.3, last.

### 2.2 6.0: CuTe calls as EagerCalls
- **Hook points.** Two, both installed inside the existing `intercepting()` scope:
  - Inductor: the `CuteDSLKernelWrapper.run` seam, added to 2.2's namespace clone.
    Why: `run` takes torch tensors, before the generated `kernel_fn` calls
    `from_dlpack` (`codegen/cutedsl/cutedsl_kernel.py:67`).
  - Direct user calls: `JitCompiledFunction.__call__` and
    `TVMFFIJitCompiledFunction.__call__`.
- **Record.** Each call becomes
  `EagerCall(("cute", compiled, options), args, {}, (), writes, reason)`, where `writes` is
  every traced tensor operand. This is the same shape as T5's fallback
  (`_host_trace_triton_launch.py:268-277`). CuTe calls return None and write their
  operands, so no fake kernel and no output metadata are needed.
- **Declines by name:**
  - the `stream=` argument is not `tr.stream`;
  - a `cute.runtime.from_dlpack` value built from a traced tensor (that needs the
    reference's `_TracedCuteValue` stand-in, deferred);
  - a tensor the trace does not track.
- **Replay.** `_run_eager` calls `compiled(*real_args, stream=current)`; memory is eager's.
  Caller-side decisions (compile-cache keys, dtype/alignment branches) ran on traced
  values, so they are already guards by hint.

### 2.3 6.1: CuTe launches in the graph
**ABI source: the host function as MLIR text** (CUTE_SPIKE.md). Pass a
`trace_finalize_hooks` callback into `cute.compile`; stock cutlass-dsl 4.6.2 has the
hook. Inside it, copy `module.operation.get_asm(print_generic_op_form=True)` (and the
pretty form for signatures and type aliases). Never touch live values: that ran the value
casters and corrupted the heap. Evaluate each `cuda.launch_ex`'s operands backwards over
the host formals (`cute_text.py` in the spike, about 400 lines):
- each kernel parameter as an expression over the host formals: a pointer
  (`get_iter(formal)` plus offset), a shape or stride leaf, a scalar formal, or a constant;
- grid, block and smem from `cuda.launch_cfg.create`, cluster from `cluster_dim`;
- `scf.if` / `cmpi` / `select` over the formals (e.g. the Ampere GEMM's raster factor);
- whether each shape/stride leaf is static (baked by the compile) or dynamic.

Unknown ops decline by name. Why: exact in the spike (add and Ampere GEMM byte for byte
over 5 shapes each), and it needs nothing from NVIDIA.

**Param layout.** Take it from `param_layout(abi, function)` (cuFuncGetParamInfo), exactly
as T4 does. The descriptor gives values, the driver gives offsets. Why: the flat
natural-packing rule is wrong for mixed i64/i32 shape leaves (cute-adoptions note); the
driver never is.

**Function handle.** Look up the kernel symbol in the loaded module of the same
`compiled` object the warm-up ran, and keep `compiled` as `owner`.

**Guards on host-side decisions.** The descriptor's expressions are evaluated on the
traced operands: SymInts give slots, and the grid is checked against `_GRID_LIMITS`. Then:
- every static leaf is pinned: `guard(sym == hint)`, the mirror of T5's constexpr check
  (`_host_trace_triton_launch.py:334-340`);
- `assumed_align` / divisibility marks become `_probe_address` alignment guards.

**Witness.** At warm-up (only when `warm_up=True`), capture the same DSL call once under
the trace's thread-local capture, on `tr.stream`. Nothing runs. Compare the captured
node's image byte-for-byte with our evaluated slots at the real arguments, skipping
padding. This is `_check_witness`'s role for fakes. A mismatch makes that call an
EagerCall with the reason recorded, never a decline of the whole trace.

**Shared vs separate.** One record kind.
- Rename `TritonLaunch` to `KernelLaunch(name, function, owner, layout, grid, block,
  smem, slots, pointer_slots, roots)`.
- T4 computes `pointer_slots` from `abi.args` at record time. Lowering reads
  `launch.abi.args` only for that (`_host_trace_lower_tape.py:303`).
- Lowering, capture, memory, replay and `_set_params` are unchanged
  (`_host_trace_replay.py:114`, `_host_trace_capture.py:189`).
- Separate: only the producer (`_host_trace_cute.py`, the hook plus the descriptor walk
  plus the witness), about 400 lines.

**Static formals can still have a slot** (Blackwell `tiled_mma`, 32 bytes of zeros), so
the driver's param info decides the layout, never the type.

**6.1 declines (they stay EagerCalls):** TMA atoms until 6.2, PDL, QuACK artifacts loaded
from its on-disk jit_cache (no finalize ran), and a walk that meets an unknown op.

**Order within 6.1.** Direct `cute.compile` users first: they need only 1b, so they can be
built while 2.2 is in flight. Inductor's CuteDSL templates follow once 2.2 is in.

### 2.3b 6.2: TMA descriptors and clusters (CUTE_SPIKE.md "TMA follow-up")
- A 128-byte, 64-aligned slot is a CUtensorMap. It is a recomputed value, not a byte image:
  `cuTensorMapEncodeTiled` on a shape/stride change (0.40 us), `cuTensorMapReplaceAddress`
  on an address change only (0.24 us). Both are documented driver calls.
- Encode arguments come from the atom type in the text: dtype from `tma_format`, box and
  dimension order from `tma_gbasis`, swizzle from the smem layout (`S<3,4,3>` 128B,
  `S<2,4,3>` 64B). L2 promotion (128B), OOB fill (none), interleave (none) and element
  strides (1) are DSL 4.6.2 constants that the witness confirms.
- The driver encode equals the DSL's on 45/45 captures except byte 8 bit 1 (DSL 0x22,
  driver 0x20), constant and correctness-neutral. Set it to the DSL value; the witness
  masks only that bit and bytes 64..127 (uninitialized), and refuses on anything else.
- Cluster dims become a kernel-node launch attribute. The spike retargeted one captured
  graph to 6 shapes by patching descriptors, coordinate slots and grid.
- No dependency on item 3: the per-replay cost is small in Python too.

### 2.3c 6.1b: witness at every trace; shared capture
- **Witness at every trace** (REVIEW_CUTE61 S1). Each traced CuTe call also runs through
  TVM-FFI over stand-in tensors (`_construct_storage_from_data_pointer` at the traced
  tensor's hint address, sizes and strides), under a relaxed capture on a pool stream
  that is not the trace's, never replayed. Its kernel nodes give each function and
  parameter layout; the evaluation at the hints must equal each node's grid, block,
  smem, cluster and the bytes its fields and descriptors set. A mismatch is an
  EagerCall. No `__class__` swap, no retry, nothing on users' eager calls, and the
  witness is always the trace's device and shapes (C4, C6). Cost: about 0.27 ms of
  capture per call, plus the evaluation the trace does anyway.
- Stand-in addresses drop the placeholder's non-canonical top (`& _PLACEHOLDER_LOW`):
  the driver's TMA encode and ReplaceAddress reject it. `TmaDescriptor.encode` masks
  the same way, so a TMA over a traced allocation can be captured at its placeholder.
- **F1.** `TmaDescriptor` keeps its last slot values and CUtensorMap: a change of the
  address alone is a `cuTensorMapReplaceAddress`, no change is a copy of the 128 bytes.
  The LRU is gone. The replay holds `(grid, slots)` per node and repacks only on change.
- **Shared capture** (`torch/cuda/_host_trace_capture.py`), for item 4's cuBLAS harvest
  too:

  ```python
  @dataclass(frozen=True)
  class KernelNode:
      name: str
      function: int                          # the CUfunction
      layout: tuple[tuple[int, int], ...]    # (offset, size) per parameter, cuFuncGetParamInfo
      grid: tuple[int, int, int]
      block: tuple[int, int, int]
      smem: int                              # dynamic shared memory bytes
      cluster: tuple[int, int, int]          # (1, 1, 1) without the attribute
      images: tuple[bytes, ...]              # each parameter's bytes as the node holds them

  def capture_kernel_nodes(fn: Callable[[torch.cuda.Stream], Any]) -> list[KernelNode]
  ```

  `fn(side)` runs with `side` current, under a relaxed `torch.cuda.CUDAGraph` capture
  (keep_graph) and `_gc_hold`; the graph is never instantiated. Nodes come back in
  capture order. A non-kernel node, or a kernel node without `kernelParams` (the
  `extra` form), raises ValueError. `pack_params(launch, slots, pointers)` is the
  inverse for a record with `fields`: the bytes the replay writes, which a witness
  compares with `images`.

### 2.4 Stream in the jit name
`BaseDSL.mangle_name` keys the compile on `str(stream)` when the handle has fewer than 8
hex digits (every GB300 stream), so each stream gets its own compile.
- **Recorded launches (6.1):** unaffected. The trace and its warm-up are already on one
  stream (`_intercept` checks `tr.stream`); the recorded CUfunction is launched by our
  graph on any replay stream; the witness captures on `tr.stream`, never on a side stream.
- **EagerCalls (6.0):** a replay on a new stream recompiles (about 0.7 s), and fails if
  that stream is being captured by someone else. Document it; no code.
- **Upstream ask to NVIDIA:** a one-line regex change `{8,16}` -> `{1,16}`. Batch it with
  the asks below.

## 3. Item 7: training

### 3.1 Inductor forward and backward (the primary path)
- **Autograd: nothing new.** AOTAutograd already hands us separate forward and backward
  graphs. Each is a plain function of tensors, run under no_grad inside its
  autograd.Function, and 2.2 installs each independently. `_output_records`' requires-grad
  decline (`_host_trace_tape.py:760`) does not fire for them.
- **Saved tensors crossing forward and backward.** They are ordinary outputs of the
  forward replay:
  fresh caching-allocator tensors each call, held by autograd's ctx, then backward inputs.
  Memory matches compile-without-cudagraphs; there is no pool, generation,
  `mark_step_begin` or "overwritten by a subsequent run" error, so several forwards before
  a backward (gradient accumulation, checkpoint recompute) just work (a paper point).
- **The backward memory gap.** Inductor's backward `del`s each saved tensor at its last
  use. `_Installation` clears `new_inputs` but holds `args` for the whole replay, so every
  saved activation lives until the backward ends. Measured cost (backward-reclaim note):
  +1.3% of peak at batch 64 and +25% at batch 4096 on an 8-layer stack.
  - **7.2 fix:** the replay takes its inputs boxed, and `_commit` drops its reference to
    each input after enqueueing the last step that reads it.
  - Inductor's own `del` semantics (another holder keeps it alive); stream order makes
    the allocator's reuse safe. Residual gap: a temporary cannot reuse an input freed in
    the same segment; donated-block planning (7.4, optional) closes it.
  - This lifts the standing "don't free inputs yet" deferral: needs Elias's OK first.

### 3.2 In-place parameter and optimizer updates
- **What blocks it today.** `torch.compile(opt.step)` graphs mutate their inputs (params,
  `exp_avg`, `exp_avg_sq`, `step`), and v1 honors the "mutated inputs" fail reason.
- **7.0 fix:** host trace skips that reason; "input storage mutation" stays a fail reason.
  - The tape already reproduces the writes: Inductor's kernels write through the input
    pointer, and a `copy_` epilogue is Triton or an EagerCall with schema aliasing.
  - The exactly-once order holds without `warm_up=False`: `trace()`'s warm-up is the real
    call and its result is returned, the symbolic run executes nothing, and R5's first
    miss is untraced.
- **Parameters as constant roots (2.4).** They are sound under in-place updates because
  the address does not change. A `.data` swap is the documented contract (STEP2 risk 8).
- **Not covered:** a non-capturable optimizer keeps `step` on the CPU, so it stays plain
  through the existing "cpu device" reason. Tests use `fused=True` or `capturable=True`.

### 3.3 AMP and GradScaler
- **Autocast** is baked into the Inductor graphs; eager tracing fixes it at trace time
  (`trace()` docstring), and the witness caught no mismatch (REVIEW_1B, "What held up").
- **GradScaler with a fused optimizer** (`_step_supports_amp_scaling`,
  `amp/grad_scaler.py:418`) is sync-free:
  - the unscale is an EagerCall of `_amp_foreach_non_finite_check_and_unscale_`;
  - the skip happens on device through `found_inf`/`grad_scale` inside `_fused_adamw_`;
  - `_amp_update_scale_` updates the scale tensor in place.
- **Non-fused optimizers** hit `found_inf.item()`. Dynamo graph-breaks there, and each
  side is installed on its own. In an eager trace it is a host read, so the whole call
  declines, per 1b's data-dependent rule. No design work; one test each.

### 3.4 Eager training step (after the Inductor path)
Inside `trace()`, `loss.backward()` fails three ways:
- the engine runs CUDA nodes on its device thread, and `_TraceMode` declines "on a thread
  other than the trace's" (`_host_trace_tape.py:692`);
- AccumulateGrad sets `.grad` Python attributes to traced tensors, which escape the tape;
- `requires_grad` args are traced as plain `_TracedTensor`s.

**7.3:** the eager step goes through the A241 route. Symbolic `make_fx` over the
functional step gives a GraphModule with no engine and no attribute writes; its in-place
`_foreach_add_` / `_fused_adamw_` on placeholders use the in-place tape form (A261), with
the loss as the only output. That GraphModule is host-traced like any function.
- A direct `.backward()` inside `trace()` declines by name.
- Why not propagate the trace to the engine thread: that fixes only the first failure.
- HF GPT-2 train declines only at `aten.slice_backward` (no kernel): item-8 work.

## 4. PR slicing (in order)

| PR | contents | tests | needs |
|---|---|---|---|
| 6.0 | `("cute", ...)` EagerCall target; CuteDSL `run` seam; `JitCompiledFunction`/TVM-FFI `__call__` hooks; stream and from_dlpack declines | Inductor cutedsl template bitwise vs eager over 3 sizes; direct `cute.compile` kernel; replay on a second stream | 1b, 2.2 |
| 7.0 | skip "mutated inputs" under host trace | compiled fused AdamW and SGD-momentum steps, 5 steps bitwise; mutation through an input view | 2.2 (2.4 for constant roots) |
| 7.1 | fwd/bwd flag agreement (STEP2 risk 2); tests only otherwise | fwd+bwd+step on an MLP and a 2-layer transformer; gradient accumulation (2 forwards, 1 backward); GradScaler fused and non-fused; dropout | 7.0, 2.3 |
| 7.2 | boxed inputs; `_commit` drops each input after its last reading step | backward peak vs compile-without-cudagraphs on the backward-reclaim 8-layer stack, batch 64 and 4096 | 7.1; Elias's OK |
| 6.1 | `KernelLaunch` rename plus `pointer_slots`; finalize-hook MLIR text and evaluator; static-leaf and alignment guards; witness | Inductor cutedsl pointwise and reduction templates as launches; a static leaf that changes retraces; witness mismatch becomes an EagerCall | 2.1, 6.0; risk 1 spike |
| 7.3 | eager step via symbolic make_fx; named decline for `.backward()` in `trace()` | module+AdamW(fused) bitwise over 5 steps; HF GPT-2 minus slice_backward | 7.0 |
| 6.2 | cluster dims and CUtensorMap re-encode / ReplaceAddress per replay (shared with Triton T4's later PRs) | a Blackwell CuTe GEMM with TMA over shapes on both sides of 128 KB | 6.1 |
| 6.3 | torch._native overrides as launches (symbolic data_ptr plus alignment guard, condition outside the mode) | rmsnorm/topk overrides vs their aten EagerCall | 6.1 |

**Why this order:**
- 6.0 is small and makes every CuTe call correct.
- Training (7.0-7.2) matters more for the paper than CuTe launch coverage, and needs no
  DSL change.
- 6.1 carries the only external dependency (risk 1).
- 7.4 (donated-block planning) is listed only if 7.2's residual gap shows up in the
  numbers.

## 5. Risks and open questions

1. **The descriptor walk without raw value reads.** The reference walk ran under the
   runtime SDK's `ir.raw_values()`, a meta-path patch, not landable. Stock 4.6.2 has
   `trace_finalize_hooks` but no `ir.raw_values` (checked). Without it, the DSL's value
   casters emit ops into the module being walked: the heap corruption A410 saw.
   Recommendation: a one-day spike under `land/scratch/step67` walking plain
   `op.operation.operands` with `MALLOC_CHECK_=3`. If casters still fire, ask NVIDIA for
   the three asks from the bridge comparison (launch geometry as data, an invocation
   hook, the kernel ABI in `kernel_info`), plus the stream regex. 6.0 covers the gap.
2. **TMA and clusters are the norm for Blackwell CuTe GEMM/attention.** 6.1 may cover
   mostly pointwise and reduction templates. Recommendation: run a decline census by
   cause on Inductor's cutedsl templates and QuACK as 6.1's first test, and size 6.2 from
   it.
3. **Backward memory until 7.2** (+25% at large batch). Recommendation: land 7.2 directly
   after 7.1, and report numbers from both states in the paper evaluation.
4. **A mid-commit `_Disagreement` on a mutating tape leaves a half-applied optimizer
   step** (REVIEW_1B finding 3, raise policy). The witness limits this to size-dependent
   lying fakes. Recommendation: keep raising, and make the message say that inputs were
   partially updated.
5. **Forward/backward mismatch.** For example, a forward under host trace and a backward
   under trees (lazy compile under different config). Both-plain is fine; trees plus host
   trace is not, because trees' generation bookkeeping sees half a step.
   Recommendation: 7.1's agreement check, falling back to plain with a logged reason.
6. **DSL version drift renames IR ops.** The walk declines unknown ops by name (never
   wrong, only slower). Recommendation: pin the tested cutlass-dsl range in 6.1's tests.
