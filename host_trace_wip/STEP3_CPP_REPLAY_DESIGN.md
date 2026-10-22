# Step 3 implementation design: C++ replay

This design moves the per-call replay loop of `HostTraceReplay` (snapshot 20260925-1bdone) into
libtorch. The Python tracing, lowering, capture and miss policy stay in Python.

The rule for every PR: the Python `_commit` loop stays in tree as the oracle, behind
`HostTraceReplay(native=False)`. Each PR is tested by running the same calls through both
paths and comparing them bitwise:
- outputs and argument mutations;
- the `held` images;
- `torch.cuda.memory_stats()` deltas.

Code references:
- current code: `land/scratch/step3/src/torch/cuda/_host_trace_{replay,memory,program,capture,lower_tape}.py`
- measured costs: `land/BOUNDARY_COST.md`
- reference only, never ported wholesale: `wt-integration-14@baef109ee1c:torch/csrc/cuda/GraphParameterProgram.h`
  (an SSA interpreter) and `Graph.cpp` (`BoxedReplayPlan`, +4467 lines, coupled to the allocator
  changes).

## 1. What moves, ranked by measured cost

GB300 box, small tensors, CPU-bound. The "C++ estimate" column is the per-item floor measured on
the prototype's native runtime.

| item | Python today | C++ estimate | scales with |
|---|---|---|---|
| `_set_params` (patch) | 4.0 isolated, 6-7.7 in the loop | 0.43 setter + 0.16 in the next launch | dirty nodes |
| contract + family lookup | 0.45/tensor (0.86 at 1 input, 101 at 222) | about 0.03/tensor | inputs |
| `evaluate` (bind + ctypes + tuple) | 8.8/variant tried (2.3 + 6.4 for 76 rows) | under 1 | inputs, rows, variants tried |
| allocate (place + empty + empty_strided) | 7.5/step | 0.6-0.8/allocation | allocations |
| eager step glue (views 8.8, check 6.6) | about 15/boundary | about 1 | boundaries |
| `graph.replay()` | 5-6/segment | about 2.5 (`cudaGraphLaunch`) | segments |
| `launch_images` + held compare | 0.84/launch | about 0.05 | launches |

The contract row comes from `scratch/step3/p_contract.py`. The rest are from BOUNDARY_COST.md.

Order: the whole commit loop moves in one direction, patch first. Why:
- `_set_params` dominates patch mode: 16 dirty nodes cost 114 us of 192.
- The loop's other items are only worth moving once values already live in C++. Otherwise each
  of them pays a Python-to-C++ conversion, which is the 6.4 us that `compiled.evaluate` spends
  today.

The contract moves last, as a separate PR (3.4). Why:
- It is cheap on the benchmark chains, but it is the largest fixed cost on real models (222
  inputs gives 101 us).
- It is independent of the loop.
- Inductor may narrow it: it knows its inputs' types (step 2, PLAN 2.4 static inputs).

## 2. The evaluator: no host compiler at runtime

`compile_program` currently calls `CppCodeCache.load` on `_source`. That needs a compiler at
runtime, the dependency STACK_SPLIT.md flags. It is replaced by an interpreter in
`torch/csrc/cuda/HostTraceProgram.{h,cpp}`:
- It runs over the existing `IntegerProgram` rows: constant, the leaf reads, add/multiply
  (checked), floordiv/ceildiv, comparisons, and/bit ops, min/max, select.
- Values are int64 only.
- Status codes match `_step`: SUCCESS 0, ADD_OVERFLOW 1, MULTIPLY_OVERFLOW 2, DIVISION_DOMAIN 3,
  BOOLEAN_DOMAIN 4. Overflow uses `c10::add_overflows` / `c10::mul_overflows`.

Why an interpreter rather than AOT codegen:
- 76 rows at a few ns each is under 0.5 us.
- Rows are validated once, at construction (operand indices earlier than the row, known opcodes).
  After that the loop does no bounds checks.
- Unlike the reference `parameter_program::Program`, there are no widths, poison bits or flags.
  The Python rows only need int64 plus a status.

Leaves are read straight from `at::Tensor`: `data_ptr`, `sizes()[d]`, `strides()[d]`,
`storage_offset()`, and boxed ints. They come from a flat leaf table built at construction,
`(input index, kind, dim)`.

The Python evaluator stays as the oracle. `compile_program` is kept for one release, under a
test-only flag, as a third opinion.

Tests: differential fuzzing of random well-formed programs against `_step`, plus every program
the existing replay tests lower. Compare the value vectors and the statuses, including the
overflow and domain edges (INT64_MIN // -1, ceildiv with d=0, a boolean domain violation).

## 3. The object boundary

There is one `torch._C._HostTraceVariant` per variant. It is created at the end of `_capture`
and stored on the Python `Variant` as `variant.native`. The Python `Variant` still owns the
`LoweredTape`, the `CapturedTape` and the `MemoryPlan`, for misses, repr and the oracle.

Handoff is one Python function, `flatten_variant(lowered, captured, memory) -> tuple`, of plain
ints, bytes, tuples and callables. The constructor parses it once. The tuple holds:
- `program`: rows as `(op, a, b, c, imm)`, the leaf table, and the `valid` and `predicate` row
  indices.
- `segments`: for each one, the `_CUDAGraph` object (a pybind shared_ptr holder, so C++ holds
  `std::shared_ptr<at::cuda::CUDAGraph>`) and its exec handle.
- `launches`: for each one:
  - node handle and function handle, block, smem and grid rows;
  - slots, as `(scalar, width, row)` or `(pointer, displacement row, address row, base index)`;
  - the initial held image, exactly the bytes Python's `held` starts from.
- `allocations`: dtype, size, stride and nbytes rows.
- `steps`: `("graph", segment, first, last)` or `("eager", index)`, each with its `StepMemory`
  (tensors, temporaries, drops).
- `eager`: for each call:
  - the callable, which is the `OpOverload` or Triton `jit.run`;
  - a leaf template: constants pre-boxed, plus tensor slots that point at a LoweredView;
  - the pytree spec;
  - the PredictedOutput rows.
- `outputs`: LoweredView rows.

Why flat tuples:
- C++ never imports or pybinds the Python dataclasses, so lowering can keep changing.
- The handoff format is one function with one round-trip test.
- The C++ object is immutable apart from `held` and the per-call scratch.

The Python `Variant.held` list goes away when native is on. The object exposes `held_images()`
for the oracle tests.

## 4. The commit loop in C++

`variant.native.commit(args) -> outputs | None` takes the place of `lowered.evaluate` +
`_replay` + `_commit` + `_outputs`. It returns None on a failed status or predicate, and Python
then tries the next variant, as it does today. The steps run in Python's order:
1. **Evaluate.** The program runs into a per-object `std::vector<int64_t>`, reused across calls.
2. **Allocate.** Allocations use `at::empty_strided` on the current device and stream. The
   temporaries use one `at::empty({total}, kByte)`, with `place` ported line for line
   (first-fit, 512-byte blocks). The `_ALLOC_ALIGNMENT` check stays as a `_Disagreement`.
3. **Patch.** For each launch, the object builds the images into a param buffer it keeps per
   launch, then does a `memcmp` against `held`. On a difference:
   - set held to invalid;
   - call `cuGraphExecKernelNodeSetParams` with the object's prebuilt `CUDA_KERNEL_NODE_PARAMS`,
     whose `kernelParams` points at that buffer;
   - copy the images into held.

   The driver entry points go into `c10/cuda/driver_api.h`. Main has no graph entries there
   today; the reference added `cuGraphExecKernelNodeSetParams` / `cuGraphKernelNodeGetParams`
   under 12.0. Why: no link against libcuda, and the same path as the existing table.
4. **Replay.** `graph->replay()` handles the generator prologues and the current stream.
   `cudaGraphLaunch` is not called directly, which keeps philox state correct for free.
5. **Eager step.**
   - C++ builds one `at::as_strided` per distinct LoweredView, keyed by view index. This is REVIEW_1B
     finding 1's fix: the same Python object for the same traced tensor.
   - It fills the leaf template, then calls back into Python with the GIL held:
     `callable(*tree_unflatten(leaves, spec))` under `at::AutoDispatchBelowADInplaceOrView`.
   - The output check runs in C++ against the PredictedOutput rows (sizes, strides, offset,
     dtype, base identity). An op that returns None gives zero outputs (REVIEW_1B finding 2).
6. **Drop.** The scratch tensor is dropped after its graph is queued, and `tensors[b]` is dropped
   per `drops`. Tensors are RAII `at::Tensor` in a per-call vector, so an exception frees them in
   the same order Python's frame teardown does.
7. **Outputs.** The object builds the output views and returns them as a tuple.

Why callbacks for eager steps first rather than `callBoxed`:
- It keeps the exact Python semantics: torch function modes, `OpOverload.__call__`, and Triton
  `jit.run` fallbacks. The Python-state-frozen contract is unchanged.
- The C++ glue alone takes about 15 us of the 19 us.

Dispatching `aten::` ops through `c10::Dispatcher` `callBoxed` is follow-up 3.5. It applies only
when the target is a plain `OpOverload` with no `py_kernels`, and is decided at flatten time. The
Python callback stays the fallback.

## 5. Memory parity with eager (the hard requirement)

- Every allocation goes through `at::empty_strided` / `at::empty` on the current stream. That is
  the same c10 caching allocator and the same pool as eager. There is no private pool, no hold
  mode and no arena.
- The alloc/free order matches the Python loop exactly. So parity with eager is whatever the
  Python loop already has; the one-scratch-per-commit choice is item 1's and is unchanged.
- Inputs are never freed or retained beyond the call. The object keeps no references to input
  tensors between calls.
- Test: for each chain in `test_cuda_host_trace_replay.py`, run the Python path and the native
  path from a fresh `reset_peak_memory_stats()`. Assert equal `allocated_bytes.all.{current,peak}`
  and equal `num_alloc_retries`. Also assert that peak is at most eager's peak, which is the
  existing parity assert.
- Pluggable allocators work unchanged, because `at::empty_strided` goes through the swapped
  `CUDAAllocator`. The alignment check keeps them sound.

## 6. What stays in Python

- `HostTraceReplay.__call__`, the RLock, warm-up, trace, lowering, capture and the miss policy (a
  miss re-traces and never runs eager silently).
- `_Disagreement` handling: C++ raises a dedicated `torch._C._HostTraceDisagreement(op, reason)`,
  and Python drops the variant and blocklists the op as today. This includes REVIEW_1B finding
  4's `if variant in family` fix.
- The contract, until 3.4. 3.4 adds `torch._C._host_trace_tensor_key(args)`, the tensor half of
  the contract as one hashable tuple built in C++. `global_state()` stays Python (0.18 us).

## 7. Build

- Code lives in libtorch: `torch/csrc/cuda/HostTrace{Program,Variant}.{h,cpp}`, bound in
  `torch/csrc/cuda/Module.cpp` next to `_CUDAGraph`. Everything is underscored `torch._C._*`, and
  nothing is public.
- It builds only with `pip install -e . -v --no-build-isolation`. When other agents share the
  binary, use the isolated build route (memory: paramgraph-upstream-isolated-build).
- It builds with no host compiler at runtime and no `CppCodeCache`, which removes the
  `torch._inductor` import from `torch.cuda`.
- No C++20 features, so the aarch64 box's toolchain question stays out of scope.

## 8. Targets (per call, CPU, GB300, same harness as BOUNDARY_COST)

| case | eager | Python replay today | target after 3.3 |
|---|---|---|---|
| N=16 K=0 hit | 261 | 78 | 8 or less |
| N=16 K=0 patch (all dirty) | 261 | 192 | 20 or less |
| N=8, each boundary K (hit) | n/a | +56/boundary | +15/boundary or less, eager op included |
| fixed (N=1 hit) | 17 | 32 | 5 or less |

Reference points to report next to these:
- plain `torch.cuda.graph` replay: 2.5-3.6;
- the prototype's native replay: chain 15.2, GPT-2 71-91;
- cudagraph trees: HF GPT-2 467, TinyLlama 802, train 2935;
- the prototype native on the same models: 118, 253, 177.

The paper goal is at most 1.5x the prototype native on GPT-2 once items 2 and 4 remove the mm
boundaries. Time trees and native in a process that never compiles (memory: output-arena note).

## 9. PR slicing (in order)

| PR | contents | tests | needs |
|---|---|---|---|
| 3.0 | `HostTraceProgram` interpreter + `torch._C._host_trace_evaluate(rows, leaves, args)`; `LoweredTape.evaluate` uses it; `compile_program` becomes test-only | differential fuzz vs `_step`; all replay tests unchanged; subprocess check that no compiler is invoked | 1b |
| 3.1 | `_HostTraceVariant` with patch + held + replay only (steps 1, 3, 4); Python still allocates and runs eager steps and passes bases in | native vs Python: `held_images()` equal after every call of 12-way variant alternation; interrupt test (exception injected after the setter) | 3.0, driver_api entries |
| 3.2 | allocate/drop/scratch/outputs in C++: the whole loop for graph-only variants | memory_stats equality vs Python and parity with eager on every chain; REVIEW_1B `p2_memory` simulator on the native path | 3.1 |
| 3.3 | eager steps: views, callback, output check, `_HostTraceDisagreement` | REVIEW_1B p1/p5/p6/p7 probes as tests, native vs Python bitwise; reentrant same-entry case | 3.2, 1b fixes 1/2/4 landed |
| 3.4 | `_host_trace_tensor_key` contract fast path | family hit/miss parity over the existing contract tests; 222-input microbench | 3.0 |
| 3.5 | optional: `callBoxed` for plain aten ops | same tests as 3.3 with dispatch forced both ways | 3.3 |

Each PR flips `native=True` as the default only for what it covers. The Python path stays
selectable, and CI runs the replay suite in both modes (`@parametrize("native", ...)`).

## 10. Risks and open questions

1. **An exception mid-commit** (a callback raising, OOM in `empty_strided`). Held stays
   consistent because it is set to invalid before each setter. The tensors are RAII. Earlier
   segments' effects are visible, as they are in Python (REVIEW_1B finding 3). Recommendation:
   identical semantics to Python, tested by injecting failures at each step kind.
2. **GIL and reentrancy.** Recommendation: hold the GIL for the whole commit. Releasing it would
   need a per-object mutex and gains nothing while eager steps call Python. A reentrant commit
   of the same object needs its own values/scratch, so the per-call state moves to a small
   stack-allocated frame; only `held` is shared, and it mirrors the exec.
3. **Pybind overhead at the boundary** (about 1-2 us for args parsing). Recommendation: take
   `args` as a raw `PyObject*` tuple via `THPUtils` and unpack tensors with
   `THPVariable_Unpack`. No `py::args` and no `isinstance` on Parameter (the adapter-lean
   lesson).
4. **Two maintained evaluators and loops.** Recommendation: the Python ones stay as the oracle,
   not as a fallback. Remove the Python `_commit` once the paper numbers are in and 3.3 has
   soaked. Until then the dual-mode CI keeps them honest.
5. **Overlap with the other team's native runtime** (`BoxedReplayPlan`). Recommendation: don't
   depend on it or share headers. If upstream lands it first, `_HostTraceVariant` becomes a thin
   client of its patch primitive. The interpreter is the piece most worth reconciling with
   `parameter_program::Program`.
6. **Driver API version floor.** `cuGraphExecKernelNodeSetParams` needs CUDA 12.0 in the
   driver table. Recommendation: without it, `native` is False and the Python path runs.
   Nothing declines.
7. **Scalar slot widths and floats.** `_SCALARS[size]` packs ints only today. Recommendation:
   port it exactly, and reject unknown widths at construction rather than at call time.

## 3.2 The whole hit in C++ (entry to return)

The directive: a replay hit runs no Python between the user's call and its return. The only
exception is a 1b eager boundary step, which C++ calls back into. The time before the kernel
launch matters most, but the time after it matters too. The target for a hit is about 5 us in
total, against 23.9 us at 3.1 (16-kernel chain) and 33.2 us all-patched.

The 3.1 breakdown shows where that time goes. There is a fixed Python cost of 15.2 us. Of that,
`_place` and `allocate` take 8.4 us, outputs and drops take about 6, and the entry takes about 7
(contract 0.8, device context plus capture check 2.1). The C++ core already takes only 3.2 us on
a hit (10.1 all-patched), and cudaGraphLaunch is about 2.5 of that. So 3.2 moves everything
around the core into C++ as well. This absorbs the table's 3.2, 3.3 (the eager-step frame; the
op call itself stays a callback) and 3.4 rows.

**Entry object.** `torch._C._HostTraceEntry` is a raw CPython type whose `tp_call` is the whole
fast path. It is subclassable: `HostTraceReplay` derives from it and has no Python `__call__`, so
calling the object enters C++ directly with no Python frame and no pybind argument parsing.
Python configures the entry once with
`_native_init(active_local, disagreement_cls, trusted, inductor)`, and adds or removes families'
variants with `_native_add(key, native)` and `_native_remove(native)`. `replays` is a C member,
so `self.replays += 1` from Python and the C++ increment count the same thing. On a build
without CUDA the type does not exist. The Python base then runs every call eagerly, with a
decline reason.

The fast path in `tp_call` runs these steps in order. Any "no" hands the call to Python's
`_call_slow(args, kwargs, searched)` unchanged:
1. No kwargs, and `_active.trace` is unset (`PyObject_GetOptionalAttr` on the tape module's
   `threading.local`; a call inside an outer trace is traced inline, in Python).
2. There is at least one native family.
3. The native contract key (below). Under trust there is no key, and every variant is in one
   family.
4. The family lookup: a linear scan over a few int64 vectors. A missing family means slow path
   with `searched=False`.
5. A capture check on the current stream (`currentStreamCaptureStatusMayInitCtx`). It is
   repeated under a `CUDAGuard` only when the variant's device differs from the current one.
6. Per variant in the family: `bind` + `evaluate` + the `valid` row, into a stack buffer. If no
   variant holds the call, it takes the slow path with `searched=True`. Python then goes straight
   to `_miss` without evaluating the family a second time.
7. Commit (below), then build the result: None, a tensor, a tuple or a list, per `result_kind`.

The native contract key encodes `argument_contract(args) + global_state()` as int64s:
- a tensor: its exact type pointer (the family holds a strong reference), dtype, device type and
  index, dim, layout, neg and conj;
- an exact int: its tag only;
- None: its tag;
- a bool: tag and value;
- an exact float: tag and bits;
- the global state: the default dtype, deterministic algorithms and deterministic fill.

Any other argument makes the call unrepresentable, and `_native_key` returns None. Python then
registers the variant without a native key, so the variant is reachable only through
`_call_slow`, which evaluates it with `native.call(args)`. Equal Python contracts give equal keys
on the representable subset, and vice versa, so the native families partition calls exactly as
the Python families do. `_native_key(args)` is exposed so that `_miss` registers the new variant
under the same key the entry computes.

**Inductor convention.** `_Installation.__call__(new_inputs)` becomes the same entry with
`inductor=(constants, nones)`:
- it takes one list argument, appends the constants into a `PyObject*` array and clears the
  list (Inductor's input-freeing contract; the frame holds the references);
- it returns a list with None reinserted at `nones`.

The slow path gets the same list. This removes the second Python frame and the tuple-to-list
copy from every Inductor hit.

**Commit in C++** (the Python `_commit`, ported step for step; the Python loop is deleted, N3):
- **Allocation.** Tensors come from `at::detail::empty_strided_cuda` on the current device and
  stream, which is the caching allocator without the dispatcher: the same block, stream and
  lifetime as eager's `torch.empty_strided`. Temporaries are placed per step by a C++ port of
  `place` (first fit, `_BLOCK` 512). It runs over at most a few temporaries and needs no cache.
  The scratch is one `at::detail::empty_cuda` byte tensor, released right after the step's graph
  is queued, as in Python. Drops reset the `at::Tensor` slots. Memory parity with eager is the
  Python plan's, with no hold mode.
- **Alignment.** A misaligned tensor before the first effect returns `Misaligned` (Python runs
  the call eagerly with the reason). After the first effect it raises the `_Disagreement` class
  passed to `_native_init`.
- **Graph steps.** Kernel nodes are patched as in 3.1. Memset nodes (N1) are patched with
  `cudaGraphExecMemsetNodeSetParams` (runtime API; `dst` = base, `width` = nbytes row,
  `elementSize` 1, `height` 1, value 0, the same as `_MemsetParams`), with held
  `(dst, nbytes)` compared first. No driver-table change is needed.
- **Direct replay (N2).** The segment's graph is replayed through the C++
  `at::cuda::CUDAGraph::replay()` when nothing observes Python replays:
  - `torch.cuda.graphs._global_replay_start_hooks` / `_end_hooks` are empty;
  - the graph's `_replay_start_hooks` / `_replay_end_hooks` are empty;
  - `_tracker` is None;
  - `_retained.sync_before_fire` is False.

  These are dict-size and identity checks on objects captured at construction: the hook dicts
  are never rebound, and the graph's instance dict is read directly. The check costs tens of ns.
  If any check fails, the step calls the Python `CUDAGraph.replay()` wrapper, so hooks, the
  liveness check and the replay-stream record for `synchronize_before_release` keep their exact
  semantics. We don't record the stream natively: its `torch.cuda.Stream` value has to be the
  one the Python holder compares, and the case is opt-in and rare.
- **Eager steps** (the 1b boundary; the only Python on a hit that has one). C++ builds the
  leaves:
  - each `LoweredView` is one `as_strided` of its base (an argument, an allocation or an earlier
    eager output), deduplicated per view;
  - a `ScalarSlot` is a Python int of its row;
  - a constant is the held object.

  It then calls the step's Python closure `run(leaves, values)`, built at flatten time, where
  `values` is a read-only memoryview released after the call. The closure is the old `_run_eager`
  tail, unchanged in semantics: the pytree unflatten, the call below ADInplaceOrView (or
  `jit.run(grid=...)` for a Triton target), and the output check against `PredictedOutput`,
  which raises `_Disagreement`. It returns `(root, tensor)` for each fresh output, and C++ stores
  the tensor and its storage base. Keeping the check in Python keeps its message and dtype and
  device reprs identical, at no cost on graph-only hits.
- **Outputs.** After the last launch: `ScalarSlot` gives an int; `LoweredView` gives
  `at::Tensor::as_strided` of its base (the same view semantics as the Python loop); an
  allocation or eager reference returns that tensor wrapped (`THPVariable_Wrap`); `("output", i)`
  aliases. Everything else is dropped by RAII when the frame ends.

**Per-call state and reentrancy.** All per-call state (rows, leaves, bases, tensors, scratch)
lives in a stack frame. Only `held` is shared, and it mirrors the exec. An eager step may call
the same entry again. The inner call gets its own frame, and it leaves the nodes it patched
recorded in `held`, so the outer call re-patches what it needs, as in Python. The entry holds a
`std::recursive_mutex` for the whole call, including the slow-path callback, which replaces the
Python RLock. It takes the lock with `try_lock` and otherwise releases the GIL while waiting, so
a thread blocked on the mutex never holds the GIL that the owner's callbacks need. The frame
holds a strong reference to its variant object, so a nested call that drops the variant (a
disagreement) cannot free it mid-commit.

**Errors.** C++ exceptions (OOM, a driver error, a width check) become Python exceptions at
`tp_call` (`HANDLE_TH_ERRORS`). For a `_Disagreement` (raised by a closure, or by C++ for
alignment), the entry calls `self._disagreed(native, exc)` before re-raising. That call drops
the variant from the Python and native families and records `exc.op` in `_meta_disagrees`.
Construction errors in the native variant are raised as AssertionError, the lowering's bug, not
a silent permanent decline (N1).

**What stays in Python:** misses (trace, lower, plan, capture, flatten), the first call,
retrace, declines and fallbacks, `_disagreed`, keyword binding, calls inside an outer trace, and
unrepresentable contracts. Of the replay path, only the eager step closure remains. Deleted:
`_commit`, `_NodeParams`, `_MemsetParams`, the runtime half of `allocate`, `_outputs`, the
`TORCH_HOST_TRACE_NATIVE` switch, and the Python `__call__` on CUDA builds. The oracle for
node patching becomes a test: `native.held_images()` must equal the Python-computed
`launch_images` after every call of the variant alternation.

**Measurement.** Report entry-to-launch CPU time and total per call, with the GPU held busy (a
long spin kernel queued first), for three cases: the 16-kernel hit, the all-patched case and one
boundary case (an eager step between two runs). Entry-to-launch is measured with a C++ timestamp
taken at the first `tp_call` instruction and just before `cudaGraphLaunch` of the first segment
(a test-only counter, `_native_timing`). Total is `perf_counter_ns` around the call.

**As built (3.2, 2026-09-25).** Deviations from the above:
- `_native_init(active_local, disagreement_cls, trusted)` has no `inductor` argument. The
  Inductor convention is not built: `_Installation.__call__` stays a Python frame over the entry
  (list-to-tuple, input freeing, None reinsertion). It is the one Python frame left on an
  Inductor hit, and the next slice.
- The timing counter is `torch._C._host_trace_timing(enable)`, which returns the last hit's
  entry-to-launch ns.
- The misaligned test drives the C++ allocation through a `MemPool` over a custom allocator that
  returns +16-byte segments, since mocking `torch.empty_strided` no longer reaches it.
- Follow-up from 4.2 (STEP4_5_OPAQUE_DESIGN section 1), not in 3.2's representation: TritonLaunch
  `fields`/`cluster`/`image`, the unified `LoweredMemset(seq, launch, slots, width, height,
  pitch)` patched with dst/value/elementSize/width/height (pitch when height > 1), `owner` as the
  provider, and opaque-key binding before effects. flatten_variant and the C++ Launch take these
  when 4.2 lands.
