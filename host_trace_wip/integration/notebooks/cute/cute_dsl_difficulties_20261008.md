# CuTe DSL under CUDA graph host tracing: the difficulties (long version)

> **AI-generated draft** (Claude Code, for Elias to review and edit). Not reviewed yet; nothing here has been sent to the CuTe DSL team. Refreshed 2026-10-08 for our internal snapshot "candidate 3" (int6 `snap/step246`), which has patch 97 (guards from TVM-FFI's argument spec), 97b(a) (structured builtin types) and 98 (cuBLAS workspace always allocated). Patch 97b(c) (layouts evaluated by the DSL's layout algebra at compile time) is **not** in it: it was pulled for heap corruption on GB300, and its redo is pending for candidate 4 (item 2).

This is the companion to the notebook "Dynamic Shape CUDA Graphs with CuTe DSL" (`cute_dsl_host_tracing_20261008_v2.ipynb`). It covers each difficulty we hit integrating CuTe DSL kernels into host tracing that comes from how CuTe DSL works (its JIT, launch, binding and descriptor design) and that an API change on their side could remove. For each: what broke, the root cause, our workaround with its size, what it costs to maintain, the request that removes it (R1-R7a, numbered as in the notebook's section 8; R7a-c are its nice-to-haves: a compile callback, stream handles out of mangled names, TMA descriptors), and a status:
- **fixed:** fixed on our side;
- **in progress;**
- **open:** an interim workaround, and the DSL API would remove it;
- **blocked on the DSL:** we can't do it soundly without the DSL.

**The rule** (user, 2026-10-07): capture facts at compile time and require them, consume structured data, and decline what doesn't provide them. Never go from full to lossy to recovered; sound is simpler even with more LOC.

**How references work:**
- `cute.py:N` is `torch/cuda/_host_trace_cute.py` in candidate 3's even top, `land/int6/snap/step246` (1643 lines, 1408 non-blank). `launch.py` and `capture.py` are the sibling `_host_trace_*.py` files there. `dsl.py` is `cutlass/base_dsl/dsl.py` in nvidia-cutlass-dsl 4.6.2.
- LOC counts are non-blank lines.
- **"Decline"** means the call is recorded as a counted eager call that the replay runs as eager would. `_intercept` turns every exception into one (`cute.py:1166-1174`). **Crash** marks where that doesn't hold.

---

### 1. Compiles are visible only through DSL internals (R7a, R1, R2): open

- **What broke:** we need each compile's host function and argument spec. A function compiled before we arm can't be traced ("its compile was not observed ...", `cute.py:1130-1132`).
- **Root cause:** the DSL. No compile observer is given the compiled object, and the compiled object doesn't carry what we need.
- **Workaround (two seams since 97, both installed once and never restored):**
  - `CompileCallable.__call__` is patched (`install`, `cute.py:1185-1215`; `_compile`, `1306-1360`), to add a trace-finalize hook for the host function. It also turns torch SymInts at compile positions into guarded ints, and makes a compile that fails under a trace a decline.
  - `CutlassBaseDSL.compile_and_cache` is wrapped to keep its inputs (`_compile_and_cache`, `1362-1367`), from which `_compile` builds the argument spec (item 12).
  - The text and the spec are tied to the result by `result.function_name`.
- **The process-wide `register_trace_finalize_hook` (`dsl.py:1765`) is not enough:** it is reached through the private `CuTeDSL._get_dsl()`; it is given a function name, not the compiled object; it runs on every jit trace; and it runs before the cute.experimental passes that add TMA operands.
- **Maintenance:**
  - A renamed class or a changed `compile_and_cache` signature is an ImportError or TypeError at import or at compile (**crash**).
  - A spec that fails to convert is recorded as the reason, and the call declines (`1357-1358`).
- **Status:** open. R1 and R2, as attributes of the compiled object, would make R7a unnecessary.

### 2. The host function is IR, and we interpret it (R1, R4): builtin types done (97b(a)); layouts still read from type text in candidate 3, a structured route pending for candidate 4; the TMA atom and `cute.assume` parts are blocked on the DSL

- **What broke:** nothing exposes the grid, block, shared memory, cluster, launch attributes or kernel parameter values as data.
- **Workaround:**
  - The hook prints the host `func.func`. For cute.experimental it first runs the DSL's passes up to `cute-to-nvvm` on a copy, through the private `dsl._get_pipeline` (`cute.py:1309-1323`).
  - `_Program` parses the text with MLIR's parser in our own context (`363-475`).
  - `_Eval` interprets about 45 ops (`523-825`).
  - In all, about 680 non-blank lines (`54-833`). Per the user (2026-10-07), interpreting compile-time MLIR through the bindings is acceptable; parsing printed text with regexes is not.
- **97b(a) (in candidate 3; `core/97b_cute_builtin_types_structured.*`):** integer, float and index types are read through MLIR's type classes (`IntegerType.width` signless, `FloatType.width`, `IndexType`; `_Builtin`, `cute.py:118-133`). `_INT_TYPE_RE` and the `"f32"`-style compares are gone. Kernel identity is the callee `SymbolRefAttr` against the launched node's name, a symbol comparison.
- **97b(c) (pulled; pending for candidate 4):** it evaluated layouts, shapes, strides, int tuples, tiles and coordinate tensors' layouts at compile time with the DSL's own layout algebra (`cute_ir.get_layout` / `get_shape` / `get_stride`, `core._unpack_x_tuple`) on a re-parsed copy of the module in the DSL's context. On GB300 it corrupted the heap ("malloc(): unaligned tcache chunk detected", rc 134) in the sm100 GEMM tests, because asking for a value runs the DSL's value caster, and it did so for every value, MMA and TMA atoms included (item 3). GH200 skips those tests, which is why it passed there first. The redo, 97b_c2 (cast only values of the layout-like types) plus 97b_c3 (a coordinate tensor's basis strides as static leaves), passes the GB300 cute tests with no corruption and is queued for candidate 4 (`core/READY.md`, `core/cute70/`). Nothing in candidate 3 uses it.
- **Text inference still left in candidate 3 (`snap/step246`):**
  - layout, int tuple, shape, stride and tile types: `_LEAF_RE` (`:111`), `_parse_tuple` (`:175`), `_static_value` (`:208`), `_memref_pattern` (`:221`), `_is_runtime` (`:229`), `_int_widths` (`:299`); this is what 97b(c) replaces;
  - `!cuda.stream` compares on a formal's type (`:461`, `:479`, `:491`, `:1140`);
  - `cute.assume`'s `<divby N>` (`:680`), because `_cute.ConstrainedIntType` has no accessors;
  - the TMA atom's `tma_format` and `tma_gbasis` (`:736-737`), with the memref alignment and smem swizzle around them, which have no structured source.

  The last two are R4, the binding accessors; the coordinator's notes call this R8. Repros: `scratch/cute_notebook/repro_structured_types.py` and `core/cute70/probe/p97b_types.py`, both without a GPU.
- **Bindings gaps found on the way (R4):**
  - no leaf accessors on `ShapeType`, `IntTupleType` and `LayoutType`;
  - `LayoutType(t).stride` on the nested layout `(4,(8,?))` raises `std::get: wrong index for variant`, which fails the compile from the trace-finalize hook.

  97b(c) and its redo avoid both by asking the DSL's Python layout algebra instead of the type. The notebook's toy evaluates layouts the same way as the redo (only values of layout-like types are taken, on a module copy).
- **Coverage gaps still open:** FlashInfer `mm_mxfp8` has about 9 gaps, about 200 LOC estimated (ISSUES U13).
- **Maintenance:** a renamed op or changed attribute becomes a decline; a wrong evaluation is caught by item 5.
- **Status:** done for builtin types. Layouts: on type text in candidate 3; structured evaluation pending for candidate 4. `cute.assume` and the TMA atom fields are blocked on the DSL (R4, or R1).

### 3. Reading live IR values corrupts the heap (R1, R4): open

- **What broke:** an earlier route that walked the live module hit "malloc(): unaligned tcache chunk detected", a SIGSEGV on the 4th compile, and a BlockArgument lifetime crash.
- **Root cause:** the DSL's Python value casters emit `get_iter` ops into the module when a MemRef value is read.
- **Workaround:** never touch live values. The interpreter reads a printed copy parsed into a private context where the DSL dialects aren't registered (`cute.py:435-436`).
- **Seen again in 97b(c) (2026-10-07):** casting every value of a module copy in the DSL's context corrupted the heap on GB300 sm100 ("malloc(): unaligned tcache chunk detected"), even though the live module was untouched. So it is the casters themselves on some values (MMA and TMA atoms), not only the ops they add. The redo casts only values of the layout-like types (item 2).
- **Maintenance:** this depends on the private module path `cutlass._mlir._mlir_libs._cutlass_ir`; if it moves, every call declines.
- **Status:** open. The early crashes are from notes (paramgraph-cute-descriptor and paramgraph-cute-bridge-comparison memory files, CUTE_SPIKE.md); the 97b(c) crash is from the CuTe lane's GB300 runs (int6 MERGE_LOG, DECISIONS 2026-10-07); the workaround is verified in the code.

### 4. The kernel parameter ABI is undocumented (R1): open

- **What broke:**
  - Memref structs carry uninitialized padding.
  - The SM100 `tiled_mma` has a static type but a 32-byte all-zero parameter.
  - `compiled.kernel_info` is `{}`.
- **Workaround (about 60 LOC):**
  - Sizes come from `cuFuncGetParamInfo` on the captured node (`capture.py`).
  - `_PARAM_STATICS` lists the static types that have a parameter (`cute.py:90`), by a prefix compare on the type text (`790-811`), and the MMA zeros are checked (`1004-1005`).
  - Padding is masked out of the compare (`994-1021`, `1058-1060`), and a parameter count mismatch declines (`993`).
- **Status:** open.

### 5. A second capture at every trace (R1): open (kept by decision)

- **What broke:** each trace also runs the call over stand-in tensors under a capture that is never replayed. It is our source of each launch's CUfunction and parameter sizes, and our byte-for-byte check of item 2. It costs about 0.27 ms per call per trace (PLAN.md, 2026-09-25).
- **Workaround:** `_stand_in` (`cute.py:1068-1083`), the capture (in `_intercept`, `1150-1157`), and the compare in `_describe` (`1049-1066`).
- **Status:** open. It is kept as the check (DECISIONS, 2026-10-07). With R1, it becomes a debug check.

### 6. TMA descriptors are encoded inline (R7c, or R1): open, blocked on the DSL for the atom fields

- **What broke:**
  - The DSL builds each `CUtensorMap` inline. It writes bytes 0-63 of 128 and sets byte 8 bit 1, which no driver encode option sets.
  - It stores byte strides in units of 16, rounded down.
  - The other encode arguments aren't in the IR.
- **Workaround (about 110 LOC):**
  - The atom's fields are read by regex from its type (`Eval.tma`, `cute.py:732-776`).
  - The replay re-encodes with `cuTensorMapEncodeTiled`, or uses `cuTensorMapReplaceAddress` when only the address moved, and ORs in that bit (`launch.py:22-24, 117-185`).
  - 45/45 captures matched in the 2026-09-25 probe.
- **Maintenance:** bytes 64-127 are never compared (unverified risk).
- **Status:** open, blocked on the DSL for the atom fields. They are still read from the atom's type text (`:736-737`), because there is no structured source (R4, or R7c/R1).

### 7. The stream a launch uses is implicit (R1, R6): open

- **What broke:**
  - A `.launch()` without `stream=` runs on the NULL stream: our old check call ran the kernel for real (REVIEW_CUTE61 C3).
  - TVM-FFI's environment stream drops the stream formal and launches on torch's current stream.
- **Workaround (about 30 LOC):**
  - Decline unless the launch config's stream operand is a `!cuda.stream` formal (`cute.py:458-463`).
  - Bind the environment stream (`479-485`).
  - Accept only the trace's stream (`1112-1124`).
  - Swap in a side stream for the check capture (`1151-1152`).
- **Status:** open.

### 8. `from_dlpack` needs a live tensor (R5): open

- **What broke:** a traced tensor has no storage to export.
  - SGLang's TGV GEMM had bound `from_dlpack` by name before we installed (ISSUES K3, prep item C6).
  - The `cute_without_tvm_ffi` switch and its `sys.modules` walk (int6 step 6) were retired by patch 70 (steps 123-124), which hooked `_Tensor.__new__`.
  - That hook can't be removed: once set, CPython keeps `slot_tp_new`, so every later eager `from_dlpack` raised "object.__new__() takes exactly one argument" (found in patch 75).
- **Workaround (about 46 LOC):**
  - During traces we replace the module global `cutlass.cute.runtime._Tensor`.
  - The replacement returns a `_DLPackArg`, which records the layout marks and `element_type` sets and rebuilds the tensor on stand-in and replay tensors (`cute.py:1247-1305`, held at `1501-1528`).
- **Status:** open.

### 9. Fake tensors, `make_ptr` and argument types (R5): open

- **What broke:** fake tensors take the DSL's own `cute.sym_int32`; `make_ptr` of a symbolic address raises a TypeError; and `torch._native` wraps inputs in `ReadOnlyTensorWrapper`.
- **Workaround (about 20 LOC):**
  - `_FakeTensor.__init__` is patched during traces (`_fake_tensor`, `cute.py:1235`).
  - The `make_ptr` TypeError becomes a labeled decline (`1636-1639`).
  - `_read_only` passes traced tensors through (`1239`).
- **Maintenance:** if `__init__` moves to a base class, `vars(owner)[name]` raises at trace entry (**crash**, `1519`).
- **Status:** open.

### 10. Functions compiled without TVM-FFI (R5, R6): open, with guards sound since 97

- **What broke:** a function compiled without TVM-FFI (SGLang's TGV) takes CuTe tensors built by cutlass's `TensorAdapter`. A direct `@cute.jit` call has no route at all.
- **Workaround:** `JitCompiledFunction.__call__` is hooked separately (`_traced_call`, `cute.py:1445-1481`; `_call_classes`, `1492`). Since 97, such a call is guarded as if it had a binder: the spec comes from the same converter, with the dtype taken as the storage width, because there is no binder to check it. Direct jit calls decline.
- **Status:** open.

### 11. TVM-FFI's argument flattening (R2, R6): mostly fixed by 97

- **What broke:** quack passes NamedTuples. TVM-FFI flattens them, drops `None`, and passes numerics by value. Our flattening once broke quack's compile (patch 22).
- **Now:** the spec has tuples and constants, so `_spec_args` drops constants and `None` according to the spec, not by position (`cute.py:875-954`). `_fields` still flattens NamedTuples for the stand-in call (`1084-1100`).
- **Status:** mostly fixed. A public spec (R2) would make it fully structured.

### 12. Sizes that share a symbol: fixed by 97

- **What broke:** with one `cute.sym_int32` shared across tensors, TVM-FFI's binder checks that the sizes are equal, but the host types print each as `?`. Before 97, `_memref` guarded each tensor alone.
- **Repro** (`repro_shared_symbol.py`):
  - **On step188 (2026-10-06):** eager raised `ValueError: Mismatched mB.shape[0] ... expected to match mA.shape[0]`, but the replay ran the call (traces=1, replays=2, eager=1, no declines). For the second half it returned `a` plus the memory past `b`'s end.
  - **On s188_97** (`core/cute70/logs/p97c/repro_after.log`): the replay raises eager's error, `ValueError: Mismatched mB.shape[0] ... expected to match mA.shape[0]`, ending at traces=2, replays=1, eager=1. The second trace is the miss's retrace, whose eager warm-up raises; replays stays at 1.
- **Fix (97, +107 lib / +48 tests):**
  - `_compile` runs the DSL's own `_tvm_ffi_args_spec_converter` on the inputs kept by the `compile_and_cache` wrapper (`cute.py:1347-1367`).
  - `_encode_spec` stores the spec as JSON (`845-874`).
  - At each traced call, `_spec_args` turns every binder check into a guard (`875-954`):
    - static sizes and strides;
    - each later use of a Var equal to its first;
    - divisibility;
    - int8/16/32 bounds;
    - alignment;
    - dtype, ndim and device;
    - constants;
    - the size-1 stride skip, as the binder makes it.
  - The type-text guards (`div=`, `align<`, element-type regexes, `_MEMREF_DTYPES`) are gone.
  - Tests: `test_a_shared_size_symbol_is_guarded`, `test_the_spec_guards_divisibility_and_alignment`.
- **Status:** fixed.

### 13. Objects loaded from disk carry no metadata (R3): open (sound since 97)

- **What broke:** quack's `jit_cache` loads `.o` objects with nothing beside them. It exports to a temporary path and renames it. Its `cute_dsl_elf_fix` replaces `ExternalBinaryModule.__init__`.
- **Workaround since 97:**
  - Sidecar format 2, `{name, text, spec, version, object digest}`, is written beside the object and under its digest (`_export_to_c`, `cute.py:1379-1389`). (97b(c) would move this to format 3; it is not in candidate 3.)
  - A load takes only a current-format record whose digest is this object's (`_record`, `1397-1413`). A record without the spec declines: "exported without TVM-FFI's argument spec: compile and export it again (clear a jit cache that holds it, e.g. quack's)" (`_exported`, `1420-1430`).
  - quack's `jit_cache` asks `has_host_function`, so its stale entries recompile (`torch/_vendor/quack/cache/jit.py`).
  - `ExternalBinaryModule.__new__` and `__getattr__` are patched (`_load`, `1390-1396`; `_lookup`, `1431-1440`).
- **Maintenance:** `_load` calls the saved `__new__(cls)` with no arguments (`1390-1396`). If the DSL gives the class a `__new__` that takes arguments, every load fails (**crash**).
- **Status:** open.

### 14. Compiling under a trace (R5, partly): open

- **What broke:** a compile that receives a torch SymInt through a caller's object raises (quack's `rmsnorm_bwd`, keyed on `T_hint`). Mostly a quack problem.
- **Workaround:** a counted decline, retried once per function and replay (patch 90, `cute.py:1330-1345`).
- **Status:** open (not DSL-side, apart from accepting int-like values).

### 15. A stream handle in the mangled function name (R7b): open, doesn't affect us

- **What broke:** `mangle_name` strips only `0x[a-f0-9]{8,16}` (`dsl.py:1037`), and GB300 stream handles have 7 hex digits. So a jit call that takes a stream compiles and loads once per stream, and a capture on a side stream runs another function.
- **Status:** open, and not hit: we trace compiled functions only.

### 16. PDL, cluster and cooperative launches (R1): open

- **What broke:** these are separate `cuda.launch_cfg.*` ops on the config value.
- **Workaround:**
  - The ops are parsed (`cute.py:72-74`, `820`).
  - PDL becomes `programmatic=True` under `trace_pdl` (`980-981`).
  - The cluster is pinned, and other nonzero attributes decline (`977-990`).
- **Status:** open.

### 17. The binder has no check-only entry point (R2): open

- **What broke:** we re-implement the binder's checks from its spec (`cute.py:875-954`). `attach_ffi_func` generates one `__tvm_ffi_<name>` whose prologue runs the checks inline before the call. There is no separate check symbol in `tvm_ffi_builder` or `tvm_ffi_provider` (READY 97).
- **Maintenance:** if a DSL release adds a check, our guards miss it until `_spec_args` learns it, and such a call would replay where eager raises.
- **Status:** open. `compiled.check_args(*args)` would let a test compare our guards with the binder.

### 18. TVM-FFI functions from outside CuTe DSL (tvm_ffi, not CuTe DSL): open

- **What broke:** FlashInfer's TVM-FFI functions read sizes in C, so a whole call declined.
- **Workaround:** `tvm_ffi.core.Function.__call__` is hooked during traces, and such a call becomes an eager step (`_Foreign`, `cute.py:1541-1620`).
- **Status:** open, and it is for the tvm_ffi maintainers.

### 19. Dynamo can't trace through the DSL (probe only; not used by host tracing)

- **What was found** (CUTE_TRACING_PROBE.md, 2026-09-11; not re-run): a wrapper around a CuTe call is not a tracing boundary. Dynamo breaks inside the DSL on `posix.getcwd`, `ContextVar.set`, `PyCSimpleType.__new__` and `CUstream`, and `Int32(n)` specializes. R5, R1, a torch stream, and SymInt scalars would help a `torch.compile` integration.

---

## Retired

- **`cute_without_tvm_ffi` and `_rebind_from_dlpack`:** patch 70.
- **Construction hooks always installed:** patch 75.
- **CuTe rollout switches:** S2.
- **Type-text guards and unversioned sidecars:** patch 97.
- **Integer and float widths read from type text:** 97b(a).

## Omitted (not caused by CuTe DSL, or nothing their API would change)

- quack's size-keyed caches (U12).
- quack's `rmsnorm_bwd` `T_hint` key.
- The retry-once policy.
- FlashInfer's module-global alpha tensor.
- SGLang's `next_power_of_2` setattr.

## Sources

- **Code:** `land/int6/snap/step246/torch/cuda/_host_trace_{cute,launch,capture}.py` (candidate 3's even top); patches `land/core/97_cute_tvm_ffi_spec_guards.int6cpp.patch`, `97b_cute_builtin_types_structured.int6cpp.patch`; pending: `97b_c3_cute_layouts_evaluated_by_the_dsl.step242.int6cpp.patch`.
- **DSL:** nvidia-cutlass-dsl 4.6.2 (`cutlass/base_dsl/dsl.py`, `tvm_ffi_builder/`, `cutlass/cute/_tvm_ffi_args_spec_converter.py`, `cutlass/cutlass_dsl/cutlass.py`).
- **Notes:**
  - `land/core/READY.md` (20-28, 70, 72-76, 90-93, 97, 97b, 97b_c3, 98);
  - `land/int6/MERGE_LOG.md` (candidate 3, 97b(c) removal);
  - `agent_space/paramgraph/DECISIONS.md` (2026-10-07);
  - `land/core/ISSUES.md` (K3, K4, U11, U13);
  - `land/core/cute70/SEAMS.md`, `CENSUS.md`, `probe/p97b_types.py`;
  - `land/CUTE_SPIKE.md`, `land/REVIEW_CUTE61.md`, `land/PLAN.md`;
  - `agent_space/paramgraph/CUTE_TRACING_PROBE.md`;
  - the paramgraph-cute-* memory notes.
