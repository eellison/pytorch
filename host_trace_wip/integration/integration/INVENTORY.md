# Torch-side change inventory vs int6 step177 (odd, Python only) / step178 (even, C++ build_cpp6)

Written 2026-10-05, CPU only, no GPU runs. Evidence comes from the snapshot sources (int6/snap/step177, step178,
build_cpp6/src), int6/MERGE_LOG.md, and import probes. The probes are inv/check_api.py (output api_step177.txt and
api_step178.txt) and inv/check_c.py. The adapter scan is inv/adapter_uses.py.

Classes:
- P = PRESENT.
- MD = MERGED-DIFFERENTLY.
- MISS = MISSING.
- N/A = adapter-only, never applied, or a C++-line item checked on the Python line.

## 1. Adapter would break (read this first)

Hard breaks (exception at import or at the first call):

| # | Adapter:line | Attribute / kwarg | int6 step that removed it | Line(s) | Effect |
|---|---|---|---|---|---|
| H1 | sglang/armF/launchstamps_cpp.py:12 | `torch._C._host_trace_launch_stamps` | S1_dead_code_csrc, step 128 (C++ side of S1 127/128) | 178 | AttributeError at import. This is a diagnostic script only. hoststamps_cpp.py:56 uses getattr and is safe. |
| H2 | sglang/armF/adapter_cpp.py:451, prep/adapter_prep_cpp.py:920 | `HostTraceReplay(static_prefix=...)` | never on the odd line | 177 | TypeError. Expected: these are C++-line adapters. vllm armV:47 probes co_varnames instead and drops it silently. |
| H3 | adapter_cpp.py:77/81, adapter_prep_cpp.py:929 | `torch._C._HostTraceBound` (+ `statics_changed`, `trust_statics=`) | never on the odd line | 177 | adapter_cpp falls back via hasattr. adapter_prep_cpp:929 does not, so PREP_BOUND on 177 raises AttributeError. Expected (C++ only). |
| H4 | adapter_cpp.py:47-49 | `memory="held"` (because `_MERGED = hasattr(R, "_PLANNED")` is True on both lines) | never on the odd line (MERGE_LOG L181: "held" and "planned_scratch" are C++-line only) | 177 | ValueError at HostTraceReplay for ARMF_MEMORY=held / arena_scratch. Default ARMF_MEMORY=run_buffer is fine. |

Silent no-ops / behaviour changes (no exception; the adapter believes a switch is on):

| # | Adapter:line | Attribute / kwarg | int6 step | Lines | What int6 does instead |
|---|---|---|---|---|---|
| S1 | sglang/armF/prep/adapter_prep_cpp.py:72 (PREP_ABC, default on under PREP_M=1) | `R._LEARN_WIDE` | never merged under that name | both | Behaviour is unconditional: int6 `_learn` is always wide (zeros for non-floating inputs, alias views, provider.operands lending). The assignment creates a dead attribute. |
| S2 | same line | `_harvest._BUDGET_GROUPS` | never merged under that name (DEFAULT_OFF.md) | both | MD as `HarvestProvider(budget_unit="group")`, default "key". The adapter's HarvestProvider at :255 does not pass it, so it gets per-key budgets. Since S5 (125/126) the default budget is None, i.e. unlimited, so this only matters if a budget is set. |
| S3 | same line | `_harvest._EXTERN_SIBLINGS` | S1 (127/128) retired the switch | both | Always on. Dead attribute. |
| S4 | sglang/armF/tgv/adapter.py:61 (ARMF_TGV, default 0) | `ht.cute_without_tvm_ffi = True` | 70 (123/124) retired the switch | both | Always on. Dead attribute. |
| S5 | sglang/armF/prep/adapter_prep_cpp.py:291 (PREP_TGV) | `tgv.from_dlpack = cute_runtime.from_dlpack` | 70 (123/124): `_rebind_from_dlpack` removed | both | Still works but is redundant: from_dlpack is intercepted through a `runtime._Tensor.__new__` hook, so a name bound early is covered. |
| S6 | vllm/armV/adapter.py:205-206 (ARMV_LEARNER_LEARNS, default 1) | `HostTraceReplay(learn_in_learning_variants=True)` | S6 (135/136) retired it | both | The co_varnames probe drops the kwarg silently. int6 keeps the old False behaviour: a learning variant whose keyed site misses retraces. Moot while oob_learn is ON (0 learning variants in the suites and proxies), but arm V results under LEARNER_LEARNS=1 no longer match. |
| S7 | vllm/armV/adapter.py:207-208 (ARMV_REDISPATCH_LEARNERS, default 0) | `HostTraceReplay(redispatch_learners=True)` | S6 (135/136) retired it | both | Dropped silently. The default path is unchanged; ARMV_REDISPATCH_LEARNERS=1 has no effect. |
| S8 | vllm/armV/adapter.py:47/200 | `static_prefix=` | not on the odd line | 177 | Dropped silently on 177 (the probe); present on 178. |
| S9 | adapter_cpp.py:254, adapter_prep_cpp.py:929 | `_HostTraceBound(trust_statics=True)` | core 61 (trusted_statics_check OFF; build_cpp6 Entry.cpp:533) | 178 | sglcpp checked a trusted entry's static layout on every call. int6 skips the check by default: re-pointing or relayouting a static without `statics_changed()` replays the old address. This is the documented contract (user's 10-04 direction). Re-enable with `torch._C._host_trace_trusted_statics_check(True)`. |

Everything else the adapters read exists on both lines. That covers:
- the stats internals: `traces`, `replays`, `eager`, `triton_fallbacks`, `_reasons`, `declines`, `_call_slow`, `variants`, `variant.learns`, `captured.lowered.steps[*].call.host`, `captured.segments`, `EagerCall.host` / `.name`, `HarvestProvider.refused`;
- `_host_trace_cute.install`, `_stand_in` and `_DLPackArg`;
- the tape, launch and capture symbols that vllm_ops_trace imports;
- `HarvestProvider(lendable=, lend_alias_groups=, budget_unit=)` and `ht.raise_unexpected`.

## 2. Summary counts

There are 88 rows in the change table below. Some rows are grouped (sgl1 7-13, sglplan 1-8, N1-N5).

| | step177 | step178 |
|---|---|---|
| PRESENT | 39 | 53 |
| MERGED-DIFFERENTLY | 18 | 20 |
| MISSING | 7 | 6 |
| N/A | 24 | 9 |

The important MISSING items (no torch fix in int6):
- CORE_ISSUES C1/T2: a hashed SymInt or SymBool pins its value. `(m, n, k) in SET` records Eq(s, hint). torch/__init__.py:936-937 still returns `hash(int(self))`, and SymBool's `__hash__` (L1139) still forces bool(). SGLang's use_bf16_splitk_gemm and the TMA test hit it. The workaround is adapter `_by_runs` (PREP_TGV).
- CORE_ISSUES C4: Triton pointer alignment on a symbolic address becomes a trace-wide guard. step178 _host_trace_triton_launch.py:151-153 still sends a SymInt `_address` through `native_specialize_impl`. The workaround is the adapter's `do_not_specialize` copy (PREP_BS_META).
- CORE_ISSUES C3: compound And/Or guards collapse to the hint. Not isolated; no torch change.
- vllmcpp 17 / 20 (learn_in_learning_variants, redispatch_learners): retired by S6, which kept the old behaviour. Moot under oob_learn=True.
- sglcpp C15 `planned_scratch` on step177 only: it is a Python-only change, but the odd line does not carry it (by design, MERGE_LOG L181).

Partially covered:
- C2 is covered by `dispatch_unit` (core 35, `dispatch_units=True`), but no adapter wraps SGLang's choice code with it.
- C5 is covered by core 28 (`symint_bit_length=True`) for bit_length-based next-power-of-2. Constexpr and launch-option pins stay by design.

## 3. int6 defaults of every switch involved

Python (`torch.cuda._host_trace`):
- ON: raise_unexpected=False (adapters set it from the env), trace_pdl=True, traced_impls=True, symint_bit_length=True, dispatch_units=True, library_impls=True, fresh_stack_chunk=True, shared_op_guards=True.
- cpp_entry=True on 178 only (step 66).
- cute_without_tvm_ffi: RETIRED at 70.

`_host_trace_capture`: keep_programmatic_edges=True (flipped at 57/58).

`_host_trace_replay`:
- oob_learn=True (75/76) and refused_eager_steps=True.
- `_PLANNED` = (planned, planned_no_reuse, planned_no_split, packed, planned_scratch) on 178. On 177 it lacks planned_scratch.
- memory: auto / eager / run_buffer / _PLANNED, plus "held" on 178.
- RETIRED: `_LEARN_WIDE` (never merged), learn_in_learning_variants (S6), redispatch_learners (S6) and the owner_sets switch (S4).

`HostTraceReplay` kwargs:
- memory='auto', handback=False (inductor turns it on at 93/94), splits='peak', trace_builtin_minmax=True, learn_pool=False, background=False, fullgraph=False, forbid_learners=False, static_shapes=().
- static_prefix=0 on 178 only.

`_host_trace_harvest`:
- `_EXTERN_CAP`=1 GiB, `_LENT`=256 MiB, `_SIBLINGS`=True, `_EXTERN_FAST`=True.
- RETIRED: `_EXTERN_SIBLINGS` (S1) and `_BUDGET_GROUPS` (never merged).

`HarvestProvider` kwargs: families=('blas',), budget=None (S5), budget_unit='key', extern_cap=None, extern_digest='full', indexed=None, lendable=(), lend_alias_groups=True (91/92).

C++ (178, build_cpp6 Sites.cpp:17-21, Entry.cpp:533, Patch.cpp:20):
- delta=True, variant_order=True, placement_reuse=True, entry_descriptors=True.
- form_held=False, trusted_statics_check=False (core 61), phase_timing=False.
- REMOVED (S1, 128): `_host_trace_launch_stamps`, `_cuda_hostTraceSegments`.

## 4. Change table

The columns are: item, source, step177, step178, int6 step / switch (int6 default), evidence.

### src_sgl1 (sglcpp 1-12 inherit these)
| item | 177 | 178 | int6 step / switch | evidence |
|---|---|---|---|---|
| sgl1 1 trunci | P | P | step 1 | step1_sgl1.diff, MERGE_LOG L36 |
| sgl1 2 PDL in cute/triton launches | P | P | step 1; trace_pdl=True | _host_trace.py |
| sgl1 3 cooperative-kernel decline | P | P | step 1 | grep in launch |
| sgl1 4 numel==0 pointer | P | P | step 1 | |
| sgl1 5 OpaqueKey.alias | P | P | step 1 | _host_trace_opaque |
| sgl1 6 extern cap / `_equal` rows of 4096 / `_EXTERN_FAST` | P | P | step 1; `_EXTERN_CAP`=1 GiB, `_EXTERN_FAST`=True | check_api |
| sgl1 7-13 | N/A | N/A | adapter-only | CHANGES "(armF/adapter.py, not in the diff)" |

### sgltgv
| item | 177 | 178 | int6 step / switch | evidence |
|---|---|---|---|---|
| tgv 14 from_dlpack bound before install | MD | MD | 70 (123/124): switch + `_rebind_from_dlpack` retired, `_Tensor.__new__` hook, always on | test_a_from_dlpack_bound_before_the_hook_is_a_launch, test_sglang_tgv_gemm present |
| tgv 15 static_shapes | N/A | N/A | adapter-only; kwarg exists | |
| tgv 16 `_concrete` NamedTuple | P | P | core 22 | |

### src_sglcpp C items (C++ line; on 177 these are N/A unless Python-only)
| item | 177 | 178 | int6 step / switch | evidence |
|---|---|---|---|---|
| C1 static_prefix fast path | N/A | P | static_prefix=0 | HostTraceReplay signature |
| C2 O(1) variant selection | N/A | P | variant_order=True | Sites.cpp:18 |
| C3 _HostTraceBound | N/A | P | | check_c: derived/none_kwargs/resets/fallback/trust_statics |
| C4 planned/held memory, dirty-record skip | N/A | P | memory "held" | memory validation |
| C5 phase_times + launch_stamps | N/A | MD | launch_stamps removed S1 (128); phase_timing=False | check_c |
| C6 one-call entry (statics_changed) | N/A | P | placement_reuse=True | `_HostTraceBound.statics_changed` |
| C7 trusted statics | N/A | MD | core 61: trusted_statics_check=False (check skipped) | Entry.cpp:533,787 |
| C8 one exec per launch config | N/A | N/A | never applied (draft) | CHANGES C8 |
| C9 delta evaluate | N/A | P | delta=True | Sites.cpp:17 |
| C10 capture check once per call | N/A | P | | EagerStep.cpp:368 frame.uncaptured |
| C11 bound fast check | N/A | P | | Entry.cpp PyType_IsSubtype L982/L1172 |
| C12 switches | N/A | P | variant_order/placement_reuse/delta=True | check_c setters |
| C13 TMA descriptors (= CORE_ISSUES T1) | N/A | P | entry_descriptors=True | Sites.cpp:21 |
| C14 held params per form exec | N/A | P | form_held=False (as built) | Sites.cpp:20 |
| C15 planned_scratch (Python only) | MISS | P | `_PLANNED` | MERGE_LOG L181/L213; odd line excludes it |

### src_sglprep, src_sglm, src_sglm_abc
| item | 177 | 178 | int6 step / switch | evidence |
|---|---|---|---|---|
| sglprep (no torch changes) | N/A | N/A | | |
| sglm install() from_dlpack rebind | MD | MD | 70 | as tgv 14 |
| sglm_abc A `_LEARN_WIDE` | MD | MD | unconditional | adapter break S1 |
| sglm_abc B `_BUDGET_GROUPS` | MD | MD | budget_unit="group" (default "key") | DEFAULT_OFF.md |
| sglm_abc C `_EXTERN_SIBLINGS` | MD | MD | always on (S1) | |
| sglm_abc operands() protocol | P | P | | TableProvider.operands |
| sglm_abc form_fix (copy of sglbs) | P | P | | capture L852 lcm |

### src_sglplan, src_sglpre, src_sglbs, src_sglfork, src_sglforkbs
| item | 177 | 178 | int6 step / switch | evidence |
|---|---|---|---|---|
| sglplan 1-8 memory planner | P | P | memory modes on both lines | _host_trace_memory.py functions present |
| sglpre P1 prepare / lendable | P | P | lendable=(), `_LENT`=256 MiB | |
| sglpre P2 extern siblings | MD | MD | always on (S1) | |
| sglpre P3 budget groups | MD | MD | budget_unit | |
| sglpre P4 `_Background` / `_Deferred` | P | P | background=False | |
| sglpre P5 save/load/_restore/_resolved/untouched | P | P | | |
| sglpre P6 form cluster lcm fix | P | P | | capture L852 |
| sglpre P7 learn-pool fix | P | P | learn_pool=False | replay L727/1093/1232 |
| sglpre P8 entry_guards | P | P | | |
| sglpre P9 extern_digest | P | P | 'full' | |
| sglpre P10 extern_cap | P | P | None | |
| sglpre P11 budget_unit | P | P | 'key' | |
| sglpre P12 remaining tests | P | P | | |
| sglbs PDL edges | P | P | step 7; keep_programmatic_edges=True | |
| sglbs TableProvider.operands test fix | P | P | | |
| sglbs form fix | P | P | | |
| sglfork F1 packed launches | P | P | step 2a; KernelLaunch.packed=False | test_cuda_host_trace_packed.py |
| sglfork SELECT | N/A | N/A | fork-side | |
| sglforkbs F2 out-of-band learn | P | P | oob_learn=True (75/76) | |

### vLLM overlays
| item | 177 | 178 | int6 step / switch | evidence |
|---|---|---|---|---|
| vllm1 14 indexed writers on one storage | P | P | step 5a (+13b asserts) | test_declared_interleaved_kv_writer, ..._dropped_v_write_is_refused |
| vllmcpp 15 sglpre P1-P5 + _form fix | P | P | | as sglpre |
| vllmcpp 16 lend alias groups | P | P | lend_alias_groups=True (91/92) | test_learned_interleaved_kv_writer |
| vllmcpp 17 learn_in_learning_variants | MISS | MISS | retired S6 (135/136), old False kept | test_learned_in_learning_variant now asserts (learned, traces) = (0, 2) |
| vllmcpp 18 item-14 tests | P | P | | |
| vllmcpp 19 relower learn pool | P | P | learn_pool kwarg | |
| vllmcpp 20 redispatch_learners | MISS | MISS | retired S6, old False kept | test_learner_redispatches_selector asserts (2, 0, 1) |
| vllmcpp 21 traced_impls | P | P | step 2c; traced_impls=True | `_host_trace.py:82`, register_traced_impl L90 |
| vllmcpp 22 F1 packed | P | P | step 2a | |

### armF/prep CORE_ISSUES.md
| item | 177 | 178 | int6 step / switch | evidence |
|---|---|---|---|---|
| C1 hashed SymInt pins bs | MISS | MISS | | torch/__init__.py:936-937 |
| C2 host choices split bs into classes | MD | MD | core 35 dispatch_unit (dispatch_units=True); not used by any adapter | _host_trace.py:104-121 |
| C3 compound And/Or guard pins m | MISS | MISS | not isolated | |
| C4 Triton alignment guard on moving addresses | MISS | MISS | | triton_launch.py:151-153 |
| C5 size-derived constexprs pinned | MD | MD | core 28 symint_bit_length=True (partial; constexpr pin by design) | torch/__init__.py:952 |
| C6 from_dlpack bound before install | MD | MD | 70 | |
| C7 form cluster change fails to instantiate | P | P | | capture L852; no unit test (gap) |
| C8 first-call cost of a new bs | N/A | N/A | perf, no fix | |
| T1 TMA descriptors | N/A | P | entry_descriptors=True | = sglcpp C13 |
| T2 SymBool hash | MISS | MISS | | torch/__init__.py:1139 |
| T3 TGV TMA bitwise check | N/A | N/A | verification only | |
| T4 TGV TMA-encode host cut | N/A | N/A | superseded by C4 dirty + C9 delta | |
| planned_scratch arena overrun fix | N/A | P | | test_planned_scratch_is_an_arena_member (178) |
| N1-N5 | N/A | N/A | not torch-side | |

### Core items
| item | 177 | 178 | int6 step / switch | evidence |
|---|---|---|---|---|
| 02 op-owned guards | P | P | shared_op_guards=True; owner_sets switch retired S4, always on | |
| 20 | MD | MD | switch retired S2, always on | |
| 21 | MD | MD | S2 | |
| 22 | MD | MD | S2 | |
| 23 | MD | MD | S2 | |
| 24 | MD | MD | S2 | |
| 25 | MD | MD | S2 | test_fastdivmod_host_setup_is_symbolic |
| 26 PDL in trace | P | P | trace_pdl=True | |
| 27 | MD | MD | S2 | test_a_host_branch... |
| 28 symint_bit_length | P | P | symint_bit_length=True | torch/__init__.py:952 |
| 29 test | P | P | | test_two_launches_share_a_moving_view_and_a_size_block |
| 35 dispatch_unit | P | P | dispatch_units=True | |
| 49 extern host choice | P | P | step 71/72 | TestExternHostChoice |
| change 15 / `_rebind_from_dlpack` | MD | MD | 70 (123/124) | `_rebind_from_dlpack` absent; `_Tensor.__new__` hook |
