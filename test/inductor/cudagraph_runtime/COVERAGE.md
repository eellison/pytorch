The catalog covers frontend, compiler-contract, import, replay and ownership
regressions. `suite.json` is authoritative for its files and case counts.
Execution results are separate from this source inventory; a passing snapshot
does not qualify a changed branch.

| Contract | Representative test files |
|---|---|
| Input/metadata provenance and output source kinds | `cpu/test_saved_input_metadata.py`, `test_output_offset_sources.py`, `test_storage_offset_sources.py` |
| Native input facts | `cuda/test_native_tensor_facts.py`: grouped/interleaved facts preserve integer, pointer and storage-offset argument order across in-place metadata changes, invalid-rank misses and later reuse |
| Typed early arithmetic and address range/guard semantics | `cpu/test_numeric_program.py`, `test_address_scalars.py`, `test_early_address_ranges.py`, `test_signed_guard_printer.py` |
| Shared expression DAGs and pointer-root uses | `cpu/test_parameter_expression_dag.py`, `test_storage_root_dag.py`: typed instruction identity, prepared output slots, retained sources, root ordering and last-use planning |
| Compiled expression evaluation | `cpu/test_compiled_evaluation.py`, `test_compiled_parameters.py`, `test_host_evaluation.py`, `test_numeric_minmax.py`: integer overflow, ordered float bits, callback order with zero/changing arity and early failures, n-ary min/max and late pointer fields |
| Integer payload simplification | `cpu/test_integer_payload_simplification.py`: retained facts, immutable contracts and partial-operation domains after folding, including cached constants and unselected conditional branches |
| Ordered capture events | `cpu/test_capture_events.py`, `cuda/test_hosttrace_sequence.py`: exact stream/capture frontiers, missing/reordered nodes, kernel/memset/kernel and kernel/H2D/kernel composition with changed shapes and addresses |
| Symbolic owned-address guards | `cpu/test_owned_address_guards.py`: actual traced allocations and offset views, exact alignment proofs without address hints, incompatible branches and division-domain rejection |
| Owned allocation invariant | `test/test_cuda_graph_owned_alignment.py`: real custom-allocator misalignment rejection before submission, cleanup, healthy replay and zero-byte/null outputs (outside the catalog) |
| Non-kernel nodes and late field slots | `cpu/test_replay_value_bindings.py`: memcpy/memset counts introduced after kernel pointer expressions, evaluated with changing inputs by the compiled early and late programs |
| CUDA host reductions on shared replay | `cuda/test_host_trace_reduce.py`: changing addresses, shapes and blocks, fixed intermediates, symbolic scalar fields, mutation, retracing and retained outputs |
| Mixed CUDA/Triton invocation import | `cpu/test_cuda_tape_import.py`, `cuda/test_mixed_cuda_host.py`: distinct symbolic formals, owned allocations and offset views, physical ABI fields, ordered calls, complete local guards including zero-launch entries, output identity and native reuse |
| Guard-specialized output identity | `cpu/test_cuda_output_identity.py`, `cuda/test_mixed_cuda_host.py`: local guarded layout proofs, repeated input/prior-output identity, offset views, retained reuse guards, invalid layout rejection and old-variant native hits |
| Triton/CuTe/CUDA composition | `cuda/test_mixed_three_frontends.py`: one four-node graph, changing shapes and addresses, offset inputs, native hits and outputs surviving close |
| Mixed reduction initialization and computed integers | `cpu/test_computed_integer_guards.py`, `cpu/test_mixed_memset_events.py`, `cuda/test_mixed_cuda_memset.py`: ordered native callbacks, local guard domains, independent invocation symbols, semaphore resets and exact per-shape launch correspondence |
| Host-trace GC lifecycle | `cpu/test_host_trace_gc.py`: no forced full collection, disabled capture window, prior-state restoration, constructor/body/finalizer errors; existing `test/test_cuda_host_trace.py` covers cyclic CUDA graph destruction |
| Programs with no kernel nodes | `cuda/test_empty_program.py`, `cpu/test_capture_events.py`: symbolic allocations and view-only outputs, changed pointers/shapes, branch retracing and old native variants, output identity/lifetimes, and rejection of unrecorded graph nodes |
| Ordinary Triton completion on a tracing decline | `cuda/test_direct_triton_declines.py`: successful launches, warmup and hooks finish the original host exactly once; ordinary errors propagate; later observations can prepare and reuse a native variant |
| CuTe converter constant snapshots | `cpu/cute_host/test_source_dependencies.py`: signed-zero configuration changes selecting different views are detected; unchanged builtin float bits remain stable |
| Mixed-host decline boundaries | `cpu/test_direct_cuda_host_declines.py`: ordinary work finishes exactly once after a recognized recording/lowering decline; no variant is published, ordinary/unclassified failures propagate, and failed observations release transient recorded resources |
| Closed scalar argument guards | `cpu/test_hosttrace_constant_contract.py`, `cuda/test_hosttrace_closed_scalar_bits.py`: signed-zero and NaN bit identity, preserved custom scalar equality, correct ordinary fallback, shape retracing and native hits with fresh addresses |
| Tape effect admission | `cpu/test_hosttrace_effect_admission.py`: unsupported closed regions and copy directions decline before capture; legacy and explicit H2D records remain accepted |
| Thread-local CUDA context before parameter updates | `test/inductor/test_cudagraph_host_trace_threads.py`: fresh-thread numeric, public-API and raw pointer-batch updates, plus a detached context on the same thread (outside the catalog) |
| CUDA host composition | `cuda/test_hosttrace_*.py`: TensorIterator, flash attention, reductions, H2D tables/copies, unary/cast/softmax, embedding/cat and RNG through the shared replay |
| H2D capture correspondence | `cpu/test_h2d_capture_helper.py`: exact pointer/count/direction and frontier checks, invalid dimensionality/arrays/positions, and CUDA query errors; the existing H2D CUDA suite uses the shared helper |
| H2D staging and lifetime | `cuda/test_hosttrace_h2d.py`: changed pointers/counts, retained outputs after close, dropped caller references, no-sync replay, pinned sources and source-rebind counters |
| Mixed host-table transport | `cpu/test_mixed_hosttrace_h2d.py`, `cuda/test_mixed_hosttrace_h2d.py`: invocation-specific table identity, symbolic pointer/scalar rebasing, table/copy order, pinned input metadata, repeated adapters, branch retracing and queued pointer-table target lifetimes |
| Native pinned-input ownership | `cuda/test_pinned_replay.py`: managed allocator reuse, external registered owners, offset views, bound storage retention, original storage and Python-owner retention for unfinished submissions after `set_`, exact wait boundaries and invalid preparation indices |
| Mixed pinned copies and input positions | `cuda/test_mixed_pinned_h2d.py`: Triton/gather/Triton replay with changing pinned views, capture-source retention through cache flush, queued source release, wait/rewrite and standalone boxed indices with closed scalar and unused CPU formals |
| RNG capture ownership | `cuda/test_hosttrace_rng_slots.py`: separate native preparations from one tape, interleaved fresh inputs, exact eager RNG state and outputs, unchanged shared tape/constants |
| Mixed RNG invocation import | `cpu/test_mixed_rng.py`, `cuda/test_mixed_rng.py`: per-invocation prefixes, multiple fields per launch, recorded offset widths, early overflow checks, zero-launch guards, Triton/CuTe and late-parameter composition, exact generator advancement and independent capture lifetimes |
| Attention scale guards | `test/inductor/test_cudagraph_host_trace_guards.py`: typed binary64 conversion, power and division, exact scale comparisons, lazy domains and strict compilation; `cuda/test_hosttrace_dropout.py` exercises ordinary attention dispatch |
| Explicit recorder declines on a miss | `test/inductor/test_cudagraph_host_trace.py`: empty inputs and negative views return the ordinary result once, consume the box, preserve cached variants, and leave unclassified errors visible |
| Triton formal ABI, selected launcher and symbolic grid | `cpu/triton/`, `cuda/generated/test_compiler_user_autotune.py`, `cuda/direct_autotune/test_autotune.py` |
| Triton shape and alignment specialization with independently changing pointers | `cuda/test_alignment_shape_guards.py`: aligned and unaligned first captures, shifted views, guard misses, revisits and native hits |
| Native Triton read-only pointer wrappers | `cuda/test_native_triton_override.py`: unchanged BMM host/JIT through an explicit adapter, fresh addresses, symbolic shape/configuration changes, shifted-view alignment guards, old-variant native hits and retained outputs; the test explicitly bypasses the host helper's Python cache |
| Triton host/device TMA replay | `cuda/triton_tma/test_host_tma_replay.py`, `test_tma_replay.py`, `test_mixed_tma_replay.py`: native tensor-map encoding, exact captured ABI bytes, owned global scratch, changing addresses/shapes/views, native reuse and retained outputs |
| Triton descriptor rank, dtype and independent stride changes | `cuda/triton_tma/test_host_tma_components.py`: rank-one fp32 and rank-three bf16, fixed-address shape/stride changes, exact ordinary ABI bytes, invalid-stride guard rejection and retained aliases |
| Native tensor-map binding domains | `cpu/triton/test_tensor_map_binding.py`: typed binding ownership, index/rank/width/domain validation and address overflow before driver invocation |
| Default compilation and policy identity | `cpu/test_policy_config.py`, `cuda/test_default_compile.py`: config snapshots and default `fullgraph=False`, with and without graph breaks |
| Aligned static-input mutation | `cuda/test_static_input_mutation.py`: registered-buffer mutation, native reuse, aliasing and held outputs |
| Ordinary input mutation and alignment writeback | `cuda/test_input_mutation.py`: aligned native reuse, shifted inputs, full backing storage, aliases and held outputs |
| Generated GEMM, MultiKernelCall and foreach | `cuda/generated/test_generated_selection.py`, `test_generated_axes.py`, `test_foreach.py`, `test_foreach_mutation.py` |
| CuTe host operations, predicates, vectors, integer properties and physical ABI | Compiler suites in `cpu/cute_host/` |
| CuTe observed selection, preprocessing/source ownership and registered cache | `cpu/cute_host/test_preprocessing_ownership.py`, `test_source_dependencies.py`, `cuda/cute/selection/`, `cuda/cute/observed_cache/` |
| CuTe registered Tensor operands and ordinary compilation | `cpu/test_invocation_operands.py`, `cpu/cute_host/test_gemm_registration.py`: two/three-Tensor calls, destination-only mutation, source/output lifetimes, invalid argument checks, and ordinary compilation without replay sidecars |
| CuTe scalar widths, pointer offsets, shared memory and actual GEMM/TMA | Scalar, offset and shared tests in `cuda/cute/`; `tensorop_gemm/` and `tma/` |
| CuTe TMA descriptors from offset views | `cuda/cute/tma_boundaries/test_descriptor_views.py`: fresh pointers, changing shapes/storage offsets, exact descriptor argument correspondence and effective-address alignment guards |
| CuTe TMA descriptor byte-stride alignment | `cpu/cute_host/test_tma_requirements.py`, `test_tma_stride_transport.py`: typed basis/source mapping, grouped/nested modes, derived views, static strides and recasts; `cuda/cute/tma/test_tma_padded_strides.py`: fp16/bf16 native reuse with independent padded strides and invalid-stride guard rejection |
| CuTe TMA encoded stride range | `cpu/cute_host/test_tma_stride_domains.py`, `cpu/test_signed_guard_printer.py`: strict byte-stride bounds, exact unsigned integer gcd for grouped modes, singleton axes with large strides, and compiled predicate arithmetic; the padded-stride CUDA fixture checks the real artifact's cold selection for input/output stride boundaries |
| CuTe TMA encoded dimensions | `cpu/cute_host/test_tma_dimensions.py`, `test_tma_dimension_consumers.py`, `test_tma_requirements.py`, `cpu/test_tma_dimension_printer.py`: exact recast shape sources, all descriptor axes, grouped order/duplicates, zero gcd, unsigned wrap and compiled domain predicates; the padded-stride CUDA fixture checks actual artifact consumer coverage |
| Symbolic rank-three allocation size/stride descriptors | `cuda/allocation_layouts/test_layouts.py` |
| Views, alignment, empty pointers, scalar-only roots and retained aliases | Standalone `cuda/` address/offset/alignment tests and `cuda/host_multi_views/test_multi_views.py` |
| Saved-input release, prepared ownership and native reuse | `cpu/test_release_steps.py`, `test_storage_roots.py`, `test_prepared_owner.py`, `cuda/test_saved_input_release.py` |
| Optional SDK imports and canonical API identity | `cpu/test_api.py`, `cpu/cute_host/test_sdk_import.py` |
| Composition of generated, user Triton and CuTe calls | `cuda/mixed_mlp/`, `cuda/direct_example/`, `cuda/direct_autotune/`, `cuda/repeated_gemm/`: repeated generated reductions/residuals, three-operand upstream CuTe GEMM and user Triton, exact physical pointers, held output views and native reuse |

The native component tests in `test/test_cuda.py` independently cover graph
inspection, parameter/pointer updates, packed arguments, invalid requests, and
graph/update-plan lifetimes.
They are not included in the component catalog counts.
That file also tests the public Python `CUDAGraph.update_kernel_params` method,
including automatic instantiation, repeated/partial updates, hooks, reset and
recapture, `keep_graph=False`, and request validation. Catalog-only function
tracing does not measure this separate public-API suite.
The separate `test/inductor/test_static_triton_launcher.py` suite covers ordinary
launcher ABI, constexprs, argument validation, lifetimes, shared memory, scratch,
fast-launcher behavior and device-TMA fallback. Its cases are also outside the
component catalog counts.

Pinned-source tests require the captured storage to survive until rebinding, and
the most recently bound storage to survive until another submission or close.
The H2D suite checks captured and served sources of older variants after `set_`
and host-cache flushes. Unfinished external copies retain their original storage
and Python owner independently of later bindings. These cases cover `set_`, which
replaces a tensor's storage; they do not cover allocation replacement inside the
same storage by a growing resize, or manual host-memory unregister.

The CuTe TMA tests cover host-built descriptors consumed by real SM10x Blackwell
kernels, using fp16/bf16 inputs, fp32 outputs, a 128x128 tile and a single-CTA cluster.
They do not establish device-side descriptor creation/replacement, multicast,
two-CTA MMA or every dtype/layout. Invalid alignment probes evaluate only the
compiled reuse guard; they never submit an invalid TMA launch.
The padded-stride tests independently vary each operand's stride at a fixed
address, then vary addresses and retain outputs after replay ownership closes.
Sixteen-byte outer descriptor stride alignment is the TMA operation contract.
Its guards use the actual constructor's typed basis and symbolic layout;
pointer alignment does not establish stride alignment. The CPU tests preserve
original-byte alignment across recasts and verify exact derived-view source
correspondence. Encoded outer strides must also be below `2**40` bytes. Grouped
modes retain their exact unsigned integer gcd; valid large source strides are
not rejected merely because a contributor exceeds the descriptor limit.
Encoded dimensions, including axis zero, must be in `[1, 2**32]`. The tests
cover this domain for the supported tiled load/store constructors. Grouped
dimensions follow the ordered unsigned recurrence over exact compiler-projected
shape/stride pairs. They do not flatten arbitrary nested descriptor groups or
extend coverage to other TMA constructor families.

The repeated GEMM fixture covers two generated shape variants sharing one
compiled CuTe owner. Its converter checks a singleton output batch and forms a
same-storage view with batch stride zero; this preserves all elements, the
pointer and storage offset while keeping the exact projected SDK signature
stable. The companion metadata test checks that signature equality. This does
not establish automatic CuTe signature-cache interception.

The Triton TMA cases exercise direct host descriptors and device-created
descriptors, including both in one kernel. Host descriptor bytes come from CUDA's
tiled encoder using the selected Triton compiler metadata and traced pointer,
shape and stride expressions. Device descriptor scratch follows the compiler's
per-CTA size and alignment, with a traced allocation sized by the symbolic grid.
The old unsupported-admission cases have been replaced by these positive cases.
Current limits include packed/FP4 and im2col maps, nonempty descriptor attributes,
multi-CTA programs, profiling scratch, and generated-wrapper host descriptors.
Predicate-only invalid-alignment probes never launch invalid TMA operations.

Historical producer-specific suites are not blindly restored. Formal/constexpr
ABI, selected configuration, symbolic grid and source-identity checks now use the
current contracts above. Obsolete read/write effects analysis and blanket raw
address rejection are not requirements of this runtime.

The historical generated reduction-axis/static GEMM and rank-three allocation
workloads now use the current tracing and replay entrypoints. Their checks include
actual selected kernel identity and native allocation size/stride descriptors.
No retired compiler producer is required to run them.

Normalization provenance remains separate work; this catalog does not qualify
original-versus-copy identity for `copy_if_misaligned`. The runtime also retains
its documented unsupported boundaries; a test count is not a claim of arbitrary
CUDA host behavior.

The explicit mixed CUDA-host adapter currently imports tensor formals, symbolic
allocations/views, kernel launches, recorded byte memsets and canonical native
integer callbacks. Host tables, copies, RNG and tensor-map events still decline
at this import boundary; their standalone
CUDA-host replay coverage remains separate. Local guards are transferred at
invocation completion, including zero-launch entries and allocations after the
last launch. Output object identity is recorded with Python `is` and checked
against the traced storage/metadata; it is never inferred from equal pointers.

Mixed reduction replay currently qualifies fixed-width semaphore initialization.
Optional saved-input reclamation keeps inputs through submission when a mixed
program contains a memset, because the release plan currently names kernel uses.
Native callback guards retain the original arithmetic domains and operation
order; their recorded hints never replace runtime computations.

Standalone CUDA host replay also returns its completed ordinary result when
recording, lowering or preparation declines. `cpu/test_standalone_host_trace_declines.py`
checks mutation exactly once, alias outputs, input-box consumption, preservation
of existing variants, subsequent successful preparation and publication cleanup.
Ordinary exceptions and unexpected preparation/publication errors propagate.
