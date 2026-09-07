# Padded quantization review and benchmarks

Date: 2026-09-02

This is an AI-assisted local engineering report. It is not intended to be
pasted into GitHub without human review and the disclosure required by
`AI_POLICY.md`.

## Status

The correctness fixes are ready for human review, but the upper
predicated-store commit should be re-scoped before submission because isolated
benchmarks removed its workload-level performance justification. No commit was
amended and nothing was submitted.

- Original two-commit worktree:
  `/data/users/eellison/pytorch/agent_space/index_put_stack`
- Integration worktree on the final PR #191974 head (`41a8d6bee91`):
  `/data/users/eellison/pytorch/agent_space/index_put_nested_integration`
- Integration commits before the uncommitted review fixes:
  - `a79f03b9bfc` - pointer-swing correctness fix
  - `a3bd451efe0` - predicated masked-fill support

The review fixes are deliberately left uncommitted in both worktrees. They
belong in the upper predicated-masked-fill commit.

## Design

The general solution remains two kernels for padded blocked scales:

1. The nested reduction computes and scatters valid scale values into the
   swizzled destination.
2. A predicated store writes zeros only to padding lanes and never reads the
   uninitialized destination.

The new internal `inductor_prims.predicated_masked_fill` makes this intent
explicit. Generic `index_put` keeps its existing read-modify-write lowering,
because that form can still fuse with consumers. A scheduler guard prevents a
consumer of the predicated-store output from being fused into the same kernel,
where store-to-load forwarding would be invalid on masked-off lanes.

The guard is dependency-precise: it rejects only when the particular masked
Scatter operation is an ancestor of the proposed consumer. An unrelated
producer that happened to fuse horizontally with the Scatter is not penalized.

The lower commit fixes an independent pre-existing mutation bug: after a
buffer has already acquired a `MutationLayoutSHOULDREMOVE`, a later
`mutate_to` may not swing the underlying data pointer and orphan the earlier
mutation.

## Adversarial review fixes

The final local diff adds the following hardening:

- Reinplace only when input and output tensor metadata match. This prevents a
  noncontiguous view from incorrectly inheriting the output's stride or
  storage offset.
- Keep complex tensors on the functional fallback path, because mutable custom
  fallback kernels are unsupported.
- Implement eager float8 behavior for E4M3/E8M0 through `where`, while retaining
  ATen `masked_fill` behavior for ordinary dtypes.
- Give both functional and in-place forms one exact contract: a zero-dimensional
  value tensor with the same dtype as the destination.
- Register the functional autograd formula and validate it through opcheck.
- Preserve exact `_unsafe_index_put` dtype semantics after excluding its
  decomposition; mismatched source and destination dtypes now fail in both
  direct and reinplaced lowerings.
- Make dependency tests prove that the conditional Triton store was actually
  emitted. The destination-read fallback uses a cross-indexed mask so an
  accidentally predicated implementation cannot pass by performing the same
  load it needs for the predicate.
- Make the template test prove epilogue fusion rather than merely finding both
  kernels in one wrapper module.

Two independent adversarial passes report no remaining blockers.

## Correctness and regression testing

All commands used the integration worktree through
`agent_space/run_current_worktree.py`.

- `test/inductor/test_padded_scatter.py`: 25/25 passed.
- `test/inductor/test_inductor_scheduler.py`: 150 run, 144 passed, 6 skipped.
- `test/inductor/test_inplacing_pass.py`: 28/28 passed.
- `test/inductor/test_torchinductor.py -k index_put`: 37 run, 36 passed,
  1 skipped. The local torchvision NMS registration shim was required by this
  machine's installed torchvision package.
- `torch.library.opcheck`: schema, autograd registration, FakeTensor, and AOT
  dynamic checks all passed.
- `spin lint` on all five changed files: clean.
- `git diff --check`: clean.

The padded scale output from every Inductor variant is byte-identical to the
existing `F.pad` formulation, including padding bytes. FlashInfer comparisons
retain the previously documented small reduction-order differences for FP4;
MXFP8 is exact at `1x4096` and within the saved logical-scale/quant tolerances
elsewhere.

## Benchmark methodology correction (updated 2026-09-03)

The original performance section below this heading was removed because its
multi-variant process produced order-dependent results. Compiling and timing
all padding variants in one process inflated variants compiled before the last
entry by 20-70%; timing them again in reverse order could not repair the
already-selected kernel configurations. The old raw matrices remain useful for
correctness and kernel counts, but not for cross-implementation timing.

The corrected method keeps the same external CUDA-graph protocol (100 calls
per capture/replay, 20 warmups, 50 samples, PDL disabled for FlashInfer) but
runs exactly one Inductor variant plus FlashInfer in each process. Full results
and the supporting order experiments are in
`agent_space/nested_quant_comparison/PADDED_SWIZZLE_FINDINGS_20260902.md`.

### Corrected small-shape FP4 result

The isolated matrix covers NVFP4 and MXFP4 at `19/99/129 x 4096`:

| tuning | predicated / FlashInfer | pad_scatter / FlashInfer |
|---|---:|---:|
| coordinate descent | 0.98-1.11x | 0.97-1.14x |
| default | 1.27-1.32x | 1.26-1.34x |

With coordinate descent, the generic predicated-store implementation is at
parity with FlashInfer. It is also 26-63% faster than the isolated `F.pad`
formulation at these shapes. `predicated` and `pad_scatter` differ by at most
3%, with the sign varying by shape and format, so the rectangular caller-side
specialization has no demonstrated benefit and should be dropped.

An independent rerun on the final integration tree at `99x4096` reproduced the
isolated result:

| format | predicated | pad_scatter | FlashInfer |
|---|---:|---:|---:|
| NVFP4 | 2.961 us | 2.892 us | 2.938/2.943 us |
| MXFP4 | 2.775 us | 2.890 us | 2.876/2.870 us |

### Large shapes

*(Corrected 2026-09-03. The numbers below supersede the earlier "NVFP4 is
1.50-1.53x FlashInfer" claim, which was measured on a stale tree and with a
benchmark that timed Inductor first and FlashInfer second. Ordering alone was
worth ~12%: adding a reversed sweep moved Inductor from 118.23 to 104.39 us on
an unchanged binary.)*

At `16000/16384 x 8192` with coordinate descent, Inductor is at parity with or
faster than FlashInfer; measured ratios span 0.92-1.06 depending on shape and
run. Representative run at `16000x8192`, NVFP4:

| variant | median | vs FlashInfer |
|---|---:|---:|
| Inductor | 105.3 us (3.19 TB/s) | 0.94x |
| FlashInfer | 112.3 us (2.99 TB/s) | 1.00x |

The duplicated-scale-computation finding described in the previous revision is
already fixed on the current stack: `_SubParentValueResolver.resolve_load`
returns the un-broadcast group-width source, so the FP8 conversion and the
reciprocal happen once per 16-element block, not once per pair. The generated
kernel reciprocates at reduced width (`tmp47`) and broadcasts into the pair
domain. The 154.68 -> 129.02 us experiment quoted previously was run against a
kernel dumped before that change landed.

Two smaller items were measured on the current tree, both with a 3 s cooldown
between samples (without it, thermal throttling makes medians swing 104-123 us
and swamps effects of this size):

- **Zero-init of the blocked scale destination is dead in the unpadded case.**
  When `rows % 128 == 0 and cols % 4 == 0` the swizzle index map is a bijection
  onto the destination, verified by coverage check at `(16000,512)`,
  `(16384,512)`, `(16000,256)`, `(128,512)`, `(256,4)`. Swapping `torch.zeros`
  for `torch.empty` under that static condition drops the fill kernel
  (kernel_count 2 -> 1), is byte-identical, and saves 1.19 us (1.13%). Since the
  condition is on static shapes, no compiler-side coverage analysis is needed.

- **Redundant `emulate_precision_casts` round-trips are not the cost.** Under
  emulation the generated kernel round-trips the same value through bf16 up to
  three times in a row, plus once more after `abs` and after the block `max`,
  where the value is provably bf16-exact. A LoopBody FX pass that drops every
  round-trip whose input is already exact in the target dtype cuts the kernel
  from 30 to 10 bf16 casts, byte-identically -- and buys only 0.6%
  (104.11 -> 103.49 us). Disabling emulation entirely is worth 14%
  (89.77 us), and the register/shared-memory footprint shows why: 48 regs /
  4096 B shared with emulation versus 31 regs / 64 B without. The cost is the
  *required* bf16 materialization forcing layout conversions through shared
  memory, not the redundant casts. The pass is saved at
  `agent_space/loopbody_cast_elision.patch` but is not worth landing on this
  evidence.

### Simpler implementation option

The isolated results also weaken the performance justification for the new
predicated masked-fill primitive itself. On current main, initializing the
blocked destination with `torch.zeros` and scattering only the valid values
already produces the same two-kernel topology. At `99x4096` with coordinate
descent it measured 2.747 us for NVFP4 and 2.952 us for MXFP4, versus
FlashInfer at 2.942/2.871 us.

That existing spelling is correct for dynamic shapes too. Current main emits a
`tl.device_assert` because `_unsafe_index_put` is decomposed to the checked
form; the integration diff's one-line decomposition exclusion removes the
assert while retaining two kernels. Therefore the minimal production path may
be:

1. zero-initialize the blocked output in the quantization caller;
2. scatter the valid scales;
3. retain only the narrow `_unsafe_index_put` lowering/decomposition fix if
   dynamic-shape code must avoid the device assert.

The independent `mutate_to` pointer-swing bug remains a valid standalone
correctness fix, but this padding formulation does not require it. The larger
predicated-store IR, scheduler, reinplacing, eager, and autograd machinery
should not land solely on the basis of this padded-quant workload unless a
separate use case demonstrates an advantage over zero-initialize-plus-scatter.

### Nested-reduction assertion from the one-kernel probe

The assertion

`expected str(d1) == str(d2), got R0_BLOCK != nested_R0_REDUCED_BLOCK`

reproduces on the stale pre-#191974 nested feature stack, including its first
commit. It occurs while replaying a `where` pointwise node in the grouped stage:
one operand is evaluated in the parent full-resolution tile and the other in
the grouped reduced tile.

It is not a current-main crash. `origin/main` at `fd23f0c97b4`, which includes
the landed #191974 correctness guard (`abaacefd143`), compiles the same probe
correctly with `nested_reduction_count=1` and `kernel_count=2`. The unsafe
fusion is declined before codegen. Achieving one kernel would require a new,
index-proven mixed-domain replay design; it is not needed for the present
small-shape parity result.

### Other open issue

A masked fill of an FP8 tensor can still emit `tl.full(..., tl.float8e4nv)` and
fail Triton LLVM verification because the backend represents FP8 as `i8`.
That is a separate standalone Inductor bug; the predicated-store path avoids it
structurally but does not fix it.

## Raw results

- `agent_space/nested_quant_comparison/results/cdiso_*.json`
- `agent_space/nested_quant_comparison/results/nocd_*.json`
- `agent_space/nested_quant_comparison/results/iso_*.json`
- `agent_space/nested_quant_comparison/results/big_*.json`
- `agent_space/nested_quant_comparison/results/recheck_iso_*_99.json`

## Recommendation

Do not amend or submit the upper predicated-store commit yet. First reduce the
target workload to zero-initialize-plus-scatter and determine whether the only
remaining core change is the narrow `_unsafe_index_put` decomposition fix.
Keep the pointer-swing correctness fix separate. Any retained performance claim
must use one variant per process and state explicitly that small-shape FP4
parity requires coordinate-descent tuning. Do not add the rectangular
`pad_scatter` special case.
