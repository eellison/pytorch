# symti: READY for candidate 4, even line (2026-10-12)

The pointwise witness and the hand-written hosts are retired: an op's own CUDA host code runs over symbolic
TensorIterators (SymHost / EagerHost templates), and its launches are recorded. This drop also adds the Flash-Next
PLE batch: clamp with a symbolic Scalar, gather, index_put_, index_copy_, searchsorted, cummax/cummin, and the
depthwise convolution (ATen's CudaDepthwise2d).

## Patches (land/core/symti/patches, md5 in MD5SUMS)

| file | md5 | applies to |
|---|---|---|
| symti_cpp.patch | 6203e0b112ac | int6/build_cpp10/src as of 10-12 07:42 (E1 + RD2 csrc). `patch -p1 -E` (deletes host_trace/Pointwise.cu and UfuncCUDA_add.cuh). Also dry-runs clean on build_cpp9/src. |
| jiterator_null_kwargs.patch | c63fde1a207a | torch/csrc/cuda/Module.cpp, kept as a one-line upstream candidate; apply with the C++. |
| symti_py.step314.patch (= symti_py.step316.patch, identical) | 961eef9cfcc7 | int6 snap step314 or step316 (even). Covers torch/cuda/_host_trace*.py, torch/_meta_registrations.py and test/test_cuda_host_trace_*.py. |

The odd line (frozen binary, Python only) cannot take this: the change is mostly C++. The Python was produced by
patches/merge_top.py <top> (a 3-way merge of core/symti/py onto the top). Rerun it for a later top; it stops on any
conflict it has no rule for.

## Gates

GB300, GPU 2/3 (idle check) or GPU 1, install15 (build_cpp10/src + symti_cpp.patch) with wt314
(step314 + symti_py.step314.patch). Every file is OK:
replay 294, python_entry 569, aten 1731, harvest 229, opaque 45, tape 46, pinned 38, fullgraph 8, hint_audit 16,
guards 12, lower_tape 6, oob 8, memory 21, capture 12, lower 24, packed 5, program 114, triton 66, triton_abi 7,
cover 20, integration 1, cute 73, float 12. Logs: land/core/symti/logs/g15a/.

The PLE repro (scratch/integration/serve/repro/aten_ple_repro.py, check fixed to reuse each call's arguments)
gives ngram_ids 1 trace, 3 replays, bitwise, no eager step, and short_conv 1 trace, 3 replays, no eager step.
Nothing was recorded as unconverted.

GH200 (gh200-b, sm_90), the same pair (build_cpp10/src + symti, step314 + the patch):
- PLE repro: the same result.
- OK: aten 1731, replay 294, opaque 45, tape 46, pinned 38, fullgraph, hint_audit, guards, lower_tape, oob,
  memory, capture, lower, packed, program 114, triton 66, triton_abi, cover.
- harvest and python_entry fail only on GH200's own set, the one candidate 4's GH200 base runs show:
  test_workspace_cache_on, test_sibling_one_capture / _unverified, and the test_persist* family, which in the
  python_entry run also includes test_persist_stale.
- Logs: gh200-b:ht/symti/logs/g17.

Trace time on GH200 (traces of the step242-based tree vs the witness base, no profiler):

| workload | witness | new |
|---|---|---|
| decode36 | 2.08 / 2.10 s | 2.00 s |
| prefill36 | 1.81 / 1.82 s | 1.74 s |
| minops decode | 13.34 / 13.38 s | 12.58 s |
| minops prefill | 10.49 s | 9.17 s |

## Changes of behavior worth a look

- **Real device allocations during a trace now go through the trace.** Under a trace's recorder
  (`at::detail::symbolic_iterators()`), a CUDA structured kernel's int-sized output (`create_out`, torchgen) is
  allocated through the dispatcher, so the trace allocates it.
  - Before this, an op whose meta read sizes as ints allocated real memory in the trace's capture pool.
  - Each such trace then left a 2 MB segment cached in a dead private pool. This is what failed the harvest
    `_private_bytes <= _POOL_KEEP` asserts.
  - Those ops now trace, with their int reads loud as decided. cat's and sort's metas are converted to SymInt
    outputs, so neither has an int read.
- **Harvest padding.** The harvest takes the bytes that an ATen pointwise launch reports as `'.'`
  (`LaunchLayout`: padding or an empty functor member) as loose, by type. It no longer depends on stack garbage
  that happens to match the smear. The new test is `test_pointwise_padding_is_loose`, with a padded-functor
  extension op.
- **Symbolic Scalar arguments** (clamp's bounds) reach functor members as fields of their expression.
  - Integer members are exact.
  - Float members use the f32 row, with the sign handled because f32div takes n >= 0.
  - Double members decline.
  - clamp's NaN and range checks read no ints.
- **Ops with no returns** (`_cummax_helper`) that trace no longer also run eagerly. The traced result was None and
  was taken as "untraced"; an `_UNTRACED` sentinel fixes it.
- **New meta registrations:** `_cummax_helper` / `_cummin_helper` (no-op) and `_conv_depthwise2d`.
- **`host_trace.select`** takes a constant condition.
- **_convolution:**
  - A symbolic input takes the SymInt shape checks and backend choice.
  - The kernel sizes the backends take as ints are guards of the weight's sizes.
  - Its depthwise kernel's launch is recorded.
  - cuDNN convolution stays a harvest matter. The "conv" harvest family checks no backend; see
    QUESTIONS_FOR_USER.md.

## Test expectations changed (all tied to ops that now trace)

- harvest:
  - an_op_outside_the_harvested_set uses cumsum through `trace()`, not a replay.
  - The interleaved KV writer tests pass again.
- opaque: the mm/mv reasons gain their CUDA host's decline. A plain trace's eager calls carry no provider reason.
- replay:
  - Several reason strings now use startswith.
  - The boxed-list test uses an accumulating index_put.
  - The index guard test: on RD tops, case 2 is the top's own (1, 3, 1, {}).
- aten: new tests for the PLE ops, the symbolic clamp bound, the depthwise convolution, the generator state
  invariant and int-read reporting.
