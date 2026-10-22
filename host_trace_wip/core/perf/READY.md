# core/perf: ready patches (perf lane, G4)

Each diff applies with `git apply -p1` (or `patch -p1`) from a tree root. 01 targets wt-sgl-cpp (the C++ entry,
host_trace/*.cpp, after 48 sglcpp); its tests and 02 target int6/snap/step46 (with or without 48 int6cpp). All
checked with `git apply --check`.

## F4b class: zero-numel / size-1 strides from a prediction (coordinator 2026-10-08) -- analysis + design, not implemented (candidate 4)

Ownership (10-08), after checking land/core/symti/DESIGN.txt:
- F4d (the empty-reduction keepdim layout) moves to the symti lane. Its sym_sizes resize_reduction / make_reduction
  (coverage step 2) gives eager's layout by construction. I sent them the repro and notes and wrote no F4d fix.
- flip (F4b) is covered by symti only if flip moves onto its route, where flip's own at::empty_like reaches
  _TraceMode's allocate, which is exact for an empty source. flip is outside symti's listed coverage (tagged +
  _UNTAGGED_ELEMENTWISE). Today it is an eager step with FZ05, which passes the flip test.
- Stays with me: the general class below (eager-step outputs a fake kernel predicts, whose address-free strides fn
  reads during the trace). symti does not touch eager steps.

Measured (CPU, step228):
- The fake/meta kernels disagree with eager on strides that address nothing:
  - `v4 = randn(34, 1)[:, 1:]` (sizes (34, 0), strides (1, 1)); `v4.sum(0, keepdim=True)` gives eager strides (1, 1),
    meta and fake strides (0, 1).
  - flip (prims.rev's meta is the empty_like decomposition, i.e. empty_permuted) is contiguous for an empty input,
    where eager's at::empty_like keeps the input's strides (sound/flip_meta.py, sound/zmeta.py).
- Upstream these are meta-function differences, not host-trace bugs.

Why seed 2022925 is wrong:
- `_check_witness` compares each eager call's predicted output with the warm-up's real one. Where they differ only in
  strides that address nothing (a size-1 dim, or any dim of an empty tensor), it patches the output's size-1 strides
  after the trace (`o._sym_strides = ...`).
- By then fn has already read the predicted value: `v6.as_strided(..., (v6.stride(0) * 2, ...))` baked 0 into v7's
  strides during the trace.
- So it is a post-hoc fix of an op's own output that misses everything derived from it. The same holds for any
  size-1-dim stride a user reads, not just empty tensors.
- F4b proper (flip traced at a contiguous empty input, replayed at (35, 1)) passes on step220/228. flip is an eager
  step (T2), and FZ05 keeps eager's tensor for an empty eager output. A stride fn reads during that trace would still
  be stale.

Design for the class (handles the class rather than the op; candidate 4):
1. Reconcile at dispatch, not after the trace. When an eager call or traced host records its fresh outputs, pair them
   with the warm-up witness's k-th call of the same key (the data `_check_witness` already uses). Adopt eager's
   strides wherever they address nothing (size-1 dims; every dim of an empty output), before fn sees the tensor.
   - The end-of-trace count check stays. A key whose call count differs between warm-up and trace declines the trace
     if any of its outputs adopted values, since the pairing is then unproven.
   - Without a witness (a trusted retrace) there is nothing to adopt from. The prediction stays, and step 2 guards it.
2. Guard what fn read. Mark a traced tensor's address-free strides as read when fn calls .stride() / as_strided /
   is_contiguous on them.
   - For an eager step, the replay compares eager's real value for a read stride with the recorded one. A mismatch
     refuses the variant (a miss that retraces at the call), instead of today's `_Disagreement` or FZ05's
     no-compare.
   - For a traced host's empty output, a read stride declines at the trace: no kernel or replay shows eager's value.
   - Strides fn never reads keep FZ05's rule (eager's tensor stands).
3. Regression tests: land/scratch/fuzz/repros/F4d_asstrided_zero_size_keepdim.py (seed 2022925 minimized: a0 (34, 1)
   float32, 3 identical calls, (0, 0) output, replay (0, 1) vs eager (2, 1); still failing on step242), and the flip test
   (test_flip_of_an_empty_input_keeps_eagers_strides, already in the S1 candidate 3 patch).
- Not trivial: it touches the eager-step record path, the replay's eager-step check (C++ EagerStep.cpp and the
  Python twin) and _TracedTensor's stride reads. So it goes to candidate 4 as the coordinator suggested.

## S1 on candidate 3 (snap/step219 odd, step220 even) + F4b check (coordinator 2026-10-07) -- ready, NOT handed to int6

Update 10-08: 219/220 were superseded (97b(c) left candidate 3). The patches are rebased onto the candidate 4 bases:
- gaps/S1_aten_inductor_sound.step228.patch (even): md5 8cf803181f2c6b2b4ebaddef4e20fe72
- gaps/S1_aten_inductor_sound.step227.patch (odd): md5 52584e617a947625248fd5e75553cbeb
- Trees: core/perf/src_s4e and src_s4o.
- Even 228 gates on gh200-b (GH200, ht/perf/s4e = ht/int6c3/cand3_s228 + the 5 files; logs ht/perf/logs/s4gh*):
  - base new tests: F4b and mse pass; own-tag and no-report fail (no report gives a wrong result); the descriptive_names
    rename gives wrong gradients;
  - patched: new tests 23 + 3 OK, replay 250 (skipped=1), tape 43, inductor 84, aten 1682 OK;
  - aten needed one more test update: test_masked_fill_of_a_device_value (new in 228). Its device-value form runs
    where.self_out, an out= overload that ATen leaves untagged (where.self is tagged), so it is now an eager step.
    Rerun 3/3 OK.
  - decode36 / prefill36: (2, 13, 1) on base and patched alike.
- Odd 227 gates (GB300 GPU 1, gpu1.lock; sound/s4o_*): new tests 25 + 3 OK, tape 43, aten 1679 OK, inductor 84 OK.
  - replay 188: only the 2 known odd failures (test_a_zero_element_eager_output_keeps_eagers_strides, which needs
    FZ05's C++, and test_a_redo_takes_a_dispatch_whose_guards_hold; both fail on odd 197..241, MERGE_LOG 2837).
- Losses to propose as yaml tags (the user's call): masked_fill_.Scalar, masked_fill_.Tensor, rsub.Tensor and
  where.self_out.
- The 219/220 gate runs below were stopped when those steps were superseded. What finished there:
  - patched new tests OK on both lines;
  - even replay 247: only FZ06's known respecs error;
  - even tape 43 OK;
  - decode36/prefill36 (2, 13, 1) on 220, base not measured (minops84 reads `respecs`, which candidate 3 removed).

S1 (the section below, written against 187/188) is rebased onto the candidate 3 tops. The bases moved, so the change
set is smaller:
- Kept:
  - item 1 (`_pointwise` is this overload's own tag; `_NULLARY` is fill_.Scalar / fill.Scalar);
  - item 2 (a kernel with no matching LaunchLayout report declines, so no witness bytes are frozen);
  - item 3 (Inductor C++ kernels found by code-cache provenance; backward primals from AOTAutograd descriptors).
- Dropped:
  - S1's 1b scalar-member gate. T2's `_witness_takes` allowlist (`_UNTAGGED_ELEMENTWISE`) now keeps
    mse_loss_backward and the other loss backwards off the pointwise host. My gate would only have added declines
    for the allowlisted *_backward ops, whose scalars are their arguments.
  - The extended `_NULLARY`. On these bases, factories that are neither tagged nor allowlisted already decline
    before my change.
- Patches:
  - gaps/S1_aten_inductor_sound.step220.patch (even): md5 443e011be7245a2266eb98cc1305c1a5
  - gaps/S1_aten_inductor_sound.step219.patch (odd): md5 22fd2f8b370ea639028a2669e95d5dc3
  - Both pass `git apply --check` on fresh copies.
  - Net LOC: library +43 / -31 (net +12); tests, even +92 / -13, odd +76 / -13.
- Still silent on step220 before this patch (measured): the add_numel op with its launch report removed is wrong at a
  second size (err 4.0). Inductor's cpp__0 kernel gives wrong gradients. The own-tag unit test fails.
  - mse_loss_backward is already correct on step219/220, through T2. The test is kept as a regression guard for it,
    and now expects T2's "ATen declares it no elementwise kernel".
- F4b (flip of an empty input, at other strides than the trace's):
  - The fuzz repro PASSES on step219 and step220, but it never replays there: every call is uncaptured, an eager
    region or a slow call (f4b/repro_21{9,0}.log).
  - So I added test_flip_of_an_empty_input_keeps_eagers_strides (even only). It traces at the contiguous empty call,
    replays at strides (35, 1), and asserts the strides equal eager's and that replays >= 2.
    - On step220 (base and patched) it passes. T2 makes flip an eager step, and FZ05 keeps eager's tensor for an
      empty eager output.
    - On step219 (odd) it FAILS LOUDLY: `_Disagreement: aten.flip.default output 0 is ([0, 32], [35, 1], ...); its
      fake kernel predicted ([0, 32], [32, 1], ...)`. The odd line's frozen int5b C++ predates FZ05's EagerStep.cpp
      half (as FZ05's READY notes).
    - The error is loud, not silent, but it reaches the user on odd. It needs the int5b C++ rebuild, so the test is
      left out of the odd patch.
- Gates (GPU 1, a few lock holds):
  - new tests: even 23/23 OK, odd 22/22 OK; inductor new 3/3 OK on both;
  - the rest of the 219/220 gates stopped when those steps were superseded (see the 10-08 update above).

## S2 nfnet: native_batch_norm in training without running statistics (coordinator 2026-10-07) -- ready, NOT handed to int6

Cause: the decline is accurate, not an inference/training mix-up. timm's ScaledStdConv2d (dm_nfnet_f0, nfnet_l0)
standardizes its conv weight on every forward with
`F.batch_norm(w.reshape(1, C, -1), None, None, weight=gain*scale, training=True, momentum=0)`. That is a batch norm in
training with no running statistics, which the traced host declined outright ("a batch norm in training").

Fix (C++, even line only; the odd line's frozen install is unchanged): the traced native_batch_norm host now mirrors
batch_norm_cuda_out's training branch when there are no running statistics.
- batch_norm_mean_var's contiguous case is batch_norm_stats_cuda_template<scalar_t, int32_t, Var>:
  - one block per channel;
  - getNumThreads(F) x max(1, 512 / tf) threads;
  - the variance goes into save_invstd.
- batch_norm_calc_invstd(save_invstd, save_invstd, eps) is the same functor as the eval path, now over the variance.
- The elementwise transform is the existing one.
- Still declined, each with its own reason:
  - training that updates running statistics (batch_norm_update_stats_and_invert);
  - a training input that batch_norm_mean_var does not take as contiguous (channels-last / general).
- Patches:
  - gaps/S2_training_batch_norm.build7_csrc.patch (against int6/build_cpp7/src, Normalization.cu), md5
    2258811c21f2d88a2bb6e8b9228c34ec, +40 / -14;
  - gaps/S2_training_batch_norm.step188_test.patch, md5 d07035bf2541ce910627b69638c0f665, +19.
  - Built isolated in core/perf/bn (src_bn = build_cpp7/src + patch; bn/build.sh, CPUs 108-143, MAX_JOBS 16,
    ccache read-only); launcher bn/python_bn.sh.
- Test test_batch_norm_in_training_without_running_statistics[float32, float16]:
  - nfnet's call at (C, F) = (64, 27) x2, (96, 300), (32, 576), so the thread choice changes. Bitwise vs eager, no
    declines, all KernelLaunch.
  - A training call with running statistics declines with its own reason.
- Gates (GPU 1; bn/install = build7 + this patch, src_bne = step188 + test):
  - the batch_norm tests: 14 OK;
  - aten: 1671 OK (+2);
  - replay: 247 OK (skipped=1).
- Candidate 3: Normalization.cu is the same file in build_cpp8/src, so the csrc patch applies unchanged. The test
  patch applies to step220.
- nfnet rows (ht_eager_conv on bn/install; coverage counts only, since GPU 1 is not a timing GPU; logs bn/out,
  bn/nfnet.log). Each model went from native_batch_norm 57 + avg_pool2d 3 eager steps (candidate 2) to avg_pool2d 3
  only:

  | model | native_batch_norm eager steps (candidate 2 -> now) | traces / replays / eager | output check |
  |---|---|---|---|
  | dm_nfnet_f0 | 57 -> 0 | 1 / 34 / 1 | bitwise, replayed |
  | nfnet_l0 | 57 -> 0 | 1 / 34 / 1 | bitwise, replayed |

  - That removes the 114 batch-norm steps, 49% of the 234 remaining in NUMBERS.md (measured on these two rows only;
    the suite was not re-run).

## S1 sound ATen + Inductor: exact pointwise tag, required launch reports, Inductor metadata (coordinator 2026-10-06) -- ready, NOT handed to int6

Base: candidate 2 (snap/step187 odd, snap/step188 even). Python only. Trees: core/perf/src_s1o, src_s1e. Scratch and
logs: core/perf/sound/.

Patches:
- gaps/S1_aten_inductor_sound.step188.patch (even): md5 2f2f8f4b28d57c7422f6bacb73c6d204
- gaps/S1_aten_inductor_sound.step187.patch (odd): md5 c76628170bc79f6134c1dc28e92b4c95
- The changes in the two patches are identical (the odd one needs fuzz on one hunk only).
- Net LOC: library +53 / -32 (net +21); tests +62 / -13 (net +49).

### Silent wrong results found, before and after (all measured)

| case | before (step188) | after |
|---|---|---|
| mse_loss_backward / smooth_l1_loss_backward / huber_loss_backward (mean), second shape | wrong at every element (max err 8e-4 / 9e-5 / 9e-5): the functor's 2 / numel is replayed at the traced numel | correct (eager step) |
| binary_cross_entropy_backward (mean), second shape (probe only) | wrong (max err 3e-2) | (same rule; not in the test) |
| Inductor C++ kernel not named `cpp_fused*` (cpp.descriptive_names=False names it cpp__0): XLNet-style CPU constant | wrong gradients (max err 62.8): the kernel ran once at the trace and never at a replay | correct (host step) |
| same, CPU rand (fallback_random=False) | wrong (the trace's draw reused) | (same fix) |
| add_numel test op with its launch report removed (simulates an unreported launch site) | wrong at a second p size (err 2.0) | correct ("has no launch report") |

### 1. `_pointwise`: this overload's own tag only

- The census (sound/pw_census_all.txt) found 90 overloads that old `_pointwise` accepted without their own tag.
  - 57 come from a sibling overload's tag. These include max.default / max.dim / min.default / min.dim (reductions) and
    where.default (nonzero).
  - 27 come from stripping a trailing "_" from the name.
  - 6 are fill packets.
- `_NULLARY` is now exact overloads, each justified by its code (cited in the comment): fill_.Scalar, fill.Scalar,
  zero_, full, ones, ones_like, new_full. All of them are FillFunctor of their value argument.
  - fill_.Tensor of a device value is a copy_, so it is left out.
- The composite redispatch (tape.py ~837) uses the exact tag directly. Routing through `_NULLARY` would redispatch
  ones_like / new_full composites, which take no symbolic size.
- Found on the way: `_elementwise` (the broadcast-shape inference for untagged ops) is the route by which the
  loss-backward ops above reach the pointwise host. An untagged op's functor scalar can come from shapes.
  - New rule: an op that is not `_pointwise` declines at the pointwise host if any of its kernels has an 's'
    (functor scalar) byte. Pointers, host-set fields and probed CPU scalars are still fine.
  - A zero-element reduction's fill is exempt, because the host already checks that the reduction's input stays
    empty.

### 2. Kernels with no LaunchLayout report

- Verified: before, a kernel whose report was missing or did not match kept the witness's bytes, scanned only for
  addresses. A size member was then frozen at the traced shape (shown with the add_numel test op).
- Now the pointwise host declines "has no launch report" / "launch report is not its node's". The CPU-buffer-only
  check and the unclassified scan branches are gone.
- Cost (measured census, sound/census_*.tsv): 0 kernels without a matching report across decode36, prefill36,
  minops84 decode/prefill, and the replay, tape and aten test files.
  - Every CUDA pointwise launch route in CUDALoops.cuh / CUDAJitLoops.cuh / Loops.cuh / jiterator.cu reports.
  - The unreported sites are ROCm-only, the vectorized_templated kernels and the manual-unroll kernel. The C++ host's
    route-name check already declines all of them.

### 3. Inductor metadata instead of name prefixes

- `cpp_fused` (cudagraph_host_trace.py:393) failed silently: see the table above.
  - Now `_cpp_kernel(v)` is used: v is the entry function (`CppPythonBindingsCodeCache.entry_function`) of a module
    the code cache itself loaded (`_loaded_module_names`), so detection depends on where the kernel was loaded, not
    on its name.
- `primals_` (:95) fails only in accounting, not results. A rename would make `held` empty, so every static input
  would count as freed. `freed_arguments` only moves run splits and peak accounting; a caller's reference still keeps
  the storage.
  - Now `PlaceholderInfo.aot_input` (new field, default False) is used. It is set from the placeholder's
    AOTAutograd descriptor (`meta["desc"]` is an AOTInput), which the partitioner copies onto backward placeholders
    that are joint graph inputs. Saved intermediates have no descriptor.
  - A graph with no descriptors at all holds every static input (the switch-off behavior) instead of guessing.

### What it loses (measured)

- decode36 / prefill36: unchanged, (1, 14, 1) each, bitwise.
- minops84 (even): decode traces 0, re-run 30, bitwise 23/23; prefill traces 0, re-run 23, bitwise 17/17. Same as
  int6's step188.
- Only tagged ops reach the pointwise host in those workloads.
- In the test files, these now run as eager steps (census hits before, per file):
  - masked_fill_.Scalar (4 in aten; also masked_fill's composite body);
  - rsub.Tensor (12 in aten);
  - max/min.dim never reached the pointwise host (other routes served them).
- Proposal for native_functions.yaml (user's call; not edited): tag `pointwise` on masked_fill_.Scalar,
  masked_fill_.Tensor and rsub.Tensor.
  - masked_fill_ is the in-place form of a tagged op.
  - rsub.Tensor's CUDA kernel is `at::sub(other, self, alpha)` (BinaryOps.cpp).
  - Tagging them restores the old coverage soundly.
- Suites: not measured. Under Inductor these ops lower into Triton kernels, so only eager-traced code and Inductor
  fallbacks are affected (estimate).

### Tests (each shows the unsound case failing on step188 and passing after)

- aten:
  - test_pointwise_kernel_without_a_launch_report_declines;
  - test_a_functor_scalar_of_an_untagged_op_is_not_replayed (x3);
  - test_pointwise_is_the_overloads_own_tag (unit).
- inductor:
  - test_cpu_constant_feeds_the_device[descriptive_names=False];
  - test_a_backward_frees_its_saved_tensors now uses aot_input.
- Updated to the new behavior:
  - test_contiguous_copy_is_a_memcpy expects masked_fill_ as an EagerCall;
  - the "rsub" pointwise case uses rsub.Scalar.
- Before (logs sound/base_new_*.log): the three loss ops are wrong in all 3000 elements; add_numel err 2.0;
  descriptive_names=False grads err 62.8.

### Gates (GPU 1, gpu1.lock)

- even (src_s1e, build_cpp7): all pass.
  - aten 1674 (+5);
  - replay 247 (skipped=1);
  - tape 43;
  - inductor 83 (+1);
  - decode36 / prefill36: (1, 14, 1) each;
  - minops84: decode re-run 30, traces 0, 23/23 bitwise; prefill re-run 23, traces 0, 17/17 bitwise.
  - Logs: sound/s1e_*.log.
- odd (src_s1o, int5b frozen):
  - new tests 22 + 3 OK;
  - replay 185 (skipped=1);
  - tape 43;
  - decode36 / prefill36: (1, 14, 1) each;
  - minops84: prefill re-run 27, decode 3 traces / re-run 1319, the same as int6's step187.
  - aten / inductor not run: stopped when the bases moved to candidate 3. Logs: sound/s1o_*.log.

## G15 check_escapes: the tape's eager outputs are not user-kept (coordinator 2026-10-06) -- ready, NOT handed to int6

This is an add-on on top of G14. It is Python only, has no options, and does not use gc.collect.

The fix is one line at the end of `_copy_tensors`: `tape.eager_outputs = copied(tape.eager_outputs)`.
- `Tape.eager_outputs` (one traced tensor per eager-step output) was the only tape record that `_copy_tensors` did not
  replace with copies. So any trace with an eager step looked like a tensor fn kept "in a list".
- The copies keep `_root`, which is all that the eager_outputs readers use (`next(o for o in ... if o._root is r)` in
  tape.py and redispatch.py). `copy_of`'s shared map keeps the copies identical to those in the EagerCall records.
- What counts as user-kept is unchanged. A tensor fn really keeps is still reported, including an eager step's output.

Correction to the G14 note below: the EagerCall `.args` / `.outputs` holders were not the tape's own records.
`_copy_tensors` already copies the EagerCalls in `tape.launches`. Those holders were in the `_Trace` itself
(`aten_calls` and others), which H pins. When the first trace in a process lazily imports torch._dynamo (from
FakeTensorMode.__init__, via torch.nested), torch.fx.wrap's self-held frame keeps the f_back chain up to
`_Trace.__init__`, and so `self`. With dynamo imported first, the remaining holder was `Tape.eager_outputs`.
Probes: core/perf/esc/probe2.py and probe_t.py.

Patches:
- gaps/G15_eager_outputs_not_kept.step179+G14.patch (odd): md5 080ee893b2759fff65e3ed89064d60b4
- gaps/G15_eager_outputs_not_kept.step180+G14.patch (even): md5 51a7d8d40fb38e77506bfe24dcac885e
- `git apply --check` is clean on step179+G14 and step180+G14. Either patch is also clean on snap/step183 and step184.
- Net LOC: library +1; tests +26.

Test `test_an_eager_steps_output_is_kept_only_if_fn_keeps_it`, parametrized on keep. Its fn is `x[i].sum(-1)`, where
index.Tensor is an eager step, run under check_escapes=True:
- keep=False: no decline, only the eager-step note, and it replays.
- keep=True: fn appends the eager output to a list, and the call still declines with "kept a traced tensor ... in a
  list".
- The test imports torch._dynamo itself (test-local H workaround), so it passes run alone (checked) as well as in the
  full file.
- keep=False fails on unpatched step180+G14 (replays 0, it declined). keep=True passes on both.

Gates (GPU 1, gpu1.lock, started only when no int6 job was on the lock):
- even (src_g15e, build_cpp6 install): replay PASS, 247 tests (skipped=1); tape PASS, 43 tests.
- odd (src_g15o, int5b frozen): replay PASS, 185 tests (skipped=1); tape PASS, 43 tests.
- The first replay run failed keep=False on both lines. Its exact replays count depended on order (1 run alone, 2 in
  the full file), so the test now asserts traces == 1, the declines list, and replays > 0.
- Logs: logs/g15_{even,odd}/tape.log and logs/g15_{even2,odd2}/replay.log.
- Conformance was not run (the minimum-GPU instruction). No options.json change.

decode36 / prefill36 with check_escapes=True (held.py on the even line). Counts are (traces, replays, eager):

| | before (step180+G14) | after (G15) |
|---|---|---|
| torch._dynamo imported first: decode36 | (8, 0, 16), declines "kept ... in a list" | (1, 14, 1) |
| torch._dynamo imported first: prefill36 | (8, 0, 16), declines "kept ... in a list" | (1, 14, 1) |
| no pre-import: decode36 | (8, 0, 16) (measured on step178+G14) | (2, 13, 2) |
| no pre-import: prefill36 | (8, 0, 16) (measured on step178+G14) | (2, 13, 2) |

- The "after" counts with dynamo imported first equal the check_escapes=False counts (1, 14, 1). With them, decode36
  has only the index.Tensor eager-step note, and prefill36 has no declines.
- Without the pre-import, the first trace still declines "kept ... in a list", because H pins that trace's `_Trace`.
  The next trace succeeds. This costs one trace, once per process, only when dynamo is not yet imported.
- Fixing H needs torch.fx.wrap not to hold its own frame. That is upstream and is noted only, not filed.
- Memory with check_escapes on is not separately measured (see G14 for the memory numbers).

## G14 cycle-free replays + no gc.collect in the escape check (coordinator 2026-10-05) -- rebased; handed to int6 as 183/184 (10-06)

Rebase (10-05 21:28). 86 changed the redispatch import to `_TraceShapeEnv, Declined, declined`, so hunk #1 of the
step177/178 patches failed. The rebased patches differ only in that import line
(`from torch.cuda._host_trace import _drop_tracebacks, _TraceShapeEnv, Declined, declined`); there are no code changes.
- gaps/G14_cycle_free_replays.step179.patch (odd, 177+85+86): md5 ad0761779d4a398943d45d5431f962fc
- gaps/G14_cycle_free_replays.step180.patch (even): md5 7a4d75bc21832a2cd1f97614d1f4d5ee
- `git apply --check` is clean on fresh copies of snap/step179 and snap/step180.
- Replay file on the rebased trees (GPU 0, gpu0.lock):
  - odd, src_g14o on install_land_int5b_frozen: PASS, 183 tests (skipped=1). Log: logs/g14r_odd179/replay.log.
  - even, src_g14e: PASS, 245 tests (skipped=1). This ran on the build_cpp6 install, because build_cpp7 was not ready.
    Log: logs/g14r_even180_b6/replay.log.
  - NOT run: snap/step182 or the build_cpp7 install. step182 did not exist before the 22:01 stop. int6's gates cover it.
- The older step177/step178 patches below are superseded.


Patches (Python only, no build):
- gaps/G14_cycle_free_replays.int6cpp.patch (md5 78f374e4db2c9b25759cb3c19f283dcc), made against int6/snap/step178.
- gaps/G14_cycle_free_replays.int6.patch (md5 37d0cbbc618693c9bd9a6849ed68788b), made against int6/snap/step177.
- Each one passes `git apply --check` on both step177 and step178.
- Trees: core/perf/src_cyc (step178 + patch, run on build_cpp6) and core/perf/src_cyco (step177 + patch, run on
  install_land_int5b_frozen).

Net LOC:
- Library: +23 / -10, so net +13, across _host_trace.py, _host_trace_replay.py, _host_trace_tape.py and
  _host_trace_redispatch.py.
- Tests: +31 / -4, so net +27.

What changed:
- `_drop_tracebacks(e)` is a new helper in _host_trace.py.
  - It clears `__traceback__` along the cause/context chain and keeps the chain objects. Line 1144 still reads
    `e.__cause__`.
  - It has a seen-set, so a `raise a from b` loop can't spin.
- A: `_miss` (replay 1116 and 1140) now calls `_drop_tracebacks(e)` instead of `e.__traceback__ = None`.
- C/F: `_add_entries` calls `_drop_tracebacks(e)` before `return e`.
- D: `_drop_tracebacks(tr.declined)` runs in `_symbolic_run`'s finally (the end of the traced run) and in `_redo`
  before its "declines again" FoldRefused. Identity is kept, so the `e is tr.declined` checks are untouched.
  - Side effect: a Declined that propagates out of a direct `trace()` call now has a traceback that starts at
    `_symbolic_run`, not at the raise site.
  - Through HostTraceReplay, tracebacks were already dropped at 1140.
- E: `trace()` now does `try: raise d from e` / `finally: del d`.
- The escape check's `gc.collect()` is deleted, together with its `if tensors:` guard. An object held only by a
  cycle now counts as kept. The trace() docstring says so.
- `_gc_hold` in trace() is kept. It does not exist only for the collect:
  - Its comment (tape 2641) says it keeps the automatic collector from finalizing a CUDAGraph during the capture.
  - The same hold is used by `_segment`, capture.py:430, harvest:966 and replay 622/711.
- B (test): `test_replay_hooks_run_on_a_native_hit` keeps the end-hook handle and removes it in a finally.
- New test: `test_a_redispatch_refusal_leaves_no_cycle`.
  - It runs with gc disabled. After a redispatch refusal, `del f` must free the replay and every segment CUDAGraph
    (checked by weakref).
  - Measured: it fails on int6/snap/step178 (the replay is still alive) and passes on src_cyc.
- One test relied on the collect: `test_a_cycle_fn_leaves_is_not_kept` tested the collect itself.
  - It is now `test_a_cycle_fn_leaves_is_kept` and asserts (traces, replays, eager) = (1, 0, 3) and "in a dict".
  - No other test needed a gc pass.

Measured. Correctness runs used GPU 0 under gpu0.lock, because GPU 1 was 4-6 deep:
- Full replay file as `__main__` (the gate's mode) on src_cyc: 3/3 runs OK, 244 tests.
  - int6's step178 gate fails it, on test_a_cycle_fn_leaves_is_not_kept.
- gc-off scan (cyc2/scan.jsonl, collect after each test, src_cyc): 0 of 243 tests leave a graph or replay that only
  gc frees.
  - Before (step172): 20 tests, 46 graphs (44 instantiated), 21 replays.
  - The hook test (B) is clean too.
  - H does not show, because the suite has torch._dynamo imported before its first trace. No test workaround is
    needed.
- decode36 / prefill36 (cyc2/held.py; 8 shapes x 2 calls, bitwise vs eager; (traces, replays, eager) = (1, 14, 1)):
  - Live memory with the replay alive (alloc / reserved MiB):

    | workload | before (step172) | after, no preimport | after, torch._dynamo pre-imported |
    |---|---|---|---|
    | prefill36 | 8.75 / 102 | 1.61 / 66 | 1.56 / 66 |
    | decode36 | 1.58 / 66 | 1.56 / 66 | 1.56 / 66 |

    So prefill no longer holds 7.2 MiB allocated / 36 MiB reserved past the calls.
  - `del f` with gc off, torch._dynamo pre-imported: graphs 3 -> 1 by refcount (the harvest keeper is left) and the
    replay is not in gc garbage, on both workloads.
  - `del f` without the pre-import: graphs stay at 3 until gc. That is H, the torch.fx.wrap self-cycle. It pins the
    one replay whose first trace imports torch._dynamo, once per process. It is upstream and only noted here;
    nothing was filed or posted.
- Gates (gate_g9.sh: the int6 gate set plus capture, triton_abi, lower, lower_tape, cover, fullgraph,
  inductor/test_cudagraph_host_trace and python_entry):
  - even (src_cyc, build_cpp6): 17/17 files pass. replay has 244 tests (+1 for the new one); python_entry has 507.
  - odd (src_cyco, int5b frozen): 16/16 files pass (no python_entry, same as int6's step177 set). replay has 182
    tests (+1); the counts match step177's for the other files.
- Conformance (my copy core/perf/conf_cyc; overlays cyc_even / cyc_odd mirror int6_s178 / int6_s177, including the
  memory.held respell):
  - cyc_even is identical to int6_s178: present 160, respelled 2, option tests 158/0/6, cases 51/0/0.
  - cyc_odd is identical to int6_s177: present 150, respelled 1, option tests 148/0/16, cases 32/0/19.
  - No options.json row change is proposed. The check_escapes spelling and default are unchanged.

Found on the way, not fixed (it was there before this change). With check_escapes=True, decode36 and prefill36
decline:
- The message is "fn kept a traced tensor past the trace, in a list ... (it stays a traced tensor)".
- (traces, replays, eager) = (8, 0, 16), the same on unpatched step178 and on src_cyc.
- Cause (cyc2/esc_probe.py): the tape's own EagerCall `.args` / `.outputs` hold the traced tensors. These are the eager
  steps (index.Tensor, sum.dim_IntList), which `_copy_tensors` does not cover.
- So check_escapes is a false positive on any trace that has an eager step. Only the check_escapes tests use the
  option, and its default is False.

## gc cycles that hold CUDAGraphs (coordinator 2026-10-05) -- diagnosis only, no fix written

All runs are on step172 (int6/snap/step172, build_cpp5) with my copies under cyc/. Correctness runs used GPU 1, or
GPU 0 when GPU 1 was queued. Scripts: cyc/scan.py (SCCs of the garbage after each test), cyc/scan2.py (per test:
the CUDAGraphs and replays that only a gc pass frees; PROBE_A / PROBE_C monkeypatch the fixes below in memory),
cyc/nat/ (the test file run as __main__ like the gate, with tagging of which test made each graph), cyc/held.py (the
decode36 / prefill36 proxies; it cuts cycle edges until none are left).

Main finding: when one of these cycles forms during a call, the cycle holds the call's frames. `_miss`,
`_redispatch` and `_call_held` keep `self`, so the cycle holds the whole HostTraceReplay: every variant, its graphs,
its pools and its traced tensors. Nothing is freed by refcount, not even on `del`.

Q2, the order dependence (measured, full file as __main__, 3/3 runs fail):
- At the start of test_a_cycle_fn_leaves_is_not_kept, 2 graphs are alive (int6's "2"):
  - 1 is live, not garbage. It comes from TestBoundCall.test_a_bound_call_from_within_a_bound_call and is not
    finalized.
  - 1 is instantiated, in cyclic garbage, from TestChainReplay.test_cudnn_attention_of_a_shape_it_does_not_take_raises
    (cycles C+D). It is the one finalized during the test. The escape check's gc.collect() inside the capture runs
    cudaGraphExecDestroy (CUDAGraph.cpp:434, "operation not permitted when stream is capturing"). The capture is
    invalidated and the calls run eagerly.
- Only instantiated graphs break it. cudaGraphDestroy of a never-instantiated graph is allowed mid-capture.
  - Pairs run with gc as is: segment_fails_to_capture + target, segment_fails_to_verify + target, and both + target
    all pass. Those graphs are never instantiated.
  - test_replay_hooks_run_on_a_native_hit + target fails.
- Which test supplies the graph depends on gc thresholds: whatever instantiated garbage is still pending when the
  target runs. 20 tests can supply one (next item), which is why adding 81's test flipped 170 to 172.
- An import-mode run (my order.py) passed only because the gc timing differed.

Q1 and Q3, the cycles (measured):
- scan2, automatic gc off, collect after each test, 241 tests:
  - 20 tests leave 46 CUDAGraphs (44 instantiated) and 21 HostTraceReplays that only gc frees.
  - With probes A+C: only 1 test leaves a graph, the hook test (scaffolding). 2 tests leave replays with no graph
    (cycles D, E).
- decode36 / prefill36 (8 shapes x 2 calls, bitwise vs eager, (traces, replays, eager) = (1, 14, 1)), after `del f`
  with gc off:
  - 0 of 2 segment graphs freed, 776 / 781 traced tensors alive.
  - The gc pass then frees both graphs. gc.garbage lists 701 / 745 _TracedTensors directly. The graphs hang off the
    cycle through an untracked holder, so they are freed by the collection without showing up in gc.garbage.
  - MemPool count is 1 before and after (the process-global harvest keeper pool), so no MemPool is in a cycle.
  - Memory held until gc:
    - decode36: alloc 1.58 -> 1.56 MiB, reserved 66 -> 66 MiB.
    - prefill36: 8.75 -> 1.56 MiB allocated and 102 -> 66 MiB reserved. That is 7.2 MiB allocated and 36 MiB reserved
      held past the calls, even while the replay is alive.
  - With probe C and torch._dynamo pre-imported, live memory is already 1.56-1.61 / 66 MiB. `del f` frees both
    segment graphs by refcount (3 -> 1, the keeper left), and the replay is not in the garbage.
  - The remaining 66 MiB is the harvest keeper (64) plus 2 MiB of default pool, both process-global.

The cycles:
- A (library, holds graphs).
  - In `_miss`'s retry loop (_host_trace_replay.py:1110-1117), `failed = e` keeps a SegmentFailed. Only its own
    `__traceback__` is cleared.
  - Its `__cause__` (the RuntimeError or Declined from `_segment_failed`'s `raise ... from e`) keeps a traceback
    through the `_segment` frame (local `graph`) up to capture_tape, `_build` and `_miss`, which holds `failed`.
  - Tests: segment_fails_to_capture / _verify (4+2 graphs, 4 instantiated; the replay's other segments too).
- C (library, the big one).
  - `_add_entries` (_host_trace_replay.py:1306-1310) returns the caught FoldRefused. Its traceback holds the
    `_add_entries` frame, whose f_back is `_redispatch` / `_call_held`, which holds `refused`.
  - It forms on every redispatch refusal: 17 tests (the composite, redo, respec, split, autotune, heuristics, ...),
    and decode36 / prefill36 on "aten.sum.dim_IntList dispatches otherwise".
- D (library).
  - `_Trace.declined` (_host_trace_tape.py:372-376) keeps the Declined that gets raised. Its traceback holds the
    `_eager_call` frame, whose cell holds the `_Trace`.
  - In redispatch it is FoldRefused's `__context__`, so walking the chain in C breaks it (cudnn test).
  - Left after A+C: test_a_python_error_of_a_host_route_raises_in_the_suites (1 replay, no graph).
- E (library). In `trace()` (_host_trace_tape.py:2969-2972), the local `d` raised `from e` has a traceback holding
  `trace()`'s frame. test_errors_of_the_trace_decline (3 replays, no graph).
- F (library). HintsFail / ZeroDivisionError inside a FoldRefused chain (_host_trace_lower_tape.py:696-702, redispatch
  rename:490). Same shape as C; the chain walk covers it.
- B (test scaffolding, holds a graph). test_replay_hooks_run_on_a_native_hit:
  `segment.graph.register_replay_end_hook(seen.append)` is never removed, so `seen` <-> graph.
- G (test scaffolding). A MagicMock cycle in test_errors_of_the_trace_decline. Harmless.
- H (upstream torch.fx, a one-time pin).
  - torch.fx.wrap (_symbolic_trace.py:1374) keeps `currentframe` = its own frame, a self-cycle whose f_back chain is
    the importing stack.
  - The first trace's FakeTensorMode init lazily imports torch._dynamo, and dynamo/utils.py:6139 calls fx.wrap. That
    pins the first replay that triggers the import: in held.py, probe C alone still left the replay in garbage until
    torch._dynamo was pre-imported.

Q4, proposed minimal fixes (not written; LOC estimated; A and C verified as in-memory probes):
- One helper, about 4 LOC, e.g. in _host_trace_replay.py:
  `def _drop_tracebacks(e): while e is not None: e.__traceback__ = None; e = e.__cause__ or e.__context__`.
  It keeps the chain objects, since line 1144 reads `type(e.__cause__)`.
- A: at line 1116, `e.__traceback__ = None` becomes `_drop_tracebacks(e)`. 1 LOC (probe A verified). The same swap at
  line 1140 is 1 LOC, for the chain of a re-raised error.
- C/F: `_drop_tracebacks(e)` before `return e` in `_add_entries`. +1 LOC (probe C verified).
- D: when the trace ends (`trace()`'s exit and `_redo` after line 435), `_drop_tracebacks(tr.declined)`. It can't be
  a copy, because callers test `e is tr.declined`. +2-3 LOC, estimated.
- E: `raise d from e` becomes `try: raise d from e` / `finally: del d`. +2 LOC, estimated.
- B (test only): keep the handle from register_replay_end_hook and remove it in the existing finally. +1-2 LOC.
- H (upstream): `del currentframe` once f_back is taken, in torch.fx.wrap. 1 LOC, a separate small upstream change.
- Independent of the cycles: run the escape check's gc.collect() after the capture exits. Otherwise any user
  garbage holding an instantiated CUDAGraph can still be finalized mid-capture. This moves the `if check_escapes:`
  block (about 12 lines) out of the `with _gc_hold, _capture(...)`, net about 0-2 LOC. I have not checked that
  `_swap_twins` needs nothing from the capture.
- Total library change: about 10-13 LOC. The test change is 1-2 LOC.

## G13: direct output views in EagerStep.cpp (coordinator 2026-10-05) -- ready, default ON (no switch; same semantics), marginal

Speedup (GPU 0 under gpu0.lock, CPU 4; scratch/viewcost/probe.py flat view: two copy_ kernels into a fresh output
returned as a transposed view; build4 = int6/build_cpp4/install vs g13 = core/perf/g13/install, same tree src_t11c).
Separate processes, interleaved, order flipped per round; per-process medians, then the median over processes.
| setting | processes per arm | hit us build4 -> g13 | outputs phase (7) ns build4 -> g13 | C++ total (8) ns |
|---|---|---|---|---|
| `z.t()` between hits (the realistic setting) | 9 | 11.75 -> 11.98 | 1306 -> 1069 (-0.24 us) | 8585 -> 8666 |
| `zc.add(1)` (CPU op) between hits | 9 | 18.52 -> 17.85 | 1491 -> 1193 (-0.30 us) | 10540 -> 9928 |
| back to back (hot loop, not the bar) | 3 | 8.0 -> 5.55 | 1244 -> 470 | 7625 -> 4829 |
| control: base output (no view), `z.t()` | 3 | 9.68 -> 10.96 | 113 -> 160 | 6273 -> 7189 |
- Saving per view output with ops between hits: 0.24 us (t) and 0.30 us (add), measured on the outputs phase timer.
  The distributions are apart: add 1425-1573 vs 942-1377 ns, t 1222-1650 vs 708-1413 ns (8 of 9 g13 processes are
  below the build4 median). That meets the 0.2 us bar, but only just.
- At the whole-hit level 0.2-0.3 us cannot be resolved: per-process noise is about 1.5-2 us (the base control, which
  G13 does not touch, moves by 1.3 us). Whole-hit medians are +0.23 (t) and -0.67 (add); neither is a measurement of G13.
- In the hot loop it saves 2.45 us per hit, more than the 0.8 us phase cut: the dispatcher as_strided on the hit was
  itself the "dispatcher op between hits" that slowed the next hit (item 4). This is not the realistic setting.
- GPU time unchanged (6.15 us every arm).

What it does (gaps/G13_direct_output_view.int6cpp_csrc.patch, +32 -1: 4 includes and HostTraceVariant::view): for an output view
whose base has no autograd meta (no requires_grad, no forward grad, not itself a view), a dense CUDA tensor with the
ADInplaceOrView key, with default TLS dispatch sets, no conj/neg bit, no Python key or dispatch mode, view replay off
and no RecordFunction callbacks (profiler traces still show aten::as_strided), it calls
`as_view(base, at::native::as_strided_tensorimpl(...), true, true, nullptr, nullptr, creation_meta)`, the calls the
dispatcher's ADInplaceOrView kernel makes, with the same CreationMeta rule. Anything else takes the dispatcher route
as before, including any argument that requires grad or is a view.

Coverage: no change in what replays. Gates (logs/g13c, logs/g13p), both green:
- even line on g13/install, src_t11c + the test, GPU 2 (before GPU 2 was retired): python_entry 502 (67 skipped),
  replay 239, harvest 219, opaque 43, memory 19, fullgraph 8, oob 3, packed 5, cover 20, tape 43, guards 9, capture 12,
  lower 18, lower_tape 6, triton_abi 7, program 102, triton 56, cute 61 (21 skipped), integration 1,
  inductor/test_cudagraph_host_trace 82. All OK.
- odd line (frozen install, src_t11p + the test): replay 178 OK. The C++ change does not reach the odd line.
- New test `test_output_views_match_eager_autograd_metadata` (G13_direct_output_view.test.patch even,
  .odd_test.patch odd): under enable_grad, no_grad and inference_mode, outputs `y, y[1:], y.view(-1, 2), x[2:], x2[2:]`
  (x2 a view of an argument; under no_grad that argument requires grad) match eager bitwise, in `_is_view`,
  `requires_grad`, `is_inference`, `_get_creation_meta` and `_base` identity, and an in-place op on each view bumps
  its base's version counter. 0 eager calls, >1 replay per mode. It also passes on build4 (the dispatcher route).

Memory: no change (same tensors, same view meta).

Patches (apply with patch -p1; csrc on int6/build_cpp4/src, tests on src_t11c / src_t11p before the test):
- G13_direct_output_view.int6cpp_csrc.patch 42da158468e243510dbc1b73a476942a
- G13_direct_output_view.test.patch 6bcc5291ed89f207ff89acb5338add30
- G13_direct_output_view.odd_test.patch ed70c9513b1de05941735f8a5f5172c9
Build: g13/build.sh (isolated; core/perf/g13/{src,build,install}), launcher g13/python_g13.sh. Timing: g13/ab.sh,
g13/ab2.sh, raw out/g13/ab.jsonl. These timing runs had no explicit in-script foreign-process check; they were
serialized only by gpu0.lock.

## Single-op row re-measured with an op between calls (coordinator 2026-10-05), measurement only

scratch/singleop/rms_interleave.py: quack RMSNorm bf16 128x4096 on src_t11c / build4, GPU 0 under gpu0.lock, CPU 4.
CPU us per call, thread_time median of 25 x 256 calls. Two processes per mode, two rounds each (4 values; range shown).
The op runs between calls and is timed with every arm: t = `z.t()` on an unrelated 64x64 CUDA tensor (a dispatcher
call, no kernel); add = `z.add_(1)` (a dispatcher call plus a kernel launch). "- op" subtracts the op timed alone in
the same process (t 0.75-0.76 us, add 7.3-8.5 us). Every call is checked: eager count unchanged, one variant.

| arm | back to back (as the old row) | with t | with t, - op | with add, - op |
|---|---|---|---|---|
| host-trace hit | 3.4-5.6 | 6.9-8.6 | 6.1-7.8 | 6.9-8.3 |
| bare tvm_ffi call | 5.9-8.1 | 10.2-10.8 | 9.5-10.0 | 9.7-12.3 |
| quack eager wrapper | 18.5-20.2 | 20.7-23.1 | 19.9-22.3 | 22.1-27.5 |
| raw graph replay (floor) | 1.8-1.9 | 4.8-5.4 | 4.1-4.7 | 2.8-3.7 |

- The old row (hit ~4, bare 5.3-6.8, eager 14-16) was a hot loop. With an unrelated op between calls every arm pays
  about 1-3.5 us more (t, after subtracting the op), including a raw `cudaGraphLaunch`, so the per-call gap is cache state, not host-trace work.
- Interleaved, the hit is 1.3-1.6x faster than the bare tvm_ffi call (hot loop 1.4-1.8x) and 2.8-3.3x faster than the
  quack wrapper (hot loop 3.3-5.6x); ratios are per process, t, after subtracting the op. The gap to a raw graph
  replay is 1.6-4.9 us across the three settings.
- The eager wrapper is 18.5-20 us back to back today vs 14-16 in the old row; same script, other day and box load.
- Raw: out/singleop/rms.jsonl.

## G11 follow-ups (coordinator 2026-10-05): harvest memory attribution, index.Tensor, G12 quiet decline, view-output hit cost

Item 1, G11 memory (quick attribution, measured, GPU 2, src_t11c = step146 + G9 + G11, memory auto):
- Not per key. scratch/g11mem/sweep.py, N distinct cuBLAS keys (x (32, 64) @ w_i (64, 64 + 8 i)), reserved over the
  process baseline after 7 calls: opaque=() 32 / 32 / 34 MiB at N = 1 / 16 / 64; default (blas) 96 / 96 / 98 MiB, all
  N keys harvested, 0 eager steps, bitwise. The difference is a flat 64 MiB (out/g11mem/sweep.jsonl).
- What holds it: the harvest MemPool's cache (_POOL_KEEP = 64 MiB, _host_trace_harvest.py:231). It is 2 segments of
  32 MiB, active 0 (the captures' cuBLAS workspaces), kept so the next harvest reuses them. empty_cache does not
  release them: they belong to a private pool. Nothing per harvested graph or key is kept.
- The harvested GEMM's workspace at replay is a planned temporary. It is the keyed site's scratch buffer, which the
  native commit places per call in the run buffer by first fit around the run's arenas. It is freed once the run's
  graph is queued, so it is not held between calls. prefill36: 36 of 108 keyed sites have a 32 MiB scratch buffer.
  decode36: no site has scratch bytes. Allocated bytes between calls are 1 MiB on both, the same as without the harvest.
  Eager matches this: on CUDA in this build TORCH_CUBLAS_WORKSPACE_CACHE defaults off, so eager also allocates and
  frees its workspace per call.
- decode36 / prefill36 (scratch/g11mem/peaks_seg.py, out/g11mem/peaks_seg.jsonl), reserved MiB, blas vs opaque=():
  - decode36: 546 vs 480. That is the 64 MiB harvest pool plus 2 MiB of default pool.
  - prefill36: 686 vs 586. That is the 64 MiB harvest pool plus 36 MiB of default pool.
  - The prefill 36 MiB (inferred from the segment list, not isolated): the run buffer, which holds the 32 MiB workspace
    plus about 6 MiB of temporaries, takes a 38 MiB segment. The cached 32 MiB block from eager's own GEMM calls cannot
    serve it, so the allocator caches both. These are ordinary cached blocks that empty_cache returns; the harvest does
    not hold them.
  - This GPU 2 probe's call sequence differs from the GPU 0 timing runs' (+64 / +144), so the totals differ.
- No fix proposed, since it does not grow with keys. The existing knob is _POOL_KEEP: 0 drops the pool after every
  harvest, at the cost of new segments per harvest. Releasing the pool once no harvest is pending would be about
  5 LOC; I have not written it.

Item 2, decode36's eager aten.index.Tensor (quick look, measured on src_t11c):
- The decline reason is only "aten.index.Tensor is not one of the harvested ops". The tape (_host_trace_tape.py
  ~880-945) finds no host for it:
  - it is not in _TRACED_ATEN;
  - it has no Python kernel;
  - it is not pointwise, so there is no pointwise host;
  - it is not kernel-less.
  So it records an EagerCall. With opaque=() its reason is None ("no traced implementation").
- It is a general index.Tensor gap, not specific to this proxy (scratch/g11mem/index_why.py). All of these are eager
  steps:
  - kv[pt] (2-D index, 3-D source);
  - kv[pt].flatten(1, 2);
  - kv[i] (1-D index, 3-D source);
  - h[i] (1-D index, 2-D source, the hidden_states[last_index] shape).
- Arms F and V show 0 boundaries because the gather is outside their traced entry, model.forward. scratch/vllm/PLAN.md
  item 8 respells hidden_states[logits_indices] as index_select, which has a host (_cuda_hostTraceIndexSelect).
- Fix (estimated, not written): a traced host for ATen's index kernel. It would be ht::index next to
  ht::index_select's port (native/cuda/Indexing.cu ~2091-2241, about 150 LOC), on TensorIteratorSym's index iterator,
  plus a pybind row (about 5 LOC) and a _TRACED_ATEN row (about 3 LOC). That is roughly 150-250 C++ LOC and needs a C++
  build.
- Rerouting index.Tensor to the index_select host in Python is not safe: index.Tensor wraps negative indices, while
  index_select device-asserts on them.

Item 3, G12: a provider names no reason for an op outside its set -- ready, green on int6 step146 / step145:
- Patches: gaps/G12_quiet_harvest_decline.int6.patch (odd, on step145 + G9 + G11) and .int6cpp.patch (even, on
  step146 + G9 + G11). The two have the same content: +16 -3, md5 0fed063c6945.
- Torch side, +5 -3:
  - HarvestProvider.accepts returns "" for an op outside its set;
  - the tape appends a provider's reason only if it is non-empty;
  - the OpaqueProvider.accepts docstring states the "" contract.
- An op outside every provider's set now records reason None. fullgraph names that "no traced implementation", the
  same as with opaque=() before G11. An op inside the set still names its decline.
- Test: test_an_op_outside_the_harvested_set_names_no_reason, which checks that x[i] under the default records
  (aten.index.Tensor, None) and is bitwise.
- Gates: green on both lines, all PASS (logs/g12c_suites.log, logs/g12p_suites.log). Even: python_entry 501, replay 238, harvest 219, opaque 43, memory 19, fullgraph 8, oob 3, packed 5, cover 20, tape 43, guards 9, capture 12, lower 18, lower_tape 6, triton_abi 7, program 102, triton 56, cute 61, integration 1, inductor test_cudagraph_host_trace 82, aten 1667. Odd: the same set without python_entry, with replay 177, harvest 221, opaque 37, cute 59, and the rest equal (aten 1667).

Item 4, the CuTe survey's +5 us per hit for view outputs and strided 3-D arguments (measured, GPU 0, CPU 4, src_t11c,
memory auto; scratch/viewcost/probe.py, out/viewcost/r1-r6.jsonl):
- Probe: two copy_ kernels into a fresh 256 x 256 fp16 output, one variant, 0 eager calls, checked equal to eager with
  eager's strides. Per-hit thread time is the median of 25 x 400 hits; C++ phase ns come from 2000 hits.
- Strided 3-D arguments cost nothing: with a base output, perm3 args (tma's permuted layout) give 3.62-4.07 us per hit
  and flat args 3.28-4.18 us.
- With a view output, a hit costs 7.6-11.0 us against 3.3-4.2 us with the base. A slice view (row-contiguous) costs the
  same as a permuted view (8.0-9.0 us), so the cost comes from returning a view, not from its layout.
- Of the +4.5 us, about 1.2 us is in the hit path: phase 7 (outputs) is 1175-1337 ns against 81-98 ns. That is
  HostTraceVariant::view's base.as_strided, a full dispatcher call (VariableType, then ADInplaceOrView, then as_view
  for the autograd view meta). perf, 300k hits each: as_strided is 1415 of the extra 2835 samples under entry_call.
- The other ~3.3 us is not the view. Any dispatcher op between two hits raises the next hit's C++ phases (launch,
  capture check, entry) by about the same amount. Controls, both rounds:

| between hits | hit us | phase 8 (C++ total) ns |
|---|---|---|
| nothing | 3.28 / 3.69 | 4265 / 5518 |
| Python only, no op (sum(range(40)), z.size(1)) | 3.83-5.02 | 3860-5234 |
| an unrelated z.t() (CUDA) | 7.78 / 8.04 | 6208 / 5499 |
| an unrelated zc.t() (CPU) | 8.65 / 11.02 | 6810 / 7550 |
| a base output, viewed in Python after the hit | 8.67 / 9.58 | 6290 / 6649 |
| a view output | 8.45 / 10.97 | 7647 / 8731 |

  With nothing between hits, the hit's code and data stay hot. A real decode step always has other ops between hits,
  so the base output's 3.3-4 us is the optimistic figure; for a view output the hit-path cost is the ~1.2 us of
  phase 7. The same pattern holds with a sync before every hit (sync_hit_us: base 8.5-11.7, view 11.8-17.1).
- GPU time is the same for all variants (6.15 us per hit, enqueued behind _sleep), so this is not queue
  backpressure.
- Cut options. C++ microbench, create plus destroy one view of a fresh CUDA base, measured hot on CPUs 108-143
  (scratch/viewcost/micro.py, out/viewcost/micro.json):
  - base.as_strided (now): 277 ns.
  - at::native::as_strided_tensorimpl + torch::autograd::as_view: 195 ns. Keeps _is_view(), _base identity and the
    shared version counter, all checked.
  - Under AutoDispatchBelowADInplaceOrView: 147 ns. Not a view.
  - make_tensor like the dtype branch: 103 ns. Not a view.
- The two non-view routes break eager's `_base` (test_output_views asserts `a._base is b._base`), so they are ruled
  out.
- The view-preserving route is about 10 LOC in EagerStep.cpp HostTraceVariant::view. It must be limited to a base
  this run allocated: for an argument base that requires grad, VariableType is what attaches the grad_fn.
  Estimated saving: about 80 ns hot, so roughly 0.3 us of the 1.2 us cold (scaled, not measured). Not implemented
  (needs a C++ build and a gate); worth doing only if the coordinator wants it.

## G9 (memory default "auto") + G11 (default blas harvest) -- ready, green on int6 step145/146
Patches in gaps/, against int6's current top. G11 goes on top of G9. Each pair was regenerated after a fresh apply,
so offsets are zero. The older step137/138 versions (same hunks) are in tmp/g9/base_137.
| patch | base | +/- | md5 |
|---|---|---|---|
| G9_memory_auto_default.int6.patch | snap/step145 | +18 -12 | 1e946dd899b8 |
| G9_memory_auto_default.int6cpp.patch | snap/step146 | +18 -12 | 4d4a96c965d7 |
| G11_default_blas_harvest.int6.patch | step145 + G9 | +25 -8 | 73505b8dd723 |
| G11_default_blas_harvest.int6cpp.patch | step146 + G9 | +28 -11 | 28e74cac757a |
- G9: HostTraceReplay(memory="auto") is the default. "eager" (and every other mode) stays selectable. Outputs were
  already their own caching-allocator allocations. auto still falls back to eager when the run buffer would exceed
  max(64 MiB, 5%). Torch +6 -6 (the kwarg default and docstrings in _host_trace_replay.py and _host_trace_memory.py).
  - Tests: one new default-path check in test_the_plan_follows_the_tape (the run buffer, the output its own
    allocation). memory="eager" is pinned in the 5 TestReplayMemory tests and in aten's test_host_replays_new_shapes,
    which assert eager's exact allocation pattern or peak.
- G11: opaque defaults to None, which means a HarvestProvider() of the entry's own (cuBLAS mm/addmm/bmm). Callers
  who pass opaque= keep their choice, and opaque=() disables it. Torch +5 -2.
  - Tests: new test_the_default_harvests_blas covers the default (4 harvests, 0 eager steps) and opaque=() (1 eager
    mm step), bitwise vs eager. opaque=() is pinned in 7 tests whose subject is an mm eager step: chain segments,
    planned tensors read by eager steps, TestBoundCall.entry and the library-impl bound call, and fullgraph's
    eager-step message.
  - Under the default, two things change for users:
    - Every eager step's decline reason now starts with "<op> is not one of the harvested ops; ". This was already
      true for anyone passing a HarvestProvider. A possible follow-up (not done) is to leave that reason out for ops
      outside a provider's set.
    - A bound call whose static weight changes layout now misses (a new cuBLAS key) and returns NotImplemented once,
      per the bound contract. With the mm as an eager step it used to hit.
- Inductor passes memory= and opaque= explicitly (cudagraph_host_trace.py), so neither flip changes it.
- Gates (GPU 2):
  - Even line, src_t11c (build_cpp4). PASS: python_entry 500, replay 238, harvest 218, opaque 43, memory 19,
    fullgraph 8, oob 3, packed 5, cover 20, tape 43, guards 9, capture 12, lower 18, lower_tape 6, triton_abi 7,
    program 102, triton 56, cute 61, integration 1, inductor test_cudagraph_host_trace 82, aten 1667. The first run
    failed 9 replay, 4 python_entry and 1 fullgraph test, all mm-as-eager-step tests. They pass with the opaque=()
    pins (rerun of those files plus harvest).
  - Odd line, src_t11p. The same set minus python_entry, all PASS (replay 177, harvest 220, opaque 37, cute 59).
- Conformance (conf_g9, a copy of int6/conformance_copy plus overlays for my trees, GPU 2):
  - The patched trees with the proposed rows show the same failures as the base trees with the current rows: odd
    148/13/18 option tests and cases 32/0/23; even 156/17/6 and 50/1/4.
  - replay.memory, replay.opaque, memory.auto and memory.eager are "present" on all four.
  - Without the row change, the patched even tree flips replay.memory ('auto' != 'eager') and replay.opaque
    (None != []).
  - Row proposal: gaps/G9_G11_options_rows.json.diff (against land/conformance/options.json) and
    gaps/G9_G11_options_rows.int6_copy.json.diff (against int6/conformance_copy/options.json). Changes:
    - replay.memory default "eager" -> "auto".
    - memory.eager meaning gains "selectable with memory=\"eager\"".
    - memory.auto meaning becomes HostTraceReplay's and Inductor's default.
    - replay.opaque default [] -> null, with values gaining "None" and the meaning explaining None and ().
- Measured, GPU 0, step146 + G9 + G11 (src_t11c, build_cpp4). scratch/prof_g9.py runs decode36/prefill36 (B: proxy
  unchanged, kwargs explicit): 240 hits each, fresh args, every hit bitwise vs eager. Two interleaved rounds. Host
  wall per hit in us, median (mean):
  | | eager + opaque=() (int6 today) | auto + opaque=() (G9 only) | eager + blas (G11 only) | auto + blas (G9 + G11) |
  |---|---|---|---|---|
  | decode36 | 1929 (1923) / 1645 (1678) | 1483 (1506) / 1386 (1416) | 628 (749) / 510 (548) | 124 (171) / 129 (182) |
  | prefill36 | 3985 (3975) / 3493 (3510) | 3657 (3661) / 3529 (3565) | 471 (509) / 424 (465) | 666 (562) / 436 (366) |
  - Speed: decode36 is 13-15x faster with both (1645-1929 -> 124-129 us). prefill36 is 6-9x faster on medians
    (3493-3985 -> 436-666 us), driven by G11.
  - G9 on its own (with blas) takes decode36 from 510-628 to 124-129 us. The cause is the eager order path's 540
    caching-allocator calls and about 330 kernel-node re-patches per hit.
  - prefill36 auto vs eager is within noise (medians 436-666 vs 424-471, means 366-562 vs 465-509). auto picks the
    run buffer for one of the 2 variants.
  - step126 / old-base runs agree (out/g9/s126_hit_*, g11_hit_*): decode auto 123-156 vs eager 511-603; G11 decode
    1281-1553 -> 118-129; prefill 3227-3359 -> 407-428.
  - Coverage, eager steps per hit:
    - decode36 37 -> 1. The remaining one is aten.index.Tensor (kv[page_table]), which is not a harvest key.
    - prefill36 108 -> 0.
    - No unharvestable (refused) keys showed up in either; refused is {} per variant.
  - Cold-start harvest (scratch/prof_g9_build.py on src_g11c = step126 + G9 + G11, 2 runs):
    - decode36: 13 harvests, 2 in the cover (38-51 ms) and 11 in the sizes phase (75-76 ms).
    - prefill36: 42 harvests, 6 in the cover (46-59 ms) and 36 in the sizes phase (574-591 ms, about 16 ms each).
    - Cover wall: decode 1.77-1.82 s (none) vs 1.78-1.99 s (default); prefill 1.45-1.60 vs 1.65-2.01 s (noisy).
  - G9 build cost, measured: the planner is 2.3-2.6 ms per variant under eager vs 5.3-5.8 ms under auto for decode,
    and 1.3-2.4 vs 3.9-4.1 ms for prefill. A whole _build is 77-110 ms either way. That adds about 9 ms per decode
    proxy run (3 variants) and about 5 ms per prefill run (2 variants), against 3.3 s of cover plus sizes.
    Negligible, so no fix proposed.
  - Memory, GPU 0, same runs:
    - Peak over start: decode auto 315500 vs eager reference 314887 KiB (+0.2%); prefill equal (520155).
    - Reserved, auto + blas vs eager + blas: decode 598 vs 638 MiB, prefill 858 vs 778 MiB.
    - Reserved, blas vs opaque=() under auto: decode 598 vs 534, prefill 858 vs 714.
    - Per-size sweep (GPU 2, step126, scratch/prof_g9_peaks.py): decode auto max peak ratio 1.0. prefill max 1.32
      at size 1932 (62 vs 47 MiB, inside auto's 64 MiB margin).

## G10: single-op replay hit profile (quack RMSNorm), measured, no change
scratch/g10/rms_prof.py on step116 (GPU 0), C++ entry phase timers plus perf (out/g10/rms.perf.data).
- Hit: 3.85-4.02 us CPU, of which g.replay (cudaGraphLaunch) is 2.09 us.
- C++ phases per hit, in us (the timer itself adds about 1.3 us):
  - launch 2.05
  - alloc 0.94, of which the output allocation is 0.69
  - sites 0.44
  - order/key 0.41
  - capture-status check 0.20
  - frees 0.11
  - outputs 0.09
  - plan 0.06
  - patch 0.04
- About 0.77 us is in the Python entry, outside the C++ total.
- No small, clean cut was found:
  - The capture check is needed for correctness, since a replay inside a user's capture must not launch.
  - The fresh output allocation is eager's contract.
  - A vectorcall entry might save about 0.3 us (estimate), but it is not small.

## P1: the C++ entry for int6's even line -- ships with its switches as built (no new default-OFF switch)
Files: p1/P1a_sglcpp_cpp.int6cpp.diff (md5 84ac1a54b433; the 9 host_trace C++ files + pyi defs), then core
48_native_safety_checks.sglcpp.patch (C++) and p1/48_native_safety_checks.int6cpp.step46.diff (md5 6e18b65a7980; the
48 int6cpp Python/tests rebased onto snap/step46, which the original needs fuzz 1 for). 02 is separate.
- What the even line lacks: its source (snap/step46 torch/csrc = src_cpp) is the int5b-level C++, while its tests
  already run on install_sgl_cpp (sglcpp C1-C15). P1a applied to src_cpp is byte-identical to wt-sgl-cpp, the
  source of that install, so P1 makes the even line's source match the binary its gates already use.
- Why: decode pre-launch with the C++ entry is 163-184 us vs A 220-242 and the Python entry 326-390 (CPP_INTERIM
  headline, measured by the SGLang C++ lane, not rerun here).
- Defaults: delta, variant_order, placement_reuse and entry_descriptors are ON. Two pieces are not on by default, and
  neither is new to int6: step46 already calls both (opaque test_form_switch_sets_what_the_forms_exec_lacks
  [form_held]; replay TestBoundCall `_HostTraceBound(..., trust_statics=True)`).
  - form_held (`_host_trace_form_held`, global, OFF): a measured trade-off (see "form_held" below). Removing it would
    drop an option int6's tests use, so it ships as built.
  - trust_statics (a `_HostTraceBound` constructor argument, False): a caller contract (statics change only through
    `bound.statics_changed()`), not a global switch. Unsound without 48 (K14); 48 adds the layout check under trust.
    Its cost vs the untrusted path is the next item (scratch/ubench_statics.py).
  - `_host_trace_phase_times` / `_host_trace_launch_stamps`: diagnostics, off.
- Tests: the sglcpp tests are already in step46 (TestStaticPrefix, TestPlannedMemory, TestDirtyRecords,
  TestBoundCall). Gate: src_gate (step46 + 48 int6cpp + 02 + the parked 01 tests) on install_grid (P1a + 48 + 01 C++,
  01 at cap 0 = the old path) vs src_base (step46) on install_sgl_cpp; logs/gate_cap0.log, logs/gate_base.log.

## P1b + gaps 6 and 3: C++ entry default with a Python option; slow_calls; kwargs and constants in C++
Files in gaps/, applied in this order, all dry-run clean on int6/work_cpp (Python) and int6/build_cpp/src (C++):
- P1_cpp_entry_switch.py.diff (md5 d7c8c1c032fc). Switch `torch.cuda._host_trace.cpp_entry` (default True), read at
  construction; False = HostTraceReplay's Python dispatch serves every call (kept as an option, per the user). Also
  moves the "inside an outer trace, run fn inline" check to the top of `_call_slow`, which the Python path needed.
- G63_slow_calls_kwargs_constants.cpp.diff (md5 0abd5ab635dd; Entry.cpp + pyi) and .py.diff (md5 081c5095b588;
  replay.py, tests). The Python half needs the C++ half (6th `_native_init` argument, `slow_calls`).
  - Gap 6: `entry.slow_calls` counts calls the C++ entry hands to Python (`_call_slow` or `_unreplayed`). The parity
    tests assert it is 0 in the steady state.
  - Gap 3 kwargs: `_native_init` gets fn's positional parameter names; a call whose keywords name exactly the next
    parameters is bound positionally in C++ and dispatched as such. Any other keyword use (keyword-only, gaps) goes
    to Python, as before.
  - Gap 3 constants: a non-tensor argument the Python contract keys by value (str, list/tuple elementwise, dtype,
    float/complex by bits, ...) gets a Constant tag in the C++ key, interned per entry at registration and compared
    by value at dispatch. The C++ partition can be finer than Python's, never coarser.
- Tests: TestEntryParity (test_cuda_host_trace_replay.py) runs each case on both paths and asserts bitwise vs eager,
  equal (traces, replays, eager) and the cpp path's slow_calls: steady state 0 through Python; kwargs; 5 constant
  kinds; a changed list; keyword-only stays eager on both. New test_cuda_host_trace_python_entry.py reruns the replay,
  opaque and harvest suites with cpp_entry=False, skipping the tests of C++-only features (static_prefix,
  _HostTraceBound, dirty records, a no-Python hit, native keys) and 6 TestReplayMemory tests: the Python dispatch
  holds its arguments during the call, so a freed or boxed argument dies after the replay, a known limit of that path.
- Measured gates (GPU 2, install_grid = P1a + 48 + 01 C++ (01 at cap 0) + G63 C++, logs/par/): replay 221 OK, python
  entry 475 OK (58 skipped), opaque 42, harvest 202, fullgraph 8, cover 11, inductor/test_cudagraph_host_trace 79
  OK. The Python path on the pre-G63 tree also passes aten 1666 OK, cover 11, fullgraph 8, integration 1
  (logs/entry_python_all.log). Before G63: P1 gate cap 0 and base all pass (logs/gate_cap0*.log, gate_base*.log).
- Not done here: gaps 2 and 4 (learners, refused keys) are the core lane's patch 56. The reviewer's P9: work_core's
  C++ predates P1a, so it needs P1a's C++ before the step45/step46 lines merge (passed to int6).

## Gap 5 (graphsafe RNG and HOST_SEED_OFFSET eager steps in C++) -- ready, default ON (pure improvement)
gaps/G5_eager_step_rng.cpp.diff (md5 53bb35387aea; EagerStep.cpp, VariantBuild.cpp, Variant.h) and
gaps/G5_eager_step_rng.py.diff (f452a43c9f84; _host_trace_native._boxed + tests). The two land together: the new
_boxed tuple has 8 entries the old C++ cannot read, and the old C++ would run a graphsafe step on the default
generator. Stacks after G63 (same build). No switch: it is the boxed step, now also for these two cases.
- `_boxed` no longer declines a step with a generator or a HOST_SEED_OFFSET op. The boxed step swaps the generator
  state into its device's default generator around the call (as _impl_graphsafe_rng, restored on throw) and moves
  the CPU seed/offset outputs to the device non_blocking (as seed_offset_on_device).
- Tests (TestNativeReplay; C++ entry only, listed in python_entry's _CPP_ONLY):
  test_a_graphsafe_rng_step_runs_no_python, test_a_host_seed_offset_step_runs_no_python. Each runs the boxed
  step and the Python step (_boxed -> None): equal (traces, replays, eager) = (1, 3, 0), outputs bitwise across paths
  and vs eager from the same generator offset, the same draw from the generator, no Python frame at a hit,
  python_calls 0 vs one per replay. Graphsafe: the default generator's offset is unchanged.
- Suites on install_grid + src_par, cpp entry, GPU 2 (logs/par_gap5_suites.log): replay 223 OK, python_entry 477 OK,
  opaque 42, harvest 202, fullgraph 8, cover 11, triton 50, aten 1666, inductor/test_cudagraph_host_trace 79 OK.
- Measured, GPU 0, host us per replay, median, 8 interleaved rounds x 300 calls, two runs (scratch/ubench_gap5.py,
  out/ubench/gap5.json): dropout under graphsafe RNG + 2 Triton kernels 24.1 / 20.5 (boxed) vs 79.0 / 71.3
  (Python step), 3.3-3.5x; efficient attention with dropout + 2 Triton kernels 43.0 / 34.8 vs 124.8 / 114.0,
  2.9-3.3x. Memory unchanged (the same op calls).

## G7 (Tensor?[] arguments in C++ eager steps) -- ready, default ON (pure improvement)
gaps/G7_optional_tensor_list.{cpp,py}.diff, applied on top of G5. G5 then G7 apply to copies of int6/src_cpp with
offsets only. The py diff's python_entry hunk needs G63's new test file. boxed_step now takes a list of optional
Tensors (ArgKind::OptionalTensorList): leaf items plus None items. EagerStep builds a c10::List<std::optional<Tensor>>.
After this, decode36's kv[page_table] (aten.index.Tensor) runs as a boxed C++ step: python_steps 0, python_calls 0.
- LOC: cpp +21 -1 (net +20), tests +8 (net +8), total +29 -1 = net +28.
- Test: TestNativeReplay.test_an_index_with_a_none_runs_no_python. It indexes with [None, idx], runs through
  _boxed_parity (python_steps 0, python_calls 0, no Python frames on a hit, bitwise vs eager and vs the Python
  step), and is C++-only in the python_entry file. It fails on the pre-G7 build (25 Python frames on a hit).
- Suites (install_grid + src_par, C++ entry, GPU 2, logs/par_g7_suites.log), all OK:
  replay 224, python_entry 478, opaque 42, harvest 202, fullgraph 8, cover 11, triton 50, aten 1666, inductor 79.
- decode36 median hit host us, GPU 0 (measured):
  - In one process (scratch/ab_g7_inproc.py, out/ab_g7/inproc*): entry A declines only index.Tensor's boxed form
    (the pre-G7 path), entry B is G7. Same args, A/B order flips each call, 16 sizes x 60 calls.
    - Device idle before each call: A 2081 / B 1964 and A 2031 / B 1743. Paired difference: median 120 / 292,
      mean 137 +- 4.5 / 355 +- 7.4.
    - Without the idle: A 2252 / B 1952 and A 2149 / B 1926. Paired difference: median 201 / 130.
  - Across builds (scratch/ab_g7.sh, install_g5 vs install_grid, ABAB, out/ab_g7/decode*), the difference is
    within run-to-run noise (+-100-150 us):
    - Random sizes, 4 rounds: g5 1748/1634/1633/1715, G7 1743/1679/1634/1874.
    - The 4 points, idle device, 4 rounds: g5 1299/1324/1516/1435, G7 1317/1298/1400/1263.
  - The saving is larger than the 68 us estimate from the closure time. That estimate left out the C++ side of
    a Python step: argument tuple, call and output unwrapping.
- Seen while measuring: each decode36 hit has 37 eager steps. A copy of the hit with every boxed form declined
  has python_steps 37. The other 36 (one per layer) were already boxed C++ steps, so the hit's host time (1.6-2.2 ms)
  is mostly those 36 eager steps between graph segments, not the replays. See "decode36's 36 per-layer eager steps".

## decode36's 36 per-layer eager steps (diagnosis, coordinator ask) + G8 (respec beside a bound site) -- G8 ready, default ON (bug fix)
(1) What they are. Each one is aten.mm.default (`(p[..., None] * k).sum(1) @ w`, [bs,64]x[64,64] fp32), one per
layer. scratch/probe_decode_steps.py lists the variant's eager steps. The reason is "no traced implementation":
mm is a closed cuBLAS op, and no opaque provider was given.
- Site: _host_trace_tape.py ~878 tries `self.opaque` first. It is empty because HostTraceReplay's `opaque` defaults
  to () (_host_trace_replay.py ~455). mm is not in _TRACED_ATEN (~2125). It then falls to
  `EagerCall(func, ..., why=None)` (~1026-1028).
- The 37th step is index.Tensor (kv[page_table]); G7 made it a boxed C++ step.
- This is the proxy's configuration, not a core refusal. next/rec_startup.py:52 builds `HostTraceReplay(f)` with no
  provider. The core's route for cuBLAS is HarvestProvider "blas", which is opt-in. Proxy not changed.
(2) Host split per decode hit, GPU 0, measured. scratch/prof_decode_mm.py, out/decode_mm/gpu0*.json: one process,
A = proxy config, B = opaque=(HarvestProvider(),), paired, device synced before each call, 16 sizes x 40 calls,
every timed call a hit, 0 Python steps and 0 slow calls on hits. Mean us from _host_trace_phase_times:
| | wall median | eager steps | alloc | patch | graph launch | frees | other |
|---|---|---|---|---|---|---|---|
| A (proxy) | 2280 | 1027 (37 steps) | 450 | 522 | 237 (37 segments) | 71 | 35 |
| B (blas harvest) | 646 | 79 (1 step) | 231 | 254 | 42 (1 graph) | 7.5 | 36 |
- Handoff to Python: 0 in both.
- Paired A-B: median 1622 us, mean 1691 +- 20 (phases on). With phases off, median 1247 us, mean 1210 +- 13; B wall
  735 vs A 1979 (2.7x).
- Staying eager costs about 1.0 ms of dispatch (about 28 us per mm step, op included). It also costs about 0.65 ms
  of segment overhead: 36 extra segment launches plus the alloc/patch/frees they split.
(3) Proposal: configure the proxy with opaque=(HarvestProvider(),). This is 0 core LOC; it puts all 36 mm steps in the
one graph as keyed sites. HarvestProvider stays opt-in in the core: a new key costs a harvest/trace (functionality
and miss-cost change), so it is not default ON.
- Miss cost, measured (out/decode_mm/miss_A.json, miss_B_g8_gpu2.json, first call at 12 unseen bs):
  - A: about 190 ms first call (a hit).
  - B: 190-950 ms first call (3 traces in 12), 10 variants vs 3.
  - Reserved memory the same (+568 MiB both).
  - B's later calls sometimes take about 150 ms on a hit (a missing key learned in _fill). Seen on GPU 2 while the
    suites shared it, not timed on GPU 0.
- Blocker found and fixed (G8): before G8, B crashed on decode36's 4-calls-per-size miss run (KeyError in
  lower_tape's pointer, on src_par and int6 build_cpp2). With G8, all 48 calls in that run match eager bitwise.
- Next host cost in B: alloc 231 + patch 254 us per hit (memory="eager").
G8: gaps/G8_respec_beside_bound_site.py.diff (v1, md5 86da244d5861, unchanged; int6 merged only its test, which passes through core 56's rebind). Python only; no C++ build. Two bugs, both on respec of a
variant whose tape has a bound opaque call (a keyed site):
- bind_opaque gave a converted call's fresh output allocations to the op before it: an EagerCall op has an empty alloc
  range, and the spans were cut at old allocation indices. A respec of that earlier op (a split sum) then dropped
  the mm's output allocation, and lower_tape raised KeyError. Fix: an op's alloc span is its own old allocations plus
  its converted calls' new ones (_host_trace_tape.py bind_opaque).
- respec moved the tape to the call's args. The built variant then seeded each site's table with the call's key mapped
  to nodes recorded for the traced key, so at a new bs the mm ran the traced bs's cuBLAS binding: wrong output, no
  error. Fix: LoweredKeyedSite records its rows at the tape's call (`traced`). respec refuses ("key at the call is not
  its traced key") when a site's key differs, and the call traces instead. A same-key respec (new S, same bs) still
  respecs.
- LOC: core +13 -3, test +17. Total +30 -3 = net +27.
- Test: TestHostTraceHarvest.test_a_respec_beside_a_bound_mm[new_key False/True], sum(1) @ w + 1 with
  HarvestProvider, bitwise vs eager, counts (traces, respecs, refusals, eager). Before G8: KeyError in both. With
  only the span fix: new_key=True fails bitwise. Passes on the python and cpp entries.
- Suites (install_grid + src_par, C++ entry, GPU 2, logs/par_g8_suites.log), all OK: harvest 204, replay 224,
  python_entry 480, opaque 42, fullgraph 8, cover 11, triton 50, cute 59, aten 1666, inductor 79.

## B miss path after core 56 + 66 (G8b dropped; coordinator asks 2026-10-04/05)
All on GPU 0. B = decode36 with opaque=(HarvestProvider(),), passed from the harness (proxy not changed).
scratch/miss_decode_mm3.py: 4 record points, then 12 unseen (bs, S), 4 calls each, every call bitwise vs eager.
- G8b (re-bind a keyed site in respec, option respec_rebinds_sites) is dropped: core 56 (int6 steps 75/76, default ON)
  already rebinds every keyed site in respec, in the variant's own env (no rename, no xreplace). No new option. The
  old combined diff is kept only for reference at tmp/g8v3d/G8_v2_combined_srcpar.diff (not for merging).
- Trees: src_s109 (= int6/snap/step109: 56, 63-68, oob_learn ON) and src_g8b (= w1/s68, also has 56, oob_learn ON).
  "v1 emulation" wraps _respec to refuse when a site's key differs (G8 v1's core), so the call traces instead.
- Measured, step109, interleaved, 2 runs (out/g8b/gpu0_B{,_norebind}_s109_{1,2}.json):
  | | bs 51 first call | first calls summed | 48 calls summed | first-call median / max | traces | variants | op re-runs (redispatch + respec) | reserved |
  |---|---|---|---|---|---|---|---|---|
  | 56 rebind (as is) | 614 / 586 ms (respec) | 1702 / 1658 ms | 1855 / 1809 ms | 78-82 / 695 ms | 2 | 3 | 290 (218 + 72) | +632 MiB |
  | v1 emulation | 688 / 671 ms (trace) | 1769 / 1733 ms | 1922 / 1885 ms | 78-82 / 688 ms | 3 | 3 | 218 (218 + 0) | +632 MiB |
  - s68 (out/g8b/*_s68_*): 56 bs 51 570 / 611 ms vs v1 774 / 693 ms; first calls summed 1628 / 1722 vs 1958 / 1762 ms.
  - Before 56 + 66 (src_par, old G8 v2 ON): bs 51 respec 3013 ms, about 33 ms per respec op re-run (sympy xreplace).
- Respec at bs 51 is now about 80 ms (12%) faster than a trace, so 66 did remove the xreplace. Where its 0.6 s goes
  (step109): 72 respec op re-runs 0.24-0.26 s (about 3.4 ms each), 108 splices 0.20 s (about 1.9 ms each, each
  O(tape)), 36 rebinds 0.09 s, build about 0.1 s, 2 bind_opaque 0.17 s.
- Possible follow-ups (not done, not asked; estimates): splice the 108 ops in one pass (about -0.2 s); skip the rebind
  of a site whose key did not change (about -0.09 s). The 1 remaining new-size trace is (10, 35), the pages=1 guard.
- Earlier, on src_par (pre-56, oob_learn OFF), why 3 of 12 new sizes traced (scratch/why_trace.py lists each variant's false guards at a traced call):
  - (51, 465): the G8 refusal. Respecs through 56 now.
  - (11, 6220): no false graph guards, but every variant was a learning variant. Learning variants are not redispatch
    candidates (redispatch_learners False), so the call traces.
  - (10, 35): `Ne(s57, 1)` (page_table.size(1) == 1 specialization) is false in every variant.
- src_par with oob_learn=True (now default ON in int6), out/g8v2/gpu0_B_{,norebind_}oob.json:
  - G8 v1: 3 traces, 3 variants, first calls summed 2372 ms, 48 calls summed 2521 ms, bs 51 709 ms.
  - old G8 v2: 2 traces (1 of the 12: the pages=1 call), 3 variants, 0 relowers, no slow hits, bs 51 2884 ms (respec, pre-66).
  - The remaining 190-220 ms first calls (bs 39/57/54) are redispatch: about 5 ms per re-run op.
(2) The ~150 ms hits (GPU 0 alone, reproduced). They are `_relower` of a learning variant: bind_opaque + `_build`
(about 110 ms for the build). It happens when the variant's mm key becomes bound (`final`). In every case it is aten.mm
[[bs,64],[64,64],[bs,64]] fp32 (the 36 per-layer sites share one key):
- bs 11 and bs 10, call 3: call 1 traced a learning variant, and call 2's eager run harvested the key. Call 3 finds
  it bound and relowers.
- bs 60, call 2: relowers the learning variant traced at the (6, 5000) record point. bind_opaque `_learn`s the bs 6
  key (2.5 ms) at that variant's traced call. It was never harvested because that variant's only call at bs 6 was its
  trace, and a trace does not learn keys unless oob_learn is on.
- bs 51, call 3 (OFF only): same pattern.
- Fix: oob_learn=True removed every slow hit in the run (above); it is default ON since int6 step 75.
(3) Per-hit alloc/patch in B (scratch/prof_alloc_patch.py, 240 hits, fresh args each call, all hits, bitwise;
out/g8v2/alloc_patch_*.json). Phase means in us:
| memory= | wall median | alloc | patch | launch | node sets / hit | peak vs eager | reserved |
|---|---|---|---|---|---|---|---|
| "eager" (default) | 526 | 208-210 | 214 | 37 | 330.8 kernel + 30 memset | 313625 vs 315590 KiB | 574 MiB |
| "planned" | 120.5 | 7.9 | 48.7 | 15.3 | 23.6 kernel + 30 memset | 316150 vs 314887 KiB (+0.4%) | 538 MiB |
| "auto" | 113.4 | 12.4 | 46.2 | 13.8 | -- | 315500 vs 314887 KiB (+0.2%) | 534 MiB |
- alloc under "eager": the eager-order path makes 540 caching-allocator allocations per variant (one per temporary),
  each allocate/free pair about 0.19 us under the device allocator lock. Plan and the unordered paths are about 0.5 us.
- patch: those addresses move with the pool's state, so about 331 kernel node sets plus 30 memset sets per hit, about
  0.6 us each. With the same args it is still 172 us (234 sets).
- Proposal: run B with memory="auto" (existing option): about 4.6x lower host time per hit (526 -> 113 us); peak
  within 0.2% of eager here, reserved lower. Default stays the coordinator's call, since it changes the memory layout.
  auto falls back to "eager" itself when a run buffer's peak exceeds max(64 MiB, 5%).

## What still goes to Python on a hit after G63 + G5 (measured only, nothing fixed)
scratch/handoffs.py on next/rec_startup.py's proxies (36 layers): record points, then 16 (decode) / 27 (prefill)
sampled sizes, 2 warm-up calls each, then 30 timed hits each. Counts: entry slow_calls per hit, and each Python eager
step run inside a C++ replay (by target). out/handoffs/. Every timed call was a hit (0 traces, 0 eager).
- prefill36 (with and without the graph cover): 0 slow calls, 0 Python steps per hit. Hit host median 4.5 ms (GPU 2).
- decode36 (warm-up variants; src_par's cover lacks ArgsAt(sizes=) for the page table): 0 slow calls per hit, one
  Python eager step per hit, aten.index.Tensor (kv[page_table]), 86.1 us each (GPU 0; includes the op itself, 18 us
  as a bare call). Hit host median 1608 us. Cause: boxed_step (VariantBuild.cpp) declines `Tensor?[] indices` (the
  list element is Optional[Tensor], and a None item is not an int constant). Fix: fix (on the benchmark path, cheap):
  about 15 C++ LOC; estimated saving about 68 us per decode hit (about 4%).
- ubench_decode (bound call): 0 slow calls on hits, 0 Python steps.
- The suites (census, scratch/with_slow_census.py): the C++ entry hands calls to Python only on the miss path, and
  `_call_held` then runs a variant through `_call_native`. That happens in replay (64), opaque (196), harvest
  (399), fullgraph (7), cover (9) and triton (14). The causes are redispatch/respec after a kernel-choice change,
  learning variants (out-of-band and background learn), keys past 64 at a keyed site going to harvest, and binding
  switches. None of these is on the benchmark hit path, so they are "leave on Python (rare or complicated)".
  They do block deleting `_call_held` (the learn/relower/respec dispatch). The hit-serving Python dispatch
  (cpp_entry=False) is not used by any benchmark hit.

## Gap 1 (Inductor statics as a static prefix) -- coded, end of queue (coordinator: Inductor speedups already good)
gaps/G1_inductor_static_prefix.py.diff (md5 e965bf7369a5), Python only, after P1 + G63. `_Installation` puts the
static inputs (a backward: primals only, since its saved tensors are new objects at every call) and the constants first
and passes `static_prefix`; the traced fn un-permutes. Test test_static_inputs_lead_as_a_static_prefix runs both
paths: equal counts (1, 4, 1), outputs bitwise across paths, cpp static_hits 3. Inductor suite 79 OK.
- Measured, GPU 0, scratch/ubench_inductor_statics.py (Inductor MLP stack, hidden 512, batch 4, 2L params), one
  process per arm, 4 interleaved rounds x 3000 calls (logs/gap1/): no measurable gain. L32: host median per call
  A 82.3-88.0 vs B 83.2-87.2 us, wall 145.6 us both (static_hits 3000 in B). L128: host 843-844 us both, wall
  851 us both (GPU-bound). Not handed to int6: no shown improvement, and the coordinator put gap 1 last.

## K14 check cost (core 48 trusted-statics check) -- measured; checked trust is no faster than untrusted
scratch/ubench_statics.py: a `_HostTraceBound` over N static 4096-float weights (decode-like prefix) and 2 Triton
kernels, host us per bound call, 12 rounds x 200 calls, mode order rotated per round. Modes: T1 trust + 48 check, T0
trust unchecked (pre-48, unsound), U untrusted (same_statics). --cold fills 192 MiB of host memory before each call.
Outputs bitwise, static hits == calls in every mode, eager unchanged. GPU 0, install_grid (P1a + 48 C++),
out/ubench/statics{,_cold}.json. Measured, median us per call:

  | statics | warm T1 / T0 / U | cold T1 / T0 / U |
  |---|---|---|
  | 64   | 6.9 / 6.1 / 7.1   | 134.8 / 125.2 / 131.1 |
  | 400  | 11.1 / 6.3 / 11.2 | 211.5 / 134.3 / 183.5 |
  | 1000 | 23.7 / 6.7 / 20.4 | 269.3 / 136.8 / 226.2 |

- The check is material: at 400 statics (about the SGLang decode prefix, 376) it costs 4.9 us warm and 77 us cold over
  unchecked trust; the decode pre-launch is 163-184 us.
- Checked trust is never faster than the untrusted path: equal warm at 64-400, slower at 1000 (+3.2 us warm) and cold
  (+4 to +43 us). With the check on, trust changes only which check runs (Entry.cpp: `trusted_statics ||
  same_statics(...)`), and same_static_layout lacks the 8-ahead prefetch that same_statics has.
- No cheaper sound check is available: TensorImpl has no layout generation counter, and the version counter is bumped
  by data writes and not by `.data =`. Adding a counter means bumping it in every metadata setter in c10 (set_,
  as_strided_, t_, resize_, set_data, swap_tensors, storage changes), a core change.
- Proposal (not applied): with the check on, run same_statics for a trusted hit and delete same_static_layout (less
  code; T1 = U by construction). Then trust_statics is a speedup only unchecked, which is unsound, so adapters
  should not set it. Next: the untrusted floor (U - T0 = 14 us warm, 89 us cold at 1000) is the miss chain
  PyObject -> TensorImpl -> StorageImpl per static. A prefetch distance sweep (8 today) is the one cheap lever left.

## 01_grid_execs (ISSUES P9 item 1, P6) -- parked, not merged (default OFF; memory trade-off)
Files: 01_grid_execs.cpp.diff (driver_api.h, __init__.pyi.in, host_trace/Entry.cpp, Patch.cpp, Sites.cpp, Variant.h),
01_grid_execs.tests.diff.
- Switch: `torch._C._host_trace_grid_execs(cap) -> (old_cap, instantiations, switches, instantiate_ns)`. cap 0 is
  off (default, the old path); cap < 0 only queries. Global, in Patch.cpp (`grid_exec_cap`).
  `torch._C._host_trace_grid_execs_frozen(frozen) -> (was, misses)`: frozen, no exec is instantiated (after
  warm-up); a configuration with no exec re-grids the form's current exec in place and counts a miss. -1 queries.
- Problem: one exec per form. A launch-configuration change (grid, block, smem or function of a kernel node,
  e.g. a bs change) re-grids the uploaded exec in place with cuGraphExecKernelNodeSetParams, every node whose
  configuration changed, on every switch. On the host that is about 0.7-0.8 us per changed node per call. The
  e2e GPU penalty (bs8: 476 -> 280 early starts after bs 64/13 churn, +180 us) did not reproduce in any synthetic
  chain (scratch/pdl_regrid.py: in-place, churn, template + fresh instantiate and cuGraphExecUpdate all keep 191/191
  early starts at equal replay time), so the measured win here is on the host.
- Fix: a segment keeps up to `cap` execs, counted across all its forms (the adopted traced exec included), keyed by the launch configuration of the form's kernel nodes
  (Patch.cpp launch_config / select_grid). A new configuration under the cap gets its own exec: the frame's
  configuration is set on the graph (cuGraphKernelNodeSetParams) and instantiated and uploaded (instantiate_grid).
  The exec of the traced configuration is adopted, not rebuilt. A switch then sets no configuration in an exec,
  only the parameters that differ (switch_grid marks them dirty). Past the cap the form's current exec is re-gridded
  in place as before. The cache fills first-come with no eviction. Execs are built at the first replay at a new configuration, on the calling thread, so with a
  warm-up over the served sizes they are all built at warm-up. There is no background instantiate and so no
  stall of another launch.
- Microbench (scratch/ubench_grid.py: a host-traced chain of Triton PDL kernels, half with a bs-dependent grid,
  alternating bs 8/13/64 per call, cap 0 vs 8 interleaved over 40 rounds; GPU 0; out/ubench/grid_k{48,450}.json).
  Measured:

  | kernels | bs | host us/call cap0 -> cap8 | GPU ms (events) cap0 -> cap8 | early starts |
  |---|---|---|---|---|
  | 450 | 8  | 388 -> 342 | 1.084 -> 1.037 | 448/449 both |
  | 450 | 13 | 419 -> 236 | 1.325 -> 1.142 | 448/449 both |
  | 450 | 64 | 364 -> 201 | 3.338 -> 3.162 | 398 -> 403 of 449 |
  | 48  | 13 | 47.6 -> 32.5 | 0.147 -> 0.130 | 46/47 both |
  | 48  | 64 | 45.2 -> 29.4 | 0.364 -> 0.346 | 39/47 both |

  The GPU event time includes the launch gap of a host-bound call, so most of the GPU drop is the host drop.
  Instantiations during the churn: 0 (all built at the first sight of each bs). The measured win comes mostly
  from per-size reuse in a 2-3 size alternation; a stream of sizes that rarely repeat is unmeasured. Outputs bitwise vs the eager
  chain; traces 1, eager 1 (the entry's first call), variants 1.
- Cost, measured: per new configuration, instantiate + upload 1.0-1.1 ms at 450 kernels (0.17-0.21 ms at 48,
  2.4-3.0 ms at 1000) on the first replay at that configuration, and 2 MiB of exec memory (cudaMemGetInfo;
  4 MiB at 1000 kernels). At cap 8: at most 7 extra execs per segment, about 14 MiB per 450-node segment.
  That memory is why the default is OFF.
- Tests: test_a_launch_configuration_gets_its_own_exec[cap 0,1,4] (replay; instantiations 0/1/2, switches iff the
  cap is on) and test_binding_switches_with_grid_execs[cap 0,2,8] (opaque: forms x sizes alternating, bitwise,
  one binding, eager 0); test_frozen_grid_execs_instantiate_nothing_after_warm_up[frozen F,T] (0 instantiations
  and one miss per replay at the unwarmed grid when frozen). Off path: every existing test at cap 0.
- Gate (install_grid = src_grid built, src_py tests, GPU 2): cap 0, all files OK (replay 156, opaque 38, capture
  12, cute 45, tape, program, lower, guards, memory, packed, triton), except harvest test_background. That test
  also fails without the patch (install_sgl_cpp, logs/tests_edges.log) and with either keep_programmatic_edges
  value; it is timing dependent (12/32, 7/37, 1/43 replay/eager splits across runs). With the whole process at
  cap 8 (scratch/with_cap.py) three tests that count node sets on the single-exec path fail as expected:
  replay test_unchanged_records_are_not_packed, test_an_error_after_a_setter_leaves_the_node_unknown, and opaque
  test_form_switch_sets_what_the_forms_exec_lacks[form_held False]. With grid execs a configuration switch
  selects an exec and sets nothing. They are tests of the cap-0 path. Logs: logs/tests_grid_cap{0,8}.log.
- Open: the distinct-configuration count over SGLang decode bs 1..130 (scratch/regrid_drive.py --sweep 1-130 is
  ready; SGLang runs wait for the measurement checkpoint). Bounds today: the per-segment cap and frozen mode; no
  process-wide memory budget (one was prototyped and dropped with the default-ON plan).

## 02_keep_programmatic_edges (ISSUES P9 item 2) -- default ON (pure improvement)
File: 02_keep_programmatic_edges.diff (torch/cuda/_host_trace_capture.py, one line plus its comment).
- Switch: `torch.cuda._host_trace_capture.keep_programmatic_edges`, False -> True. False is still supported.
- Problem: instantiate_form puts each spliced piece (an opaque binding replacing a chain, e.g. split-K GEMM
  pieces) on default edges. A programmatic edge into or out of the piece is dropped, so the piece's first
  kernel cannot start early and neither can the kernel after it, though eager launches both with PDL. The
  default False came from the int6 step 7 merge, to keep int5b behaviour; MERGE_LOG records no failure with True.
- Fix: default True. instantiate_form already keeps the edge into a piece only where the binding's first kernel
  carries the PDL attribute and is not a memset, and the edge out where the piece's last node is not a memset.
  That is eager's launch structure.
- Microbench (scratch/ubench_piece_pdl.py: a captured chain of Triton PDL kernels, every 2nd kernel spliced as a
  piece of the same kernel, so only the edges differ; plain capture vs pieces with edges dropped vs kept; GPU 2
  smoke run, 64 kernels, 32 pieces): early starts 45 plain, 31 dropped, 46 kept; 286.5 / 288.7 / 286.6 us per
  replay. GPU 0, 128 kernels, median of 20 interleaved rounds x 50 replays (out/ubench/piece_n*_e*.json), measured:

  | elements/kernel | pieces | us/replay plain / dropped / kept | early starts plain / dropped / kept |
  |---|---|---|---|
  | 65536 (decode-sized, ~1 us kernels) | 64 | 120.8 / 143.5 / 118.9 | 126 / 66 / 126 |
  | 65536 | 16 | 120.7 / 127.0 / 120.8 | 126 / 111 / 126 |
  | 524288 | 64 | 565.4 / 571.5 / 565.1 | 101 / 64 / 107 |
  | 524288 | 16 | 565.4 / 573.9 / 565.4 | 103 / 82 / 102 |

  Dropping the edges costs about 0.35-0.4 us per piece with short kernels (+19% at 64 pieces); kept equals the
  plain capture.
- Tests: capture test_a_piece_keeps_a_programmatic_edge_behind_the_switch covers both values (the formed node's
  attribute and the output). Gate with True (install_sgl_cpp, GPU 2, logs/tests_edges.log): replay 153, cute 45,
  tape 35, program 102, lower 18, guards 6, memory 19, packed 5, aten 1665, triton 38, capture 12, opaque 35 OK;
  harvest test_background fails as without it; test_persist failed once in that run and passes on rerun with
  both values.

## form_held (ISSUES P6) -- not ready: stays default OFF (measured trade-off, not a pure improvement)
No diff. Candidate 03 (flip `form_held_enabled` to true) withdrawn: scratch/03_form_held_default_on.withdrawn.diff.
- scratch/ubench_formheld.py (64 x mul_chain, opaque TableProvider "mixed", forms x sizes alternating per call,
  4 modes interleaved over 100 rounds, GPU 0, out/ubench/formheld_{held,planned}_r100*.json). Measured, host us per
  call off -> form_held: memory=held 100.2 -> 95.7 (106.9 -> 100.7 on a rerun), kernel node sets 161 -> 98;
  memory=planned 110.5 -> 122.7 (157.0 -> 169.9 on a loaded rerun), sets 192.5 -> 160.8. Single-mode perf
  profiles (out/perf/fhL{0,1}.data, planned, 120k calls): swap_held + kernel_row add about the samples that the
  avoided cuGraphExecKernelNodeSetParams save in libcuda. swap_held saves and restores every kernel record's held
  image on each switch, so it costs O(records) while it saves only records whose parameters repeat across the
  form's visits: a win when addresses are stable, a loss when they move per call.
- Grid execs (01, cap 8) dominate both: held 79.1 us / 16.6 sets, planned 103.6 us / 128 sets, and the keyed path
  bypasses form_held. Next step if P6 matters without grid execs: keep the held images per form in place
  (index the row's held buffer by form) so a switch costs O(1), and measure again.
- Gate with form_held=1 (logs/tests_formheld1.log): all files OK except harvest test_background (pre-existing).
