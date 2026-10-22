# Design note: the learn transient (d), learn stand-ins (f), per-shape planned arenas (e)

Lane mem_eager, 2026-10-14. Base: step332 (+ option B in progress). The snapshots are the gh200-a lane's
(land/scratch/talk/suites_mem/data/out/20261014-snap*, c4_290, so line numbers below are step290's).
Scripts: land/core/mem_eager/scratch/free_at_peak.py, peak_wo_harvest_cache.py.
Every "estimate" below is a counterfactual computed from the recorded device trace, not a run. The before/after needs
gh200-a reruns of the patched tree.

## (d) What the conv-net peak actually is: verify's working set, not captures or plugs

Snapshot attribution at the whole-run reserved peak, ours eager:

| model | reserved peak | default peak | free in reserved at the peak | of which: segments verify made (default pool, idle) |
|---|---:|---:|---:|---:|
| mobilenetv3_large_100 | 7496 MiB | 3184 MiB | 6587 MiB | 5296 MiB (checked: same / fresh_outputs / blocks) |
| inception_v3 | 3820 MiB | 2330 MiB | 2844 MiB | 2038 MiB |

- The peak event is a harvest capture allocating in the private pool on a harvest stream. The free bytes are the default
  pool's segments, which an earlier key's verify allocated, freed and left cached. A private pool cannot use them.
- Plugs and capture outputs are small at the peak: 2 x 1 MiB plugs and a 220 MiB capture buffer. The finally in _harvest
  already resets the graphs and frees plugs and kept outputs after every key, and drops the pool above _POOL_KEEP.
  - So (d1) as worded (trim captures and plugs) does not touch the 5.3 GB.
- With verify's idle segments counted as returned (peak_wo_harvest_cache.py), the peak is still 5672 MiB (mobilenetv3)
  and 2992 MiB (inception). That moment is inside verify's own eager self-check, with harvest bytes live:
  - mobilenetv3: ref_out 784 + the second eager run 784 + same()'s two clones 1568 + torch.equal's bool tensor 784
    + _learn's probe output 784 + refs blocks 196 MiB = 4.9 GB, about 6 output-sized buffers at once.
  - inception: the same pattern at 337 MiB per buffer.

Proposals (d'), cheapest first:
1. same(a, b) without copies: torch.equal(a.view(_BITS[n]), b.view(_BITS[n])) on the same-itemsize integer view. That
   view works for any strides, is NaN-safe and bitwise, and needs no contiguous clone.
   - Removes the 2 clones, and cuts equal's temporary to 1/8 of the output.
   - Estimate: mobilenetv3 -2.3 GB, inception -0.95 GB at verify's moment.
2. Run verify's tensors (refs, outputs, scratch) from the harvest pool (use_mem_pool(dev.pool)).
   - With option B's per-harvest pool release, they go back to the device after each key, instead of sitting idle in the
     default pool while the next key's captures grow the private pool.
   - Removes the 5.3 GB idle term (mobilenetv3), 2.0 GB (inception).
   - Cost: a cudaMalloc per harvest (the note measured +1-3 ms per harvest).
3. Free _learn's eager probe outputs before the captures where only their metadata is used (conv, attention), and keep
   them for in-place GEMMs, whose verify writes into them. -784 / -337 MiB.

- Estimate with 1-3: mobilenetv3 5672 - 2254 - 784 = about 2630 MiB vs default 3184; inception 2992 - 970 - 337 = about 1690 MiB
  vs default 2330.
  - The peak moves to max(default's own peak, model live + one key's reduced verify set).
- Risk: (1) is a pure rewrite of a comparison. (2) changes which pool verify's mirrors-adjacent scratch comes from: the
  mirrors are cuMem-mapped by the harvest itself, so they are untouched.
- learnfast batchlearn takes the same _harvest and verify path, so it inherits all three. That needs checking on its
  benchmark once implemented.

## (f) Learn stand-ins beside the trace's live tensors (HF, dynamic rows)

- At bind time (tape _bind_opaque -> replay learn -> _learn), the trace has just run the op eagerly, so its real
  operands and its output are live.
- _learn still allocates stand-in inputs (:869) and a probe output (:893). On MobileBert that is 1048 MiB of stand-ins
  beside the seam's 1000 MiB logits.
- (f1) Lend the real operands: pass the call's tensors to _learn (the `lent` path).
  - For _OUT ops (mm/addmm/bmm), also lend the real output as out=, so the probe and verify's in-place GEMM check write
    into the call's own output, as verify's in_place path already does. Eager reruns into it on a mismatch.
  - Zero extra bytes for GEMM keys. Stand-ins remain only for an op that writes an input (mutates), which must not touch
    the caller's tensors.
  - Cost: a few lines in _learn plus passing the call's tensors through learn().
  - Risk: alias groups. The key's alias layout comes from those same tensors, so it matches by construction.
- (f2) Defer: queue the learn and run it after the trace call returns. This is the existing learner and background
  machinery, so the variant learns at its next call.
  - It removes coexistence with the trace's values for every op, but adds one learning call per key.
  - Recommended only for what (f1) cannot lend: mutating ops.

## (e) Planned arenas over a fragmented cache and per-shape arenas

- (e1) "fall back to per-temporary placement when no cached block fits" is the only option at or below default in every
  shape order, but it is not cheap here:
  - A relocated temporary's pointers are rewritten at relocate() to arena base + offset row (_moved: PointerSlot on the
    arena allocation). A per-temporary fallback needs the un-relocated slot forms too: two forms of every relocated
    slot, keyed site operand and scratch, with patch selecting by call.
  - Or it needs a second (eager) variant of the same tape, with the entry choosing by fit.
  - Either way it also needs a c10 query, "a cached free block of n bytes on this stream exists" (a lower_bound on the
    stream's pool, like the project's getSegmentReleaseCount).
  - Scope: Python (dual rows) + C++ (selection, patch) + c10 (query).
- Cheaper partial options:
  - (e4) high-water arena per run: smaller shapes use a prefix. Reserved = max arena. The first growth leaves one smaller
    block cached.
  - (e3) release cached blocks once at the first planned call. This is a sync, so once only, never per call.
  - (e4) + (e3)-at-growth bounds unet at about max arena (2.7 GB) + model, vs 7.7 GB now and 7.0 GB default.
    It is not at default in every order.
- Note: under the approved policy, Inductor inference resolves to "auto" (eager or run_buffer), not "planned". "planned"
  arenas are reached only when memory="planned" is set explicitly (suite arm host_trace_planned, vLLM/SGLang
  integrations).
- Recommendation:
  - Do (e4) now, for the explicit-planned users.
  - Do (e1) only if planned becomes a default anywhere: it is the only one that never exceeds default, but it is the
    dual-form mechanism above.

## Order of work proposed
1. (d') 1 + 2 + 3 and (f1). All Python, harvest/replay only, on top of option B's snapshot. Gates: harvest, python_entry,
   opaque, aten. Then gh200-a reruns mobilenetv3_large_100, inception_v3, MobileBert, MT5, DistilBert, T5Small.
2. (e4). Python + small C++ (the arena size row becomes a max over seen shapes).
3. (e1) on request.

## (B') The dead window for retired qwords (proposal, not implemented)

Problem: verify's mirrors prove that a retired qword's value does not change the binding's output. They do not prove that
no kernel dereferences it.
- With the harvest pool released per learn (option B), a retired qword can point at memory that is unmapped at replay.
- A kernel that reads but ignores it would then fault.

Proposal: at bind time, rewrite each retired qword to one fixed offset inside a process-lifetime window per device.
- The window is 66 MiB of VA ([-2 MiB, +64 MiB] around the rewritten address, the same span verify's mirrors cover).
- One small physical allocation (2 MiB, the granularity) is cuMemMap-ped 33 times across that VA.
- So every replay reads mapped memory, at about 2 MiB physical per device.

Q1: what if a kernel writes through a retired pointer? All retired qwords alias the same physical page, so concurrent
writers would race on garbage.
- verify already refuses that binding. Each check launch fills the mirrors with 0x00 then 0xFF and fails the key if any
  mirror byte changed ("wrote into memory a constant address points to", _Mirrors.touched).
- A retired qword therefore has a proof per learn that the binding neither reads it in a way that changes the output
  (bitwise check under poison) nor writes through it (mirror untouched).
- Replays run the same kernels at the same key, so no replay writes to the window. Concurrent replays (two streams,
  two variants) only read it.
- If a write happened anyway, through a kernel path verify never exercised, the write lands in garbage nobody reads
  meaningfully. Under the proof, the output does not depend on those bytes, so there is no corruption of real data. It is
  strictly safer than today, where such a write would land in whatever block the stale address now holds, possibly a
  live tensor.
- Optionally map the window PROT_READ, so a write faults loudly instead. I recommend READWRITE: a fault is a sticky
  context error that kills the process, while a write into the window is harmless by construction.

Q2: does 66 MiB of VA per device ever collide with the caching allocator's expandable segments?
- No. The window is a cuMemAddressReserve'd range. Expandable segments also reserve their VA with cuMemAddressReserve,
  as do verify's mirrors and the harvest's operand arena.
- The driver never hands out overlapping reservations, so neither side can map into the other's range.
- VA is not scarce: 66 MiB per device of the 2^49-byte (or larger) VA space.
- Expandable segments reserve large ranges up front (sized to device memory), and the driver places the window outside
  them.
- The only shared resource is physical memory: 2 MiB per device, allocated once, never freed.

## (B'') Retired qwords: what verify proves, and why the binding keeps the captured bytes (decided 2026-10-14)

The dead window (B') was declined. The follow-up "rewrite each verify-proven retired qword to 0 at bind time" was tried,
first as the whole qword and then as the high half only. It was dropped: the binding keeps capture A's bytes, as first
approved for option B.

What a retired qword is:
- A kernel-parameter qword of a cuBLAS key (op in _OUT) with no role: not an operand's, not a scratch block's, and not
  constant across the captures.
- It is shaped like a device address (high half set, low half 256-aligned) in every capture. Since option B it no longer
  needs to be mapped.
- verify forces it into the mirrored set. Each check launch points it into a harvest-owned window at a 4 GiB-multiple
  shift (low half kept), filled 0x00 and then 0xFF, and requires the output to be bitwise and the window untouched.

What that proves: value-independence under that poisoning. It does not prove deadness.
- The mirror keeps the low half by design, for the int32-under-a-stale-high-half case. So the low half is never varied.
- The high half is only varied to the shifted value. Address shape also matches a packed int pair whose high int is
  live, and a shifted high int can still be harmless while 0 is not.

Evidence, from the F tree on GPU 3 with only the rewrite changing:
- Whole-qword zero fails test_fp32_rebinds_bitwise (tf32 mm and addmm), the trace_learns tests (fp32 sgemm: cutlass simt
  + splitKreduce), and 5 attention/RNG equality checks.
- High-half zero fails opaque test_keys_past_64_harvest (990/2048 elements wrong, max abs diff 137) and inductor
  test_cudnn_attention_at_query_length_1.
- No rewrite: all of these pass.

Consequences:
- A retired qword is a value the kernel's result does not depend on under verify's poisoning. It is not a pointer we may
  rewrite. Any rewrite is an unverified value.
- The only sound rewrite would be one verify itself launches: re-check with the rewritten value and keep it only if
  bitwise. That is not worth a check launch per retired key, and a dereferenced null would fault rather than mismatch.
- The user's intent still holds: no VMM window is kept, and a stale address that some kernel did dereference at replay
  would point at unmapped memory and fault loudly, never silently read live data.
- The low-address false refusal in python_entry was a separate issue. An int pair (M-1, K-1) read as an address because
  mirror VA reserve/free churn handed low VA to the caching allocator. It is fixed by keeping mirror spans reserved for
  the process (_MIRROR_SPANS), not by loosening verify.

## (iv) Alternative for later: keep small int pairs from aliasing mapped memory by reserving the low VA range

Both VA-dependent false refusals of the day were small int pairs read as device addresses:
- 0xd_00000fff in nvjet: (M-1, K-1).
- 0x8_000060b2 in a cuDNN attention struct.
Each read as mapped because an allocation sat at that low VA at that moment.
- The first case came from mirror reserve/free churn handing low VA to cudaMalloc, fixed with persistent _MIRROR_SPANS.
- The second, a 1-vs-3 capture sibling fallback in about 3/21 runs, is still unexplained.

Proposal: at host-trace init, reserve the low VA range so that no allocation can ever land there. It would be reserved,
never mapped. Then a qword whose high half is small always reads as "no CUDA address".

Probe on this GB300, a fresh process after torch init (scratch/dbg/lowva.py):
- Ordinary allocations land high (0xfff9_5de0_0000...).
- cuMemAddressReserve(64 GiB, align 2 MiB, hint 0) returns 0x3_4000_0000. So the driver does not hand out VA below about
  13 GiB, and the context or driver already holds or excludes that part.
- A hint of 0x1_0000_0000 (4 GiB) is not honored: it also returns 0x3_4000_0000.
- Hints of 0x4_0000_0000 and 0x8_0000_0000 are honored exactly.
- So a fixed-address reserve works where the range is free: the hint is taken when available and is not a guarantee.
  The code should check the returned address and fall back to what it got.

Design:
- Reserve [0x3_4000_0000, 64 GiB) once per device, about 51 GiB of VA, which is free on a 2^47+ space.
- Int pairs with a high int of 4..63 then never alias. Pairs with high 1..3 fall in the context's own sub-13 GiB region,
  which this cannot cover. An ordinary process has nothing mapped there that we know of (unverified).
- Mirror spans would then be placed above 64 GiB by the driver; they need 4 GiB alignment, not low addresses.
- It retires the 0xd_00000fff class even without the persistent spans.

Cost: one reservation per device, no physical memory.

Risks:
- Another component that needs low VA, such as a fixed-address IPC import or a library's own cuMemAddressReserve with a
  low hint, would fail or move. I know of none in our stack, but this needs a check with NCCL/symmetric-memory users
  before defaulting it on.

## (B''') Correction: the cuDNN fields are stale and never dereferenced; the B-era fault was an int pair verify took for an address (2026-10-14 late)

The user's question was whether the faulting cuDNN fields are the call's live workspace, which verify's mirror test cannot
tell from a stale pointer. Answer: they are neither our memory nor a workspace. They are stale and never dereferenced.
My earlier attribution of the B fault ("constant pointers into the released capture pool, dereferenced") was wrong.

Method, untimed, on GPU 3:
- Probes: scratch/dbg/provenance.py, conv_bwd_probe.py, deref_child.py + scratch/deref.sh, const_suite.py and
  mirror_log.py.
- Configurations: conv forward sibling (batch 4 -> 6), conv backward bf16 and tf32 (the test_backward keys), and cudnn
  attention forward and backward, both the batch-4 template and batch 16 (sibling and full).

(1) Provenance against the capture-time allocation map:
- The map is every capture's own allocations from _cuda_hostTraceHarvestCapture, its fresh outputs, the plugs, every
  capture-pool block, and the operand arena.
- 0 non-slot qwords of any of these bindings point into it.
- The device-shaped (0xF...) non-slot qwords that exist come in two kinds:
  - Most (950 to 1022 per full-suite run) are host addresses or small ints. The caching allocator never held them,
    they are unmapped on the device, and they vary run to run.
  - A few in conv backward (CUTLASS wgrad/dgrad param 0, bytes 576/1008/1640/1656) point into default-pool blocks:
    temporaries an earlier call freed, e.g. a test's .contiguous() copy, found by stack (assertReplays:1329). These are
    leftovers of cuDNN's uncaptured path, not this call's allocation.

(2) Writes and reads:
- Mirroring every such qword and launching with fills 0xA5/0x00/0xFF: no mirror byte changed, and the outputs were
  bitwise across fills and against eager.
- Dereference: one child process per field set it to reserved, never-mapped VA. Control: operand 0 at the same kind of
  address faults with cudaErrorIllegalAddress in both the conv and the attention binding.
  - conv backward bf16 9/9 fields, tf32 16/16, attention forward 3/3, attention backward 4/4: all launched clean and
    bitwise.
  - conv forward has no such fields.

(3) Verdict per field: stale for all of them; none read-only, none workspace-written.
- The real workspaces are already slots (scratch blocks). No slot fix is needed.

The B fault itself, caught with CUDA_LAUNCH_BLOCKING=1 on the B tree:
- verify's own launch faulted ("operation not supported on global/shared address space") right after mirroring q =
  0x9_47ae147b.
- The driver placed that inside a mapped 1.4 GiB caching-allocator segment at 0x9_14c0_0000, low VA, which B's mirror
  reserve/free churn handed to cudaMalloc.
- The qword is a packed (9, 0x47ae147b) field, not a pointer. The mirror's 4 GiB shift changed the high int from 9 to
  something else, and the kernel built a bad address from it.
- This is the same class as the python_entry false refusal (0xd_00000fff) and the sibling fallback (0xc_00000fff),
  ending in a fault instead of a mismatch.
- The persistent mirror spans and the high reservations (_reserve_high) remove the churn. (iv), reserving the low VA
  range, removes the class.

Guard, as the user asked:
- verify now refuses any non-slot, non-retired qword that points into the harvest's own memory: a capture's allocation,
  the capture pool, or the operand arena. The refusal names the region.
- With the capture pool released after the captures again (B's c), on the G tree: 3/3 full harvest runs green, 0 of
  these refusals, 0 faults.

### (iv) is a verify-safety measure too

A packed int pair that reads as a mapped address gets mirrored. That changes its high int by a multiple of 4 GiB (in
address terms) for verify's launches. The kernel then computes with a wrong integer, such as a size or an index base:
- Usually the result is a wrong output (a false refusal) or a fault (the B fault above).
- If the wrong integer instead produces an address in mapped memory, a verify launch can read or write there.
- A write lands outside the mirrors and outside the binding's operands and scratch, so neither the bitwise check nor
  mirror.touched() sees it. The learn would silently scribble on a live block of the process.

How likely is that today, with the persistent high reservations (_MIRROR_SPANS and the operand arena at
0xFF00_0000_0000)?
- It takes two things.
  - (a) A mapped allocation at low VA, so that a small int pair (hi < 64) aliases it.
    - We no longer reserve or free anything low, and the caching allocator's segments land at 0xfff9... in every run
      seen here.
    - Over 12 + 3 instrumented full-suite runs since the fix: 0 strays, 0 int pairs mirrored.
  - (b) The corrupted integer must turn a computed address into one that is mapped and live.
    - The shift changes the high int by at least 1, which moves an address by at least 4 GiB times whatever the
      kernel scales it by. Past the low region, the only mapped memory sits near 0xfff9_xxxx_xxxx (one band of a few
      hundred GiB in a 2^48+ space). Landing there instead of in unmapped VA (a fault) needs a coincidence.
- Neither (a) nor (b) has been observed since the fix. (a) did occur three times before it, all from our own churn.
- Estimate: very low today. It is not zero: (a) can come back if another component maps low VA. Examples are an IPC
  import, a library's low-hint reserve, or a process whose high band is exhausted.
- (iv) closes (a) outright, which closes the write hazard with it. It should count as part of verify's soundness, not
  as a false-refusal fix. Recommendation: do (iv) next, with a startup check that logs where the low range went if the
  reservation is not honored.

## (C) Stale high halves in cuDNN parameter structs: verified canonicalization (decided 2026-10-15)

What happened. With P (captures into an address-only span at the 0xFF00_0000_0000 hint), harvest's
`TestHostTraceHarvestConv.test_backward_channels_last_True_dtype_bf16_case_strided_dilated` hung the GPU at verify's
first launch of the binding, deterministically in most runs. P2 (span moved to its own hint 0xFE00_0000_0000) hung in
1 of 5 harvest stress runs.

Minimal repro: that test (convolution_backward, channels_last, bf16, case strided_dilated: 32 -> 64 channels, stride 2,
dilation 2, 20x20, batch 4; cudnn deterministic, benchmark off). Node 0 is
`cutlass3x_sm100_tensorop_s128x128x16implicit_gemm_strided_dgrad_bf16_bf16_f32_...`. Its parameter 0 byte 1424 qword
is an int32 0x20 (low half) under 4 bytes that are not that field's: the high half of some address that was on the
heap when cuDNN built the struct (cuDNN builds its parameter structs before the captures and reuses them, so the bytes
are the same in every capture and classify sees a constant; mallopt(M_PERTURB) during the captures leaves them as they
are). Observed values: 0xff00 (H: the operand arena's high half), 0xff11 (P), 0x0000 (another P run).

Measured on H's tree by forcing the high half before verify's launch: 0x0000 and 0xff00 bind; 0x0001 and 0xff11 hang
the kernel. Bit 32 of the qword (bit 0 of byte 1428) behaves as a flag. This looks like a latent cuDNN/CUTLASS issue:
the kernel reads uninitialized host-struct bytes as a flag. Eager runs pass because its leftovers happen to be benign.
Nothing is filed upstream; that is the user's call.

Why it is fatal for us. A verify launch that hangs this way is not recoverable: verify's hang handling zeroes the
scratch and the mirrors from another stream, which does not release it; the live path's eager() restore (and the
copying path's mirrors.close) then wait on the stuck stream forever. H, me4 and step332 have the same exposure and
bind only because 0xff00 is benign.

Decision (coordinator, 2026-10-15): (a) as a verified canonicalization.
- `_canonical(binding, stale)`: every constant qword of a kernel's parameters (alike in every capture, not within 8
  bytes of a slot, not loose) whose value looks like a stale address (high half past the lowest device address the
  harvest knows, i.e. its reservations and the call's operands, or inside the brk heap; not a mapped CUDA address; not
  a retired cuBLASLt qword; not a stack address) gets its high half zeroed, and those 4 bytes become loose (a restore's
  fresh capture holds its own leftovers there).
- The full harvest verifies the canonical form first: verify launches exactly what replay will run, so unlike the
  earlier unverified rewrites (B', B'') this is sound. If verify refuses it, the captured bytes are verified as before.
- A sibling zeroes the high halves its template zeroed (4-byte loose runs at 4 mod 8 that are zero in the template),
  and sibling verify judges it.
- The coordinator's "eager on the same inputs before a launch with a canonicalized field" already holds: in the live
  path the call itself ran just before, and the copying path computes its eager reference before every launch.
- P2's separate hint for the address-only span stays.
- Remaining exposure: a canonicalized qword that is in fact a pointer the kernel dereferences would now fault (low
  address) where the captured form pointed into unmapped memory or, for a constant into the harvest's own memory, was
  redirected to a poisoned mirror. No such dereference has been seen (the 2026-10-14 provenance and dereference tests
  found the cuDNN fields dead).

Result: under P's layout (span at the 0xFF00 hint), the repro binds 4/4 with byte 1424 = 0x20 in every verify launch.
Tests: test_stale_high_halves_are_zeroed (unit), test_strided_dgrad_binds_no_stale_high_half.

## (D) Constant inputs at verify: a random stand-in, and the bound's allowance (decided 2026-10-15)

Live verify (H) runs a binding on the call's own tensors. An input of one value (an attention mask with no padding:
all zeros) cannot check a binding on it, so it used to send the whole op to the copying path (every input copied,
three output-sized buffers): HF's SDPA forwards ran 4-6x eager's peak there.

Why the constant input still needs values. The captures prove its base pointer: operand sets sit in distinct 2 MiB
windows, and a qword is that operand's slot only if it tracks the operand's address in every capture. A host
decision on its values is impossible inside a capture (a device-to-host read fails the capture), and device-side
branching on values is the captured kernels' own, right for any values. What provenance does not prove is the scalars
that index the input: strides, broadcast flags, batch/head offsets, an internal pad's extents. A stale or wrong one
(a cuDNN struct leftover, alike in every capture) reads the same value from every place of a constant input and
passes; only non-constant values catch it. Non-constant live inputs carry their own; a constant one needs a stand-in.

Decision: in live verify, each constant floating input (bitwise one value, via its integer view) takes a random
stand-in at its layout and alignment (its span; a broadcast mask's is its storage). Eager on the stand-ins gives the
outputs' digests (and an unordered output's saved copy and spread); the binding launches into the call's outputs; the
outputs are put back from eager on the real inputs after the check. Nothing of the user's is written but the call's
own outputs, which hold eager's results again. In the copying path (dropout forwards and other ops that write
nothing), operands stay live but for the alias groups of constant inputs.

Bound per learned key (TestMemoryGate): learn peak - eager peak <= workspace + 64 MiB + the constant inputs' stand-in
bytes. A materialized mask (SDPA pads an unaligned bias) makes the stand-in the mask's size.

Found on the way: the fused digest (me10) built a list of a non-dense tensor's copied pieces (all alive at once: a
384 MiB broadcast mask copied whole); it now takes them one at a time (test_a_digest_holds_one_piece_at_a_time,
red on me10/me11).

## (E) The remaining training leftovers are placement (2026-10-15)

gh200-a's me12 rows: whole-process reserved at default on all 10 training rows; leftovers mobilenetv2_100 steady
+197, mobilenet_v2 compiled +375 / steady +149, ghostnet_100 compiled +619 / steady +50 (GH200). diag/harness_phases.py
splits the dynamo harness's process at its warmup() phases: in every phase the live peak is the same in both arms (or
lower for ours); each leftover is extra segments mapped in one phase (mobilenetv2_100: the first warm replay, +186 MiB
GH200 / +148 GB300; ghostnet_100: the cold start, +590 GH200, equal on GB300; its warm replay +44 / +66).

Attributed on GB300 (harness_phases --detail, diag/phase_compare.py): mobilenetv2_100's +148 MiB is one new segment for
a 147 MiB buffer both arms allocate (a (128,96,56,56) fp32 backward buffer), which in default lands in an existing
segment. Default's sequence holds allocations ours does not: cuDNN workspaces inside eager convolution_backward
(2 x 20.25 MiB in the depthwise calls, 3.5 MiB at another), and a few buffers placed differently. Harvested bindings
allocate only the scratch their kernels reference (unreferenced cuDNN workspaces dropped; dgrad/wgrad share a block),
so the replay allocates less than eager, the cache's free blocks differ when the large request arrives, and it maps a
fresh segment. Live memory is at parity; the extra reserved is placement, from fewer workspace allocations.

Decision (coordinator): accept as placement. The exact-parity option, if strict sequence parity is wanted later:
eager-mode bindings allocate every captured allocation as scratch (as cuBLAS's workspace already is) and each scratch
block takes eager's lifetime (allocated before its first node, freed after its last), so the replay's allocator
sequence is eager's. It needs the native commit to free a site's scratch mid-binding (one allocation per site today)
and gives back part of scratch sharing's peak win where blocks are unreferenced.
