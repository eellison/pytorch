# peak: a run's temporaries live in a later tensor's bytes (replay peak = eager's for pointwise chains)

STATUS: READY (results below). Base: int6 step242 / step241 + hintfix hf1-hf8 (`base/s242`, `base/s241`); tops `s242`, `s241`.

Patches: `peak1_host_temporaries.int6cpp.patch` (even, md5 a3af7b860a54), `.int6.patch` (odd, md5 a4764d9068ed;
`_host_trace_memory.py` ported by hand, the odd line has no "held"). Net non-blank: product +57, test +63 (both lines).

## Root cause

`sigmoid(x) * y + 2` at m=7 (fp32, 768 wide; each tensor 21,504 B):
- eager: t1 = sigmoid(x); t2 = t1 * y; t1 is freed (refcount) before `+ 2` allocates out, which takes t1's block.
  Peak t1 + t2 = 43,008.
- replay, "run_buffer" (what "auto" picks for small runs: within its documented margin of eager's peak) and the
  planned modes ("planned", "packed"): every tensor a run makes is allocated at the run's start, before the graph,
  and its temporaries are in one buffer freed after the graph is queued. So out is live with t1 and t2:
  64,512 (+50%). A tensor could never take a temporary's bytes ("A tensor cannot reuse a temporary's bytes",
  the run_buffer docstring).
- "eager" mode already matched (43,008): it interleaves the run's allocations and frees in tape order. The
  roots-based liveness (FZ06) and check_plan were correct; the gap was placement, not liveness.

Measured on GH200, step242 + hf1-hf8, per chain x mode (`probe/peak.py`): `eager` == eager. `auto`, `run_buffer`,
`planned` and `packed` were over by one tensor on every chain with a dead intermediate (sigmoid_mul_add,
add_mul_exp, four ops, two outputs); x.abs()+1 was already equal.

## Fix

A run's temporary that dies before a tensor the run makes is first touched lives in that tensor's bytes, at
offset 0. That is what eager's caching allocator does when it gives the tensor the freed temporary's block.
- `plan_memory` (every mode but "eager" and "held") records each base's first use. `_hosts` then gives each
  tensor, in first-use order, a chain of the run's temporaries that meet three conditions: the same bytes row
  (equal bytes at every call), a last use before the tensor's first use, and no overlap with each other.
  `MemoryPlan.hosted` (k -> host, seq, last).
- `relocate` moves a hosted temporary's pointer slots and keyed-site rows to the host (`MemoryPlan.moves`, also
  used by fold/redispatch entries). `arena_addresses` gives it the host's placeholder for the capture. `_build`
  relocates whenever a plan hosts or plans.
- `check_plan` checks the invariant: a hosted temporary is live with its host, and the host's first use comes
  after the temporary's last use.
- Fewer temporaries means a smaller run buffer and no extra allocator call. Nothing changes on hits.
- Switch: `torch.cuda._host_trace.host_temporaries` (default on; off is the old placement). No mode was removed
  and no default was flipped: "auto" still picks run_buffer vs eager by its margin, and the run buffer now
  matches eager on these chains.
- Pre-existing bug fixed on the way: `relocate` mapped a captured segment's launches back by object identity.
  An RNG kernel's captured launch is a replaced object (its capture's images), so the planned modes raised
  KeyError on any RNG run. Segments' launches are now moved directly.

Not covered: a tensor smaller than the run's temporaries (a reduction's output) is still made at the run's
start, above eager by its block. The test bounds it to the m-element output and partial sum, 2 x 512 B.

## Tests

- `test_cuda_host_trace_memory.py::TestPeakMatchesEager::test_a_chains_peak_is_eagers`: 4 chains x 5 modes
  (auto, eager, run_buffer, planned, packed); traced at m=7, replayed at 7/33/7. Asserts bitwise outputs and
  replay peak == eager peak (the reduction chain within 2 x 512 B outside "eager").
- `test_without_hosting_the_run_buffer_holds_the_intermediate`: the switch off gives eager + one tensor.
- Updated: `test_the_plan_follows_the_tape` (the unused allocation is now hosted in the output) and
  `test_greedy_by_size` (runs with hosting off: it tests the arena's placement).

## Results

Strict hint audit on. Base for the failing sets: hintfix's runs on the same box.
- GH200 even:
  - memory OK (42), replay OK (257), tape OK, opaque OK, oob OK, capture OK, guards OK, triton OK,
    lower_tape OK, aten OK (1676).
  - harvest: 11 F + 1 E, the base's set.
  - integration: 1 test, skipped, same as base.
  - Repros F17, F2, F10, F11: PASS.
- GB300 odd:
  - memory OK (42), tape OK (45), opaque OK (37).
  - replay: the base's 2 known failures.
  - aten OK (1674). GB300 even memory is queued on gpu1.lock (`logs/pk_e.txt`); the GH200 even run passes.
- minops84 (GH200 even): decode traces 3, redispatches 16, re-run 892, bitwise 16/16; prefill traces 2,
  redispatches 17, re-run 384, bitwise 17/17. Identical to hf1-hf8.
- `probe/peak.py` (GH200): all five chains equal eager in auto, run_buffer, planned, packed and eager at m=7
  (before: four of five chains +21,504 B in all but eager).
