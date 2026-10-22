# hintfix: zero hint reads at the candidate-4 core sites (this lane's scope)

STATUS: READY. hf1-hf7 are queued for candidate 4 (int6/CANDIDATE4_QUEUE.md). The coordinator approved checks before the graph and keeping n <= 64 in the slice fix (10-08). hf8 is comment-only.

Base: int6 snap/step242 (even, build_cpp8) and snap/step241 (odd), each + the hint audit
(`core/hintaudit/hint_audit.int6cpp.patch` / `.int6.patch`), copies in `base/s242`, `base/s241`
(host-trace files) and full trees `b242`, `b241` (untouched baselines). Working tops: `s242`, `s241`.

## Patches (apply in order, on top of the audit patch)

`hintfix/hf{1..6}_*.int6cpp.patch` (even) and `.int6.patch` (odd), copies of `split/out/`. md5 (12):

```
082441b8821a  hf1_at_call.int6cpp.patch
c67fa2cd5a02  hf1_at_call.int6.patch
b74c1a373309  hf2_tape.int6cpp.patch
2107ad0e8277  hf2_tape.int6.patch
922f392f2a3b  hf3_opaque.int6cpp.patch
3e409aacd0bd  hf3_opaque.int6.patch
c87fa2051881  hf4_memory.int6cpp.patch
0fa47ae79286  hf4_memory.int6.patch
33d9fbcc8fb9  hf5_checks.int6cpp.patch
dbd19072a6e7  hf5_checks.int6.patch
2a880b7e6fa1  hf6_tags.int6cpp.patch
e9f6ed90a7fc  hf6_tags.int6.patch
fd6c211b4eea  hf7_slice.int6cpp.patch
f0f9ea1b4025  hf7_slice.int6.patch
4330824cb24d  hf8_offender_owners.int6cpp.patch
4330824cb24d  hf8_offender_owners.int6.patch
```

The odd series is the even one applied
stage by stage onto base/s241. One known hunk (LoweredKeyedSite.alias comment: the odd line has no
`arena_scratch`) is re-applied by hand. The odd top equals `s241`. Each stage imports, and its OFFENDERS
shrink: 32, 23, 17, 15, 15, 15. ALLOWLIST goes 5 -> 3 at hf5.

| patch | what | net non-blank (product / test) |
|---|---|---|
| hf1_at_call | `CallValues` (a trace's values at a call, from the call's leaves, via Ctx.transfer / xreplace); `_Trace.values`; `_Root.index` + `_placeholder()`; replay `_build`, redispatch `_tag`/`_redo.root`/`_redo` (each made symbol keeps its call value, `at_call`), capture `_check_trace_values` (evaluated at the tape's call, not at hints), `_bind_opaque` first_alloc; `_symbolic_run`'s empty-size guard | +66 / 0 |
| hf2_tape | F17: `_check_witness` compares layouts at the call; a size-1 disagreement gets `select(z == 1, rule, s)`, rule = the op's Python meta at size 1 under eager's empty_like (`_EagerMeta`, its conditions guarded where z == 1), checked to equal eager's at the warm-up; with no rule the fake's stride stays and an opaque call on that root runs eagerly. `_call_key`/`_order_key` at call values; `_contiguous_strides` = sym_max; `_dense_terms` ordered by call values (the terms guard it); pins `int(v)`; `pending` by call values | +76 / +15 |
| hf3_opaque | `aliasing` structural (first operand per storage, symbolic distances; `_opaque_key` takes the lowest per call); `trace_key` = `_opaque_key` over the key's values at the call (the rows a bound site evaluates); `key_guards` from operand offsets; `bind_at_trace`: no distance pins for any provider, `bool(offset != 0)` declines; `record_binding.address` = root + offset (as the site's operand rows) | +2 / 0 |
| hf4_memory | F11: an arena whose conditions fail at the traced call declines (was an AssertionError); `_canonical`/`_forms` already read `program.values` | -1 / 0 |
| hf5_checks | validity checks (user decision): `_check_as_strided` and allocate's negative size become `_Trace.check` conditions, not guards. They are checked at the call's values, ANDed into the variant's `valid` row (`Tape.checks`, also in fold / redispatch entries), and deduped when the domains or a recorded guard decide them. Flag `torch.cuda._host_trace.validity_checks` (default on; off: a guard, as before). Messages use call values. ALLOWLIST: the two error-text sites removed; bit_length/select/f32_div kept, justified | +48 / +17 |
| hf6_tags | inplace_view: `_fake_call` compares each traced argument's twin metadata after the fake call (tag check kept as an early decline). nondeterministic_seeded: `takes_generator(op)` (op or an overload of its packet takes a Generator) in harvest `_draws` and replay's learn bounds; a harvest capture of an op not declared drawing that takes philox offsets is refused | +25 / +15 |
| hf7_slice | coordinator's case (ii), vLLM `slot_mappings[:, :num_tokens_padded]`: Python indexing (C++ applySlice) returns `self` for a slice it statically knows covers its dim (`TORCH_STATICALLY_KNOWN_TRUE(length <= stop)`), and the IR guards that read (`n < W` at n=8, a graph-level retrace at n=W). `_TracedTensor.__getitem__`/`__setitem__` run indexing with C++ static reads decided by the domains alone (`_host_trace_ir.static_reads`); `_TraceMode` restores guarding inside every dispatch. What's left: `n <= W` (the fake slice's own clamp, true on [1, W]; past W eager's size is W, so a retrace is right) and the trivial `n != INT64_MAX` | +32 / +21 |
| hf8_offender_owners | comment-only: the owner of each of the 15 remaining OFFENDERS entries. symti lane (land/core/symti): `_elementwise` and `_Trace._pointwise_host` + its 3 locals (5). Triton lane (land/core/next/triton_hint): `_witness`, `_launch.<locals>.launcher`, `_intercept`, `_SymbolicSpecialization.__call__`, `_resolve`, `record_launch` (6). CuTe lane (land/core/cute70): `_describe`, `_intercept`, `_intercept_foreign`, `_pin` (4) | 0 / 0 |

## Decisions and snags

- **Validity checks run before the graph, not after it (reported to the coordinator, waiting for an answer).**
  All four cases are memory-safety checks: negative stride, negative storage offset, a view past an
  allocation's bytes, a negative size. Running the graph first with failing values is an out-of-bounds launch
  or an allocation that can't be made. They are requirement rows in `valid`, evaluated with the program
  before launch (no hit-path cost). A failing call misses, and the retrace's warm-up raises eager's error with
  the call's values. A true after-graph check would also need native replay (C++) changes.
- **F17 rule.** The fake kernel can't be eager's rule at size 1. torch's meta/fake empty_like (refs,
  elementwise perm) differs from eager's C++ empty_like on size-1 dims, even concretely. The rule therefore
  runs the op's Python meta (meta_table) on plain meta twins with eager's empty_like
  (`_empty_like_strides`, the trace's own model). sort/argsort get `Where(s0 == 1, input stride, s)`.
  Where no rule matches eager's warm-up value, nothing from the warm-up is written. The fake's stride stays
  (the replay's check ignores size-1 dims), and an OpaqueCall on that root becomes an EagerCall, so no key
  reads an inexact stride.
- **`_dense_terms` / `pending` / witness keys read call values (CallValues), not guards.** Each is a choice
  that guards already settle (the dense terms), or one where any outcome is sound (bind now vs after the
  trace), or a check against the warm-up. The values come from the call's leaves through the IR's own
  arithmetic, not a node's hint.
- **record_binding.address** is root + offset with no numel test, the same address the KeyedSite's
  operand rows give at every other key. `select(numel == 0, 0, ...)` would not lower: a pointer must be one
  root plus an offset.
- **Coordination.** The symti lane deletes `_traced_pointwise` / `_pointwise` / the pointwise witness, and
  took over the tag check at tape.py:1258 and the `_pointwise` meta-check skip (detection by Generator
  argument / generator TLS; its structured meta guards). This lane did harvest `_draws` and replay:858.
  My `_eager_call` hunks (the provider block, `_fake_call`) are disjoint from theirs.

## Results

hf7: repro case (ii) on GH200: base traces 2 with retrace cause ('graph', (n - W) < 0); new traces 1, 1 redispatch (the mul's own op guard: its operand is contiguous at the full width), retrace_causes {}. GH200 strict tape/replay/opaque/oob/capture/memory/guards/triton/aten OK, harvest = base set. GB300 odd (s241 with hf7): tape OK 45, replay = the base's 2 known failures (195 tests; the new full-width test passes), opaque OK 37, aten OK 1674. Returning self vs a view differs only in object identity (`out is buf`, `_is_view()`): same storage, sizes, strides, offset, version counter; in-place writes alias either way. The JIT tracer skips the shortcut for the same reason (disable_slice_optimization).

Hint audit strict (`TORCH_HOST_TRACE_HINT_AUDIT=strict`). Sites before/after, summed over the GH200 suite logs
(`ht/hintfix/logs/{base,it5}`) and minops (`base_meas`, `it5_meas`):
- every site in this scope (capture x2, replay x2, redispatch x3, opaque x6, memory x2, tape `_check_witness`,
  `_call_key`, `_order_key`, `_contiguous_strides`, `_dense_terms`, `_eager_call` x2, `_bind_opaque`,
  `_check_as_strided`) plus `_symbolic_run`: 0 reads. minops decode+prefill hint reads: 1,152,017 -> 95,820
  (what is left: `_elementwise` / `_pointwise_host` (symti lane) and triton `_intercept`/`_witness`/launcher).

Suites, failing sets identical to base where not OK (base = b242 / b241 + audit, same box):

| file | GH200 even base / new | GB300 odd base / new | GB300 even base / new |
|---|---|---|---|
| hint_audit | - / OK | OK / OK | OK / OK |
| tape | OK 43 / OK 44 | OK 43 / OK 44 | OK 43 / OK 44 |
| guards | OK / OK | OK / OK | OK / OK |
| capture | OK / OK | OK / OK | OK / OK |
| memory | OK / OK | OK / OK | OK / OK |
| opaque | OK / OK | OK 37 / OK 37 | OK / OK |
| replay | OK 254 / OK 256 | 2 F / same 2 F (known odd: zero-element sort strides, redo dispatch count) | OK 254 / OK 256 |
| oob | OK 7 / OK 8 | OK 7 / OK 8 | OK 7 / OK 8 |
| harvest | 11 F + 1 E / same (known GH200 pool bytes etc.) | 5 F / same | 5 F / same |
| triton | OK / OK | - | - |
| aten | OK 1676 / OK 1676 | OK 1674 / OK 1674 | OK 1676 / OK 1676 |

Intermediate stages (GH200, each stage tree = b242 + stage files): tape, replay, opaque, oob, capture,
memory all OK at stages 1-5.

Repros: F17, F2, F10, F11 PASS before and after (GH200); F17 PASS on GB300, both lines.

minops84 (36 layers; counts are the same before and after on every line):
- GH200 even, decode: traces 3, redispatches 16, re-run 892, bitwise 16/16. prefill: traces 2, redispatches 17,
  re-run 384, bitwise 17/17. Retrace causes identical.
- GB300 even: the same numbers as GH200 even, before and after.
- GB300 odd, decode: traces 6, redispatches 20, re-run 2185, bitwise 23/23. prefill: traces 2,
  redispatches 17, re-run 388, bitwise 17/17. Retrace causes identical.

Tooling: `scripts/` (gate.sh / meas.sh on GB300, r029.sh / gate029.sh / meas029.sh on gh200-b, port.sh for
the odd line). minops84 copy `scripts/minops84.py`: a redo keeps its made symbols' call values (`at_call`),
so the copy reads those where the tree has them. `split/build.py` + `split/patches.sh` regenerate the patch
series from the labeled hunks of base/s242 -> s242.
