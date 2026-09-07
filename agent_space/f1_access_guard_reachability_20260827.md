# F1 `_AccessGuard` reachability audit

Date: 2026-08-27

This is an AI-assisted scratch report for human review. No production files,
commits, or GitHub state were changed by this audit.

## Question

Does exact indexed forwarding need its own mask/fill compatibility model, or can
it follow the existing Triton masked-load behavior?

## Existing in-tree behavior

`TritonKernelOverrides.masked()` already owns masked-load semantics:

- For a direct external load, it sets `kernel._load_mask` and
  `kernel._load_other`, and the physical `tl.load` carries the predicate/fill.
- If the body may return an in-kernel store-cache value, it sets
  `_load_other=None` and emits an outer `tl.where(mask, result, fill)`.

The pre-F1 sub-parent resolver also recorded only unmasked source loads. F1's
`_AccessGuard`, `_SourceAccessKey`, and `_apply_consumer_guard()` duplicate part
of the first mechanism so an external unguarded value can replace a masked
physical load and then have its fill reconstructed manually.

## Instrumented reachability

I instrumented every planned source record, consumer resolution, and successful
forward in `test_nested_reduction.py`, with both caches disabled and the full
worktree overlay.

Full suite result: 399 tests ran, 8 skipped, all passed.

| Event | Count |
| --- | ---: |
| Planned source records | 562 |
| Unguarded source records | 556 |
| Guarded source records | 6 |
| Successful forwards | 943 |
| Successful forwards from guarded sources | 0 |
| Unguarded source -> unmasked consumer | 709 |
| Unguarded source -> masked consumer with `_load_other=None` | 234 |
| Successful forwards with an explicit consumer fill | 0 |

The six guarded records come only from two external-input tests:

- cat inputs loaded under fill `0.0`;
- a deliberately mismatched padded input loaded under fill `0.0` and consumed
  under fill `1.0`.

None is forwarded. The 12 explicit-fill consumer attempts are all optional
external relations and all use the normal physical-load fallback.

The target kernels are narrower still:

- MXFP6 4:3, scale swizzle, and preshuffle only use unguarded sources. Their
  masked consumers always have `_load_other=None`, so the existing outer
  `tl.where` owns the predicate.
- NVFP4 and MXFP4 use only unguarded sources and unmasked consumers in the
  exercised paths.

## Conservative-policy experiment

The tested policy was:

1. Record a planned source only when `kernel._load_mask is None`.
2. Forward it only when `kernel._load_other is None`.
3. Otherwise use the ordinary physical-load fallback for an external source;
   retain the existing loud miss for a required in-kernel source.

This makes `TritonKernelOverrides.masked()` the sole owner of fill/where
semantics. It removes the need for `_AccessGuard`, guard-keyed caches, typed
fill identity, guard transition matching, and `_apply_consumer_guard()`.

Results:

- The complete nested-reduction suite under this policy ran 399 tests, passed
  all of them, and skipped 8.
- The applied no-guard worktree then repeated the ten-case source capture;
  every case emitted one kernel and matched the published hash exactly.
- All ten protected kernel hashes are byte-identical, including persistent and
  looped MXFP6 4:3, internal-source, reduced-broadcast, scale-swizzle,
  preshuffle, and DCN-preshuffle forms.
- The cat producer test passes in persistent and looped modes.
- The mismatched masked-source test passes in persistent and looped modes and
  continues to use the correct fallback.
- MXFP6 4:3 numeric/exact tests pass in persistent and looped modes.

I also constructed the missing reachable case: an unguarded full-width input is
reused by a padded sub-parent consumer with an explicit fill. Current F1 avoids
two loads by splitting the parent register and rebuilding the mask with
`_apply_consumer_guard()`. Under the conservative policy, the resolver declines
that optional reuse and ordinary Triton codegen emits two masked loads instead.
The graph still emits one staged kernel and is numerically correct. This is the
only demonstrated capability lost by removing the guard machinery, and it is
not used by the protected workload corpus.

The protected hashes exactly match the published F1 table:

```text
factor2-True       98b111d6f7798393c16439e9ed2c770ad98ddb8dee6cb731cb2d1640a75bc59c
mxfp6-True         b0fe88ce8d8fb83adc55d9d9e7c4c16bc360160bd14db3afe6343b42b57a79
internal-True      2efc561292dd1ba54ae8576515638189fcf7671122fcafbb64c0f987f3e9c887
factor2-False      6edb0230b628013056895dcbddd3643228608bb4be95d40250de15d7ec906961
mxfp6-False        fcd5bda8ac08084965dfc92ad7142b0dadde8f3615469f3c4a8b28a44f6a976e
internal-False     fb769f5dca3089dce0684cef2b3f4fcc6a5d9c232d736570d89aedd434be52ba
reduced-broadcast  a907b3fbb5d83fdf8a61c7f292a4729d26aa01d4c8f9c9ed61f8562bb83b8258
scale-swizzle      7f72ab3924703863f8e92edcf6849f6c70f9270e31f696cde87fd115571b48e9
preshuffle         710d6b80635a0add97076be321f2c19dd82301319f8cc85224f6421c99871497
dcn-preshuffle     f3a02e5c076058f991b61a2ebb5bb8a54fb5d7c0e8f08f110018c2f99a869d19
```

## Recommendation

Remove the custom guard algebra from F1. Keep exact access matching, but map
`MemoryDep` directly to its unguarded live value. When `_load_other` is not
`None`, do not intercept the load; let ordinary Triton load/CSE code preserve
the fill. A required-source miss remains a compiler assertion, which is the
correct fail-closed behavior for an unsupported in-kernel guarded producer.

This is both smaller and more consistent with the tree. It gives up only an
unobserved optimization: reusing a previously materialized external value in a
direct masked-load context instead of issuing the normal load (which ordinary
CSE can still deduplicate when the emitted load expression is identical).
