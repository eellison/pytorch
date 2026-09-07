# #191974 - Deduplicate nested reduction test helpers

Local commit `3f62ae27baa` - #191974. The reviewed fixes keep this commit
test-only; source fixes remain with their owning PRs.

The isolated commit remains test-only. The feature regressions belong to the
lower commits that introduce the corresponding behavior, not this cleanup.

Test-only cleanup. It changes no scheduler or codegen source and adds no test
case; the exact-index scheduler coverage now lands in #191775 with the API it
tests.

## What changes

`test/inductor/test_nested_reduction.py` now:

- imports `torch.nn.functional as F` once at module scope instead of in each
  helper and test
- shares `E2M1X2_PACK_ASM` across the NVFP4/MXFP4 pack constructions
- keeps FileCheck expectations literal, so generated-code checks are not built
  from the same constant as the code under test
- records why `_float_to_mxfp6_e2m3` is intentionally longer than an equivalent
  expression

The last point is load-bearing test setup. Shortening the software E2M3
producer changes realization and makes the preshuffled MXFP6 staged fusion stop
forming. The comment prevents a future cleanup from silently weakening those
tests.

## Stack evolution

An earlier version of this commit restored
`test_fusable_read_and_write_requires_exact_index_match`. That was the wrong
commit boundary: #191775 removes the old `allow_index_equivalence` API and must
replace its test at the same time. The final #191974 diff therefore contains
only the one-file deduplication above.

## Review checklist

1. Confirm the diff is confined to `test_nested_reduction.py`.
2. Confirm all shared assembly call sites still use the same operand order and
   constraints.
3. Keep generated-source assertions literal rather than referencing the shared
   constant.
4. Do not shorten `_float_to_mxfp6_e2m3` without proving the preshuffled tests
   still form a staged kernel.

## Verification

At this commit:

```text
python test/inductor/test_nested_reduction.py
  450 passed, 12 skipped

python test/inductor/test_inductor_scheduler.py
  88 passed, 6 skipped
```

The adversarial replay review compared all 154 test definitions and decorators
before and after this commit and found no semantic coverage change.
