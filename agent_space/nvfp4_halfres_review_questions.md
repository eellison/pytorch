# NVFP4 Half-Resolution Epilogue Review Questions

This note captures the main questions we worked through while hardening the
NVFP4 half-resolution nested-reduction path.

## What was the actual bug?

The half-resolution epilogue was not being treated as a leaf stage.

The fused kernel computes half-resolution outputs such as `even` and `odd`, but
their materialized stores are emitted at the tail of the kernel. Fusion legality
only checked domain compatibility (`PARENT_HALF`, `PARENT_FULL`, reduced output,
etc.), so a later fused node could read those output buffers before the stores
had happened. That was the silent wrong-result case.

The inline-asm crash was the same contract violation in a different form: the
full-resolution sibling recomputed `even` inline instead of reading the stored
half output. That bypassed a pure read-before-write check, fused an incompatible
`reduction + half-pack + full-res sibling` group, and codegen crashed with an
unexpected group.

## Should this ever silently produce incorrect results?

No. If the half-resolution epilogue cannot safely stay fused with a consumer, it
must fall back to another kernel. Silent incorrectness is worse than losing the
fusion.

The invariant we now enforce is:

> Once a half-resolution epilogue is present, its materialized outputs are a
> leaf stage for this implementation.

Same-stage half-resolution consumers may stay fused because they reuse values
in-register. Reduced-output epilogues may also stay fused. Non-half consumers of
half outputs, and full-resolution siblings in the standalone half-epilogue
kernel, must split out.

## Why was a buffer-read guard not enough?

It caught the materialized re-read bug, but not the inline-asm case. In that
case, the full-resolution consumer did not read the half output buffer; it
recomputed the half expression inline. The real issue was not only "does this
node read the output buffer?" but "does this fused group contain a
half-resolution epilogue plus a non-half full-resolution sibling that this
codegen path does not model?"

That is why the standalone guard uses half-resolution candidate detection before
requiring the full epilogue plan to succeed.

## Why extract `_half_resolution_epilogue_candidate_nodes`?

The old legality path asked for a complete
`half_resolution_epilogue_plan(...)`. Invalid groups can fail the full plan
before the leaf guard gets to reject them, which is how the inline-asm crash
escaped.

The extracted helper answers only the lighter question: "does this node set
contain half-resolution epilogue candidates for this reduction?" The full plan
still performs the source-dependency and ambiguity checks before codegen. This
keeps the guard and the actual plan aligned without weakening the real legality
checks.

## Why are producer and standalone guards different?

The producer path already has nested-reduction domain classification and can
support valid `PARENT_FULL` consumers in the surrounding nested schedule. Its
new guard is intentionally narrower: reject non-half nodes that read
half-resolution output buffers.

The standalone path builds a special reduction plus half-epilogue kernel. It
does not model arbitrary full-resolution siblings alongside the half epilogue,
even if they do not read the materialized half output. Its guard therefore also
rejects non-half, non-reduced siblings once a half-resolution epilogue candidate
is present.

## How should persistent and non-persistent differ?

They should both support the valid standalone half-resolution pattern in a
single kernel. The difference is how the parent tile is obtained:

- persistent: the reduction sees the whole tile at once, so the half epilogue
  reuses that live tile and splits it
- non-persistent: the reduction is looped, so the first-pass tile is not live
  after the loop; the half epilogue runs as a second loop, reloads the parent
  tile once, then splits it

The non-persistent form is therefore not a fallback. It is the same fused
kernel contract with an epilogue reload, matching the existing nested-reduction
producer path.

That should not be confused with an optimal single-load target. If the goal is
to avoid the second input load, the implementation must either choose the
persistent form or explicitly stage the parent tile somewhere. A looped
non-persistent reduction cannot compute the scale first and later pack the
original values without either reloading those values or storing them.

## Did this over-reject the important NVFP4 pattern?

No. The production pattern remains:

```text
reduction/scale -> even/odd half-resolution values -> inline_asm pack
```

Those half-resolution nodes stay in the same kernel. The rejected cases are
downstream or sibling consumers that need full-resolution work not represented
by the half-epilogue kernel contract.

## What about mutation renames and aliasing?

The guards compare names through `mutation_renames`. Mutation and aliasing cases
do not enter the half-resolution plan itself. The adversarial review specifically
checked that the new guard does not introduce a stale-name hole.

## What tests now cover the issue?

New regressions cover:

- producer path: half output reconstructed at full resolution
- producer path: half output read by a downstream reduction
- standalone path: half output reconstructed at full resolution
- standalone path: half output read by a downstream reduction
- standalone NVFP4 inline-asm path: pack plus full-resolution sibling

The valid standalone half-resolution and NVFP4 patterns now assert one fused
kernel for both persistent choices. The kernel-form tests also assert that
non-persistent reloads the parent tile once in the epilogue and still emits a
single `tl.split`.

## What was verified?

- `python test/inductor/test_nested_reduction.py -k fullres_reader -v`: 6/6
- `python test/inductor/test_nested_reduction.py -k standalone -v`: 22/22
- `python test/inductor/test_nested_reduction.py -k nvfp4_inline -v`: 14/14
- `python test/inductor/test_nested_reduction.py`: 218/218
- `python -m py_compile ...`
- `git diff --check ...`
- Three adversarial reviews after the fix found no must-fix issues.

Repo lint via `spin` was not run because `spin` was unavailable in this checkout
and no `.venv` was present.

## What remains as future work?

The current fix is conservative and correct: split unsafe consumers into another
kernel. A more aggressive future design could route in-kernel consumers of a
half output to the in-register value, or emit half stores before dependents.
That would allow more fusion, but it needs explicit codegen support rather than
scheduler legality pretending the current tail-store ordering is ordinary.
