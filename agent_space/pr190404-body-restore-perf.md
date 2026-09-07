Stack from [ghstack](https://github.com/ezyang/ghstack/tree/0.15.0) (oldest at bottom):
* #190596
* #190595
* #190594
* #190593
* __->__ #190404
* #190403
* #190402

Use Triton's PropagateNan.ALL operations in the shared min/max helpers. This preserves NaN and signed-zero behavior while allowing SM80+ to replace compare/select sequences with native instructions.

Add direct elementwise and reduction coverage for NaNs, infinities, and signed zero.

Authored by Codex.

## Local B200 performance

Measured the isolated two-file delta from this PR on top of the scalar online-softmax Phase B tree (`0e40fb08157`).

Method: 4x NVIDIA B200, 4 workers/GPU, fresh caches, coordinate descent disabled, exact shape-aligned A/B. The head completed 4,977/4,977 corpus points with zero failures in 961.7s; supplemental model/Opacus points produced 5,127 exactly matched kernel-shape pairs.

- Fusible-kernel geomean: **+0.288%**
- Shared min/max helper-family geomean: **about +0.24%**
- Projected model E2E geomean: **+0.138% (provisional)**

The model projection is provisional because the flash-attention-backward extern/materialization accounting needs correction; the shape-aligned kernel result is unaffected and is the result used for this PR. This is a small additive win on top of Phase B. A full coordinate-descent-enabled sweep has not been run.

Validation on the combined tree: new helper tests 3/3 passed; full online-softmax tests 60/60 passed; `compileall` and `git diff --check` passed.
