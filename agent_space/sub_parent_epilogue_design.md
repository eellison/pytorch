# Sub-parent nested-reduction epilogues

This patch extends nested-reduction epilogue fusion to consumers whose iteration
domain is a proper sub-domain of the reduction parent tile. The important
contract is not "half" specifically; it is that the scheduler proves how each
consumer read maps back to the original parent source without materializing a
compiler-created intermediate in HBM.

Currently there are two supported source layouts:

1. `INTERLEAVED`: parent axis is split as `child, lane`, so
   `parent_r = factor * child_r + lane`. This is the NVFP4 even/odd packing
   layout.
2. `CONTIGUOUS`: parent axis is split as `lane, child`, so
   `parent_r = lane * child_extent + child_r`. This is the RMSNorm
   `chunk(2)`/`chunk(4)` SwiGLU-style layout.

The scheduler records `source_layouts: dict[name, SubParentSourceLayout]` in the
epilogue plan. That layout kind is load-bearing: codegen uses it to pick the
right in-register materialization, and rejects a source name if different
consumers require different layouts.

There are two codegen forms:

1. Persistent reductions materialize sub-parent source reads from the live
   parent tile. This is the single-load form: original inputs are loaded once,
   the reduced value is computed, and sub-parent consumers reuse the live values.
2. Looped reductions emit a second in-kernel sub-parent loop. This may reload
   original inputs, but it still keeps compiler-created intermediates inside the
   kernel: no normalized tensor, scale tensor, gate/up tensor, or other fused
   intermediate is stored to HBM and then loaded back.

The implementation intentionally accepts only the source mappings we can prove:
static power-of-two parent reduction sizes, power-of-two sub-parent factors, and
R-axis grouped reductions. Dynamic sub-parent shapes are left as follow-up work.
Unsupported cases fall back instead of storing fused intermediate values.

Future generalization should add new source-layout proofs and materializers
behind the same `SubParentSourceLayout` abstraction. It should not add another
parallel name set such as "split half source"; the layout proof must stay
attached to the source name through legality and codegen.
