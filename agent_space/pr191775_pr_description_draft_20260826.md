# PR #191775 description draft

Human review is required before using this text on GitHub. Suggested human
context: "I reviewed the implementation and the validation below. The quoted
draft summarizes the final design."

> AI-generated draft:
>
> Extend staged reduction planning and Triton codegen to fuse MXFP6's 4-to-3
> packing epilogue. Four full-resolution E2M3 values produce three packed
> bytes, so the epilogue runs in a derived sub-parent iteration domain and is
> replayed once per output lane.
>
> Fusion is fail-closed. The planner records exact producer/read access pairs
> for cross-domain dependencies, preserves the parent X/R boundary, and
> rejects any residual dependency it cannot prove. Existing grouped-stage
> equivalence remains scoped to the inherited nested-reduction path. Codegen
> rebuilds the plan from the final fused topology rather than relying on a
> cached fusion-time plan.
>
> For looped reductions, full-resolution producers needed by the packing
> epilogue are placed in the final reduction pass. This avoids materializing
> and reloading the source while keeping the ordinary node-schedule machinery
> responsible for reduction-loop boundaries. Persistent kernels retain the
> values directly.
>
> The staged representation supports ordered output groups, recursive
> power-of-two splitting, and derived-family mask propagation. An already
> formed standalone staged reduction may also append an exactly matched
> parent-stage consumer, allowing an aligned scale preshuffle to remain in the
> same kernel. Shifted or otherwise unproved scale accesses still decline.
>
> The resulting graph composes with a full-resolution
> `inline_asm_elementwise(..., pack=2)` E2M3 conversion. Local B200 validation
> produced one nested kernel with one native conversion site for ordinary,
> looped, persistent, odd-tail, and aligned DCN-preshuffle cases.
>
> Test Plan:
>
> ```bash
> conda run --no-capture-output -n pytorch-3.12 \
>   python test/inductor/test_inductor_scheduler.py
> conda run --no-capture-output -n pytorch-3.12 \
>   python test/inductor/test_nested_reduction.py
> conda run --no-capture-output -n pytorch-3.12 spin quicklint
> git diff --check HEAD
> ```
>
> Local results: 108 scheduler tests passed with 6 skipped; 395 nested-reduction
> tests passed with 8 skipped. Native MXFP6 was exact and faster than the
> extracted AITER comparison across four representative shapes. The aligned
> 2048x3072 DCN-preshuffle case emitted one kernel with 32 registers and no
> spills.
>
> Authored with assistance from Codex.
