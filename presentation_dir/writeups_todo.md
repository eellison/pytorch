Notes on fusion improvement:

Loop reordering : note - in-tree landed test/inductor/test_loop_ordering.py
  → writeup: ./loop_reordering.md

Loop reindexing : note in-tree landed https://github.com/pytorch/pytorch/pull/176927
  → writeup: ./loop_reindexing.md

Index Inversion: see test/inductor/test_fp8.py def test_mx_fusion(self): you can use agent_space/test_reindex_deps.py to show real example
  → writeup: ./index_inversion.md

Inline ASM Integration - see test/higher_order_ops/test_inline_asm_elementwise.py - u can use you can use agent_space/test_reindex_deps.py to show real example
  → writeup: ./inline_asm.md

Mix Order Reduciton: test/inductor//test_mix_order_reduction.py MixOrderReductionTest
  → writeup: ./mix_order_reduction.md

Horizontal Fusion (Combo Kernels) - pytorch/test/inductor/test_combo_kernels.py (use combo_kernel_per_subkernel_blocks)
  → writeup: ./combo_kernels.md

Nested Reduction (different-size dependent reductions): test/inductor/test_nested_reduction.py
  → writeup: ./nested_reduction.md

Other Misc:

Helion Epilogue Fusion - see test_external_template_prologue_epilogue_fusion, https://github.com/pytorch/helion/pull/1324
  → writeup: ./helion_epilogue_fusion.md

https://github.com/pytorch/helion/issues/1346 ( we can just reference this issue, not actually do anything here)

Cute DSL Template Integration (Flex Attention) - lets just reference - i'll do it i suppose:

https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/



Potential Future Project:

Inductor codegen is too inflexible, and tightly coupled. SIMD IR -> Lower level, looped IR, post fusion -> codegen

Symmetric Memory Fusions / kernels. Unclear if through native codegen or template 

