from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401
import torch
from torch.testing._internal.jit_utils import JitTestCase, enable_cpu_fuser, disable_autodiff_subgraph_inlining
torch._C._jit_override_can_fuse_on_cpu(True)
import torchvision.models as models
import time
with enable_profiling_mode():

    class Mod(torch.nn.Module):
        def forward(self, x):
            # if x.size() == [1, 2, 3, 4]:
            #     return x * x * x * x * x
            # else:
            return x * x * x * x * x * x * x

    m = torch.jit.script(Mod())
    start_time = time.time()
    m(torch.randn([4, 3, 224, 224], requires_grad=True))
    # your code
    elapsed_time = time.time() - start_time
    print(elapsed_time)

    start_time = time.time()

    m(torch.randn([4, 3, 224, 224], requires_grad=True))
    elapsed_time = time.time() - start_time
    print("Elapsed time 2", elapsed_time)
    g = torch.jit.last_executed_optimized_graph()
    import pdb; pdb.set_trace()

    print(m.graph)
        # with disable_autodiff_subgraph_inlining():
        #     @torch.jit.script
        #     def foo(x, iters: int):
        #         ht = x[0]
        #         for k in range(iters):
        #             ht = ht + k
        #         return ht

        #     foo(torch.rand(5, 5), 10)
        #     foo(torch.rand(5, 5), 12)
        #     foo(torch.rand(5, 5), 14)
        #     print()
