import torch

@torch.jit.script
def test_batch_mm(n: int):
    T3 = torch.zeros((n, n))
    T4 = torch.zeros((n, n))
    T5 = torch.zeros((n, n))
    T6 = torch.zeros((n, n))
    T7 = torch.zeros((n, n))
    T8 = torch.zeros((n, n))
    result = (
        torch.mm(T5, T6)
        + torch.mm(T7, T8)
    )
    return result

# FileCheck().check_count("aten::mm", 4, exactly=True).run(test_batch_mm.graph)
torch._C._jit_pass_batch_mm(test_batch_mm.graph)
# FileCheck().check_count("aten::mm", 4, exactly=True).check_not(
#     "prim::MMTreeReduce"
# ).run(test_batch_mm.graph)
print(test_batch_mm.graph)

