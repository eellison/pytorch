
import torch


cu = torch.jit.CompilationUnit('''
    def foo(x: List[int]):
        return 1 if len(x) == 0 else x[0] if len(x) == 1 else x[1]
''')
print(cu.foo([]))
