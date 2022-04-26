import torch
from typing import Dict, Union
from torch.testing import FileCheck
from torch._C import parse_ir


class SubgraphRewriterWrapper(object):
    def __init__(self):
        self.rewriter = torch._C.SubgraphRewriter()
        self.dummy_name_count = 0
        self.dummy_nodes_replacement_map = {}
        self.dummy_nodes_pattern_map = {}

    def register_rewrite_pattern(
        self, pattern: str, replacement: Union[str, torch._C.Graph], value_name_pair=[]
    ):
        if isinstance(replacement, torch._C.Graph):
            dummy_name = "prim::dummy_node_" + str(self.dummy_name_count)
            self.dummy_nodes_replacement_map[dummy_name] = replacement
            self.dummy_nodes_pattern_map[dummy_name] = pattern
            mutated_graph = replacement.copy()
            self.dummy_name_count += 1
            assert len(list(mutated_graph.nodes())) == 1
            node = next(mutated_graph.nodes())
            node.replaceWithNewSymbol(dummy_name)
            node.destroy()
            replacement = str(mutated_graph)
        self.rewriter.register_rewrite_pattern(pattern, replacement, value_name_pair)

    def run_on_graph(self, graph, filters, insert_type_check_node=True):
        self.rewriter.run_on_graph(graph, filters)
        for name, pattern in self.dummy_nodes_replacement_map.items():
            original_pattern = self.dummy_nodes_pattern_map[name]
            original_pattern_graph = parse_ir(original_pattern)
            for node in graph.findAllNodes(name):
                with graph.insert_point_guard(node):
                    outputs = graph.insertGraph(pattern, list(node.inputs()))
                    assert len(outputs) > 0 and len(outputs) == len(list(node.outputs()))
                    new_node = outputs[0].node()
                    node.replaceAllUsesWith(new_node)
                    node.destroy()
                    if insert_type_check_node:
                        new_node.g_("Subgraph", original_pattern_graph)
                        torch._C._jit_insert_type_check_node(new_node)
                        new_node.removeAttribute("Subgraph")

def foo(x):
    return x + torch.relu(torch.mm(x, x))

foo_scripted = torch.jit.script(foo)
# dont fuse with nnc
torch._C._jit_set_texpr_fuser_enabled(False)

pattern = r"""
graph(%original_input.1 : Tensor,
      %original_input.2 : Tensor):
  %z.1 : Tensor = aten::mm(%original_input.1, %original_input.2)
  %6 : Tensor = aten::relu(%z.1)
  return (%6)"""

count = 0

@torch.jit.ignore
def python_func(x, y):
    global count
    count += 1
    return torch.relu_(torch.mm(x, y))

# TODO: automate wrapping of ignored function
@torch.jit.script
def wrapper_fn(x, y):
    return python_func(x, y)


rewriter = SubgraphRewriterWrapper()
rewriter.register_rewrite_pattern(pattern, wrapper_fn.graph)

def match_filter(match: torch._C.Match, name_map: Dict[str, torch._C.Value]):
    # import pdb; pdb.set_trace()
    profiled_values = match.values_map.values()
    all_complete = all(value.isCompleteTensor() for value in profiled_values)
    if not all_complete:
        return False
    all_cpu = all(
        value.type().device() == torch.device("cpu") for value in profiled_values
    )
    if not all_cpu:
        return False
    all_continguous = all(
        value.type().contiguous() == value.type() for value in profiled_values
    )
    if not all_continguous:
        return False
    return True

def rewrite_mm_relu_pass(graph):
    rewriter.run_on_graph(graph, [match_filter])

torch._C._jit_register_custom_post_past("mm_counter", rewrite_mm_relu_pass)
# proflie
inp = torch.rand([4, 4])
foo_scripted(inp)
# run
torch.testing.assert_close(foo_scripted(inp), foo(inp))
assert count == 1
# type guards should fail
foo(torch.rand([4, 4]).cuda())
assert count == 1

FileCheck().check("python_func").check("fallback").run(torch.jit.last_executed_optimized_graph())

@torch.jit.script
def fresh_func(x):
    return x + torch.relu(torch.mm(x, x))

# try running with dis-contiguous inputs - should fail match
fresh_func(torch.rand([4, 4]).T)
fresh_func(torch.rand([4, 4]).T)
assert count == 1

FileCheck().check_not("python_func").run(torch.jit.last_executed_optimized_graph())
