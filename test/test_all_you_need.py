import torch

# download model from https://drive.google.com/file/d/1FKNQ4TYBXU5ArhLr3a5szjAymqNz4Gnm/view?usp=sharing
model = torch.jit.load("./attention_is_all_you_need.pt")

froze = model
inputs = list(froze.graph.inputs())

# firs inp is batch size
inputs[1].setType(inputs[1].type().with_sizes([None, None]))
inputs[2].setType(inputs[1].type())

for node in froze.graph.findAllNodes("prim::Constant"):
    ival = node.output().toIValue()
    if not isinstance(ival, torch.Tensor):
        continue
    node.output().inferTypeFrom(ival)

torch._C._jit_pass_propagate_shapes_on_graph(froze.graph)
# import pdb; pdb.set_trace()
torch._C._jit_pass_canonicalize_for_shape_analysis(froze.graph)
# import pdb; pdb.set_trace()
compute = torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(froze.graph)

eval_g = compute.partial_eval_shape_graph()
mapping = compute.graph_output_to_symbolic_shape_dim()

[node.destroy() for node in eval_g.findAllNodes("prim::RaiseException")]
torch._C._jit_pass_dce(eval_g)
if True:
    # slightly cheating a bit here but this value will always be False
    # or the model throws
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Models.py#L45
    # x.size(1) > pos_table lenght will throw
    #
    # TODO: rely on compiler
    cons = eval_g.insertConstant(False)
    ge = eval_g.findNode("aten::ge")
    ge_inps = list(ge.inputs())
    assert ge_inps[1].toIValue() == 200
    assert ge_inps[0].node().kind() == "aten::__getitem__"
    cons.node().moveBefore(ge)
    ge.replaceAllUsesWith(cons.node())
    torch._C._jit_pass_constant_propagation(eval_g)

eval_g.makeMultiOutputIntoTuple()
func = torch._C._create_function_from_graph("partial_eval_graph", eval_g)
import pdb; pdb.set_trace()
print(func.code)
# print(eval_g)
# [node.destroy() for node in eval_g.findAllNodes("prim::RaiseException")]
# torch._C._jit_pass_dce(eval_g)
# print(eval_g)
# print(mapping)
# TODO: assumption that the input(1) size < 200
# gets clean graph