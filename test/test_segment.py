import torch
model = torch.jit.load("/private/home/eellison/pytorch/xirp_20a.ptl")

from z3 import *

add_nodes = list(model.graph.findAllNodes("aten::add"))
for node in add_nodes:
    node_inputs = list(node.inputs())
    if len(list(node_inputs[0].uses())) == 1:
        node.replaceWithNewSymbol("aten::add_")
        node.destroy()
    elif len(list(node_inputs[1].uses())) == 1:
        node_inp = list(node.inputs())
        node.replaceInput(0, node_inputs[1])
        node.replaceInput(1, node_inputs[0])
        node.replaceWithNewSymbol("aten::add_")
        node.destroy()

def convert_graph_to_z3(graph):
    # we are going to make some modifications
    graph = graph.copy()
    input_len_mappings = {}
    input_set = list(graph.inputs())

    # TODO: maybe do from JIT level and refine list type of input
    def add_input_len_from_assertions(exception):
        owning_node = exception.owningBlock().owningNode()
        # for now only handle assertions whos If node is on graph block
        if not owning_node or owning_node.owningBlock().owningNode() is not None:
            return
        # handle assertions of the form != and ==
        for block_index, node_name in (1, "aten::eq"), (0, "aten::neq"):
            if exception.owningBlock() != list(owning_node.blocks())[block_index]:
                continue
            if owning_node.input().node().kind() != node_name:
                continue
            eq_node_inputs = list(owning_node.input().node().inputs())
            for i in range(2):
                non_const_value = eq_node_inputs[1 - i]
                if non_const_value.node().kind() != "aten::len":
                    continue
                len_input = non_const_value.node().input()
                if isinstance(eq_node_inputs[i].toIValue(), int) and len_input in input_set:
                    input_len_mappings[len_input.offset()] = eq_node_inputs[i].toIValue()
                    exception.destroy()
                    return

    for exception in graph.findAllNodes("prim::RaiseException"):
        add_input_len_from_assertions(exception)

    torch._C._jit_pass_dce(graph)
    s = Solver()
    names_to_z3 = {}
    value_to_len = {}

    # only handling Booleans/Integers & Fixed Length Lists initially
    for i, value in enumerate(graph.inputs()):
        if len(value.uses()):
            if value.type() == torch._C.IntType.get():
                names_to_z3[value.debugName()] = Int(value.debugName())
            elif value.type() == torch._C.BoolType.get():
                names_to_z3[value.debugName()] = Bool(value.debugName())
            elif isinstance(value.type(), torch._C.ListType):
                contained_type = value.type().getElementType()
                if contained_type != torch._C.IntType.get() and contained_type != torch._C.BoolType.get():
                    raise Exception("Unhandled input type with uses", value.type())
                if i in input_len_mappings:
                    li_len = input_len_mappings[i]
                else:
                    raise Exception("not enough len information on input list")
                names_to_z3[value.debugName()] = IntVector(value.debugName(), li_len)
                value_to_len[value.debugName()] = li_len
                for i in range(li_len):
                    # tensors must be >=0
                    s.add(names_to_z3[value.debugName()][i] >= 0)
                # term = names_to_z3[value.debugName()][0]
                # for i in range(1, li_len):
                #     term = term * names_to_z3[value.debugName()][i]
                # s.add(term >= 4000)
            else:
                raise Exception("Unhandled input type with uses", value.type())

    # forced_exception
    # for an exception to occur,

    def add_block(block):
        for node in block.nodes():
            if node.kind() == "prim::RaiseException":
                owning_node = node.owningBlock().owningNode()
                # we're throwing in the graph block, graph is unsat
                if owning_node is None:
                    s.add(0 == 1)
                    continue

                blocks = list(owning_node.blocks())
                node_input = names_to_z3[owning_node.input().debugName()]
                if node.owningBlock() == blocks[0]:
                    # if we're throwing in the true branch, if node input must be 0
                    s.add(node_input == False)
                elif node.owningBlock() == blocks[1]:
                    # otherwise if node input must be 1
                    s.add(node_input == True)
                else:
                    assert False
                continue

            map_integer_bool_binary_op = {
              "aten::add": lambda a, b: a + b,
              "aten::sub": lambda a, b: a - b,
              "aten::mul": lambda a, b: a * b,
              "aten::floordiv": lambda a, b: a / b,
              "aten::remainder": lambda a, b: a % b, # todo: fix for negative values
              "aten::ge": lambda a, b: (a - b >= 0),
              "aten::gt": lambda a, b: a > b,
              "aten::lt": lambda a, b: (b - a > 0),
              "aten::eq": lambda a, b: And(a - b >=0, b - a >= 0),
              "aten::ne": lambda a, b: a != b,
            }

            def construct_z3_val(val):
                if val.type() == torch._C.IntType.get():
                    z3_val = Int(val.debugName())
                elif val.type() == torch._C.BoolType.get():
                    z3_val = Bool(val.debugName())
                else:
                    assert False
                names_to_z3[val.debugName()] = z3_val
                return z3_val

            inputs = list(node.inputs())
            try:
                inputs_z3 = [names_to_z3[inp.debugName()] for inp in inputs]
            except:
                import pdb; pdb.set_trace()
            if node.kind() == "prim::Constant":
                if node.output().type() == torch._C.IntType.get() or node.output().type() == torch._C.BoolType.get():
                    ival = node.output().toIValue()
                    ival = bool(ival) if node.output().type() == torch._C.BoolType.get() else ival
                    s.add(construct_z3_val(node.output()) == ival)
            elif node.kind() == "aten::__getitem__":
                z3_val = Int(node.output().debugName())
                z3_val = construct_z3_val(node.output())
                li_len = value_to_len[inputs[0].debugName()]
                index = inputs[1].toIValue()
                index = index if index >= 0 else index + li_len
                s.add(z3_val == inputs_z3[0][index])
                names_to_z3[node.output().debugName()] = z3_val
            elif node.kind() == "aten::__not__":
                z3_val = construct_z3_val(node.output())
                s.add(z3_val ==  Not(inputs_z3[0]))
            elif node.kind() in map_integer_bool_binary_op:
                z3_val = construct_z3_val(node.output())
                s.add(z3_val == map_integer_bool_binary_op[node.kind()](inputs_z3[0], inputs_z3[1]))
            elif node.kind() == "prim::If":
                blocks = list(node.blocks())
                add_block(blocks[0])
                add_block(blocks[1])
                n_inp = inputs_z3[0]
                for node_output, true_output, false_output in zip(node.outputs(), blocks[0].outputs(), blocks[1].outputs()):
                    z3_val = construct_z3_val(node_output)
                    true_z3_val = names_to_z3[true_output.debugName()]
                    false_z3_val = names_to_z3[false_output.debugName()]
                    s.add(z3_val == If(n_inp, true_z3_val, false_z3_val)) # (n_inp * true_z3_val) + (1 - n_inp) * false_z3_val)
                    names_to_z3[node_output.debugName()] = z3_val
            if node.kind() == "prim::ListConstruct":
                vec_value = IntVector(node.output().debugName(), len(inputs))
                names_to_z3[node.output().debugName()] = vec_value
                for i in range(len(inputs_z3)):
                    s.add(vec_value[i] == inputs_z3[i])

    add_block(graph)
    assert s.check()
    return s


# out = model(torch.rand([1, 4, 256, 256]))
inps = list(model.graph.inputs())
inps[1].setType(inps[1].type().with_sizes([1, 4, None, None]))
torch._C._jit_pass_constant_propagation(model.graph)
shape_compute_graph = torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(model.graph)
partial_eval = shape_compute_graph.partial_eval_shape_graph()
# for node in partial_eval.findAllNodes("prim::RaiseException"):
#     node.destroy()
# torch._C._jit_pass_dce(partial_eval)

# with torch.jit._hide_source_ranges():
# mapping = shape_compute_graph.graph_output_to_symbolic_shape_dim()
str_constant = partial_eval.insertConstant('empty_assertion')
str_constant.node().moveBefore(next(partial_eval.nodes()))
# for node in partial_eval.findAllNodes("prim::RaiseException"):
#     node.replaceInputWith(node.input(), str_constant)

# print(mapping)/
# for val, shape in mapping.items():
#     if "prim::If" in str(val) or "prim::If" in str(shape):
#         # import pdb; pdb.set_trace()
#         print(val, shape)

torch._C._augment_with_length(partial_eval, 0, 4)
modeled_z3 = convert_graph_to_z3(partial_eval)
import pdb; pdb.set_trace()
print(partial_eval)
partial_eval.makeMultiOutputIntoTuple()
func = torch._C._create_function_from_graph("forward", partial_eval)
# import pdb; pdb.set_trace()
print(func([1, 3, 256, 256]))
# print(model.graph)
model.graph.eraseInput(0)
with torch.jit._hide_source_ranges():
    print(model.graph)
    print(func.code)
    print(torch._C._jit_trace_graph(model.graph, (torch.rand([1, 4, 256, 256]),)))



# i = 0

# # while i < len(list(partial_eval.outputs())):
# #     outs = list(partial_eval.outputs())
# #     if "803" in str(outs[i].debugName()):
# #         i += 1
# #     else:
# #         partial_eval.eraseOutput(i)
# # torch._C._jit_pass_dce(partial_eval)
# # # import pdb; pdb.set_trace()
# # print(func.code)
# # print(model.graph)
#     # print(model.graphs)

