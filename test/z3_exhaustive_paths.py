import torch
from z3 import *
from typing import List

exception_map = {}

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

    to_process = []


    for exception in graph.findAllNodes("prim::RaiseException"):
        if exception.input().toIValue() == "Input Shape Augment":
            continue
        to_process.append(exception)

    for curr_exception in to_process:
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
                    if i == 0:
                        for i in range(li_len):
                            # for now add this constraint to gen inputs with actual data
                            s.add(names_to_z3[value.debugName()][i] >= 0)
                else:
                    raise Exception("Unhandled input type with uses", value.type())

        def add_block(block):
            for node in block.nodes():
                if node.kind() == "prim::RaiseException":
                    owning_node = node.owningBlock().owningNode()
                    # we're throwing in the graph block, graph is unsat
                    if owning_node is None:
                        s.add(0 == 1)
                        continue

                    assert node in to_process

                    # if we have already constrained an exception so far,
                    # leave this exception unconstrained
                    # global exception_found
                    # if exception_found:
                    #     continue
                    throwing = False
                    if node == curr_exception:
                        throwing = True

                    blocks = list(owning_node.blocks())
                    node_input = names_to_z3[owning_node.input().debugName()]
                    if node.owningBlock() == blocks[0]:
                        # if we're throwing in the true branch, if node input must be 0
                                # unless we are constraining this to be the exception node
                        s.add(node_input == throwing)
                    elif node.owningBlock() == blocks[1]:
                        # otherwise if node input must be 1
                        s.add(node_input == (not throwing))
                    else:
                        assert False
                    if throwing:
                        return True
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
                inputs_z3 = [names_to_z3[inp.debugName()] for inp in inputs]
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
                    if add_block(blocks[0]):
                        return True
                    if add_block(blocks[1]):
                        return True
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
        # import pdb; pdb.set_trace()
        is_sat = s.check()
        model = s
        # import pdb; pdb.set_trace()
        # assert str(is_sat) == "sat"
        model = s.model()
        input_values = []
        for i, inp in enumerate(list(graph.inputs())):
            if not len(inp.uses()):
                continue
            z3_val = names_to_z3[inp.debugName()]
            if i in input_len_mappings:
                value = []
                for j in range(input_len_mappings[i]):
                    try:
                        val = model[z3_val[j]]
                        if val is None:
                            value.append(0)
                        else:
                            value.append(model[z3_val[j]].as_long())
                    except Exception as e:
                        val = model[z3_val[j]]
                        import pdb; pdb.set_trace()
                        print(val)
                input_values.append(value)
                continue
            try:
                val = model[z3_val]
                # TODO: not generating value for bool input... strange
                if val is None:
                    assert inp.type() == torch._C.BoolType.get()
                    input_values.append(True)
                else:
                    assert inp.type() == torch._C.BoolType.get()
                    input_values.append(is_true(val))
            except Exception as e:
                import pdb; pdb.set_trace()
                val = model[z3_val[j]]
                import pdb; pdb.set_trace()
                print(val)

        # import pdb; pdb.set_trace()
        exception_map[curr_exception] = input_values

    # for key, value in exception_map.items():
    #     import pdb; pdb.set_trace()
    #     print(key)
    #     print(value)




    import pdb; pdb.set_trace()
    return exception_map.values()

@torch.jit.script
def foo(input, list_int: List[int]):
	return input.expand(list_int)

shape_compute_graph = torch._C._jit_get_shape_compute_graph_for_node(foo.graph.findNode("aten::expand"))
# import pdb; pdb.set_trace()
torch._C._augment_with_length(shape_compute_graph, 0, 4)
for i in range(1, 2):
	torch._C._augment_with_length(shape_compute_graph, i, 4)
print(shape_compute_graph)
# import pdb; pdb.set_trace()

changed = True
while changed:
	changed = torch._C._jit_pass_shape_graph_cleanup_passes(shape_compute_graph)
print(shape_compute_graph)

output = convert_graph_to_z3(shape_compute_graph)

for out in output:
    # import pdb; pdb.set_trace()
    inputs = [torch.rand(*out[0])] + out[1:]
    try:
        inputs[0].expand(inputs[1])
    except Exception as e:
        print(e)

# import pdb; pdb.set_trace()
# print(output)
