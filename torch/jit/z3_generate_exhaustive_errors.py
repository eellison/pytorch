import torch
from typing import List, Union, Any

from z3 import *

def generate_exception_inputs(graph, node_or_graph):
    # Takes a Node or Graph and its accompaning partial eval'd shape graph and generates
    # all possible exception inputs
    exception_map = {}
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

        node_inputs = list(node_or_graph.inputs())
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
                        raise Exception("not enough len information on input list:" + str(i))
                    names_to_z3[value.debugName()] = IntVector(value.debugName(), li_len)
                    value_to_len[value.debugName()] = li_len
                    if node_inputs[i].type() == torch._C.TensorType.get():
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
        is_sat = s.check()
        assert str(is_sat) == "sat"
        model = s.model()
        input_values = generate_input_values(graph, names_to_z3, model, input_len_mappings)

        exception_map[curr_exception] = input_values

    return translate_z3_vals_to_inputs(exception_map.values(), node_or_graph)

def gatherThrowingBlocks(block, throwing_blocks):
    for n in block.nodes():
        assert n.kind() != "prim::Loop"
        if n.kind() == "prim::RaiseException":
            throwing_blocks.add(block)
        if n.kind() == "prim::If":
            blocks = list(n.blocks())
            gatherThrowingBlocks(blocks[0], throwing_blocks)
            gatherThrowingBlocks(blocks[1], throwing_blocks)
            if blocks[0] in throwing_blocks and blocks[1] in throwing_blocks:
                throwing_blocks.add(block)
        if n.kind() == "prim::Loop":
            assert False, str(n)

def generate_input_values(graph, names_to_z3, model, input_len_mappings):
    input_values = []
    for i, inp in enumerate(list(graph.inputs())):
        z3_val = names_to_z3[inp.debugName()]
        if i in input_len_mappings:
            value = []
            for j in range(input_len_mappings[i]):
                index = model[z3_val[j]]
                if index is None:
                    value.append(0)
                else:
                    value.append(model[z3_val[j]].as_long())
            input_values.append(value)
            continue
        val = model[z3_val]
        # TODO: not generating value for bool input... strange
        if val is None:
            if inp.type() == torch._C.BoolType.get():
                input_values.append(True)
            else:
                assert inp.type() == torch._C.IntType.get()
                input_values.append(0)
        else:
            if inp.type() == torch._C.BoolType.get():
                input_values.append(is_true(val))
            else:
                assert inp.type() == torch._C.IntType.get()
                input_values.append(val.as_long())
    return input_values

def is_true_block(block):
    owning_node = block.owningNode()
    assert owning_node
    if_blocks = list(owning_node.blocks())
    if block is if_blocks[0]:
        return True
    elif block is if_blocks[1]:
        return False
    else:
        assert False

def generate_all_paths(graph, node_or_graph, force_non_empty_tensors=True):
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
    throwing_blocks = set()
    gatherThrowingBlocks(graph, throwing_blocks)
    non_throwing_blocks = set()
    for if_node in graph.findAllNodes("prim::If"):
      for block in if_node.blocks():
        if block not in throwing_blocks:
          non_throwing_blocks.add(block)

    outputs = []
    # import pdb; pdb.set_trace()

    while len(non_throwing_blocks) != 0:
      constrained_block = non_throwing_blocks.pop()

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
                  if list(node_or_graph.inputs())[i].type().isSubtypeOf(torch._C.TensorType.get()):
                    for j in range(li_len):
                        # tensors must be >=0
                        s.add(names_to_z3[value.debugName()][j] >= (1 if force_non_empty_tensors else 0))
              else:
                  raise Exception("Unhandled input type with uses", value.type())

      # forced_exception
      # for an exception to occur,

      def add_block(block):
          if block is constrained_block:
            # if owning_node.input.debugName()
            owning_node = block.owningNode()
            assert owning_node
            if_blocks = list(owning_node.blocks())
            node_input = names_to_z3[owning_node.input().debugName()]
            if block is if_blocks[0]:
              s.add(node_input == True)
            elif block is if_blocks[1]:
              s.add(node_input == False)
            else:
              assert False

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
      # it's possible we constrained a path to be taken that is unsat
      if str(s.check()) != "sat":
        continue
      modeled = s.model()
      # now that we have a concrete path taken, we can remove all of the blocks that have
      # been taken from this path
      def path_taken(block):
        owning_node = block.owningNode()
        if owning_node is None:
          return True
        node_input = names_to_z3[owning_node.input().debugName()]
        z3_val = modeled[node_input]
        if is_true_block(block):
          return is_true(z3_val) and path_taken(owning_node.owningBlock())
        else:
          return is_false(z3_val) and path_taken(owning_node.owningBlock())

      non_throwing_blocks_list = list(non_throwing_blocks)
      for non_throwing_block in non_throwing_blocks_list:
        if path_taken(non_throwing_block):
          non_throwing_blocks.remove(non_throwing_block)

      input_values_gen = generate_input_values(graph, names_to_z3, modeled, input_len_mappings)
      outputs.append(input_values_gen)

    return translate_z3_vals_to_inputs(outputs, node_or_graph)

def translate_z3_vals_to_inputs(z3_outputs: List[List[Any]], node_or_graph) -> List[List[Any]]:
    # since the shape functions translate Tensor -> List[int], we need to look at the graph or
    # node input type to retranslate back those inputs to Tensors
    translated_inputs = []
    node_or_graph_inputs = list(node_or_graph.inputs())
    for out in z3_outputs:
        inputs = []
        for i, elem in enumerate(out):
            if node_or_graph_inputs[i].type() == torch._C.TensorType.get():
                inputs.append(torch.rand(*elem))
            else:
                inputs.append(elem)
        translated_inputs.append(inputs)
    return translated_inputs


