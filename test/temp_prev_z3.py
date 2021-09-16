import torch

# graph taken from partially evaluated the shape graph of nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
specialized_graph = """
def max_pool2d(input: List[int],
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    ceil_mode: bool) -> List[int]:
  _0 = "AssertionError: stride should not be zeero"
  if torch.eq(2, len(dilation)):
    pass
  else:
    raise Exception("Input Shape Augment")
  if torch.eq(2, len(padding)):
    pass
  else:
    raise Exception("Input Shape Augment")
  if torch.eq(2, len(stride)):
    pass
  else:
    raise Exception("Input Shape Augment")
  if torch.eq(2, len(kernel_size)):
    pass
  else:
    raise Exception("Input Shape Augment")
  if torch.eq(4, len(input)):
    pass
  else:
    raise Exception("Input Shape Augment")
  kH = kernel_size[0]
  _1 = kernel_size[1]
  _2 = stride[0]
  _3 = stride[1]
  padH = padding[0]
  _4 = padding[1]
  dilationH = dilation[0]
  _5 = dilation[1]
  _6 = input[-4]
  nInputPlane = input[-3]
  inputHeight = input[-2]
  inputWidth = input[-1]
  if torch.ne(_2, 0):
    pass
  else:
    raise Exception("AssertionError: ")
  _7 = torch.add(inputHeight, padH)
  _8 = torch.add(_7, padH)
  _9 = torch.mul(dilationH, torch.sub(kH, 1))
  _10 = torch.sub(torch.sub(_8, _9), 1)
  if ceil_mode:
    _11 = torch.sub(_2, 1)
  else:
    _11 = 0
  _12 = (torch.add(_10, _11) // _2)
  outputSize = torch.add(_12, 1)
  if ceil_mode:
    if torch.ge(torch.mul(_12, _2), _7):
      outputSize0 = _12
    else:
      outputSize0 = outputSize
    outputHeight = outputSize0
  else:
    outputHeight = outputSize
  if torch.ne(_3, 0):
    pass
  else:
    raise Exception("AssertionError: ")
  _13 = torch.add(inputWidth, _4)
  _14 = torch.sub(torch.add(_13, _4), torch.mul(_5, torch.sub(_1, 1)))
  _15 = torch.sub(_14, 1)
  if ceil_mode:
    _16 = torch.sub(_3, 1)
  else:
    _16 = 0
  _17 = (torch.add(_15, _16) // _3)
  outputSize1 = torch.add(_17, 1)
  if ceil_mode:
    if torch.ge(torch.mul(_17, _3), _13):
      outputSize2 = _17
    else:
      outputSize2 = outputSize1
    outputWidth = outputSize2
  else:
    outputWidth = outputSize1
  if torch.gt(_1, 0):
    _18 = torch.gt(kH, 0)
  else:
    _18 = False
  if _18:
    pass
  else:
    raise Exception("AssertionError: ")
  if torch.gt(_3, 0):
    _19 = torch.gt(_2, 0)
  else:
    _19 = False
  if _19:
    pass
  else:
    raise Exception("AssertionError: ")
  if torch.gt(dilationH, 0):
    _20 = torch.gt(_5, 0)
  else:
    _20 = False
  if _20:
    pass
  else:
    raise Exception("AssertionError: ")
  if torch.ne(input[1], 0):
    valid_dims = torch.ne(input[2], 0)
  else:
    valid_dims = False
  if valid_dims:
    _21 = torch.ne(input[3], 0)
  else:
    _21 = False
  if _21:
    pass
  else:
    raise Exception("AssertionError: ")
  if torch.ge((_1 // 2), _4):
    _23 = torch.ge((kH // 2), padH)
    _22 = _23
  else:
    _22 = False
  if _22:
    pass
  else:
    raise Exception("AssertionError: ")
  if torch.ge(outputWidth, 1):
    _24 = torch.ge(outputHeight, 1)
  else:
    _24 = False
  if _24:
    pass
  else:
    raise Exception("AssertionError: ")
  _25 = [_6, nInputPlane, outputHeight, outputWidth]
  return _25
"""

from z3 import *

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
	exception_map = {}

	for exception in graph.findAllNodes("prim::RaiseException"):
		if exception.input().toIValue() == "Input Shape Augment":
			continue
		to_process.append(exception)


	for i in range(len(exception_map)):
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
            # for now add this constraint to gen inputs with actual data
            s.add(names_to_z3[value.debugName()][i] >= 1)
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
    assert s.check()
    return s


s = convert_graph_to_z3(torch.jit.CompilationUnit(specialized_graph).max_pool2d.graph)
print(s.model())
