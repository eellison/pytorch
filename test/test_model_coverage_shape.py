1
# run with PYTORCH_JIT_TYPE_VERBOSITY=4 python example_model.py

import torch
import torchvision

models = torchvision.models

model_pairs = (
    # ("mobilenet_v3_small", models.mobilenet_v3_small),
    ("mobilenet_v2", models.mobilenet_v2()),
    # ("inception_v3", models.inception_v3()),
    # ("resnet", models.resnet18()),
    # ("shufflenet_v2_x1_0", models.shufflenet_v2_x1_0()),
    # ("deeplab", models.segmentation.deeplabv3_resnet50()), # TODO: need a couple more ops
)

def compute_number_of_sym_shapes(graph):
    num_symbols = set()

    for node in graph.nodes():
        vals = list(node.outputs()) + list(node.inputs())
        for val in vals:
            if "Tensor" not in str(val.type()):
                continue
            ss = val.type().symbolic_sizes()
            for shape in ss:
                if shape < 0:
                    num_symbols.add(shape)
    print("Number of symbols", len(num_symbols))

for name, model in model_pairs:

    model_frozen = torch.jit.freeze(torch.jit.script(model.eval()))

    # until https://github.com/pytorch/pytorch/issues/65643 lands to clean up control flow..

    inps = list(model_frozen.graph.inputs())
    # None creates a new dynamic dimension,
    # alternative inputs:
    # [None, 3, 255, 255] - batch dimension not specified
    # sym_shape = torch._C._new_symbolic_shape_symbol(); [1, 3, sym_shape, sym_shape] - same width/height dimension
    inps[1].setType(inps[1].type().with_sizes([None, None, None, None]))

    torch._C._jit_pass_canonicalize_graph_fuser_ops(model_frozen.graph)
    torch._C._jit_pass_propagate_shapes_on_graph(model_frozen.graph)
    torch._C._jit_pass_canonicalize_for_shape_analysis(model_frozen.graph)
    # torch._C._jit_pass_propagate_shapes_on_graph(model_frozen.graph)
    torch._C._jit_pass_canonicalize_for_shape_analysis(model_frozen.graph)
    torch._C._jit_pass_peephole(model_frozen.graph)
    torch._C._jit_pass_constant_propagation(model_frozen.graph)

    # import pdb; pdb.set_trace()

    # unused tuple output, manipulate so we are just returning tensor
    if name == "inception_v3":
        output = next(model_frozen.graph.outputs())
        assert output.node().kind() == "prim::TupleConstruct"
        model_frozen.graph.eraseOutput(0)
        model_frozen.graph.registerOutput(next(output.node().inputs()))
        model_frozen.graph.findNode("aten::warn").destroy()
        torch._C._jit_pass_dce(model_frozen.graph)

    if name == "deeplab":
        # remove error handling
        for node in model_frozen.graph.findAllNodes("prim::RaiseException"):
            node.destroy()
        torch._C._jit_pass_dce(model_frozen.graph)
        torch._C._jit_pass_remove_mutation(model_frozen.graph)
        torch._C._jit_pass_peephole(model_frozen.graph)
        # TODO - need to add handling of a couple ops, doen't quite work yet

    if name == "deeplab":
        # rewrite slice(size) to [size(0), size(1)]
        output = next(model_frozen.graph.outputs())
        assert output.node().kind() == "prim::DictConstruct"
        # dict with a single element
        assert len(list(output.node().inputs())) == 2
        assert len(list(output.uses())) == 1
        tensor_input = list(output.node().inputs())[1]
        model_frozen.graph.eraseOutput(0)
        model_frozen.graph.registerOutput(tensor_input)
        torch._C._jit_pass_peephole(model_frozen.graph)
        torch._C._jit_pass_dce(model_frozen.graph)

    torch._C._jit_pass_canonicalize_for_shape_analysis(model_frozen.graph)

    # inps[1].setType(inps[1].type().with_sizes([None, 3, 255, 255]))
    torch._C.Graph.set_global_print_source_ranges(False)
    shape_compute_graph = torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(model_frozen.graph)
    g = shape_compute_graph.partial_eval_shape_graph()
    print("model GRAPH \n\n")
    print(model_frozen.graph)
    print("Shape Compute Graph \n\n")
    # here is the encoding of shape arithmetic from the input, represnted as TS graph
    for node in g.findAllNodes("prim::RaiseException"):
        node.destroy()
    torch._C._jit_pass_dce(g)
    print(g)
    print("Mapping from Sym Dim to Shape Compute Graph Output")
    for key, value in shape_compute_graph.graph_output_to_symbolic_shape_dim().items():
        print(str(value) + " :", key)

    # model_frozen.graph.eraseInput(0)
    # with torch.jit._hide_source_ranges():
    #     out = (torch._C._jit_trace_graph(model_frozen.graph, (torch.rand([1, 3, 256, 256]),)))
    #     print(out)

    # to execute jit function it must have a single output
    g.makeMultiOutputIntoTuple()
    func = torch._C._create_function_from_graph("partial_eval_graph", g)
    print(func.code)
    print("Calculating dims from input", func([1, 3, 255, 255]))
