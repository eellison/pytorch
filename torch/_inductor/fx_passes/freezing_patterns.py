import functools

import torch
from torch._inductor.compile_fx import fake_tensor_prop
from ..._dynamo.utils import counters

from typing import List
from torch._inductor.utils import is_ones, is_zeros
from .. import config
from ..pattern_matcher import (
    _return_true,
    CallFunction,
    Ignored,
    MULTIPLE,
    inference_graph,
    init_once_fakemode,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
    register_replacement,
    stable_topological_sort,
)

aten = torch.ops.aten

# First pass_patterns[0] are applied, then [1], then [2]
pass_patterns = [
    PatternMatcherPass(),
    PatternMatcherPass(),
    PatternMatcherPass(),
]

binary_folding_pass = PatternMatcherPass()


def freezing_passes(gm: torch.fx.GraphModule, aot_example_inputs):
    """
    Passes that are applied to the graph to freeze pass.
    """

    from ..freezing import constant_fold

    lazy_init()
    # We need a few rounds of binary folding to get rid of all the
    # unnecessary nodes, but may need a good method to chose the rounds number.
    # works like: conv+binary+binary.
    binary_folding = counters["inductor"]["binary_folding"]
    fake_tensor_prop(gm, aot_example_inputs, True)

    torch._inductor.fx_passes.binary_folding.mark_mixed_dtype_allowed_convs(gm)
    for _ in range(4):
        constant_fold(gm)
        # Make sure meta['val'] is properly set for all nodes
        fake_tensor_prop(gm, aot_example_inputs, True)
        binary_folding_pass.apply(gm.graph)
        # If we don't have binary folding, we don't need to run the pass again.
        # TODO: remove the need to run fake_tensor_prop on the whole model.
        if counters["inductor"]["binary_folding"] == binary_folding:
            break
        binary_folding = counters["inductor"]["binary_folding"]

    torch._inductor.fx_passes.binary_folding.recover_original_precision_folded_convs(gm)

    constant_fold(gm)
    fake_tensor_prop(gm, aot_example_inputs, True)

    for pattern in pass_patterns:
        pattern.apply(gm.graph)

    # The CPU weight packing always assume the conv's weight is channels last,
    # So make sure the layout_optimization is on when doing it.
    if (
        torch._C._has_mkldnn
        and config.cpp.weight_prepack
        and config.layout_optimization
    ):
        from .mkldnn_fusion import _eliminate_duplicate_packed_nodes

        _eliminate_duplicate_packed_nodes(gm)

    stable_topological_sort(gm.graph)
    gm.recompile()
    gm.graph.lint()


@init_once_fakemode
def lazy_init():
    if torch._C._has_mkldnn and config.cpp.weight_prepack:
        from .mkldnn_fusion import _mkldnn_weight_pack_init

        _mkldnn_weight_pack_init()

    from .binary_folding import binary_folding_init

    addmm_patterns_init()
    binary_folding_init()


def register_freezing_graph_pattern(pattern, extra_check=_return_true, pass_number=0):
    return register_graph_pattern(
        pattern,
        extra_check=extra_check,
        pass_dict=pass_patterns[pass_number],
    )


def register_binary_folding_pattern(pattern, extra_check=_return_true):
    return register_graph_pattern(
        pattern,
        extra_check=extra_check,
        pass_dict=binary_folding_pass,
    )


@functools.lru_cache(None)
def addmm_patterns_init():
    if torch.cuda.is_available():
        # workaround https://github.com/pytorch/pytorch/issues/97894
        device = "cuda"
    else:
        device = "cpu"
    val = functools.partial(torch.empty, (10, 10), device=device, requires_grad=False)

    def check_concat_weights(match):
        weights = [
            match.kwargs["w1"],
            match.kwargs["w2"],
            match.kwargs["w3"],
        ]
        return all(
            w.op == "get_attr" and w.meta["val"].shape == weights[0].meta["val"].shape
            for w in weights
        )

    def matmul_fuse_pattern(inp, w1, w2, w3):
        return (inp @ w1, inp @ w2, inp @ w3)

    def matmul_replacement(inp, w1, w2, w3):
        cat_t = torch.cat((w1, w2, w3), dim=1)
        mm = inp @ cat_t
        return mm.chunk(3, dim=1)

    register_replacement(
        matmul_fuse_pattern,
        matmul_replacement,
        [val(), val(), val(), val()],
        inference_graph,
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2", "w3"),
    )

    def addmm_fuse_pattern_second(inp, w1, w2, w3, b1, b2, b3):
        return (
            aten.addmm(b1, inp, w1),
            aten.addmm(b2, inp, w2),
            aten.addmm(b3, inp, w3),
        )

    def addmm_fuse_replacement_second(inp, w1, w2, w3, b1, b2, b3):
        cat_w = torch.cat((w1, w2, w3), dim=1)
        cat_b = torch.cat((b1, b2, b3))
        return aten.addmm(cat_b, inp, cat_w).chunk(3, dim=1)

    register_replacement(
        addmm_fuse_pattern_second,
        addmm_fuse_replacement_second,
        [val() for _ in range(7)],
        inference_graph,
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2", "w3", "b1", "b2", "b3"),
    )


def same_dtype(match):
    return match.output_node().args[0].meta["val"].dtype == match.kwargs["dtype"]


@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type.default,
        Ignored(),
        KeywordArg("dtype"),
    ),
    pass_dict=pass_patterns[0],
    extra_check=same_dtype,
)
def unnecessary_dtype_convert(match: Match, **kwargs):
    """Remove unnecessary dtype conversion op, probably left as a result of Conv-Bn folding"""
    graph = match.graph
    node = match.output_node()
    node.replace_all_uses_with(node.args[0])
    graph.erase_node(node)


def is_conv_1v1(
    inp,
    weight,
    bias,
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    transposed: bool,
    output_padding: List[int],
    groups: int,
):

    if not inp.meta["val"].is_cuda or not inp.meta["val"].is_contiguous(memory_format=torch.channels_last):
        return False

    if not weight.meta["val"].is_contiguous(memory_format=torch.channels_last):
        return False

    out_chan, in_chan, *kernel_shape = weight.meta["val"].shape

    if not (
        is_ones(kernel_shape)
        and is_ones(stride)
        and is_zeros(padding)
        and is_ones(dilation)
        and not transposed
        and is_zeros(output_padding)
        and groups == 1
    ):
        return False

    return True

    CallFunction(
        aten.convolution.default, 
        CallFunction(
            aten.relu.default,
            CallFunction(
                aten.add.Tensor,
                CallFunction(aten.convolution.default, *[Ignored() for _ in range(9)]),
                Ignored(),
            )
        )
        *[Ignored() for _ in range(8)]
    )



@register_graph_pattern(
    CallFunction(
        aten.mm.default, 
        KeywordArg("arg0"),
        KeywordArg("arg1"),
    ),
    pass_dict=pass_patterns[2],
)
def mm_bandwidth_bound(match: Match, arg0, arg1):
    from torch._inductor.fx_passes.pad_mm import is_mm_compute_bound

    if not is_mm_compute_bound(arg0.meta["val"].shape[0], arg0.meta["val"].shape[1], arg1.meta["val"].shape[1], arg1.meta["val"].dtype):
        print("Potentially Bandwidth bound mm", arg0.meta["val"].shape[0], arg0.meta["val"].shape[1], arg1.meta["val"].shape[1])


@register_graph_pattern(
    CallFunction(
        aten.bmm.default, 
        KeywordArg("arg0"),
        KeywordArg("arg1"),
    ),
    pass_dict=pass_patterns[2],
)
def mm_bandwidth_bound(match: Match, arg0, arg1):
    from torch._inductor.fx_passes.pad_mm import is_mm_compute_bound
    
    m, k, n = arg0.meta["val"].shape[1], arg0.meta["val"].shape[2], arg1.meta["val"].shape[2]
    if not is_mm_compute_bound(m, k, n, arg1.meta["val"].dtype):
        print("Potentially Bandwidth bound bmm", m, k, n)



@register_graph_pattern(
    CallFunction(
        aten.addmm.default, 
        Ignored(),
        KeywordArg("arg0"),
        KeywordArg("arg1"),
    ),
    pass_dict=pass_patterns[2],
)
def mm_bandwidth_bound(match: Match, arg0, arg1):
    from torch._inductor.fx_passes.pad_mm import is_mm_compute_bound

    if not is_mm_compute_bound(arg0.meta["val"].shape[0], arg0.meta["val"].shape[1], arg1.meta["val"].shape[1], arg1.meta["val"].dtype):
        print("Potentially Bandwidth bound mm", arg0.meta["val"].shape[0], arg0.meta["val"].shape[1], arg1.meta["val"].shape[1])


@register_graph_pattern(
    CallFunction(
        aten.sum.dim_IntList, 
        CallFunction(
            aten.addmm.default, 
            Ignored(),
            Ignored(),
            Ignored(),
        ),
        Ignored(),
        Ignored(),
    ),
    pass_dict=pass_patterns[2],
)
def mm_bandwidth_bound(match: Match, arg0, arg1):
    print("matmul -> sum")


@register_graph_pattern(
    CallFunction(
        aten.sum.dim_IntList, 
        CallFunction(
            aten.mm.default, 
            Ignored(),
            Ignored(),
        ),
        Ignored(),
        Ignored(),
    ),
    pass_dict=pass_patterns[2],
)
def mm_bandwidth_bound(match: Match, arg0, arg1):
    print("matmul -> sum")

# # TODO handle ndim == 3
# def repl(x, weight, bias):
#     # special case for 1x1 convolution, which is actually just a matmul
#     rank = len(weight.size())
#     weight = weight.squeeze(-1).squeeze(-1)
#     weight = weight.permute(1, 0)
#     x_permute = list(range(rank))
#     x_permute.append(x_permute.pop(1))
#     x = x.permute(*x_permute)
#     *sizes, in_chan = x.size()
#     x = x.reshape(-1, in_chan)
#     if bias is None:
#         result = torch.mm(x, weight)
#     else:
#         result = torch.addmm(bias, x, weight)
#     result = result.reshape(*sizes, -1)
#     result_permute = list(range(rank))
#     result_permute.insert(1, result_permute.pop(-1))
#     return result.permute(*result_permute)
    # with V.fake_mode:
    #     match.replace_by_example(repl, [inp, weight, bias])

@register_graph_pattern(
    CallFunction(
        aten.convolution.default, 
        CallFunction(
            aten.relu.default,
            CallFunction(
                aten.add.Tensor,
                CallFunction(aten.convolution.default, *[Ignored() for _ in range(9)]),
                Ignored(),
            ),
            _users=MULTIPLE,
        ),
        *[Ignored() for _ in range(8)],
    ),
    pass_dict=pass_patterns[2],
)
def conv1v1_plus_conv1v1(match: Match):
    if not is_conv_1v1(*match.output_node().args):
        return

    if not is_conv_1v1(*match.output_node().args[0].args[0].args[0].args):
        return

    # breakpoint()
    
    print("found pattern")
    # breakpoint()
    return
        
    # return inductor.kernel.mm_plus_mm.tuned_mm_plus_mm(mat1, mat2, mat3, mat4)
