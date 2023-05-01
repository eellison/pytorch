import warnings

import torch
import torch.utils._pytree as pytree

def replace_node_with_constant(gm, node, constant):
    g = gm.graph

    i = 0
    while True:
        qualname = f"_frozen_param{i}"
        if not hasattr(gm, qualname):
            break
        i += 1

    with g.inserting_before(node):
        new_input_node = g.create_node("get_attr", qualname, (), {})
        node.replace_all_uses_with(new_input_node)
        new_input_node.meta.update(node.meta)
        g.erase_node(node)

    # needed to suppress `does not reference an nn.Module, nn.Parameter, or buffer` warning
    gm.register_buffer(qualname, constant)
    setattr(gm, qualname, constant)


def replace_params_with_constants(fake_gm, real_inputs, example_inputs_, fw_metadata):
    fake_inp_nodes = [node for (_, node) in zip(real_inputs, fake_gm.graph.nodes)]

    g = fake_gm.graph

    preserved_arg_indices = []

    for i, (real_input, fake_input, node) in enumerate(
        zip(real_inputs, example_inputs_, fake_inp_nodes)
    ):
        assert real_input.shape == fake_input.shape

        if i in fw_metadata.mutated_inp_indices:
            preserved_arg_indices.append(i)
            continue

        replace_node_with_constant(fake_gm, node, real_input)

    # add on non param inputs
    preserved_arg_indices.extend(range(len(real_inputs), len(example_inputs_)))

    g.lint()
    # is this necessary ?
    fake_gm.recompile()
    return fake_gm, preserved_arg_indices


@torch.utils._python_dispatch._disable_current_modes()
def constant_fold(gm, num_inputs):
    unknown_value = object()

    node_replacements = {}

    class ConstantFolder(torch.fx.Interpreter):
        def run_node(self, node):
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            if unknown_value in pytree.tree_flatten((args, kwargs))[0]:
                return unknown_value

            # TODO - skip nondeterminism
            # We shouldnt technically have to skip mutation (post functionalization), 
            # but might as well double check

            out = super().run_node(node)

            # Can I modify the graph while running it ?
            # Not sure if I need tensor check here
            if node.op != "get_attr" and isinstance(out, torch.Tensor):
                node_replacements[node] = out
            return out

    ConstantFolder(gm).run(*[unknown_value for _ in range(num_inputs)])

    for node, constant in node_replacements.items():
        replace_node_with_constant(gm, node, constant)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()


def optimize_for_inference(
    original_gm: torch.fx.GraphModule,
    fake_gm: torch.fx.GraphModule,
    example_inputs_,
    fw_metadata,
):
    # currently just freezes and runs constant folding, no special passes
    print(original_gm)

    params = {
        **dict(original_gm.named_parameters(remove_duplicate=False)),
        **dict(original_gm.named_buffers(remove_duplicate=False)),
    }
    params_flat, _ = pytree.tree_flatten(params)
    params_flat = tuple(params_flat)

    fake_gm, preserved_arg_indices = replace_params_with_constants(
        fake_gm, params_flat, example_inputs_, fw_metadata
    )

    constant_fold(fake_gm, len(preserved_arg_indices))

    return fake_gm, preserved_arg_indices
