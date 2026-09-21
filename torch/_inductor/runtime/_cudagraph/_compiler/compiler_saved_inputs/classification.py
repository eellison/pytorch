from dataclasses import dataclass

import torch
from torch._functorch._aot_autograd.descriptors import (
    AOTOutput,
    SavedForBackwardsAOTOutput,
    SavedForBackwardsNoVcCheckAOTOutput,
    TangentAOTInput,
)
from torch._functorch._aot_autograd.utils import contain_metadata_mutation_ops
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule, Node
from torch.multiprocessing.reductions import StorageWeakRef


class ClassificationDeclined(ValueError):
    pass


@dataclass(frozen=True)
class BackwardInput:
    boxed_index: int
    name: str
    kind: str
    forward_output: int | None = None
    saved_index: int | None = None


@dataclass(frozen=True)
class SavedActivations:
    inputs: tuple[BackwardInput, ...]

    @property
    def candidate_indices(self):
        return tuple(row.boxed_index for row in self.inputs if row.kind == "activation")


def _outputs(module):
    outputs = [node for node in module.graph.nodes if node.op == "output"]
    if len(outputs) != 1 or len(outputs[0].args) != 1:
        raise ClassificationDeclined("Expected one flat AOT output")
    values = outputs[0].args[0]
    if not isinstance(values, (tuple, list)):
        raise ClassificationDeclined("Expected a flat AOT output sequence")
    return outputs[0], tuple(values)


def _value(node):
    if isinstance(node, Node):
        if "val" not in node.meta:
            raise ClassificationDeclined("Missing final AOT value metadata")
        return node.meta["val"]
    return node


def _storage(value):
    if type(value) is FakeTensor and value.layout is torch.strided:
        return StorageWeakRef(value.untyped_storage())
    if type(value) in (type(None), bool, int, float, complex, torch.SymInt, torch.SymFloat, torch.SymBool):
        return None
    raise ClassificationDeclined("Unknown tensor or opaque boundary storage provenance")


def classify_saved_activations(forward, backward):
    """Classify final AOT callback graphs that preserve shared FakeTensor identities.

    Saved tensors must retain both their node names and exact metadata objects
    across the paired callbacks. Indices address AOT's backward compiler box,
    before any downstream compiler reordering. No runtime owners are returned.
    """
    if not isinstance(forward, GraphModule) or not isinstance(backward, GraphModule):
        raise ClassificationDeclined("Expected final AOT compiler callback graphs")
    if contain_metadata_mutation_ops(forward) or contain_metadata_mutation_ops(backward):
        raise ClassificationDeclined("Metadata mutation invalidates storage provenance")
    output, fw_values = _outputs(forward)
    _, bw_values = _outputs(backward)
    descriptors = output.meta.get("desc")
    if not isinstance(descriptors, (tuple, list)) or len(descriptors) != len(fw_values):
        raise ClassificationDeclined("Missing complete final AOT output descriptors")
    if any(not isinstance(desc, AOTOutput) for desc in descriptors):
        raise ClassificationDeclined("Expected typed AOT output descriptors")
    saved_types = (SavedForBackwardsAOTOutput, SavedForBackwardsNoVcCheckAOTOutput)
    saved = [(index, value, desc) for index, (value, desc) in enumerate(zip(fw_values, descriptors, strict=True))
             if type(desc) in saved_types]
    if any(type(desc.idx) is not int for _, _, desc in saved):
        raise ClassificationDeclined("Malformed saved-output ordinal")
    if [desc.idx for _, _, desc in saved] != list(range(len(saved))):
        raise ClassificationDeclined("Saved-output ordinals are not canonical")

    fw_inputs = [node for node in forward.graph.nodes if node.op in ("placeholder", "get_attr")]
    input_storage = {_storage(_value(node)) for node in fw_inputs} - {None}
    visible = [node for node, desc in zip(fw_values, descriptors, strict=True) if type(desc) not in saved_types]
    output_storage = {_storage(_value(node)) for node in (*visible, *bw_values)} - {None}
    placeholders = [node for node in backward.graph.nodes if node.op == "placeholder"]
    rows = []
    for index, node in enumerate(placeholders):
        value = _value(node)
        if isinstance(value, torch.Tensor):
            _storage(value)
            kind = "tangent" if type(node.meta.get("desc")) is TangentAOTInput else "other_tensor"
        elif type(value) in (torch.SymInt, torch.SymFloat, torch.SymBool):
            kind = "symbol"
        elif type(value) in (type(None), bool, int, float, complex):
            kind = "scalar"
        else:
            kind = "opaque"
        rows.append(BackwardInput(index, node.name, kind))

    joined = set()
    for output_index, saved_node, desc in saved:
        value = _value(saved_node)
        storage = _storage(value)
        if storage is None:
            continue
        if not isinstance(saved_node, Node):
            raise ClassificationDeclined("Saved tensor lacks its final AOT node")
        matches = [index for index, node in enumerate(placeholders)
                   if node.name == saved_node.name and _value(node) is value]
        if len(matches) != 1 or matches[0] in joined:
            raise ClassificationDeclined("Saved tensor lost its exact backward identity/name association")
        index = matches[0]
        if rows[index].kind == "tangent":
            raise ClassificationDeclined("Saved tensor unexpectedly joined a tangent")
        joined.add(index)
        if storage in input_storage:
            kind = "input_alias"
        elif storage in output_storage:
            kind = "output_alias"
        else:
            kind = "activation"
        rows[index] = BackwardInput(index, saved_node.name, kind, output_index, desc.idx)
    return SavedActivations(tuple(rows))
