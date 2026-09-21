"""Compact cold AOT forward facts for the later backward compiler callback."""

from dataclasses import dataclass

import torch
from torch._functorch._aot_autograd.descriptors import (
    AOTOutput, SavedForBackwardsAOTOutput, SavedForBackwardsNoVcCheckAOTOutput, TangentAOTInput,
)
from torch._functorch._aot_autograd.utils import contain_metadata_mutation_ops
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule, Node
from torch.multiprocessing.reductions import StorageWeakRef

from .classification import (
    _outputs, _storage, _value, BackwardInput, ClassificationDeclined, SavedActivations,
)


@dataclass(frozen=True, eq=False)
class SavedTensorOrigin:
    forward_output: int
    saved_index: int
    name: str
    value: FakeTensor
    storage: StorageWeakRef
    forward_kind: str


@dataclass(frozen=True, eq=False)
class ForwardOrigins:
    saved: tuple[SavedTensorOrigin, ...]


def capture_forward_origins(forward):
    """Capture final AOT facts, then isolate forward metadata dictionary writes.

    Only saved FakeTensor values and their storage identities remain owned.
    The original module and graph remain the forward compilation carrier.
    """
    if not isinstance(forward, GraphModule):
        raise ClassificationDeclined("Expected the final AOT forward compiler graph")
    if contain_metadata_mutation_ops(forward):
        raise ClassificationDeclined("Metadata mutation invalidates storage provenance")
    output, values = _outputs(forward)
    descriptors = output.meta.get("desc")
    if not isinstance(descriptors, (tuple, list)) or len(descriptors) != len(values):
        raise ClassificationDeclined("Missing complete final AOT output descriptors")
    if any(not isinstance(desc, AOTOutput) for desc in descriptors):
        raise ClassificationDeclined("Expected typed AOT output descriptors")
    saved_types = (SavedForBackwardsAOTOutput, SavedForBackwardsNoVcCheckAOTOutput)
    saved = [(index, node, desc) for index, (node, desc) in enumerate(zip(values, descriptors, strict=True))
             if type(desc) in saved_types]
    if any(type(desc.idx) is not int for _, _, desc in saved):
        raise ClassificationDeclined("Malformed saved-output ordinal")
    if [desc.idx for _, _, desc in saved] != list(range(len(saved))):
        raise ClassificationDeclined("Saved-output ordinals are not canonical")
    inputs = [node for node in forward.graph.nodes if node.op in ("placeholder", "get_attr")]
    input_storage = {_storage(_value(node)) for node in inputs} - {None}
    visible = [node for node, desc in zip(values, descriptors, strict=True) if type(desc) not in saved_types]
    output_storage = {_storage(_value(node)) for node in visible} - {None}
    rows = []
    for output_index, node, desc in saved:
        value = _value(node)
        storage = _storage(value)
        if storage is None:
            continue
        if not isinstance(node, Node):
            raise ClassificationDeclined("Saved tensor lacks its final AOT node")
        kind = "input_alias" if storage in input_storage else (
            "output_alias" if storage in output_storage else "activation")
        rows.append(SavedTensorOrigin(output_index, desc.idx, node.name, value, storage, kind))
    result = ForwardOrigins(tuple(rows))
    # AOT can share these dictionaries with backward placeholders.
    for node in forward.graph.nodes:
        node.meta = node.meta.copy()
    return result


def join_backward_origins(origins, backward):
    if type(origins) is not ForwardOrigins or not isinstance(backward, GraphModule):
        raise ClassificationDeclined("Expected compact forward origins and the final AOT backward graph")
    if contain_metadata_mutation_ops(backward):
        raise ClassificationDeclined("Metadata mutation invalidates storage provenance")
    _, outputs = _outputs(backward)
    output_storage = {_storage(_value(node)) for node in outputs} - {None}
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
    for saved in origins.saved:
        if _storage(saved.value) != saved.storage:
            raise ClassificationDeclined("Saved FakeTensor storage changed after forward capture")
        matches = [index for index, node in enumerate(placeholders)
                   if node.name == saved.name and _value(node) is saved.value]
        if len(matches) != 1 or matches[0] in joined:
            raise ClassificationDeclined("Saved tensor lost its exact backward identity/name association")
        index = matches[0]
        if rows[index].kind == "tangent":
            raise ClassificationDeclined("Saved tensor unexpectedly joined a tangent")
        joined.add(index)
        kind = saved.forward_kind
        if kind == "activation" and saved.storage in output_storage:
            kind = "output_alias"
        rows[index] = BackwardInput(index, saved.name, kind, saved.forward_output, saved.saved_index)
    return SavedActivations(tuple(rows))
