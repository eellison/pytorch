"""Optional cold provenance for an unchanged, unpartitioned backward input box."""

from dataclasses import dataclass

from torch._inductor.runtime._cudagraph._compiler.compiler_metadata_handoff.metadata import _check_metadata, MetadataDeclined
from torch._inductor import config
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch.fx import Graph, GraphModule, Node

from .classification import ClassificationDeclined, classify_saved_activations, SavedActivations


@dataclass(frozen=True, eq=False)
class AOTInputOrigins:
    module: GraphModule
    graph: Graph
    placeholders: tuple[Node, ...]
    names: tuple[str, ...]
    targets: tuple[str, ...]
    classification: SavedActivations


@dataclass(frozen=True)
class FinalSavedInputs:
    input_names: tuple[str, ...]
    classification: SavedActivations

    @property
    def candidate_indices(self):
        return self.classification.candidate_indices


def capture_aot_inputs(forward, backward):
    """Capture before downstream fake propagation changes AOT value metadata."""
    try:
        classification = classify_saved_activations(forward, backward)
    except ClassificationDeclined:
        return None
    placeholders = tuple(node for node in backward.graph.nodes if node.op == "placeholder")
    targets = tuple(node.target for node in placeholders)
    if any(type(target) is not str for target in targets):
        return None
    return AOTInputOrigins(
        backward, backward.graph, placeholders, tuple(node.name for node in placeholders),
        targets, classification,
    )


def same_aot_inputs(origin, module):
    if (type(origin) is not AOTInputOrigins or module is not origin.module
            or module.graph is not origin.graph):
        return False
    placeholders = tuple(node for node in origin.graph.nodes if node.op == "placeholder")
    return (
        len(placeholders) == len(origin.placeholders)
        and all(actual is saved for actual, saved in zip(placeholders, origin.placeholders, strict=True))
        and tuple(node.name for node in placeholders) == origin.names
        and tuple(node.target for node in placeholders) == origin.targets
    )


def collect_saved_inputs(origin, wrapper, records, metadata):
    """Join at the live collect_metadata return boundary; retain no compiler owners.

    The caller supplies that collector's records and result. Only exact original
    placeholder identities and ordering are supported. No cache reconstruction
    or post-propagation FakeTensor identity recovery is attempted.
    """
    if type(origin) is not AOTInputOrigins or metadata is None or type(wrapper) is not PythonWrapperCodegen:
        return None
    graph = V.graph
    if (type(graph) is not GraphLowering or not graph.is_backward or graph.cpp_wrapper
            or graph.aot_mode or graph.partition_maps or config.graph_partition or graph.name is not None
            or not same_aot_inputs(origin, graph.module)
            or tuple(wrapper.get_graph_input_names()) != origin.targets):
        return None
    try:
        _check_metadata(metadata, records)
    except MetadataDeclined:
        return None
    if (records.input_names != origin.targets
            or len(origin.classification.inputs) != len(records.input_names)
            or any(metadata.inputs.kinds[index] != "tensor" for index in origin.classification.candidate_indices)):
        return None
    return FinalSavedInputs(records.input_names, origin.classification)
