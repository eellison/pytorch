"""Optional saved-input metadata preserves exact boxed input provenance."""

import gc
from dataclasses import replace
from unittest import mock
from weakref import ref


import torch
from torch._inductor.runtime._cudagraph._compiler.compiler_saved_inputs import transport
from torch._inductor.runtime._cudagraph._compiler.compiler_saved_inputs.classification import BackwardInput, SavedActivations
from torch._inductor.runtime._cudagraph._compiler.compiler_saved_inputs.handoff import AOTInputOrigins
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import FXTraceDeclined, InputContract, IntegerRange, TensorInput
from torch._inductor.runtime._cudagraph.metadata import (
    _check_metadata, attach_terminal_metadata, check_terminal_attachment, emit_terminal_metadata,
    MetadataDeclined, read_terminal_metadata, TerminalMetadata,
)
from torch._inductor.runtime._cudagraph.policy import Installation
from torch._dynamo import config as dynamo_config
from torch._functorch import config as aot_config
from torch._inductor import compile_fx as compiler, config
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.graph import GraphLowering
from torch._inductor.output_code import CompiledFxGraph
from torch._inductor.utils import _InputAlignmentWrapper, align_inputs_from_check_idxs, IndentedBuffer
from torch._inductor.virtualized import V
from torch.fx import Graph, GraphModule
from torch.utils._ordered_set import OrderedSet
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


@instantiate_parametrized_tests
class TestSavedInputMetadata(TestCase):
    def setUp(self):
        super().setUp()
        self.inputs = InputContract(
            ("integer", "tensor", "tensor"),
            tuple(TensorInput(index, torch.float32, (8,), (1,)) for index in (1, 2)),
            (IntegerRange(0, 1, 8),),
        )


    def collector_fixture(self):
        self.enterContext(config.patch(
            cudagraph_saved_input_schedule=True, force_disable_caches=True, save_args=False,
            graph_partition=False, cpp_wrapper=False, fx_wrapper=False, **{"trace.enabled": False},
        ))
        self.enterContext(aot_config.patch(enable_autograd_cache=False, bundled_autograd_cache=False))
        self.enterContext(dynamo_config.patch(repro_after=None, caching_precompile=False))
        self.enterContext(mock.patch.object(compiler, "fx_compile_mode", compiler.FxCompileMode.NORMAL))
        self.enterContext(mock.patch.object(compiler, "fx_compile_async", False))
        self.enterContext(mock.patch.object(compiler, "fx_compile_progressive", False))
        self.enterContext(V.set_aot_compilation(False))
        self.assertTrue(transport.eligible(compiler.compile_fx_inner))
        graph = Graph()
        names = ("size", "activation", "tangent")
        placeholders = tuple(graph.placeholder(name) for name in names)
        graph.output((placeholders[1],))
        module = GraphModule(torch.nn.Module(), graph)
        classification = SavedActivations((
            BackwardInput(0, "size", "symbol"),
            BackwardInput(1, "activation", "activation", 1, 0),
            BackwardInput(2, "tangent", "tangent"),
        ))
        origin = AOTInputOrigins(module, graph, placeholders, names, names, classification)
        module.meta[transport.ORIGIN_KEY] = transport._origin_key(origin)
        token = transport._active_origin.set(origin)
        self.addCleanup(transport._active_origin.reset, token)
        lowering = GraphLowering.__new__(GraphLowering)
        lowering.module = module
        lowering.is_backward = True
        lowering.cpp_wrapper = lowering.aot_mode = False
        lowering.partition_maps = None
        lowering.name = None
        lowering.graph_input_names = list(names)
        self.enterContext(V.set_graph_handler(lowering))
        return PythonWrapperCodegen.__new__(PythonWrapperCodegen), lowering

    def test_default_has_no_saved_inputs(self):
        metadata = TerminalMetadata(self.inputs)
        _check_metadata(metadata)
        self.assertEqual(metadata.saved_input_indices, ())

    @parametrize("indices", ((0,), (True,), (1, 1)))
    def test_invalid_saved_input_slots(self, indices):
        with self.assertRaisesRegex(MetadataDeclined, "distinct Tensor boxed slots"):
            _check_metadata(TerminalMetadata(self.inputs, indices))

    @parametrize("case", ("valid", "disabled", "absent", "reordered", "cache_key"))
    def test_collector_requires_exact_active_origin(self, case):
        wrapper, graph = self.collector_fixture()
        if case == "disabled":
            self.enterContext(config.patch(cudagraph_saved_input_schedule=False))
        elif case == "absent":
            transport._active_origin.set(None)
        elif case == "reordered":
            graph.graph_input_names.reverse()
        elif case == "cache_key":
            graph.module.meta[transport.ORIGIN_KEY] = ("saved-input-origin", 1, None)
        self.assertEqual(transport.collect_terminal_saved_inputs(wrapper, self.inputs),
                         (1,) if case == "valid" else ())

    def test_generated_attachment_preserves_saved_tuple(self):
        metadata = TerminalMetadata(self.inputs, (1,))
        source = IndentedBuffer()
        emit_terminal_metadata(source, "call", metadata)
        namespace = {}
        exec("def call(args):\n    return args\n" + source.getvalue(), namespace)
        function = namespace["call"]
        restored = check_terminal_attachment(function, function._cudagraph_terminal_attachment)
        self.assertEqual(restored, metadata)
        self.assertIsNot(restored, metadata)
        self.assertIs(type(restored.saved_input_indices), tuple)
        self.assertEqual(restored.saved_input_indices, (1,))

    def alignment_fixture(self):
        namespace = {}
        exec("def call(args):\n    args.clear()\n    return (7,)\n", namespace)
        original = namespace["call"]
        metadata = TerminalMetadata(self.inputs)
        attach_terminal_metadata(original, metadata)
        ordinary = align_inputs_from_check_idxs(original, (1,), OrderedSet((1,)))
        artifact = CompiledFxGraph.__new__(CompiledFxGraph)
        artifact.compiled_fn_runner = None
        artifact.current_callable = ordinary
        artifact._cudagraph_original_callable = original
        artifact.fx_kwargs = {"static_input_idxs": ()}
        return artifact, original, metadata

    def test_exact_alignment_wrapper_preserves_attachment(self):
        artifact, original, metadata = self.alignment_fixture()
        self.assertIs(type(artifact.current_callable), _InputAlignmentWrapper)
        self.assertIs(artifact.current_callable.model, original)
        function, restored = read_terminal_metadata(artifact)
        self.assertIs(function, original)
        self.assertIs(restored, metadata)

    @parametrize("wrapper", ("different_model", "arbitrary"))
    def test_unrecognized_alignment_wrapper_declines(self, wrapper):
        artifact, original, _ = self.alignment_fixture()

        def other(args):
            return original(args)

        artifact.current_callable = (replace(artifact.current_callable, model=other)
                                     if wrapper == "different_model" else other)
        with self.assertRaisesRegex(MetadataDeclined, "original generated function"):
            read_terminal_metadata(artifact)

    @parametrize("field,indices", (
        ("inputs_to_check", (0,)),
        ("inputs_to_check", (1, 1)),
        ("mutated_input_idxs", (True,)),
        ("mutated_input_idxs", (3,)),
    ))
    def test_alignment_wrapper_indices_require_tensor_slots(self, field, indices):
        artifact, _, _ = self.alignment_fixture()
        artifact.current_callable = replace(artifact.current_callable, **{field: indices})
        with self.assertRaisesRegex(MetadataDeclined, "compiler tensor slots"):
            read_terminal_metadata(artifact)

    @parametrize("transition", ("close", "decline"))
    def test_alignment_wrapper_restored_and_owner_released(self, transition):
        artifact, original, metadata = self.alignment_fixture()
        ordinary_ref = ref(artifact.current_callable)
        prepare = mock.Mock(side_effect=FXTraceDeclined("test preparation decline"))
        installation = Installation(artifact, original, metadata, prepare)
        self.addCleanup(installation.close)
        artifact.current_callable = installation
        gc.collect()
        self.assertIsNotNone(ordinary_ref())
        self.assertIs(installation.ordinary, ordinary_ref())
        if transition == "close":
            installation.close()
            prepare.assert_not_called()
        else:
            box = [2, torch.empty(8), torch.empty(8)]
            self.assertEqual(installation(box), (7,))
            self.assertEqual(box, [])
            self.assertEqual(installation.status, "declined")
            self.assertEqual(installation.decline, "test preparation decline")
            self.assertEqual(prepare.call_count, 1)
            self.assertIs(prepare.call_args.args[0], original)
            self.assertEqual(prepare.call_args.kwargs["alignment_inputs"], (1,))
        self.assertIs(artifact.current_callable, ordinary_ref())
        self.assertIsNone(installation._ordinary_owner)
        artifact.current_callable = original
        gc.collect()
        self.assertIsNone(ordinary_ref())
        self.assertIsNone(installation.ordinary)


if __name__ == "__main__":
    run_tests()
