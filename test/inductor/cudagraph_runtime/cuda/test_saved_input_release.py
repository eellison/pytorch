"""Compiler saved activations reach terminal replay with ordinary storage lifetime."""

from dataclasses import asdict
import functools
import json
from pathlib import Path
import sys
import threading
from unittest import mock


import torch._inductor.runtime._cudagraph._compiler.compiler_saved_inputs.transport as saved_transport
from torch._inductor.runtime._cudagraph._compiler.host_program import Allocate, Normalize
import torch._inductor.runtime._cudagraph.policy as policy_module
import torch._inductor.runtime._cudagraph.replay as terminal_replay
from torch._inductor.runtime._cudagraph.api import NativeTerminalPolicy
from torch._inductor.runtime._cudagraph.frontend import TerminalCall
import torch
from torch._dynamo.utils import counters
from torch._functorch import config as aot_config
from torch._inductor import config
from torch._inductor.output_code import CompiledFxGraph
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource, InputSource, IntExpr, OwnedBuffer, PointerSource,
)
from torch._inductor.runtime.static_triton_launcher import StaticallyLaunchedCudaKernel
from torch._inductor.virtualized import V
from torch.multiprocessing.reductions import StorageWeakRef
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts


SIGMOID_CONTEXT = functools.partial(create_selective_checkpoint_contexts, [torch.ops.aten.sigmoid.default])


def sigmoid_weighted(value, weight):
    return value.sigmoid() * weight


class SavedFanout(torch.nn.Module):
    def forward(self, value, residual, weight):
        shared = value.sin() + residual.cos()
        return checkpoint(sigmoid_weighted, shared, weight, use_reentrant=False, context_fn=SIGMOID_CONTEXT)


class NativeObservation:
    def __init__(self, installation, saved_indices, keep_aliases):
        self.artifact = installation.artifact
        self.original_code = installation.original.__code__
        self.terminal_directory = str(Path(policy_module.__file__).parent) + "/"
        self.saved_indices = saved_indices
        self.keep_aliases = keep_aliases
        self.pending = {}
        self.calls = []
        self.frames = []
        self.errors = []
        self.aliases = []

    def profile(self, frame, event, result):
        try:
            code = frame.f_code
            if event == "call" and (code is self.original_code
                                    or code.co_filename.startswith(self.terminal_directory)):
                self.frames.append((code.co_filename, code.co_name))
            if code is not CompiledFxGraph.__call__.__code__ or frame.f_locals.get("self") is not self.artifact:
                return
            thread = threading.get_ident()
            if event == "call":
                if thread in self.pending:
                    raise AssertionError("Nested backward artifact invocation")
                inputs = frame.f_locals["inputs"]
                self.pending[thread] = (tuple(
                    (index, value.data_ptr(), StorageWeakRef(value.untyped_storage()))
                    for index, value in enumerate(inputs) if isinstance(value, torch.Tensor)
                ), tuple((index, value) for index, value in enumerate(inputs) if type(value) is int))
                if self.keep_aliases:
                    for index in self.saved_indices:
                        value = inputs[index]
                        self.aliases.append((value.detach(), value.detach().clone(),
                                             StorageWeakRef(value.untyped_storage())))
            elif event == "return":
                inputs, integers = self.pending.pop(thread)
                if result is None:
                    return
                if type(result) not in (tuple, list) or any(
                        value is not None and type(value) is not torch.Tensor for value in result):
                    raise AssertionError("Expected complete generated backward Tensor/None output slots")
                self.calls.append({
                    "inputs": [(index, pointer, weak.expired()) for index, pointer, weak in inputs],
                    "integers": integers,
                    "output_pointers": [None if value is None else value.data_ptr() for value in result],
                    "output_shapes": [None if value is None else tuple(value.shape) for value in result],
                })
        except BaseException as error:
            self.errors.append(f"{type(error).__name__}: {error}")

    def __enter__(self):
        if sys.getprofile() is not None or threading.getprofile() is not None:
            raise AssertionError("Native observation requires an unprofiled test process")
        threading.setprofile_all_threads(self.profile)
        return self

    def __exit__(self, kind, error, traceback):
        threading.setprofile_all_threads(None)
        if error is not None and self.errors:
            error.add_note("Native observation errors: " + "; ".join(self.errors))


class TestTerminalSavedInputRelease(TestCase):
    @parametrize("mode", ("release", "disabled", "retain_graph", "external_alias"))
    def test_generated_backward_release(self, device, mode):
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        self.assertFalse(config.cudagraph_saved_input_schedule)
        policy = NativeTerminalPolicy()
        self.addCleanup(policy.close)
        self.enterContext(torch.cuda.device(device))
        self.enterContext(torch._dynamo.config.patch(caching_precompile=False, repro_after=None))
        self.enterContext(aot_config.patch(enable_autograd_cache=False, bundled_autograd_cache=False, donated_buffer=False))
        settings = {
            "cudagraph_policy": policy, "graph_partition": False, "cpp_wrapper": False,
            "fx_graph_cache": False, "force_disable_caches": True, "allow_buffer_reuse": False,
            "inplace_buffers": False, "max_fusion_size": 1, "realize_reads_threshold": 0,
            "normalize_static_input_alignment": True, "use_static_triton_launcher": True,
            "use_fast_triton_launcher": True, "triton.cudagraphs": False,
        }
        if mode != "disabled":
            settings["cudagraph_saved_input_schedule"] = True
        self.enterContext(config.patch(settings))
        compiler_origins, bridge_calls = [], []
        collect_saved = saved_transport.collect_terminal_saved_inputs
        make_replay = terminal_replay._make_replay

        def observe_saved(wrapper, inputs):
            indices = collect_saved(wrapper, inputs)
            if V.graph.is_backward:
                origin = saved_transport._active_origin.get()
                if origin is None:
                    raise AssertionError("Opt-in backward lost its active AOT classification")
                compiler_origins.append({
                    "targets": tuple(wrapper.get_graph_input_names()),
                    "origin_targets": origin.targets,
                    "classification": tuple(asdict(row) for row in origin.classification.inputs),
                    "candidates": origin.classification.candidate_indices,
                    "collected": indices,
                    "kinds": inputs.kinds,
                })
            return indices

        def observe_bridge(*args, **kwargs):
            entry = make_replay(*args, **kwargs)
            program, = kwargs["resources"]
            numeric = kwargs["numeric"]
            bridge_calls.append({
                "entry_id": id(entry), "program_id": id(program),
                "release_steps": kwargs.get("release_steps"),
                "integer_indices": numeric.integer_indices,
                "numeric_tape": tuple(numeric.instructions),
            })
            return entry

        self.enterContext(mock.patch.object(saved_transport, "collect_terminal_saved_inputs", new=observe_saved))
        self.enterContext(mock.patch.object(terminal_replay, "_make_replay", new=observe_bridge))
        model = SavedFanout()
        outer_compilations = counters["stats"]["unique_graphs"]
        compiled = torch.compile(model, fullgraph=True, dynamic=True)
        held, expected_held = [], []

        def invoke(iteration, width, observer=None):
            shape = (32, width)
            value = torch.linspace(-0.8, 0.9, 32 * width, device=device).reshape(shape).add(iteration / 32).requires_grad_()
            residual = torch.linspace(-0.6, 0.7, 32 * width, device=device).reshape(shape).sub(iteration / 64).requires_grad_()
            weight = torch.linspace(0.5, 1.0, 32 * width, device=device).reshape(shape).requires_grad_()
            for argument in (value, residual, weight):
                torch._dynamo.mark_static(argument, 0)
            expected_value = value.detach().clone().requires_grad_()
            expected_residual = residual.detach().clone().requires_grad_()
            expected_weight = weight.detach().clone().requires_grad_()
            expected = (expected_value.sin() + expected_residual.cos()).sigmoid() * expected_weight
            actual = compiled(value, residual, weight)
            self.assertEqual(actual, expected)
            tangent = torch.ones_like(actual)
            expected_gradients = torch.autograd.grad(expected, (expected_value, expected_residual, expected_weight), tangent)
            retains = (True, False) if observer is not None and mode == "retain_graph" else (False,)
            for retain in retains:
                if observer is None:
                    gradients = torch.autograd.grad(actual, (value, residual, weight), tangent, retain_graph=retain)
                else:
                    before = len(observer.calls)
                    with observer:
                        gradients = torch.autograd.grad(actual, (value, residual, weight), tangent, retain_graph=retain)
                    self.assertEqual(observer.errors, [])
                    self.assertEqual(observer.pending, {})
                    self.assertEqual(len(observer.calls), before + 1)
                    observer.calls[-1]["retain_graph"] = retain
                    observer.calls[-1]["width"] = width
                self.assertEqual(gradients, expected_gradients)
                held.append((actual, gradients))
                expected_held.append((expected.detach(), tuple(value.detach() for value in expected_gradients)))

        invoke(0, 13)
        self.assertEqual(counters["stats"]["unique_graphs"] - outer_compilations, 1)
        backwards = tuple(item for item in policy.installations if item.artifact.fx_kwargs["is_backward"])
        self.assertEqual(len(backwards), 1, f"Actual backward missing: {policy.declines}")
        backward, = backwards
        self.assertEqual(backward.status, "ready", backward.decline)
        self.assertIsNone(backward.decline)
        self.assertEqual(len(backward.variants), 1)
        variant, = backward.variants
        program, entry = variant.program, variant.entry
        self.assertIs(type(entry), torch._C._CUDAGraphBoxedReplay)
        self.assertIs(backward.artifact.current_callable, backward.entry)
        self.assertIs(program.origin.wrapper, backward.original)
        self.assertIs(program.origin.attachment.metadata, backward.metadata)
        self.assertEqual(program.saved_input_indices, backward.metadata.saved_input_indices)
        saved_indices = program.saved_input_indices
        if mode == "disabled":
            self.assertEqual(compiler_origins, [])
            self.assertEqual(saved_indices, ())
        else:
            self.assertEqual(len(compiler_origins), 1)
            compiler_origin, = compiler_origins
            self.assertEqual(compiler_origin["targets"], compiler_origin["origin_targets"])
            self.assertEqual(compiler_origin["kinds"], program.contract.kinds)
            classified = tuple(row["boxed_index"] for row in compiler_origin["classification"]
                               if row["kind"] == "activation")
            self.assertTrue(classified)
            self.assertEqual(saved_indices, classified)
            self.assertEqual(saved_indices, compiler_origin["candidates"])
            self.assertEqual(saved_indices, compiler_origin["collected"])

        self.assertEqual(len(program.integer_inputs), 1)
        integer_index = program.integer_inputs[0].boxed_index
        extent = IntExpr("boxed", integer_index)
        tensor_indices = tuple(index for index, kind in enumerate(program.contract.kinds) if kind == "tensor")
        for layout in program.allocations:
            self.assertEqual((layout.size, layout.stride), ((32, extent), (extent, 1)))
        calls = tuple(event for event in program.events if type(event) is TerminalCall)
        self.assertGreater(len(calls), 1)
        self.assertTrue(all(type(event) in (Allocate, Normalize, TerminalCall) for event in program.events))
        self.assertEqual(sum(output is not None for output in program.outputs), 3)
        self.assertTrue(any(output is None for output in program.outputs))
        self.assertTrue(all(output is None or type(output) is OwnedBuffer for output in program.outputs))
        launchers = []
        for call in calls:
            launcher, = call.provider.launchers
            module = launcher.__globals__["runner"].__self__
            self.assertIs(type(module), StaticallyLaunchedCudaKernel)
            launchers.append(type(module).__name__)
        bridge, = (row for row in bridge_calls if row["entry_id"] == id(entry))
        self.assertEqual(bridge["program_id"], id(program))
        self.assertEqual(bridge["integer_indices"], (integer_index,))
        steps = bridge["release_steps"]
        early_pairs = []
        if mode == "disabled":
            self.assertIsNone(steps)
        else:
            self.assertIsNotNone(steps)
            self.assertEqual(tuple(index for kind, index in steps if kind == "allocate"),
                             tuple(range(len(program.allocations))))
            self.assertEqual(tuple(index for kind, index in steps if kind == "kernel"), tuple(range(len(calls))))
            self.assertCountEqual(tuple(index for kind, index in steps if kind == "drop"), saved_indices)
            expected_order, ordinal = [], 0
            for event in program.events:
                if type(event) is Allocate:
                    expected_order.append(("allocate", next(index for index, layout in enumerate(program.allocations)
                                                            if layout.source == BufferSource(event.value_id))))
                elif type(event) is TerminalCall:
                    expected_order.append(("kernel", ordinal))
                    ordinal += 1
            self.assertEqual(tuple(step for step in steps if step[0] != "drop"), tuple(expected_order))
            for index in saved_indices:
                uses = []
                for ordinal, call in enumerate(calls):
                    roots = tuple(argument.source.root if type(argument.source) is PointerSource else argument.source
                                  for argument in call.arguments)
                    if InputSource(index) in roots:
                        uses.append(ordinal)
                self.assertTrue(uses, "This workload requires a consumed saved activation")
                position = steps.index(("drop", index))
                self.assertEqual(next(step for step in reversed(steps[:position]) if step[0] != "drop"),
                                 ("kernel", uses[-1]))
                for kind, number in steps[position + 1:]:
                    if kind == "allocate" and program.allocations[number] in program.outputs:
                        early_pairs.append((index, program.outputs.index(program.allocations[number])))
            self.assertTrue(early_pairs, "Saved input release must precede a returned gradient allocation")

        observer = NativeObservation(backward, saved_indices, mode == "external_alias")
        installation_count, bridge_count = len(policy.installations), len(bridge_calls)
        for iteration, width in enumerate((2, 7, 35, 36), start=1):
            invoke(iteration, width, observer)
            self.assertEqual(counters["stats"]["unique_graphs"] - outer_compilations, 1)
            self.assertEqual(len(policy.installations), installation_count)
            self.assertEqual(len(bridge_calls), bridge_count)
            self.assertEqual(len(backward.variants), 1)
            self.assertIs(backward.variants[0], variant)
            self.assertIs(backward.artifact.current_callable, backward.entry)
            self.assertEqual(backward.status, "ready")
            self.assertFalse(entry.failed)
        self.assertEqual(observer.frames, [])
        self.assertEqual(len(observer.calls), 8 if mode == "retain_graph" else 4)
        reuse = []
        for call_index, call in enumerate(observer.calls):
            inputs = {index: (pointer, expired) for index, pointer, expired in call["inputs"]}
            self.assertEqual(tuple(inputs), tensor_indices)
            self.assertEqual(call["integers"], ((integer_index, call["width"]),))
            self.assertEqual(tuple(pointer is None for pointer in call["output_pointers"]),
                             tuple(output is None for output in program.outputs))
            self.assertEqual(call["output_shapes"],
                             [None if output is None else (32, call["width"]) for output in program.outputs])
            retained = call["retain_graph"] or mode == "external_alias"
            for index in saved_indices:
                self.assertEqual(inputs[index][1], not retained)
                if retained:
                    self.assertNotIn(inputs[index][0], call["output_pointers"])
            if mode == "disabled":
                self.assertTrue({pointer for pointer, _ in inputs.values()}.isdisjoint(call["output_pointers"]))
            for input_index, output_index in early_pairs:
                pointer, expired = inputs[input_index]
                if pointer == call["output_pointers"][output_index]:
                    self.assertTrue(expired)
                    self.assertFalse(retained)
                    reuse.append((call_index, input_index, output_index))
        if mode in ("release", "retain_graph"):
            self.assertTrue(reuse, "No released saved storage was reused for a later gradient")
        else:
            self.assertEqual(reuse, [])
        aliases_released = 0
        if mode == "external_alias":
            self.assertEqual(len(observer.aliases), len(saved_indices) * len(observer.calls))
            self.assertEqual([alias for alias, _, _ in observer.aliases],
                             [expected for _, expected, _ in observer.aliases])
            weak_aliases = tuple(weak for _, _, weak in observer.aliases)
            self.assertTrue(all(not weak.expired() for weak in weak_aliases))
            observer.aliases.clear()
            self.assertTrue(all(weak.expired() for weak in weak_aliases))
            aliases_released = len(weak_aliases)
        torch.cuda.synchronize()
        self.assertEqual(held, expected_held)
        forward_states = [item.status for item in policy.installations if not item.artifact.fx_kwargs["is_backward"]]
        policy.close()
        self.assertTrue(entry.closed)
        self.assertIs(backward.artifact.current_callable, backward.original)
        self.assertEqual(held, expected_held)
        print("TERMINAL_SAVED_INPUT_RELEASE_RESULT=" + json.dumps({
            "accepted": True, "mode": mode,


            "compiler_origins": compiler_origins,
            "saved_input_indices": saved_indices, "integer_indices": [integer_index],
            "tensor_indices": tensor_indices, "release_steps": steps,
            "early_output_pairs": early_pairs, "observed_reuse": reuse,
            "numeric_tape": bridge["numeric_tape"], "launchers": launchers,
            "outer_compilations": counters["stats"]["unique_graphs"] - outer_compilations,
            "widths": [13, 2, 7, 35, 36], "backward_calls": len(calls),
            "native_backward_hits": len(observer.calls), "watched_host_frames": observer.frames,
            "observations": observer.calls, "installations": installation_count,
            "bridge_constructions": bridge_count, "backward_variants": 1,
            "forward_installation_states": forward_states,
            "external_aliases_released": aliases_released,
            "held_outputs_after_close": len(held), "closed": entry.closed,
        }, sort_keys=True))


instantiate_device_type_tests(TestTerminalSavedInputRelease, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
