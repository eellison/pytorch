import dataclasses
import functools
import itertools
import logging
import sys
import threading

import functorch
from functorch._src.aot_autograd import make_boxed_func
from functorch.compile import min_cut_rematerialization_partition

import weakref
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from  torch.utils import _pytree as pytree
from . import config, overrides
from .debug import DebugContext
from .decomposition import select_decomp_table
from .graph import GraphLowering
from .utils import (
    dynamo_logging,
    dynamo_optimizations,
    dynamo_utils,
    has_incompatible_cudagraph_ops,
)
from .virtualized import V
from typing import Optional, Tuple, List, Any

Ref = Any

log = logging.getLogger(__name__)
ALIGNMENT = 16

aot_autograd = dynamo_optimizations.backends.aot_autograd
normalize_ir = dynamo_optimizations.normalize.normalize_ir
is_aot_autograd_safe_to_run = dynamo_optimizations.training.is_aot_autograd_safe_to_run
count_calls = dynamo_utils.count_calls


@dataclasses.dataclass
class BoxedBool:
    value: bool

    def __bool__(self):
        return self.value

    @staticmethod
    def disable(obj):
        if isinstance(obj, BoxedBool):
            obj.value = False
            return obj
        return False


# copy_ fails when trying to write to tensors with memory overlap,
# for expanded dimensions (a dimension which used to have size 1 -> ?)
# we can select one element from that dimension and write to it
# to achieve writing to all values of that dimension of the input tensor
def get_expanded_dims(t):
    return [i for i in range(t.ndim) if t.stride(i) == 0 and t.size(i) != 1]


def index_expanded_dims(t, expanded_dims):
    for expanded_dim in expanded_dims:
        t = torch.ops.aten.slice(t, expanded_dim, 0, 1)
    return t


def complex_memory_overlap(t):
    # if torch._debug_has_internal_overlap thinks this tensor potentially has
    # memory overlap internally, let's dig deeper to find out whether it's true.
    if torch._debug_has_internal_overlap(t) != 0:
        strides = t.stride()
        sizes = t.shape
        indices = list(range(len(strides)))
        indices = [x for _, x in sorted(zip(strides, indices))]
        for i in range(len(strides)):
            prev_stride = 1 if i == 0 else strides[indices[i - 1]]
            prev_size = 1 if i == 0 else sizes[indices[i - 1]]
            if strides[indices[i]] < prev_stride * prev_size:
                return True
    return False


@functools.lru_cache(None)
def _step_logger():
    return dynamo_logging.get_step_logger(log)


@DebugContext.wrap
@torch.utils._python_dispatch._disable_current_modes()
def compile_fx_inner(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    cudagraphs=None,
    num_fixed=0,
    is_backward=False,
    graph_id=None,
):
    if dynamo_utils.count_calls(gm.graph) == 0:
        return make_boxed_func(gm.forward)

    # lift the maximum depth of the Python interpreter stack
    # to adapt large/deep models
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))

    _step_logger()(
        logging.INFO,
        "torchinductor compiling "
        f"{'BACKWARDS' if is_backward else 'FORWARDS'} "
        f"graph {graph_id}",
    )

    V.debug.fx_graph(gm, example_inputs)

    if cudagraphs is None:
        cudagraphs = config.triton.cudagraphs
    shape_env = None
    for inp in example_inputs:
        if isinstance(inp, FakeTensor) and inp.fake_mode.shape_env is not None:
            shape_env = inp.fake_mode.shape_env

    graph = GraphLowering(gm, shape_env=shape_env, num_static_inputs=num_fixed)
    with V.set_graph_handler(graph):
        graph.run(*example_inputs)
        compiled_fn = graph.compile_to_fn()

    if cudagraphs:
        complex_memory_overlap_inputs = any(
            complex_memory_overlap(t) for t in example_inputs
        )

        if (
            set(graph.device_types) == {"cuda"}
            and not graph.mutated_inputs
            and not has_incompatible_cudagraph_ops(gm)
            and not complex_memory_overlap_inputs
        ):
            compiled_fn = cudagraphify(
                compiled_fn, example_inputs, static_input_idxs=range(num_fixed)
            )
        else:
            BoxedBool.disable(cudagraphs)

            if len(set(graph.device_types)) > 1:
                log.warning("skipping cudagraphs due to multiple devices")
            elif set(graph.device_types) == {"cuda"}:
                if graph.mutated_inputs:
                    log.warning("skipping cudagraphs due to input mutation")
                elif complex_memory_overlap_inputs:
                    log.warning("skipping cudagraphs due to complex input striding")

    result = align_inputs(compiled_fn, example_inputs, range(num_fixed))
    _step_logger()(
        logging.INFO,
        "torchinductor done compiling "
        f"{'BACKWARDS' if is_backward else 'FORWARDS'} "
        f"graph {graph_id}",
    )

    # aot autograd needs to know to pass in inputs as a list
    result._boxed_call = True
    return result


def clone_preserve_strides(x):
    needed_size = (
        sum((shape - 1) * stride for shape, stride in zip(x.size(), x.stride())) + 1
    )
    buffer = torch.as_strided(x, (needed_size,), (1,)).clone()
    return torch.as_strided(buffer, x.size(), x.stride())


def align_inputs(model, inputs, static_input_idxs=()):
    check_inputs = [
        i
        for i in range(len(inputs))
        if (i not in static_input_idxs or (inputs[i].data_ptr() % ALIGNMENT) != 0)
        and inputs[i].device.type == "cuda"
    ]

    if len(check_inputs) == 0:
        return model

    def run(new_inputs):
        for i in check_inputs:
            if new_inputs[i].data_ptr() % ALIGNMENT:
                new_inputs[i] = clone_preserve_strides(new_inputs[i])
        return model(new_inputs)

    return run


@dynamo_utils.dynamo_timed
def cudagraphify(model, inputs, static_input_idxs=()):
    # if using fake tensors, defer cudagraphs until we get real inputs at runtime
    if not any(isinstance(inp, FakeTensor) for inp in inputs):
        return cudagraphify_impl(model, inputs, static_input_idxs)

    compiled_fn = None

    def run(new_inputs):
        nonlocal compiled_fn
        if compiled_fn is None:
            with dynamo_utils.preserve_rng_state():
                compiled_fn = cudagraphify_impl(model, new_inputs, static_input_idxs)

        return compiled_fn(new_inputs)

    return run


def remove_unaligned_input_idxs(inputs, static_input_idxs):
    """
    We require all inputs to be aligned, so introduce a copy for any
    that aren't.
    """
    aligned_static_input_idxs = {
        idx for idx in static_input_idxs if (inputs[idx].data_ptr() % ALIGNMENT) == 0
    }
    if len(aligned_static_input_idxs) != len(static_input_idxs):
        return aligned_static_input_idxs
    return static_input_idxs


# TODO make thread local
# TODO allow different order of threads
# We should allow differnt tapes 


# manages separate tapes
class CUDAGraphTapeManger(object):
    def __init__(self):
        self.recorded_tapes = []
        self.active_tape = []
        self.counter = itertools.count(0)
        self.cuda_graphs_thread_pool = torch.cuda.graph_pool_handle()


class CUDAGraphTape(object):
    def __init__(self):
        
        self.recorded_tape: List[int] = []
        self.executed_tape: List[int] = []

        self.recorded_outputs_weakrefs: List[List[Ref[torch.Tensor]]] = []
        self.executed_outputs_weakrefs: List[List[Ref[torch.Tensor]]] = []

        # identify tensors by their graph id # and output #
        # for the end, we only need to check which outputs died that
        # are cudagraph managed data pointers
        # in the future, it would be better to check storages
        self.recorded_liveness_before_graph : List[List[bool]] = []
        self.recorded_liveness_after_graph : List[List[bool]] = []

        # indices into graph i for output j
        # we check the delta of tensors which are no longer live
        self.expected_dead_indices_before_graph : List[Tuple[int, int]] = []
        self.expected_dead_indices_after_graph : List[Tuple[int, int]] = []

        # on the first invocation, we need to set up which indices we need to check
        # for liveness from the final recording
        self.has_executed = False

    def increment_recording_tape(self) -> int:
        cnt = next(self.counter)
        self.recorded_tape.append(cnt)

        if len(self.recorded_tape == 1):
            self.recorded_liveness_before_graph.append([])
            self.expected_dead_indices_before_graph.append([])
            return cnt

        previous_liveness = self.get_liveness(self.recorded_liveness_after_graph[-1])
        curr_liveness = self.get_liveness(self.recorded_outputs_weakrefs)

        liveness_delta = self.liveness_delta(previous_liveness, curr_liveness)

        self.recorded_liveness_before_graph.append(curr_liveness)
        self.expected_dead_indices_before_graph.append(liveness_delta)

        return cnt

    def record_graph_outputs(self, outputs):

        prev_liveness = self.recorded_liveness_before_graph[-1]
        curr_liveness = self.get_liveness(self.recorded_outputs_weakrefs)

        delta = self.liveness_delta(prev_liveness, curr_liveness)
        self.expected_dead_indices_after_graph.append(delta)

        self.recorded_outputs_weakrefs.append([self.map_to_ref(o) for o in outputs])
        self.recorded_liveness_after_graph.append(self.get_liveness(self.recorded_outputs_weakrefs))

    @staticmethod
    def map_to_ref(t):
        if not isinstance(t, torch.Tensor):
            assert t is None
            return None
        return weakref.ref(t)

    def increment_and_check_execution_tape(self, graph_id):

        # first execution freezes the tape
        if not self.has_executed:
            self.has_executed = True

            # on first execution, we need to check that all previous outputs are dead
            # TODO condition could be relaxed with a config
            live_outputs = []
            final_recorded_liveness = self.recorded_liveness_after_graph[-1]
            for i in range(len(final_recorded_liveness)):
                for j in range(len(final_recorded_liveness[i])):
                    if final_recorded_liveness[i][j]:
                        live_outputs.append((i, j))
            self.expected_dead_indices_before_graph[0] = live_outputs

        if (len(self.executed_tape) > len(self.reorded_tape)):
            self.executed_tape = []


        if not self.check_liveness(self.expected_dead_indices_before_graph[0], self.executed_outputs_weakrefs):
            return False

        self.executed_tape.append(graph_id)
        if self.executed_tape[-1] != self.recorded_tape[len(self.executed_tape) - 1]:
            return False
        if len(self.executed_tape) == len(self.recorded_tape):
            # TODO
            # need to restart all the other executed state as well
            # for the first input, need to check that the last outputs are dead
            self.executed_tape = []

        # check that the dead indices are now dead
        for i, j in self.expected_dead_indices_before_graph(len(self.executed_tape) - 1):
            if self.executed_outputs_weakrefs[i][j]() is not None:
                return False

        return True

    def add_outputs_and_check_liveness(self, outputs):
        self.executed_outputs_weakrefs.append([self.map_to_ref(t) for t in outputs])

        idx = len(self.executed_tape)
        return self.check_liveness(self.expected_dead_indices_after_graph[idx -1], self.executed_outputs_weakrefs)


    @staticmethod
    def get_liveness(output_weakrefs) -> List[List[bool]]:
        def is_live(weak_ref):
            if weak_ref is None:
                return False
            return weak_ref() is not None

        return [pytree.tree_map(outputs, is_live) for outputs in output_weakrefs]

    @staticmethod
    def liveness_delta(prev: List[List[bool]], curr: List[List[bool]]) -> List[Tuple[int, int]]:
        dead_indices = []
        assert len(prev) <= len(curr)
        for i, outputs1, outputs2 in enumerate(zip(prev, curr)):
            assert len(outputs1) == len(outputs2)
            for j, output1, output2 in enumerate(zip(outputs1, outputs2)):
                if output1 != output2:
                    dead_indices.append(i, j)

        return dead_indices

    @staticmethod
    def check_liveness(self, indices: List[Tuple[int, int]], output_refs: List[List[bool]]):
        for i, j in indices:
            if output_refs[i][j]() is not None:
                return False
        return True


    def is_cuda_graph_managed_tensors(self, t):
        for weak_ref in pytree.tree_flatten(self.recorded_outputs_weakrefs):
            if weak_ref is not None and weak_ref() is not None:
                if weak_ref().data_ptr() == t.data_ptr():
                    return True

        return False

    
    def __repr__(self):

        return f"""
            recorded_tape: {self.recorded_tape}, 
            executed_tape: {self.executed_tape},
            expected_dead_indices_before_graph: {self.expected_dead_indices_before_graph},
            expected_dead_indices_after_graph: {self.expected_dead_indices_before_graph},
            "executed outputs weakrefs: {self.executed_outputs_weakrefs}
            """

tape = CUDAGraphTape()

class CudaGraphify(object):
    def __init__(self, model, inputs, static_input_idxs=()):

        assert isinstance(inputs, (list, tuple))
        
        
        self.graph_id = tape.increment_recording_tape()
        self.model = model
        static_input_idxs = remove_unaligned_input_idxs(inputs, static_input_idxs)

        # assume these data ptrs stay constant - id(inp) is not stable, but memory address is
        self.cudagraph_managed_idxs =  [
            idx
            for idx, inp in enumerate(inputs)
            if tape.is_cuda_graph_managed_tensors(inp)
        ]

        static_input_idxs = list(
            set(static_input_idxs) | set(self.cudagraph_managed_idxs)
        )
        self.static_input_idxs = static_input_idxs

        self.static_input_data_ptrs = [
            (inputs[i] if i in static_input_idxs else None) for i in range(len(inputs))
        ]
        self.expanded_dims = [
            get_expanded_dims(x) if idx not in static_input_idxs else []
            for idx, x in enumerate(inputs)
        ]

        stream = torch.cuda.Stream()

        # graph needs to be kept alive
        self.inps_alloc_graph = torch.cuda.CUDAGraph()
        recording_inputs = self.allocate_recording_inputs(inputs)

        self.non_static_inputs_metadata = [
            (self.tensor_metadata(x) if idx not in self.static_input_idxs else None)
            for idx, x in enumerate(recording_inputs)
        ]

        self.warmup(model, stream, recording_inputs)

        self.graph = torch.cuda.CUDAGraph()

        # on the first invocation, return the first recorded outputs, because their memory
        # is correctly accounted for in the CUDAGraphs caching allocator, so on subsequent cudagraph
        # recording we are tracing with a valid caching allocator state
        # TODO - consider checkpointing the current cudagraph memory pool state,
        # so that cudagraphs can multiple different have multiple, different
        # cudagraph invocation paths
        self.outputs = self.record(model, stream, recording_inputs)
        self.outputs_metadata = []
        for out in self.outputs:
            if isinstance(out, torch.Tensor):
                self.outputs_metadata.append(self.tensor_metadata(out, ignore_storage_offset=False))
            else:
                assert out is None
                self.outputs_metadata.append(None)

    def allocate_recording_inputs(self, inputs):
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        recording_inputs = []

        # inputs should be allocated in the cuda graph memory pool
        with torch.cuda.graph(
            self.inps_alloc_graph, pool=tape.cuda_graphs_thread_pool, stream=stream
        ):
            for i, inp in enumerate(inputs):
                if i not in self.static_input_idxs:
                    recording_inputs.append(self.static_input(inp))
                else:
                    recording_inputs.append(inp)

        # TODO - assert that no kernels were launched here
        # TODO: more memory efficient to allocate new input and deallocate
        # old input, one by one

        # Now that the Graph is no longer recording, zero out inputs
        # since they may be used in indexing in graph warmup
        for i, inp in enumerate(recording_inputs):
            if i not in self.static_input_idxs:
                inp.zero_()

        return recording_inputs

    def warmup(self, model, stream, inps):
        # TODO - optimize memory pattern
        torch.cuda.synchronize()
        stream.wait_stream(torch.cuda.current_stream())
        # copy inputs because list will get cleared in model invocation
        with torch.cuda.stream(stream):
            model(list(inps))
        stream.synchronize()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

    def record(self, model, stream, inputs):
        tape.record_

        with torch.cuda.graph(self.graph, stream=stream, pool=tape.cuda_graphs_thread_pool):
            static_outputs = model(inputs)

        # running model should reclaim memory
        assert len(inputs) == 0

        if not isinstance(static_outputs, (list, tuple)):
            static_outputs = (static_outputs,)

        return static_outputs

    def run(self, new_inputs):
        assert len(self.static_input_data_ptrs) == len(new_inputs)
        for idx, inp in enumerate(self.static_input_data_ptrs):
            if inp is not None:
                assert inp.data_ptr() == new_inputs[idx].data_ptr()
            else:
                dst = self.reconstruct_from_tensor_metadata(
                    self.non_static_inputs_metadata[idx]
                )
                src = new_inputs[idx]
                expanded_dims = self.expanded_dims[idx]

                dst = index_expanded_dims(dst, expanded_dims)
                src = index_expanded_dims(src, expanded_dims)
                # TODO - one jit kernel acorss multiple inputs
                dst.copy_(src)

        new_inputs.clear()
        self.graph.replay()

        if self.outputs is not None:
            outputs = self.outputs
            self.outputs = None
            # tape.
            return outputs

        # TODO - share the same storage object across aliased outputs
        return [
            self.reconstruct_from_tensor_metadata(metadata)
            for metadata in self.outputs_metadata
        ]

    def check_invariants(self, inputs):
        # same order of execution
        if not tape.increment_and_check_execution_tape(self.graph_id):
            return False

        # previously managed data pointers remain stable
        for idx in self.cudagraph_managed_idxs:
            if inputs[idx].data_ptr() != self.static_input_data_ptrs[idx]:
                return False

        # the same delta of live tensors that we observed at recording is true now
        # check the same delta of dead tensors from previous cudagraph is dead now
        # if they die early, that's not really a problem

        # if we're the first graph, we should check that all tensors are dead
        # i suppose

    @staticmethod
    def static_input(x):
        """
        Copy input while preserving strides
        """
        needed_size = (
            sum((shape - 1) * stride for shape, stride in zip(x.size(), x.stride())) + 1
        )
        buffer = torch.empty((needed_size,), dtype=x.dtype, device=x.device)
        return torch.as_strided(buffer, x.size(), x.stride())

    @staticmethod
    def tensor_metadata(x, ignore_storage_offset=True):
        assert isinstance(x, torch.Tensor)
        # We ignore the storage offset for inputs, but not for outputs
        # TODO: - should we make the storage resizable ?
        return {
            "nbytes": x.storage().nbytes(),
            "data_ptr": x.storage().data_ptr(),
            "size": x.shape,
            "stride": x.stride(),
            "dtype": x.dtype,
            "device": x.device,
            "storage_offset": x.storage_offset() if not ignore_storage_offset else 0,
        }

    @staticmethod
    def reconstruct_from_tensor_metadata(metadata):
        s = torch._C._construct_storage_from_data_pointer(
            metadata["data_ptr"], metadata["device"], metadata["nbytes"]
        )
        t = torch.empty([0], device=metadata["device"], dtype=metadata["dtype"])
        t.set_(
            source=s,
            storage_offset=metadata["storage_offset"],
            size=metadata["size"],
            stride=metadata["stride"],
        )
        return t


def cudagraphify_impl(model, inputs, static_input_idxs=()):
    return CudaGraphify(model, inputs, static_input_idxs).run


def count_tangents(fx_g: torch.fx.GraphModule):
    """
    Infers which inputs are static for a backwards graph
    """

    def is_not_gradout(x):
        return "tangents" not in x.name

    arg_count = 0
    static_arg_idxs = []
    for n in fx_g.graph.nodes:
        if n.op == "placeholder":
            if is_not_gradout(n):
                static_arg_idxs.append(arg_count)
            arg_count += 1

    assert static_arg_idxs == list(range(len(static_arg_idxs)))
    return len(static_arg_idxs)


_graph_counter = itertools.count(0)


def compile_fx(model_: torch.fx.GraphModule, example_inputs_: List[torch.Tensor]):
    """Main entrypoint to a compile given FX graph"""

    if not is_aot_autograd_safe_to_run(model_, example_inputs_):
        log.warning("Aot Autograd is not safe to run, so falling back to eager")
        return model_

    functorch.compile.config.use_functionalize = True
    functorch.compile.config.use_fake_tensor = True

    with overrides.patch_functions():
        model_ = normalize_ir(model_, example_inputs_)
        model_ = overrides.replace_fx(model_)
        model_ = overrides.fuse_fx(model_, example_inputs_)
    num_example_inputs = len(example_inputs_)
    cudagraphs = BoxedBool(config.triton.cudagraphs and not config.dynamic_shapes)

    graph_id = next(_graph_counter)

    @dynamo_utils.dynamo_timed
    def fw_compiler(model: torch.fx.GraphModule, example_inputs):
        fixed = len(example_inputs) - num_example_inputs
        return compile_fx_inner(
            model,
            example_inputs,
            num_fixed=fixed,
            cudagraphs=cudagraphs,
            graph_id=graph_id,
        )

    @dynamo_utils.dynamo_timed
    def bw_compiler(model: torch.fx.GraphModule, example_inputs):
        fixed = count_tangents(model)
        return compile_fx_inner(
            model,
            example_inputs,
            num_fixed=fixed,
            cudagraphs=cudagraphs,
            is_backward=True,
            graph_id=graph_id,
        )

    with overrides.patch_functions():

        # TODO: can add logging before/after the call to create_aot_dispatcher_function
        # in functorch/_src/aot_autograd.py::aot_module_simplified::aot_function_simplified::new_func
        # once torchdynamo is merged into pytorch
        return aot_autograd(
            model_,
            example_inputs_,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            decompositions=select_decomp_table(),
            partition_fn=functools.partial(
                min_cut_rematerialization_partition, compiler="inductor"
            ),
        )
