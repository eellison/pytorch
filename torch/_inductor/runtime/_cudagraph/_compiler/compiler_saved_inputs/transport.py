"""Optional cold transport between paired AOT callbacks and live wrapper IR."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial

from torch._dynamo import config as dynamo_config
from torch._functorch import config as aot_config
from torch._inductor import config
from torch._inductor.runtime.cudagraph_arg_mapping import WrapperCallRecords
from torch._inductor.virtualized import V

from .classification import ClassificationDeclined
from .handoff import AOTInputOrigins, collect_saved_inputs, same_aot_inputs
from .origins import capture_forward_origins, ForwardOrigins, join_backward_origins
from .schedule import collect_release_schedule


ORIGIN_KEY = "_cudagraph_saved_input_origin"
_NO_ORIGIN = ("saved-input-origin", 1, None)
_MISSING = object()
_active_origin: ContextVar[AOTInputOrigins | None] = ContextVar("saved_input_origin", default=None)


def eligible(inner_compile):
    from torch._inductor import compile_fx as compiler

    standard = inner_compile is compiler.compile_fx_inner or (
        type(inner_compile) is partial and inner_compile.func is compiler.compile_fx_inner
        and not inner_compile.args and inner_compile.keywords.keys() == {"compile_region_name"}
        and inner_compile.keywords["compile_region_name"] is None
    )
    return bool(
        standard and config.cudagraph_saved_input_schedule
        and (config.force_disable_caches or (
            config.fx_graph_cache is True and config.fx_graph_remote_cache is False
            and aot_config.enable_remote_autograd_cache is False
        ))
        and not aot_config.enable_autograd_cache and not aot_config.bundled_autograd_cache
        and not dynamo_config.caching_precompile and dynamo_config.repro_after is None
        and not config.save_args and not config.trace.enabled
        and not config.graph_partition and not config.cpp_wrapper and not config.fx_wrapper
        and not V.aot_compilation
        and compiler.fx_compile_mode is compiler.FxCompileMode.NORMAL
        and not compiler.fx_compile_async and not compiler.fx_compile_progressive
    )


def _origin_key(origin):
    return (
        "saved-input-origin", 1, origin.names, origin.targets,
        tuple((row.boxed_index, row.name, row.kind, row.forward_output, row.saved_index)
              for row in origin.classification.inputs),
    )


@contextmanager
def saved_input_cache_key(graph):
    from torch._inductor import compile_fx as compiler

    previous = graph.meta.get(ORIGIN_KEY, _MISSING)
    origin = _active_origin.get()
    if (eligible(compiler.compile_fx_inner) and same_aot_inputs(origin, graph)
            and type(previous) is tuple and previous == _origin_key(origin)):
        key = previous
    else:
        key = _NO_ORIGIN
        if type(origin) is AOTInputOrigins and origin.module is graph:
            _active_origin.set(None)
    graph.meta[ORIGIN_KEY] = key
    try:
        yield
    finally:
        if previous is _MISSING:
            graph.meta.pop(ORIGIN_KEY, None)
        else:
            graph.meta[ORIGIN_KEY] = previous


class SavedInputCapture:
    def __init__(self, inner_compile):
        self.inner_compile = inner_compile
        self.origins: ForwardOrigins | None = None

    def clear(self):
        self.origins = None

    def forward(self, graph, *, is_inference):
        self.clear()
        if is_inference or not eligible(self.inner_compile):
            return
        try:
            self.origins = capture_forward_origins(graph)
        except ClassificationDeclined:
            pass

    @contextmanager
    def backward(self, graph):
        token = _active_origin.set(None)
        try:
            origins, self.origins = self.origins, None
            if origins is None or not eligible(self.inner_compile):
                del origins
                yield
                return
            try:
                classification = join_backward_origins(origins, graph)
            except ClassificationDeclined:
                classification = None
            del origins
            if classification is None or not classification.candidate_indices:
                yield
                return
            placeholders = tuple(node for node in graph.graph.nodes if node.op == "placeholder")
            targets = tuple(node.target for node in placeholders)
            if any(type(target) is not str for target in targets):
                yield
                return
            if ORIGIN_KEY in graph.meta:
                raise RuntimeError("Saved-input origin metadata is already present")
            origin = AOTInputOrigins(
                graph, graph.graph, placeholders, tuple(node.name for node in placeholders), targets, classification,
            )
            _active_origin.set(origin)
            graph.meta[ORIGIN_KEY] = _origin_key(origin)
            try:
                yield
            finally:
                graph.meta.pop(ORIGIN_KEY, None)
        finally:
            _active_origin.reset(token)


def collect_optional_schedule(wrapper, records, metadata):
    from torch._inductor import compile_fx as compiler

    if (not eligible(compiler.compile_fx_inner) or metadata is None
            or type(records) is not WrapperCallRecords or type(records.version) is not int
            or records.version not in (3, 4)):
        return None
    origin = _active_origin.get()
    if (not same_aot_inputs(origin, V.graph.module)
            or V.graph.module.meta.get(ORIGIN_KEY) != _origin_key(origin)):
        return None
    saved = collect_saved_inputs(origin, wrapper, records, metadata)
    if saved is None or not saved.candidate_indices:
        return None
    return collect_release_schedule(saved, wrapper, records)


def collect_terminal_saved_inputs(wrapper, inputs):
    from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import InputContract
    from torch._inductor import compile_fx as compiler
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.graph import GraphLowering

    if (not eligible(compiler.compile_fx_inner) or type(wrapper) is not PythonWrapperCodegen
            or type(inputs) is not InputContract or type(inputs.kinds) is not tuple):
        return ()
    origin, graph = _active_origin.get(), V.graph
    if (type(graph) is not GraphLowering or not graph.is_backward or graph.cpp_wrapper
            or graph.aot_mode or graph.partition_maps or config.graph_partition or graph.name is not None
            or not same_aot_inputs(origin, graph.module)
            or graph.module.meta.get(ORIGIN_KEY) != _origin_key(origin)
            or tuple(wrapper.get_graph_input_names()) != origin.targets
            or len(origin.classification.inputs) != len(inputs.kinds)):
        return ()
    indices = origin.classification.candidate_indices
    if any(type(index) is not int or not 0 <= index < len(inputs.kinds)
           or inputs.kinds[index] != "tensor" for index in indices):
        return ()
    return indices
