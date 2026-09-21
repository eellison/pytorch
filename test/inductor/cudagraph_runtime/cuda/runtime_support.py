"""Shared configuration and native observation for generated runtime tests."""

from pathlib import Path
import sys
import threading
from unittest import mock

import torch
import torch._inductor.runtime._cudagraph.policy as terminal_policy
import torch._inductor.runtime._cudagraph.replay as terminal_replay
from torch._inductor.runtime._cudagraph.api import NativeTerminalPolicy
from torch._inductor.runtime._cudagraph.frontend import TerminalCall
from torch._dynamo.utils import counters
from torch._functorch import config as aot_config
from torch._inductor import config
from torch._inductor.output_code import CompiledFxGraph
from torch._inductor.utils import fresh_cache
from torch.multiprocessing.reductions import StorageWeakRef
from torch.testing._internal.common_utils import TestCase


POLICY = NativeTerminalPolicy()
IMPLEMENTATION = tuple(Path(terminal_policy.__file__).with_name(name) for name in (
    "frontend.py", "extraction.py", "trace_views.py", "provider_facts.py", "replay.py",
    "guard_export.py", "policy.py", "metadata.py", "cute_adapter.py", "cute_types.py"))


def original_reference(test, compiled, artifact, original, entry, arguments):
    test.assertIs(artifact.current_callable, entry)
    test.assertIs(artifact._cudagraph_original_callable, original)
    calls = 0

    def invoke(box):
        nonlocal calls
        calls += 1
        return original(box)

    with mock.patch.object(artifact, "current_callable", invoke):
        result = compiled(*arguments)
    test.assertEqual(calls, 1)
    test.assertIs(artifact.current_callable, entry)
    return result


class NativeObservation:
    def __init__(self, installation):
        self.artifact = installation.artifact
        self.original_code = installation.original.__code__
        self.pending = {}
        self.calls = []
        self.frames = {}
        self.errors = []
        self.forbidden_files = {sys.modules[name].__file__ for name in (
            "torch._inductor.runtime.cudagraph_boxed_replay",
            "torch._inductor.runtime._cudagraph._compiler.fx_adapter.extraction",
            "torch._inductor.runtime._cudagraph._compiler.fx_adapter.lowering",
            "torch._inductor.runtime._cudagraph._compiler.retained_ir.prototype",
        ) if name in sys.modules}

    def profile(self, frame, event, result):
        try:
            code = frame.f_code
            if event == "call" and (code is self.original_code or code.co_filename in self.forbidden_files):
                name = (code.co_filename, code.co_name)
                self.frames[name] = self.frames.get(name, 0) + 1
            if code is not CompiledFxGraph.__call__.__code__ or frame.f_locals.get("self") is not self.artifact:
                return
            thread = threading.get_ident()
            if event == "call":
                if thread in self.pending:
                    raise AssertionError("Nested backward artifact invocation")
                # Never retain the box, a Tensor, the frame, or its locals dictionary.
                self.pending[thread] = (tuple(
                    (index, value.data_ptr(), StorageWeakRef(value.untyped_storage()))
                    for index, value in enumerate(frame.f_locals["inputs"])
                    if isinstance(value, torch.Tensor)
                ), tuple((index, value) for index, value in enumerate(frame.f_locals["inputs"])
                         if type(value) is int))
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


class MixedTestCase(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        self.enterContext(torch.no_grad())
        self.enterContext(fresh_cache())
        self.enterContext(config.patch({
            "triton.cudagraphs": False, "cudagraph_policy": None,
            "graph_partition": False, "cpp_wrapper": False,
            "use_static_triton_launcher": True, "fx_graph_cache": False,
            "compile_threads": 1, "wrap_inductor_compiled_regions": False,
        }))
        self.modules = {}
        self.addCleanup(self.cleanup_modules)
        self.enterContext(torch._dynamo.config.patch(caching_precompile=False, repro_after=None))
        self.enterContext(aot_config.patch(enable_autograd_cache=False, enable_remote_autograd_cache=False,
                                          bundled_autograd_cache=False))
        self.enterContext(config.patch(force_disable_caches=True, fx_graph_cache=False, fx_graph_remote_cache=False,
            cudagraph_saved_input_schedule=False, allow_buffer_reuse=False, inplace_buffers=False,
            use_fast_triton_launcher=True, normalize_static_input_alignment=True,
            autotune_local_cache=False, autotune_remote_cache=False,
            **{"triton.autotune_at_compile_time": False}))

    def cleanup_modules(self):
        torch.cuda.synchronize()
        for module in self.modules.values():
            module.close()
