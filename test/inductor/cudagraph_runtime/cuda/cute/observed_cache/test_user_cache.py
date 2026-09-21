"""Observed user-cache entries retain exact compiled artifacts through native reuse."""

import ctypes
import hashlib
import json
from pathlib import Path
import sys
from unittest import mock
import weakref

import model

from cutlass import cute
from cutlass.base_dsl.jit_executor import JitCompiledFunction, JitExecutor
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall, CuteInvokeEvent
import torch._inductor.runtime._cudagraph.direct_host as direct_host
from torch._inductor.runtime._cudagraph.frontend import DirectKernelCall
import torch._inductor.runtime._cudagraph.replay as replay
import torch
from torch._inductor.runtime.cudagraph_launch_association import associate_kernel_launches
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


def close_owners(references):
    for reference in references.values():
        owner = reference()
        if owner is not None:
            owner.close()


class TestObservedUserCache(TestCase):
    def check_variants(self, runtime, owner_refs, selected_refs, capsule_refs):
        for variant, key in zip(runtime.variants, (1, 2), strict=True):
            variant.program.check()
            calls = tuple(event for event in variant.program.events
                          if type(event) in (DirectKernelCall, CuTeCall))
            self.assertEqual(tuple(type(call) for call in calls),
                             (DirectKernelCall, CuTeCall, DirectKernelCall))
            self.assertEqual(len(variant.program.allocations), 2)
            owner = owner_refs[key]()
            selected = selected_refs[key]()
            self.assertIsNotNone(owner)
            self.assertIsNotNone(selected)
            call = calls[1]
            capsule = call.receipt.invocation.compilation
            self.assertIs(call.receipt.borrow.owner, owner)
            self.assertIs(capsule.entry_owner, owner)
            self.assertIs(capsule.selected, selected)
            self.assertIs(owner.selected, selected)
            self.assertIs(capsule, capsule_refs[key]())
            self.assertEqual(owner._capture.calls, 1)
            self.assertTrue(owner._native_borrows)
            self.assertIsNotNone(owner._owned_executor)
            self.assertIsNotNone(selected.jit_module)
            event, = (event for event in variant.program.guards.trace.events
                      if type(event) is CuteInvokeEvent)
            self.assertIs(event.entry.owner, owner)
            self.assertIs(event.conversion.conversion, owner.conversion)
            self.assertIs(owner.conversion.function, model.kernels.convert_arguments)
            self.assertEqual(len(event.conversion.outputs), 3)
            guard = variant.guard
            self.assertIsNotNone(guard)
            self.assertEqual(guard.boxed_integer_indices, (0,))
            predicate = ctypes.CFUNCTYPE(ctypes.c_int8, ctypes.POINTER(ctypes.c_int64),
                                        ctypes.POINTER(ctypes.c_double))(guard.function_address)
            for rows in (2, 5, 16, 17, 35, 127):
                self.assertEqual(predicate((ctypes.c_int64 * 1)(rows), None),
                                 int((rows > 16) == (key == 2)))

    def test_cache_selection_and_eviction(self, device):
        with torch.cuda.device(device):
            runtime, add = model.make_runtime(torch.cuda.current_device())
            owner_refs = {key: weakref.ref(owner) for key, owner in model.OWNERS.items()}
            self.addCleanup(close_owners, owner_refs)
            self.addCleanup(add.close)
            self.addCleanup(runtime.close)
            self.enterContext(mock.patch.object(torch, "compile", side_effect=AssertionError("No host compiler")))
            self.enterContext(mock.patch.object(torch.func, "functionalize", side_effect=AssertionError("No functionalization")))
            samples = [(rows, torch.randn((rows, 128), device=device))
                       for rows in (5, 7, 35, 64, 13, 17, 8, 48)]
            self.assertEqual(len({source.data_ptr() for _, source in samples}), len(samples))
            selected_refs, capsule_refs, receipt_refs = {}, {}, {}
            sdk_returns, sdk_compile_calls, ordinary_calls, traces, captures, frames = [], [], [], [], [], []
            ordinary_results, held = [], []
            compile_only_checks, reuse_checks, mutation_checks = [], [], []
            compile_owner = model.ObservedOrdinaryEntry.compile
            compile_sdk = cute.compile
            observe_direct, trace_host, make_replay = direct_host._observe_direct, direct_host.trace_host, replay._make_replay

            def sdk_compile(*args, **kwargs):
                sdk_compile_calls.append(args[0].__name__)
                selected = compile_sdk(*args, **kwargs)
                sdk_returns.append(weakref.ref(selected))
                return selected

            def compile_only(owner, *args, **kwargs):
                previous_profiler = sys.getprofile()
                sys.setprofile(None)
                try:
                    with mock.patch.object(JitCompiledFunction, "run_compiled_program",
                                           side_effect=AssertionError("Compile-only executed selected code")), \
                         mock.patch.object(JitExecutor, "run_compiled_program",
                                           side_effect=AssertionError("Compile-only executed an SDK executor")), \
                         mock.patch.object(JitCompiledFunction, "to",
                                           side_effect=AssertionError("Compile-only created an SDK executor")):
                        selected = compile_owner(owner, *args, **kwargs)
                        self.assertIsInstance(selected, JitCompiledFunction)
                        self.assertIs(selected, sdk_returns[-1]())
                        self.assertIs(owner.selected, selected)
                        self.assertIsNone(selected.jit_module)
                        self.assertIsNone(owner._owned_executor)
                        self.assertIsNone(owner._calls)
                        key = 2 if args[2] > 16 else 1
                        self.assertIs(owner, owner_refs[key]())
                        selected_refs[key] = weakref.ref(selected)
                        compile_only_checks.append(key)
                        if key == 1:
                            before = len(sdk_compile_calls)
                            self.assertIs(compile_owner(owner, *args, **kwargs), selected)
                            self.assertEqual(len(sdk_compile_calls), before)
                            self.assertEqual(owner._capture.calls, 1)
                            reuse_checks.append(key)
                            name = selected.function_name
                            try:
                                selected.function_name = name + "_unobserved_change"
                                with self.assertRaisesRegex(RuntimeError, "selected CuTe code or execution owner changed"):
                                    compile_owner(owner, *args, **kwargs)
                            finally:
                                selected.function_name = name
                            owner.check()
                            self.assertEqual(len(sdk_compile_calls), before)
                            mutation_checks.append(key)
                        return selected
                finally:
                    sys.setprofile(previous_profiler)

            def count_ordinary(frame, event, result):
                if event == "call" and frame.f_code is model.host.__code__:
                    ordinary_calls.append(frame.f_locals["box"][0])

            def observe(*args, **kwargs):
                try:
                    sys.setprofile(count_ordinary)
                    result = observe_direct(*args, **kwargs)
                finally:
                    sys.setprofile(None)
                ordinary_results.append(result)
                return result

            def trace(*args, **kwargs):
                traces.append(args[2][0])
                return trace_host(*args, **kwargs)

            def capture(graph, input_count, allocations, outputs, copies, calls, launches, buffers, stream, **kwargs):
                rows = kwargs["capture_inputs"][0]
                key = 2 if rows > 16 else 1
                self.assertEqual(input_count, 2)
                self.assertEqual(len(allocations), 2)
                self.assertEqual(len(calls), 3)
                nodes = tuple(launch.after[3][0][0] for launch in launches)
                associated = associate_kernel_launches(
                    tuple(launches), graph._inspect_captured_kernel_nodes(nodes))
                grid, block = associated[1].snapshot[4:6]
                self.assertEqual(grid, ((rows + key - 1) // key, 1, 1))
                self.assertEqual(block, (128 * key, 1, 1))
                captures.append({"key": key, "rows": rows, "grid": grid, "block": block})
                return make_replay(graph, input_count, allocations, outputs, copies, calls,
                                   launches, buffers, stream, **kwargs)

            def profile(frame, event, result):
                if event == "call":
                    frames.append((frame.f_code.co_filename, frame.f_code.co_name))

            native_calls = ordinary_references = 0
            with mock.patch.object(cute, "compile", sdk_compile), \
                 mock.patch.object(model.ObservedOrdinaryEntry, "compile", compile_only), \
                 mock.patch.object(direct_host, "_observe_direct", observe), \
                 mock.patch.object(direct_host, "trace_host", trace), \
                 mock.patch.object(replay, "_make_replay", capture):
                for index, (rows, source) in enumerate(samples):
                    if index == 6:
                        self.check_variants(runtime, owner_refs, selected_refs, capsule_refs)
                        model.CACHE.clear()
                        model.ADAPTERS.clear()
                        model.OWNERS.clear()
                        self.assertEqual((model.CACHE, model.ADAPTERS, model.OWNERS), ({}, {}, {}))
                        self.check_variants(runtime, owner_refs, selected_refs, capsule_refs)
                    miss = index in (0, 2)
                    before = len(ordinary_calls)
                    box = [rows, source]
                    if index == 0:
                        actual = runtime(box)
                        native_entry = runtime.entry
                    elif miss:
                        actual = native_entry(box)
                    else:
                        try:
                            sys.setprofile(profile)
                            actual = native_entry(box)
                        finally:
                            sys.setprofile(None)
                        self.assertEqual(frames, [])
                        native_calls += 1
                    self.assertEqual(box, [])
                    self.assertIs(runtime.entry, native_entry)
                    self.assertEqual(len(ordinary_calls) - before, int(miss))
                    self.assertEqual(len(runtime.variants), 1 if index < 2 else 2)
                    if miss:
                        self.assertIs(actual, ordinary_results[-1])
                        key = 2 if rows > 16 else 1
                        capsule_refs[key] = weakref.ref(next(
                            call for call in runtime.variants[-1].program.events if type(call) is CuTeCall
                        ).receipt.invocation.compilation)
                        self.assertIsNotNone(owner_refs[key]()._owned_executor)
                        self.assertIsNotNone(selected_refs[key]().jit_module)
                        receipt_refs[key] = weakref.ref(next(
                            call for call in runtime.variants[-1].program.events if type(call) is CuTeCall
                        ).receipt)
                    if index < 6:
                        key = 2 if rows > 16 else 1
                        self.assertIs(model.CACHE[key], selected_refs[key]())
                        self.assertIs(model.ADAPTERS[model.CACHE[key]].owner, owner_refs[key]())
                        ordinary = model.host([rows, source])
                        self.assertEqual(actual, ordinary)
                        self.assertNotEqual(actual[0].data_ptr(), ordinary[0].data_ptr())
                        ordinary_references += 1
                        self.assertIs(model.CACHE[key], selected_refs[key]())
                    else:
                        self.assertEqual((model.CACHE, model.ADAPTERS, model.OWNERS), ({}, {}, {}))
                    self.assertEqual(actual, ((source + 1) * 2 + rows + 1,))
                    held.append((actual, tuple(value.clone() for value in actual)))

            self.assertEqual(ordinary_calls, [5, 35])
            self.assertEqual(traces, [5, 35])
            self.assertEqual(sdk_compile_calls, ["launch_one", "launch_two"])
            self.assertEqual(compile_only_checks, [1, 2])
            self.assertEqual(reuse_checks, [1])
            self.assertEqual(mutation_checks, [1])
            self.assertEqual(native_calls, 6)
            self.assertEqual(ordinary_references, 6)
            self.assertEqual(len(captures), 2)
            self.assertIsNot(selected_refs[1](), selected_refs[2]())
            self.assertNotEqual(owner_refs[1]().compiled_sha256, owner_refs[2]().compiled_sha256)
            self.assertEqual(len({actual[0].data_ptr() for actual, _ in held}), len(held))
            self.check_variants(runtime, owner_refs, selected_refs, capsule_refs)
            runtime.close()
            self.assertTrue(runtime.closed)
            self.assertTrue(native_entry.closed)
            for key in (1, 2):
                self.assertTrue(receipt_refs[key]().closed)
                self.assertEqual(owner_refs[key]()._native_borrows, set())
            add.close()
            close_owners(owner_refs)
            for actual, expected in held:
                self.assertEqual(actual, expected)
            root = Path(__file__).resolve().parent
            report = {
                "accepted": True,

                "sources": {str(root / name): hashlib.sha256((root / name).read_bytes()).hexdigest()
                            for name in ("kernels.py", "model.py", Path(__file__).name)},
                "samples": [rows for rows, _ in samples],
                "ordinary_compilations": len(sdk_compile_calls),
                "ordinary_misses": len(ordinary_calls),
                "preparations": len(traces),
                "captures": captures,
                "variants": len(runtime.variants),
                "native_hits": native_calls,
                "ordinary_references": ordinary_references,
                "torch_references": len(samples),
                "compile_only_no_launch_checks": len(compile_only_checks),
                "same_owner_compile_reuse_checks": len(reuse_checks),
                "selected_state_mutation_checks": len(mutation_checks),
                "post_eviction_native_hits": 2,
                "user_caches_empty": not model.CACHE and not model.ADAPTERS and not model.OWNERS,
                "python_frames_on_native_hits": frames,
                "held_outputs": len(held),
                "held_outputs_after_close": True,
            }
            print("OBSERVED_USER_CACHE_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


instantiate_device_type_tests(TestObservedUserCache, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()

