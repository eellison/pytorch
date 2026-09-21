"""Pin observed CuTe globals at first compilation, after ordinary owner setup."""

import struct
import sys
from pathlib import Path

import torch
from torch._inductor.runtime._cudagraph import _sdk


_sdk.activate()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cuda/cute"))
from fixture_support import load_source

fixture = load_source("_source_dependency_fixture", ROOT / "cuda/direct_autotune/cute_fixture.py")

from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute.runtime import from_dlpack
from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.conversion_trace import (
    ConversionFunction,
)
from torch._inductor.runtime._cudagraph._compiler.entry_signature import SignaturePolicy
from torch._inductor.runtime._cudagraph._compiler.frontend import _TRACE_LOCK
from torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner import ObservedOrdinaryEntry
from torch._inductor.runtime._cudagraph.direct_cute import DirectCuTe
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


def replacement(*args):
    return None


class TestSourceDependencies(TestCase):
    def make_owner(self):
        entry, kernel = fixture.make_fixture()
        namespace = entry.target.__wrapped__.__globals__
        namespace["ADAPTER"] = None
        namespace["UNRELATED"] = object()
        owner = ObservedOrdinaryEntry(entry, kernel, policy=SignaturePolicy(32, 64, 16, "stream"),
                                      conversion=namespace["convert_arguments"])
        self.addCleanup(owner.close)
        return owner, namespace

    def test_setup_binding_is_pinned_before_sdk_preprocessing(self):
        owner, namespace = self.make_owner()
        namespace["ADAPTER"] = DirectCuTe(owner)
        namespace["UNRELATED"] = object()
        owner.check()
        self.assertEqual(owner._globals, ())
        function = owner.entry.target.__wrapped__
        code = function.__code__
        with _TRACE_LOCK, owner._compilation_context():
            saved, = (values for current, values in owner._globals if current is namespace)
            self.assertIs(saved["ADAPTER"], namespace["ADAPTER"])
            self.assertIs(saved["UNRELATED"], namespace["UNRELATED"])
            BaseDSL._preprocess_and_replace_code(function)
        self.assertIsNot(function.__code__, code)
        owner.check()
        self.assertIsNone(owner.selected)

    @parametrize("change", ("replace", "delete"))
    def test_existing_binding_changes_after_pinning_still_fail(self, change):
        owner, namespace = self.make_owner()
        with _TRACE_LOCK, owner._compilation_context():
            pass
        if change == "replace":
            namespace["UNRELATED"] = object()
        else:
            del namespace["UNRELATED"]
        with self.assertRaisesRegex(RuntimeError, "Original CuTe source globals changed"):
            owner.check()

    def test_new_namespace_names_keep_existing_semantics(self):
        owner, namespace = self.make_owner()
        with _TRACE_LOCK, owner._compilation_context():
            pass
        namespace["ADDED_AFTER_PINNING"] = object()
        owner.check()

    def test_failed_compilation_context_keeps_attempted_snapshot(self):
        owner, namespace = self.make_owner()
        original = namespace["UNRELATED"]
        with self.assertRaisesRegex(RuntimeError, "compilation failed"):
            with _TRACE_LOCK, owner._compilation_context():
                saved = owner._globals
                raise RuntimeError("compilation failed")
        self.assertIsNone(owner.selected)
        self.assertIs(owner._globals, saved)
        namespace["UNRELATED"] = object()
        with self.assertRaisesRegex(RuntimeError, "Original CuTe source globals changed"):
            owner.check()
        namespace["UNRELATED"] = original
        owner.check()
        with _TRACE_LOCK, owner._compilation_context():
            self.assertIs(owner._globals, saved)

    @parametrize("change", ("kernel_code", "host_defaults", "converter_code"))
    def test_callable_and_converter_checks_start_at_construction(self, change):
        owner, namespace = self.make_owner()
        if change == "kernel_code":
            owner.kernel.__wrapped__.__code__ = replacement.__code__
        elif change == "host_defaults":
            owner.entry.target.__wrapped__.__defaults__ = (None,)
        else:
            namespace["convert_arguments"].__code__ = replacement.__code__
        self.assertEqual(owner._globals, ())
        with self.assertRaisesRegex(RuntimeError, "callable code, defaults or receiver changed|conversion code or its bindings changed"):
            owner.check()

    @parametrize("nested", (False, True))
    def test_converter_signed_zero_changes_view(self, nested):
        config = (0.0,) if nested else 0.0

        def converter(value):
            number = config[0] if type(config) is tuple else config
            view = value[1:] if str(number).startswith("-") else value[:-1]
            return (from_dlpack(view, assumed_align=8),)

        conversion = ConversionFunction(converter)
        value = FakeTensorMode().from_tensor(torch.arange(8, dtype=torch.int64))
        trace = conversion.trace((value,))
        self.assertEqual(trace.outputs[0].tensor.storage_offset(), 0)
        self.assertEqual(trace.outputs[0].tensor.shape, (7,))
        config = (float("0.0"),) if nested else float("0.0")
        conversion.check()
        trace.check()

        config = (-0.0,) if nested else -0.0
        reason = "conversion code or its bindings changed"
        with self.assertRaisesRegex(RuntimeError, reason):
            conversion.check()
        with self.assertRaisesRegex(RuntimeError, reason):
            trace.check()
        conversion = ConversionFunction(converter)
        trace = conversion.trace((value,))
        self.assertEqual(trace.outputs[0].tensor.storage_offset(), 1)
        self.assertEqual(trace.outputs[0].tensor.shape, (7,))
        config = (float("-0.0"),) if nested else float("-0.0")
        conversion.check()
        trace.check()

    def test_converter_unchanged_nan_payload_is_stable(self):
        bits = bytes.fromhex("420000000000f87f")
        config = struct.unpack("<d", bits)[0]

        def converter(value):
            view = value[1:] if config != config else value[:-1]
            return (from_dlpack(view, assumed_align=8),)

        conversion = ConversionFunction(converter)
        value = FakeTensorMode().from_tensor(torch.arange(8, dtype=torch.int64))
        trace = conversion.trace((value,))
        self.assertEqual(trace.outputs[0].tensor.storage_offset(), 1)
        config = struct.unpack("<d", bits)[0]
        conversion.check()
        trace.check()


instantiate_parametrized_tests(TestSourceDependencies)

if __name__ == "__main__":
    run_tests()
