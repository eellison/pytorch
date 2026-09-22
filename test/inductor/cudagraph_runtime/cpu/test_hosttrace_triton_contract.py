# Owner(s): ["module: inductor"]
import gc
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.runtime.jit import create_function_from_signature

import torch
from torch._inductor.runtime._cudagraph.direct_hosttrace import (
    _HostTraceTritonModule,
    HostTraceLoweringDeclined,
)
from torch.cuda import _host_trace_triton as HOOK
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@triton.jit
def ordinary_integer(n):
    pass


@triton.jit(do_not_specialize=["n"])
def generic_integer(n):
    pass


@triton.jit(do_not_specialize_on_alignment=["n"])
def integer_without_alignment(n):
    pass


@triton.jit
def explicit_i64(n: tl.int64):
    pass


@triton.jit
def ordinary_pointer(p):
    pass


@triton.jit(do_not_specialize_on_alignment=["p"])
def pointer_without_alignment(p):
    pass


class Declined(ValueError):
    pass


def decline(message):
    raise Declined(message)


@instantiate_parametrized_tests
class TestRecordedTritonSpecialization(TestCase):
    def setUp(self):
        super().setUp()
        # The binder is target-independent for these argument facts. No device,
        # compiler, loaded kernel, or driver launch is used by this CPU probe.
        self.backend = CUDABackend(GPUTarget("cuda", 80, 32))
        self.assertFalse(torch.cuda.is_initialized())

    def attributes(self, kernel, argument):
        binder = create_function_from_signature(
            kernel.signature, kernel.params, self.backend
        )
        _, (specialization,), _ = binder(argument)
        kind, descriptor = specialization
        attributes = (
            tuple(tuple(row) for row in self.backend.parse_attr(descriptor))
            if isinstance(descriptor, str)
            else ()
        )
        return kind, attributes

    @parametrize(
        "kernel,value",
        (
            (ordinary_integer, 3),
            (ordinary_integer, 16),
            (generic_integer, 1),
            (generic_integer, 16),
            (integer_without_alignment, 16),
            (explicit_i64, 3),
            (explicit_i64, 1),
            (explicit_i64, 2**33),
        ),
    )
    def test_selected_integer_signature_is_admitted(self, kernel, value):
        kind, attributes = self.attributes(kernel, value)
        self.assertIn(kind, ("i32", "i64"))
        row = SimpleNamespace(formal="n", attributes=attributes)
        try:
            HOOK._int_guards(
                row, value, {"i32": 32, "i64": 64}[kind], decline, kernel.params[0]
            )
        except Declined as error:
            self.fail(
                f"The actual binder selected {kind}, {attributes} for "
                f"{kernel.fn.__name__}({value}), but its recorder rejected it: {error}"
            )

    @parametrize(
        "kernel,address",
        (
            (ordinary_pointer, 4096),
            (pointer_without_alignment, 4096),
            (pointer_without_alignment, 4100),
        ),
    )
    def test_selected_pointer_signature_is_admitted(self, kernel, address):
        pointer = HOOK._PointerStandIn(torch.float32, address)
        kind, attributes = self.attributes(kernel, pointer)
        self.assertEqual(kind, "*fp32")
        alignment = max((1, *(value for _, value in attributes)))
        tensor = SimpleNamespace(_root=SimpleNamespace(allocation=False))
        try:
            HOOK._alignment_guard(
                "p", tensor, address, 0, alignment, decline, kernel.params[0]
            )
        except Declined as error:
            self.fail(
                f"The actual binder selected {kind}, {attributes} for "
                f"{kernel.fn.__name__} at {address}, but its recorder rejected it: {error}"
            )

    @parametrize("kind", ("integer", "pointer", "int64_range"))
    def test_required_selected_constraints_still_decline(self, kind):
        if kind == "integer":
            _, attributes = self.attributes(ordinary_integer, 16)
            row = SimpleNamespace(formal="n", attributes=attributes)
            with self.assertRaisesRegex(Declined, "divisibility"):
                HOOK._int_guards(row, 3, 32, decline, ordinary_integer.params[0])
        elif kind == "pointer":
            tensor = SimpleNamespace(_root=SimpleNamespace(allocation=False))
            with self.assertRaisesRegex(Declined, "alignment"):
                HOOK._alignment_guard(
                    "p", tensor, 4100, 0, 16, decline, ordinary_pointer.params[0]
                )
        else:
            row = SimpleNamespace(formal="n", attributes=())
            with self.assertRaisesRegex(Declined, "int64 range"):
                HOOK._int_guards(row, 2**63 + 3, 64, decline, explicit_i64.params[0])


class _Binary:
    module = 101
    function = 202
    name = "borrowed_kernel"


class _StaticModule:
    num_warps = 4
    shared = 0

    def _borrow_for_cudagraph(self):
        return self


@instantiate_parametrized_tests
class TestEagerTritonModuleOwnership(TestCase):
    def module(self):
        from cuda.bindings import driver

        record = SimpleNamespace(
            binary=_Binary(),
            owner=SimpleNamespace(
                device_index=0,
                module=_StaticModule(),
                abi_layout=(),
                check=lambda: None,
            ),
        )
        with (
            mock.patch.object(torch.cuda, "device", return_value=nullcontext()),
            mock.patch.object(
                driver,
                "cuCtxGetCurrent",
                return_value=(driver.CUresult.CUDA_SUCCESS, 1),
            ),
        ):
            return _HostTraceTritonModule(record, 0)

    @parametrize("field", ("module", "function"))
    def test_replaced_loaded_handle_declines(self, field):
        module = self.module()
        setattr(module.record.binary, field, 303)
        with self.assertRaisesRegex(HostTraceLoweringDeclined, "loaded function"):
            module.check()

    def test_graph_borrow_retains_the_launched_binary(self):
        module = self.module()
        reference = weakref.ref(module.record.binary)
        borrowed = module._borrow_for_cudagraph()
        del module
        gc.collect()
        self.assertIsNotNone(reference())
        del borrowed
        gc.collect()
        self.assertIsNone(reference())


if __name__ == "__main__":
    run_tests()
