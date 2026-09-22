# Owner(s): ["module: inductor"]
import gc
import weakref
from contextlib import nullcontext
from types import FunctionType, SimpleNamespace
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
    """The recorder's Triton specialization is Triton's own binder run on the values
    of the trace (torch/cuda/_host_trace_triton.py `_select`: the kernel's generated
    binder cloned with `specialize_impl` replaced by `_SymbolicSpecialization`);
    the record admits a launch when the symbolic run's resolved specialization is
    the concrete binder's entry by entry, and declines it by name otherwise."""

    def setUp(self):
        super().setUp()
        # The binder is target-independent for these argument facts. No device,
        # compiler, loaded kernel, or driver launch is used by this CPU probe.
        self.backend = CUDABackend(GPUTarget("cuda", 80, 32))
        self.assertFalse(torch.cuda.is_initialized())

    def sym(self, value):
        from torch._dynamo.source import ConstantSource
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        env = ShapeEnv()
        symbol = env.create_symbol(value, ConstantSource("n"))
        return env.create_symintnode(symbol, hint=value)

    def binders(self, kernel):
        binder = create_function_from_signature(
            kernel.signature, kernel.params, self.backend
        )
        symbolic = FunctionType(
            binder.__code__,
            {
                **binder.__globals__,
                "specialize_impl": HOOK._SymbolicSpecialization(self.backend),
            },
            binder.__name__,
            binder.__defaults__,
            binder.__closure__,
        )
        return binder, symbolic

    def specializations(self, kernel, concrete, symbolic):
        binder, symbolic_binder = self.binders(kernel)
        _, (selected,), _ = binder(concrete)
        _, (entry,), _ = symbolic_binder(symbolic)
        return selected, HOOK._resolve(entry)

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
        selected, recorded = self.specializations(kernel, value, self.sym(value))
        self.assertIn(selected[0], ("i32", "i64", "constexpr"))
        self.assertEqual(
            recorded,
            selected,
            f"The actual binder selected {selected} for "
            f"{kernel.fn.__name__}({value}), but its recorder derived {recorded}",
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
        selected, recorded = self.specializations(
            kernel,
            HOOK._PointerStandIn(torch.float32, address),
            HOOK._PointerStandIn(torch.float32, self.sym(address)),
        )
        self.assertEqual(selected[0], "*fp32")
        self.assertEqual(
            recorded,
            selected,
            f"The actual binder selected {selected} for "
            f"{kernel.fn.__name__} at {address}, but its recorder derived {recorded}",
        )

    @parametrize("kind", ("integer", "pointer", "int64_range"))
    def test_required_selected_constraints_still_decline(self, kind):
        # a value outside the class the compilation was selected under is another
        # specialization: `_select` declines it by name ("is not the symbolic run's")
        if kind == "integer":
            selected, recorded = self.specializations(ordinary_integer, 16, self.sym(3))
            self.assertEqual(selected, ("i32", "D"))
            self.assertNotEqual(recorded, selected)
        elif kind == "pointer":
            selected, recorded = self.specializations(
                ordinary_pointer,
                HOOK._PointerStandIn(torch.float32, 4096),
                HOOK._PointerStandIn(torch.float32, self.sym(4100)),
            )
            self.assertEqual(selected, ("*fp32", "D"))
            self.assertNotEqual(recorded, selected)
        else:
            # the width class of a value above 2**63 - 1 is u64: outside the
            # 64-bit slot a declared tl.int64 binds, which the record declines
            self.assertEqual(HOOK._int_type(self.sym(2**63 + 3)), "u64")
            self.assertEqual(HOOK._int_type(self.sym(2**33)), "i64")
            self.assertEqual(HOOK._int_type(self.sym(3)), "i32")


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
