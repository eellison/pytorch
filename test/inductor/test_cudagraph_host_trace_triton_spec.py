# Owner(s): ["module: inductor"]
"""The installed Triton's specialization axes against the host trace's symbolic run of
its binder (torch/cuda/_host_trace_triton.py): what native_specialize_impl reads of an
argument and how it classifies it, what the generated binder does with each declaration
flag, and how the compile key is composed. Every check enumerates from the installed
Triton itself and fails on an axis the symbolic run does not cover. CPU only."""

import hashlib
import inspect
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


try:
    import triton

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    import triton.language as tl
    from triton._C.libtriton import native_specialize_impl
    from triton.backends.compiler import BaseBackend
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.runtime import jit as triton_jit
    from triton.runtime.jit import (
        compute_cache_key,
        create_function_from_signature,
        JITFunction,
        KernelParam,
        serialize_specialization_data,
    )

    from torch.cuda import _host_trace_triton as htt

# The sources of the functions the symbolic run relies on, per Triton version: a new
# version fails here until its axes are re-derived (TRITON_SPEC_GUARDS.md) and the hash
# updated; the behavioral checks below are the enumeration itself.
_REVIEWED = {
    "3.8.0": "d9ac55b2e83be1787f0b83adf8187f99d6b556c7650013b4bf7a19aee363cfc0",
}

_INT_BOUNDARIES = (
    0,
    1,
    2,
    15,
    16,
    17,
    31,
    32,
    33,
    255,
    256,
    257,
    2**31 - 2,
    2**31 - 1,
    2**31,
    2**31 + 1,
    2**32 - 1,
    2**32,
    2**32 + 1,
    2**63 - 2,
    2**63 - 1,
    2**63,
    2**63 + 1,
    2**64 - 2,
    2**64 - 1,
)
_INT_GRID = sorted(
    set(range(-70, 71))
    | set(_INT_BOUNDARIES)
    | {-v for v in _INT_BOUNDARIES}
    | {v + d for v in _INT_BOUNDARIES for d in (-16, 16)}
    | {-(v + d) for v in _INT_BOUNDARIES for d in (-16, 16)}
)
_FLAGS = [
    (c, s, a) for c in (False, True) for s in (False, True) for a in (False, True)
]


class _Reads:
    """A tensor stand-in recording every attribute native_specialize_impl reads."""

    def __init__(self, dtype, address):
        object.__setattr__(self, "_dtype", dtype)
        object.__setattr__(self, "_address", address)
        object.__setattr__(self, "reads", [])

    def __getattribute__(self, name):
        if name not in ("reads", "_dtype", "_address"):
            object.__getattribute__(self, "reads").append(name)
        if name == "dtype":
            return object.__getattribute__(self, "_dtype")
        return object.__getattribute__(self, name)

    def data_ptr(self):
        return object.__getattribute__(self, "_address")


class _BackendReads:
    supports_native_tensor_specialization = False

    def __init__(self):
        self.reads = []

    def __getattribute__(self, name):
        if name != "reads":
            object.__getattribute__(self, "reads").append(name)
        if name == "get_tensor_specialization":
            return BaseBackend.get_tensor_specialization
        return object.__getattribute__(self, name)


def _int_specialization(v, specialize, align):
    # the symbolic run's integer rules (htt._SymbolicSpecialization / htt._int_type) on
    # a plain int: what the C++ must agree with on every class boundary
    if specialize and v == 1:
        return ("constexpr", 1)
    ty = htt._int_type(v)
    if not specialize:
        return (ty, None)
    return (ty, BaseBackend.get_int_specialization(v, align=align))


# the pointer types the runtime's owner admits (direct_triton._POINTER_DTYPES), by dtype; this
# Triton names a bool pointer *u1 where the owner's table reads *i1
_DTYPES = {
    torch.bool: "u1",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.uint8: "u8",
    torch.uint16: "u16",
    torch.uint32: "u32",
    torch.uint64: "u64",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
    torch.float64: "fp64",
}


def _dtype_name(dtype):
    return _DTYPES[dtype]


@unittest.skipUnless(HAS_TRITON, "Triton required")
class TestTritonSpecializationAxes(TestCase):
    def test_reviewed_triton_version(self):
        pieces = [
            inspect.getsource(create_function_from_signature),
            inspect.getsource(JITFunction.run),
            inspect.getsource(JITFunction._pack_args),
            inspect.getsource(compute_cache_key),
            inspect.getsource(serialize_specialization_data),
            inspect.getsource(BaseBackend.parse_attr),
            inspect.getsource(BaseBackend.get_int_specialization),
            inspect.getsource(BaseBackend.get_tensor_specialization),
            inspect.getsource(KernelParam),
        ]
        digest = hashlib.sha256("\n".join(pieces).encode()).hexdigest()
        self.assertEqual(
            _REVIEWED.get(triton.__version__),
            digest,
            f"Triton {triton.__version__}'s specialization pipeline is not the reviewed one "
            f"(sha256 {digest}): re-derive the axes the symbolic binder run covers and record it",
        )

    def test_integer_classes_agree_with_the_native_specialization(self):
        # the C++ reads a C long: its equal-to-1, width and divisibility classes must be
        # the symbolic run's on every boundary and its outputs from the known alphabet
        types_seen, specs_seen = set(), set()
        for v in _INT_GRID:
            for is_const, specialize, align in _FLAGS:
                if v < -(2**63) or v > 2**64 - 1:
                    with self.assertRaises(OverflowError):
                        native_specialize_impl(
                            BaseBackend, v, is_const, specialize, align
                        )
                    with self.assertRaises(OverflowError):
                        _int_specialization(v, specialize, align)
                    continue
                native = native_specialize_impl(
                    BaseBackend, v, is_const, specialize, align
                )
                self.assertEqual(
                    native,
                    _int_specialization(v, specialize, align),
                    (v, is_const, specialize, align),
                )
                types_seen.add(native[0])
                specs_seen.add(native[1])
        self.assertEqual(types_seen, {"constexpr", "i32", "i64", "u64"})
        self.assertEqual(specs_seen, {1, "D", "", None})
        # a bool or a float has no value axis; None is a constexpr
        for flags in _FLAGS:
            self.assertEqual(
                native_specialize_impl(BaseBackend, True, *flags), ("u1", None)
            )
            self.assertEqual(
                native_specialize_impl(BaseBackend, 2.5, *flags), ("fp32", None)
            )
            self.assertEqual(
                native_specialize_impl(BaseBackend, None, *flags), ("constexpr", None)
            )

        # an integer type the C++ takes by value only: no dunder of a subclass is called
        class Int(int):
            calls = []

            def __mod__(self, other):
                Int.calls.append("__mod__")
                return int(self) % other

            def __eq__(self, other):
                Int.calls.append("__eq__")
                return int(self) == other

            __hash__ = int.__hash__

        self.assertEqual(
            native_specialize_impl(BaseBackend, Int(32), False, True, True),
            ("i32", "D"),
        )
        self.assertEqual(Int.calls, [])
        with self.assertRaisesRegex(
            TypeError, "failed to specialize argument of type: SymInt"
        ):
            from torch.fx.experimental.symbolic_shapes import ShapeEnv

            native_specialize_impl(
                BaseBackend, ShapeEnv().create_unbacked_symint(), False, True, True
            )

    def test_tensor_reads_and_classes_agree_with_the_native_specialization(self):
        # of a tensor argument the C++ reads the dtype and, where the declaration
        # specializes it, data_ptr(); the address's only class is 16-byte divisibility
        for dtype in _DTYPES:
            for address in (*range(65), 2**40, 2**40 + 8, 2**63 - 16):
                for is_const, specialize, align in _FLAGS:
                    stand_in = _Reads(dtype, address)
                    native = native_specialize_impl(
                        BaseBackend, stand_in, is_const, specialize, align
                    )
                    reads = set(stand_in.reads) - {"__class__"}
                    self.assertEqual(reads, {"dtype", "data_ptr"}, (dtype, address))
                    ty = "*" + ("k" if is_const else "") + _dtype_name(dtype)
                    spec = (
                        None
                        if not specialize
                        else ("D" if address % 16 == 0 and align else "")
                    )
                    self.assertEqual(
                        native,
                        (ty, spec),
                        (dtype, address, is_const, specialize, align),
                    )
                    # the backend view the symbolic run uses: the C++ reads two attributes
                    # of it and hands the argument to Triton's own Python
                    view = _BackendReads()
                    routed = native_specialize_impl(
                        view, _Reads(dtype, address), is_const, specialize, align
                    )
                    self.assertEqual(routed, native)
                    self.assertLessEqual(
                        set(view.reads),
                        {
                            "supports_native_tensor_specialization",
                            "get_tensor_specialization",
                        },
                    )
                    self.assertEqual(
                        "get_tensor_specialization" in view.reads, bool(specialize)
                    )
        self.assertEqual(BaseBackend.parse_attr("D"), [["tt.divisibility", 16]])
        self.assertEqual(BaseBackend.parse_attr(""), [])
        for name in (
            "parse_attr",
            "get_int_specialization",
            "get_tensor_specialization",
        ):
            self.assertIs(getattr(CUDABackend, name), getattr(BaseBackend, name))
        self.assertTrue(CUDABackend.supports_native_tensor_specialization)
        self.assertTrue(
            htt._TensorSpecializationInPython.supports_native_tensor_specialization
            is False
        )

    def test_binder_flags_are_the_declarations(self):
        # the generated binder's per-parameter calls and what it keeps of each result,
        # from a signature with every declaration kind, through a recording specialize_impl
        @triton.jit(do_not_specialize=["dns"], do_not_specialize_on_alignment=["dnsa"])
        def kernel(
            ptr,
            kptr: tl.const,
            dns,
            dnsa,
            plain,
            ann32: tl.int32,
            ann64: tl.int64,
            annf: tl.float32,
            annb: tl.int1,
            BLOCK: tl.constexpr,
            opt: tl.constexpr = 8,
        ):
            pass

        calls = []

        def recording(backend, arg, is_const, specialize, align):
            calls.append((arg, is_const, specialize, align))
            return (f"T{arg}", f"S{arg}")

        binder = create_function_from_signature(
            kernel.signature, kernel.params, BaseBackend
        )
        symbolic = type(binder)(
            binder.__code__,
            {**binder.__globals__, "specialize_impl": recording},
            binder.__name__,
            binder.__defaults__,
        )
        names = [
            "ptr",
            "kptr",
            "dns",
            "dnsa",
            "plain",
            "ann32",
            "ann64",
            "annf",
            "annb",
            "BLOCK",
        ]
        params, specialization, options = symbolic(*names, num_warps=2)
        self.assertEqual(params, {**{n: n for n in names}, "opt": 8})
        self.assertEqual(options, {"num_warps": 2})
        self.assertEqual(
            calls,
            [
                ("ptr", False, True, True),
                ("kptr", True, True, True),
                ("dns", False, False, True),
                ("dnsa", False, True, False),
                ("plain", False, True, True),
                ("ann32", False, True, True),
                ("ann64", False, True, True),
            ],
        )
        self.assertEqual(
            specialization,
            [
                ("Tptr", "Sptr"),
                ("Tkptr", "Skptr"),
                ("Tdns", "Sdns"),
                ("Tdnsa", "Sdnsa"),
                ("Tplain", "Splain"),
                ("i32", "Sann32"),
                ("i64", "Sann64"),
                ("fp32", None),
                ("u1", None),
                ("constexpr", "BLOCK"),
                ("constexpr", 8),
            ],
        )
        # the module's only uses of the per-argument specialization
        source = inspect.getsource(triton_jit)
        self.assertEqual(
            source.count("specialize_impl("), 2
        )  # mangle_type, the generator
        self.assertIn("specialize_impl = native_specialize_impl", source)
        # the flags the generator reads of a KernelParam
        generator = inspect.getsource(create_function_from_signature)
        for flag in (
            "is_constexpr",
            "is_const",
            "do_not_specialize",
            "do_not_specialize_on_alignment",
            "annotation_type",
        ):
            self.assertIn(f"kp.{flag}", generator)
        self.assertIn('["fp", "bf"]', generator)
        self.assertIn('== "u1"', generator)

    def test_compile_key_composition(self):
        run = inspect.getsource(JITFunction.run)
        self.assertIn(
            "bound_args, specialization, options = binder(*args, **kwargs)", run
        )
        self.assertIn(
            "key = compute_cache_key(kernel_key_cache, specialization, options)", run
        )
        self.assertIn(
            'kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug',
            run,
        )
        self.assertIn(
            'kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode',
            run,
        )
        self.assertIn("knobs.runtime.add_stages_inspection_hook", run)
        key = inspect.getsource(compute_cache_key)
        self.assertIn(
            "cache_key = str(replace_callables(specialization)) + str(options)", key
        )
        pack = inspect.getsource(JITFunction._pack_args)
        self.assertIn("backend.parse_attr(get_iterable_path(attrvals, k))", pack)
        self.assertIn('lambda _, val: val == "constexpr"', pack)
        data = inspect.getsource(serialize_specialization_data)
        for field in (
            "'name'",
            "'signature'",
            "'constant_keys'",
            "'constant_vals'",
            "'attrs_keys'",
            "'attrs_vals'",
            "'options'",
            "'key'",
            "'target'",
        ):
            self.assertIn(field, data)
        # the symbolic run's key of a resolved specialization is Triton's own
        entries = [
            ("*fp32", "D"),
            ("*fp32", ""),
            ("i32", "D"),
            ("i32", ""),
            ("constexpr", 1024),
        ]
        self.assertEqual(
            compute_cache_key({}, entries, {"num_warps": 4}),
            str(entries) + str({"num_warps": 4}),
        )

    def test_symbolic_specialization_on_plain_values_is_the_native_one(self):
        # a constant of the call takes the C++'s reading; a tuple the element-wise one
        spec = htt._SymbolicSpecialization(BaseBackend)
        for v in (0, 1, 16, 17, 2**31, True, 2.5, None, (1, 16, 3)):
            for flags in _FLAGS:
                self.assertEqual(
                    spec(BaseBackend, v, *flags),
                    native_specialize_impl(BaseBackend, v, *flags),
                )
        self.assertEqual(htt._resolve(("i32", "D")), ("i32", "D"))


if __name__ == "__main__":
    run_tests()
