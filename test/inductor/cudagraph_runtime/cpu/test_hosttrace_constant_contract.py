# Owner(s): ["module: inductor"]

import struct
from types import SimpleNamespace

import torch
from torch._inductor.runtime._cudagraph.direct_hosttrace import LoweredTape
from torch.cuda import _host_trace
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TaggedFloat(float):
    tag: str

    def __new__(cls, value, tag):
        result = super().__new__(cls, value)
        result.tag = tag
        return result

    def __eq__(self, other):
        return (
            type(other) is type(self)
            and super().__eq__(other)
            and self.tag == other.tag
        )


class TaggedComplex(complex):
    tag: str

    def __new__(cls, value, tag):
        result = super().__new__(cls, value)
        result.tag = tag
        return result

    def __eq__(self, other):
        return (
            type(other) is type(self)
            and super().__eq__(other)
            and self.tag == other.tag
        )


@instantiate_parametrized_tests
class TestHostTraceConstantContract(TestCase):
    def setUp(self):
        super().setUp()
        self.tensor = torch.empty(0)

    def contract_holds(self, before, after):
        args = (self.tensor, before)
        tape = SimpleNamespace(
            nargs=2,
            positions=[0],
            constants=_host_trace._constants(args, [0]),
        )
        return LoweredTape.contract_holds(
            SimpleNamespace(tape=tape), (self.tensor, after)
        )

    @parametrize(
        "before,after",
        (
            (0.0, -0.0),
            (complex(0.0, 1.0), complex(-0.0, 1.0)),
            (complex(1.0, 0.0), complex(1.0, -0.0)),
            ([None, (0.0,)], [None, (-0.0,)]),
        ),
    )
    def test_signed_zero_changes_miss(self, before, after):
        self.assertFalse(self.contract_holds(before, after))
        self.assertFalse(self.contract_holds(after, before))

    @parametrize("kind", ("float", "complex_real", "complex_imag"))
    def test_fresh_nan_with_same_bits_hits(self, kind):
        before = float("nan")
        after = float("nan")
        if kind == "complex_real":
            before, after = complex(before, 1.0), complex(after, 1.0)
        elif kind == "complex_imag":
            before, after = complex(1.0, before), complex(1.0, after)
        self.assertTrue(self.contract_holds(before, after))

    @parametrize(
        "bits",
        (0xFFF8000000000000, 0x7FF8000000000001, 0x7FF0000000000001),
    )
    def test_changed_nan_bits_miss(self, bits):
        before = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000000))[0]
        after = struct.unpack("<d", struct.pack("<Q", bits))[0]
        self.assertFalse(self.contract_holds(before, after))

    @parametrize("before,after", ((0, 0.0), (False, 0), (True, 1.0)))
    def test_numeric_type_changes_miss(self, before, after):
        self.assertFalse(self.contract_holds(before, after))

    @parametrize("scalar_type", (TaggedFloat, TaggedComplex))
    def test_scalar_subclass_equality_is_preserved(self, scalar_type):
        before = scalar_type(1, "original")
        self.assertTrue(self.contract_holds(before, scalar_type(1, "original")))
        self.assertFalse(self.contract_holds(before, scalar_type(1, "changed")))

    def test_list_tuple_size_equivalence_is_preserved(self):
        self.assertTrue(self.contract_holds([None, -0.0], (None, -0.0)))
        self.assertTrue(self.contract_holds([2, 3], torch.Size((2, 3))))


if __name__ == "__main__":
    run_tests()
