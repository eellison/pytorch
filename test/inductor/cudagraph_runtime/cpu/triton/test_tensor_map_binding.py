"""Tensor-map binding validation must finish before calling the CUDA encoder."""

import unittest

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


FIELDS = {
    "node": 0,
    "argument": 0,
    "pointer_index": 0,
    "address_offset_value_index": 0,
    "dimensions": (1, 2),
    "strides": (3,),
    "box_dimensions": (16, 16),
    "data_type": 0,
    "swizzle": 0,
    "nan_fill": False,
}
VALUES = (0, 32, 32, 32)


@unittest.skipIf(torch.version.cuda is None, "requires a CUDA build, without a GPU")
@instantiate_parametrized_tests
class TestTensorMapBinding(TestCase):
    def test_binding_is_immutable_and_owns_its_vectors(self):
        dimensions = [1, 2]
        binding = torch._C._CUDAGraphTensorMapBinding(**{**FIELDS, "dimensions": dimensions})
        dimensions[0] = 99
        returned = binding.dimensions
        returned[0] = 99
        self.assertEqual(tuple(binding.dimensions), (1, 2))
        self.assertEqual(tuple(binding.strides), (3,))
        self.assertEqual(tuple(binding.box_dimensions), (16, 16))
        with self.assertRaises(AttributeError):
            binding.pointer_index = 1

    @parametrize("field", ("node", "pointer_index", "data_type"))
    def test_negative_unsigned_constructor_fields(self, field):
        with self.assertRaises(TypeError):
            torch._C._CUDAGraphTensorMapBinding(**{**FIELDS, field: -1})

    @parametrize("changes,error,message", (
        ({"dimensions": (), "strides": (), "box_dimensions": ()}, ValueError, "matching rank"),
        ({"dimensions": (1,) * 6, "strides": (3,) * 5, "box_dimensions": (16,) * 6}, ValueError, "matching rank"),
        ({"strides": ()}, ValueError, "matching rank"),
        ({"box_dimensions": (16,)}, ValueError, "matching rank"),
        ({"pointer_index": 1}, IndexError, "pointer index"),
        ({"address_offset_value_index": 4}, IndexError, "offset index"),
        ({"dimensions": (1, 4)}, IndexError, "dimension index"),
        ({"strides": (4,)}, IndexError, "stride index"),
        ({"data_type": 13}, ValueError, "unpacked CUDA data type"),
        ({"swizzle": 4}, ValueError, "swizzle"),
        ({"box_dimensions": (0, 16)}, ValueError, "box dimensions"),
        ({"box_dimensions": (257, 16)}, ValueError, "box dimensions"),
    ))
    def test_binding_domains_before_driver(self, changes, error, message):
        binding = torch._C._CUDAGraphTensorMapBinding(**{**FIELDS, **changes})
        with self.assertRaisesRegex(error, message):
            torch._C._cuda_encode_tensor_map(binding, VALUES, (16,))

    @parametrize("values,message", (
        ((0, 0, 32, 32), "dimension"),
        ((0, (1 << 32) + 1, 32, 32), "dimension"),
        ((0, 32, 32, -1), "byte stride"),
        ((0, 32, 32, 1 << 40), "byte stride"),
    ))
    def test_numeric_domains_before_driver(self, values, message):
        binding = torch._C._CUDAGraphTensorMapBinding(**FIELDS)
        with self.assertRaisesRegex(ValueError, message):
            torch._C._cuda_encode_tensor_map(binding, values, (16,))

    @parametrize("pointer,offset,message", (
        ((1 << 64) - 1, 1, "addition overflowed"),
        (8, -9, "subtraction underflowed"),
    ))
    def test_pointer_displacement_before_driver(self, pointer, offset, message):
        binding = torch._C._CUDAGraphTensorMapBinding(**FIELDS)
        with self.assertRaisesRegex(ValueError, message):
            torch._C._cuda_encode_tensor_map(binding, (offset, *VALUES[1:]), (pointer,))

    @parametrize("values,pointers", ((VALUES, (True,)), ((False, *VALUES[1:]), (16,))))
    def test_exact_integer_transport(self, values, pointers):
        binding = torch._C._CUDAGraphTensorMapBinding(**FIELDS)
        with self.assertRaisesRegex(TypeError, "exact integers"):
            torch._C._cuda_encode_tensor_map(binding, values, pointers)


if __name__ == "__main__":
    run_tests()
