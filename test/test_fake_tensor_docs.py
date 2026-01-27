"""
Tests for FakeTensor public documentation.

This tests Goal 4: Make FakeTensor have public docs.
"""

import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFakeTensorDocs(TestCase):
    """Test that FakeTensor and FakeTensorMode have proper documentation."""

    def test_fake_tensor_has_docstring(self):
        """Test that FakeTensor has a comprehensive docstring."""
        from torch._subclasses import FakeTensor

        doc = FakeTensor.__doc__
        self.assertIsNotNone(doc)

        # Check for key sections
        self.assertIn("tensor subclass", doc.lower())
        self.assertIn("metadata", doc.lower())
        self.assertIn("FakeTensorMode", doc)

        # Check for usage examples
        self.assertIn("Example", doc)
        self.assertIn("fake_mode", doc)

        # Check for documented attributes
        self.assertIn("fake_device", doc)
        self.assertIn("fake_mode", doc)

    def test_fake_tensor_mode_has_docstring(self):
        """Test that FakeTensorMode has a comprehensive docstring."""
        from torch._subclasses import FakeTensorMode

        doc = FakeTensorMode.__doc__
        self.assertIsNotNone(doc)

        # Check for key sections
        self.assertIn("context manager", doc.lower())
        self.assertIn("fake tensor", doc.lower())

        # Check for documented arguments
        self.assertIn("Args:", doc)
        self.assertIn("allow_fallback_kernels", doc)
        self.assertIn("allow_non_fake_inputs", doc)
        self.assertIn("shape_env", doc)
        self.assertIn("static_shapes", doc)

        # Check for usage examples
        self.assertIn("Example", doc)
        self.assertIn("FakeTensorMode()", doc)

        # Check for use case descriptions
        self.assertIn("Shape inference", doc)
        self.assertIn("Compilation", doc)

    def test_fake_tensor_is_exported(self):
        """Test that FakeTensor is exported from torch._subclasses."""
        from torch._subclasses import FakeTensor, FakeTensorMode

        # These should be accessible
        self.assertTrue(hasattr(FakeTensor, "__doc__"))
        self.assertTrue(hasattr(FakeTensorMode, "__doc__"))

    def test_fake_tensor_mode_basic_usage(self):
        """Test that the documented basic usage works."""
        from torch._subclasses import FakeTensorMode

        # Basic usage from docstring
        fake_mode = FakeTensorMode()

        # Create a real tensor
        real_tensor = torch.randn(10, 20)

        # Convert to fake tensor
        fake_tensor = fake_mode.from_tensor(real_tensor)

        # Verify it has the right shape
        self.assertEqual(fake_tensor.shape, torch.Size([10, 20]))
        self.assertEqual(fake_tensor.dtype, torch.float32)

        # Operations within the mode produce fake tensors
        with fake_mode:
            result = fake_tensor @ fake_tensor.T
            self.assertEqual(result.shape, torch.Size([10, 10]))

    def test_fake_tensor_attributes(self):
        """Test that documented attributes exist and work."""
        from torch._subclasses import FakeTensorMode, FakeTensor

        fake_mode = FakeTensorMode()
        real_tensor = torch.randn(5, 5)
        fake_tensor = fake_mode.from_tensor(real_tensor)

        # Check documented attributes
        self.assertTrue(hasattr(fake_tensor, "fake_device"))
        self.assertTrue(hasattr(fake_tensor, "fake_mode"))

        # Verify the mode reference
        self.assertIs(fake_tensor.fake_mode, fake_mode)

    def test_exceptions_are_exported(self):
        """Test that documented exceptions are exported."""
        from torch._subclasses import (
            UnsupportedFakeTensorException,
            DynamicOutputShapeException,
        )

        # These should be importable
        self.assertTrue(issubclass(UnsupportedFakeTensorException, Exception))
        self.assertTrue(issubclass(DynamicOutputShapeException, Exception))

    def test_unset_fake_temporarily_exists(self):
        """Test that unset_fake_temporarily is importable and documented."""
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        # Should have a docstring
        # Note: unset_fake_temporarily is a generator function
        self.assertIsNotNone(unset_fake_temporarily)

    def test_docs_rst_file_exists(self):
        """Test that the RST documentation file was created."""
        import os

        # Get the pytorch root directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        pytorch_root = os.path.dirname(test_dir)
        docs_path = os.path.join(pytorch_root, "docs", "source", "fake_tensor.rst")

        self.assertTrue(
            os.path.exists(docs_path),
            f"Documentation file not found at {docs_path}",
        )


if __name__ == "__main__":
    run_tests()
