# Owner(s): ["module: inductor"]
"""Run CUDA contracts serially, with each script owning a fresh interpreter."""

import unittest

from cudagraph_runtime.runner import catalog, run_script
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests, parametrize, run_tests, TestCase, TEST_CUDA,
)


@unittest.skipIf(not TEST_CUDA, "CUDA is required")
@instantiate_parametrized_tests
class TestCUDAGraphRuntimeCUDA(TestCase):
    @parametrize("test", catalog("cuda"), name_fn=lambda test: test["id"])
    def test_script(self, test):
        run_script(test)


if __name__ == "__main__":
    run_tests()
