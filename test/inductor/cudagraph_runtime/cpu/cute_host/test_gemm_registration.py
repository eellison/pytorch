"""Register the existing three-Tensor GEMM without compiling or launching it."""

from pathlib import Path
import sys


REPO = next(path for path in Path(__file__).resolve().parents if (path / "torch/__init__.py").is_file())
FIXTURES = REPO / "test/inductor/cudagraph_runtime/cuda/cute"
sys.path.insert(0, str(FIXTURES))
from fixture_support import load_source

from torch._inductor.runtime._cudagraph import _sdk

_sdk.activate()

from torch._inductor.runtime._cudagraph._compiler.compiler_cute_handoff.invocation import register_cute_entry
from torch.testing._internal.common_utils import run_tests, TestCase


class TestGemmRegistration(TestCase):
    def test_three_tensor_owner(self):
        prototype = load_source("_three_tensor_registration", FIXTURES / "tensorop_gemm/host_prototype.py")
        owner = prototype.make_owner(
            REPO / "third_party/cutlass/examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py"
        )
        self.addCleanup(owner.close)
        self.assertIsNone(owner.selected)
        entry = register_cute_entry(owner, owned_executor=True)
        self.addCleanup(entry.close)
        self.assertEqual(entry.formals, ("a", "b", "c"))
        self.assertIs(entry.provider.owner, owner)
        entry.check()
        self.assertIsNone(owner.selected)
        self.assertEqual(owner._native_borrows, set())


if __name__ == "__main__":
    run_tests()
