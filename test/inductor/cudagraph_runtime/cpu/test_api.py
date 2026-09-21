"""The runtime facade preserves optional imports and canonical identities."""

from importlib import import_module
from importlib.util import find_spec
import subprocess
import sys


import_module("torch._inductor.runtime._cudagraph")

BEFORE_FACADE = set(sys.modules)
import torch._inductor.runtime._cudagraph.api as api

AFTER_FACADE = set(sys.modules)
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


TRITON_WITHOUT_CUTE = r"""
import sys

class NoCuTe:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cutlass" or fullname.startswith("cutlass."):
            raise ModuleNotFoundError("CuTe is unavailable in this import check")

sys.meta_path.insert(0, NoCuTe())
from torch._inductor.runtime._cudagraph.api import (
    DirectHost, DirectTriton, InputContract, IntegerRange, IntExpr,
    NativeTerminalPolicy, TensorInput, prepare_direct,
)
if any(name == "cutlass" or name.startswith("cutlass.") for name in sys.modules):
    raise AssertionError("Triton-only entrypoints imported CuTe")
print("TRITON_ONLY_IMPORT_OK")
"""


class TestExperimentalApi(TestCase):
    def test_import_does_not_activate_optional_frontends(self):
        for prefix in ("cutlass", "triton"):
            self.assertFalse(any(name == prefix or name.startswith(prefix + ".")
                                 for name in AFTER_FACADE - BEFORE_FACADE))
        result = subprocess.run([sys.executable, "-c", TRITON_WITHOUT_CUTE],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("TRITON_ONLY_IMPORT_OK", result.stdout)

    @parametrize("name,module", (
        ("DirectHost", "torch._inductor.runtime._cudagraph.direct_host"),
        ("prepare_direct", "torch._inductor.runtime._cudagraph.direct_host"),
        ("DirectCuTe", "torch._inductor.runtime._cudagraph.direct_cute"),
        ("DirectTriton", "torch._inductor.runtime._cudagraph.direct_triton"),
        ("ObservedOrdinaryEntry", "torch._inductor.runtime._cudagraph._compiler.ordinary_artifact_capture.owner"),
        ("PythonEntry", "torch._inductor.runtime._cudagraph._compiler.python_entry"),
        ("SignaturePolicy", "torch._inductor.runtime._cudagraph._compiler.entry_signature"),
        ("InputContract", "torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract"),
        ("TensorInput", "torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract"),
        ("IntegerRange", "torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract"),
        ("IntExpr", "torch._inductor.runtime.cudagraph_arg_mapping"),
        ("NativeTerminalPolicy", "torch._inductor.runtime._cudagraph.policy"),
    ))
    def test_export_is_the_existing_canonical_object(self, name, module):
        if name in ("ObservedOrdinaryEntry", "SignaturePolicy"):
            if find_spec("cutlass") is None:
                self.skipTest("CuTe SDK is not installed (cutlass package unavailable)")
            from torch._inductor.runtime._cudagraph import _sdk

            _sdk.activate()
        self.assertIn(name, api.__all__)
        self.assertIs(getattr(api, name), getattr(import_module(module), name))


instantiate_parametrized_tests(TestExperimentalApi)

if __name__ == "__main__":
    run_tests()
