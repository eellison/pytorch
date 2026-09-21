import json
from pathlib import Path
import subprocess
import sys
import tempfile

from torch._inductor.runtime._cudagraph._sdk import activate, raw_values
activate()
import cutlass.cute as cute
from cutlass._mlir import ir
from torch.testing._internal.common_utils import TestCase, instantiate_parametrized_tests, parametrize, run_tests


OWNER_SOURCE = '''
from cutlass import cute
from cutlass._mlir import ir
from torch._inductor.runtime._cudagraph.api import (
    DirectCuTe, ObservedOrdinaryEntry, PythonEntry, SignaturePolicy,
)
from torch._inductor.runtime._cudagraph._compiler.cute_dispatch.entry import TensorPolicy
from torch._inductor.runtime._cudagraph._compiler.compiler_cute_handoff.invocation import _CuTeProvider

@cute.kernel
def kernel(source: cute.Tensor, destination: cute.Tensor):
    pass

@cute.jit
def host(source: cute.Tensor, destination: cute.Tensor, stream):
    pass

def make_owner():
    return ObservedOrdinaryEntry(
        PythonEntry(host), kernel, policy=SignaturePolicy(32, 64, 16, "stream"),
        tensor_policies={name: TensorPolicy((None, 128), (0, 1))
                         for name in ("source", "destination")},
    )
'''


@instantiate_parametrized_tests
class TestSDKImport(TestCase):
    def run_child(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sdk_import_case.py"
            path.write_text("import json\n" + source)
            result = subprocess.run([sys.executable, str(path)], capture_output=True,
                                    text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout.splitlines()[-1])

    @parametrize("runtime_first", (False, True))
    def test_activation_before_sdk_import(self, runtime_first):
        source = "from torch._inductor.runtime._cudagraph import _sdk\n"
        if runtime_first:
            source += "from torch._inductor.runtime._cudagraph.api import DirectCuTe, DirectTriton\n"
        source += "_sdk.activate()\n" + OWNER_SOURCE + '''
owner = make_owner()
try:
    DirectCuTe(owner).check()
    _CuTeProvider(owner, owned_executor=True).check()
    _sdk.activate()
    _sdk.require_active()
    print(json.dumps({"active": ir.raw_values is _sdk.raw_values}))
finally:
    owner.close()
'''
        self.assertEqual(self.run_child(source), {"active": True})

    @parametrize("boundary", ("owner", "activate"))
    def test_sdk_first_import_is_rejected(self, boundary):
        source = OWNER_SOURCE + '''
from torch._inductor.runtime._cudagraph import _sdk
caster = ir.register_value_caster
try:
    ACTION
except RuntimeError as error:
    print(json.dumps({"error": str(error), "unchanged": ir.register_value_caster is caster,
                      "active": hasattr(ir, "raw_values")}))
else:
    raise AssertionError("Unactivated CuTe preparation was accepted")
'''.replace("ACTION", "make_owner()" if boundary == "owner" else "_sdk.activate()")
        result = self.run_child(source)
        self.assertIn("_sdk.activate()", result["error"])
        self.assertIn("before importing cutlass", result["error"])
        self.assertIn("restart the process", result["error"])
        self.assertTrue(result["unchanged"])
        self.assertFalse(result["active"])

    def test_unactivated_check_preserves_triton_independence(self):
        result = self.run_child('''
import sys
class NoCuTe:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cutlass" or fullname.startswith("cutlass."):
            raise ModuleNotFoundError("CuTe is unavailable")
sys.meta_path.insert(0, NoCuTe())
from torch._inductor.runtime._cudagraph import _sdk
from torch._inductor.runtime._cudagraph.api import DirectHost, DirectTriton, NativeTerminalPolicy
before = tuple(sys.meta_path)
try:
    _sdk.require_active()
except RuntimeError as error:
    print(json.dumps({"error": str(error), "unchanged": tuple(sys.meta_path) == before,
                      "imported_sdk": any(name == "cutlass" or name.startswith("cutlass.")
                                          for name in sys.modules)}))
else:
    raise AssertionError("Unactivated SDK check succeeded")
''')
        self.assertIn("_sdk.activate()", result["error"])
        self.assertTrue(result["unchanged"])
        self.assertFalse(result["imported_sdk"])

    def test_unsupported_sdk_version_does_not_install_hook(self):
        result = self.run_child('''
import sys
from unittest.mock import patch
from torch._inductor.runtime._cudagraph import _sdk
before = tuple(sys.meta_path)
with patch.object(_sdk.importlib.metadata, "version", return_value="4.6.3"):
    try:
        _sdk.activate()
    except RuntimeError as error:
        print(json.dumps({"error": str(error), "unchanged": tuple(sys.meta_path) == before,
                          "imported_sdk": "cutlass" in sys.modules}))
    else:
        raise AssertionError("An unsupported SDK version was accepted")
''')
        self.assertIn("requires nvidia-cutlass-dsl 4.6.2", result["error"])
        self.assertTrue(result["unchanged"])
        self.assertFalse(result["imported_sdk"])

    @parametrize("kind,spelling,wrapper", [
        ("tensor", '!cute.memref<f32, gmem, align<16>, "(?,128):(128,1)">', cute.Tensor),
        ("pointer", "!cute.ptr<f32, gmem, align<16>>", cute.Pointer),
    ])
    def test_raw_scope_preserves_typed_caster(self, kind, spelling, wrapper):
        activate()
        self.assertIs(ir.raw_values, raw_values)
        with ir.Context(), ir.Location.unknown():
            module = ir.Module.parse(f"module {{ func.func @test(%arg0: {spelling}) {{ return }} }}")
            function, = module.body.operations
            block = function.regions[0].blocks[0]
            before = str(module)
            with raw_values():
                value = block.arguments[0]
                self.assertIsInstance(value, ir.Value)
                self.assertEqual(str(value.type), spelling)
                with raw_values():
                    self.assertIsInstance(block.arguments[0], ir.Value)
                self.assertIsInstance(block.arguments[0], ir.Value)
            self.assertEqual(str(module), before)
            with self.assertRaisesRegex(RuntimeError, "scope exit"):
                with raw_values():
                    raise RuntimeError("scope exit")
            with ir.InsertionPoint(tuple(block.operations)[-1]):
                converted = block.arguments[0]
            self.assertIsInstance(converted, wrapper)
            with raw_values():
                self.assertIsInstance(block.arguments[0], ir.Value)
            self.assertTrue(module.operation.verify())
            print("SDK_CASTER_RESULT " + json.dumps({"kind":kind,"raw_value":type(value).__name__,"normal_value":type(converted).__name__,"nested_and_exception_restore":True}), flush=True)

if __name__ == "__main__":
    run_tests()
