# Owner(s): ["module: inductor"]

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from torch._inductor.runtime._cudagraph._sdk import activate


activate()

from torch._vendor.quack.cache import async_compile, jit
from torch.cuda import _host_trace_cute_dsl as hook
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class _Compiled:
    def export_to_c(self, *, object_file_path, function_name):
        Path(object_file_path).write_bytes(b"compiled object")


@instantiate_parametrized_tests
class TestCuTeCachePair(TestCase):
    def setUp(self):
        super().setUp()
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.compilations = 0
        self.loaded = object()

        def compile_kernel(rows):
            self.compilations += 1
            return _Compiled()

        self.compile = jit.jit_cache(compile_kernel)
        enter = self.enterContext
        enter(patch.object(jit, "get_cache_path", return_value=self.root))
        enter(patch.object(jit, "_compute_source_fingerprint", return_value="source"))
        enter(patch.object(jit._state, "CACHE_ENABLED", True))
        enter(patch.object(async_compile, "get_active_pool", return_value=None))
        enter(patch.object(hook, "install"))
        enter(patch.object(hook, "descriptors_available", return_value=True))
        enter(patch.object(hook, "descriptor_json", return_value="typed payload"))
        module = {"func": self.loaded}
        self.load = enter(
            patch.object(jit.cute.runtime, "load_module", return_value=module)
        )
        self.bind = enter(
            patch.object(hook, "loaded_program", return_value=self.loaded)
        )
        self.compile(4)
        (self.object_path,) = (self.root / "source").glob("*.o")
        self.descriptor_path = self.object_path.with_suffix(jit.DESCRIPTOR_SUFFIX)
        self.compile.cache_clear()

    def test_matching_object_loads_existing_payload(self):
        self.assertIs(self.compile(4), self.loaded)
        self.assertEqual(self.compilations, 1)
        self.load.assert_called_once()
        self.assertEqual(self.bind.call_args.args[2], "typed payload")
        envelope = json.loads(self.descriptor_path.read_text())
        digest = hashlib.sha256(self.object_path.read_bytes()).hexdigest()
        self.assertEqual(envelope["object_sha256"], digest)

    @parametrize("damage", ("object", "hash", "missing", "legacy", "malformed"))
    def test_unpaired_payload_recompiles_before_loading(self, damage):
        if damage == "object":
            self.object_path.write_bytes(b"another compiled object")
        elif damage == "hash":
            envelope = json.loads(self.descriptor_path.read_text())
            envelope["object_sha256"] = "0" * 64
            self.descriptor_path.write_text(json.dumps(envelope))
        elif damage == "missing":
            self.descriptor_path.unlink()
        elif damage == "legacy":
            self.descriptor_path.write_text('{"function_name": "old descriptor"}')
        else:
            self.descriptor_path.write_text("{")
        self.assertIsInstance(self.compile(4), _Compiled)
        self.assertEqual(self.compilations, 2)
        self.load.assert_not_called()
        self.bind.assert_not_called()
        envelope = json.loads(self.descriptor_path.read_text())
        digest = hashlib.sha256(self.object_path.read_bytes()).hexdigest()
        self.assertEqual(envelope["object_sha256"], digest)
        self.compile.cache_clear()
        self.assertIs(self.compile(4), self.loaded)
        self.assertEqual(self.compilations, 2)


if __name__ == "__main__":
    run_tests()
