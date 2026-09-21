# Owner(s): ["module: inductor"]
"""Run CPU contracts in isolated interpreters through the standard test runner."""

import os
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from cudagraph_runtime.runner import catalog, ROOT, run_script
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests, parametrize, run_tests, TestCase,
)


@instantiate_parametrized_tests
class TestCUDAGraphRuntimeCPU(TestCase):
    @parametrize("test", catalog("cpu"), name_fn=lambda test: test["id"])
    def test_script(self, test):
        run_script(test)


class TestCUDAGraphRuntimeLauncher(TestCase):
    def test_catalog_and_standard_discovery_are_complete(self):
        tests = (*catalog("cpu"), *catalog("cuda"))
        self.assertEqual(len({test["id"] for test in tests}), len(tests))
        scripts = {test["script"] for test in tests}
        self.assertEqual(len(scripts), len(tests))
        self.assertEqual(scripts, {
            str(path.relative_to(ROOT))
            for device in ("cpu", "cuda")
            for path in (ROOT / device).rglob("test_*.py")
        })
        discovery = Path(__file__).resolve().parents[2] / "tools/testing/discover_tests.py"
        discovered = runpy.run_path(str(discovery))["TESTS"]
        self.assertEqual(
            {name for name in discovered if name.startswith("inductor/test_cudagraph_runtime")},
            {"inductor/test_cudagraph_runtime_cpu", "inductor/test_cudagraph_runtime_cuda"},
        )
        self.assertFalse(any(name.startswith("inductor/cudagraph_runtime/") for name in discovered))

    def test_child_imports_and_environment_are_isolated(self):
        with TemporaryDirectory() as directory:
            for value in ("first", "second"):
                folder = Path(directory) / value
                folder.mkdir()
                (folder / "helper.py").write_text(f"VALUE = {value!r}\n")
                script = folder / "test_child.py"
                script.write_text(
                    "import helper, os\n"
                    "print(helper.VALUE)\n"
                    "print(repr(os.environ['CUDA_VISIBLE_DEVICES']))\n"
                    "print('PYTEST_CURRENT_TEST' in os.environ)\n"
                )
                with mock.patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "parent", "CUDA_VISIBLE_DEVICES": "7"}):
                    result = run_script({"script": str(script), "device": "cpu"})
                    self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")
                    self.assertEqual(os.environ["PYTEST_CURRENT_TEST"], "parent")
                self.assertEqual(result.stdout.splitlines(), [value, "''", "False"])

    def test_child_failure_preserves_stdout_and_stderr(self):
        with TemporaryDirectory() as directory:
            script = Path(directory) / "test_failure.py"
            script.write_text("import sys\nprint('output marker')\nprint('error marker', file=sys.stderr)\nsys.exit(7)\n")
            with self.assertRaisesRegex(AssertionError, "(?s)exited 7.*output marker.*error marker"):
                run_script({"script": str(script), "device": "cpu"})

    def test_missing_cute_only_skips_cute_scripts(self):
        with mock.patch("cudagraph_runtime.runner.find_spec", return_value=None), \
             mock.patch("cudagraph_runtime.runner.subprocess.run") as launch:
            with self.assertRaisesRegex(unittest.SkipTest, "CuTe SDK is not installed"):
                run_script({"script": "must_not_import.py", "device": "cpu", "requires_cute": True})
            launch.assert_not_called()
        with TemporaryDirectory() as directory:
            script = Path(directory) / "test_without_cute.py"
            script.write_text("print('no CuTe needed')\n")
            with mock.patch("cudagraph_runtime.runner.find_spec", return_value=None):
                result = run_script({"script": str(script), "device": "cpu"})
            self.assertEqual(result.stdout, "no CuTe needed\n")


if __name__ == "__main__":
    run_tests()
