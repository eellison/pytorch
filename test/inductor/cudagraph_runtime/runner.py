"""Run catalog scripts in fresh interpreters without importing their fixtures."""

from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parent


def catalog(device):
    return tuple(
        test for test in json.loads((ROOT / "suite.json").read_text())
        if test["device"] == device
    )


def run_script(test, *, log_directory=None):
    if test.get("requires_cute", False) and find_spec("cutlass") is None:
        raise unittest.SkipTest("CuTe SDK is not installed (cutlass package unavailable)")
    path = ROOT / test["script"]
    environment = os.environ.copy()
    environment.pop("PYTEST_CURRENT_TEST", None)
    if test["device"] == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    try:
        result = subprocess.run(
            [sys.executable, str(path), "-v"], env=environment,
            capture_output=True, text=True,
            timeout=180 if test["device"] == "cpu" else None,
        )
    except subprocess.TimeoutExpired as error:
        raise AssertionError(
            f"{path} timed out\nstdout:\n{error.stdout}\nstderr:\n{error.stderr}"
        ) from error
    if log_directory is not None:
        (log_directory / "stdout.log").write_text(result.stdout)
        (log_directory / "stderr.log").write_text(result.stderr)
    if result.returncode:
        raise AssertionError(
            f"{path} exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    print(result.stdout, end="", flush=True)
    print(result.stderr, end="", file=sys.stderr, flush=True)
    return result
