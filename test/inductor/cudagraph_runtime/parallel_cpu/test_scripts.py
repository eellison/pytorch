"""Distribute common_utils CPU scripts across isolated xdist jobs."""

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from cudagraph_runtime.runner import catalog, run_script


CPU_SCRIPTS = catalog("cpu")


@pytest.mark.parametrize("test", CPU_SCRIPTS, ids=[test["id"] for test in CPU_SCRIPTS])
def test_cpu_script(test, tmp_path):
    run_script(test, log_directory=tmp_path)
