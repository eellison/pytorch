"""Fresh source modules for compiler ownership and configuration tests."""

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[5]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None or name in sys.modules:
        raise RuntimeError("The fixture requires a fresh source module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
