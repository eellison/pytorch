"""Run the CUDA component tests serially in separate processes."""

import unittest

from runner import catalog, run_script


if __name__ == "__main__":
    for test in catalog("cuda"):
        print(f"Running {test['id']}: {test['script']}", flush=True)
        try:
            run_script(test)
        except unittest.SkipTest as error:
            print(f"Skipped {test['id']}: {error}", flush=True)
