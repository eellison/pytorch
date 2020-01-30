import unittest
import os
import sys
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.testing import FileCheck
from collections import OrderedDict

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, _tmp_donotuse_dont_inline_everything

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestFunctionalBlocks(JitTestCase):
    def test_simple_no_merge(self):
        # o: autodiff supported. x: not autodiff supported.
        # o --> x
        def fn(x, y, z):
            x = x + 1
            y = y + 1
            z = z + 1
            z.add_(2)
            z = z * z
            y = y * z
            return x + y + z
            return

        graph = torch.jit.script(fn).graph
        self.run_pass('create_functional_blocks', graph)
        print(graph)
