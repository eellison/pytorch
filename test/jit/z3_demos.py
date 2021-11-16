import torch

import torch
from torch.testing._internal.jit_utils import JitTestCase, execWrapper
import operator


from torch.testing import FileCheck
from torch.testing._internal.common_utils import make_tensor
from torch.testing._internal.common_methods_invocations import sample_inputs_cat_concat
from torch import nn
from torch.testing._internal.common_utils import run_tests
from typing import List, Optional, Any
from torch.jit import generate_exception_inputs, generate_all_paths


from textwrap import dedent

from z3 import *


# XXX: still in prototype
class TestZ3Demos(JitTestCase):
    def setUp(self):
        schema_mapping = {}
        for schema in torch._C._jit_get_all_schemas():
            if schema.name not in schema_mapping:
                schema_mapping[schema.name] = []
            schema_mapping[schema.name].append(schema)
        self.schema_mapping = schema_mapping

    def test_generate_max_pool2d_input_errors(self):
        # TODO: below isnt working for some reason
        # graph = None
        # for schema in self.schema_mapping["aten::conv2d"]:
        #     graph = torch._C._jit_shape_compute_graph_for_schema(schema)
        #     if graph is not None:
        #         break
        # assert graph is not None
        @torch.jit.script
        def foo(input, kernel: List[int], stride: List[int], padding: List[int], dilation: List[int], ceil_mode: bool):
            return torch.max_pool2d(input, kernel, stride, padding, dilation, ceil_mode)
        shape_compute_graph = torch._C._jit_get_shape_compute_graph_for_node(foo.graph.findNode("aten::max_pool2d"))
        torch._C._augment_with_length(shape_compute_graph, 0, 4)
        for i in range(1, 5):
            torch._C._augment_with_length(shape_compute_graph, i, 2)
        changed = True
        while changed:
            changed = torch._C._jit_pass_shape_graph_cleanup_passes(shape_compute_graph)

        out = generate_exception_inputs(shape_compute_graph, foo.graph.findNode("aten::max_pool2d"))
        foo(*out[0])
        # print(shape_compute_graph)

    def test_generate_max_pool2d_input_successful(self):
        @torch.jit.script
        def foo(input, kernel: List[int], stride: List[int], padding: List[int], dilation: List[int], ceil_mode: bool):
            return torch.max_pool2d(input, kernel, stride, padding, dilation, ceil_mode)
        shape_compute_graph = torch._C._jit_get_shape_compute_graph_for_node(foo.graph.findNode("aten::max_pool2d"))
        # print(shape_compute_graph)

        torch._C._augment_with_length(shape_compute_graph, 0, 4)
        for i in range(1, 5):
            torch._C._augment_with_length(shape_compute_graph, i, 2)
        changed = True
        while changed:
            changed = torch._C._jit_pass_shape_graph_cleanup_passes(shape_compute_graph)

        out = generate_all_paths(shape_compute_graph, foo.graph.findNode("aten::max_pool2d"))
        for inp in out:
            print(foo(*inp))

    def test_generate_avg_pool2d_input_successful(self):
        @torch.jit.script
        def foo(input, kernel: List[int], stride: List[int], padding: List[int], ceil_mode: bool):
            return torch.nn.functional.avg_pool2d(input, kernel, stride, padding, ceil_mode)
        torch._C._jit_pass_inline(foo.graph)
        shape_compute_graph = torch._C._jit_get_shape_compute_graph_for_node(foo.graph.findNode("aten::max_pool2d"))
        # print(shape_compute_graph)

        torch._C._augment_with_length(shape_compute_graph, 0, 4)
        for i in range(1, 5):
            torch._C._augment_with_length(shape_compute_graph, i, 2)
        changed = True
        while changed:
            changed = torch._C._jit_pass_shape_graph_cleanup_passes(shape_compute_graph)

        out = generate_all_paths(shape_compute_graph, foo.graph.findNode("aten::max_pool2d"))
        for inp in out:
            print(foo(*inp))


if __name__ == '__main__':
    run_tests()
