import torch

import torch
from torch.testing._internal.jit_utils import JitTestCase, execWrapper
import operator


from torch.testing import FileCheck
from torch.testing._internal.common_utils import make_tensor
from torch.testing._internal.common_methods_invocations import sample_inputs_cat_concat
from torch import nn
from torch.testing._internal.common_utils import run_tests


from textwrap import dedent

# XXX: still in prototype
class TestZ3Demos(JitTestCase):
    def setUp(self):
        schema_mapping = {}
        for schema in torch._C._jit_get_all_schemas():
            if schema.name not in schema_mapping:
                schema_mapping[schema.name] = []
            schema_mapping[schema.name].append(schema)
        self.schema_mapping = schema_mapping

    def test_generate_conv_input_errors(self):
        # TODO: below isnt working for some reason
        # graph = None
        # for schema in self.schema_mapping["aten::conv2d"]:
        #     graph = torch._C._jit_shape_compute_graph_for_schema(schema)
        #     if graph is not None:
        #         break
        # assert graph is not None
        import pdb; pdb.set_trace()
        graph = grpah.copy()



if __name__ == '__main__':
    run_tests()



