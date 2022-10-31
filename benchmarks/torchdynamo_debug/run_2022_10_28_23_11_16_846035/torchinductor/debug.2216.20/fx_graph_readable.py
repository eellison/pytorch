class GraphModule(torch.nn.Module):
    def forward(self, primals_1: f32[4, 12, 512, 512], primals_2: f32[4, 12, 512, 64]):
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:697, code: context_layer = torch.matmul(attention_probs, value_layer)
        expand: f32[4, 12, 512, 512] = torch.ops.aten.expand.default(primals_1, [4, 12, 512, 512]);  primals_1 = None
        view: f32[48, 512, 512] = torch.ops.aten.view.default(expand, [48, 512, 512]);  expand = None
        expand_1: f32[4, 12, 512, 64] = torch.ops.aten.expand.default(primals_2, [4, 12, 512, 64]);  primals_2 = None
        clone: f32[4, 12, 512, 64] = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        _unsafe_view: f32[48, 512, 64] = torch.ops.aten._unsafe_view.default(clone, [48, 512, 64]);  clone = None
        bmm: f32[48, 512, 64] = torch.ops.aten.bmm.default(view, _unsafe_view)
        _unsafe_view_1: f32[4, 12, 512, 64] = torch.ops.aten._unsafe_view.default(bmm, [4, 12, 512, 64]);  bmm = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:698, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute: f32[4, 512, 12, 64] = torch.ops.aten.permute.default(_unsafe_view_1, [0, 2, 1, 3]);  _unsafe_view_1 = None
        clone_1: f32[4, 512, 12, 64] = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:700, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1: f32[4, 512, 768] = torch.ops.aten.view.default(clone_1, [4, 512, -1]);  clone_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:697, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_2: f32[48, 512, 512] = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        permute_3: f32[48, 64, 512] = torch.ops.aten.permute.default(_unsafe_view, [0, 2, 1]);  _unsafe_view = None
        return [view_1, permute_2, permute_3]
        