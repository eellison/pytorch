class GraphModule(torch.nn.Module):
    def forward(self, primals_1: f32[768], primals_2: f32[768], primals_3: f32[2304, 768], primals_4: f32[4, 512, 768]):
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:654, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
        permute: f32[768, 2304] = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        view: f32[2048, 768] = torch.ops.aten.view.default(primals_4, [2048, 768]);  primals_4 = None
        mm: f32[2048, 2304] = torch.ops.aten.mm.default(view, permute)
        _unsafe_view: f32[4, 512, 2304] = torch.ops.aten._unsafe_view.default(mm, [4, 512, 2304]);  mm = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:612, code: x = x.view(new_x_shape)
        view_1: f32[4, 512, 12, 192] = torch.ops.aten.view.default(_unsafe_view, [4, 512, 12, -1]);  _unsafe_view = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:613, code: return x.permute(0, 2, 1, 3)
        permute_1: f32[4, 12, 512, 192] = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:655, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
        split = torch.ops.aten.split.Tensor(permute_1, 64, -1);  permute_1 = None
        getitem: f32[4, 12, 512, 64] = split[0]
        getitem_1: f32[4, 12, 512, 64] = split[1]
        getitem_2: f32[4, 12, 512, 64] = split[2];  split = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:672, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
        unsqueeze: f32[1, 768] = torch.ops.aten.unsqueeze.default(primals_1, 0);  primals_1 = None
        unsqueeze_1: f32[1, 1, 768] = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        slice_1: f32[1, 1, 768] = torch.ops.aten.slice.Tensor(unsqueeze_1, 2, 0, 9223372036854775807);  unsqueeze_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:612, code: x = x.view(new_x_shape)
        view_2: f32[1, 1, 12, 64] = torch.ops.aten.view.default(slice_1, [1, 1, 12, -1]);  slice_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:613, code: return x.permute(0, 2, 1, 3)
        permute_2: f32[1, 12, 1, 64] = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:672, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
        add: f32[4, 12, 512, 64] = torch.ops.aten.add.Tensor(getitem, permute_2);  getitem = permute_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:673, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
        unsqueeze_2: f32[1, 768] = torch.ops.aten.unsqueeze.default(primals_2, 0);  primals_2 = None
        unsqueeze_3: f32[1, 1, 768] = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        slice_2: f32[1, 1, 768] = torch.ops.aten.slice.Tensor(unsqueeze_3, 2, 0, 9223372036854775807);  unsqueeze_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:612, code: x = x.view(new_x_shape)
        view_3: f32[1, 1, 12, 64] = torch.ops.aten.view.default(slice_2, [1, 1, 12, -1]);  slice_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:613, code: return x.permute(0, 2, 1, 3)
        permute_3: f32[1, 12, 1, 64] = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:673, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
        add_1: f32[4, 12, 512, 64] = torch.ops.aten.add.Tensor(getitem_2, permute_3);  getitem_2 = permute_3 = None
        
        # No stacktrace found for following nodes
        _tensor_constant0 = self._tensor_constant0
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:678, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        mul: f32[] = torch.ops.aten.mul.Tensor(lift_fresh_copy, 1);  lift_fresh_copy = None
        sqrt: f32[] = torch.ops.aten.sqrt.default(mul);  mul = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:679, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
        div: f32[4, 12, 512, 64] = torch.ops.aten.div.Tensor(add, sqrt);  add = sqrt = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:680, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_4: f32[4, 12, 64, 512] = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
        expand: f32[4, 12, 512, 64] = torch.ops.aten.expand.default(div, [4, 12, 512, 64]);  div = None
        clone: f32[4, 12, 512, 64] = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        _unsafe_view_1: f32[48, 512, 64] = torch.ops.aten._unsafe_view.default(clone, [48, 512, 64]);  clone = None
        expand_1: f32[4, 12, 64, 512] = torch.ops.aten.expand.default(permute_4, [4, 12, 64, 512]);  permute_4 = None
        clone_1: f32[4, 12, 64, 512] = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        _unsafe_view_2: f32[48, 64, 512] = torch.ops.aten._unsafe_view.default(clone_1, [48, 64, 512]);  clone_1 = None
        bmm: f32[48, 512, 512] = torch.ops.aten.bmm.default(_unsafe_view_1, _unsafe_view_2)
        _unsafe_view_3: f32[4, 12, 512, 512] = torch.ops.aten._unsafe_view.default(bmm, [4, 12, 512, 512]);  bmm = None
        permute_5: f32[48, 64, 512] = torch.ops.aten.permute.default(_unsafe_view_1, [0, 2, 1]);  _unsafe_view_1 = None
        permute_6: f32[48, 512, 64] = torch.ops.aten.permute.default(_unsafe_view_2, [0, 2, 1]);  _unsafe_view_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:654, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
        permute_13: f32[2304, 768] = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return [_unsafe_view_3, add_1, view, permute_5, permute_6, permute_13]
        