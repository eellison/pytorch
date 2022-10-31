class GraphModule(torch.nn.Module):
    def forward(self, primals_1: f32[768, 768], primals_2: f32[768], primals_3: f32[4, 512, 768]):
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:308, code: hidden_states = self.dense(hidden_states)
        permute: f32[768, 768] = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        view: f32[2048, 768] = torch.ops.aten.view.default(primals_3, [2048, 768]);  primals_3 = None
        addmm: f32[2048, 768] = torch.ops.aten.addmm.default(primals_2, view, permute);  primals_2 = None
        view_1: f32[4, 512, 768] = torch.ops.aten.view.default(addmm, [4, 512, 768]);  addmm = None
        permute_1: f32[768, 768] = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return [view_1, view, permute_1]
        