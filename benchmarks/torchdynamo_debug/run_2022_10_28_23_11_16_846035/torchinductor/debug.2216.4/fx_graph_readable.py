class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: f32[4, 512]):
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:435, code: extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        unsqueeze: f32[4, 1, 512] = torch.ops.aten.unsqueeze.default(arg0_1, 1);  arg0_1 = None
        unsqueeze_1: f32[4, 1, 1, 512] = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:436, code: attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        squeeze: f32[4, 1, 512] = torch.ops.aten.squeeze.dim(unsqueeze_1, -2)
        unsqueeze_2: f32[4, 1, 512, 1] = torch.ops.aten.unsqueeze.default(squeeze, -1);  squeeze = None
        mul: f32[4, 1, 512, 512] = torch.ops.aten.mul.Tensor(unsqueeze_1, unsqueeze_2);  unsqueeze_1 = unsqueeze_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:437, code: attention_mask = attention_mask.byte()
        _to_copy: u8[4, 1, 512, 512] = torch.ops.aten._to_copy.default(mul, dtype = torch.uint8);  mul = None
        return (_to_copy,)
        