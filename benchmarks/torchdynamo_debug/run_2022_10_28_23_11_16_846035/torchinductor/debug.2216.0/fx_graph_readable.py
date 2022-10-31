class <lambda>(torch.nn.Module):
    def forward(self):
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:987, code: attention_mask = torch.ones(input_shape, device=device)
        ones: f32[4, 512] = torch.ops.aten.ones.default([4, 512], device = device(type='cuda', index=0), pin_memory = False)
        alias: f32[4, 512] = torch.ops.aten.alias.default(ones);  ones = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:989, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        zeros: i64[4, 512] = torch.ops.aten.zeros.default([4, 512], dtype = torch.int64, device = device(type='cuda', index=0), pin_memory = False)
        alias_1: i64[4, 512] = torch.ops.aten.alias.default(zeros);  zeros = None
        return (alias_1, alias)
        