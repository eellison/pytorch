class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: f32[4, 12, 512, 512], arg1_1: b8[4, 12, 512, 512]):
        # No stacktrace found for following nodes
        _tensor_constant0 = self._tensor_constant0
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:201, code: return input.masked_fill(mask, 0) * ctx.scale
        lift_fresh_copy: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        where: f32[4, 12, 512, 512] = torch.ops.aten.where.self(arg1_1, lift_fresh_copy, arg0_1);  arg1_1 = lift_fresh_copy = arg0_1 = None
        mul: f32[4, 12, 512, 512] = torch.ops.aten.mul.Tensor(where, 1.1111111111111112);  where = None
        return (mul,)
        