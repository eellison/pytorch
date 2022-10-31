class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: f32[4, 12, 512, 512], arg1_1: u8[4, 1, 512, 512]):
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:133, code: rmask = ~(mask.to(torch.bool))
        _to_copy: b8[4, 1, 512, 512] = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.bool);  arg1_1 = None
        bitwise_not: b8[4, 1, 512, 512] = torch.ops.aten.bitwise_not.default(_to_copy);  _to_copy = None
        
        # No stacktrace found for following nodes
        _tensor_constant0 = self._tensor_constant0
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:135, code: output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
        lift_fresh_copy: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        _to_copy_1: f32[] = torch.ops.aten._to_copy.default(lift_fresh_copy, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0));  lift_fresh_copy = None
        where: f32[4, 12, 512, 512] = torch.ops.aten.where.self(bitwise_not, _to_copy_1, arg0_1);  _to_copy_1 = arg0_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:136, code: output = torch.softmax(output, self.dim)
        amax: f32[4, 12, 512, 1] = torch.ops.aten.amax.default(where, [-1], True)
        sub: f32[4, 12, 512, 512] = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
        exp: f32[4, 12, 512, 512] = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: f32[4, 12, 512, 1] = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: f32[4, 12, 512, 512] = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        
        # No stacktrace found for following nodes
        _tensor_constant1 = self._tensor_constant1
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:137, code: output.masked_fill_(rmask, 0)
        lift_fresh_copy_1: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        where_1: f32[4, 12, 512, 512] = torch.ops.aten.where.self(bitwise_not, lift_fresh_copy_1, div);  bitwise_not = lift_fresh_copy_1 = None
        copy_: f32[4, 12, 512, 512] = torch.ops.aten.copy_.default(div, where_1);  div = where_1 = None
        return (copy_,)
        