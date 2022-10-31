class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: f32[4, 512, 768]):
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:183, code: mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).to(torch.bool)
        empty_like: f32[4, 512, 768] = torch.ops.aten.empty_like.default(arg0_1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  arg0_1 = None
        alias: f32[4, 512, 768] = torch.ops.aten.alias.default(empty_like);  empty_like = None
        rand_like: f32[4, 512, 768] = torch.ops.aten.rand_like.default(alias, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_1: f32[4, 512, 768] = torch.ops.aten.alias.default(rand_like);  rand_like = None
        lt: b8[4, 512, 768] = torch.ops.aten.lt.Scalar(alias_1, 0.9);  alias_1 = None
        copy_: f32[4, 512, 768] = torch.ops.aten.copy_.default(alias, lt);  alias = lt = None
        
        # No stacktrace found for following nodes
        _tensor_constant0 = self._tensor_constant0
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:183, code: mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).to(torch.bool)
        lift_fresh_copy: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        sub: f32[4, 512, 768] = torch.ops.aten.sub.Tensor(lift_fresh_copy, copy_);  lift_fresh_copy = copy_ = None
        _to_copy: b8[4, 512, 768] = torch.ops.aten._to_copy.default(sub, dtype = torch.bool);  sub = None
        return (_to_copy,)
        