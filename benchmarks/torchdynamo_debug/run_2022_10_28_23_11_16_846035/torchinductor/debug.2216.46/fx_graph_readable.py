class GraphModule(torch.nn.Module):
    def forward(self, primals_1: f32[3072, 768], primals_2: f32[3072], primals_3: f32[4, 512, 768]):
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
        permute: f32[768, 3072] = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        view: f32[2048, 768] = torch.ops.aten.view.default(primals_3, [2048, 768]);  primals_3 = None
        addmm: f32[2048, 3072] = torch.ops.aten.addmm.default(primals_2, view, permute);  primals_2 = None
        view_1: f32[4, 512, 3072] = torch.ops.aten.view.default(addmm, [4, 512, 3072]);  addmm = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(view_1, 0.5)
        mul_1: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(view_1, 0.7071067811865476)
        sign: f32[4, 512, 3072] = torch.ops.aten.sign.default(mul_1)
        abs_1: f32[4, 512, 3072] = torch.ops.aten.abs.default(mul_1);  mul_1 = None
        mul_2: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(abs_1, 0.3275911)
        add: f32[4, 512, 3072] = torch.ops.aten.add.Tensor(mul_2, 1.0);  mul_2 = None
        reciprocal: f32[4, 512, 3072] = torch.ops.aten.reciprocal.default(add);  add = None
        mul_3: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(reciprocal, 1.0);  reciprocal = None
        mul_4: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(mul_3, 1.061405429)
        add_1: f32[4, 512, 3072] = torch.ops.aten.add.Tensor(mul_4, -1.453152027);  mul_4 = None
        mul_5: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(add_1, mul_3);  add_1 = None
        add_2: f32[4, 512, 3072] = torch.ops.aten.add.Tensor(mul_5, 1.421413741);  mul_5 = None
        mul_6: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(add_2, mul_3);  add_2 = None
        add_3: f32[4, 512, 3072] = torch.ops.aten.add.Tensor(mul_6, -0.284496736);  mul_6 = None
        mul_7: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(add_3, mul_3);  add_3 = None
        add_4: f32[4, 512, 3072] = torch.ops.aten.add.Tensor(mul_7, 0.254829592);  mul_7 = None
        mul_8: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(add_4, mul_3);  add_4 = mul_3 = None
        neg: f32[4, 512, 3072] = torch.ops.aten.neg.default(abs_1)
        mul_9: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(neg, abs_1);  neg = abs_1 = None
        exp: f32[4, 512, 3072] = torch.ops.aten.exp.default(mul_9);  mul_9 = None
        mul_10: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(mul_8, exp);  mul_8 = exp = None
        
        # No stacktrace found for following nodes
        _tensor_constant0 = self._tensor_constant0
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        sub: f32[4, 512, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy, mul_10);  lift_fresh_copy = None
        mul_11: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(sign, sub);  sub = None
        add_5: f32[4, 512, 3072] = torch.ops.aten.add.Tensor(mul_11, 1);  mul_11 = None
        mul_12: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(mul, add_5);  mul = add_5 = None
        
        # No stacktrace found for following nodes
        _tensor_constant1 = self._tensor_constant1
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_1: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        sub_1: f32[4, 512, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_1, mul_10);  lift_fresh_copy_1 = mul_10 = None
        mul_23: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(sign, sub_1);  sign = sub_1 = None
        add_11: f32[4, 512, 3072] = torch.ops.aten.add.Tensor(mul_23, 1);  mul_23 = None
        mul_24: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(add_11, 0.5);  add_11 = None
        mul_25: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(view_1, view_1)
        mul_26: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(mul_25, -0.5);  mul_25 = None
        exp_2: f32[4, 512, 3072] = torch.ops.aten.exp.default(mul_26);  mul_26 = None
        mul_27: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_28: f32[4, 512, 3072] = torch.ops.aten.mul.Tensor(view_1, mul_27);  view_1 = mul_27 = None
        add_12: f32[4, 512, 3072] = torch.ops.aten.add.Tensor(mul_24, mul_28);  mul_24 = mul_28 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
        permute_1: f32[3072, 768] = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return [mul_12, view, add_12, permute_1]
        