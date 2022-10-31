class GraphModule(torch.nn.Module):
    def forward(self, primals_1: f32[768], primals_2: f32[768], primals_3: f32[4, 512, 768], primals_4: f32[4, 512, 768]):
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:377, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add: f32[4, 512, 768] = torch.ops.aten.add.Tensor(primals_3, primals_4);  primals_3 = primals_4 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:292, code: mean = hidden_states.mean(-1, keepdim=True)
        mean: f32[4, 512, 1] = torch.ops.aten.mean.dim(add, [-1], True)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:293, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        sub: f32[4, 512, 768] = torch.ops.aten.sub.Tensor(add, mean);  add = mean = None
        pow_1: f32[4, 512, 768] = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
        mean_1: f32[4, 512, 1] = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
        add_1: f32[4, 512, 1] = torch.ops.aten.add.Tensor(mean_1, 1e-07);  mean_1 = None
        sqrt: f32[4, 512, 1] = torch.ops.aten.sqrt.default(add_1);  add_1 = None
        div: f32[4, 512, 768] = torch.ops.aten.div.Tensor(sub, sqrt)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: y = self.weight * hidden_states + self.bias
        mul: f32[4, 512, 768] = torch.ops.aten.mul.Tensor(primals_1, div);  div = None
        add_2: f32[4, 512, 768] = torch.ops.aten.add.Tensor(mul, primals_2);  mul = primals_2 = None
        return [add_2, primals_1, sub, sqrt]
        