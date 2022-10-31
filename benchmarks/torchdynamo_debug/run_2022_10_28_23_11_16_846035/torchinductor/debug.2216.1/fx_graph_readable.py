class GraphModule(torch.nn.Module):
    def forward(self, primals_1: f32[768], primals_2: f32[768], primals_3: f32[50265, 768], primals_4: f32[512, 768], primals_5: i64[1, 512], primals_6: i64[4, 512], primals_7: f32[4, 512]):
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:794, code: position_ids = self.position_ids[:, :seq_length]
        slice_1: i64[1, 512] = torch.ops.aten.slice.Tensor(primals_5, 0, 0, 9223372036854775807);  primals_5 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:800, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: f32[4, 512, 768] = torch.ops.aten.embedding.default(primals_3, primals_6, 0);  primals_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:803, code: position_embeddings = self.position_embeddings(position_ids.long())
        embedding_1: f32[1, 512, 768] = torch.ops.aten.embedding.default(primals_4, slice_1);  primals_4 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:809, code: embeddings += position_embeddings
        add: f32[4, 512, 768] = torch.ops.aten.add.Tensor(embedding, embedding_1)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:292, code: mean = hidden_states.mean(-1, keepdim=True)
        mean: f32[4, 512, 1] = torch.ops.aten.mean.dim(add, [-1], True)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:293, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        sub: f32[4, 512, 768] = torch.ops.aten.sub.Tensor(add, mean);  add = None
        pow_1: f32[4, 512, 768] = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
        mean_1: f32[4, 512, 1] = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
        add_1: f32[4, 512, 1] = torch.ops.aten.add.Tensor(mean_1, 1e-07);  mean_1 = None
        sqrt: f32[4, 512, 1] = torch.ops.aten.sqrt.default(add_1);  add_1 = None
        div: f32[4, 512, 768] = torch.ops.aten.div.Tensor(sub, sqrt);  sub = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: y = self.weight * hidden_states + self.bias
        mul: f32[4, 512, 768] = torch.ops.aten.mul.Tensor(primals_1, div);  div = None
        add_2: f32[4, 512, 768] = torch.ops.aten.add.Tensor(mul, primals_2);  mul = primals_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:823, code: mask = mask.unsqueeze(2)
        unsqueeze: f32[4, 512, 1] = torch.ops.aten.unsqueeze.default(primals_7, 2)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:826, code: embeddings = embeddings * mask
        mul_1: f32[4, 512, 768] = torch.ops.aten.mul.Tensor(add_2, unsqueeze);  add_2 = unsqueeze = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:803, code: position_embeddings = self.position_embeddings(position_ids.long())
        view_3: i64[512] = torch.ops.aten.view.default(slice_1, [512]);  slice_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:800, code: inputs_embeds = self.word_embeddings(input_ids)
        view_5: i64[2048] = torch.ops.aten.view.default(primals_6, [2048]);  primals_6 = None
        return [mul_1, primals_1, primals_7, embedding, embedding_1, mean, sqrt, view_3, view_5]
        