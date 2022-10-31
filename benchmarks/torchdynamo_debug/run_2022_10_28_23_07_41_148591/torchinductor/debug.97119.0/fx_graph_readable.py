class GraphModule(torch.nn.Module):
    def forward(self, primals_1: f32[512], primals_2: f32[512], primals_3: f32[512], primals_4: f32[512], primals_5: f32[512], primals_6: f32[512], primals_7: f32[512], primals_8: f32[512], primals_9: f32[512], primals_10: f32[512], primals_11: f32[512], primals_12: f32[512], primals_13: f32[512], primals_14: f32[512], primals_15: f32[512], primals_16: f32[512], primals_17: f32[512], primals_18: f32[512], primals_19: f32[512], primals_20: f32[512], primals_21: f32[512], primals_22: f32[512], primals_23: f32[512], primals_24: f32[512], primals_25: f32[512], primals_26: f32[512], primals_27: f32[512], primals_28: f32[512], primals_29: f32[512], primals_30: f32[512], primals_31: f32[512], primals_32: f32[512], primals_33: f32[512], primals_34: f32[512], primals_35: f32[512], primals_36: f32[512], primals_37: f32[512], primals_38: f32[512], primals_39: f32[512], primals_40: f32[512], primals_41: f32[512], primals_42: f32[512], primals_43: f32[250112, 512], primals_44: f32[384, 512], primals_45: f32[384, 512], primals_46: f32[384, 512], primals_47: f32[32, 6], primals_48: f32[512, 384], primals_49: f32[1024, 512], primals_50: f32[1024, 512], primals_51: f32[512, 1024], primals_52: f32[384, 512], primals_53: f32[384, 512], primals_54: f32[384, 512], primals_55: f32[512, 384], primals_56: f32[1024, 512], primals_57: f32[1024, 512], primals_58: f32[512, 1024], primals_59: f32[384, 512], primals_60: f32[384, 512], primals_61: f32[384, 512], primals_62: f32[512, 384], primals_63: f32[1024, 512], primals_64: f32[1024, 512], primals_65: f32[512, 1024], primals_66: f32[384, 512], primals_67: f32[384, 512], primals_68: f32[384, 512], primals_69: f32[512, 384], primals_70: f32[1024, 512], primals_71: f32[1024, 512], primals_72: f32[512, 1024], primals_73: f32[384, 512], primals_74: f32[384, 512], primals_75: f32[384, 512], primals_76: f32[512, 384], primals_77: f32[1024, 512], primals_78: f32[1024, 512], primals_79: f32[512, 1024], primals_80: f32[384, 512], primals_81: f32[384, 512], primals_82: f32[384, 512], primals_83: f32[512, 384], primals_84: f32[1024, 512], primals_85: f32[1024, 512], primals_86: f32[512, 1024], primals_87: f32[384, 512], primals_88: f32[384, 512], primals_89: f32[384, 512], primals_90: f32[512, 384], primals_91: f32[1024, 512], primals_92: f32[1024, 512], primals_93: f32[512, 1024], primals_94: f32[384, 512], primals_95: f32[384, 512], primals_96: f32[384, 512], primals_97: f32[512, 384], primals_98: f32[1024, 512], primals_99: f32[1024, 512], primals_100: f32[512, 1024], primals_101: f32[384, 512], primals_102: f32[384, 512], primals_103: f32[384, 512], primals_104: f32[32, 6], primals_105: f32[512, 384], primals_106: f32[384, 512], primals_107: f32[384, 512], primals_108: f32[384, 512], primals_109: f32[512, 384], primals_110: f32[1024, 512], primals_111: f32[1024, 512], primals_112: f32[512, 1024], primals_113: f32[384, 512], primals_114: f32[384, 512], primals_115: f32[384, 512], primals_116: f32[512, 384], primals_117: f32[384, 512], primals_118: f32[384, 512], primals_119: f32[384, 512], primals_120: f32[512, 384], primals_121: f32[1024, 512], primals_122: f32[1024, 512], primals_123: f32[512, 1024], primals_124: f32[384, 512], primals_125: f32[384, 512], primals_126: f32[384, 512], primals_127: f32[512, 384], primals_128: f32[384, 512], primals_129: f32[384, 512], primals_130: f32[384, 512], primals_131: f32[512, 384], primals_132: f32[1024, 512], primals_133: f32[1024, 512], primals_134: f32[512, 1024], primals_135: f32[384, 512], primals_136: f32[384, 512], primals_137: f32[384, 512], primals_138: f32[512, 384], primals_139: f32[384, 512], primals_140: f32[384, 512], primals_141: f32[384, 512], primals_142: f32[512, 384], primals_143: f32[1024, 512], primals_144: f32[1024, 512], primals_145: f32[512, 1024], primals_146: f32[384, 512], primals_147: f32[384, 512], primals_148: f32[384, 512], primals_149: f32[512, 384], primals_150: f32[384, 512], primals_151: f32[384, 512], primals_152: f32[384, 512], primals_153: f32[512, 384], primals_154: f32[1024, 512], primals_155: f32[1024, 512], primals_156: f32[512, 1024], primals_157: f32[384, 512], primals_158: f32[384, 512], primals_159: f32[384, 512], primals_160: f32[512, 384], primals_161: f32[384, 512], primals_162: f32[384, 512], primals_163: f32[384, 512], primals_164: f32[512, 384], primals_165: f32[1024, 512], primals_166: f32[1024, 512], primals_167: f32[512, 1024], primals_168: f32[384, 512], primals_169: f32[384, 512], primals_170: f32[384, 512], primals_171: f32[512, 384], primals_172: f32[384, 512], primals_173: f32[384, 512], primals_174: f32[384, 512], primals_175: f32[512, 384], primals_176: f32[1024, 512], primals_177: f32[1024, 512], primals_178: f32[512, 1024], primals_179: f32[384, 512], primals_180: f32[384, 512], primals_181: f32[384, 512], primals_182: f32[512, 384], primals_183: f32[384, 512], primals_184: f32[384, 512], primals_185: f32[384, 512], primals_186: f32[512, 384], primals_187: f32[1024, 512], primals_188: f32[1024, 512], primals_189: f32[512, 1024], primals_190: f32[250112, 512], primals_191: i64[2, 128], primals_192: i64[2, 128], primals_193: i64[2, 128]):
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:932, code: input_ids = input_ids.view(-1, input_shape[-1])
        view: i64[2, 128] = torch.ops.aten.view.default(primals_191, [-1, 128]);  primals_191 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: f32[2, 128, 512] = torch.ops.aten.embedding.default(primals_43, view)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:952, code: attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        ones: f32[2, 128] = torch.ops.aten.ones.default([2, 128], device = device(type='cuda', index=0), pin_memory = False)
        alias: f32[2, 128] = torch.ops.aten.alias.default(ones);  ones = None
        alias_1: f32[2, 128] = torch.ops.aten.alias.default(alias);  alias = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/modeling_utils.py:779, code: extended_attention_mask = attention_mask[:, None, None, :]
        slice_1: f32[2, 128] = torch.ops.aten.slice.Tensor(alias_1, 0, 0, 9223372036854775807);  alias_1 = None
        unsqueeze: f32[2, 1, 128] = torch.ops.aten.unsqueeze.default(slice_1, 1);  slice_1 = None
        unsqueeze_1: f32[2, 1, 1, 128] = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        slice_2: f32[2, 1, 1, 128] = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
        
        # No stacktrace found for following nodes
        _tensor_constant0 = self._tensor_constant0
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/modeling_utils.py:791, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        lift_fresh_copy: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        sub: f32[2, 1, 1, 128] = torch.ops.aten.sub.Tensor(lift_fresh_copy, slice_2);  lift_fresh_copy = None
        mul: f32[2, 1, 1, 128] = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:988, code: hidden_states = self.dropout(inputs_embeds)
        rand_like: f32[2, 128, 512] = torch.ops.aten.rand_like.default(embedding, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_2: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like);  rand_like = None
        gt: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_2, 0.1);  alias_2 = None
        mul_1: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt, embedding)
        mul_2: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_1, 1.1111111111111112);  mul_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_1: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(mul_2, 2)
        mean: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        sqrt: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul_3: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_2, reciprocal)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_4: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_1, mul_3);  mul_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute: f32[512, 384] = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        view_1: f32[256, 512] = torch.ops.aten.view.default(mul_4, [256, 512]);  mul_4 = None
        mm: f32[256, 384] = torch.ops.aten.mm.default(view_1, permute)
        _unsafe_view: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm, [2, 128, 384]);  mm = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_2: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view, [2, -1, 6, 64]);  _unsafe_view = None
        permute_1: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_2: f32[512, 384] = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        mm_1: f32[256, 384] = torch.ops.aten.mm.default(view_1, permute_2)
        _unsafe_view_1: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_1, [2, 128, 384]);  mm_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_4: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_1, [2, -1, 6, 64]);  _unsafe_view_1 = None
        permute_3: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_4: f32[512, 384] = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
        mm_2: f32[256, 384] = torch.ops.aten.mm.default(view_1, permute_4);  view_1 = None
        _unsafe_view_2: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_2, [2, 128, 384]);  mm_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_6: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_2, [2, -1, 6, 64]);  _unsafe_view_2 = None
        permute_5: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_6: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_1, [2, 6, 128, 64]);  permute_1 = None
        clone: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        _unsafe_view_3: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone, [12, 128, 64]);  clone = None
        expand_1: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_6, [2, 6, 64, 128]);  permute_6 = None
        clone_1: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        _unsafe_view_4: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_1, [12, 64, 128]);  clone_1 = None
        bmm: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_3, _unsafe_view_4)
        _unsafe_view_5: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm, [2, 6, 128, 128]);  bmm = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:425, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        arange: i64[128] = torch.ops.aten.arange.default(128, dtype = torch.int64, device = device(type='cuda', index=0), pin_memory = False)
        alias_6: i64[128] = torch.ops.aten.alias.default(arange);  arange = None
        alias_7: i64[128] = torch.ops.aten.alias.default(alias_6);  alias_6 = None
        slice_3: i64[128] = torch.ops.aten.slice.Tensor(alias_7, 0, 0, 9223372036854775807)
        unsqueeze_2: i64[128, 1] = torch.ops.aten.unsqueeze.default(slice_3, 1);  slice_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:426, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        unsqueeze_3: i64[1, 128] = torch.ops.aten.unsqueeze.default(alias_7, 0);  alias_7 = None
        slice_4: i64[1, 128] = torch.ops.aten.slice.Tensor(unsqueeze_3, 1, 0, 9223372036854775807);  unsqueeze_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:427, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
        sub_1: i64[128, 128] = torch.ops.aten.sub.Tensor(slice_4, unsqueeze_2);  slice_4 = unsqueeze_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1611, code: encoder_outputs = self.encoder(
        gt_1: b8[128, 128] = torch.ops.aten.gt.Scalar(sub_1, 0)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:398, code: relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        _to_copy: i64[128, 128] = torch.ops.aten._to_copy.default(gt_1, dtype = torch.int64);  gt_1 = None
        mul_5: i64[128, 128] = torch.ops.aten.mul.Tensor(_to_copy, 16);  _to_copy = None
        add_1: i64[128, 128] = torch.ops.aten.add.Tensor(mul_5, 0);  mul_5 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:399, code: relative_position = torch.abs(relative_position)
        abs_1: i64[128, 128] = torch.ops.aten.abs.default(sub_1)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1611, code: encoder_outputs = self.encoder(
        lt: b8[128, 128] = torch.ops.aten.lt.Scalar(abs_1, 8)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:410, code: torch.log(relative_position.float() / max_exact)
        _to_copy_1: f32[128, 128] = torch.ops.aten._to_copy.default(abs_1, dtype = torch.float32)
        div: f32[128, 128] = torch.ops.aten.div.Tensor(_to_copy_1, 8);  _to_copy_1 = None
        log: f32[128, 128] = torch.ops.aten.log.default(div);  div = None
        div_1: f32[128, 128] = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
        mul_6: f32[128, 128] = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:413, code: ).to(torch.long)
        _to_copy_2: i64[128, 128] = torch.ops.aten._to_copy.default(mul_6, dtype = torch.int64);  mul_6 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:409, code: relative_position_if_large = max_exact + (
        add_2: i64[128, 128] = torch.ops.aten.add.Tensor(_to_copy_2, 8);  _to_copy_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:415, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_like: i64[128, 128] = torch.ops.aten.full_like.default(add_2, 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_10: i64[128, 128] = torch.ops.aten.alias.default(full_like);  full_like = None
        alias_11: i64[128, 128] = torch.ops.aten.alias.default(alias_10);  alias_10 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:414, code: relative_position_if_large = torch.min(
        minimum: i64[128, 128] = torch.ops.aten.minimum.default(add_2, alias_11);  add_2 = alias_11 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:418, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where: i64[128, 128] = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
        add_3: i64[128, 128] = torch.ops.aten.add.Tensor(add_1, where);  add_1 = where = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:434, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_1: f32[128, 128, 6] = torch.ops.aten.embedding.default(primals_47, add_3);  primals_47 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:435, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_7: f32[6, 128, 128] = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
        unsqueeze_4: f32[1, 6, 128, 128] = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:529, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
        add_4: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(unsqueeze_4, mul);  unsqueeze_4 = mul = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_5: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_5, add_4);  _unsafe_view_5 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_5, [-1], True)
        sub_2: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_5, amax);  add_5 = amax = None
        exp: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_seed_like: i32[] = torch.ops.prims.philox_seed_like.default(div_2)
        philox_rand_like: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_2, philox_seed_like, 0)
        gt_2: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like, 0.1);  philox_rand_like = None
        _to_copy_3: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_2, dtype = torch.float32);  gt_2 = None
        mul_7: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_3, div_2);  _to_copy_3 = None
        mul_8: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112);  mul_7 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_2: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_8, [2, 6, 128, 128]);  mul_8 = None
        view_7: f32[12, 128, 128] = torch.ops.aten.view.default(expand_2, [12, 128, 128]);  expand_2 = None
        expand_3: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_5, [2, 6, 128, 64]);  permute_5 = None
        clone_2: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        _unsafe_view_6: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_2, [12, 128, 64]);  clone_2 = None
        bmm_1: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_7, _unsafe_view_6)
        _unsafe_view_7: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_1, [2, 6, 128, 64]);  bmm_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_8: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_7, [0, 2, 1, 3]);  _unsafe_view_7 = None
        clone_3: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_8: f32[2, 128, 384] = torch.ops.aten.view.default(clone_3, [2, -1, 384]);  clone_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_9: f32[384, 512] = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        view_9: f32[256, 384] = torch.ops.aten.view.default(view_8, [256, 384]);  view_8 = None
        mm_3: f32[256, 512] = torch.ops.aten.mm.default(view_9, permute_9)
        _unsafe_view_8: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_3, [2, 128, 512]);  mm_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        rand_like_1: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_8, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_15: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_1);  rand_like_1 = None
        gt_3: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_15, 0.1);  alias_15 = None
        mul_9: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_3, _unsafe_view_8);  _unsafe_view_8 = None
        mul_10: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_9, 1.1111111111111112);  mul_9 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_6: f32[2, 128, 512] = torch.ops.aten.add.Tensor(mul_2, mul_10);  mul_2 = mul_10 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_2: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_6, 2)
        mean_1: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_7: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        sqrt_1: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_7);  add_7 = None
        reciprocal_1: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_11: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_6, reciprocal_1)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_12: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_2, mul_11);  mul_11 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_10: f32[512, 1024] = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        view_10: f32[256, 512] = torch.ops.aten.view.default(mul_12, [256, 512]);  mul_12 = None
        mm_4: f32[256, 1024] = torch.ops.aten.mm.default(view_10, permute_10)
        _unsafe_view_9: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_4, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:35, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_13: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(_unsafe_view_9, 0.5)
        pow_3: f32[2, 128, 1024] = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_9, 3.0)
        mul_14: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_8: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(_unsafe_view_9, mul_14);  _unsafe_view_9 = mul_14 = None
        mul_15: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(add_8, 0.7978845608028654);  add_8 = None
        mul_16: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_15, -2.0);  mul_15 = None
        exp_1: f32[2, 128, 1024] = torch.ops.aten.exp.default(mul_16);  mul_16 = None
        add_9: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(exp_1, 1.0);  exp_1 = None
        reciprocal_2: f32[2, 128, 1024] = torch.ops.aten.reciprocal.default(add_9);  add_9 = None
        mul_17: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(reciprocal_2, 2.0);  reciprocal_2 = None
        sub_3: f32[2, 128, 1024] = torch.ops.aten.sub.Tensor(mul_17, 1.0);  mul_17 = None
        add_10: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(sub_3, 1.0)
        mul_18: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_13, add_10);  mul_13 = add_10 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_11: f32[512, 1024] = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        mm_5: f32[256, 1024] = torch.ops.aten.mm.default(view_10, permute_11);  view_10 = None
        _unsafe_view_10: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_5, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:309, code: hidden_states = hidden_gelu * hidden_linear
        mul_19: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_18, _unsafe_view_10);  mul_18 = _unsafe_view_10 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:310, code: hidden_states = self.dropout(hidden_states)
        rand_like_2: f32[2, 128, 1024] = torch.ops.aten.rand_like.default(mul_19, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_22: f32[2, 128, 1024] = torch.ops.aten.alias.default(rand_like_2);  rand_like_2 = None
        gt_4: b8[2, 128, 1024] = torch.ops.aten.gt.Scalar(alias_22, 0.1);  alias_22 = None
        mul_20: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(gt_4, mul_19);  mul_19 = None
        mul_21: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_20, 1.1111111111111112);  mul_20 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_12: f32[1024, 512] = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        view_12: f32[256, 1024] = torch.ops.aten.view.default(mul_21, [256, 1024]);  mul_21 = None
        mm_6: f32[256, 512] = torch.ops.aten.mm.default(view_12, permute_12)
        _unsafe_view_11: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_6, [2, 128, 512]);  mm_6 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        rand_like_3: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_11, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_23: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_3);  rand_like_3 = None
        gt_5: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_23, 0.1);  alias_23 = None
        mul_22: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_5, _unsafe_view_11);  _unsafe_view_11 = None
        mul_23: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_22, 1.1111111111111112);  mul_22 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_11: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_6, mul_23);  mul_23 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_4: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_11, 2)
        mean_2: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_12: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        sqrt_2: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_12);  add_12 = None
        reciprocal_3: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_24: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_11, reciprocal_3)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_25: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_3, mul_24);  mul_24 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_13: f32[512, 384] = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
        view_13: f32[256, 512] = torch.ops.aten.view.default(mul_25, [256, 512]);  mul_25 = None
        mm_7: f32[256, 384] = torch.ops.aten.mm.default(view_13, permute_13)
        _unsafe_view_12: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_7, [2, 128, 384]);  mm_7 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_14: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_12, [2, -1, 6, 64]);  _unsafe_view_12 = None
        permute_14: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_15: f32[512, 384] = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
        mm_8: f32[256, 384] = torch.ops.aten.mm.default(view_13, permute_15)
        _unsafe_view_13: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_8, [2, 128, 384]);  mm_8 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_16: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_13, [2, -1, 6, 64]);  _unsafe_view_13 = None
        permute_16: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_17: f32[512, 384] = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
        mm_9: f32[256, 384] = torch.ops.aten.mm.default(view_13, permute_17);  view_13 = None
        _unsafe_view_14: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_9, [2, 128, 384]);  mm_9 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_18: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_14, [2, -1, 6, 64]);  _unsafe_view_14 = None
        permute_18: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_19: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2]);  permute_16 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_4: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_14, [2, 6, 128, 64]);  permute_14 = None
        clone_4: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        _unsafe_view_15: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_4, [12, 128, 64]);  clone_4 = None
        expand_5: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_19, [2, 6, 64, 128]);  permute_19 = None
        clone_5: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        _unsafe_view_16: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_5, [12, 64, 128]);  clone_5 = None
        bmm_2: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_15, _unsafe_view_16)
        _unsafe_view_17: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_2, [2, 6, 128, 128]);  bmm_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_13: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_17, add_4);  _unsafe_view_17 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_1: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_13, [-1], True)
        sub_4: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_13, amax_1);  add_13 = amax_1 = None
        exp_2: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_3: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_2, sum_2);  exp_2 = sum_2 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_1: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_3, philox_seed_like, 196608)
        gt_6: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_1, 0.1);  philox_rand_like_1 = None
        _to_copy_4: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_6, dtype = torch.float32);  gt_6 = None
        mul_26: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_4, div_3);  _to_copy_4 = None
        mul_27: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_26, 1.1111111111111112);  mul_26 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_6: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_27, [2, 6, 128, 128]);  mul_27 = None
        view_19: f32[12, 128, 128] = torch.ops.aten.view.default(expand_6, [12, 128, 128]);  expand_6 = None
        expand_7: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_18, [2, 6, 128, 64]);  permute_18 = None
        clone_6: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        _unsafe_view_18: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_6, [12, 128, 64]);  clone_6 = None
        bmm_3: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_19, _unsafe_view_18)
        _unsafe_view_19: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_3, [2, 6, 128, 64]);  bmm_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_20: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_19, [0, 2, 1, 3]);  _unsafe_view_19 = None
        clone_7: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
        view_20: f32[2, 128, 384] = torch.ops.aten.view.default(clone_7, [2, -1, 384]);  clone_7 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_21: f32[384, 512] = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        view_21: f32[256, 384] = torch.ops.aten.view.default(view_20, [256, 384]);  view_20 = None
        mm_10: f32[256, 512] = torch.ops.aten.mm.default(view_21, permute_21)
        _unsafe_view_20: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_10, [2, 128, 512]);  mm_10 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        rand_like_4: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_20, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_30: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_4);  rand_like_4 = None
        gt_7: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_30, 0.1);  alias_30 = None
        mul_28: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_7, _unsafe_view_20);  _unsafe_view_20 = None
        mul_29: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_28, 1.1111111111111112);  mul_28 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_14: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_11, mul_29);  mul_29 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_5: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_14, 2)
        mean_3: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_15: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        sqrt_3: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_15);  add_15 = None
        reciprocal_4: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_30: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_14, reciprocal_4)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_31: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_4, mul_30);  mul_30 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_22: f32[512, 1024] = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        view_22: f32[256, 512] = torch.ops.aten.view.default(mul_31, [256, 512]);  mul_31 = None
        mm_11: f32[256, 1024] = torch.ops.aten.mm.default(view_22, permute_22)
        _unsafe_view_21: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_11, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:35, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_32: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(_unsafe_view_21, 0.5)
        pow_6: f32[2, 128, 1024] = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_21, 3.0)
        mul_33: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_16: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(_unsafe_view_21, mul_33);  _unsafe_view_21 = mul_33 = None
        mul_34: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(add_16, 0.7978845608028654);  add_16 = None
        mul_35: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_34, -2.0);  mul_34 = None
        exp_3: f32[2, 128, 1024] = torch.ops.aten.exp.default(mul_35);  mul_35 = None
        add_17: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(exp_3, 1.0);  exp_3 = None
        reciprocal_5: f32[2, 128, 1024] = torch.ops.aten.reciprocal.default(add_17);  add_17 = None
        mul_36: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(reciprocal_5, 2.0);  reciprocal_5 = None
        sub_5: f32[2, 128, 1024] = torch.ops.aten.sub.Tensor(mul_36, 1.0);  mul_36 = None
        add_18: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(sub_5, 1.0)
        mul_37: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_32, add_18);  mul_32 = add_18 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_23: f32[512, 1024] = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        mm_12: f32[256, 1024] = torch.ops.aten.mm.default(view_22, permute_23);  view_22 = None
        _unsafe_view_22: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_12, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:309, code: hidden_states = hidden_gelu * hidden_linear
        mul_38: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_37, _unsafe_view_22);  mul_37 = _unsafe_view_22 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:310, code: hidden_states = self.dropout(hidden_states)
        rand_like_5: f32[2, 128, 1024] = torch.ops.aten.rand_like.default(mul_38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_37: f32[2, 128, 1024] = torch.ops.aten.alias.default(rand_like_5);  rand_like_5 = None
        gt_8: b8[2, 128, 1024] = torch.ops.aten.gt.Scalar(alias_37, 0.1);  alias_37 = None
        mul_39: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(gt_8, mul_38);  mul_38 = None
        mul_40: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_39, 1.1111111111111112);  mul_39 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_24: f32[1024, 512] = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        view_24: f32[256, 1024] = torch.ops.aten.view.default(mul_40, [256, 1024]);  mul_40 = None
        mm_13: f32[256, 512] = torch.ops.aten.mm.default(view_24, permute_24)
        _unsafe_view_23: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_13, [2, 128, 512]);  mm_13 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        rand_like_6: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_23, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_38: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_6);  rand_like_6 = None
        gt_9: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_38, 0.1);  alias_38 = None
        mul_41: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_9, _unsafe_view_23);  _unsafe_view_23 = None
        mul_42: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_41, 1.1111111111111112);  mul_41 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_19: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_14, mul_42);  mul_42 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_7: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_19, 2)
        mean_4: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_20: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        sqrt_4: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_20);  add_20 = None
        reciprocal_6: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_43: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_19, reciprocal_6)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_44: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_5, mul_43);  mul_43 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_25: f32[512, 384] = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        view_25: f32[256, 512] = torch.ops.aten.view.default(mul_44, [256, 512]);  mul_44 = None
        mm_14: f32[256, 384] = torch.ops.aten.mm.default(view_25, permute_25)
        _unsafe_view_24: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_14, [2, 128, 384]);  mm_14 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_26: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_24, [2, -1, 6, 64]);  _unsafe_view_24 = None
        permute_26: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_27: f32[512, 384] = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        mm_15: f32[256, 384] = torch.ops.aten.mm.default(view_25, permute_27)
        _unsafe_view_25: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_15, [2, 128, 384]);  mm_15 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_28: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_25, [2, -1, 6, 64]);  _unsafe_view_25 = None
        permute_28: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_29: f32[512, 384] = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        mm_16: f32[256, 384] = torch.ops.aten.mm.default(view_25, permute_29);  view_25 = None
        _unsafe_view_26: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_16, [2, 128, 384]);  mm_16 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_30: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_26, [2, -1, 6, 64]);  _unsafe_view_26 = None
        permute_30: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_31: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_28, [0, 1, 3, 2]);  permute_28 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_8: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_26, [2, 6, 128, 64]);  permute_26 = None
        clone_8: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        _unsafe_view_27: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_8, [12, 128, 64]);  clone_8 = None
        expand_9: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_31, [2, 6, 64, 128]);  permute_31 = None
        clone_9: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        _unsafe_view_28: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_9, [12, 64, 128]);  clone_9 = None
        bmm_4: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_27, _unsafe_view_28)
        _unsafe_view_29: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_4, [2, 6, 128, 128]);  bmm_4 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_21: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_29, add_4);  _unsafe_view_29 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_2: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_21, [-1], True)
        sub_6: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_21, amax_2);  add_21 = amax_2 = None
        exp_4: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_3: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_4, sum_3);  exp_4 = sum_3 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_2: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_4, philox_seed_like, 393216)
        gt_10: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_2, 0.1);  philox_rand_like_2 = None
        _to_copy_5: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_10, dtype = torch.float32);  gt_10 = None
        mul_45: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_5, div_4);  _to_copy_5 = None
        mul_46: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_45, 1.1111111111111112);  mul_45 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_10: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_46, [2, 6, 128, 128]);  mul_46 = None
        view_31: f32[12, 128, 128] = torch.ops.aten.view.default(expand_10, [12, 128, 128]);  expand_10 = None
        expand_11: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_30, [2, 6, 128, 64]);  permute_30 = None
        clone_10: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        _unsafe_view_30: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_10, [12, 128, 64]);  clone_10 = None
        bmm_5: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_31, _unsafe_view_30)
        _unsafe_view_31: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_5, [2, 6, 128, 64]);  bmm_5 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_32: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_31, [0, 2, 1, 3]);  _unsafe_view_31 = None
        clone_11: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_32: f32[2, 128, 384] = torch.ops.aten.view.default(clone_11, [2, -1, 384]);  clone_11 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_33: f32[384, 512] = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        view_33: f32[256, 384] = torch.ops.aten.view.default(view_32, [256, 384]);  view_32 = None
        mm_17: f32[256, 512] = torch.ops.aten.mm.default(view_33, permute_33)
        _unsafe_view_32: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_17, [2, 128, 512]);  mm_17 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        rand_like_7: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_32, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_45: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_7);  rand_like_7 = None
        gt_11: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_45, 0.1);  alias_45 = None
        mul_47: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_11, _unsafe_view_32);  _unsafe_view_32 = None
        mul_48: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_47, 1.1111111111111112);  mul_47 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_22: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_19, mul_48);  mul_48 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_8: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_22, 2)
        mean_5: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_23: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        sqrt_5: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_23);  add_23 = None
        reciprocal_7: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_49: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_22, reciprocal_7)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_50: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_6, mul_49);  mul_49 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_34: f32[512, 1024] = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        view_34: f32[256, 512] = torch.ops.aten.view.default(mul_50, [256, 512]);  mul_50 = None
        mm_18: f32[256, 1024] = torch.ops.aten.mm.default(view_34, permute_34)
        _unsafe_view_33: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_18, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:35, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_51: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(_unsafe_view_33, 0.5)
        pow_9: f32[2, 128, 1024] = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_33, 3.0)
        mul_52: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_24: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(_unsafe_view_33, mul_52);  _unsafe_view_33 = mul_52 = None
        mul_53: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(add_24, 0.7978845608028654);  add_24 = None
        mul_54: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_53, -2.0);  mul_53 = None
        exp_5: f32[2, 128, 1024] = torch.ops.aten.exp.default(mul_54);  mul_54 = None
        add_25: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(exp_5, 1.0);  exp_5 = None
        reciprocal_8: f32[2, 128, 1024] = torch.ops.aten.reciprocal.default(add_25);  add_25 = None
        mul_55: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(reciprocal_8, 2.0);  reciprocal_8 = None
        sub_7: f32[2, 128, 1024] = torch.ops.aten.sub.Tensor(mul_55, 1.0);  mul_55 = None
        add_26: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(sub_7, 1.0)
        mul_56: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_51, add_26);  mul_51 = add_26 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_35: f32[512, 1024] = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        mm_19: f32[256, 1024] = torch.ops.aten.mm.default(view_34, permute_35);  view_34 = None
        _unsafe_view_34: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_19, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:309, code: hidden_states = hidden_gelu * hidden_linear
        mul_57: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_56, _unsafe_view_34);  mul_56 = _unsafe_view_34 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:310, code: hidden_states = self.dropout(hidden_states)
        rand_like_8: f32[2, 128, 1024] = torch.ops.aten.rand_like.default(mul_57, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_52: f32[2, 128, 1024] = torch.ops.aten.alias.default(rand_like_8);  rand_like_8 = None
        gt_12: b8[2, 128, 1024] = torch.ops.aten.gt.Scalar(alias_52, 0.1);  alias_52 = None
        mul_58: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(gt_12, mul_57);  mul_57 = None
        mul_59: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_58, 1.1111111111111112);  mul_58 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_36: f32[1024, 512] = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
        view_36: f32[256, 1024] = torch.ops.aten.view.default(mul_59, [256, 1024]);  mul_59 = None
        mm_20: f32[256, 512] = torch.ops.aten.mm.default(view_36, permute_36)
        _unsafe_view_35: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_20, [2, 128, 512]);  mm_20 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        rand_like_9: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_35, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_53: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_9);  rand_like_9 = None
        gt_13: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_53, 0.1);  alias_53 = None
        mul_60: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_13, _unsafe_view_35);  _unsafe_view_35 = None
        mul_61: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_60, 1.1111111111111112);  mul_60 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_27: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_22, mul_61);  mul_61 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_10: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_27, 2)
        mean_6: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_28: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        sqrt_6: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_28);  add_28 = None
        reciprocal_9: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        mul_62: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_27, reciprocal_9)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_63: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_7, mul_62);  mul_62 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_37: f32[512, 384] = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        view_37: f32[256, 512] = torch.ops.aten.view.default(mul_63, [256, 512]);  mul_63 = None
        mm_21: f32[256, 384] = torch.ops.aten.mm.default(view_37, permute_37)
        _unsafe_view_36: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_21, [2, 128, 384]);  mm_21 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_38: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_36, [2, -1, 6, 64]);  _unsafe_view_36 = None
        permute_38: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_39: f32[512, 384] = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        mm_22: f32[256, 384] = torch.ops.aten.mm.default(view_37, permute_39)
        _unsafe_view_37: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_22, [2, 128, 384]);  mm_22 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_40: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_37, [2, -1, 6, 64]);  _unsafe_view_37 = None
        permute_40: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_41: f32[512, 384] = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
        mm_23: f32[256, 384] = torch.ops.aten.mm.default(view_37, permute_41);  view_37 = None
        _unsafe_view_38: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_23, [2, 128, 384]);  mm_23 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_42: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_38, [2, -1, 6, 64]);  _unsafe_view_38 = None
        permute_42: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_43: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_40, [0, 1, 3, 2]);  permute_40 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_12: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_38, [2, 6, 128, 64]);  permute_38 = None
        clone_12: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        _unsafe_view_39: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_12, [12, 128, 64]);  clone_12 = None
        expand_13: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_43, [2, 6, 64, 128]);  permute_43 = None
        clone_13: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        _unsafe_view_40: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_13, [12, 64, 128]);  clone_13 = None
        bmm_6: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_39, _unsafe_view_40)
        _unsafe_view_41: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_6, [2, 6, 128, 128]);  bmm_6 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_29: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_41, add_4);  _unsafe_view_41 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_3: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_29, [-1], True)
        sub_8: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_29, amax_3);  add_29 = amax_3 = None
        exp_6: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_4: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_5: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_6, sum_4);  exp_6 = sum_4 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_3: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_5, philox_seed_like, 589824)
        gt_14: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_3, 0.1);  philox_rand_like_3 = None
        _to_copy_6: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_14, dtype = torch.float32);  gt_14 = None
        mul_64: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_6, div_5);  _to_copy_6 = None
        mul_65: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_64, 1.1111111111111112);  mul_64 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_14: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_65, [2, 6, 128, 128]);  mul_65 = None
        view_43: f32[12, 128, 128] = torch.ops.aten.view.default(expand_14, [12, 128, 128]);  expand_14 = None
        expand_15: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_42, [2, 6, 128, 64]);  permute_42 = None
        clone_14: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        _unsafe_view_42: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_14, [12, 128, 64]);  clone_14 = None
        bmm_7: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_43, _unsafe_view_42)
        _unsafe_view_43: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_7, [2, 6, 128, 64]);  bmm_7 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_44: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_43, [0, 2, 1, 3]);  _unsafe_view_43 = None
        clone_15: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_44: f32[2, 128, 384] = torch.ops.aten.view.default(clone_15, [2, -1, 384]);  clone_15 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_45: f32[384, 512] = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
        view_45: f32[256, 384] = torch.ops.aten.view.default(view_44, [256, 384]);  view_44 = None
        mm_24: f32[256, 512] = torch.ops.aten.mm.default(view_45, permute_45)
        _unsafe_view_44: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_24, [2, 128, 512]);  mm_24 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        rand_like_10: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_44, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_60: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_10);  rand_like_10 = None
        gt_15: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_60, 0.1);  alias_60 = None
        mul_66: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_15, _unsafe_view_44);  _unsafe_view_44 = None
        mul_67: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_66, 1.1111111111111112);  mul_66 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_30: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_27, mul_67);  mul_67 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_11: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_30, 2)
        mean_7: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_31: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        sqrt_7: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_31);  add_31 = None
        reciprocal_10: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        mul_68: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_30, reciprocal_10)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_69: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_8, mul_68);  mul_68 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_46: f32[512, 1024] = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        view_46: f32[256, 512] = torch.ops.aten.view.default(mul_69, [256, 512]);  mul_69 = None
        mm_25: f32[256, 1024] = torch.ops.aten.mm.default(view_46, permute_46)
        _unsafe_view_45: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_25, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:35, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_70: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(_unsafe_view_45, 0.5)
        pow_12: f32[2, 128, 1024] = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_45, 3.0)
        mul_71: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_32: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(_unsafe_view_45, mul_71);  _unsafe_view_45 = mul_71 = None
        mul_72: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(add_32, 0.7978845608028654);  add_32 = None
        mul_73: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_72, -2.0);  mul_72 = None
        exp_7: f32[2, 128, 1024] = torch.ops.aten.exp.default(mul_73);  mul_73 = None
        add_33: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(exp_7, 1.0);  exp_7 = None
        reciprocal_11: f32[2, 128, 1024] = torch.ops.aten.reciprocal.default(add_33);  add_33 = None
        mul_74: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(reciprocal_11, 2.0);  reciprocal_11 = None
        sub_9: f32[2, 128, 1024] = torch.ops.aten.sub.Tensor(mul_74, 1.0);  mul_74 = None
        add_34: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(sub_9, 1.0)
        mul_75: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_70, add_34);  mul_70 = add_34 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_47: f32[512, 1024] = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
        mm_26: f32[256, 1024] = torch.ops.aten.mm.default(view_46, permute_47);  view_46 = None
        _unsafe_view_46: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_26, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:309, code: hidden_states = hidden_gelu * hidden_linear
        mul_76: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_75, _unsafe_view_46);  mul_75 = _unsafe_view_46 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:310, code: hidden_states = self.dropout(hidden_states)
        rand_like_11: f32[2, 128, 1024] = torch.ops.aten.rand_like.default(mul_76, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_67: f32[2, 128, 1024] = torch.ops.aten.alias.default(rand_like_11);  rand_like_11 = None
        gt_16: b8[2, 128, 1024] = torch.ops.aten.gt.Scalar(alias_67, 0.1);  alias_67 = None
        mul_77: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(gt_16, mul_76);  mul_76 = None
        mul_78: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_77, 1.1111111111111112);  mul_77 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_48: f32[1024, 512] = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        view_48: f32[256, 1024] = torch.ops.aten.view.default(mul_78, [256, 1024]);  mul_78 = None
        mm_27: f32[256, 512] = torch.ops.aten.mm.default(view_48, permute_48)
        _unsafe_view_47: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_27, [2, 128, 512]);  mm_27 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        rand_like_12: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_47, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_68: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_12);  rand_like_12 = None
        gt_17: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_68, 0.1);  alias_68 = None
        mul_79: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_17, _unsafe_view_47);  _unsafe_view_47 = None
        mul_80: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_79, 1.1111111111111112);  mul_79 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_35: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_30, mul_80);  mul_80 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_13: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_35, 2)
        mean_8: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_36: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        sqrt_8: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_36);  add_36 = None
        reciprocal_12: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        mul_81: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_35, reciprocal_12)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_82: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_9, mul_81);  mul_81 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_49: f32[512, 384] = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        view_49: f32[256, 512] = torch.ops.aten.view.default(mul_82, [256, 512]);  mul_82 = None
        mm_28: f32[256, 384] = torch.ops.aten.mm.default(view_49, permute_49)
        _unsafe_view_48: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_28, [2, 128, 384]);  mm_28 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_50: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_48, [2, -1, 6, 64]);  _unsafe_view_48 = None
        permute_50: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_51: f32[512, 384] = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        mm_29: f32[256, 384] = torch.ops.aten.mm.default(view_49, permute_51)
        _unsafe_view_49: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_29, [2, 128, 384]);  mm_29 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_52: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_49, [2, -1, 6, 64]);  _unsafe_view_49 = None
        permute_52: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_53: f32[512, 384] = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        mm_30: f32[256, 384] = torch.ops.aten.mm.default(view_49, permute_53);  view_49 = None
        _unsafe_view_50: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_30, [2, 128, 384]);  mm_30 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_54: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_50, [2, -1, 6, 64]);  _unsafe_view_50 = None
        permute_54: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_55: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_52, [0, 1, 3, 2]);  permute_52 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_16: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_50, [2, 6, 128, 64]);  permute_50 = None
        clone_16: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        _unsafe_view_51: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_16, [12, 128, 64]);  clone_16 = None
        expand_17: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_55, [2, 6, 64, 128]);  permute_55 = None
        clone_17: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        _unsafe_view_52: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_17, [12, 64, 128]);  clone_17 = None
        bmm_8: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_51, _unsafe_view_52)
        _unsafe_view_53: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_8, [2, 6, 128, 128]);  bmm_8 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_37: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_53, add_4);  _unsafe_view_53 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_4: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_37, [-1], True)
        sub_10: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_37, amax_4);  add_37 = amax_4 = None
        exp_8: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_5: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_6: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_8, sum_5);  exp_8 = sum_5 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_4: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_6, philox_seed_like, 786432)
        gt_18: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_4, 0.1);  philox_rand_like_4 = None
        _to_copy_7: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_18, dtype = torch.float32);  gt_18 = None
        mul_83: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_7, div_6);  _to_copy_7 = None
        mul_84: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_83, 1.1111111111111112);  mul_83 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_18: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_84, [2, 6, 128, 128]);  mul_84 = None
        view_55: f32[12, 128, 128] = torch.ops.aten.view.default(expand_18, [12, 128, 128]);  expand_18 = None
        expand_19: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_54, [2, 6, 128, 64]);  permute_54 = None
        clone_18: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        _unsafe_view_54: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_18, [12, 128, 64]);  clone_18 = None
        bmm_9: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_55, _unsafe_view_54)
        _unsafe_view_55: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_9, [2, 6, 128, 64]);  bmm_9 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_56: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_55, [0, 2, 1, 3]);  _unsafe_view_55 = None
        clone_19: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_56: f32[2, 128, 384] = torch.ops.aten.view.default(clone_19, [2, -1, 384]);  clone_19 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_57: f32[384, 512] = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        view_57: f32[256, 384] = torch.ops.aten.view.default(view_56, [256, 384]);  view_56 = None
        mm_31: f32[256, 512] = torch.ops.aten.mm.default(view_57, permute_57)
        _unsafe_view_56: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_31, [2, 128, 512]);  mm_31 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        rand_like_13: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_56, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_75: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_13);  rand_like_13 = None
        gt_19: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_75, 0.1);  alias_75 = None
        mul_85: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_19, _unsafe_view_56);  _unsafe_view_56 = None
        mul_86: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_85, 1.1111111111111112);  mul_85 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_38: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_35, mul_86);  mul_86 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_14: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_38, 2)
        mean_9: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_39: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        sqrt_9: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_39);  add_39 = None
        reciprocal_13: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        mul_87: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_38, reciprocal_13)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_88: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_10, mul_87);  mul_87 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_58: f32[512, 1024] = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
        view_58: f32[256, 512] = torch.ops.aten.view.default(mul_88, [256, 512]);  mul_88 = None
        mm_32: f32[256, 1024] = torch.ops.aten.mm.default(view_58, permute_58)
        _unsafe_view_57: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_32, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:35, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_89: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(_unsafe_view_57, 0.5)
        pow_15: f32[2, 128, 1024] = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_57, 3.0)
        mul_90: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_40: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(_unsafe_view_57, mul_90);  _unsafe_view_57 = mul_90 = None
        mul_91: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(add_40, 0.7978845608028654);  add_40 = None
        mul_92: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_91, -2.0);  mul_91 = None
        exp_9: f32[2, 128, 1024] = torch.ops.aten.exp.default(mul_92);  mul_92 = None
        add_41: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(exp_9, 1.0);  exp_9 = None
        reciprocal_14: f32[2, 128, 1024] = torch.ops.aten.reciprocal.default(add_41);  add_41 = None
        mul_93: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(reciprocal_14, 2.0);  reciprocal_14 = None
        sub_11: f32[2, 128, 1024] = torch.ops.aten.sub.Tensor(mul_93, 1.0);  mul_93 = None
        add_42: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(sub_11, 1.0)
        mul_94: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_89, add_42);  mul_89 = add_42 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_59: f32[512, 1024] = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
        mm_33: f32[256, 1024] = torch.ops.aten.mm.default(view_58, permute_59);  view_58 = None
        _unsafe_view_58: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_33, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:309, code: hidden_states = hidden_gelu * hidden_linear
        mul_95: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_94, _unsafe_view_58);  mul_94 = _unsafe_view_58 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:310, code: hidden_states = self.dropout(hidden_states)
        rand_like_14: f32[2, 128, 1024] = torch.ops.aten.rand_like.default(mul_95, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_82: f32[2, 128, 1024] = torch.ops.aten.alias.default(rand_like_14);  rand_like_14 = None
        gt_20: b8[2, 128, 1024] = torch.ops.aten.gt.Scalar(alias_82, 0.1);  alias_82 = None
        mul_96: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(gt_20, mul_95);  mul_95 = None
        mul_97: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_96, 1.1111111111111112);  mul_96 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_60: f32[1024, 512] = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        view_60: f32[256, 1024] = torch.ops.aten.view.default(mul_97, [256, 1024]);  mul_97 = None
        mm_34: f32[256, 512] = torch.ops.aten.mm.default(view_60, permute_60)
        _unsafe_view_59: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_34, [2, 128, 512]);  mm_34 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        rand_like_15: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_59, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_83: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_15);  rand_like_15 = None
        gt_21: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_83, 0.1);  alias_83 = None
        mul_98: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_21, _unsafe_view_59);  _unsafe_view_59 = None
        mul_99: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_98, 1.1111111111111112);  mul_98 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_43: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_38, mul_99);  mul_99 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_16: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_43, 2)
        mean_10: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_44: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        sqrt_10: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_44);  add_44 = None
        reciprocal_15: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        mul_100: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_43, reciprocal_15)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_101: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_11, mul_100);  mul_100 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_61: f32[512, 384] = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        view_61: f32[256, 512] = torch.ops.aten.view.default(mul_101, [256, 512]);  mul_101 = None
        mm_35: f32[256, 384] = torch.ops.aten.mm.default(view_61, permute_61)
        _unsafe_view_60: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_35, [2, 128, 384]);  mm_35 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_62: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_60, [2, -1, 6, 64]);  _unsafe_view_60 = None
        permute_62: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_62, [0, 2, 1, 3]);  view_62 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_63: f32[512, 384] = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
        mm_36: f32[256, 384] = torch.ops.aten.mm.default(view_61, permute_63)
        _unsafe_view_61: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_36, [2, 128, 384]);  mm_36 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_64: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_61, [2, -1, 6, 64]);  _unsafe_view_61 = None
        permute_64: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_65: f32[512, 384] = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
        mm_37: f32[256, 384] = torch.ops.aten.mm.default(view_61, permute_65);  view_61 = None
        _unsafe_view_62: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_37, [2, 128, 384]);  mm_37 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_66: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_62, [2, -1, 6, 64]);  _unsafe_view_62 = None
        permute_66: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_67: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_64, [0, 1, 3, 2]);  permute_64 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_20: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_62, [2, 6, 128, 64]);  permute_62 = None
        clone_20: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        _unsafe_view_63: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_20, [12, 128, 64]);  clone_20 = None
        expand_21: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_67, [2, 6, 64, 128]);  permute_67 = None
        clone_21: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        _unsafe_view_64: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_21, [12, 64, 128]);  clone_21 = None
        bmm_10: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_63, _unsafe_view_64)
        _unsafe_view_65: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_10, [2, 6, 128, 128]);  bmm_10 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_45: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_65, add_4);  _unsafe_view_65 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_5: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_45, [-1], True)
        sub_12: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_45, amax_5);  add_45 = amax_5 = None
        exp_10: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_6: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_7: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_10, sum_6);  exp_10 = sum_6 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_5: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_7, philox_seed_like, 983040)
        gt_22: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_5, 0.1);  philox_rand_like_5 = None
        _to_copy_8: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_22, dtype = torch.float32);  gt_22 = None
        mul_102: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_8, div_7);  _to_copy_8 = None
        mul_103: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_102, 1.1111111111111112);  mul_102 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_22: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_103, [2, 6, 128, 128]);  mul_103 = None
        view_67: f32[12, 128, 128] = torch.ops.aten.view.default(expand_22, [12, 128, 128]);  expand_22 = None
        expand_23: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_66, [2, 6, 128, 64]);  permute_66 = None
        clone_22: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        _unsafe_view_66: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_22, [12, 128, 64]);  clone_22 = None
        bmm_11: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_67, _unsafe_view_66)
        _unsafe_view_67: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_11, [2, 6, 128, 64]);  bmm_11 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_68: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_67, [0, 2, 1, 3]);  _unsafe_view_67 = None
        clone_23: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_68: f32[2, 128, 384] = torch.ops.aten.view.default(clone_23, [2, -1, 384]);  clone_23 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_69: f32[384, 512] = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
        view_69: f32[256, 384] = torch.ops.aten.view.default(view_68, [256, 384]);  view_68 = None
        mm_38: f32[256, 512] = torch.ops.aten.mm.default(view_69, permute_69)
        _unsafe_view_68: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_38, [2, 128, 512]);  mm_38 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        rand_like_16: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_68, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_90: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_16);  rand_like_16 = None
        gt_23: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_90, 0.1);  alias_90 = None
        mul_104: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_23, _unsafe_view_68);  _unsafe_view_68 = None
        mul_105: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_104, 1.1111111111111112);  mul_104 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_46: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_43, mul_105);  mul_105 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_17: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_46, 2)
        mean_11: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_47: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        sqrt_11: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_47);  add_47 = None
        reciprocal_16: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        mul_106: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_46, reciprocal_16)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_107: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_12, mul_106);  mul_106 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_70: f32[512, 1024] = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
        view_70: f32[256, 512] = torch.ops.aten.view.default(mul_107, [256, 512]);  mul_107 = None
        mm_39: f32[256, 1024] = torch.ops.aten.mm.default(view_70, permute_70)
        _unsafe_view_69: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_39, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:35, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_108: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(_unsafe_view_69, 0.5)
        pow_18: f32[2, 128, 1024] = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_69, 3.0)
        mul_109: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_48: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(_unsafe_view_69, mul_109);  _unsafe_view_69 = mul_109 = None
        mul_110: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(add_48, 0.7978845608028654);  add_48 = None
        mul_111: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_110, -2.0);  mul_110 = None
        exp_11: f32[2, 128, 1024] = torch.ops.aten.exp.default(mul_111);  mul_111 = None
        add_49: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(exp_11, 1.0);  exp_11 = None
        reciprocal_17: f32[2, 128, 1024] = torch.ops.aten.reciprocal.default(add_49);  add_49 = None
        mul_112: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(reciprocal_17, 2.0);  reciprocal_17 = None
        sub_13: f32[2, 128, 1024] = torch.ops.aten.sub.Tensor(mul_112, 1.0);  mul_112 = None
        add_50: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(sub_13, 1.0)
        mul_113: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_108, add_50);  mul_108 = add_50 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_71: f32[512, 1024] = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        mm_40: f32[256, 1024] = torch.ops.aten.mm.default(view_70, permute_71);  view_70 = None
        _unsafe_view_70: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_40, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:309, code: hidden_states = hidden_gelu * hidden_linear
        mul_114: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_113, _unsafe_view_70);  mul_113 = _unsafe_view_70 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:310, code: hidden_states = self.dropout(hidden_states)
        rand_like_17: f32[2, 128, 1024] = torch.ops.aten.rand_like.default(mul_114, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_97: f32[2, 128, 1024] = torch.ops.aten.alias.default(rand_like_17);  rand_like_17 = None
        gt_24: b8[2, 128, 1024] = torch.ops.aten.gt.Scalar(alias_97, 0.1);  alias_97 = None
        mul_115: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(gt_24, mul_114);  mul_114 = None
        mul_116: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_115, 1.1111111111111112);  mul_115 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_72: f32[1024, 512] = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
        view_72: f32[256, 1024] = torch.ops.aten.view.default(mul_116, [256, 1024]);  mul_116 = None
        mm_41: f32[256, 512] = torch.ops.aten.mm.default(view_72, permute_72)
        _unsafe_view_71: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_41, [2, 128, 512]);  mm_41 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        rand_like_18: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_71, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_98: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_18);  rand_like_18 = None
        gt_25: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_98, 0.1);  alias_98 = None
        mul_117: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_25, _unsafe_view_71);  _unsafe_view_71 = None
        mul_118: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_117, 1.1111111111111112);  mul_117 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_51: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_46, mul_118);  mul_118 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_19: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_51, 2)
        mean_12: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_52: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        sqrt_12: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_52);  add_52 = None
        reciprocal_18: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        mul_119: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_51, reciprocal_18)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_120: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_13, mul_119);  mul_119 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_73: f32[512, 384] = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        view_73: f32[256, 512] = torch.ops.aten.view.default(mul_120, [256, 512]);  mul_120 = None
        mm_42: f32[256, 384] = torch.ops.aten.mm.default(view_73, permute_73)
        _unsafe_view_72: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_42, [2, 128, 384]);  mm_42 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_74: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_72, [2, -1, 6, 64]);  _unsafe_view_72 = None
        permute_74: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_75: f32[512, 384] = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        mm_43: f32[256, 384] = torch.ops.aten.mm.default(view_73, permute_75)
        _unsafe_view_73: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_43, [2, 128, 384]);  mm_43 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_76: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_73, [2, -1, 6, 64]);  _unsafe_view_73 = None
        permute_76: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_77: f32[512, 384] = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
        mm_44: f32[256, 384] = torch.ops.aten.mm.default(view_73, permute_77);  view_73 = None
        _unsafe_view_74: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_44, [2, 128, 384]);  mm_44 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_78: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_74, [2, -1, 6, 64]);  _unsafe_view_74 = None
        permute_78: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_79: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_76, [0, 1, 3, 2]);  permute_76 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_24: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_74, [2, 6, 128, 64]);  permute_74 = None
        clone_24: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        _unsafe_view_75: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_24, [12, 128, 64]);  clone_24 = None
        expand_25: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_79, [2, 6, 64, 128]);  permute_79 = None
        clone_25: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        _unsafe_view_76: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_25, [12, 64, 128]);  clone_25 = None
        bmm_12: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_75, _unsafe_view_76)
        _unsafe_view_77: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_12, [2, 6, 128, 128]);  bmm_12 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_53: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_77, add_4);  _unsafe_view_77 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_6: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_53, [-1], True)
        sub_14: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_53, amax_6);  add_53 = amax_6 = None
        exp_12: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_7: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_8: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_12, sum_7);  exp_12 = sum_7 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_6: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_8, philox_seed_like, 1179648)
        gt_26: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_6, 0.1);  philox_rand_like_6 = None
        _to_copy_9: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_26, dtype = torch.float32);  gt_26 = None
        mul_121: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_9, div_8);  _to_copy_9 = None
        mul_122: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_121, 1.1111111111111112);  mul_121 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_26: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_122, [2, 6, 128, 128]);  mul_122 = None
        view_79: f32[12, 128, 128] = torch.ops.aten.view.default(expand_26, [12, 128, 128]);  expand_26 = None
        expand_27: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_78, [2, 6, 128, 64]);  permute_78 = None
        clone_26: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        _unsafe_view_78: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_26, [12, 128, 64]);  clone_26 = None
        bmm_13: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_79, _unsafe_view_78)
        _unsafe_view_79: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_13, [2, 6, 128, 64]);  bmm_13 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_80: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_79, [0, 2, 1, 3]);  _unsafe_view_79 = None
        clone_27: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_80: f32[2, 128, 384] = torch.ops.aten.view.default(clone_27, [2, -1, 384]);  clone_27 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_81: f32[384, 512] = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
        view_81: f32[256, 384] = torch.ops.aten.view.default(view_80, [256, 384]);  view_80 = None
        mm_45: f32[256, 512] = torch.ops.aten.mm.default(view_81, permute_81)
        _unsafe_view_80: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_45, [2, 128, 512]);  mm_45 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        rand_like_19: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_80, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_105: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_19);  rand_like_19 = None
        gt_27: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_105, 0.1);  alias_105 = None
        mul_123: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_27, _unsafe_view_80);  _unsafe_view_80 = None
        mul_124: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_123, 1.1111111111111112);  mul_123 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_54: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_51, mul_124);  mul_124 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_20: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_54, 2)
        mean_13: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_55: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        sqrt_13: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_55);  add_55 = None
        reciprocal_19: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        mul_125: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_54, reciprocal_19)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_126: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_14, mul_125);  mul_125 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_82: f32[512, 1024] = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        view_82: f32[256, 512] = torch.ops.aten.view.default(mul_126, [256, 512]);  mul_126 = None
        mm_46: f32[256, 1024] = torch.ops.aten.mm.default(view_82, permute_82)
        _unsafe_view_81: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_46, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:35, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_127: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(_unsafe_view_81, 0.5)
        pow_21: f32[2, 128, 1024] = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_81, 3.0)
        mul_128: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_56: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(_unsafe_view_81, mul_128);  _unsafe_view_81 = mul_128 = None
        mul_129: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(add_56, 0.7978845608028654);  add_56 = None
        mul_130: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_129, -2.0);  mul_129 = None
        exp_13: f32[2, 128, 1024] = torch.ops.aten.exp.default(mul_130);  mul_130 = None
        add_57: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(exp_13, 1.0);  exp_13 = None
        reciprocal_20: f32[2, 128, 1024] = torch.ops.aten.reciprocal.default(add_57);  add_57 = None
        mul_131: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(reciprocal_20, 2.0);  reciprocal_20 = None
        sub_15: f32[2, 128, 1024] = torch.ops.aten.sub.Tensor(mul_131, 1.0);  mul_131 = None
        add_58: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(sub_15, 1.0)
        mul_132: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_127, add_58);  mul_127 = add_58 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_83: f32[512, 1024] = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
        mm_47: f32[256, 1024] = torch.ops.aten.mm.default(view_82, permute_83);  view_82 = None
        _unsafe_view_82: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_47, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:309, code: hidden_states = hidden_gelu * hidden_linear
        mul_133: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_132, _unsafe_view_82);  mul_132 = _unsafe_view_82 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:310, code: hidden_states = self.dropout(hidden_states)
        rand_like_20: f32[2, 128, 1024] = torch.ops.aten.rand_like.default(mul_133, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_112: f32[2, 128, 1024] = torch.ops.aten.alias.default(rand_like_20);  rand_like_20 = None
        gt_28: b8[2, 128, 1024] = torch.ops.aten.gt.Scalar(alias_112, 0.1);  alias_112 = None
        mul_134: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(gt_28, mul_133);  mul_133 = None
        mul_135: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_134, 1.1111111111111112);  mul_134 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_84: f32[1024, 512] = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
        view_84: f32[256, 1024] = torch.ops.aten.view.default(mul_135, [256, 1024]);  mul_135 = None
        mm_48: f32[256, 512] = torch.ops.aten.mm.default(view_84, permute_84)
        _unsafe_view_83: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_48, [2, 128, 512]);  mm_48 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        rand_like_21: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_83, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_113: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_21);  rand_like_21 = None
        gt_29: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_113, 0.1);  alias_113 = None
        mul_136: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_29, _unsafe_view_83);  _unsafe_view_83 = None
        mul_137: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_136, 1.1111111111111112);  mul_136 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_59: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_54, mul_137);  mul_137 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_22: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_59, 2)
        mean_14: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_60: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        sqrt_14: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_60);  add_60 = None
        reciprocal_21: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        mul_138: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_59, reciprocal_21)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_139: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_15, mul_138);  mul_138 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_85: f32[512, 384] = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
        view_85: f32[256, 512] = torch.ops.aten.view.default(mul_139, [256, 512]);  mul_139 = None
        mm_49: f32[256, 384] = torch.ops.aten.mm.default(view_85, permute_85)
        _unsafe_view_84: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_49, [2, 128, 384]);  mm_49 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_86: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_84, [2, -1, 6, 64]);  _unsafe_view_84 = None
        permute_86: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_87: f32[512, 384] = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
        mm_50: f32[256, 384] = torch.ops.aten.mm.default(view_85, permute_87)
        _unsafe_view_85: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_50, [2, 128, 384]);  mm_50 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_88: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_85, [2, -1, 6, 64]);  _unsafe_view_85 = None
        permute_88: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_89: f32[512, 384] = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
        mm_51: f32[256, 384] = torch.ops.aten.mm.default(view_85, permute_89);  view_85 = None
        _unsafe_view_86: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_51, [2, 128, 384]);  mm_51 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_90: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_86, [2, -1, 6, 64]);  _unsafe_view_86 = None
        permute_90: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_91: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_88, [0, 1, 3, 2]);  permute_88 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_28: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_86, [2, 6, 128, 64]);  permute_86 = None
        clone_28: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        _unsafe_view_87: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_28, [12, 128, 64]);  clone_28 = None
        expand_29: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_91, [2, 6, 64, 128]);  permute_91 = None
        clone_29: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        _unsafe_view_88: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_29, [12, 64, 128]);  clone_29 = None
        bmm_14: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_87, _unsafe_view_88)
        _unsafe_view_89: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_14, [2, 6, 128, 128]);  bmm_14 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_61: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_89, add_4);  _unsafe_view_89 = add_4 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_7: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_61, [-1], True)
        sub_16: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_61, amax_7);  add_61 = amax_7 = None
        exp_14: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_8: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_9: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_14, sum_8);  exp_14 = sum_8 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_7: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_9, philox_seed_like, 1376256)
        gt_30: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_7, 0.1);  philox_rand_like_7 = None
        _to_copy_10: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_30, dtype = torch.float32);  gt_30 = None
        mul_140: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_10, div_9);  _to_copy_10 = None
        mul_141: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_140, 1.1111111111111112);  mul_140 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_30: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_141, [2, 6, 128, 128]);  mul_141 = None
        view_91: f32[12, 128, 128] = torch.ops.aten.view.default(expand_30, [12, 128, 128]);  expand_30 = None
        expand_31: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_90, [2, 6, 128, 64]);  permute_90 = None
        clone_30: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        _unsafe_view_90: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_30, [12, 128, 64]);  clone_30 = None
        bmm_15: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_91, _unsafe_view_90)
        _unsafe_view_91: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_15, [2, 6, 128, 64]);  bmm_15 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_92: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_91, [0, 2, 1, 3]);  _unsafe_view_91 = None
        clone_31: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_92: f32[2, 128, 384] = torch.ops.aten.view.default(clone_31, [2, -1, 384]);  clone_31 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_93: f32[384, 512] = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
        view_93: f32[256, 384] = torch.ops.aten.view.default(view_92, [256, 384]);  view_92 = None
        mm_52: f32[256, 512] = torch.ops.aten.mm.default(view_93, permute_93)
        _unsafe_view_92: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_52, [2, 128, 512]);  mm_52 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        rand_like_22: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_92, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_120: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_22);  rand_like_22 = None
        gt_31: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_120, 0.1);  alias_120 = None
        mul_142: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_31, _unsafe_view_92);  _unsafe_view_92 = None
        mul_143: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_142, 1.1111111111111112);  mul_142 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_62: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_59, mul_143);  mul_143 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_23: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_62, 2)
        mean_15: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_63: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        sqrt_15: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_63);  add_63 = None
        reciprocal_22: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        mul_144: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_62, reciprocal_22)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_145: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_16, mul_144);  mul_144 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_94: f32[512, 1024] = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
        view_94: f32[256, 512] = torch.ops.aten.view.default(mul_145, [256, 512]);  mul_145 = None
        mm_53: f32[256, 1024] = torch.ops.aten.mm.default(view_94, permute_94)
        _unsafe_view_93: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_53, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:35, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_146: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(_unsafe_view_93, 0.5)
        pow_24: f32[2, 128, 1024] = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_93, 3.0)
        mul_147: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_64: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(_unsafe_view_93, mul_147);  _unsafe_view_93 = mul_147 = None
        mul_148: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(add_64, 0.7978845608028654);  add_64 = None
        mul_149: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_148, -2.0);  mul_148 = None
        exp_15: f32[2, 128, 1024] = torch.ops.aten.exp.default(mul_149);  mul_149 = None
        add_65: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(exp_15, 1.0);  exp_15 = None
        reciprocal_23: f32[2, 128, 1024] = torch.ops.aten.reciprocal.default(add_65);  add_65 = None
        mul_150: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(reciprocal_23, 2.0);  reciprocal_23 = None
        sub_17: f32[2, 128, 1024] = torch.ops.aten.sub.Tensor(mul_150, 1.0);  mul_150 = None
        add_66: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(sub_17, 1.0)
        mul_151: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_146, add_66);  mul_146 = add_66 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_95: f32[512, 1024] = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        mm_54: f32[256, 1024] = torch.ops.aten.mm.default(view_94, permute_95);  view_94 = None
        _unsafe_view_94: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_54, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:309, code: hidden_states = hidden_gelu * hidden_linear
        mul_152: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_151, _unsafe_view_94);  mul_151 = _unsafe_view_94 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:310, code: hidden_states = self.dropout(hidden_states)
        rand_like_23: f32[2, 128, 1024] = torch.ops.aten.rand_like.default(mul_152, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_127: f32[2, 128, 1024] = torch.ops.aten.alias.default(rand_like_23);  rand_like_23 = None
        gt_32: b8[2, 128, 1024] = torch.ops.aten.gt.Scalar(alias_127, 0.1);  alias_127 = None
        mul_153: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(gt_32, mul_152);  mul_152 = None
        mul_154: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_153, 1.1111111111111112);  mul_153 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_96: f32[1024, 512] = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
        view_96: f32[256, 1024] = torch.ops.aten.view.default(mul_154, [256, 1024]);  mul_154 = None
        mm_55: f32[256, 512] = torch.ops.aten.mm.default(view_96, permute_96)
        _unsafe_view_95: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_55, [2, 128, 512]);  mm_55 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        rand_like_24: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_95, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_128: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_24);  rand_like_24 = None
        gt_33: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_128, 0.1);  alias_128 = None
        mul_155: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_33, _unsafe_view_95);  _unsafe_view_95 = None
        mul_156: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_155, 1.1111111111111112);  mul_155 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_67: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_62, mul_156);  mul_156 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_25: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_67, 2)
        mean_16: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_68: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        sqrt_16: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_68);  add_68 = None
        reciprocal_24: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        mul_157: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_67, reciprocal_24)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_158: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_17, mul_157);  mul_157 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1083, code: hidden_states = self.dropout(hidden_states)
        rand_like_25: f32[2, 128, 512] = torch.ops.aten.rand_like.default(mul_158, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_132: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_25);  rand_like_25 = None
        gt_34: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_132, 0.1);  alias_132 = None
        mul_159: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_34, mul_158);  mul_158 = None
        mul_160: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_159, 1.1111111111111112);  mul_159 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:932, code: input_ids = input_ids.view(-1, input_shape[-1])
        view_97: i64[2, 128] = torch.ops.aten.view.default(primals_192, [-1, 128]);  primals_192 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_2: f32[2, 128, 512] = torch.ops.aten.embedding.default(primals_43, view_97);  primals_43 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:955, code: encoder_attention_mask = torch.ones(
        ones_2: i64[2, 128] = torch.ops.aten.ones.default([2, 128], dtype = torch.int64, device = device(type='cuda', index=0), pin_memory = False)
        alias_135: i64[2, 128] = torch.ops.aten.alias.default(ones_2);  ones_2 = None
        alias_136: i64[2, 128] = torch.ops.aten.alias.default(alias_135);  alias_135 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/modeling_utils.py:723, code: seq_ids = torch.arange(seq_length, device=device)
        arange_2: i64[128] = torch.ops.aten.arange.default(128, device = device(type='cuda', index=0), pin_memory = False)
        alias_137: i64[128] = torch.ops.aten.alias.default(arange_2);  arange_2 = None
        alias_138: i64[128] = torch.ops.aten.alias.default(alias_137);  alias_137 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/modeling_utils.py:724, code: causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        unsqueeze_5: i64[1, 128] = torch.ops.aten.unsqueeze.default(alias_138, 0);  alias_138 = None
        unsqueeze_6: i64[1, 1, 128] = torch.ops.aten.unsqueeze.default(unsqueeze_5, 1)
        slice_5: i64[1, 1, 128] = torch.ops.aten.slice.Tensor(unsqueeze_6, 2, 0, 9223372036854775807);  unsqueeze_6 = None
        repeat: i64[2, 128, 128] = torch.ops.aten.repeat.default(slice_5, [2, 128, 1]);  slice_5 = None
        slice_6: i64[1, 128] = torch.ops.aten.slice.Tensor(unsqueeze_5, 1, 0, 9223372036854775807);  unsqueeze_5 = None
        unsqueeze_8: i64[1, 128, 1] = torch.ops.aten.unsqueeze.default(slice_6, 2);  slice_6 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1648, code: decoder_outputs = self.decoder(
        le: b8[2, 128, 128] = torch.ops.aten.le.Tensor(repeat, unsqueeze_8);  repeat = unsqueeze_8 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/modeling_utils.py:727, code: causal_mask = causal_mask.to(attention_mask.dtype)
        _to_copy_11: f32[2, 128, 128] = torch.ops.aten._to_copy.default(le, dtype = torch.float32);  le = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/modeling_utils.py:739, code: extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        slice_7: f32[2, 128, 128] = torch.ops.aten.slice.Tensor(_to_copy_11, 0, 0, 9223372036854775807);  _to_copy_11 = None
        unsqueeze_9: f32[2, 1, 128, 128] = torch.ops.aten.unsqueeze.default(slice_7, 1);  slice_7 = None
        slice_8: f32[2, 1, 128, 128] = torch.ops.aten.slice.Tensor(unsqueeze_9, 2, 0, 9223372036854775807);  unsqueeze_9 = None
        slice_9: f32[2, 1, 128, 128] = torch.ops.aten.slice.Tensor(slice_8, 3, 0, 9223372036854775807);  slice_8 = None
        mul_161: f32[2, 1, 128, 128] = torch.ops.aten.mul.Tensor(slice_9, slice_2);  slice_9 = slice_2 = None
        
        # No stacktrace found for following nodes
        _tensor_constant1 = self._tensor_constant1
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/modeling_utils.py:791, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        lift_fresh_copy_1: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        sub_18: f32[2, 1, 128, 128] = torch.ops.aten.sub.Tensor(lift_fresh_copy_1, mul_161);  lift_fresh_copy_1 = mul_161 = None
        mul_162: f32[2, 1, 128, 128] = torch.ops.aten.mul.Tensor(sub_18, -3.4028234663852886e+38);  sub_18 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/modeling_utils.py:703, code: encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        slice_12: i64[2, 128] = torch.ops.aten.slice.Tensor(alias_136, 0, 0, 9223372036854775807);  alias_136 = None
        unsqueeze_12: i64[2, 1, 128] = torch.ops.aten.unsqueeze.default(slice_12, 1);  slice_12 = None
        unsqueeze_13: i64[2, 1, 1, 128] = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
        slice_13: i64[2, 1, 1, 128] = torch.ops.aten.slice.Tensor(unsqueeze_13, 3, 0, 9223372036854775807);  unsqueeze_13 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/modeling_utils.py:709, code: encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        _to_copy_12: f32[2, 1, 1, 128] = torch.ops.aten._to_copy.default(slice_13, dtype = torch.float32);  slice_13 = None
        
        # No stacktrace found for following nodes
        _tensor_constant2 = self._tensor_constant2
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/modeling_utils.py:710, code: encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min
        lift_fresh_copy_2: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        sub_19: f32[2, 1, 1, 128] = torch.ops.aten.sub.Tensor(lift_fresh_copy_2, _to_copy_12);  lift_fresh_copy_2 = _to_copy_12 = None
        mul_163: f32[2, 1, 1, 128] = torch.ops.aten.mul.Tensor(sub_19, -3.4028234663852886e+38);  sub_19 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:988, code: hidden_states = self.dropout(inputs_embeds)
        rand_like_26: f32[2, 128, 512] = torch.ops.aten.rand_like.default(embedding_2, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_139: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_26);  rand_like_26 = None
        gt_35: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_139, 0.1);  alias_139 = None
        mul_164: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_35, embedding_2)
        mul_165: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_164, 1.1111111111111112);  mul_164 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_26: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(mul_165, 2)
        mean_17: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_69: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        sqrt_17: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_69);  add_69 = None
        reciprocal_25: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
        mul_166: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_165, reciprocal_25)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_167: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_18, mul_166);  mul_166 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_97: f32[512, 384] = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
        view_98: f32[256, 512] = torch.ops.aten.view.default(mul_167, [256, 512]);  mul_167 = None
        mm_56: f32[256, 384] = torch.ops.aten.mm.default(view_98, permute_97)
        _unsafe_view_96: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_56, [2, 128, 384]);  mm_56 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_99: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_96, [2, -1, 6, 64]);  _unsafe_view_96 = None
        permute_98: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_99: f32[512, 384] = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
        mm_57: f32[256, 384] = torch.ops.aten.mm.default(view_98, permute_99)
        _unsafe_view_97: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_57, [2, 128, 384]);  mm_57 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_101: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_97, [2, -1, 6, 64]);  _unsafe_view_97 = None
        permute_100: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_101: f32[512, 384] = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
        mm_58: f32[256, 384] = torch.ops.aten.mm.default(view_98, permute_101);  view_98 = None
        _unsafe_view_98: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_58, [2, 128, 384]);  mm_58 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_103: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_98, [2, -1, 6, 64]);  _unsafe_view_98 = None
        permute_102: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_103, [0, 2, 1, 3]);  view_103 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_103: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_32: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_98, [2, 6, 128, 64]);  permute_98 = None
        clone_32: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        _unsafe_view_99: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_32, [12, 128, 64]);  clone_32 = None
        expand_33: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_103, [2, 6, 64, 128]);  permute_103 = None
        clone_33: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        _unsafe_view_100: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_33, [12, 64, 128]);  clone_33 = None
        bmm_16: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_99, _unsafe_view_100)
        _unsafe_view_101: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_16, [2, 6, 128, 128]);  bmm_16 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:401, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        zeros_like: i64[128, 128] = torch.ops.aten.zeros_like.default(sub_1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_147: i64[128, 128] = torch.ops.aten.alias.default(zeros_like);  zeros_like = None
        alias_148: i64[128, 128] = torch.ops.aten.alias.default(alias_147);  alias_147 = None
        minimum_1: i64[128, 128] = torch.ops.aten.minimum.default(sub_1, alias_148);  sub_1 = alias_148 = None
        neg: i64[128, 128] = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1648, code: decoder_outputs = self.decoder(
        lt_1: b8[128, 128] = torch.ops.aten.lt.Scalar(neg, 16)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:410, code: torch.log(relative_position.float() / max_exact)
        _to_copy_13: f32[128, 128] = torch.ops.aten._to_copy.default(neg, dtype = torch.float32)
        div_10: f32[128, 128] = torch.ops.aten.div.Tensor(_to_copy_13, 16);  _to_copy_13 = None
        log_1: f32[128, 128] = torch.ops.aten.log.default(div_10);  div_10 = None
        div_11: f32[128, 128] = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
        mul_168: f32[128, 128] = torch.ops.aten.mul.Tensor(div_11, 16);  div_11 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:413, code: ).to(torch.long)
        _to_copy_14: i64[128, 128] = torch.ops.aten._to_copy.default(mul_168, dtype = torch.int64);  mul_168 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:409, code: relative_position_if_large = max_exact + (
        add_70: i64[128, 128] = torch.ops.aten.add.Tensor(_to_copy_14, 16);  _to_copy_14 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:415, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        full_like_1: i64[128, 128] = torch.ops.aten.full_like.default(add_70, 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_149: i64[128, 128] = torch.ops.aten.alias.default(full_like_1);  full_like_1 = None
        alias_150: i64[128, 128] = torch.ops.aten.alias.default(alias_149);  alias_149 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:414, code: relative_position_if_large = torch.min(
        minimum_2: i64[128, 128] = torch.ops.aten.minimum.default(add_70, alias_150);  add_70 = alias_150 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:418, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        where_1: i64[128, 128] = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
        add_71: i64[128, 128] = torch.ops.aten.add.Tensor(where_1, 0);  where_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:434, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        embedding_3: f32[128, 128, 6] = torch.ops.aten.embedding.default(primals_104, add_71);  primals_104 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:435, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_104: f32[6, 128, 128] = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
        unsqueeze_16: f32[1, 6, 128, 128] = torch.ops.aten.unsqueeze.default(permute_104, 0);  permute_104 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:529, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
        add_72: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(unsqueeze_16, mul_162);  unsqueeze_16 = mul_162 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_73: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_101, add_72);  _unsafe_view_101 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_8: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_73, [-1], True)
        sub_21: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_73, amax_8);  add_73 = amax_8 = None
        exp_16: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_9: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_12: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_16, sum_9);  exp_16 = sum_9 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_8: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_12, philox_seed_like, 1572864)
        gt_36: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_8, 0.1);  philox_rand_like_8 = None
        _to_copy_15: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_36, dtype = torch.float32);  gt_36 = None
        mul_169: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_15, div_12);  _to_copy_15 = None
        mul_170: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_169, 1.1111111111111112);  mul_169 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_34: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_170, [2, 6, 128, 128]);  mul_170 = None
        view_104: f32[12, 128, 128] = torch.ops.aten.view.default(expand_34, [12, 128, 128]);  expand_34 = None
        expand_35: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_102, [2, 6, 128, 64])
        clone_34: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        _unsafe_view_102: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_34, [12, 128, 64]);  clone_34 = None
        bmm_17: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_104, _unsafe_view_102)
        _unsafe_view_103: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_17, [2, 6, 128, 64]);  bmm_17 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_105: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_103, [0, 2, 1, 3]);  _unsafe_view_103 = None
        clone_35: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        view_105: f32[2, 128, 384] = torch.ops.aten.view.default(clone_35, [2, -1, 384]);  clone_35 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_106: f32[384, 512] = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        view_106: f32[256, 384] = torch.ops.aten.view.default(view_105, [256, 384]);  view_105 = None
        mm_59: f32[256, 512] = torch.ops.aten.mm.default(view_106, permute_106)
        _unsafe_view_104: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_59, [2, 128, 512]);  mm_59 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        rand_like_27: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_104, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_154: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_27);  rand_like_27 = None
        gt_37: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_154, 0.1);  alias_154 = None
        mul_171: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_37, _unsafe_view_104);  _unsafe_view_104 = None
        mul_172: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_171, 1.1111111111111112);  mul_171 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_74: f32[2, 128, 512] = torch.ops.aten.add.Tensor(mul_165, mul_172);  mul_165 = mul_172 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_27: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_74, 2)
        mean_18: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_75: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        sqrt_18: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_75);  add_75 = None
        reciprocal_26: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
        mul_173: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_74, reciprocal_26)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_174: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_19, mul_173);  mul_173 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_107: f32[512, 384] = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
        view_107: f32[256, 512] = torch.ops.aten.view.default(mul_174, [256, 512]);  mul_174 = None
        mm_60: f32[256, 384] = torch.ops.aten.mm.default(view_107, permute_107);  view_107 = None
        _unsafe_view_105: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_60, [2, 128, 384]);  mm_60 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_108: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_105, [2, -1, 6, 64]);  _unsafe_view_105 = None
        permute_108: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_109: f32[512, 384] = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
        view_109: f32[256, 512] = torch.ops.aten.view.default(mul_160, [256, 512])
        mm_61: f32[256, 384] = torch.ops.aten.mm.default(view_109, permute_109)
        _unsafe_view_106: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_61, [2, 128, 384]);  mm_61 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_110: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_106, [2, -1, 6, 64]);  _unsafe_view_106 = None
        permute_110: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_111: f32[512, 384] = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
        mm_62: f32[256, 384] = torch.ops.aten.mm.default(view_109, permute_111)
        _unsafe_view_107: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_62, [2, 128, 384]);  mm_62 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_112: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_107, [2, -1, 6, 64]);  _unsafe_view_107 = None
        permute_112: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_113: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_110, [0, 1, 3, 2])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_36: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_108, [2, 6, 128, 64]);  permute_108 = None
        clone_36: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        _unsafe_view_108: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_36, [12, 128, 64]);  clone_36 = None
        expand_37: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_113, [2, 6, 64, 128]);  permute_113 = None
        clone_37: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        _unsafe_view_109: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_37, [12, 64, 128]);  clone_37 = None
        bmm_18: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_108, _unsafe_view_109)
        _unsafe_view_110: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_18, [2, 6, 128, 128]);  bmm_18 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:515, code: position_bias = torch.zeros(
        zeros: f32[1, 6, 128, 128] = torch.ops.aten.zeros.default([1, 6, 128, 128], dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        alias_158: f32[1, 6, 128, 128] = torch.ops.aten.alias.default(zeros);  zeros = None
        alias_159: f32[1, 6, 128, 128] = torch.ops.aten.alias.default(alias_158);  alias_158 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:529, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
        add_76: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(alias_159, mul_163);  alias_159 = mul_163 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_77: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_110, add_76);  _unsafe_view_110 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_9: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_77, [-1], True)
        sub_22: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_77, amax_9);  add_77 = amax_9 = None
        exp_17: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_10: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_13: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_17, sum_10);  exp_17 = sum_10 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_9: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_13, philox_seed_like, 1769472)
        gt_38: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_9, 0.1);  philox_rand_like_9 = None
        _to_copy_16: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_38, dtype = torch.float32);  gt_38 = None
        mul_175: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_16, div_13);  _to_copy_16 = None
        mul_176: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_175, 1.1111111111111112);  mul_175 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_38: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_176, [2, 6, 128, 128]);  mul_176 = None
        view_113: f32[12, 128, 128] = torch.ops.aten.view.default(expand_38, [12, 128, 128]);  expand_38 = None
        expand_39: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_112, [2, 6, 128, 64])
        clone_38: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        _unsafe_view_111: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_38, [12, 128, 64]);  clone_38 = None
        bmm_19: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_113, _unsafe_view_111)
        _unsafe_view_112: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_19, [2, 6, 128, 64]);  bmm_19 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_114: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_112, [0, 2, 1, 3]);  _unsafe_view_112 = None
        clone_39: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        view_114: f32[2, 128, 384] = torch.ops.aten.view.default(clone_39, [2, -1, 384]);  clone_39 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_115: f32[384, 512] = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        view_115: f32[256, 384] = torch.ops.aten.view.default(view_114, [256, 384]);  view_114 = None
        mm_63: f32[256, 512] = torch.ops.aten.mm.default(view_115, permute_115)
        _unsafe_view_113: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_63, [2, 128, 512]);  mm_63 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:624, code: layer_output = hidden_states + self.dropout(attention_output[0])
        rand_like_28: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_113, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_163: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_28);  rand_like_28 = None
        gt_39: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_163, 0.1);  alias_163 = None
        mul_177: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_39, _unsafe_view_113);  _unsafe_view_113 = None
        mul_178: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_177, 1.1111111111111112);  mul_177 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:624, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_78: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_74, mul_178);  mul_178 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_28: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_78, 2)
        mean_19: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_79: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        sqrt_19: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_79);  add_79 = None
        reciprocal_27: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
        mul_179: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_78, reciprocal_27)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_180: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_20, mul_179);  mul_179 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_116: f32[512, 1024] = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
        view_116: f32[256, 512] = torch.ops.aten.view.default(mul_180, [256, 512]);  mul_180 = None
        mm_64: f32[256, 1024] = torch.ops.aten.mm.default(view_116, permute_116)
        _unsafe_view_114: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_64, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:35, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_181: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(_unsafe_view_114, 0.5)
        pow_29: f32[2, 128, 1024] = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_114, 3.0)
        mul_182: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
        add_80: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(_unsafe_view_114, mul_182);  _unsafe_view_114 = mul_182 = None
        mul_183: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(add_80, 0.7978845608028654);  add_80 = None
        mul_184: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_183, -2.0);  mul_183 = None
        exp_18: f32[2, 128, 1024] = torch.ops.aten.exp.default(mul_184);  mul_184 = None
        add_81: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(exp_18, 1.0);  exp_18 = None
        reciprocal_28: f32[2, 128, 1024] = torch.ops.aten.reciprocal.default(add_81);  add_81 = None
        mul_185: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(reciprocal_28, 2.0);  reciprocal_28 = None
        sub_23: f32[2, 128, 1024] = torch.ops.aten.sub.Tensor(mul_185, 1.0);  mul_185 = None
        add_82: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(sub_23, 1.0)
        mul_186: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_181, add_82);  mul_181 = add_82 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_117: f32[512, 1024] = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
        mm_65: f32[256, 1024] = torch.ops.aten.mm.default(view_116, permute_117);  view_116 = None
        _unsafe_view_115: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_65, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:309, code: hidden_states = hidden_gelu * hidden_linear
        mul_187: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_186, _unsafe_view_115);  mul_186 = _unsafe_view_115 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:310, code: hidden_states = self.dropout(hidden_states)
        rand_like_29: f32[2, 128, 1024] = torch.ops.aten.rand_like.default(mul_187, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_170: f32[2, 128, 1024] = torch.ops.aten.alias.default(rand_like_29);  rand_like_29 = None
        gt_40: b8[2, 128, 1024] = torch.ops.aten.gt.Scalar(alias_170, 0.1);  alias_170 = None
        mul_188: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(gt_40, mul_187);  mul_187 = None
        mul_189: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_188, 1.1111111111111112);  mul_188 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_118: f32[1024, 512] = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
        view_118: f32[256, 1024] = torch.ops.aten.view.default(mul_189, [256, 1024]);  mul_189 = None
        mm_66: f32[256, 512] = torch.ops.aten.mm.default(view_118, permute_118)
        _unsafe_view_116: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_66, [2, 128, 512]);  mm_66 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        rand_like_30: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_116, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_171: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_30);  rand_like_30 = None
        gt_41: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_171, 0.1);  alias_171 = None
        mul_190: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_41, _unsafe_view_116);  _unsafe_view_116 = None
        mul_191: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_190, 1.1111111111111112);  mul_190 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_83: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_78, mul_191);  mul_191 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_30: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_83, 2)
        mean_20: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_84: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        sqrt_20: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_84);  add_84 = None
        reciprocal_29: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
        mul_192: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_83, reciprocal_29)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_193: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_21, mul_192);  mul_192 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_119: f32[512, 384] = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
        view_119: f32[256, 512] = torch.ops.aten.view.default(mul_193, [256, 512]);  mul_193 = None
        mm_67: f32[256, 384] = torch.ops.aten.mm.default(view_119, permute_119)
        _unsafe_view_117: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_67, [2, 128, 384]);  mm_67 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_120: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_117, [2, -1, 6, 64]);  _unsafe_view_117 = None
        permute_120: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_121: f32[512, 384] = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
        mm_68: f32[256, 384] = torch.ops.aten.mm.default(view_119, permute_121)
        _unsafe_view_118: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_68, [2, 128, 384]);  mm_68 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_122: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_118, [2, -1, 6, 64]);  _unsafe_view_118 = None
        permute_122: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_123: f32[512, 384] = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        mm_69: f32[256, 384] = torch.ops.aten.mm.default(view_119, permute_123);  view_119 = None
        _unsafe_view_119: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_69, [2, 128, 384]);  mm_69 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_124: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_119, [2, -1, 6, 64]);  _unsafe_view_119 = None
        permute_124: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_125: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_122, [0, 1, 3, 2])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_40: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_120, [2, 6, 128, 64]);  permute_120 = None
        clone_40: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        _unsafe_view_120: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_40, [12, 128, 64]);  clone_40 = None
        expand_41: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_125, [2, 6, 64, 128]);  permute_125 = None
        clone_41: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        _unsafe_view_121: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_41, [12, 64, 128]);  clone_41 = None
        bmm_20: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_120, _unsafe_view_121)
        _unsafe_view_122: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_20, [2, 6, 128, 128]);  bmm_20 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_85: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_122, add_72);  _unsafe_view_122 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_10: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_85, [-1], True)
        sub_24: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_85, amax_10);  add_85 = amax_10 = None
        exp_19: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_24);  sub_24 = None
        sum_11: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_14: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_19, sum_11);  exp_19 = sum_11 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_10: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_14, philox_seed_like, 1966080)
        gt_42: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_10, 0.1);  philox_rand_like_10 = None
        _to_copy_17: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_42, dtype = torch.float32);  gt_42 = None
        mul_194: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_17, div_14);  _to_copy_17 = None
        mul_195: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_194, 1.1111111111111112);  mul_194 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_42: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_195, [2, 6, 128, 128]);  mul_195 = None
        view_125: f32[12, 128, 128] = torch.ops.aten.view.default(expand_42, [12, 128, 128]);  expand_42 = None
        expand_43: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_124, [2, 6, 128, 64])
        clone_42: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        _unsafe_view_123: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_42, [12, 128, 64]);  clone_42 = None
        bmm_21: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_125, _unsafe_view_123)
        _unsafe_view_124: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_21, [2, 6, 128, 64]);  bmm_21 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_126: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_124, [0, 2, 1, 3]);  _unsafe_view_124 = None
        clone_43: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_126: f32[2, 128, 384] = torch.ops.aten.view.default(clone_43, [2, -1, 384]);  clone_43 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_127: f32[384, 512] = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
        view_127: f32[256, 384] = torch.ops.aten.view.default(view_126, [256, 384]);  view_126 = None
        mm_70: f32[256, 512] = torch.ops.aten.mm.default(view_127, permute_127)
        _unsafe_view_125: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_70, [2, 128, 512]);  mm_70 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        rand_like_31: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_125, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_178: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_31);  rand_like_31 = None
        gt_43: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_178, 0.1);  alias_178 = None
        mul_196: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_43, _unsafe_view_125);  _unsafe_view_125 = None
        mul_197: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_196, 1.1111111111111112);  mul_196 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_86: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_83, mul_197);  mul_197 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_31: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_86, 2)
        mean_21: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_87: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        sqrt_21: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_87);  add_87 = None
        reciprocal_30: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
        mul_198: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_86, reciprocal_30)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_199: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_22, mul_198);  mul_198 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_128: f32[512, 384] = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
        view_128: f32[256, 512] = torch.ops.aten.view.default(mul_199, [256, 512]);  mul_199 = None
        mm_71: f32[256, 384] = torch.ops.aten.mm.default(view_128, permute_128);  view_128 = None
        _unsafe_view_126: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_71, [2, 128, 384]);  mm_71 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_129: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_126, [2, -1, 6, 64]);  _unsafe_view_126 = None
        permute_129: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_130: f32[512, 384] = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
        mm_72: f32[256, 384] = torch.ops.aten.mm.default(view_109, permute_130)
        _unsafe_view_127: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_72, [2, 128, 384]);  mm_72 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_131: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_127, [2, -1, 6, 64]);  _unsafe_view_127 = None
        permute_131: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_132: f32[512, 384] = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
        mm_73: f32[256, 384] = torch.ops.aten.mm.default(view_109, permute_132)
        _unsafe_view_128: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_73, [2, 128, 384]);  mm_73 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_133: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_128, [2, -1, 6, 64]);  _unsafe_view_128 = None
        permute_133: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_134: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_44: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_129, [2, 6, 128, 64]);  permute_129 = None
        clone_44: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        _unsafe_view_129: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_44, [12, 128, 64]);  clone_44 = None
        expand_45: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_134, [2, 6, 64, 128]);  permute_134 = None
        clone_45: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        _unsafe_view_130: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_45, [12, 64, 128]);  clone_45 = None
        bmm_22: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_129, _unsafe_view_130)
        _unsafe_view_131: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_22, [2, 6, 128, 128]);  bmm_22 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_88: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_131, add_76);  _unsafe_view_131 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_11: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_88, [-1], True)
        sub_25: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_88, amax_11);  add_88 = amax_11 = None
        exp_20: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_12: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_15: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_20, sum_12);  exp_20 = sum_12 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_11: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_15, philox_seed_like, 2162688)
        gt_44: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_11, 0.1);  philox_rand_like_11 = None
        _to_copy_18: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_44, dtype = torch.float32);  gt_44 = None
        mul_200: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_18, div_15);  _to_copy_18 = None
        mul_201: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_200, 1.1111111111111112);  mul_200 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_46: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_201, [2, 6, 128, 128]);  mul_201 = None
        view_134: f32[12, 128, 128] = torch.ops.aten.view.default(expand_46, [12, 128, 128]);  expand_46 = None
        expand_47: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_133, [2, 6, 128, 64])
        clone_46: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        _unsafe_view_132: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_46, [12, 128, 64]);  clone_46 = None
        bmm_23: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_134, _unsafe_view_132)
        _unsafe_view_133: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_23, [2, 6, 128, 64]);  bmm_23 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_135: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_133, [0, 2, 1, 3]);  _unsafe_view_133 = None
        clone_47: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_135: f32[2, 128, 384] = torch.ops.aten.view.default(clone_47, [2, -1, 384]);  clone_47 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_136: f32[384, 512] = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        view_136: f32[256, 384] = torch.ops.aten.view.default(view_135, [256, 384]);  view_135 = None
        mm_74: f32[256, 512] = torch.ops.aten.mm.default(view_136, permute_136)
        _unsafe_view_134: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_74, [2, 128, 512]);  mm_74 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:624, code: layer_output = hidden_states + self.dropout(attention_output[0])
        rand_like_32: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_134, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_185: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_32);  rand_like_32 = None
        gt_45: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_185, 0.1);  alias_185 = None
        mul_202: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_45, _unsafe_view_134);  _unsafe_view_134 = None
        mul_203: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_202, 1.1111111111111112);  mul_202 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:624, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_89: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_86, mul_203);  mul_203 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_32: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_89, 2)
        mean_22: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_90: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        sqrt_22: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_90);  add_90 = None
        reciprocal_31: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
        mul_204: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_89, reciprocal_31)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_205: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_23, mul_204);  mul_204 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_137: f32[512, 1024] = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
        view_137: f32[256, 512] = torch.ops.aten.view.default(mul_205, [256, 512]);  mul_205 = None
        mm_75: f32[256, 1024] = torch.ops.aten.mm.default(view_137, permute_137)
        _unsafe_view_135: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_75, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:35, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_206: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(_unsafe_view_135, 0.5)
        pow_33: f32[2, 128, 1024] = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_135, 3.0)
        mul_207: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
        add_91: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(_unsafe_view_135, mul_207);  _unsafe_view_135 = mul_207 = None
        mul_208: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(add_91, 0.7978845608028654);  add_91 = None
        mul_209: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_208, -2.0);  mul_208 = None
        exp_21: f32[2, 128, 1024] = torch.ops.aten.exp.default(mul_209);  mul_209 = None
        add_92: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(exp_21, 1.0);  exp_21 = None
        reciprocal_32: f32[2, 128, 1024] = torch.ops.aten.reciprocal.default(add_92);  add_92 = None
        mul_210: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(reciprocal_32, 2.0);  reciprocal_32 = None
        sub_26: f32[2, 128, 1024] = torch.ops.aten.sub.Tensor(mul_210, 1.0);  mul_210 = None
        add_93: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(sub_26, 1.0)
        mul_211: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_206, add_93);  mul_206 = add_93 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_138: f32[512, 1024] = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
        mm_76: f32[256, 1024] = torch.ops.aten.mm.default(view_137, permute_138);  view_137 = None
        _unsafe_view_136: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_76, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:309, code: hidden_states = hidden_gelu * hidden_linear
        mul_212: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_211, _unsafe_view_136);  mul_211 = _unsafe_view_136 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:310, code: hidden_states = self.dropout(hidden_states)
        rand_like_33: f32[2, 128, 1024] = torch.ops.aten.rand_like.default(mul_212, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_192: f32[2, 128, 1024] = torch.ops.aten.alias.default(rand_like_33);  rand_like_33 = None
        gt_46: b8[2, 128, 1024] = torch.ops.aten.gt.Scalar(alias_192, 0.1);  alias_192 = None
        mul_213: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(gt_46, mul_212);  mul_212 = None
        mul_214: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_213, 1.1111111111111112);  mul_213 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_139: f32[1024, 512] = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
        view_139: f32[256, 1024] = torch.ops.aten.view.default(mul_214, [256, 1024]);  mul_214 = None
        mm_77: f32[256, 512] = torch.ops.aten.mm.default(view_139, permute_139)
        _unsafe_view_137: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_77, [2, 128, 512]);  mm_77 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        rand_like_34: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_137, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_193: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_34);  rand_like_34 = None
        gt_47: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_193, 0.1);  alias_193 = None
        mul_215: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_47, _unsafe_view_137);  _unsafe_view_137 = None
        mul_216: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_215, 1.1111111111111112);  mul_215 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_94: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_89, mul_216);  mul_216 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_34: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_94, 2)
        mean_23: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_95: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        sqrt_23: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_95);  add_95 = None
        reciprocal_33: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
        mul_217: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_94, reciprocal_33)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_218: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_24, mul_217);  mul_217 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_140: f32[512, 384] = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
        view_140: f32[256, 512] = torch.ops.aten.view.default(mul_218, [256, 512]);  mul_218 = None
        mm_78: f32[256, 384] = torch.ops.aten.mm.default(view_140, permute_140)
        _unsafe_view_138: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_78, [2, 128, 384]);  mm_78 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_141: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_138, [2, -1, 6, 64]);  _unsafe_view_138 = None
        permute_141: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_142: f32[512, 384] = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
        mm_79: f32[256, 384] = torch.ops.aten.mm.default(view_140, permute_142)
        _unsafe_view_139: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_79, [2, 128, 384]);  mm_79 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_143: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_139, [2, -1, 6, 64]);  _unsafe_view_139 = None
        permute_143: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_143, [0, 2, 1, 3]);  view_143 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_144: f32[512, 384] = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
        mm_80: f32[256, 384] = torch.ops.aten.mm.default(view_140, permute_144);  view_140 = None
        _unsafe_view_140: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_80, [2, 128, 384]);  mm_80 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_145: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_140, [2, -1, 6, 64]);  _unsafe_view_140 = None
        permute_145: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_146: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_143, [0, 1, 3, 2])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_48: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_141, [2, 6, 128, 64]);  permute_141 = None
        clone_48: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        _unsafe_view_141: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_48, [12, 128, 64]);  clone_48 = None
        expand_49: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_146, [2, 6, 64, 128]);  permute_146 = None
        clone_49: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        _unsafe_view_142: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_49, [12, 64, 128]);  clone_49 = None
        bmm_24: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_141, _unsafe_view_142)
        _unsafe_view_143: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_24, [2, 6, 128, 128]);  bmm_24 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_96: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_143, add_72);  _unsafe_view_143 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_12: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_96, [-1], True)
        sub_27: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_96, amax_12);  add_96 = amax_12 = None
        exp_22: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_13: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_16: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_22, sum_13);  exp_22 = sum_13 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_12: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_16, philox_seed_like, 2359296)
        gt_48: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_12, 0.1);  philox_rand_like_12 = None
        _to_copy_19: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_48, dtype = torch.float32);  gt_48 = None
        mul_219: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_19, div_16);  _to_copy_19 = None
        mul_220: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_219, 1.1111111111111112);  mul_219 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_50: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_220, [2, 6, 128, 128]);  mul_220 = None
        view_146: f32[12, 128, 128] = torch.ops.aten.view.default(expand_50, [12, 128, 128]);  expand_50 = None
        expand_51: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_145, [2, 6, 128, 64])
        clone_50: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        _unsafe_view_144: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_50, [12, 128, 64]);  clone_50 = None
        bmm_25: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_146, _unsafe_view_144)
        _unsafe_view_145: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_25, [2, 6, 128, 64]);  bmm_25 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_147: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_145, [0, 2, 1, 3]);  _unsafe_view_145 = None
        clone_51: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
        view_147: f32[2, 128, 384] = torch.ops.aten.view.default(clone_51, [2, -1, 384]);  clone_51 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_148: f32[384, 512] = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        view_148: f32[256, 384] = torch.ops.aten.view.default(view_147, [256, 384]);  view_147 = None
        mm_81: f32[256, 512] = torch.ops.aten.mm.default(view_148, permute_148)
        _unsafe_view_146: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_81, [2, 128, 512]);  mm_81 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        rand_like_35: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_146, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_200: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_35);  rand_like_35 = None
        gt_49: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_200, 0.1);  alias_200 = None
        mul_221: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_49, _unsafe_view_146);  _unsafe_view_146 = None
        mul_222: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_221, 1.1111111111111112);  mul_221 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_97: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_94, mul_222);  mul_222 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_35: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_97, 2)
        mean_24: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_98: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        sqrt_24: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_98);  add_98 = None
        reciprocal_34: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
        mul_223: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_97, reciprocal_34)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_224: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_25, mul_223);  mul_223 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_149: f32[512, 384] = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        view_149: f32[256, 512] = torch.ops.aten.view.default(mul_224, [256, 512]);  mul_224 = None
        mm_82: f32[256, 384] = torch.ops.aten.mm.default(view_149, permute_149);  view_149 = None
        _unsafe_view_147: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_82, [2, 128, 384]);  mm_82 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_150: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_147, [2, -1, 6, 64]);  _unsafe_view_147 = None
        permute_150: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_151: f32[512, 384] = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
        mm_83: f32[256, 384] = torch.ops.aten.mm.default(view_109, permute_151)
        _unsafe_view_148: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_83, [2, 128, 384]);  mm_83 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_152: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_148, [2, -1, 6, 64]);  _unsafe_view_148 = None
        permute_152: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_153: f32[512, 384] = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
        mm_84: f32[256, 384] = torch.ops.aten.mm.default(view_109, permute_153)
        _unsafe_view_149: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_84, [2, 128, 384]);  mm_84 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_154: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_149, [2, -1, 6, 64]);  _unsafe_view_149 = None
        permute_154: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_155: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_152, [0, 1, 3, 2])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_52: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_150, [2, 6, 128, 64]);  permute_150 = None
        clone_52: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        _unsafe_view_150: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_52, [12, 128, 64]);  clone_52 = None
        expand_53: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_155, [2, 6, 64, 128]);  permute_155 = None
        clone_53: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        _unsafe_view_151: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_53, [12, 64, 128]);  clone_53 = None
        bmm_26: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_150, _unsafe_view_151)
        _unsafe_view_152: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_26, [2, 6, 128, 128]);  bmm_26 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_99: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_152, add_76);  _unsafe_view_152 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_13: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_99, [-1], True)
        sub_28: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_99, amax_13);  add_99 = amax_13 = None
        exp_23: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_14: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_17: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_23, sum_14);  exp_23 = sum_14 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_13: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_17, philox_seed_like, 2555904)
        gt_50: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_13, 0.1);  philox_rand_like_13 = None
        _to_copy_20: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_50, dtype = torch.float32);  gt_50 = None
        mul_225: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_20, div_17);  _to_copy_20 = None
        mul_226: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_225, 1.1111111111111112);  mul_225 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_54: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_226, [2, 6, 128, 128]);  mul_226 = None
        view_155: f32[12, 128, 128] = torch.ops.aten.view.default(expand_54, [12, 128, 128]);  expand_54 = None
        expand_55: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_154, [2, 6, 128, 64])
        clone_54: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        _unsafe_view_153: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_54, [12, 128, 64]);  clone_54 = None
        bmm_27: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_155, _unsafe_view_153)
        _unsafe_view_154: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_27, [2, 6, 128, 64]);  bmm_27 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_156: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_154, [0, 2, 1, 3]);  _unsafe_view_154 = None
        clone_55: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
        view_156: f32[2, 128, 384] = torch.ops.aten.view.default(clone_55, [2, -1, 384]);  clone_55 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_157: f32[384, 512] = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
        view_157: f32[256, 384] = torch.ops.aten.view.default(view_156, [256, 384]);  view_156 = None
        mm_85: f32[256, 512] = torch.ops.aten.mm.default(view_157, permute_157)
        _unsafe_view_155: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_85, [2, 128, 512]);  mm_85 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:624, code: layer_output = hidden_states + self.dropout(attention_output[0])
        rand_like_36: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_155, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_207: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_36);  rand_like_36 = None
        gt_51: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_207, 0.1);  alias_207 = None
        mul_227: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_51, _unsafe_view_155);  _unsafe_view_155 = None
        mul_228: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_227, 1.1111111111111112);  mul_227 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:624, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_100: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_97, mul_228);  mul_228 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_36: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_100, 2)
        mean_25: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_101: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        sqrt_25: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_101);  add_101 = None
        reciprocal_35: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
        mul_229: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_100, reciprocal_35)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_230: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_26, mul_229);  mul_229 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_158: f32[512, 1024] = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
        view_158: f32[256, 512] = torch.ops.aten.view.default(mul_230, [256, 512]);  mul_230 = None
        mm_86: f32[256, 1024] = torch.ops.aten.mm.default(view_158, permute_158)
        _unsafe_view_156: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_86, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:35, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_231: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(_unsafe_view_156, 0.5)
        pow_37: f32[2, 128, 1024] = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_156, 3.0)
        mul_232: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(pow_37, 0.044715);  pow_37 = None
        add_102: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(_unsafe_view_156, mul_232);  _unsafe_view_156 = mul_232 = None
        mul_233: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(add_102, 0.7978845608028654);  add_102 = None
        mul_234: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_233, -2.0);  mul_233 = None
        exp_24: f32[2, 128, 1024] = torch.ops.aten.exp.default(mul_234);  mul_234 = None
        add_103: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(exp_24, 1.0);  exp_24 = None
        reciprocal_36: f32[2, 128, 1024] = torch.ops.aten.reciprocal.default(add_103);  add_103 = None
        mul_235: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(reciprocal_36, 2.0);  reciprocal_36 = None
        sub_29: f32[2, 128, 1024] = torch.ops.aten.sub.Tensor(mul_235, 1.0);  mul_235 = None
        add_104: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(sub_29, 1.0)
        mul_236: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_231, add_104);  mul_231 = add_104 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_159: f32[512, 1024] = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
        mm_87: f32[256, 1024] = torch.ops.aten.mm.default(view_158, permute_159);  view_158 = None
        _unsafe_view_157: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_87, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:309, code: hidden_states = hidden_gelu * hidden_linear
        mul_237: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_236, _unsafe_view_157);  mul_236 = _unsafe_view_157 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:310, code: hidden_states = self.dropout(hidden_states)
        rand_like_37: f32[2, 128, 1024] = torch.ops.aten.rand_like.default(mul_237, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_214: f32[2, 128, 1024] = torch.ops.aten.alias.default(rand_like_37);  rand_like_37 = None
        gt_52: b8[2, 128, 1024] = torch.ops.aten.gt.Scalar(alias_214, 0.1);  alias_214 = None
        mul_238: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(gt_52, mul_237);  mul_237 = None
        mul_239: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_238, 1.1111111111111112);  mul_238 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_160: f32[1024, 512] = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
        view_160: f32[256, 1024] = torch.ops.aten.view.default(mul_239, [256, 1024]);  mul_239 = None
        mm_88: f32[256, 512] = torch.ops.aten.mm.default(view_160, permute_160)
        _unsafe_view_158: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_88, [2, 128, 512]);  mm_88 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        rand_like_38: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_158, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_215: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_38);  rand_like_38 = None
        gt_53: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_215, 0.1);  alias_215 = None
        mul_240: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_53, _unsafe_view_158);  _unsafe_view_158 = None
        mul_241: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_240, 1.1111111111111112);  mul_240 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_105: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_100, mul_241);  mul_241 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_38: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_105, 2)
        mean_26: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_106: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        sqrt_26: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_106);  add_106 = None
        reciprocal_37: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
        mul_242: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_105, reciprocal_37)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_243: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_27, mul_242);  mul_242 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_161: f32[512, 384] = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
        view_161: f32[256, 512] = torch.ops.aten.view.default(mul_243, [256, 512]);  mul_243 = None
        mm_89: f32[256, 384] = torch.ops.aten.mm.default(view_161, permute_161)
        _unsafe_view_159: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_89, [2, 128, 384]);  mm_89 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_162: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_159, [2, -1, 6, 64]);  _unsafe_view_159 = None
        permute_162: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_163: f32[512, 384] = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
        mm_90: f32[256, 384] = torch.ops.aten.mm.default(view_161, permute_163)
        _unsafe_view_160: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_90, [2, 128, 384]);  mm_90 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_164: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_160, [2, -1, 6, 64]);  _unsafe_view_160 = None
        permute_164: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_165: f32[512, 384] = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
        mm_91: f32[256, 384] = torch.ops.aten.mm.default(view_161, permute_165);  view_161 = None
        _unsafe_view_161: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_91, [2, 128, 384]);  mm_91 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_166: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_161, [2, -1, 6, 64]);  _unsafe_view_161 = None
        permute_166: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_167: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_164, [0, 1, 3, 2])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_56: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_162, [2, 6, 128, 64]);  permute_162 = None
        clone_56: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        _unsafe_view_162: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_56, [12, 128, 64]);  clone_56 = None
        expand_57: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_167, [2, 6, 64, 128]);  permute_167 = None
        clone_57: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        _unsafe_view_163: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_57, [12, 64, 128]);  clone_57 = None
        bmm_28: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_162, _unsafe_view_163)
        _unsafe_view_164: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_28, [2, 6, 128, 128]);  bmm_28 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_107: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_164, add_72);  _unsafe_view_164 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_14: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_107, [-1], True)
        sub_30: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_107, amax_14);  add_107 = amax_14 = None
        exp_25: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        sum_15: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_18: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_25, sum_15);  exp_25 = sum_15 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_14: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_18, philox_seed_like, 2752512)
        gt_54: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_14, 0.1);  philox_rand_like_14 = None
        _to_copy_21: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_54, dtype = torch.float32);  gt_54 = None
        mul_244: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_21, div_18);  _to_copy_21 = None
        mul_245: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_244, 1.1111111111111112);  mul_244 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_58: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_245, [2, 6, 128, 128]);  mul_245 = None
        view_167: f32[12, 128, 128] = torch.ops.aten.view.default(expand_58, [12, 128, 128]);  expand_58 = None
        expand_59: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_166, [2, 6, 128, 64])
        clone_58: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        _unsafe_view_165: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_58, [12, 128, 64]);  clone_58 = None
        bmm_29: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_167, _unsafe_view_165)
        _unsafe_view_166: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_29, [2, 6, 128, 64]);  bmm_29 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_168: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_166, [0, 2, 1, 3]);  _unsafe_view_166 = None
        clone_59: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
        view_168: f32[2, 128, 384] = torch.ops.aten.view.default(clone_59, [2, -1, 384]);  clone_59 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_169: f32[384, 512] = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
        view_169: f32[256, 384] = torch.ops.aten.view.default(view_168, [256, 384]);  view_168 = None
        mm_92: f32[256, 512] = torch.ops.aten.mm.default(view_169, permute_169)
        _unsafe_view_167: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_92, [2, 128, 512]);  mm_92 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        rand_like_39: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_167, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_222: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_39);  rand_like_39 = None
        gt_55: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_222, 0.1);  alias_222 = None
        mul_246: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_55, _unsafe_view_167);  _unsafe_view_167 = None
        mul_247: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_246, 1.1111111111111112);  mul_246 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_108: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_105, mul_247);  mul_247 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_39: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_108, 2)
        mean_27: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_109: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        sqrt_27: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_109);  add_109 = None
        reciprocal_38: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
        mul_248: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_108, reciprocal_38)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_249: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_28, mul_248);  mul_248 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_170: f32[512, 384] = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
        view_170: f32[256, 512] = torch.ops.aten.view.default(mul_249, [256, 512]);  mul_249 = None
        mm_93: f32[256, 384] = torch.ops.aten.mm.default(view_170, permute_170);  view_170 = None
        _unsafe_view_168: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_93, [2, 128, 384]);  mm_93 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_171: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_168, [2, -1, 6, 64]);  _unsafe_view_168 = None
        permute_171: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_172: f32[512, 384] = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
        mm_94: f32[256, 384] = torch.ops.aten.mm.default(view_109, permute_172)
        _unsafe_view_169: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_94, [2, 128, 384]);  mm_94 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_173: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_169, [2, -1, 6, 64]);  _unsafe_view_169 = None
        permute_173: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_174: f32[512, 384] = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        mm_95: f32[256, 384] = torch.ops.aten.mm.default(view_109, permute_174)
        _unsafe_view_170: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_95, [2, 128, 384]);  mm_95 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_175: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_170, [2, -1, 6, 64]);  _unsafe_view_170 = None
        permute_175: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_176: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_173, [0, 1, 3, 2])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_60: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_171, [2, 6, 128, 64]);  permute_171 = None
        clone_60: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        _unsafe_view_171: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_60, [12, 128, 64]);  clone_60 = None
        expand_61: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_176, [2, 6, 64, 128]);  permute_176 = None
        clone_61: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        _unsafe_view_172: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_61, [12, 64, 128]);  clone_61 = None
        bmm_30: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_171, _unsafe_view_172)
        _unsafe_view_173: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_30, [2, 6, 128, 128]);  bmm_30 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_110: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_173, add_76);  _unsafe_view_173 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_15: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_110, [-1], True)
        sub_31: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_110, amax_15);  add_110 = amax_15 = None
        exp_26: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_16: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_19: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_26, sum_16);  exp_26 = sum_16 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_15: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_19, philox_seed_like, 2949120)
        gt_56: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_15, 0.1);  philox_rand_like_15 = None
        _to_copy_22: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_56, dtype = torch.float32);  gt_56 = None
        mul_250: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_22, div_19);  _to_copy_22 = None
        mul_251: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_250, 1.1111111111111112);  mul_250 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_62: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_251, [2, 6, 128, 128]);  mul_251 = None
        view_176: f32[12, 128, 128] = torch.ops.aten.view.default(expand_62, [12, 128, 128]);  expand_62 = None
        expand_63: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_175, [2, 6, 128, 64])
        clone_62: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        _unsafe_view_174: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_62, [12, 128, 64]);  clone_62 = None
        bmm_31: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_176, _unsafe_view_174)
        _unsafe_view_175: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_31, [2, 6, 128, 64]);  bmm_31 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_177: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_175, [0, 2, 1, 3]);  _unsafe_view_175 = None
        clone_63: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_177: f32[2, 128, 384] = torch.ops.aten.view.default(clone_63, [2, -1, 384]);  clone_63 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_178: f32[384, 512] = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
        view_178: f32[256, 384] = torch.ops.aten.view.default(view_177, [256, 384]);  view_177 = None
        mm_96: f32[256, 512] = torch.ops.aten.mm.default(view_178, permute_178)
        _unsafe_view_176: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_96, [2, 128, 512]);  mm_96 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:624, code: layer_output = hidden_states + self.dropout(attention_output[0])
        rand_like_40: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_176, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_229: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_40);  rand_like_40 = None
        gt_57: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_229, 0.1);  alias_229 = None
        mul_252: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_57, _unsafe_view_176);  _unsafe_view_176 = None
        mul_253: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_252, 1.1111111111111112);  mul_252 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:624, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_111: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_108, mul_253);  mul_253 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_40: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_111, 2)
        mean_28: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_112: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        sqrt_28: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_112);  add_112 = None
        reciprocal_39: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        mul_254: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_111, reciprocal_39)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_255: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_29, mul_254);  mul_254 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_179: f32[512, 1024] = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
        view_179: f32[256, 512] = torch.ops.aten.view.default(mul_255, [256, 512]);  mul_255 = None
        mm_97: f32[256, 1024] = torch.ops.aten.mm.default(view_179, permute_179)
        _unsafe_view_177: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_97, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:35, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_256: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(_unsafe_view_177, 0.5)
        pow_41: f32[2, 128, 1024] = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_177, 3.0)
        mul_257: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(pow_41, 0.044715);  pow_41 = None
        add_113: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(_unsafe_view_177, mul_257);  _unsafe_view_177 = mul_257 = None
        mul_258: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(add_113, 0.7978845608028654);  add_113 = None
        mul_259: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_258, -2.0);  mul_258 = None
        exp_27: f32[2, 128, 1024] = torch.ops.aten.exp.default(mul_259);  mul_259 = None
        add_114: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(exp_27, 1.0);  exp_27 = None
        reciprocal_40: f32[2, 128, 1024] = torch.ops.aten.reciprocal.default(add_114);  add_114 = None
        mul_260: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(reciprocal_40, 2.0);  reciprocal_40 = None
        sub_32: f32[2, 128, 1024] = torch.ops.aten.sub.Tensor(mul_260, 1.0);  mul_260 = None
        add_115: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(sub_32, 1.0)
        mul_261: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_256, add_115);  mul_256 = add_115 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_180: f32[512, 1024] = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
        mm_98: f32[256, 1024] = torch.ops.aten.mm.default(view_179, permute_180);  view_179 = None
        _unsafe_view_178: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_98, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:309, code: hidden_states = hidden_gelu * hidden_linear
        mul_262: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_261, _unsafe_view_178);  mul_261 = _unsafe_view_178 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:310, code: hidden_states = self.dropout(hidden_states)
        rand_like_41: f32[2, 128, 1024] = torch.ops.aten.rand_like.default(mul_262, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_236: f32[2, 128, 1024] = torch.ops.aten.alias.default(rand_like_41);  rand_like_41 = None
        gt_58: b8[2, 128, 1024] = torch.ops.aten.gt.Scalar(alias_236, 0.1);  alias_236 = None
        mul_263: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(gt_58, mul_262);  mul_262 = None
        mul_264: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_263, 1.1111111111111112);  mul_263 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_181: f32[1024, 512] = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
        view_181: f32[256, 1024] = torch.ops.aten.view.default(mul_264, [256, 1024]);  mul_264 = None
        mm_99: f32[256, 512] = torch.ops.aten.mm.default(view_181, permute_181)
        _unsafe_view_179: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_99, [2, 128, 512]);  mm_99 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        rand_like_42: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_179, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_237: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_42);  rand_like_42 = None
        gt_59: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_237, 0.1);  alias_237 = None
        mul_265: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_59, _unsafe_view_179);  _unsafe_view_179 = None
        mul_266: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_265, 1.1111111111111112);  mul_265 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_116: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_111, mul_266);  mul_266 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_42: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_116, 2)
        mean_29: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_117: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        sqrt_29: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_117);  add_117 = None
        reciprocal_41: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        mul_267: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_116, reciprocal_41)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_268: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_30, mul_267);  mul_267 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_182: f32[512, 384] = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
        view_182: f32[256, 512] = torch.ops.aten.view.default(mul_268, [256, 512]);  mul_268 = None
        mm_100: f32[256, 384] = torch.ops.aten.mm.default(view_182, permute_182)
        _unsafe_view_180: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_100, [2, 128, 384]);  mm_100 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_183: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_180, [2, -1, 6, 64]);  _unsafe_view_180 = None
        permute_183: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_184: f32[512, 384] = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
        mm_101: f32[256, 384] = torch.ops.aten.mm.default(view_182, permute_184)
        _unsafe_view_181: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_101, [2, 128, 384]);  mm_101 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_185: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_181, [2, -1, 6, 64]);  _unsafe_view_181 = None
        permute_185: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_186: f32[512, 384] = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
        mm_102: f32[256, 384] = torch.ops.aten.mm.default(view_182, permute_186);  view_182 = None
        _unsafe_view_182: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_102, [2, 128, 384]);  mm_102 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_187: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_182, [2, -1, 6, 64]);  _unsafe_view_182 = None
        permute_187: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_188: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_185, [0, 1, 3, 2])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_64: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_183, [2, 6, 128, 64]);  permute_183 = None
        clone_64: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        _unsafe_view_183: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_64, [12, 128, 64]);  clone_64 = None
        expand_65: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_188, [2, 6, 64, 128]);  permute_188 = None
        clone_65: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        _unsafe_view_184: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_65, [12, 64, 128]);  clone_65 = None
        bmm_32: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_183, _unsafe_view_184)
        _unsafe_view_185: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_32, [2, 6, 128, 128]);  bmm_32 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_118: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_185, add_72);  _unsafe_view_185 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_16: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_118, [-1], True)
        sub_33: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_118, amax_16);  add_118 = amax_16 = None
        exp_28: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_33);  sub_33 = None
        sum_17: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
        div_20: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_28, sum_17);  exp_28 = sum_17 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_16: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_20, philox_seed_like, 3145728)
        gt_60: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_16, 0.1);  philox_rand_like_16 = None
        _to_copy_23: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_60, dtype = torch.float32);  gt_60 = None
        mul_269: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_23, div_20);  _to_copy_23 = None
        mul_270: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_269, 1.1111111111111112);  mul_269 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_66: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_270, [2, 6, 128, 128]);  mul_270 = None
        view_188: f32[12, 128, 128] = torch.ops.aten.view.default(expand_66, [12, 128, 128]);  expand_66 = None
        expand_67: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_187, [2, 6, 128, 64])
        clone_66: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        _unsafe_view_186: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_66, [12, 128, 64]);  clone_66 = None
        bmm_33: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_188, _unsafe_view_186)
        _unsafe_view_187: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_33, [2, 6, 128, 64]);  bmm_33 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_189: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_187, [0, 2, 1, 3]);  _unsafe_view_187 = None
        clone_67: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_189: f32[2, 128, 384] = torch.ops.aten.view.default(clone_67, [2, -1, 384]);  clone_67 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_190: f32[384, 512] = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
        view_190: f32[256, 384] = torch.ops.aten.view.default(view_189, [256, 384]);  view_189 = None
        mm_103: f32[256, 512] = torch.ops.aten.mm.default(view_190, permute_190)
        _unsafe_view_188: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_103, [2, 128, 512]);  mm_103 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        rand_like_43: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_188, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_244: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_43);  rand_like_43 = None
        gt_61: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_244, 0.1);  alias_244 = None
        mul_271: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_61, _unsafe_view_188);  _unsafe_view_188 = None
        mul_272: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_271, 1.1111111111111112);  mul_271 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_119: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_116, mul_272);  mul_272 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_43: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_119, 2)
        mean_30: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_120: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        sqrt_30: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_120);  add_120 = None
        reciprocal_42: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        mul_273: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_119, reciprocal_42)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_274: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_31, mul_273);  mul_273 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_191: f32[512, 384] = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
        view_191: f32[256, 512] = torch.ops.aten.view.default(mul_274, [256, 512]);  mul_274 = None
        mm_104: f32[256, 384] = torch.ops.aten.mm.default(view_191, permute_191);  view_191 = None
        _unsafe_view_189: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_104, [2, 128, 384]);  mm_104 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_192: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_189, [2, -1, 6, 64]);  _unsafe_view_189 = None
        permute_192: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_193: f32[512, 384] = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
        mm_105: f32[256, 384] = torch.ops.aten.mm.default(view_109, permute_193)
        _unsafe_view_190: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_105, [2, 128, 384]);  mm_105 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_194: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_190, [2, -1, 6, 64]);  _unsafe_view_190 = None
        permute_194: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_195: f32[512, 384] = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
        mm_106: f32[256, 384] = torch.ops.aten.mm.default(view_109, permute_195)
        _unsafe_view_191: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_106, [2, 128, 384]);  mm_106 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_196: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_191, [2, -1, 6, 64]);  _unsafe_view_191 = None
        permute_196: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_196, [0, 2, 1, 3]);  view_196 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_197: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_194, [0, 1, 3, 2])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_68: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_192, [2, 6, 128, 64]);  permute_192 = None
        clone_68: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        _unsafe_view_192: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_68, [12, 128, 64]);  clone_68 = None
        expand_69: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_197, [2, 6, 64, 128]);  permute_197 = None
        clone_69: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        _unsafe_view_193: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_69, [12, 64, 128]);  clone_69 = None
        bmm_34: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_192, _unsafe_view_193)
        _unsafe_view_194: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_34, [2, 6, 128, 128]);  bmm_34 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_121: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_194, add_76);  _unsafe_view_194 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_17: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_121, [-1], True)
        sub_34: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_121, amax_17);  add_121 = amax_17 = None
        exp_29: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_18: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_29, [-1], True)
        div_21: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_29, sum_18);  exp_29 = sum_18 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_17: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_21, philox_seed_like, 3342336)
        gt_62: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_17, 0.1);  philox_rand_like_17 = None
        _to_copy_24: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_62, dtype = torch.float32);  gt_62 = None
        mul_275: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_24, div_21);  _to_copy_24 = None
        mul_276: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_275, 1.1111111111111112);  mul_275 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_70: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_276, [2, 6, 128, 128]);  mul_276 = None
        view_197: f32[12, 128, 128] = torch.ops.aten.view.default(expand_70, [12, 128, 128]);  expand_70 = None
        expand_71: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_196, [2, 6, 128, 64])
        clone_70: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        _unsafe_view_195: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_70, [12, 128, 64]);  clone_70 = None
        bmm_35: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_197, _unsafe_view_195)
        _unsafe_view_196: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_35, [2, 6, 128, 64]);  bmm_35 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_198: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_196, [0, 2, 1, 3]);  _unsafe_view_196 = None
        clone_71: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
        view_198: f32[2, 128, 384] = torch.ops.aten.view.default(clone_71, [2, -1, 384]);  clone_71 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_199: f32[384, 512] = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
        view_199: f32[256, 384] = torch.ops.aten.view.default(view_198, [256, 384]);  view_198 = None
        mm_107: f32[256, 512] = torch.ops.aten.mm.default(view_199, permute_199)
        _unsafe_view_197: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_107, [2, 128, 512]);  mm_107 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:624, code: layer_output = hidden_states + self.dropout(attention_output[0])
        rand_like_44: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_197, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_251: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_44);  rand_like_44 = None
        gt_63: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_251, 0.1);  alias_251 = None
        mul_277: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_63, _unsafe_view_197);  _unsafe_view_197 = None
        mul_278: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_277, 1.1111111111111112);  mul_277 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:624, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_122: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_119, mul_278);  mul_278 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_44: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_122, 2)
        mean_31: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_123: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        sqrt_31: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_123);  add_123 = None
        reciprocal_43: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_279: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_122, reciprocal_43)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_280: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_32, mul_279);  mul_279 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_200: f32[512, 1024] = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
        view_200: f32[256, 512] = torch.ops.aten.view.default(mul_280, [256, 512]);  mul_280 = None
        mm_108: f32[256, 1024] = torch.ops.aten.mm.default(view_200, permute_200)
        _unsafe_view_198: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_108, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:35, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_281: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(_unsafe_view_198, 0.5)
        pow_45: f32[2, 128, 1024] = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_198, 3.0)
        mul_282: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(pow_45, 0.044715);  pow_45 = None
        add_124: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(_unsafe_view_198, mul_282);  _unsafe_view_198 = mul_282 = None
        mul_283: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(add_124, 0.7978845608028654);  add_124 = None
        mul_284: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_283, -2.0);  mul_283 = None
        exp_30: f32[2, 128, 1024] = torch.ops.aten.exp.default(mul_284);  mul_284 = None
        add_125: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(exp_30, 1.0);  exp_30 = None
        reciprocal_44: f32[2, 128, 1024] = torch.ops.aten.reciprocal.default(add_125);  add_125 = None
        mul_285: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(reciprocal_44, 2.0);  reciprocal_44 = None
        sub_35: f32[2, 128, 1024] = torch.ops.aten.sub.Tensor(mul_285, 1.0);  mul_285 = None
        add_126: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(sub_35, 1.0)
        mul_286: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_281, add_126);  mul_281 = add_126 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_201: f32[512, 1024] = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
        mm_109: f32[256, 1024] = torch.ops.aten.mm.default(view_200, permute_201);  view_200 = None
        _unsafe_view_199: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_109, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:309, code: hidden_states = hidden_gelu * hidden_linear
        mul_287: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_286, _unsafe_view_199);  mul_286 = _unsafe_view_199 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:310, code: hidden_states = self.dropout(hidden_states)
        rand_like_45: f32[2, 128, 1024] = torch.ops.aten.rand_like.default(mul_287, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_258: f32[2, 128, 1024] = torch.ops.aten.alias.default(rand_like_45);  rand_like_45 = None
        gt_64: b8[2, 128, 1024] = torch.ops.aten.gt.Scalar(alias_258, 0.1);  alias_258 = None
        mul_288: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(gt_64, mul_287);  mul_287 = None
        mul_289: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_288, 1.1111111111111112);  mul_288 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_202: f32[1024, 512] = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
        view_202: f32[256, 1024] = torch.ops.aten.view.default(mul_289, [256, 1024]);  mul_289 = None
        mm_110: f32[256, 512] = torch.ops.aten.mm.default(view_202, permute_202)
        _unsafe_view_200: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_110, [2, 128, 512]);  mm_110 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        rand_like_46: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_200, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_259: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_46);  rand_like_46 = None
        gt_65: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_259, 0.1);  alias_259 = None
        mul_290: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_65, _unsafe_view_200);  _unsafe_view_200 = None
        mul_291: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_290, 1.1111111111111112);  mul_290 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_127: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_122, mul_291);  mul_291 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_46: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_127, 2)
        mean_32: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_128: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None
        sqrt_32: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_128);  add_128 = None
        reciprocal_45: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_292: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_127, reciprocal_45)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_293: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_33, mul_292);  mul_292 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_203: f32[512, 384] = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
        view_203: f32[256, 512] = torch.ops.aten.view.default(mul_293, [256, 512]);  mul_293 = None
        mm_111: f32[256, 384] = torch.ops.aten.mm.default(view_203, permute_203)
        _unsafe_view_201: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_111, [2, 128, 384]);  mm_111 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_204: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_201, [2, -1, 6, 64]);  _unsafe_view_201 = None
        permute_204: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_205: f32[512, 384] = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
        mm_112: f32[256, 384] = torch.ops.aten.mm.default(view_203, permute_205)
        _unsafe_view_202: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_112, [2, 128, 384]);  mm_112 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_206: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_202, [2, -1, 6, 64]);  _unsafe_view_202 = None
        permute_206: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_207: f32[512, 384] = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
        mm_113: f32[256, 384] = torch.ops.aten.mm.default(view_203, permute_207);  view_203 = None
        _unsafe_view_203: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_113, [2, 128, 384]);  mm_113 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_208: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_203, [2, -1, 6, 64]);  _unsafe_view_203 = None
        permute_208: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_209: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_206, [0, 1, 3, 2])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_72: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_204, [2, 6, 128, 64]);  permute_204 = None
        clone_72: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        _unsafe_view_204: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_72, [12, 128, 64]);  clone_72 = None
        expand_73: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_209, [2, 6, 64, 128]);  permute_209 = None
        clone_73: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        _unsafe_view_205: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_73, [12, 64, 128]);  clone_73 = None
        bmm_36: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_204, _unsafe_view_205)
        _unsafe_view_206: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_36, [2, 6, 128, 128]);  bmm_36 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_129: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_206, add_72);  _unsafe_view_206 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_18: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_129, [-1], True)
        sub_36: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_129, amax_18);  add_129 = amax_18 = None
        exp_31: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        sum_19: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_31, [-1], True)
        div_22: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_31, sum_19);  exp_31 = sum_19 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_18: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_22, philox_seed_like, 3538944)
        gt_66: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_18, 0.1);  philox_rand_like_18 = None
        _to_copy_25: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_66, dtype = torch.float32);  gt_66 = None
        mul_294: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_25, div_22);  _to_copy_25 = None
        mul_295: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_294, 1.1111111111111112);  mul_294 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_74: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_295, [2, 6, 128, 128]);  mul_295 = None
        view_209: f32[12, 128, 128] = torch.ops.aten.view.default(expand_74, [12, 128, 128]);  expand_74 = None
        expand_75: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_208, [2, 6, 128, 64])
        clone_74: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        _unsafe_view_207: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_74, [12, 128, 64]);  clone_74 = None
        bmm_37: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_209, _unsafe_view_207)
        _unsafe_view_208: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_37, [2, 6, 128, 64]);  bmm_37 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_210: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_208, [0, 2, 1, 3]);  _unsafe_view_208 = None
        clone_75: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_210: f32[2, 128, 384] = torch.ops.aten.view.default(clone_75, [2, -1, 384]);  clone_75 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_211: f32[384, 512] = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
        view_211: f32[256, 384] = torch.ops.aten.view.default(view_210, [256, 384]);  view_210 = None
        mm_114: f32[256, 512] = torch.ops.aten.mm.default(view_211, permute_211)
        _unsafe_view_209: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_114, [2, 128, 512]);  mm_114 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        rand_like_47: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_209, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_266: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_47);  rand_like_47 = None
        gt_67: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_266, 0.1);  alias_266 = None
        mul_296: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_67, _unsafe_view_209);  _unsafe_view_209 = None
        mul_297: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_296, 1.1111111111111112);  mul_296 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_130: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_127, mul_297);  mul_297 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_47: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_130, 2)
        mean_33: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_131: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None
        sqrt_33: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_131);  add_131 = None
        reciprocal_46: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_298: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_130, reciprocal_46)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_299: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_34, mul_298);  mul_298 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_212: f32[512, 384] = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
        view_212: f32[256, 512] = torch.ops.aten.view.default(mul_299, [256, 512]);  mul_299 = None
        mm_115: f32[256, 384] = torch.ops.aten.mm.default(view_212, permute_212);  view_212 = None
        _unsafe_view_210: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_115, [2, 128, 384]);  mm_115 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_213: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_210, [2, -1, 6, 64]);  _unsafe_view_210 = None
        permute_213: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_214: f32[512, 384] = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
        mm_116: f32[256, 384] = torch.ops.aten.mm.default(view_109, permute_214)
        _unsafe_view_211: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_116, [2, 128, 384]);  mm_116 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_215: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_211, [2, -1, 6, 64]);  _unsafe_view_211 = None
        permute_215: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_216: f32[512, 384] = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
        mm_117: f32[256, 384] = torch.ops.aten.mm.default(view_109, permute_216)
        _unsafe_view_212: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_117, [2, 128, 384]);  mm_117 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_217: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_212, [2, -1, 6, 64]);  _unsafe_view_212 = None
        permute_217: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_217, [0, 2, 1, 3]);  view_217 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_218: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_215, [0, 1, 3, 2])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_76: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_213, [2, 6, 128, 64]);  permute_213 = None
        clone_76: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        _unsafe_view_213: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_76, [12, 128, 64]);  clone_76 = None
        expand_77: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_218, [2, 6, 64, 128]);  permute_218 = None
        clone_77: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        _unsafe_view_214: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_77, [12, 64, 128]);  clone_77 = None
        bmm_38: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_213, _unsafe_view_214)
        _unsafe_view_215: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_38, [2, 6, 128, 128]);  bmm_38 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_132: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_215, add_76);  _unsafe_view_215 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_19: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_132, [-1], True)
        sub_37: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_132, amax_19);  add_132 = amax_19 = None
        exp_32: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_20: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
        div_23: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_32, sum_20);  exp_32 = sum_20 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_19: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_23, philox_seed_like, 3735552)
        gt_68: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_19, 0.1);  philox_rand_like_19 = None
        _to_copy_26: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_68, dtype = torch.float32);  gt_68 = None
        mul_300: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_26, div_23);  _to_copy_26 = None
        mul_301: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_300, 1.1111111111111112);  mul_300 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_78: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_301, [2, 6, 128, 128]);  mul_301 = None
        view_218: f32[12, 128, 128] = torch.ops.aten.view.default(expand_78, [12, 128, 128]);  expand_78 = None
        expand_79: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_217, [2, 6, 128, 64])
        clone_78: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        _unsafe_view_216: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_78, [12, 128, 64]);  clone_78 = None
        bmm_39: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_218, _unsafe_view_216)
        _unsafe_view_217: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_39, [2, 6, 128, 64]);  bmm_39 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_219: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_217, [0, 2, 1, 3]);  _unsafe_view_217 = None
        clone_79: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
        view_219: f32[2, 128, 384] = torch.ops.aten.view.default(clone_79, [2, -1, 384]);  clone_79 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_220: f32[384, 512] = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
        view_220: f32[256, 384] = torch.ops.aten.view.default(view_219, [256, 384]);  view_219 = None
        mm_118: f32[256, 512] = torch.ops.aten.mm.default(view_220, permute_220)
        _unsafe_view_218: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_118, [2, 128, 512]);  mm_118 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:624, code: layer_output = hidden_states + self.dropout(attention_output[0])
        rand_like_48: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_218, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_273: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_48);  rand_like_48 = None
        gt_69: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_273, 0.1);  alias_273 = None
        mul_302: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_69, _unsafe_view_218);  _unsafe_view_218 = None
        mul_303: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_302, 1.1111111111111112);  mul_302 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:624, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_133: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_130, mul_303);  mul_303 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_48: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_133, 2)
        mean_34: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_134: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None
        sqrt_34: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_47: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_304: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_133, reciprocal_47)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_305: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_35, mul_304);  mul_304 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_221: f32[512, 1024] = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
        view_221: f32[256, 512] = torch.ops.aten.view.default(mul_305, [256, 512]);  mul_305 = None
        mm_119: f32[256, 1024] = torch.ops.aten.mm.default(view_221, permute_221)
        _unsafe_view_219: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_119, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:35, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_306: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(_unsafe_view_219, 0.5)
        pow_49: f32[2, 128, 1024] = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_219, 3.0)
        mul_307: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(pow_49, 0.044715);  pow_49 = None
        add_135: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(_unsafe_view_219, mul_307);  _unsafe_view_219 = mul_307 = None
        mul_308: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(add_135, 0.7978845608028654);  add_135 = None
        mul_309: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_308, -2.0);  mul_308 = None
        exp_33: f32[2, 128, 1024] = torch.ops.aten.exp.default(mul_309);  mul_309 = None
        add_136: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(exp_33, 1.0);  exp_33 = None
        reciprocal_48: f32[2, 128, 1024] = torch.ops.aten.reciprocal.default(add_136);  add_136 = None
        mul_310: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(reciprocal_48, 2.0);  reciprocal_48 = None
        sub_38: f32[2, 128, 1024] = torch.ops.aten.sub.Tensor(mul_310, 1.0);  mul_310 = None
        add_137: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(sub_38, 1.0)
        mul_311: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_306, add_137);  mul_306 = add_137 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_222: f32[512, 1024] = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
        mm_120: f32[256, 1024] = torch.ops.aten.mm.default(view_221, permute_222);  view_221 = None
        _unsafe_view_220: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_120, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:309, code: hidden_states = hidden_gelu * hidden_linear
        mul_312: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_311, _unsafe_view_220);  mul_311 = _unsafe_view_220 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:310, code: hidden_states = self.dropout(hidden_states)
        rand_like_49: f32[2, 128, 1024] = torch.ops.aten.rand_like.default(mul_312, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_280: f32[2, 128, 1024] = torch.ops.aten.alias.default(rand_like_49);  rand_like_49 = None
        gt_70: b8[2, 128, 1024] = torch.ops.aten.gt.Scalar(alias_280, 0.1);  alias_280 = None
        mul_313: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(gt_70, mul_312);  mul_312 = None
        mul_314: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_313, 1.1111111111111112);  mul_313 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_223: f32[1024, 512] = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
        view_223: f32[256, 1024] = torch.ops.aten.view.default(mul_314, [256, 1024]);  mul_314 = None
        mm_121: f32[256, 512] = torch.ops.aten.mm.default(view_223, permute_223)
        _unsafe_view_221: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_121, [2, 128, 512]);  mm_121 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        rand_like_50: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_221, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_281: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_50);  rand_like_50 = None
        gt_71: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_281, 0.1);  alias_281 = None
        mul_315: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_71, _unsafe_view_221);  _unsafe_view_221 = None
        mul_316: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_315, 1.1111111111111112);  mul_315 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_138: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_133, mul_316);  mul_316 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_50: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_138, 2)
        mean_35: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_139: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None
        sqrt_35: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_139);  add_139 = None
        reciprocal_49: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_317: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_138, reciprocal_49)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_318: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_36, mul_317);  mul_317 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_224: f32[512, 384] = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
        view_224: f32[256, 512] = torch.ops.aten.view.default(mul_318, [256, 512]);  mul_318 = None
        mm_122: f32[256, 384] = torch.ops.aten.mm.default(view_224, permute_224)
        _unsafe_view_222: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_122, [2, 128, 384]);  mm_122 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_225: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_222, [2, -1, 6, 64]);  _unsafe_view_222 = None
        permute_225: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_226: f32[512, 384] = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
        mm_123: f32[256, 384] = torch.ops.aten.mm.default(view_224, permute_226)
        _unsafe_view_223: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_123, [2, 128, 384]);  mm_123 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_227: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_223, [2, -1, 6, 64]);  _unsafe_view_223 = None
        permute_227: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_228: f32[512, 384] = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
        mm_124: f32[256, 384] = torch.ops.aten.mm.default(view_224, permute_228);  view_224 = None
        _unsafe_view_224: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_124, [2, 128, 384]);  mm_124 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_229: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_224, [2, -1, 6, 64]);  _unsafe_view_224 = None
        permute_229: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_230: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_227, [0, 1, 3, 2])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_80: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_225, [2, 6, 128, 64]);  permute_225 = None
        clone_80: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        _unsafe_view_225: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_80, [12, 128, 64]);  clone_80 = None
        expand_81: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_230, [2, 6, 64, 128]);  permute_230 = None
        clone_81: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        _unsafe_view_226: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_81, [12, 64, 128]);  clone_81 = None
        bmm_40: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_225, _unsafe_view_226)
        _unsafe_view_227: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_40, [2, 6, 128, 128]);  bmm_40 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_140: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_227, add_72);  _unsafe_view_227 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_20: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_140, [-1], True)
        sub_39: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_140, amax_20);  add_140 = amax_20 = None
        exp_34: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        sum_21: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
        div_24: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_34, sum_21);  exp_34 = sum_21 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_20: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_24, philox_seed_like, 3932160)
        gt_72: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_20, 0.1);  philox_rand_like_20 = None
        _to_copy_27: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_72, dtype = torch.float32);  gt_72 = None
        mul_319: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_27, div_24);  _to_copy_27 = None
        mul_320: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_319, 1.1111111111111112);  mul_319 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_82: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_320, [2, 6, 128, 128]);  mul_320 = None
        view_230: f32[12, 128, 128] = torch.ops.aten.view.default(expand_82, [12, 128, 128]);  expand_82 = None
        expand_83: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_229, [2, 6, 128, 64])
        clone_82: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        _unsafe_view_228: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_82, [12, 128, 64]);  clone_82 = None
        bmm_41: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_230, _unsafe_view_228)
        _unsafe_view_229: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_41, [2, 6, 128, 64]);  bmm_41 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_231: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_229, [0, 2, 1, 3]);  _unsafe_view_229 = None
        clone_83: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
        view_231: f32[2, 128, 384] = torch.ops.aten.view.default(clone_83, [2, -1, 384]);  clone_83 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_232: f32[384, 512] = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
        view_232: f32[256, 384] = torch.ops.aten.view.default(view_231, [256, 384]);  view_231 = None
        mm_125: f32[256, 512] = torch.ops.aten.mm.default(view_232, permute_232)
        _unsafe_view_230: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_125, [2, 128, 512]);  mm_125 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        rand_like_51: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_230, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_288: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_51);  rand_like_51 = None
        gt_73: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_288, 0.1);  alias_288 = None
        mul_321: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_73, _unsafe_view_230);  _unsafe_view_230 = None
        mul_322: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_321, 1.1111111111111112);  mul_321 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_141: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_138, mul_322);  mul_322 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_51: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_141, 2)
        mean_36: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_142: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None
        sqrt_36: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_142);  add_142 = None
        reciprocal_50: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_323: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_141, reciprocal_50)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_324: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_37, mul_323);  mul_323 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_233: f32[512, 384] = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
        view_233: f32[256, 512] = torch.ops.aten.view.default(mul_324, [256, 512]);  mul_324 = None
        mm_126: f32[256, 384] = torch.ops.aten.mm.default(view_233, permute_233);  view_233 = None
        _unsafe_view_231: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_126, [2, 128, 384]);  mm_126 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_234: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_231, [2, -1, 6, 64]);  _unsafe_view_231 = None
        permute_234: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_235: f32[512, 384] = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
        mm_127: f32[256, 384] = torch.ops.aten.mm.default(view_109, permute_235)
        _unsafe_view_232: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_127, [2, 128, 384]);  mm_127 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_236: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_232, [2, -1, 6, 64]);  _unsafe_view_232 = None
        permute_236: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_237: f32[512, 384] = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
        mm_128: f32[256, 384] = torch.ops.aten.mm.default(view_109, permute_237)
        _unsafe_view_233: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_128, [2, 128, 384]);  mm_128 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_238: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_233, [2, -1, 6, 64]);  _unsafe_view_233 = None
        permute_238: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_239: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_236, [0, 1, 3, 2])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_84: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_234, [2, 6, 128, 64]);  permute_234 = None
        clone_84: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        _unsafe_view_234: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_84, [12, 128, 64]);  clone_84 = None
        expand_85: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_239, [2, 6, 64, 128]);  permute_239 = None
        clone_85: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        _unsafe_view_235: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_85, [12, 64, 128]);  clone_85 = None
        bmm_42: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_234, _unsafe_view_235)
        _unsafe_view_236: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_42, [2, 6, 128, 128]);  bmm_42 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_143: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_236, add_76);  _unsafe_view_236 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_21: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_143, [-1], True)
        sub_40: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_143, amax_21);  add_143 = amax_21 = None
        exp_35: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        sum_22: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_35, [-1], True)
        div_25: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_35, sum_22);  exp_35 = sum_22 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_21: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_25, philox_seed_like, 4128768)
        gt_74: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_21, 0.1);  philox_rand_like_21 = None
        _to_copy_28: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_74, dtype = torch.float32);  gt_74 = None
        mul_325: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_28, div_25);  _to_copy_28 = None
        mul_326: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_325, 1.1111111111111112);  mul_325 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_86: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_326, [2, 6, 128, 128]);  mul_326 = None
        view_239: f32[12, 128, 128] = torch.ops.aten.view.default(expand_86, [12, 128, 128]);  expand_86 = None
        expand_87: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_238, [2, 6, 128, 64])
        clone_86: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        _unsafe_view_237: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_86, [12, 128, 64]);  clone_86 = None
        bmm_43: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_239, _unsafe_view_237)
        _unsafe_view_238: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_43, [2, 6, 128, 64]);  bmm_43 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_240: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_238, [0, 2, 1, 3]);  _unsafe_view_238 = None
        clone_87: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
        view_240: f32[2, 128, 384] = torch.ops.aten.view.default(clone_87, [2, -1, 384]);  clone_87 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_241: f32[384, 512] = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
        view_241: f32[256, 384] = torch.ops.aten.view.default(view_240, [256, 384]);  view_240 = None
        mm_129: f32[256, 512] = torch.ops.aten.mm.default(view_241, permute_241)
        _unsafe_view_239: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_129, [2, 128, 512]);  mm_129 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:624, code: layer_output = hidden_states + self.dropout(attention_output[0])
        rand_like_52: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_239, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_295: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_52);  rand_like_52 = None
        gt_75: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_295, 0.1);  alias_295 = None
        mul_327: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_75, _unsafe_view_239);  _unsafe_view_239 = None
        mul_328: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_327, 1.1111111111111112);  mul_327 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:624, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_144: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_141, mul_328);  mul_328 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_52: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_144, 2)
        mean_37: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_145: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None
        sqrt_37: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_145);  add_145 = None
        reciprocal_51: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_329: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_144, reciprocal_51)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_330: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_38, mul_329);  mul_329 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_242: f32[512, 1024] = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
        view_242: f32[256, 512] = torch.ops.aten.view.default(mul_330, [256, 512]);  mul_330 = None
        mm_130: f32[256, 1024] = torch.ops.aten.mm.default(view_242, permute_242)
        _unsafe_view_240: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_130, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:35, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_331: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(_unsafe_view_240, 0.5)
        pow_53: f32[2, 128, 1024] = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_240, 3.0)
        mul_332: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(pow_53, 0.044715);  pow_53 = None
        add_146: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(_unsafe_view_240, mul_332);  _unsafe_view_240 = mul_332 = None
        mul_333: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(add_146, 0.7978845608028654);  add_146 = None
        mul_334: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_333, -2.0);  mul_333 = None
        exp_36: f32[2, 128, 1024] = torch.ops.aten.exp.default(mul_334);  mul_334 = None
        add_147: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(exp_36, 1.0);  exp_36 = None
        reciprocal_52: f32[2, 128, 1024] = torch.ops.aten.reciprocal.default(add_147);  add_147 = None
        mul_335: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(reciprocal_52, 2.0);  reciprocal_52 = None
        sub_41: f32[2, 128, 1024] = torch.ops.aten.sub.Tensor(mul_335, 1.0);  mul_335 = None
        add_148: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(sub_41, 1.0)
        mul_336: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_331, add_148);  mul_331 = add_148 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_243: f32[512, 1024] = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
        mm_131: f32[256, 1024] = torch.ops.aten.mm.default(view_242, permute_243);  view_242 = None
        _unsafe_view_241: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_131, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:309, code: hidden_states = hidden_gelu * hidden_linear
        mul_337: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_336, _unsafe_view_241);  mul_336 = _unsafe_view_241 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:310, code: hidden_states = self.dropout(hidden_states)
        rand_like_53: f32[2, 128, 1024] = torch.ops.aten.rand_like.default(mul_337, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_302: f32[2, 128, 1024] = torch.ops.aten.alias.default(rand_like_53);  rand_like_53 = None
        gt_76: b8[2, 128, 1024] = torch.ops.aten.gt.Scalar(alias_302, 0.1);  alias_302 = None
        mul_338: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(gt_76, mul_337);  mul_337 = None
        mul_339: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_338, 1.1111111111111112);  mul_338 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_244: f32[1024, 512] = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
        view_244: f32[256, 1024] = torch.ops.aten.view.default(mul_339, [256, 1024]);  mul_339 = None
        mm_132: f32[256, 512] = torch.ops.aten.mm.default(view_244, permute_244)
        _unsafe_view_242: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_132, [2, 128, 512]);  mm_132 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        rand_like_54: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_242, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_303: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_54);  rand_like_54 = None
        gt_77: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_303, 0.1);  alias_303 = None
        mul_340: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_77, _unsafe_view_242);  _unsafe_view_242 = None
        mul_341: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_340, 1.1111111111111112);  mul_340 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_149: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_144, mul_341);  mul_341 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_54: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_149, 2)
        mean_38: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_150: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None
        sqrt_38: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_150);  add_150 = None
        reciprocal_53: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_342: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_149, reciprocal_53)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_343: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_39, mul_342);  mul_342 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_245: f32[512, 384] = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
        view_245: f32[256, 512] = torch.ops.aten.view.default(mul_343, [256, 512]);  mul_343 = None
        mm_133: f32[256, 384] = torch.ops.aten.mm.default(view_245, permute_245)
        _unsafe_view_243: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_133, [2, 128, 384]);  mm_133 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_246: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_243, [2, -1, 6, 64]);  _unsafe_view_243 = None
        permute_246: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_247: f32[512, 384] = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
        mm_134: f32[256, 384] = torch.ops.aten.mm.default(view_245, permute_247)
        _unsafe_view_244: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_134, [2, 128, 384]);  mm_134 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_248: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_244, [2, -1, 6, 64]);  _unsafe_view_244 = None
        permute_248: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_249: f32[512, 384] = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
        mm_135: f32[256, 384] = torch.ops.aten.mm.default(view_245, permute_249);  view_245 = None
        _unsafe_view_245: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_135, [2, 128, 384]);  mm_135 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_250: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_245, [2, -1, 6, 64]);  _unsafe_view_245 = None
        permute_250: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_251: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_248, [0, 1, 3, 2])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_88: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_246, [2, 6, 128, 64]);  permute_246 = None
        clone_88: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        _unsafe_view_246: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_88, [12, 128, 64]);  clone_88 = None
        expand_89: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_251, [2, 6, 64, 128]);  permute_251 = None
        clone_89: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        _unsafe_view_247: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_89, [12, 64, 128]);  clone_89 = None
        bmm_44: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_246, _unsafe_view_247)
        _unsafe_view_248: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_44, [2, 6, 128, 128]);  bmm_44 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_151: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_248, add_72);  _unsafe_view_248 = add_72 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_22: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_151, [-1], True)
        sub_42: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_151, amax_22);  add_151 = amax_22 = None
        exp_37: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_42);  sub_42 = None
        sum_23: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_37, [-1], True)
        div_26: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_37, sum_23);  exp_37 = sum_23 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_22: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_26, philox_seed_like, 4325376)
        gt_78: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_22, 0.1);  philox_rand_like_22 = None
        _to_copy_29: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_78, dtype = torch.float32);  gt_78 = None
        mul_344: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_29, div_26);  _to_copy_29 = None
        mul_345: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_344, 1.1111111111111112);  mul_344 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_90: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_345, [2, 6, 128, 128]);  mul_345 = None
        view_251: f32[12, 128, 128] = torch.ops.aten.view.default(expand_90, [12, 128, 128]);  expand_90 = None
        expand_91: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_250, [2, 6, 128, 64])
        clone_90: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        _unsafe_view_249: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_90, [12, 128, 64]);  clone_90 = None
        bmm_45: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_251, _unsafe_view_249)
        _unsafe_view_250: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_45, [2, 6, 128, 64]);  bmm_45 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_252: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_250, [0, 2, 1, 3]);  _unsafe_view_250 = None
        clone_91: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None
        view_252: f32[2, 128, 384] = torch.ops.aten.view.default(clone_91, [2, -1, 384]);  clone_91 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_253: f32[384, 512] = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
        view_253: f32[256, 384] = torch.ops.aten.view.default(view_252, [256, 384]);  view_252 = None
        mm_136: f32[256, 512] = torch.ops.aten.mm.default(view_253, permute_253)
        _unsafe_view_251: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_136, [2, 128, 512]);  mm_136 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        rand_like_55: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_251, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_310: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_55);  rand_like_55 = None
        gt_79: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_310, 0.1);  alias_310 = None
        mul_346: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_79, _unsafe_view_251);  _unsafe_view_251 = None
        mul_347: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_346, 1.1111111111111112);  mul_346 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:588, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_152: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_149, mul_347);  mul_347 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_55: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_152, 2)
        mean_39: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_153: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None
        sqrt_39: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_153);  add_153 = None
        reciprocal_54: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_348: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_152, reciprocal_54)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_349: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_40, mul_348);  mul_348 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_254: f32[512, 384] = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
        view_254: f32[256, 512] = torch.ops.aten.view.default(mul_349, [256, 512]);  mul_349 = None
        mm_137: f32[256, 384] = torch.ops.aten.mm.default(view_254, permute_254);  view_254 = None
        _unsafe_view_252: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_137, [2, 128, 384]);  mm_137 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_255: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_252, [2, -1, 6, 64]);  _unsafe_view_252 = None
        permute_255: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_256: f32[512, 384] = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
        mm_138: f32[256, 384] = torch.ops.aten.mm.default(view_109, permute_256)
        _unsafe_view_253: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_138, [2, 128, 384]);  mm_138 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_257: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_253, [2, -1, 6, 64]);  _unsafe_view_253 = None
        permute_257: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_258: f32[512, 384] = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
        mm_139: f32[256, 384] = torch.ops.aten.mm.default(view_109, permute_258);  view_109 = None
        _unsafe_view_254: f32[2, 128, 384] = torch.ops.aten._unsafe_view.default(mm_139, [2, 128, 384]);  mm_139 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:470, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        view_259: f32[2, 128, 6, 64] = torch.ops.aten.view.default(_unsafe_view_254, [2, -1, 6, 64]);  _unsafe_view_254 = None
        permute_259: f32[2, 6, 128, 64] = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:510, code: query_states, key_states.transpose(3, 2)
        permute_260: f32[2, 6, 64, 128] = torch.ops.aten.permute.default(permute_257, [0, 1, 3, 2])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        expand_92: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_255, [2, 6, 128, 64]);  permute_255 = None
        clone_92: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        _unsafe_view_255: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_92, [12, 128, 64]);  clone_92 = None
        expand_93: f32[2, 6, 64, 128] = torch.ops.aten.expand.default(permute_260, [2, 6, 64, 128]);  permute_260 = None
        clone_93: f32[2, 6, 64, 128] = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        _unsafe_view_256: f32[12, 64, 128] = torch.ops.aten._unsafe_view.default(clone_93, [12, 64, 128]);  clone_93 = None
        bmm_46: f32[12, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_255, _unsafe_view_256)
        _unsafe_view_257: f32[2, 6, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_46, [2, 6, 128, 128]);  bmm_46 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: scores += position_bias_masked
        add_154: f32[2, 6, 128, 128] = torch.ops.aten.add.Tensor(_unsafe_view_257, add_76);  _unsafe_view_257 = add_76 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:539, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_23: f32[2, 6, 128, 1] = torch.ops.aten.amax.default(add_154, [-1], True)
        sub_43: f32[2, 6, 128, 128] = torch.ops.aten.sub.Tensor(add_154, amax_23);  add_154 = amax_23 = None
        exp_38: f32[2, 6, 128, 128] = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_24: f32[2, 6, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_38, [-1], True)
        div_27: f32[2, 6, 128, 128] = torch.ops.aten.div.Tensor(exp_38, sum_24);  exp_38 = sum_24 = None
        
        # Gradient addition node due to multiple use of tensor around:, File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        philox_rand_like_23: f32[2, 6, 128, 128] = torch.ops.prims.philox_rand_like.default(div_27, philox_seed_like, 4521984)
        gt_80: b8[2, 6, 128, 128] = torch.ops.aten.gt.Scalar(philox_rand_like_23, 0.1);  philox_rand_like_23 = None
        _to_copy_30: f32[2, 6, 128, 128] = torch.ops.aten._to_copy.default(gt_80, dtype = torch.float32);  gt_80 = None
        mul_350: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(_to_copy_30, div_27);  _to_copy_30 = None
        mul_351: f32[2, 6, 128, 128] = torch.ops.aten.mul.Tensor(mul_350, 1.1111111111111112);  mul_350 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        expand_94: f32[2, 6, 128, 128] = torch.ops.aten.expand.default(mul_351, [2, 6, 128, 128]);  mul_351 = None
        view_260: f32[12, 128, 128] = torch.ops.aten.view.default(expand_94, [12, 128, 128]);  expand_94 = None
        expand_95: f32[2, 6, 128, 64] = torch.ops.aten.expand.default(permute_259, [2, 6, 128, 64])
        clone_94: f32[2, 6, 128, 64] = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        _unsafe_view_258: f32[12, 128, 64] = torch.ops.aten._unsafe_view.default(clone_94, [12, 128, 64]);  clone_94 = None
        bmm_47: f32[12, 128, 64] = torch.ops.aten.bmm.default(view_260, _unsafe_view_258)
        _unsafe_view_259: f32[2, 6, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_47, [2, 6, 128, 64]);  bmm_47 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:474, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        permute_261: f32[2, 128, 6, 64] = torch.ops.aten.permute.default(_unsafe_view_259, [0, 2, 1, 3]);  _unsafe_view_259 = None
        clone_95: f32[2, 128, 6, 64] = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
        view_261: f32[2, 128, 384] = torch.ops.aten.view.default(clone_95, [2, -1, 384]);  clone_95 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_262: f32[384, 512] = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
        view_262: f32[256, 384] = torch.ops.aten.view.default(view_261, [256, 384]);  view_261 = None
        mm_140: f32[256, 512] = torch.ops.aten.mm.default(view_262, permute_262)
        _unsafe_view_260: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_140, [2, 128, 512]);  mm_140 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:624, code: layer_output = hidden_states + self.dropout(attention_output[0])
        rand_like_56: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_260, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_317: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_56);  rand_like_56 = None
        gt_81: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_317, 0.1);  alias_317 = None
        mul_352: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_81, _unsafe_view_260);  _unsafe_view_260 = None
        mul_353: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_352, 1.1111111111111112);  mul_352 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:624, code: layer_output = hidden_states + self.dropout(attention_output[0])
        add_155: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_152, mul_353);  mul_353 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_56: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_155, 2)
        mean_40: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_156: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None
        sqrt_40: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_156);  add_156 = None
        reciprocal_55: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_354: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_155, reciprocal_55)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_355: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_41, mul_354);  mul_354 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_263: f32[512, 1024] = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
        view_263: f32[256, 512] = torch.ops.aten.view.default(mul_355, [256, 512]);  mul_355 = None
        mm_141: f32[256, 1024] = torch.ops.aten.mm.default(view_263, permute_263)
        _unsafe_view_261: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_141, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:35, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_356: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(_unsafe_view_261, 0.5)
        pow_57: f32[2, 128, 1024] = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_261, 3.0)
        mul_357: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(pow_57, 0.044715);  pow_57 = None
        add_157: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(_unsafe_view_261, mul_357);  _unsafe_view_261 = mul_357 = None
        mul_358: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(add_157, 0.7978845608028654);  add_157 = None
        mul_359: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_358, -2.0);  mul_358 = None
        exp_39: f32[2, 128, 1024] = torch.ops.aten.exp.default(mul_359);  mul_359 = None
        add_158: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(exp_39, 1.0);  exp_39 = None
        reciprocal_56: f32[2, 128, 1024] = torch.ops.aten.reciprocal.default(add_158);  add_158 = None
        mul_360: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(reciprocal_56, 2.0);  reciprocal_56 = None
        sub_44: f32[2, 128, 1024] = torch.ops.aten.sub.Tensor(mul_360, 1.0);  mul_360 = None
        add_159: f32[2, 128, 1024] = torch.ops.aten.add.Tensor(sub_44, 1.0)
        mul_361: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_356, add_159);  mul_356 = add_159 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_264: f32[512, 1024] = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
        mm_142: f32[256, 1024] = torch.ops.aten.mm.default(view_263, permute_264);  view_263 = None
        _unsafe_view_262: f32[2, 128, 1024] = torch.ops.aten._unsafe_view.default(mm_142, [2, 128, 1024])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:309, code: hidden_states = hidden_gelu * hidden_linear
        mul_362: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_361, _unsafe_view_262);  mul_361 = _unsafe_view_262 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:310, code: hidden_states = self.dropout(hidden_states)
        rand_like_57: f32[2, 128, 1024] = torch.ops.aten.rand_like.default(mul_362, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_324: f32[2, 128, 1024] = torch.ops.aten.alias.default(rand_like_57);  rand_like_57 = None
        gt_82: b8[2, 128, 1024] = torch.ops.aten.gt.Scalar(alias_324, 0.1);  alias_324 = None
        mul_363: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(gt_82, mul_362);  mul_362 = None
        mul_364: f32[2, 128, 1024] = torch.ops.aten.mul.Tensor(mul_363, 1.1111111111111112);  mul_363 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_265: f32[1024, 512] = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
        view_265: f32[256, 1024] = torch.ops.aten.view.default(mul_364, [256, 1024]);  mul_364 = None
        mm_143: f32[256, 512] = torch.ops.aten.mm.default(view_265, permute_265)
        _unsafe_view_263: f32[2, 128, 512] = torch.ops.aten._unsafe_view.default(mm_143, [2, 128, 512]);  mm_143 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        rand_like_58: f32[2, 128, 512] = torch.ops.aten.rand_like.default(_unsafe_view_263, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_325: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_58);  rand_like_58 = None
        gt_83: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_325, 0.1);  alias_325 = None
        mul_365: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_83, _unsafe_view_263);  _unsafe_view_263 = None
        mul_366: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_365, 1.1111111111111112);  mul_365 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:329, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        add_160: f32[2, 128, 512] = torch.ops.aten.add.Tensor(add_155, mul_366);  mul_366 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_58: f32[2, 128, 512] = torch.ops.aten.pow.Tensor_Scalar(add_160, 2)
        mean_41: f32[2, 128, 1] = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:256, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_161: f32[2, 128, 1] = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None
        sqrt_41: f32[2, 128, 1] = torch.ops.aten.sqrt.default(add_161);  add_161 = None
        reciprocal_57: f32[2, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_367: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(add_160, reciprocal_57)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:262, code: return self.weight * hidden_states
        mul_368: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(primals_42, mul_367);  mul_367 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1083, code: hidden_states = self.dropout(hidden_states)
        rand_like_59: f32[2, 128, 512] = torch.ops.aten.rand_like.default(mul_368, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_329: f32[2, 128, 512] = torch.ops.aten.alias.default(rand_like_59);  rand_like_59 = None
        gt_84: b8[2, 128, 512] = torch.ops.aten.gt.Scalar(alias_329, 0.1);  alias_329 = None
        mul_369: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(gt_84, mul_368);  mul_368 = None
        mul_370: f32[2, 128, 512] = torch.ops.aten.mul.Tensor(mul_369, 1.1111111111111112);  mul_369 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1676, code: lm_logits = self.lm_head(sequence_output)
        permute_266: f32[512, 250112] = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
        view_266: f32[256, 512] = torch.ops.aten.view.default(mul_370, [256, 512]);  mul_370 = None
        mm_144: f32[256, 250112] = torch.ops.aten.mm.default(view_266, permute_266)
        _unsafe_view_264: f32[2, 128, 250112] = torch.ops.aten._unsafe_view.default(mm_144, [2, 128, 250112]);  mm_144 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1681, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        view_267: f32[256, 250112] = torch.ops.aten.view.default(_unsafe_view_264, [-1, 250112])
        view_268: i64[256] = torch.ops.aten.view.default(primals_193, [-1]);  primals_193 = None
        amax_24: f32[256, 1] = torch.ops.aten.amax.default(view_267, [1], True)
        sub_45: f32[256, 250112] = torch.ops.aten.sub.Tensor(view_267, amax_24);  view_267 = amax_24 = None
        exp_40: f32[256, 250112] = torch.ops.aten.exp.default(sub_45)
        sum_25: f32[256, 1] = torch.ops.aten.sum.dim_IntList(exp_40, [1], True);  exp_40 = None
        log_2: f32[256, 1] = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_46: f32[256, 250112] = torch.ops.aten.sub.Tensor(sub_45, log_2);  sub_45 = log_2 = None
        unsqueeze_17: i64[256, 1] = torch.ops.aten.unsqueeze.default(view_268, 1);  view_268 = None
        gather: f32[256, 1] = torch.ops.aten.gather.default(sub_46, 1, unsqueeze_17)
        squeeze: f32[256] = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_1: f32[256] = torch.ops.aten.neg.default(squeeze);  squeeze = None
        mean_42: f32[] = torch.ops.aten.mean.default(neg_1);  neg_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1676, code: lm_logits = self.lm_head(sequence_output)
        permute_269: f32[250112, 512] = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_273: f32[512, 1024] = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_277: f32[1024, 512] = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_281: f32[1024, 512] = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_285: f32[512, 384] = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_288: f32[12, 128, 128] = torch.ops.aten.permute.default(view_260, [0, 2, 1]);  view_260 = None
        permute_289: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_258, [0, 2, 1]);  _unsafe_view_258 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_290: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_255, [0, 2, 1]);  _unsafe_view_255 = None
        permute_291: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_256, [0, 2, 1]);  _unsafe_view_256 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_296: f32[384, 512] = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_301: f32[384, 512] = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_306: f32[384, 512] = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_310: f32[512, 384] = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_313: f32[12, 128, 128] = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
        permute_314: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_249, [0, 2, 1]);  _unsafe_view_249 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_315: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_246, [0, 2, 1]);  _unsafe_view_246 = None
        permute_316: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_247, [0, 2, 1]);  _unsafe_view_247 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_321: f32[384, 512] = torch.ops.aten.permute.default(permute_249, [1, 0]);  permute_249 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_326: f32[384, 512] = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_331: f32[384, 512] = torch.ops.aten.permute.default(permute_245, [1, 0]);  permute_245 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_335: f32[512, 1024] = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_339: f32[1024, 512] = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_343: f32[1024, 512] = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_347: f32[512, 384] = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_350: f32[12, 128, 128] = torch.ops.aten.permute.default(view_239, [0, 2, 1]);  view_239 = None
        permute_351: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_237, [0, 2, 1]);  _unsafe_view_237 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_352: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_234, [0, 2, 1]);  _unsafe_view_234 = None
        permute_353: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_235, [0, 2, 1]);  _unsafe_view_235 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_358: f32[384, 512] = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_363: f32[384, 512] = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_368: f32[384, 512] = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_372: f32[512, 384] = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_375: f32[12, 128, 128] = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
        permute_376: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_228, [0, 2, 1]);  _unsafe_view_228 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_377: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_225, [0, 2, 1]);  _unsafe_view_225 = None
        permute_378: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_226, [0, 2, 1]);  _unsafe_view_226 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_383: f32[384, 512] = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_388: f32[384, 512] = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_393: f32[384, 512] = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_397: f32[512, 1024] = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_401: f32[1024, 512] = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_405: f32[1024, 512] = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_409: f32[512, 384] = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_412: f32[12, 128, 128] = torch.ops.aten.permute.default(view_218, [0, 2, 1]);  view_218 = None
        permute_413: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_216, [0, 2, 1]);  _unsafe_view_216 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_414: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_213, [0, 2, 1]);  _unsafe_view_213 = None
        permute_415: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_214, [0, 2, 1]);  _unsafe_view_214 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_420: f32[384, 512] = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_425: f32[384, 512] = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_430: f32[384, 512] = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_434: f32[512, 384] = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_437: f32[12, 128, 128] = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
        permute_438: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_207, [0, 2, 1]);  _unsafe_view_207 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_439: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_204, [0, 2, 1]);  _unsafe_view_204 = None
        permute_440: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_205, [0, 2, 1]);  _unsafe_view_205 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_445: f32[384, 512] = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_450: f32[384, 512] = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_455: f32[384, 512] = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_459: f32[512, 1024] = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_463: f32[1024, 512] = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_467: f32[1024, 512] = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_471: f32[512, 384] = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_474: f32[12, 128, 128] = torch.ops.aten.permute.default(view_197, [0, 2, 1]);  view_197 = None
        permute_475: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_195, [0, 2, 1]);  _unsafe_view_195 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_476: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_192, [0, 2, 1]);  _unsafe_view_192 = None
        permute_477: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_193, [0, 2, 1]);  _unsafe_view_193 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_482: f32[384, 512] = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_487: f32[384, 512] = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_492: f32[384, 512] = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_496: f32[512, 384] = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_499: f32[12, 128, 128] = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        permute_500: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_186, [0, 2, 1]);  _unsafe_view_186 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_501: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_183, [0, 2, 1]);  _unsafe_view_183 = None
        permute_502: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_184, [0, 2, 1]);  _unsafe_view_184 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_507: f32[384, 512] = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_512: f32[384, 512] = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_517: f32[384, 512] = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_521: f32[512, 1024] = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_525: f32[1024, 512] = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_529: f32[1024, 512] = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_533: f32[512, 384] = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_536: f32[12, 128, 128] = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None
        permute_537: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_174, [0, 2, 1]);  _unsafe_view_174 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_538: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_171, [0, 2, 1]);  _unsafe_view_171 = None
        permute_539: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_172, [0, 2, 1]);  _unsafe_view_172 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_544: f32[384, 512] = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_549: f32[384, 512] = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_554: f32[384, 512] = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_558: f32[512, 384] = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_561: f32[12, 128, 128] = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
        permute_562: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_165, [0, 2, 1]);  _unsafe_view_165 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_563: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_162, [0, 2, 1]);  _unsafe_view_162 = None
        permute_564: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_163, [0, 2, 1]);  _unsafe_view_163 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_569: f32[384, 512] = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_574: f32[384, 512] = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_579: f32[384, 512] = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_583: f32[512, 1024] = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_587: f32[1024, 512] = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_591: f32[1024, 512] = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_595: f32[512, 384] = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_598: f32[12, 128, 128] = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
        permute_599: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_153, [0, 2, 1]);  _unsafe_view_153 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_600: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_150, [0, 2, 1]);  _unsafe_view_150 = None
        permute_601: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_151, [0, 2, 1]);  _unsafe_view_151 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_606: f32[384, 512] = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_611: f32[384, 512] = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_616: f32[384, 512] = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_620: f32[512, 384] = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_623: f32[12, 128, 128] = torch.ops.aten.permute.default(view_146, [0, 2, 1]);  view_146 = None
        permute_624: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_144, [0, 2, 1]);  _unsafe_view_144 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_625: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_141, [0, 2, 1]);  _unsafe_view_141 = None
        permute_626: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_142, [0, 2, 1]);  _unsafe_view_142 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_631: f32[384, 512] = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_636: f32[384, 512] = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_641: f32[384, 512] = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_645: f32[512, 1024] = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_649: f32[1024, 512] = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_653: f32[1024, 512] = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_657: f32[512, 384] = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_660: f32[12, 128, 128] = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
        permute_661: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_132, [0, 2, 1]);  _unsafe_view_132 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_662: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_129, [0, 2, 1]);  _unsafe_view_129 = None
        permute_663: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_130, [0, 2, 1]);  _unsafe_view_130 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_668: f32[384, 512] = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_673: f32[384, 512] = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_678: f32[384, 512] = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_682: f32[512, 384] = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_685: f32[12, 128, 128] = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
        permute_686: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_123, [0, 2, 1]);  _unsafe_view_123 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_687: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_120, [0, 2, 1]);  _unsafe_view_120 = None
        permute_688: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_121, [0, 2, 1]);  _unsafe_view_121 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_693: f32[384, 512] = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_698: f32[384, 512] = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_703: f32[384, 512] = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_707: f32[512, 1024] = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_711: f32[1024, 512] = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_715: f32[1024, 512] = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_719: f32[512, 384] = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_722: f32[12, 128, 128] = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
        permute_723: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_111, [0, 2, 1]);  _unsafe_view_111 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_724: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_108, [0, 2, 1]);  _unsafe_view_108 = None
        permute_725: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_109, [0, 2, 1]);  _unsafe_view_109 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_730: f32[384, 512] = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:485, code: hidden_states = shape(proj_layer(key_value_states))
        permute_735: f32[384, 512] = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_740: f32[384, 512] = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_744: f32[512, 384] = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_747: f32[12, 128, 128] = torch.ops.aten.permute.default(view_104, [0, 2, 1]);  view_104 = None
        permute_748: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_102, [0, 2, 1]);  _unsafe_view_102 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:434, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        view_560: i64[16384] = torch.ops.aten.view.default(add_71, [16384]);  add_71 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_750: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_99, [0, 2, 1]);  _unsafe_view_99 = None
        permute_751: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_100, [0, 2, 1]);  _unsafe_view_100 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_756: f32[384, 512] = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_761: f32[384, 512] = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_766: f32[384, 512] = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        view_572: i64[256] = torch.ops.aten.view.default(view_97, [256]);  view_97 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_770: f32[512, 1024] = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_774: f32[1024, 512] = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_778: f32[1024, 512] = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_782: f32[512, 384] = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_785: f32[12, 128, 128] = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_786: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_90, [0, 2, 1]);  _unsafe_view_90 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_787: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_87, [0, 2, 1]);  _unsafe_view_87 = None
        permute_788: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_88, [0, 2, 1]);  _unsafe_view_88 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_793: f32[384, 512] = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_798: f32[384, 512] = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_803: f32[384, 512] = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_807: f32[512, 1024] = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_811: f32[1024, 512] = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_815: f32[1024, 512] = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_819: f32[512, 384] = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_822: f32[12, 128, 128] = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
        permute_823: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_78, [0, 2, 1]);  _unsafe_view_78 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_824: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_75, [0, 2, 1]);  _unsafe_view_75 = None
        permute_825: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_76, [0, 2, 1]);  _unsafe_view_76 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_830: f32[384, 512] = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_835: f32[384, 512] = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_840: f32[384, 512] = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_844: f32[512, 1024] = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_848: f32[1024, 512] = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_852: f32[1024, 512] = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_856: f32[512, 384] = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_859: f32[12, 128, 128] = torch.ops.aten.permute.default(view_67, [0, 2, 1]);  view_67 = None
        permute_860: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_66, [0, 2, 1]);  _unsafe_view_66 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_861: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_63, [0, 2, 1]);  _unsafe_view_63 = None
        permute_862: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_64, [0, 2, 1]);  _unsafe_view_64 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_867: f32[384, 512] = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_872: f32[384, 512] = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_877: f32[384, 512] = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_881: f32[512, 1024] = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_885: f32[1024, 512] = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_889: f32[1024, 512] = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_893: f32[512, 384] = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_896: f32[12, 128, 128] = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
        permute_897: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_54, [0, 2, 1]);  _unsafe_view_54 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_898: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_51, [0, 2, 1]);  _unsafe_view_51 = None
        permute_899: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_52, [0, 2, 1]);  _unsafe_view_52 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_904: f32[384, 512] = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_909: f32[384, 512] = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_914: f32[384, 512] = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_918: f32[512, 1024] = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_922: f32[1024, 512] = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_926: f32[1024, 512] = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_930: f32[512, 384] = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_933: f32[12, 128, 128] = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
        permute_934: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_42, [0, 2, 1]);  _unsafe_view_42 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_935: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_39, [0, 2, 1]);  _unsafe_view_39 = None
        permute_936: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_40, [0, 2, 1]);  _unsafe_view_40 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_941: f32[384, 512] = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_946: f32[384, 512] = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_951: f32[384, 512] = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_955: f32[512, 1024] = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_959: f32[1024, 512] = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_963: f32[1024, 512] = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_967: f32[512, 384] = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_970: f32[12, 128, 128] = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_971: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_30, [0, 2, 1]);  _unsafe_view_30 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_972: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_27, [0, 2, 1]);  _unsafe_view_27 = None
        permute_973: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_28, [0, 2, 1]);  _unsafe_view_28 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_978: f32[384, 512] = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_983: f32[384, 512] = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_988: f32[384, 512] = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_992: f32[512, 1024] = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_996: f32[1024, 512] = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_1000: f32[1024, 512] = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_1004: f32[512, 384] = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_1007: f32[12, 128, 128] = torch.ops.aten.permute.default(view_19, [0, 2, 1]);  view_19 = None
        permute_1008: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_18, [0, 2, 1]);  _unsafe_view_18 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_1009: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_15, [0, 2, 1]);  _unsafe_view_15 = None
        permute_1010: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_16, [0, 2, 1]);  _unsafe_view_16 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_1015: f32[384, 512] = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_1020: f32[384, 512] = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_1025: f32[384, 512] = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:311, code: hidden_states = self.wo(hidden_states)
        permute_1029: f32[512, 1024] = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:308, code: hidden_linear = self.wi_1(hidden_states)
        permute_1033: f32[1024, 512] = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:307, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        permute_1037: f32[1024, 512] = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:551, code: attn_output = self.o(attn_output)
        permute_1041: f32[512, 384] = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:550, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        permute_1044: f32[12, 128, 128] = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        permute_1045: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_6, [0, 2, 1]);  _unsafe_view_6 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:434, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        view_741: i64[16384] = torch.ops.aten.view.default(add_3, [16384]);  add_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:509, code: scores = torch.matmul(
        permute_1047: f32[12, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_3, [0, 2, 1]);  _unsafe_view_3 = None
        permute_1048: f32[12, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_4, [0, 2, 1]);  _unsafe_view_4 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_1053: f32[384, 512] = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:481, code: hidden_states = shape(proj_layer(hidden_states))
        permute_1058: f32[384, 512] = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        permute_1063: f32[384, 512] = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:941, code: inputs_embeds = self.embed_tokens(input_ids)
        view_753: i64[256] = torch.ops.aten.view.default(view, [256]);  view = None
        return [mean_42, _unsafe_view_264, permute_100, permute_102, permute_110, permute_112, permute_122, permute_124, permute_131, permute_133, permute_143, permute_145, permute_152, permute_154, permute_164, permute_166, permute_173, permute_175, permute_185, permute_187, permute_194, permute_196, permute_206, permute_208, permute_215, permute_217, permute_227, permute_229, permute_236, permute_238, permute_248, permute_250, permute_257, permute_259, mul_160, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, embedding, gt, reciprocal, div_2, philox_seed_like, view_9, gt_3, add_6, reciprocal_1, mm_4, sub_3, mm_5, gt_4, view_12, gt_5, add_11, reciprocal_3, div_3, view_21, gt_7, add_14, reciprocal_4, mm_11, sub_5, mm_12, gt_8, view_24, gt_9, add_19, reciprocal_6, div_4, view_33, gt_11, add_22, reciprocal_7, mm_18, sub_7, mm_19, gt_12, view_36, gt_13, add_27, reciprocal_9, div_5, view_45, gt_15, add_30, reciprocal_10, mm_25, sub_9, mm_26, gt_16, view_48, gt_17, add_35, reciprocal_12, div_6, view_57, gt_19, add_38, reciprocal_13, mm_32, sub_11, mm_33, gt_20, view_60, gt_21, add_43, reciprocal_15, div_7, view_69, gt_23, add_46, reciprocal_16, mm_39, sub_13, mm_40, gt_24, view_72, gt_25, add_51, reciprocal_18, div_8, view_81, gt_27, add_54, reciprocal_19, mm_46, sub_15, mm_47, gt_28, view_84, gt_29, add_59, reciprocal_21, div_9, view_93, gt_31, add_62, reciprocal_22, mm_53, sub_17, mm_54, gt_32, view_96, gt_33, add_67, reciprocal_24, gt_34, embedding_2, gt_35, reciprocal_25, div_12, view_106, gt_37, add_74, reciprocal_26, div_13, view_115, gt_39, add_78, reciprocal_27, mm_64, sub_23, mm_65, gt_40, view_118, gt_41, add_83, reciprocal_29, div_14, view_127, gt_43, add_86, reciprocal_30, div_15, view_136, gt_45, add_89, reciprocal_31, mm_75, sub_26, mm_76, gt_46, view_139, gt_47, add_94, reciprocal_33, div_16, view_148, gt_49, add_97, reciprocal_34, div_17, view_157, gt_51, add_100, reciprocal_35, mm_86, sub_29, mm_87, gt_52, view_160, gt_53, add_105, reciprocal_37, div_18, view_169, gt_55, add_108, reciprocal_38, div_19, view_178, gt_57, add_111, reciprocal_39, mm_97, sub_32, mm_98, gt_58, view_181, gt_59, add_116, reciprocal_41, div_20, view_190, gt_61, add_119, reciprocal_42, div_21, view_199, gt_63, add_122, reciprocal_43, mm_108, sub_35, mm_109, gt_64, view_202, gt_65, add_127, reciprocal_45, div_22, view_211, gt_67, add_130, reciprocal_46, div_23, view_220, gt_69, add_133, reciprocal_47, mm_119, sub_38, mm_120, gt_70, view_223, gt_71, add_138, reciprocal_49, div_24, view_232, gt_73, add_141, reciprocal_50, div_25, view_241, gt_75, add_144, reciprocal_51, mm_130, sub_41, mm_131, gt_76, view_244, gt_77, add_149, reciprocal_53, div_26, view_253, gt_79, add_152, reciprocal_54, div_27, view_262, gt_81, add_155, reciprocal_55, mm_141, sub_44, mm_142, gt_82, view_265, gt_83, add_160, reciprocal_57, gt_84, view_266, sub_46, unsqueeze_17, permute_269, permute_273, permute_277, permute_281, permute_285, permute_288, permute_289, permute_290, permute_291, permute_296, permute_301, permute_306, permute_310, permute_313, permute_314, permute_315, permute_316, permute_321, permute_326, permute_331, permute_335, permute_339, permute_343, permute_347, permute_350, permute_351, permute_352, permute_353, permute_358, permute_363, permute_368, permute_372, permute_375, permute_376, permute_377, permute_378, permute_383, permute_388, permute_393, permute_397, permute_401, permute_405, permute_409, permute_412, permute_413, permute_414, permute_415, permute_420, permute_425, permute_430, permute_434, permute_437, permute_438, permute_439, permute_440, permute_445, permute_450, permute_455, permute_459, permute_463, permute_467, permute_471, permute_474, permute_475, permute_476, permute_477, permute_482, permute_487, permute_492, permute_496, permute_499, permute_500, permute_501, permute_502, permute_507, permute_512, permute_517, permute_521, permute_525, permute_529, permute_533, permute_536, permute_537, permute_538, permute_539, permute_544, permute_549, permute_554, permute_558, permute_561, permute_562, permute_563, permute_564, permute_569, permute_574, permute_579, permute_583, permute_587, permute_591, permute_595, permute_598, permute_599, permute_600, permute_601, permute_606, permute_611, permute_616, permute_620, permute_623, permute_624, permute_625, permute_626, permute_631, permute_636, permute_641, permute_645, permute_649, permute_653, permute_657, permute_660, permute_661, permute_662, permute_663, permute_668, permute_673, permute_678, permute_682, permute_685, permute_686, permute_687, permute_688, permute_693, permute_698, permute_703, permute_707, permute_711, permute_715, permute_719, permute_722, permute_723, permute_724, permute_725, permute_730, permute_735, permute_740, permute_744, permute_747, permute_748, view_560, permute_750, permute_751, permute_756, permute_761, permute_766, view_572, permute_770, permute_774, permute_778, permute_782, permute_785, permute_786, permute_787, permute_788, permute_793, permute_798, permute_803, permute_807, permute_811, permute_815, permute_819, permute_822, permute_823, permute_824, permute_825, permute_830, permute_835, permute_840, permute_844, permute_848, permute_852, permute_856, permute_859, permute_860, permute_861, permute_862, permute_867, permute_872, permute_877, permute_881, permute_885, permute_889, permute_893, permute_896, permute_897, permute_898, permute_899, permute_904, permute_909, permute_914, permute_918, permute_922, permute_926, permute_930, permute_933, permute_934, permute_935, permute_936, permute_941, permute_946, permute_951, permute_955, permute_959, permute_963, permute_967, permute_970, permute_971, permute_972, permute_973, permute_978, permute_983, permute_988, permute_992, permute_996, permute_1000, permute_1004, permute_1007, permute_1008, permute_1009, permute_1010, permute_1015, permute_1020, permute_1025, permute_1029, permute_1033, permute_1037, permute_1041, permute_1044, permute_1045, view_741, permute_1047, permute_1048, permute_1053, permute_1058, permute_1063, view_753]
        