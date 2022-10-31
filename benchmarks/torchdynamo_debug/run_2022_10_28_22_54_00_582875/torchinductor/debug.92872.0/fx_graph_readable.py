class GraphModule(torch.nn.Module):
    def forward(self, primals_1: f32[30522, 768], primals_2: f32[2, 768], primals_3: f32[512, 768], primals_4: f32[768], primals_5: f32[768], primals_6: f32[768, 768], primals_7: f32[768], primals_8: f32[768, 768], primals_9: f32[768], primals_10: f32[768, 768], primals_11: f32[768], primals_12: f32[768, 768], primals_13: f32[768], primals_14: f32[768], primals_15: f32[768], primals_16: f32[3072, 768], primals_17: f32[3072], primals_18: f32[768, 3072], primals_19: f32[768], primals_20: f32[768], primals_21: f32[768], primals_22: f32[768, 768], primals_23: f32[768], primals_24: f32[768, 768], primals_25: f32[768], primals_26: f32[768, 768], primals_27: f32[768], primals_28: f32[768, 768], primals_29: f32[768], primals_30: f32[768], primals_31: f32[768], primals_32: f32[3072, 768], primals_33: f32[3072], primals_34: f32[768, 3072], primals_35: f32[768], primals_36: f32[768], primals_37: f32[768], primals_38: f32[768, 768], primals_39: f32[768], primals_40: f32[768, 768], primals_41: f32[768], primals_42: f32[768, 768], primals_43: f32[768], primals_44: f32[768, 768], primals_45: f32[768], primals_46: f32[768], primals_47: f32[768], primals_48: f32[3072, 768], primals_49: f32[3072], primals_50: f32[768, 3072], primals_51: f32[768], primals_52: f32[768], primals_53: f32[768], primals_54: f32[768, 768], primals_55: f32[768], primals_56: f32[768, 768], primals_57: f32[768], primals_58: f32[768, 768], primals_59: f32[768], primals_60: f32[768, 768], primals_61: f32[768], primals_62: f32[768], primals_63: f32[768], primals_64: f32[3072, 768], primals_65: f32[3072], primals_66: f32[768, 3072], primals_67: f32[768], primals_68: f32[768], primals_69: f32[768], primals_70: f32[768, 768], primals_71: f32[768], primals_72: f32[768, 768], primals_73: f32[768], primals_74: f32[768, 768], primals_75: f32[768], primals_76: f32[768, 768], primals_77: f32[768], primals_78: f32[768], primals_79: f32[768], primals_80: f32[3072, 768], primals_81: f32[3072], primals_82: f32[768, 3072], primals_83: f32[768], primals_84: f32[768], primals_85: f32[768], primals_86: f32[768, 768], primals_87: f32[768], primals_88: f32[768, 768], primals_89: f32[768], primals_90: f32[768, 768], primals_91: f32[768], primals_92: f32[768, 768], primals_93: f32[768], primals_94: f32[768], primals_95: f32[768], primals_96: f32[3072, 768], primals_97: f32[3072], primals_98: f32[768, 3072], primals_99: f32[768], primals_100: f32[768], primals_101: f32[768], primals_102: f32[768, 768], primals_103: f32[768], primals_104: f32[768, 768], primals_105: f32[768], primals_106: f32[768, 768], primals_107: f32[768], primals_108: f32[768, 768], primals_109: f32[768], primals_110: f32[768], primals_111: f32[768], primals_112: f32[3072, 768], primals_113: f32[3072], primals_114: f32[768, 3072], primals_115: f32[768], primals_116: f32[768], primals_117: f32[768], primals_118: f32[768, 768], primals_119: f32[768], primals_120: f32[768, 768], primals_121: f32[768], primals_122: f32[768, 768], primals_123: f32[768], primals_124: f32[768, 768], primals_125: f32[768], primals_126: f32[768], primals_127: f32[768], primals_128: f32[3072, 768], primals_129: f32[3072], primals_130: f32[768, 3072], primals_131: f32[768], primals_132: f32[768], primals_133: f32[768], primals_134: f32[768, 768], primals_135: f32[768], primals_136: f32[768, 768], primals_137: f32[768], primals_138: f32[768, 768], primals_139: f32[768], primals_140: f32[768, 768], primals_141: f32[768], primals_142: f32[768], primals_143: f32[768], primals_144: f32[3072, 768], primals_145: f32[3072], primals_146: f32[768, 3072], primals_147: f32[768], primals_148: f32[768], primals_149: f32[768], primals_150: f32[768, 768], primals_151: f32[768], primals_152: f32[768, 768], primals_153: f32[768], primals_154: f32[768, 768], primals_155: f32[768], primals_156: f32[768, 768], primals_157: f32[768], primals_158: f32[768], primals_159: f32[768], primals_160: f32[3072, 768], primals_161: f32[3072], primals_162: f32[768, 3072], primals_163: f32[768], primals_164: f32[768], primals_165: f32[768], primals_166: f32[768, 768], primals_167: f32[768], primals_168: f32[768, 768], primals_169: f32[768], primals_170: f32[768, 768], primals_171: f32[768], primals_172: f32[768, 768], primals_173: f32[768], primals_174: f32[768], primals_175: f32[768], primals_176: f32[3072, 768], primals_177: f32[3072], primals_178: f32[768, 3072], primals_179: f32[768], primals_180: f32[768], primals_181: f32[768], primals_182: f32[768, 768], primals_183: f32[768], primals_184: f32[768, 768], primals_185: f32[768], primals_186: f32[768, 768], primals_187: f32[768], primals_188: f32[768, 768], primals_189: f32[768], primals_190: f32[768], primals_191: f32[768], primals_192: f32[3072, 768], primals_193: f32[3072], primals_194: f32[768, 3072], primals_195: f32[768], primals_196: f32[768], primals_197: f32[768], primals_198: f32[768, 768], primals_199: f32[768], primals_200: f32[768], primals_201: f32[768], primals_202: f32[30522], primals_203: i64[1, 512], primals_204: i64[1, 512], primals_205: i64[64, 128], primals_206: i64[64, 128]):
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:975, code: attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        ones: f32[64, 128] = torch.ops.aten.ones.default([64, 128], device = device(type='cuda', index=0), pin_memory = False)
        alias: f32[64, 128] = torch.ops.aten.alias.default(ones);  ones = None
        alias_1: f32[64, 128] = torch.ops.aten.alias.default(alias);  alias = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:979, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
        slice_1: i64[1, 512] = torch.ops.aten.slice.Tensor(primals_203, 0, 0, 9223372036854775807);  primals_203 = None
        slice_2: i64[1, 128] = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 128);  slice_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:980, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand: i64[64, 128] = torch.ops.aten.expand.default(slice_2, [64, 128])
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/modeling_utils.py:779, code: extended_attention_mask = attention_mask[:, None, None, :]
        slice_3: f32[64, 128] = torch.ops.aten.slice.Tensor(alias_1, 0, 0, 9223372036854775807);  alias_1 = None
        unsqueeze: f32[64, 1, 128] = torch.ops.aten.unsqueeze.default(slice_3, 1);  slice_3 = None
        unsqueeze_1: f32[64, 1, 1, 128] = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        slice_4: f32[64, 1, 1, 128] = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
        
        # No stacktrace found for following nodes
        _tensor_constant0 = self._tensor_constant0
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/modeling_utils.py:791, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        lift_fresh_copy: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        sub: f32[64, 1, 1, 128] = torch.ops.aten.sub.Tensor(lift_fresh_copy, slice_4);  lift_fresh_copy = slice_4 = None
        mul: f32[64, 1, 1, 128] = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:217, code: position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        slice_5: i64[1, 512] = torch.ops.aten.slice.Tensor(primals_204, 0, 0, 9223372036854775807);  primals_204 = None
        slice_6: i64[1, 128] = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 128);  slice_5 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:231, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: f32[64, 128, 768] = torch.ops.aten.embedding.default(primals_1, primals_205, 0)
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:232, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: f32[64, 128, 768] = torch.ops.aten.embedding.default(primals_2, expand);  primals_2 = expand = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:234, code: embeddings = inputs_embeds + token_type_embeddings
        add: f32[64, 128, 768] = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:236, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: f32[1, 128, 768] = torch.ops.aten.embedding.default(primals_3, slice_6);  primals_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:237, code: embeddings += position_embeddings
        add_1: f32[64, 128, 768] = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:238, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: f32[64, 128, 1] = var_mean[0]
        getitem_1: f32[64, 128, 1] = var_mean[1];  var_mean = None
        add_2: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        sqrt: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_2);  add_2 = None
        reciprocal: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        sub_1: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul_1: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_1, reciprocal);  sub_1 = None
        mul_2: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_1, primals_4)
        add_3: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_2, primals_5);  mul_2 = primals_5 = None
        convert_element_type: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_3, torch.float32);  add_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:239, code: embeddings = self.dropout(embeddings)
        rand_like: f32[64, 128, 768] = torch.ops.aten.rand_like.default(convert_element_type, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_2: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like);  rand_like = None
        gt: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_2, 0.1);  alias_2 = None
        mul_3: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt, convert_element_type);  convert_element_type = None
        mul_4: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute: f32[768, 768] = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        view: f32[8192, 768] = torch.ops.aten.view.default(mul_4, [8192, 768])
        addmm: f32[8192, 768] = torch.ops.aten.addmm.default(primals_7, view, permute);  primals_7 = None
        view_1: f32[64, 128, 768] = torch.ops.aten.view.default(addmm, [64, 128, 768]);  addmm = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_1: f32[768, 768] = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        addmm_1: f32[8192, 768] = torch.ops.aten.addmm.default(primals_9, view, permute_1);  primals_9 = None
        view_3: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_1, [64, 128, 768]);  addmm_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_4: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_3, [64, 128, 12, 64]);  view_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_2: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_3: f32[768, 768] = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        addmm_2: f32[8192, 768] = torch.ops.aten.addmm.default(primals_11, view, permute_3);  primals_11 = None
        view_6: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_2, [64, 128, 768]);  addmm_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_7: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_6, [64, 128, 12, 64]);  view_6 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_4: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_8: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_1, [64, 128, 12, 64]);  view_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_5: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_6: f32[64, 12, 64, 128] = torch.ops.aten.permute.default(permute_2, [0, 1, 3, 2]);  permute_2 = None
        expand_1: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_5, [64, 12, 128, 64]);  permute_5 = None
        clone: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        _unsafe_view: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone, [768, 128, 64]);  clone = None
        expand_2: f32[64, 12, 64, 128] = torch.ops.aten.expand.default(permute_6, [64, 12, 64, 128]);  permute_6 = None
        clone_1: f32[64, 12, 64, 128] = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        _unsafe_view_1: f32[768, 64, 128] = torch.ops.aten._unsafe_view.default(clone_1, [768, 64, 128]);  clone_1 = None
        bmm: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view, _unsafe_view_1)
        _unsafe_view_2: f32[64, 12, 128, 128] = torch.ops.aten._unsafe_view.default(bmm, [64, 12, 128, 128]);  bmm = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(_unsafe_view_2, 8.0);  _unsafe_view_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:344, code: attention_scores = attention_scores + attention_mask
        add_4: f32[64, 12, 128, 128] = torch.ops.aten.add.Tensor(div, mul);  div = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax: f32[64, 12, 128, 1] = torch.ops.aten.amax.default(add_4, [-1], True)
        sub_2: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(add_4, amax);  add_4 = amax = None
        exp: f32[64, 12, 128, 128] = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        alias_4: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(div_1)
        alias_5: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_4);  alias_4 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        rand_like_1: f32[64, 12, 128, 128] = torch.ops.aten.rand_like.default(div_1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_6: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(rand_like_1);  rand_like_1 = None
        gt_1: b8[64, 12, 128, 128] = torch.ops.aten.gt.Scalar(alias_6, 0.1);  alias_6 = None
        mul_5: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(gt_1, div_1);  div_1 = None
        mul_6: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_5, 1.1111111111111112);  mul_5 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        expand_3: f32[64, 12, 128, 128] = torch.ops.aten.expand.default(mul_6, [64, 12, 128, 128]);  mul_6 = None
        view_9: f32[768, 128, 128] = torch.ops.aten.view.default(expand_3, [768, 128, 128]);  expand_3 = None
        expand_4: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_4, [64, 12, 128, 64]);  permute_4 = None
        clone_2: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        _unsafe_view_3: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_2, [768, 128, 64]);  clone_2 = None
        bmm_1: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_9, _unsafe_view_3)
        _unsafe_view_4: f32[64, 12, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_1, [64, 12, 128, 64]);  bmm_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_7: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(_unsafe_view_4, [0, 2, 1, 3]);  _unsafe_view_4 = None
        clone_3: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_10: f32[64, 128, 768] = torch.ops.aten.view.default(clone_3, [64, 128, 768]);  clone_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_8: f32[768, 768] = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        view_11: f32[8192, 768] = torch.ops.aten.view.default(view_10, [8192, 768]);  view_10 = None
        addmm_3: f32[8192, 768] = torch.ops.aten.addmm.default(primals_13, view_11, permute_8);  primals_13 = None
        view_12: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_3, [64, 128, 768]);  addmm_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        rand_like_2: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_12, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_7: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_2);  rand_like_2 = None
        gt_2: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_7, 0.1);  alias_7 = None
        mul_7: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_2, view_12);  view_12 = None
        mul_8: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112);  mul_7 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_5: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_8, mul_4);  mul_8 = mul_4 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2: f32[64, 128, 1] = var_mean_1[0]
        getitem_3: f32[64, 128, 1] = var_mean_1[1];  var_mean_1 = None
        add_6: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        sqrt_1: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_6);  add_6 = None
        reciprocal_1: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        sub_3: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
        mul_9: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_3, reciprocal_1);  sub_3 = None
        mul_10: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_9, primals_14)
        add_7: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_10, primals_15);  mul_10 = primals_15 = None
        convert_element_type_1: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_7, torch.float32);  add_7 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_9: f32[768, 3072] = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        view_13: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_1, [8192, 768])
        addmm_4: f32[8192, 3072] = torch.ops.aten.addmm.default(primals_17, view_13, permute_9);  primals_17 = None
        view_14: f32[64, 128, 3072] = torch.ops.aten.view.default(addmm_4, [64, 128, 3072]);  addmm_4 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_11: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_14, 0.5)
        mul_12: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476)
        sign: f32[64, 128, 3072] = torch.ops.aten.sign.default(mul_12)
        abs_1: f32[64, 128, 3072] = torch.ops.aten.abs.default(mul_12);  mul_12 = None
        mul_13: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(abs_1, 0.3275911)
        add_8: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_13, 1.0);  mul_13 = None
        reciprocal_2: f32[64, 128, 3072] = torch.ops.aten.reciprocal.default(add_8);  add_8 = None
        mul_14: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(reciprocal_2, 1.0);  reciprocal_2 = None
        mul_15: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_14, 1.061405429)
        add_9: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_15, -1.453152027);  mul_15 = None
        mul_16: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_9, mul_14);  add_9 = None
        add_10: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_16, 1.421413741);  mul_16 = None
        mul_17: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_10, mul_14);  add_10 = None
        add_11: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_17, -0.284496736);  mul_17 = None
        mul_18: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_11, mul_14);  add_11 = None
        add_12: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_18, 0.254829592);  mul_18 = None
        mul_19: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_12, mul_14);  add_12 = mul_14 = None
        neg: f32[64, 128, 3072] = torch.ops.aten.neg.default(abs_1)
        mul_20: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(neg, abs_1);  neg = abs_1 = None
        exp_1: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_20);  mul_20 = None
        mul_21: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_19, exp_1);  mul_19 = exp_1 = None
        
        # No stacktrace found for following nodes
        _tensor_constant1 = self._tensor_constant1
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_1: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        sub_4: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_1, mul_21);  lift_fresh_copy_1 = None
        mul_22: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign, sub_4);  sub_4 = None
        add_13: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_22, 1);  mul_22 = None
        mul_23: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_11, add_13);  mul_11 = add_13 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_10: f32[3072, 768] = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        view_15: f32[8192, 3072] = torch.ops.aten.view.default(mul_23, [8192, 3072]);  mul_23 = None
        addmm_5: f32[8192, 768] = torch.ops.aten.addmm.default(primals_19, view_15, permute_10);  primals_19 = None
        view_16: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_5, [64, 128, 768]);  addmm_5 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        rand_like_3: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_16, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_8: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_3);  rand_like_3 = None
        gt_3: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_8, 0.1);  alias_8 = None
        mul_24: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_3, view_16);  view_16 = None
        mul_25: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_24, 1.1111111111111112);  mul_24 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_14: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_25, convert_element_type_1);  mul_25 = convert_element_type_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_4: f32[64, 128, 1] = var_mean_2[0]
        getitem_5: f32[64, 128, 1] = var_mean_2[1];  var_mean_2 = None
        add_15: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        sqrt_2: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_15);  add_15 = None
        reciprocal_3: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        sub_5: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_14, getitem_5);  add_14 = getitem_5 = None
        mul_26: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_5, reciprocal_3);  sub_5 = None
        mul_27: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_26, primals_20)
        add_16: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_27, primals_21);  mul_27 = primals_21 = None
        convert_element_type_2: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_11: f32[768, 768] = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
        view_17: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_2, [8192, 768])
        addmm_6: f32[8192, 768] = torch.ops.aten.addmm.default(primals_23, view_17, permute_11);  primals_23 = None
        view_18: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_6, [64, 128, 768]);  addmm_6 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_12: f32[768, 768] = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        addmm_7: f32[8192, 768] = torch.ops.aten.addmm.default(primals_25, view_17, permute_12);  primals_25 = None
        view_20: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_7, [64, 128, 768]);  addmm_7 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_21: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_20, [64, 128, 12, 64]);  view_20 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_13: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_14: f32[768, 768] = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        addmm_8: f32[8192, 768] = torch.ops.aten.addmm.default(primals_27, view_17, permute_14);  primals_27 = None
        view_23: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_8, [64, 128, 768]);  addmm_8 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_24: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_23, [64, 128, 12, 64]);  view_23 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_15: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_25: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_18, [64, 128, 12, 64]);  view_18 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_16: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_17: f32[64, 12, 64, 128] = torch.ops.aten.permute.default(permute_13, [0, 1, 3, 2]);  permute_13 = None
        expand_5: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_16, [64, 12, 128, 64]);  permute_16 = None
        clone_4: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        _unsafe_view_5: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_4, [768, 128, 64]);  clone_4 = None
        expand_6: f32[64, 12, 64, 128] = torch.ops.aten.expand.default(permute_17, [64, 12, 64, 128]);  permute_17 = None
        clone_5: f32[64, 12, 64, 128] = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        _unsafe_view_6: f32[768, 64, 128] = torch.ops.aten._unsafe_view.default(clone_5, [768, 64, 128]);  clone_5 = None
        bmm_2: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_5, _unsafe_view_6)
        _unsafe_view_7: f32[64, 12, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_2, [64, 12, 128, 128]);  bmm_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_2: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(_unsafe_view_7, 8.0);  _unsafe_view_7 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:344, code: attention_scores = attention_scores + attention_mask
        add_17: f32[64, 12, 128, 128] = torch.ops.aten.add.Tensor(div_2, mul);  div_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_1: f32[64, 12, 128, 1] = torch.ops.aten.amax.default(add_17, [-1], True)
        sub_6: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(add_17, amax_1);  add_17 = amax_1 = None
        exp_2: f32[64, 12, 128, 128] = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_2: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_3: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(exp_2, sum_2);  exp_2 = sum_2 = None
        alias_10: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(div_3)
        alias_11: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_10);  alias_10 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        rand_like_4: f32[64, 12, 128, 128] = torch.ops.aten.rand_like.default(div_3, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_12: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(rand_like_4);  rand_like_4 = None
        gt_4: b8[64, 12, 128, 128] = torch.ops.aten.gt.Scalar(alias_12, 0.1);  alias_12 = None
        mul_28: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(gt_4, div_3);  div_3 = None
        mul_29: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_28, 1.1111111111111112);  mul_28 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        expand_7: f32[64, 12, 128, 128] = torch.ops.aten.expand.default(mul_29, [64, 12, 128, 128]);  mul_29 = None
        view_26: f32[768, 128, 128] = torch.ops.aten.view.default(expand_7, [768, 128, 128]);  expand_7 = None
        expand_8: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_15, [64, 12, 128, 64]);  permute_15 = None
        clone_6: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        _unsafe_view_8: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_6, [768, 128, 64]);  clone_6 = None
        bmm_3: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_26, _unsafe_view_8)
        _unsafe_view_9: f32[64, 12, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_3, [64, 12, 128, 64]);  bmm_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_18: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(_unsafe_view_9, [0, 2, 1, 3]);  _unsafe_view_9 = None
        clone_7: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_27: f32[64, 128, 768] = torch.ops.aten.view.default(clone_7, [64, 128, 768]);  clone_7 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_19: f32[768, 768] = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        view_28: f32[8192, 768] = torch.ops.aten.view.default(view_27, [8192, 768]);  view_27 = None
        addmm_9: f32[8192, 768] = torch.ops.aten.addmm.default(primals_29, view_28, permute_19);  primals_29 = None
        view_29: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_9, [64, 128, 768]);  addmm_9 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        rand_like_5: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_29, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_13: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_5);  rand_like_5 = None
        gt_5: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_13, 0.1);  alias_13 = None
        mul_30: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_5, view_29);  view_29 = None
        mul_31: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_30, 1.1111111111111112);  mul_30 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_18: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_31, convert_element_type_2);  mul_31 = convert_element_type_2 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
        getitem_6: f32[64, 128, 1] = var_mean_3[0]
        getitem_7: f32[64, 128, 1] = var_mean_3[1];  var_mean_3 = None
        add_19: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        sqrt_3: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_19);  add_19 = None
        reciprocal_4: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        sub_7: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_18, getitem_7);  add_18 = getitem_7 = None
        mul_32: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_7, reciprocal_4);  sub_7 = None
        mul_33: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_32, primals_30)
        add_20: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_33, primals_31);  mul_33 = primals_31 = None
        convert_element_type_3: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_20, torch.float32);  add_20 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_20: f32[768, 3072] = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        view_30: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_3, [8192, 768])
        addmm_10: f32[8192, 3072] = torch.ops.aten.addmm.default(primals_33, view_30, permute_20);  primals_33 = None
        view_31: f32[64, 128, 3072] = torch.ops.aten.view.default(addmm_10, [64, 128, 3072]);  addmm_10 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_34: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_31, 0.5)
        mul_35: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476)
        sign_1: f32[64, 128, 3072] = torch.ops.aten.sign.default(mul_35)
        abs_2: f32[64, 128, 3072] = torch.ops.aten.abs.default(mul_35);  mul_35 = None
        mul_36: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(abs_2, 0.3275911)
        add_21: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_36, 1.0);  mul_36 = None
        reciprocal_5: f32[64, 128, 3072] = torch.ops.aten.reciprocal.default(add_21);  add_21 = None
        mul_37: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(reciprocal_5, 1.0);  reciprocal_5 = None
        mul_38: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_37, 1.061405429)
        add_22: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_38, -1.453152027);  mul_38 = None
        mul_39: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_22, mul_37);  add_22 = None
        add_23: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_39, 1.421413741);  mul_39 = None
        mul_40: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_23, mul_37);  add_23 = None
        add_24: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_40, -0.284496736);  mul_40 = None
        mul_41: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_24, mul_37);  add_24 = None
        add_25: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_41, 0.254829592);  mul_41 = None
        mul_42: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_25, mul_37);  add_25 = mul_37 = None
        neg_1: f32[64, 128, 3072] = torch.ops.aten.neg.default(abs_2)
        mul_43: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(neg_1, abs_2);  neg_1 = abs_2 = None
        exp_3: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_43);  mul_43 = None
        mul_44: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_42, exp_3);  mul_42 = exp_3 = None
        
        # No stacktrace found for following nodes
        _tensor_constant2 = self._tensor_constant2
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_2: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        sub_8: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_2, mul_44);  lift_fresh_copy_2 = None
        mul_45: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_1, sub_8);  sub_8 = None
        add_26: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_45, 1);  mul_45 = None
        mul_46: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_34, add_26);  mul_34 = add_26 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_21: f32[3072, 768] = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        view_32: f32[8192, 3072] = torch.ops.aten.view.default(mul_46, [8192, 3072]);  mul_46 = None
        addmm_11: f32[8192, 768] = torch.ops.aten.addmm.default(primals_35, view_32, permute_21);  primals_35 = None
        view_33: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_11, [64, 128, 768]);  addmm_11 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        rand_like_6: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_33, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_14: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_6);  rand_like_6 = None
        gt_6: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_14, 0.1);  alias_14 = None
        mul_47: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_6, view_33);  view_33 = None
        mul_48: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_47, 1.1111111111111112);  mul_47 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_27: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_48, convert_element_type_3);  mul_48 = convert_element_type_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_8: f32[64, 128, 1] = var_mean_4[0]
        getitem_9: f32[64, 128, 1] = var_mean_4[1];  var_mean_4 = None
        add_28: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        sqrt_4: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_28);  add_28 = None
        reciprocal_6: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        sub_9: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_27, getitem_9);  add_27 = getitem_9 = None
        mul_49: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_9, reciprocal_6);  sub_9 = None
        mul_50: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_49, primals_36)
        add_29: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_50, primals_37);  mul_50 = primals_37 = None
        convert_element_type_4: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_29, torch.float32);  add_29 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_22: f32[768, 768] = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        view_34: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_4, [8192, 768])
        addmm_12: f32[8192, 768] = torch.ops.aten.addmm.default(primals_39, view_34, permute_22);  primals_39 = None
        view_35: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_12, [64, 128, 768]);  addmm_12 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_23: f32[768, 768] = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        addmm_13: f32[8192, 768] = torch.ops.aten.addmm.default(primals_41, view_34, permute_23);  primals_41 = None
        view_37: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_13, [64, 128, 768]);  addmm_13 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_38: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_37, [64, 128, 12, 64]);  view_37 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_24: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_25: f32[768, 768] = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        addmm_14: f32[8192, 768] = torch.ops.aten.addmm.default(primals_43, view_34, permute_25);  primals_43 = None
        view_40: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_14, [64, 128, 768]);  addmm_14 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_41: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_40, [64, 128, 12, 64]);  view_40 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_26: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_42: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_35, [64, 128, 12, 64]);  view_35 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_27: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_28: f32[64, 12, 64, 128] = torch.ops.aten.permute.default(permute_24, [0, 1, 3, 2]);  permute_24 = None
        expand_9: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_27, [64, 12, 128, 64]);  permute_27 = None
        clone_8: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        _unsafe_view_10: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_8, [768, 128, 64]);  clone_8 = None
        expand_10: f32[64, 12, 64, 128] = torch.ops.aten.expand.default(permute_28, [64, 12, 64, 128]);  permute_28 = None
        clone_9: f32[64, 12, 64, 128] = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        _unsafe_view_11: f32[768, 64, 128] = torch.ops.aten._unsafe_view.default(clone_9, [768, 64, 128]);  clone_9 = None
        bmm_4: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_10, _unsafe_view_11)
        _unsafe_view_12: f32[64, 12, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_4, [64, 12, 128, 128]);  bmm_4 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_4: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(_unsafe_view_12, 8.0);  _unsafe_view_12 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:344, code: attention_scores = attention_scores + attention_mask
        add_30: f32[64, 12, 128, 128] = torch.ops.aten.add.Tensor(div_4, mul);  div_4 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_2: f32[64, 12, 128, 1] = torch.ops.aten.amax.default(add_30, [-1], True)
        sub_10: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(add_30, amax_2);  add_30 = amax_2 = None
        exp_4: f32[64, 12, 128, 128] = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_3: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_5: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(exp_4, sum_3);  exp_4 = sum_3 = None
        alias_16: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(div_5)
        alias_17: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_16);  alias_16 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        rand_like_7: f32[64, 12, 128, 128] = torch.ops.aten.rand_like.default(div_5, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_18: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(rand_like_7);  rand_like_7 = None
        gt_7: b8[64, 12, 128, 128] = torch.ops.aten.gt.Scalar(alias_18, 0.1);  alias_18 = None
        mul_51: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(gt_7, div_5);  div_5 = None
        mul_52: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_51, 1.1111111111111112);  mul_51 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        expand_11: f32[64, 12, 128, 128] = torch.ops.aten.expand.default(mul_52, [64, 12, 128, 128]);  mul_52 = None
        view_43: f32[768, 128, 128] = torch.ops.aten.view.default(expand_11, [768, 128, 128]);  expand_11 = None
        expand_12: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_26, [64, 12, 128, 64]);  permute_26 = None
        clone_10: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        _unsafe_view_13: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_10, [768, 128, 64]);  clone_10 = None
        bmm_5: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_43, _unsafe_view_13)
        _unsafe_view_14: f32[64, 12, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_5, [64, 12, 128, 64]);  bmm_5 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_29: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(_unsafe_view_14, [0, 2, 1, 3]);  _unsafe_view_14 = None
        clone_11: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_44: f32[64, 128, 768] = torch.ops.aten.view.default(clone_11, [64, 128, 768]);  clone_11 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_30: f32[768, 768] = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        view_45: f32[8192, 768] = torch.ops.aten.view.default(view_44, [8192, 768]);  view_44 = None
        addmm_15: f32[8192, 768] = torch.ops.aten.addmm.default(primals_45, view_45, permute_30);  primals_45 = None
        view_46: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_15, [64, 128, 768]);  addmm_15 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        rand_like_8: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_46, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_19: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_8);  rand_like_8 = None
        gt_8: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_19, 0.1);  alias_19 = None
        mul_53: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_8, view_46);  view_46 = None
        mul_54: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_53, 1.1111111111111112);  mul_53 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_31: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_54, convert_element_type_4);  mul_54 = convert_element_type_4 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_10: f32[64, 128, 1] = var_mean_5[0]
        getitem_11: f32[64, 128, 1] = var_mean_5[1];  var_mean_5 = None
        add_32: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        sqrt_5: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_32);  add_32 = None
        reciprocal_7: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        sub_11: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_31, getitem_11);  add_31 = getitem_11 = None
        mul_55: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_11, reciprocal_7);  sub_11 = None
        mul_56: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_55, primals_46)
        add_33: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_56, primals_47);  mul_56 = primals_47 = None
        convert_element_type_5: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_33, torch.float32);  add_33 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_31: f32[768, 3072] = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        view_47: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_5, [8192, 768])
        addmm_16: f32[8192, 3072] = torch.ops.aten.addmm.default(primals_49, view_47, permute_31);  primals_49 = None
        view_48: f32[64, 128, 3072] = torch.ops.aten.view.default(addmm_16, [64, 128, 3072]);  addmm_16 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_57: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_48, 0.5)
        mul_58: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_48, 0.7071067811865476)
        sign_2: f32[64, 128, 3072] = torch.ops.aten.sign.default(mul_58)
        abs_3: f32[64, 128, 3072] = torch.ops.aten.abs.default(mul_58);  mul_58 = None
        mul_59: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(abs_3, 0.3275911)
        add_34: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_59, 1.0);  mul_59 = None
        reciprocal_8: f32[64, 128, 3072] = torch.ops.aten.reciprocal.default(add_34);  add_34 = None
        mul_60: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(reciprocal_8, 1.0);  reciprocal_8 = None
        mul_61: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_60, 1.061405429)
        add_35: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_61, -1.453152027);  mul_61 = None
        mul_62: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_35, mul_60);  add_35 = None
        add_36: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_62, 1.421413741);  mul_62 = None
        mul_63: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_36, mul_60);  add_36 = None
        add_37: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_63, -0.284496736);  mul_63 = None
        mul_64: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_37, mul_60);  add_37 = None
        add_38: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_64, 0.254829592);  mul_64 = None
        mul_65: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_38, mul_60);  add_38 = mul_60 = None
        neg_2: f32[64, 128, 3072] = torch.ops.aten.neg.default(abs_3)
        mul_66: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(neg_2, abs_3);  neg_2 = abs_3 = None
        exp_5: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_66);  mul_66 = None
        mul_67: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_65, exp_5);  mul_65 = exp_5 = None
        
        # No stacktrace found for following nodes
        _tensor_constant3 = self._tensor_constant3
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_3: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
        sub_12: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_3, mul_67);  lift_fresh_copy_3 = None
        mul_68: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_2, sub_12);  sub_12 = None
        add_39: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_68, 1);  mul_68 = None
        mul_69: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_57, add_39);  mul_57 = add_39 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_32: f32[3072, 768] = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        view_49: f32[8192, 3072] = torch.ops.aten.view.default(mul_69, [8192, 3072]);  mul_69 = None
        addmm_17: f32[8192, 768] = torch.ops.aten.addmm.default(primals_51, view_49, permute_32);  primals_51 = None
        view_50: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_17, [64, 128, 768]);  addmm_17 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        rand_like_9: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_50, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_20: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_9);  rand_like_9 = None
        gt_9: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_20, 0.1);  alias_20 = None
        mul_70: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_9, view_50);  view_50 = None
        mul_71: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_70, 1.1111111111111112);  mul_70 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_40: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_71, convert_element_type_5);  mul_71 = convert_element_type_5 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_12: f32[64, 128, 1] = var_mean_6[0]
        getitem_13: f32[64, 128, 1] = var_mean_6[1];  var_mean_6 = None
        add_41: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        sqrt_6: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_41);  add_41 = None
        reciprocal_9: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        sub_13: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_40, getitem_13);  add_40 = getitem_13 = None
        mul_72: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_13, reciprocal_9);  sub_13 = None
        mul_73: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_72, primals_52)
        add_42: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_73, primals_53);  mul_73 = primals_53 = None
        convert_element_type_6: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_42, torch.float32);  add_42 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_33: f32[768, 768] = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
        view_51: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_6, [8192, 768])
        addmm_18: f32[8192, 768] = torch.ops.aten.addmm.default(primals_55, view_51, permute_33);  primals_55 = None
        view_52: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_18, [64, 128, 768]);  addmm_18 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_34: f32[768, 768] = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        addmm_19: f32[8192, 768] = torch.ops.aten.addmm.default(primals_57, view_51, permute_34);  primals_57 = None
        view_54: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_19, [64, 128, 768]);  addmm_19 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_55: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_54, [64, 128, 12, 64]);  view_54 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_35: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_36: f32[768, 768] = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        addmm_20: f32[8192, 768] = torch.ops.aten.addmm.default(primals_59, view_51, permute_36);  primals_59 = None
        view_57: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_20, [64, 128, 768]);  addmm_20 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_58: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_57, [64, 128, 12, 64]);  view_57 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_37: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_59: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_52, [64, 128, 12, 64]);  view_52 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_38: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_39: f32[64, 12, 64, 128] = torch.ops.aten.permute.default(permute_35, [0, 1, 3, 2]);  permute_35 = None
        expand_13: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_38, [64, 12, 128, 64]);  permute_38 = None
        clone_12: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        _unsafe_view_15: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_12, [768, 128, 64]);  clone_12 = None
        expand_14: f32[64, 12, 64, 128] = torch.ops.aten.expand.default(permute_39, [64, 12, 64, 128]);  permute_39 = None
        clone_13: f32[64, 12, 64, 128] = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        _unsafe_view_16: f32[768, 64, 128] = torch.ops.aten._unsafe_view.default(clone_13, [768, 64, 128]);  clone_13 = None
        bmm_6: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_15, _unsafe_view_16)
        _unsafe_view_17: f32[64, 12, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_6, [64, 12, 128, 128]);  bmm_6 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_6: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(_unsafe_view_17, 8.0);  _unsafe_view_17 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:344, code: attention_scores = attention_scores + attention_mask
        add_43: f32[64, 12, 128, 128] = torch.ops.aten.add.Tensor(div_6, mul);  div_6 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_3: f32[64, 12, 128, 1] = torch.ops.aten.amax.default(add_43, [-1], True)
        sub_14: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(add_43, amax_3);  add_43 = amax_3 = None
        exp_6: f32[64, 12, 128, 128] = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_4: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_7: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(exp_6, sum_4);  exp_6 = sum_4 = None
        alias_22: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(div_7)
        alias_23: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_22);  alias_22 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        rand_like_10: f32[64, 12, 128, 128] = torch.ops.aten.rand_like.default(div_7, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_24: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(rand_like_10);  rand_like_10 = None
        gt_10: b8[64, 12, 128, 128] = torch.ops.aten.gt.Scalar(alias_24, 0.1);  alias_24 = None
        mul_74: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(gt_10, div_7);  div_7 = None
        mul_75: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_74, 1.1111111111111112);  mul_74 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        expand_15: f32[64, 12, 128, 128] = torch.ops.aten.expand.default(mul_75, [64, 12, 128, 128]);  mul_75 = None
        view_60: f32[768, 128, 128] = torch.ops.aten.view.default(expand_15, [768, 128, 128]);  expand_15 = None
        expand_16: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_37, [64, 12, 128, 64]);  permute_37 = None
        clone_14: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        _unsafe_view_18: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_14, [768, 128, 64]);  clone_14 = None
        bmm_7: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_60, _unsafe_view_18)
        _unsafe_view_19: f32[64, 12, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_7, [64, 12, 128, 64]);  bmm_7 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_40: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(_unsafe_view_19, [0, 2, 1, 3]);  _unsafe_view_19 = None
        clone_15: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_61: f32[64, 128, 768] = torch.ops.aten.view.default(clone_15, [64, 128, 768]);  clone_15 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_41: f32[768, 768] = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        view_62: f32[8192, 768] = torch.ops.aten.view.default(view_61, [8192, 768]);  view_61 = None
        addmm_21: f32[8192, 768] = torch.ops.aten.addmm.default(primals_61, view_62, permute_41);  primals_61 = None
        view_63: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_21, [64, 128, 768]);  addmm_21 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        rand_like_11: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_63, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_25: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_11);  rand_like_11 = None
        gt_11: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_25, 0.1);  alias_25 = None
        mul_76: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_11, view_63);  view_63 = None
        mul_77: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_76, 1.1111111111111112);  mul_76 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_44: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_77, convert_element_type_6);  mul_77 = convert_element_type_6 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_14: f32[64, 128, 1] = var_mean_7[0]
        getitem_15: f32[64, 128, 1] = var_mean_7[1];  var_mean_7 = None
        add_45: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        sqrt_7: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_45);  add_45 = None
        reciprocal_10: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        sub_15: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_44, getitem_15);  add_44 = getitem_15 = None
        mul_78: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_15, reciprocal_10);  sub_15 = None
        mul_79: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_78, primals_62)
        add_46: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_79, primals_63);  mul_79 = primals_63 = None
        convert_element_type_7: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_46, torch.float32);  add_46 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_42: f32[768, 3072] = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        view_64: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_7, [8192, 768])
        addmm_22: f32[8192, 3072] = torch.ops.aten.addmm.default(primals_65, view_64, permute_42);  primals_65 = None
        view_65: f32[64, 128, 3072] = torch.ops.aten.view.default(addmm_22, [64, 128, 3072]);  addmm_22 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_80: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_65, 0.5)
        mul_81: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476)
        sign_3: f32[64, 128, 3072] = torch.ops.aten.sign.default(mul_81)
        abs_4: f32[64, 128, 3072] = torch.ops.aten.abs.default(mul_81);  mul_81 = None
        mul_82: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(abs_4, 0.3275911)
        add_47: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_82, 1.0);  mul_82 = None
        reciprocal_11: f32[64, 128, 3072] = torch.ops.aten.reciprocal.default(add_47);  add_47 = None
        mul_83: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(reciprocal_11, 1.0);  reciprocal_11 = None
        mul_84: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_83, 1.061405429)
        add_48: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_84, -1.453152027);  mul_84 = None
        mul_85: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_48, mul_83);  add_48 = None
        add_49: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_85, 1.421413741);  mul_85 = None
        mul_86: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_49, mul_83);  add_49 = None
        add_50: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_86, -0.284496736);  mul_86 = None
        mul_87: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_50, mul_83);  add_50 = None
        add_51: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_87, 0.254829592);  mul_87 = None
        mul_88: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_51, mul_83);  add_51 = mul_83 = None
        neg_3: f32[64, 128, 3072] = torch.ops.aten.neg.default(abs_4)
        mul_89: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(neg_3, abs_4);  neg_3 = abs_4 = None
        exp_7: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_89);  mul_89 = None
        mul_90: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_88, exp_7);  mul_88 = exp_7 = None
        
        # No stacktrace found for following nodes
        _tensor_constant4 = self._tensor_constant4
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_4: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
        sub_16: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_4, mul_90);  lift_fresh_copy_4 = None
        mul_91: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_3, sub_16);  sub_16 = None
        add_52: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_91, 1);  mul_91 = None
        mul_92: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_80, add_52);  mul_80 = add_52 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_43: f32[3072, 768] = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        view_66: f32[8192, 3072] = torch.ops.aten.view.default(mul_92, [8192, 3072]);  mul_92 = None
        addmm_23: f32[8192, 768] = torch.ops.aten.addmm.default(primals_67, view_66, permute_43);  primals_67 = None
        view_67: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_23, [64, 128, 768]);  addmm_23 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        rand_like_12: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_67, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_26: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_12);  rand_like_12 = None
        gt_12: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_26, 0.1);  alias_26 = None
        mul_93: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_12, view_67);  view_67 = None
        mul_94: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_93, 1.1111111111111112);  mul_93 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_53: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_94, convert_element_type_7);  mul_94 = convert_element_type_7 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_16: f32[64, 128, 1] = var_mean_8[0]
        getitem_17: f32[64, 128, 1] = var_mean_8[1];  var_mean_8 = None
        add_54: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        sqrt_8: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_54);  add_54 = None
        reciprocal_12: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        sub_17: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_53, getitem_17);  add_53 = getitem_17 = None
        mul_95: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_17, reciprocal_12);  sub_17 = None
        mul_96: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_95, primals_68)
        add_55: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_96, primals_69);  mul_96 = primals_69 = None
        convert_element_type_8: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_55, torch.float32);  add_55 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_44: f32[768, 768] = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        view_68: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_8, [8192, 768])
        addmm_24: f32[8192, 768] = torch.ops.aten.addmm.default(primals_71, view_68, permute_44);  primals_71 = None
        view_69: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_24, [64, 128, 768]);  addmm_24 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_45: f32[768, 768] = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        addmm_25: f32[8192, 768] = torch.ops.aten.addmm.default(primals_73, view_68, permute_45);  primals_73 = None
        view_71: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_25, [64, 128, 768]);  addmm_25 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_72: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_71, [64, 128, 12, 64]);  view_71 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_46: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_47: f32[768, 768] = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        addmm_26: f32[8192, 768] = torch.ops.aten.addmm.default(primals_75, view_68, permute_47);  primals_75 = None
        view_74: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_26, [64, 128, 768]);  addmm_26 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_75: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_74, [64, 128, 12, 64]);  view_74 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_48: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_76: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_69, [64, 128, 12, 64]);  view_69 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_49: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_50: f32[64, 12, 64, 128] = torch.ops.aten.permute.default(permute_46, [0, 1, 3, 2]);  permute_46 = None
        expand_17: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_49, [64, 12, 128, 64]);  permute_49 = None
        clone_16: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        _unsafe_view_20: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_16, [768, 128, 64]);  clone_16 = None
        expand_18: f32[64, 12, 64, 128] = torch.ops.aten.expand.default(permute_50, [64, 12, 64, 128]);  permute_50 = None
        clone_17: f32[64, 12, 64, 128] = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        _unsafe_view_21: f32[768, 64, 128] = torch.ops.aten._unsafe_view.default(clone_17, [768, 64, 128]);  clone_17 = None
        bmm_8: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_20, _unsafe_view_21)
        _unsafe_view_22: f32[64, 12, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_8, [64, 12, 128, 128]);  bmm_8 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_8: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(_unsafe_view_22, 8.0);  _unsafe_view_22 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:344, code: attention_scores = attention_scores + attention_mask
        add_56: f32[64, 12, 128, 128] = torch.ops.aten.add.Tensor(div_8, mul);  div_8 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_4: f32[64, 12, 128, 1] = torch.ops.aten.amax.default(add_56, [-1], True)
        sub_18: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(add_56, amax_4);  add_56 = amax_4 = None
        exp_8: f32[64, 12, 128, 128] = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_5: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_9: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(exp_8, sum_5);  exp_8 = sum_5 = None
        alias_28: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(div_9)
        alias_29: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_28);  alias_28 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        rand_like_13: f32[64, 12, 128, 128] = torch.ops.aten.rand_like.default(div_9, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_30: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(rand_like_13);  rand_like_13 = None
        gt_13: b8[64, 12, 128, 128] = torch.ops.aten.gt.Scalar(alias_30, 0.1);  alias_30 = None
        mul_97: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(gt_13, div_9);  div_9 = None
        mul_98: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_97, 1.1111111111111112);  mul_97 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        expand_19: f32[64, 12, 128, 128] = torch.ops.aten.expand.default(mul_98, [64, 12, 128, 128]);  mul_98 = None
        view_77: f32[768, 128, 128] = torch.ops.aten.view.default(expand_19, [768, 128, 128]);  expand_19 = None
        expand_20: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_48, [64, 12, 128, 64]);  permute_48 = None
        clone_18: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        _unsafe_view_23: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_18, [768, 128, 64]);  clone_18 = None
        bmm_9: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_77, _unsafe_view_23)
        _unsafe_view_24: f32[64, 12, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_9, [64, 12, 128, 64]);  bmm_9 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_51: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(_unsafe_view_24, [0, 2, 1, 3]);  _unsafe_view_24 = None
        clone_19: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_78: f32[64, 128, 768] = torch.ops.aten.view.default(clone_19, [64, 128, 768]);  clone_19 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_52: f32[768, 768] = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        view_79: f32[8192, 768] = torch.ops.aten.view.default(view_78, [8192, 768]);  view_78 = None
        addmm_27: f32[8192, 768] = torch.ops.aten.addmm.default(primals_77, view_79, permute_52);  primals_77 = None
        view_80: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_27, [64, 128, 768]);  addmm_27 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        rand_like_14: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_80, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_31: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_14);  rand_like_14 = None
        gt_14: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_31, 0.1);  alias_31 = None
        mul_99: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_14, view_80);  view_80 = None
        mul_100: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_99, 1.1111111111111112);  mul_99 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_57: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_100, convert_element_type_8);  mul_100 = convert_element_type_8 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_18: f32[64, 128, 1] = var_mean_9[0]
        getitem_19: f32[64, 128, 1] = var_mean_9[1];  var_mean_9 = None
        add_58: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        sqrt_9: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_58);  add_58 = None
        reciprocal_13: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        sub_19: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_57, getitem_19);  add_57 = getitem_19 = None
        mul_101: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_19, reciprocal_13);  sub_19 = None
        mul_102: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_101, primals_78)
        add_59: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_102, primals_79);  mul_102 = primals_79 = None
        convert_element_type_9: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_59, torch.float32);  add_59 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_53: f32[768, 3072] = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        view_81: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_9, [8192, 768])
        addmm_28: f32[8192, 3072] = torch.ops.aten.addmm.default(primals_81, view_81, permute_53);  primals_81 = None
        view_82: f32[64, 128, 3072] = torch.ops.aten.view.default(addmm_28, [64, 128, 3072]);  addmm_28 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_103: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_82, 0.5)
        mul_104: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_82, 0.7071067811865476)
        sign_4: f32[64, 128, 3072] = torch.ops.aten.sign.default(mul_104)
        abs_5: f32[64, 128, 3072] = torch.ops.aten.abs.default(mul_104);  mul_104 = None
        mul_105: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(abs_5, 0.3275911)
        add_60: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_105, 1.0);  mul_105 = None
        reciprocal_14: f32[64, 128, 3072] = torch.ops.aten.reciprocal.default(add_60);  add_60 = None
        mul_106: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(reciprocal_14, 1.0);  reciprocal_14 = None
        mul_107: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_106, 1.061405429)
        add_61: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_107, -1.453152027);  mul_107 = None
        mul_108: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_61, mul_106);  add_61 = None
        add_62: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_108, 1.421413741);  mul_108 = None
        mul_109: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_62, mul_106);  add_62 = None
        add_63: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_109, -0.284496736);  mul_109 = None
        mul_110: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_63, mul_106);  add_63 = None
        add_64: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_110, 0.254829592);  mul_110 = None
        mul_111: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_64, mul_106);  add_64 = mul_106 = None
        neg_4: f32[64, 128, 3072] = torch.ops.aten.neg.default(abs_5)
        mul_112: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(neg_4, abs_5);  neg_4 = abs_5 = None
        exp_9: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_112);  mul_112 = None
        mul_113: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_111, exp_9);  mul_111 = exp_9 = None
        
        # No stacktrace found for following nodes
        _tensor_constant5 = self._tensor_constant5
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_5: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = None
        sub_20: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_5, mul_113);  lift_fresh_copy_5 = None
        mul_114: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_4, sub_20);  sub_20 = None
        add_65: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_114, 1);  mul_114 = None
        mul_115: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_103, add_65);  mul_103 = add_65 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_54: f32[3072, 768] = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
        view_83: f32[8192, 3072] = torch.ops.aten.view.default(mul_115, [8192, 3072]);  mul_115 = None
        addmm_29: f32[8192, 768] = torch.ops.aten.addmm.default(primals_83, view_83, permute_54);  primals_83 = None
        view_84: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_29, [64, 128, 768]);  addmm_29 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        rand_like_15: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_84, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_32: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_15);  rand_like_15 = None
        gt_15: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_32, 0.1);  alias_32 = None
        mul_116: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_15, view_84);  view_84 = None
        mul_117: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_116, 1.1111111111111112);  mul_116 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_66: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_117, convert_element_type_9);  mul_117 = convert_element_type_9 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
        getitem_20: f32[64, 128, 1] = var_mean_10[0]
        getitem_21: f32[64, 128, 1] = var_mean_10[1];  var_mean_10 = None
        add_67: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        sqrt_10: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_67);  add_67 = None
        reciprocal_15: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        sub_21: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_66, getitem_21);  add_66 = getitem_21 = None
        mul_118: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_21, reciprocal_15);  sub_21 = None
        mul_119: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_118, primals_84)
        add_68: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_119, primals_85);  mul_119 = primals_85 = None
        convert_element_type_10: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_68, torch.float32);  add_68 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_55: f32[768, 768] = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
        view_85: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_10, [8192, 768])
        addmm_30: f32[8192, 768] = torch.ops.aten.addmm.default(primals_87, view_85, permute_55);  primals_87 = None
        view_86: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_30, [64, 128, 768]);  addmm_30 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_56: f32[768, 768] = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        addmm_31: f32[8192, 768] = torch.ops.aten.addmm.default(primals_89, view_85, permute_56);  primals_89 = None
        view_88: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_31, [64, 128, 768]);  addmm_31 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_89: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_88, [64, 128, 12, 64]);  view_88 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_57: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_58: f32[768, 768] = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
        addmm_32: f32[8192, 768] = torch.ops.aten.addmm.default(primals_91, view_85, permute_58);  primals_91 = None
        view_91: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_32, [64, 128, 768]);  addmm_32 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_92: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_91, [64, 128, 12, 64]);  view_91 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_59: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_93: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_86, [64, 128, 12, 64]);  view_86 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_60: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_61: f32[64, 12, 64, 128] = torch.ops.aten.permute.default(permute_57, [0, 1, 3, 2]);  permute_57 = None
        expand_21: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_60, [64, 12, 128, 64]);  permute_60 = None
        clone_20: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        _unsafe_view_25: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_20, [768, 128, 64]);  clone_20 = None
        expand_22: f32[64, 12, 64, 128] = torch.ops.aten.expand.default(permute_61, [64, 12, 64, 128]);  permute_61 = None
        clone_21: f32[64, 12, 64, 128] = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        _unsafe_view_26: f32[768, 64, 128] = torch.ops.aten._unsafe_view.default(clone_21, [768, 64, 128]);  clone_21 = None
        bmm_10: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_25, _unsafe_view_26)
        _unsafe_view_27: f32[64, 12, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_10, [64, 12, 128, 128]);  bmm_10 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_10: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(_unsafe_view_27, 8.0);  _unsafe_view_27 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:344, code: attention_scores = attention_scores + attention_mask
        add_69: f32[64, 12, 128, 128] = torch.ops.aten.add.Tensor(div_10, mul);  div_10 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_5: f32[64, 12, 128, 1] = torch.ops.aten.amax.default(add_69, [-1], True)
        sub_22: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(add_69, amax_5);  add_69 = amax_5 = None
        exp_10: f32[64, 12, 128, 128] = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_6: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_11: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(exp_10, sum_6);  exp_10 = sum_6 = None
        alias_34: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(div_11)
        alias_35: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_34);  alias_34 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        rand_like_16: f32[64, 12, 128, 128] = torch.ops.aten.rand_like.default(div_11, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_36: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(rand_like_16);  rand_like_16 = None
        gt_16: b8[64, 12, 128, 128] = torch.ops.aten.gt.Scalar(alias_36, 0.1);  alias_36 = None
        mul_120: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(gt_16, div_11);  div_11 = None
        mul_121: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_120, 1.1111111111111112);  mul_120 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        expand_23: f32[64, 12, 128, 128] = torch.ops.aten.expand.default(mul_121, [64, 12, 128, 128]);  mul_121 = None
        view_94: f32[768, 128, 128] = torch.ops.aten.view.default(expand_23, [768, 128, 128]);  expand_23 = None
        expand_24: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_59, [64, 12, 128, 64]);  permute_59 = None
        clone_22: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        _unsafe_view_28: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_22, [768, 128, 64]);  clone_22 = None
        bmm_11: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_94, _unsafe_view_28)
        _unsafe_view_29: f32[64, 12, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_11, [64, 12, 128, 64]);  bmm_11 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_62: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(_unsafe_view_29, [0, 2, 1, 3]);  _unsafe_view_29 = None
        clone_23: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_95: f32[64, 128, 768] = torch.ops.aten.view.default(clone_23, [64, 128, 768]);  clone_23 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_63: f32[768, 768] = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
        view_96: f32[8192, 768] = torch.ops.aten.view.default(view_95, [8192, 768]);  view_95 = None
        addmm_33: f32[8192, 768] = torch.ops.aten.addmm.default(primals_93, view_96, permute_63);  primals_93 = None
        view_97: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_33, [64, 128, 768]);  addmm_33 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        rand_like_17: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_97, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_37: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_17);  rand_like_17 = None
        gt_17: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_37, 0.1);  alias_37 = None
        mul_122: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_17, view_97);  view_97 = None
        mul_123: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_122, 1.1111111111111112);  mul_122 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_70: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_123, convert_element_type_10);  mul_123 = convert_element_type_10 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_22: f32[64, 128, 1] = var_mean_11[0]
        getitem_23: f32[64, 128, 1] = var_mean_11[1];  var_mean_11 = None
        add_71: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        sqrt_11: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_71);  add_71 = None
        reciprocal_16: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        sub_23: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_70, getitem_23);  add_70 = getitem_23 = None
        mul_124: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_23, reciprocal_16);  sub_23 = None
        mul_125: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_124, primals_94)
        add_72: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_125, primals_95);  mul_125 = primals_95 = None
        convert_element_type_11: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_72, torch.float32);  add_72 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_64: f32[768, 3072] = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
        view_98: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_11, [8192, 768])
        addmm_34: f32[8192, 3072] = torch.ops.aten.addmm.default(primals_97, view_98, permute_64);  primals_97 = None
        view_99: f32[64, 128, 3072] = torch.ops.aten.view.default(addmm_34, [64, 128, 3072]);  addmm_34 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_126: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_99, 0.5)
        mul_127: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_99, 0.7071067811865476)
        sign_5: f32[64, 128, 3072] = torch.ops.aten.sign.default(mul_127)
        abs_6: f32[64, 128, 3072] = torch.ops.aten.abs.default(mul_127);  mul_127 = None
        mul_128: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(abs_6, 0.3275911)
        add_73: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_128, 1.0);  mul_128 = None
        reciprocal_17: f32[64, 128, 3072] = torch.ops.aten.reciprocal.default(add_73);  add_73 = None
        mul_129: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(reciprocal_17, 1.0);  reciprocal_17 = None
        mul_130: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_129, 1.061405429)
        add_74: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_130, -1.453152027);  mul_130 = None
        mul_131: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_74, mul_129);  add_74 = None
        add_75: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_131, 1.421413741);  mul_131 = None
        mul_132: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_75, mul_129);  add_75 = None
        add_76: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_132, -0.284496736);  mul_132 = None
        mul_133: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_76, mul_129);  add_76 = None
        add_77: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_133, 0.254829592);  mul_133 = None
        mul_134: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_77, mul_129);  add_77 = mul_129 = None
        neg_5: f32[64, 128, 3072] = torch.ops.aten.neg.default(abs_6)
        mul_135: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(neg_5, abs_6);  neg_5 = abs_6 = None
        exp_11: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_135);  mul_135 = None
        mul_136: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_134, exp_11);  mul_134 = exp_11 = None
        
        # No stacktrace found for following nodes
        _tensor_constant6 = self._tensor_constant6
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_6: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant6);  _tensor_constant6 = None
        sub_24: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_6, mul_136);  lift_fresh_copy_6 = None
        mul_137: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_5, sub_24);  sub_24 = None
        add_78: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_137, 1);  mul_137 = None
        mul_138: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_126, add_78);  mul_126 = add_78 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_65: f32[3072, 768] = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
        view_100: f32[8192, 3072] = torch.ops.aten.view.default(mul_138, [8192, 3072]);  mul_138 = None
        addmm_35: f32[8192, 768] = torch.ops.aten.addmm.default(primals_99, view_100, permute_65);  primals_99 = None
        view_101: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_35, [64, 128, 768]);  addmm_35 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        rand_like_18: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_101, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_38: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_18);  rand_like_18 = None
        gt_18: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_38, 0.1);  alias_38 = None
        mul_139: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_18, view_101);  view_101 = None
        mul_140: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_139, 1.1111111111111112);  mul_139 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_79: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_140, convert_element_type_11);  mul_140 = convert_element_type_11 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
        getitem_24: f32[64, 128, 1] = var_mean_12[0]
        getitem_25: f32[64, 128, 1] = var_mean_12[1];  var_mean_12 = None
        add_80: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        sqrt_12: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_80);  add_80 = None
        reciprocal_18: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        sub_25: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_79, getitem_25);  add_79 = getitem_25 = None
        mul_141: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_25, reciprocal_18);  sub_25 = None
        mul_142: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_141, primals_100)
        add_81: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_142, primals_101);  mul_142 = primals_101 = None
        convert_element_type_12: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_81, torch.float32);  add_81 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_66: f32[768, 768] = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
        view_102: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_12, [8192, 768])
        addmm_36: f32[8192, 768] = torch.ops.aten.addmm.default(primals_103, view_102, permute_66);  primals_103 = None
        view_103: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_36, [64, 128, 768]);  addmm_36 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_67: f32[768, 768] = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
        addmm_37: f32[8192, 768] = torch.ops.aten.addmm.default(primals_105, view_102, permute_67);  primals_105 = None
        view_105: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_37, [64, 128, 768]);  addmm_37 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_106: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_105, [64, 128, 12, 64]);  view_105 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_68: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_69: f32[768, 768] = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
        addmm_38: f32[8192, 768] = torch.ops.aten.addmm.default(primals_107, view_102, permute_69);  primals_107 = None
        view_108: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_38, [64, 128, 768]);  addmm_38 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_109: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_108, [64, 128, 12, 64]);  view_108 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_70: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_110: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_103, [64, 128, 12, 64]);  view_103 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_71: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_72: f32[64, 12, 64, 128] = torch.ops.aten.permute.default(permute_68, [0, 1, 3, 2]);  permute_68 = None
        expand_25: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_71, [64, 12, 128, 64]);  permute_71 = None
        clone_24: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        _unsafe_view_30: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_24, [768, 128, 64]);  clone_24 = None
        expand_26: f32[64, 12, 64, 128] = torch.ops.aten.expand.default(permute_72, [64, 12, 64, 128]);  permute_72 = None
        clone_25: f32[64, 12, 64, 128] = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        _unsafe_view_31: f32[768, 64, 128] = torch.ops.aten._unsafe_view.default(clone_25, [768, 64, 128]);  clone_25 = None
        bmm_12: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_30, _unsafe_view_31)
        _unsafe_view_32: f32[64, 12, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_12, [64, 12, 128, 128]);  bmm_12 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_12: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(_unsafe_view_32, 8.0);  _unsafe_view_32 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:344, code: attention_scores = attention_scores + attention_mask
        add_82: f32[64, 12, 128, 128] = torch.ops.aten.add.Tensor(div_12, mul);  div_12 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_6: f32[64, 12, 128, 1] = torch.ops.aten.amax.default(add_82, [-1], True)
        sub_26: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(add_82, amax_6);  add_82 = amax_6 = None
        exp_12: f32[64, 12, 128, 128] = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        sum_7: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_13: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(exp_12, sum_7);  exp_12 = sum_7 = None
        alias_40: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(div_13)
        alias_41: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_40);  alias_40 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        rand_like_19: f32[64, 12, 128, 128] = torch.ops.aten.rand_like.default(div_13, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_42: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(rand_like_19);  rand_like_19 = None
        gt_19: b8[64, 12, 128, 128] = torch.ops.aten.gt.Scalar(alias_42, 0.1);  alias_42 = None
        mul_143: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(gt_19, div_13);  div_13 = None
        mul_144: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_143, 1.1111111111111112);  mul_143 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        expand_27: f32[64, 12, 128, 128] = torch.ops.aten.expand.default(mul_144, [64, 12, 128, 128]);  mul_144 = None
        view_111: f32[768, 128, 128] = torch.ops.aten.view.default(expand_27, [768, 128, 128]);  expand_27 = None
        expand_28: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_70, [64, 12, 128, 64]);  permute_70 = None
        clone_26: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        _unsafe_view_33: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_26, [768, 128, 64]);  clone_26 = None
        bmm_13: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_111, _unsafe_view_33)
        _unsafe_view_34: f32[64, 12, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_13, [64, 12, 128, 64]);  bmm_13 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_73: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(_unsafe_view_34, [0, 2, 1, 3]);  _unsafe_view_34 = None
        clone_27: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_112: f32[64, 128, 768] = torch.ops.aten.view.default(clone_27, [64, 128, 768]);  clone_27 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_74: f32[768, 768] = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
        view_113: f32[8192, 768] = torch.ops.aten.view.default(view_112, [8192, 768]);  view_112 = None
        addmm_39: f32[8192, 768] = torch.ops.aten.addmm.default(primals_109, view_113, permute_74);  primals_109 = None
        view_114: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_39, [64, 128, 768]);  addmm_39 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        rand_like_20: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_114, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_43: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_20);  rand_like_20 = None
        gt_20: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_43, 0.1);  alias_43 = None
        mul_145: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_20, view_114);  view_114 = None
        mul_146: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_145, 1.1111111111111112);  mul_145 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_83: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_146, convert_element_type_12);  mul_146 = convert_element_type_12 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_26: f32[64, 128, 1] = var_mean_13[0]
        getitem_27: f32[64, 128, 1] = var_mean_13[1];  var_mean_13 = None
        add_84: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        sqrt_13: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_84);  add_84 = None
        reciprocal_19: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        sub_27: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_83, getitem_27);  add_83 = getitem_27 = None
        mul_147: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_27, reciprocal_19);  sub_27 = None
        mul_148: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_147, primals_110)
        add_85: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_148, primals_111);  mul_148 = primals_111 = None
        convert_element_type_13: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_85, torch.float32);  add_85 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_75: f32[768, 3072] = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
        view_115: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_13, [8192, 768])
        addmm_40: f32[8192, 3072] = torch.ops.aten.addmm.default(primals_113, view_115, permute_75);  primals_113 = None
        view_116: f32[64, 128, 3072] = torch.ops.aten.view.default(addmm_40, [64, 128, 3072]);  addmm_40 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_149: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_116, 0.5)
        mul_150: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_116, 0.7071067811865476)
        sign_6: f32[64, 128, 3072] = torch.ops.aten.sign.default(mul_150)
        abs_7: f32[64, 128, 3072] = torch.ops.aten.abs.default(mul_150);  mul_150 = None
        mul_151: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(abs_7, 0.3275911)
        add_86: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_151, 1.0);  mul_151 = None
        reciprocal_20: f32[64, 128, 3072] = torch.ops.aten.reciprocal.default(add_86);  add_86 = None
        mul_152: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(reciprocal_20, 1.0);  reciprocal_20 = None
        mul_153: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_152, 1.061405429)
        add_87: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_153, -1.453152027);  mul_153 = None
        mul_154: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_87, mul_152);  add_87 = None
        add_88: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_154, 1.421413741);  mul_154 = None
        mul_155: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_88, mul_152);  add_88 = None
        add_89: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_155, -0.284496736);  mul_155 = None
        mul_156: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_89, mul_152);  add_89 = None
        add_90: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_156, 0.254829592);  mul_156 = None
        mul_157: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_90, mul_152);  add_90 = mul_152 = None
        neg_6: f32[64, 128, 3072] = torch.ops.aten.neg.default(abs_7)
        mul_158: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(neg_6, abs_7);  neg_6 = abs_7 = None
        exp_13: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_158);  mul_158 = None
        mul_159: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_157, exp_13);  mul_157 = exp_13 = None
        
        # No stacktrace found for following nodes
        _tensor_constant7 = self._tensor_constant7
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_7: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant7);  _tensor_constant7 = None
        sub_28: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_7, mul_159);  lift_fresh_copy_7 = None
        mul_160: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_6, sub_28);  sub_28 = None
        add_91: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_160, 1);  mul_160 = None
        mul_161: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_149, add_91);  mul_149 = add_91 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_76: f32[3072, 768] = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
        view_117: f32[8192, 3072] = torch.ops.aten.view.default(mul_161, [8192, 3072]);  mul_161 = None
        addmm_41: f32[8192, 768] = torch.ops.aten.addmm.default(primals_115, view_117, permute_76);  primals_115 = None
        view_118: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_41, [64, 128, 768]);  addmm_41 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        rand_like_21: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_118, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_44: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_21);  rand_like_21 = None
        gt_21: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_44, 0.1);  alias_44 = None
        mul_162: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_21, view_118);  view_118 = None
        mul_163: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_162, 1.1111111111111112);  mul_162 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_92: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_163, convert_element_type_13);  mul_163 = convert_element_type_13 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_28: f32[64, 128, 1] = var_mean_14[0]
        getitem_29: f32[64, 128, 1] = var_mean_14[1];  var_mean_14 = None
        add_93: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        sqrt_14: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_93);  add_93 = None
        reciprocal_21: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        sub_29: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_92, getitem_29);  add_92 = getitem_29 = None
        mul_164: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_29, reciprocal_21);  sub_29 = None
        mul_165: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_164, primals_116)
        add_94: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_165, primals_117);  mul_165 = primals_117 = None
        convert_element_type_14: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_94, torch.float32);  add_94 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_77: f32[768, 768] = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
        view_119: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_14, [8192, 768])
        addmm_42: f32[8192, 768] = torch.ops.aten.addmm.default(primals_119, view_119, permute_77);  primals_119 = None
        view_120: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_42, [64, 128, 768]);  addmm_42 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_78: f32[768, 768] = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        addmm_43: f32[8192, 768] = torch.ops.aten.addmm.default(primals_121, view_119, permute_78);  primals_121 = None
        view_122: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_43, [64, 128, 768]);  addmm_43 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_123: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_122, [64, 128, 12, 64]);  view_122 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_79: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_80: f32[768, 768] = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
        addmm_44: f32[8192, 768] = torch.ops.aten.addmm.default(primals_123, view_119, permute_80);  primals_123 = None
        view_125: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_44, [64, 128, 768]);  addmm_44 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_126: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_125, [64, 128, 12, 64]);  view_125 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_81: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_127: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_120, [64, 128, 12, 64]);  view_120 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_82: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_127, [0, 2, 1, 3]);  view_127 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_83: f32[64, 12, 64, 128] = torch.ops.aten.permute.default(permute_79, [0, 1, 3, 2]);  permute_79 = None
        expand_29: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_82, [64, 12, 128, 64]);  permute_82 = None
        clone_28: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        _unsafe_view_35: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_28, [768, 128, 64]);  clone_28 = None
        expand_30: f32[64, 12, 64, 128] = torch.ops.aten.expand.default(permute_83, [64, 12, 64, 128]);  permute_83 = None
        clone_29: f32[64, 12, 64, 128] = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        _unsafe_view_36: f32[768, 64, 128] = torch.ops.aten._unsafe_view.default(clone_29, [768, 64, 128]);  clone_29 = None
        bmm_14: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_35, _unsafe_view_36)
        _unsafe_view_37: f32[64, 12, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_14, [64, 12, 128, 128]);  bmm_14 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_14: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(_unsafe_view_37, 8.0);  _unsafe_view_37 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:344, code: attention_scores = attention_scores + attention_mask
        add_95: f32[64, 12, 128, 128] = torch.ops.aten.add.Tensor(div_14, mul);  div_14 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_7: f32[64, 12, 128, 1] = torch.ops.aten.amax.default(add_95, [-1], True)
        sub_30: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(add_95, amax_7);  add_95 = amax_7 = None
        exp_14: f32[64, 12, 128, 128] = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        sum_8: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_15: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(exp_14, sum_8);  exp_14 = sum_8 = None
        alias_46: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(div_15)
        alias_47: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_46);  alias_46 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        rand_like_22: f32[64, 12, 128, 128] = torch.ops.aten.rand_like.default(div_15, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_48: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(rand_like_22);  rand_like_22 = None
        gt_22: b8[64, 12, 128, 128] = torch.ops.aten.gt.Scalar(alias_48, 0.1);  alias_48 = None
        mul_166: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(gt_22, div_15);  div_15 = None
        mul_167: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_166, 1.1111111111111112);  mul_166 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        expand_31: f32[64, 12, 128, 128] = torch.ops.aten.expand.default(mul_167, [64, 12, 128, 128]);  mul_167 = None
        view_128: f32[768, 128, 128] = torch.ops.aten.view.default(expand_31, [768, 128, 128]);  expand_31 = None
        expand_32: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_81, [64, 12, 128, 64]);  permute_81 = None
        clone_30: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        _unsafe_view_38: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_30, [768, 128, 64]);  clone_30 = None
        bmm_15: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_128, _unsafe_view_38)
        _unsafe_view_39: f32[64, 12, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_15, [64, 12, 128, 64]);  bmm_15 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_84: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(_unsafe_view_39, [0, 2, 1, 3]);  _unsafe_view_39 = None
        clone_31: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_129: f32[64, 128, 768] = torch.ops.aten.view.default(clone_31, [64, 128, 768]);  clone_31 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_85: f32[768, 768] = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
        view_130: f32[8192, 768] = torch.ops.aten.view.default(view_129, [8192, 768]);  view_129 = None
        addmm_45: f32[8192, 768] = torch.ops.aten.addmm.default(primals_125, view_130, permute_85);  primals_125 = None
        view_131: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_45, [64, 128, 768]);  addmm_45 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        rand_like_23: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_131, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_49: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_23);  rand_like_23 = None
        gt_23: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_49, 0.1);  alias_49 = None
        mul_168: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_23, view_131);  view_131 = None
        mul_169: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_168, 1.1111111111111112);  mul_168 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_96: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_169, convert_element_type_14);  mul_169 = convert_element_type_14 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
        getitem_30: f32[64, 128, 1] = var_mean_15[0]
        getitem_31: f32[64, 128, 1] = var_mean_15[1];  var_mean_15 = None
        add_97: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        sqrt_15: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_97);  add_97 = None
        reciprocal_22: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        sub_31: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_96, getitem_31);  add_96 = getitem_31 = None
        mul_170: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_31, reciprocal_22);  sub_31 = None
        mul_171: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_170, primals_126)
        add_98: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_171, primals_127);  mul_171 = primals_127 = None
        convert_element_type_15: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_98, torch.float32);  add_98 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_86: f32[768, 3072] = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        view_132: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_15, [8192, 768])
        addmm_46: f32[8192, 3072] = torch.ops.aten.addmm.default(primals_129, view_132, permute_86);  primals_129 = None
        view_133: f32[64, 128, 3072] = torch.ops.aten.view.default(addmm_46, [64, 128, 3072]);  addmm_46 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_172: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_133, 0.5)
        mul_173: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_133, 0.7071067811865476)
        sign_7: f32[64, 128, 3072] = torch.ops.aten.sign.default(mul_173)
        abs_8: f32[64, 128, 3072] = torch.ops.aten.abs.default(mul_173);  mul_173 = None
        mul_174: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(abs_8, 0.3275911)
        add_99: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_174, 1.0);  mul_174 = None
        reciprocal_23: f32[64, 128, 3072] = torch.ops.aten.reciprocal.default(add_99);  add_99 = None
        mul_175: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(reciprocal_23, 1.0);  reciprocal_23 = None
        mul_176: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_175, 1.061405429)
        add_100: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_176, -1.453152027);  mul_176 = None
        mul_177: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_100, mul_175);  add_100 = None
        add_101: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_177, 1.421413741);  mul_177 = None
        mul_178: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_101, mul_175);  add_101 = None
        add_102: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_178, -0.284496736);  mul_178 = None
        mul_179: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_102, mul_175);  add_102 = None
        add_103: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_179, 0.254829592);  mul_179 = None
        mul_180: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_103, mul_175);  add_103 = mul_175 = None
        neg_7: f32[64, 128, 3072] = torch.ops.aten.neg.default(abs_8)
        mul_181: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(neg_7, abs_8);  neg_7 = abs_8 = None
        exp_15: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_181);  mul_181 = None
        mul_182: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_180, exp_15);  mul_180 = exp_15 = None
        
        # No stacktrace found for following nodes
        _tensor_constant8 = self._tensor_constant8
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_8: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant8);  _tensor_constant8 = None
        sub_32: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_8, mul_182);  lift_fresh_copy_8 = None
        mul_183: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_7, sub_32);  sub_32 = None
        add_104: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_183, 1);  mul_183 = None
        mul_184: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_172, add_104);  mul_172 = add_104 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_87: f32[3072, 768] = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
        view_134: f32[8192, 3072] = torch.ops.aten.view.default(mul_184, [8192, 3072]);  mul_184 = None
        addmm_47: f32[8192, 768] = torch.ops.aten.addmm.default(primals_131, view_134, permute_87);  primals_131 = None
        view_135: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_47, [64, 128, 768]);  addmm_47 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        rand_like_24: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_135, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_50: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_24);  rand_like_24 = None
        gt_24: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_50, 0.1);  alias_50 = None
        mul_185: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_24, view_135);  view_135 = None
        mul_186: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_185, 1.1111111111111112);  mul_185 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_105: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_186, convert_element_type_15);  mul_186 = convert_element_type_15 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
        getitem_32: f32[64, 128, 1] = var_mean_16[0]
        getitem_33: f32[64, 128, 1] = var_mean_16[1];  var_mean_16 = None
        add_106: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        sqrt_16: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_106);  add_106 = None
        reciprocal_24: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        sub_33: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_105, getitem_33);  add_105 = getitem_33 = None
        mul_187: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_33, reciprocal_24);  sub_33 = None
        mul_188: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_187, primals_132)
        add_107: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_188, primals_133);  mul_188 = primals_133 = None
        convert_element_type_16: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_107, torch.float32);  add_107 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_88: f32[768, 768] = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
        view_136: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_16, [8192, 768])
        addmm_48: f32[8192, 768] = torch.ops.aten.addmm.default(primals_135, view_136, permute_88);  primals_135 = None
        view_137: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_48, [64, 128, 768]);  addmm_48 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_89: f32[768, 768] = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
        addmm_49: f32[8192, 768] = torch.ops.aten.addmm.default(primals_137, view_136, permute_89);  primals_137 = None
        view_139: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_49, [64, 128, 768]);  addmm_49 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_140: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_139, [64, 128, 12, 64]);  view_139 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_90: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_91: f32[768, 768] = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
        addmm_50: f32[8192, 768] = torch.ops.aten.addmm.default(primals_139, view_136, permute_91);  primals_139 = None
        view_142: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_50, [64, 128, 768]);  addmm_50 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_143: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_142, [64, 128, 12, 64]);  view_142 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_92: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_143, [0, 2, 1, 3]);  view_143 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_144: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_137, [64, 128, 12, 64]);  view_137 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_93: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_94: f32[64, 12, 64, 128] = torch.ops.aten.permute.default(permute_90, [0, 1, 3, 2]);  permute_90 = None
        expand_33: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_93, [64, 12, 128, 64]);  permute_93 = None
        clone_32: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        _unsafe_view_40: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_32, [768, 128, 64]);  clone_32 = None
        expand_34: f32[64, 12, 64, 128] = torch.ops.aten.expand.default(permute_94, [64, 12, 64, 128]);  permute_94 = None
        clone_33: f32[64, 12, 64, 128] = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        _unsafe_view_41: f32[768, 64, 128] = torch.ops.aten._unsafe_view.default(clone_33, [768, 64, 128]);  clone_33 = None
        bmm_16: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_40, _unsafe_view_41)
        _unsafe_view_42: f32[64, 12, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_16, [64, 12, 128, 128]);  bmm_16 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_16: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(_unsafe_view_42, 8.0);  _unsafe_view_42 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:344, code: attention_scores = attention_scores + attention_mask
        add_108: f32[64, 12, 128, 128] = torch.ops.aten.add.Tensor(div_16, mul);  div_16 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_8: f32[64, 12, 128, 1] = torch.ops.aten.amax.default(add_108, [-1], True)
        sub_34: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(add_108, amax_8);  add_108 = amax_8 = None
        exp_16: f32[64, 12, 128, 128] = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_9: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_17: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(exp_16, sum_9);  exp_16 = sum_9 = None
        alias_52: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(div_17)
        alias_53: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_52);  alias_52 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        rand_like_25: f32[64, 12, 128, 128] = torch.ops.aten.rand_like.default(div_17, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_54: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(rand_like_25);  rand_like_25 = None
        gt_25: b8[64, 12, 128, 128] = torch.ops.aten.gt.Scalar(alias_54, 0.1);  alias_54 = None
        mul_189: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(gt_25, div_17);  div_17 = None
        mul_190: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_189, 1.1111111111111112);  mul_189 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        expand_35: f32[64, 12, 128, 128] = torch.ops.aten.expand.default(mul_190, [64, 12, 128, 128]);  mul_190 = None
        view_145: f32[768, 128, 128] = torch.ops.aten.view.default(expand_35, [768, 128, 128]);  expand_35 = None
        expand_36: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_92, [64, 12, 128, 64]);  permute_92 = None
        clone_34: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        _unsafe_view_43: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_34, [768, 128, 64]);  clone_34 = None
        bmm_17: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_145, _unsafe_view_43)
        _unsafe_view_44: f32[64, 12, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_17, [64, 12, 128, 64]);  bmm_17 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_95: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(_unsafe_view_44, [0, 2, 1, 3]);  _unsafe_view_44 = None
        clone_35: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_146: f32[64, 128, 768] = torch.ops.aten.view.default(clone_35, [64, 128, 768]);  clone_35 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_96: f32[768, 768] = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
        view_147: f32[8192, 768] = torch.ops.aten.view.default(view_146, [8192, 768]);  view_146 = None
        addmm_51: f32[8192, 768] = torch.ops.aten.addmm.default(primals_141, view_147, permute_96);  primals_141 = None
        view_148: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_51, [64, 128, 768]);  addmm_51 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        rand_like_26: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_148, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_55: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_26);  rand_like_26 = None
        gt_26: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_55, 0.1);  alias_55 = None
        mul_191: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_26, view_148);  view_148 = None
        mul_192: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_191, 1.1111111111111112);  mul_191 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_109: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_192, convert_element_type_16);  mul_192 = convert_element_type_16 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
        getitem_34: f32[64, 128, 1] = var_mean_17[0]
        getitem_35: f32[64, 128, 1] = var_mean_17[1];  var_mean_17 = None
        add_110: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        sqrt_17: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_110);  add_110 = None
        reciprocal_25: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
        sub_35: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_109, getitem_35);  add_109 = getitem_35 = None
        mul_193: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_35, reciprocal_25);  sub_35 = None
        mul_194: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_193, primals_142)
        add_111: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_194, primals_143);  mul_194 = primals_143 = None
        convert_element_type_17: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_111, torch.float32);  add_111 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_97: f32[768, 3072] = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
        view_149: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_17, [8192, 768])
        addmm_52: f32[8192, 3072] = torch.ops.aten.addmm.default(primals_145, view_149, permute_97);  primals_145 = None
        view_150: f32[64, 128, 3072] = torch.ops.aten.view.default(addmm_52, [64, 128, 3072]);  addmm_52 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_195: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_150, 0.5)
        mul_196: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_150, 0.7071067811865476)
        sign_8: f32[64, 128, 3072] = torch.ops.aten.sign.default(mul_196)
        abs_9: f32[64, 128, 3072] = torch.ops.aten.abs.default(mul_196);  mul_196 = None
        mul_197: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(abs_9, 0.3275911)
        add_112: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_197, 1.0);  mul_197 = None
        reciprocal_26: f32[64, 128, 3072] = torch.ops.aten.reciprocal.default(add_112);  add_112 = None
        mul_198: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(reciprocal_26, 1.0);  reciprocal_26 = None
        mul_199: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_198, 1.061405429)
        add_113: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_199, -1.453152027);  mul_199 = None
        mul_200: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_113, mul_198);  add_113 = None
        add_114: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_200, 1.421413741);  mul_200 = None
        mul_201: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_114, mul_198);  add_114 = None
        add_115: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_201, -0.284496736);  mul_201 = None
        mul_202: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_115, mul_198);  add_115 = None
        add_116: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_202, 0.254829592);  mul_202 = None
        mul_203: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_116, mul_198);  add_116 = mul_198 = None
        neg_8: f32[64, 128, 3072] = torch.ops.aten.neg.default(abs_9)
        mul_204: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(neg_8, abs_9);  neg_8 = abs_9 = None
        exp_17: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_204);  mul_204 = None
        mul_205: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_203, exp_17);  mul_203 = exp_17 = None
        
        # No stacktrace found for following nodes
        _tensor_constant9 = self._tensor_constant9
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_9: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant9);  _tensor_constant9 = None
        sub_36: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_9, mul_205);  lift_fresh_copy_9 = None
        mul_206: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_8, sub_36);  sub_36 = None
        add_117: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_206, 1);  mul_206 = None
        mul_207: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_195, add_117);  mul_195 = add_117 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_98: f32[3072, 768] = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
        view_151: f32[8192, 3072] = torch.ops.aten.view.default(mul_207, [8192, 3072]);  mul_207 = None
        addmm_53: f32[8192, 768] = torch.ops.aten.addmm.default(primals_147, view_151, permute_98);  primals_147 = None
        view_152: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_53, [64, 128, 768]);  addmm_53 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        rand_like_27: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_152, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_56: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_27);  rand_like_27 = None
        gt_27: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_56, 0.1);  alias_56 = None
        mul_208: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_27, view_152);  view_152 = None
        mul_209: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_208, 1.1111111111111112);  mul_208 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_118: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_209, convert_element_type_17);  mul_209 = convert_element_type_17 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_118, [2], correction = 0, keepdim = True)
        getitem_36: f32[64, 128, 1] = var_mean_18[0]
        getitem_37: f32[64, 128, 1] = var_mean_18[1];  var_mean_18 = None
        add_119: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        sqrt_18: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_119);  add_119 = None
        reciprocal_27: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
        sub_37: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_118, getitem_37);  add_118 = getitem_37 = None
        mul_210: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_37, reciprocal_27);  sub_37 = None
        mul_211: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_210, primals_148)
        add_120: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_211, primals_149);  mul_211 = primals_149 = None
        convert_element_type_18: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_120, torch.float32);  add_120 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_99: f32[768, 768] = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
        view_153: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_18, [8192, 768])
        addmm_54: f32[8192, 768] = torch.ops.aten.addmm.default(primals_151, view_153, permute_99);  primals_151 = None
        view_154: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_54, [64, 128, 768]);  addmm_54 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_100: f32[768, 768] = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
        addmm_55: f32[8192, 768] = torch.ops.aten.addmm.default(primals_153, view_153, permute_100);  primals_153 = None
        view_156: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_55, [64, 128, 768]);  addmm_55 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_157: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_156, [64, 128, 12, 64]);  view_156 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_101: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_102: f32[768, 768] = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
        addmm_56: f32[8192, 768] = torch.ops.aten.addmm.default(primals_155, view_153, permute_102);  primals_155 = None
        view_159: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_56, [64, 128, 768]);  addmm_56 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_160: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_159, [64, 128, 12, 64]);  view_159 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_103: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_161: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_154, [64, 128, 12, 64]);  view_154 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_104: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_105: f32[64, 12, 64, 128] = torch.ops.aten.permute.default(permute_101, [0, 1, 3, 2]);  permute_101 = None
        expand_37: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_104, [64, 12, 128, 64]);  permute_104 = None
        clone_36: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        _unsafe_view_45: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_36, [768, 128, 64]);  clone_36 = None
        expand_38: f32[64, 12, 64, 128] = torch.ops.aten.expand.default(permute_105, [64, 12, 64, 128]);  permute_105 = None
        clone_37: f32[64, 12, 64, 128] = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        _unsafe_view_46: f32[768, 64, 128] = torch.ops.aten._unsafe_view.default(clone_37, [768, 64, 128]);  clone_37 = None
        bmm_18: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_45, _unsafe_view_46)
        _unsafe_view_47: f32[64, 12, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_18, [64, 12, 128, 128]);  bmm_18 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_18: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(_unsafe_view_47, 8.0);  _unsafe_view_47 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:344, code: attention_scores = attention_scores + attention_mask
        add_121: f32[64, 12, 128, 128] = torch.ops.aten.add.Tensor(div_18, mul);  div_18 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_9: f32[64, 12, 128, 1] = torch.ops.aten.amax.default(add_121, [-1], True)
        sub_38: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(add_121, amax_9);  add_121 = amax_9 = None
        exp_18: f32[64, 12, 128, 128] = torch.ops.aten.exp.default(sub_38);  sub_38 = None
        sum_10: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_19: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(exp_18, sum_10);  exp_18 = sum_10 = None
        alias_58: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(div_19)
        alias_59: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_58);  alias_58 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        rand_like_28: f32[64, 12, 128, 128] = torch.ops.aten.rand_like.default(div_19, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_60: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(rand_like_28);  rand_like_28 = None
        gt_28: b8[64, 12, 128, 128] = torch.ops.aten.gt.Scalar(alias_60, 0.1);  alias_60 = None
        mul_212: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(gt_28, div_19);  div_19 = None
        mul_213: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_212, 1.1111111111111112);  mul_212 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        expand_39: f32[64, 12, 128, 128] = torch.ops.aten.expand.default(mul_213, [64, 12, 128, 128]);  mul_213 = None
        view_162: f32[768, 128, 128] = torch.ops.aten.view.default(expand_39, [768, 128, 128]);  expand_39 = None
        expand_40: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_103, [64, 12, 128, 64]);  permute_103 = None
        clone_38: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        _unsafe_view_48: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_38, [768, 128, 64]);  clone_38 = None
        bmm_19: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_162, _unsafe_view_48)
        _unsafe_view_49: f32[64, 12, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_19, [64, 12, 128, 64]);  bmm_19 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_106: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(_unsafe_view_49, [0, 2, 1, 3]);  _unsafe_view_49 = None
        clone_39: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_163: f32[64, 128, 768] = torch.ops.aten.view.default(clone_39, [64, 128, 768]);  clone_39 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_107: f32[768, 768] = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
        view_164: f32[8192, 768] = torch.ops.aten.view.default(view_163, [8192, 768]);  view_163 = None
        addmm_57: f32[8192, 768] = torch.ops.aten.addmm.default(primals_157, view_164, permute_107);  primals_157 = None
        view_165: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_57, [64, 128, 768]);  addmm_57 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        rand_like_29: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_165, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_61: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_29);  rand_like_29 = None
        gt_29: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_61, 0.1);  alias_61 = None
        mul_214: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_29, view_165);  view_165 = None
        mul_215: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_214, 1.1111111111111112);  mul_214 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_122: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_215, convert_element_type_18);  mul_215 = convert_element_type_18 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_122, [2], correction = 0, keepdim = True)
        getitem_38: f32[64, 128, 1] = var_mean_19[0]
        getitem_39: f32[64, 128, 1] = var_mean_19[1];  var_mean_19 = None
        add_123: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        sqrt_19: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_123);  add_123 = None
        reciprocal_28: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
        sub_39: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_122, getitem_39);  add_122 = getitem_39 = None
        mul_216: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_39, reciprocal_28);  sub_39 = None
        mul_217: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_216, primals_158)
        add_124: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_217, primals_159);  mul_217 = primals_159 = None
        convert_element_type_19: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_124, torch.float32);  add_124 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_108: f32[768, 3072] = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
        view_166: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_19, [8192, 768])
        addmm_58: f32[8192, 3072] = torch.ops.aten.addmm.default(primals_161, view_166, permute_108);  primals_161 = None
        view_167: f32[64, 128, 3072] = torch.ops.aten.view.default(addmm_58, [64, 128, 3072]);  addmm_58 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_218: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_167, 0.5)
        mul_219: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_167, 0.7071067811865476)
        sign_9: f32[64, 128, 3072] = torch.ops.aten.sign.default(mul_219)
        abs_10: f32[64, 128, 3072] = torch.ops.aten.abs.default(mul_219);  mul_219 = None
        mul_220: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(abs_10, 0.3275911)
        add_125: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_220, 1.0);  mul_220 = None
        reciprocal_29: f32[64, 128, 3072] = torch.ops.aten.reciprocal.default(add_125);  add_125 = None
        mul_221: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(reciprocal_29, 1.0);  reciprocal_29 = None
        mul_222: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_221, 1.061405429)
        add_126: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_222, -1.453152027);  mul_222 = None
        mul_223: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_126, mul_221);  add_126 = None
        add_127: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_223, 1.421413741);  mul_223 = None
        mul_224: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_127, mul_221);  add_127 = None
        add_128: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_224, -0.284496736);  mul_224 = None
        mul_225: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_128, mul_221);  add_128 = None
        add_129: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_225, 0.254829592);  mul_225 = None
        mul_226: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_129, mul_221);  add_129 = mul_221 = None
        neg_9: f32[64, 128, 3072] = torch.ops.aten.neg.default(abs_10)
        mul_227: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(neg_9, abs_10);  neg_9 = abs_10 = None
        exp_19: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_227);  mul_227 = None
        mul_228: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_226, exp_19);  mul_226 = exp_19 = None
        
        # No stacktrace found for following nodes
        _tensor_constant10 = self._tensor_constant10
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_10: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant10);  _tensor_constant10 = None
        sub_40: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_10, mul_228);  lift_fresh_copy_10 = None
        mul_229: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_9, sub_40);  sub_40 = None
        add_130: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_229, 1);  mul_229 = None
        mul_230: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_218, add_130);  mul_218 = add_130 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_109: f32[3072, 768] = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
        view_168: f32[8192, 3072] = torch.ops.aten.view.default(mul_230, [8192, 3072]);  mul_230 = None
        addmm_59: f32[8192, 768] = torch.ops.aten.addmm.default(primals_163, view_168, permute_109);  primals_163 = None
        view_169: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_59, [64, 128, 768]);  addmm_59 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        rand_like_30: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_169, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_62: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_30);  rand_like_30 = None
        gt_30: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_62, 0.1);  alias_62 = None
        mul_231: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_30, view_169);  view_169 = None
        mul_232: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_231, 1.1111111111111112);  mul_231 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_131: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_232, convert_element_type_19);  mul_232 = convert_element_type_19 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_131, [2], correction = 0, keepdim = True)
        getitem_40: f32[64, 128, 1] = var_mean_20[0]
        getitem_41: f32[64, 128, 1] = var_mean_20[1];  var_mean_20 = None
        add_132: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        sqrt_20: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_30: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
        sub_41: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_131, getitem_41);  add_131 = getitem_41 = None
        mul_233: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_41, reciprocal_30);  sub_41 = None
        mul_234: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_233, primals_164)
        add_133: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_234, primals_165);  mul_234 = primals_165 = None
        convert_element_type_20: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_133, torch.float32);  add_133 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_110: f32[768, 768] = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
        view_170: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_20, [8192, 768])
        addmm_60: f32[8192, 768] = torch.ops.aten.addmm.default(primals_167, view_170, permute_110);  primals_167 = None
        view_171: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_60, [64, 128, 768]);  addmm_60 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_111: f32[768, 768] = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
        addmm_61: f32[8192, 768] = torch.ops.aten.addmm.default(primals_169, view_170, permute_111);  primals_169 = None
        view_173: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_61, [64, 128, 768]);  addmm_61 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_174: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_173, [64, 128, 12, 64]);  view_173 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_112: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_113: f32[768, 768] = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
        addmm_62: f32[8192, 768] = torch.ops.aten.addmm.default(primals_171, view_170, permute_113);  primals_171 = None
        view_176: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_62, [64, 128, 768]);  addmm_62 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_177: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_176, [64, 128, 12, 64]);  view_176 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_114: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_178: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_171, [64, 128, 12, 64]);  view_171 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_115: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_116: f32[64, 12, 64, 128] = torch.ops.aten.permute.default(permute_112, [0, 1, 3, 2]);  permute_112 = None
        expand_41: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_115, [64, 12, 128, 64]);  permute_115 = None
        clone_40: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        _unsafe_view_50: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_40, [768, 128, 64]);  clone_40 = None
        expand_42: f32[64, 12, 64, 128] = torch.ops.aten.expand.default(permute_116, [64, 12, 64, 128]);  permute_116 = None
        clone_41: f32[64, 12, 64, 128] = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        _unsafe_view_51: f32[768, 64, 128] = torch.ops.aten._unsafe_view.default(clone_41, [768, 64, 128]);  clone_41 = None
        bmm_20: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_50, _unsafe_view_51)
        _unsafe_view_52: f32[64, 12, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_20, [64, 12, 128, 128]);  bmm_20 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_20: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(_unsafe_view_52, 8.0);  _unsafe_view_52 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:344, code: attention_scores = attention_scores + attention_mask
        add_134: f32[64, 12, 128, 128] = torch.ops.aten.add.Tensor(div_20, mul);  div_20 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_10: f32[64, 12, 128, 1] = torch.ops.aten.amax.default(add_134, [-1], True)
        sub_42: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(add_134, amax_10);  add_134 = amax_10 = None
        exp_20: f32[64, 12, 128, 128] = torch.ops.aten.exp.default(sub_42);  sub_42 = None
        sum_11: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_21: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(exp_20, sum_11);  exp_20 = sum_11 = None
        alias_64: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(div_21)
        alias_65: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_64);  alias_64 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        rand_like_31: f32[64, 12, 128, 128] = torch.ops.aten.rand_like.default(div_21, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_66: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(rand_like_31);  rand_like_31 = None
        gt_31: b8[64, 12, 128, 128] = torch.ops.aten.gt.Scalar(alias_66, 0.1);  alias_66 = None
        mul_235: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(gt_31, div_21);  div_21 = None
        mul_236: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_235, 1.1111111111111112);  mul_235 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        expand_43: f32[64, 12, 128, 128] = torch.ops.aten.expand.default(mul_236, [64, 12, 128, 128]);  mul_236 = None
        view_179: f32[768, 128, 128] = torch.ops.aten.view.default(expand_43, [768, 128, 128]);  expand_43 = None
        expand_44: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_114, [64, 12, 128, 64]);  permute_114 = None
        clone_42: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        _unsafe_view_53: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_42, [768, 128, 64]);  clone_42 = None
        bmm_21: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_179, _unsafe_view_53)
        _unsafe_view_54: f32[64, 12, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_21, [64, 12, 128, 64]);  bmm_21 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_117: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(_unsafe_view_54, [0, 2, 1, 3]);  _unsafe_view_54 = None
        clone_43: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_180: f32[64, 128, 768] = torch.ops.aten.view.default(clone_43, [64, 128, 768]);  clone_43 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_118: f32[768, 768] = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
        view_181: f32[8192, 768] = torch.ops.aten.view.default(view_180, [8192, 768]);  view_180 = None
        addmm_63: f32[8192, 768] = torch.ops.aten.addmm.default(primals_173, view_181, permute_118);  primals_173 = None
        view_182: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_63, [64, 128, 768]);  addmm_63 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        rand_like_32: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_182, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_67: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_32);  rand_like_32 = None
        gt_32: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_67, 0.1);  alias_67 = None
        mul_237: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_32, view_182);  view_182 = None
        mul_238: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_237, 1.1111111111111112);  mul_237 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_135: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_238, convert_element_type_20);  mul_238 = convert_element_type_20 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_135, [2], correction = 0, keepdim = True)
        getitem_42: f32[64, 128, 1] = var_mean_21[0]
        getitem_43: f32[64, 128, 1] = var_mean_21[1];  var_mean_21 = None
        add_136: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        sqrt_21: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_136);  add_136 = None
        reciprocal_31: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
        sub_43: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_135, getitem_43);  add_135 = getitem_43 = None
        mul_239: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_43, reciprocal_31);  sub_43 = None
        mul_240: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_239, primals_174)
        add_137: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_240, primals_175);  mul_240 = primals_175 = None
        convert_element_type_21: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_137, torch.float32);  add_137 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_119: f32[768, 3072] = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
        view_183: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_21, [8192, 768])
        addmm_64: f32[8192, 3072] = torch.ops.aten.addmm.default(primals_177, view_183, permute_119);  primals_177 = None
        view_184: f32[64, 128, 3072] = torch.ops.aten.view.default(addmm_64, [64, 128, 3072]);  addmm_64 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_241: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_184, 0.5)
        mul_242: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_184, 0.7071067811865476)
        sign_10: f32[64, 128, 3072] = torch.ops.aten.sign.default(mul_242)
        abs_11: f32[64, 128, 3072] = torch.ops.aten.abs.default(mul_242);  mul_242 = None
        mul_243: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(abs_11, 0.3275911)
        add_138: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_243, 1.0);  mul_243 = None
        reciprocal_32: f32[64, 128, 3072] = torch.ops.aten.reciprocal.default(add_138);  add_138 = None
        mul_244: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(reciprocal_32, 1.0);  reciprocal_32 = None
        mul_245: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_244, 1.061405429)
        add_139: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_245, -1.453152027);  mul_245 = None
        mul_246: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_139, mul_244);  add_139 = None
        add_140: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_246, 1.421413741);  mul_246 = None
        mul_247: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_140, mul_244);  add_140 = None
        add_141: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_247, -0.284496736);  mul_247 = None
        mul_248: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_141, mul_244);  add_141 = None
        add_142: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_248, 0.254829592);  mul_248 = None
        mul_249: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_142, mul_244);  add_142 = mul_244 = None
        neg_10: f32[64, 128, 3072] = torch.ops.aten.neg.default(abs_11)
        mul_250: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(neg_10, abs_11);  neg_10 = abs_11 = None
        exp_21: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_250);  mul_250 = None
        mul_251: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_249, exp_21);  mul_249 = exp_21 = None
        
        # No stacktrace found for following nodes
        _tensor_constant11 = self._tensor_constant11
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_11: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant11);  _tensor_constant11 = None
        sub_44: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_11, mul_251);  lift_fresh_copy_11 = None
        mul_252: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_10, sub_44);  sub_44 = None
        add_143: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_252, 1);  mul_252 = None
        mul_253: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_241, add_143);  mul_241 = add_143 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_120: f32[3072, 768] = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
        view_185: f32[8192, 3072] = torch.ops.aten.view.default(mul_253, [8192, 3072]);  mul_253 = None
        addmm_65: f32[8192, 768] = torch.ops.aten.addmm.default(primals_179, view_185, permute_120);  primals_179 = None
        view_186: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_65, [64, 128, 768]);  addmm_65 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        rand_like_33: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_186, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_68: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_33);  rand_like_33 = None
        gt_33: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_68, 0.1);  alias_68 = None
        mul_254: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_33, view_186);  view_186 = None
        mul_255: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_254, 1.1111111111111112);  mul_254 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_144: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_255, convert_element_type_21);  mul_255 = convert_element_type_21 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_144, [2], correction = 0, keepdim = True)
        getitem_44: f32[64, 128, 1] = var_mean_22[0]
        getitem_45: f32[64, 128, 1] = var_mean_22[1];  var_mean_22 = None
        add_145: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        sqrt_22: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_145);  add_145 = None
        reciprocal_33: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
        sub_45: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_144, getitem_45);  add_144 = getitem_45 = None
        mul_256: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_45, reciprocal_33);  sub_45 = None
        mul_257: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_256, primals_180)
        add_146: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_257, primals_181);  mul_257 = primals_181 = None
        convert_element_type_22: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_146, torch.float32);  add_146 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_121: f32[768, 768] = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
        view_187: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_22, [8192, 768])
        addmm_66: f32[8192, 768] = torch.ops.aten.addmm.default(primals_183, view_187, permute_121);  primals_183 = None
        view_188: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_66, [64, 128, 768]);  addmm_66 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_122: f32[768, 768] = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
        addmm_67: f32[8192, 768] = torch.ops.aten.addmm.default(primals_185, view_187, permute_122);  primals_185 = None
        view_190: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_67, [64, 128, 768]);  addmm_67 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_191: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_190, [64, 128, 12, 64]);  view_190 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_123: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_124: f32[768, 768] = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
        addmm_68: f32[8192, 768] = torch.ops.aten.addmm.default(primals_187, view_187, permute_124);  primals_187 = None
        view_193: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_68, [64, 128, 768]);  addmm_68 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_194: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_193, [64, 128, 12, 64]);  view_193 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_125: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:272, code: x = x.view(new_x_shape)
        view_195: f32[64, 128, 12, 64] = torch.ops.aten.view.default(view_188, [64, 128, 12, 64]);  view_188 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: return x.permute(0, 2, 1, 3)
        permute_126: f32[64, 12, 128, 64] = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_127: f32[64, 12, 64, 128] = torch.ops.aten.permute.default(permute_123, [0, 1, 3, 2]);  permute_123 = None
        expand_45: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_126, [64, 12, 128, 64]);  permute_126 = None
        clone_44: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        _unsafe_view_55: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_44, [768, 128, 64]);  clone_44 = None
        expand_46: f32[64, 12, 64, 128] = torch.ops.aten.expand.default(permute_127, [64, 12, 64, 128]);  permute_127 = None
        clone_45: f32[64, 12, 64, 128] = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        _unsafe_view_56: f32[768, 64, 128] = torch.ops.aten._unsafe_view.default(clone_45, [768, 64, 128]);  clone_45 = None
        bmm_22: f32[768, 128, 128] = torch.ops.aten.bmm.default(_unsafe_view_55, _unsafe_view_56)
        _unsafe_view_57: f32[64, 12, 128, 128] = torch.ops.aten._unsafe_view.default(bmm_22, [64, 12, 128, 128]);  bmm_22 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:341, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_22: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(_unsafe_view_57, 8.0);  _unsafe_view_57 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:344, code: attention_scores = attention_scores + attention_mask
        add_147: f32[64, 12, 128, 128] = torch.ops.aten.add.Tensor(div_22, mul);  div_22 = mul = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        amax_11: f32[64, 12, 128, 1] = torch.ops.aten.amax.default(add_147, [-1], True)
        sub_46: f32[64, 12, 128, 128] = torch.ops.aten.sub.Tensor(add_147, amax_11);  add_147 = amax_11 = None
        exp_22: f32[64, 12, 128, 128] = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_12: f32[64, 12, 128, 1] = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_23: f32[64, 12, 128, 128] = torch.ops.aten.div.Tensor(exp_22, sum_12);  exp_22 = sum_12 = None
        alias_70: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(div_23)
        alias_71: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_70);  alias_70 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:351, code: attention_probs = self.dropout(attention_probs)
        rand_like_34: f32[64, 12, 128, 128] = torch.ops.aten.rand_like.default(div_23, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_72: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(rand_like_34);  rand_like_34 = None
        gt_34: b8[64, 12, 128, 128] = torch.ops.aten.gt.Scalar(alias_72, 0.1);  alias_72 = None
        mul_258: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(gt_34, div_23);  div_23 = None
        mul_259: f32[64, 12, 128, 128] = torch.ops.aten.mul.Tensor(mul_258, 1.1111111111111112);  mul_258 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        expand_47: f32[64, 12, 128, 128] = torch.ops.aten.expand.default(mul_259, [64, 12, 128, 128]);  mul_259 = None
        view_196: f32[768, 128, 128] = torch.ops.aten.view.default(expand_47, [768, 128, 128]);  expand_47 = None
        expand_48: f32[64, 12, 128, 64] = torch.ops.aten.expand.default(permute_125, [64, 12, 128, 64]);  permute_125 = None
        clone_46: f32[64, 12, 128, 64] = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        _unsafe_view_58: f32[768, 128, 64] = torch.ops.aten._unsafe_view.default(clone_46, [768, 128, 64]);  clone_46 = None
        bmm_23: f32[768, 128, 64] = torch.ops.aten.bmm.default(view_196, _unsafe_view_58)
        _unsafe_view_59: f32[64, 12, 128, 64] = torch.ops.aten._unsafe_view.default(bmm_23, [64, 12, 128, 64]);  bmm_23 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_128: f32[64, 128, 12, 64] = torch.ops.aten.permute.default(_unsafe_view_59, [0, 2, 1, 3]);  _unsafe_view_59 = None
        clone_47: f32[64, 128, 12, 64] = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:361, code: context_layer = context_layer.view(new_context_layer_shape)
        view_197: f32[64, 128, 768] = torch.ops.aten.view.default(clone_47, [64, 128, 768]);  clone_47 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_129: f32[768, 768] = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
        view_198: f32[8192, 768] = torch.ops.aten.view.default(view_197, [8192, 768]);  view_197 = None
        addmm_69: f32[8192, 768] = torch.ops.aten.addmm.default(primals_189, view_198, permute_129);  primals_189 = None
        view_199: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_69, [64, 128, 768]);  addmm_69 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:379, code: hidden_states = self.dropout(hidden_states)
        rand_like_35: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_199, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_73: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_35);  rand_like_35 = None
        gt_35: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_73, 0.1);  alias_73 = None
        mul_260: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_35, view_199);  view_199 = None
        mul_261: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_260, 1.1111111111111112);  mul_260 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_148: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_261, convert_element_type_22);  mul_261 = convert_element_type_22 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_148, [2], correction = 0, keepdim = True)
        getitem_46: f32[64, 128, 1] = var_mean_23[0]
        getitem_47: f32[64, 128, 1] = var_mean_23[1];  var_mean_23 = None
        add_149: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        sqrt_23: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_149);  add_149 = None
        reciprocal_34: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
        sub_47: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_148, getitem_47);  add_148 = getitem_47 = None
        mul_262: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_47, reciprocal_34);  sub_47 = None
        mul_263: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_262, primals_190)
        add_150: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_263, primals_191);  mul_263 = primals_191 = None
        convert_element_type_23: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_150, torch.float32);  add_150 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_130: f32[768, 3072] = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
        view_200: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_23, [8192, 768])
        addmm_70: f32[8192, 3072] = torch.ops.aten.addmm.default(primals_193, view_200, permute_130);  primals_193 = None
        view_201: f32[64, 128, 3072] = torch.ops.aten.view.default(addmm_70, [64, 128, 3072]);  addmm_70 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_264: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_201, 0.5)
        mul_265: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_201, 0.7071067811865476)
        sign_11: f32[64, 128, 3072] = torch.ops.aten.sign.default(mul_265)
        abs_12: f32[64, 128, 3072] = torch.ops.aten.abs.default(mul_265);  mul_265 = None
        mul_266: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(abs_12, 0.3275911)
        add_151: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_266, 1.0);  mul_266 = None
        reciprocal_35: f32[64, 128, 3072] = torch.ops.aten.reciprocal.default(add_151);  add_151 = None
        mul_267: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(reciprocal_35, 1.0);  reciprocal_35 = None
        mul_268: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_267, 1.061405429)
        add_152: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_268, -1.453152027);  mul_268 = None
        mul_269: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_152, mul_267);  add_152 = None
        add_153: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_269, 1.421413741);  mul_269 = None
        mul_270: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_153, mul_267);  add_153 = None
        add_154: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_270, -0.284496736);  mul_270 = None
        mul_271: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_154, mul_267);  add_154 = None
        add_155: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_271, 0.254829592);  mul_271 = None
        mul_272: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_155, mul_267);  add_155 = mul_267 = None
        neg_11: f32[64, 128, 3072] = torch.ops.aten.neg.default(abs_12)
        mul_273: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(neg_11, abs_12);  neg_11 = abs_12 = None
        exp_23: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_273);  mul_273 = None
        mul_274: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_272, exp_23);  mul_272 = exp_23 = None
        
        # No stacktrace found for following nodes
        _tensor_constant12 = self._tensor_constant12
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_12: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant12);  _tensor_constant12 = None
        sub_48: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_12, mul_274);  lift_fresh_copy_12 = None
        mul_275: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_11, sub_48);  sub_48 = None
        add_156: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_275, 1);  mul_275 = None
        mul_276: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_264, add_156);  mul_264 = add_156 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_131: f32[3072, 768] = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
        view_202: f32[8192, 3072] = torch.ops.aten.view.default(mul_276, [8192, 3072]);  mul_276 = None
        addmm_71: f32[8192, 768] = torch.ops.aten.addmm.default(primals_195, view_202, permute_131);  primals_195 = None
        view_203: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_71, [64, 128, 768]);  addmm_71 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:457, code: hidden_states = self.dropout(hidden_states)
        rand_like_36: f32[64, 128, 768] = torch.ops.aten.rand_like.default(view_203, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_74: f32[64, 128, 768] = torch.ops.aten.alias.default(rand_like_36);  rand_like_36 = None
        gt_36: b8[64, 128, 768] = torch.ops.aten.gt.Scalar(alias_74, 0.1);  alias_74 = None
        mul_277: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(gt_36, view_203);  view_203 = None
        mul_278: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_277, 1.1111111111111112);  mul_277 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_157: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_278, convert_element_type_23);  mul_278 = convert_element_type_23 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
        getitem_48: f32[64, 128, 1] = var_mean_24[0]
        getitem_49: f32[64, 128, 1] = var_mean_24[1];  var_mean_24 = None
        add_158: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        sqrt_24: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_158);  add_158 = None
        reciprocal_36: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
        sub_49: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(add_157, getitem_49);  add_157 = getitem_49 = None
        mul_279: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_49, reciprocal_36);  sub_49 = None
        mul_280: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_279, primals_196)
        add_159: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_280, primals_197);  mul_280 = primals_197 = None
        convert_element_type_24: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_159, torch.float32);  add_159 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:671, code: hidden_states = self.dense(hidden_states)
        permute_132: f32[768, 768] = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
        view_204: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_24, [8192, 768]);  convert_element_type_24 = None
        addmm_72: f32[8192, 768] = torch.ops.aten.addmm.default(primals_199, view_204, permute_132);  primals_199 = None
        view_205: f32[64, 128, 768] = torch.ops.aten.view.default(addmm_72, [64, 128, 768]);  addmm_72 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        mul_281: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(view_205, 0.5)
        mul_282: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(view_205, 0.7071067811865476)
        sign_12: f32[64, 128, 768] = torch.ops.aten.sign.default(mul_282)
        abs_13: f32[64, 128, 768] = torch.ops.aten.abs.default(mul_282);  mul_282 = None
        mul_283: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(abs_13, 0.3275911)
        add_160: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_283, 1.0);  mul_283 = None
        reciprocal_37: f32[64, 128, 768] = torch.ops.aten.reciprocal.default(add_160);  add_160 = None
        mul_284: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(reciprocal_37, 1.0);  reciprocal_37 = None
        mul_285: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_284, 1.061405429)
        add_161: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_285, -1.453152027);  mul_285 = None
        mul_286: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_161, mul_284);  add_161 = None
        add_162: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_286, 1.421413741);  mul_286 = None
        mul_287: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_162, mul_284);  add_162 = None
        add_163: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_287, -0.284496736);  mul_287 = None
        mul_288: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_163, mul_284);  add_163 = None
        add_164: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_288, 0.254829592);  mul_288 = None
        mul_289: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_164, mul_284);  add_164 = mul_284 = None
        neg_12: f32[64, 128, 768] = torch.ops.aten.neg.default(abs_13)
        mul_290: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(neg_12, abs_13);  neg_12 = abs_13 = None
        exp_24: f32[64, 128, 768] = torch.ops.aten.exp.default(mul_290);  mul_290 = None
        mul_291: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_289, exp_24);  mul_289 = exp_24 = None
        
        # No stacktrace found for following nodes
        _tensor_constant13 = self._tensor_constant13
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_13: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant13);  _tensor_constant13 = None
        sub_50: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(lift_fresh_copy_13, mul_291);  lift_fresh_copy_13 = None
        mul_292: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sign_12, sub_50);  sub_50 = None
        add_165: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_292, 1);  mul_292 = None
        mul_293: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_281, add_165);  mul_281 = add_165 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:673, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean_25 = torch.ops.aten.var_mean.correction(mul_293, [2], correction = 0, keepdim = True)
        getitem_50: f32[64, 128, 1] = var_mean_25[0]
        getitem_51: f32[64, 128, 1] = var_mean_25[1];  var_mean_25 = None
        add_166: f32[64, 128, 1] = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        sqrt_25: f32[64, 128, 1] = torch.ops.aten.sqrt.default(add_166);  add_166 = None
        reciprocal_38: f32[64, 128, 1] = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
        sub_51: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(mul_293, getitem_51);  mul_293 = getitem_51 = None
        mul_294: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sub_51, reciprocal_38);  sub_51 = None
        mul_295: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_294, primals_200)
        add_167: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_295, primals_201);  mul_295 = primals_201 = None
        convert_element_type_25: f32[64, 128, 768] = torch.ops.prims.convert_element_type.default(add_167, torch.float32);  add_167 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:693, code: hidden_states = self.decoder(hidden_states)
        permute_133: f32[768, 30522] = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        view_206: f32[8192, 768] = torch.ops.aten.view.default(convert_element_type_25, [8192, 768]);  convert_element_type_25 = None
        addmm_73: f32[8192, 30522] = torch.ops.aten.addmm.default(primals_202, view_206, permute_133);  primals_202 = None
        view_207: f32[64, 128, 30522] = torch.ops.aten.view.default(addmm_73, [64, 128, 30522]);  addmm_73 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1367, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_208: f32[8192, 30522] = torch.ops.aten.view.default(view_207, [-1, 30522])
        view_209: i64[8192] = torch.ops.aten.view.default(primals_206, [-1]);  primals_206 = None
        amax_12: f32[8192, 1] = torch.ops.aten.amax.default(view_208, [1], True)
        sub_52: f32[8192, 30522] = torch.ops.aten.sub.Tensor(view_208, amax_12);  view_208 = amax_12 = None
        exp_25: f32[8192, 30522] = torch.ops.aten.exp.default(sub_52)
        sum_13: f32[8192, 1] = torch.ops.aten.sum.dim_IntList(exp_25, [1], True);  exp_25 = None
        log: f32[8192, 1] = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_53: f32[8192, 30522] = torch.ops.aten.sub.Tensor(sub_52, log);  sub_52 = log = None
        unsqueeze_2: i64[8192, 1] = torch.ops.aten.unsqueeze.default(view_209, 1);  view_209 = None
        gather: f32[8192, 1] = torch.ops.aten.gather.default(sub_53, 1, unsqueeze_2)
        squeeze: f32[8192] = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_13: f32[8192] = torch.ops.aten.neg.default(squeeze);  squeeze = None
        mean: f32[] = torch.ops.aten.mean.default(neg_13);  neg_13 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:693, code: hidden_states = self.decoder(hidden_states)
        permute_134: f32[30522, 768] = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:673, code: hidden_states = self.LayerNorm(hidden_states)
        div_25: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_38, 768);  reciprocal_38 = None
        
        # No stacktrace found for following nodes
        _tensor_constant14 = self._tensor_constant14
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_14: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant14);  _tensor_constant14 = None
        sub_58: f32[64, 128, 768] = torch.ops.aten.sub.Tensor(lift_fresh_copy_14, mul_291);  lift_fresh_copy_14 = mul_291 = None
        mul_315: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(sign_12, sub_58);  sign_12 = sub_58 = None
        add_174: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_315, 1);  mul_315 = None
        mul_316: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(add_174, 0.5);  add_174 = None
        mul_317: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(view_205, view_205)
        mul_318: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(mul_317, -0.5);  mul_317 = None
        exp_28: f32[64, 128, 768] = torch.ops.aten.exp.default(mul_318);  mul_318 = None
        mul_319: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
        mul_320: f32[64, 128, 768] = torch.ops.aten.mul.Tensor(view_205, mul_319);  view_205 = mul_319 = None
        add_175: f32[64, 128, 768] = torch.ops.aten.add.Tensor(mul_316, mul_320);  mul_316 = mul_320 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:671, code: hidden_states = self.dense(hidden_states)
        permute_138: f32[768, 768] = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_26: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_36, 768);  reciprocal_36 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_142: f32[768, 3072] = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        
        # No stacktrace found for following nodes
        _tensor_constant15 = self._tensor_constant15
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_15: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant15);  _tensor_constant15 = None
        sub_62: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_15, mul_274);  lift_fresh_copy_15 = mul_274 = None
        mul_341: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_11, sub_62);  sign_11 = sub_62 = None
        add_181: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_341, 1);  mul_341 = None
        mul_342: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_181, 0.5);  add_181 = None
        mul_343: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_201, view_201)
        mul_344: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_343, -0.5);  mul_343 = None
        exp_30: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_344);  mul_344 = None
        mul_345: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
        mul_346: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_201, mul_345);  view_201 = mul_345 = None
        add_182: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_342, mul_346);  mul_342 = mul_346 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_146: f32[3072, 768] = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_27: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_34, 768);  reciprocal_34 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_150: f32[768, 768] = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_155: f32[768, 128, 128] = torch.ops.aten.permute.default(view_196, [0, 2, 1]);  view_196 = None
        permute_156: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_58, [0, 2, 1]);  _unsafe_view_58 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        alias_82: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_71);  alias_71 = None
        alias_83: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_82);  alias_82 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_157: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_55, [0, 2, 1]);  _unsafe_view_55 = None
        permute_158: f32[768, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_56, [0, 2, 1]);  _unsafe_view_56 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_162: f32[768, 768] = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_167: f32[768, 768] = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_171: f32[768, 768] = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_29: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_33, 768);  reciprocal_33 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_175: f32[768, 3072] = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        
        # No stacktrace found for following nodes
        _tensor_constant16 = self._tensor_constant16
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_16: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant16);  _tensor_constant16 = None
        sub_70: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_16, mul_251);  lift_fresh_copy_16 = mul_251 = None
        mul_380: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_10, sub_70);  sign_10 = sub_70 = None
        add_192: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_380, 1);  mul_380 = None
        mul_381: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_192, 0.5);  add_192 = None
        mul_382: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_184, view_184)
        mul_383: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_382, -0.5);  mul_382 = None
        exp_32: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_383);  mul_383 = None
        mul_384: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
        mul_385: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_184, mul_384);  view_184 = mul_384 = None
        add_193: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_381, mul_385);  mul_381 = mul_385 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_179: f32[3072, 768] = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_30: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_31, 768);  reciprocal_31 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_183: f32[768, 768] = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_188: f32[768, 128, 128] = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None
        permute_189: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_53, [0, 2, 1]);  _unsafe_view_53 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        alias_84: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_65);  alias_65 = None
        alias_85: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_84);  alias_84 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_190: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_50, [0, 2, 1]);  _unsafe_view_50 = None
        permute_191: f32[768, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_51, [0, 2, 1]);  _unsafe_view_51 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_195: f32[768, 768] = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_200: f32[768, 768] = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_204: f32[768, 768] = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_32: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_30, 768);  reciprocal_30 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_208: f32[768, 3072] = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        
        # No stacktrace found for following nodes
        _tensor_constant17 = self._tensor_constant17
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_17: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant17);  _tensor_constant17 = None
        sub_78: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_17, mul_228);  lift_fresh_copy_17 = mul_228 = None
        mul_419: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_9, sub_78);  sign_9 = sub_78 = None
        add_203: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_419, 1);  mul_419 = None
        mul_420: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_203, 0.5);  add_203 = None
        mul_421: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_167, view_167)
        mul_422: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_421, -0.5);  mul_421 = None
        exp_34: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_422);  mul_422 = None
        mul_423: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
        mul_424: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_167, mul_423);  view_167 = mul_423 = None
        add_204: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_420, mul_424);  mul_420 = mul_424 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_212: f32[3072, 768] = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_33: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_28, 768);  reciprocal_28 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_216: f32[768, 768] = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_221: f32[768, 128, 128] = torch.ops.aten.permute.default(view_162, [0, 2, 1]);  view_162 = None
        permute_222: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_48, [0, 2, 1]);  _unsafe_view_48 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        alias_86: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_59);  alias_59 = None
        alias_87: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_86);  alias_86 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_223: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_45, [0, 2, 1]);  _unsafe_view_45 = None
        permute_224: f32[768, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_46, [0, 2, 1]);  _unsafe_view_46 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_228: f32[768, 768] = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_233: f32[768, 768] = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_237: f32[768, 768] = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_35: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_27, 768);  reciprocal_27 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_241: f32[768, 3072] = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        
        # No stacktrace found for following nodes
        _tensor_constant18 = self._tensor_constant18
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_18: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant18);  _tensor_constant18 = None
        sub_86: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_18, mul_205);  lift_fresh_copy_18 = mul_205 = None
        mul_458: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_8, sub_86);  sign_8 = sub_86 = None
        add_214: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_458, 1);  mul_458 = None
        mul_459: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_214, 0.5);  add_214 = None
        mul_460: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_150, view_150)
        mul_461: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_460, -0.5);  mul_460 = None
        exp_36: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_461);  mul_461 = None
        mul_462: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
        mul_463: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_150, mul_462);  view_150 = mul_462 = None
        add_215: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_459, mul_463);  mul_459 = mul_463 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_245: f32[3072, 768] = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_36: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_25, 768);  reciprocal_25 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_249: f32[768, 768] = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_254: f32[768, 128, 128] = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        permute_255: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_43, [0, 2, 1]);  _unsafe_view_43 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        alias_88: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_53);  alias_53 = None
        alias_89: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_88);  alias_88 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_256: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_40, [0, 2, 1]);  _unsafe_view_40 = None
        permute_257: f32[768, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_41, [0, 2, 1]);  _unsafe_view_41 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_261: f32[768, 768] = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_266: f32[768, 768] = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_270: f32[768, 768] = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_38: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_24, 768);  reciprocal_24 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_274: f32[768, 3072] = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        
        # No stacktrace found for following nodes
        _tensor_constant19 = self._tensor_constant19
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_19: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant19);  _tensor_constant19 = None
        sub_94: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_19, mul_182);  lift_fresh_copy_19 = mul_182 = None
        mul_497: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_7, sub_94);  sign_7 = sub_94 = None
        add_225: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_497, 1);  mul_497 = None
        mul_498: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_225, 0.5);  add_225 = None
        mul_499: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_133, view_133)
        mul_500: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_499, -0.5);  mul_499 = None
        exp_38: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_500);  mul_500 = None
        mul_501: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
        mul_502: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_133, mul_501);  view_133 = mul_501 = None
        add_226: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_498, mul_502);  mul_498 = mul_502 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_278: f32[3072, 768] = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_39: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_22, 768);  reciprocal_22 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_282: f32[768, 768] = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_287: f32[768, 128, 128] = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None
        permute_288: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_38, [0, 2, 1]);  _unsafe_view_38 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        alias_90: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_47);  alias_47 = None
        alias_91: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_90);  alias_90 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_289: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_35, [0, 2, 1]);  _unsafe_view_35 = None
        permute_290: f32[768, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_36, [0, 2, 1]);  _unsafe_view_36 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_294: f32[768, 768] = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_299: f32[768, 768] = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_303: f32[768, 768] = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_41: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_21, 768);  reciprocal_21 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_307: f32[768, 3072] = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        
        # No stacktrace found for following nodes
        _tensor_constant20 = self._tensor_constant20
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_20: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant20);  _tensor_constant20 = None
        sub_102: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_20, mul_159);  lift_fresh_copy_20 = mul_159 = None
        mul_536: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_6, sub_102);  sign_6 = sub_102 = None
        add_236: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_536, 1);  mul_536 = None
        mul_537: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_236, 0.5);  add_236 = None
        mul_538: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_116, view_116)
        mul_539: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_538, -0.5);  mul_538 = None
        exp_40: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_539);  mul_539 = None
        mul_540: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
        mul_541: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_116, mul_540);  view_116 = mul_540 = None
        add_237: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_537, mul_541);  mul_537 = mul_541 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_311: f32[3072, 768] = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_42: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_19, 768);  reciprocal_19 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_315: f32[768, 768] = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_320: f32[768, 128, 128] = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
        permute_321: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_33, [0, 2, 1]);  _unsafe_view_33 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        alias_92: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_41);  alias_41 = None
        alias_93: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_92);  alias_92 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_322: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_30, [0, 2, 1]);  _unsafe_view_30 = None
        permute_323: f32[768, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_31, [0, 2, 1]);  _unsafe_view_31 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_327: f32[768, 768] = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_332: f32[768, 768] = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_336: f32[768, 768] = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_44: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_18, 768);  reciprocal_18 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_340: f32[768, 3072] = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        
        # No stacktrace found for following nodes
        _tensor_constant21 = self._tensor_constant21
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_21: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant21);  _tensor_constant21 = None
        sub_110: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_21, mul_136);  lift_fresh_copy_21 = mul_136 = None
        mul_575: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_5, sub_110);  sign_5 = sub_110 = None
        add_247: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_575, 1);  mul_575 = None
        mul_576: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_247, 0.5);  add_247 = None
        mul_577: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_99, view_99)
        mul_578: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_577, -0.5);  mul_577 = None
        exp_42: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_578);  mul_578 = None
        mul_579: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
        mul_580: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_99, mul_579);  view_99 = mul_579 = None
        add_248: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_576, mul_580);  mul_576 = mul_580 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_344: f32[3072, 768] = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_45: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_16, 768);  reciprocal_16 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_348: f32[768, 768] = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_353: f32[768, 128, 128] = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
        permute_354: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_28, [0, 2, 1]);  _unsafe_view_28 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        alias_94: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_35);  alias_35 = None
        alias_95: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_94);  alias_94 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_355: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_25, [0, 2, 1]);  _unsafe_view_25 = None
        permute_356: f32[768, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_26, [0, 2, 1]);  _unsafe_view_26 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_360: f32[768, 768] = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_365: f32[768, 768] = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_369: f32[768, 768] = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_47: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_15, 768);  reciprocal_15 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_373: f32[768, 3072] = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        
        # No stacktrace found for following nodes
        _tensor_constant22 = self._tensor_constant22
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_22: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant22);  _tensor_constant22 = None
        sub_118: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_22, mul_113);  lift_fresh_copy_22 = mul_113 = None
        mul_614: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_4, sub_118);  sign_4 = sub_118 = None
        add_258: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_614, 1);  mul_614 = None
        mul_615: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_258, 0.5);  add_258 = None
        mul_616: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_82, view_82)
        mul_617: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_616, -0.5);  mul_616 = None
        exp_44: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_617);  mul_617 = None
        mul_618: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
        mul_619: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_82, mul_618);  view_82 = mul_618 = None
        add_259: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_615, mul_619);  mul_615 = mul_619 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_377: f32[3072, 768] = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_48: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_13, 768);  reciprocal_13 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_381: f32[768, 768] = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_386: f32[768, 128, 128] = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
        permute_387: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_23, [0, 2, 1]);  _unsafe_view_23 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        alias_96: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_29);  alias_29 = None
        alias_97: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_96);  alias_96 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_388: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_20, [0, 2, 1]);  _unsafe_view_20 = None
        permute_389: f32[768, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_21, [0, 2, 1]);  _unsafe_view_21 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_393: f32[768, 768] = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_398: f32[768, 768] = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_402: f32[768, 768] = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_50: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_12, 768);  reciprocal_12 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_406: f32[768, 3072] = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        
        # No stacktrace found for following nodes
        _tensor_constant23 = self._tensor_constant23
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_23: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant23);  _tensor_constant23 = None
        sub_126: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_23, mul_90);  lift_fresh_copy_23 = mul_90 = None
        mul_653: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_3, sub_126);  sign_3 = sub_126 = None
        add_269: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_653, 1);  mul_653 = None
        mul_654: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_269, 0.5);  add_269 = None
        mul_655: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_65, view_65)
        mul_656: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_655, -0.5);  mul_655 = None
        exp_46: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_656);  mul_656 = None
        mul_657: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
        mul_658: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_65, mul_657);  view_65 = mul_657 = None
        add_270: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_654, mul_658);  mul_654 = mul_658 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_410: f32[3072, 768] = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_51: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_10, 768);  reciprocal_10 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_414: f32[768, 768] = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_419: f32[768, 128, 128] = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
        permute_420: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_18, [0, 2, 1]);  _unsafe_view_18 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        alias_98: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_23);  alias_23 = None
        alias_99: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_98);  alias_98 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_421: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_15, [0, 2, 1]);  _unsafe_view_15 = None
        permute_422: f32[768, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_16, [0, 2, 1]);  _unsafe_view_16 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_426: f32[768, 768] = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_431: f32[768, 768] = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_435: f32[768, 768] = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_53: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_9, 768);  reciprocal_9 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_439: f32[768, 3072] = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        
        # No stacktrace found for following nodes
        _tensor_constant24 = self._tensor_constant24
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_24: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant24);  _tensor_constant24 = None
        sub_134: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_24, mul_67);  lift_fresh_copy_24 = mul_67 = None
        mul_692: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_2, sub_134);  sign_2 = sub_134 = None
        add_280: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_692, 1);  mul_692 = None
        mul_693: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_280, 0.5);  add_280 = None
        mul_694: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_48, view_48)
        mul_695: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_694, -0.5);  mul_694 = None
        exp_48: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_695);  mul_695 = None
        mul_696: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(exp_48, 0.3989422804014327);  exp_48 = None
        mul_697: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_48, mul_696);  view_48 = mul_696 = None
        add_281: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_693, mul_697);  mul_693 = mul_697 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_443: f32[3072, 768] = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_54: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_7, 768);  reciprocal_7 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_447: f32[768, 768] = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_452: f32[768, 128, 128] = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
        permute_453: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_13, [0, 2, 1]);  _unsafe_view_13 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        alias_100: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_17);  alias_17 = None
        alias_101: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_100);  alias_100 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_454: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_10, [0, 2, 1]);  _unsafe_view_10 = None
        permute_455: f32[768, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_11, [0, 2, 1]);  _unsafe_view_11 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_459: f32[768, 768] = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_464: f32[768, 768] = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_468: f32[768, 768] = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_56: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_6, 768);  reciprocal_6 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_472: f32[768, 3072] = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        
        # No stacktrace found for following nodes
        _tensor_constant25 = self._tensor_constant25
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_25: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant25);  _tensor_constant25 = None
        sub_142: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_25, mul_44);  lift_fresh_copy_25 = mul_44 = None
        mul_731: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign_1, sub_142);  sign_1 = sub_142 = None
        add_291: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_731, 1);  mul_731 = None
        mul_732: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_291, 0.5);  add_291 = None
        mul_733: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_31, view_31)
        mul_734: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_733, -0.5);  mul_733 = None
        exp_50: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_734);  mul_734 = None
        mul_735: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(exp_50, 0.3989422804014327);  exp_50 = None
        mul_736: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_31, mul_735);  view_31 = mul_735 = None
        add_292: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_732, mul_736);  mul_732 = mul_736 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_476: f32[3072, 768] = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_57: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_4, 768);  reciprocal_4 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_480: f32[768, 768] = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_485: f32[768, 128, 128] = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
        permute_486: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_8, [0, 2, 1]);  _unsafe_view_8 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        alias_102: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_11);  alias_11 = None
        alias_103: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_102);  alias_102 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_487: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_5, [0, 2, 1]);  _unsafe_view_5 = None
        permute_488: f32[768, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_6, [0, 2, 1]);  _unsafe_view_6 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_492: f32[768, 768] = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_497: f32[768, 768] = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_501: f32[768, 768] = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:458, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_59: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_3, 768);  reciprocal_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:456, code: hidden_states = self.dense(hidden_states)
        permute_505: f32[768, 3072] = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        
        # No stacktrace found for following nodes
        _tensor_constant26 = self._tensor_constant26
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/activations.py:57, code: return self.act(input)
        lift_fresh_copy_26: f32[] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant26);  _tensor_constant26 = None
        sub_150: f32[64, 128, 3072] = torch.ops.aten.sub.Tensor(lift_fresh_copy_26, mul_21);  lift_fresh_copy_26 = mul_21 = None
        mul_770: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(sign, sub_150);  sign = sub_150 = None
        add_302: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_770, 1);  mul_770 = None
        mul_771: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(add_302, 0.5);  add_302 = None
        mul_772: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_14, view_14)
        mul_773: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(mul_772, -0.5);  mul_772 = None
        exp_52: f32[64, 128, 3072] = torch.ops.aten.exp.default(mul_773);  mul_773 = None
        mul_774: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(exp_52, 0.3989422804014327);  exp_52 = None
        mul_775: f32[64, 128, 3072] = torch.ops.aten.mul.Tensor(view_14, mul_774);  view_14 = mul_774 = None
        add_303: f32[64, 128, 3072] = torch.ops.aten.add.Tensor(mul_771, mul_775);  mul_771 = mul_775 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:443, code: hidden_states = self.dense(hidden_states)
        permute_509: f32[3072, 768] = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:380, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_60: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal_1, 768);  reciprocal_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:378, code: hidden_states = self.dense(hidden_states)
        permute_513: f32[768, 768] = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:357, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_518: f32[768, 128, 128] = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_519: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view_3, [0, 2, 1]);  _unsafe_view_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:347, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        alias_104: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_5);  alias_5 = None
        alias_105: f32[64, 12, 128, 128] = torch.ops.aten.alias.default(alias_104);  alias_104 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:323, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_520: f32[768, 64, 128] = torch.ops.aten.permute.default(_unsafe_view, [0, 2, 1]);  _unsafe_view = None
        permute_521: f32[768, 128, 64] = torch.ops.aten.permute.default(_unsafe_view_1, [0, 2, 1]);  _unsafe_view_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        permute_525: f32[768, 768] = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:307, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        permute_530: f32[768, 768] = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:285, code: mixed_query_layer = self.query(hidden_states)
        permute_534: f32[768, 768] = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:238, code: embeddings = self.LayerNorm(embeddings)
        div_62: f32[64, 128, 1] = torch.ops.aten.div.Tensor(reciprocal, 768);  reciprocal = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:236, code: position_embeddings = self.position_embeddings(position_ids)
        view_506: i64[128] = torch.ops.aten.view.default(slice_6, [128]);  slice_6 = None
        
        # File: /scratch/eellison/work/newest_env/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:231, code: inputs_embeds = self.word_embeddings(input_ids)
        view_509: i64[8192] = torch.ops.aten.view.default(primals_205, [8192]);  primals_205 = None
        return [mean, view_207, primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, slice_2, mul_1, gt, view, gt_1, view_11, gt_2, mul_9, view_13, view_15, gt_3, mul_26, view_17, gt_4, view_28, gt_5, mul_32, view_30, view_32, gt_6, mul_49, view_34, gt_7, view_45, gt_8, mul_55, view_47, view_49, gt_9, mul_72, view_51, gt_10, view_62, gt_11, mul_78, view_64, view_66, gt_12, mul_95, view_68, gt_13, view_79, gt_14, mul_101, view_81, view_83, gt_15, mul_118, view_85, gt_16, view_96, gt_17, mul_124, view_98, view_100, gt_18, mul_141, view_102, gt_19, view_113, gt_20, mul_147, view_115, view_117, gt_21, mul_164, view_119, gt_22, view_130, gt_23, mul_170, view_132, view_134, gt_24, mul_187, view_136, gt_25, view_147, gt_26, mul_193, view_149, view_151, gt_27, mul_210, view_153, gt_28, view_164, gt_29, mul_216, view_166, view_168, gt_30, mul_233, view_170, gt_31, view_181, gt_32, mul_239, view_183, view_185, gt_33, mul_256, view_187, gt_34, view_198, gt_35, mul_262, view_200, view_202, gt_36, mul_279, view_204, mul_294, view_206, sub_53, unsqueeze_2, permute_134, div_25, add_175, permute_138, div_26, permute_142, add_182, permute_146, div_27, permute_150, permute_155, permute_156, alias_83, permute_157, permute_158, permute_162, permute_167, permute_171, div_29, permute_175, add_193, permute_179, div_30, permute_183, permute_188, permute_189, alias_85, permute_190, permute_191, permute_195, permute_200, permute_204, div_32, permute_208, add_204, permute_212, div_33, permute_216, permute_221, permute_222, alias_87, permute_223, permute_224, permute_228, permute_233, permute_237, div_35, permute_241, add_215, permute_245, div_36, permute_249, permute_254, permute_255, alias_89, permute_256, permute_257, permute_261, permute_266, permute_270, div_38, permute_274, add_226, permute_278, div_39, permute_282, permute_287, permute_288, alias_91, permute_289, permute_290, permute_294, permute_299, permute_303, div_41, permute_307, add_237, permute_311, div_42, permute_315, permute_320, permute_321, alias_93, permute_322, permute_323, permute_327, permute_332, permute_336, div_44, permute_340, add_248, permute_344, div_45, permute_348, permute_353, permute_354, alias_95, permute_355, permute_356, permute_360, permute_365, permute_369, div_47, permute_373, add_259, permute_377, div_48, permute_381, permute_386, permute_387, alias_97, permute_388, permute_389, permute_393, permute_398, permute_402, div_50, permute_406, add_270, permute_410, div_51, permute_414, permute_419, permute_420, alias_99, permute_421, permute_422, permute_426, permute_431, permute_435, div_53, permute_439, add_281, permute_443, div_54, permute_447, permute_452, permute_453, alias_101, permute_454, permute_455, permute_459, permute_464, permute_468, div_56, permute_472, add_292, permute_476, div_57, permute_480, permute_485, permute_486, alias_103, permute_487, permute_488, permute_492, permute_497, permute_501, div_59, permute_505, add_303, permute_509, div_60, permute_513, permute_518, permute_519, alias_105, permute_520, permute_521, permute_525, permute_530, permute_534, div_62, view_506, view_509]
        